"""Independent verifier for structural-adapter output (increments 1 and 2).

``verify(doc, record, trace_sha256=...)`` re-checks an adapter output against
the raw instrumented record and returns a list of error strings (empty when the
output passes).  It is written separately from the adapter and shares only the
locator/hash resolver (``_semantic_locator``).  Every check is re-derived from
the raw record:

  * shape: the eight 0.3 collections, no claims/propositions/states, only
    ``delivers`` links, ``$unresolved`` only on the origin of a paired model
    call whose agent origin the raw record does not establish (and its argument
    evidence's source), unique ids and integer ``order``;
  * events: every raw seq exists and belongs to at most one event; model events
    are one ``llm_start``/``llm_end`` pair of the origin agent; tool events are
    the start/end(/commit) of one ``tool_call_id`` for the origin tool;
    framework eviction events are one ``file_write``/``tool_result_committed``
    pair whose eviction checks are recomputed here; ``order`` is the first raw
    seq and ``raw_ref.start_seq``/``end_seq`` bound the event's raw seqs;
  * evidence: each locator resolves inside its producer's records and its digest
    and size reproduce; eviction substitutes are the committed content of a
    verified eviction; the question is external and equals the record's field;
  * delivers: each link's locator resolves inside the source event's rendered
    input, the value equals the target exactly, and the structural join
    (``tool_call_id``, ``delegation_id`` for a child's human message, or a root
    agent's human message for the question) holds and is unique in that input;
  * accounting: every raw record is in an event or a skipped record entry, and
    the raw irregularities found here (unpaired calls, missing ids, null
    ``offered_tools``, evictions, the question, system prompts) are surfaced;
  * profile_local: ``claims_extracted`` is false, the delivery-telemetry flag
    matches the record, ``profile_local.order`` has basis ``first_raw_seq`` and
    lists exactly the overlapping event pairs, and no block that would enable
    negative inferences (``coverage``, ``size_bounds``, ``annotation``) is present.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from typing import Any

from agent.deep_agent._semantic_locator import LocatorError, digest, resolve, size

COLLECTIONS = ("run", "entities", "events", "evidence", "claims", "propositions", "states", "links")
ENTITY_KINDS = {"agent", "tool", "service", "framework", "environment"}
NEGATIVE_GATES = ("coverage", "size_bounds", "annotation")

_MSG = re.compile(r"^events\[seq=(\d+)\]\.input_batches\[0\]\[(\d+)\](.*)$")
_ARG = re.compile(r"^\.tool_calls\[(\d+)\]\.args\.([A-Za-z_]\w*)$")
_PRODUCED_ARG = re.compile(r"^events\[seq=(\d+)\]\.message\.tool_calls\[(\d+)\]\.args\.([A-Za-z_]\w*)$")

# The eviction code of the deepagents versions read for this verifier (FilesystemMiddleware._process_large_message).
EVICTION_VERSIONS = {"0.5.3"}
EVICTION_HEAD = (
    "Tool result too large, the result of this tool call {tool_call_id} was saved in the filesystem at this "
    "path: {file_path}\n\nYou can read the result from the filesystem by using the read_file tool, but make "
    "sure to only read part of the result at a time.\n\nYou can do this by specifying an offset and limit in "
    "the read_file tool call. For example, to read the first 100 lines, you can use the read_file tool with "
    "offset=0 and limit=100.\n\nHere is a preview showing the head and tail of the result (lines of the form "
    "`... [N lines truncated] ...` indicate omitted lines in the middle of the content):\n\n"
)


def _text_blocks(output: Any) -> str | None:
    if isinstance(output, str):
        return output
    if not isinstance(output, list) or not output:
        return None
    texts = []
    for b in output:
        if not isinstance(b, dict) or b.get("type") != "text" or not isinstance(b.get("text"), str):
            return None
        texts.append(b["text"])
    return "\n".join(texts)


def _unresolved(value: Any) -> bool:
    return isinstance(value, dict) and "$unresolved" in value


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


class _Verifier:
    def __init__(self, doc: dict, record: dict, trace_sha256: str | None):
        self.doc = doc
        self.rec = record
        self.trace_sha256 = trace_sha256
        self.errors: list[str] = []
        self.by_seq = {r.get("seq"): r for r in record.get("events", [])}

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def run(self) -> list[str]:
        sem = self.doc.get("semantic")
        if not isinstance(sem, dict) or set(sem) != set(COLLECTIONS):
            self.err("semantic must hold exactly the eight 0.3 collections")
            return self.errors
        self.sem = sem
        self._shape()
        self._run()
        self._entities()
        self._events()
        self._evidence()
        self._links()
        self._profile_local()
        self._accounting()
        return self.errors

    # --- shape ---------------------------------------------------------------------

    def _shape(self) -> None:
        for coll in ("claims", "propositions", "states"):
            if self.sem[coll]:
                self.err(f"{coll} must be empty in increment 1")
        self.unresolved_paths: dict[str, dict] = {}
        for coll, field in (("events", "origin"), ("evidence", "source")):
            for o in self.sem[coll]:
                v = o.get(field)
                if isinstance(v, dict) and "$unresolved" in v:
                    self.unresolved_paths[f"semantic.{coll}[{o.get('id')}].{field}"] = v
        allowed = sum(1 for _ in self.unresolved_paths)
        if json.dumps(self.sem).count('"$unresolved"') != allowed:
            self.err("$unresolved occurs outside event origin and argument-evidence source")
        ids = [o.get("id") for coll in ("entities", "events", "evidence") for o in self.sem[coll]]
        dup = [i for i, n in Counter(ids).items() if n > 1]
        if dup or None in ids:
            self.err(f"ids must be present and unique: {dup}")
        self.entities = {e.get("id"): e for e in self.sem["entities"]}
        self.events = {e.get("id"): e for e in self.sem["events"]}
        self.evidence = {e.get("id"): e for e in self.sem["evidence"]}

    def _run(self) -> None:
        run = self.sem["run"]
        ref = run.get("raw_ref", {})
        if run.get("id") != self.rec.get("run_id"):
            self.err("run.id differs from the record's run_id")
        if run.get("outputs") != "UNKNOWN":
            self.err("run.outputs must be UNKNOWN: product markers are OPEN (0.3 §4.1)")
        if ref.get("raw_schema") != self.rec.get("schema"):
            self.err("run.raw_ref.raw_schema differs from the record")
        if self.trace_sha256 is not None and ref.get("trace_sha256") != self.trace_sha256:
            self.err("run.raw_ref.trace_sha256 does not match the trace file")

    def _entities(self) -> None:
        for eid, e in self.entities.items():
            if e.get("kind") not in ENTITY_KINDS:
                self.err(f"{eid}: kind {e.get('kind')!r}")
                continue
            loc = (e.get("raw_ref") or {}).get("locator")
            try:
                target = resolve(self.rec, loc)
            except LocatorError as exc:
                self.err(f"{eid}: {exc}")
                continue
            if e["kind"] == "agent":
                if not isinstance(target, dict) or target.get("kind") != e.get("role") or e.get("model") != self.rec.get("model"):
                    self.err(f"{eid}: role or model does not match the raw agent")
            elif e["kind"] == "tool":
                if target != e["raw_ref"].get("name"):
                    self.err(f"{eid}: tool name does not match the raw record")

    def _agent_of(self, entity_id: str) -> str | None:
        e = self.entities.get(entity_id) or {}
        if e.get("kind") != "agent":
            return None
        try:
            return resolve(self.rec, e["raw_ref"]["locator"]).get("agent_id")
        except (LocatorError, KeyError, AttributeError):
            return None

    # --- events --------------------------------------------------------------------

    def _events(self) -> None:
        self.event_seqs: dict[str, list[int]] = {}
        self.unresolved_event_starts: set = set()
        self.evictions: dict[str, dict] = {}  # tool_call_id -> verified eviction event
        orders = [e.get("order") for e in self.events.values()]
        if any(not isinstance(o, int) or isinstance(o, bool) for o in orders):
            self.err("every order must be an integer")
            return
        if len(set(orders)) != len(orders):
            self.err("order values must be unique")
        used: Counter = Counter()
        for eid, e in self.events.items():
            ent = self.entities.get(e.get("origin")) if isinstance(e.get("origin"), str) else None
            if ent is not None and ent.get("kind") == "framework":
                self._eviction_event(eid, e, ent)
        evict_commits = {ev["raw_ref"]["seq"][-1] for ev in self.evictions.values()}
        lc_count = Counter((r.get("type"), r.get("lc_run_id")) for r in self.by_seq.values())
        id_count = Counter((r.get("type"), r.get("tool_call_id")) for r in self.by_seq.values())
        for eid, e in self.events.items():
            seqs = (e.get("raw_ref") or {}).get("seq")
            if not isinstance(seqs, list) or not seqs or any(s not in self.by_seq for s in seqs):
                self.err(f"{eid}: raw_ref.seq must list existing raw seqs")
                continue
            used.update(seqs)
            self.event_seqs[eid] = seqs
            if e["order"] != seqs[0] or seqs[0] != min(seqs):
                self.err(f"{eid}: order must be the first raw seq")
            if e["raw_ref"].get("start_seq") != min(seqs) or e["raw_ref"].get("end_seq") != max(seqs):
                self.err(f"{eid}: start_seq/end_seq must be the minimum and maximum raw seq")
            if _unresolved(e.get("origin")):
                if self._unresolved_event(eid, e, seqs):
                    self._hashes(eid, e, seqs)
                continue
            origin = self.entities.get(e.get("origin")) if isinstance(e.get("origin"), str) else None
            if origin is None:
                self.err(f"{eid}: origin is not an entity")
                continue
            recs = [self.by_seq[s] for s in seqs]
            types = [r.get("type") for r in recs]
            if origin["kind"] == "agent":
                aid = self._agent_of(origin["id"])
                lc = recs[0].get("lc_run_id")
                ok = (
                    types == ["llm_start", "llm_end"]
                    and lc is not None
                    and all(r.get("lc_run_id") == lc and r.get("agent_id") == aid for r in recs)
                    and lc_count[("llm_start", lc)] == 1
                    and lc_count[("llm_end", lc)] == 1
                )
                if not ok:
                    self.err(f"{eid}: not one llm_start/llm_end pair of its origin agent")
                elif not self._origin_established().get(seqs[0]):
                    self.err(f"{eid}: agent origin is not established from the raw record")
                else:
                    self._hashes(eid, e, seqs)
                basis = e["raw_ref"].get("origin_basis") or {}
                rule = "offered_tools_present" if "offered_tools" in recs[0] else "agent_loop_signature"
                if not isinstance(basis, dict) or basis.get("label") != "[INF]" or basis.get("rule") != rule:
                    self.err(f"{eid}: origin_basis must be an [INF] record of rule {rule}")
            elif origin["kind"] == "tool":
                tcid = recs[0].get("tool_call_id")
                commits = [s for s, r in self.by_seq.items() if r.get("type") == "tool_result_committed" and r.get("tool_call_id") == tcid]
                moved = len(commits) == 1 and commits[0] in evict_commits and types[2:] == []
                ok = (
                    types[:2] == ["tool_start", "tool_end"]
                    and types[2:] in ([], ["tool_result_committed"])
                    and tcid is not None
                    and all(r.get("tool_call_id") == tcid and r.get("tool") == origin["raw_ref"].get("name") for r in recs)
                    and id_count[("tool_start", tcid)] == 1
                    and id_count[("tool_end", tcid)] == 1
                    and id_count[("tool_result_committed", tcid)] <= 1
                    and (len(seqs) == 2 + id_count[("tool_result_committed", tcid)] or moved)
                    and e["raw_ref"].get("tool_call_id") == tcid
                )
                if not ok:
                    self.err(f"{eid}: not the start/end(/commit) of one tool_call_id for its origin tool")
                elif tcid in self.evictions and recs[1]["seq"] >= self.evictions[tcid]["raw_ref"]["seq"][0]:
                    self.err(f"{eid}: tool_end must precede the eviction's file_write")
            elif origin["kind"] == "framework":
                pass  # checked in _eviction_event
            else:
                self.err(f"{eid}: only agent-, tool- and framework-eviction events are emitted")
        for s, n in used.items():
            if n > 1:
                self.err(f"raw seq {s} belongs to more than one event")

    def _eviction_event(self, eid: str, e: dict, ent: dict) -> None:
        """Recompute every eviction check from raw records (adapter rule eviction_event)."""
        ref = e.get("raw_ref") or {}
        seqs = ref.get("seq")
        if not isinstance(seqs, list) or len(seqs) != 2 or any(s not in self.by_seq for s in seqs):
            self.err(f"{eid}: eviction raw_ref.seq must be [file_write, tool_result_committed]")
            return
        w, c = self.by_seq[seqs[0]], self.by_seq[seqs[1]]
        tcid = ref.get("tool_call_id")
        version = (self.rec.get("versions") or {}).get("deepagents")
        ends = [r for r in self.by_seq.values() if r.get("type") == "tool_end" and r.get("tool_call_id") == tcid]
        starts = [r for r in self.by_seq.values() if r.get("type") == "tool_start" and r.get("tool_call_id") == tcid]
        writes = [r for r in self.by_seq.values() if r.get("type") == "file_write" and r.get("tool_call_id") == tcid]
        path = w.get("path")
        same_path = [r for r in self.by_seq.values() if r.get("type") == "file_write" and r.get("path") == path]
        content = c.get("content")
        problems = []
        if ent.get("id") != "framework:deepagents" or version not in EVICTION_VERSIONS:
            problems.append("framework or version")
        if w.get("type") != "file_write" or c.get("type") != "tool_result_committed" or tcid is None:
            problems.append("record types")
        if c.get("tool_call_id") != tcid or c.get("differs_from_raw") is not True or len(writes) != 1 or len(ends) != 1 or len(starts) != 1:
            problems.append("joins")
        if w.get("attribution") != "tool_call" or w.get("error") is not None or not isinstance(path, str) \
                or w.get("result_path") != path or len(same_path) != 1:
            problems.append("file_write")
        elif path.split("/")[-1] != tcid.replace(".", "_").replace("/", "_").replace("\\", "_"):
            problems.append("file name")
        if not problems:
            end = ends[0]
            agent = starts[0].get("agent_id")
            text = _text_blocks(end.get("output"))
            head = EVICTION_HEAD.format(tool_call_id=tcid, file_path=path)
            if not end["seq"] < w["seq"] < c["seq"]:
                problems.append("seq order")
            if w.get("agent_id") != agent or c.get("agent_id") != agent:
                problems.append("agent_id")
            if text is None or hashlib.sha256(text.encode("utf-8")).hexdigest() != w.get("sha256") or len(text) != w.get("size"):
                problems.append("file digest")
            if not isinstance(content, str) or digest(content) != c.get("sha256") or not content.startswith(head) \
                    or not content.endswith("\n") or len(content) <= len(head) + 1:
                problems.append("template")
        basis = ref.get("origin_basis") or {}
        if basis.get("label") != "[INF]" or basis.get("rule") != "eviction_event":
            problems.append("origin_basis")
        if ref.get("file_path") != path:
            problems.append("file_path provenance")
        if problems:
            self.err(f"{eid}: eviction event not established from the raw record ({', '.join(problems)})")
            return
        if tcid in self.evictions:
            self.err(f"{eid}: more than one eviction event for one tool_call_id")
        self.evictions[tcid] = e

    def _origin_established(self) -> dict:
        """llm_start seq -> whether an agent origin is established, recomputed from raw records only."""
        if hasattr(self, "_origin_cache"):
            return self._origin_cache
        recs = sorted(self.by_seq.values(), key=lambda r: r["seq"])
        starts = [r for r in recs if r.get("type") == "llm_start"]
        end_of: dict = {}
        n_start, n_end, n_parent = Counter(), Counter(), Counter()
        for r in recs:
            if r.get("type") == "llm_start":
                n_start[r.get("lc_run_id")] += 1
                n_parent[r.get("lc_parent_run_id")] += 1
            elif r.get("type") == "llm_end":
                n_end[r.get("lc_run_id")] += 1
                end_of[r.get("lc_run_id")] = r
        known = {a.get("agent_id") for a in self.rec.get("agents", [])} - {None}
        out: dict = {}
        last: dict = {}  # agent_id -> previous llm_start
        first: dict = {}  # agent_id -> first llm_start
        for s in starts:
            aid, lc = s.get("agent_id"), s.get("lc_run_id")
            prev = last.get(aid)
            first.setdefault(aid, s)
            last[aid] = s
            end = end_of.get(lc)
            if (lc is None or n_start[lc] != 1 or n_end[lc] != 1 or end["seq"] < s["seq"]
                    or end.get("agent_id") != aid or aid not in known):
                out[s["seq"]] = False
                continue
            if "offered_tools" in s:
                out[s["seq"]] = s["offered_tools"] is not None
                continue
            out[s["seq"]] = self._signature_holds(s, end, prev, first[aid], out, end_of, n_parent)
        self._origin_cache = out
        return out

    @staticmethod
    def _signature_holds(s, end, prev, first, out, end_of, n_parent) -> bool:
        def msgs(r):
            b = r.get("input_batches")
            return b[0] if isinstance(b, list) and len(b) == 1 and isinstance(b[0], list) else None

        cur = msgs(s)
        if not cur:
            return False
        par = s.get("lc_parent_run_id")
        if par is None or n_parent[par] != 1 or end.get("lc_parent_run_id") != par:
            return False
        ref = msgs(first) or []
        if not isinstance(cur[0], dict) or cur[0].get("role") != "system" or not ref or _canon(ref[0]) != _canon(cur[0]):
            return False
        if prev is None:
            rest = cur[1:]
            return bool(rest) and all(isinstance(m, dict) and m.get("role") == "human" for m in rest)
        if not out.get(prev["seq"]):
            return False
        before = msgs(prev) or []
        k = len(before)
        if len(cur) <= k or _canon(cur[:k]) != _canon(before):
            return False
        produced = (end_of.get(prev.get("lc_run_id")) or {}).get("message") or {}
        nxt = cur[k]
        if not isinstance(nxt, dict) or nxt.get("role") != "ai":
            return False
        if any(_canon(nxt.get(f)) != _canon(produced.get(f)) for f in ("role", "content", "tool_calls")):
            return False
        expected = [c.get("id") for c in produced.get("tool_calls") or []]
        replies = [m.get("tool_call_id") if isinstance(m, dict) and m.get("role") == "tool" else None for m in cur[k + 1:]]
        return bool(expected) and replies == expected

    def _paired_seqs(self) -> dict:
        """llm_start seq -> llm_end, for calls paired one-to-one by lc_run_id with the end after the start."""
        n = Counter((r.get("type"), r.get("lc_run_id")) for r in self.by_seq.values())
        ends = {r.get("lc_run_id"): r for r in self.by_seq.values() if r.get("type") == "llm_end"}
        out = {}
        for r in self.by_seq.values():
            lc = r.get("lc_run_id")
            if r.get("type") == "llm_start" and lc is not None and n[("llm_start", lc)] == 1 and n[("llm_end", lc)] == 1:
                if ends[lc]["seq"] > r["seq"]:
                    out[r["seq"]] = ends[lc]
        return out

    def _unresolved_seqs(self) -> set:
        """Paired calls whose agent origin is not established: these must be unresolved-origin events."""
        est = self._origin_established()
        return {s for s in self._paired_seqs() if not est.get(s)}

    def _expected_rule(self, start: dict, end: dict) -> str:
        known = {a.get("agent_id") for a in self.rec.get("agents", [])} - {None}
        if start.get("agent_id") not in known or end.get("agent_id") != start.get("agent_id"):
            return "agent_id_unattributed"
        if "offered_tools" in start:
            return "offered_tools_null"
        if "input_batches" not in start:
            return "no_rendered_input"
        return "agent_loop_signature_failed"

    def _unresolved_event(self, eid: str, e: dict, seqs: list) -> bool:
        self.unresolved_event_starts.add(seqs[0])
        start = self.by_seq[seqs[0]]
        end = self._paired_seqs().get(seqs[0])
        if seqs[0] not in self._unresolved_seqs() or end is None or seqs != [start["seq"], end["seq"]]:
            self.err(f"{eid}: unresolved origin on a call whose agent origin is established, or not a paired call")
            return False
        raw_ids = {start.get("agent_id"), end.get("agent_id")} - {None}
        expected = {x for x, ent in self.entities.items() if ent.get("kind") == "agent" and self._agent_of(x) in raw_ids}
        expected |= {x for x, ent in self.entities.items() if ent.get("kind") == "framework"}
        expected.add("$absent")
        cands = e["origin"].get("candidates") or []
        basis = e["raw_ref"].get("origin_basis") or {}
        ok = (
            set(cands) == expected
            and len(cands) == len(set(cands))
            and f"semantic.events[{eid}].origin" in self.unresolved_paths
            and e["raw_ref"].get("input_telemetry") == ("present" if "input_batches" in start else "absent")
            and basis.get("label") == "UNRESOLVED"
            and basis.get("rule") == self._expected_rule(start, end)
        )
        if not ok:
            self.err(f"{eid}: unresolved origin, candidates or rule do not match the raw record")
        return ok

    def _hashes(self, eid: str, e: dict, seqs: list) -> None:
        start, end = self.by_seq[seqs[0]], self.by_seq[seqs[-1]]
        ref = e["raw_ref"]
        if ref.get("output_sha256") != digest(end.get("message")):
            self.err(f"{eid}: output_sha256 does not reproduce")
        if "input_batches" in start and ref.get("input_sha256") != digest(start["input_batches"]):
            self.err(f"{eid}: input_sha256 does not reproduce")

    def _delegation_ids(self) -> set:
        ids = {a.get("delegation_id") for a in self.rec.get("agents", [])}
        ids |= {d.get("delegation_id") for d in self.rec.get("delegations") or []}
        return ids - {None}

    # --- evidence ----------------------------------------------------------------

    def _evidence(self) -> None:
        self.values: dict[str, Any] = {}
        self.question_ids: set[str] = set()
        self.substitutes: dict[str, str] = {}  # evidence id -> tool_call_id
        for xid, x in self.evidence.items():
            ref = x.get("raw_ref") or {}
            if "produced_by" not in x:
                self._external(xid, x, ref)
                continue
            prod = self.events.get(x.get("produced_by"))
            if prod is None:
                self.err(f"{xid}: produced_by is not an event")
                continue
            try:
                value = resolve(self.rec, ref.get("locator"))
            except LocatorError as exc:
                self.err(f"{xid}: {exc}")
                continue
            self.values[xid] = value
            if digest(value) != ref.get("sha256") or size(value) != ref.get("size"):
                self.err(f"{xid}: sha256 or size does not reproduce")
            seq = ref.get("seq")
            if seq not in self.event_seqs.get(prod["id"], []) or not ref["locator"].startswith(f"events[seq={seq}]."):
                self.err(f"{xid}: locator is outside its producer's records")
                continue
            if _unresolved(prod.get("origin")):
                kind = "unresolved"
            else:
                kind = (self.entities.get(prod.get("origin")) or {}).get("kind")
            raw = self.by_seq[seq]
            if kind == "framework":
                tcid = prod["raw_ref"].get("tool_call_id")
                src = x.get("source")
                ok = (
                    self.evictions.get(tcid) is prod
                    and raw.get("type") == "tool_result_committed"
                    and seq == prod["raw_ref"]["seq"][-1]
                    and ref["locator"] == f"events[seq={seq}].content"
                    and ref.get("tool_call_id") == tcid
                    and _unresolved(src)
                    and src.get("candidates") == ["framework", "tool"]
                    and f"semantic.evidence[{xid}].source" in self.unresolved_paths
                )
                if ok:
                    if tcid in self.substitutes.values():
                        self.err(f"{xid}: more than one substitute for one eviction")
                    self.substitutes[xid] = tcid
            elif kind == "tool":
                if raw.get("tool_call_id") in self._delegation_ids():
                    self.err(f"{xid}: result of a delegating tool call is not emitted in increment 1 (0.3 §4.4)")
                ok = (
                    x.get("source") == "tool"
                    and raw.get("type") == "tool_end"
                    and ref["locator"] == f"events[seq={seq}].output"
                    and ref.get("tool_call_id") == raw.get("tool_call_id") == prod["raw_ref"].get("tool_call_id")
                )
            elif kind in ("agent", "unresolved"):
                m = _PRODUCED_ARG.match(ref["locator"])
                src = x.get("source")
                if kind == "agent":
                    src_ok = src == "agent"
                else:
                    src_ok = (
                        _unresolved(src)
                        and src.get("candidates") == ["agent", "framework"]
                        and f"semantic.evidence[{xid}].source" in self.unresolved_paths
                    )
                ok = bool(m) and src_ok and raw.get("type") == "llm_end"
                if ok:
                    tc = raw["message"]["tool_calls"][int(m.group(2))]
                    ok = tc.get("id") == ref.get("tool_call_id") and m.group(3) == ref.get("argument")
            else:
                ok = False
            if not ok:
                self.err(f"{xid}: source, locator or tool_call_id inconsistent with its producer")

    def _external(self, xid: str, x: dict, ref: dict) -> None:
        """The only external item emitted is the record's question."""
        q = self.rec.get("question")
        src = x.get("source")
        ok = (
            q is not None
            and ref.get("external") is True
            and ref.get("locator") == "question"
            and "seq" not in ref
            and digest(q) == ref.get("sha256")
            and size(q) == ref.get("size")
            and _unresolved(src)
            and src.get("candidates") == ["user", "benchmark"]
            and f"semantic.evidence[{xid}].source" in self.unresolved_paths
            and not self.question_ids
        )
        if not ok:
            self.err(f"{xid}: external evidence is not the record's question with an unresolved user/benchmark source")
            return
        self.question_ids.add(xid)
        self.values[xid] = q

    # --- links ---------------------------------------------------------------------

    def _links(self) -> None:
        seen = set()
        for i, l in enumerate(self.sem["links"]):
            tag = f"links[{i}]"
            if l.get("relation") != "delivers":
                self.err(f"{tag}: only delivers is emitted in increment 1")
                continue
            src, tgt = self.events.get(l.get("source")), self.evidence.get(l.get("target"))
            if src is None or tgt is None:
                self.err(f"{tag}: source must be an event and target an evidence item")
                continue
            if (src["id"], tgt["id"]) in seen:
                self.err(f"{tag}: duplicate link")
            seen.add((src["id"], tgt["id"]))
            o = src.get("origin")
            origin = (self.entities.get(o) if isinstance(o, str) else None) or {}
            if origin.get("kind") != "agent":
                self.err(f"{tag}: delivers must start at an agent-input event (0.3 §5.1)")
                continue
            self._link(tag, l, src, tgt, origin)

    def _link(self, tag: str, l: dict, src: dict, tgt: dict, origin: dict) -> None:
        loc = (l.get("raw_ref") or {}).get("locator") or ""
        m = _MSG.match(loc)
        start = self.by_seq.get(self.event_seqs.get(src["id"], [None])[0])
        if not m or start is None or int(m.group(1)) != start.get("seq"):
            self.err(f"{tag}: locator is not inside the source event's rendered input")
            return
        batches = start.get("input_batches")
        if not isinstance(batches, list) or len(batches) != 1:
            self.err(f"{tag}: source input is not one rendered batch")
            return
        try:
            value = resolve(self.rec, loc)
        except LocatorError as exc:
            self.err(f"{tag}: {exc}")
            return
        expected = self.values.get(tgt["id"])
        if digest(value) != tgt["raw_ref"].get("sha256") or _canon(value) != _canon(expected):
            self.err(f"{tag}: rendered value is not exactly equal to the target")
            return
        batch = batches[0]
        idx, rest = int(m.group(2)), m.group(3)
        msg = batch[idx]
        tcid = tgt["raw_ref"].get("tool_call_id")
        role = msg.get("role")
        if role == "tool":
            same = [x for x in batch if isinstance(x, dict) and x.get("role") == "tool" and x.get("tool_call_id") == tcid]
            kind_ok = tgt.get("source") == "tool" or self.substitutes.get(tgt["id"]) == tcid
            ok = rest == ".content" and kind_ok and msg.get("tool_call_id") == tcid and len(same) == 1
        elif role == "ai":
            a = _ARG.match(rest)
            entries = [tc for x in batch if isinstance(x, dict) and x.get("role") == "ai" for tc in x.get("tool_calls") or [] if tc.get("id") == tcid]
            ok = (
                bool(a)
                and tgt.get("source") == "agent"
                and msg["tool_calls"][int(a.group(1))].get("id") == tcid
                and a.group(2) == tgt["raw_ref"].get("argument")
                and len(entries) == 1
            )
        elif role == "human":
            aid = self._agent_of(origin["id"])
            agent = next((a for a in self.rec.get("agents", []) if a.get("agent_id") == aid), {})
            equal = [x for x in batch if isinstance(x, dict) and x.get("role") == "human" and _canon(x.get("content")) == _canon(expected)]
            if tgt["id"] in self.question_ids:
                ok = (
                    rest == ".content"
                    and agent.get("parent_agent_id") is None
                    and agent.get("delegation_id") is None
                    and len(equal) == 1
                )
            else:
                ok = (
                    rest == ".content"
                    and tgt.get("source") == "agent"
                    and agent.get("delegation_id") is not None
                    and agent.get("delegation_id") == tcid
                    and len(equal) == 1
                )
        else:
            ok = False
        if not ok:
            self.err(f"{tag}: structural join does not hold or is not unique")

    # --- profile_local and accounting --------------------------------------------

    def _profile_local(self) -> None:
        pl = self.doc.get("profile_local") or {}
        for key in NEGATIVE_GATES:
            if key in pl:
                self.err(f"profile_local.{key} must be absent: it would enable negative inferences")
        ad = pl.get("adapter")
        if not isinstance(ad, dict):
            self.err("profile_local.adapter is missing")
            self.adapter = {"skipped": []}
            return
        self.adapter = ad
        if ad.get("claims_extracted") is not False:
            self.err("claims_extracted must be false")
        ambs = pl.get("ambiguities") or []
        amb_ids = [a.get("id") for a in ambs]
        if ad.get("unresolved") != amb_ids:
            self.err("profile_local.adapter.unresolved must list the ambiguity ids")
        if sorted(a.get("field_path") for a in ambs) != sorted(self.unresolved_paths):
            self.err("ambiguities do not match the $unresolved fields one-to-one")
        for a in ambs:
            v = self.unresolved_paths.get(a.get("field_path"))
            if v is None or v.get("$unresolved") != a.get("id") or v.get("candidates") != a.get("candidates"):
                self.err(f"ambiguity {a.get('id')}: id or candidates differ from the field")
        conf = pl.get("conformance") or {}
        if conf.get("canonical_complete") is not (not ambs) or sorted(conf.get("unresolved") or []) != sorted(self.unresolved_paths):
            self.err("conformance does not reflect the $unresolved fields")
        has_input = any("input_batches" in r for r in self.by_seq.values() if r.get("type") == "llm_start")
        if ad.get("delivery_telemetry") != ("present" if has_input else "absent"):
            self.err("delivery_telemetry does not match the record")
        if not has_input and self.sem["links"]:
            self.err("delivers links in a run without rendered inputs")
        if not (ad.get("order") or {}).get("caveat"):
            self.err("the order caveat is missing")
        order = pl.get("order") or {}
        spans = sorted((min(s), max(s), eid) for eid, s in self.event_seqs.items())
        pairs = [[a, b] for i, (_, hi, a) in enumerate(spans) for (lo2, _, b) in spans[i + 1:] if lo2 <= hi]
        if order.get("basis") != "first_raw_seq" or not order.get("caveat"):
            self.err("profile_local.order must have basis first_raw_seq and a caveat")
        if sorted(map(tuple, order.get("overlaps") or [])) != sorted(map(tuple, pairs)):
            self.err("profile_local.order.overlaps does not list exactly the overlapping event pairs")

    def _accounting(self) -> None:
        skipped = self.adapter.get("skipped") or []
        rec_skipped = Counter(s for e in skipped if e.get("scope") == "record" for s in e.get("seq", []))
        in_events = Counter(s for seqs in self.event_seqs.values() for s in seqs)
        for s in self.by_seq:
            n = in_events[s] + rec_skipped[s]
            if n != 1:
                self.err(f"raw seq {s} is accounted for {n} times (events + skipped records)")
        for s in rec_skipped:
            if s not in self.by_seq:
                self.err(f"skipped record seq {s} does not exist")

        by_kind: dict[str, list[dict]] = defaultdict(list)
        for e in skipped:
            by_kind[e.get("kind")].append(e)
        for s in sorted(self._unresolved_seqs() - self.unresolved_event_starts):
            self.err(f"llm_start {s}: paired call with unestablished agent origin is not preserved as an event")
        evicted = {s for e in by_kind["eviction_substitute"] for s in e.get("seq", [])}
        represented = {ev["raw_ref"]["seq"][-1] for ev in self.evictions.values()}
        for s, r in self.by_seq.items():
            if r.get("type") == "tool_result_committed" and r.get("differs_from_raw") is True:
                if (s in evicted) == (s in represented):
                    self.err(f"eviction at seq {s} must be either an eviction event or a skipped item, exactly once")
        q = self.rec.get("question")
        if q is not None and not self.question_ids:
            self.err("the question is not emitted as external evidence")
        prompts = {(e.get("agent_id"), e.get("sha256")) for e in by_kind["system_prompt"]}
        for eid, seqs in self.event_seqs.items():
            start = self.by_seq[seqs[0]]
            if start.get("type") != "llm_start":
                continue
            for msg in (start.get("input_batches") or [[]])[0]:
                if isinstance(msg, dict) and msg.get("role") == "system":
                    if (start.get("agent_id"), digest(msg.get("content"))) not in prompts:
                        self.err(f"{eid}: system prompt in its input is not surfaced")


def verify(doc: dict, record: dict, *, trace_sha256: str | None = None) -> list[str]:
    """Return the list of verification errors (empty if the output passes)."""
    return _Verifier(doc, record, trace_sha256).run()
