"""Deterministic structural adapter: instrumented DeepAgents record -> agenttrace.semantic/0.3.

Increments 1 and 2 of automated semantic extraction.  No model, network or database is
used, and the raw record is never modified.  The output is::

    {"profile": ADAPTER, "semantic": {<the eight 0.3 collections>},
     "profile_local": {"adapter": {...}}}

What is emitted (0.3 §4, §5.1):

  * entities: one ``agent`` per ``agents[]`` entry, one ``tool`` per tool name of
    an emitted tool event, and the ``framework`` named in ``versions``;
  * events: one model call per ``llm_start``/``llm_end`` pair (joined by
    ``lc_run_id``), origin the calling agent; one tool call per
    ``tool_start``/``tool_end``(/``tool_result_committed``) set (joined by
    ``tool_call_id``), origin the tool; one framework eviction per
    ``file_write``/``tool_result_committed`` pair that passes every
    ``eviction_event`` check, origin the framework (increment 2);
  * evidence: each tool result (``tool_end.output``, source ``tool``), each
    top-level tool-call argument value of a model output
    (``llm_end.message.tool_calls[j].args.<key>``, source ``agent``), each
    eviction substitute (the committed tool message) and the record's question
    (external, no producer) (increment 2);
  * ``delivers(E, X)`` only where the rendered input of model call E holds an
    occurrence of X found by a structural join plus exact equality:

      - a tool message with X's ``tool_call_id`` whose content equals X (a tool
        result, or the substitute that replaced it);
      - an AI message whose ``tool_calls`` entry has X's ``tool_call_id`` and an
        equal value under X's argument key;
      - in the input of the agent whose ``delegation_id`` is X's
        ``tool_call_id``, a human message whose content equals X;
      - in the input of a root agent (no parent, no delegation), exactly one
        human message equal to the question.

    A join that matches more than once, or not exactly, gives no link and is
    surfaced in ``profile_local.adapter.skipped``.

Never emitted: claims, propositions, states, links other than ``delivers``,
non-delivery, hierarchy, file lineage, causal or precedence relations, the
evicted file as evidence (its path is provenance only), and the system prompt
(recorded as a skipped item).

``order`` is the seq of an event's first raw record: observation order only
(decision E1, viz-profile D1 rev 3).  Each event's ``raw_ref`` carries
``start_seq``/``end_seq``; ``profile_local.order`` lists every overlapping pair.
It is profile-local and leaves 0.3 unchanged.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from agent.deep_agent._semantic_locator import DIGEST_BASIS, digest, is_name, size

ADAPTER = "agenttrace.adapter.deepagents/0.1"
INCREMENT = 2
RAW_SCHEMA = "assetopsbench.deepagent.instrumented/0.1"
SEMANTIC_SCHEMA = "agenttrace.semantic/0.3"

ORDER_BASIS = "first_raw_seq"
ORDER_RULE = (
    "order = the seq of the event's first raw record (llm_start, tool_start or file_write); the seqs of all the "
    "event's records are in raw_ref.seq, and raw_ref.start_seq/end_seq are their minimum and maximum. This is "
    "the order in which events were first observed by the capture."
)
ORDER_CAVEAT = (
    "Observation order only. X observably precedes Y only if end_seq(X) < start_seq(Y). Events whose spans "
    "intersect (parallel tool calls issued by one model output, a delegation tool call spanning its child's "
    "events) are overlapping observations, listed in profile_local.order.overlaps; overlap is not evidence of "
    "simultaneous execution. Between overlapping events, order supports no precedence, causation or "
    "reiteration relation, and no inference may rely on it as completion or dependency order (0.3 §4.3, §12: "
    "partial order is DEFERRED; viz-profile D1 rev 3)."
)

# Framework eviction (deepagents FilesystemMiddleware), verified in the installed source for these versions
# only: wrap_tool_call -> _intercept_large_tool_result -> _process_large_message writes the joined text blocks
# of the tool result to a file named by the sanitized tool_call_id, then replaces the message content with
# TOO_LARGE_TOOL_MSG. A failed write keeps the original message. Any other version fails closed.
EVICTION_VERIFIED_VERSIONS = ("0.5.3",)
EVICTION_CODE = ("deepagents/middleware/filesystem.py: FilesystemMiddleware.wrap_tool_call -> "
                 "_intercept_large_tool_result -> _process_large_message (backend write, then TOO_LARGE_TOOL_MSG)")
EVICTION_TEMPLATE_HEAD = (
    "Tool result too large, the result of this tool call {tool_call_id} was saved in the filesystem at this "
    "path: {file_path}\n\nYou can read the result from the filesystem by using the read_file tool, but make "
    "sure to only read part of the result at a time.\n\nYou can do this by specifying an offset and limit in "
    "the read_file tool call. For example, to read the first 100 lines, you can use the read_file tool with "
    "offset=0 and limit=100.\n\nHere is a preview showing the head and tail of the result (lines of the form "
    "`... [N lines truncated] ...` indicate omitted lines in the middle of the content):\n\n"
)
EVICTION_NOTE = (
    "One framework procedure: that file_write and tool_result_committed belong to one call of the eviction "
    "code is an inference from that code plus the raw joins, not an observation. tool_result_committed is "
    "emitted by the capture when the tools node finishes, so adjacency is not required. The file path is "
    "provenance only: the evicted file is not an evidence item, and no lineage from the raw result to the "
    "substitute or the file is stored (0.3 §5.3, §12). input_locator and file_sha256 are integrity provenance: "
    "that the file digest equals the raw result's joined text shows content equality at most (exposure, 0.3 "
    "§10.1), never lineage or derivation."
)

ORIGIN_NOTE = (
    "Inference, not observation: agent_id is assigned by the capture's walk of the LangChain run tree, and "
    "no check here proves that the agent, rather than framework middleware inside it, initiated the call."
)
LINK_STATUS = (
    "The occurrence of the target in the rendered input is observed; that the source is an agent-input event is "
    "an [INF] resting on the source event's raw_ref.origin_basis."
)
CHAIN_NOTE = (
    "previous_* locators are inputs to the continuity check only; they state no iteration, step or loop order "
    "(0.3 §12 DEFERRED)."
)
CONTINUITY_KEYS = ("role", "content", "tool_calls")  # other message metadata (e.g. name) is not compared

RULES = {
    "agent_entity": "One agent per agents[] entry; role = agents[].kind, model = the run-level model field. "
    "Parent and delegation ids stay in raw_ref (hierarchy is DEFERRED).",
    "tool_entity": "One tool entity per tool name of an emitted tool event, framework built-ins included.",
    "framework_entity": "One framework entity per versions entry naming the agent framework.",
    "model_event": "llm_start and llm_end joined by lc_run_id, exactly one of each, end after start. Origin is "
    "the agent named by both records' agent_id, as an inference, when offered_tools is present and non-null, or "
    "when offered_tools is absent and agent_loop_signature holds. Every other paired call follows "
    "unresolved_model_event. A call with no llm_end has no observed output and is skipped (0.3 §4.3); a call "
    "with an output but no captured input is kept, because its input existed but was not logged (a coverage "
    "matter, §11 r4).",
    "unresolved_model_event": "A paired call whose agent origin is not established is kept as an event with "
    "origin $unresolved: no rendered input and no offered_tools (no_rendered_input), offered_tools null "
    "(offered_tools_null), a failed agent_loop_signature (agent_loop_signature_failed), or an agent_id missing, "
    "unknown or different at start and end (agent_id_unattributed). Candidates: the agents named by the raw "
    "agent_id values that exist, the framework entity, and $absent. The raw agent_id stays in raw_ref; input and "
    "output keep their locators and digests. It is not an agent-input event, so it sources no delivers. Its "
    "argument evidence is kept with source $unresolved (agent, framework): an extractor choice, since 0.3 does "
    "not fix the author class of model output in a framework-initiated call. Order is its llm_start seq. "
    "input_sha256/output_sha256 are integrity provenance only: comparing them across events establishes no "
    "input equivalence, delivery or lineage.",
    "agent_loop_signature": "Used only when llm_start has no offered_tools. All must hold: (1) the call's "
    "lc_parent_run_id is shared by no other llm_start and equals the llm_end's; (2) message 0 is a system "
    "message equal to message 0 of the agent's first model call; (3) continuity: the agent's first call has "
    "only human messages after the system message; any later call's previous model call (by seq, same agent) "
    "was itself established, and the input is that call's input unchanged, then its output (role, content and "
    "tool_calls equal), then exactly one tool message per output tool call, in order. Post-summary calls fail "
    "(3), and so does every later call of that agent: they fail closed.",
    "tool_event": "tool_start, tool_end and at most one tool_result_committed joined by tool_call_id, exactly "
    "one start and one end, equal lc_run_id, tool and agent_id; origin is the tool.",
    "tool_result_evidence": "tool_end.output, source tool, produced_by the tool event. Not emitted for a "
    "delegating tool call (its tool_call_id is a delegation_id): its content may originate in the child.",
    "argument_evidence": "Each top-level argument value of each llm_end tool call, source agent, produced_by "
    "the model call.",
    "eviction_event": "Kept only when every check holds: the framework version is one whose eviction code was "
    "verified; one tool event of the tool_call_id with a tool_result_committed whose differs_from_raw is true, "
    "string content and a reproducing sha256; exactly one file_write with that tool_call_id (attribution "
    "tool_call, no error, path equal to result_path, file name equal to the sanitized tool_call_id, the only "
    "write to that path) and the same agent_id; tool_end seq < file_write seq < committed seq; the raw result is "
    "text only and its joined text blocks reproduce the file's sha256 and size; the committed content is the "
    "eviction template for that tool_call_id and path followed by a non-empty preview. Origin is the framework "
    "entity ([INF], see origin_basis). raw_ref.seq is [file_write, tool_result_committed]; the committed record "
    "leaves the tool event. Any failed check: no event and no substitute, recorded as an eviction_substitute "
    "skip, and the records stay where increment 1 put them.",
    "substitute_evidence": "The committed tool message content of a kept eviction, produced_by the eviction "
    "event, source $unresolved (framework, tool): the template is framework text and the preview is tool "
    "content, an extractor choice.",
    "question_evidence": "The record's question field, external evidence (no produced_by, raw_ref.external), "
    "source $unresolved (user, benchmark): the record does not say which authored it.",
    "delivers": "Positive only. Structural join (tool_call_id, delegation_id for the child's human messages, "
    "or a root agent's human message for the question) plus exact equality: canonical JSON equal and equal "
    "capture digest. More than one candidate occurrence, or an inexact one, gives no link. An exact question "
    "match in a non-root agent's input gives no link: delivery to subagents is not inferred.",
    "digest": DIGEST_BASIS,
}

NOT_EMITTED = [
    "claims, propositions and states (claims_extracted is false)",
    "links other than delivers: propagates_to, checks, depends_on, supported_by, contradicted_by",
    "non-delivery of any kind (0.3 §5.1, §11 r4); input coverage is not established",
    "parent/child hierarchy (DEFERRED)",
    "file lineage: file_read and other file_write records are not events; an eviction's file_write is part of "
    "its eviction event, and the evicted file is not an evidence item",
    "the system prompt (no record field holds it)",
    "model text parts and thinking blocks",
    "precedence or causal order between events",
    "$unresolved values, except the origin of a model call whose agent origin is not established, the "
    "source of its argument evidence, the source of eviction substitutes and the source of the question",
]

_LABEL_TYPES = {"agent_start", "agent_end", "agent_error", "delegation_start", "delegation_end", "delegation_error"}
_FILE_TYPES = {"file_write", "file_read"}


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _same(a: Any, b: Any) -> bool:
    return _canon(a) == _canon(b) and digest(a) == digest(b)


class _Adapter:
    def __init__(self, record: dict, trace_ref: str, trace_sha256: str):
        self.rec = record
        self.trace_ref = trace_ref
        self.trace_sha256 = trace_sha256
        self.skipped: list[dict] = []
        self.entities: dict[str, dict] = {}
        self.events: dict[str, dict] = {}
        self.evidence: dict[str, dict] = {}
        self.links: list[dict] = []
        self.model_end: dict[str, dict] = {}  # event id -> llm_end record
        self.model_start: dict[str, dict] = {}  # event id -> llm_start record
        self.tool_items: dict[str, dict] = {}  # tool_call_id -> tool-result evidence
        self.arg_items: dict[str, list[dict]] = defaultdict(list)  # tool_call_id -> argument evidence
        self.issued: dict[str, str] = {}  # tool_call_id -> raw locator of the issuing tool_calls entry
        self.by_seq: dict[int, dict] = {}
        self.ambiguities: list[dict] = []
        self.unresolved_events: set[str] = set()
        self.tool_event_of: dict[str, str] = {}  # tool_call_id -> tool event id
        self.commits: dict[str, dict] = {}  # tool_call_id -> tool_result_committed of an emitted tool event
        self.sub_items: dict[str, dict] = {}  # tool_call_id -> substitute evidence
        self.eviction_records: set[int] = set()  # file_write seqs inside eviction events
        self.question_item: dict | None = None

    # --- helpers -----------------------------------------------------------------

    def skip(self, scope: str, kind: str, seqs: list[int], reason: str, **extra: Any) -> None:
        self.skipped.append({"scope": scope, "kind": kind, "seq": sorted(seqs), "reason": reason, **extra})

    # --- build -----------------------------------------------------------------------

    def run(self) -> dict:
        rec = self.rec
        if rec.get("schema") != RAW_SCHEMA:
            raise ValueError(f"unsupported raw schema {rec.get('schema')!r}; expected {RAW_SCHEMA!r}")
        records = rec.get("events")
        seqs = [r.get("seq") for r in records]
        if not all(isinstance(s, int) for s in seqs) or len(set(seqs)) != len(seqs):
            raise ValueError("raw record seqs must be unique integers")
        agent_ids = [a.get("agent_id") for a in rec.get("agents", [])]
        if len(set(agent_ids)) != len(agent_ids) or None in agent_ids:
            raise ValueError("agents[].agent_id must be unique and present")
        self.agents = {a["agent_id"]: a for a in rec["agents"]}
        self.by_seq = {r["seq"]: r for r in records}
        self.delegation_ids = {a["delegation_id"] for a in rec["agents"] if a.get("delegation_id") is not None}
        self.delegation_ids |= {d["delegation_id"] for d in rec.get("delegations") or [] if d.get("delegation_id") is not None}

        self._agent_entities()
        self._framework_entity()
        by_type: dict[str, list[dict]] = defaultdict(list)
        for r in sorted(records, key=lambda r: r["seq"]):
            by_type[r.get("type")].append(r)
        self._model_events(by_type["llm_start"], by_type["llm_end"])
        self._tool_events(by_type["tool_start"], by_type["tool_end"], by_type["tool_result_committed"])
        self._argument_evidence()
        self._evictions(by_type["file_write"])
        self._other_records(records)
        self._question()
        self._system_prompts()
        self._delivers()
        return self._output()

    def _agent_entities(self) -> None:
        for aid, a in self.agents.items():
            self.entities[f"agent:{aid}"] = {
                "id": f"agent:{aid}",
                "kind": "agent",
                "role": a.get("kind"),
                "model": self.rec.get("model"),
                "raw_ref": {
                    "locator": f"agents[agent_id={aid}]",
                    "name": a.get("name"),
                    "role_locator": f"agents[agent_id={aid}].kind",
                    "model_locator": "model",
                    "parent_agent_id": a.get("parent_agent_id"),
                    "delegation_id": a.get("delegation_id"),
                    "note": "Run-level model field; the record has no per-agent model. Hierarchy is kept here only.",
                },
            }

    def _framework_entity(self) -> None:
        versions = self.rec.get("versions") or {}
        if "deepagents" in versions:
            self.entities["framework:deepagents"] = {
                "id": "framework:deepagents",
                "kind": "framework",
                "raw_ref": {"locator": "versions.deepagents", "version": versions["deepagents"]},
            }

    def _model_events(self, starts: list[dict], ends: list[dict]) -> None:
        groups: dict[Any, dict[str, list[dict]]] = defaultdict(lambda: {"start": [], "end": []})
        for r in starts:
            groups[r.get("lc_run_id")]["start"].append(r)
        for r in ends:
            groups[r.get("lc_run_id")]["end"].append(r)
        parent_count: dict[Any, int] = defaultdict(int)
        for r in starts:
            parent_count[r.get("lc_parent_run_id")] += 1
        paired: dict[int, dict] = {}  # llm_start seq -> llm_end, for calls paired one-to-one
        for lc, g in groups.items():
            all_seqs = [r["seq"] for r in g["start"] + g["end"]]
            if lc is None or len(g["start"]) != 1 or len(g["end"]) != 1 or g["end"][0]["seq"] < g["start"][0]["seq"]:
                extra: dict[str, Any] = {}
                if len(g["start"]) == 1:
                    extra = {"raw_agent_id": g["start"][0].get("agent_id")}
                    if "input_batches" in g["start"][0]:
                        extra["input_locator"] = f"events[seq={g['start'][0]['seq']}].input_batches[0]"
                self.skip("record", "model_call_unpaired", all_seqs,
                          "llm_start/llm_end not paired one-to-one by lc_run_id (or the end precedes the start); an "
                          "event needs an observed input and output (0.3 §4.3)",
                          lc_run_id=lc, starts=len(g["start"]), ends=len(g["end"]), **extra)
                continue
            paired[g["start"][0]["seq"]] = g["end"][0]

        # Origin, decided per agent in seq order: established ([INF]) or kept with an unresolved origin.
        by_agent: dict[Any, list[dict]] = defaultdict(list)
        for r in starts:
            by_agent[r.get("agent_id")].append(r)
        for aid, calls in by_agent.items():
            accepted: dict[int, bool] = {}
            first = calls[0] if calls else None
            for k, s in enumerate(calls):
                accepted[s["seq"]] = False
                e = paired.get(s["seq"])
                if e is None:
                    continue
                prev = calls[k - 1] if k else None
                if aid is None or aid not in self.agents or e.get("agent_id") != aid:
                    self._emit_unresolved_model_event(
                        s, e, "agent_id_unattributed", {"agent_id_consistent": False},
                        "agent_id is missing, unknown or different between llm_start and llm_end")
                    continue
                if "offered_tools" in s and s["offered_tools"] is None:
                    self._emit_unresolved_model_event(
                        s, e, "offered_tools_null", {"offered_tools_present": True, "offered_tools_non_null": False},
                        "offered_tools is null: the call was not made with the agent's tools, so whether the agent or "
                        "framework middleware inside it initiated the call is not established (0.3 §4.3)")
                    continue
                if "offered_tools" in s:
                    basis = {
                        "label": "[INF]",
                        "rule": "offered_tools_present",
                        "checks": {"offered_tools_non_null": True},
                        "raw_refs": {"offered_tools_locator": f"events[seq={s['seq']}].offered_tools",
                                     "agent_id_locator": f"events[seq={s['seq']}].agent_id"},
                        "note": ORIGIN_NOTE,
                    }
                elif "input_batches" not in s:
                    self._emit_unresolved_model_event(
                        s, e, "no_rendered_input", {"input_batches_present": False, "offered_tools_present": False},
                        "llm_start has neither input_batches nor offered_tools (observed absence), so neither the "
                        "agent_loop_signature nor the offered_tools rule can be applied")
                    continue
                else:
                    basis, reason = self._signature(s, prev, accepted, paired, first, parent_count)
                    if basis is None:
                        self._emit_unresolved_model_event(
                            s, e, "agent_loop_signature_failed", {"offered_tools_present": False, "signature_holds": False},
                            "offered_tools absent and the agent-loop signature does not hold: " + reason)
                        continue
                accepted[s["seq"]] = True
                self._emit_model_event(s, e, basis)

    @staticmethod
    def _input_ref(s: dict) -> dict:
        return {"input_locator": f"events[seq={s['seq']}].input_batches[0]"} if "input_batches" in s else {}

    def _ambiguity(self, amb_id: str, field_path: str, candidates: list, reason: str, raw_ref: dict,
                   basis: str = "OBS") -> dict:
        self.ambiguities.append({"id": amb_id, "field_path": field_path, "candidates": candidates,
                                 "reason": reason, "raw_ref": raw_ref, "basis": basis})
        return {"$unresolved": amb_id, "candidates": candidates}

    def _emit_unresolved_model_event(self, s: dict, e: dict, rule: str, checks: dict, reason: str) -> None:
        """A paired call whose agent origin is not established: kept, origin $unresolved (decisions 2026-10-20)."""
        eid = f"ev:llm:{s['seq']}"
        raw_agents = []
        for a in (s.get("agent_id"), e.get("agent_id")):
            if a in self.agents and f"agent:{a}" not in raw_agents:
                raw_agents.append(f"agent:{a}")
        framework = ["framework:deepagents"] if "framework:deepagents" in self.entities else []
        candidates = raw_agents + framework + ["$absent"]
        origin = self._ambiguity(
            f"amb:origin:{s['seq']}", f"semantic.events[{eid}].origin", candidates,
            f"{rule}: {reason}; the capture's run-tree agent_id is kept as a raw value only",
            {"seq": [s["seq"], e["seq"]], "agent_id_locator": f"events[seq={s['seq']}].agent_id", **self._input_ref(s),
             **({"offered_tools_locator": f"events[seq={s['seq']}].offered_tools"} if "offered_tools" in s else {})},
        )
        raw_ref = {
            "seq": [s["seq"], e["seq"]],
            "lc_run_id": s.get("lc_run_id"),
            "output_locator": f"events[seq={e['seq']}].message",
            "output_sha256": digest(e.get("message")),
            "input_telemetry": "present" if "input_batches" in s else "absent",
            "raw_agent_id": s.get("agent_id"),
            "origin_basis": {
                "label": "UNRESOLVED",
                "rule": rule,
                "checks": checks,
                "raw_refs": {"agent_id_locator": f"events[seq={s['seq']}].agent_id",
                             **({"offered_tools_locator": f"events[seq={s['seq']}].offered_tools"}
                                if "offered_tools" in s else {})},
                "note": "Not an agent-input event: no delivers, claims, states, direct links or uptake may be "
                        "derived from it.",
            },
        }
        if e.get("agent_id") != s.get("agent_id"):
            raw_ref["raw_end_agent_id"] = e.get("agent_id")
        raw_ref.update(self._input_ref(s))
        if "input_batches" in s:
            raw_ref["input_sha256"] = digest(s["input_batches"])
        self.events[eid] = {"id": eid, "origin": origin, "order": s["seq"], "raw_ref": raw_ref}
        self.unresolved_events.add(eid)
        self.model_start[eid] = s
        self.model_end[eid] = e

    def _emit_model_event(self, s: dict, e: dict, basis: dict) -> None:
        eid = f"ev:llm:{s['seq']}"
        raw_ref = {"seq": [s["seq"], e["seq"]], "lc_run_id": s.get("lc_run_id"),
                   "output_locator": f"events[seq={e['seq']}].message", "origin_basis": basis,
                   "output_sha256": digest(e.get("message")),
                   "input_telemetry": "present" if "input_batches" in s else "absent"}
        if "input_batches" in s:
            raw_ref["input_sha256"] = digest(s["input_batches"])
        raw_ref.update(self._input_ref(s))
        self.events[eid] = {"id": eid, "origin": f"agent:{s['agent_id']}", "order": s["seq"], "raw_ref": raw_ref}
        self.model_start[eid] = s
        self.model_end[eid] = e

    def _signature(self, s, prev, accepted, paired, first, parent_count) -> tuple[dict | None, str]:
        """Agent-loop signature for a call without offered_tools (rule ``agent_loop_signature``).

        Returns (origin_basis, "") when every check holds, else (None, reason).
        """
        batches = s.get("input_batches")
        if not isinstance(batches, list) or len(batches) != 1 or not isinstance(batches[0], list) or not batches[0]:
            return None, "no single rendered input to check"
        msgs = batches[0]
        seq = s["seq"]
        refs: dict[str, Any] = {"input_locator": f"events[seq={seq}].input_batches[0]",
                                "system_prompt_locator": f"events[seq={seq}].input_batches[0][0]"}
        failed: list[str] = []
        # Parent-run consistency.
        parent = s.get("lc_parent_run_id")
        end = paired[seq]
        if parent is None or parent_count[parent] != 1 or end.get("lc_parent_run_id") != parent:
            failed.append("parent run missing, shared with another model call, or different at llm_end")
        else:
            refs["lc_parent_run_id"] = parent
        # System prompt.
        head = msgs[0]
        ref_msgs = ((first or {}).get("input_batches") or [[]])[0]
        if not isinstance(head, dict) or head.get("role") != "system":
            failed.append("message 0 is not a system message")
        elif not ref_msgs or _canon(ref_msgs[0]) != _canon(head):
            failed.append("message 0 differs from the system message of the agent's first model call")
        else:
            refs["reference_prompt_locator"] = f"events[seq={first['seq']}].input_batches[0][0]"
        # Continuity.
        continuity = None
        if prev is None:
            roles = [m.get("role") if isinstance(m, dict) else None for m in msgs[1:]]
            if not roles or any(r != "human" for r in roles):
                failed.append("first call input is not a system message followed only by human messages")
            else:
                continuity = "first_call"
        elif not accepted.get(prev["seq"]):
            failed.append(f"the agent's previous model call (llm_start {prev['seq']}) is not established")
        else:
            p_in = prev["input_batches"][0]
            p_out = paired[prev["seq"]].get("message") or {}
            n = len(p_in)
            want = [tc.get("id") for tc in p_out.get("tool_calls") or []]
            ai = msgs[n] if len(msgs) > n else None
            got = [m.get("tool_call_id") if isinstance(m, dict) and m.get("role") == "tool" else None for m in msgs[n + 1:]]
            if len(msgs) <= n or any(_canon(a) != _canon(b) for a, b in zip(msgs[:n], p_in)):
                failed.append(f"input does not extend the previous input (llm_start {prev['seq']}) unchanged")
            elif not isinstance(ai, dict) or ai.get("role") != "ai" or any(
                _canon(ai.get(k)) != _canon(p_out.get(k)) for k in CONTINUITY_KEYS
            ):
                failed.append(f"message {n} is not the previous call's output (llm_end {paired[prev['seq']]['seq']})")
            elif not want or got != want:
                failed.append("the messages after the previous output are not exactly one tool message per previous "
                              "tool call, in order")
            else:
                continuity = "extends_previous"
                refs["previous_input_locator"] = f"events[seq={prev['seq']}].input_batches[0]"
                refs["previous_output_locator"] = f"events[seq={paired[prev['seq']]['seq']}].message"
        if failed:
            return None, "; ".join(failed)
        return {
            "label": "[INF]",
            "rule": "agent_loop_signature",
            "checks": {"parent_run_unique": True, "system_prompt_matches_first_call": True, "continuity": continuity},
            "raw_refs": refs,
            "note": ORIGIN_NOTE,
            "chain_note": CHAIN_NOTE,
        }, ""

    def _tool_events(self, starts: list[dict], ends: list[dict], commits: list[dict]) -> None:
        groups: dict[Any, dict[str, list[dict]]] = defaultdict(lambda: {"start": [], "end": [], "commit": []})
        for key, rs in (("start", starts), ("end", ends), ("commit", commits)):
            for r in rs:
                groups[r.get("tool_call_id")][key].append(r)
        no_id = groups.pop(None, None)
        if no_id:
            for key, rs in no_id.items():
                for r in rs:
                    candidates = sorted(
                        x["seq"] for x in starts + ends if x is not r and x.get("lc_run_id") == r.get("lc_run_id")
                    ) if key != "commit" else []
                    self.skip("record", "tool_record_without_tool_call_id", [r["seq"]],
                              f"tool_{key} record has no tool_call_id, so it cannot be joined; not repaired",
                              raw_type=r.get("type"), lc_run_id_candidates=candidates)
        for tcid, g in sorted(groups.items(), key=lambda kv: min(r["seq"] for rs in kv[1].values() for r in rs)):
            all_seqs = [r["seq"] for rs in g.values() for r in rs]
            if len(g["start"]) != 1 or len(g["end"]) != 1 or len(g["commit"]) > 1:
                self.skip("record", "tool_call_unjoined", all_seqs,
                          "tool_call_id does not join exactly one tool_start, one tool_end and at most one "
                          "tool_result_committed", tool_call_id=tcid,
                          counts={k: len(v) for k, v in g.items()})
                continue
            s, e = g["start"][0], g["end"][0]
            c = g["commit"][0] if g["commit"] else None
            consistent = (
                s.get("lc_run_id") == e.get("lc_run_id")
                and s.get("tool") == e.get("tool")
                and s.get("agent_id") == e.get("agent_id")
                and s["seq"] < e["seq"]
                and (c is None or (c.get("tool") == s.get("tool") and c["seq"] > e["seq"]))
            )
            if not consistent or "output" not in e:
                self.skip("record", "tool_call_inconsistent", all_seqs,
                          "joined records disagree on lc_run_id, tool, agent_id or seq order, or the end has no output",
                          tool_call_id=tcid)
                continue
            out = e["output"]
            if "output_sha256" in e and e["output_sha256"] != digest(out):
                self.skip("record", "tool_call_digest_mismatch", all_seqs,
                          "the capture's output_sha256 does not reproduce", tool_call_id=tcid)
                continue
            name = s.get("tool")
            tool_id = f"tool:{name}"
            self.entities.setdefault(tool_id, {
                "id": tool_id,
                "kind": "tool",
                "raw_ref": {"locator": f"events[seq={s['seq']}].tool", "name": name},
            })
            eid = f"ev:tool:{s['seq']}"
            seqs = [s["seq"], e["seq"]] + ([c["seq"]] if c else [])
            self.events[eid] = {
                "id": eid,
                "origin": tool_id,
                "order": s["seq"],
                "raw_ref": {
                    "seq": seqs,
                    "tool_call_id": tcid,
                    "agent_context": s.get("agent_id"),
                    "args_locator": f"events[seq={s['seq']}].args",
                    "lc_run_id": s.get("lc_run_id"),
                },
            }
            xid = f"evi:tool_result:{e['seq']}"
            item = {
                "id": xid,
                "source": "tool",
                "produced_by": eid,
                "raw_ref": {
                    "seq": e["seq"],
                    "locator": f"events[seq={e['seq']}].output",
                    "tool_call_id": tcid,
                    "sha256": digest(out),
                    "size": size(out),
                    "sha256_basis": DIGEST_BASIS,
                },
            }
            self.tool_event_of[tcid] = eid
            if c is not None:
                item["raw_ref"]["committed_seq"] = c["seq"]
                item["raw_ref"]["committed_differs_from_raw"] = c.get("differs_from_raw")
                self.commits[tcid] = c
            if tcid in self.delegation_ids:
                self.skip("item", "delegation_result", [e["seq"]],
                          "result of a delegating tool call: its content may first occur in the child's output, so "
                          "neither source tool nor produced_by the tool event is established (0.3 §4.4); not represented",
                          tool_call_id=tcid)
                continue
            self.evidence[xid] = item
            self.tool_items[tcid] = item

    def _argument_evidence(self) -> None:
        call_count: dict[str, int] = defaultdict(int)
        for e in self.model_end.values():
            for tc in (e.get("message") or {}).get("tool_calls") or []:
                call_count[tc.get("id")] += 1
        for eid in sorted(self.model_end, key=lambda k: self.events[k]["order"]):
            e = self.model_end[eid]
            calls = (e.get("message") or {}).get("tool_calls") or []
            for j, tc in enumerate(calls):
                tcid = tc.get("id")
                args = tc.get("args")
                if tcid is None or call_count[tcid] != 1 or not isinstance(args, dict):
                    self.skip("item", "tool_call_arguments_unjoinable", [e["seq"]],
                              "tool call id missing or repeated across model outputs, or args not an object",
                              tool_call_id=tcid, index=j)
                    continue
                self.issued[tcid] = f"events[seq={e['seq']}].message.tool_calls[{j}]"
                for key in sorted(args):
                    if not is_name(key):
                        self.skip("item", "argument_key_unlocatable", [e["seq"]],
                                  "argument key cannot be written as a locator step", tool_call_id=tcid, index=j)
                        continue
                    value = args[key]
                    xid = f"evi:tool_arg:{e['seq']}:{j}:{key}"
                    source: Any = "agent"
                    if eid in self.unresolved_events:
                        source = self._ambiguity(
                            f"amb:source:{xid}", f"semantic.evidence[{xid}].source", ["agent", "framework"],
                            "the producing model call has an unresolved origin, so the author class of its output "
                            "is not established",
                            {"seq": e["seq"], "locator": f"events[seq={e['seq']}].message.tool_calls[{j}].args.{key}"},
                            basis="INF",
                        )
                    item = {
                        "id": xid,
                        "source": source,
                        "produced_by": eid,
                        "raw_ref": {
                            "seq": e["seq"],
                            "locator": f"events[seq={e['seq']}].message.tool_calls[{j}].args.{key}",
                            "tool_call_id": tcid,
                            "tool": tc.get("name"),
                            "index": j,
                            "argument": key,
                            "sha256": digest(value),
                            "size": size(value),
                            "sha256_basis": DIGEST_BASIS,
                        },
                    }
                    self.evidence[xid] = item
                    self.arg_items[tcid].append(item)
        for ev in self.events.values():
            tcid = ev["raw_ref"].get("tool_call_id")
            if tcid in self.issued:
                ev["raw_ref"]["issued_at"] = self.issued[tcid]

    # --- framework eviction (increment 2) ---------------------------------------------

    def _evictions(self, writes: list[dict]) -> None:
        by_call: dict[Any, list[dict]] = defaultdict(list)
        by_path: dict[Any, int] = defaultdict(int)
        for w in writes:
            by_call[w.get("tool_call_id")].append(w)
            by_path[w.get("path")] += 1
        version = (self.rec.get("versions") or {}).get("deepagents")
        for tcid, c in sorted(self.commits.items(), key=lambda kv: kv[1]["seq"]):
            if c.get("differs_from_raw") is not True:
                continue
            tool_eid = self.tool_event_of[tcid]
            tool_ev = self.events[tool_eid]
            s, e = (self.by_seq[x] for x in tool_ev["raw_ref"]["seq"][:2])
            failed: list[str] = []
            if "framework:deepagents" not in self.entities or version not in EVICTION_VERIFIED_VERSIONS:
                failed.append(f"framework version {version!r} is not one whose eviction code was verified")
            content = c.get("content")
            if not isinstance(content, str) or c.get("sha256") != digest(content):
                failed.append("committed content is missing, not a string, or its sha256 does not reproduce")
            cands = by_call.get(tcid, [])
            w = cands[0] if len(cands) == 1 else None
            if w is None:
                failed.append(f"{len(cands)} file_write records carry this tool_call_id; exactly one is required")
            else:
                path = w.get("path")
                if (w.get("attribution") != "tool_call" or w.get("error") is not None or not isinstance(path, str)
                        or w.get("result_path") != path or by_path[path] != 1):
                    failed.append("file_write is not an error-free tool_call write that alone names its path")
                elif path.rsplit("/", 1)[-1] != _sanitize(tcid):
                    failed.append("file name is not the sanitized tool_call_id")
                if w.get("agent_id") != s.get("agent_id") or c.get("agent_id") != s.get("agent_id"):
                    failed.append("file_write or tool_result_committed has a different agent_id from the tool call")
                if not e["seq"] < w["seq"] < c["seq"]:
                    failed.append("seqs are not tool_end < file_write < tool_result_committed")
                text = _joined_text(e.get("output"))
                if text is None:
                    failed.append("raw result is not text only")
                elif _sha_text(text) != w.get("sha256") or len(text) != w.get("size"):
                    failed.append("joined text of the raw result does not reproduce the file's sha256 and size")
                if isinstance(path, str) and isinstance(content, str):
                    head = EVICTION_TEMPLATE_HEAD.format(tool_call_id=tcid, file_path=path)
                    if not content.startswith(head) or not content.endswith("\n") or len(content) <= len(head) + 1:
                        failed.append("committed content is not the eviction template for this call and path")
            if failed:
                self.skip("item", "eviction_substitute", [c["seq"]],
                          "committed tool message differs from the raw result, but the eviction is not established: "
                          + "; ".join(failed) + ". No eviction event or substitute is emitted (fail closed)",
                          tool_call_id=tcid, committed_sha256=c.get("sha256"))
                continue
            tool_ev["raw_ref"]["seq"] = [x for x in tool_ev["raw_ref"]["seq"] if x != c["seq"]]
            eid = f"ev:evict:{w['seq']}"
            self.events[eid] = {
                "id": eid,
                "origin": "framework:deepagents",
                "order": w["seq"],
                "raw_ref": {
                    "seq": [w["seq"], c["seq"]],
                    "tool_call_id": tcid,
                    "agent_context": s.get("agent_id"),
                    "input_locator": f"events[seq={e['seq']}].output",
                    "output_locator": f"events[seq={c['seq']}].content",
                    "file_path": path,
                    "file_path_status": "provenance only; the file is not an evidence item",
                    "file_sha256": w.get("sha256"),
                    "file_size": w.get("size"),
                    "origin_basis": {
                        "label": "[INF]",
                        "rule": "eviction_event",
                        "code": EVICTION_CODE,
                        "code_version": version,
                        "checks": {"version_verified": True, "single_file_write": True, "path_is_sanitized_call_id": True,
                                   "seq_order": True, "file_digest_reproduces": True, "template_matches": True},
                        "raw_refs": {"file_write_locator": f"events[seq={w['seq']}]",
                                     "committed_locator": f"events[seq={c['seq']}]",
                                     "tool_end_locator": f"events[seq={e['seq']}]"},
                        "note": EVICTION_NOTE,
                    },
                },
            }
            self.eviction_records.add(w["seq"])
            xid = f"evi:evict_sub:{c['seq']}"
            source = self._ambiguity(
                f"amb:source:{xid}", f"semantic.evidence[{xid}].source", ["framework", "tool"],
                "eviction substitute: the template text is framework-authored and the preview is tool content, so "
                "one author class is not established",
                {"seq": c["seq"], "locator": f"events[seq={c['seq']}].content"}, basis="INF",
            )
            item = {
                "id": xid,
                "source": source,
                "produced_by": eid,
                "raw_ref": {
                    "seq": c["seq"],
                    "locator": f"events[seq={c['seq']}].content",
                    "tool_call_id": tcid,
                    "sha256": digest(content),
                    "size": size(content),
                    "sha256_basis": DIGEST_BASIS,
                },
            }
            self.evidence[xid] = item
            self.sub_items[tcid] = item

    def _other_records(self, records: list[dict]) -> None:
        labels = sorted(r["seq"] for r in records if r.get("type") in _LABEL_TYPES)
        if labels:
            self.skip("record", "framework_label", labels,
                      "lifecycle labels (agent/delegation start, end, error) stay in raw data; framework labels are "
                      "never semantic structure (0.3 §11 r9)")
        files = sorted(r["seq"] for r in records if r.get("type") in _FILE_TYPES and r["seq"] not in self.eviction_records)
        if files:
            self.skip("record", "file_record", files,
                      "file_write/file_read records outside an eviction event are not represented; no file lineage "
                      "is stored")
        known = {"llm_start", "llm_end", "tool_start", "tool_end", "tool_result_committed"} | _LABEL_TYPES | _FILE_TYPES
        other = sorted(r["seq"] for r in records if r.get("type") not in known)
        if other:
            self.skip("record", "unknown_record_type", other, "raw record type not handled by the adapter")

    def _question(self) -> None:
        q = self.rec.get("question")
        if q is None:
            return
        xid = "evi:question"
        source = self._ambiguity(
            f"amb:source:{xid}", f"semantic.evidence[{xid}].source", ["user", "benchmark"],
            "the record's question field does not say whether a user or the benchmark authored it",
            {"locator": "question"}, basis="INF",
        )
        self.question_item = {
            "id": xid,
            "source": source,
            "raw_ref": {
                "external": True,
                "locator": "question",
                "note": "External evidence: no event in the trace outputs it (0.3 §4.4).",
                "sha256": digest(q),
                "size": size(q),
                "sha256_basis": DIGEST_BASIS,
            },
        }
        self.evidence[xid] = self.question_item

    def _system_prompts(self) -> None:
        seen: dict[tuple[str, str], int] = {}
        for eid, s in self.model_start.items():
            batches = s.get("input_batches")
            if not batches:
                continue
            for msg in batches[0]:
                if isinstance(msg, dict) and msg.get("role") == "system":
                    k = (s.get("agent_id"), digest(msg.get("content")))
                    seen.setdefault(k, s["seq"])
        for (aid, sha), first in sorted(seen.items(), key=lambda kv: kv[1]):
            self.skip("item", "system_prompt", [],
                      "system message seen in rendered inputs; no record field holds it, so it is not represented "
                      "(E3)", agent_id=aid, sha256=sha, first_input_seq=first)

    # --- delivers --------------------------------------------------------------------

    def _delivers(self) -> None:
        for eid in sorted(self.model_start, key=lambda k: self.events[k]["order"]):
            s = self.model_start[eid]
            if "input_batches" not in s or eid in self.unresolved_events:
                continue
            batches = s["input_batches"]
            if not isinstance(batches, list) or len(batches) != 1 or not isinstance(batches[0], list):
                self.skip("join", "input_not_single_batch", [s["seq"]],
                          "rendered input is not exactly one message list; no delivery is read from it")
                continue
            self._scan_input(eid, s, batches[0])

    def _link(self, eid: str, item: dict, locator: str, join: dict) -> None:
        self.links.append({
            "relation": "delivers",
            "source": eid,
            "target": item["id"],
            "raw_ref": {
                "seq": self.events[eid]["order"],
                "locator": locator,
                "join": join,
                "sha256": item["raw_ref"]["sha256"],
                "match": "exact",
                "evidence_status": LINK_STATUS,
            },
        })

    def _scan_input(self, eid: str, s: dict, msgs: list) -> None:
        seq = s["seq"]
        base = f"events[seq={seq}].input_batches[0]"
        tool_msgs: dict[Any, list[int]] = defaultdict(list)
        ai_calls: dict[Any, list[tuple[int, int]]] = defaultdict(list)
        human: list[int] = []
        for m, msg in enumerate(msgs):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role == "tool":
                tool_msgs[msg.get("tool_call_id")].append(m)
            elif role == "ai":
                for j, tc in enumerate(msg.get("tool_calls") or []):
                    ai_calls[tc.get("id")].append((m, j))
            elif role == "human":
                human.append(m)

        # Tool results.
        for tcid, idxs in sorted(tool_msgs.items(), key=lambda kv: kv[1][0]):
            item = self.tool_items.get(tcid)
            if item is None:
                self.skip("join", "rendered_tool_message_unjoined", [seq],
                          "rendered tool message whose tool_call_id joins no emitted tool result",
                          tool_call_id=tcid, messages=idxs)
                continue
            if len(idxs) != 1:
                self.skip("join", "ambiguous_join", [seq], "more than one tool message carries this tool_call_id",
                          tool_call_id=tcid, target=item["id"], messages=idxs)
                continue
            m = idxs[0]
            content = msgs[m].get("content")
            out = self._raw_output(item)
            sub = self.sub_items.get(tcid)
            commit = self.commits.get(tcid)
            if _same(content, out):
                self._link(eid, item, f"{base}[{m}].content", {"tool_call_id": tcid})
            elif sub is not None and _same(content, self.by_seq[sub["raw_ref"]["seq"]]["content"]):
                self._link(eid, sub, f"{base}[{m}].content", {"tool_call_id": tcid})
            elif commit is not None and commit.get("differs_from_raw") is True and _same(content, commit.get("content")):
                self.skip("join", "eviction_substitute_unrepresented", [seq],
                          "tool message equals a committed substitute whose eviction is not established; no link",
                          tool_call_id=tcid, message=m)
            else:
                self.skip("join", "content_mismatch", [seq],
                          "tool message joined by tool_call_id is not exactly equal to the raw result or substitute",
                          tool_call_id=tcid, target=item["id"], message=m)

        # Tool-call arguments, in the issuing agent's later inputs.
        for tcid, occ in sorted(ai_calls.items(), key=lambda kv: kv[1][0]):
            items = self.arg_items.get(tcid)
            if not items:
                continue
            if len(occ) != 1:
                self.skip("join", "ambiguous_join", [seq], "more than one AI tool_calls entry carries this id",
                          tool_call_id=tcid, messages=[m for m, _ in occ])
                continue
            m, j = occ[0]
            rendered = (msgs[m]["tool_calls"][j] or {}).get("args")
            for item in items:
                key = item["raw_ref"]["argument"]
                value = self._raw_argument(item)
                if isinstance(rendered, dict) and key in rendered and _same(rendered[key], value):
                    self._link(eid, item, f"{base}[{m}].tool_calls[{j}].args.{key}", {"tool_call_id": tcid})
                else:
                    self.skip("join", "content_mismatch", [seq],
                              "AI tool_calls entry joined by id does not hold an exactly equal argument value",
                              tool_call_id=tcid, target=item["id"], message=m)

        # The question, in a root agent's input only.
        aid = s.get("agent_id")
        if self.question_item is not None:
            q = self.rec.get("question")
            matches = [m for m in human if _same(msgs[m].get("content"), q)]
            agent = self.agents[aid]
            if matches and (agent.get("parent_agent_id") is not None or agent.get("delegation_id") is not None):
                self.skip("join", "question_in_non_root_input", [seq],
                          "a human message equals the question in a non-root agent's input; delivery to subagents is "
                          "not inferred", messages=matches)
            elif len(matches) > 1:
                self.skip("join", "ambiguous_join", [seq], "more than one human message equals the question",
                          target=self.question_item["id"], messages=matches)
            elif matches:
                self._link(eid, self.question_item, f"{base}[{matches[0]}].content",
                           {"external": "question", "role": "human", "agent_scope": "root"})

        # Delegation: the child's human messages against the delegating call's arguments.
        dele = self.agents[aid].get("delegation_id")
        if dele is None or not self.arg_items.get(dele) or not self._delegation_consistent(aid, dele):
            return
        items = self.arg_items[dele]
        hits: dict[str, list[int]] = defaultdict(list)
        for m in human:
            content = msgs[m].get("content")
            matched = [it for it in items if _same(content, self._raw_argument(it))]
            if len(matched) > 1:
                self.skip("join", "ambiguous_join", [seq], "human message equals more than one delegation argument",
                          delegation_id=dele, message=m, targets=[it["id"] for it in matched])
                continue
            for it in matched:
                hits[it["id"]].append(m)
        for it in items:
            ms = hits.get(it["id"], [])
            if len(ms) > 1:
                self.skip("join", "ambiguous_join", [seq], "more than one human message equals this delegation argument",
                          delegation_id=dele, target=it["id"], messages=ms)
            elif ms:
                self._link(eid, it, f"{base}[{ms[0]}].content",
                           {"delegation_id": dele, "join_locator": f"agents[agent_id={aid}].delegation_id", "role": "human"})

    def _delegation_consistent(self, aid: str, dele: str) -> bool:
        entries = [d for d in self.rec.get("delegations") or [] if d.get("delegation_id") == dele]
        return all(d.get("subagent_agent_id") in (None, aid) for d in entries) and len(entries) <= 1

    def _raw_output(self, item: dict) -> Any:
        return self.by_seq[item["raw_ref"]["seq"]]["output"]

    def _raw_argument(self, item: dict) -> Any:
        ref = item["raw_ref"]
        return self.by_seq[ref["seq"]]["message"]["tool_calls"][ref["index"]]["args"][ref["argument"]]

    # --- output ------------------------------------------------------------------------

    def _overlaps(self) -> list[list[str]]:
        """Every pair of events whose [start_seq, end_seq] spans intersect, in (start, start) order."""
        spans = sorted((e["raw_ref"]["start_seq"], e["raw_ref"]["end_seq"], eid) for eid, e in self.events.items())
        out = []
        for i, (lo, hi, eid) in enumerate(spans):
            out.extend([eid, other] for (lo2, _, other) in spans[i + 1:] if lo2 <= hi)
        return out

    def _output(self) -> dict:
        rec = self.rec
        for e in self.events.values():
            e["raw_ref"]["start_seq"] = min(e["raw_ref"]["seq"])
            e["raw_ref"]["end_seq"] = max(e["raw_ref"]["seq"])
        has_input = any("input_batches" in r for r in rec["events"] if r.get("type") == "llm_start")
        has_offered = any("offered_tools" in r for r in rec["events"] if r.get("type") == "llm_start")
        run_ref = {
            "trace": self.trace_ref,
            "trace_sha256": self.trace_sha256,
            "raw_schema": rec.get("schema"),
            "semantic_schema": SEMANTIC_SCHEMA,
            "status_locator": "status",
            "status": rec.get("status"),
        }
        if isinstance(rec.get("error"), dict):
            run_ref["error_type"] = rec["error"].get("type")
        semantic = {
            "run": {
                "id": rec.get("run_id"),
                "raw_ref": run_ref,
                "outputs": "UNKNOWN",
            },
            "entities": sorted(self.entities.values(), key=lambda x: x["id"]),
            "events": sorted(self.events.values(), key=lambda x: x["order"]),
            "evidence": sorted(self.evidence.values(), key=lambda x: (x["raw_ref"].get("seq", -1), x["id"])),
            "claims": [],
            "propositions": [],
            "states": [],
            "links": sorted(self.links, key=lambda l: (self.events[l["source"]]["order"], l["raw_ref"]["locator"], l["target"])),
        }
        kinds: dict[str, int] = defaultdict(int)
        for sk in self.skipped:
            kinds[sk["kind"]] += 1
        adapter = {
            "adapter": ADAPTER,
            "increment": INCREMENT,
            "raw_schema": RAW_SCHEMA,
            "claims_extracted": False,
            "delivery_telemetry": "present" if has_input else "absent",
            "offered_tools_telemetry": "present" if has_offered else "absent",
            "telemetry_limits": self._limits(has_input, has_offered),
            "order": {"rule": ORDER_RULE, "caveat": ORDER_CAVEAT, "see": "profile_local.order"},
            "rules": RULES,
            "not_emitted": NOT_EMITTED,
            "unresolved": [a["id"] for a in self.ambiguities],
            "skipped": sorted(self.skipped, key=lambda x: (x["seq"][:1] or [-1], x["scope"], x["kind"], _canon(x))),
            "counts": {
                "entities": len(self.entities),
                "events": len(self.events),
                "evidence": len(self.evidence),
                "delivers": len(self.links),
                "unresolved_origin_events": len(self.unresolved_events),
                "eviction_events": len(self.sub_items),
                "skipped_by_kind": dict(sorted(kinds.items())),
            },
        }
        conformance = {"canonical_complete": not self.ambiguities,
                       "unresolved": [a["field_path"] for a in self.ambiguities]}
        order = {"basis": ORDER_BASIS, "caveat": ORDER_CAVEAT, "overlaps": self._overlaps()}
        return {"profile": ADAPTER, "semantic": semantic,
                "profile_local": {"adapter": adapter, "order": order, "ambiguities": self.ambiguities,
                                  "conformance": conformance}}

    @staticmethod
    def _limits(has_input: bool, has_offered: bool) -> list[str]:
        out = [
            "Input coverage is not established for any event: every absent delivers link means UNKNOWN, never "
            "non-delivery.",
            "Delivery is checked only for the emitted item kinds (tool results, eviction substitutes, tool-call "
            "arguments and the question in root inputs).",
        ]
        if not has_input:
            out.append("No llm_start carries a rendered input (input_batches): no delivery can be observed in this run.")
        if not has_offered:
            out.append("llm_start carries no offered_tools field: where a rendered input exists, agent origin rests "
                       "on the agent_loop_signature inference; calls that fail it, or that have no input, are kept with "
                       "an unresolved origin. A framework call that reproduced the agent's exact "
                       "conversation state would pass, and cannot be excluded from this telemetry.")
        return out


def _sanitize(tool_call_id: Any) -> str:
    """The eviction code's file name for a tool_call_id (deepagents backends.utils.sanitize_tool_call_id)."""
    return str(tool_call_id).replace(".", "_").replace("/", "_").replace("\\", "_")


def _joined_text(output: Any) -> str | None:
    """The text the eviction code writes: the string, or the text blocks joined by newlines; None otherwise."""
    if isinstance(output, str):
        return output
    if isinstance(output, list) and output and all(
        isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str) for b in output
    ):
        return "\n".join(b["text"] for b in output)
    return None


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def adapt(record: dict, *, trace_ref: str, trace_sha256: str) -> dict:
    """Adapt one instrumented record.  ``record`` is not modified."""
    return _Adapter(record, trace_ref, trace_sha256).run()


def adapt_file(path: str | Path, *, trace_ref: str | None = None) -> dict:
    path = Path(path)
    data = path.read_bytes()
    return adapt(json.loads(data), trace_ref=trace_ref or path.name, trace_sha256=hashlib.sha256(data).hexdigest())


def dumps(doc: dict) -> str:
    """Deterministic serialization."""
    return json.dumps(doc, sort_keys=True, ensure_ascii=False, indent=1) + "\n"
