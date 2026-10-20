"""Offline checks of the two hand-grounded agenttrace.semantic/0.3 fixtures.

The fixtures live in ``fixtures/agenttrace_semantic_0_3/`` (see its README).
Every check reads the original raw traces in the AP research repo; nothing is
written.  The tests skip when the AP repo is not present.  No model, network
or database is used.

Set ``AGENTTRACE_AP_ROOT`` to point at the AP repo; by default it is the
``AP`` directory next to this worktree.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "agenttrace_semantic_0_3"
AP_ROOT = Path(os.environ.get("AGENTTRACE_AP_ROOT", Path(__file__).resolve().parents[5] / "AP"))
SCHEMA_PATH = AP_ROOT / "docs" / "agenttrace.semantic.0.3.md"
SCHEMA_SHA256 = "2273d210090d19c17f75e98bb06875e0c55ef10b9c26528ea2471cee25c01c06"

FIXTURES = {"621": "aob-621.semantic.json", "sc3": "aob-sc3-B.semantic.json"}

pytestmark = pytest.mark.skipif(
    not (AP_ROOT / "data" / "assetopsbench" / "runs").is_dir(),
    reason="AP research repo with raw runs not found (set AGENTTRACE_AP_ROOT)",
)


# --- helpers ---------------------------------------------------------------


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path: Path) -> str:
    with open(path, "rb") as fh:
        return _sha256_bytes(fh.read())


def _digest(value: Any) -> str:
    """Hash content the way the capture does (``_subagent_capture._content_digest``)."""
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return _sha256_bytes(text.encode("utf-8"))


_TOKEN = re.compile(r"([A-Za-z_]\w*)|\[(\w+)=([^\]]+)\]|\[(\d+)\]")


def resolve(record: dict, locator: str) -> Any:
    """Resolve a raw_ref locator such as ``events[seq=125].input_batches[0][46].content``."""
    obj: Any = record
    for name, key, val, idx in _TOKEN.findall(locator):
        if name:
            obj = obj[name]
        elif key:
            matches = [x for x in obj if str(x.get(key)) == val]
            assert len(matches) == 1, f"{locator}: {len(matches)} matches for [{key}={val}]"
            obj = matches[0]
        else:
            obj = obj[int(idx)]
    return obj


def _seqs(raw_ref: dict) -> list[int]:
    seq = raw_ref.get("seq")
    if seq is None:
        return []
    return list(seq) if isinstance(seq, list) else [seq]


def _walk_raw_refs(fx: dict):
    """Yield (path, raw_ref) for every raw_ref in a fixture."""
    yield "run", fx["run"]["raw_ref"]
    for coll in ("entities", "events", "evidence", "claims", "links"):
        for i, obj in enumerate(fx[coll]):
            yield f"{coll}[{i}]", obj["raw_ref"]


# --- fixtures --------------------------------------------------------------


@pytest.fixture(scope="module")
def fixtures() -> dict[str, dict]:
    return {k: json.loads((FIXTURE_DIR / v).read_text(encoding="utf-8")) for k, v in FIXTURES.items()}


@pytest.fixture(scope="module")
def traces(fixtures):
    """Raw traces, loaded read-only; their bytes are re-hashed after the module."""
    out, before = {}, {}
    for key, fx in fixtures.items():
        path = AP_ROOT / fx["run"]["raw_ref"]["trace"]
        with open(path, "rb") as fh:
            data = fh.read()
        before[key] = (path, _sha256_bytes(data))
        out[key] = json.loads(data)
    yield out
    for key, (path, digest) in before.items():
        assert _sha256_file(path) == digest, f"raw trace {path} changed during the tests"


def _events_by_seq(record: dict) -> dict[int, dict]:
    return {e["seq"]: e for e in record["events"]}


# --- provenance of the sources -----------------------------------------------


def test_canonical_schema_unchanged():
    assert _sha256_file(SCHEMA_PATH) == SCHEMA_SHA256


@pytest.mark.parametrize("key", list(FIXTURES))
def test_trace_hash_and_run_identity(key, fixtures, traces):
    fx, rec = fixtures[key], traces[key]
    rr = fx["run"]["raw_ref"]
    assert _sha256_file(AP_ROOT / rr["trace"]) == rr["trace_sha256"]
    assert rr["semantic_schema_sha256"] == SCHEMA_SHA256
    assert fx["run"]["id"] == rec["run_id"]
    assert rr["raw_schema"] == rec["schema"]
    assert rr["status"] == resolve(rec, rr["status_locator"])


# --- schema compliance (0.3 §3-§5, §8) ---------------------------------------

ENTITY_KINDS = {"agent", "tool", "service", "framework", "environment"}
SOURCES = {"user", "agent", "tool", "service", "framework", "environment", "benchmark"}
STATE_VALUES = {"TRUE", "FALSE", "UNKNOWN", "UNCERTAIN"}
RELATIONS = {"delivers", "checks", "propagates_to", "depends_on", "supported_by", "contradicted_by"}
COLLECTIONS = {"run", "entities", "events", "evidence", "claims", "propositions", "states", "links"}


@pytest.mark.parametrize("key", list(FIXTURES))
def test_schema_compliance(key, fixtures):
    fx = fixtures[key]
    assert set(fx) == COLLECTIONS  # §3: the eight collections, nothing else

    run = fx["run"]
    assert set(run) == {"id", "raw_ref", "outputs"}  # §4.1
    if run["outputs"] != "UNKNOWN":
        assert isinstance(run["outputs"], list)
        assert all(set(o) == {"evidence", "raw_ref"} for o in run["outputs"])

    ids: list[str] = []
    entities = {}
    for ent in fx["entities"]:  # §4.2
        assert {"id", "kind", "raw_ref"} <= set(ent) <= {"id", "kind", "role", "model", "raw_ref"}
        assert ent["kind"] in ENTITY_KINDS
        if ent["kind"] == "agent":
            assert isinstance(ent["role"], str) and isinstance(ent["model"], str)
        else:
            assert "role" not in ent and "model" not in ent
        entities[ent["id"]] = ent
        ids.append(ent["id"])

    events = {}
    for ev in fx["events"]:  # §4.3: no type or phase field
        assert set(ev) == {"id", "origin", "order", "raw_ref"}
        assert ev["origin"] is None or ev["origin"] in entities
        assert isinstance(ev["order"], int)
        events[ev["id"]] = ev
        ids.append(ev["id"])
    orders = [ev["order"] for ev in fx["events"]]
    assert len(orders) == len(set(orders))  # a total order within the slice

    evidence = {}
    for it in fx["evidence"]:  # §4.4
        assert {"id", "source", "raw_ref"} <= set(it) <= {"id", "source", "produced_by", "raw_ref"}
        assert it["source"] in SOURCES
        if "produced_by" in it:
            assert it["produced_by"] in events
        evidence[it["id"]] = it
        ids.append(it["id"])

    propositions = {}
    for p in fx["propositions"]:  # §4.6
        assert set(p) == {"id", "form"}
        assert isinstance(p["form"], str) and p["form"]
        propositions[p["id"]] = p
        ids.append(p["id"])

    claims = {}
    for c in fx["claims"]:  # §4.5
        assert set(c) == {"id", "evidence", "raw_ref", "proposition"}
        assert c["evidence"] in evidence and c["proposition"] in propositions
        claims[c["id"]] = c
        ids.append(c["id"])

    assert len(ids) == len(set(ids))

    def speaker(claim):  # §4.5: derived, not stored
        ev_item = evidence[claim["evidence"]]
        if ev_item["source"] == "agent":
            return events[ev_item["produced_by"]]["origin"]
        return ev_item["source"]

    for st in fx["states"]:  # §4.7, §8
        assert set(st) == {"proposition", "value", "claims"}
        assert st["value"] in STATE_VALUES
        assert st["proposition"] in propositions
        assert st["claims"] and all(c in claims for c in st["claims"])
        speakers = {speaker(claims[c]) for c in st["claims"]}
        producers = {evidence[claims[c]["evidence"]].get("produced_by") for c in st["claims"]}
        assert len(speakers) == 1 and len(producers) == 1
        (holder,) = speakers
        assert holder in entities  # the holder must be an entity
        assert None not in producers
        # Fixture rule (stricter than 0.3, which leaves this tie open): the
        # grounding claims assert the state's proposition.
        assert all(claims[c]["proposition"] == st["proposition"] for c in st["claims"])

    for link in fx["links"]:  # §4.8, §5
        assert set(link) == {"relation", "source", "target", "raw_ref"}
        assert link["relation"] in RELATIONS
        rel, src, tgt = link["relation"], link["source"], link["target"]
        if rel == "delivers":  # §5.1: agent-input event -> evidence
            assert src in events and tgt in evidence
            origin = events[src]["origin"]
            assert origin is not None and entities[origin]["kind"] == "agent"
        elif rel == "propagates_to":  # §5.3
            assert src in evidence and tgt in evidence
            producer = evidence[tgt]["produced_by"]
            assert {"relation": "delivers", "source": producer, "target": src} in [
                {k: lk[k] for k in ("relation", "source", "target")} for lk in fx["links"]
            ], "propagates_to needs delivers(B.produced_by, A)"
            assert evidence[src].get("produced_by") != producer
            assert events[evidence[src]["produced_by"]]["order"] < events[producer]["order"]
        elif rel == "checks":
            assert src in events and tgt in propositions
        elif rel == "depends_on":
            assert src in propositions and tgt in propositions
        else:  # supported_by / contradicted_by
            assert src in propositions and tgt in evidence


# --- raw references resolve against the original traces -----------------------


@pytest.mark.parametrize("key", list(FIXTURES))
def test_raw_seq_references_exist(key, fixtures, traces):
    by_seq = _events_by_seq(traces[key])
    for path, rr in _walk_raw_refs(fixtures[key]):
        seqs = _seqs(rr)
        for s in seqs:
            assert s in by_seq, f"{path}: seq {s} not in trace"
        tcid = rr.get("tool_call_id")
        if tcid and seqs:
            carrying = [by_seq[s] for s in seqs if "tool_call_id" in by_seq[s]]
            if carrying:  # tool / file events carry the id as a field
                assert all(e["tool_call_id"] == tcid for e in carrying), path
            else:  # model calls carry it inside a message or a tool call
                assert any(tcid in json.dumps(by_seq[s], default=str) for s in seqs), path


@pytest.mark.parametrize("key", list(FIXTURES))
def test_locators_and_hashes(key, fixtures, traces):
    rec = traces[key]
    checked = 0
    for path, rr in _walk_raw_refs(fixtures[key]):
        for loc_key, sha_key in (("locator", "sha256"), ("text_part_locator", "text_part_sha256")):
            if loc_key in rr and sha_key in rr:
                assert _digest(resolve(rec, rr[loc_key])) == rr[sha_key], f"{path}.{loc_key}"
                checked += 1
            elif loc_key in rr:
                resolve(rec, rr[loc_key])  # must still resolve
        if "size" in rr and "locator" in rr:
            value = resolve(rec, rr["locator"])
            text = value if isinstance(value, str) else json.dumps(value, default=str)
            if "sha256_basis" not in rr:
                assert len(text) == rr["size"], path
    assert checked >= 4


@pytest.mark.parametrize("key", list(FIXTURES))
def test_evidence_production_anchors(key, fixtures, traces):
    """Each evidence item's first occurrence lies in its producing event's raw span,
    and committed tool results hash-equal the evidence."""
    fx, by_seq = fixtures[key], _events_by_seq(traces[key])
    events = {e["id"]: e for e in fx["events"]}
    for it in fx["evidence"]:
        rr = it["raw_ref"]
        if "produced_by" in it:
            assert set(_seqs(rr)) <= set(_seqs(events[it["produced_by"]]["raw_ref"])), it["id"]
        if "committed_seq" in rr:
            committed = by_seq[rr["committed_seq"]]
            assert committed["type"] == "tool_result_committed"
            assert committed["tool_call_id"] == rr["tool_call_id"]
            assert committed["sha256"] == rr["sha256"]
            assert committed["differs_from_raw"] is False


@pytest.mark.parametrize("key", list(FIXTURES))
def test_event_origins_match_raw(key, fixtures, traces):
    """Agent-origin events are model calls of that agent; tool-origin events are
    tool calls of that tool."""
    fx, by_seq = fixtures[key], _events_by_seq(traces[key])
    entities = {e["id"]: e for e in fx["entities"]}
    for ev in fx["events"]:
        first = by_seq[_seqs(ev["raw_ref"])[0]]
        ent = entities[ev["origin"]]
        if ent["kind"] == "agent":
            raw_agent = resolve(traces[key], ent["raw_ref"]["locator"])["agent_id"]
            seqs = _seqs(ev["raw_ref"])
            assert [by_seq[s]["type"] for s in seqs] == ["llm_start", "llm_end"]
            assert all(by_seq[s]["agent_id"] == raw_agent for s in seqs)
            assert by_seq[seqs[0]]["lc_run_id"] == by_seq[seqs[1]]["lc_run_id"]
        elif ent["kind"] == "tool":
            assert first["type"] == "tool_start"
            assert first["tool"] == resolve(traces[key], ent["raw_ref"]["locator"])
            assert by_seq[_seqs(ev["raw_ref"])[-1]]["type"] in {"tool_end", "tool_result_committed"}


# --- exposure anchors ----------------------------------------------------------


def _first_llm_start_with(record: dict, agent_id: str, predicate) -> int | None:
    for e in record["events"]:
        if e["type"] == "llm_start" and e["agent_id"] == agent_id:
            if any(predicate(m) for m in e["input_batches"][0]):
                return e["seq"]
    return None


FIRST_EXPOSURE = {
    # (fixture, link target) -> seq of the first rendered input containing it
    ("621", "evi:621:R"): 125,
    ("sc3", "evi:sc3:D"): 96,
    ("sc3", "evi:sc3:G"): 386,
    ("sc3", "evi:sc3:R1"): 392,
    ("sc3", "evi:sc3:R2"): 398,
    ("sc3", "evi:sc3:R3"): 404,
}


@pytest.mark.parametrize("key", list(FIXTURES))
def test_delivers_links_resolve(key, fixtures, traces):
    fx, rec = fixtures[key], traces[key]
    evidence = {e["id"]: e for e in fx["evidence"]}
    events = {e["id"]: e for e in fx["events"]}
    for link in fx["links"]:
        if link["relation"] != "delivers":
            continue
        rr = link["raw_ref"]
        assert rr["seq"] == _seqs(events[link["source"]]["raw_ref"])[0]
        assert rr["locator"].startswith(f"events[seq={rr['seq']}].input_batches[0]")
        assert _digest(resolve(rec, rr["locator"])) == evidence[link["target"]]["raw_ref"]["sha256"]
        if "tool_call_id" in rr:
            msg = resolve(rec, rr["locator"].rsplit(".content", 1)[0])
            assert msg["role"] == "tool" and msg["tool_call_id"] == rr["tool_call_id"]


@pytest.mark.parametrize(("key", "target"), list(FIRST_EXPOSURE))
def test_first_exposure_anchor(key, target, fixtures, traces):
    fx, rec = fixtures[key], traces[key]
    by_seq = _events_by_seq(rec)
    want = FIRST_EXPOSURE[(key, target)]
    link = next(lk for lk in fx["links"] if lk["relation"] == "delivers" and lk["target"] == target)
    assert link["raw_ref"]["seq"] == want
    agent_id = by_seq[want]["agent_id"]
    sha = next(e for e in fx["evidence"] if e["id"] == target)["raw_ref"]["sha256"]
    first = _first_llm_start_with(rec, agent_id, lambda m: _digest(m.get("content")) == sha)
    assert first == want


def test_sc3_summarizer_input_lacks_grep_result(traces):
    """G is not in the summarizer's input (383); 386 is its first delivery."""
    by_seq = _events_by_seq(traces["sc3"])
    e383 = by_seq[383]
    assert e383["type"] == "llm_start" and e383["agent_id"] == "sub-1"
    assert e383.get("offered_tools") is None
    assert all(m.get("tool_call_id") != "toolu_0157EnXeA9v8fjFWocGXB6XP" for m in e383["input_batches"][0])


# --- run 621: claim span, carryover and uniqueness (tie rule) --------------------


def _no_signatures(obj: Any) -> Any:
    """Drop opaque thinking signatures (base64), which are not rendered text."""
    if isinstance(obj, dict):
        return {k: _no_signatures(v) for k, v in obj.items() if k != "signature"}
    if isinstance(obj, list):
        return [_no_signatures(v) for v in obj]
    return obj


def test_621_claim_and_carryover_spans(fixtures, traces):
    fx, rec = fixtures["621"], traces["621"]
    answer = resolve(rec, "events[seq=158].message.content[1].text")
    claim = fx["claims"][0]["raw_ref"]
    a, b = claim["span"]
    assert answer[a:b] == claim["text"]
    c, d = claim["subject_context_span"]
    assert answer[c:d] == claim["subject_context_text"]
    h0, h1 = claim["heading_subject_span"]
    assert answer[h0:h1] == claim["heading_subject_text"]

    prop = next(lk for lk in fx["links"] if lk["relation"] == "propagates_to")["raw_ref"]
    t0, t1 = prop["in_target"]["span"]
    target_text = answer[t0:t1]
    assert target_text == prop["in_target"]["text"]
    assert answer.count(target_text) == 1
    assert a <= t0 and t1 <= b  # the carried span lies inside the claim span

    source = resolve(rec, prop["in_source"]["locator"])
    values = [source[s:e] for s, e in prop["in_source"]["spans"]]
    assert values == prop["in_source"]["text"]
    # [INF] rounding rule: nearest integer with a thousands separator.
    assert [f"{round(float(v)):,}" for v in values] == target_text.split(" → ")


def test_621_values_unique_among_delivered_items(fixtures, traces):
    """§5.3(4b): no other item delivered to the answer call carries the values."""
    fx, rec = fixtures["621"], traces["621"]
    prop = next(lk for lk in fx["links"] if lk["relation"] == "propagates_to")["raw_ref"]
    delivered_idx = int(re.search(r"\[0\]\[(\d+)\]", next(
        lk for lk in fx["links"] if lk["relation"] == "delivers" and lk["source"] == "ev:621:llm:157"
    )["raw_ref"]["locator"]).group(1))
    messages = resolve(rec, "events[seq=157].input_batches[0]")
    rendered = [json.dumps(_no_signatures(m), ensure_ascii=False, default=str) for m in messages]

    for value in prop["in_source"]["text"]:
        whole, frac = value.split(".")
        exact = re.compile(rf"(?<![\d.]){whole}\.{frac[:3]}")
        rounded = round(float(value))
        approx = re.compile(rf"(?<![\d.,]){rounded // 1000},?{rounded % 1000:03d}(?![\d])")
        hits_exact = [i for i, r in enumerate(rendered) if exact.search(r)]
        hits_approx = [i for i, r in enumerate(rendered) if approx.search(r)]
        assert hits_exact == [delivered_idx], (value, hits_exact)
        # Rounded forms may match inside R itself (e.g. "1281.16"), never elsewhere.
        assert set(hits_approx) <= {delivered_idx}, (value, hits_approx)

    # The excluded next value really is tied (several delivered sources).
    tied = [i for i, r in enumerate(rendered) if re.search(r"(?<![\d.])5476\.77", r)]
    assert delivered_idx in tied and len(tied) >= 2


def test_621_native_answer_marker_equals_evidence(fixtures, traces):
    marker = fixtures["621"]["run"]["raw_ref"]["harness_answer_marker"]
    path = AP_ROOT / marker["file"]
    assert _sha256_file(path) == marker["file_sha256"]
    native = json.loads(path.read_text(encoding="utf-8"))
    assert native[marker["locator"]] == resolve(traces["621"], "events[seq=158].message.content[1].text")
    assert fixtures["621"]["run"]["outputs"] == "UNKNOWN"  # OPEN §4.1: harness marker not used


# --- prohibited objects and links -------------------------------------------------


def test_621_only_permitted_links(fixtures):
    fx = fixtures["621"]
    got = sorted((lk["relation"], lk["source"], lk["target"]) for lk in fx["links"])
    assert got == sorted([
        ("delivers", "ev:621:llm:125", "evi:621:R"),
        ("delivers", "ev:621:llm:157", "evi:621:R"),
        ("propagates_to", "evi:621:R", "evi:621:A"),
    ])
    # Upstream items (evicted history result, framework file, substitute) are
    # not represented, so no tool -> file -> tool hop can be linked.
    assert {e["id"] for e in fx["evidence"]} == {"evi:621:R", "evi:621:A"}
    assert len(fx["claims"]) == len(fx["propositions"]) == len(fx["states"]) == 1


def test_sc3_no_claims_states_or_lineage(fixtures):
    fx = fixtures["sc3"]
    assert fx["claims"] == [] and fx["propositions"] == [] and fx["states"] == []
    assert {lk["relation"] for lk in fx["links"]} == {"delivers"}
    linked = {lk["source"] for lk in fx["links"]} | {lk["target"] for lk in fx["links"]}
    assert "evi:sc3:O" not in linked and "evi:sc3:F" not in linked  # no file lineage
    entity_ids = {e["id"] for e in fx["entities"]}
    assert not linked & entity_ids  # no parent/child or availability links
    assert fx["run"]["outputs"] == "UNKNOWN"


def test_sc3_eviction_attributed_to_framework(fixtures, traces):
    fx, by_seq = fixtures["sc3"], _events_by_seq(traces["sc3"])
    ev = next(e for e in fx["events"] if e["id"] == "ev:sc3:evict-write:01CD6j")
    assert ev["origin"] == "framework:deepagents"
    raw = by_seq[40]
    assert raw["type"] == "file_write" and raw["agent_id"] == ev["raw_ref"]["raw_agent_id"]
    f = next(e for e in fx["evidence"] if e["id"] == "evi:sc3:F")["raw_ref"]
    assert (raw["path"], raw["sha256"], raw["size"]) == (f["path"], f["sha256"], f["size"])
    o = next(e for e in fx["evidence"] if e["id"] == "evi:sc3:O")["raw_ref"]
    assert o["text_part_sha256"] == f["sha256"]


def test_sc3_child_reads_target_root_file(fixtures, traces):
    fx, by_seq = fixtures["sc3"], _events_by_seq(traces["sc3"])
    f_path = next(e for e in fx["evidence"] if e["id"] == "evi:sc3:F")["raw_ref"]["path"]
    reads = [e for e in fx["events"] if "file_read" in e["raw_ref"]]
    assert len(reads) == 3
    for ev in reads:
        fr = ev["raw_ref"]["file_read"]
        raw = by_seq[fr["seq"]]
        assert raw["type"] == "file_read" and raw["agent_id"] == "sub-1"
        for k in ("path", "offset", "limit", "window_sha256", "created_at"):
            assert raw[k] == fr[k], (fr["seq"], k)
        assert raw["path"] == f_path
    grep_out = resolve(traces["sc3"], "events[seq=381].output")
    assert f_path in (grep_out if isinstance(grep_out, str) else json.dumps(grep_out))


def test_sc3_identical_sha_confound_documented(fixtures, traces):
    """The child's own eviction file has the same sha as root's file F, so a value
    match alone cannot separate them; only the path / tool_call_id join can."""
    by_seq = _events_by_seq(traces["sc3"])
    f = next(e for e in fixtures["sc3"]["evidence"] if e["id"] == "evi:sc3:F")["raw_ref"]
    child = by_seq[135]
    assert child["type"] == "file_write" and child["agent_id"] == "sub-1"
    assert child["sha256"] == f["sha256"] and child["path"] != f["path"]
    readme = (FIXTURE_DIR / "README.md").read_text(encoding="utf-8")
    assert "toolu_01VwZ82BKDyeSFwC5oMi3FwH" in readme


def test_sc3_budget_stop_documented(fixtures, traces):
    rec, rr = traces["sc3"], fixtures["sc3"]["run"]["raw_ref"]
    assert resolve(rec, rr["error_locator"]) == rr["error_type"]
    assert rec["budget_guard"]["outcome"] == rr["budget_guard_outcome"]
    assert rec["budget_guard"]["stop_reason"] == rr["budget_guard_stop_reason"]
    by_seq = _events_by_seq(rec)
    refused = by_seq[416]
    assert refused["type"] == "llm_start"
    assert not any(e["type"] == "llm_end" and e["lc_run_id"] == refused["lc_run_id"] for e in rec["events"])
    assert [by_seq[s]["type"] for s in (417, 418, 419)] == ["agent_error", "delegation_error", "agent_error"]
    assert rec["delegations"][0]["returned_to_parent"] is None
