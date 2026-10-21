"""Tests for the increment-1 structural adapter and its independent verifier.

The adapter (``_semantic_adapter``) turns an instrumented AssetOpsBench
DeepAgents record into an agenttrace.semantic/0.3 slice with entities, model
and tool events, tool-result and tool-call-argument evidence, and positive
``delivers`` links found by exact matching.  The verifier (``_semantic_verify``)
re-checks that output against the raw record and shares only the locator/hash
resolver (``_semantic_locator``) with the adapter.

Expected values come from the hand-grounded fixtures
(``fixtures/agenttrace_semantic_0_3``, ``fixtures/agenttrace_compare_0_1``),
never from the adapter.  The raw traces are read-only; they are re-hashed after
the module.  No model, network or database is used.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import tokenize
from pathlib import Path

import pytest

from agent.deep_agent import _semantic_adapter as adapter
from agent.deep_agent import _semantic_verify as verifier
from agent.deep_agent._semantic_compare import Comparator
from agent.deep_agent._semantic_locator import resolve

HERE = Path(__file__).parent
SEMANTIC_FIXTURES = HERE / "fixtures" / "agenttrace_semantic_0_3"
COMPARE_FIXTURES = HERE / "fixtures" / "agenttrace_compare_0_1"
AP_ROOT = Path(os.environ.get("AGENTTRACE_AP_ROOT", Path(__file__).resolve().parents[5] / "AP"))
RUNS_DIR = AP_ROOT / "data" / "assetopsbench" / "runs"

RUNS = {
    "620f": "aob-620-sonnet5-deep-fileobs-20261013",
    "621": "aob-621-sonnet5-deep-fileobs-20261013",
    "sc3": "aob-sc3-sonnet5-deep-spec-B-20261014",
    "607": "aob-607-sonnet5-deep-20261005",
    "612": "aob-612-sonnet5-deep-20261005",
    "615": "aob-615-sonnet5-deep-20261005",
    "617": "aob-617-sonnet5-deep-20261005",
    "620": "aob-620-sonnet5-deep-20261005",
}
WITH_INPUT = ("620f", "621", "sc3")
NO_INPUT = ("607", "612", "615", "617", "620")

pytestmark = pytest.mark.skipif(
    not RUNS_DIR.is_dir(),
    reason="AP research repo with raw runs not found (set AGENTTRACE_AP_ROOT)",
)


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _trace_path(key: str) -> Path:
    return RUNS_DIR / f"{RUNS[key]}.instrumented.json"


@pytest.fixture(scope="module")
def raw():
    """Raw records and their file hashes; every file is re-hashed after the module."""
    out = {}
    for key in RUNS:
        data = _trace_path(key).read_bytes()
        out[key] = (json.loads(data), _sha256(data))
    yield out
    for key, (_, digest) in out.items():
        assert _sha256(_trace_path(key).read_bytes()) == digest, f"raw trace {key} changed during the tests"


@pytest.fixture(scope="module")
def docs(raw):
    return {
        key: adapter.adapt(
            copy.deepcopy(rec),
            trace_ref=f"data/assetopsbench/runs/{RUNS[key]}.instrumented.json",
            trace_sha256=sha,
        )
        for key, (rec, sha) in raw.items()
    }


def _ids(doc: dict, coll: str) -> dict:
    return {o["id"]: o for o in doc["semantic"][coll]}


def _delivers(doc: dict) -> list[dict]:
    return [l for l in doc["semantic"]["links"] if l["relation"] == "delivers"]


def _skipped(doc: dict, kind: str) -> list[dict]:
    return [s for s in doc["profile_local"]["adapter"]["skipped"] if s["kind"] == kind]


# --- the verifier accepts every adapter output ---------------------------------


@pytest.mark.parametrize("key", list(RUNS))
def test_verifier_accepts_adapter_output(key, raw, docs):
    rec, sha = raw[key]
    assert verifier.verify(docs[key], rec, trace_sha256=sha) == []


@pytest.mark.parametrize("key", list(RUNS))
def test_output_is_increment_1_only(key, docs):
    doc = docs[key]
    sem = doc["semantic"]
    assert set(sem) == {"run", "entities", "events", "evidence", "claims", "propositions", "states", "links"}
    assert sem["claims"] == [] and sem["propositions"] == [] and sem["states"] == []
    assert {l["relation"] for l in sem["links"]} <= {"delivers"}
    # $unresolved only on unresolved-origin model calls and their argument evidence
    n_unres = json.dumps(sem).count('"$unresolved"')
    n_expected = sum(1 for e in sem["events"] if isinstance(e["origin"], dict)) + sum(
        1 for x in sem["evidence"] if isinstance(x["source"], dict))
    assert n_unres == n_expected
    assert (n_unres == 0) is (key in ("620f", "621"))
    assert sem["run"]["outputs"] == "UNKNOWN"
    orders = [e["order"] for e in sem["events"]]
    assert all(isinstance(o, int) for o in orders) and len(set(orders)) == len(orders)
    pl = doc["profile_local"]["adapter"]
    assert pl["claims_extracted"] is False
    assert not {"coverage", "size_bounds", "annotation"} & set(doc["profile_local"])
    kinds = {e["kind"] for e in sem["entities"]}
    assert {"agent", "tool", "framework"} <= kinds
    assert all(isinstance(e["origin"], dict) or e["origin"] in {x["id"] for x in sem["entities"]} for e in sem["events"])


# --- recovery of the hand-grounded delivers links --------------------------------


def _fixture_pairs(fx_sem: dict) -> list[tuple[int, str, str, str]]:
    """(source order, target locator, target sha, link locator) for each fixture delivers link."""
    events = {e["id"]: e for e in fx_sem["events"]}
    evidence = {e["id"]: e for e in fx_sem["evidence"]}
    out = []
    for l in fx_sem["links"]:
        if l["relation"] != "delivers":
            continue
        tgt = evidence[l["target"]]["raw_ref"]
        out.append((events[l["source"]]["order"], tgt["locator"], tgt["sha256"], l["raw_ref"]["locator"]))
    return out


def _adapter_pairs(doc: dict) -> set[tuple[int, str, str, str]]:
    events = _ids(doc, "events")
    evidence = _ids(doc, "evidence")
    return {
        (
            events[l["source"]]["order"],
            evidence[l["target"]]["raw_ref"]["locator"],
            evidence[l["target"]]["raw_ref"]["sha256"],
            l["raw_ref"]["locator"],
        )
        for l in _delivers(doc)
    }


FIXTURE_SOURCES = [
    ("621", SEMANTIC_FIXTURES / "aob-621.semantic.json", 2),
    ("sc3", SEMANTIC_FIXTURES / "aob-sc3-B.semantic.json", 5),
    ("620f", COMPARE_FIXTURES / "aob-620-compare.trace.json", 2),
]


@pytest.mark.parametrize("key,path,n", FIXTURE_SOURCES)
def test_hand_grounded_delivers_recovered(key, path, n, docs):
    fx = json.loads(path.read_text(encoding="utf-8"))
    fx_sem = fx.get("semantic", fx)
    pairs = _fixture_pairs(fx_sem)
    assert len(pairs) == n
    missing = [p for p in pairs if p not in _adapter_pairs(docs[key])]
    assert missing == []


def test_sc3_first_deliveries_match_fixture(docs):
    """All five sc3 fixture links are first deliveries; the adapter's earliest link agrees."""
    fx = json.loads((SEMANTIC_FIXTURES / "aob-sc3-B.semantic.json").read_text(encoding="utf-8"))
    doc = docs["sc3"]
    events = _ids(doc, "events")
    by_locator = {e["raw_ref"]["locator"]: e["id"] for e in doc["semantic"]["evidence"]}
    for order, tgt_locator, _, _ in _fixture_pairs(fx):
        item = by_locator[tgt_locator]
        first = min(events[l["source"]]["order"] for l in _delivers(doc) if l["target"] == item)
        assert first == order


def test_621_first_delivery_of_R(docs):
    doc = docs["621"]
    events = _ids(doc, "events")
    item = next(e["id"] for e in doc["semantic"]["evidence"] if e["raw_ref"]["locator"] == "events[seq=123].output")
    assert min(events[l["source"]]["order"] for l in _delivers(doc) if l["target"] == item) == 125


def test_sc3_delegation_description_joined_by_delegation_id(docs):
    doc = docs["sc3"]
    evidence = _ids(doc, "evidence")
    links = [
        l
        for l in _delivers(doc)
        if evidence[l["target"]]["raw_ref"]["locator"] == "events[seq=93].message.tool_calls[0].args.description"
    ]
    child = [l for l in links if l["raw_ref"]["join"].get("delegation_id")]
    assert child, "description never joined to the child"
    assert all(l["raw_ref"]["join"]["delegation_id"] == evidence[l["target"]]["raw_ref"]["tool_call_id"] for l in child)
    # The task call's other arguments never reached the child as a message.
    others = {
        e["id"]
        for e in doc["semantic"]["evidence"]
        if e["raw_ref"]["locator"].startswith("events[seq=93].message.tool_calls[0].args.")
        and not e["raw_ref"]["locator"].endswith(".description")
    }
    assert not [l for l in _delivers(doc) if l["target"] in others and l["raw_ref"]["join"].get("delegation_id")]


# --- no-input runs ---------------------------------------------------------------


@pytest.mark.parametrize("key", NO_INPUT)
def test_no_input_runs_flag_absent_and_emit_no_delivers(key, docs):
    doc = docs[key]
    assert doc["profile_local"]["adapter"]["delivery_telemetry"] == "absent"
    assert _delivers(doc) == []
    assert doc["semantic"]["events"] and doc["semantic"]["evidence"]


@pytest.mark.parametrize("key", WITH_INPUT)
def test_input_runs_flag_present(key, docs):
    assert docs[key]["profile_local"]["adapter"]["delivery_telemetry"] == "present"
    assert _delivers(docs[key])


# --- irregularities are surfaced, not repaired ------------------------------------


def test_sc3_irregularities_surfaced(docs):
    doc = docs["sc3"]
    skipped_seqs = {s: e["kind"] for e in doc["profile_local"]["adapter"]["skipped"] if e["scope"] == "record" for s in e["seq"]}
    event_seqs = {s for e in doc["semantic"]["events"] for s in e["raw_ref"]["seq"]}
    # tool_end 364 has no tool_call_id: no tool event, no evidence for that call.
    assert 364 in skipped_seqs and 364 not in event_seqs
    assert 363 in skipped_seqs and 363 not in event_seqs
    # 383 was made with offered_tools null: kept, origin unresolved.
    assert 383 not in skipped_seqs
    ev383 = next(e for e in doc["semantic"]["events"] if e["order"] == 383)
    assert isinstance(ev383["origin"], dict) and ev383["raw_ref"]["origin_basis"]["rule"] == "offered_tools_null"
    # 416 has an input and no output.
    assert skipped_seqs.get(416) == "model_call_unpaired"
    # Evictions: committed content differs from the raw result (E2 deferred).
    assert len(_skipped(doc, "eviction_substitute")) == 6
    assert _skipped(doc, "question") and _skipped(doc, "system_prompt")


def test_621_eviction_results_have_no_delivers_and_stay_unknown(docs):
    doc = docs["621"]
    evicted = {s["tool_call_id"] for s in _skipped(doc, "eviction_substitute")}
    assert len(evicted) == 3
    items = [e["id"] for e in doc["semantic"]["evidence"] if e["raw_ref"].get("tool_call_id") in evicted and e["source"] == "tool"]
    assert len(items) == 3
    assert not [l for l in _delivers(doc) if l["target"] in items]
    comp = Comparator(doc["semantic"], doc["profile_local"])
    model_events = [e["id"] for e in doc["semantic"]["events"] if e["origin"].startswith("agent:")]
    for item in items:
        for ev in model_events:
            assert comp.exposure_at(ev, item)["whole_item"] == "UNKNOWN"


@pytest.mark.parametrize("key", list(RUNS))
def test_every_raw_record_accounted_for(key, raw, docs):
    rec, _ = raw[key]
    doc = docs[key]
    in_events = [s for e in doc["semantic"]["events"] for s in e["raw_ref"]["seq"]]
    in_skipped = [s for e in doc["profile_local"]["adapter"]["skipped"] if e["scope"] == "record" for s in e["seq"]]
    assert sorted(in_events + in_skipped) == sorted(e["seq"] for e in rec["events"])


# --- order: observation order only ---------------------------------------------


@pytest.mark.parametrize("key", list(RUNS))
def test_order_is_first_raw_seq_with_caveat(key, docs):
    doc = docs[key]
    for e in doc["semantic"]["events"]:
        assert e["order"] == e["raw_ref"]["seq"][0] == min(e["raw_ref"]["seq"])
    order = doc["profile_local"]["adapter"]["order"]
    text = order["caveat"].lower()
    for word in ("precedence", "causation", "reiteration"):
        assert word in text


def test_overlapping_siblings_recorded(docs):
    overlaps = docs["sc3"]["profile_local"]["adapter"]["order"]["overlaps"]
    assert overlaps
    events = _ids(docs["sc3"], "events")
    for o in overlaps:
        a = events[o["event"]]["raw_ref"]["seq"]
        for b_id in o["overlapped_by"]:
            b = events[b_id]["raw_ref"]["seq"]
            assert a[0] < b[0] <= max(a)


# --- generality: no run constants; renaming IDs and seqs changes nothing ----------


def test_adapter_and_verifier_hold_no_run_constants():
    allowed_numbers = {"0", "1", "2", "3"}  # structural indices only; every fixture seq and count is larger or absent
    for mod in (adapter, verifier):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.NUMBER:
                assert tok.string in allowed_numbers, f"{mod.__name__}: numeric literal {tok.string} at line {tok.start[0]}"
            if tok.type == tokenize.STRING:
                low = tok.string.lower()
                for needle in ("toolu_", "aob-", "sc3", "621", "620", "large_tool_results", "chiller", "sub-1"):
                    assert needle not in low, f"{mod.__name__}: run-specific string {tok.string[:40]} at line {tok.start[0]}"


def _rename(rec: dict, offset: int) -> dict:
    """Rename run, agent, tool-call and LangChain ids and shift every seq."""
    rec = copy.deepcopy(rec)
    agent = {a["agent_id"]: f"agent-{i}" for i, a in enumerate(rec["agents"])}
    calls: dict = {}
    runs: dict = {}

    def call(x):
        return None if x is None else calls.setdefault(x, f"call-{len(calls)}")

    def lc(x):
        return None if x is None else runs.setdefault(x, f"lc-{len(runs)}")

    rec["run_id"] = "renamed-run"
    for a in rec["agents"]:
        a["agent_id"] = agent[a["agent_id"]]
        a["parent_agent_id"] = agent.get(a.get("parent_agent_id"))
        a["delegation_id"] = call(a.get("delegation_id"))
    for d in rec.get("delegations", []):
        d["delegation_id"] = call(d["delegation_id"])
        d["parent_agent_id"] = agent.get(d.get("parent_agent_id"))
        d["subagent_agent_id"] = agent.get(d.get("subagent_agent_id"))
    for e in rec["events"]:
        e["seq"] += offset
        if e.get("agent_id") is not None:
            e["agent_id"] = agent[e["agent_id"]]
        for k in ("tool_call_id", "delegation_id"):
            if k in e:
                e[k] = call(e[k])
        for k in ("lc_run_id", "lc_parent_run_id"):
            if k in e:
                e[k] = lc(e[k])
        for tc in (e.get("message") or {}).get("tool_calls") or []:
            tc["id"] = call(tc["id"])
        for batch in e.get("input_batches") or []:
            for msg in batch:
                if "tool_call_id" in msg:
                    msg["tool_call_id"] = call(msg["tool_call_id"])
                for tc in msg.get("tool_calls") or []:
                    tc["id"] = call(tc["id"])
    return rec


def _shape(doc: dict) -> dict:
    """The output with every id and seq replaced by its rank."""
    sem = doc["semantic"]
    events = sorted(sem["events"], key=lambda e: e["order"])
    rank = {e["id"]: i for i, e in enumerate(events)}
    kinds = {e["id"]: (e["kind"], e.get("role")) for e in sem["entities"]}
    ev_shape = [(("unresolved", len(e["origin"]["candidates"])) if isinstance(e["origin"], dict) else kinds[e["origin"]],
                 len(e["raw_ref"]["seq"])) for e in events]
    evidence = sorted(sem["evidence"], key=lambda x: (rank[x["produced_by"]], x["raw_ref"]["sha256"], x["raw_ref"]["locator"].split(".")[-1]))
    erank = {x["id"]: i for i, x in enumerate(evidence)}
    evi_shape = [(json.dumps(x["source"]) if not isinstance(x["source"], dict) else "unresolved", rank[x["produced_by"]],
                  x["raw_ref"]["sha256"], x["raw_ref"]["size"]) for x in evidence]
    links = sorted((rank[l["source"]], erank[l["target"]], bool(l["raw_ref"]["join"].get("delegation_id"))) for l in sem["links"])
    pl = doc["profile_local"]["adapter"]
    skipped = sorted((s["kind"], len(s["seq"])) for s in pl["skipped"])
    return {"events": ev_shape, "evidence": evi_shape, "links": links, "skipped": skipped,
            "telemetry": (pl["delivery_telemetry"], pl["offered_tools_telemetry"])}


@pytest.mark.parametrize("key", ["621", "sc3", "607"])
def test_renamed_ids_and_shifted_seqs_give_same_shape(key, raw, docs):
    rec, sha = raw[key]
    renamed = _rename(rec, offset=1000)
    doc = adapter.adapt(renamed, trace_ref="renamed.json", trace_sha256=sha)
    assert verifier.verify(doc, renamed, trace_sha256=sha) == []
    assert _shape(doc) == _shape(docs[key])


# --- determinism -----------------------------------------------------------------


@pytest.mark.parametrize("key", ["621", "sc3"])
def test_byte_identical_reruns(key, raw):
    rec, sha = raw[key]
    ref = f"data/assetopsbench/runs/{RUNS[key]}.instrumented.json"
    a = adapter.dumps(adapter.adapt(copy.deepcopy(rec), trace_ref=ref, trace_sha256=sha))
    b = adapter.dumps(adapter.adapt(copy.deepcopy(rec), trace_ref=ref, trace_sha256=sha))
    assert a == b


def test_adapt_file_matches_adapt_and_leaves_trace_unchanged(raw, docs):
    path = _trace_path("621")
    before = _sha256(path.read_bytes())
    doc = adapter.adapt_file(path, trace_ref=f"data/assetopsbench/runs/{RUNS['621']}.instrumented.json")
    assert _sha256(path.read_bytes()) == before
    assert adapter.dumps(doc) == adapter.dumps(docs["621"])


def test_adapter_does_not_mutate_its_input(raw):
    rec, sha = raw["sc3"]
    snapshot = json.dumps(rec, sort_keys=True)
    adapter.adapt(rec, trace_ref="x", trace_sha256=sha)
    assert json.dumps(rec, sort_keys=True) == snapshot


# --- fail closed ----------------------------------------------------------------


def _first_tool_message(rec: dict) -> tuple[int, int, str]:
    for e in rec["events"]:
        if e["type"] == "llm_start" and e.get("input_batches"):
            for m, msg in enumerate(e["input_batches"][0]):
                if msg.get("role") == "tool":
                    return e["seq"], m, msg["tool_call_id"]
    raise AssertionError("no tool message")


def _links_at(doc: dict, seq: int, tool_call_id: str) -> list[dict]:
    events = _ids(doc, "events")
    evidence = _ids(doc, "evidence")
    return [
        l
        for l in _delivers(doc)
        if events[l["source"]]["order"] == seq
        and evidence[l["target"]]["source"] == "tool"
        and evidence[l["target"]]["raw_ref"]["tool_call_id"] == tool_call_id
    ]


def test_duplicate_tool_message_fails_closed(raw):
    rec, sha = raw["sc3"]  # offered_tools present: the join is tested without the signature rule
    rec = copy.deepcopy(rec)
    seq, m, tcid = _first_tool_message(rec)
    base = adapter.adapt(copy.deepcopy(rec), trace_ref="x", trace_sha256=sha)
    assert _links_at(base, seq, tcid)
    start = next(e for e in rec["events"] if e["seq"] == seq)
    start["input_batches"][0].append(copy.deepcopy(start["input_batches"][0][m]))
    doc = adapter.adapt(rec, trace_ref="x", trace_sha256=sha)
    assert _links_at(doc, seq, tcid) == []
    amb = _skipped(doc, "ambiguous_join")
    assert any(s["seq"] == [seq] and s["tool_call_id"] == tcid for s in amb)
    assert verifier.verify(doc, rec, trace_sha256=sha) == []


def test_one_byte_change_removes_only_that_link(raw):
    rec, sha = raw["sc3"]  # offered_tools present: the join is tested without the signature rule
    rec = copy.deepcopy(rec)
    seq, m, tcid = _first_tool_message(rec)
    base = adapter.adapt(copy.deepcopy(rec), trace_ref="x", trace_sha256=sha)
    start = next(e for e in rec["events"] if e["seq"] == seq)
    msg = start["input_batches"][0][m]
    if isinstance(msg["content"], str):
        msg["content"] += " "
    else:
        msg["content"][0]["text"] += " "
    doc = adapter.adapt(rec, trace_ref="x", trace_sha256=sha)
    assert _links_at(doc, seq, tcid) == []
    assert len(_delivers(doc)) == len(_delivers(base)) - 1
    assert any(s["seq"] == [seq] and s["tool_call_id"] == tcid for s in _skipped(doc, "content_mismatch"))
    assert verifier.verify(doc, rec, trace_sha256=sha) == []


def test_duplicate_tool_end_drops_the_call(raw):
    rec, sha = raw["621"]
    rec = copy.deepcopy(rec)
    end = next(e for e in rec["events"] if e["type"] == "tool_end")
    dup = copy.deepcopy(end)
    dup["seq"] = max(e["seq"] for e in rec["events"]) + 1
    rec["events"].append(dup)
    doc = adapter.adapt(rec, trace_ref="x", trace_sha256=sha)
    assert not [e for e in doc["semantic"]["events"] if e["raw_ref"].get("tool_call_id") == end["tool_call_id"]]
    assert not [x for x in doc["semantic"]["evidence"] if x["source"] == "tool" and x["raw_ref"]["tool_call_id"] == end["tool_call_id"]]
    assert verifier.verify(doc, rec, trace_sha256=sha) == []


def test_unknown_raw_schema_is_rejected(raw):
    rec, sha = raw["607"]
    rec = copy.deepcopy(rec)
    rec["schema"] = "something-else/9"
    with pytest.raises(ValueError):
        adapter.adapt(rec, trace_ref="x", trace_sha256=sha)


# --- the verifier is independent: it catches tampered output -----------------------


def _tamper_cases():
    def sha(doc):
        doc["semantic"]["evidence"][0]["raw_ref"]["sha256"] = "0" * 64

    def link_locator(doc):
        l = next(l for l in doc["semantic"]["links"])
        head, _, _ = l["raw_ref"]["locator"].partition("input_batches[0][")
        l["raw_ref"]["locator"] = head + "input_batches[0][0].content"

    def link_target(doc):
        links = doc["semantic"]["links"]
        other = next(x["id"] for x in doc["semantic"]["evidence"] if x["id"] != links[0]["target"])
        links[0]["target"] = other

    def add_claim(doc):
        doc["semantic"]["claims"].append({"id": "c", "evidence": doc["semantic"]["evidence"][0]["id"], "raw_ref": {}, "proposition": "p"})

    def non_delivers(doc):
        doc["semantic"]["links"][0]["relation"] = "propagates_to"

    def unresolved_order(doc):
        doc["semantic"]["events"][0]["order"] = {"$unresolved": "x", "candidates": [1, 2]}

    def tool_source(doc):
        ev = next(e for e in doc["semantic"]["events"] if e["origin"].startswith("tool:"))
        doc["semantic"]["links"][0]["source"] = ev["id"]

    def drop_skip(doc):
        sk = doc["profile_local"]["adapter"]["skipped"]
        sk.remove(next(s for s in sk if s["kind"] == "model_call_unpaired"))

    def claims_flag(doc):
        doc["profile_local"]["adapter"]["claims_extracted"] = True

    def coverage(doc):
        doc["profile_local"]["coverage"] = [{"event": doc["semantic"]["events"][0]["id"], "status": "established"}]

    def origin(doc):
        ev = next(e for e in doc["semantic"]["events"] if e["origin"].startswith("tool:"))
        ev["origin"] = next(x["id"] for x in doc["semantic"]["entities"] if x["kind"] == "agent")

    def order_shift(doc):
        doc["semantic"]["events"][0]["order"] += 1

    return {k: v for k, v in locals().items() if callable(v)}


@pytest.mark.parametrize("case", sorted(_tamper_cases()))
def test_verifier_rejects_tampered_output(case, raw, docs):
    rec, sha = raw["sc3"]
    doc = copy.deepcopy(docs["sc3"])
    _tamper_cases()[case](doc)
    assert verifier.verify(doc, rec, trace_sha256=sha) != []


def test_verifier_rejects_wrong_trace_hash(raw, docs):
    rec, _ = raw["621"]
    assert verifier.verify(docs["621"], rec, trace_sha256="0" * 64) != []


def test_verifier_shares_only_the_locator_module():
    for mod in (adapter, verifier):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        imports = [l for l in src.splitlines() if l.startswith(("import ", "from "))]
        local = [l for l in imports if "deep_agent" in l or l.startswith("from .")]
        assert all("_semantic_locator" in l for l in local), local
    assert "_semantic_adapter" not in Path(verifier.__file__).read_text(encoding="utf-8")


# --- O2 compatibility -------------------------------------------------------------


def test_o2_exposure_on_adapter_output(docs):
    doc = docs["621"]
    comp = Comparator(doc["semantic"], doc["profile_local"])
    events = sorted(doc["semantic"]["events"], key=lambda e: e["order"])
    by_order = {e["order"]: e["id"] for e in events}
    item = next(e["id"] for e in doc["semantic"]["evidence"] if e["raw_ref"]["locator"] == "events[seq=123].output")
    assert comp.exposure_at(by_order[125], item)["whole_item"] == "DELIVERED"
    assert comp.exposure_at(by_order[157], item)["whole_item"] == "DELIVERED"
    # Before the read, nothing negative may be concluded: coverage is not established.
    earlier = [e["id"] for e in events if e["order"] < 121 and e["origin"].startswith("agent:")]
    assert earlier
    for ev in earlier:
        assert comp.exposure_at(ev, item)["whole_item"] == "UNKNOWN"
    tool_ev = next(e["id"] for e in events if e["origin"].startswith("tool:"))
    assert comp.exposure_at(tool_ev, item)["whole_item"] == "NOT_APPLICABLE"


# --- review follow-ups (schema-auditor) -------------------------------------------


def test_delegating_call_result_is_not_emitted(raw):
    """A tool call whose id is a delegation_id may return the child's text: no tool-result item (0.3 §4.4)."""
    rec, sha = raw["621"]
    rec = copy.deepcopy(rec)
    start = next(e for e in rec["events"] if e["type"] == "tool_start")
    rec["agents"].append({"agent_id": "child", "name": "x", "kind": "subagent", "parent_agent_id": rec["agents"][0]["agent_id"],
                          "delegation_id": start["tool_call_id"], "status": "ok"})
    doc = adapter.adapt(rec, trace_ref="x", trace_sha256=sha)
    assert not [x for x in doc["semantic"]["evidence"] if x["source"] == "tool" and x["raw_ref"]["tool_call_id"] == start["tool_call_id"]]
    assert any(s["tool_call_id"] == start["tool_call_id"] for s in _skipped(doc, "delegation_result"))
    assert verifier.verify(doc, rec, trace_sha256=sha) == []
    base = adapter.adapt(raw["621"][0], trace_ref="x", trace_sha256=sha)
    assert verifier.verify(base, rec, trace_sha256=sha) != []


@pytest.mark.parametrize("key,rule", [("621", "agent_loop_signature"), ("620f", "agent_loop_signature"), ("sc3", "offered_tools_present")])
def test_model_event_origin_basis_is_an_inference(key, rule, docs):
    for e in docs[key]["semantic"]["events"]:
        if isinstance(e["origin"], str) and e["origin"].startswith("agent:"):
            basis = e["raw_ref"]["origin_basis"]
            assert basis["label"] == "[INF]" and basis["rule"] == rule
            assert "OBS" not in json.dumps(basis)
            assert "not" in basis["note"] and "prove" in basis["note"]


def test_raw_refs_name_no_semantic_ids(docs):
    """raw_ref carries raw locators, not pointers to other semantic objects (0.3 §5, §11 r9)."""
    sem = docs["sc3"]["semantic"]
    ids = {o["id"] for c in ("entities", "events", "evidence") for o in sem[c]}
    for c in ("events", "evidence"):
        for o in sem[c]:
            assert not ids & {v for v in o["raw_ref"].values() if isinstance(v, str)}
    assert all("via" not in l["raw_ref"]["join"] for l in sem["links"])


# --- agent-origin attribution without offered_tools (agent_loop_signature) -----------


@pytest.mark.parametrize("key,n", [("620f", 7), ("621", 24)])
def test_all_calls_pass_the_agent_loop_signature(key, n, raw, docs):
    rec, _ = raw[key]
    starts = [e["seq"] for e in rec["events"] if e["type"] == "llm_start"]
    assert len(starts) == n and all("offered_tools" not in e for e in rec["events"] if e["type"] == "llm_start")
    model = [e for e in docs[key]["semantic"]["events"] if e["origin"].startswith("agent:")]
    assert sorted(e["order"] for e in model) == starts
    checks = [e["raw_ref"]["origin_basis"]["checks"] for e in model]
    assert [c["continuity"] for c in checks] == ["first_call"] + ["extends_previous"] * (n - 1)
    assert all(c["parent_run_unique"] and c["system_prompt_matches_first_call"] for c in checks)
    assert not [e for e in docs[key]["semantic"]["events"] if isinstance(e["origin"], dict)]
    for e in model:
        for loc in e["raw_ref"]["origin_basis"]["raw_refs"].values():
            if isinstance(loc, str) and loc.startswith("events["):
                resolve(rec, loc)


def test_delivery_links_unchanged_by_the_signature_rule(raw, docs):
    """The links of 620f/621 equal those produced when agent origin is not gated at all."""
    for key, path, n in FIXTURE_SOURCES:
        if key in ("621", "620f"):
            assert len(_fixture_pairs(json.loads(path.read_text(encoding="utf-8")).get("semantic", json.loads(path.read_text(encoding="utf-8"))))) == n
    for key, expected in (("620f", 172), ("621", 1376)):
        assert len(_delivers(docs[key])) == expected


def _unresolved_orders(doc: dict) -> set:
    return {e["order"] for e in doc["semantic"]["events"] if isinstance(e["origin"], dict)}


def _ambiguity_for(doc: dict, order: int) -> dict:
    eid = next(e["id"] for e in doc["semantic"]["events"] if e["order"] == order)
    return next(a for a in doc["profile_local"]["ambiguities"] if a["field_path"] == f"semantic.events[{eid}].origin")


def _without_offered_tools(rec: dict) -> dict:
    rec = copy.deepcopy(rec)
    for e in rec["events"]:
        e.pop("offered_tools", None)
    return rec


def test_summarizer_shaped_call_is_rejected(raw):
    """sc3 with offered_tools removed: the summarizer call 383 fails the signature."""
    rec, sha = raw["sc3"]
    rec = _without_offered_tools(rec)
    doc = adapter.adapt(rec, trace_ref="x", trace_sha256=sha)
    assert 383 in _unresolved_orders(doc)
    amb = _ambiguity_for(doc, 383)
    assert amb["reason"].startswith("agent_loop_signature_failed") and "system message" in amb["reason"]
    assert amb["raw_ref"]["input_locator"] == "events[seq=383].input_batches[0]"
    assert verifier.verify(doc, rec, trace_sha256=sha) == []


def test_post_summary_calls_fail_closed(raw):
    """After the summary every later call of that agent is unestablished; other agents are unaffected."""
    rec, sha = raw["sc3"]
    rec = _without_offered_tools(rec)
    doc = adapter.adapt(rec, trace_ref="x", trace_sha256=sha)
    rejected = _unresolved_orders(doc)
    child_after = [e["seq"] for e in rec["events"] if e["type"] == "llm_start" and e["agent_id"] == "sub-1" and 383 < e["seq"] < 416]
    assert child_after and set(child_after) <= rejected
    assert "previous model call" in _ambiguity_for(doc, 386)["reason"]
    events = _ids(doc, "events")
    assert not [l for l in _delivers(doc) if events[l["source"]]["order"] in rejected]
    root_calls = [e["seq"] for e in rec["events"] if e["type"] == "llm_start" and e["agent_id"] == "root"]
    model_orders = {e["order"] for e in doc["semantic"]["events"] if e["origin"] == "agent:root"}
    assert set(child_after) <= {e["order"] for e in doc["semantic"]["events"]}  # kept, not dropped
    assert set(root_calls) <= model_orders
    child_before = [e["seq"] for e in rec["events"] if e["type"] == "llm_start" and e["agent_id"] == "sub-1" and e["seq"] < 383]
    assert set(child_before) <= {e["order"] for e in doc["semantic"]["events"] if e["origin"] == "agent:sub-1"}
    assert verifier.verify(doc, rec, trace_sha256=sha) == []


@pytest.mark.parametrize("mutation", ["shared_parent", "changed_prompt", "extra_message", "edited_history", "first_call_with_ai"])
def test_signature_violations_fail_closed(mutation, raw):
    rec, sha = raw["621"]
    rec = copy.deepcopy(rec)
    starts = [e for e in rec["events"] if e["type"] == "llm_start"]
    target = starts[5]
    msgs = target["input_batches"][0]
    if mutation == "shared_parent":
        target["lc_parent_run_id"] = starts[4]["lc_parent_run_id"]
        expect = {target["seq"], starts[4]["seq"]}
    elif mutation == "changed_prompt":
        msgs[0] = copy.deepcopy(msgs[0])
        msgs[0]["content"] = "a different prompt"
        expect = {target["seq"]}
    elif mutation == "extra_message":
        msgs.append({"role": "human", "content": "injected"})
        expect = {target["seq"]}
    elif mutation == "edited_history":
        msgs[1] = {"role": "human", "content": "edited"}
        expect = {target["seq"]}
    else:
        target = starts[0]
        target["input_batches"][0].append({"role": "ai", "content": "x"})
        expect = {target["seq"]}
    doc = adapter.adapt(rec, trace_ref="x", trace_sha256=sha)
    rejected = _unresolved_orders(doc)
    assert expect <= rejected
    later = [s["seq"] for s in starts if s["seq"] > min(expect)]
    assert set(later) <= rejected  # the chain stays broken: later calls fail closed
    assert verifier.verify(doc, rec, trace_sha256=sha) == []


def test_verifier_recomputes_origin_rather_than_trusting_metadata(raw, docs):
    """Promoting a kept unresolved call to an agent origin, with well-formed metadata, is caught from raw records."""
    rec, sha = raw["sc3"]
    rec = _without_offered_tools(rec)
    doc = adapter.adapt(rec, trace_ref="x", trace_sha256=sha)
    good = next(e for e in doc["semantic"]["events"] if e["origin"] == "agent:sub-1")
    forged = next(e for e in doc["semantic"]["events"] if e["order"] == 386)
    forged["origin"] = "agent:sub-1"
    forged["raw_ref"]["origin_basis"] = copy.deepcopy(good["raw_ref"]["origin_basis"])
    path = f"semantic.events[{forged['id']}].origin"
    pl = doc["profile_local"]
    gone = next(a for a in pl["ambiguities"] if a["field_path"] == path)
    pl["ambiguities"].remove(gone)
    pl["adapter"]["unresolved"].remove(gone["id"])
    pl["conformance"]["unresolved"].remove(path)
    errors = verifier.verify(doc, rec, trace_sha256=sha)
    assert any("not established from the raw record" in e for e in errors)


def test_refused_call_stays_out_with_provenance(docs, raw):
    rec, _ = raw["sc3"]
    doc = docs["sc3"]
    assert 416 not in {s for e in doc["semantic"]["events"] for s in e["raw_ref"]["seq"]}
    entry = next(x for x in _skipped(doc, "model_call_unpaired") if x["seq"] == [416])
    assert entry["raw_agent_id"] == "sub-1" and entry["starts"] == 1 and entry["ends"] == 0
    assert resolve(rec, entry["input_locator"]) == next(e for e in rec["events"] if e["seq"] == 416)["input_batches"][0]


def test_links_declare_inferred_source_status(docs):
    for key in ("621", "sc3"):
        for l in _delivers(docs[key]):
            assert "[INF]" in l["raw_ref"]["evidence_status"] and "observed" in l["raw_ref"]["evidence_status"]


# --- no-input runs: paired model calls kept with unresolved origin -----------------------


def _expected_no_input(rec: dict) -> tuple[list[int], int]:
    """Independent count from raw records: llm_start seqs, and top-level tool-call arguments of their outputs."""
    ends = {e["lc_run_id"]: e for e in rec["events"] if e["type"] == "llm_end"}
    starts = [e["seq"] for e in rec["events"] if e["type"] == "llm_start" and e["lc_run_id"] in ends]
    n_args = sum(len(tc["args"]) for e in ends.values() for tc in e["message"].get("tool_calls") or [])
    return starts, n_args


NO_INPUT_COUNTS = {}  # filled from raw records, never from the adapter


@pytest.mark.parametrize("key", NO_INPUT)
def test_no_input_model_calls_kept_with_unresolved_origin(key, raw, docs):
    rec, sha = raw[key]
    doc = docs[key]
    sem, pl = doc["semantic"], doc["profile_local"]
    starts, n_args = _expected_no_input(rec)
    assert starts and all("input_batches" not in e and "offered_tools" not in e for e in rec["events"] if e["type"] == "llm_start")
    model = [e for e in sem["events"] if isinstance(e["origin"], dict)]
    assert sorted(e["order"] for e in model) == starts
    for e in model:
        o = e["origin"]
        assert o["candidates"] == [f"agent:{e['raw_ref']['raw_agent_id']}", "framework:deepagents", "$absent"]
        assert e["raw_ref"]["input_telemetry"] == "absent"
        assert e["raw_ref"]["origin_basis"]["label"] == "UNRESOLVED"
        assert resolve(rec, e["raw_ref"]["origin_basis"]["raw_refs"]["agent_id_locator"]) == e["raw_ref"]["raw_agent_id"]
        assert e["order"] == e["raw_ref"]["seq"][0]
    args = [x for x in sem["evidence"] if isinstance(x["source"], dict)]
    assert len(args) == n_args > 0
    for x in args:
        assert x["source"]["candidates"] == ["agent", "framework"]
        assert x["produced_by"] in {e["id"] for e in model}
        value = resolve(rec, x["raw_ref"]["locator"])
        assert x["raw_ref"]["sha256"] == hashlib.sha256(
            (value if isinstance(value, str) else json.dumps(value, default=str)).encode()).hexdigest()
    assert _delivers(doc) == []
    assert _skipped(doc, "model_call_origin_unresolved") == []
    ambs = pl["ambiguities"]
    assert len(ambs) == len(model) + len(args) == len(pl["adapter"]["unresolved"])
    assert pl["conformance"] == {"canonical_complete": False, "unresolved": [a["field_path"] for a in ambs]}
    assert pl["adapter"]["delivery_telemetry"] == "absent"
    assert verifier.verify(doc, rec, trace_sha256=sha) == []


@pytest.mark.parametrize("key", NO_INPUT)
def test_no_input_events_give_no_exposure(key, docs):
    doc = docs[key]
    comp = Comparator(doc["semantic"], doc["profile_local"])
    model = [e["id"] for e in doc["semantic"]["events"] if isinstance(e["origin"], dict)]
    items = [x["id"] for x in doc["semantic"]["evidence"]]
    for ev in model:
        for item in items[:5]:
            assert comp.exposure_at(ev, item)["whole_item"] == "UNRESOLVED"


BASELINE_LINKS = {  # recorded before the no-input correction
    "620f": (172, "749460394cac633f319fad5b8b15550f3a80a35ce7f792e3cee6b66560ae8f10"),
    "621": (1376, "9d42af1ebbe64f91e5e468cc26a31e9ecf7369228453400bb1b49a8153e05487"),
    "sc3": (4546, "bd2033737cb845ea75832e3e4d720d185744effc2a66525cd5c047bba9732b3c"),
}


@pytest.mark.parametrize("key", WITH_INPUT)
def test_input_run_deliveries_unchanged(key, docs):
    sem = docs[key]["semantic"]
    ev = {e["id"]: e for e in sem["events"]}
    evi = {e["id"]: e for e in sem["evidence"]}
    rows = sorted(json.dumps([ev[l["source"]]["order"], evi[l["target"]]["raw_ref"]["locator"],
                              evi[l["target"]]["raw_ref"]["sha256"], l["raw_ref"]["locator"]]) for l in sem["links"])
    assert (len(rows), _sha256("\n".join(rows).encode())) == BASELINE_LINKS[key]
    expected = [] if key != "sc3" else ["semantic.events[ev:llm:383].origin"]
    assert docs[key]["profile_local"]["conformance"] == {"canonical_complete": not expected, "unresolved": expected}


def test_verifier_rejects_misused_unresolved_origin(raw, docs):
    rec, sha = raw["621"]
    doc = copy.deepcopy(docs["621"])
    ev = next(e for e in doc["semantic"]["events"] if e["origin"] == "agent:root")
    ev["origin"] = {"$unresolved": "amb:x", "candidates": ["agent:root", "$absent"]}
    assert verifier.verify(doc, rec, trace_sha256=sha) != []


def test_verifier_requires_no_input_calls_preserved(raw, docs):
    rec, sha = raw["607"]
    doc = copy.deepcopy(docs["607"])
    victim = next(e for e in doc["semantic"]["events"] if isinstance(e["origin"], dict))
    doc["semantic"]["events"].remove(victim)
    doc["semantic"]["evidence"] = [x for x in doc["semantic"]["evidence"] if x["produced_by"] != victim["id"]]
    errors = verifier.verify(doc, rec, trace_sha256=sha)
    assert any("not preserved" in e for e in errors)


def test_verifier_rejects_links_from_unresolved_origin(raw, docs):
    rec, sha = raw["607"]
    doc = copy.deepcopy(docs["607"])
    ev = next(e for e in doc["semantic"]["events"] if isinstance(e["origin"], dict))
    item = doc["semantic"]["evidence"][0]
    doc["semantic"]["links"].append({"relation": "delivers", "source": ev["id"], "target": item["id"],
                                     "raw_ref": {"locator": f"events[seq={ev['order']}].input_batches[0][0].content", "join": {}}})
    assert verifier.verify(doc, rec, trace_sha256=sha) != []


# --- model calls whose agent origin is not established are kept (decision 2026-10-20) -------


def test_sc3_summarizer_383_kept_with_unresolved_origin(raw, docs):
    rec, sha = raw["sc3"]
    doc = docs["sc3"]
    ev = next(e for e in doc["semantic"]["events"] if e["order"] == 383)
    assert ev["origin"]["candidates"] == ["agent:sub-1", "framework:deepagents", "$absent"]
    ref = ev["raw_ref"]
    assert ref["seq"] == [383, 385] and ref["input_telemetry"] == "present" and ref["raw_agent_id"] == "sub-1"
    by_seq = {e["seq"]: e for e in rec["events"]}
    assert resolve(rec, ref["input_locator"]) == by_seq[383]["input_batches"][0]
    assert ref["input_sha256"] == hashlib.sha256(json.dumps(by_seq[383]["input_batches"], default=str).encode()).hexdigest()
    assert ref["output_sha256"] == hashlib.sha256(json.dumps(by_seq[385]["message"], default=str).encode()).hexdigest()
    amb = _ambiguity_for(doc, 383)
    assert amb["reason"].startswith("offered_tools_null") and amb["candidates"] == ev["origin"]["candidates"]
    assert not [l for l in _delivers(doc) if l["source"] == ev["id"]]
    assert not [x for x in doc["semantic"]["evidence"] if x["produced_by"] == ev["id"]]  # 383 issued no tool calls
    comp = Comparator(doc["semantic"], doc["profile_local"])
    item = doc["semantic"]["evidence"][0]["id"]
    assert comp.exposure_at(ev["id"], item)["whole_item"] == "UNRESOLVED"
    assert verifier.verify(doc, rec, trace_sha256=sha) == []


def test_unattributed_call_kept_with_both_raw_agents(raw):
    rec, sha = raw["sc3"]
    rec = copy.deepcopy(rec)
    end = next(e for e in rec["events"] if e["type"] == "llm_end" and e["agent_id"] == "sub-1")
    end["agent_id"] = "root"
    start_seq = next(e["seq"] for e in rec["events"] if e["type"] == "llm_start" and e["lc_run_id"] == end["lc_run_id"])
    doc = adapter.adapt(rec, trace_ref="x", trace_sha256=sha)
    ev = next(e for e in doc["semantic"]["events"] if e["order"] == start_seq)
    assert ev["origin"]["candidates"] == ["agent:sub-1", "agent:root", "framework:deepagents", "$absent"]
    assert ev["raw_ref"]["origin_basis"]["rule"] == "agent_id_unattributed"
    assert verifier.verify(doc, rec, trace_sha256=sha) == []


@pytest.mark.parametrize("change", ["drop_absent", "add_agent", "wrong_rule", "wrong_hash"])
def test_verifier_rejects_wrong_unresolved_metadata(change, raw, docs):
    rec, sha = raw["sc3"]
    doc = copy.deepcopy(docs["sc3"])
    ev = next(e for e in doc["semantic"]["events"] if e["order"] == 383)
    amb = _ambiguity_for(doc, 383)
    if change == "drop_absent":
        ev["origin"]["candidates"].remove("$absent")
        amb["candidates"] = list(ev["origin"]["candidates"])
    elif change == "add_agent":
        ev["origin"]["candidates"].insert(0, "agent:root")
        amb["candidates"] = list(ev["origin"]["candidates"])
    elif change == "wrong_rule":
        ev["raw_ref"]["origin_basis"]["rule"] = "no_rendered_input"
    else:
        ev["raw_ref"]["input_sha256"] = "0" * 64
    assert verifier.verify(doc, rec, trace_sha256=sha) != []


def test_no_origin_unresolved_skips_remain(docs):
    for key in RUNS:
        assert _skipped(docs[key], "model_call_origin_unresolved") == []
