"""File-observability instrumentation: tool-result commits, delegation file
hand-back, and backend file operations attributed by ``tool_call_id``.

Reuses the scripted scenarios of ``test_offloading_probe``.  As there, every
model "decision" is scripted: these tests establish observable information
flow only.  Runner tests go through ``DeepAgentRunner``; direct-graph tests
call ``create_deep_agent`` themselves to compare final graph state (including
``files``), which the runner does not expose.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.backends.utils import create_file_data
from langchain_core.messages import AIMessage

from agent.deep_agent._file_observer import observing_backend_factory
from agent.deep_agent._subagent_capture import SubagentCaptureHandler
from agent.deep_agent.runner import DeepAgentRunner
from observability import runspan as _runspan
from observability import set_run_context

from .test_offloading_probe import (
    _CHILD_PATH,
    _EVICT_PREFIX,
    _READ_LIMIT,
    _READ_OFFSET,
    _USAGE,
    CRITICAL_MARKER,
    CRITICAL_ORIGINAL,
    ORIGINAL,
    QUESTION,
    _ai,
    _consequential_script,
    _llm_inputs,
    _read_call,
    _ScriptedModel,
    _tool_end,
    _tool_msg,
    get_event_log,
    get_history,
)

# deepagents 0.5.3 warns on every resolution of a callable backend (see _file_observer).
pytestmark = pytest.mark.filterwarnings("ignore:Passing a callable:DeprecationWarning")

_FILE_EVENTS = {"file_write", "file_edit", "file_read", "tool_result_committed"}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.fixture(autouse=True)
def _reset_ctx():
    _runspan._run_id_var.set(None)
    _runspan._scenario_id_var.set(None)
    yield
    _runspan._run_id_var.set(None)
    _runspan._scenario_id_var.set(None)


# -- runner / direct-graph harnesses -------------------------------------------


async def _run_runner(out_dir: Path, script, monkeypatch, *, capture: bool, question: str = QUESTION):
    monkeypatch.setenv("AGENT_TRAJECTORY_DIR", str(out_dir))
    if capture:
        monkeypatch.setenv("AGENTTRACE_CAPTURE_SUBAGENTS", "true")
    else:
        monkeypatch.delenv("AGENTTRACE_CAPTURE_SUBAGENTS", raising=False)
    set_run_context(run_id="run-1", scenario_id="s1")
    client = MagicMock()
    client.get_tools = AsyncMock(return_value=[get_history, get_event_log])
    with (
        patch("agent.deep_agent.runner._build_chat_model", return_value=_ScriptedModel(messages=iter(script))),
        patch("langchain_mcp_adapters.client.MultiServerMCPClient", return_value=client),
        patch("deepagents.create_deep_agent", wraps=create_deep_agent) as create,
    ):
        result = await DeepAgentRunner(server_paths={"iot": "iot-mcp-server"}).run(question)
    rec_path = out_dir / "run-1.instrumented.json"
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else None
    return result, rec, create


async def _run_graph(script, *, backend, capture=None, files=None, question: str = QUESTION) -> dict:
    agent = create_deep_agent(
        model=_ScriptedModel(messages=iter(script)),
        tools=[get_history, get_event_log],
        system_prompt="sys",
        backend=backend,
    )
    state: dict = {"messages": [{"role": "user", "content": question}]}
    if files is not None:
        state["files"] = files
    config = {"callbacks": [capture]} if capture is not None else {}
    return await agent.ainvoke(state, config=config)


def _final_messages(state: dict) -> list[tuple]:
    return [(m.type, m.content, getattr(m, "tool_call_id", None)) for m in state["messages"]]


def _final_files(state: dict) -> dict[str, tuple]:
    return {path: (_sha(fd["content"]), "created_at" in fd) for path, fd in state.get("files", {}).items()}


def _native(out_dir: Path) -> dict:
    record = json.loads((out_dir / "run-1.json").read_text())
    record["trajectory"].pop("started_at")
    return record


def _events(events: list[dict], type_: str, **match) -> list[dict]:
    return [e for e in events if e["type"] == type_ and all(e.get(k) == v for k, v in match.items())]


# -- 1. native execution equivalence (runner) ----------------------------------


@pytest.mark.anyio
async def test_runner_native_output_identical_and_no_backend_when_capture_off(tmp_path, monkeypatch):
    off, _, create_off = await _run_runner(tmp_path / "off", _consequential_script(), monkeypatch, capture=False)
    on, rec, create_on = await _run_runner(tmp_path / "on", _consequential_script(), monkeypatch, capture=True)

    assert on.answer == off.answer
    assert _native(tmp_path / "on") == _native(tmp_path / "off")
    assert rec is not None and _events(rec["events"], "file_write")

    # Capture off: create_deep_agent is called exactly as before, with no backend.
    [off_call] = create_off.call_args_list
    assert "backend" not in off_call.kwargs
    assert set(off_call.kwargs) == {"model", "tools", "system_prompt"}
    [on_call] = create_on.call_args_list
    assert callable(on_call.kwargs["backend"])


# -- 2. graph equivalence (direct) ---------------------------------------------


@pytest.mark.anyio
async def test_graph_state_identical_with_observed_backend():
    plain = await _run_graph(_consequential_script(), backend=StateBackend())
    capture = SubagentCaptureHandler()
    observed = await _run_graph(
        _consequential_script(), backend=observing_backend_factory(StateBackend(), capture), capture=capture,
    )

    assert _final_messages(observed) == _final_messages(plain)
    assert _final_files(observed) == _final_files(plain)
    assert _final_files(plain) == {_CHILD_PATH: (_sha(CRITICAL_ORIGINAL), True)}
    assert _events(capture.events, "file_write") and _events(capture.events, "file_read")


# -- 3-5, 9. consequential lockout scenario through the runner ------------------


@pytest.fixture
async def lockout(tmp_path, monkeypatch):
    result, rec, _ = await _run_runner(tmp_path, _consequential_script(), monkeypatch, capture=True)
    return result, rec


@pytest.mark.anyio
async def test_consequential_parent_retrieval_file_events(lockout):
    result, rec = lockout
    events = rec["events"]

    [write] = _events(events, "file_write")
    assert write["tool_call_id"] == "call_log_child"
    assert write["attribution"] == "tool_call"
    assert write["agent_id"] == "sub-1"  # via the tool_call_id -> tool_start join
    assert write["path"] == write["result_path"] == _CHILD_PATH
    assert write["sha256"] == _sha(CRITICAL_ORIGINAL)
    assert write["size"] == len(CRITICAL_ORIGINAL)
    assert write["error"] is None
    assert write["lc_run_id"] is None

    [read] = _events(events, "file_read")
    assert read["tool_call_id"] == "call_read_parent"
    assert read["attribution"] == "tool_call"
    assert read["agent_id"] == "root"
    assert read["path"] == _CHILD_PATH
    assert (read["offset"], read["limit"]) == (_READ_OFFSET, _READ_LIMIT)
    assert read["error"] is None
    assert read["created_at"] and read["modified_at"]
    raw_window = "\n".join(CRITICAL_ORIGINAL.splitlines()[_READ_OFFSET:_READ_OFFSET + _READ_LIMIT])
    assert CRITICAL_MARKER in raw_window
    assert read["window_sha256"] == _sha(raw_window)
    assert read["window_size"] == len(raw_window)

    # Ordering: write (child) < delegation_end < read (parent) < read_file tool_end.
    seq = {e["type"]: e["seq"] for e in events if e["type"] in {"file_write", "delegation_end", "file_read"}}
    assert seq["file_write"] < seq["delegation_end"] < seq["file_read"] < _tool_end(rec, "call_read_parent")["seq"]
    assert result.answer.startswith("Do not restart Chiller 6")


@pytest.mark.anyio
async def test_original_substitute_pairing(lockout):
    _, rec = lockout
    events = rec["events"]

    raw = _tool_end(rec, "call_log_child")
    assert raw["output"] == CRITICAL_ORIGINAL
    assert raw["output_sha256"] == _sha(CRITICAL_ORIGINAL)
    assert raw["output_size"] == len(CRITICAL_ORIGINAL)

    [committed] = _events(events, "tool_result_committed", tool_call_id="call_log_child")
    assert committed["agent_id"] == "sub-1"
    assert committed["tool"] == "get_event_log"
    assert committed["differs_from_raw"] is True
    assert committed["sha256"] != raw["output_sha256"]
    assert f"{_EVICT_PREFIX}call_log_child" in committed["content"]
    assert CRITICAL_MARKER not in committed["content"]
    # The committed substitute is what the child's next model input received.
    assert committed["content"] == _tool_msg(_llm_inputs(rec, "sub-1")[1], "call_log_child")["content"]

    # Small result: committed unchanged, so no content is duplicated.
    [small] = _events(events, "tool_result_committed", tool_call_id="call_read_parent")
    assert small["agent_id"] == "root"
    assert small["differs_from_raw"] is False
    assert small["sha256"] == _tool_end(rec, "call_read_parent")["output_sha256"]
    assert "content" not in small

    # task: no raw tool_end hash exists for a delegation.
    [task] = _events(events, "tool_result_committed", tool_call_id="call_task_1")
    assert task["differs_from_raw"] is None and "content" not in task


@pytest.mark.anyio
async def test_delegation_file_handback(lockout):
    _, rec = lockout
    [d] = rec["delegations"]
    expected = {"sha256": _sha(CRITICAL_ORIGINAL), "size": len(CRITICAL_ORIGINAL), "encoding": "utf-8"}

    meta = d["files_returned"][_CHILD_PATH]
    assert {k: meta[k] for k in expected} == expected
    assert meta["created_at"] and meta["modified_at"]
    assert "content" not in meta
    assert d["update_keys"] == ["files", "messages"]

    [end] = _events(rec["events"], "delegation_end")
    assert end["files_returned"] == d["files_returned"]
    assert end["update_keys"] == d["update_keys"]
    assert CRITICAL_MARKER not in json.dumps(end)


@pytest.mark.anyio
async def test_storage_is_not_delivery(lockout):
    _, rec = lockout
    events = rec["events"]
    assert not any("input_batches" in e for e in events if e["type"] in _FILE_EVENTS)

    root = [e for e in events if e["agent_id"] == "root"]
    handback_idx = next(i for i, e in enumerate(root) if e["type"] == "delegation_end")
    read_end_seq = _tool_end(rec, "call_read_parent")["seq"]
    read_end_idx = next(i for i, e in enumerate(root) if e["seq"] == read_end_seq)
    assert _CHILD_PATH in root[handback_idx]["files_returned"]  # stored and handed back...

    starts = [i for i, e in enumerate(root) if e["type"] == "llm_start"]
    before = [i for i in starts if i < read_end_idx]
    assert any(i > handback_idx for i in before)  # ...including a parent input after the hand-back
    for i in before:
        assert CRITICAL_MARKER not in json.dumps(root[i]["input_batches"])  # ...but not delivered

    first = next(i for i in starts if CRITICAL_MARKER in json.dumps(root[i]["input_batches"]))
    assert first > read_end_idx
    batch = root[first]["input_batches"][0]
    holders = [m for m in batch if CRITICAL_MARKER in json.dumps(m)]
    assert holders == [_tool_msg(batch, "call_read_parent")]


# -- 6. parallel tool-call attribution -------------------------------------------


@pytest.mark.anyio
async def test_parallel_tool_calls_attributed_by_runtime_id():
    # IDs contain "." so the sanitized eviction path differs from the id itself.
    calls = [
        {"name": "get_event_log", "args": {"asset": "Chiller 6"}, "id": "call.log.a"},
        {"name": "get_history", "args": {"asset": "Chiller 6"}, "id": "call.hist.b"},
    ]
    script = [AIMessage(content="", tool_calls=calls, usage_metadata=_USAGE), _ai("done")]
    capture = SubagentCaptureHandler()
    await _run_graph(script, backend=observing_backend_factory(StateBackend(), capture), capture=capture)

    writes = {e["tool_call_id"]: e for e in _events(capture.events, "file_write")}
    assert set(writes) == {"call.log.a", "call.hist.b"}
    assert writes["call.log.a"]["sha256"] == _sha(CRITICAL_ORIGINAL)
    assert writes["call.hist.b"]["sha256"] == _sha(ORIGINAL)
    assert writes["call.log.a"]["path"] == f"{_EVICT_PREFIX}call_log_a"
    assert writes["call.hist.b"]["path"] == f"{_EVICT_PREFIX}call_hist_b"
    assert all(w["agent_id"] == "root" and w["attribution"] == "tool_call" for w in writes.values())
    for call_id, w in writes.items():
        assert w["sha256"] == _events(capture.events, "tool_end", tool_call_id=call_id)[0]["output_sha256"]


# -- 7. failed eviction ----------------------------------------------------------


@pytest.mark.anyio
async def test_failed_eviction_passes_original_through():
    path = f"{_EVICT_PREFIX}call_x"
    script = [_ai(name="get_event_log", args={"asset": "Chiller 6"}, id="call_x"), _ai("done")]
    capture = SubagentCaptureHandler()
    state = await _run_graph(
        script,
        backend=observing_backend_factory(StateBackend(), capture),
        capture=capture,
        files={path: create_file_data("pre-existing")},
    )

    [write] = _events(capture.events, "file_write", tool_call_id="call_x")
    assert write["error"] and "already exists" in write["error"]
    assert write["result_path"] is None
    [committed] = _events(capture.events, "tool_result_committed", tool_call_id="call_x")
    assert committed["differs_from_raw"] is False and "content" not in committed

    # deepagents returns the original message unevicted; the model receives it.
    [_, second] = [e for e in capture.events if e["type"] == "llm_start"]
    assert _tool_msg(second["input_batches"][0], "call_x")["content"] == CRITICAL_ORIGINAL
    assert state["files"][path]["content"] == "pre-existing"


# -- 8. observer failure -----------------------------------------------------------


def _missing_read_script():
    return [_ai(**_read_call("call_missing", "/does/not/exist.txt")), _ai("done")]


@pytest.mark.anyio
async def test_observer_failure_never_affects_run(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("sink down")

    for script in (_consequential_script, _missing_read_script):
        plain = await _run_graph(script(), backend=StateBackend())
        capture = SubagentCaptureHandler()
        monkeypatch.setattr(capture, "record_file_op", boom)
        observed = await _run_graph(
            script(), backend=observing_backend_factory(StateBackend(), capture), capture=capture,
        )
        assert _final_messages(observed) == _final_messages(plain)
        assert _final_files(observed) == _final_files(plain)
        assert not any(e["type"].startswith("file_") for e in capture.events)

    # Inner backend errors surface exactly as without the observer, and are recorded.
    [missing] = [m for m in plain["messages"] if getattr(m, "tool_call_id", None) == "call_missing"]
    assert "not found" in missing.content
    capture = SubagentCaptureHandler()
    await _run_graph(_missing_read_script(), backend=observing_backend_factory(StateBackend(), capture),
                     capture=capture)
    [read] = _events(capture.events, "file_read", tool_call_id="call_missing")
    assert "not found" in read["error"] and read["window_sha256"] is None
