"""Probe: how deepagents offloads an oversized tool result, and what the
sub-agent capture side channel observes of it.

Runs the real ``create_deep_agent`` graph (via ``DeepAgentRunner``) against a
scripted fake chat model.  The eviction itself is not mocked: deepagents'
``FilesystemMiddleware.awrap_tool_call`` writes any tool result over
``NUM_CHARS_PER_TOKEN * tool_token_limit_before_evict`` characters (default
4 * 20,000) to ``/large_tool_results/<tool_call_id>`` in graph state and
substitutes a head/tail preview plus a file reference.

A unique marker sits in the middle of a >90K-character tool result, outside
the preview's 5 head and 5 tail lines.  The tests compare, for the same tool
call: the original output, the ``tool_end`` callback output, the next
``llm_start`` input, the tool message left in graph state, and the content
recovered with the framework's own ``read_file`` tool.

Assertions are joined by ``tool_call_id`` / ``lc_run_id`` and by which
``llm_start`` comes next for the same agent, never by global callback order.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from agent.deep_agent.runner import DeepAgentRunner
from observability import runspan as _runspan
from observability import set_run_context

_USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

_N_LINES = 1200
_MARKER_LINE = 600  # 0-indexed; far from the 5 head / 5 tail preview lines
MARKER = "AGENTTRACE-PROBE-MARKER-7Q3ZK9 peak_supply_temp=47.913"
_EVICT_PREFIX = "/large_tool_results/"
_READ_OFFSET, _READ_LIMIT = _MARKER_LINE - 5, 10


def _big_history(marker: str) -> str:
    lines = [
        f"obs {i:05d} | 2026-10-05T{i // 60 % 24:02d}:{i % 60:02d}:00Z | "
        f"Chiller 6 | Supply Temperature | {40 + (i % 97) / 10:.1f} | quality=good"
        for i in range(_N_LINES)
    ]
    lines[_MARKER_LINE] = marker
    return "\n".join(lines)


ORIGINAL = _big_history(MARKER)
_HEAD_LINE = ORIGINAL.splitlines()[0]
_TAIL_LINE = ORIGINAL.splitlines()[-1]


@tool
def get_history(asset: str) -> str:
    """Return the full sensor history of an asset."""
    return ORIGINAL


# Decision-critical variant: the only place in the run that says a restart is
# not permitted is one line in the middle of an oversized tool result.
LOCKOUT_ID = "LOCKOUT-RX7Q3"
CRITICAL_MARKER = f"{LOCKOUT_ID} compressor_lockout=ACTIVE restart_permitted=NO"
CRITICAL_ORIGINAL = _big_history(CRITICAL_MARKER)


@tool
def get_event_log(asset: str) -> str:
    """Return the full operational event log of an asset."""
    return CRITICAL_ORIGINAL


class _ScriptedModel(GenericFakeChatModel):
    """Fake chat model that accepts ``bind_tools``."""

    def bind_tools(self, tools, **kwargs):
        return self


def _ai(content: str = "", **tool_call) -> AIMessage:
    calls = [tool_call] if tool_call else []
    return AIMessage(content=content, tool_calls=calls, usage_metadata=_USAGE)


def _read_call(call_id: str, path: str) -> dict:
    return {
        "name": "read_file",
        "args": {"file_path": path, "offset": _READ_OFFSET, "limit": _READ_LIMIT},
        "id": call_id,
    }


async def _run(out_dir: Path, script, monkeypatch, question: str = "Peak supply temperature?"):
    monkeypatch.setenv("AGENT_TRAJECTORY_DIR", str(out_dir))
    monkeypatch.setenv("AGENTTRACE_CAPTURE_SUBAGENTS", "true")
    set_run_context(run_id="run-1", scenario_id="s1")
    client = MagicMock()
    client.get_tools = AsyncMock(return_value=[get_history, get_event_log])
    with (
        patch("agent.deep_agent.runner._build_chat_model", return_value=_ScriptedModel(messages=iter(script))),
        patch("langchain_mcp_adapters.client.MultiServerMCPClient", return_value=client),
    ):
        result = await DeepAgentRunner(server_paths={"iot": "iot-mcp-server"}).run(question)
    rec = json.loads((out_dir / "run-1.instrumented.json").read_text())
    return result, rec


@pytest.fixture(autouse=True)
def _reset_ctx():
    _runspan._run_id_var.set(None)
    _runspan._scenario_id_var.set(None)
    yield
    _runspan._run_id_var.set(None)
    _runspan._scenario_id_var.set(None)


# -- helpers over the instrumented record ------------------------------------


def _tool_end(rec: dict, tool_call_id: str) -> dict:
    """The ``tool_end`` event for *tool_call_id*, joined via its ``tool_start`` run id."""
    [start] = [e for e in rec["events"] if e["type"] == "tool_start" and e["tool_call_id"] == tool_call_id]
    [end] = [e for e in rec["events"] if e["type"] == "tool_end" and e["lc_run_id"] == start["lc_run_id"]]
    return end


def _llm_inputs(rec: dict, agent_id: str) -> list[list[dict]]:
    """Each ``llm_start`` input (first batch) of *agent_id*, in that agent's call order."""
    return [e["input_batches"][0] for e in rec["events"] if e["type"] == "llm_start" and e["agent_id"] == agent_id]


def _tool_msg(messages: list[dict], tool_call_id: str) -> dict:
    [msg] = [m for m in messages if m.get("role") == "tool" and m.get("tool_call_id") == tool_call_id]
    return msg


def _first_input_with(inputs: list[list[dict]], tool_call_id: str) -> list[dict]:
    """The first model input of an agent that contains the tool message for *tool_call_id*."""
    return next(batch for batch in inputs if any(m.get("tool_call_id") == tool_call_id for m in batch))


def _assert_substituted(content: str, tool_call_id: str, marker: str = MARKER) -> None:
    """*content* is deepagents' eviction substitute for *tool_call_id*, not the original."""
    assert marker not in content
    assert f"{_EVICT_PREFIX}{tool_call_id}" in content
    assert _HEAD_LINE in content and _TAIL_LINE in content  # head/tail preview
    assert f"[{_N_LINES - 10} lines truncated]" in content
    assert len(content) < len(ORIGINAL) // 10


def _assert_recovered(content: str, marker: str = MARKER) -> None:
    """*content* is a ``read_file`` window around the marker, with line numbers."""
    assert marker in content
    assert f"{_MARKER_LINE + 1}\t{marker}" in content  # cat -n style, 1-indexed
    assert len(content) < 2_000


# -- single agent --------------------------------------------------------------


def _single_agent_script():
    return [
        _ai(name="get_history", args={"asset": "Chiller 6"}, id="call_big_1"),
        _ai(**_read_call("call_read_1", f"{_EVICT_PREFIX}call_big_1")),
        _ai("Peak supply temperature was 47.913."),
    ]


def test_synthetic_result_exceeds_eviction_threshold():
    from deepagents.middleware.filesystem import NUM_CHARS_PER_TOKEN

    assert len(ORIGINAL) > 90_000 > NUM_CHARS_PER_TOKEN * 20_000
    assert MARKER not in "\n".join(ORIGINAL.splitlines()[:5] + ORIGINAL.splitlines()[-5:])


@pytest.mark.anyio
async def test_offloaded_result_observability_single_agent(tmp_path, monkeypatch):
    result, rec = await _run(tmp_path, _single_agent_script(), monkeypatch)
    root_inputs = _llm_inputs(rec, "root")
    assert len(root_inputs) == 3

    # 1. on_tool_end sees the ORIGINAL output: eviction happens in
    #    awrap_tool_call after the tool (and its callbacks) have finished.
    assert _tool_end(rec, "call_big_1")["output"] == ORIGINAL

    # 2. The next model input carries only the substitute: preview + file path.
    next_input = _first_input_with(root_inputs, "call_big_1")
    assert next_input is root_inputs[1]
    _assert_substituted(_tool_msg(next_input, "call_big_1")["content"], "call_big_1")
    assert MARKER not in json.dumps(next_input)

    # 3. The tool message left in graph state (native trajectory) is the substitute too.
    [big_call] = [c for c in result.trajectory.all_tool_calls if c.id == "call_big_1"]
    _assert_substituted(big_call.output, "call_big_1")

    # 4. read_file on the offload path recovers the marker line verbatim...
    recovered = _tool_end(rec, "call_read_1")["output"]
    _assert_recovered(recovered)

    # 5. ...and that recovery reaches the next model input, unchanged.
    after_read = _first_input_with(root_inputs, "call_read_1")
    assert after_read is root_inputs[2]
    assert _tool_msg(after_read, "call_read_1")["content"] == recovered

    # The earlier substitute is still what the model sees for call_big_1.
    _assert_substituted(_tool_msg(after_read, "call_big_1")["content"], "call_big_1")

    # Callbacks alone do not record the eviction itself; it is visible only through
    # the file-observability events (tested in test_file_observability.py).
    file_events = {"file_write", "file_edit", "file_read", "tool_result_committed"}
    assert not any(e["type"] not in {"agent_start", "agent_end", "llm_start", "llm_end", "tool_start", "tool_end"}
                   for e in rec["events"] if e["type"] not in file_events)


# -- parent -> child delegation --------------------------------------------------


def _delegation_script():
    # Parent and general-purpose sub-agent share the model, so responses are
    # consumed in execution order.
    child_path = f"{_EVICT_PREFIX}call_big_child"
    return [
        _ai(name="task", args={"description": "Find the peak supply temperature of Chiller 6",
                               "subagent_type": "general-purpose"}, id="call_task_1"),
        _ai(name="get_history", args={"asset": "Chiller 6"}, id="call_big_child"),  # child
        _ai(**_read_call("call_read_child", child_path)),  # child
        _ai("Peak supply temperature was 47.913."),  # child final -> returned to parent
        _ai(**_read_call("call_read_parent", child_path)),  # parent reads the child's offload file
        _ai("Peak supply temperature was 47.913."),  # parent final
    ]


@pytest.mark.anyio
async def test_offloaded_result_observability_with_delegation(tmp_path, monkeypatch):
    result, rec = await _run(tmp_path, _delegation_script(), monkeypatch)
    [d] = rec["delegations"]
    assert d["delegation_id"] == "call_task_1" and d["subagent_agent_id"] == "sub-1"
    child_inputs = _llm_inputs(rec, "sub-1")
    parent_inputs = _llm_inputs(rec, "root")
    assert len(child_inputs) == 3 and len(parent_inputs) == 3

    # Child: tool_end has the original; its next model input has the substitute.
    big_end = _tool_end(rec, "call_big_child")
    assert big_end["agent_id"] == "sub-1"
    assert big_end["output"] == ORIGINAL
    child_next = _first_input_with(child_inputs, "call_big_child")
    assert child_next is child_inputs[1]
    _assert_substituted(_tool_msg(child_next, "call_big_child")["content"], "call_big_child")
    assert MARKER not in json.dumps(child_next)

    # Child's final message list (captured at agent_end) holds the substitute.
    _assert_substituted(_tool_msg(d["subagent_messages"], "call_big_child")["content"], "call_big_child")

    # Child recovers the marker with read_file, and it reaches the child's next input.
    child_read = _tool_end(rec, "call_read_child")
    assert child_read["agent_id"] == "sub-1"
    _assert_recovered(child_read["output"])
    assert _tool_msg(child_inputs[2], "call_read_child")["content"] == child_read["output"]

    # Handoff: only the child's final text returns to the parent.
    assert d["returned_to_parent"] == "Peak supply temperature was 47.913."
    assert all(MARKER not in json.dumps(batch) for batch in parent_inputs[:2])

    # The child's offload file is in the parent's state after the delegation:
    # the parent's own read_file on the child's path recovers the same lines.
    parent_read = _tool_end(rec, "call_read_parent")
    assert parent_read["agent_id"] == "root"
    _assert_recovered(parent_read["output"])
    assert parent_read["output"] == child_read["output"]
    assert _tool_msg(parent_inputs[2], "call_read_parent")["content"] == parent_read["output"]

    # Neither the original nor the substitute of the child's big result ever
    # reaches a parent model input; only the read_file window does.
    for batch in parent_inputs:
        assert not any(m.get("tool_call_id") == "call_big_child" for m in batch)
        assert _HEAD_LINE not in json.dumps(batch) and "lines truncated" not in json.dumps(batch)
    assert not any(c.id == "call_big_child" for c in result.trajectory.all_tool_calls)


# -- consequential flow: the parent's decision rests on a fact the handoff omits --
#
# The scripted model makes every "decision" here; nothing below is evidence
# that an agent reasoned its way to reading the file.  The test establishes
# only the observable information flow: where the decisive line was, which
# model inputs it did and did not reach, and through which tool output it
# reached the input that precedes the parent's final answer.

_CHILD_PATH = f"{_EVICT_PREFIX}call_log_child"
HANDOFF = (
    "Retrieved the Chiller 6 operational event log (1200 entries). "
    f"The full log was saved at {_CHILD_PATH}; I reviewed only the preview."
)
QUESTION = "Is it safe to restart Chiller 6 now?"
FINAL = f"Do not restart Chiller 6: {LOCKOUT_ID} is active (restart_permitted=NO)."


def _consequential_script():
    return [
        # P1 (parent): delegate.
        _ai(name="task", args={"description": "Fetch the Chiller 6 event log",
                               "subagent_type": "general-purpose"}, id="call_task_1"),
        # C1 (child): oversized tool result, evicted by the framework.
        _ai(name="get_event_log", args={"asset": "Chiller 6"}, id="call_log_child"),
        # C2 (child): hand off without reading the file.
        _ai(HANDOFF),
        # P2 (parent): read the child's offloaded file itself.
        _ai(**_read_call("call_read_parent", _CHILD_PATH)),
        # P3 (parent): final answer that states the recovered fact.
        _ai(FINAL),
    ]


def _agent_events(rec: dict, agent_id: str) -> list[dict]:
    return [e for e in rec["events"] if e["agent_id"] == agent_id]


@pytest.mark.anyio
async def test_parent_decision_depends_on_fact_recovered_from_child_offload(tmp_path, monkeypatch):
    result, rec = await _run(tmp_path, _consequential_script(), monkeypatch, question=QUESTION)
    [d] = rec["delegations"]
    assert (d["delegation_id"], d["parent_agent_id"], d["subagent_agent_id"]) == ("call_task_1", "root", "sub-1")

    meaning = (LOCKOUT_ID, "lockout", "restart_permitted", "restart")

    # -- child: original -> substitute -> handoff ------------------------------
    child_inputs = _llm_inputs(rec, "sub-1")
    assert len(child_inputs) == 2  # C1, C2

    original = _tool_end(rec, "call_log_child")
    assert original["agent_id"] == "sub-1"
    assert original["output"] == CRITICAL_ORIGINAL

    substitute = _tool_msg(child_inputs[1], "call_log_child")["content"]
    _assert_substituted(substitute, "call_log_child", CRITICAL_MARKER)
    assert _CHILD_PATH in substitute
    # The child's model never received the decisive line in any input.
    assert all(CRITICAL_MARKER not in json.dumps(batch) for batch in child_inputs)

    child_final = [e for e in _agent_events(rec, "sub-1") if e["type"] == "llm_end"][-1]
    assert child_final["message"]["content"] == HANDOFF
    assert d["returned_to_parent"] == HANDOFF
    assert not any(word.lower() in HANDOFF.lower() for word in meaning)
    assert _CHILD_PATH in HANDOFF  # carried from the substitute

    # -- parent: model inputs before and after its own read_file ---------------
    parent_inputs = _llm_inputs(rec, "root")
    assert len(parent_inputs) == 3  # P1, P2, P3
    p1, p2, p3 = parent_inputs

    # The handoff is delivered to P2 as the task tool's message, verbatim.
    assert _tool_msg(p2, "call_task_1")["content"] == HANDOFF

    # The marker is absent from every parent input before the read.
    for batch in (p1, p2):
        assert CRITICAL_MARKER not in json.dumps(batch)
        assert LOCKOUT_ID not in json.dumps(batch)

    # P2's output is the read_file call on the path the handoff named.
    parent_events = _agent_events(rec, "root")
    p2_end = [e for e in parent_events if e["type"] == "llm_end"][1]
    assert p2_end["message"]["tool_calls"] == [
        {"id": "call_read_parent", "name": "read_file",
         "args": {"file_path": _CHILD_PATH, "offset": _READ_OFFSET, "limit": _READ_LIMIT}},
    ]

    read = _tool_end(rec, "call_read_parent")
    assert read["agent_id"] == "root"
    _assert_recovered(read["output"], CRITICAL_MARKER)

    # First appearance: within the parent's own event sequence, the first
    # llm_start whose input holds the marker comes after the read_file
    # tool_end, and the marker sits only in that tool's message.
    starts = [i for i, e in enumerate(parent_events) if e["type"] == "llm_start"]
    first_with_marker = next(i for i in starts if CRITICAL_MARKER in json.dumps(parent_events[i]["input_batches"]))
    read_end_idx = next(i for i, e in enumerate(parent_events) if e is read)
    assert read_end_idx < first_with_marker
    assert parent_events[first_with_marker]["input_batches"][0] == p3
    holders = [m for m in p3 if CRITICAL_MARKER in json.dumps(m)]
    assert holders == [_tool_msg(p3, "call_read_parent")]
    assert holders[0]["content"] == read["output"]

    # -- parent final response states the recovered fact ------------------------
    final_end = [e for e in parent_events if e["type"] == "llm_end"][-1]
    assert final_end["message"]["content"] == FINAL == result.answer
    assert LOCKOUT_ID in FINAL and "restart_permitted=NO" in FINAL
    assert "restart_permitted=NO" in read["output"]  # the only parent input source of that span

    # Native trajectory agrees on the parent side; the child's big tool call is not in it.
    calls = {c.id: c for c in result.trajectory.all_tool_calls}
    assert calls["call_task_1"].output == HANDOFF
    assert calls["call_read_parent"].output == read["output"]
    assert "call_log_child" not in calls
