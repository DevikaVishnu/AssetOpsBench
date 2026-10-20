"""Tests for the optional sub-agent capture side channel.

These run the real ``create_deep_agent`` graph against a scripted fake chat
model, so the ``task`` delegation and the general-purpose sub-agent actually
execute.  No network calls or MCP subprocesses are started.
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


class _ScriptedModel(GenericFakeChatModel):
    """Fake chat model that accepts ``bind_tools`` and raises on ``"BOOM"``."""

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, *args, **kwargs):
        result = super()._generate(*args, **kwargs)
        if result.generations[0].message.content == "BOOM":
            raise RuntimeError("provider down")
        return result


@tool
def get_sensors(asset: str) -> str:
    """Return the sensors on an asset."""
    return "Supply Temperature, Return Temperature"


def _script(subagent_final: str = "Chiller 6 has 2 sensors."):
    # The parent and the general-purpose sub-agent share the model instance,
    # so responses are consumed in execution order.
    return iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {
                            "description": "List the sensors on Chiller 6",
                            "subagent_type": "general-purpose",
                        },
                        "id": "call_task_1",
                    }
                ],
                usage_metadata=_USAGE,
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_sensors", "args": {"asset": "Chiller 6"}, "id": "call_sub_1"}
                ],
                usage_metadata=_USAGE,
            ),
            AIMessage(content=subagent_final, usage_metadata=_USAGE),
            AIMessage(content="Chiller 6 has 2 sensors.", usage_metadata=_USAGE),
        ]
    )


def _client():
    client = MagicMock()
    client.get_tools = AsyncMock(return_value=[get_sensors])
    return client


async def _run(out_dir: Path, *, capture: bool, monkeypatch, script=None):
    monkeypatch.setenv("AGENT_TRAJECTORY_DIR", str(out_dir))
    if capture:
        monkeypatch.setenv("AGENTTRACE_CAPTURE_SUBAGENTS", "true")
    else:
        monkeypatch.delenv("AGENTTRACE_CAPTURE_SUBAGENTS", raising=False)
    set_run_context(run_id="run-1", scenario_id="s1")
    model = _ScriptedModel(messages=script if script is not None else _script())
    with (
        patch("agent.deep_agent.runner._build_chat_model", return_value=model),
        patch("langchain_mcp_adapters.client.MultiServerMCPClient", return_value=_client()),
    ):
        runner = DeepAgentRunner(server_paths={"iot": "iot-mcp-server"})
        return await runner.run("What sensors are on Chiller 6?")


def _native(out_dir: Path) -> dict:
    record = json.loads((out_dir / "run-1.json").read_text())
    record["trajectory"].pop("started_at")  # wall-clock, differs between runs
    return record


@pytest.fixture(autouse=True)
def _reset_ctx():
    _runspan._run_id_var.set(None)
    _runspan._scenario_id_var.set(None)
    yield
    _runspan._run_id_var.set(None)
    _runspan._scenario_id_var.set(None)


@pytest.mark.anyio
async def test_capture_off_writes_no_side_channel(tmp_path, monkeypatch):
    await _run(tmp_path, capture=False, monkeypatch=monkeypatch)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run-1.json"]


@pytest.mark.anyio
async def test_native_output_identical_with_capture_on_and_off(tmp_path, monkeypatch):
    off_dir, on_dir = tmp_path / "off", tmp_path / "on"
    off = await _run(off_dir, capture=False, monkeypatch=monkeypatch)
    on = await _run(on_dir, capture=True, monkeypatch=monkeypatch)

    assert on.answer == off.answer == "Chiller 6 has 2 sensors."
    assert _native(on_dir) == _native(off_dir)
    assert sorted(p.name for p in on_dir.iterdir()) == ["run-1.instrumented.json", "run-1.json"]


@pytest.mark.anyio
async def test_capture_records_delegation_and_subagent_internals(tmp_path, monkeypatch):
    await _run(tmp_path, capture=True, monkeypatch=monkeypatch)
    rec = json.loads((tmp_path / "run-1.instrumented.json").read_text())

    assert rec["schema"] == "assetopsbench.deepagent.instrumented/0.1"
    assert rec["status"] == "ok" and rec["error"] is None
    assert rec["native_trajectory_file"] == "run-1.json"

    agents = {a["agent_id"]: a for a in rec["agents"]}
    assert agents["root"]["kind"] == "root"
    assert agents["sub-1"] == {
        **agents["sub-1"],
        "name": "general-purpose",
        "kind": "subagent",
        "parent_agent_id": "root",
        "delegation_id": "call_task_1",
        "status": "ok",
    }

    [d] = rec["delegations"]
    assert d["delegation_id"] == "call_task_1"
    assert d["parent_agent_id"] == "root"
    assert d["parent_turn_index"] == 0
    assert d["subagent_type"] == "general-purpose"
    assert d["subagent_agent_id"] == "sub-1"
    assert d["description"] == "List the sensors on Chiller 6"
    assert d["status"] == "ok"
    assert d["returned_to_parent"] == "Chiller 6 has 2 sensors."
    assert [m["role"] for m in d["subagent_messages"]] == ["human", "ai", "tool", "ai"]
    assert d["subagent_messages"][0]["content"] == "List the sensors on Chiller 6"
    assert d["subagent_messages"][2]["tool_call_id"] == "call_sub_1"

    events = rec["events"]
    assert [e["seq"] for e in events] == list(range(len(events)))
    sub_tool = [e for e in events if e["agent_id"] == "sub-1" and e["type"] in ("tool_start", "tool_end")]
    assert [(e["type"], e["tool"]) for e in sub_tool] == [
        ("tool_start", "get_sensors"),
        ("tool_end", "get_sensors"),
    ]
    assert sub_tool[0]["args"] == {"asset": "Chiller 6"}
    assert sub_tool[1]["output"] == "Supply Temperature, Return Temperature"
    assert all(e["delegation_id"] == "call_task_1" for e in events if e["agent_id"] == "sub-1")

    # Delegation brackets the sub-agent's lifecycle in arrival order.
    order = [(e["type"], e["agent_id"]) for e in events]
    assert order.index(("delegation_start", "root")) < order.index(("agent_start", "sub-1"))
    assert order.index(("agent_end", "sub-1")) < order.index(("delegation_end", "root"))
    sub_llm_ends = [e for e in events if e["type"] == "llm_end" and e["agent_id"] == "sub-1"]
    assert len(sub_llm_ends) == 2
    assert sub_llm_ends[0]["usage"]["input_tokens"] == 10


@pytest.mark.anyio
async def test_failure_writes_partial_trace_and_reraises(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="provider down"):
        await _run(tmp_path, capture=True, monkeypatch=monkeypatch, script=_script("BOOM"))

    # Native behavior unchanged: no native trajectory on failure.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run-1.instrumented.json"]

    rec = json.loads((tmp_path / "run-1.instrumented.json").read_text())
    assert rec["status"] == "error"
    assert rec["error"]["type"] == "RuntimeError"
    assert rec["error"]["message"] == "provider down"
    assert rec["error"]["traceback"]

    [d] = rec["delegations"]
    assert d["subagent_type"] == "general-purpose"
    assert d["status"] == "error"
    assert d["parent_turn_index"] is None
    agents = {a["agent_id"]: a for a in rec["agents"]}
    assert agents["sub-1"]["status"] == "error"
    assert agents["root"]["status"] == "error"

    types = [(e["type"], e["agent_id"]) for e in rec["events"]]
    assert ("tool_end", "sub-1") in types  # captured before the failure
    assert ("llm_error", "sub-1") in types
    assert types[-1] == ("agent_error", "root")


@pytest.mark.anyio
async def test_failure_without_capture_writes_nothing(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="provider down"):
        await _run(tmp_path, capture=False, monkeypatch=monkeypatch, script=_script("BOOM"))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.anyio
async def test_rendered_model_inputs_captured(tmp_path, monkeypatch):
    """Verify parent and child model-input visibility."""
    await _run(tmp_path, capture=True, monkeypatch=monkeypatch)

    rec = json.loads(
        (tmp_path / "run-1.instrumented.json").read_text()
    )

    starts = [
        event for event in rec["events"]
        if event["type"] == "llm_start"
    ]

    assert len(starts) == 4
    assert all("input_batches" in event for event in starts)

    def contains(event, needle):
        return needle in json.dumps(event["input_batches"])

    child = [
        event for event in starts
        if event["agent_id"] == "sub-1"
    ]
    parent = [
        event for event in starts
        if event["agent_id"] == "root"
    ]

    assert len(child) == 2
    assert len(parent) == 2

    assert any(
        contains(event, "List the sensors on Chiller 6")
        for event in child
    )

    assert any(
        contains(event, "Supply Temperature, Return Temperature")
        for event in child
    )

    assert any(
        contains(event, "Chiller 6 has 2 sensors.")
        for event in parent
    )

    assert any(
        message["role"] == "system"
        for event in starts
        for batch in event["input_batches"]
        for message in batch
    )
