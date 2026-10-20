"""Tests for the optional ``AGENTTRACE_SPECIALISTS`` research configuration.

M1 (specialist sub-agents), M4 (read-only WO connection) and M6 (offered-tool
capture).  The runner tests use the real ``create_deep_agent`` graph with a
scripted fake chat model and a fake MCP client.  The M4 server test starts the
real WO MCP server over stdio and only lists tools / calls an unregistered
tool, so no CouchDB request is made.  No network or LLM calls.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import deepagents
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from agent.deep_agent import _specialists as spec
from agent.deep_agent._subagent_capture import _bound_tool_names
from agent.deep_agent.runner import DeepAgentRunner, _build_mcp_connections
from agent.runner import DEFAULT_SERVER_PATHS
from observability import runspan as _runspan
from observability import set_run_context

_SRC = Path(__file__).resolve().parents[3]
_USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
_FRAMEWORK_TOOLS = {"write_todos", "ls", "read_file", "write_file", "edit_file", "glob", "grep"}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# -- fakes ---------------------------------------------------------------------


@tool
def iot_tool(asset: str) -> str:
    """Fake IoT tool."""
    return f"iot:{asset}"


@tool
def tsfm_tool(x: str) -> str:
    """Fake TSFM tool."""
    return "tsfm"


@tool
def fmsr_tool(x: str) -> str:
    """Fake FMSR tool."""
    return "fmsr"


@tool
def wo_tool(wonum: str) -> str:
    """Fake WO read tool."""
    return f"WO {wonum}: status WAPPR"


@tool
def util_tool(x: str) -> str:
    """Fake utilities tool."""
    return "util"


@tool
def vib_tool(x: str) -> str:
    """Fake vibration tool."""
    return "vib"


_BY_SERVER = {
    "iot": [iot_tool],
    "utilities": [util_tool],
    "fmsr": [fmsr_tool],
    "tsfm": [tsfm_tool],
    "wo": [wo_tool],
    "vibration": [vib_tool],
}


def _client():
    async def get_tools(*, server_name=None):
        if server_name is None:
            return [t for name in DEFAULT_SERVER_PATHS for t in _BY_SERVER[name]]
        return list(_BY_SERVER[server_name])

    client = MagicMock()
    client.get_tools = AsyncMock(side_effect=get_tools)
    return client


class _BindingModel(GenericFakeChatModel):
    """Fake chat model whose bound tools reach ``invocation_params`` like a real
    provider model's (``bind(tools=...)``), in Anthropic's ``{"name"}`` shape."""

    def bind_tools(self, tools, **kwargs):
        names = [t["name"] if isinstance(t, dict) else t.name for t in tools]
        return self.bind(tools=[{"name": n} for n in names])


def _script():
    # Parent, specialist and general-purpose share the model: execution order.
    return iter(
        [
            AIMessage(  # parent -> wo specialist
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {"description": "Look up work order 42", "subagent_type": "wo"},
                        "id": "call_task_wo",
                    }
                ],
                usage_metadata=_USAGE,
            ),
            AIMessage(  # wo specialist uses its tool
                content="",
                tool_calls=[{"name": "wo_tool", "args": {"wonum": "42"}, "id": "call_wo_1"}],
                usage_metadata=_USAGE,
            ),
            AIMessage(content="Work order 42 is WAPPR.", usage_metadata=_USAGE),  # wo final
            AIMessage(  # parent -> general-purpose
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {"description": "Summarise", "subagent_type": "general-purpose"},
                        "id": "call_task_gp",
                    }
                ],
                usage_metadata=_USAGE,
            ),
            AIMessage(content="Summary done.", usage_metadata=_USAGE),  # gp final
            AIMessage(content="WO 42 is awaiting approval.", usage_metadata=_USAGE),  # parent final
        ]
    )


@pytest.fixture(autouse=True)
def _reset_ctx():
    _runspan._run_id_var.set(None)
    _runspan._scenario_id_var.set(None)
    yield
    _runspan._run_id_var.set(None)
    _runspan._scenario_id_var.set(None)


async def _run(out_dir: Path, monkeypatch, *, specialists: str | None, capture: bool, script=None):
    monkeypatch.setenv("AGENT_TRAJECTORY_DIR", str(out_dir))
    if capture:
        monkeypatch.setenv("AGENTTRACE_CAPTURE_SUBAGENTS", "1")
    else:
        monkeypatch.delenv("AGENTTRACE_CAPTURE_SUBAGENTS", raising=False)
    if specialists is None:
        monkeypatch.delenv(spec.SPECIALISTS_ENV, raising=False)
    else:
        monkeypatch.setenv(spec.SPECIALISTS_ENV, specialists)
    set_run_context(run_id="run-1", scenario_id="s1")

    model = _BindingModel(messages=script if script is not None else _script())
    client = _client()
    real_create = deepagents.create_deep_agent
    with (
        patch("agent.deep_agent.runner._build_chat_model", return_value=model),
        patch("langchain_mcp_adapters.client.MultiServerMCPClient", return_value=client) as mcp_cls,
        patch("deepagents.create_deep_agent", wraps=real_create) as create,
    ):
        result = await DeepAgentRunner().run("What is the status of WO 42?")
    return result, create, mcp_cls, client


def _flag_off_script():
    return iter(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "wo_tool", "args": {"wonum": "42"}, "id": "call_wo_1"}],
                usage_metadata=_USAGE,
            ),
            AIMessage(content="WO 42 is awaiting approval.", usage_metadata=_USAGE),
        ]
    )


def _strip_volatile(obj):
    """Drop wall-clock / process-local fields so two runs can be compared."""
    volatile = {"ts", "mono_ns", "lc_run_id", "lc_parent_run_id", "started_at", "ended_at",
                "checkpoint_ns", "versions"}
    if isinstance(obj, dict):
        return {k: _strip_volatile(v) for k, v in obj.items() if k not in volatile}
    if isinstance(obj, list):
        return [_strip_volatile(v) for v in obj]
    return obj


# -- flag-off compatibility ------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "0", "false", "off"])
def test_flag_parsing_off(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(spec.SPECIALISTS_ENV, raising=False)
    else:
        monkeypatch.setenv(spec.SPECIALISTS_ENV, value)
    assert spec.specialists_enabled() is False


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.anyio
async def test_flag_off_construction_unchanged(tmp_path, monkeypatch, capture):
    _, create, mcp_cls, client = await _run(
        tmp_path, monkeypatch, specialists=None, capture=capture, script=_flag_off_script()
    )
    # Same connections as before (no env on any server), one get_tools() call.
    mcp_cls.assert_called_once_with(_build_mcp_connections(DEFAULT_SERVER_PATHS))
    client.get_tools.assert_awaited_once_with()
    kwargs = create.call_args.kwargs
    expected = {"model", "tools", "system_prompt"} | ({"backend"} if capture else set())
    assert set(kwargs) == expected
    assert [t.name for t in kwargs["tools"]] == [
        "iot_tool", "util_tool", "fmsr_tool", "tsfm_tool", "wo_tool", "vib_tool"
    ]


@pytest.mark.anyio
async def test_flag_off_outputs_identical_to_unset(tmp_path, monkeypatch):
    unset, zero = tmp_path / "unset", tmp_path / "zero"
    await _run(unset, monkeypatch, specialists=None, capture=True, script=_flag_off_script())
    await _run(zero, monkeypatch, specialists="0", capture=True, script=_flag_off_script())

    for name in ("run-1.json", "run-1.instrumented.json"):
        a = _strip_volatile(json.loads((unset / name).read_text()))
        b = _strip_volatile(json.loads((zero / name).read_text()))
        assert a == b
    rec = json.loads((unset / "run-1.instrumented.json").read_text())
    assert "tool_availability" not in rec
    assert all("offered_tools" not in e for e in rec["events"])


# -- M1: specialist definitions and partitioning ----------------------------------


def test_build_specialists_partition():
    specs = spec.build_specialists(_BY_SERVER)
    assert [s["name"] for s in specs] == ["iot", "tsfm", "fmsr", "wo"]
    tools = {s["name"]: [t.name for t in s["tools"]] for s in specs}
    assert tools == {
        "iot": ["iot_tool", "util_tool"],
        "tsfm": ["tsfm_tool", "util_tool"],
        "fmsr": ["fmsr_tool", "util_tool"],
        "wo": ["wo_tool", "util_tool"],
    }
    assert all("vib_tool" not in names for names in tools.values())
    assert all("model" not in s for s in specs)  # inherited from the coordinator


def test_build_specialists_skips_missing_servers():
    specs = spec.build_specialists({"wo": [wo_tool]})
    assert [(s["name"], [t.name for t in s["tools"]]) for s in specs] == [("wo", ["wo_tool"])]


def test_scope_lines_verbatim_from_server_code():
    """Each scope equals the server's ``FastMCP(instructions=...)`` (parsed, not imported)."""
    for name, scope in spec.SERVER_SCOPES.items():
        tree = ast.parse((_SRC / "servers" / name / "main.py").read_text())
        found = [
            ast.literal_eval(kw.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "FastMCP"
            for kw in node.keywords
            if kw.arg == "instructions"
        ]
        assert found == [scope], name


def test_specialist_prompts_are_neutral():
    template = spec._PROMPT_TEMPLATE.lower()
    for word in ("chiller", "scenario", "1000045", "error", "file", "csv", "json", "anomal"):
        assert word not in template
    for name in ("iot", "tsfm", "fmsr"):
        assert spec.SERVER_SCOPES[name] in spec.specialist_prompt(name)


def test_wo_specialist_not_told_it_can_write():
    by_name = {s["name"]: s for s in spec.build_specialists(_BY_SERVER)}
    for text in (by_name["wo"]["system_prompt"], by_name["wo"]["description"]):
        assert spec.READONLY_SCOPES["wo"] in text
        assert spec.SERVER_SCOPES["wo"] not in text
        for verb in ("create", "approve", "assign", "close", "cancel", "update", "generate"):
            assert verb not in text.lower()
    # Provenance: the verbatim server text is kept and recorded by hash.
    ta = spec.tool_availability([wo_tool], list(by_name.values()), {})
    wo = next(a for a in ta["agents"] if a["name"] == "wo")
    assert wo["scope_source"] == "readonly_edit"
    assert wo["server_instructions_sha256"] == _sha(spec.SERVER_SCOPES["wo"])
    iot = next(a for a in ta["agents"] if a["name"] == "iot")
    assert iot["scope_source"] == "server_instructions"


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.anyio
async def test_flag_on_both_construction_paths_get_subagents(tmp_path, monkeypatch, capture):
    _, create, mcp_cls, client = await _run(tmp_path, monkeypatch, specialists="1", capture=capture)
    kwargs = create.call_args.kwargs
    assert [s["name"] for s in kwargs["subagents"]] == ["iot", "tsfm", "fmsr", "wo"]
    # Coordinator keeps every original tool, in the original order, and its prompt.
    assert [t.name for t in kwargs["tools"]] == [
        "iot_tool", "util_tool", "fmsr_tool", "tsfm_tool", "wo_tool", "vib_tool"
    ]
    from agent._prompts import AGENT_SYSTEM_PROMPT

    assert kwargs["system_prompt"] == AGENT_SYSTEM_PROMPT
    assert ("backend" in kwargs) is capture
    # Tools are loaded per server, never via the all-servers call.
    assert sorted(c.kwargs["server_name"] for c in client.get_tools.await_args_list) == sorted(
        DEFAULT_SERVER_PATHS
    )


# -- M4: read-only WO ------------------------------------------------------------------


def _clear_server_env(monkeypatch):
    for names in spec.SERVER_ENV.values():
        for k in names:
            monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("AOB_READONLY", raising=False)


@pytest.mark.anyio
async def test_flag_on_only_wo_connection_carries_readonly_env(tmp_path, monkeypatch):
    _clear_server_env(monkeypatch)
    _, _, mcp_cls, _ = await _run(tmp_path, monkeypatch, specialists="1", capture=False)
    connections = mcp_cls.call_args.args[0]
    assert connections["wo"]["env"] == {"AOB_READONLY": "1"}
    assert all("env" not in c for n, c in connections.items() if n != "wo")
    # Everything else on the WO connection is unchanged.
    base = _build_mcp_connections(DEFAULT_SERVER_PATHS)
    assert {k: v for k, v in connections["wo"].items() if k != "env"} == base["wo"]


_FAKE_ENV = {
    "COUCHDB_URL": "http://couch.invalid:5984",
    "COUCHDB_USERNAME": "user-sentinel",
    "COUCHDB_PASSWORD": "pw-sentinel-7f3a",
    "IOT_DBNAME": "iot",
    "WO_DBNAME": "workorder",
    "FAILURE_CODE_DBNAME": "failure_code",
    "ANTHROPIC_API_KEY": "anthropic-sentinel-91c2",
    "WATSONX_APIKEY": "watsonx-sentinel",
    "LITELLM_API_KEY": "litellm-sentinel",
    "OTEL_TRACES_FILE": "/should/not/forward.jsonl",
    "AOB_READONLY": "0",
}


def test_forwarded_env_is_per_server_allowlist():
    base = _build_mcp_connections(DEFAULT_SERVER_PATHS)
    snapshot = json.loads(json.dumps(base))
    conns = spec.with_readonly_wo(spec.with_forwarded_env(base, _FAKE_ENV))
    assert base == snapshot  # input not mutated
    couch = {"COUCHDB_URL", "COUCHDB_USERNAME", "COUCHDB_PASSWORD"}
    assert set(conns["iot"]["env"]) == couch | {"IOT_DBNAME"}
    assert set(conns["wo"]["env"]) == couch | {"WO_DBNAME", "FAILURE_CODE_DBNAME", "AOB_READONLY"}
    assert conns["wo"]["env"]["AOB_READONLY"] == "1"  # M4 wins over the runner's value
    for name in ("utilities", "fmsr", "tsfm", "vibration"):
        assert set(conns[name]["env"]) == couch
    for conn in conns.values():
        for k in ("ANTHROPIC_API_KEY", "WATSONX_APIKEY", "LITELLM_API_KEY", "OTEL_TRACES_FILE"):
            assert k not in conn["env"]
    # Everything except env is unchanged.
    assert all({k: v for k, v in conns[n].items() if k != "env"} == base[n] for n in base)


def test_forwarded_env_skips_unset_and_empty():
    conns = spec.with_forwarded_env(
        _build_mcp_connections({"fmsr": "fmsr-mcp-server"}), {"COUCHDB_URL": "", "LOG_LEVEL": "INFO"}
    )
    assert conns["fmsr"]["env"] == {"LOG_LEVEL": "INFO"}


@pytest.mark.anyio
async def test_forwarded_values_never_recorded(tmp_path, monkeypatch):
    _clear_server_env(monkeypatch)
    for k, v in _FAKE_ENV.items():
        if k != "OTEL_TRACES_FILE":
            monkeypatch.setenv(k, v)
    _, _, mcp_cls, _ = await _run(tmp_path, monkeypatch, specialists="1", capture=True)
    conns = mcp_cls.call_args.args[0]
    assert conns["wo"]["env"]["COUCHDB_PASSWORD"] == "pw-sentinel-7f3a"  # forwarded...
    for path in tmp_path.iterdir():  # ...but never written to any output file
        text = path.read_text()
        for secret in ("pw-sentinel-7f3a", "user-sentinel", "anthropic-sentinel-91c2", "couch.invalid"):
            assert secret not in text, (path.name, secret)
    rec = json.loads((tmp_path / "run-1.instrumented.json").read_text())
    assert rec["tool_availability"]["forwarded_env_names"]["wo"] == [
        "COUCHDB_PASSWORD", "COUCHDB_URL", "COUCHDB_USERNAME", "FAILURE_CODE_DBNAME", "WO_DBNAME"
    ]


def test_server_env_allowlist_covers_what_servers_read():
    """Every env var a server's code reads is either forwarded or deliberately excluded."""
    import re

    excluded = {"AOB_READONLY", "FMSR_MODEL_ID", "WATSONX_APIKEY", "TSFM_STORE", "TSFM_SEEDS_DIR"}
    pattern = re.compile(r"""(?:environ(?:\.get)?\(\s*|environ\[|getenv\(\s*|COLLECTION_ENV_VAR\s*=\s*)["']([A-Z_0-9]+)["']""")
    for name, allowed in spec.SERVER_ENV.items():
        read = set()
        for f in (_SRC / "servers" / name).rglob("*.py"):
            if "tests" not in f.parts:
                read |= set(pattern.findall(f.read_text()))
        assert read - set(allowed) - excluded == set(), name


def test_with_readonly_wo_does_not_mutate_input():
    base = _build_mcp_connections(DEFAULT_SERVER_PATHS)
    snapshot = json.loads(json.dumps(base))
    spec.with_readonly_wo(base)
    assert base == snapshot


_WO_READ = {
    "list_workorders", "get_workorder", "get_workorder_tasks", "get_workorder_costs",
    "get_workorder_actuals_vs_planned", "get_workorder_kpis", "get_schedule_calendar",
    "get_my_assigned_workorders", "get_failure_codes",
}
_WO_WRITE = {
    "generate_work_order", "update_workorder", "approve_workorder", "assign_technician",
    "close_workorder", "cancel_workorder",
}


def _wo_stdio_connection(readonly: bool) -> dict:
    """The runner's WO connection, run with this interpreter instead of ``uv run``.

    ``PYTHONPATH`` is added because stdio servers get only a default env; the
    runner's ``uv run`` resolves the package itself.  No CouchDB variables are set.
    """
    conns = _build_mcp_connections({"wo": "wo-mcp-server"})
    if readonly:
        conns = spec.with_readonly_wo(conns)
    conn = conns["wo"]
    conn["command"] = sys.executable
    conn["args"] = ["-m", "servers.wo.main"]
    conn["env"] = {**(conn.get("env") or {}), "PYTHONPATH": str(_SRC)}
    return conn


@pytest.mark.anyio
async def test_wo_server_with_readonly_env_exposes_only_read_tools():
    from langchain_mcp_adapters.client import MultiServerMCPClient

    ro = MultiServerMCPClient({"wo": _wo_stdio_connection(readonly=True)})
    ro_names = {t.name for t in await ro.get_tools(server_name="wo")}
    assert ro_names == _WO_READ

    rw = MultiServerMCPClient({"wo": _wo_stdio_connection(readonly=False)})
    assert {t.name for t in await rw.get_tools(server_name="wo")} == _WO_READ | _WO_WRITE

    # A write tool cannot be invoked: it is not registered on the server.
    async with ro.session("wo") as session:
        result = await session.call_tool("cancel_workorder", {"wonum": "0"})
    assert result.isError
    assert "unknown tool" in str(result.content).lower()


# -- M6 + delegation capture with specialist names ------------------------------------


def test_bound_tool_names_formats():
    assert _bound_tool_names(None) is None
    assert _bound_tool_names({"model": "x"}) is None
    assert _bound_tool_names({"tools": [{"name": "a"}, {"type": "function", "function": {"name": "b"}}]}) == [
        "a", "b"
    ]
    assert _bound_tool_names({"tools": [iot_tool]}) == ["iot_tool"]


@pytest.mark.anyio
async def test_specialist_delegation_capture(tmp_path, monkeypatch):
    result, *_ = await _run(tmp_path, monkeypatch, specialists="1", capture=True)
    assert result.answer == "WO 42 is awaiting approval."
    rec = json.loads((tmp_path / "run-1.instrumented.json").read_text())
    events = rec["events"]

    # Distinct agent identities.
    agents = {a["agent_id"]: a for a in rec["agents"]}
    assert {k: (a["name"], a["kind"], a["parent_agent_id"]) for k, a in agents.items()} == {
        "root": ("deep-agent", "root", None),
        "sub-1": ("wo", "subagent", "root"),
        "sub-2": ("general-purpose", "subagent", "root"),
    }

    # delegation_start / delegation_end correlation.
    starts = [e for e in events if e["type"] == "delegation_start"]
    ends = [e for e in events if e["type"] == "delegation_end"]
    assert [(e["tool_call_id"], e["subagent_type"]) for e in starts] == [
        ("call_task_wo", "wo"), ("call_task_gp", "general-purpose")
    ]
    assert [e["lc_run_id"] for e in ends] == [e["lc_run_id"] for e in starts]
    delegations = {d["delegation_id"]: d for d in rec["delegations"]}
    assert delegations["call_task_wo"]["subagent_agent_id"] == "sub-1"
    assert delegations["call_task_wo"]["parent_turn_index"] == 0
    assert agents["sub-1"]["delegation_id"] == "call_task_wo"

    # Child rendered input: own agent_id, specialist prompt, description as HumanMessage.
    wo_llm = [e for e in events if e["type"] == "llm_start" and e["agent_id"] == "sub-1"]
    assert len(wo_llm) == 2
    first = wo_llm[0]["input_batches"][0]
    assert first[0]["role"] == "system"
    content = first[0]["content"]
    system_text = content if isinstance(content, str) else "".join(b.get("text", "") for b in content)
    assert spec.specialist_prompt("wo") in system_text
    humans = [m for m in first if m["role"] == "human"]
    assert len(humans) == 1 and _sha(humans[0]["content"]) == _sha(starts[0]["description"])
    wo_calls = [e for e in events if e["type"] == "tool_start" and e["tool"] == "wo_tool"]
    assert [e["agent_id"] for e in wo_calls] == ["sub-1"]

    # Child -> parent delivery, verified from the parent's next rendered input.
    returned = delegations["call_task_wo"]["returned_to_parent"]
    assert returned == "Work order 42 is WAPPR."
    end_seq = ends[0]["seq"]
    next_parent = next(
        e for e in events if e["type"] == "llm_start" and e["agent_id"] == "root" and e["seq"] > end_seq
    )
    tool_msgs = [
        m for m in next_parent["input_batches"][0]
        if m["role"] == "tool" and m["tool_call_id"] == "call_task_wo"
    ]
    assert len(tool_msgs) == 1 and _sha(tool_msgs[0]["content"]) == _sha(returned)

    # M6: offered tool names per model call.
    offered = {}
    for e in events:
        if e["type"] == "llm_start":
            offered.setdefault(e["agent_id"], set(e["offered_tools"]))
            assert set(e["offered_tools"]) == offered[e["agent_id"]]  # stable per agent
    mcp_all = {"iot_tool", "util_tool", "fmsr_tool", "tsfm_tool", "wo_tool", "vib_tool"}
    assert mcp_all | {"task"} <= offered["root"]
    assert offered["sub-1"] - _FRAMEWORK_TOOLS == {"wo_tool", "util_tool"}
    assert "task" not in offered["sub-1"]
    assert offered["sub-2"] - _FRAMEWORK_TOOLS == mcp_all

    # M6: configured tools per agent; names only, no env values.
    ta = rec["tool_availability"]
    assert ta["readonly_servers"] == ["wo"]
    by_name = {a["name"]: a for a in ta["agents"]}
    assert list(by_name) == ["deep-agent", "general-purpose", "iot", "tsfm", "fmsr", "wo"]
    assert by_name["wo"]["configured_tools"] == ["wo_tool", "util_tool"]
    assert by_name["wo"]["system_prompt_sha256"] == _sha(spec.specialist_prompt("wo"))
    assert by_name["deep-agent"]["configured_tools"] == [
        "iot_tool", "util_tool", "fmsr_tool", "tsfm_tool", "wo_tool", "vib_tool"
    ]
    blob = json.dumps(ta).lower()
    assert "couchdb" not in blob and "api_key" not in blob and "password" not in blob
