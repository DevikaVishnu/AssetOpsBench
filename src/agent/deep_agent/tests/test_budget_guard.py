"""Tests for the optional ``AGENTTRACE_BUDGET_GUARD`` call/cost guard.

Unit tests drive the callback directly.  The runner tests use the real
``create_deep_agent`` graph with specialists, a scripted fake chat model that
yields to the event loop before answering (so parallel sub-agent calls are in
flight together) and a fake MCP client.  No network or LLM calls.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import deepagents
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.tools import tool

from agent.deep_agent import _budget_guard as bg
from agent.deep_agent import _specialists as spec
from agent.deep_agent.runner import DeepAgentRunner
from agent.runner import DEFAULT_SERVER_PATHS
from observability import runspan as _runspan
from observability import set_run_context

_USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
_PRICES = bg.Prices(input=2.0, cache_write_5m=2.5, cache_write_1h=4.0, output=10.0)
_ENV = {
    bg.MAX_CALLS_ENV: "200",
    bg.MAX_COST_ENV: "10",
    bg.PRICE_ENV["input"]: "2",
    bg.PRICE_ENV["cache_write_5m"]: "2.5",
    bg.PRICE_ENV["cache_write_1h"]: "4",
    bg.PRICE_ENV["output"]: "10",
}


def _guard(**kw) -> bg.BudgetGuard:
    return bg.BudgetGuard(**{"max_calls": 200, "max_cost_usd": 10.0, "prices": _PRICES, **kw})


def _start(guard, *, max_tokens=4096, text="hi", run_id=None):
    run_id = run_id or uuid4()
    guard.on_chat_model_start(
        {}, [[HumanMessage(content=text)]], run_id=run_id,
        invocation_params={"max_tokens": max_tokens, "tools": []},
    )
    return run_id


def _end(guard, run_id, usage=_USAGE):
    msg = AIMessage(content="ok", usage_metadata=usage) if usage else AIMessage(content="ok")
    guard.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]), run_id=run_id)


# -- configuration -----------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "0", "false", "off"])
def test_flag_parsing_off(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(bg.GUARD_ENV, raising=False)
    else:
        monkeypatch.setenv(bg.GUARD_ENV, value)
    assert bg.guard_enabled() is False


def test_from_env_reads_explicit_values():
    g = bg.BudgetGuard.from_env(_ENV)
    assert (g.max_calls, g.max_cost_usd, g.prices) == (200, 10.0, _PRICES)


@pytest.mark.parametrize("missing", [bg.MAX_CALLS_ENV, bg.MAX_COST_ENV, *bg.PRICE_ENV.values()])
def test_from_env_fails_closed_without_a_value(missing):
    env = {k: v for k, v in _ENV.items() if k != missing}
    with pytest.raises(ValueError, match=missing):
        bg.BudgetGuard.from_env(env)


@pytest.mark.parametrize("bad", [{bg.MAX_CALLS_ENV: "0"}, {bg.MAX_COST_ENV: "-1"},
                                 {bg.PRICE_ENV["output"]: "0"}, {bg.MAX_CALLS_ENV: "1.5"}])
def test_from_env_rejects_invalid_values(bad):
    with pytest.raises(ValueError):
        bg.BudgetGuard.from_env({**_ENV, **bad})


# -- limits ------------------------------------------------------------------------


def test_call_limit_allows_exactly_max_calls():
    g = _guard(max_calls=200, max_cost_usd=1e9)
    for _ in range(200):
        _end(g, _start(g))
    with pytest.raises(bg.BudgetExceeded, match="BUDGET-STOPPED: max_model_calls"):
        _start(g)
    s = g.summary()
    assert (s["calls_started"], s["calls_settled"], s["stop_reason"]) == (200, 200, "max_model_calls")
    assert s["outcome"] == bg.OUTCOME and len(s["blocked"]) == 1


def test_stop_is_sticky():
    g = _guard(max_calls=1, max_cost_usd=1e9)
    _end(g, _start(g))
    for _ in range(2):
        with pytest.raises(bg.BudgetExceeded):
            _start(g)
    assert [b["reason"] for b in g.blocked] == ["max_model_calls", "already_stopped"]
    assert g.stop_reason == "max_model_calls"


def test_cost_limit_counts_inflight_reservations():
    g = _guard()
    one = g.reservation([[HumanMessage(content="hi")]], {"max_tokens": 4096, "tools": []})
    g.max_cost_usd = one * 2.5  # room for two reservations, not three
    a, b = _start(g), _start(g)
    with pytest.raises(bg.BudgetExceeded, match="max_cost_usd"):
        _start(g)
    blocked = g.blocked[0]
    assert blocked["inflight_reserved_usd"] == pytest.approx(2 * one, abs=1e-6)
    assert g.calls_started == 2
    _end(g, a), _end(g, b)
    assert g.summary()["calls_unsettled"] == 0


def test_reservation_is_an_upper_bound_on_its_inputs():
    g = _guard()
    text = "é" * 1000  # 2 UTF-8 bytes each
    r = g.reservation([[HumanMessage(content=text)]], {"max_tokens": 100, "tools": [{"name": "t"}]})
    payload = len(json.dumps({"content": text, "tool_calls": None}, ensure_ascii=False).encode())
    tools = len(json.dumps([{"name": "t"}]).encode())
    tokens = payload + tools + bg.INPUT_OVERHEAD_TOKENS + bg.PER_MESSAGE_OVERHEAD_TOKENS
    assert r == pytest.approx((tokens * 4.0 + 100 * 10.0) / 1e6)  # input at the max price


@pytest.mark.parametrize("params", [{}, {"max_tokens": None}, {"max_tokens": 0}, None])
def test_unbounded_output_fails_closed(params):
    g = _guard()
    with pytest.raises(bg.BudgetExceeded, match="unbounded_output"):
        g.on_chat_model_start({}, [[HumanMessage(content="x")]], run_id=uuid4(), invocation_params=params)
    assert g.calls_started == 0


def test_usage_cost_prices_cache_tokens_as_uncached_or_write():
    g = _guard()
    usage = {
        "input_tokens": 1_000_000,  # langchain-anthropic total: plain + cache read + cache write
        "output_tokens": 100_000,
        "input_token_details": {"cache_read": 600_000, "ephemeral_5m_input_tokens": 100_000,
                                "ephemeral_1h_input_tokens": 50_000, "cache_creation": 0},
    }
    # 850k plain/read at $2, 100k at $2.5, 50k at $4, 100k out at $10.
    assert g.usage_cost(usage) == pytest.approx(1.7 + 0.25 + 0.2 + 1.0)


def test_missing_usage_and_errors_keep_the_reservation():
    g = _guard()
    r1 = _start(g)
    res = g._inflight[r1]
    _end(g, r1, usage=None)
    r2 = _start(g)
    g.on_llm_error(RuntimeError("boom"), run_id=r2)
    s = g.summary()
    assert s["committed_usd_estimate"] == pytest.approx(2 * res, abs=1e-6)
    assert (s["calls_without_usage"], s["calls_errored"]) == (1, 1)


def test_concurrent_reservations_are_atomic():
    g = _guard()
    one = g.reservation([[HumanMessage(content="hi")]], {"max_tokens": 4096, "tools": []})
    g.max_cost_usd = one * 5.5
    barrier = threading.Barrier(32)
    accepted, rejected = [], []

    def worker():
        barrier.wait()
        try:
            accepted.append(_start(g))
        except bg.BudgetExceeded:
            rejected.append(1)

    threads = [threading.Thread(target=worker) for _ in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert (len(accepted), len(rejected)) == (5, 27)
    assert g.summary()["peak_reserved_usd"] <= g.max_cost_usd


def test_attach_replaces_an_earlier_guard():
    model = GenericFakeChatModel(messages=iter([]))
    first, second = _guard(), _guard()
    first.attach(model)
    second.attach(model)
    assert model.callbacks == [second]


# -- runner: real deepagents graph with specialists ------------------------------------


@tool
def iot_tool(asset: str) -> str:
    """Fake IoT tool."""
    return f"sensors for {asset}"


@tool
def wo_tool(wonum: str) -> str:
    """Fake WO read tool."""
    return f"WO {wonum}: WAPPR"


@tool
def util_tool(x: str) -> str:
    """Fake utilities tool."""
    return x


_BY_SERVER = {"iot": [iot_tool], "utilities": [util_tool], "fmsr": [], "tsfm": [],
              "wo": [wo_tool], "vibration": []}


def _client():
    async def get_tools(*, server_name=None):
        if server_name is None:
            return [t for name in DEFAULT_SERVER_PATHS for t in _BY_SERVER[name]]
        return list(_BY_SERVER[server_name])

    client = MagicMock()
    client.get_tools = AsyncMock(side_effect=get_tools)
    return client


class _SlowModel(GenericFakeChatModel):
    """Binds tools and ``max_tokens`` like a provider model and yields to the
    event loop before answering, so concurrent calls overlap.  ``provider_calls``
    counts calls that reached the "provider" (``_agenerate``)."""

    provider_calls: int = 0
    max_out: int | None = 1000

    def bind_tools(self, tools, **kwargs):
        names = [t["name"] if isinstance(t, dict) else t.name for t in tools]
        extra = {} if self.max_out is None else {"max_tokens": self.max_out}
        return self.bind(tools=[{"name": n} for n in names], **extra)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        self.provider_calls += 1
        await asyncio.sleep(0.05)
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _parallel_script():
    return iter(
        [
            AIMessage(  # parent -> iot and wo specialists in parallel
                content="",
                tool_calls=[
                    {"name": "task", "args": {"description": "Sensors on Chiller 6", "subagent_type": "iot"},
                     "id": "call_task_iot"},
                    {"name": "task", "args": {"description": "Look up WO 42", "subagent_type": "wo"},
                     "id": "call_task_wo"},
                ],
                usage_metadata=_USAGE,
            ),
            AIMessage(content="Specialist answer A.", usage_metadata=_USAGE),
            AIMessage(content="Specialist answer B.", usage_metadata=_USAGE),
            AIMessage(content="Done.", usage_metadata=_USAGE),  # parent final
        ]
    )


def _sequential_script():
    return iter(
        [
            AIMessage(content="", tool_calls=[{"name": "task", "args": {
                "description": "Look up WO 42", "subagent_type": "wo"}, "id": "call_task_wo"}],
                usage_metadata=_USAGE),
            AIMessage(content="", tool_calls=[{"name": "wo_tool", "args": {"wonum": "42"},
                                               "id": "call_wo_1"}], usage_metadata=_USAGE),
            AIMessage(content="WO 42 is WAPPR.", usage_metadata=_USAGE),
            AIMessage(content="Done.", usage_metadata=_USAGE),
        ]
    )


@pytest.fixture(autouse=True)
def _reset_ctx():
    _runspan._run_id_var.set(None)
    _runspan._scenario_id_var.set(None)
    yield
    _runspan._run_id_var.set(None)
    _runspan._scenario_id_var.set(None)


async def _run(out_dir: Path, monkeypatch, *, guard_env: dict | None, script, max_out=1000):
    monkeypatch.setenv("AGENT_TRAJECTORY_DIR", str(out_dir))
    monkeypatch.setenv("AGENTTRACE_CAPTURE_SUBAGENTS", "1")
    monkeypatch.setenv(spec.SPECIALISTS_ENV, "1")
    for name in (bg.GUARD_ENV, bg.MAX_CALLS_ENV, bg.MAX_COST_ENV, *bg.PRICE_ENV.values()):
        monkeypatch.delenv(name, raising=False)
    for name, value in (guard_env or {}).items():
        monkeypatch.setenv(name, value)
    set_run_context(run_id="run-1", scenario_id="s1")

    model = _SlowModel(messages=script, max_out=max_out)
    real_create = deepagents.create_deep_agent
    error = None
    with (
        patch("agent.deep_agent.runner._build_chat_model", return_value=model),
        patch("langchain_mcp_adapters.client.MultiServerMCPClient", return_value=_client()),
        patch("deepagents.create_deep_agent", wraps=real_create),
    ):
        try:
            await DeepAgentRunner().run("What is going on with Chiller 6?")
        except Exception as exc:  # noqa: BLE001 - inspected by the tests
            error = exc
    record = json.loads((out_dir / "run-1.instrumented.json").read_text())
    return model, record, error


def _llm_starts(record):
    return [e for e in record["events"] if e["type"] == "llm_start"]


@pytest.mark.anyio
async def test_guard_off_attaches_nothing_and_records_nothing(tmp_path, monkeypatch):
    model, record, error = await _run(tmp_path, monkeypatch, guard_env=None, script=_parallel_script())
    assert error is None and record["status"] == "ok"
    assert model.callbacks is None
    assert "budget_guard" not in record


@pytest.mark.anyio
async def test_guard_values_without_flag_do_nothing(tmp_path, monkeypatch):
    model, record, error = await _run(tmp_path, monkeypatch, guard_env=dict(_ENV), script=_parallel_script())
    assert error is None and model.callbacks is None and "budget_guard" not in record


@pytest.mark.anyio
async def test_parallel_specialists_share_one_guard(tmp_path, monkeypatch):
    model, record, error = await _run(
        tmp_path, monkeypatch, guard_env={bg.GUARD_ENV: "1", **_ENV}, script=_parallel_script()
    )
    assert error is None and record["status"] == "ok"
    g = record["budget_guard"]
    starts = _llm_starts(record)
    agents = {e["agent_name"] for e in starts}
    assert {"iot", "wo"} <= agents  # both specialists' calls went through the guard
    assert g["calls_started"] == len(starts) == model.provider_calls == 4
    assert g["calls_settled"] == 4 and g["calls_unsettled"] == 0
    assert g["peak_inflight_calls"] == 2  # the two specialist calls overlapped
    assert g["stopped"] is False and g["outcome"] is None and g["blocked"] == []
    assert g["calls_over_reservation"] == 0
    assert g["committed_usd_estimate"] == pytest.approx(4 * (10 * 2.0 + 5 * 10.0) / 1e6)


@pytest.mark.anyio
async def test_call_limit_stops_inside_a_specialist_before_the_provider(tmp_path, monkeypatch):
    env = {bg.GUARD_ENV: "1", **_ENV, bg.MAX_CALLS_ENV: "2"}
    model, record, error = await _run(tmp_path, monkeypatch, guard_env=env, script=_sequential_script())
    assert isinstance(error, bg.BudgetExceeded)
    assert model.provider_calls == 2  # third call never reached the provider
    assert record["status"] == "error" and record["error"]["type"] == "BudgetExceeded"
    g = record["budget_guard"]
    assert (g["outcome"], g["stop_reason"], g["calls_started"]) == (bg.OUTCOME, "max_model_calls", 2)
    # The refused call is the wo specialist's second turn; its llm_start has no llm_end.
    blocked_id = g["blocked"][0]["lc_run_id"]
    blocked = [e for e in _llm_starts(record) if e["lc_run_id"] == blocked_id]
    assert len(blocked) == 1 and blocked[0]["agent_name"] == "wo"
    assert not [e for e in record["events"] if e["type"] == "llm_end" and e["lc_run_id"] == blocked_id]
    assert record["delegations"][0]["status"] == "error"  # task tool saw the exception


@pytest.mark.anyio
async def test_cost_limit_refuses_the_second_parallel_specialist(tmp_path, monkeypatch):
    # Output dominates: each call reserves about max_tokens * $1000/MTok = $1.
    env = {bg.GUARD_ENV: "1", **_ENV, bg.PRICE_ENV["output"]: "1000",
           bg.PRICE_ENV["input"]: "0.000001", bg.PRICE_ENV["cache_write_5m"]: "0.000001",
           bg.PRICE_ENV["cache_write_1h"]: "0.000001", bg.MAX_COST_ENV: "1.5"}
    model, record, error = await _run(tmp_path, monkeypatch, guard_env=env, script=_parallel_script())
    assert isinstance(error, bg.BudgetExceeded)
    g = record["budget_guard"]
    assert g["stop_reason"] == "max_cost_usd"
    assert g["blocked"][0]["inflight_reserved_usd"] > 0.9  # sibling's reservation was counted
    assert g["calls_started"] == model.provider_calls == 2  # parent + one specialist
    assert record["status"] == "error"
    blocked_agent = {e["agent_name"] for e in _llm_starts(record)
                     if e["lc_run_id"] == g["blocked"][0]["lc_run_id"]}
    assert blocked_agent <= {"iot", "wo"} and len(blocked_agent) == 1


@pytest.mark.anyio
async def test_unbounded_output_stops_before_the_first_call(tmp_path, monkeypatch):
    model, record, error = await _run(
        tmp_path, monkeypatch, guard_env={bg.GUARD_ENV: "1", **_ENV},
        script=_parallel_script(), max_out=None,
    )
    assert isinstance(error, bg.BudgetExceeded) and model.provider_calls == 0
    assert record["budget_guard"]["stop_reason"] == "unbounded_output"


@pytest.mark.anyio
async def test_missing_configuration_fails_before_any_call(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_TRAJECTORY_DIR", str(tmp_path))
    monkeypatch.setenv(bg.GUARD_ENV, "1")
    for name in (bg.MAX_CALLS_ENV, bg.MAX_COST_ENV, *bg.PRICE_ENV.values()):
        monkeypatch.delenv(name, raising=False)
    model = _SlowModel(messages=_parallel_script())
    with (
        patch("agent.deep_agent.runner._build_chat_model", return_value=model),
        patch("langchain_mcp_adapters.client.MultiServerMCPClient", return_value=_client()) as mcp,
    ):
        with pytest.raises(ValueError, match=bg.MAX_CALLS_ENV):
            await DeepAgentRunner().run("q")
    assert model.provider_calls == 0 and not mcp.called


# -- real ChatAnthropic class, no network (closed localhost port, dummy key) ---------


def _anthropic(guard):
    from langchain_anthropic import ChatAnthropic

    model = ChatAnthropic(model="claude-sonnet-5", api_key="sk-test-not-a-key",
                          base_url="http://127.0.0.1:9", max_retries=0)
    guard.attach(model)
    return model.bind_tools([{"name": "t", "description": "d",
                              "input_schema": {"type": "object", "properties": {}}}])


@pytest.mark.anyio
async def test_real_chat_anthropic_is_refused_before_any_request():
    g = _guard(max_cost_usd=1e-9)
    with patch("anthropic.AsyncAnthropic.post", side_effect=AssertionError("request attempted")) as post:
        with pytest.raises(bg.BudgetExceeded, match="max_cost_usd"):
            await _anthropic(g).ainvoke("hello")
    post.assert_not_called()
    assert g.calls_started == 0


@pytest.mark.anyio
async def test_real_chat_anthropic_exposes_max_tokens_to_the_guard():
    g = _guard()
    with pytest.raises(Exception) as info:  # connection refused on 127.0.0.1:9
        await _anthropic(g).ainvoke("hello")
    assert not isinstance(info.value, bg.BudgetExceeded)
    s = g.summary()
    assert (s["calls_started"], s["calls_errored"]) == (1, 1)  # reserved, then kept on error
    # max_tokens (4096 default for this model id) was bounded; reservation includes it.
    assert s["max_call_reservation_usd"] > 4096 * 10.0 / 1e6
