"""Optional research guard: one shared cap on model calls and estimated spend.

Enabled only when ``AGENTTRACE_BUDGET_GUARD`` is truthy.  When off, nothing is
attached and the runner behaves exactly as before.  When on, a single
:class:`BudgetGuard` is attached to the runner's chat model.  The coordinator,
every specialist, ``general-purpose`` and summarization all call that same model
instance, so every model call passes through the one guard, whether or not the
callback config reaches the call.

Before each call (``on_chat_model_start``), under a lock, the guard reserves an
upper-bound cost estimate for that call and refuses the call, by raising
:class:`BudgetExceeded` before the provider request, when

  * ``calls_started + 1`` would exceed ``AGENTTRACE_MAX_MODEL_CALLS``, or
  * committed + in-flight reservations + this reservation would exceed
    ``AGENTTRACE_MAX_COST_USD``, or
  * the call's ``max_tokens`` is unknown (no output bound: fail closed).

When a call ends, its reservation is replaced by its cost from the provider's
``usage_metadata``.  All input tokens are priced as uncached (cache reads at the
full input price; cache writes at their higher write price).  If a call errors
or reports no usage, its full reservation is kept as its cost.

This bounds an *estimate*.  It is not a provider-enforced dollar cap:

  * the input bound assumes a token is at least one UTF-8 byte of the serialized
    messages and tool schemas, plus a fixed overhead [INF];
  * the client's internal retries (``max_retries``) happen inside one guarded
    call and are invisible to callbacks;
  * prices are whatever the launcher passes; they are not fetched.

The guard is invisible to the model unless it fires.  It never edits a request.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage

_log = logging.getLogger(__name__)

GUARD_ENV = "AGENTTRACE_BUDGET_GUARD"
MAX_CALLS_ENV = "AGENTTRACE_MAX_MODEL_CALLS"
MAX_COST_ENV = "AGENTTRACE_MAX_COST_USD"
PRICE_ENV = {
    "input": "AGENTTRACE_PRICE_INPUT_USD_PER_MTOK",
    "cache_write_5m": "AGENTTRACE_PRICE_CACHE_WRITE_5M_USD_PER_MTOK",
    "cache_write_1h": "AGENTTRACE_PRICE_CACHE_WRITE_1H_USD_PER_MTOK",
    "output": "AGENTTRACE_PRICE_OUTPUT_USD_PER_MTOK",
}
OUTCOME = "BUDGET-STOPPED"

# Fixed allowance for provider-side framing (tool-use preamble, message and
# role markers) on top of the serialized bytes [INF].
INPUT_OVERHEAD_TOKENS = 2048
PER_MESSAGE_OVERHEAD_TOKENS = 16


def guard_enabled() -> bool:
    """True when ``AGENTTRACE_BUDGET_GUARD`` is set to a truthy value."""
    return os.environ.get(GUARD_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


class BudgetExceeded(RuntimeError):
    """Raised before a model call that would exceed a limit."""


@dataclass(frozen=True)
class Prices:
    """USD per million tokens."""

    input: float
    cache_write_5m: float
    cache_write_1h: float
    output: float

    @property
    def reserve_input(self) -> float:
        # Any input token could turn out to be a cache write.
        return max(self.input, self.cache_write_5m, self.cache_write_1h)


def _message_payload(msg: Any) -> Any:
    if isinstance(msg, BaseMessage):
        return {"content": msg.content, "tool_calls": getattr(msg, "tool_calls", None) or None}
    return msg


def _utf8_len(obj: Any) -> int:
    return len(json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"))


class BudgetGuard(BaseCallbackHandler):
    """Shared call/cost guard; see the module docstring."""

    raise_error = True  # the only way a callback can stop a call
    run_inline = True

    def __init__(self, *, max_calls: int, max_cost_usd: float, prices: Prices) -> None:
        if max_calls < 1 or max_cost_usd <= 0:
            raise ValueError("budget guard limits must be positive")
        if min(asdict(prices).values()) <= 0:
            raise ValueError("budget guard prices must be positive")
        self.max_calls = max_calls
        self.max_cost_usd = max_cost_usd
        self.prices = prices
        self._lock = threading.Lock()
        self._inflight: dict[UUID, float] = {}
        self.calls_started = 0
        self.calls_settled = 0
        self.calls_errored = 0
        self.calls_without_usage = 0
        self.calls_over_reservation = 0  # actual > reserved: the [INF] bound failed
        self.committed_usd = 0.0
        self.peak_inflight = 0
        self.peak_reserved_usd = 0.0
        self.max_call_reservation_usd = 0.0
        self.stop_reason: str | None = None
        self.blocked: list[dict[str, Any]] = []

    # -- construction --------------------------------------------------------------

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> BudgetGuard:
        """Build from the environment.  Every limit and price must be set
        explicitly (no defaults), else :class:`ValueError` before any call."""
        environ = os.environ if environ is None else environ

        def need(name: str, cast):
            raw = environ.get(name, "").strip()
            if not raw:
                raise ValueError(f"{GUARD_ENV} is on but {name} is not set")
            try:
                return cast(raw)
            except ValueError:
                raise ValueError(f"{name} is not a valid number") from None

        return cls(
            max_calls=need(MAX_CALLS_ENV, int),
            max_cost_usd=need(MAX_COST_ENV, float),
            prices=Prices(**{k: need(v, float) for k, v in PRICE_ENV.items()}),
        )

    def attach(self, model: Any) -> None:
        """Add this guard to *model*'s own callbacks, replacing any earlier guard."""
        existing = model.callbacks
        if existing is not None and not isinstance(existing, list):
            raise ValueError("budget guard: model.callbacks is not a list")
        model.callbacks = [c for c in existing or [] if not isinstance(c, BudgetGuard)] + [self]

    # -- estimates -----------------------------------------------------------------

    def reservation(self, messages: list[list[Any]], invocation_params: Mapping[str, Any] | None) -> float | None:
        """Upper-bound cost of one call, or None when its output is unbounded."""
        params = invocation_params or {}
        max_out = params.get("max_tokens")
        if not isinstance(max_out, int) or isinstance(max_out, bool) or max_out <= 0:
            return None
        batch = [m for b in messages for m in b]
        input_bound = (
            sum(_utf8_len(_message_payload(m)) for m in batch)
            + _utf8_len(params.get("tools") or [])
            + INPUT_OVERHEAD_TOKENS
            + PER_MESSAGE_OVERHEAD_TOKENS * len(batch)
        )
        return (input_bound * self.prices.reserve_input + max_out * self.prices.output) / 1e6

    def usage_cost(self, usage: Mapping[str, Any]) -> float:
        """Cost of a finished call at uncached prices (cache writes at write price)."""
        details = usage.get("input_token_details") or {}
        w5 = int(details.get("ephemeral_5m_input_tokens") or 0)
        w1h = int(details.get("ephemeral_1h_input_tokens") or 0)
        w_generic = int(details.get("cache_creation") or 0)
        total_in = int(usage.get("input_tokens") or 0)
        plain = max(total_in - w5 - w1h - w_generic, 0)  # includes cache reads, at full price
        p = self.prices
        return (
            plain * p.input
            + w5 * p.cache_write_5m
            + w1h * p.cache_write_1h
            + w_generic * max(p.cache_write_5m, p.cache_write_1h)
            + int(usage.get("output_tokens") or 0) * p.output
        ) / 1e6

    # -- callbacks -------------------------------------------------------------------

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
        try:
            reserve = self.reservation(messages, kwargs.get("invocation_params"))
            estimate_failed = False
        except Exception:
            _log.exception("budget guard: reservation estimate failed")
            reserve, estimate_failed = None, True
        with self._lock:
            reserved = sum(self._inflight.values())
            if self.stop_reason is not None:
                reason = "already_stopped"
            elif self.calls_started + 1 > self.max_calls:
                reason = "max_model_calls"
            elif estimate_failed:
                reason = "estimate_failed"
            elif reserve is None:
                reason = "unbounded_output"
            elif self.committed_usd + reserved + reserve > self.max_cost_usd:
                reason = "max_cost_usd"
            else:
                reason = None
            if reason is not None:
                if self.stop_reason is None:
                    self.stop_reason = reason
                self.blocked.append(
                    {
                        "lc_run_id": str(run_id),
                        "lc_parent_run_id": str(parent_run_id) if parent_run_id else None,
                        "reason": reason,
                        "calls_started": self.calls_started,
                        "committed_usd": round(self.committed_usd, 6),
                        "inflight_reserved_usd": round(reserved, 6),
                        "call_reservation_usd": None if reserve is None else round(reserve, 6),
                    }
                )
                _log.warning("%s: %s (calls started %d)", OUTCOME, reason, self.calls_started)
                raise BudgetExceeded(f"{OUTCOME}: {reason}")
            self._inflight[run_id] = reserve
            self.calls_started += 1
            self.peak_inflight = max(self.peak_inflight, len(self._inflight))
            self.peak_reserved_usd = max(self.peak_reserved_usd, reserved + reserve)
            self.max_call_reservation_usd = max(self.max_call_reservation_usd, reserve)

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        try:
            usage = None
            gens = response.generations[0] if response.generations else []
            message = getattr(gens[0], "message", None) if gens else None
            if message is not None:
                usage = getattr(message, "usage_metadata", None)
            cost = self.usage_cost(usage) if usage else None
        except Exception:  # never let accounting break a finished call
            _log.exception("budget guard: usage accounting failed")
            cost = None
        with self._lock:
            reserve = self._inflight.pop(run_id, None)
            if reserve is None:
                return
            if cost is None:
                self.calls_without_usage += 1
                cost = reserve
            elif cost > reserve:
                self.calls_over_reservation += 1
            self.committed_usd += cost
            self.calls_settled += 1

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        with self._lock:
            reserve = self._inflight.pop(run_id, None)
            if reserve is None:
                return
            self.committed_usd += reserve  # billing unknown: keep the bound
            self.calls_errored += 1

    # -- record ------------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        with self._lock:
            inflight = sum(self._inflight.values())
            return {
                "source": GUARD_ENV,
                "outcome": OUTCOME if self.stop_reason is not None else None,
                "stopped": self.stop_reason is not None,
                "stop_reason": self.stop_reason,
                "limits": {"max_model_calls": self.max_calls, "max_cost_usd": self.max_cost_usd},
                "prices_usd_per_mtok": asdict(self.prices),
                "calls_started": self.calls_started,
                "calls_settled": self.calls_settled,
                "calls_errored": self.calls_errored,
                "calls_without_usage": self.calls_without_usage,
                "calls_over_reservation": self.calls_over_reservation,
                "calls_unsettled": len(self._inflight),
                "committed_usd_estimate": round(self.committed_usd, 6),
                "unsettled_reserved_usd": round(inflight, 6),
                "total_usd_estimate": round(self.committed_usd + inflight, 6),
                "peak_inflight_calls": self.peak_inflight,
                "peak_reserved_usd": round(self.peak_reserved_usd, 6),
                "max_call_reservation_usd": round(self.max_call_reservation_usd, 6),
                "blocked": list(self.blocked),
                "basis": (
                    "estimate at uncached prices; reservation = UTF-8 bytes of messages and "
                    "tool schemas + fixed overhead, plus max_tokens; not a provider-enforced cap"
                ),
            }
