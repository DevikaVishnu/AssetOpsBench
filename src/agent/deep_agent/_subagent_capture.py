"""Optional side-channel capture of deep-agent sub-agent internals.

The native trajectory (:func:`agent.deep_agent.runner._build_trajectory`)
only sees the parent agent's message list, so a ``task`` delegation shows up
as one tool call whose output is the sub-agent's final text.  This module
records what happened *inside* the delegation by attaching a read-only
LangChain callback handler to the parent ``ainvoke`` config.  LangChain
propagates that config into the sub-agent graph that deepagents' ``task``
tool invokes, so the handler observes the sub-agent's LLM and tool calls.

Enabled only when ``AGENTTRACE_CAPTURE_SUBAGENTS`` is truthy.  Output goes to
``$AGENT_TRAJECTORY_DIR/{run_id}.instrumented.json``, next to (and never
replacing) the native ``{run_id}.json``.

What is taken directly from LangChain / LangGraph / deepagents:
  * ``task`` tool inputs (``description``, ``subagent_type``) and ``tool_call_id``
  * sub-agent name (the chain name deepagents passes to ``create_agent``)
  * LangChain ``run_id`` / ``parent_run_id`` and LangGraph ``checkpoint_ns``
  * LLM output messages with usage, tool names / args / outputs
  * the sub-agent's final message list and the text returned to the parent

What this module infers:
  * which agent an event belongs to (walk of the LangChain run tree; a chain
    whose parent run is a ``task`` tool run is a new sub-agent)
  * ``agent_id`` labels, treating a ``task`` call as a delegation
  * ``parent_turn_index`` (``tool_call_id`` matched against the native trajectory)
  * ``ts`` / ``mono_ns`` (local clock at callback receipt) and ``seq``
    (callback arrival order)

File-mediated flow (metadata and hashes, not content, unless noted):
  * ``tool_end`` carries ``output_sha256`` / ``output_size`` of the raw output
  * ``tool_result_committed``: each tool message a ``tools`` node wrote to
    state, after deepagents' eviction; content only when it differs from raw
  * ``delegation_end.files_returned``: the ``files`` dict in the ``task``
    hand-back ``Command`` (includes files the child inherited from the parent)
  * ``file_write`` / ``file_edit`` / ``file_read`` via :meth:`record_file_op`,
    from ``_file_observer.ObservedBackend``

Tool availability (only when the runner enables it, i.e. ``AGENTTRACE_SPECIALISTS``):
  * ``llm_start.offered_tools``: names of the tools bound to that model call,
    from LangChain's ``invocation_params["tools"]`` (names only, no schemas)
  * top-level ``tool_availability``: tools configured per agent by the runner
"""

from __future__ import annotations

import datetime as _dt
import functools
import hashlib
import importlib.metadata
import json
import logging
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.types import Overwrite

from observability.runspan import _run_id_var, _scenario_id_var

_log = logging.getLogger(__name__)

SCHEMA = "assetopsbench.deepagent.instrumented/0.1"
CAPTURE_ENV = "AGENTTRACE_CAPTURE_SUBAGENTS"
_TRAJECTORY_DIR_ENV = "AGENT_TRAJECTORY_DIR"
_TASK_TOOL = "task"
_ROOT = "root"


def capture_enabled() -> bool:
    """True when ``AGENTTRACE_CAPTURE_SUBAGENTS`` is set to a truthy value."""
    return os.environ.get(CAPTURE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


def _never_raise(fn):
    """Swallow and log handler errors so capture can never affect the run."""

    @functools.wraps(fn)
    async def wrapper(self, *args, **kwargs):
        try:
            fn(self, *args, **kwargs)
        except Exception:
            _log.exception("subagent capture: %s failed", fn.__name__)

    return wrapper


def _serialize_message(msg: Any) -> Any:
    if not isinstance(msg, BaseMessage):
        return msg
    out: dict[str, Any] = {"role": msg.type, "content": msg.content}
    if msg.name:
        out["name"] = msg.name
    if isinstance(msg, AIMessage):
        if msg.tool_calls:
            out["tool_calls"] = [
                {"id": tc.get("id"), "name": tc.get("name"), "args": tc.get("args")}
                for tc in msg.tool_calls
            ]
        if msg.usage_metadata:
            out["usage"] = dict(msg.usage_metadata)
    if isinstance(msg, ToolMessage):
        out["tool_call_id"] = msg.tool_call_id
        out["status"] = msg.status
    return out


def _tool_output(output: Any) -> Any:
    """Plain content of a tool result (``ToolMessage``, ``Command`` or raw)."""
    if isinstance(output, ToolMessage):
        return output.content
    update = getattr(output, "update", None)
    if isinstance(update, dict):
        msgs = [m for m in update.get("messages", []) if isinstance(m, ToolMessage)]
        if msgs:
            return msgs[-1].content
    return output


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _content_digest(content: Any) -> dict[str, Any]:
    """``sha256`` / ``size`` of tool content; non-string content is hashed as JSON."""
    if isinstance(content, str):
        return {"sha256": _sha256(content), "size": len(content)}
    text = json.dumps(content, default=str)
    return {
        "sha256": _sha256(text),
        "size": len(text),
        "content_kind": "list" if isinstance(content, list) else type(content).__name__,
    }


def _file_meta(fd: Any) -> dict[str, Any]:
    """Metadata of a deepagents ``FileData`` (no content); ``None`` is a deletion."""
    if fd is None:
        return {"deleted": True}
    content = fd.get("content")
    if isinstance(content, list):  # legacy v1 storage: list of lines
        content = "\n".join(content)
    return {
        "sha256": _sha256(content) if isinstance(content, str) else None,
        "size": len(content) if isinstance(content, str) else None,
        "encoding": fd.get("encoding"),
        "created_at": fd.get("created_at"),
        "modified_at": fd.get("modified_at"),
    }


def _committed_tool_messages(outputs: Any) -> list[ToolMessage]:
    """``ToolMessage``s a tools node committed: a ``{"messages": ...}`` dict
    (possibly wrapped in ``Overwrite``) or a list of messages / ``Command``s."""
    if isinstance(outputs, dict):
        msgs = outputs.get("messages")
        if isinstance(msgs, Overwrite):
            msgs = msgs.value
        items = msgs if isinstance(msgs, list) else []
    elif isinstance(outputs, list):
        items = []
        for item in outputs:
            update = getattr(item, "update", None)
            if isinstance(update, dict) and isinstance(update.get("messages"), list):
                items.extend(update["messages"])
            else:
                items.append(item)
    else:
        return []
    return [m for m in items if isinstance(m, ToolMessage)]


def _bound_tool_names(invocation_params: Any) -> list[str] | None:
    """Names of the tools bound to a chat-model call, or ``None`` if unknown.

    Handles Anthropic (``{"name"}``), OpenAI (``{"function": {"name"}}``) and
    ``BaseTool`` entries.  Nothing else from ``invocation_params`` is kept.
    """
    if not isinstance(invocation_params, dict) or "tools" not in invocation_params:
        return None
    names: list[str] = []
    for tool in invocation_params.get("tools") or []:
        if isinstance(tool, dict):
            fn = tool.get("function")
            name = fn.get("name") if isinstance(fn, dict) else tool.get("name")
        else:
            name = getattr(tool, "name", None)
        names.append(name if isinstance(name, str) else None)
    return names


def _error_info(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exception(exc),
    }


class SubagentCaptureHandler(AsyncCallbackHandler):
    """Read-only callback handler that records agent / delegation structure
    and an ordered event log.  Never mutates inputs and never raises."""

    raise_error = False
    run_inline = True

    def __init__(self, *, offered_tools: bool = False) -> None:
        self._offered_tools = offered_tools
        self.tool_availability: dict[str, Any] | None = None
        self.budget_guard: Any = None  # _budget_guard.BudgetGuard when that flag is on
        self._t0 = time.monotonic_ns()
        self.started_at = _now()
        self.events: list[dict[str, Any]] = []
        self.agents: dict[str, dict[str, Any]] = {}
        self.delegations: dict[str, dict[str, Any]] = {}
        self._run_agent: dict[UUID, str] = {}  # any LangChain run -> owning agent_id
        self._agent_runs: dict[UUID, str] = {}  # agent's top-level graph run -> agent_id
        self._task_runs: dict[UUID, str] = {}  # task tool run -> delegation_id
        self._tool_names: dict[UUID, str] = {}
        self._tools_node_runs: dict[UUID, str] = {}  # "tools" node run -> owning agent_id
        self._raw_sha_by_call: dict[str, str] = {}  # tool_call_id -> sha256 of tool_end output
        self._call_agent: dict[str, str | None] = {}  # tool_call_id -> calling agent_id
        self._n_sub = 0
        self._lock = threading.Lock()  # record_file_op may be called from worker threads

    # -- helpers -----------------------------------------------------------

    def _emit(self, type_: str, agent_id: str | None, run_id: UUID | None, parent_run_id: UUID | None, **payload: Any) -> None:
        agent = self.agents.get(agent_id or "", {})
        with self._lock:
            self.events.append(
                {
                    "seq": len(self.events),
                    "ts": _now(),
                    "mono_ns": time.monotonic_ns() - self._t0,
                    "type": type_,
                    "agent_id": agent_id,
                    "agent_name": agent.get("name"),
                    "delegation_id": agent.get("delegation_id"),
                    "lc_run_id": str(run_id) if run_id else None,
                    "lc_parent_run_id": str(parent_run_id) if parent_run_id else None,
                    **payload,
                }
            )

    def _owner(self, parent_run_id: UUID | None) -> str | None:
        return self._run_agent.get(parent_run_id) if parent_run_id else None

    # -- chains (agent graphs) ---------------------------------------------

    @_never_raise
    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, metadata=None, **kwargs):
        metadata = metadata or {}
        if parent_run_id is None:
            agent_id = _ROOT
            self.agents[agent_id] = {
                "agent_id": agent_id,
                "name": "deep-agent",
                "kind": "root",
                "parent_agent_id": None,
                "delegation_id": None,
            }
        elif parent_run_id in self._task_runs:
            delegation_id = self._task_runs[parent_run_id]
            delegation = self.delegations[delegation_id]
            self._n_sub += 1
            agent_id = f"sub-{self._n_sub}"
            self.agents[agent_id] = {
                "agent_id": agent_id,
                "name": kwargs.get("name") or metadata.get("lc_agent_name"),
                "kind": "subagent",
                "parent_agent_id": delegation["parent_agent_id"],
                "delegation_id": delegation_id,
                "checkpoint_ns": metadata.get("checkpoint_ns"),
            }
            delegation["subagent_agent_id"] = agent_id
        else:
            owner = self._owner(parent_run_id)
            if owner is not None:
                self._run_agent[run_id] = owner
                if kwargs.get("name") == "tools":
                    self._tools_node_runs[run_id] = owner
            return
        self._run_agent[run_id] = agent_id
        self._agent_runs[run_id] = agent_id
        self._emit("agent_start", agent_id, run_id, parent_run_id)

    @_never_raise
    def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs):
        owner = self._tools_node_runs.pop(run_id, None)
        if owner is not None:
            self._emit_committed(outputs, owner, run_id, parent_run_id)
            return
        agent_id = self._agent_runs.get(run_id)
        if agent_id is None:
            return
        agent = self.agents[agent_id]
        agent["status"] = "ok"
        if agent["delegation_id"] and isinstance(outputs, dict):
            self.delegations[agent["delegation_id"]]["subagent_messages"] = [
                _serialize_message(m) for m in outputs.get("messages", [])
            ]
        self._emit("agent_end", agent_id, run_id, parent_run_id)

    def _emit_committed(self, outputs: Any, agent_id: str, run_id: UUID, parent_run_id: UUID | None) -> None:
        """One ``tool_result_committed`` per tool message the tools node wrote to
        state, i.e. after deepagents' eviction.  Content is kept only when it
        differs from the raw ``tool_end`` output for the same ``tool_call_id``."""
        for msg in _committed_tool_messages(outputs):
            digest = _content_digest(msg.content)
            raw_sha = self._raw_sha_by_call.get(msg.tool_call_id)
            differs = None if raw_sha is None else digest["sha256"] != raw_sha
            payload: dict[str, Any] = {
                "tool_call_id": msg.tool_call_id,
                "tool": msg.name,
                **digest,
                "status": msg.status,
                "differs_from_raw": differs,
            }
            if differs:
                payload["content"] = msg.content
            self._emit("tool_result_committed", agent_id, run_id, parent_run_id, **payload)

    @_never_raise
    def on_chain_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        agent_id = self._agent_runs.get(run_id)
        if agent_id is None:
            return
        self.agents[agent_id]["status"] = "error"
        self._emit("agent_error", agent_id, run_id, parent_run_id, error=_error_info(error))

    # -- LLM calls -----------------------------------------------------------

    @_never_raise
    def on_chat_model_start(
        self, serialized, messages, *, run_id, parent_run_id=None, **kwargs
    ):
        agent_id = self._owner(parent_run_id)
        self._run_agent[run_id] = agent_id

        input_batches = [
            [_serialize_message(msg) for msg in batch]
            for batch in messages
        ]

        extra: dict[str, Any] = {}
        if self._offered_tools:
            extra["offered_tools"] = _bound_tool_names(kwargs.get("invocation_params"))

        self._emit(
            "llm_start",
            agent_id,
            run_id,
            parent_run_id,
            n_input_messages=len(messages[0]) if messages else 0,
            input_batches=input_batches,
            **extra,
        )

    @_never_raise
    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        gen = response.generations[0][0] if response.generations and response.generations[0] else None
        message = getattr(gen, "message", None)
        payload: dict[str, Any] = {"message": _serialize_message(message)}
        if isinstance(message, AIMessage) and message.usage_metadata:
            payload["usage"] = dict(message.usage_metadata)
        self._emit("llm_end", self._run_agent.get(run_id), run_id, parent_run_id, **payload)

    @_never_raise
    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self._emit("llm_error", self._run_agent.get(run_id), run_id, parent_run_id, error=_error_info(error))

    # -- tool calls / delegations ---------------------------------------------

    @_never_raise
    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, inputs=None, **kwargs):
        agent_id = self._owner(parent_run_id)
        self._run_agent[run_id] = agent_id
        name = kwargs.get("name") or (serialized or {}).get("name")
        self._tool_names[run_id] = name
        tool_call_id = kwargs.get("tool_call_id")
        if tool_call_id:
            self._call_agent[tool_call_id] = agent_id
        args = inputs if inputs is not None else input_str
        if name == _TASK_TOOL and isinstance(inputs, dict) and "subagent_type" in inputs:
            delegation_id = tool_call_id or str(run_id)
            self._task_runs[run_id] = delegation_id
            self.delegations[delegation_id] = {
                "delegation_id": delegation_id,
                "parent_agent_id": agent_id,
                "parent_turn_index": None,
                "subagent_type": inputs.get("subagent_type"),
                "subagent_agent_id": None,
                "description": inputs.get("description"),
                "started_at": _now(),
                "ended_at": None,
                "status": "running",
                "returned_to_parent": None,
                "subagent_messages": None,
                "files_returned": None,
                "update_keys": None,
            }
            self._emit(
                "delegation_start", agent_id, run_id, parent_run_id,
                tool_call_id=tool_call_id,
                subagent_type=inputs.get("subagent_type"),
                description=inputs.get("description"),
            )
            return
        self._emit("tool_start", agent_id, run_id, parent_run_id, tool=name, tool_call_id=tool_call_id, args=args)

    @_never_raise
    def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        agent_id = self._run_agent.get(run_id)
        content = _tool_output(output)
        delegation_id = self._task_runs.get(run_id)
        if delegation_id is not None:
            delegation = self.delegations[delegation_id]
            delegation["ended_at"] = _now()
            delegation["status"] = "ok"
            delegation["returned_to_parent"] = content
            # The hand-back Command carries the child's whole ``files`` dict,
            # including files inherited from the parent: metadata only.
            update = getattr(output, "update", None)
            if isinstance(update, dict):
                delegation["update_keys"] = sorted(update)
                files = update.get("files")
                if isinstance(files, dict):
                    delegation["files_returned"] = {path: _file_meta(fd) for path, fd in files.items()}
            self._emit(
                "delegation_end", agent_id, run_id, parent_run_id,
                returned_to_parent=content,
                files_returned=delegation["files_returned"],
                update_keys=delegation["update_keys"],
            )
            return
        digest = _content_digest(content)
        payload: dict[str, Any] = {
            "tool": self._tool_names.get(run_id),
            "output": content,
            "output_sha256": digest["sha256"],
            "output_size": digest["size"],
        }
        if "content_kind" in digest:
            payload["content_kind"] = digest["content_kind"]
        if isinstance(output, ToolMessage):
            payload["tool_call_id"] = output.tool_call_id
            payload["status"] = output.status
            self._raw_sha_by_call[output.tool_call_id] = digest["sha256"]
        self._emit("tool_end", agent_id, run_id, parent_run_id, **payload)

    @_never_raise
    def on_tool_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        agent_id = self._run_agent.get(run_id)
        delegation_id = self._task_runs.get(run_id)
        if delegation_id is not None:
            delegation = self.delegations[delegation_id]
            delegation["ended_at"] = _now()
            delegation["status"] = "error"
            self._emit("delegation_error", agent_id, run_id, parent_run_id, error=_error_info(error))
            return
        self._emit(
            "tool_error", agent_id, run_id, parent_run_id,
            tool=self._tool_names.get(run_id), error=_error_info(error),
        )

    # -- framework file operations (fed by ``_file_observer.ObservedBackend``) --

    def record_file_op(self, type_: str, **payload: Any) -> None:
        """Record a backend file operation; synchronous and never raises.

        ``agent_id`` is resolved from the operation's ``tool_call_id`` (the
        backend runtime's, never parsed from a path); there is no LangChain run.
        """
        try:
            tool_call_id = payload.get("tool_call_id")
            agent_id = self._call_agent.get(tool_call_id) if tool_call_id else None
            self._emit(type_, agent_id, None, None, **payload)
        except Exception:
            _log.exception("subagent capture: record_file_op(%s) failed", type_)


def _versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for pkg in ("deepagents", "langgraph", "langchain-core"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def _link_parent_turns(capture: SubagentCaptureHandler, trajectory: Any) -> None:
    """Fill ``parent_turn_index`` by matching delegation ids to native tool calls."""
    for turn in getattr(trajectory, "turns", None) or []:
        for call in turn.tool_calls:
            delegation = capture.delegations.get(call.id)
            if delegation is not None and delegation["parent_agent_id"] == _ROOT:
                delegation["parent_turn_index"] = turn.index


def write_instrumented(
    capture: SubagentCaptureHandler,
    *,
    model: str,
    question: str,
    trajectory: Any = None,
    error: BaseException | None = None,
) -> Path | None:
    """Write ``{run_id}.instrumented.json``; never raises.

    Pass *error* when the run failed: the record is written with
    ``"status": "error"`` and whatever was captured before the failure.
    """
    try:
        dir_env = os.environ.get(_TRAJECTORY_DIR_ENV)
        run_id = _run_id_var.get()
        if not dir_env or not run_id:
            _log.warning(
                "%s is set but %s or run_id is missing; skipping instrumented trace",
                CAPTURE_ENV,
                _TRAJECTORY_DIR_ENV,
            )
            return None

        if trajectory is not None:
            _link_parent_turns(capture, trajectory)
        if error is not None:
            for delegation in capture.delegations.values():
                if delegation["status"] == "running":
                    delegation["status"] = "incomplete"

        record = {
            "schema": SCHEMA,
            "status": "error" if error is not None else "ok",
            "error": _error_info(error) if error is not None else None,
            "run_id": run_id,
            "scenario_id": _scenario_id_var.get(),
            "runner": "deep-agent",
            "model": model,
            "question": question,
            "native_trajectory_file": f"{run_id}.json",
            "started_at": capture.started_at,
            "ended_at": _now(),
            "versions": _versions(),
            "agents": list(capture.agents.values()),
            "delegations": list(capture.delegations.values()),
            "events": capture.events,
        }
        if capture.tool_availability is not None:
            record["tool_availability"] = capture.tool_availability
        if capture.budget_guard is not None:
            record["budget_guard"] = capture.budget_guard.summary()
        out_dir = Path(dir_env)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{run_id}.instrumented.json"
        out_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        return out_path
    except Exception:
        _log.exception("write_instrumented failed")
        return None
