"""Optional research configuration: per-server specialist sub-agents.

Enabled only when ``AGENTTRACE_SPECIALISTS`` is truthy.  When off, the runner
behaves exactly as before.  When on:

  * M1 — four ``SubAgent`` specs (``iot``, ``tsfm``, ``fmsr``, ``wo``) are
    passed to ``create_deep_agent``.  Each gets its own server's tools plus the
    ``utilities`` tools; ``vibration`` stays with the coordinator only.  The
    model is inherited.  The coordinator keeps all of its original tools and
    system prompt, and deepagents still adds its ``general-purpose`` sub-agent.
  * M4 — the ``wo`` stdio connection carries ``env={"AOB_READONLY": "1"}``, so
    the WO server registers only its read tools.  stdio servers inherit only a
    fixed default environment, so the variable must be on the connection.
  * Credentials — stdio servers do not inherit the runner's environment and a
    worktree has no ``.env`` for their ``load_dotenv()``, so
    :func:`with_forwarded_env` copies onto each connection only the variables
    that server reads (:data:`SERVER_ENV`), taken from the runner process.
    Model-provider keys are never forwarded.
  * M6 — :func:`tool_availability` describes which tool names were configured
    for each agent (names only; no schemas, no environment values).

Specialist prompts are neutral: the scope line is each server's FastMCP
``instructions`` copied verbatim (checked by a test against the server code),
except for ``wo``.  Its server text advertises create/approve/assign/close/
cancel, which the read-only connection does not register, so the prompt and
description use :data:`READONLY_SCOPES` instead.  The verbatim text stays in
:data:`SERVER_SCOPES` for provenance.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from collections.abc import Mapping
from typing import Any

SPECIALISTS_ENV = "AGENTTRACE_SPECIALISTS"
COORDINATOR_NAME = "deep-agent"
SPECIALIST_SERVERS = ("iot", "tsfm", "fmsr", "wo")
SHARED_SERVERS = ("utilities",)
READONLY_SERVER = "wo"

# Variables each server reads (``os.environ`` in ``src/servers/<name>``), minus
# LLM credentials: FMSR's ``generate_*`` tools stay disabled, as with the current
# ``.env`` whose WatsonX key is empty.  AOB_READONLY is never forwarded; M4 sets it.
_COUCH_ENV = ("COUCHDB_URL", "COUCHDB_USERNAME", "COUCHDB_PASSWORD", "LOG_LEVEL")
SERVER_ENV: dict[str, tuple[str, ...]] = {
    "iot": _COUCH_ENV + ("IOT_DBNAME", "ASSET_DBNAME"),
    "utilities": _COUCH_ENV + ("CATALOG_DBNAME",),
    "fmsr": _COUCH_ENV + ("FAILURE_MODE_DBNAME",),
    "tsfm": _COUCH_ENV + ("MODEL_CATALOG_DBNAME", "FEATURE_CATALOG_DBNAME", "TSFM_WORKDIR"),
    "wo": _COUCH_ENV + ("WO_DBNAME", "FAILURE_CODE_DBNAME"),
    "vibration": _COUCH_ENV + ("VIBRATION_DBNAME",),
}
READONLY_ENV = {"AOB_READONLY": "1"}

# Verbatim copies of each server's ``FastMCP(instructions=...)``.
SERVER_SCOPES: dict[str, str] = {
    "iot": (
        "Read-only IoT sensor data and asset registry tools. Browse sites, assets, and "
        "sensors; inspect asset details; compare installed and measured sensors; and query "
        "telemetry."
    ),
    "tsfm": (
        "The TSFM server provides task discovery, file-pointer evidence tools, recipe execution, "
        "and CouchDB-backed model and feature catalog tools for forecasting and anomaly detection. "
        "For anomaly detection, follow this workflow: 1) call `recipe_template` to confirm the "
        "recipe shape; 2) choose a detector with `find_models(task_id=\"tsfm_anomaly_detection\")` "
        "or `search_models`; 3) call `run_recipe` with `recipe={\"task\": "
        "\"tsfm_anomaly_detection\", \"estimator\": {\"model_id\": \"<model_id>\"}}`; "
        "4) report the anomalous segment from returned labels, indices, or `results_file`."
    ),
    "fmsr": (
        "Failure-mode catalog tools for industrial asset classes. Exposes stored "
        "failure-mode lookup, LLM failure-mode generation, and failure-mode "
        "persistence. Failure-mode/sensor mapping is intentionally disabled and "
        "is not available as an FMSR tool. Agent should use LLM domain knowledge "
        "to get mapping by themselves."
    ),
    "wo": (
        "Work order lifecycle for industrial assets, backed by CouchDB. Query, "
        "create, approve, assign, close, and cancel work orders; compute KPIs, "
        "costs, and schedules. Documents use IBM Maximo mxwo field names."
    ),
}

# Scope lines used in place of the server text under M4.  Edit of the ``wo``
# text: write verbs removed, everything else kept.
READONLY_SCOPES: dict[str, str] = {
    "wo": (
        "Read-only work order access for industrial assets, backed by CouchDB. Query "
        "work orders; compute KPIs, costs, and schedules. Documents use IBM Maximo "
        "mxwo field names."
    ),
}

_PROMPT_TEMPLATE = (
    "You are the {name} specialist agent. Your tools come from the {name} server: "
    "{scope}\n\n"
    "Complete the task you are given using your tools. Your final message is "
    "returned to the agent that delegated the task."
)


def specialists_enabled() -> bool:
    """True when ``AGENTTRACE_SPECIALISTS`` is set to a truthy value."""
    return os.environ.get(SPECIALISTS_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def with_forwarded_env(
    connections: dict[str, dict], environ: Mapping[str, str] | None = None
) -> dict[str, dict]:
    """Copy of *connections* with each server's :data:`SERVER_ENV` variables that
    are set in *environ* (default ``os.environ``) added to its ``env``."""
    environ = os.environ if environ is None else environ
    out = {name: dict(conn) for name, conn in connections.items()}
    for name, conn in out.items():
        forwarded = {k: environ[k] for k in SERVER_ENV.get(name, ()) if environ.get(k)}
        if forwarded:
            conn["env"] = {**(conn.get("env") or {}), **forwarded}
    return out


def with_readonly_wo(connections: dict[str, dict]) -> dict[str, dict]:
    """Copy of *connections* whose ``wo`` entry carries ``AOB_READONLY=1``."""
    out = {name: dict(conn) for name, conn in connections.items()}
    if READONLY_SERVER in out:
        out[READONLY_SERVER]["env"] = {**(out[READONLY_SERVER].get("env") or {}), **READONLY_ENV}
    return out


def specialist_scope(name: str) -> str:
    """Scope line shown to agents: the read-only edit if any, else the server text."""
    return READONLY_SCOPES.get(name, SERVER_SCOPES[name])


def specialist_prompt(name: str) -> str:
    return _PROMPT_TEMPLATE.format(name=name, scope=specialist_scope(name))


async def load_tools_by_server(client: Any, server_names: list[str]) -> dict[str, list]:
    """``client.get_tools(server_name=...)`` for each server, concurrently.

    Concatenated in *server_names* order this equals ``client.get_tools()``,
    which gathers the same per-server loads in connection order.
    """
    results = await asyncio.gather(*(client.get_tools(server_name=n) for n in server_names))
    return dict(zip(server_names, results))


def build_specialists(tools_by_server: dict[str, list]) -> list[dict[str, Any]]:
    """deepagents ``SubAgent`` specs for every specialist server that is connected."""
    shared = [t for s in SHARED_SERVERS for t in tools_by_server.get(s, [])]
    specs: list[dict[str, Any]] = []
    for name in SPECIALIST_SERVERS:
        if name not in tools_by_server:
            continue
        specs.append(
            {
                "name": name,
                "description": f"Specialist agent for the {name} server: {specialist_scope(name)}",
                "system_prompt": specialist_prompt(name),
                "tools": list(tools_by_server[name]) + shared,
            }
        )
    return specs


def tool_availability(
    parent_tools: list, specialists: list[dict[str, Any]], connections: dict[str, dict]
) -> dict[str, Any]:
    """Configured tool names per agent (M6).

    Covers the tools the runner passes in.  Framework tools that deepagents
    middleware adds (``task``, ``write_todos``, filesystem tools) are recorded
    per model call from the bound tools instead (``llm_start.offered_tools``).
    """
    parent_names = [t.name for t in parent_tools]
    agents = [
        {"name": COORDINATOR_NAME, "role": "coordinator", "configured_tools": parent_names},
        # deepagents gives general-purpose the coordinator's tools (graph.py).
        {"name": "general-purpose", "role": "framework_subagent", "configured_tools": parent_names},
    ]
    for spec in specialists:
        agents.append(
            {
                "name": spec["name"],
                "role": "specialist",
                "servers": [spec["name"], *SHARED_SERVERS],
                "configured_tools": [t.name for t in spec["tools"]],
                "scope_source": "readonly_edit" if spec["name"] in READONLY_SCOPES else "server_instructions",
                "server_instructions_sha256": _sha256(SERVER_SCOPES[spec["name"]]),
                "system_prompt_sha256": _sha256(spec["system_prompt"]),
                "description_sha256": _sha256(spec["description"]),
            }
        )
    return {
        "source": SPECIALISTS_ENV,
        # Names only; values are never recorded.
        "forwarded_env_names": {
            n: sorted(k for k in (c.get("env") or {}) if k != "AOB_READONLY")
            for n, c in connections.items()
        },
        "readonly_servers": sorted(
            n for n, c in connections.items() if (c.get("env") or {}).get("AOB_READONLY") == "1"
        ),
        "agents": agents,
    }
