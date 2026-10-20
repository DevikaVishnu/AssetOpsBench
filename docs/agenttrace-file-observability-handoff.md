# AgentTrace: file-observability instrumentation handoff

Written 2026-10-13 at the end of an investigation session. The next session's job is to **implement** the instrumentation in §5, following the plan in §7. The investigation is finished. Don't repeat it and don't redesign the schema.

Labels used throughout:

- **[CONFIRMED]**: verified in this session, either by reading the installed source at the stated line or by running code.
- **[INFERRED]**: follows from confirmed facts but wasn't directly run.
- **[UNVERIFIED]**: an assumption that the next session should check before relying on it.
- **[LIMITATION]**: a known boundary of the design.

---

## 1. Repository state

### 1.1 Two repositories (keep them separate)

| Repo | Path | Role |
|---|---|---|
| **AssetOpsBench** (execution) | `AssetOpsBench` | IBM's benchmark and agent runners. **All implementation work happens here.** Remote `origin` = `https://github.com/IBM/AssetOpsBench.git` (upstream IBM repo; never push). |
| **AP / AgentTrace** (research) | `AP` | AgentTrace schema, docs and the Belief Explorer prototype. Remote: `DevikaVishnu/IBM_Project2`. Canonical schema: `docs/agenttrace.semantic.0.3.md`. **Read-only for this task.** |

AP state at handoff [CONFIRMED]:
- On `main` at `3f7f28b`, in sync with `origin/main`. Claude co-author trailers were removed from history and force-pushed this session.
- `data/assetopsbench/runs/` is untracked and must stay that way.
- There is a local backup branch, `backup-before-remove-claude-coauthor`, at the old `c303c8d`.

### 1.2 AssetOpsBench git state [CONFIRMED]

- Branch: `local/direct-provider-models`. HEAD: `6ec35fb` (Merge pull request #601 from IBM/fix/missing-db-guard).
- `git status --short` at handoff:

```
 M src/agent/claude_agent/runner.py
 M src/agent/deep_agent/runner.py
 M src/llm/litellm.py
?? docs/agenttrace-file-observability-handoff.md      <- this file
?? src/agent/deep_agent/_subagent_capture.py
?? src/agent/deep_agent/_subagent_capture.py.bak
?? src/agent/deep_agent/tests/test_offloading_probe.py
?? src/agent/deep_agent/tests/test_subagent_capture.py
```

All of these are the user's work in progress. **Preserve them.** The `.bak` file differs from `_subagent_capture.py`; leave both alone. The `claude_agent/runner.py` and `llm/litellm.py` changes are unrelated to this task; don't touch them.

### 1.3 Relevant files

| File | What it is |
|---|---|
| `src/agent/deep_agent/runner.py` | `DeepAgentRunner`. The uncommitted diff (+32/−4) adds optional capture: it builds a `SubagentCaptureHandler` when `capture_enabled()`, passes it in `config["callbacks"]`, and calls `write_instrumented` on success and on error. **Ordering detail:** `create_deep_agent(...)` is called at about line 201, *before* `capture` is constructed at about line 214. Part C needs `capture` to exist before `create_deep_agent`. |
| `src/agent/deep_agent/_subagent_capture.py` (399 lines, untracked) | `SubagentCaptureHandler(AsyncCallbackHandler)` plus `write_instrumented`. Raw schema `assetopsbench.deepagent.instrumented/0.1`. Enabled by env `AGENTTRACE_CAPTURE_SUBAGENTS` (truthy values `1/true/yes/on`). Output goes to `$AGENT_TRAJECTORY_DIR/{run_id}.instrumented.json`. |
| `src/agent/deep_agent/tests/test_subagent_capture.py` (untracked) | 6 tests: capture on/off, native-output equivalence, delegation internals, failure path, rendered model inputs. |
| `src/agent/deep_agent/tests/test_offloading_probe.py` (untracked, 397 lines, written this session) | 4 tests on eviction, recovery and the consequential lockout scenario (§3). |
| `src/agent/deep_agent/tests/test_runner.py` (tracked) | 19 runner unit tests. |

Key internals of `_subagent_capture.py` the implementation builds on [CONFIRMED]:

- **`_never_raise(fn)` decorator.** It turns a sync method body into an `async` callback wrapper that swallows and logs exceptions. Every handler method uses it.
- **`_emit(type_, agent_id, run_id, parent_run_id, **payload)`.** Appends an event with these fields: `seq`, `ts`, `mono_ns`, `type`, `agent_id`, `agent_name`, `delegation_id`, `lc_run_id`, `lc_parent_run_id`, plus the payload.
- **Agent attribution.**
  - `on_chain_start` with `parent_run_id is None` creates `root`.
  - A chain whose parent run is a `task` tool run creates `sub-N`.
  - Any other chain inherits its owner through `self._run_agent`.
  - Only agent graph runs (stored in `self._agent_runs`) emit events. Node chains such as `"tools"` and `"model"` are tracked for ownership but **not recorded**.
- **`_tool_output(output)`.** If the output is a `ToolMessage`, it returns `.content`. If it is something with `.update["messages"]` (a `Command`), it returns the last `ToolMessage`'s content and **discards the rest of the update, including `files`**.
- **Event types emitted today:** `agent_start`/`agent_end`/`agent_error`, `llm_start` (with `input_batches`)/`llm_end`/`llm_error`, `tool_start`/`tool_end`/`tool_error`, `delegation_start`/`delegation_end`/`delegation_error`.

### 1.4 Environment and versions [CONFIRMED]

- Python 3.12.14 in `.venv`.
- Package versions: deepagents **0.5.3**, langchain 1.2.15, langgraph 1.1.9, langgraph-prebuilt 1.0.10, langchain-core 1.3.0, langchain-mcp-adapters 0.2.2, pytest 9.0.2, anyio 4.12.1.
- `pyproject.toml` has `[tool.pytest.ini_options]` with `pythonpath = ["src"]` and `anyio_mode = "auto"`. Tests use `@pytest.mark.anyio`, which runs on the asyncio backend only.
- Test command (run from the AssetOpsBench root):

```bash
.venv/bin/python -m pytest src/agent/deep_agent/tests/ -q -p no:cacheprovider
```

Latest result: **29 passed** (4 probe + 6 capture + 19 runner).

---

## 2. Research objective

AgentTrace is a **framework-independent semantic observability layer** for agentic AI. It is part of an IBM Research collaboration on *Mission Control for Agentic AI* (also CSE 523, by Devika Vishnu and Komalika Acharya). It sits above raw telemetry, never replaces it, and answers:
- what information existed;
- what actually reached each agent's rendered model input;
- what agents claimed;
- what was verified;
- how content propagated;
- where the answer must be UNKNOWN because telemetry is missing.

Why file-mediated flow matters: deep agents move information through **framework-managed files**. Oversized tool results are evicted to `/large_tool_results/<id>`, files are handed back from child to parent through delegation state, and agents retrieve them with `read_file`. If those steps aren't observed:
- a final decision can rest on content whose origin can't be traced;
- "the child never handed it over" can't be told apart from "it was never available."

`agenttrace.semantic/0.3` (in the AP repo) is **experimental and unchanged** by this work. Under 0.3:
- `delivers` covers rendered model inputs only.
- `propagates_to` can't cross events that aren't agent inputs, so tool → file → tool lineage is a known limit (§5.3, §12 of the schema).

The goal here is **raw-telemetry observability only**. It does not change the schema.

---

## 3. Experiments completed (all with scripted fake models, no network)

Shared setup [CONFIRMED]:
- The tests run the real `DeepAgentRunner` → `create_deep_agent`, with `agent.deep_agent.runner._build_chat_model` patched to return `_ScriptedModel(GenericFakeChatModel)`.
- `bind_tools` returns self.
- `langchain_mcp_adapters.client.MultiServerMCPClient` is patched to a mock whose `get_tools` returns local `@tool` functions.
- The parent and the general-purpose sub-agent **share one model instance**, so scripted responses are consumed in execution order.

**Scripted-model caveat (applies to everything below):** every model "decision" (delegate, read the file, final wording) is scripted. These tests establish **observable information flow only**, never autonomous reasoning or uptake.

### 3.1 Parent-child delegation and rendered inputs (pre-existing, `test_subagent_capture.py`)

- `task` delegation is captured as `delegation_start`/`delegation_end` with `delegation_id = tool_call_id` (`call_task_1`).
- The child appears as `sub-1` (name `general-purpose`, `parent_agent_id` `root`).
- `llm_start.input_batches` holds the post-middleware message list, including system messages, for both parent and child.

### 3.2 Large-result eviction and `read_file` recovery (`test_offloading_probe.py`)

- Synthetic result: 1,200 lines, 105,566 chars, with the marker `AGENTTRACE-PROBE-MARKER-7Q3ZK9 peak_supply_temp=47.913` at line index 600 (1-indexed line 601).
- Eviction threshold [CONFIRMED]: `NUM_CHARS_PER_TOKEN * tool_token_limit_before_evict` = 4 × 20,000 = 80,000 chars.

Observations [CONFIRMED by assertions]:

1. `on_tool_end` receives the **original** byte-for-byte. Eviction runs in `awrap_tool_call` after the tool and its callbacks return.
2. The next `llm_start` input of the same agent holds the **substitute**:
   - about 1,600 chars;
   - the text "Tool result too large, the result of this tool call {id} was saved in the filesystem at this path: /large_tool_results/{id}";
   - the first and last 5 lines with line numbers, and `... [1190 lines truncated] ...`;
   - no marker.
3. The native trajectory (`call.output`) and the child's final `messages` also hold the substitute. **The instrumented `tool_end` and the native trajectory disagree about "the tool output."**
4. `read_file(file_path="/large_tool_results/<id>", offset=595, limit=10)` returns lines 596–605 in `cat -n` format (`"   601\t<marker>"`). That result appears verbatim in the calling agent's next `llm_start`.
5. With delegation:
   - the child's evicted file **is present in parent state** after `task` returns;
   - the parent's own `read_file` on the child's path returns output identical to the child's read;
   - neither the original nor the substitute of the child's result ever reaches a parent model input.

### 3.3 Consequential Chiller 6 lockout scenario (`test_parent_decision_depends_on_fact_recovered_from_child_offload`)

- `CRITICAL_MARKER = "LOCKOUT-RX7Q3 compressor_lockout=ACTIVE restart_permitted=NO"` sits at line 601 of a 105,572-char event log (tool `get_event_log`).
- The question is "Is it safe to restart Chiller 6 now?"

Observed chain [CONFIRMED by assertions; joined by tool-call IDs and per-agent order]:

| # | Agent / step | Observation | Lockout line? |
|---|---|---|---|
| 1 | root P1 input | question → `task` call `call_task_1` | no |
| 2 | sub-1 tool | `tool_end(call_log_child).output == CRITICAL_ORIGINAL` | yes (tool output only) |
| 3 | sub-1 C2 input | substitute naming `/large_tool_results/call_log_child` | no (absent from all child inputs) |
| 4 | sub-1 C2 output = handoff | "Retrieved the Chiller 6 operational event log (1200 entries). The full log was saved at /large_tool_results/call_log_child; I reviewed only the preview." Equals `returned_to_parent`. | no; also no "lockout" or "restart" |
| 5 | root P2 input | handoff verbatim as the `call_task_1` tool message | no |
| 6 | root P2 output | `read_file` call `call_read_parent` with args `{file_path: /large_tool_results/call_log_child, offset: 595, limit: 10}` | — |
| 7 | root tool | `tool_end(call_read_parent)` holds the window including line 601 | yes |
| 8 | root P3 input | first parent input containing the marker; it appears **only** in the `call_read_parent` message, which comes after that `tool_end` in root's event sequence | yes |
| 9 | root P3 output | "Do not restart Chiller 6: LOCKOUT-RX7Q3 is active (restart_permitted=NO)." Equals `result.answer`. | yes |

Mapping to 0.3, analysis only (no schema change):
- `delivers(P2, handoff)` and `delivers(P3, read_result)` separate the two routes.
- Storable `propagates_to` links: substitute → handoff (path span); handoff → P2's read call (path span); read_result → final (`LOCKOUT-RX7Q3`, `restart_permitted=NO`).
- Not storable: original → substitute, and original → file → read_result. Causes: telemetry (no eviction or file events) and schema (non-agent-input events).
- Non-delivery to P1/P2 stays UNKNOWN under the strict 0.3 rules, because the fake model's fixed token counts give no coverage basis.

---

## 4. Confirmed DeepAgents 0.5.3 execution path

Paths are relative to `.venv/lib/python3.12/site-packages/`. Line numbers were confirmed in deepagents 0.5.3 and langgraph-prebuilt 1.0.10.

### 4.1 Source anchors [CONFIRMED]

`deepagents/middleware/filesystem.py`

| Line | Item |
|---|---|
| 75 | `NUM_CHARS_PER_TOKEN = 4` |
| 78 | `_file_data_reducer(left, right)`: right wins per key; `None` deletes. Used at line 117: `files: Annotated[NotRequired[dict[str, FileData]], _file_data_reducer]` |
| 333 | `supports_execution(backend)`: `isinstance(backend, SandboxBackendProtocol)` (or the `CompositeBackend` default) |
| 374 | `TOOLS_EXCLUDED_FROM_EVICTION = ("ls","glob","grep","read_file","edit_file","write_file")` |
| 384 | `TOO_LARGE_TOOL_MSG` template |
| 454 | `_create_content_preview(content_str, *, head_lines=5, tail_lines=5)` |
| 581 | `__init__(..., tool_token_limit_before_evict: int \| None = 20000, ...)` |
| 610 | `artifacts_root` from `isinstance(self.backend, CompositeBackend)`; otherwise `"/"` |
| 632 | `_get_backend(self, runtime)`: if `callable(self.backend)`, emits `DeprecationWarning` ("Passing a callable (factory) as `backend` is deprecated and will be removed in v0.7…") and returns `self.backend(runtime)`; otherwise returns `self.backend` |
| 709 | `_create_read_file_tool()`. The async read calls `resolved_backend.aread(validated_path, offset=offset, limit=limit)`, then line-number formatting and truncation. `DEFAULT_READ_LIMIT = 100` |
| 1207, 1272 | `wrap_model_call` / `awrap_model_call` call `self._get_backend(request.runtime)` when an `execute` tool is present. **`request.runtime` here is a `Runtime`, not a `ToolRuntime`, and may have no `tool_call_id` attribute** |
| 1372 | `async _aprocess_large_message(message, resolved_backend) -> tuple[ToolMessage, bool]`. If `len(text) > 4 * limit`: `file_path = f"{prefix}/{sanitize_tool_call_id(message.tool_call_id)}"`, then `await resolved_backend.awrite(file_path, content_str)`. **If `result.error`, it returns the original message unevicted.** Otherwise it builds a new `ToolMessage` with the same `tool_call_id`, `name`, `id`, `artifact`, `status`, `additional_kwargs` and `response_metadata`, whose content is the template plus the preview |
| 1419 | `_get_backend_from_runtime(state, runtime)`: for a factory, builds `ToolRuntime(..., tool_call_id=None)` |
| 1617 | `_aintercept_large_tool_result(tool_result, runtime)`: handles a `ToolMessage`, or a `Command` whose `update["messages"]` contains `ToolMessage`s |
| 1672 | `awrap_tool_call(request, handler)`: skips excluded tools; otherwise `tool_result = await handler(request)`, then `_aintercept_large_tool_result(tool_result, request.runtime)` |

`deepagents/backends/utils.py`:
- line 99: `sanitize_tool_call_id(id)` replaces `.`, `/` and `\` with `_`;
- line 218: `create_file_data(content, created_at=None, encoding="utf-8") -> FileData` stamps `created_at` and `modified_at` with `datetime.now(UTC).isoformat()`.

`deepagents/backends/protocol.py`:

| Line | Item |
|---|---|
| 148 | `FileData(TypedDict)`: `content: str`, `encoding: str`, `created_at: NotRequired[str]`, `modified_at: NotRequired[str]` |
| 165 | `@dataclass ReadResult`: `error: str \| None = None`, `file_data: FileData \| None = None` |
| 200 | `WriteResult(error=None, path=None)` |
| 229 | `EditResult(error, path, occurrences)` |
| 301 | `class BackendProtocol(abc.ABC)` |
| 810 | `BackendFactory = Callable[[ToolRuntime], BackendProtocol]` |

`BackendProtocol` methods (sync and async pairs; the async defaults use `asyncio.to_thread` on the sync method):
- `ls(path)` / `als`
- `read(file_path, offset=0, limit=2000) -> ReadResult` / `aread`
- `grep(...)` / `agrep`
- `glob(pattern, path="/")` / `aglob`
- `write(file_path, content) -> WriteResult` / `awrite`
- `edit(file_path, old_string, new_string, replace_all=False) -> EditResult` / `aedit`
- `upload_files(files)` / `aupload_files`
- `download_files(paths)` / `adownload_files`
- `ls_info` / `als_info`, `glob_info` / `aglob_info`, `grep_raw` / `agrep_raw`

`execute` / `aexecute` / `id` exist only on `SandboxBackendProtocol` (line 730).

`deepagents/backends/state.py` (`class StateBackend(BackendProtocol)`, line 38):
- `_read_files()` reads the `files` channel through `CONFIG_KEY_READ` with `fresh=False`, which gives the value as of the start of the superstep.
- `_send_files_update(update)` queues `send([("files", update)])` through `CONFIG_KEY_SEND`. It is applied at the node boundary.
- `write(file_path, content)` (line 242) returns `WriteResult(error="Cannot write to … already exists…")` if the path exists. Otherwise it sends `{path: create_file_data(content)}` and returns `WriteResult(path=…)`.
- It does **not** override `awrite`/`aread`, so they go through `asyncio.to_thread`, which copies contextvars so `get_config()` still works.
- `StateBackend(runtime=…)` is deprecated; use `StateBackend()`.

`deepagents/middleware/subagents.py`:

| Line | Item |
|---|---|
| 176 | `_EXCLUDED_STATE_KEYS = {"messages","todos","structured_response","skills_metadata","memory_contents"}` |
| 374 | `_return_command_with_state_update(result, tool_call_id)` returns `Command(update={**{k: v for k, v in result.items() if k not in excluded}, "messages": [ToolMessage(<last message text, rstripped>, tool_call_id=tool_call_id)]})`. **This carries the child's entire `files` dict, including files inherited from the parent** |
| 405 | `_validate_and_prepare_state`: the child state is the parent `runtime.state` minus excluded keys, plus `messages=[HumanMessage(description)]`, so the child inherits the parent's files |
| 428, 440 | `atask` runs `result = await subagent.ainvoke(subagent_state)` |

`deepagents/graph.py` (`create_deep_agent`, line 218; signature includes `middleware: Sequence[AgentMiddleware] = ()`, `subagents`, `backend: BackendProtocol | BackendFactory | None = None`):
- Line 440: `backend = backend if backend is not None else StateBackend()`. **The same backend object is passed to the parent `FilesystemMiddleware`, `SubAgentMiddleware`, the summarization middleware, and the general-purpose sub-agent's `FilesystemMiddleware`.**
- The general-purpose sub-agent stack (about line 443) is fixed: `[TodoList, Filesystem, Summarization, PatchToolCalls, (extra), AnthropicPromptCaching]`. **User `middleware=` is appended only to the parent stack (line 583), after `FilesystemMiddleware`.**

`deepagents/middleware/summarization.py`:
- line 294: `artifacts_root` uses `isinstance(backend, CompositeBackend)`;
- line 352: `_get_backend` builds `ToolRuntime(..., tool_call_id=None)` for a factory;
- default trigger without a model profile: 170,000 tokens.

`deepagents/middleware/patch_tool_calls.py`, line 44: returns `{"messages": Overwrite(patched_messages)}`, where `Overwrite` is `langgraph.types.Overwrite` (line 832). **This appears in node `on_chain_end` outputs.**

`langgraph/prebuilt/tool_node.py`:
- `ToolRuntime` (line 1559) fields: `state`, `context`, `config: RunnableConfig`, `stream_writer`, `tool_call_id: str | None`, `store`, `execution_info=None`, `server_info=None`.
- One `ToolRuntime` is built per tool call, with `tool_call_id=call["id"]` and a per-call `config` from `get_config_list` (about line 804).
- `ToolCallRequest` (line 131): `tool_call`, `tool`, `state`, `runtime: ToolRuntime`.
- `_arun_one` (line 1153) calls `self._awrap_tool_call(tool_request, execute)`.
- Inside the wrapper, the ambient `get_config()` is the tools node's config, **shared by every tool call in the node**. The only per-call identity is `request.runtime.tool_call_id`.

### 4.2 Callback payloads observed (scratch spy, consequential scenario) [CONFIRMED]

Event names below are the LangChain `name` kwarg passed to `on_chain_start`.

```
tool_end    get_event_log  -> ToolMessage(tool_call_id=call_log_child, len=105572, sha256[:10]=070970c5cc)     # original
chain_end   tools (child)  -> {'messages': [ToolMessage(call_log_child, len=1600, sha=567da9e12e)]}            # substitute, as committed
chain_start model (child)  inputs['files'] = {'/large_tool_results/call_log_child': (105572, 070970c5cc)}       # file visible next step
chain_end   general-purpose -> {'messages': [...], 'files': {'/large_tool_results/call_log_child': (105572, 070970c5cc)}}
tool_end    task           -> Command(update keys=['files','messages'], files={... same path/sha ...},
                                      messages=[ToolMessage(call_task_1, len=152)])                           # hand-back
chain_end   tools (parent) -> [Command(... same as above ...)]                                                  # a LIST of Command
chain_start model (parent) inputs['files'] = {... same path/sha ...}                                            # merged into parent state
tool_end    read_file      -> ToolMessage(call_read_parent, len=922)
chain_end   tools (parent) -> {'messages': [ToolMessage(call_read_parent, len=922)]}
chain_end   LangGraph(root)-> {'messages': [...], 'files': {...}}
```

Facts from this run:
- The output of the tools-node `on_chain_end` is a `dict` with `messages` for plain results, but a **`list[Command]`** when the result is a `Command` (as with `task`).
- Some node outputs contain `Overwrite` objects. A naive iteration crashed the spy with `TypeError: 'Overwrite' object is not iterable`.
- The root graph's chain name is `"LangGraph"` (agent `name=None`). The child's is `"general-purpose"`.
- **[UNVERIFIED]** That a tool run's `parent_run_id` is the tools-node run of the same agent. Ownership through `_run_agent` already works for `tool_start`/`tool_end`, so join by `tool_call_id` rather than by run nesting.

### 4.3 Backend factory experiment (scratch) [CONFIRMED]

- A callable `backend=lambda rt: Observing(StateBackend_shared, rt)` was passed to `create_deep_agent` directly.
- `Observing` forwarded attributes with `__getattr__` and wrapped `awrite`/`aread`.

Recorded calls:

```
('awrite', tool_call_id='call_log_child',  path='/large_tool_results/call_log_child', size=105572, sha=070970c5cc, error=None)
('aread',  tool_call_id='call_read_parent', path='/large_tool_results/call_log_child', offset=595, limit=10,
           modified_at='2026-10-13T12:08:21.745638+00:00', error=None)
```

- **Equivalence:** final `messages` (type, content, `tool_call_id`) and final `files` (path → sha256) were **identical** with and without the factory.
- The factory produced **8 `DeprecationWarning`s** in this scenario (one per backend resolution, including model-call resolutions); the plain run produced 0.
- `runtime.config["metadata"]["checkpoint_ns"]` values (`tools:<uuid>`) **do not reliably identify the agent**. Use the `tool_call_id` join to `tool_start` events instead.

The scratch scripts are in the previous session's scratchpad and **won't persist**. The essential code shape is reproduced in §5.

### 4.4 The seven transitions

| # | Transition | Mechanism | Directly observed now | In existing callbacks but discarded | Inferable now | Missing telemetry |
|---|---|---|---|---|---|---|
| T1 | child tool → original | tool runs inside `handler(request)` | `tool_end.output` (full content) | — | — | hash and size not recorded (convenience only) |
| T2 | original → substitute | `awrap_tool_call` → `_aprocess_large_message` | — | child tools-node `on_chain_end` → `{'messages':[ToolMessage(same id, substitute)]}` | substitute in the next `llm_start` of that agent, matched by `tool_call_id` | no explicit eviction event (Part A.2 covers it) |
| T3 | original → stored file | `resolved_backend.awrite(path, content)` → `StateBackend.write` → queued `send` | — | file appears in the next node's `chain_start` inputs and the child graph's `chain_end` outputs (snapshots, no attribution) | only by path naming or hash equality, **neither enough on its own** | a write event attributed to the tool call (Part B) |
| T4 | child state → `task` result | `_return_command_with_state_update` | `returned_to_parent` text only | `on_tool_end(task)` receives the full `Command`, including `update["files"]` | — | hand-back file metadata (Part A.1) |
| T5 | `task` result → parent state | parent ToolNode applies `Command`; `_file_data_reducer` merges | — | parent model-node `chain_start` inputs show the merged `files` (snapshot) | before/after snapshots | the merge itself isn't an event. **[LIMITATION]**: stays inferred |
| T6 | parent state → `read_file` result | `read_file` → `backend.aread(path, offset, limit)` → format and truncate | `tool_start.args`, `tool_end.output` | — | file version only by path | read event with `tool_call_id`, path, window and `created_at`/`modified_at` (Part B) |
| T7 | read result → parent model input | normal message flow | `llm_start.input_batches` | — | — | none |

---

## 5. Confirmed instrumentation design

General rules for every part:
- **Read-only:** never mutate inputs, outputs, errors or state.
- **Never raise:** reuse `_never_raise`, or wrap observer bookkeeping in `try/except`, which must never wrap the delegated backend call itself.
- **Hashes:** sha256 of the UTF-8 string content.
- **Content:** store content only where §5 says so.
- **Storage is not delivery:** file events are separate types and never go into `input_batches` or delivery fields.

### Part A: extend `_subagent_capture.py`

**A.1 Delegation file hand-back.**
- In `on_tool_end`, in the delegation branch (`delegation_id is not None`), when `getattr(output, "update", None)` is a dict containing `"files"`, build `files_returned = {path: _file_meta(fd) for path, fd in update["files"].items()}`.
- `_file_meta(fd)` gives `{"sha256", "size", "encoding", "created_at", "modified_at"}`. Use `None` for missing keys. A `None` value in the update means a deletion marker; record it as `{"deleted": True}`.
- Store it on the delegation (`delegation["files_returned"]`) and in the `delegation_end` payload.
- Initialise `"files_returned": None` in `on_tool_start`'s delegation dict.
- Also record `update_keys` (sorted). **No file content.**
- Note: the update includes files the child **inherited** from the parent. Don't label them "created by child."

**A.2 Committed tool results.**
- In `on_chain_start`, when a chain is owned by an agent and `kwargs.get("name") == "tools"`, remember `self._tools_node_runs[run_id] = owner`.
- In `on_chain_end` for those runs, normalise `outputs` into a list of `ToolMessage`s:
  - `dict` → `outputs.get("messages")`; if that is an `Overwrite`, use its wrapped value [UNVERIFIED attribute name; check `langgraph.types.Overwrite` at line 832 before use];
  - `list` → each item that is a `ToolMessage`, or a `Command` whose `.update["messages"]` contains `ToolMessage`s;
  - anything else → ignore.
- Emit one `tool_result_committed` event per `ToolMessage` with: `tool_call_id`, `tool` (message `.name`), `sha256`, `size`, `status`, and `content` **only when** its sha differs from the raw `tool_end` sha for the same `tool_call_id`. Keep a `self._raw_sha_by_call` dict from A.3; a differing sha signals substitution.
- Also emit `differs_from_raw: True/False/None`, where `None` means no raw `tool_end` was seen.
- Don't infer the eviction path from the content.

**A.3 Raw tool-result hashes.** In the non-delegation `on_tool_end` branch:
- add `output_sha256` and `output_size` to the `tool_end` payload (string content; for list content, hash the JSON dump and mark `content_kind: "list"`);
- store `self._raw_sha_by_call[tool_call_id]`.
- Keep `output` unchanged (full original, as today).

**A.4 Robustness.** Every new code path must tolerate `Command`, `Overwrite`, `list` content and missing keys without raising (the decorator already protects the run, but a crash loses the event).

### Part B: new `src/agent/deep_agent/_file_observer.py`

Shape, confirmed workable in scratch (adapt names freely):

```python
class ObservedBackend(BackendProtocol):          # subclass BackendProtocol, NOT SandboxBackendProtocol
    def __init__(self, inner: StateBackend, sink, tool_call_id: str | None): ...
    # Forward EVERY BackendProtocol method explicitly (sync + async) to `inner`,
    # returning inner's result object unchanged (same instance, not a copy).
    # Record events for: write/awrite, edit/aedit, read/aread (and optionally upload_files).
    # ls/glob/grep(+info/raw)/download: forward only (optionally record a lightweight "file_list" etc.; not required).

def observing_backend_factory(inner: StateBackend, sink) -> Callable[[Any], BackendProtocol]:
    def factory(runtime) -> BackendProtocol:
        return ObservedBackend(inner, sink, getattr(runtime, "tool_call_id", None))
    return factory
```

Requirements:

- **Single shared inner.** Create one `StateBackend()` per run and share it. `StateBackend` is stateless apart from `file_format`, and reads and writes go through `get_config()`.
- **Explicit forwarding**, not `__getattr__`, so `isinstance(…, BackendProtocol)` holds and `supports_execution` stays `False`, the same as a plain `StateBackend`. Then:
  - the `execute` tool is filtered identically (`wrap_model_call`, line 1207);
  - `artifacts_root` stays `"/"` (filesystem line 610, summarization line 294).
- **Attribution:** use `getattr(runtime, "tool_call_id", None)`.
  - In a tool call (`awrap_tool_call` eviction, `read_file`, `write_file`, `edit_file`) it is the tool-call ID [CONFIRMED for eviction and `read_file`].
  - From `wrap_model_call`, summarization and `_get_backend_from_runtime` it is `None` or absent. Record those as `"attribution": "unattributed"`.
  - **Never parse the path to get an ID.**
- **Event payloads** go to the capture handler's event list through a sink method such as `capture.record_file_op(type_, **payload)`. That method must call `_emit`-equivalent logic with `agent_id` resolved by looking up `tool_call_id` in a `tool_call_id → agent_id` map (fill it in `on_tool_start`), and `None` if unknown. No `lc_run_id` is available; set it to `None`.

| Event type | Payload |
|---|---|
| `file_write` | `tool_call_id`, `attribution` (`"tool_call"` or `"unattributed"`), `path`, `sha256`, `size`, `error` (`WriteResult.error`), `result_path` |
| `file_edit` | `tool_call_id`, `attribution`, `path`, `old_sha256`, `new_sha256` (of the strings), `replace_all`, `error`, `occurrences` |
| `file_read` | `tool_call_id`, `attribution`, `path`, `offset`, `limit`, `error`, plus returned `file_data` metadata: `encoding`, `created_at`, `modified_at`, `window_sha256`, `window_size` (of the **raw** window content before line-number formatting) |

- **Queued vs committed:** a `file_write` is a queued update (`state.py`), not proof of commit. Name and document it so.
- **Observer bookkeeping failures** must be caught and logged, never re-raised. **Errors from `inner` must propagate unchanged.** Call `inner`, record inside `try/except`, then return or raise exactly what `inner` produced.
- **Thread safety:** `awrite`/`aread` reach the sync methods through `asyncio.to_thread` only if you call `inner.awrite` (which does it). Prefer recording in the async wrapper *after* `await inner.awrite(...)`, in the event-loop thread. If recording from sync methods, which run in worker threads, guard the event append with a `threading.Lock` [INFERRED need].

### Part C: runner integration (`runner.py`)

- Move `capture = SubagentCaptureHandler() if capture_enabled() else None` **before** `create_deep_agent(...)`.
- Only when `capture is not None`, pass `backend=observing_backend_factory(StateBackend(), capture)` to `create_deep_agent`.
- When capture is off, call `create_deep_agent(model=..., tools=..., system_prompt=...)` **exactly as today**: no `backend` argument, so the default `StateBackend()` applies (graph.py line 440).
- Import `StateBackend` from `deepagents.backends` lazily, inside `run()` next to `create_deep_agent`, matching the existing lazy import.
- Don't change `ainvoke`, the model, tools, prompts or the result. No model-visible content changes.
- Optionally suppress the factory `DeprecationWarning` only in tests (pytest `filterwarnings` marker on the new tests). Don't add global filters in production code.

---

## 6. Known limitations

- **Deprecated factory API.** In deepagents 0.5.3, a callable backend triggers a `DeprecationWarning` on every resolution and is scheduled for removal in v0.7. Without it there is no per-tool-call identity for backend writes (§4.1, `tool_node.py`). Pin and check the version: record `deepagents` in `_versions()`, which already happens. A v0.7 upgrade will need a new hook.
- **Queued vs committed.** `file_write` records a queued channel write. Commit is inferred from later snapshots (`tool_result_committed`, delegation `files_returned`, later `file_read` metadata).
- **Parent reducer merge is not observed.** T5 stays an inference: before (the `task` `Command` files) → after (a later read's `created_at`/`modified_at` and sha).
- **Operations outside observed middleware.** Tools or custom middleware that return `Command(update={"files": ...})` directly bypass the backend. They're visible only through Part A, and only for `task` hand-backs and tools-node outputs.
- **Unattributed framework writes.** Summarization offload and human-message eviction (`awrap_model_call`) resolve the backend without a tool-call ID. They're recorded as unattributed and never assigned by path or timing.
- **Hash or path equality is corroboration only.** Provenance claims rest on the execution-context `tool_call_id` at the backend call.
- **No evidence about hidden reasoning.** With scripted models, nothing shows attention or decision-making. Even with real models, uptake shows only through observable outputs.
- **No schema change.** These synthetic tests don't justify any change to `agenttrace.semantic/0.3`. Whether a real trace plus a concrete Mission Control question ever does is a separate, later decision in the AP repo.
- **Version-specific facts.** Line numbers, `Overwrite` in node outputs, the eviction-on-write-error passthrough, and the `list[Command]` tools-node output shape were all confirmed only for the versions in §1.4.

---

## 7. Implementation and testing plan

### 7.1 Code changes

1. **`src/agent/deep_agent/_subagent_capture.py`** (A.1–A.4):
   - `__init__`: add `self._tools_node_runs: dict[UUID, str] = {}`, `self._raw_sha_by_call: dict[str, str] = {}`, `self._call_agent: dict[str, str | None] = {}`.
   - `on_chain_start`: in the `else` branch, when an owner exists and the name is `"tools"`, record it in `_tools_node_runs`.
   - `on_chain_end`: before the `_agent_runs` lookup, handle `_tools_node_runs` and emit `tool_result_committed`.
   - `on_tool_start`: record `_call_agent[tool_call_id] = agent_id`; add `"files_returned": None` to the delegation dict.
   - `on_tool_end`: add the hashes (A.3) and the delegation `files_returned` (A.1).
   - New helpers: `_sha256(text)`, `_file_meta(fd)`, `_committed_tool_messages(outputs)`.
   - New public method: `record_file_op(type_: str, **payload)`. It is synchronous, never raises, and resolves `agent_id` from `_call_agent`.
   - Leave `SCHEMA` at `assetopsbench.deepagent.instrumented/0.1` unless the user decides to bump it. New fields and event types are additive. **Ask before bumping.**
2. **`src/agent/deep_agent/_file_observer.py`** (new, Part B): `ObservedBackend`, `observing_backend_factory`.
3. **`src/agent/deep_agent/runner.py`** (Part C): reorder the capture construction and pass `backend=` conditionally. Keep the user's existing diff intact otherwise.

### 7.2 Tests

Put these in a new file, `src/agent/deep_agent/tests/test_file_observability.py`. Reuse the patterns and helpers of `test_offloading_probe.py`; import its module-level fixtures and data (`CRITICAL_ORIGINAL`, `_consequential_script`, `_ScriptedModel`, `get_event_log`, `_read_call`) or copy them.

For direct-graph tests, build `create_deep_agent(model=_ScriptedModel(messages=iter(script)), tools=[...], system_prompt="sys", backend=...)` and call `ainvoke(..., config={"callbacks": [capture]})`. That returns the final state, including `files`, which `DeepAgentRunner.run` doesn't expose.

| # | Test | Assertions |
|---|---|---|
| 1 | **Native execution equivalence (runner)** | With capture off vs on through `DeepAgentRunner`: native `run-1.json` is identical except for `started_at` (existing `_native` helper); `answer` is identical. With capture off, `create_deep_agent` is called **without** a `backend` kwarg (patch it with `wraps=` and inspect the call kwargs). |
| 2 | **Graph equivalence (direct)** | Plain `StateBackend()` vs `observing_backend_factory(StateBackend(), capture)`: final messages (type, content, `tool_call_id`) and final `files` (path → sha256 and `created_at` presence) are identical. |
| 3 | **Consequential parent retrieval** | Lockout scenario through the runner with capture on: there is a `file_write` with `tool_call_id == "call_log_child"`, `attribution == "tool_call"`, `agent_id == "sub-1"` (from the ID join), `sha256 == sha(CRITICAL_ORIGINAL)` and `error is None`; there is a `file_read` with `tool_call_id == "call_read_parent"`, `agent_id == "root"`, the same path, `offset == 595`, `limit == 10`, non-null `created_at`/`modified_at`, and `window_sha256` matching the raw lines 595–604; all existing probe assertions still hold. |
| 4 | **Original/substitute pairing** | `tool_end(call_log_child).output_sha256 == sha(original)`; `tool_result_committed(call_log_child)` has `differs_from_raw is True`, and its content contains `/large_tool_results/call_log_child` and not the marker; for a small tool (`read_file`), the committed result has `differs_from_raw is False` and no `content`. |
| 5 | **Delegation file hand-back** | `delegations[0]["files_returned"]` contains `/large_tool_results/call_log_child` with `sha256 == sha(original)` and `size == len(original)`; `delegation_end` carries the same; there is no content field. |
| 6 | **Parallel tool-call attribution** | One AI message with two `get_event_log`-style calls with different IDs and different marker payloads (two tools or args) → two `file_write` events whose `tool_call_id` matches each original's sha. This proves attribution doesn't depend on path parsing (also check that the event's `tool_call_id` comes from the runtime, e.g. by using IDs containing `.` so the sanitized path differs from the ID). |
| 7 | **Failed eviction** | Direct graph with initial state `{"messages": [...], "files": {"/large_tool_results/call_x": create_file_data("pre-existing")}}` → `file_write.error` is non-null, `tool_result_committed(call_x).differs_from_raw is False`, and the model's next input holds the **original** (documents the passthrough). |
| 8 | **Observer failure** | Monkeypatch `capture.record_file_op` (or the sink) to raise → the run completes; final messages and files are identical to the plain run; inner backend errors still propagate as before (e.g. reading a missing path returns the same `"Error: File … not found"` tool output). |
| 9 | **Storage is not delivery** | No `file_*` or `tool_result_committed` event has `input_batches`; across all parent `llm_start` events before `call_read_parent`'s `tool_end`, the marker is absent even though `files_returned` (storage) already contains the file; the marker first appears in a parent `llm_start` only through the `call_read_parent` tool message. |

Then run the full suite:

```bash
.venv/bin/python -m pytest src/agent/deep_agent/tests/ -q -p no:cacheprovider
```

Expected: the 29 existing tests plus the new ones, all passing. Report the warnings count, including the factory `DeprecationWarning`s.

---

## 8. Important constraints

- Do **not** modify AgentTrace schema 0.3 (`AP/docs/agenttrace.semantic.0.3.md`) or anything in the AP repo.
- Do **not** modify installed packages in `.venv/.../site-packages`.
- Do **not** modify, move or publish existing raw runs: AP `data/assetopsbench/runs/`, and any `AGENT_TRAJECTORY_DIR` outputs.
- **Preserve all existing uncommitted changes** listed in §1.2, including `_subagent_capture.py.bak`.
- Do **not** stage, commit or push without explicit authorization. Never `git add .`; stage named files only when asked. AssetOpsBench's `origin` is IBM's upstream: never push there.
- Do **not** add Claude co-author trailers or any Claude attribution to commits.
- Do **not** redesign telemetry or the schema based on speculation. Implement §5 as specified; if a blocker forces a deviation, stop and report it.
- Prefer minimal, tested instrumentation. Model inputs, tool outputs and backend behavior must stay unchanged when capture is on, and the code path must be identical when capture is off.

---

## 9. Exact next action

1. Read this handoff, then `src/agent/deep_agent/_subagent_capture.py`, `src/agent/deep_agent/runner.py`, `src/agent/deep_agent/tests/test_offloading_probe.py` and `test_subagent_capture.py`.
2. Run a brief compatibility check, with no exploratory research:
   - `.venv/bin/python -c "import importlib.metadata as m; print(m.version('deepagents'))"` should print `0.5.3`;
   - grep for `def _get_backend`, `async def awrap_tool_call`, `async def _aprocess_large_message` and `def _return_command_with_state_update` in the installed deepagents and confirm they match §4.1;
   - check `langgraph.types.Overwrite`'s value attribute name (§5 A.2, the one [UNVERIFIED] item);
   - run the existing suite and expect 29 passed.
3. Implement Parts A, B and C (§5, §7.1).
4. Add `test_file_observability.py` (§7.2) and run the full deep-agent suite.
5. Report: files changed, test results, the observed event sequence for the lockout scenario (including `file_write`, `tool_result_committed`, `files_returned` and `file_read`), and any deviation from this plan. Stop there; don't stage or commit.

Do not do further exploratory research unless a concrete implementation blocker appears.
