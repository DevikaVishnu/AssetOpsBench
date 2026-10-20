# AgentTrace research checkpoint — 2026-10-13

This checkpoint **supersedes** `docs/agenttrace-file-observability-handoff.md` for continuing the research. That older document stays in place unchanged: it is still the detailed reference for DeepAgents internals (its §4) and the instrumentation design (its §5). The implementation work it describes is done.

Evidence labels:

- **[OBS]**: observed directly, either on disk, in raw telemetry, or in a test assertion, during the session that wrote this file.
- **[INF]**: follows from observed facts by a stated rule, but was not observed directly.
- **[HYP]**: a hypothesis that has not been tested.

All repository facts below were re-verified from disk when this file was written.

---

## 1. Repositories, environment, versions

### 1.1 AssetOpsBench: execution repo, where all code lives [OBS]

- **Path:** `AssetOpsBench`
- **Branch:** `local/direct-provider-models`. HEAD is `6ec35fb` ("Merge pull request #601 from IBM/fix/missing-db-guard"). A local `main` branch also exists.
- **Remote:** `origin` = `https://github.com/IBM/AssetOpsBench.git`, IBM upstream. **Never push.**
- **`git status --short`:**

```
 M src/agent/claude_agent/runner.py        # user WIP, unrelated to this research; don't touch
 M src/agent/deep_agent/runner.py          # Part C (capture + observed backend wiring)
 M src/llm/litellm.py                      # user WIP, unrelated to this research; don't touch
?? docs/agenttrace-file-observability-handoff.md      # previous handoff (keep)
?? docs/agenttrace-research-checkpoint-20261013.md    # this file
?? src/agent/deep_agent/_file_observer.py             # Part B (227 lines)
?? src/agent/deep_agent/_subagent_capture.py          # capture handler + Part A (535 lines)
?? src/agent/deep_agent/_subagent_capture.py.bak      # user's backup; leave alone
?? src/agent/deep_agent/tests/test_file_observability.py  # 9 tests (362 lines)
?? src/agent/deep_agent/tests/test_offloading_probe.py    # 4 tests (398 lines)
?? src/agent/deep_agent/tests/test_subagent_capture.py    # 6 tests (271 lines)
```

- **`git diff --stat`:** 3 files, +55/−12. Breakdown: `deep_agent/runner.py` 55 changed lines, `claude_agent/runner.py` 6, `llm/litellm.py` 6.
- **Nothing is staged or committed.** Never `git add .`. Never add Claude co-author or attribution trailers.

### 1.2 AP / AgentTrace: research repo, read-only for this work [OBS]

- **Path:** `AP`
- **Branch:** `main` at `3f7f28b` ("docs: update AgentTrace research overview").
- **Other branches:** `backup-before-remove-claude-coauthor` and `claude/inspiring-lumiere-0e232e`.
- **`git status`:** only `?? data/assetopsbench/runs/`. That directory must stay untracked.
- **Canonical schema:** `docs/agenttrace.semantic.0.3.md`, 913 lines, sha256 `2273d210…5c74cd`. **Unchanged. Do not modify.**

### 1.3 Environment [OBS]

- **Python:** 3.12.14 in `AssetOpsBench/.venv`.
- **Packages:** deepagents **0.5.3**, langchain 1.2.15, langgraph 1.1.9, langgraph-prebuilt 1.0.10, langchain-core 1.3.0, langchain-mcp-adapters 0.2.2, langchain-anthropic 1.4.1, pytest 9.0.2, anyio 4.12.1.
- **`AssetOpsBench/.env`.** Only variable *names* and *non-secret* values are recorded here; never copy key values.
  - `ANTHROPIC_API_KEY` is set.
  - `LITELLM_*` and `TOKENROUTER_*` are empty. The only usable model route is therefore native `anthropic:<model>`. The CLI default, `litellm_proxy/aws/claude-opus-4-6`, would **not** work.
  - `COUCHDB_URL=http://localhost:5984`, `IOT_DBNAME=iot`, `WO_DBNAME=workorder`.
  - `AGENT_TRAJECTORY_DIR=<projects>/AP/data/assetopsbench/runs`
  - `OTEL_TRACES_FILE=<projects>/AP/data/assetopsbench/runs/traces.jsonl` (see §1.4)
- **Local data limits.** CouchDB was up during the run.
  - IoT site `MAIN` has the assets `Chiller 6`, `Motor_01`, `PUMP3`, `hyd_1` and `mp_1`.
  - Chiller 6 telemetry covers **2020-06-01T00:00 to 2020-06-30T23:45 only**: 2,896 records at ~15-minute intervals.
  - Chiller 9 and any April/May 2020 window return **0 observations**. Scenarios that name them (420–435, 501–520) cannot behave as designed locally.
  - The WO MCP server has **no** alert/anomaly event tools. It exposes `list_workorders`, `get_failure_codes`, `generate_work_order` and similar.
  - The `workorder` DB held 7 documents before and after the 621 run.

### 1.4 Issue: `traces.jsonl` was appended. Needs a user decision [OBS]

- `.env` routes OTEL spans to the **existing** `AP/data/assetopsbench/runs/traces.jsonl`. The scenario-621 run therefore appended to it.
  - **Before:** 79,114 bytes, 50 lines (last modified Oct 5).
  - **After:** 101,154 bytes, 69 lines (last modified 2026-10-13 09:17).
- **The appended region is lines 51–69 (byte offset 79,114 onward).**
  - Every appended span has `service.name = deep-agent`.
  - `aob-621` first appears on line 69, at byte offset 99,036.
  - The first 79,114 bytes end with a newline and contain no `aob-621`.
- [INF] The exporter appends, so the original 50 lines are probably intact. **There is no backup**, so whether they are byte-identical to the original cannot be verified.
- **Do not truncate, restore or otherwise modify `traces.jsonl` without explicit user approval.**
- **Before any future real run:** point `OTEL_TRACES_FILE` at a new, per-run file in the environment of that one command. Do not edit `.env` without approval.

---

## 2. Research goal and constraints

- **What AgentTrace is.** `agenttrace.semantic` is a framework-independent semantic observability layer for agentic AI. It is part of the IBM Research *Mission Control for Agentic AI* collaboration, and also CSE 523 (Devika Vishnu, Komalika Acharya).
- **Questions it answers:**
  - what content existed;
  - what reached each agent's *rendered* model input;
  - what agents claimed;
  - what was verified;
  - how content propagated;
  - where the answer must be UNKNOWN.
- **Current question:** can 0.3 represent file-mediated information flow, in particular the *delegated* lineage child tool result → framework-managed file → parent `read_file` result → parent final answer?
- **0.3 rules that bear on this:**
  - `delivers` is defined only for agent-input events and their actual rendered input (schema §5.1).
  - `propagates_to` requires `delivers(E, A)` with E the producer of B. It therefore cannot cross events that aren't agent inputs (§5.3 "Known limit", §12).
  - Lineage through tool → file → tool dataflow is **DEFERRED** (§12) and listed as **OPEN** after run 620 (§12 "OPEN after cross-trace validation").
  - First-class delegation and hierarchy are also OPEN (§12).
  - `run.outputs` populated from harness markers is OPEN (§4.1).
  - Exposed spans (§10.1) records exposure only and "never establishes derivation or lineage".
- **Standing constraints:**
  - Do not modify schema 0.3, the installed site-packages, or any existing raw run.
  - Do not stage, commit or push without explicit authorization.
  - Do not change production agent behavior to force delegation.
  - Do not fabricate tool outputs.
  - Do not put secrets or complete model inputs in documents.
  - **Do not design 0.4.**

---

## 3. Instrumentation (implemented) and test status

The capture switch is the env var `AGENTTRACE_CAPTURE_SUBAGENTS` (truthy values `1/true/yes/on`). The output is `$AGENT_TRAJECTORY_DIR/{run_id}.instrumented.json` with raw schema `assetopsbench.deepagent.instrumented/0.1`, unchanged. The native `{run_id}.json` is still written, and is identical whether capture is on or off [OBS, test].

| Part | File | What it records [OBS, code] |
|---|---|---|
| Base | `_subagent_capture.py` | `agent_*`, `llm_start` (with `input_batches` = post-middleware rendered messages incl. system), `llm_end` (message + usage), `tool_start/end/error`, `delegation_start/end/error` (`delegation_id` = `task` tool_call_id; child agents `sub-N`) |
| A.1 | `_subagent_capture.py` ~l.378–412 | `delegation_end.files_returned` = `{path: {sha256,size,encoding,created_at,modified_at}}` (or `{"deleted": True}`) and `update_keys`; **no content**; includes files the child merely inherited |
| A.2 | `_subagent_capture.py` ~l.288–305 | `tool_result_committed` per ToolMessage leaving an agent's `tools` node: `tool_call_id`, `tool`, `sha256`, `size`, `status`, `differs_from_raw`, and `content` **only when** it differs from the raw `tool_end` (i.e. the eviction substitute) |
| A.3 | `_subagent_capture.py` | `tool_end.output_sha256`, `output_size`, `content_kind` (`"list"` → hash of JSON dump); `output` keeps the full original |
| B | `_file_observer.py` | `ObservedBackend(BackendProtocol)` + `observing_backend_factory(inner, sink)`; explicit forwarding of every backend method; emits `file_write` (`path, sha256, size, error, result_path`), `file_edit`, `file_read` (`path, offset, limit, error, encoding, created_at, modified_at, window_sha256, window_size`), each with `tool_call_id` from the runtime and `attribution` `tool_call`/`unattributed`; sink is `capture.record_file_op` (never raises; lock-guarded) |
| C | `runner.py` | capture built before `create_deep_agent`; `backend=observing_backend_factory(StateBackend(), capture)` **only** when capture is on; capture off → exact original call, no `backend` kwarg |

**Test suite:** `.venv/bin/python -m pytest src/agent/deep_agent/tests/ -q -p no:cacheprovider` gives **38 passed, 45 warnings** [OBS, re-run while writing this file]. The breakdown is 19 runner, 6 capture, 4 offloading probe and 9 file observability (`test_runner_native_output_identical_and_no_backend_when_capture_off`, `test_graph_state_identical_with_observed_backend`, `test_consequential_parent_retrieval_file_events`, `test_original_substitute_pairing`, `test_delegation_file_handback`, `test_storage_is_not_delivery`, `test_parallel_tool_calls_attributed_by_runtime_id`, `test_failed_eviction_passes_original_through`, `test_observer_failure_never_affects_run`).

Known limits of the telemetry:

- The callable-backend factory API is deprecated in deepagents 0.5.3 and scheduled for removal in v0.7.
  - [OBS] The 621 run's captured stderr had 0 `DeprecationWarning` lines.
  - [HYP] The CLI may filter warnings.
- `file_write` records a **queued** state write. A commit is only inferred, from later evidence: a `file_read` whose metadata matches, `files_returned`, or later snapshots.
- The merge of a child's files into parent state (T5 below) is not an event.
- Framework writes that happen outside a tool call (summarization offload, model-call eviction) would be `unattributed`. No such writes occurred in run 621.

---

## 4. Synthetic consequential delegated experiment (Chiller 6 lockout)

- **Test:** `test_offloading_probe.py::test_parent_decision_depends_on_fact_recovered_from_child_offload`, extended by `test_file_observability.py::test_consequential_parent_retrieval_file_events`.
- **Scripted setup.** The model is `_ScriptedModel(GenericFakeChatModel)`, shared by parent and child. MCP is patched to local tools. `get_event_log` returns a 105,572-char log whose line 601 is `LOCKOUT-RX7Q3 compressor_lockout=ACTIVE restart_permitted=NO`. The question is "Is it safe to restart Chiller 6 now?"
- **Observed chain** [OBS, assertions]:
  1. root delegates via `call_task_1`.
  2. The `sub-1` tool `call_log_child` returns the original.
  3. `file_write` (`tool_call_id=call_log_child`, `agent_id=sub-1`, sha = original).
  4. The child's next input holds only the substitute.
  5. The child's handoff text names the path but not the lockout.
  6. `files_returned` carries the path, sha and size.
  7. The parent's `read_file` `call_read_parent` (offset 595, limit 10) produces a `file_read` with a matching `window_sha256` and the same `created_at`.
  8. The lockout line first appears in a parent `llm_start` only through that read's tool message.
  9. The final answer is "Do not restart Chiller 6: LOCKOUT-RX7Q3 is active (restart_permitted=NO)."
- **Limits:**
  - Every model decision was **scripted**: to delegate, which file to read, which window, and the final wording. This proves observable information flow and instrumentation correctness only. It proves **nothing** about natural agent behavior, uptake or causation.
  - The fake model's fixed token counts give no coverage basis, so non-delivery stays UNKNOWN under 0.3.
  - The lockout fact was planted.

---

## 5. Natural scenario 621 (real model)

### 5.1 Configuration [OBS]

| Item | Value |
|---|---|
| Run ID | `aob-621-sonnet5-deep-fileobs-20261013` |
| Scenario | 621 (`AP/data/assetopsbench/scenarios/all_utterance.jsonl`, type `multiagent`): "Can you check if there are any anomalies for chiller tonnage, and do you think I should create a work order after reviewing the anomalies?" Used verbatim. |
| Command | from AssetOpsBench root, `.env` sourced, `AGENTTRACE_CAPTURE_SUBAGENTS=1`: `.venv/bin/deep-agent --model-id anthropic:claude-sonnet-5 --run-id aob-621-sonnet5-deep-fileobs-20261013 --scenario-id 621 --json "<question>"` |
| Model | `anthropic:claude-sonnet-5` (same route as runs 607–620) |
| System prompt / tools | unchanged `AGENT_SYSTEM_PROMPT`; all default MCP servers (iot, utilities, fmsr, tsfm, wo, vibration); default deepagents middleware incl. `task` tool with `general-purpose` sub-agent |
| Timing | `started_at` 2026-10-13T13:14:28Z, `ended_at` 13:17:15Z; status `ok`, error `None` |
| Usage | 24 model turns; total input tokens 1,123,772 (summed per turn), output 12,123 |
| Raw files | `AP/data/assetopsbench/runs/aob-621-sonnet5-deep-fileobs-20261013.json` (68,091 B, sha256 `8645198b…0130ef`); `…instrumented.json` (2,102,943 B, sha256 `aa8531f2…c74cd`) |
| Side effects | `workorder` DB doc count 7 → 7 (no WO created); `traces.jsonl` appended (§1.4) |

Why this scenario was chosen:

- It is a real benchmark task with several steps: find the data, detect anomalies, decide on a work order.
- A single-sensor history page is about 103–130k chars, above the 80,000-char eviction threshold (4 × 20,000 tokens).
- It ends in a decision.
- The local data supports it (Chiller 6, June 2020).

### 5.2 Delegation: none [OBS]

- `agents` = `[root]` and `delegations` = `[]`. There is no `task` tool call and no `sub-N` agent.
- The `task` tool **was offered.** Its description ("## `task` (subagent spawner)" and "Remember to use the `task` tool to silo independent tasks…") is in root's rendered system message at `llm_start` seq 1.
- The model chose not to delegate.
- [OBS] Across natural real-model runs so far (607, 612, 615, 617, 620, 621, all `anthropic:claude-sonnet-5`), **0 of 6 delegated.**

### 5.3 Offloading and retrieval (single agent) [OBS]

Event numbers are `seq` values in `…instrumented.json`. Tool call IDs are abbreviated: `016q` = `toolu_016qXfh1yoXBzXpoixb8LJmS`.

| seq | Event | Fact |
|---|---|---|
| 24/25 | `history` page 1, `016q` (sensor `Chiller 6 Tonnage`, limit 1000) | original text 130,849 chars (list content; `output_size` 145,917 is the JSON dump) |
| 26 | `file_write` `/large_tool_results/016q`, `attribution=tool_call`, `error=None` | sha256 `38478099…` = sha of the original text |
| 27 | `tool_result_committed` `016q`, `differs_from_raw=True`, size 967 | substitute: "Tool result too large…" + lines 1–5 and 4992–4996; **no** `next_cursor` line, no June-10 values |
| 28 | `llm_start` | input tokens 38,268 vs 37,394 at seq 22 (+874) |
| 30–32 | `read_file` `toolu_015GAp2u…` offset 4975 limit 25 on `016q` | `window_sha256` matches raw lines; `created_at` = write time 13:14:48.274Z; yields page-1 `next_cursor` |
| 36–39 | `history` page 2 `toolu_01DTF5DY…` (cursor arg) → `file_write` (117,428 chars) → substitute | |
| 42–44 | `read_file` `toolu_01WuZEop…` offset 4990 limit 20 on page-2 file | shows zero tonnage around 2020-06-21T16:00 |
| 48–51 | `history` page 3 `toolu_011WGwSE…` → `file_write` (105,213 chars) → substitute | **never read** |
| 54–55 | `profile_series(dataset_path="/large_tool_results/016q")` (tsfm) | error "No such file or directory": real-disk tool can't see framework state files |
| 59–66 | `write_file /tmp/test.csv` (framework state; `file_write` recorded) then `profile_series` on it | same error; no tool→file→tool flow succeeded |
| 17/20 | `sensor_stats` Tonnage | max 5476.770123504672, mean 736.04, stddev 1257.93 (small result, not evicted) |
| 108–118 | `grep "5476.77"` in `/large_tool_results` → `grep … output_mode=content` | `016q: 4717: "Chiller 6 Tonnage": 5476.770123504672` |
| 121–123 | `read_file` `toolu_01S66jRc…` offset 4695 limit 40 on `016q` | `window_sha256` `8fe92fd5…` matches raw lines 4696–4735; same `created_at` as the write |
| 125 | `llm_start` | **first** model input containing the June-10 values (e.g. `1821.7971284756836`, `76.51830258275933`, `4521.621182991773`) |
| 157/158 | final `llm_start` / `llm_end` | final answer (identical to native `answer`) |

- The June-10 values occur in exactly **two** tool outputs: the original page-1 `history` result (seq 25), which never appears in any `llm_start`, and the `read_file` result (seq 123).
- The page-1 `next_cursor` occurs in the `read_file` result (seq 32) but not in the substitute (seq 27). It first appears in a model input at seq 34, and is then used as the `cursor` argument of the page-2 call (seq 36).

### 5.4 Final answer [OBS]

- **Recommendation:** "Yes, I'd open a work order" for a tonnage-metering data-integrity fault (CM, priority 2–3). The answer offers to file it and does not create it.
- **Anomaly point 3** says "1,822 → 1,115 → 3,186 → 1,281 → 5,477 → 3,173 → 77 → 4,522 tons across consecutive 15-min samples on June 10". These are rounded values from file lines 4697–4732, timestamps 2020-06-10T16:15 to 18:00.
- **Other evidence in the answer has other sources:**
  - the 00:00–00:45 table comes from the small direct `history` call `toolu_01DC8KJ6…` (seq 133–134);
  - "193%" and "5,477"/mean/stddev come from `sensor_stats`;
  - the 6/21 and 6/30 off periods come from direct `history` calls (seq 151–156). The 6/21 zeros also appear in the seq-44 read.
- [INF] The read-retrieved sequence is **one of four** supporting observations for the recommendation. Whether the decision depended on it is not established. There is no counterfactual, so contribution is at most a hypothesis (schema §10.4).

---

## 6. The seven file-mediated transitions

T1–T7 are as defined in the older handoff §4.4, where "child" means the agent that ran the tool. "Synthetic" means the scripted delegated test. "621" means the natural single-agent run.

| # | Transition | Telemetry now | Synthetic (delegated) | 621 (single agent) | Status |
|---|---|---|---|---|---|
| T1 | tool → original result | `tool_end.output` + `output_sha256` | OBS | OBS (seq 25) | observed |
| T2 | original → substitute in agent context | `tool_result_committed.differs_from_raw=True` + content; next `llm_start` | OBS | OBS (seq 27, 28) | observed |
| T3 | original → stored file | `file_write` with runtime `tool_call_id`, sha | OBS | OBS (seq 26) | observed as a **queued** write; commit INF from later `file_read.created_at` + window sha |
| T4 | child state → `task` result | `delegation_end.files_returned` (meta only) | OBS | **did not occur** | observed (synthetic only) |
| T5 | `task` result → parent state (reducer merge) | none | INF from `files_returned` → later `file_read` metadata | **did not occur** | inferred only (LIMITATION) |
| T6 | state file → `read_file` result | `file_read` (path, window, `window_sha256`, timestamps) + `tool_end` | OBS | OBS (seq 122–123) | observed |
| T7 | read result → model input | `llm_start.input_batches` | OBS | OBS (seq 125) | observed |

---

## 7. A demonstrated 0.3 limitation vs evidence sufficient for 0.4

**What run 621 demonstrates** [OBS + INF under 0.3 rules]:

- **Representable:**
  - `delivers(P@28…, substitute)`, `delivers(P@119, grep result)` and `delivers(P@125…157, read result)`, observed in rendered inputs.
  - Whole-item non-delivery of the page-1 original, by quantitative bound (§10.1): +874 input tokens versus a 130,849-char item.
  - `propagates_to(read result → final answer)`, through the spans specific to the read (seven values plus "June 10"). The value "5,477" supports no link, by the tie rule, because the independent `sensor_stats` result was also delivered.
  - `propagates_to(page-1 read result → P@34 output)`, through the cursor value.
  - Exposed spans of the original at P@125, as exposure only.
- **Not representable:**
  - original → file → read result. No agent-input event lies between them, so `propagates_to` cannot be stored (§5.3 known limit, §12 DEFERRED). In 0.3 the read result is a tool-produced item with **no storable origin**.
  - The final answer as a product. `run.outputs` stays UNKNOWN because harness `answer` markers are OPEN (§4.1).
- **Coverage was not established.** No Anthropic tokenizer check was run, so non-delivery of specific spans stays UNKNOWN.
- **Change from run 620:** the gap is now **schema-side only**. Every raw step is observed and joined by tool-call ID and hash. In 620 the fragments "did not affect the run's product". In 621 retrieved fragments **are restated in the final answer**.

**What this does NOT establish:**

- It is **not a counterexample** to 0.3. 0.3 already *declares* this limit (DEFERRED/OPEN). A declared limit that a trace exhibits confirms the declaration. It does not falsify the schema.
- It is **not** the delegated lineage. No hierarchy or hand-back was exercised (T4 and T5 are absent).
- It does **not** show that the file-mediated fact *decided* the outcome. It was one of four observations, with no contrast evidence.

**Evidence sufficient to justify a 0.4 change.** This summarises what schema §13 requires ("each … must be representable with no core change, or must justify a specific change"). It is not a design. It needs at least all of the following:

1. A real, natural trace where the information flow in question occurs, ideally **genuine delegation** with the child's offloaded result retrieved by the parent (T1–T7 all present).
2. Complete raw telemetry for every step, so the failure can't be attributed to missing telemetry. This is now satisfied by the instrumentation.
3. A concrete Mission Control question that 0.3 must answer UNKNOWN or wrongly. One example: "from which tool call does the evidence behind this final claim originate?"
4. A demonstration that no existing inference rule (Exposed spans, the tie rule, delivery, claims) answers that question **without** violating a stated rule. Exposed spans explicitly forbids treating overlap as lineage.
5. Ideally, a consequential path: the retrieved fact is decisive, or there is contrast evidence that it mattered. That lets the question matter for diagnosis, not just provenance.
6. A separate decision, recorded in the AP repo, that this justifies a change. Not taken.

Run 621 satisfies (2), satisfies (3)/(4) only for the single-agent case, partly satisfies (5), and does not satisfy (1) for delegation.

---

## 8. Outstanding research questions

1. Will a natural real-model run delegate at all under the unchanged production prompt? So far 0 of 6 (Sonnet 5).
2. When a child's result is evicted, does a natural parent retrieve from the child's file? Or does it rely on the child's text handoff?
3. Does T5 (the parent reducer merge) need direct observation, or are `files_returned` plus `file_read` metadata enough?
4. Can 0.3 represent delegation and hierarchy with full telemetry? (§12 OPEN; HyperAgent trace 25 lacked rendered inputs.)
5. Lineage through tools and files (§12 OPEN): does any stated inference rule cover original → read result? Current reading: no.
6. Do benchmark-harness `answer` markers count for `run.outputs` (§4.1 OPEN)?
7. Can input coverage for Anthropic models be established through token-count residuals? Not attempted.
8. Availability and access (§12): unread data that would have mattered, for example page 3 of 621, which was never read.
9. Real-disk tools versus framework state files: the agent passed `/large_tool_results/...` to `profile_series`, which failed. Is that a separate class of failed flow worth representing?
10. Version risk: the backend factory hook breaks in deepagents v0.7.
11. Should the raw schema `assetopsbench.deepagent.instrumented/0.1` be bumped for the new additive fields? Ask the user; the version has not been bumped.
12. Deferred from the start: claims and states for stance, `depends_on`, and partial ordering for the parallel tool calls observed in 621 (for example seq 94–97).

---

## 9. Source files and raw traces

**AssetOpsBench (code):**

- `src/agent/deep_agent/runner.py`: `DeepAgentRunner.run`, Part C.
- `src/agent/deep_agent/_subagent_capture.py`: handler, `write_instrumented`, `record_file_op`, Part A.
- `src/agent/deep_agent/_file_observer.py`: Part B.
- `src/agent/deep_agent/tests/{test_file_observability,test_offloading_probe,test_subagent_capture,test_runner}.py`
- `src/agent/_prompts.py` (`AGENT_SYSTEM_PROMPT`), `src/agent/_cli_common.py` (`--run-id`, `--scenario-id`), `src/agent/runner.py` (`DEFAULT_SERVER_PATHS`).
- `src/servers/iot/main.py` (`history`, l.706; `PAGE_SIZE=1000`, l.92) and `src/servers/wo/main.py` (`generate_work_order`, l.281, a **mutating** tool).
- `src/observability/persistence.py` (`AGENT_TRAJECTORY_DIR`).
- Installed DeepAgents anchors: older handoff §4.1, under `.venv/lib/python3.12/site-packages/deepagents/`.
- `docs/agenttrace-file-observability-handoff.md`: the previous handoff, still the detailed internals reference.

**AP (research):**

- `docs/agenttrace.semantic.0.3.md`: canonical schema. Relevant sections are §5.1, §5.3, §10.1, §12, §13.1 and §4.1.
- `data/assetopsbench/scenarios/all_utterance.jsonl`: 152 scenarios.
- `data/assetopsbench/runs/` (untracked):
  - `aob-607/612/615/617/620-sonnet5-deep-20261005{.json,.instrumented.json}`: pre-file-instrumentation runs with no delegation. 620 has offloads.
  - `aob-621-sonnet5-deep-fileobs-20261013{.json,.instrumented.json}`: this checkpoint's natural run.
  - `traces.jsonl`: appended by 621 (§1.4).

---

## 10. Next recommended experiment

**Goal:** one natural, unscripted real-model run with **genuine delegation** in which a child's large tool result is evicted to `/large_tool_results/…`, handed back (`files_returned`), and retrieved by the parent with `read_file` or `grep`, with retrieved content restated in the final answer.

**Requirements:**

- Use an existing benchmark scenario verbatim.
- Keep the unchanged production prompt, tools and model route.
- Do not plant facts.

**Preparation before running anything:**

1. **Pre-register** the scenario, the success criteria (delegation occurred; a child `file_write`; parent `file_read` on the child's path; read content in a parent `llm_start`; a span of it carried into the final answer) and the stopping rule (one run, report honestly).
2. **Choose a scenario** whose text has independent sub-objectives, each needing a large Chiller-6 June-2020 result. Candidates to evaluate: 606/616 (temperature *and* power-input detectable failure modes), 607, 617 and 620.
   - None has delegated before (all ran with no delegation), but 617 and 620 ran before file instrumentation existed.
   - [HYP] Multi-sensor, multi-part wording raises the chance of delegation. This is untested.
3. **Use a fresh `OTEL_TRACES_FILE`** for that run (§1.4), a unique run ID, and record the WO DB doc count before and after.
4. **If it again does not delegate,** report that (natural rate 0/7). Then **ask the user** whether a *declared, non-production* experimental condition is acceptable, before doing anything that alters agent behavior. One example would be a different configured model, if credentials are added.

---

## NEXT_SESSION_START_HERE

1. Read this file, then `AP/docs/agenttrace.semantic.0.3.md` §5.1, §5.3, §10.1, §12 and §13. Use `docs/agenttrace-file-observability-handoff.md` §4 only for DeepAgents internals.
2. Verify the state without changing it:
   - `git status --short` in both repos should match §1.1 and §1.2;
   - `.venv/bin/python -m pytest src/agent/deep_agent/tests/ -q -p no:cacheprovider` should give 38 passed;
   - `curl http://localhost:5984/` should be up.
3. **Do not touch `AP/data/assetopsbench/runs/traces.jsonl`.** Ask the user whether to restore it to its first 79,114 bytes (50 lines) or leave it.
4. **Exact next task:** prepare, but do not yet run, the §10 experiment:
   - choose one existing scenario from 606, 607, 616, 617 or 620, using the criteria in §10;
   - write the pre-registration (scenario text, success criteria, run ID such as `aob-<id>-sonnet5-deep-fileobs-<date>`, a fresh per-run `OTEL_TRACES_FILE`, WO DB count check);
   - **show it to the user for approval;**
   - only after approval, execute exactly one run with `AGENTTRACE_CAPTURE_SUBAGENTS=1` and `--model-id anthropic:claude-sonnet-5`.
5. Analyse it exactly as in §5–§7: delegation IDs, `files_returned`, `file_write` and `file_read` joined by `tool_call_id`, first `llm_start` containing the retrieved content, and final-answer spans. Evaluate against 0.3 and report one of: schema counterexample / incomplete evidence / no counterexample.

Do not modify schema 0.3, design 0.4, stage, commit, push, or add co-author trailers.
