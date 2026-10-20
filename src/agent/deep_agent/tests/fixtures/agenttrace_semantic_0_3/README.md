# Hand-grounded agenttrace.semantic/0.3 fixtures

Two minimal fixtures, written by hand from real DeepAgents traces and checked
offline by `../../test_semantic_fixtures.py`. They use only the eight 0.3
collections and the fields listed in 0.3 §4. Everything else (locators,
hashes, adapter decisions, notes) is in `raw_ref`.

- Canonical schema: `AP/docs/agenttrace.semantic.0.3.md`, sha256 `2273d210…25c01c06`. It is unchanged.
- The raw traces are read-only evidence in `AP/data/assetopsbench/runs/`. The tests re-hash them after reading.
- Each fixture is a **slice**. An object missing from it is not a claim about the run.

## `aob-621.semantic.json`: run 621, single agent

| Object | Anchor |
|---|---|
| R, the `read_file` result (`toolu_01S66j…`) | `tool_end` 123, sha `0734f555…`; committed 124 |
| `delivers(llm 125, R)` | first rendered input with R, message 46 |
| `delivers(llm 157, R)` | the input of the answer call, message 46 |
| A, the final answer | `llm_end` 158, `content[1].text`, sha `293d820e…`, 2,715 chars |
| claim, proposition, state (root, TRUE) | A span [1304:1428]: "with jumps like 1,822 → … → 4,522 tons across consecutive 15-min samples on June 10" |
| `propagates_to(R → A)` | A span [1320:1349] `1,822 → 1,115 → 3,186 → 1,281` ← R spans of `1821.797…`, `1114.934…`, `3185.564…`, `1281.168…` |

**Why `propagates_to` is stored (0.3 §5.3).**
- (1) A is produced by the llm 157 call.
- (2) That call's rendered input delivers R.
- (3) The span restates R's specific values.
- (4a) The values are not generic.
- (4b) None of the four values, at full precision or rounded, occurs in any other message of the 59-message input of llm_start 157. The test checks this, ignoring opaque thinking signatures.

**Limits on this link.**
- **[INF] Rounding.** The model rounded each value to the nearest integer and added a thousands separator. The trace does not record this rule.
- **Where the span stops.** The next value, 5,477 (5476.77), is also in four other messages delivered to llm 157: the `sensor_stats` result (msg 9), the model's own `grep` arguments (msgs 40, 43) and the `grep` result (msg 44). The tie rule therefore stops the span before it.

**Not represented, by design.**
- **Upstream hops.** The chain is: page-1 `history` result (`tool_end` 25) → framework `file_write` 26 → 967-char substitute (27, delivered at 28) → R. These hops are tool → file → tool, which 0.3 §5.3 (known limit) and §12 do not let you link.
- **`run.outputs` is UNKNOWN.** The harness `answer` field equals A exactly, but harness markers are OPEN (§4.1), so the marker stays in `raw_ref`.
- **Hash correction (2026-10-20).** The recorded `tool_end` 25 is the list form, 145,917 chars, sha `5d37d836…`. The figures 130,849 chars / sha `38478099…` belong to its joined text, which is what `file_write` 26 stored.

## `aob-sc3-B.semantic.json`: scenario_3 condition B, root + `general-purpose` child

**Represented.**
- Root's `history` result O (38/39). The framework's eviction write of it to file F (40).
- Root's delegation description D (`llm_end` 93).
- The child's `grep` result G (380–382) and three `read_file` results R1–R3 (388–391, 394–397, 400–403). All three reads target root's file F.
- The first deliveries, which are the only links:

| Link | Seq | Message |
|---|---|---|
| `delivers(sub-1 llm, D)` | 96 | 1 (human) |
| `delivers(sub-1 llm, G)` | 386 | 7 |
| `delivers(sub-1 llm, R1)` | 392 | 9 |
| `delivers(sub-1 llm, R2)` | 398 | 11 |
| `delivers(sub-1 llm, R3)` | 404 | 13 |

Later re-deliveries are not listed. They are at 392–410, and at 416, the refused call. **Whether 416 counts as an event is undecided in 0.3 §4.3.**

**The three layers of the handoff.**

1. **Availability: not represented.**
   - The child's state copy at delegation has no event.
   - Availability is DEFERRED/OPEN (0.3 §10.1, §11 r3, §12).
   - The first observable sign is the path appearing inside G (381).
2. **Retrieval: evidence only.**
   - G and R1–R3 are evidence produced by the child's tool events.
   - The joins to root's file are path + `tool_call_id` + `created_at` (equal to root's own `file_read` 45) + window hash. They live in `raw_ref` only.
   - No O → F or F → R_i link exists: lineage cannot cross non-agent-input events (§5.3, §12).
3. **Model exposure: stored.** It is the five `delivers` links above.

**Not represented, by design.**
- **No claims, propositions or states.** The child's model outputs after 386 contain only empty thinking blocks and tool calls.
- **No `propagates_to`.**
- **No parent/child relation.** Hierarchy is DEFERRED; it is kept in entity `raw_ref` only.
- **`delegation_start` 94** stays a framework label.

**Identical-sha confound.**
- The child's own eviction file `/large_tool_results/toolu_01VwZ82BKDyeSFwC5oMi3FwH` (`file_write` 135, sub-1) has the **same sha256 `00ab1468…` as root's file F**.
- Value or hash matching alone therefore cannot tell root-created content from child-created content. Only the path and `tool_call_id` join can.
- Under §5.3(4b), any later child restatement of these values would tie between the two items, and no link would be stored.

**Budget-stopped execution.**
- The run status is `error` (`BudgetExceeded`, guard outcome BUDGET-STOPPED, `max_cost_usd`).
- The guard refused the child's call at `llm_start` 416 before any provider request; it has no `llm_end`. Then come `agent_error` 417, `delegation_error` 418 and `agent_error` 419.
- The child returned nothing (`returned_to_parent` is null), so the return path of the delegation is unexercised.
- There is no product marker, so `run.outputs` is UNKNOWN.

## Adapter decisions (in `raw_ref`, not in the core)

- **DeepAgents `read_file` and `grep` are `tool` entities.**
- **The eviction write `file_write` 40 has origin `framework:deepagents`.**
  - The capture records `agent_id: root`, because the write runs inside root's tool-call context.
  - The write itself is `FilesystemMiddleware`'s large-result eviction (deepagents 0.5.3 `middleware/filesystem.py:1344-1347`).
  - F's `source` is `tool`, because its content is the `history` tool's text (§4.4: source is the author class, `produced_by` the producing event).
- **The summarizer call 383 is not represented, and its origin is left UNRESOLVED.**
  - The record attributes it to sub-1 with `offered_tools: null`.
  - The code suggests the framework's summarization middleware initiates it. The trace does not establish this, and 0.3 does not fix the origin of framework-initiated model calls inside an agent.
  - No delivery in this slice depends on it: G first appears at 386, in the verbatim tail kept after the summary, and 383's input does not contain G.
- **Per-agent `model`** comes from the record's run-level `model` field.

## Schema ambiguities met (no change to 0.3)

1. **`raw_ref` on propositions and states.** §3 and §11 r1 say inferred objects "(claims, propositions, states, links)" carry `raw_ref`, but the field tables in §4.6 and §4.7 do not list it. The fixtures follow the field tables, so states reach raw data through their claims.
2. **Claim proposition vs state proposition.** 0.3 does not require a state's grounding claims to assert the state's proposition. The test applies this as a stricter fixture rule.
3. **`order` for multi-record events.** `order` is "temporal order". The fixtures use the seq of each event's first raw record; within these slices that gives a total order.
4. **Content first seen only in a rendered input** (for example the eviction substitute) has no `raw_ref` rule in §4.4. The fixtures avoid such items.
5. **Origin of a framework-initiated model call inside an agent** (summarizer 383). §4.3 does not decide it, and the choice decides whether 383 is an agent-input event. Left UNRESOLVED and not represented.
6. **Refused call with an input and no output** (416). §4.3 defines an event as having an input and an output, so it is undecided whether 416 is an event and whether anything is delivered to it. Not represented.
7. **"Not observable" premise in §12 (OPEN, lineage through tools and files).** That text says file writes and reads are not observable events. In this fileobs capture they are (`file_write` 40, `file_read` 389 …). Lineage F → R_i is still not storable, because `delivers` needs an agent-input source (§5.3 known limit). The outcome is unchanged, but the premise does not hold for this capture.
