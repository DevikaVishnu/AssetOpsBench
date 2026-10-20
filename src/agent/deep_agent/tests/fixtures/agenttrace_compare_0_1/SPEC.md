# agenttrace.compare/0.1: derived exposure and state-vs-evidence comparison

Status: **EXPERIMENTAL, revision 3.** This is consumer-side inference over `agenttrace.semantic/0.3` objects. It is not part of the canonical schema.

- **Canonical schema:** `AP/docs/agenttrace.semantic.0.3.md`, sha256 `2273d210…25c01c06`. It is frozen; nothing here changes its semantics.
- **Scope.** Three operations, and no C1–C6:
  - O2 `exposure_at`;
  - claim delivery `claim_delivery`;
  - O3 `state_vs_evidence`.
- **Never stored.** No result is written into `semantic`. Results are returned to the caller; fixture expectations live in `*.expected.json` (viz-profile §7 convention).
- **Basis labels.**
  - **[0.3]** restates a canonical rule and cites its §.
  - **[PD]** (profile-derived) is a choice 0.3 does not make.
  - **[0.3+PD]** is a canonical rule with a profile-derived condition added.

  Every result carries `basis` and `provenance` (0.3 §10, §11 r1).
- **Forbidden outputs.** No result names or implies:
  - `conflicts_with` or `revises` (§6);
  - causation (§11 r12);
  - uptake or attention (§5.1, §11 r5);
  - belief persistence (§8);
  - "first exposure" or any other ordering judgment (§4.3, §12).
- **Approved decisions applied.**
  - D1: numeric matching rules are declared in advance.
  - D2: verified size bounds, recorded in `profile_local`.
  - D3: separate comparison fixtures.
  - D4: no external benchmark evidence for run 620.
  - D5: no negative verdicts that depend on completeness until independent annotation coverage exists.

Revision 2 applies the pre-implementation schema-auditor review (2026-10-20): MUST-FIX 1–3 and every SHOULD-FIX item.

Revision 3 applies the post-implementation review (2026-10-20):
- MUST-FIX 1: unresolved hedges in testimony (§5.3);
- MUST-FIX 2: precision is stated in the proposition form (§6.3);
- SHOULD-FIX items: the ungated bound exclusion (§3.2), span validation (§3), underivable speakers (§2), the `direct` basis (§5.1), and per-speaker stances (§5.3).

## 1. Input layout [PD]

A comparison fixture `<name>.trace.json` has these keys.

| Key | Contents | Status |
|---|---|---|
| `profile` | `"agenttrace.compare/0.1"` | identifier |
| `semantic` | the eight 0.3 collections with the fields of 0.3 §4 | **canonical 0.3 instance** |
| `profile_local` | `fixture_class`, synthetic notice, coverage, size bounds, numeric matching, extraction notes, link assessments, ambiguities, annotation | **non-canonical** |

**Invariants.**
- Deleting `profile_local` and `*.expected.json` leaves a complete 0.3 serialization.
- Each fixture is a **slice**: an object missing from it is not a claim about the run.

**Content and coordinates.** 0.3 stores no raw content. Operations that match content take a `texts` map, from evidence id to the item's text.
- `texts[X]` is the text of X's **produced occurrence**, resolved by the caller from `X.raw_ref`.
- If `X.raw_ref` names a `text_sha256` (or a `sha256` of the text form), the caller must check the resolved text against it. Fixture tests do this.
- A claim span `C.raw_ref.span` is a pair of code-point offsets into `texts[C.evidence]` (viz-profile M3: the span is located in the produced occurrence).
- If the claim's evidence is external (it has no `produced_by`), its span coordinates are not defined in 0.1, and span-based results are `UNKNOWN`.
- A missing text never counts as "no match". It makes the affected part `UNKNOWN`.

**Undecided values** use the viz-profile §2 form `{"$unresolved": "<amb id>", "candidates": [...]}`. Consumers report them as UNRESOLVED and never read them as UNKNOWN.

## 2. Derived anchors

- **speaker(C)** [0.3 §4.5]. Let e be C's evidence.
  - If `e.source = agent`, the speaker is the `origin` of `e.produced_by`.
  - Otherwise it is the source class, written `source:<class>`.
  - If any value on that path is `$unresolved`, the speaker is UNRESOLVED.
  - **Underivable speaker.** If agent evidence has no producer, or its producer has no origin, the speaker is underivable. The claim is listed as unresolved (`speaker-underivable:<evidence id>`) and is never pooled under `source:agent`.
- **holder(S) and event(S)** [0.3 §4.7] are the shared speaker and the shared `produced_by` of S's grounding claims.
- **State validity** [PD, viz-profile M1]:
  - every grounding claim's evidence has `source: agent`;
  - every such evidence item has a `produced_by` event;
  - that event is one shared event;
  - its origin is an `agent` entity.

  0.3 itself requires only that the holder be an entity.
- **Malformed input.** A serialized state whose anchor is `$unresolved` (in claims, their evidence, `source`, `produced_by` or origin), or which violates M1, is **malformed input**. The operation raises `InvalidInput`; this is not an inference result. M1 says such a state must be omitted.
- **UNRESOLVED results.** Only a `$unresolved` proposition or value on the state produces UNRESOLVED results (§5).

## 3. O2 `exposure_at(E, X, spans=None)`

E is an event and X an evidence item. `spans` is an optional list of `[start, end]` code-point spans in `texts[X]`.

**Span validation.** Each span must be two integers with 0 ≤ start < end, and must end within the text when the text is known. A malformed span raises `InvalidInput`. A `$unresolved` span inside a link's `raw_ref.spans` makes that link pending with reason UNRESOLVED (§5.2).

### 3.1 Applicability [PD]
1. If E's `origin` is `$unresolved`, the result is `whole_item = UNRESOLVED`.
2. If E has no origin, or its origin is not an `agent` entity, the result is `whole_item = NOT_APPLICABLE`. `delivers` starts only at agent-input events [0.3 §4.3, §5].

### 3.2 Negative-result gate (MUST-FIX 1, D5) [PD]

A negative statement about E's input is allowed only if `negatives_enabled(E)` holds, which requires both of the following.
- **Coverage.** `profile_local.coverage` has an entry for E with:
  - `status: "established"`;
  - a non-empty `basis`, naming the §10.1 reconstruction (tokenizer, residual, memory flags, template slots);
  - `input_exhausted: true`. This asserts that the fixture's delivered items for E, plus the recorded `residual` text, exhaust E's reconstructed input.
- **Annotation.** `profile_local.annotation.completeness` is `"established"`, with at least two independent annotators.

The negative statements are W3, `NOT_EXPOSED_EXACT`, claim delivery `FALSE`, and "excluded" in O3.

Otherwise every such statement is UNKNOWN.

**Ungated exclusion.** A whole-item size bound (W4) is quantitative, not completeness-dependent (0.3 §10.1 bound row). Excluding an *absence-grounded* link through WHOLE_NOT_DELIVERED (§5.2) therefore needs no gate. It is the only negative that is ungated, and it never applies to subsets or spans.

`status` takes the values `established`, `not_established` and `unassessed`. Coverage provenance is inference provenance, never a 0.3 field [0.3 §10.1, §12].

### 3.3 Whole item

The first matching row wins.

| # | Condition | `whole_item` | Basis |
|---|---|---|---|
| W1 | a stored `delivers(E, X)` with resolved relation and endpoints | `DELIVERED` | [0.3 §5.1] |
| W2 | a link with at least one `$unresolved` field, some completion of which is `delivers(E, X)`: relation `delivers` or a candidate including it, source E or a candidate set containing E, target X or a candidate set containing X | `UNRESOLVED` (amb ids) | [PD] |
| W3 | `negatives_enabled(E)` | `NOT_DELIVERED` | [0.3+PD]: §10.1 Non-delivery and §11 r4, plus the §3.2 gate |
| W4 | a verified size bound for (E, X) with `bound < size` (§6.2) | `WHOLE_NOT_DELIVERED` | [0.3 §10.1 bound row] |
| W5 | otherwise | `UNKNOWN` | [0.3 §5.1, §11 r4] |

- W4 is evaluated even when W3 applies, and both are then reported in the provenance.
- W4 never decides anything about a subset of X.
- UNRESOLVED is never collapsed to UNKNOWN or NOT_DELIVERED.
- Absence is never inferred from a missing link, silence, timing, names, roles or counters [0.3 §11 r4, r11].

### 3.4 Exposed spans (only when `spans` is given)

- **Delivered set.** The items D ≠ X with a resolved stored `delivers(E, D)`.
- **`exposed`** [0.3 §10.1 Exposed spans]. A requested span s is exposed iff `texts[X][s]` occurs as an **exact** substring of `texts[D]` for some D in the delivered set. `exposed` lists `{span, via: D}`. No normalization of whitespace, JSON escaping, tokens or numbers is applied.
- **`others`**, the requested spans not found [PD, built on §11 r4]:
  - `NOT_EXPOSED_EXACT` only if `negatives_enabled(E)` holds, the texts of X, of every delivered item and of the coverage `residual` are available, and the span is in none of them;
  - otherwise `UNKNOWN`.

  A failed exact match is not absence of the content in other forms.
- **If W1 holds**, the spans result is `{"via": "whole_item"}`.
- **Meaning.** This is exposure only. A content match establishes no derivation, lineage or uptake [0.3 §10.1, §5.3], and there is no `partial_delivers`.
- **Caller's responsibility.** Short or generic spans such as `0.0` can match by coincidence, so callers ask only for spans that carry the content in question.

## 4. Claim delivery `claim_delivery(E, C)` [0.3 §10.1]

Claim delivery runs O2 on (E, C.evidence) with `spans = [C.raw_ref.span]`. `value` takes only the 0.3 values TRUE, FALSE and UNKNOWN, plus the PD values UNRESOLVED and NOT_APPLICABLE.

| Condition | `value` | Basis |
|---|---|---|
| `whole_item = DELIVERED` | `TRUE` | [0.3] |
| `whole_item` is UNRESOLVED or NOT_APPLICABLE, or C's evidence or span is `$unresolved` | that value | [PD] |
| `whole_item = NOT_DELIVERED` and the claim span is `NOT_EXPOSED_EXACT` | `FALSE` | [0.3+PD]: 0.3 "only under coverage", plus the §3.2 gate and the span condition |
| otherwise, including `WHOLE_NOT_DELIVERED` (the bound says nothing about subsets) | `UNKNOWN` | [0.3] |

**`content_exposed_via`** [PD] is a separate field. When the claim span is exposed through another delivered item, it lists those items.
- It is **not** claim delivery: C.evidence did not reach E, and the relaying item carries its own claims (§7.3).
- It never changes `value`.

## 5. O3 `state_vs_evidence(S)`

S = (P, v, claims), with holder H and event E (§2). If P or v is `$unresolved`, every part of the result is `UNRESOLVED` [PD].

O3 returns three separate results that are **never merged** [0.3 §9, §7.2 r6].

### 5.1 `direct` [0.3 §9, §5.5, §5.6]

The direct links are the stored `supported_by` and `contradicted_by` links whose source is P.

| Links | `value` |
|---|---|
| only `supported_by` | `TRUE` |
| only `contradicted_by` | `FALSE` |
| both | `UNCERTAIN` |
| neither | `UNKNOWN` |

**Open links** [PD]. A link is *open* when its relation or source is `$unresolved` and some completion makes it a direct link on P.
- A link whose relation and source P are resolved is a definite direct link even if its target is `$unresolved`, because the table depends only on the relation.
- The table is evaluated over every completion of the open links. In each completion, an open link takes each candidate relation, and is left out only when a candidate (non-direct relation, or a source other than P) allows that.
- If all completions agree, that value is returned with `stable_over_unresolved: true`. Otherwise the value is `UNRESOLVED`.
- When open links exist, the basis of `direct` is `PD`. It is `0.3` only when every direct link is definite.

**`comparison`** [PD] sets `direct` against v.

| Condition | `comparison` |
|---|---|
| v ∉ {TRUE, FALSE} | `NOT_APPLICABLE`; the values are reported side by side |
| `direct` is UNKNOWN or UNRESOLVED | the same value |
| `direct = UNCERTAIN` | `AMBIGUOUS` |
| `direct = v` | `AGREES` |
| otherwise | `DIFFERS_FROM_DIRECT`, read as "within the stored links". It is never rendered as conflict, error or falsity of the holder |

`direct` is an assessment of P relative to the stored links. It says nothing about link completeness, and a state's value is never read as truth [0.3 §8, §9].

### 5.2 `delivered_direct` [PD, built on 0.3 §9 and §10.1]

For each definite direct link L on P with a resolved target X, run O2 on (E, X), with `spans` set to L's grounding spans. L's `raw_ref` holds either `spans` or `whole_item: true` (absence-grounded).

| L's grounding | O2 at E | Class |
|---|---|---|
| any | `DELIVERED` | **counted**, `grounding: delivered` |
| spans | not DELIVERED, every grounding span `exposed` | **counted**, `grounding: exposed-span` |
| spans | `NOT_DELIVERED`, some span `NOT_EXPOSED_EXACT` | **excluded** (requires the §3.2 gate) |
| `whole_item` | `NOT_DELIVERED` or `WHOLE_NOT_DELIVERED` | **excluded** |
| anything else (UNKNOWN, UNRESOLVED, WHOLE_NOT_DELIVERED with spans not all exposed, NOT_APPLICABLE) | | **pending** |

Links with an unresolved target, and open links, are **pending** as well.

**Value.**
- `counted_value` is the §9 table over the counted links.
- `value = counted_value` if no subset of the pending links could change the table result.
- Otherwise `value` is `UNRESOLVED` if any pending link is pending because of `$unresolved`, and `UNKNOWN` if none is. Both cases carry `exposure_incomplete: true`.

The value is relative to the stored links. It does not imply that the stored links are complete (D5).

**Guards.**
- `delivered_direct` is never uptake [0.3 §11 r5].
- It never feeds Unexposed or Gate-miss without established coverage [0.3 §10.4].
- A completeness-dependent negative such as NO_DELIVERED_DIRECT_SUPPORT is **not produced in 0.1** (D5).

### 5.3 `testimonial` [0.3 §9, §7.2 r7–r8, §7.3; aggregation PD]

**Inputs.**
- The resolved claims whose proposition is P, or whose proposition form is syntactically `not(<P.form>)`. Identity is syntactic (viz-profile D3); double negation is never computed.
- Claims with an UNRESOLVED speaker are listed under `unresolved` and not counted.

**Exclusion.** Claims by H are excluded [0.3 §7.2 r7].

**Lineage groups** [PD].
- The remaining claims are grouped by speaker.
- Speaker groups whose claims' evidence items are connected through stored `propagates_to` links (undirected, transitive) form one lineage group [0.3 §7.2 r8].
- A group connected in the same way to the evidence of any of H's claims is flagged `lineage_connected_to_holder: true`.
  - The flag is undirected and evidence-level only. It says neither which way content flowed nor that the claim span itself was carried (§5.3, §10.3).
  - Such a group is not counted as independent of the holder.
- **Non-agent speakers** [0.3 §4.5]. The speaker is a source class, so distinct documents or tools with the same class share one group.

**Contribution per group** [PD]. The first matching rule wins.
1. A claim grounds a state whose value is UNCERTAIN or UNKNOWN (a hedge, §4.5, §8): the contribution is `UNCERTAIN`.
2. A claim grounds a state with value FALSE: the claim is listed under `flagged_stance`. It is not counted, because stance is unexercised (§12 known limit; fixture-README ambiguity 2).
3. The group's claims assert both P and not(P): the contribution is `UNCERTAIN` [0.3 §9, a testifier who contradicts themselves].
4. All the claims assert P: `TRUE`.
5. All the claims assert not(P): `FALSE`.

**Unresolved state values.** If a member claim grounds a state whose value is `$unresolved`, rules 1–5 are evaluated over every completion of those values.
- If the completions agree, that contribution is used.
- Otherwise the contribution is `UNRESOLVED`, with the amb ids.
- It is never collapsed to the resolved part.

**Per-speaker stances.** Each group also reports `per_speaker`, the rule 1–5 result for each speaker alone (`NOT_COUNTED` when nothing counts). Lineage merging deduplicates relayed testimony; it does not hide a speaker who asserts the negation.

**Independence.** With two or more groups, independence is `UNKNOWN`. 0.1 never establishes independence [0.3 §9].

**Output.**
- `value` is `UNKNOWN` when no testimony remains.
- Otherwise it is `REPORTED`, with the per-group contributions.
- **There is no aggregate truth value**; 0.3 defines none.
- The PROVISIONAL implication extension through `depends_on` is excluded [0.3 §12].

## 6. `profile_local` blocks [PD]

### 6.1 `coverage`

Each entry has the form `{event, status, basis, input_exhausted, residual}` (§3.2).

**No real fixture in 0.1 establishes coverage.** No tokenizer reconstruction has been run on any AssetOpsBench trace.

### 6.2 `size_bounds` (D2)

Each entry has the form `{event, item, bound: {quantity, value}, size: {quantity, value}, basis, verified}`.

An entry is used only if all of these hold:
- `verified` is true;
- both `quantity` values are equal;
- `basis` names the raw records behind both numbers: the logged usage of E and of the previous call, the content already accounted for, and the tokenizer used to size X.

W4 applies iff `bound.value < size.value`. Any other entry is ignored.

### 6.3 `numeric_matching` (D1)

Each entry has the form `{proposition, rule, justification, declared, values}`.

- **Default rule: `exact`.**
- **`nearest_integer`.** The stated integer n matches the source value x iff |x − n| < 0.5. It is allowed only when the claim states integers and the justification cites the claim's own display precision.
- **No percentage or range tolerance.**
- **The form states the precision** (for example `values_rounded_to_integer=[…]`). A stored `supported_by` link must then be sound from `semantic` alone; `profile_local` only records the declaration and its justification.
- **`declared`** records where and when the rule was fixed. It also records whether the raw data had already been inspected, which is always true for a fixture built after the run.

The operations do not apply these rules; they consume links. Fixture tests re-check every value listed in `values` against raw data under the declared rule.

### 6.4 `link_assessment`, `annotation`, `ambiguities`

- **`link_assessment`** records each delivered item that was assessed for direct bearing on P, and why a link was or was not extracted.
- **`annotation`** has the form `{annotators: [...], completeness: "not_established" | "established"}` and feeds the §3.2 gate.
- **`ambiguities`** has the form `{id, field_path, reason, basis}`.

## 7. Output shape [PD]

Every result has the shape `{"op", "inputs", <op fields>, "basis": {<field>: "0.3" | "PD" | "0.3+PD"}, "provenance": [...]}`.

- `provenance` lists the stored links, the `raw_ref` locators and the `profile_local` entries used.
- Results are deterministic: the same inputs give identical JSON under `sort_keys`.
- Operations never mutate their inputs.

## 8. Known limits of 0.1

- **No ordering judgments.** "Before", earliest delivery and first exposure are not provided, because partial order is DEFERRED (§4.3, §12).
- **No relations between propositions.** `depends_on` is unused, and implication is DEFERRED.
- **No proposition normalization.** There is no negation normalization and no grouping across versions.
- **Testimonial independence is never established.**
- **Exact matching only.** Exposed spans use exact substrings with no normalization.
- **No completeness-dependent negatives on real fixtures.** No real fixture can produce one until coverage is established and a second annotator has worked.
