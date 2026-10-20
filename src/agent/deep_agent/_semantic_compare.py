"""Offline derived comparisons over agenttrace.semantic/0.3 objects.

Implements ``agenttrace.compare/0.1`` (spec:
``tests/fixtures/agenttrace_compare_0_1/SPEC.md``):

  * O2 :meth:`Comparator.exposure_at` -- was evidence item X in the rendered
    input of agent-input event E, in whole or by exact spans;
  * :meth:`Comparator.claim_delivery` -- 0.3 §10.1 claim delivery;
  * O3 :meth:`Comparator.state_vs_evidence` -- a holder's state against the
    stored direct links (``direct``), the subset delivered to the state's event
    (``delivered_direct``) and other testifiers' claims (``testimonial``).

The operations only read.  Inputs are deep-copied; nothing is written back into
the 0.3 objects, and no result names ``conflicts_with``, ``revises``, causation,
uptake, ordering or belief persistence.  Negative results (non-delivery,
non-exposure, claim delivery FALSE, excluded links) are gated on established
coverage *and* established annotation completeness (spec §3.2); without both,
the answer is UNKNOWN.  ``$unresolved`` values are reported as UNRESOLVED and
never collapsed to UNKNOWN.  Malformed states raise :class:`InvalidInput`.

No model, network or database is used.
"""

from __future__ import annotations

import copy
import itertools
from typing import Any, Iterable, Mapping

PROFILE = "agenttrace.compare/0.1"
UNRESOLVED_KEY = "$unresolved"
DIRECT = ("supported_by", "contradicted_by")
_EXCLUDED = None  # completion choice: the open link is not a direct link on P

TRUE, FALSE, UNKNOWN, UNCERTAIN = "TRUE", "FALSE", "UNKNOWN", "UNCERTAIN"
UNRESOLVED, NOT_APPLICABLE = "UNRESOLVED", "NOT_APPLICABLE"


class InvalidInput(ValueError):
    """The input is not a valid 0.3 object for the operation (not an inference result)."""


# --- $unresolved helpers ------------------------------------------------------


def is_unresolved(value: Any) -> bool:
    return isinstance(value, dict) and UNRESOLVED_KEY in value


def candidates(value: Any) -> list:
    """Possible values of a field: its candidates if unresolved, else itself."""
    return list(value.get("candidates", [])) if is_unresolved(value) else [value]


def _amb(value: Any) -> str:
    return value[UNRESOLVED_KEY]


def _table(relations: Iterable[str]) -> str:
    """0.3 §9 direct-assessment table."""
    rels = set(relations)
    sup, con = "supported_by" in rels, "contradicted_by" in rels
    if sup and con:
        return UNCERTAIN
    if sup:
        return TRUE
    if con:
        return FALSE
    return UNKNOWN


def _one(outcomes: set) -> str:
    """Collapse per-completion outcomes: one value, UNRESOLVED if they differ, NOT_COUNTED if none."""
    if len(outcomes) != 1:
        return UNRESOLVED
    value = next(iter(outcomes))
    return "NOT_COUNTED" if value is None else value


class _Unres:
    """Marker for a derived value that depends on ``$unresolved`` fields."""

    def __init__(self, ambs: Iterable[str]):
        self.ambs = sorted(set(ambs))


# --- comparator -----------------------------------------------------------------


class Comparator:
    """Deterministic comparison operations over one 0.3 slice.

    ``semantic`` holds the eight 0.3 collections; ``profile_local`` the
    non-canonical blocks of spec §6; ``texts`` maps evidence ids to the text of
    each item's produced occurrence (resolved and hash-checked by the caller).
    """

    def __init__(
        self,
        semantic: Mapping[str, Any],
        profile_local: Mapping[str, Any] | None = None,
        texts: Mapping[str, str] | None = None,
    ):
        self.semantic = copy.deepcopy(dict(semantic))
        self.profile_local = copy.deepcopy(dict(profile_local or {}))
        self.texts = dict(texts or {})
        s = self.semantic
        self.entities = {e["id"]: e for e in s.get("entities", [])}
        self.events = {e["id"]: e for e in s.get("events", [])}
        self.evidence = {e["id"]: e for e in s.get("evidence", [])}
        self.claims = {c["id"]: c for c in s.get("claims", [])}
        self.propositions = {p["id"]: p for p in s.get("propositions", [])}
        self.states = list(s.get("states", []))
        self.links = list(s.get("links", []))

    # --- lookups ----------------------------------------------------------------

    def _event(self, event_id: str) -> dict:
        if event_id not in self.events:
            raise InvalidInput(f"unknown event {event_id!r}")
        return self.events[event_id]

    def _item(self, item_id: str) -> dict:
        if item_id not in self.evidence:
            raise InvalidInput(f"unknown evidence {item_id!r}")
        return self.evidence[item_id]

    def _resolved_links(self, relation: str) -> list[dict]:
        return [
            l
            for l in self.links
            if l["relation"] == relation and not is_unresolved(l["source"]) and not is_unresolved(l["target"])
        ]

    def _delivered_items(self, event_id: str) -> list[str]:
        return sorted({l["target"] for l in self._resolved_links("delivers") if l["source"] == event_id})

    # --- derived anchors (spec §2) ---------------------------------------------

    def speaker(self, claim: dict) -> str | _Unres:
        """0.3 §4.5: origin of the producing event for agent evidence, else the source class."""
        ev_id = claim["evidence"]
        if is_unresolved(ev_id):
            return _Unres([_amb(ev_id)])
        item = self._item(ev_id)
        source = item["source"]
        if is_unresolved(source):
            return _Unres([_amb(source)])
        if source != "agent":
            return f"source:{source}"
        producer = item.get("produced_by")
        if producer is None:
            return _Unres([f"speaker-underivable:{ev_id}"])
        if is_unresolved(producer):
            return _Unres([_amb(producer)])
        origin = self._event(producer).get("origin")
        if is_unresolved(origin):
            return _Unres([_amb(origin)])
        return origin if origin is not None else _Unres([f"speaker-underivable:{ev_id}"])

    def state_anchor(self, state: dict) -> tuple[str, str]:
        """Holder and event of a state (0.3 §4.7) under viz-profile M1; raise if malformed."""
        claim_ids = state.get("claims")
        if is_unresolved(claim_ids) or not claim_ids:
            raise InvalidInput("state has unresolved or empty grounding claims")
        events, holders = set(), set()
        for cid in claim_ids:
            if is_unresolved(cid) or cid not in self.claims:
                raise InvalidInput(f"state grounding claim {cid!r} is unresolved or unknown")
            ev_id = self.claims[cid]["evidence"]
            if is_unresolved(ev_id):
                raise InvalidInput(f"claim {cid} has unresolved evidence")
            item = self._item(ev_id)
            if item["source"] != "agent":
                raise InvalidInput(f"claim {cid}: grounding evidence source is not 'agent' (M1)")
            producer = item.get("produced_by")
            if producer is None or is_unresolved(producer):
                raise InvalidInput(f"claim {cid}: grounding evidence has no resolved producer (M1)")
            origin = self._event(producer).get("origin")
            if origin is None or is_unresolved(origin):
                raise InvalidInput(f"claim {cid}: producing event has no resolved origin (M1)")
            kind = self.entities.get(origin, {}).get("kind")
            if kind != "agent":
                raise InvalidInput(f"claim {cid}: producing event origin {origin!r} is not an agent entity (M1)")
            events.add(producer)
            holders.add(origin)
        if len(events) != 1 or len(holders) != 1:
            raise InvalidInput("state grounding claims do not share one producing event and speaker (0.3 §4.7)")
        return holders.pop(), events.pop()

    # --- profile_local gates (spec §3.2, §6) -------------------------------------

    def _coverage(self, event_id: str) -> dict | None:
        for entry in self.profile_local.get("coverage", []):
            if (
                entry.get("event") == event_id
                and entry.get("status") == "established"
                and entry.get("basis")
                and entry.get("input_exhausted") is True
            ):
                return entry
        return None

    def _annotation_established(self) -> bool:
        ann = self.profile_local.get("annotation", {})
        return ann.get("completeness") == "established" and len(set(ann.get("annotators", []))) >= 2

    def negatives_enabled(self, event_id: str) -> bool:
        return self._coverage(event_id) is not None and self._annotation_established()

    def _size_bound(self, event_id: str, item_id: str) -> dict | None:
        for entry in self.profile_local.get("size_bounds", []):
            if entry.get("event") != event_id or entry.get("item") != item_id:
                continue
            bound, size = entry.get("bound", {}), entry.get("size", {})
            if (
                entry.get("verified") is True
                and entry.get("basis")
                and bound.get("quantity")
                and bound.get("quantity") == size.get("quantity")
                and isinstance(bound.get("value"), (int, float))
                and isinstance(size.get("value"), (int, float))
                and bound["value"] < size["value"]
            ):
                return entry
        return None

    # --- O2 (spec §3) -------------------------------------------------------------

    def exposure_at(self, event_id: str, item_id: str, spans: list[list[int]] | None = None) -> dict:
        event = self._event(event_id)
        self._item(item_id)
        if spans is not None:
            self._check_spans(item_id, spans)
        prov: list = []
        basis: dict = {}
        result: dict = {"op": "O2", "inputs": {"event": event_id, "item": item_id, "spans": spans}}

        origin = event.get("origin")
        if is_unresolved(origin):
            result.update(whole_item=UNRESOLVED, unresolved=[_amb(origin)])
            basis["whole_item"] = "PD"
            prov.append({"event_origin": origin})
        elif origin is None or self.entities.get(origin, {}).get("kind") != "agent":
            kind = self.entities.get(origin, {}).get("kind") if origin else None
            result["whole_item"] = NOT_APPLICABLE
            basis["whole_item"] = "PD"
            prov.append({"event_origin": origin, "origin_kind": kind, "rule": "delivers starts at agent-input events (0.3 §4.3, §5)"})
        else:
            self._whole_item(event_id, item_id, result, basis, prov)

        if spans is not None:
            result["spans"] = self._spans(event_id, item_id, spans, result["whole_item"], basis, prov)
        result["basis"] = basis
        result["provenance"] = prov
        return result

    def _whole_item(self, event_id: str, item_id: str, result: dict, basis: dict, prov: list) -> None:
        definite = [l for l in self._resolved_links("delivers") if l["source"] == event_id and l["target"] == item_id]
        if definite:  # W1
            result["whole_item"] = "DELIVERED"
            basis["whole_item"] = "0.3"
            prov.extend({"link": "delivers", "raw_ref": l["raw_ref"]} for l in definite)
            return
        open_ambs = []
        for l in self.links:  # W2
            fields = (l["relation"], l["source"], l["target"])
            if not any(is_unresolved(f) for f in fields):
                continue
            if "delivers" in candidates(l["relation"]) and event_id in candidates(l["source"]) and item_id in candidates(l["target"]):
                open_ambs.extend(_amb(f) for f in fields if is_unresolved(f))
        if open_ambs:
            result.update(whole_item=UNRESOLVED, unresolved=sorted(set(open_ambs)))
            basis["whole_item"] = "PD"
            return
        bound = self._size_bound(event_id, item_id)
        if bound:
            prov.append({"size_bound": bound})
        if self.negatives_enabled(event_id):  # W3
            result["whole_item"] = "NOT_DELIVERED"
            basis["whole_item"] = "0.3+PD"
            prov.append({"coverage": self._coverage(event_id), "annotation": self.profile_local.get("annotation")})
        elif bound:  # W4
            result["whole_item"] = "WHOLE_NOT_DELIVERED"
            basis["whole_item"] = "0.3"
        else:  # W5
            result["whole_item"] = UNKNOWN
            basis["whole_item"] = "0.3"

    def _check_spans(self, item_id: str, spans: list) -> None:
        text = self.texts.get(item_id)
        for span in spans:
            if (
                not isinstance(span, (list, tuple))
                or len(span) != 2
                or not all(isinstance(v, int) and not isinstance(v, bool) for v in span)
                or not 0 <= span[0] < span[1]
                or (text is not None and span[1] > len(text))
            ):
                raise InvalidInput(f"invalid span {span!r} for {item_id}")

    def _spans(self, event_id: str, item_id: str, spans: list, whole: str, basis: dict, prov: list) -> dict:
        if whole == "DELIVERED":
            basis["spans"] = "0.3"
            return {"via": "whole_item"}
        if whole == NOT_APPLICABLE:
            basis["spans"] = "PD"
            return {"via": NOT_APPLICABLE}
        basis["spans"] = "0.3+PD"
        delivered = [d for d in self._delivered_items(event_id) if d != item_id]
        x_text = self.texts.get(item_id)
        missing = sorted(d for d in delivered if d not in self.texts)
        coverage = self._coverage(event_id)
        residual = coverage.get("residual") if coverage else None
        can_negate = (
            whole == "NOT_DELIVERED" and x_text is not None and not missing and isinstance(residual, str)
        )
        exposed, others = [], []
        for span in spans:
            start, end = span
            piece = x_text[start:end] if x_text is not None else None
            via = next((d for d in delivered if piece and d in self.texts and piece in self.texts[d]), None)
            if via is not None:
                exposed.append({"span": [start, end], "via": via})
            elif can_negate and piece and piece not in residual:
                others.append({"span": [start, end], "value": "NOT_EXPOSED_EXACT"})
            else:
                others.append({"span": [start, end], "value": UNKNOWN})
        out = {"exposed": exposed, "others": others}
        if x_text is None:
            out["text_missing"] = [item_id]
        elif missing:
            out["text_missing"] = missing
        prov.append({"delivered_set": delivered, "match": "exact substring, no normalization"})
        return out

    # --- claim delivery (spec §4) ---------------------------------------------------

    def claim_delivery(self, event_id: str, claim_id: str) -> dict:
        claim = self.claims.get(claim_id)
        if claim is None:
            raise InvalidInput(f"unknown claim {claim_id!r}")
        result: dict = {"op": "claim_delivery", "inputs": {"event": event_id, "claim": claim_id}}
        ev_id = claim["evidence"]
        raw_ref = claim.get("raw_ref")
        span = raw_ref.get("span") if isinstance(raw_ref, dict) and not is_unresolved(raw_ref) else None
        if is_unresolved(ev_id) or is_unresolved(raw_ref) or is_unresolved(span):
            ambs = [_amb(v) for v in (ev_id, raw_ref, span) if is_unresolved(v)]
            result.update(value=UNRESOLVED, unresolved=ambs, basis={"value": "PD"}, provenance=[])
            return result
        external = self._item(ev_id).get("produced_by") is None
        o2 = self.exposure_at(event_id, ev_id, spans=None if external or span is None else [span])
        whole = o2["whole_item"]
        span_res = o2.get("spans", {})
        if whole == "DELIVERED":
            value, b = TRUE, "0.3"
        elif whole in (UNRESOLVED, NOT_APPLICABLE):
            value, b = whole, "PD"
        elif whole == "NOT_DELIVERED" and any(o["value"] == "NOT_EXPOSED_EXACT" for o in span_res.get("others", [])):
            value, b = FALSE, "0.3+PD"
        else:
            value, b = UNKNOWN, "0.3"
        result.update(value=value, o2=o2, basis={"value": b})
        via = sorted({e["via"] for e in span_res.get("exposed", [])}) if whole != "DELIVERED" else []
        if via:
            result["content_exposed_via"] = via
            result["basis"]["content_exposed_via"] = "PD"
        if external:
            result["note"] = "external evidence: span coordinates undefined in 0.1"
        result["provenance"] = o2["provenance"]
        return result

    # --- O3 (spec §5) -----------------------------------------------------------------

    def state_vs_evidence(self, state: dict | int) -> dict:
        if isinstance(state, int):
            state = self.states[state]
        holder, event_id = self.state_anchor(state)
        prop, value = state["proposition"], state["value"]
        result: dict = {
            "op": "O3",
            "inputs": {"proposition": prop, "value": value, "claims": state["claims"]},
            "holder": holder,
            "event": event_id,
        }
        if is_unresolved(prop) or is_unresolved(value):
            ambs = [_amb(v) for v in (prop, value) if is_unresolved(v)]
            for part in ("direct", "delivered_direct", "testimonial"):
                result[part] = {"value": UNRESOLVED, "unresolved": ambs}
            result["comparison"] = UNRESOLVED
            result["basis"] = {"direct": "PD", "delivered_direct": "PD", "testimonial": "PD", "comparison": "PD"}
            return result
        definite, open_links = self._direct_links(prop)
        result["direct"] = self._direct(definite, open_links)
        result["comparison"] = self._comparison(value, result["direct"]["value"])
        result["delivered_direct"] = self._delivered_direct(event_id, definite, open_links)
        result["testimonial"] = self._testimonial(prop, holder)
        result["basis"] = {
            "direct": "PD" if open_links else "0.3",
            "comparison": "PD",
            "delivered_direct": "PD",
            "testimonial": "0.3+PD",
        }
        return result

    def _direct_links(self, prop: str) -> tuple[list[dict], list[tuple[dict, list]]]:
        definite, open_links = [], []
        for l in self.links:
            rel, src = l["relation"], l["source"]
            if not is_unresolved(rel) and not is_unresolved(src):
                if rel in DIRECT and src == prop:
                    definite.append(l)
                continue
            rel_c, src_c = candidates(rel), candidates(src)
            if prop not in src_c or not any(r in DIRECT for r in rel_c):
                continue
            options = sorted({r for r in rel_c if r in DIRECT})
            can_drop = any(r not in DIRECT for r in rel_c) or any(s != prop for s in src_c)
            open_links.append((l, options + ([_EXCLUDED] if can_drop else [])))
        return definite, open_links

    @staticmethod
    def _outcomes(fixed: list[str], variable: list[list]) -> set[str]:
        return {
            _table(fixed + [c for c in combo if c is not _EXCLUDED]) for combo in itertools.product(*variable)
        } if variable else {_table(fixed)}

    def _direct(self, definite: list[dict], open_links: list) -> dict:
        fixed = [l["relation"] for l in definite]
        outcomes = self._outcomes(fixed, [opts for _, opts in open_links])
        out: dict = {
            "links": sorted(
                ({"relation": l["relation"], "target": l["target"] if not is_unresolved(l["target"]) else UNRESOLVED}
                 for l in definite),
                key=lambda d: (d["relation"], str(d["target"])),
            ),
            "provenance": [{"link": l["relation"], "raw_ref": l["raw_ref"]} for l in definite],
        }
        if open_links:
            out["open_links"] = sorted(
                {a for l, _ in open_links for a in (_amb(f) for f in (l["relation"], l["source"], l["target"]) if is_unresolved(f))}
            )
            if len(outcomes) == 1:
                out.update(value=outcomes.pop(), stable_over_unresolved=True)
            else:
                out["value"] = UNRESOLVED
        else:
            out["value"] = outcomes.pop()
        out["scope"] = "relative to stored links"
        return out

    @staticmethod
    def _comparison(value: str, direct: str) -> str:
        if value not in (TRUE, FALSE):
            return NOT_APPLICABLE
        if direct in (UNKNOWN, UNRESOLVED):
            return direct
        if direct == UNCERTAIN:
            return "AMBIGUOUS"
        return "AGREES" if direct == value else "DIFFERS_FROM_DIRECT"

    def _delivered_direct(self, event_id: str, definite: list[dict], open_links: list) -> dict:
        counted, excluded, pending = [], [], []
        for l in definite:
            target, rel, rr = l["target"], l["relation"], l.get("raw_ref", {})
            entry = {"relation": rel, "target": target if not is_unresolved(target) else UNRESOLVED}
            if is_unresolved(target) or is_unresolved(rr):
                pending.append(dict(entry, reason=UNRESOLVED, options=[rel, _EXCLUDED]))
                continue
            whole_item = rr.get("whole_item") is True
            spans = rr.get("spans")
            if is_unresolved(spans) or (isinstance(spans, list) and any(is_unresolved(x) for x in spans)):
                pending.append(dict(entry, reason=UNRESOLVED, options=[rel, _EXCLUDED]))
                continue
            if not whole_item and not spans:
                pending.append(dict(entry, reason="link raw_ref names no grounding", options=[rel, _EXCLUDED]))
                continue
            o2 = self.exposure_at(event_id, target, spans=None if whole_item else spans)
            whole = o2["whole_item"]
            entry["o2_whole_item"] = whole
            if whole == "DELIVERED":
                counted.append(dict(entry, grounding="delivered"))
            elif whole_item:
                if whole in ("NOT_DELIVERED", "WHOLE_NOT_DELIVERED"):
                    excluded.append(dict(entry, reason=f"absence-grounded, {whole}"))
                else:
                    pending.append(dict(entry, reason=whole, options=[rel, _EXCLUDED]))
            else:
                sp = o2["spans"]
                if sp.get("exposed") and not sp.get("others"):
                    counted.append(dict(entry, grounding="exposed-span", via=sorted({e["via"] for e in sp["exposed"]})))
                elif whole == "NOT_DELIVERED" and any(o["value"] == "NOT_EXPOSED_EXACT" for o in sp["others"]):
                    excluded.append(dict(entry, reason="NOT_DELIVERED and span NOT_EXPOSED_EXACT"))
                else:
                    pending.append(dict(entry, reason=whole, options=[rel, _EXCLUDED]))
        for l, opts in open_links:
            pending.append({"relation": UNRESOLVED, "target": UNRESOLVED, "reason": UNRESOLVED, "options": opts})

        counted_value = _table(e["relation"] for e in counted)
        outcomes = self._outcomes([e["relation"] for e in counted], [p["options"] for p in pending])
        out: dict = {"counted_value": counted_value, "counted": counted, "excluded": excluded}
        if len(outcomes) == 1:
            out["value"] = counted_value
        else:
            unresolved_pending = any(p["reason"] == UNRESOLVED for p in pending)
            out["value"] = UNRESOLVED if unresolved_pending else UNKNOWN
            out["exposure_incomplete"] = True
        out["pending"] = [{k: v for k, v in p.items() if k != "options"} for p in pending]
        out["scope"] = "relative to stored links; never uptake; no completeness-dependent negative (D5)"
        return out

    @staticmethod
    def _contribution(members: list[tuple[str, str]], values: dict[str, list[str]], flagged: dict) -> str | None:
        """Spec §5.3 contribution rules 1-5 for one completion of state values (None: nothing counted)."""
        if any(v in (UNCERTAIN, UNKNOWN) for cid, _ in members for v in values.get(cid, [])):
            return UNCERTAIN
        counted = []
        for cid, stance in members:
            if FALSE in values.get(cid, []):
                flagged[cid] = {"claim": cid, "reason": "claim grounds a FALSE state; stance unexercised (0.3 §12)"}
            else:
                counted.append(stance)
        stances = set(counted)
        if not stances:
            return None
        if stances == {"P", "notP"}:
            return UNCERTAIN
        return TRUE if stances == {"P"} else FALSE

    def _testimonial(self, prop: str, holder: str) -> dict:
        form = self.propositions.get(prop, {}).get("form")
        neg_ids = {p["id"] for p in self.propositions.values() if form is not None and p.get("form") == f"not({form})"}
        claims_on, unresolved = [], []
        for cid in sorted(self.claims):
            c = self.claims[cid]
            p = c["proposition"]
            if is_unresolved(p):
                if prop in candidates(p) or neg_ids & set(candidates(p)):
                    unresolved.append({"claim": cid, "unresolved": [_amb(p)]})
                continue
            if p == prop or p in neg_ids:
                claims_on.append((cid, "P" if p == prop else "notP"))

        by_speaker: dict[str, list] = {}
        holder_claims, own = [], []
        for cid, stance in claims_on:
            spk = self.speaker(self.claims[cid])
            if isinstance(spk, _Unres):
                unresolved.append({"claim": cid, "unresolved": spk.ambs})
            elif spk == holder:
                own.append(cid)
                holder_claims.append(cid)
            else:
                by_speaker.setdefault(spk, []).append((cid, stance))

        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for l in self._resolved_links("propagates_to"):
            parent[find(l["source"])] = find(l["target"])
        ev_of = lambda cid: self.claims[cid]["evidence"]  # noqa: E731
        spk_roots = {s: {find(ev_of(cid)) for cid, _ in cl} for s, cl in by_speaker.items()}
        holder_roots = {find(ev_of(cid)) for cid in holder_claims}

        groups: list[list[str]] = []
        for s in sorted(by_speaker):
            merged = [g for g in groups if any(spk_roots[s] & spk_roots[t] for t in g)]
            new = [s] + [t for g in merged for t in g]
            groups = [g for g in groups if g not in merged] + [sorted(new)]

        resolved_values: dict[str, list[str]] = {}
        open_states: list[tuple[list[str], list[str], str]] = []  # (claims, candidate values, amb)
        for st in self.states:
            if is_unresolved(st.get("claims")):
                continue
            if is_unresolved(st.get("value")):
                open_states.append((list(st["claims"]), candidates(st["value"]), _amb(st["value"])))
                continue
            for cid in st["claims"]:
                resolved_values.setdefault(cid, []).append(st["value"])

        contributions, flagged = [], {}
        for g in sorted(groups):
            members = [(cid, stance) for s in g for cid, stance in by_speaker[s]]
            ids = {cid for cid, _ in members}
            relevant = [o for o in open_states if ids & set(o[0])]
            outcomes, per_speaker_outcomes = set(), {s: set() for s in g}
            combos = list(itertools.product(*(o[1] for o in relevant))) if relevant else [()]
            flag_counts: dict[str, int] = {}
            for combo in combos:
                values = {cid: list(resolved_values.get(cid, [])) for cid in ids}
                for (cl, _, _), v in zip(relevant, combo):
                    for cid in ids & set(cl):
                        values[cid].append(v)
                seen: dict = {}
                outcomes.add(self._contribution(members, values, seen))
                for s in g:
                    per_speaker_outcomes[s].add(self._contribution(by_speaker[s], values, {}))
                for cid in seen:
                    flag_counts[cid] = flag_counts.get(cid, 0) + 1
            for cid, n in flag_counts.items():
                entry = {"claim": cid, "reason": "claim grounds a FALSE state; stance unexercised (0.3 §12)"}
                if n < len(combos):  # flagged only in some completions of unresolved state values
                    entry["unresolved"] = sorted({o[2] for o in relevant if cid in o[0]} or {o[2] for o in relevant})
                flagged[cid] = entry
            if outcomes == {None}:
                continue
            entry = {
                "speakers": g,
                "claims": sorted(ids),
                "contribution": outcomes.pop() if len(outcomes) == 1 else UNRESOLVED,
                "per_speaker": {s: _one(v) for s, v in per_speaker_outcomes.items()},
                "lineage_connected_to_holder": bool(set().union(*(spk_roots[s] for s in g)) & holder_roots),
            }
            if relevant:
                entry["unresolved"] = sorted({o[2] for o in relevant})
            contributions.append(entry)
        flagged = list(flagged.values())
        out: dict = {
            "value": "REPORTED" if contributions else UNKNOWN,
            "contributions": contributions,
            "excluded_holder_claims": sorted(own),
        }
        if len(contributions) >= 2:
            out["independence"] = UNKNOWN
        if unresolved:
            out["unresolved"] = unresolved
        if flagged:
            out["flagged_stance"] = sorted(flagged, key=lambda f: f["claim"])
        out["note"] = "no aggregate truth value; depends_on/implication excluded (0.3 §12 PROVISIONAL)"
        return out
