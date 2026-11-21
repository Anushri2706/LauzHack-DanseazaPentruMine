"""Claim guard: deterministic validation and template rendering of explanation claims.

Pipeline (no second LLM involved):
 1. ``build_evidence`` turns completed, versioned results (and an exact comparison, sensitivity rows and
    the registry entries they used) into citable facts with stable ids.
 2. The LLM (or any client) proposes structured claims that cite fact ids.
 3. ``check_claims`` rejects a claim when: the fact id does not exist (fabricated metric / result id), the
    claim type does not fit the fact, a comparison's direction is wrong, a stated unit / basis / horizon
    does not match, or free text contains numbers, certainty or causal language.
 4. ``render`` writes every accepted claim from a template using backend values only. Free LLM text is
    never shown for numerical facts.
 5. If nothing valid remains, ``fallback_claims`` picks claims deterministically from the evidence.
"""
from __future__ import annotations

import logging
import re

from ..engine.montecarlo import scalar_facts

log = logging.getLogger("game.ai.claim_guard")

CLAIM_TYPES = {"comparison": "comparison", "level": "level", "probability": "probability",
               "sensitivity": "sensitivity", "assumption": "assumption"}
CERTAINTY = re.compile(r"\b(guarantee\w*|definitely|certain(ly)?|for sure|will (be|have|make|earn|reach)|"
                       r"no doubt|always|never fail|risk[- ]free|proven)\b", re.I)
CAUSAL = re.compile(r"\b(because of|caus(e|es|ed|ing)|leads? to|results? in|makes? you (richer|poorer)|"
                    r"due to)\b", re.I)
METRIC_LABELS = {"cash": "cash balance", "net_worth_nominal": "nominal net worth",
                 "net_worth_real": "real net worth (today's CHF)", "months_below_reserve": "months below the emergency reserve"}
STAT_LABELS = {"p10": "10th-percentile", "p50": "median", "p90": "90th-percentile", "mean": "mean"}
DISCLAIMER = ("These are simulated outcomes conditional on the stated assumptions and seed, not predictions, "
              "guarantees or financial advice.")


def chf(v: float) -> str:
    return f"CHF {v:,.0f}"


def pct(v: float) -> str:
    return f"{100 * v:.1f}%"


def build_evidence(results: list[dict], comparison: dict | None = None, sensitivity: dict | None = None) -> dict:
    """results: [{"result_id", "branch_id", "branch_name", "summary"}] (completed, versioned)."""
    facts: dict[str, dict] = {}
    names: dict[str, str] = {}
    assumptions: dict[str, dict] = {}
    for r in results:
        rid, s = r["result_id"], r["summary"]
        names[rid] = r.get("branch_name") or r["branch_id"]
        for fid, f in scalar_facts(s).items():
            kind = "probability" if f["unit"] == "probability" else "level"
            facts[f"{rid}:{fid}"] = {**f, "kind": kind, "result_id": rid, "branch": names[rid]}
        for a in s.get("assumptions_used", []):
            assumptions[a["id"]] = a
    if comparison:
        ra, rb = comparison["a"]["result_id"], comparison["b"]["result_id"]
        for key, row in comparison["differences"].items():
            facts[f"cmp:{ra}:{rb}:{key}"] = {**row, "kind": "comparison", "metric_key": key,
                                             "a_name": names.get(ra, ra), "b_name": names.get(rb, rb),
                                             "a_result": ra, "b_result": rb}
    if sensitivity:
        rid = sensitivity.get("result_id", "sens")
        for row in sensitivity.get("rows", [])[:5]:
            facts[f"sens:{rid}:{row['assumption_id']}"] = {**row, "kind": "sensitivity", "metric": sensitivity["metric"],
                                                          "unit": sensitivity.get("metric_unit", "CHF")}
            assumptions.setdefault(row["assumption_id"], {"id": row["assumption_id"], "source_status": row["source_status"],
                                                          "value_used": row["baseline_value"], "unit": row["unit"]})
    sens_ids = {row["assumption_id"] for row in (sensitivity or {}).get("rows", [])[:5]}
    for aid, a in assumptions.items():
        status = a.get("source_status")
        relevant = (aid in sens_ids or status == "user_override"
                    or (status in ("illustrative", "placeholder") and a.get("table") in ("game", "market")))
        if relevant and status in ("estimated", "illustrative", "placeholder", "user_override"):
            facts[f"assumption:{aid}"] = {"kind": "assumption", "assumption_id": aid, "value": a.get("value_used"),
                                          "unit": a.get("unit"), "source_status": a.get("source_status"),
                                          "geography": a.get("geography"), "table": a.get("table")}
    return {"facts": facts, "branch_names": names}


def check_claims(claims, evidence: dict) -> tuple[list[dict], list[dict]]:
    facts = evidence["facts"]
    accepted, rejected = [], []
    for i, c in enumerate(claims if isinstance(claims, list) else []):
        reason = None
        if not isinstance(c, dict):
            reason = "claim is not an object"
        else:
            ctype, fid = c.get("type"), c.get("fact")
            f = facts.get(fid) if isinstance(fid, str) else None
            text = c.get("text") or ""
            if ctype not in CLAIM_TYPES:
                reason = f"unknown claim type {ctype!r}"
            elif f is None:
                reason = f"fact id {fid!r} does not exist in the supplied results (fabricated or wrong branch)"
            elif CLAIM_TYPES[ctype] != f["kind"]:
                reason = f"claim type {ctype} does not fit fact kind {f['kind']}"
            elif ctype == "comparison" and c.get("direction") not in (None, f["direction"]):
                reason = f"direction {c.get('direction')!r} contradicts computed direction {f['direction']!r}"
            elif c.get("unit") not in (None, f.get("unit")):
                reason = f"unit {c.get('unit')!r} does not match {f.get('unit')!r}"
            elif c.get("basis") not in (None, f.get("basis")):
                reason = f"basis {c.get('basis')!r} does not match {f.get('basis')!r}"
            elif c.get("horizon_months") not in (None, f.get("horizon_months")):
                reason = f"horizon {c.get('horizon_months')!r} does not match {f.get('horizon_months')!r}"
            elif c.get("value") is not None and not _same_value(c.get("value"), f):
                reason = f"value {c.get('value')!r} does not match the computed value"
            elif re.search(r"\d", text):
                reason = "numbers in free text are not allowed (they are rendered from backend data)"
            elif CERTAINTY.search(text):
                reason = "certainty / guarantee language"
            elif CAUSAL.search(text):
                reason = "unsupported causal language"
        if reason:
            rejected.append({"index": i, "claim": c, "reason": reason})
            log.info("claim rejected index=%s fact=%s reason=%s", i, c.get("fact") if isinstance(c, dict) else None, reason)
        else:
            accepted.append(c)
    return accepted, rejected


def _same_value(v, f) -> bool:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return False
    target = f.get("difference_b_minus_a", f.get("value"))
    if target is None:
        return False
    tol = 1e-4 if f.get("unit") == "probability" else 1.0
    return abs(x - float(target)) <= tol


def check_text(text: str | None) -> str | None:
    """Free framing text is shown only if it has no numbers, certainty or causal language."""
    if not text or not isinstance(text, str):
        return None
    if re.search(r"\d", text) or CERTAINTY.search(text) or CAUSAL.search(text) or len(text) > 400:
        return None
    return text.strip()


def _metric_label(fid_key: str) -> tuple[str, str]:
    metric, _, rest = fid_key.partition(".final.")
    if rest:
        return METRIC_LABELS.get(metric, metric), STAT_LABELS.get(rest, rest)
    metric, _, stat = fid_key.partition(".")
    return METRIC_LABELS.get(metric, metric), STAT_LABELS.get(stat, stat)


def _fmt(v, unit):
    if unit == "CHF":
        return chf(v)
    if unit == "probability":
        return pct(v)
    if unit == "months":
        return f"{v:g} months"
    return f"{v:g} {unit or ''}".strip()


def render(claim: dict, evidence: dict) -> str:
    f = evidence["facts"][claim["fact"]]
    k = f["kind"]
    if k == "comparison":
        key = f["metric_key"]
        h = f["horizon_months"]
        if key in ("p_cash_ever_below_zero", "p_ever_below_reserve"):
            what = "probability of cash falling below zero" if key == "p_cash_ever_below_zero" else \
                "probability of falling below the emergency reserve"
            rel = {"b_higher": "a higher", "b_lower": "a lower", "equal": "the same"}[f["direction"]]
            return (f"Under the assumptions used, {f['b_name']} has {rel} simulated {what} within {h} months than "
                    f"{f['a_name']} ({pct(f['b'])} vs {pct(f['a'])}).")
        label, stat = _metric_label(key)
        rel = {"b_higher": "a higher", "b_lower": "a lower", "equal": "about the same"}[f["direction"]]
        return (f"Under the assumptions used, {f['b_name']} has {rel} {stat} {label} at month {h} than {f['a_name']} "
                f"({_fmt(f['b'], f['unit'])} vs {_fmt(f['a'], f['unit'])}; difference {_fmt(f['difference_b_minus_a'], f['unit'])}).")
    if k == "probability":
        if f["metric"] == "p_ever_below_reserve":
            reserve = evidence["facts"].get(f"{f['result_id']}:reserve_chf", {}).get("value")
            res = f" of {chf(reserve)}" if reserve is not None else ""
            return (f"The simulated probability of falling below the specified reserve{res} at some point within "
                    f"{f['horizon_months']} months is {pct(f['value'])} in {f['branch']}.")
        return (f"In {f['branch']}, {pct(f['value'])} of simulated paths have negative cash at some point within "
                f"{f['horizon_months']} months.")
    if k == "level":
        key = claim["fact"].split(":", 1)[1]
        label, stat = _metric_label(key)
        return (f"In {f['branch']}, the simulated {stat} {label} at month {f['horizon_months']} is "
                f"{_fmt(f['value'], f['unit'])}.")
    if k == "sensitivity":
        label, stat = _metric_label(f["metric"])
        return (f"This result is sensitive to assumption '{f['assumption_id']}' (labelled {f['source_status']}): "
                f"changing it by ±10% moves the {stat} {label} by {_fmt(f['swing'], f['unit'])} in total.")
    if k == "assumption":
        geo = f" for {f['geography']}" if f.get("geography") else ""
        return (f"Assumption '{f['assumption_id']}' = {f['value']:g} {f.get('unit') or ''}{geo} is labelled "
                f"{f['source_status']}.").replace("  ", " ")
    raise ValueError(k)  # pragma: no cover


def fallback_claims(evidence: dict) -> list[dict]:
    facts = evidence["facts"]
    out = []
    for key in ("net_worth_real.final.p50", "p_ever_below_reserve", "net_worth_real.final.p10"):
        out += [{"type": "comparison", "fact": fid} for fid, f in facts.items()
                if f["kind"] == "comparison" and f["metric_key"] == key]
    if not out:
        out += [{"type": "level", "fact": fid} for fid, f in facts.items()
                if f["kind"] == "level" and fid.endswith("net_worth_real.final.p50")]
        out += [{"type": "probability", "fact": fid} for fid, f in facts.items()
                if f["kind"] == "probability" and f["metric"] == "p_ever_below_reserve"]
    sens = sorted((fid for fid, f in facts.items() if f["kind"] == "sensitivity"),
                  key=lambda fid: -facts[fid]["swing"])[:3]
    out += [{"type": "sensitivity", "fact": fid} for fid in sens]
    rank = lambda f: (f["source_status"] != "user_override", f.get("table") != "game",
                      f["source_status"] != "placeholder")
    flagged = sorted((fid for fid, f in facts.items() if f["kind"] == "assumption"
                      and f["source_status"] in ("illustrative", "placeholder", "user_override")),
                     key=lambda fid: rank(facts[fid]))[:3]
    out += [{"type": "assumption", "fact": fid} for fid in flagged]
    return out
