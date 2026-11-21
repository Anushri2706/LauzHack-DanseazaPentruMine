"""Deterministic validation of extracted facts. This, not the prompt, decides what is accepted.

Every fact ends with one status:
  valid                 well-formed, unit/period/basis known, plausible -> may be proposed to the user
  needs_clarification   something required is missing (period, gross/net, currency) -> ask
  needs_confirmation    well-formed but unusual -> ask the user to confirm; never corrected silently
  conflict              contradicts another fact in the same message
  rejected              unknown field, wrong type, out of range, about someone else, or not in the text

Accepted facts are PROPOSALS: the profile only changes when the user submits it (``source`` becomes
``confirmed_user``). Model confidence is never used as evidence.
"""
from __future__ import annotations

import logging
import re

from ..engine.decisions import CANTON_CHOICES
from ..engine.profile import EMPLOYMENT, HOUSEHOLD

log = logging.getLogger("game.ai.validate")

FLOW_FIELDS = {"income": "income_amount_chf", "rent": "rent_chf_month", "other_expenses": "other_expenses_chf_month"}
STOCK_FIELDS = {"cash": "cash_chf", "investments": "investments_chf", "debt": "debt_chf"}
ENUM_FIELDS = {"employment": EMPLOYMENT, "household": HOUSEHOLD, "canton": CANTON_CHOICES}
KNOWN_FIELDS = set(FLOW_FIELDS) | set(STOCK_FIELDS) | set(ENUM_FIELDS) | {"age", "debt_rate_percent"}

# Plausibility limits (CHF/month for flows, CHF for stocks). Above "confirm" -> needs_confirmation;
# above "max" -> rejected. These are input sanity bounds, not financial assumptions.
LIMITS = {"income": (40_000, 1_000_000), "rent": (10_000, 100_000), "other_expenses": (20_000, 200_000),
          "cash": (5_000_000, 1e9), "investments": (20_000_000, 1e10), "debt": (5_000_000, 1e9)}


def _num(v):
    if isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) != float("inf") else None


def _in_text(value: float, text: str) -> bool:
    """Is this number actually present in the user's message (allowing 5k / 5'000 / 5,000 forms)?"""
    for m in re.finditer(r"(\d[\d'’,.\s]*\d|\d)\s*([kKmM])?(?![a-zA-Z])", text):
        raw = re.sub(r"[’'\s,]", "", m.group(1))
        try:
            n = float(raw)
        except ValueError:
            continue
        mult = {"k": 1e3, "m": 1e6}.get((m.group(2) or "").lower(), 1)
        if abs(n * mult - value) < 1e-6 or abs(n - value) < 1e-6:
            return True
    return False


def validate_facts(facts: list[dict], text: str) -> dict:
    out, questions = [], []
    seen: dict[str, dict] = {}
    for raw in facts:
        f = {k: raw.get(k) for k in ("field", "value", "currency", "period", "basis", "subject", "span")}
        f["source"] = "user"
        field = f["field"]
        problems: list[str] = []
        status = "valid"
        if field not in KNOWN_FIELDS:
            status, problems = "rejected", [f"unknown field {field!r}"]
        elif f.get("subject") == "other":
            status, problems = "rejected", ["this describes someone else; your profile is not changed from it"]
        elif field in ENUM_FIELDS:
            if f["value"] not in ENUM_FIELDS[field]:
                status, problems = "rejected", [f"{field} must be one of {list(ENUM_FIELDS[field])}"]
        else:
            v = _num(f["value"])
            if v is None:
                status, problems = "rejected", [f"{field} must be a number"]
            elif not _in_text(v, text):
                status, problems = "rejected", [f"{v:g} does not appear in the message (not invented)"]
            elif field == "age":
                if v != int(v) or not 18 <= v <= 100:
                    status, problems = "rejected", ["age must be a whole number between 18 and 100"]
                f["value"], f["unit"] = int(v), "years"
            elif field == "debt_rate_percent":
                if not 0 <= v <= 50:
                    status, problems = "rejected", ["interest rate must be between 0% and 50%"]
                f["unit"] = "percent per year"
            else:
                if v < 0:
                    status, problems = "rejected", ["amount cannot be negative"]
                cur = (f.get("currency") or "").upper() or None
                f["unit"] = "CHF"
                label = field.replace("_", " ")
                missing = []
                if cur is not None and "/" in cur:
                    status = "needs_clarification"
                    questions.append(f"Is your {label} of {v:,.0f} in {' or '.join(cur.split('/'))}? The game works in "
                                     "CHF only and has no exchange-rate assumption.")
                elif cur not in (None, "CHF"):
                    status = "needs_clarification"
                    questions.append(f"Your {label} was given in {cur}. The game works in CHF only and has no "
                                     "exchange-rate assumption: what is the amount in CHF?")
                elif cur is None:
                    missing.append("in CHF")
                if field in FLOW_FIELDS and f.get("period") not in ("monthly", "yearly"):
                    missing.append("per month or per year")
                if field == "income" and f.get("basis") not in ("gross", "net"):
                    missing.append("gross (before tax) or net (what reaches your account)")
                if missing:
                    status = "needs_clarification"
                    questions.append(f"For your {label} of {v:,.0f}: is it " + ", ".join(missing[:-1])
                                     + (" and " if len(missing) > 1 else "") + missing[-1] + "?")
                if status != "rejected" and field in LIMITS:
                    monthly = v / 12 if (field in FLOW_FIELDS and f.get("period") == "yearly") else v
                    confirm, hard = LIMITS[field]
                    if monthly > hard:
                        status, problems = "rejected", [f"{field} of {v:,.0f} is outside the supported range"]
                    elif monthly > confirm and status == "valid":
                        status = "needs_confirmation"
                        questions.append(f"{field.replace('_', ' ').capitalize()} of CHF {v:,.0f}"
                                         f"{' per ' + ('month' if f.get('period') == 'monthly' else 'year') if field in FLOW_FIELDS else ''}"
                                         " is unusually high. Please confirm it is correct.")
        if status not in ("rejected",) and field in seen and seen[field]["value"] != f["value"]:
            status = "conflict"
            seen[field]["status"] = "conflict"
            questions.append(f"You gave two different values for {field.replace('_', ' ')}. Which one is right?")
        f["status"] = status
        f["problems"] = problems
        if status == "rejected":
            log.info("fact rejected field=%s reason=%s", field, problems[0] if problems else "")
        if field in KNOWN_FIELDS:
            seen.setdefault(field, f)
        out.append(f)
    proposed = {}
    for f in out:
        if f["status"] != "valid":
            continue
        field = f["field"]
        if field in FLOW_FIELDS:
            proposed[FLOW_FIELDS[field]] = f["value"]
            if field == "income":
                proposed["income_period"] = f["period"]
                proposed["income_basis"] = f["basis"]
            elif f["period"] == "yearly":       # rent / other expenses are monthly in the profile
                proposed[FLOW_FIELDS[field]] = round(f["value"] / 12, 2)
        elif field in STOCK_FIELDS:
            proposed[STOCK_FIELDS[field]] = f["value"]
        elif field == "debt_rate_percent":
            proposed["debt_annual_rate"] = f["value"] / 100
        else:
            proposed[field] = f["value"]
    # dedupe questions, keep order
    qs = list(dict.fromkeys(questions))
    return {"facts": out, "questions": qs, "proposed_profile_updates": proposed,
            "requires_user_confirmation": True}
