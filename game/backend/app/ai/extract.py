"""Structured extraction: user text -> candidate facts, intent and at most one decision proposal.

Two extractors produce the SAME candidate format and both go through ``validate.validate_facts``:
  rule_extract  deterministic regex/keyword extractor (always available; used when no LLM is configured)
  llm_extract   the LLM with ``prompts.EXTRACT_PROMPT``; its output is untrusted input like any other
Only the message text is sent to the LLM (no profile values, no identifiers).
"""
from __future__ import annotations

import re

from ..engine.decisions import CATALOGUE, DecisionError, validate_decision
from .llm import LLMUnavailable, chat_json
from .prompts import EXTRACT_PROMPT
from .validate import KNOWN_FIELDS, validate_facts

CITY_TO_CANTON = {"zurich": "ZH", "zürich": "ZH", "geneva": "GE", "genève": "GE", "lausanne": "VD",
                  "vaud": "VD", "bern": "BE", "basel": "BS", "lugano": "TI", "ticino": "TI", "fribourg": "FR",
                  "neuchatel": "NE", "neuchâtel": "NE", "lucerne": "LU", "luzern": "LU", "st. gallen": "SG",
                  "st gallen": "SG"}

FIELD_WORDS = [
    ("income", r"\b(earns?|earning|earned|salary|income|makes?|paid|wages?|pay)\b"),
    ("debt", r"\b(debt|loan|owe|mortgage|credit)\b"),
    ("rent", r"\b(rent)\b"),
    ("investments", r"\b(invest\w*|portfolio|etf|stocks|shares|pillar 3a|3a)\b"),
    ("other_expenses", r"\b(expenses|spend|spending|costs?)\b"),
    ("cash", r"\b(savings|saved|cash|in the bank|bank account|have)\b"),
]
OTHER_PERSON = r"\b(my (friend|partner|wife|husband|brother|sister|colleague|dad|father|mom|mother|boss|neighbou?r)|" \
               r"a friend|someone|colleague|they earn|he earns|she earns)\b"
AMOUNT = re.compile(r"(?P<pre>chf|fr\.?|sfr|eur|€|usd|\$)?\s*(?P<num>\d[\d'’,.]*)\s*(?P<mult>k|m|mio)?\b\s*"
                    r"(?P<post>chf|francs?|eur|euros?|€|usd|dollars?)?", re.I)

CURRENCY = {"chf": "CHF", "fr": "CHF", "fr.": "CHF", "sfr": "CHF", "franc": "CHF", "francs": "CHF",
            "eur": "EUR", "euro": "EUR", "euros": "EUR", "€": "EUR", "usd": "USD", "$": "USD",
            "dollar": "USD", "dollars": "USD"}

INTENTS = [
    ("manipulation", r"(ignore (the |all )?(calculation|numbers|simulation|model)s?|make (the |my )?.{0,40}"
                     r"(successful|worth it|pay off|win|richer)|just say|pretend|say that|tell me .{0,30} makes me richer|"
                     r"change the (result|numbers|assumptions) so)"),
    ("certainty_request", r"\b(definitely|for sure|guarantee\w*|exactly how much|certain(ly)?|100%)\b"),
    ("assumption_request", r"\b(average|typical|standard) (swiss )?(tax|salary|rent|inflation)|\btax rate\b"),
    ("missing_info", r"what (information|info|data) (is|are) missing|what do you (still )?need"),
    ("explain_request", r"\b(why|explain|differ\w*|compare|which assumptions? matter)\b"),
]

DECISION_WORDS = [
    ("masters", r"\b(master'?s|msc|ma degree|go back to (school|university)|study)\b"),
    ("move_abroad", r"\b(abroad|germany|move to (another country|europe))\b"),
    ("relocate", r"\b(move|moving|relocat\w*)\b"),
    ("part_time", r"\b(part[- ]?time|reduce my hours|work less|80 ?%|60 ?%)\b"),
    ("self_employed", r"\b(self[- ]employed|freelanc\w*|start a business|own company)\b"),
    ("family", r"\b(have a (child|baby)|start a family)\b"),
    ("parental_leave", r"\b(parental leave|maternity|paternity)\b"),
    ("job_search", r"\b(quit|leave my job|take a break from work)\b"),
    ("start_working", r"\b(start working|get a job|first job)\b"),
    ("continue_employment", r"\b(stay (in|at) my job|keep my job|continue working)\b"),
]


def _clauses(text: str) -> list[str]:
    return [c for c in re.split(r"(?<!\d)[.;!?](?!\d)|,(?!\d{3})|\band\b|\bbut\b", text) if c and c.strip()]


def _period(c: str):
    if re.search(r"(per|a|/|each)\s*(month|mo)\b|\bmonthly\b", c, re.I):
        return "monthly"
    if re.search(r"(per|a|/|each)\s*(year|yr|annum)\b|\b(yearly|annual(ly)?|p\.?a\.?)\b", c, re.I):
        return "yearly"
    return None


def _basis(c: str):
    if re.search(r"\b(gross|before tax(es)?|brut)\b", c, re.I):
        return "gross"
    if re.search(r"\b(net|after tax(es)?|take[- ]home)\b", c, re.I):
        return "net"
    return None


def rule_extract(text: str) -> dict:
    facts = []
    for clause in _clauses(text):
        low = clause.lower()
        other = bool(re.search(OTHER_PERSON, low))
        field = next((f for f, pat in FIELD_WORDS if re.search(pat, low)), None)
        if re.search(r"interest|rate of|% (interest|rate)", low):
            m = re.search(r"(\d+(?:\.\d+)?)\s*%", low)
            if m:
                facts.append({"field": "debt_rate_percent", "value": float(m.group(1)), "subject": "other" if other else "user",
                              "span": m.group(0).strip()})
                continue
        age = re.search(r"\b(i'?m|i am|aged?)\s*(\d{2})\b(?!\s*(k|%|chf|francs?|eur))|\b(\d{2}) years old", low)
        if age:
            facts.append({"field": "age", "value": int(age.group(2) or age.group(5)),
                          "subject": "other" if other else "user", "span": age.group(0)})
            continue
        if field is None:
            continue
        for m in AMOUNT.finditer(clause):
            num = re.sub(r"[’',]", "", m.group("num")).rstrip(".")
            try:
                v = float(num)
            except ValueError:
                continue
            mult = (m.group("mult") or "").lower()
            v *= {"k": 1e3, "m": 1e6, "mio": 1e6}.get(mult, 1)
            if v < 10 and not mult:
                continue
            cur_raw = (m.group("pre") or m.group("post") or "").lower()
            currency = CURRENCY.get(cur_raw)
            mentioned = {CURRENCY[c] for c in CURRENCY if re.search(rf"(?<![a-z]){re.escape(c)}(?![a-z])", low)}
            if not mentioned:       # currency stated elsewhere in the message ("... it's in EUR or CHF")
                mentioned = {CURRENCY[c] for c in CURRENCY
                             if re.search(rf"(?<![a-z]){re.escape(c)}(?![a-z])", text.lower())}
            if currency is None and len(mentioned) == 1:
                currency = next(iter(mentioned))
            if len(mentioned) > 1:
                currency = "/".join(sorted(mentioned))       # ambiguous: validation asks
            facts.append({"field": field, "value": v, "currency": currency, "period": _period(clause),
                          "basis": _basis(clause), "subject": "other" if other else "user", "span": m.group(0).strip()})
            break
    low = text.lower()
    for city, canton in CITY_TO_CANTON.items():
        if re.search(rf"\b(live|living|based) in {re.escape(city)}\b", low):
            facts.append({"field": "canton", "value": canton, "subject": "user", "span": city})
            break
    for word, status in (("student", "student"), ("unemployed", "unemployed"), ("retired", "retired"),
                         ("self-employed", "self_employed"), ("part-time", "part_time")):
        if re.search(rf"\bi('?m| am) (a |an )?(currently )?{word}\b", low):
            facts.append({"field": "employment", "value": status, "subject": "user", "span": word})
            break
    return {"facts": facts, "decision": rule_decision(text)}


def rule_decision(text: str) -> dict | None:
    low = text.lower()
    for did, pat in DECISION_WORDS:
        if re.search(pat, low):
            params = {}
            if did == "relocate":
                to = next((c for city, c in CITY_TO_CANTON.items() if re.search(rf"\bto {re.escape(city)}\b", low)), None)
                to = to or next((c for c in ("ZH", "GE", "VD", "BE", "BS", "TI", "FR", "NE", "LU", "SG")
                                 if re.search(rf"\bto {c.lower()}\b", low)), None)
                if to:
                    params["to_canton"] = to
            m = re.search(r"in (\d+) (month|year)s?", low)
            start = None if not m else int(m.group(1)) * (12 if m.group(2) == "year" else 1)
            return {"decision_id": did, "start_month": start, "params": params}
    return None


def intents(text: str) -> list[str]:
    low = text.lower()
    return [name for name, pat in INTENTS if re.search(pat, low)]


def llm_extract(text: str) -> dict:
    raw = chat_json([{"role": "system", "content": EXTRACT_PROMPT}, {"role": "user", "content": text}])
    facts = raw.get("facts") if isinstance(raw.get("facts"), list) else []
    dec = raw.get("decision") if isinstance(raw.get("decision"), dict) else None
    return {"facts": [f for f in facts if isinstance(f, dict)], "decision": dec}


def propose_decision(dec: dict | None, *, initial_state: str, horizon_months: int, assumptions,
                     current_canton: str | None, min_start_month: int = 0) -> dict | None:
    """Validate a proposed decision with the backend rules. Never applies it."""
    if not dec:
        return None
    did = dec.get("decision_id")
    if did not in CATALOGUE:
        return {"decision_id": did, "status": "unsupported",
                "message": f"'{did}' is not a supported decision. Supported: {sorted(CATALOGUE)}"}
    start = dec.get("start_month")
    params = {k: v for k, v in (dec.get("params") or {}).items() if v is not None}
    out = {"decision_id": did, "name": CATALOGUE[did].name, "start_month": start, "params": params}
    if start is None:
        out.update(status="needs_clarification", message="When should this start? (month from today)")
        return out
    try:
        d = validate_decision(did, start, params, initial_state=initial_state, horizon_months=horizon_months,
                              assumptions=assumptions, current_canton=current_canton, min_start_month=min_start_month)
    except DecisionError as e:
        out.update(status="invalid", errors=e.errors)
        return out
    out.update(status="ready_for_confirmation", validated=d.to_dict(),
               message="Proposed only. Nothing is applied until you confirm.")
    return out


def extract(text: str, *, use_llm: bool = True) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("message must be non-empty text")
    if len(text) > 2000:
        raise ValueError("message too long (max 2000 characters)")
    source = "rules"
    cand = rule_extract(text)
    if use_llm:
        try:
            llm = llm_extract(text)
            cand = {"facts": llm["facts"], "decision": llm["decision"] or cand["decision"]}
            source = "llm" if llm["decision"] or not cand["decision"] else "llm (decision: rules)"
        except LLMUnavailable:
            source = "rules (LLM unavailable)"
    facts = [f for f in cand["facts"] if f.get("field") in KNOWN_FIELDS or f.get("field")]
    checked = validate_facts(facts, text)
    return {"extractor": source, "intents": intents(text), "decision_candidate": cand.get("decision"), **checked}
