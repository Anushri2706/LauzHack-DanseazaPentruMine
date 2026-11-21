"""Language agents around the engine. The LLM reads and writes language; code checks everything.

understand(): message -> structured inputs + scenario (LLM), then validate() (code)
explain():    engine results -> plain language (LLM), then check_numbers() (code)
"""
import json
import re

from .llm import LLMUnavailable, chat, chat_json
from .router import CITY_TO_CANTON, route

CANTONS = ["ZH", "GE", "VD", "BE", "BS", "TI", "FR", "NE", "LU", "SG"]
EMPLOYMENT = ["employed", "part_time", "self_employed", "student", "unemployed", "retired"]
HOUSEHOLD = ["single", "couple", "family", "shared_housing"]
SCENARIOS = ["masters", "job_loss", "relocate", "part_time", "self_employed", "family", "explore", "none"]

def _one_of(xs):
    return "one of " + " | ".join(f'"{x}"' for x in xs)


UNDERSTAND_PROMPT = f"""You extract facts for a personal-finance simulator. Return ONLY a JSON object:
{{
 "inputs": {{
   "cash": number|null,            "income": number|null, "income_period": "monthly"|"yearly"|null,
   "rent": number|null,            "rent_period": "monthly"|"yearly"|null,
   "age": integer|null,            "employment": {_one_of(EMPLOYMENT)} or null,
   "household": {_one_of(HOUSEHOLD)} or null,  "canton": {_one_of(CANTONS)} or null,
   "portfolio": number|null,       "has_debt": true|false|null,
   "debt_amount": number|null,     "debt_rate_percent": number|null,
   "goal_name": string|null,       "goal_target": number|null, "goal_years": number|null
 }},
 "scenario": {_one_of(SCENARIOS)},
 "params": {{"years": integer|null, "months": integer|null, "to_canton": {_one_of(CANTONS)} or null}},
 "estimate_requested": list of fields the user asks you to estimate for them, from "rent" (else [])
}}
Rules:
- Only fill a field if the user actually said it in THIS message. Otherwise null. Never guess.
- employment: only if the user states their CURRENT situation. Wanting to study (a master's) does NOT
  mean they are a student now; leave employment null unless they say so.
- If the twin just asked for a field and the user replies with a bare amount (e.g. "1000chf"),
  that amount is the value of the field that was asked for.
- If the user asks you to use an average or typical value for a field, put it in estimate_requested
  and leave the field null.
- Amounts in CHF as plain numbers: "5k" -> 5000, "5'200" -> 5200, "60k a year" -> 60000 with period "yearly".
- canton is where the user lives now; params.to_canton is where they want to move.
  Map cities to cantons: Zurich ZH, Geneva GE, Lausanne VD, Bern BE, Basel BS, Lugano TI,
  Fribourg FR, Neuchatel NE, Lucerne LU, St. Gallen SG.
- scenario: "masters" (study, degree, university), "job_loss" (losing job, laid off, unemployed for a while),
  "relocate" (moving), "part_time" (working part-time, fewer hours), "self_employed" (freelancing,
  starting a business), "family" (having a child, starting a family), "explore" (wants to compare all
  options, see trade-offs, or find good combinations of choices), otherwise "none".
- params.years = study length; params.months = length of job loss (1 year = 12).
- If the user would move to another city for their studies, set scenario "masters" and params.to_canton."""

EXPLAIN_PROMPT = """You explain the output of a financial simulator to the person it describes.
Rules:
- Use ONLY numbers that appear in the JSON you are given. Do not compute new numbers.
- Explain the trade-off between the baseline and the what-if: end wealth range (P10 to P90, median),
  chance of needing financing, months below the cash buffer, and goal probabilities.
- Say which inputs were estimated or defaulted rather than provided.
- If a peer_path is given, say in one sentence where people at the same percentile are at older ages,
  and that these are different people, not a forecast.
- State each scenario's "notes" (its key assumptions, e.g. an assumed starting salary) in plain words.
- Mention every warning given, in plain words.
- Never recommend a choice and never say what the person should do. Describe trade-offs only.
- End with one sentence: these are illustrative simulations on synthetic data and assumptions,
  not financial advice or a forecast.
- Plain language, about 150-200 words, short paragraphs or bullets. Amounts as CHF with thousands separators."""


def _num(x, lo, hi):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if lo <= v <= hi else None


def validate(raw):
    """Turn the LLM's JSON into safe inputs. Returns (inputs, scenario, params, problems)."""
    problems = []
    src = raw.get("inputs") or {}
    out = {}

    def monthly(field, period, hi):
        v = _num(src.get(field), 0, hi * 12)
        if v is None:
            if src.get(field) is not None:
                problems.append(f"{field} '{src.get(field)}' looks wrong, ignored")
            return
        if src.get(period) == "yearly":
            v = v / 12
        if v > hi:
            problems.append(f"{field} {v:,.0f}/month looks too high, ignored")
            return
        out[field] = round(v, 2)

    monthly("income", "income_period", 50000)
    monthly("rent", "rent_period", 20000)
    for field, hi in (("cash", 1e8), ("portfolio", 1e8)):
        v = _num(src.get(field), 0, hi)
        if v is not None:
            out[field] = v
    age = _num(src.get("age"), 16, 100)
    if age is not None:
        out["age"] = int(age)
    for field, allowed in (("employment", EMPLOYMENT), ("household", HOUSEHOLD), ("canton", CANTONS)):
        v = src.get(field)
        if isinstance(v, list):
            v = v[0] if v else None
        if isinstance(v, str):
            v = CITY_TO_CANTON.get(v.lower(), v) if field == "canton" else v
            if v in allowed:
                out[field] = v
            else:
                problems.append(f"{field} '{v}' not recognised, ignored")
    if src.get("has_debt") is False:
        out["has_debt"] = False
    amount = _num(src.get("debt_amount"), 1, 1e7)
    if amount is not None:
        rate = _num(src.get("debt_rate_percent"), 0, 30)
        out["loans"] = [[amount, (rate if rate is not None else 5.0) / 100]]
        if rate is None:
            problems.append("debt interest rate not given, assumed 5%")
    target, years = _num(src.get("goal_target"), 1, 1e8), _num(src.get("goal_years"), 0.1, 10)
    if target is not None and years is not None:
        out["goals"] = [{"name": src.get("goal_name") or "Goal", "target": target,
                         "month": int(round(years * 12)), "priority": "high"}]

    scenario = raw.get("scenario")
    if isinstance(scenario, list):  # some models wrap single values in a list
        scenario = scenario[0] if scenario else None
    scenario = scenario if scenario in SCENARIOS else "none"
    p = raw.get("params") or {}
    params = {}
    if scenario == "masters" and _num(p.get("years"), 1, 6):
        params["years"] = int(float(p["years"]))
    if scenario == "job_loss":
        params["months"] = int(_num(p.get("months"), 1, 60) or 6)
    dest = p.get("to_canton")
    dest = dest[0] if isinstance(dest, list) and dest else dest
    dest = CITY_TO_CANTON.get(dest.lower(), dest) if isinstance(dest, str) else None
    if scenario in ("relocate", "masters") and dest in CANTONS:
        params["to_canton"] = dest  # for a master's: studying in another canton
    elif scenario == "relocate":
        problems.append("move destination not recognised; supported: " + ", ".join(CANTONS))
        scenario = "none"
    est = raw.get("estimate_requested") or []
    estimates = [f for f in (est if isinstance(est, list) else [est]) if f == "rent"]
    return out, scenario, params, problems, estimates


def understand(message, known=None, awaiting=None):
    """LLM extraction + code validation. Falls back to the keyword router if the LLM is unavailable.

    awaiting: fields the twin asked for in its last reply, so a bare "1000chf" can be interpreted.
    """
    context = f"Already known (do not repeat): {json.dumps(known or {})}\n"
    if awaiting:
        context += f"The twin just asked the user for: {', '.join(awaiting)}\n"
    msgs = [{"role": "system", "content": UNDERSTAND_PROMPT},
            {"role": "user", "content": context + f"User message: {message}"}]
    try:
        raw = chat_json(msgs)
        inputs, scenario, params, problems, estimates = validate(raw)
        return {"inputs": inputs, "scenario": scenario, "params": params, "estimates": estimates,
                "problems": problems, "source": "llm"}
    except LLMUnavailable as e:
        key, params, _ = route(message)
        return {"inputs": {}, "scenario": key or "none", "params": params, "estimates": [],
                "problems": [f"LLM unavailable ({e}); used keyword router, numbers not extracted"],
                "source": "rules"}


# ---- explanation -----------------------------------------------------------------------------

def _allowed_numbers(obj):
    found = set()

    def walk(x):
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            found.add(abs(float(x)))
            if abs(x) <= 1:
                found.add(round(abs(x) * 100, 1))  # probabilities written as percentages
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            for m in re.findall(r"\d[\d,'.]*", x):
                try:
                    found.add(float(m.replace(",", "").replace("'", "")))
                except ValueError:
                    pass
    walk(obj)
    return found


def check_numbers(text, payload):
    """Return numbers in the explanation that do not match any number in the engine output."""
    allowed = _allowed_numbers(payload)
    bad = []
    for m in re.finditer(r"(?<![A-Za-z\d])(\d[\d,']*(?:\.\d+)?)\s*([kK]\b|%)?", text):
        raw = m.group(1).rstrip(",").replace(",", "").replace("'", "")
        try:
            v = float(raw) * (1000 if m.group(2) in ("k", "K") else 1)
        except ValueError:
            continue
        if m.group(2) == "%":  # every percentage is checked, however small
            if not any(abs(v - a) <= 0.5 for a in allowed):
                bad.append(m.group(0).strip())
            continue
        if v < 13:  # small counts (years, '3-month buffer') are not checked
            continue
        # Full numbers must match exactly (±1 for rounding); "395k" may round to the nearest thousand.
        tol = 1000.0 if m.group(2) in ("k", "K") else 1.0
        if not any(abs(v - a) <= tol for a in allowed):
            bad.append(m.group(0).strip())
    return bad


def _round(x):
    if isinstance(x, float):
        return round(x, 3) if abs(x) < 1 else round(x)
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v) for v in x]
    return x


def _readable(row):
    """Engine row -> clearly named fields, probabilities already as whole percentages."""
    return {
        "scenario": row["scenario"],
        "assumption_notes": row["notes"],
        "end_wealth_chf": {"p10": row["nw_p10"], "median": row["nw_p50"], "p90": row["nw_p90"]},
        "chance_of_needing_financing_percent": round(row["p_shortfall"] * 100),
        "median_months_with_negative_cash": row["median_short_months"],
        "bad_case_peak_shortfall_chf": row["p90_peak_shortfall"],
        "median_months_below_cash_buffer": row["median_buffer_months"],
        "goal_reached_on_time_percent": {g: round(v * 100) for g, v in row["goals"].items()},
        "median_wealth_chf_by_year": {f"year {y['year']}": y["p50"] for y in row.get("years", [])
                                      if y["year"] in (0, 5, 10)},
    }


def explain(p, results, question):
    provided = [k for k, v in p["tags"].items() if str(v).startswith("provided")]
    other = {k: v for k, v in p["tags"].items() if not str(v).startswith("provided")}
    payload = _round({
        "question": question,
        "person": {k: p[k] for k in ("age", "employment", "household", "canton", "cash", "income", "rent")},
        "inputs_provided_by_user": provided,
        "inputs_estimated_or_default": other,
        "scenarios": [_readable(r) for r in results["rows"]],
        "median_end_wealth_difference_chf": results.get("median_end_wealth_difference"),
        "warnings": results["warnings"],
        "peer_path": ({"your_percentile_among_same_age_people": round(results["peer_percentile"] * 100),
                       "points": [{"age": q["age"], "median_chf": q["mid"]} for q in results["peer_trajectory"]
                                  if q["year"] in (0, 5, 10)]}
                      if results.get("peer_trajectory") else None),
        "simulated_futures": results["paths"],
        "horizon_months": results["horizon_months"],
    })
    msgs = [{"role": "system", "content": EXPLAIN_PROMPT},
            {"role": "user", "content": json.dumps(payload)}]
    try:
        text = chat(msgs, temperature=0.3)
    except LLMUnavailable as e:
        return None, [f"LLM unavailable ({e})"]
    return text.strip(), check_numbers(text, payload)


# ---- trade-off explanation (NSGA-II front) ----------------------------------------------------

FRONTIER_PROMPT = """You explain the result of a trade-off search to the person it describes.
The search tried many plans: one life decision plus two levers (cutting spending by up to 20%, a side
income of up to CHF 800/month) and kept the plans where no goal can improve without another getting worse.
Goals: higher median wealth after 10 years, fewer months with cash below a 3-month buffer, and less
sacrifice (CHF per month cut or earned on purpose).
Rules:
- Use ONLY numbers in the JSON. Do not compute new numbers.
- First compare the decisions with no sacrifice (each_decision.no_sacrifice): wealth, months below buffer,
  chance of negative cash. Then say how the levers change the picture (highlights).
- Say which decisions appear on the trade-off front and which never do (on_overall_front = 0 means another
  plan is at least as good on every goal).
- Mention every warning once, in plain words (one of them explains the saving calibration).
- Never recommend a choice and never say what the person should do. Describe trade-offs only.
- End with one sentence: illustrative simulations on synthetic data and assumptions, not advice or a forecast.
- Plain language, about 180-230 words, short paragraphs or bullets. Amounts as CHF with thousands separators."""


def _plan(m):
    return {"decision": m["decision"], "expense_cut_percent": round(m["expense_cut"] * 100),
            "side_income_chf_month": m["extra_income"], "sacrifice_chf_month": m["sacrifice_chf_month"],
            "median_wealth_chf": m["wealth_median"], "p10_wealth_chf": m["wealth_p10"],
            "p90_wealth_chf": m["wealth_p90"], "months_below_buffer": m["months_below_buffer"],
            "chance_cash_negative_percent": round(m["chance_cash_negative"] * 100)}


def explain_frontier(p, front, reference, per_decision, highlights, warnings, question):
    payload = _round({
        "question": question,
        "person": {k: p[k] for k in ("age", "employment", "household", "canton", "cash", "income", "rent")},
        "saving_calibration_keep_percent": round(p.get("keep_factor", 1.0) * 100),
        "inputs_estimated_or_default": {k: v for k, v in p["tags"].items() if not str(v).startswith("provided")},
        "as_is": _plan(reference),
        "plans_on_trade_off_front": len(front),
        "each_decision": {d: {"no_sacrifice": _plan(r["no_sacrifice"]), "best_found": _plan(r["most_wealth"]),
                              "on_overall_front": r["on_overall_front"]} for d, r in per_decision.items()},
        "highlights": {k: _plan(m) for k, m in highlights.items()},
        "warnings": warnings,
    })
    msgs = [{"role": "system", "content": FRONTIER_PROMPT}, {"role": "user", "content": json.dumps(payload)}]
    try:
        text = chat(msgs, temperature=0.3)
    except LLMUnavailable as e:
        return None, [f"LLM unavailable ({e})"]
    return text.strip(), check_numbers(text, payload)
