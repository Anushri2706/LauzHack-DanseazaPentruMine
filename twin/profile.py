"""Load one synthetic profile and tag every field as provided / estimated / default."""
import csv
import statistics as st
from collections import defaultdict
from datetime import date
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "Data"


@lru_cache(maxsize=None)
def _rows(name):
    """All rows of a data file, read once and cached (files are small)."""
    with open(DATA / name, newline="") as f:
        return tuple(csv.DictReader(f))


def canton_rent_index():
    """Median reference rent per canton from the synthetic profiles (NOT official statistics)."""
    by = defaultdict(list)
    for r in _rows("synthetic-profiles.csv"):
        by[r["canton"]].append(float(r["reference_monthly_rent_chf"]))
    return {k: st.median(v) for k, v in by.items()}


def _months_from(start: date, deadline: str) -> int:
    y, m, _ = (int(x) for x in deadline.split("-"))
    return (y - start.year) * 12 + (m - start.month)


def load_profile(profile_id: str) -> dict:
    prof = next((r for r in _rows("synthetic-profiles.csv") if r["profile_id"] == profile_id), None)
    if prof is None:
        raise SystemExit(f"Unknown profile {profile_id}")
    start = date.fromisoformat(prof["as_of_date"])
    income = float(prof["reference_monthly_net_income_chf"])
    rent = float(prof["reference_monthly_rent_chf"])

    from .learned import fit  # local import avoids a circular import
    nets = [float(r["net_transactions_chf"]) for r in _rows("synthetic-cash-balances.csv")
            if r["profile_id"] == profile_id]
    mean_net = st.mean(nets) if nets else None
    # Volatility of the person's own history with the learned seasonal pattern removed
    sd_net = fit()["resid_sd"].get(profile_id, 0.0)
    # Ledger net = income - rent - other spending  =>  other = income - rent - net
    other = max(0.0, income - rent - mean_net) if mean_net is not None else 0.4 * income

    portfolio = sum(float(r["value_chf"]) for r in _rows("synthetic-assets.csv")
                    if r["profile_id"] == profile_id and r["asset_type"] == "investment_portfolio")
    loans = [(float(r["opening_principal_chf"]), float(r["annual_interest_rate"]))
             for r in _rows("synthetic-liabilities.csv") if r["profile_id"] == profile_id]
    goals = [{"name": r["goal_name"], "target": float(r["target_amount_chf"]),
              "month": _months_from(start, r["target_date"]), "priority": r["priority"]}
             for r in _rows("synthetic-goals.csv") if r["profile_id"] == profile_id]
    s1 = {(r["parameter"]): float(r["value"]) for r in _rows("scenario-assumptions.csv")
          if r["profile_id"] == profile_id and r["scenario_id"] == "S001"}

    p = {
        "id": profile_id, "name": prof["display_name"], "age": int(prof["age_at_start"]),
        "city": prof["city"], "canton": prof["canton"], "employment": prof["employment_status"],
        "household": prof["household_type"],
        "cash": float(prof["opening_cash_chf"]), "income": income, "rent": rent,
        "other_spend": other, "net_sd": sd_net, "portfolio": portfolio, "loans": loans,
        "goals": goals, "master_params": s1, "horizon": 120,
    }
    tags = {
        "who (age, city, canton, employment, household)": "provided",
        "cash (liquidity)": "provided", "income, rent (cash-flow baseline)": "provided",
        "other spending": "estimated from ledger history" if nets else "default (40% of income)",
        "assets": "provided" + ("" if portfolio else " (cash only)"),
        "debt": "provided" if loans else "none on record",
        "goals": "provided" if goals else "none on record",
        "history": f"{len(nets)} months" if nets else "missing",
    }
    p["tags"] = tags
    return p


REQUIRED = ("cash", "income", "rent")
DEFAULTS = {"age": 30, "employment": "employed", "household": "single", "canton": "VD", "city": "Lausanne"}


def profile_from_inputs(inputs: dict) -> dict:
    """Build a profile for a NEW user from whatever they provide.

    Required: cash, income, rent (monthly CHF). Everything else is optional:
    background fields fall back to stated defaults, financial fields to learned estimates.
    Optional keys: age, employment, household, canton, city, other_spend, portfolio,
    loans [[principal, rate], ...], has_debt (bool), goals [{name, target, month, priority}].
    """
    from .learned import estimate
    missing = [k for k in REQUIRED if k not in inputs]
    if missing:
        raise ValueError(f"Required inputs missing: {', '.join(missing)}")
    tags = {}
    p = {"id": "USER", "name": inputs.get("name", "You"), "horizon": 120, "master_params": {}}
    for k in ("cash", "income", "rent"):
        p[k] = float(inputs[k])
        tags[k] = "provided"
    capitals = {"ZH": "Zurich", "GE": "Geneva", "VD": "Lausanne", "BE": "Bern", "BS": "Basel",
                "TI": "Lugano", "FR": "Fribourg", "NE": "Neuchatel", "LU": "Lucerne", "SG": "St. Gallen"}
    for k, v in DEFAULTS.items():
        if k == "city" and "city" not in inputs and inputs.get("canton") in capitals:
            v = capitals[inputs["canton"]]  # default city follows the canton the user gave
        p[k] = inputs.get(k, v)
        tags[k] = "provided" if k in inputs else f"default ({v})"
    p["age"] = int(p["age"])
    for field in ("other_spend", "portfolio"):
        if field in inputs:
            p[field] = float(inputs[field])
            tags[field] = "provided"
        else:
            p[field], how = estimate(field, p["age"], p["employment"], p["household"])
            tags[field] = f"estimated: {how}"
    from .assumptions import A
    p["net_sd"] = A("noise_chf")
    if "loans" in inputs:
        p["loans"] = [tuple(map(float, x)) for x in inputs["loans"]]
        tags["debt"] = "provided"
    else:
        p["loans"] = []
        tags["debt"] = ("none (user said no debt)" if inputs.get("has_debt") is False
                        else "unknown, assumed none - ask the user")
    p["goals"] = inputs.get("goals", [])
    tags["goals"] = "provided" if p["goals"] else "none given"
    tags["history"] = "none (new user)"
    for k, how in (inputs.get("estimated") or {}).items():  # values the twin estimated on request
        tags[k] = f"estimated: {how}"
    p["tags"] = tags
    return p
