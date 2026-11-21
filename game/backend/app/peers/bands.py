"""Peer benchmark: older synthetic profiles with a similar background form a band.

Migrated from twin/peers.py (same distance weights, windows and quantile rule). Limitations, shown to the
user with every response:
* The population is the synthetic challenge data (Data/), not real people or official statistics.
* It is cross-sectional: different people at an older age, NOT a forecast of what this user becomes.
* Wealth here = opening cash + investment portfolio - opening debt principal in that table.
"""
from __future__ import annotations

import math
import statistics as st
from functools import lru_cache

import pandas as pd

from ..data.synthetic import DATA_DIR

W_CANTON, W_HOUSEHOLD, W_INCOME, W_RENT = 0.5, 1.0, 1.0, 0.5
WINDOWS = (3, 5, 8)
LIMITATIONS = ("Synthetic challenge population, not real people or official statistics.",
               "Cross-sectional: other people at an older age, not a forecast of your future.",
               "Matched on canton, household, net income and rent; age is not a matching feature.")


@lru_cache(maxsize=1)
def population() -> tuple[dict, ...]:
    pr = pd.read_csv(DATA_DIR / "synthetic-profiles.csv")
    assets = pd.read_csv(DATA_DIR / "synthetic-assets.csv")
    li = pd.read_csv(DATA_DIR / "synthetic-liabilities.csv")
    port = assets.loc[assets["asset_type"] == "investment_portfolio"].groupby("profile_id")["value_chf"].sum()
    debt = li.groupby("profile_id")["opening_principal_chf"].sum()
    out = []
    for r in pr.itertuples(index=False):
        p, d = float(port.get(r.profile_id, 0.0)), float(debt.get(r.profile_id, 0.0))
        cash = float(r.opening_cash_chf)
        out.append({"id": r.profile_id, "age": int(r.age_at_start), "canton": r.canton, "household": r.household_type,
                    "income": max(1.0, float(r.reference_monthly_net_income_chf)),
                    "rent": max(1.0, float(r.reference_monthly_rent_chf)), "cash": cash, "portfolio": p, "debt": d,
                    "wealth": cash + p - d})
    return tuple(out)


def distance(a: dict, b: dict) -> float:
    return (W_CANTON * (a.get("canton") != b["canton"]) + W_HOUSEHOLD * (a.get("household") != b["household"])
            + W_INCOME * abs(math.log(max(1.0, a["income"]) / b["income"]))
            + W_RENT * abs(math.log(max(1.0, a["rent"]) / b["rent"])))


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    pos = q * (len(xs) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def peer_band(me: dict, years: int = 10, k: int = 25) -> dict:
    """me: {age, canton, household, income (net CHF/month), rent}."""
    center = me["age"] + years
    pool = population()
    for w in WINDOWS:
        cands = [q for q in pool if abs(q["age"] - center) <= w]
        if len(cands) >= k:
            break
    ranked = sorted(cands, key=lambda q: distance(me, q))[:k]
    vals = [q["wealth"] for q in ranked]
    return {"center_age": center, "age_window": w, "n_peers": len(ranked), "pool_size": len(cands),
            "wealth_chf": {"p25": quantile(vals, .25), "p50": quantile(vals, .50), "p75": quantile(vals, .75)},
            "mean_distance": st.mean(distance(me, q) for q in ranked) if ranked else None,
            "basis": "nominal CHF in the synthetic table (as of its own date)",
            "limitations": list(LIMITATIONS), "data_status": "synthetic"}
