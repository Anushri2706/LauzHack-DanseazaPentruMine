"""Peer benchmark: older profiles with a similar background (age excluded) form a target band.

This is a cross-sectional benchmark, NOT a forecast of what the user will become.
"""
import math
import statistics as st
from collections import defaultdict
from functools import lru_cache

from .profile import _rows

# Distance weights (documented assumptions). Age is deliberately not a matching feature.
W_CANTON, W_HOUSEHOLD, W_INCOME, W_RENT = 0.5, 1.0, 1.0, 0.5
WINDOWS = (3, 5, 8)  # +/- years around the target age, widened if too few peers


@lru_cache(maxsize=1)
def population():
    port, debt, goals, gnames = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(list)
    for r in _rows("synthetic-assets.csv"):
        if r["asset_type"] == "investment_portfolio":
            port[r["profile_id"]] += float(r["value_chf"])
    for r in _rows("synthetic-liabilities.csv"):
        debt[r["profile_id"]] += float(r["opening_principal_chf"])
    for r in _rows("synthetic-goals.csv"):
        goals[r["profile_id"]] += float(r["target_amount_chf"])
        gnames[r["profile_id"]].append(r["goal_name"])
    pop = []
    for r in _rows("synthetic-profiles.csv"):
        i = r["profile_id"]
        cash = float(r["opening_cash_chf"])
        pop.append({"id": i, "age": int(r["age_at_start"]), "canton": r["canton"],
                    "household": r["household_type"], "employment": r["employment_status"],
                    "income": max(1.0, float(r["reference_monthly_net_income_chf"])),
                    "rent": max(1.0, float(r["reference_monthly_rent_chf"])),
                    "cash": cash, "portfolio": port[i], "debt": debt[i],
                    "wealth": cash + port[i] - debt[i], "goal_total": goals[i], "goals": gnames[i]})
    return pop


def distance(a, b, income_a=None, ignore=()):
    """Background distance between profiles a and b. `ignore` drops features (e.g. what a decision changes)."""
    inc = max(1.0, income_a or a["income"])
    d = 0.0
    if "canton" not in ignore:
        d += W_CANTON * (a["canton"] != b["canton"])
    if "household" not in ignore:
        d += W_HOUSEHOLD * (a["household"] != b["household"])
    if "income" not in ignore:
        d += W_INCOME * abs(math.log(inc / b["income"]))
    if "rent" not in ignore:
        d += W_RENT * abs(math.log(max(1.0, a["rent"]) / b["rent"]))
    return d


def find_peers(p, years=10, k=25, ref_income=None, pool=None, state=None, ignore=()):
    """p: dict with id, age, canton, household, income, rent. Returns peers sorted by similarity.
    state: only people with these field values (e.g. {"employment": "part_time"}).
    ignore: features left out of the match."""
    center = p["age"] + years
    pool = pool if pool is not None else population()
    state = state or {}
    for w in WINDOWS:
        cands = [q for q in pool if q["id"] != p["id"] and abs(q["age"] - center) <= w
                 and all(q[f] == v for f, v in state.items())]
        if len(cands) >= k:
            break
    dist = lambda q: distance(p, q, ref_income, ignore)
    ranked = sorted(cands, key=dist)[:k]
    return {"peers": ranked, "window": w, "center_age": center, "pool_size": len(cands),
            "mean_distance": st.mean(dist(q) for q in ranked) if ranked else None}


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    pos = q * (len(xs) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def bands(peers):
    out = {}
    for key in ("wealth", "cash", "portfolio", "goal_total"):
        v = [q[key] for q in peers]
        out[key] = {"p25": quantile(v, .25), "p50": quantile(v, .50), "p75": quantile(v, .75)}
    return out


def compare(samples, band):
    """Share of simulated futures whose end net worth reaches each peer quantile."""
    n = len(samples)
    return {lbl: sum(s >= band[lbl] for s in samples) / n for lbl in ("p25", "p50", "p75")}
