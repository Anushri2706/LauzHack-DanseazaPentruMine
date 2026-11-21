"""Learned layer: patterns estimated from the synthetic data, not hand-set.

From the 24-month history (time patterns):
  - each person's month-to-month noise (resid_sd): their own swings after removing the typical
    seasonal pattern of their employment type (seasonality itself is not simulated)
  - financing-need rate per employment type (used only to validate the simulator)
From the 1,200-profile snapshot (cross-person patterns):
  - estimates for missing inputs: cash and portfolio by age band and employment,
    non-rent spending by employment and household, debt incidence and size

These are patterns in SYNTHETIC data. They describe how the data was generated, not Swiss households.
"""
import statistics as st
from collections import defaultdict
from functools import lru_cache

from .profile import _rows

AGE_BANDS = [(0, 30), (30, 45), (45, 65), (65, 200)]
MIN_GROUP = 15  # smallest group used before falling back to a broader one


def age_band(age):
    return next(i for i, (lo, hi) in enumerate(AGE_BANDS) if lo <= age < hi)


@lru_cache(maxsize=1)
def fit():
    prof = {r["profile_id"]: r for r in _rows("synthetic-profiles.csv")}
    nets, cal = defaultdict(list), defaultdict(list)
    neg, tot = defaultdict(int), defaultdict(int)
    for r in _rows("synthetic-cash-balances.csv"):
        pid = r["profile_id"]
        nets[pid].append(float(r["net_transactions_chf"]))
        cal[pid].append(int(r["month"][-2:]) - 1)  # 0 = January
        emp = prof[pid]["employment_status"]
        tot[emp] += 1
        neg[emp] += r["status"] != "ok"

    # Seasonality per employment: mean over people of (net - own mean) / income, by calendar month
    dev = defaultdict(lambda: defaultdict(list))
    for pid, v in nets.items():
        inc = max(1.0, float(prof[pid]["reference_monthly_net_income_chf"]))
        mean = st.mean(v)
        per_m = defaultdict(list)
        for x, c in zip(v, cal[pid]):
            per_m[c].append((x - mean) / inc)
        for c, xs in per_m.items():
            dev[prof[pid]["employment_status"]][c].append(st.mean(xs))
    season = {}
    for emp, by_m in dev.items():
        # median, not mean: ratios explode for people with tiny incomes (some unemployed earn < CHF 50)
        s = [st.median(by_m[c]) if by_m[c] else 0.0 for c in range(12)]
        mu = st.mean(s)
        season[emp] = [x - mu for x in s]  # zero-mean so it does not change the yearly total

    # Residual volatility (after seasonality) per person
    resid_sd = {}
    for pid, v in nets.items():
        p = prof[pid]
        inc = max(1.0, float(p["reference_monthly_net_income_chf"]))
        s = season[p["employment_status"]]
        mean = st.mean(v)
        res = [x - mean - s[c] * inc for x, c in zip(v, cal[pid])]
        resid_sd[pid] = st.pstdev(res)

    # Cross-person estimates for missing inputs
    port, debt = defaultdict(float), defaultdict(float)
    for r in _rows("synthetic-assets.csv"):
        if r["asset_type"] == "investment_portfolio":
            port[r["profile_id"]] += float(r["value_chf"])
    for r in _rows("synthetic-liabilities.csv"):
        debt[r["profile_id"]] += float(r["opening_principal_chf"])
    cash_g, port_g = defaultdict(list), defaultdict(list)
    cash_b, port_b = defaultdict(list), defaultdict(list)
    spend_eh, spend_e = defaultdict(list), defaultdict(list)
    for pid, p in prof.items():
        b, e, h = age_band(int(p["age_at_start"])), p["employment_status"], p["household_type"]
        cash_g[(b, e)].append(float(p["opening_cash_chf"]))
        port_g[(b, e)].append(port[pid])
        cash_b[b].append(float(p["opening_cash_chf"]))
        port_b[b].append(port[pid])
        inc = float(p["reference_monthly_net_income_chf"])
        other = max(0.0, inc - float(p["reference_monthly_rent_chf"]) - st.mean(nets[pid]))
        spend_eh[(e, h)].append(other)
        spend_e[e].append(other)
    debts = [debt[pid] for pid in prof if debt[pid] > 0]

    return {
        "resid_sd": resid_sd,
        "financing_rate": {k: neg[k] / tot[k] for k in tot},
        "cash": (cash_g, cash_b),
        "portfolio": (port_g, port_b),
        "spend": (spend_eh, spend_e),
        "debt_incidence": len(debts) / len(prof),
        "debt_median": st.median(debts),
    }


def estimate(field, age=None, employment=None, household=None):
    """Learned estimate of a missing input. Returns (value, how it was estimated)."""
    f = fit()
    if field in ("cash", "portfolio"):
        g, b = f[field]
        band = age_band(age)
        key = (band, employment)
        if len(g[key]) >= MIN_GROUP:
            return st.median(g[key]), f"median of {len(g[key])} profiles, same age band and employment"
        return st.median(b[band]), f"median of {len(b[band])} profiles, same age band"
    if field == "other_spend":
        eh, e = f["spend"]
        if len(eh[(employment, household)]) >= MIN_GROUP:
            n = len(eh[(employment, household)])
            return st.median(eh[(employment, household)]), f"median of {n} profiles, same employment and household"
        return st.median(e[employment]), f"median of {len(e[employment])} profiles, same employment"
    raise KeyError(field)
