"""Monte Carlo simulation with common random numbers so scenarios are compared fairly."""
import math
import random
import statistics as st
from .assumptions import A


def make_shocks(p, n_paths, seed=7):
    """One set of random futures (inflation regime path, returns, income noise) reused by all scenarios."""
    rng = random.Random(seed)
    H = p["horizon"]
    names = ["low", "base", "high"]
    rates = A("regime_inflation")
    T = A("regime_transitions")
    shocks = []
    for _ in range(n_paths):
        state, infl = 1, []
        for m in range(H):
            if m and m % 12 == 0:
                state = rng.choices(range(3), weights=T[state])[0]
            infl.append(rates[names[state]])
        ret = [rng.gauss(A("portfolio_return") / 12, A("portfolio_vol") / math.sqrt(12)) for _ in range(H)]
        noise = [rng.gauss(0, p["net_sd"]) for _ in range(H)]
        shocks.append((infl, ret, noise))
    return shocks


def run_path(p, plan, shock):
    """One simulated future, month by month. The whole financial model is the one line marked CORE.

    CORE:  cash += income - spending - one-off costs - loan interest + noise
      income    = plan income x wage growth (1%/yr)
      spending  = (plan spending + one-offs) x price level (inflation regime of this future)
      noise     = a random swing in CHF (fixed amount, not a share of income), x plan["noise_scale"]
                  (only self-employment changes it)
    Saving calibration (calibration.py, on by default in the CLI and chat): calibration_rate = 1 - keep
    factor, the share of any monthly surplus spent beyond the ledger, while cash >= 0.
    plan["deliberate"] (explore.py levers: expense cut + side income, today's CHF) is added after the
    calibration, so a saving the person chooses on purpose is kept in full.
    Existing investments grow by a random monthly return. Debt principal stays constant.
    Negative cash is reported as an unfunded shortfall; no loan or interest is invented (starter kit rule).
    """
    infl, ret, noise = shock
    H = p["horizon"]
    cash, port = p["cash"], p["portfolio"]
    debt = sum(x for x, _ in p["loans"])
    loan_interest = sum(x * r for x, r in p["loans"]) / 12
    price, wage = 1.0, 1.0
    cash_path, nw_path, buf_breach, short = [], [], 0, 0
    peak_short = 0.0
    nw_years = [cash + port - debt]  # year 0 = today
    for m in range(H):
        price *= 1 + infl[m] / 12
        wage *= 1 + A("wage_growth_annual") / 12
        income = plan["income"][m] * wage
        spend = (plan["spend"][m] + plan["oneoff"][m]) * price
        rate = p.get("calibration_rate", 0.0)
        if rate and income > spend and cash >= 0:  # saving calibration: a share of any surplus is spent
            spend += rate * (income - spend)
        cash += income - spend - loan_interest + noise[m] * plan.get("noise_scale", 1.0)  # CORE
        if "deliberate" in plan:  # chosen expense cuts / side income: kept in full, not calibrated away
            cash += plan["deliberate"][m] * price
        if cash < 0:
            short += 1
            peak_short = max(peak_short, -cash)
        port *= 1 + ret[m]
        buffer = A("buffer_months") * plan["spend"][m] * price
        if cash < buffer:
            buf_breach += 1
        cash_path.append(cash)
        nw_path.append(cash + port - debt)
        if (m + 1) % 12 == 0:  # yearly point of the trajectory, in today's CHF
            nw_years.append(nw_path[-1] / price)
    goals_met = {g["name"]: (cash_path[min(g["month"], H - 1)] + port_at(p, ret, g["month"]) >= g["target"] * 1.0)
                 for g in p["goals"]}
    return {"final_nw_real": nw_path[-1] / price, "min_cash": min(cash_path),
            "short_months": short, "buffer_months": buf_breach, "peak_shortfall": peak_short,
            "goals": goals_met, "nw_years": nw_years}


def port_at(p, ret, month):
    v = p["portfolio"]
    for r in ret[: min(month, len(ret) - 1) + 1]:
        v *= 1 + r
    return v


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def summarize(results, p):
    fin = [r["final_nw_real"] for r in results]
    out = {
        "nw_p10": pct(fin, .10), "nw_p50": pct(fin, .50), "nw_p90": pct(fin, .90),
        "p_shortfall": sum(r["short_months"] > 0 for r in results) / len(results),
        "median_short_months": st.median(r["short_months"] for r in results),
        "p90_peak_shortfall": pct([r["peak_shortfall"] for r in results], .90),
        "median_buffer_months": st.median(r["buffer_months"] for r in results),
        "goals": {},
        "nw_samples": fin,
        # year-by-year trajectory band (today's CHF): year 0 = today
        "years": [{"year": y, "p10": pct(col, .10), "p50": pct(col, .50), "p90": pct(col, .90)}
                  for y, col in enumerate(zip(*[r["nw_years"] for r in results]))],
    }
    for g in p["goals"]:
        out["goals"][g["name"]] = sum(r["goals"][g["name"]] for r in results) / len(results)
    return out
