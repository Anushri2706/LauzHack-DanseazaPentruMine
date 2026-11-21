"""Monte Carlo "worlds": seeded random shocks, shared by every branch (scenario isolation).

A world is generated ONLY from (uncertainty assumptions, horizon, seed, world index). It never
depends on the decisions, so two branches compared under the same seed see exactly the same
market moves, income-loss spells and rent rises; differences come from decisions, not luck.

Distributions (all our labelled assumptions, see assumptions.py):
  monthly return  r_t = mu + sigma/sqrt(12) * z,  z ~ N(0, 1)
                  mu is set so the expected annual return, after expected drops, equals return_annual
  market drop     per year with prob crash_prob_annual, one month loses crash_size
  income loss     per year with prob job_loss_prob_annual, a 3-9 month spell on
                  unemployment_benefit_rate x income (only bites while working/abroad)
  rent rise       per year with prob rent_shock_prob_annual, rent x (1 + rent_shock_size) from then on
Percentiles are "ranges under our assumptions", NOT real-world probabilities.
"""
from functools import lru_cache

import numpy as np

from .metrics import check_goals
from .simulate import run_branch
from .state import Decision, State

UNCERTAINTY_KEYS = ("return_annual", "return_volatility", "crash_prob_annual", "crash_size",
                    "job_loss_prob_annual", "unemployment_benefit_rate", "rent_shock_prob_annual",
                    "rent_shock_size", "horizon_months", "start_age")


def _age(p_start_age: float, month: int) -> float:
    return round(p_start_age + month / 12, 1)


@lru_cache(maxsize=64)
def _worlds(key: tuple, n: int, seed: int) -> tuple:
    p = dict(zip(UNCERTAINTY_KEYS, key))
    horizon = int(p["horizon_months"])
    years = (horizon + 11) // 12
    expected_drop = p["crash_prob_annual"] * p["crash_size"]
    mu = ((1 + p["return_annual"]) / (1 - expected_drop)) ** (1 / 12) - 1
    sigma = p["return_volatility"] / np.sqrt(12)
    worlds = []
    for k in range(n):
        rng = np.random.default_rng([seed, k])
        ret = mu + sigma * rng.standard_normal(horizon)
        income = np.ones(horizon)
        rent = np.ones(horizon)
        events = []
        for y in range(years):
            if rng.random() < p["crash_prob_annual"]:
                m = min(horizon - 1, y * 12 + int(rng.integers(12)))
                ret[m] = (1 + ret[m]) * (1 - p["crash_size"]) - 1
                events.append({"month": m, "kind": "market_drop",
                               "text": f"Markets drop {p['crash_size']:.0%} at age {_age(p['start_age'], m)}"})
            if rng.random() < p["job_loss_prob_annual"]:
                m = y * 12 + int(rng.integers(12))
                dur = int(rng.integers(3, 10))
                if m < horizon:
                    income[m:m + dur] = p["unemployment_benefit_rate"]
                    events.append({"month": m, "kind": "income_loss",
                                   "text": f"{dur} months without a job from age {_age(p['start_age'], m)} "
                                           f"(if working)"})
            if rng.random() < p["rent_shock_prob_annual"]:
                m = min(horizon - 1, y * 12 + int(rng.integers(12)))
                rent[m:] *= 1 + p["rent_shock_size"]
                events.append({"month": m, "kind": "rent_rise",
                               "text": f"Rent rises {p['rent_shock_size']:.0%} at age {_age(p['start_age'], m)}"})
        events.sort(key=lambda e: e["month"])
        worlds.append({"ret": ret, "income_mult": income, "rent_mult": rent, "events": events})
    return tuple(worlds)


def make_worlds(p: dict, n: int, seed: int) -> tuple:
    return _worlds(tuple(float(p[k]) for k in UNCERTAINTY_KEYS), n, seed)


def stress_world(p: dict, kind: str, month: int) -> dict:
    """Named 'what could be' ghost scenarios on top of the expected path."""
    from .finance import monthly_compound_rate
    horizon = int(p["horizon_months"])
    ret = np.full(horizon, monthly_compound_rate(p["return_annual"]))
    income, rent = np.ones(horizon), np.ones(horizon)
    month = max(0, min(horizon - 1, month))
    if kind == "market_drop":
        ret[month] = (1 + ret[month]) * (1 - 0.2) - 1
        text = f"If markets drop 20% at age {_age(p['start_age'], month)}"
    elif kind == "income_loss":
        income[month:month + 6] = p["unemployment_benefit_rate"]
        text = f"If you are without a job for 6 months from age {_age(p['start_age'], month)}"
    elif kind == "rent_rise":
        rent[month:] *= 1.2
        text = f"If rent rises 20% at age {_age(p['start_age'], month)}"
    else:
        raise ValueError(kind)
    return {"ret": ret, "income_mult": income, "rent_mult": rent, "text": text, "kind": kind}


def run_montecarlo(p: dict, decisions: list[Decision], start: State, goals: list[dict],
                   n: int = 300, seed: int = 42, samples: int = 3) -> dict:
    worlds = make_worlds(p, n, seed)
    horizon = int(p["horizon_months"])
    nw = np.empty((n, horizon))
    cash = np.empty((n, horizon))
    goal_hits = {g["goal_id"]: 0 for g in goals}
    for k, w in enumerate(worlds):
        rows = run_branch(p, decisions, start=start, world=w).rows
        nw[k] = [r["net_worth"] for r in rows]
        cash[k] = [r["cash"] for r in rows]
        for g in check_goals(goals, rows):
            if g["status"] == "met":
                goal_hits[g["goal_id"]] += 1
    pct = lambda a: np.percentile(a, [10, 50, 90], axis=0)  # noqa: E731
    nw_p, cash_p = pct(nw), pct(cash)
    shortfall = (cash < 0).any(axis=1)
    final = nw[:, -1]
    sample_ids = list(range(min(samples, n)))
    return {
        "n": n, "seed": seed,
        "net_worth": {"p10": nw_p[0].tolist(), "p50": nw_p[1].tolist(), "p90": nw_p[2].tolist()},
        "cash": {"p10": cash_p[0].tolist(), "p50": cash_p[1].tolist(), "p90": cash_p[2].tolist()},
        "final_net_worth": {"p10": float(np.percentile(final, 10)), "p50": float(np.median(final)),
                            "p90": float(np.percentile(final, 90))},
        "share_worlds_shortfall": float(shortfall.mean()),
        "goal_met_share": {gid: hits / n for gid, hits in goal_hits.items()},
        "samples": [{"world": k, "net_worth": nw[k].tolist(), "cash": cash[k].tolist(),
                     "events": worlds[k]["events"]} for k in sample_ids],
    }


def sample_world(p: dict, decisions: list[Decision], start: State, seed: int, world: int) -> dict:
    """One concrete world ("roll the dice"): its path and the events that happened in it."""
    w = make_worlds(p, world + 1, seed)[world]
    rows = run_branch(p, decisions, start=start, world=w).rows
    return {"world": world, "net_worth": [r["net_worth"] for r in rows], "cash": [r["cash"] for r in rows],
            "events": w["events"]}
