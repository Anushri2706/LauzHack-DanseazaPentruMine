"""Bridge to the combined regime-switching HMM engine (`app/regime`, migrated from feature/combined-engine).

The town game simulates the player's own month-by-month choices. This module answers a different question
with the HMM engine: "if I took one big life decision, how would my cash fan out over many simulated lives?"
Each decision is a preset list of `Intervention`s (do() on the regime chain) plus optional one-off costs,
taken from the teammate's run_alexia_*.py scripts. All paths share the seed, so differences come from
the decision, not from luck.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache

import numpy as np

from .regime.accounting import OneOff, run as run_cash  # noqa: E402
from .regime.compat import apply_fractional_drop, goal_hit_rate, load_profile  # noqa: E402
from .regime.regime import RegimeEngine, fit_emissions  # noqa: E402
from .regime.transitions import load_transition_priors  # noqa: E402
from .regime.types import STATES, Intervention  # noqa: E402

START = date(2025, 1, 1)
OVERRIDES = {"scaled_spending": 1}   # the expense-priors spending recipe, as in all teammate scripts
DISCLAIMER = ("Simulated under the stated assumptions (regime HMM, transition hazards and income table in "
              "app/regime/assumptions/*.csv). Not a prediction or financial advice. Synthetic data only.")
WORKING = {"employed", "part_time", "job_search"}


@dataclass(frozen=True)
class Preset:
    id: str
    label: str
    icon: str
    text: str
    for_states: frozenset          # initial regimes for which the decision makes sense
    build: object = field(repr=False)  # (s: start month, params) -> (interventions, one_offs, deposit_month|None)


def _masters(s: int) -> list[Intervention]:
    return [Intervention("force_state", s, 24, state="student"),
            Intervention("set_attribute", s + 24, attribute=("has_masters", True))]


ALL = frozenset(STATES)
ACTIVE = frozenset(STATES) - {"retired"}
PRESETS: list[Preset] = [
    Preset("stay", "Keep going as you are", "🏠", "No decision: the regime chain runs on its own hazards.",
           ALL, lambda s, p: ([], [], None)),
    Preset("masters_now", "Master's now", "🎓", "Study 24 months, then work with the master's premium.",
           ACTIVE, lambda s, p: (_masters(s), [], None)),
    Preset("masters_later", "Work 2 years, then a master's", "⏳",
           "Employed 24 months, then study 24 months, then the master's premium.",
           ACTIVE, lambda s, p: ([Intervention("force_state", s, 24, state="employed")] + _masters(s + 24), [], None)),
    Preset("part_time", "Part-time for 2 years", "🌗",
           "Part-time (part_time_fraction of full pay) for 24 months, then full-time.",
           ACTIVE, lambda s, p: ([Intervention("force_state", s, 24, state="part_time"),
                                  Intervention("force_state", s + 24, None, state="employed")], [], None)),
    Preset("job_break", "6-month job search", "🔎",
           "A 6-month job-search spell in a year, on unemployment insurance (70–80% of the last wage).",
           ACTIVE, lambda s, p: ([Intervention("force_state", s + 12, 6, state="job_search")], [], None)),
    Preset("parental_leave", "Parental leave", "👶",
           "A parental-leave pin in a year: parental_leave_replacement of the last wage, then back to work.",
           ACTIVE, lambda s, p: ([Intervention("parental_leave", s + 12)], [], None)),
    Preset("move_zurich", "Move to Zurich", "🏙️", "Housing × zurich_housing_multiplier, one-off move cost.",
           ALL, lambda s, p: ([Intervention("set_attribute", s, attribute=("zurich", True))],
                              [OneOff(s, float(p.get("zurich_move_cost_chf", 4000)), "move to Zurich")], None)),
    Preset("move_germany", "Move to Germany", "🇩🇪",
           "Wage × abroad_income_multiplier, costs × abroad_cost_multiplier, 4 months job search, move cost.",
           ACTIVE, lambda s, p: ([Intervention("set_attribute", s, attribute=("abroad", True)),
                                  Intervention("force_state", s, 4, state="job_search")],
                                 [OneOff(s, float(p.get("germany_move_cost_chf", 5000)), "move to Germany")], None)),
    Preset("rent_step", "Bigger flat", "🛋️", "Rent × rent_step_factor from now on.",
           ALL, lambda s, p: ([Intervention("set_attribute", s, attribute=("rent_step", True))], [], None)),
    Preset("buy_home", "Buy a home", "🔑",
           "Pay purchase_deposit_fraction of cash as a deposit; the housing bill stays at today's rent.",
           ALL, lambda s, p: ([Intervention("set_attribute", s, attribute=("owns_home", True))], [], s)),
]
BY_ID = {x.id: x for x in PRESETS}


@lru_cache(maxsize=1)
def engine() -> RegimeEngine:
    return RegimeEngine(fit_emissions(), load_transition_priors())


@lru_cache(maxsize=256)
def profile(pid: str):
    return load_profile(pid)


def _params() -> dict:
    return engine().income_table.with_overrides(OVERRIDES).params


# In the teammate's chain `student` is absorbing (graduation is a pinned decision, not a hazard), so for a
# student every non-study decision starts by finishing school: one pinned `employed` month at the start.
STUDENT_SELF_PINNED = {"masters_now", "masters_later", "part_time", "move_germany"}


def _label(pre: Preset, state: str) -> tuple[str, str]:
    if pre.id == "stay" and state == "student":
        return "Finish studies, start working", "Employed from now on; the regime chain runs on its own hazards."
    return pre.label, pre.text


def _build(pre: Preset, state: str, s: int, params: dict):
    ivs, one_offs, deposit_month = pre.build(s, params)
    if state == "student" and pre.id not in STUDENT_SELF_PINNED:
        ivs = [Intervention("force_state", s, None, state="employed")] + ivs
    return ivs, one_offs, deposit_month


def decisions_for(pid: str) -> dict:
    prof = profile(pid)
    st = prof.initial_state
    return {"persona_id": pid, "initial_state": st, "states": STATES,
            "decisions": [{"id": x.id, "label": _label(x, st)[0], "icon": x.icon, "text": _label(x, st)[1],
                           "available": st in x.for_states} for x in PRESETS],
            "disclaimer": DISCLAIMER}


def _q(a: np.ndarray, qs=(10, 50, 90)) -> list[list[float]]:
    return np.percentile(a, qs, axis=0).round(2).tolist()


@lru_cache(maxsize=128)
def _fan(pid: str, ids: tuple[str, ...], horizon: int, n_paths: int, seed: int, start_month: int,
         goals: tuple[tuple[str, str, float, str], ...]) -> dict:
    prof = profile(pid)
    eng = engine()
    params = _params()
    out = []
    for did in ids:
        if did not in BY_ID:
            raise ValueError(f"Unknown decision {did!r}")
        pre = BY_ID[did]
        if prof.initial_state not in pre.for_states:
            raise ValueError(f"“{pre.label}” does not apply to someone who is {prof.initial_state}")
        ivs, one_offs, deposit_month = _build(pre, prof.initial_state, start_month, params)
        ivs = [iv for iv in ivs if iv.start_month < horizon]
        one_offs = [o for o in one_offs if o.month < horizon]
        paths = eng.simulate(prof, horizon_months=horizon, n_paths=n_paths, interventions=ivs,
                             assumption_overrides=OVERRIDES, seed=seed)
        cp = run_cash(prof, paths, one_offs)
        deposit = None
        if deposit_month is not None and deposit_month < horizon:
            cp, dep = apply_fractional_drop(cp, deposit_month, float(params.get("purchase_deposit_fraction", 0.15)))
            deposit = float(np.median(dep))
        cash = cp.cash
        p10, p50, p90 = _q(cash)
        inc = _q(paths.income_chf, (50,))[0]
        months_in = {s: float((paths.states == k).sum(axis=1).mean()) for k, s in enumerate(STATES)}
        goal_rows = []
        for gid, name, amount, gdate in goals:
            g = goal_hit_rate(cash, START, (amount, gdate))
            goal_rows.append({"goal_id": gid, "name": name, "target": amount, "target_date": gdate,
                              "hit_rate": float(g["hit_rate"]), "evaluated_at": g["evaluated_at"],
                              "goal_month": int(g["goal_month"]),
                              "median_cash_at_goal": float(g["median_cash_at_goal"])})
        closing = cash[:, -1]
        out.append({
            "id": did, "label": _label(pre, prof.initial_state)[0], "icon": pre.icon,
            "cash": {"p10": p10, "p50": p50, "p90": p90},
            "income_p50": inc,
            "occupancy": {s: paths.occupancy[:, k].round(4).tolist() for k, s in enumerate(STATES)},
            "months_in_state": months_in,
            "p_ever_negative": float((cash < 0).any(axis=1).mean()),
            "closing": {"p10": float(np.percentile(closing, 10)), "p50": float(np.median(closing)),
                        "p90": float(np.percentile(closing, 90))},
            "one_offs": [{"month": o.month, "amount": o.amount_chf, "label": o.label} for o in one_offs]
                        + ([{"month": deposit_month, "amount": deposit, "label": "home deposit (median)"}]
                           if deposit is not None else []),
            "goals": goal_rows,
        })
    return {"persona_id": pid, "name": prof.name, "age": prof.age, "initial_state": prof.initial_state,
            "opening_cash": prof.opening_cash, "horizon": horizon, "n_paths": n_paths, "seed": seed,
            "start": START.isoformat(), "start_month": start_month, "states": STATES, "decisions": out,
            "note": f"P10–P90 of {n_paths} simulated lives per decision, same seed for all. Month 0 = Jan 2025.",
            "disclaimer": DISCLAIMER}


def fan(pid: str, ids: list[str], horizon: int = 240, n_paths: int = 2000, seed: int = 0, start_month: int = 0,
        goals: list[dict] | None = None) -> dict:
    if not 12 <= horizon <= 480:
        raise ValueError("horizon must be between 12 and 480 months")
    if not 100 <= n_paths <= 10000:
        raise ValueError("n_paths must be between 100 and 10000")
    if not 0 <= start_month < horizon:
        raise ValueError("start_month must be inside the horizon")
    if not ids:
        raise ValueError("Pick at least one decision")
    g = tuple((str(x["goal_id"]), str(x["name"]), float(x["target"]), str(x["target_date"])) for x in goals or [])
    return _fan(pid, tuple(dict.fromkeys(ids)), horizon, n_paths, seed, start_month, g)
