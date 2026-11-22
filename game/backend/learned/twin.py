"""The learned twin for the game: HMM life states x learned money behaviour, from the player's current state.

    life states and incomes   the teammates' regime HMM (feature/hmm-and-frontend), from the player's state, age,
                              degree and country now; sourced Swiss hazards, retirement at 65, pensions
    monthly cash flows        the learned engine (behaviour.py): GMM spending per state, seasons, noise, and how
                              this persona's own 24 months in the data differ from people like them
    settings                  chosen on the validation people (learned/artifacts/backtest.json)

It answers "where do people like you, living through these life events, end up?" in liquid savings. It has no
investment returns (the data's cash flows have none), so it is not the town engine's net worth.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .behaviour import Behaviour
from .data import ledger
from .hmm import STATE_INDEX, STATES, Intervention, engine, profile_for

ART = Path(__file__).resolve().parent / "artifacts"
GAME_TO_STATE = {"working": "employed", "abroad": "employed", "unemployed": "job_search", "studying": "student",
                 "retired": "retired"}


@lru_cache(maxsize=1)
def report() -> dict:
    return json.loads((ART / "backtest.json").read_text())


@lru_cache(maxsize=1)
def behaviour() -> Behaviour:
    L = ledger()
    tr = L.rows("train")
    cfg = report()["chosen_on_validation"]
    b = Behaviour(n_components=int(cfg["n_components"])).fit(L.people.iloc[tr], L.flows[tr], L.month_of_year)
    b.kappa, b.noise_scale = float(cfg["kappa"]), float(cfg["noise_scale"])
    return b


@lru_cache(maxsize=1)
def hmm():
    L = ledger()
    return engine(tuple(L.people.index[L.rows("train")]))


def model_card() -> dict:
    r = report()
    keep = ("mae_cash_end_chf", "median_abs_error_chf", "r2_cash_end", "range_hit_rate_p10_p90",
            "monthly_range_coverage", "mean_range_width_chf", "n")
    return {"task": r["task"], "sizes": r["sizes"], "chosen_on_validation": r["chosen_on_validation"],
            "test": {m: {k: v[k] for k in keep} for m, v in r["test"].items()},
            "test_life_events_on": {m: {k: v[k] for k in keep} for m, v in r["test_life_events_on"].items()},
            "learned": r["learned"]}


def _q(a: np.ndarray, step: int) -> dict:
    p = np.percentile(a, [10, 50, 90], axis=0)
    return {"p10": p[0][::step].round(0).tolist(), "p50": p[1][::step].round(0).tolist(),
            "p90": p[2][::step].round(0).tolist()}


def fan(persona_id: str, *, month: int, age: float, status: str, work_share: float, income: float, rent: float,
        liquid: float, has_master: bool, abroad: bool, program_months_left: int | None, horizon_months: int,
        inflation: float = 0.01, months_to_degree: int = 12, n_paths: int = 300, seed: int = 0) -> dict:
    L = ledger()
    row = L.people.loc[persona_id]
    state = GAME_TO_STATE[status]
    # Reduced hours stay "employed" at the current (reduced) pay: the HMM's part_time state treats its income as
    # a fraction of a higher full-time wage, which would invent a raise when the chain returns to full time.
    T = max(1, int(horizon_months))
    first_month = month % 12 + 1
    ivs: list[Intervention] = []
    assumed_degree = None
    if state == "student" and program_months_left:
        k = int(program_months_left)
        ivs += [Intervention("force_state", 0, k, state="student"),
                Intervention("set_attribute", k, attribute=("has_masters", True))]
    elif state == "student":
        # In the HMM "student" only ends by a decision. An enrolled student without a programme end is assumed
        # to finish the degree and start working after `months_to_degree` (a labelled assumption).
        assumed_degree = int(months_to_degree)
        ivs += [Intervention("force_state", 0, assumed_degree, state="student"),
                Intervention("force_state", assumed_degree, state="employed")]
    ref = income if income > 0 else float(row.income)
    prof = profile_for(row, age=int(age), opening_cash=liquid, employment=state, ref_income=ref, rent=rent,
                       start_month_of_year=first_month, attributes={"has_masters": has_master, "abroad": abroad})
    paths = hmm().simulate(prof, horizon_months=T, n_paths=n_paths, interventions=ivs, seed=seed)
    beh = behaviour()
    # How this persona's real 24 months differ from people like them (only while in that same life state).
    home = STATE_INDEX[state]
    data_state = {"self_employed": "employed", "unemployed": "job_search"}.get(row.status, row.status)
    offset = 0.0
    if data_state == state:
        offset, _ = beh.history_offset(row.status, float(row.income), float(row.rent), L.flows[L.people.index.get_loc(persona_id)],
                                       L.month_of_year)
    # Spending follows rent in the data; rent rises with inflation (the HMM's wages are nominal too).
    rent_path = (rent * (1 + inflation) ** (np.arange(T) / 12))[None, :].repeat(n_paths, 0)
    flows = beh.flows(paths.states, paths.income_chf, rent_path, first_month,
                      self_employed=row.status == "self_employed", rng=np.random.default_rng(seed + 11),
                      offset=offset, home_state=home)
    wealth = liquid + np.cumsum(flows, 1)
    step = 1 if T <= 240 else 3
    shares = {s: (paths.states == k).mean(0)[::step].round(3).tolist() for k, s in enumerate(STATES)}
    js, ret = STATE_INDEX["job_search"], STATE_INDEX["retired"]
    to65 = int(max(0, min(T, round((65 - age) * 12))))
    spell = (paths.states[:, :to65] == js).any(1) if to65 else np.zeros(n_paths, bool)
    first_ret = np.where((paths.states == ret).any(1), (paths.states == ret).argmax(1), -1)
    at = lambda m_: wealth[:, min(T - 1, m_)] if m_ >= 0 else wealth[:, 0]  # noqa: E731
    return {
        "from_month": month, "step": step, "assumed_degree_after_months": assumed_degree, "n_paths": n_paths, "start_state": state, "offset_chf_per_month": round(offset),
        "wealth": _q(wealth, step), "states": shares,
        "summary": {
            "flow_now_p50": float(np.median(flows[:, 0])),
            "share_lives_with_job_search_before_65": float(spell.mean()) if to65 else None,
            "months_in_job_search_before_65_p50": float(np.median((paths.states[:, :to65] == js).sum(1))) if to65 else None,
            "retirement_age_p50": float(np.median(age + first_ret[first_ret >= 0] / 12)) if (first_ret >= 0).any() else None,
            "wealth_at_65": ({k: float(v) for k, v in zip(("p10", "p50", "p90"), np.percentile(at(to65 - 1), [10, 50, 90]))}
                             if to65 else None),
            "wealth_at_end": {k: float(v) for k, v in zip(("p10", "p50", "p90"), np.percentile(wealth[:, -1], [10, 50, 90]))},
            "share_lives_ever_below_zero": float((wealth < 0).any(1).mean()),
        },
        "basis": "liquid savings (cash + investments now, then learned monthly cash flows; no investment returns)",
        "note": ("HMM life states and incomes (sourced Swiss hazards, retirement at 65) x money behaviour learned from "
                 "958 training people (GMM spending per state, seasons, noise) and this persona's own history. "
                 "Ranges over simulated lives, not probabilities about you."),
    }
