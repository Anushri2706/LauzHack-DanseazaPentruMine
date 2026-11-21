"""Validation checks to show judges (docs/hmm-digital-twin.md §3.5).

Implemented now: deterministic equivalence with the starter kit, analytic occupancy rows,
transition-matrix sanity, cash reconciliation. TODO(Victor): emission fit and calibration.
"""
from __future__ import annotations

import importlib.util
import json

import numpy as np

from . import cash_engine
from .cash_engine import OneOff
from .data import STARTER_KIT_DIR
from .regime_engine import analytic_occupancy
from .transitions import TransitionPriors
from .types import K, Profile, RegimePaths


def _starter_kit():
    spec = importlib.util.spec_from_file_location("financial_twin", STARTER_KIT_DIR / "financial_twin.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def paths_from_starter_scenario(scenario: dict, horizon: int) -> tuple[RegimePaths, list[OneOff]]:
    """Degenerate RegimePaths (1 path, A = I, no noise) encoding a starter-kit phase scenario."""
    income = np.zeros(horizon)
    expenses = np.zeros(horizon)
    for ph in scenario["phases"]:
        income[ph["start_month"]:ph["end_month"]] = ph["monthly_income_chf"]
        expenses[ph["start_month"]:ph["end_month"]] = ph["monthly_expenses_chf"]
    rate = np.where(income != 0, (income - expenses) / np.where(income != 0, income, 1), 0.0)
    net = income - expenses
    paths = RegimePaths(states=np.zeros((1, horizon), dtype=int), income_chf=income[None], savings_rate=rate[None],
                        net_flow_chf=net[None], occupancy=np.tile(np.eye(K)[0], (horizon, 1)),
                        transition_draws=np.eye(K)[None])
    offs = [OneOff(o["month"], o["amount_chf"]) for o in scenario.get("one_off_costs", [])]
    return paths, offs


def deterministic_equivalence(tol: float = 0.01) -> list[dict]:
    """Noise off, A = I: our cash engine must reproduce starter-kit output for every scenario."""
    ft = _starter_kit()
    profile_json = json.loads((STARTER_KIT_DIR / "sample_profile.json").read_text())
    config = json.loads((STARTER_KIT_DIR / "scenario_config.json").read_text())
    T = profile_json["horizon_months"]
    profile = Profile(age=profile_json["age_at_start"], employment="student", ref_income=0.0, rent=0.0,
                      opening_cash=profile_json["opening_cash_chf"])
    out = []
    for sc in config["scenarios"]:
        ref = ft.simulate(profile_json, sc)
        paths, offs = paths_from_starter_scenario(sc, T)
        cp = cash_engine.run(profile, paths, offs)
        ours = cp.cash[0]
        fn = int(cp.first_negative_month[0])
        out.append({
            "scenario": sc["name"],
            "max_abs_diff_chf": float(np.max(np.abs(ours - np.array(ref.cash_by_month_chf)))),
            "closing_ours": float(ours[-1]), "closing_starter": ref.closing_cash_chf,
            "min_ours": float(cp.minimum[0]), "min_starter": ref.minimum_cash_chf,
            "first_negative_ours": None if fn < 0 else fn, "first_negative_starter": ref.first_negative_month,
        })
        r = out[-1]
        r["ok"] = (r["max_abs_diff_chf"] <= tol and abs(r["min_ours"] - r["min_starter"]) <= tol
                   and r["first_negative_ours"] == r["first_negative_starter"])
    return out


def check_transition_matrix(priors: TransitionPriors, overrides: dict | None = None) -> dict:
    A = priors.matrix(overrides)
    rng = np.random.default_rng(0)
    draws = priors.sample(rng, 200, overrides)
    return {"rows_sum_to_one": bool(np.allclose(A.sum(1), 1.0)), "non_negative": bool((A >= 0).all()),
            "draws_rows_sum_to_one": bool(np.allclose(draws.sum(2), 1.0)),
            "draw_mean_close_to_A": float(np.abs(draws.mean(0) - A).max())}


def check_occupancy(priors: TransitionPriors, pi0: np.ndarray, horizon: int = 60) -> dict:
    occ = analytic_occupancy(pi0, priors.matrix(), horizon)
    return {"rows_sum_to_one": bool(np.allclose(occ.sum(1), 1.0)), "final": occ[-1].round(4).tolist()}


def emission_fit_report(holdout_frac: float = 0.2, seed: int = 0) -> dict:
    """TODO(Victor): per regime, KS statistic and QQ quantiles of simulated vs held-out ratios."""
    raise NotImplementedError


def calibration_backtest(level: float = 0.9) -> dict:
    """TODO(Victor): filter on 2025, simulate 2026 per held-out profile; share of real 2026
    closing cash inside the central ``level`` band (target ≈ level). Also report band width."""
    raise NotImplementedError


def simulated_vs_analytic_occupancy(paths: RegimePaths, pi0: np.ndarray, A: np.ndarray) -> float:
    """Max abs gap between simulated occupancy and pi0 A^t (should shrink as 1/sqrt(n))."""
    return float(np.abs(paths.occupancy - analytic_occupancy(pi0, A, paths.horizon)).max())
