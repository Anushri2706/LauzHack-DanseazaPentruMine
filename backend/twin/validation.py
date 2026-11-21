"""Validation checks to show judges (docs/hmm-digital-twin.md §3.5).

Deterministic equivalence with the starter kit, transition and occupancy checks,
the emission fit against the held-out profiles, and a 2025-filter / 2026-simulate
calibration backtest.
"""
from __future__ import annotations

import importlib.util
import json
from dataclasses import replace

import numpy as np

from . import cash_engine
from .cash_engine import OneOff
from .data import STARTER_KIT_DIR, prepare_data
from .regime_engine import RegimeEngine, analytic_occupancy, fit_emissions
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


def emission_fit_report(holdout_frac: float = 0.2, seed: int = 0, model=None) -> dict:
    """Per regime: Student-t parameters, KS statistic and QQ quantiles on the held-out profiles.

    The comparison is household-adjusted spend/rent, outliers and Decembers removed, so it
    tests the body that was fit. Self-employed is reported separately from employees.
    """
    if model is None:
        model = fit_emissions(family="student_t", holdout_frac=holdout_frac, seed=seed)
    diag = model.diagnostics
    by_state = {}
    for state, block in diag["by_state"].items():
        by_state[state] = {
            "n_train": block["n_train"],
            "n_holdout": block["n"],
            "median_spend_ratio": block["median_spend_ratio"],
            "df": block["df"],
            "loc": block["loc"],
            "scale": block["scale"],
            "within_sd": block["within_sd"],
            "ks_stat": block["ks_stat"],
            "ks_pvalue": block["ks_pvalue"],
            "qq_probabilities": block["qq_probabilities"],
            "qq_holdout": block["qq_holdout"],
            "qq_model": block["qq_model"],
        }
    return {
        "family": model.family,
        "by_state": by_state,
        "self_employed": diag["self_employed"],
        "self_employed_scale_ratio": diag["self_employed_scale_ratio"],
        "self_employed_within_sd_ratio": diag["self_employed_within_sd_ratio"],
        "household_shift": diag["household_shift"],
        "december_bonus_median": diag["december_bonus_median"],
        "holdout_profiles": {s: len(v) for s, v in diag["holdout_ids"].items()},
    }


def calibration_backtest(level: float = 0.9, n_paths: int = 250, seed: int = 0, model=None,
                         prepared=None) -> dict:
    """Filter each held-out profile on 2025, simulate 2026, and score the real closing cash.

    The ledger has no regime switches, so the simulation uses A = I and no retirement.
    The emission is shrunk to that profile's 2025 non-December spend ratios (kappa = 6).
    ``level`` is the central band (0.90 -> P5 to P95). The target coverage is ``level``.
    """
    if prepared is None:
        prepared = prepare_data()
    if model is None:
        model = fit_emissions(prepared=prepared)
    lo = (1.0 - level) / 2.0
    hi = 1.0 - lo
    pm = prepared.person_months
    ids = [pid for group in prepared.holdout_ids.values() for pid in group]
    priors = TransitionPriors(
        hazards=np.zeros((K, K)),
        concentration=np.full(K, 1e6),
        names={},
        age_bounds={},
        table=None,
        age_hazards=[],
        age_names={},
    )
    engine = RegimeEngine(model, priors, mode="hmm")
    flat = {"annual_experience_growth": 0.0, "experience_growth_30_40": 0.0,
            "experience_growth_40_50": 0.0, "experience_growth_50_plus": 0.0,
            "retirement_age": 200.0, "dirichlet": 0.0}
    inside = []
    widths = []
    filter_hits = []
    by_state: dict[str, list[tuple[bool, float]]] = {}
    for i, pid in enumerate(ids):
        g = pm.loc[pm["profile_id"] == pid].sort_values("t")
        if len(g) < 24:
            continue
        row = g.iloc[0]
        state = str(row["state"])
        flows = g["net_transactions_chf"].to_numpy(dtype=float)
        y2025 = g.loc[(g["t"] < 12) & ~g["is_december"] & ~g["is_outlier"], "spend_ratio"].to_numpy()
        household = row["household_type"]
        se = str(row["employment_status"]) == "self_employed"
        personal = model.personalize(state, y2025, household=household, self_employed=se)
        opening = float(g.loc[g["t"] == 11, "closing_cash_chf"].iloc[0])
        actual = float(g.loc[g["t"] == 23, "closing_cash_chf"].iloc[0])
        profile = Profile(
            age=int(row["age_at_start"]),
            employment=str(row["employment_status"]),
            ref_income=float(row["reference_monthly_net_income_chf"]),
            rent=float(row["reference_monthly_rent_chf"]),
            attributes={"household_type": household, "has_masters": False, "abroad": False,
                        "canton": row["canton"]},
            profile_id=pid,
            opening_cash=opening,
            start_month_of_year=1,
        )
        engine.emissions = model
        belief = engine.filter(profile, flows[:12].tolist(), flat)
        filter_hits.append(max(belief, key=belief.get) == state)
        engine.emissions = personal
        future = replace(profile, age=profile.age + 1)
        paths = engine.simulate(future, horizon_months=12, n_paths=n_paths, assumption_overrides=flat,
                                initial_belief=belief, seed=seed + i)
        cash = opening + np.cumsum(paths.net_flow_chf, axis=1)[:, -1]
        p_lo, p_hi = np.quantile(cash, [lo, hi])
        hit = bool(p_lo <= actual <= p_hi)
        width = float(p_hi - p_lo)
        inside.append(hit)
        widths.append(width)
        by_state.setdefault(state, []).append((hit, width))
    def _pack(rows: list[tuple[bool, float]]) -> dict:
        if not rows:
            return {"n": 0, "coverage": None, "median_band_width_chf": None}
        return {"n": len(rows), "coverage": float(np.mean([h for h, _ in rows])),
                "median_band_width_chf": float(np.median([w for _, w in rows]))}
    return {
        "level": level,
        "n_profiles": len(inside),
        "n_paths": n_paths,
        "coverage": float(np.mean(inside)) if inside else None,
        "median_band_width_chf": float(np.median(widths)) if widths else None,
        "mean_band_width_chf": float(np.mean(widths)) if widths else None,
        "by_state": {s: _pack(rows) for s, rows in by_state.items()},
        "filter_accuracy": float(np.mean(filter_hits)) if filter_hits else None,
        "personalised": True,
        "transitions": "A = I and retirement off; the ledger has no regime changes",
    }


def simulated_vs_analytic_occupancy(paths: RegimePaths, pi0: np.ndarray, A: np.ndarray) -> float:
    """Max abs gap between simulated occupancy and pi0 A^t (should shrink as 1/sqrt(n))."""
    return float(np.abs(paths.occupancy - analytic_occupancy(pi0, A, paths.horizon)).max())
