"""Monte Carlo result contract: distributions and risk metrics from a ``SimulationRun``.

``summarise`` returns a JSON-serialisable dict. ``scalar_facts`` flattens the headline numbers into
stable fact ids (e.g. ``net_worth_real.final.p50``) that comparisons and the AI claim guard cite.
Floats are rounded for transport only (CHF to 0.01, probabilities to 1e-4); the frontend and the AI
never recompute them.
"""
from __future__ import annotations

import numpy as np

from .metrics import METRICS, PERCENTILE_NOTE
from .simulate import SimulationRun
from .types import STATES

QUANTILES = (0.10, 0.50, 0.90)
MIN_PATHS_FOR_PERCENTILES = 20
DISCLAIMER = ("Simulated outcomes conditional on the stated assumptions and seed; not a prediction, "
              "a guarantee or financial advice.")


class ResultError(ValueError):
    pass


def _bands(x: np.ndarray) -> dict[str, list[float]]:
    q = np.quantile(x, QUANTILES, axis=0)
    if not (np.all(q[0] <= q[1] + 1e-9) and np.all(q[1] <= q[2] + 1e-9)):
        raise ResultError("percentiles are not ordered")  # pragma: no cover - quantiles are monotone
    return {"p10": [round(float(v), 2) for v in q[0]], "p50": [round(float(v), 2) for v in q[1]],
            "p90": [round(float(v), 2) for v in q[2]]}


def _final(x: np.ndarray) -> dict[str, float]:
    q = np.quantile(x, QUANTILES)
    return {"p10": round(float(q[0]), 2), "p50": round(float(q[1]), 2), "p90": round(float(q[2]), 2)}


def summarise(run: SimulationRun) -> dict:
    n, T = run.n_paths, run.horizon
    warnings = list(run.warnings)
    if n < MIN_PATHS_FOR_PERCENTILES:
        warnings.append(f"Only {n} simulated paths: percentiles are very noisy.")
    cash, nw, nwr = run.cash, run.net_worth_nominal, run.net_worth_real
    neg = cash < 0
    below = cash < run.reserve_chf
    months_below = below.sum(axis=1)
    out = {
        "n_paths": int(n),
        "horizon_months": int(T),
        "seed": run.seed,
        "assumptions_version": run.assumptions_version,
        "config": run.config.to_dict(),
        "reserve_chf": round(run.reserve_chf, 2),
        "opening": {"cash_chf": run.opening_cash, "portfolio_chf": run.opening_portfolio, "debt_chf": run.debt,
                    "net_worth_chf": round(run.opening_cash + run.opening_portfolio - run.debt, 2)},
        "monthly": {
            "cash": _bands(cash),
            "net_worth_nominal": _bands(nw),
            "net_worth_real": _bands(nwr),
            "p_cash_below_zero": [round(float(v), 4) for v in neg.mean(axis=0)],
            "p_below_reserve": [round(float(v), 4) for v in below.mean(axis=0)],
        },
        "final": {
            "cash": _final(cash[:, -1]),
            "net_worth_nominal": _final(nw[:, -1]),
            "net_worth_real": _final(nwr[:, -1]),
        },
        "risk": {
            "p_cash_ever_below_zero": round(float(neg.any(axis=1).mean()), 4),
            "p_ever_below_reserve": round(float(below.any(axis=1).mean()), 4),
            "months_below_reserve": {**_final(months_below), "mean": round(float(months_below.mean()), 2)},
        },
        "expected_months_in_state": {s: round(float((run.states == k).sum(axis=1).mean()), 2)
                                     for k, s in enumerate(STATES)},
        "events": {k: {"mean_occurrences": round(float(v.mean()), 3)} for k, v in run.event_counts.items()},
        "metric_definitions": METRICS,
        "percentile_note": PERCENTILE_NOTE,
        "warnings": warnings,
        "notes": run.notes,
        "disclaimer": DISCLAIMER,
        "reconciles": run.reconciles(),
        "fingerprint": run.fingerprint(),
    }
    return out


def scalar_facts(summary: dict) -> dict[str, dict]:
    """Flat, citable facts: id -> {value, unit, basis, horizon_months, metric}."""
    T = summary["horizon_months"]
    facts: dict[str, dict] = {}
    for metric in ("cash", "net_worth_nominal", "net_worth_real"):
        md = METRICS[metric]
        for stat, v in summary["final"][metric].items():
            facts[f"{metric}.final.{stat}"] = {"value": v, "unit": "CHF", "basis": md["basis"],
                                               "horizon_months": T, "metric": metric, "stat": stat}
    for key in ("p_cash_ever_below_zero", "p_ever_below_reserve"):
        facts[f"{key}"] = {"value": summary["risk"][key], "unit": "probability", "basis": "share of paths",
                           "horizon_months": T, "metric": key, "stat": "share"}
    for stat, v in summary["risk"]["months_below_reserve"].items():
        facts[f"months_below_reserve.{stat}"] = {"value": v, "unit": "months", "basis": "count per path",
                                                 "horizon_months": T, "metric": "months_below_reserve", "stat": stat}
    facts["reserve_chf"] = {"value": summary["reserve_chf"], "unit": "CHF", "basis": "nominal",
                            "horizon_months": T, "metric": "reserve_chf", "stat": "input"}
    return facts


# Which direction is "better" is NOT asserted; comparisons only state higher/lower.
COMPARABLE = ("cash.final.p10", "cash.final.p50", "cash.final.p90", "net_worth_nominal.final.p50",
              "net_worth_real.final.p10", "net_worth_real.final.p50", "net_worth_real.final.p90",
              "p_cash_ever_below_zero", "p_ever_below_reserve", "months_below_reserve.p50",
              "months_below_reserve.mean")


def compare_summaries(a: dict, b: dict) -> dict:
    """Deterministic differences b - a on matching facts. Requires compatible runs."""
    problems = []
    for key in ("horizon_months", "n_paths", "seed"):
        if a[key] != b[key]:
            problems.append(f"{key} differs ({a[key]} vs {b[key]})")
    if a["opening"] != b["opening"]:
        problems.append("initial conditions differ")
    if a["reserve_chf"] != b["reserve_chf"]:
        problems.append("emergency reserve differs")
    if problems:
        raise ResultError("branches are not comparable: " + "; ".join(problems))
    fa, fb = scalar_facts(a), scalar_facts(b)
    rows = {}
    for k in COMPARABLE:
        va, vb = fa[k]["value"], fb[k]["value"]
        d = vb - va
        unit = fa[k]["unit"]
        tol = 0.5 if unit == "CHF" else 1e-9
        rows[k] = {"a": va, "b": vb, "difference_b_minus_a": round(d, 4 if unit == "probability" else 2),
                   "direction": "equal" if abs(d) <= tol else ("b_higher" if d > 0 else "b_lower"),
                   "unit": unit, "basis": fa[k]["basis"], "horizon_months": a["horizon_months"]}
    return {"differences": rows,
            "note": "Differences of summary statistics (e.g. difference of medians, not the median of "
                    "path-wise differences), under the stated assumptions. Not a proven causal effect."}
