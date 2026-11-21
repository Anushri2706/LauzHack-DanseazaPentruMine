"""One-at-a-time sensitivity (+/-10%) and break-even search.

Sensitivity is not optimisation: the NSGA-II trade-off search (former backend/regime/explore.py, commit 33470ff) is a separate
tool and was not migrated (see docs/AUDIT.md).

Noise control: every evaluation uses the SAME seed as the baseline (common random numbers), so a
difference between a perturbed and a baseline run comes from the assumption, plus the HMM-stream
realignment described in engine/simulate.py, not from fresh Monte Carlo noise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from .assumptions import AssumptionError, AssumptionSet
from .montecarlo import scalar_facts

DEFAULT_IDS = ("entry_salary_chf", "annual_experience_growth", "job_loss_hazard", "job_search_exit_rate",
               "masters_premium_pct", "ui_replacement_rate", "portfolio_return", "inflation_base",
               "student_income_chf", "employed_rent_chf")


def perturbed_values(assumptions: AssumptionSet, aid: str, rel: float) -> tuple[float, float, list[str]]:
    """(low, high, warnings). Probabilities/rates are clamped to their bounds and the clamp is reported."""
    a = assumptions.registry.get(aid)
    if not a.sensitivity_eligible:
        raise AssumptionError("not_eligible", aid, f"{aid} ({a.kind}) is not a continuous assumption")
    v0 = assumptions.value(aid)
    lo, hi = v0 * (1 - rel), v0 * (1 + rel)
    warnings = []
    if a.upper is not None and hi > a.upper:
        warnings.append(f"{aid}: +{rel:.0%} clamped to {a.upper}")
        hi = a.upper
    if a.lower is not None and lo < a.lower:
        warnings.append(f"{aid}: -{rel:.0%} clamped to {a.lower}")
        lo = a.lower
    return lo, hi, warnings


def one_at_a_time(assumptions: AssumptionSet, evaluate: Callable[[AssumptionSet], dict],
                  ids: list[str] | None = None, rel: float = 0.10,
                  metric: str = "net_worth_real.final.p50") -> dict:
    """``evaluate(assumption_set) -> summary`` runs the branch with a fixed seed."""
    if not 0 < rel < 1:
        raise ValueError("rel must be in (0, 1)")
    base = evaluate(assumptions)
    facts = scalar_facts(base)
    if metric not in facts:
        raise ValueError(f"unknown metric {metric!r}")
    b = facts[metric]["value"]
    rows, skipped = [], []
    for aid in ids or DEFAULT_IDS:
        a = assumptions.registry.get(aid)
        if not a.sensitivity_eligible:
            skipped.append({"assumption_id": aid, "reason": f"{a.kind} assumptions are not perturbed"})
            continue
        lo, hi, warns = perturbed_values(assumptions, aid, rel)
        out = {}
        for label, v in (("low", lo), ("high", hi)):
            try:
                s = evaluate(assumptions.with_overrides({aid: v}))
            except AssumptionError as e:     # e.g. a transition row would no longer be stochastic
                skipped.append({"assumption_id": aid, "reason": str(e)})
                out = None
                break
            val = scalar_facts(s)[metric]["value"]
            out[label] = {"value": v, "outcome": val, "difference": round(val - b, 2)}
        if out is None:
            continue
        swing = out["high"]["outcome"] - out["low"]["outcome"]
        rows.append({
            "assumption_id": aid, "baseline_value": assumptions.value(aid), "unit": a.unit, **out,
            "swing": round(abs(swing), 2),
            "direction": "increases" if swing > 0 else ("decreases" if swing < 0 else "no_effect"),
            "source_status": a.source_status, "source_name": a.source_name, "warnings": warns,
        })
    rows.sort(key=lambda r: r["swing"], reverse=True)
    return {"metric": metric, "metric_unit": facts[metric]["unit"], "baseline_outcome": b, "rel_change": rel,
            "rows": rows, "skipped": skipped,
            "noise_control": "common random numbers: every run uses the baseline seed",
            "note": "One-at-a-time: interactions between assumptions are not captured."}


@dataclass
class BreakEven:
    status: str            # converged | no_break_even_in_range | undefined | max_iterations
    value: float | None
    lower: float
    upper: float
    f_lower: float | None
    f_upper: float | None
    iterations: int
    tolerance: float
    evaluations: list[tuple[float, float]]

    def to_dict(self) -> dict:
        return {**self.__dict__, "evaluations": [{"x": x, "f": f} for x, f in self.evaluations]}


def bisect(f: Callable[[float], float], lower: float, upper: float, tol: float = 1e-3,
           max_iter: int = 40) -> BreakEven:
    """Find x in [lower, upper] with f(x) = 0. Requires a sign change at the endpoints; never extrapolates."""
    if not (math.isfinite(lower) and math.isfinite(upper)) or lower >= upper:
        raise ValueError("need finite lower < upper")
    evals: list[tuple[float, float]] = []

    def call(x):
        y = float(f(x))
        evals.append((x, y))
        return y

    fl, fu = call(lower), call(upper)
    if not (math.isfinite(fl) and math.isfinite(fu)):
        return BreakEven("undefined", None, lower, upper, fl, fu, 0, tol, evals)
    if fl == 0:
        return BreakEven("converged", lower, lower, upper, fl, fu, 0, tol, evals)
    if fu == 0:
        return BreakEven("converged", upper, lower, upper, fl, fu, 0, tol, evals)
    if (fl > 0) == (fu > 0):
        return BreakEven("no_break_even_in_range", None, lower, upper, fl, fu, 0, tol, evals)
    a, b, fa = lower, upper, fl
    for i in range(1, max_iter + 1):
        m = 0.5 * (a + b)
        fm = call(m)
        if not math.isfinite(fm):
            return BreakEven("undefined", None, lower, upper, fl, fu, i, tol, evals)
        if fm == 0 or (b - a) / 2 < tol:
            return BreakEven("converged", m, lower, upper, fl, fu, i, tol, evals)
        if (fm > 0) == (fa > 0):
            a, fa = m, fm
        else:
            b = m
    return BreakEven("max_iterations", 0.5 * (a + b), lower, upper, fl, fu, max_iter, tol, evals)
