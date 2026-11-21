"""Metrics over simulated cash and regime paths.

Percentiles are of simulated outcomes under stated assumptions, not real-world probabilities
(evaluation/responsible-ai-and-data.md). Every summary carries that label.
"""
from __future__ import annotations

import numpy as np

from .cash_engine import CashPaths
from .types import STATES, RegimePaths

DEFAULT_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
DISCLAIMER = "Simulated under the stated assumptions; not a forecast or a real-world probability."


def fan_quantiles(cash: CashPaths, qs=DEFAULT_QUANTILES) -> dict[str, np.ndarray]:
    """{"p05": [T], ...} cash quantiles per month."""
    q = np.quantile(cash.cash, qs, axis=0)
    return {f"p{int(round(x * 100)):02d}": q[i] for i, x in enumerate(qs)}


def prob_below(cash: CashPaths, threshold_chf: float) -> np.ndarray:
    """[T] share of paths with cash < threshold in each month."""
    return (cash.cash < threshold_chf).mean(axis=0)


def prob_ever_below(cash: CashPaths, threshold_chf: float) -> float:
    return float((cash.cash < threshold_chf).any(axis=1).mean())


def prob_goal_met(cash: CashPaths, amount_chf: float, month: int) -> float:
    """Share of paths with cash >= amount at ``month`` (clipped to the horizon)."""
    m = min(max(month, 0), cash.cash.shape[1] - 1)
    return float((cash.cash[:, m] >= amount_chf).mean())


def first_negative_summary(cash: CashPaths) -> dict:
    fn = cash.first_negative_month
    hit = fn[fn >= 0]
    return {"share_paths_ever_negative": float((fn >= 0).mean()),
            "median_first_negative_month": int(np.median(hit)) if hit.size else None}


def expected_months_in_state(paths: RegimePaths) -> dict[str, float]:
    return {s: float((paths.states == k).sum(axis=1).mean()) for k, s in enumerate(STATES)}


def compute_metrics(cash: CashPaths, paths: RegimePaths, reserve_chf: float = 0.0,
                    goals: list[dict] | None = None) -> dict:
    """Full metric bundle. ``goals``: [{"name", "amount_chf", "month"}]."""
    fan = fan_quantiles(cash)
    out = {
        "fan": fan,
        "closing": {k: float(v[-1]) for k, v in fan.items()},
        "minimum_cash_p05": float(np.quantile(cash.minimum, 0.05)),
        "minimum_cash_p50": float(np.quantile(cash.minimum, 0.5)),
        "p_below_reserve_by_month": prob_below(cash, reserve_chf),
        "p_ever_below_reserve": prob_ever_below(cash, reserve_chf),
        "reserve_chf": reserve_chf,
        **first_negative_summary(cash),
        "expected_months_in_state": expected_months_in_state(paths),
        "occupancy": paths.occupancy,
        "goals": [{**g, "p_met": prob_goal_met(cash, g["amount_chf"], g["month"])} for g in goals or []],
        "n_paths": paths.n_paths,
        "horizon_months": paths.horizon,
        "disclaimer": DISCLAIMER,
    }
    return out
