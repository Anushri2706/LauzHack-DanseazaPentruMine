"""Metrics over simulated cash and regime paths.

Percentiles are of simulated outcomes under stated assumptions, not real-world probabilities
(evaluation/responsible-ai-and-data.md). Every summary carries that label.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime

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


def _year_month_day(value) -> tuple[int, int, int | None]:
    """Year, month, and day (or None when the text is only YYYY-MM)."""
    if isinstance(value, datetime):
        return value.year, value.month, value.day
    if isinstance(value, date):
        return value.year, value.month, value.day
    text = str(value).strip().split()[0]
    parts = text.split("-")
    if len(parts) < 2:
        raise ValueError(f"Cannot read a month from {value!r}")
    year, month = int(parts[0]), int(parts[1])
    if not 1 <= month <= 12:
        raise ValueError(f"Cannot read a month from {value!r}")
    day = int(parts[2]) if len(parts) >= 3 and parts[2] else None
    return year, month, day


def _shift_month(year: int, month: int, offset: int) -> str:
    total = year * 12 + (month - 1) + int(offset)
    return f"{total // 12:04d}-{total % 12 + 1:02d}"


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


def _goal_mapping(goal) -> dict:
    """One CSV goal row, a mapping, or an ``(amount, date)`` pair."""
    if isinstance(goal, tuple) and len(goal) == 2 and not hasattr(goal, "_fields"):
        amount, when = goal
        return {"target_amount_chf": amount, "target_date": when}
    if hasattr(goal, "columns") and hasattr(goal, "iloc"):
        raise TypeError("pass one goal row from the CSV, not the whole goals table")
    if hasattr(goal, "to_dict") and not isinstance(goal, Mapping):
        return dict(goal.to_dict())
    if hasattr(goal, "_asdict"):
        return dict(goal._asdict())
    if isinstance(goal, Mapping):
        return dict(goal)
    raise TypeError("goal must be a CSV row, a mapping with amount and date, or (amount, date)")


def _goal_field(goal: dict, *names: str):
    for name in names:
        if name in goal and not _is_missing(goal[name]):
            return goal[name]
    return None


def goal_hit_rate(cash: np.ndarray | CashPaths, start, goal) -> dict:
    """Share of cash paths that meet a savings goal, and when the median stays above it.

    ``cash`` has shape ``(n_paths, n_months)`` (a ``CashPaths`` is accepted too).
    Month 0 is ``start`` (``"2025-01"`` or a date). ``goal`` is one row of
    ``Data/synthetic-goals.csv`` (a mapping or a named row) with
    ``target_amount_chf`` and ``target_date``. ``amount_chf`` / ``amount`` and
    ``date`` are accepted under those names. A pair ``(amount, date)`` works too.

    Let ``A`` be the target amount and ``T`` the number of simulated months. The
    goal's month index, with the day ignored, is

        g = (Y_goal - Y_start) * 12 + (M_goal - M_start).

    The cash month ``m`` is ``g`` when ``0 <= g < T`` (``evaluated_at`` is
    ``"goal_month"``). If the horizon ends first (``g >= T``), ``m = T - 1`` and
    ``evaluated_at`` is ``"last_simulated_month"``. If the goal date is before
    ``start``, ``m = 0`` and ``evaluated_at`` is ``"before_start"``.

        hit_rate = (1 / N) * count(C[:, m] >= A)

    The median path is the cross-section ``median_t = median_i C[i, t]``, not one
    sampled path. ``median_cross_month`` is the smallest ``t`` such that
    ``median_s >= A`` for every ``s`` from ``t`` through ``T - 1``, or null when
    ``median_{T-1} < A``.

        median_cash_at_goal = quantile(C[:, m], 0.50)
        p10_cash_at_goal    = quantile(C[:, m], 0.10)
        p90_cash_at_goal    = quantile(C[:, m], 0.90)

    Quantiles use NumPy's linear method. The three cash figures are taken at
    ``m``, the same month as ``hit_rate``.
    """
    if isinstance(cash, CashPaths):
        values = np.asarray(cash.cash, dtype=float)
    else:
        values = np.asarray(cash, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"cash must have shape (n_paths, n_months), got {values.shape}")
    n_paths, horizon = values.shape
    if n_paths == 0 or horizon == 0:
        raise ValueError("cash must contain at least one path and one month")

    fields = _goal_mapping(goal)
    amount_raw = _goal_field(fields, "target_amount_chf", "amount_chf", "amount")
    when_raw = _goal_field(fields, "target_date", "date")
    if amount_raw is None or when_raw is None:
        raise KeyError("goal needs an amount (target_amount_chf) and a date (target_date)")
    amount = float(amount_raw)
    start_y, start_m, _start_d = _year_month_day(start)
    goal_y, goal_m, goal_d = _year_month_day(when_raw)
    goal_month = (goal_y - start_y) * 12 + (goal_m - start_m)
    if goal_month < 0:
        evaluated = 0
        evaluated_at = "before_start"
    elif goal_month >= horizon:
        evaluated = horizon - 1
        evaluated_at = "last_simulated_month"
    else:
        evaluated = goal_month
        evaluated_at = "goal_month"

    at_goal = values[:, evaluated]
    hit_rate = float(np.mean(at_goal >= amount))
    p10, median_cash, p90 = (float(v) for v in np.quantile(at_goal, [0.10, 0.50, 0.90]))
    median_path = np.median(values, axis=0)
    above = median_path >= amount
    if bool(above[-1]):
        below = np.flatnonzero(~above)
        cross = int(below[-1] + 1) if below.size else 0
    else:
        cross = None

    target_date = f"{goal_y:04d}-{goal_m:02d}" if goal_d is None else f"{goal_y:04d}-{goal_m:02d}-{goal_d:02d}"
    name = _goal_field(fields, "goal_name", "name")
    priority = _goal_field(fields, "priority")
    return {
        "goal_name": None if name is None else str(name),
        "priority": None if priority is None else str(priority),
        "target_amount_chf": amount,
        "target_date": target_date,
        "start": f"{start_y:04d}-{start_m:02d}",
        "goal_month": int(goal_month),
        "evaluated_month": int(evaluated),
        "evaluated_at": evaluated_at,
        "evaluated_label": _shift_month(start_y, start_m, evaluated),
        "hit_rate": hit_rate,
        "median_cross_month": cross,
        "median_cross_label": None if cross is None else _shift_month(start_y, start_m, cross),
        "median_cash_at_goal": median_cash,
        "p10_cash_at_goal": p10,
        "p90_cash_at_goal": p90,
        "n_paths": int(n_paths),
        "horizon_months": int(horizon),
        "disclaimer": DISCLAIMER,
    }


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
