"""Helpers the town game's /hmm bridge needs on top of the combined engine.

`goal_hit_rate`, `apply_fractional_drop` and `load_profile` are copied unchanged from the earlier
`backend/twin` modules (metrics.py, cash_engine.py, data.py); the combined engine did not carry them.
"""
from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

from .accounting import CashPaths
from .data import DATA_DIR, load_cash_balances, load_profiles
from pathlib import Path
from .types import Profile

DISCLAIMER = "Simulated under the stated assumptions; not a forecast or a real-world probability."


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




def apply_fractional_drop(cp: CashPaths, month: int, fraction: float) -> tuple[CashPaths, np.ndarray]:
    """Drop ``fraction`` of each path's end-of-month cash at ``month``.

    The balance used is the one before this drop. That amount is a one-off cost in
    ``month``, and every later month keeps the same reduction. A non-positive balance
    drops nothing: a deposit cannot be negative.
    """
    if cp.cash.ndim != 2:
        raise ValueError("cash must be [n_paths, T]")
    n, T = cp.cash.shape
    if not 0 <= int(month) < T:
        raise ValueError(f"month {month} outside horizon")
    if fraction < 0:
        raise ValueError("fraction cannot be negative")
    deposit = float(fraction) * np.maximum(cp.cash[:, int(month)], 0.0)
    cash = np.array(cp.cash, dtype=float, copy=True)
    cash[:, int(month):] -= deposit[:, None]
    flows = {key: np.array(value, dtype=float, copy=True) for key, value in cp.flows.items()}
    if flows["one_off_costs_chf"].shape != (n, T):
        flows["one_off_costs_chf"] = np.broadcast_to(flows["one_off_costs_chf"], (n, T)).copy()
    flows["one_off_costs_chf"][:, int(month)] += deposit
    return CashPaths(cp.opening_cash_chf, cash, flows), deposit


def load_profile(profile_id: str = "P00001", data_dir: Path = DATA_DIR) -> Profile:
    """Build a ``Profile`` from the synthetic data (default: Alexia, P00001)."""
    pr = load_profiles(data_dir)
    row = pr.loc[pr["profile_id"] == profile_id]
    if row.empty:
        raise KeyError(f"Unknown profile_id {profile_id}")
    r = row.iloc[0]
    cb = load_cash_balances(data_dir)
    flows = cb.loc[cb["profile_id"] == profile_id, "net_transactions_chf"]
    ref_income = float(r["reference_monthly_net_income_chf"])
    mean_rate = float(flows.mean() / ref_income) if len(flows) and ref_income > 0 else None
    return Profile(
        age=int(r["age_at_start"]),
        employment=str(r["employment_status"]),
        ref_income=ref_income,
        rent=float(r["reference_monthly_rent_chf"]),
        attributes={"has_masters": False, "abroad": False, "canton": r["canton"],
                    "household_type": r["household_type"]},
        profile_id=profile_id,
        name=str(r["display_name"]),
        opening_cash=float(r["opening_cash_chf"]),
        mean_savings_rate=mean_rate,
    )
