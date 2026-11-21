"""Gross pay -> what lands in the account (assumptions in assumptions/tax_assumptions.csv).

Two kinds of income are taxed here:
- "wage" (employed, part-time, parental leave): AHV/IV/EO, ALV up to the ceiling, non-occupational
  accident insurance, the employee half of the BVG age credit, and income tax.
- "ui" (unemployment benefits): AHV/IV/EO and accident insurance are deducted, then income tax.
Student stipends, the synthetic profiles' own net incomes and pensions are not taxed again.

Income tax is one effective rate on gross annual pay, interpolated between the tax_rate_at_<CHF>
points. Monthly amounts are annualised x12 (the December bonus is handled elsewhere).
"""
from __future__ import annotations

import numpy as np

BVG_BANDS = ((25, 35, "bvg_credit_25_34"), (35, 45, "bvg_credit_35_44"),
             (45, 55, "bvg_credit_45_54"), (55, 65, "bvg_credit_55_64"))


def enabled(params: dict) -> bool:
    return float(params.get("apply_income_tax", 0.0)) >= 0.5


def tax_curve(params: dict) -> tuple[np.ndarray, np.ndarray]:
    pts = sorted((float(k[len("tax_rate_at_"):]), float(v)) for k, v in params.items()
                 if k.startswith("tax_rate_at_"))
    if not pts:
        return np.array([0.0]), np.array([0.0])
    xs, ys = zip(*pts)
    return np.array(xs), np.array(ys)


def bvg_employee(annual: np.ndarray, age: float, params: dict) -> np.ndarray:
    credit = next((params[key] for lo, hi, key in BVG_BANDS if lo <= age < hi), 0.0)
    if credit == 0.0:
        return np.zeros_like(annual)
    coordinated = np.clip(annual - params["bvg_coordination_deduction_chf"],
                          params["bvg_min_coordinated_chf"], params["bvg_max_coordinated_chf"])
    covered = annual >= params["bvg_entry_threshold_chf"]
    return np.where(covered, 0.5 * credit * coordinated, 0.0)


def net_from_gross(gross_monthly, age: float, kind: str, params: dict) -> np.ndarray:
    """Monthly net for monthly gross. kind is "wage" or "ui". Unchanged if the switch is off."""
    g = np.asarray(gross_monthly, dtype=float)
    if not enabled(params):
        return g.copy()
    annual = np.maximum(g, 0.0) * 12.0
    social = (params["ahv_iv_eo_rate"] + params["nbu_rate"]) * annual
    if kind == "wage":
        social = social + params["alv_rate"] * np.minimum(annual, params["alv_ceiling_chf"])
        social = social + bvg_employee(annual, age, params)
    elif kind != "ui":
        raise ValueError(f"unknown income kind {kind!r}")
    xs, ys = tax_curve(params)
    tax = np.interp(annual, xs, ys) * annual
    return np.where(g > 0, (annual - social - tax) / 12.0, g)


def gross_from_net(net_monthly: float, age: float, params: dict) -> float:
    """Wage gross that nets to `net_monthly` (bisection; net rises with gross)."""
    target = float(net_monthly)
    if not enabled(params) or target <= 0:
        return target
    lo, hi = target, target * 3.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float(net_from_gross(mid, age, "wage", params)) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
