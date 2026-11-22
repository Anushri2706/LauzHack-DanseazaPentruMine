"""Cash engine: vectorised starter-kit accounting over sampled regime paths.

Semantics follow starter-kit/financial_twin.py: C_{t+1} = C_t + net_flow_t + tuition_accrual_t - one_offs_t,
balances are end-of-month, minimum includes the opening balance, the first negative month is
reported and negative cash is never clamped (it must be explained as financing need).

TODO(team): investments (I_{t+1} = I_t(1+r) + A - W - fees), debt amortisation, tax, cash
interest, inflation / real values. Keep each as a separate explicit flow so reconciliation holds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import Profile, RegimePaths


@dataclass
class OneOff:
    month: int
    amount_chf: float            # positive = cost (starter-kit convention)
    label: str = ""


@dataclass
class CashPaths:
    opening_cash_chf: float
    cash: np.ndarray             # [n_paths, T] end-of-month balances
    flows: dict[str, np.ndarray] = field(default_factory=dict)  # named [n_paths, T] flows for reconciliation

    @property
    def minimum(self) -> np.ndarray:
        return np.minimum(self.cash.min(axis=1), self.opening_cash_chf)

    @property
    def closing(self) -> np.ndarray:
        return self.cash[:, -1]

    @property
    def first_negative_month(self) -> np.ndarray:
        """[n_paths] first month with cash < 0, or -1 if never."""
        neg = self.cash < 0
        return np.where(neg.any(axis=1), neg.argmax(axis=1), -1)


def run(profile: Profile, paths: RegimePaths, one_offs: list[OneOff] | None = None,
        opening_cash: float | None = None) -> CashPaths:
    T = paths.horizon
    costs = np.zeros(T)
    for item in one_offs or []:
        if not 0 <= item.month < T:
            raise ValueError(f"One-off month {item.month} outside horizon")
        if item.amount_chf < 0:
            raise ValueError("One-off costs cannot be negative")
        costs[item.month] += item.amount_chf
    opening = profile.opening_cash if opening_cash is None else float(opening_cash)
    accrual = paths.tuition_accrual_chf
    if accrual is None:
        accrual = np.zeros_like(paths.net_flow_chf, dtype=float)
    else:
        accrual = np.asarray(accrual, dtype=float)
    # The accrual is loan proceeds added to cash and to debt; living costs stay in the net flow.
    total = paths.net_flow_chf + accrual - costs[None, :]
    cash = opening + np.cumsum(total, axis=1)
    flows = {
        "net_flow_chf": paths.net_flow_chf,
        "one_off_costs_chf": np.broadcast_to(costs, total.shape).copy(),
        "tuition_accrual_chf": accrual,
    }
    return CashPaths(opening, cash, flows)


def reconciles(cp: CashPaths, tol: float = 1e-6) -> bool:
    """Opening + sum of named flows == closing on every path."""
    accrual = cp.flows.get("tuition_accrual_chf", 0.0)
    total = cp.flows["net_flow_chf"] - cp.flows["one_off_costs_chf"] + accrual
    return bool(np.allclose(cp.opening_cash_chf + total.sum(axis=1), cp.closing, atol=tol))


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
