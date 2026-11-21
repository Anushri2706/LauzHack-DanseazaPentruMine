"""Cash engine: vectorised starter-kit accounting over sampled regime paths.

Semantics follow starter-kit/financial_twin.py: C_{t+1} = C_t + net_flow_t - one_offs_t,
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
    total = paths.net_flow_chf - costs[None, :]
    cash = opening + np.cumsum(total, axis=1)
    flows = {"net_flow_chf": paths.net_flow_chf, "one_off_costs_chf": np.broadcast_to(costs, total.shape)}
    return CashPaths(opening, cash, flows)


def reconciles(cp: CashPaths, tol: float = 1e-6) -> bool:
    """Opening + sum of named flows == closing on every path."""
    total = cp.flows["net_flow_chf"] - cp.flows["one_off_costs_chf"]
    return bool(np.allclose(cp.opening_cash_chf + total.sum(axis=1), cp.closing, atol=tol))

