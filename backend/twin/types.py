"""Shared types for the regime-switching financial twin.

All amounts are CHF per month. Months are zero-indexed from the simulation start.
See docs/hmm-regime-engine-api.md (Project store) for the interface contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STATES: list[str] = ["employed", "part_time", "job_search", "student", "retired"]
STATE_INDEX: dict[str, int] = {s: i for i, s in enumerate(STATES)}
K = len(STATES)

# Data/synthetic-profiles.csv employment_status -> regime. self_employed is pooled
# with employed (no separate regime); revisit if its emission differs materially.
EMPLOYMENT_TO_STATE: dict[str, str] = {
    "employed": "employed",
    "self_employed": "employed",
    "self-employed": "employed",
    "part_time": "part_time",
    "part-time": "part_time",
    "unemployed": "job_search",
    "student": "student",
    "retired": "retired",
}


def state_of(employment: str) -> str:
    try:
        return EMPLOYMENT_TO_STATE[employment]
    except KeyError as exc:
        raise ValueError(f"Unknown employment status {employment!r}") from exc


@dataclass
class Profile:
    age: int
    employment: str                    # initial regime hint (raw status or a STATES name)
    ref_income: float                  # CHF/month, reference net income
    rent: float                        # CHF/month
    attributes: dict = field(default_factory=dict)  # e.g. {"has_masters": False, "abroad": False}
    # Extensions to the contract (all optional):
    profile_id: str | None = None
    name: str | None = None
    opening_cash: float = 0.0
    mean_savings_rate: float | None = None   # per-profile mean net_flow / ref_income from the ledger
    start_month_of_year: int = 1             # 1 = January; used for the December (13th salary) offset
    # New tuition debt. False uses the ETH student-loan rate (0). True uses this profile's personal loan.
    use_existing_loan_rate: bool = False

    @property
    def initial_state(self) -> str:
        return self.employment if self.employment in STATE_INDEX else state_of(self.employment)


@dataclass
class Intervention:
    """A decision applied as do() on the regime chain."""
    kind: str                  # "force_state" | "scale_hazard" | "set_attribute" | "parental_leave"
    start_month: int
    duration: int | None = None
    state: str | None = None   # for force_state
    transition: tuple[str, str] | None = None  # for scale_hazard, e.g. ("employed", "job_search")
    factor: float | None = None
    attribute: tuple[str, object] | None = None

    def active(self, t: int) -> bool:
        return t >= self.start_month and (self.duration is None or t < self.start_month + self.duration)


@dataclass
class RegimePaths:
    states: np.ndarray           # [n_paths, T] int, index into STATES
    income_chf: np.ndarray       # [n_paths, T] absolute monthly net income in CHF
    savings_rate: np.ndarray     # [n_paths, T] net flow / income_chf
    net_flow_chf: np.ndarray     # [n_paths, T] savings_rate * income_chf (one-offs are added downstream)
    occupancy: np.ndarray        # [T, K] simulated P(state_t = k); equals pi0 A^t when A is fixed and nothing is pinned
    transition_draws: np.ndarray # [n_paths, K, K] sampled A (parameter uncertainty)
    lump_chf: np.ndarray | None = None  # [n_paths, T] holiday + emergency, one draw per year; None if off
    # Extension. Tuition debt is a running total, not a regime. None on paths built outside simulate.
    debt_chf: np.ndarray | None = None            # [n_paths, T] end-of-month tuition-debt total
    tuition_accrual_chf: np.ndarray | None = None  # [n_paths, T] loan principal booked that month
    # Extension. Investments are a running total of positive net flow, not a regime and not a mortgage.
    investments_chf: np.ndarray | None = None     # [n_paths, T] end-of-month investments total

    @property
    def n_paths(self) -> int:
        return self.states.shape[0]

    @property
    def horizon(self) -> int:
        return self.states.shape[1]


@dataclass
class DecisionSnapshot:
    """Frozen record the caller stores. ``RegimeEngine`` does not hold it.

    Restore replays ``filter`` on ``observed_net_flows``, then the next ``simulate``
    uses ``initial_belief=belief`` and these interventions. There is no balance-sheet
    field: opening cash lives only on ``profile``. ``snapshot_id`` and ``parent_id``
    are the branch tree the phone draws. ``one_offs`` are ``cash_engine.OneOff``.
    """

    month: int
    profile: Profile
    belief: dict[str, float]
    observed_net_flows: list[float]
    interventions: list[Intervention]
    one_offs: list
    assumption_overrides: dict
    seed: int
    parent_id: str | None
    snapshot_id: str
