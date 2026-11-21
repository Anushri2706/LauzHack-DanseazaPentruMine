"""Independent life events: shocks that are NOT already represented by the HMM.

Ownership rule (no double counting)
-----------------------------------
The regime HMM owns every employment-state change: job loss (employed/part_time -> job_search),
re-employment, part-time switches and retirement (age rule). Those risks are listed in
``HMM_OWNED_RISKS`` and ``validate_definition`` rejects any event that declares one of them or whose
impact would force a regime. An event may only add a distinct cash shock.

Time basis
----------
Monthly probability, annual probability and hazard rate are different things. Each definition states
its ``time_basis`` and the conversion to a per-month probability is explicit:
    annual probability p_a      -> p_m = 1 - (1 - p_a) ** (1 / 12)
    annual hazard rate lambda   -> p_m = 1 - exp(-lambda / 12)
    monthly probability p_m     -> used as is

Randomness
----------
Events draw from their own ``numpy.random.Generator`` seeded with ``SeedSequence([seed, EVENT_STREAM, i])``
(i = event index in the catalogue). Turning an event on therefore never shifts the HMM or market streams,
and two branches with the same seed see the same event draws (common random numbers).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .assumptions import AssumptionSet
from .types import STATE_INDEX, STATES

EVENT_STREAM = 0xE7E7

# Risks the HMM (or the age rule inside it) already models. Events must not re-declare them.
HMM_OWNED_RISKS: dict[str, str] = {
    "job_loss": "employed -> job_search (job_loss_hazard and age bands)",
    "part_time_job_loss": "part_time -> job_search (part_time_job_loss_hazard)",
    "re_employment": "job_search -> employed / part_time (job_search_exit_rate)",
    "part_time_switch": "employed <-> part_time",
    "retirement": "age rule at retirement_age",
    "unemployment": "any move into job_search",
}

TIME_BASES = ("monthly_probability", "annual_probability", "annual_hazard_rate")


def annual_to_monthly_probability(p_annual: float) -> float:
    if not 0.0 <= p_annual <= 1.0:
        raise ValueError("annual probability must be in [0, 1]")
    return 1.0 - (1.0 - p_annual) ** (1.0 / 12.0)


def monthly_to_annual_probability(p_month: float) -> float:
    if not 0.0 <= p_month <= 1.0:
        raise ValueError("monthly probability must be in [0, 1]")
    return 1.0 - (1.0 - p_month) ** 12


def annual_hazard_to_monthly_probability(rate: float) -> float:
    if rate < 0:
        raise ValueError("hazard rate cannot be negative")
    return 1.0 - math.exp(-rate / 12.0)


@dataclass(frozen=True)
class EventDefinition:
    id: str
    description: str
    risk: str                               # what is being modelled (must not be in HMM_OWNED_RISKS)
    time_basis: str                         # one of TIME_BASES
    probability_assumption: str             # registry id holding the probability / rate
    impact_type: str                        # "one_off_cost" (today's CHF, inflated by the price level)
    impact_assumption: str                  # registry id holding the cost
    eligible_states: tuple[str, ...] = tuple(STATES)
    min_age: float | None = None
    max_age: float | None = None
    window: tuple[int, int | None] = (0, None)   # [start, end) months; end None = horizon
    recurrence: str = "can_recur"           # can_recur | once
    duration_months: int = 1                # cost is charged in the month of occurrence
    player_response: str = "Paid from cash; negative cash is reported as a financing need."
    status: str = "illustrative"
    default_enabled: bool = False

    @property
    def assumption_ids(self) -> tuple[str, ...]:
        return (self.probability_assumption, self.impact_assumption)

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["assumption_ids"] = list(self.assumption_ids)
        return d


def validate_definition(ev: EventDefinition) -> None:
    if ev.risk in HMM_OWNED_RISKS:
        raise ValueError(f"event {ev.id!r} re-declares HMM-owned risk {ev.risk!r} ({HMM_OWNED_RISKS[ev.risk]})")
    if ev.time_basis not in TIME_BASES:
        raise ValueError(f"event {ev.id!r}: unknown time basis {ev.time_basis!r}")
    if ev.impact_type != "one_off_cost":
        raise ValueError(f"event {ev.id!r}: impact type {ev.impact_type!r} not supported (regime changes belong to the HMM)")
    for s in ev.eligible_states:
        if s not in STATE_INDEX:
            raise ValueError(f"event {ev.id!r}: unknown state {s!r}")


CATALOGUE: dict[str, EventDefinition] = {e.id: e for e in [
    EventDefinition(
        id="unexpected_major_expense",
        description="An unplanned major expense (repair, medical excess, replacing an appliance).",
        risk="irregular_major_expense",
        time_basis="annual_probability",
        probability_assumption="event_major_expense_annual_prob",
        impact_type="one_off_cost",
        impact_assumption="event_major_expense_cost_chf",
        status="placeholder",
    ),
]}
for _ev in CATALOGUE.values():
    validate_definition(_ev)

# Shocks that exist inside the migrated engine and are therefore NOT sampled here (listed for transparency).
ENGINE_INTERNAL_SHOCKS = {
    "yearly_irregular_lump": "regime.sample_yearly_lump: one lognormal irregular-spend shock per year, "
                             "only when scaled_spending=1 (off by default).",
    "december_bonus": "regime.sample_december_bonus: 13th-salary bump in December while employed.",
}


def monthly_probability(ev: EventDefinition, assumptions: AssumptionSet) -> float:
    v = assumptions.value(ev.probability_assumption)
    if ev.time_basis == "monthly_probability":
        return float(v)
    if ev.time_basis == "annual_probability":
        return annual_to_monthly_probability(v)
    return annual_hazard_to_monthly_probability(v)


@dataclass
class EventDraws:
    costs: np.ndarray                                   # [n, T] today's CHF
    counts: dict[str, np.ndarray] = field(default_factory=dict)   # event id -> [n] occurrences


def sample_events(enabled: list[str], states: np.ndarray, age0: float, assumptions: AssumptionSet,
                  seed: int) -> EventDraws:
    n, T = states.shape
    costs = np.zeros((n, T))
    counts: dict[str, np.ndarray] = {}
    ids = list(CATALOGUE)
    for eid in enabled:
        if eid not in CATALOGUE:
            raise ValueError(f"unknown event {eid!r}")
        ev = CATALOGUE[eid]
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), EVENT_STREAM, ids.index(eid)])))
        p = monthly_probability(ev, assumptions)
        u = rng.random((n, T))                          # drawn for every cell: alignment does not depend on states
        ages = age0 + np.arange(T) / 12.0
        eligible = np.isin(states, [STATE_INDEX[s] for s in ev.eligible_states])
        in_age = np.ones(T, dtype=bool)
        if ev.min_age is not None:
            in_age &= ages >= ev.min_age
        if ev.max_age is not None:
            in_age &= ages < ev.max_age
        lo, hi = ev.window
        in_window = np.zeros(T, dtype=bool)
        in_window[lo:(T if hi is None else min(hi, T))] = True
        hit = (u < p) & eligible & in_age[None, :] & in_window[None, :]
        if ev.recurrence == "once":
            first = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
            hit = np.zeros_like(hit)
            rows = np.flatnonzero(first >= 0)
            hit[rows, first[rows]] = True
        costs += hit * assumptions.value(ev.impact_assumption)
        counts[eid] = hit.sum(axis=1)
    return EventDraws(costs, counts)
