"""Orchestration: profile + decisions + assumptions + config + seed -> simulated monthly arrays.

This module only wires the migrated pieces together; it does no financial maths of its own beyond
naming the flows so that cash reconciles.

Monthly operation order (as computed by the migrated engine, vectorised over paths)
--------------------------------------------------------------------------------
For each month t = 0..T-1:
 1. Decisions scheduled for month t take effect: pinned regimes (force_state / parental leave),
    attribute changes (has_masters, abroad, rent factors) and one-off costs dated t.
 2. Life regime: if t is not pinned, s_t is drawn from row s_{t-1} of the month's transition matrix
    (path's Dirichlet draw x age-band hazards x decision scalings x long-term-unemployment decay);
    working paths at or past retirement_age become 'retired'.
 3. Independent events (engine/events.py) eligible in s_t are drawn from their own stream.
 4. Income: gross pay for s_t (wage, UI, stipend, pension), then net pay (tax.py); raises at year ends.
 5. Recurring spending: sampled spend/rent ratio x rent (or the expense-priors recipe) and the December bonus.
 6. Market: price level (yearly inflation regime switch) and portfolio return.
 7. Cash flow: kept net flow (keep factor on surplus months) - spending inflation adjustment
    - decision one-offs x price - event costs x price - loan interest.
 8. cash_t = cash_{t-1} + flow_t ; portfolio_t = portfolio_{t-1} x (1 + r_t) ; debt constant (interest-only).
 9. net worth_t = cash_t + portfolio_t - debt ; real = nominal / price_t. Recorded with provenance.
A job loss and a one-off in the same month are both charged in that month (tested).

Note on the requested ``step_month`` API: the migrated HMM samples the whole state matrix month by month
inside ``regime.sample_regime_paths`` and the income/housing tables walk months with path history
(experience, UI clock, leases). Re-implementing them as a separate per-month stepper would create a second
engine, which the brief forbids. ``run_branch`` / ``run_monte_carlo`` are provided; ``trace`` returns the
per-month state changes of any path.

Randomness: every stream is an explicit ``numpy.random.Generator``:
  HMM       default_rng(seed)                  (inside RegimeEngine.simulate, unchanged)
  market    default_rng(10_000 + seed)         (inside wealth.sample_market, unchanged)
  events    SeedSequence([seed, EVENT_STREAM, i])
No module-global random state is used.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np

from . import wealth as wealth_mod
from .assumptions import AssumptionSet
from .decisions import Decision, to_engine
from .events import CATALOGUE as EVENT_CATALOGUE
from .events import sample_events
from .profile import EngineProfile
from .regime import RegimeEngine, fit_emissions
from .transitions import load_transition_priors
from .types import STATES

CORE_ASSUMPTIONS = (
    "student_income_chf", "entry_salary_chf", "annual_experience_growth", "experience_growth_30_40",
    "experience_growth_40_50", "experience_growth_50_plus", "part_time_fraction", "ui_replacement_rate",
    "ui_replacement_with_dependents", "ui_low_earnings_threshold_chf", "ui_insured_earnings_cap_chf",
    "ui_max_months", "ui_max_months_under_25", "job_search_floor_chf", "graduate_waiting_months",
    "graduate_benefit_months", "graduate_benefit_under_25_chf", "graduate_benefit_chf",
    "long_term_unemployment_multiplier", "long_term_unemployment_months", "retired_income_fraction",
    "retirement_age", "student_rent_chf", "employed_rent_chf", "scaled_spending", "keep_factor",
    "inflation_low", "inflation_base", "inflation_high", "inflation_stay_low", "inflation_low_to_base",
    "inflation_base_to_low", "inflation_base_to_high", "inflation_high_to_low", "inflation_high_to_base",
    "portfolio_return", "portfolio_vol", "apply_income_tax", "ahv_iv_eo_rate", "alv_rate", "alv_ceiling_chf",
    "nbu_rate", "bvg_entry_threshold_chf", "bvg_coordination_deduction_chf", "bvg_min_coordinated_chf",
    "bvg_max_coordinated_chf", "bvg_credit_25_34", "bvg_credit_35_44", "bvg_credit_45_54", "bvg_credit_55_64",
    "tax_rate_at_0", "tax_rate_at_20000", "tax_rate_at_40000", "tax_rate_at_60000", "tax_rate_at_80000",
    "tax_rate_at_100000", "tax_rate_at_150000", "tax_rate_at_200000", "tax_rate_at_300000",
)


class SimulationError(ValueError):
    pass


@dataclass(frozen=True)
class SimulationConfig:
    horizon_months: int = 120
    n_paths: int = 500
    inflation: bool = True
    enabled_events: tuple[str, ...] = ()
    engine_mode: str = "hmm"          # "hmm" (fitted) or "stub" (deterministic regimes, for tests)

    def validate(self) -> None:
        if not 12 <= int(self.horizon_months) <= 600:
            raise SimulationError("horizon_months must be in [12, 600]")
        if not 1 <= int(self.n_paths) <= 20_000:
            raise SimulationError("n_paths must be in [1, 20000]")
        for e in self.enabled_events:
            if e not in EVENT_CATALOGUE:
                raise SimulationError(f"unknown event {e!r}")
        if self.engine_mode not in ("hmm", "stub"):
            raise SimulationError("engine_mode must be 'hmm' or 'stub'")

    def to_dict(self) -> dict:
        return {"horizon_months": self.horizon_months, "n_paths": self.n_paths, "inflation": self.inflation,
                "enabled_events": list(self.enabled_events), "engine_mode": self.engine_mode}


@lru_cache(maxsize=2)
def base_engine(mode: str = "hmm") -> RegimeEngine:
    """The fitted engine (emissions fitted on the training split of Data/, sourced transition priors)."""
    if mode == "stub":
        return RegimeEngine.stub()
    return RegimeEngine(fit_emissions(), load_transition_priors())


def engine_for(ep: EngineProfile, other_expenses: float | None, mode: str = "hmm") -> RegimeEngine:
    """Personalise today's regime spending level to the user's stated budget (twin/engine.engine_for)."""
    base = base_engine(mode)
    prof = ep.profile
    if mode == "stub" or other_expenses is None or prof.rent <= 0:
        return base
    obs = [(prof.rent + other_expenses) / prof.rent] * 24   # stated, so it dominates the prior (kappa=6)
    em = base.emissions.personalize(prof.initial_state, np.asarray(obs), prof.attributes.get("household_type"),
                                    self_employed=prof.employment.replace("-", "_") == "self_employed")
    return RegimeEngine(em, base.priors, base.income_table)


@dataclass
class SimulationRun:
    seed: int
    config: SimulationConfig
    assumptions_version: str
    decisions: list[dict]
    states: np.ndarray            # [n, T] regime index
    gross_income: np.ndarray      # [n, T] CHF nominal
    net_income: np.ndarray        # [n, T] CHF nominal
    spending: np.ndarray          # [n, T] CHF nominal (engine spending x price level)
    keep_adjustment: np.ndarray   # [n, T] CHF nominal (<= 0; keep factor on surplus months)
    one_off_costs: np.ndarray     # [n, T] CHF nominal
    event_costs: np.ndarray       # [n, T] CHF nominal
    loan_interest: float          # CHF/month nominal
    cash: np.ndarray              # [n, T] end-of-month CHF nominal
    portfolio: np.ndarray         # [n, T] CHF nominal
    price: np.ndarray             # [n, T] price level (1 = today)
    debt: float                   # CHF nominal, constant
    opening_cash: float
    opening_portfolio: float
    reserve_chf: float
    event_counts: dict
    assumption_ids: list[str]
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def n_paths(self) -> int:
        return self.cash.shape[0]

    @property
    def horizon(self) -> int:
        return self.cash.shape[1]

    @property
    def net_worth_nominal(self) -> np.ndarray:
        return self.cash + self.portfolio - self.debt

    @property
    def net_worth_real(self) -> np.ndarray:
        return self.net_worth_nominal / self.price

    def total_flow(self) -> np.ndarray:
        return (self.net_income - self.spending + self.keep_adjustment - self.one_off_costs
                - self.event_costs - self.loan_interest)

    def reconciles(self, tol: float = 1e-6) -> bool:
        """opening + cumulative named flows == cash, every path and every month."""
        rebuilt = self.opening_cash + np.cumsum(self.total_flow(), axis=1)
        return bool(np.allclose(rebuilt, self.cash, atol=tol, rtol=0))

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for a in (self.states, self.cash, self.portfolio, self.price, self.net_income):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()[:16]

    def trace(self, path: int) -> list[dict]:
        if not 0 <= path < self.n_paths:
            raise IndexError("path out of range")
        nw = self.net_worth_nominal[path]
        out, prev = [], None
        for t in range(self.horizon):
            s = STATES[int(self.states[path, t])]
            out.append({
                "month": t, "regime": s, "regime_changed": prev is not None and s != prev,
                "gross_income_chf": float(self.gross_income[path, t]), "net_income_chf": float(self.net_income[path, t]),
                "spending_chf": float(self.spending[path, t]), "keep_adjustment_chf": float(self.keep_adjustment[path, t]),
                "one_off_costs_chf": float(self.one_off_costs[path, t]), "event_costs_chf": float(self.event_costs[path, t]),
                "loan_interest_chf": float(self.loan_interest), "cash_chf": float(self.cash[path, t]),
                "portfolio_chf": float(self.portfolio[path, t]), "price_level": float(self.price[path, t]),
                "net_worth_nominal_chf": float(nw[t]), "net_worth_real_chf": float(nw[t] / self.price[path, t]),
            })
            prev = s
        return out


def _used_assumptions(assumptions: AssumptionSet, decisions: list[Decision], config: SimulationConfig,
                      ep: EngineProfile) -> list[str]:
    from .decisions import CATALOGUE as DEC
    ids = set(CORE_ASSUMPTIONS) | set(ep.assumption_ids)
    ids |= {a.id for a in assumptions.registry.rows.values() if a.table == "transitions"}
    if assumptions.value("scaled_spending") >= 0.5:
        ids |= {a.id for a in assumptions.registry.rows.values() if a.table == "expense"}
    for d in decisions:
        spec_defaults = {spec.default_assumption: spec.name for spec in DEC[d.decision_id].params if spec.default_assumption}
        for aid in DEC[d.decision_id].required_assumptions:
            # a parameter default is "used" only when the player did not give the value
            if aid in spec_defaults and spec_defaults[aid] not in d.defaults_used:
                continue
            ids.add(aid)
        for aid, name in spec_defaults.items():
            if name in d.defaults_used:
                ids.add(aid)
    for e in config.enabled_events:
        ids |= set(EVENT_CATALOGUE[e].assumption_ids)
    if ep.profile.loans:
        pass  # the loan rate is a user fact, not a registry assumption
    return sorted(i for i in ids if i in assumptions.registry)


def run_monte_carlo(ep: EngineProfile, decisions: list[Decision], assumptions: AssumptionSet,
                    config: SimulationConfig, n_runs: int, seed: int, engine: RegimeEngine,
                    canton_rent_ratio=None) -> SimulationRun:
    """Run ``n_runs`` independent futures of one branch. Pure: inputs are not mutated."""
    config = replace(config, n_paths=int(n_runs))
    config.validate()
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise SimulationError("seed must be a non-negative integer")
    T, n = int(config.horizon_months), int(n_runs)
    for d in decisions:
        if not 0 <= d.start_month < T:
            raise SimulationError(f"decision {d.decision_id} starts outside the horizon")
    ei = to_engine(list(decisions), T, canton_rent_ratio=canton_rent_ratio)
    profile = ep.profile
    if ei.profile_changes:
        profile = replace(profile, attributes={**profile.attributes, **ei.profile_changes})
    overrides = assumptions.engine_overrides()
    paths = engine.simulate(profile, horizon_months=T, n_paths=n, interventions=ei.interventions,
                            assumption_overrides=overrides, seed=int(seed))
    params = engine.income_table.with_overrides(overrides).params
    ev = sample_events(list(config.enabled_events), paths.states, float(profile.age), assumptions, int(seed))
    keep = float(assumptions.value("keep_factor"))
    w = wealth_mod.build(profile, paths, ei.one_offs, params, int(seed), keep, config.inflation,
                         ev.costs if config.enabled_events else None)
    net = w.net()
    kept = np.where(net > 0, keep * net, net)
    gross = paths.gross_income_chf if paths.gross_income_chf is not None else paths.income_chf
    run = SimulationRun(
        seed=int(seed), config=config, assumptions_version=assumptions.version,
        decisions=[d.to_dict() for d in decisions], states=paths.states, gross_income=np.asarray(gross, float),
        net_income=np.asarray(w.income, float), spending=w.spend * w.price, keep_adjustment=kept - net,
        one_off_costs=np.broadcast_to(w.oneoff[None, :], w.price.shape) * w.price,
        event_costs=ev.costs * w.price, loan_interest=float(w.loan_interest), cash=w.cash(),
        portfolio=w.port, price=w.price, debt=float(w.debt), opening_cash=float(w.opening),
        opening_portfolio=float(w.port0), reserve_chf=float(ep.reserve_chf),
        event_counts={k: v for k, v in ev.counts.items()},
        assumption_ids=_used_assumptions(assumptions, list(decisions), config, ep),
        warnings=list(ep.warnings), notes=list(ei.notes))
    if not run.reconciles():
        raise SimulationError("cash does not reconcile with the named flows")  # pragma: no cover
    return run


def run_branch(ep: EngineProfile, decisions: list[Decision], assumptions: AssumptionSet,
               config: SimulationConfig, seed: int, engine: RegimeEngine, canton_rent_ratio=None) -> SimulationRun:
    """One future of one branch (a single path). Use ``trace(0)`` for its month-by-month record."""
    return run_monte_carlo(ep, decisions, assumptions, config, 1, seed, engine, canton_rent_ratio)
