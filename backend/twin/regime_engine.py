"""Regime engine: a regime-switching (input-output HMM) scenario generator.

Answers one question: given who this person is, what they decide and our assumptions, what
life regime are they likely in each month, and what monthly net cash flow does that produce?
It does no cash accounting, tax, investments, goals, UI or narration (see cash_engine, metrics).

Contract: docs/hmm-regime-engine-api.md. Design: docs/hmm-digital-twin.md (Project store).

Two modes:
- ``RegimeEngine.stub()``: deterministic, fixed regimes (initial state + force_state
  interventions + retirement age), constant per-state savings rate. Runs today; the team builds
  cash engine / metrics / UI / agent against it.
- ``RegimeEngine(emissions, priors)``: the real HMM. Needs the TODO functions below.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .income import IncomeTable
from .transitions import TransitionPriors, load_transition_priors
from .types import K, STATE_INDEX, STATES, Intervention, Profile, RegimePaths

# Where a scheduled (force_state with duration) phase exits to. Graduation -> employed,
# end of a forced job search -> employed; anything else returns to the pre-intervention state.
EXIT_STATE: dict[str, str] = {"student": "employed", "job_search": "employed"}


# ---------------------------------------------------------------------------
# Emissions
# ---------------------------------------------------------------------------

@dataclass
class EmissionModel:
    """Per-regime distribution of savings_rate = net_flow / income.

    params[state] holds the fitted family parameters, e.g. Student-t
    ``{"loc": .., "scale": .., "df": ..}`` or 2-GMM ``{"w": [..], "mu": [..], "sigma": [..]}``.
    ``december_offset[state]`` is the additive 13th-salary uplift in December.
    """
    family: str                                   # "student_t" | "gmm2" | "constant"
    params: dict[str, dict]
    december_offset: dict[str, float] = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)  # KS stats, n per state, holdout ids

    def mean(self, state: str) -> float:
        p = self.params[state]
        if "loc" in p:
            return float(p["loc"])
        return float(np.dot(p["w"], p["mu"]))

    @classmethod
    def placeholder(cls) -> "EmissionModel":
        """Means measured in docs/hmm-digital-twin.md §1 (employed / part-time / retired).

        student and job_search are PLACEHOLDERS: the raw data ratios (-2.6x, heavy burn) are
        relative to tiny reference incomes and must be refitted against income_chf (see plan).
        """
        means = {"employed": 0.52, "part_time": 0.11, "job_search": -0.5,
                 "student": -0.75, "retired": 0.04}
        return cls("constant", {s: {"loc": m, "scale": 0.0, "df": np.inf} for s, m in means.items()},
                   {"employed": 0.35, "part_time": 0.35})


def fit_emissions(cash_balances: pd.DataFrame, profiles: pd.DataFrame, family: str = "student_t",
                  holdout_frac: float = 0.2, seed: int = 0) -> EmissionModel:
    """Fit per-regime emission distributions on the 28.8k person-months.

    TODO(Victor):
    1. Build person-months (data.person_months): ratio = net / reference income, state from
       employment_status, is_december flag. Hold out ``holdout_frac`` of *profiles* (not rows).
    2. December offset per state = median(ratio | Dec) - median(ratio | not Dec); fit the
       distribution on December-adjusted ratios.
    3. Per state, fit Student-t (scipy.stats.t.fit) or a 2-component GMM (sklearn) on the raw
       pooled ratios, so between-person spread stays in the emission (hierarchical variant later).
    4. Robustness: winsorise at P0.5/P99.5 for the fit, keep the raw tails for the shock state.
    5. Store diagnostics: n per state, KS stat/p on the holdout, QQ quantiles.
    """
    raise NotImplementedError("fit_emissions: see TODO in docstring")


def emission_loglik(model: EmissionModel, ratios: np.ndarray, month_of_year: np.ndarray) -> np.ndarray:
    """log p(o_t | s_t = k) for each observed month. Returns [T, K].

    TODO(Victor): subtract december_offset[k] where month_of_year == 12, evaluate the fitted
    density (scipy.stats.t.logpdf / GMM logsumexp). Floor at a minimum log-density so a single
    outlier cannot zero out a state.
    """
    raise NotImplementedError("emission_loglik: see TODO in docstring")


# ---------------------------------------------------------------------------
# Filtering and sampling (real HMM)
# ---------------------------------------------------------------------------

def forward_filter(log_lik: np.ndarray, A: np.ndarray, pi0: np.ndarray) -> tuple[np.ndarray, float]:
    """Scaled forward algorithm. Returns (filtered [T, K] = P(s_t | o_1:t), total log-likelihood).

    ``A`` is [K, K] or time-varying [T, K, K].

    TODO(Victor): alpha_0 ∝ pi0 * exp(ll_0); alpha_t ∝ (alpha_{t-1} @ A_t) * exp(ll_t), with
    max-subtraction in log space and per-step normalisation; accumulate log normalisers.
    """
    raise NotImplementedError("forward_filter: see TODO in docstring")


def sample_regime_paths(rng: np.random.Generator, pi0: np.ndarray, A_draws: np.ndarray, horizon: int,
                        profile: Profile, priors: TransitionPriors, interventions: list[Intervention],
                        overrides: dict, retirement_age: float = 65.0) -> np.ndarray:
    """Sample [n_paths, T] state indices.

    TODO(Victor):
    - s_0 ~ pi0 (filtered belief or one-hot initial state).
    - Per step: build A_t per path = A_draw with age bounds and active scale_hazard
      interventions applied (priors.matrix logic, vectorised); sample s_{t+1} by inverse CDF on
      the row of each path's current state.
    - force_state with duration = scheduled HSMM phase: state pinned for the window, then resume
      from EXIT_STATE (or the pre-intervention state). Without duration: a jump into the state
      at start_month, after which the chain evolves freely under A_t (job-loss stress test exits
      job search via job_search_exit_rate). The stub has no transitions, so a jump persists.
    - Deterministic retirement when age_t >= retirement_age for employed / part_time.
    - Optional HSMM duration counters for states with non-geometric dwell (see plan).
    """
    raise NotImplementedError("sample_regime_paths: see TODO in docstring")


def sample_emissions(rng: np.random.Generator, model: EmissionModel, states: np.ndarray,
                     month_of_year: np.ndarray, overrides: dict | None = None) -> np.ndarray:
    """Sample savings_rate [n_paths, T] given states.

    TODO(Victor): draw from the fitted per-state family (vectorised by state mask), add
    december_offset in December, apply ``savings_rate_<state>`` overrides as a mean shift.
    """
    raise NotImplementedError("sample_emissions: see TODO in docstring")


def analytic_occupancy(pi0: np.ndarray, A: np.ndarray, horizon: int) -> np.ndarray:
    """P(s_t = k) = pi0 A^t for t = 0..T-1. ``A`` is [K, K] or [T, K, K]. Returns [T, K]."""
    occ = np.empty((horizon, len(pi0)))
    p = np.asarray(pi0, dtype=float)
    for t in range(horizon):
        occ[t] = p
        At = A[t] if A.ndim == 3 else A
        p = p @ At
    return occ


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def _one_hot(state: str) -> np.ndarray:
    v = np.zeros(K)
    v[STATE_INDEX[state]] = 1.0
    return v


def scheduled_states(profile: Profile, horizon: int, interventions: list[Intervention],
                     retirement_age: float) -> np.ndarray:
    """Deterministic state sequence [T]: initial state, force_state windows, retirement age."""
    base = STATE_INDEX[profile.initial_state]
    seq = np.full(horizon, base)
    for iv in sorted((i for i in interventions if i.kind == "force_state"), key=lambda i: i.start_month):
        end = horizon if iv.duration is None else min(horizon, iv.start_month + iv.duration)
        prev = seq[iv.start_month] if iv.start_month < horizon else base
        seq[iv.start_month:end] = STATE_INDEX[iv.state]
        if iv.duration is not None and end < horizon:
            exit_state = EXIT_STATE.get(iv.state)
            seq[end:] = STATE_INDEX[exit_state] if exit_state else prev
    ages = profile.age + np.arange(horizon) / 12.0
    working = np.isin(seq, [STATE_INDEX["employed"], STATE_INDEX["part_time"]])
    seq[working & (ages >= retirement_age)] = STATE_INDEX["retired"]
    return seq


class RegimeEngine:
    def __init__(self, emissions: EmissionModel, priors: TransitionPriors,
                 income_table: IncomeTable | None = None, mode: str = "hmm"):
        if mode not in ("hmm", "stub"):
            raise ValueError("mode must be 'hmm' or 'stub'")
        self.emissions = emissions
        self.priors = priors
        self.income_table = income_table or IncomeTable.from_csv()
        self.mode = mode

    @classmethod
    def stub(cls, priors: TransitionPriors | None = None, income_table: IncomeTable | None = None) -> "RegimeEngine":
        return cls(EmissionModel.placeholder(), priors or load_transition_priors(), income_table, mode="stub")

    # "Twin syncs with reality"
    def filter(self, profile: Profile, observed_net_flows: list[float],
               assumption_overrides: dict | None = None) -> dict[str, float]:
        """Belief over the current regime after observing monthly net flows (CHF)."""
        if self.mode == "stub":
            belief = {s: 0.1 / (K - 1) for s in STATES}
            belief[profile.initial_state] = 0.9
            return belief
        # TODO(Victor): ratios = flows / profile.ref_income; month_of_year from profile start;
        # ll = emission_loglik(...); A = priors.matrix(overrides, age); pi0 = one-hot(initial) mixed
        # with 5% uniform; filtered, _ = forward_filter(ll, A, pi0); return dict(zip(STATES, filtered[-1])).
        raise NotImplementedError("RegimeEngine.filter (hmm mode)")

    # Main call the team uses
    def simulate(self, profile: Profile, horizon_months: int = 60, n_paths: int = 5000,
                 interventions: list[Intervention] | None = None, assumption_overrides: dict | None = None,
                 initial_belief: dict | None = None, seed: int = 0) -> RegimePaths:
        interventions = list(interventions or [])
        overrides = dict(assumption_overrides or {})
        income_table = self.income_table.with_overrides(overrides)
        if self.mode == "stub":
            return self._simulate_stub(profile, horizon_months, n_paths, interventions, overrides, income_table)
        return self._simulate_hmm(profile, horizon_months, n_paths, interventions, overrides,
                                  initial_belief, seed, income_table)

    def savings_rate_for(self, state: str, profile: Profile, overrides: dict) -> float:
        key = f"savings_rate_{state}"
        if key in overrides:
            return float(overrides[key])
        if state == profile.initial_state and profile.mean_savings_rate is not None:
            return float(profile.mean_savings_rate)
        return self.emissions.mean(state)

    def _simulate_stub(self, profile, T, n, interventions, overrides, income_table) -> RegimePaths:
        seq = scheduled_states(profile, T, interventions, income_table.params["retirement_age"])
        states = np.repeat(seq[None, :], n, axis=0)
        income = np.repeat(income_table.income_paths(seq[None, :], profile, interventions), n, axis=0)
        rate_by_state = np.array([self.savings_rate_for(s, profile, overrides) for s in STATES])
        savings_rate = rate_by_state[states]
        occupancy = np.zeros((T, K))
        occupancy[np.arange(T), seq] = 1.0
        A = self.priors.matrix(overrides)
        return RegimePaths(
            states=states,
            income_chf=income,
            savings_rate=savings_rate,
            net_flow_chf=savings_rate * income,
            occupancy=occupancy,
            transition_draws=np.broadcast_to(A, (n, K, K)),
        )

    def _simulate_hmm(self, profile, T, n, interventions, overrides, initial_belief, seed, income_table) -> RegimePaths:
        rng = np.random.default_rng(seed)
        pi0 = (np.array([initial_belief.get(s, 0.0) for s in STATES]) if initial_belief
               else _one_hot(profile.initial_state))
        A_draws = self.priors.sample(rng, n, overrides)
        states = sample_regime_paths(rng, pi0, A_draws, T, profile, self.priors, interventions,
                                     overrides, income_table.params["retirement_age"])
        month_of_year = (profile.start_month_of_year - 1 + np.arange(T)) % 12 + 1
        savings_rate = sample_emissions(rng, self.emissions, states, month_of_year, overrides)
        income = income_table.income_paths(states, profile, interventions)
        occupancy = np.stack([(states == k).mean(axis=0) for k in range(K)], axis=1)
        # TODO(Victor): report analytic_occupancy(pi0, A_t) alongside and assert agreement in validation.
        return RegimePaths(states, income, savings_rate, savings_rate * income, occupancy, A_draws)
