"""Engine behaviour: determinism, purity, reconciliation, constraints, horizon, events, percentiles."""
import copy

import numpy as np
import pytest

from app.engine.decisions import validate_decision
from app.engine.events import (CATALOGUE as EVENTS, HMM_OWNED_RISKS, EventDefinition, annual_hazard_to_monthly_probability,
                               annual_to_monthly_probability, monthly_to_annual_probability, sample_events,
                               validate_definition)
from app.engine.montecarlo import summarise
from app.engine.simulate import SimulationConfig, SimulationError, run_branch, run_monte_carlo
from app.engine.types import STATE_INDEX, STATES

CFG = SimulationConfig(horizon_months=120, n_paths=300)


def run(ep, A, eng, decisions=(), seed=1, n=300, cfg=CFG):
    return run_monte_carlo(ep, list(decisions), A, cfg, n, seed, eng)


def dec(A, kind, t0=0, initial="employed", **params):
    return validate_decision(kind, t0, params, initial_state=initial, horizon_months=CFG.horizon_months,
                             assumptions=A, current_canton="VD")


def test_same_seed_identical(engine_profile, assumptions, hmm_engine):
    a, b = run(engine_profile, assumptions, hmm_engine), run(engine_profile, assumptions, hmm_engine)
    assert a.fingerprint() == b.fingerprint()
    assert summarise(a) == summarise(b)


def test_different_seeds_differ(engine_profile, assumptions, hmm_engine):
    a, b = run(engine_profile, assumptions, hmm_engine, seed=1), run(engine_profile, assumptions, hmm_engine, seed=2)
    assert not np.array_equal(a.cash, b.cash)


def test_caller_state_not_mutated(engine_profile, assumptions, hmm_engine):
    ep_before = copy.deepcopy(engine_profile)
    d = dec(assumptions, "masters", 3)
    d_before = d.to_dict()
    overrides_before = dict(assumptions.overrides)
    run(engine_profile, assumptions, hmm_engine, [d])
    assert engine_profile.profile == ep_before.profile
    assert d.to_dict() == d_before
    assert dict(assumptions.overrides) == overrides_before


def test_cash_and_net_worth_reconcile(engine_profile, assumptions, hmm_engine):
    r = run(engine_profile, assumptions.with_overrides({"keep_factor": 0.5}), hmm_engine,
            [dec(assumptions, "one_off", 7, amount_chf=4000)])
    assert r.reconciles()
    np.testing.assert_allclose(r.net_worth_nominal, r.cash + r.portfolio - r.debt)
    np.testing.assert_allclose(r.net_worth_real * r.price, r.net_worth_nominal)
    # month-by-month: cash_t - cash_{t-1} == named flow_t
    diffs = np.diff(np.hstack([np.full((r.n_paths, 1), r.opening_cash), r.cash]), axis=1)
    np.testing.assert_allclose(diffs, r.total_flow(), atol=1e-6)


def test_debt_and_portfolio_constraints(engine_profile, assumptions, hmm_engine):
    r = run(engine_profile, assumptions, hmm_engine)
    assert r.debt == 10000.0                      # interest-only: principal constant
    assert r.loan_interest == pytest.approx(10000 * 0.04 / 12)
    assert np.all(r.portfolio > 0)                # multiplicative returns keep a positive portfolio positive
    assert np.all(r.price > 0)


def test_transition_matrices_valid(hmm_engine):
    pri = hmm_engine.priors
    for age in (18, 22, 30, 45, 60, 64):
        A = pri.matrix({}, age=age)
        assert np.all(A >= 0) and np.allclose(A.sum(axis=1), 1)
    draws = pri.sample(np.random.default_rng(0), 50)
    assert np.all(draws >= 0) and np.allclose(draws.sum(axis=2), 1)


def test_horizon_respected(engine_profile, assumptions, hmm_engine):
    cfg = SimulationConfig(horizon_months=36, n_paths=50)
    r = run_monte_carlo(engine_profile, [], assumptions, cfg, 50, 0, hmm_engine)
    assert r.cash.shape == (50, 36)
    s = summarise(r)
    assert len(s["monthly"]["cash"]["p50"]) == 36
    with pytest.raises(Exception):
        validate_decision("one_off", 36, {"amount_chf": 1}, initial_state="employed", horizon_months=36,
                          assumptions=assumptions)
    with pytest.raises(SimulationError):
        SimulationConfig(horizon_months=5).validate()


def test_percentiles_ordered(engine_profile, assumptions, hmm_engine):
    s = summarise(run(engine_profile, assumptions, hmm_engine))
    for m in ("cash", "net_worth_nominal", "net_worth_real"):
        band = s["monthly"][m]
        assert all(a <= b <= c for a, b, c in zip(band["p10"], band["p50"], band["p90"]))
        f = s["final"][m]
        assert f["p10"] <= f["p50"] <= f["p90"]
    assert s["n_paths"] == 300


def test_job_loss_and_one_off_same_month(engine_profile, assumptions, hmm_engine):
    """A forced job search starting in month 10 and a one-off in month 10 are both charged in month 10."""
    base = run(engine_profile, assumptions, hmm_engine, [dec(assumptions, "job_search", 10, duration_months=3)], n=50)
    both = run(engine_profile, assumptions, hmm_engine,
               [dec(assumptions, "job_search", 10, duration_months=3), dec(assumptions, "one_off", 10, amount_chf=5000)], n=50)
    assert np.all(both.states[:, 10:13] == STATE_INDEX["job_search"])
    np.testing.assert_array_equal(base.states, both.states)            # a one-off does not touch the HMM
    np.testing.assert_allclose(both.one_off_costs[:, 10], 5000 * both.price[:, 10])
    np.testing.assert_allclose(base.cash[:, 9], both.cash[:, 9])
    np.testing.assert_allclose(base.cash[:, 10] - both.cash[:, 10], 5000 * both.price[:, 10])
    assert both.reconciles()


def test_run_branch_single_trace(engine_profile, assumptions, hmm_engine):
    r = run_branch(engine_profile, [dec(assumptions, "masters", 0)], assumptions, CFG, 4, hmm_engine)
    tr = r.trace(0)
    assert len(tr) == CFG.horizon_months
    assert tr[0]["regime"] == "student" and tr[24]["regime"] == "employed"
    assert tr[0]["one_off_costs_chf"] == pytest.approx(8000 * tr[0]["price_level"])


# ----------------------------------------------------------------------------- events

def test_time_basis_conversions():
    assert annual_to_monthly_probability(0.0) == 0.0
    assert monthly_to_annual_probability(annual_to_monthly_probability(0.1)) == pytest.approx(0.1)
    assert annual_hazard_to_monthly_probability(0.12) == pytest.approx(1 - np.exp(-0.01))
    with pytest.raises(ValueError):
        annual_to_monthly_probability(1.5)


def test_events_cannot_redeclare_hmm_risks():
    for risk in HMM_OWNED_RISKS:
        bad = EventDefinition("x", "dup", risk, "annual_probability", "event_major_expense_annual_prob",
                              "one_off_cost", "event_major_expense_cost_chf")
        with pytest.raises(ValueError, match="HMM-owned"):
            validate_definition(bad)
    bad = EventDefinition("y", "forces a regime", "other", "annual_probability", "event_major_expense_annual_prob",
                          "force_state", "event_major_expense_cost_chf")
    with pytest.raises(ValueError):
        validate_definition(bad)
    assert all(e.risk not in HMM_OWNED_RISKS for e in EVENTS.values())


def test_enabling_events_does_not_change_regimes(engine_profile, assumptions, hmm_engine):
    """No double counting of unemployment: events never touch the regime path or its job-search months."""
    off = run(engine_profile, assumptions, hmm_engine)
    on = run_monte_carlo(engine_profile, [], assumptions,
                         SimulationConfig(120, 300, enabled_events=("unexpected_major_expense",)), 300, 1, hmm_engine)
    np.testing.assert_array_equal(off.states, on.states)
    np.testing.assert_array_equal(off.net_income, on.net_income)
    assert on.event_costs.sum() > 0 and off.event_costs.sum() == 0
    assert on.reconciles()


def test_event_frequency_matches_annual_probability(assumptions):
    states = np.zeros((20000, 120), dtype=int)
    ev = sample_events(["unexpected_major_expense"], states, 30.0, assumptions, seed=0)
    per_year = ev.counts["unexpected_major_expense"].mean() / 10
    p_m = annual_to_monthly_probability(0.10)
    assert per_year == pytest.approx(12 * p_m, rel=0.03)


def test_event_eligibility_window(assumptions):
    from dataclasses import replace
    import app.engine.events as events
    ev = replace(EVENTS["unexpected_major_expense"], id="tmp_window", eligible_states=("student",), window=(12, 24))
    events.CATALOGUE["tmp_window"] = ev
    try:
        states = np.full((500, 48), STATE_INDEX["employed"])
        states[:250] = STATE_INDEX["student"]
        d = sample_events(["tmp_window"], states, 25.0, assumptions.with_overrides({"event_major_expense_annual_prob": 1.0}), 0)
        assert d.costs[250:].sum() == 0                     # wrong state: never
        assert d.costs[:250, :12].sum() == 0 and d.costs[:250, 24:].sum() == 0
        assert np.all(d.costs[:250, 12:24] > 0)             # p_annual = 1 -> every eligible month
    finally:
        del events.CATALOGUE["tmp_window"]


def test_assumptions_used_reflect_defaults_actually_used(engine_profile, assumptions, hmm_engine):
    explicit = run(engine_profile, assumptions, hmm_engine, [dec(assumptions, "masters", 0, annual_tuition_chf=1460)], n=20)
    default = run(engine_profile, assumptions, hmm_engine, [dec(assumptions, "masters", 0)], n=20)
    assert "masters_annual_tuition_chf" not in explicit.assumption_ids
    assert "masters_annual_tuition_chf" in default.assumption_ids
    assert "masters_premium_pct" in explicit.assumption_ids
