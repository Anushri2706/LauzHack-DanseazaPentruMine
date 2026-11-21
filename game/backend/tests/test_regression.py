"""Regression: the game engine reproduces the original backend/regime + twin engines exactly.

The original outputs were frozen into fixtures/legacy_reference.json (see make_legacy_reference.py)
before backend/ and twin/ were removed. Any drift in HMM fitting, path sampling, income, tax, the
wealth layer, decision mapping or peer distances fails these tests.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from app.engine import tax
from app.engine import wealth as game_wealth
from app.engine.accounting import OneOff
from app.engine.regime import RegimeEngine, fit_emissions
from app.engine.transitions import load_transition_priors
from app.engine.types import Intervention, Profile

from .fixtures.make_legacy_reference import digest
from .regression_cases import INTERVENTION_SETS, peer_query, profile_kwargs

REF = json.loads((Path(__file__).parent / "fixtures" / "legacy_reference.json").read_text())


def plain(x):
    """Same JSON round-trip the fixture went through (tuples -> lists, numpy scalars -> float)."""
    return json.loads(json.dumps(x, default=lambda o: list(o) if isinstance(o, tuple) else float(o)))


def _ivs(spec):
    return [Intervention(k, t0, d, **kw) for k, t0, d, kw in spec]


@pytest.fixture(scope="module")
def engine():
    return RegimeEngine(fit_emissions(), load_transition_priors())


def test_emission_fit_identical(engine):
    em = engine.emissions
    assert plain({"params": em.params, "household_shift": em.household_shift,
                  "december_bonus_quantiles": em.december_bonus_quantiles}) == REF["emissions"]


@pytest.mark.parametrize("name", sorted(INTERVENTION_SETS))
@pytest.mark.parametrize("employment", ["employed", "student", "part_time"])
def test_hmm_paths_identical(engine, name, employment):
    p = engine.simulate(Profile(**profile_kwargs(employment)), 96, 300, _ivs(INTERVENTION_SETS[name]),
                        {"job_loss_hazard": 0.006}, seed=5)
    for f, expected in REF["hmm_paths"][f"{name}/{employment}"].items():
        assert digest(getattr(p, f)) == expected, f


def test_stub_identical():
    p = RegimeEngine.stub().simulate(Profile(**profile_kwargs()), 60, 3, _ivs(INTERVENTION_SETS["masters"]))
    assert digest(p.net_flow_chf) == REF["stub_net_flow"]


def test_net_pay_identical(engine):
    params = engine.income_table.params
    g = np.linspace(0, 30000, 61)
    for kind in ("wage", "ui"):
        assert digest(tax.net_from_gross(g, 33, kind, params)) == REF["net_pay"][kind]
    assert tax.gross_from_net(5000, 30, params) == REF["net_pay"]["gross_from_net_5000_30"]


def test_wealth_layer_identical(engine):
    prof = Profile(**profile_kwargs())
    p = engine.simulate(prof, 120, 200, _ivs(INTERVENTION_SETS["masters"]), seed=9)
    w = game_wealth.build(prof, p, [OneOff(0, 8000.0), OneOff(12, 8000.0)], engine.income_table.params, 9, 0.7)
    assert digest(w.cash()) == REF["wealth"]["cash"]
    assert digest(w.net_worth_years()) == REF["wealth"]["net_worth_years"]
    assert plain(w.summary()) == REF["wealth"]["summary"]


def test_full_pipeline_matches_twin_session(engine, assumptions):
    """game run_monte_carlo == original TwinSession(wealth=True) for the same profile, decision and seed."""
    from app.engine.decisions import validate_decision
    from app.engine.profile import EngineProfile
    from app.engine.simulate import SimulationConfig, run_monte_carlo

    ep = EngineProfile(Profile(**profile_kwargs()), 6100.0, 0.0, {}, [], [])
    d = validate_decision("masters", 6, {"duration_months": 24, "annual_tuition_chf": 9000},
                          initial_state="employed", horizon_months=120, assumptions=assumptions)
    run = run_monte_carlo(ep, [d], assumptions, SimulationConfig(120, 250), 250, 3, engine)
    assert digest(run.cash) == REF["pipeline"]["cash"]
    np.testing.assert_allclose(run.net_worth_real[:, 11::12], np.array(REF["pipeline"]["net_worth_years"]),
                               rtol=0, atol=1e-6)


def test_peer_distance_matches_twin():
    from app.peers.bands import distance, population
    pop = {p["id"]: p for p in population()}
    me = peer_query()
    for q in REF["peers"]:
        n = pop[q["id"]]
        assert (n["wealth"], n["income"], n["rent"]) == (q["wealth"], q["income"], q["rent"])
        assert distance(me, n) == q["distance"]
