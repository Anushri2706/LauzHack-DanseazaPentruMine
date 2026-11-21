"""Branching: inheritance, isolation, replay, staleness, comparability, break-even through branches."""
import pytest

from app.engine.decisions import DecisionError
from app.engine.montecarlo import ResultError
from app.engine.simulate import SimulationConfig
from app.game_service import GameService
from app.storage import Conflict, NotFound


@pytest.fixture
def svc():
    return GameService()


@pytest.fixture
def game(svc, player_raw):
    return svc.start(player_raw, SimulationConfig(horizon_months=72, n_paths=120), seed=11)


def test_child_inherits_parent(svc, game):
    a = svc.create_branch(game.id, game.root_id, "A")
    svc.add_decision(game.id, a.id, "masters", 0, {})
    a1 = svc.create_branch(game.id, a.id, "A1")
    eff = svc.effective_decisions(game, a1.id)
    assert [d.decision_id for d in eff] == ["masters"]
    assert a1.fork_month == 0
    with pytest.raises(DecisionError):        # negative start month
        svc.add_decision(game.id, a1.id, "one_off", -1, {"amount_chf": 10})


def test_child_changes_do_not_affect_parent_or_sibling(svc, game):
    root = game.root_id
    a = svc.create_branch(game.id, root, "A")
    b = svc.create_branch(game.id, root, "B")
    svc.add_decision(game.id, a.id, "masters", 0, {})
    svc.add_decision(game.id, b.id, "continue_employment", 0, {})
    r_root, r_a, r_b = (svc.simulate(game.id, x) for x in (root, a.id, b.id))
    before = {x: svc.repo.branch(game.id, x) for x in (root, a.id, b.id)}
    a1 = svc.create_branch(game.id, a.id, "A1", fork_month=30)
    svc.add_decision(game.id, a1.id, "one_off", 30, {"amount_chf": 20000})
    svc.simulate(game.id, a1.id)
    for x in (root, a.id, b.id):
        assert svc.repo.branch(game.id, x) == before[x]
        res = svc.latest(game.id, x)
        assert not svc.is_stale(game, res)
    for x, r in ((root, r_root), (a.id, r_a), (b.id, r_b)):
        assert svc.latest(game.id, x).summary["fingerprint"] == r.summary["fingerprint"]


def test_decision_on_parent_invalidates_descendants_only(svc, game):
    a = svc.create_branch(game.id, game.root_id, "A")
    b = svc.create_branch(game.id, game.root_id, "B")
    a1 = svc.create_branch(game.id, a.id, "A1")
    for x in (a.id, b.id, a1.id):
        svc.simulate(game.id, x)
    svc.add_decision(game.id, a.id, "one_off", 3, {"amount_chf": 500})
    assert svc.is_stale(game, svc.latest(game.id, a.id))
    assert svc.is_stale(game, svc.latest(game.id, a1.id))
    assert not svc.is_stale(game, svc.latest(game.id, b.id))


def test_assumption_override_invalidates_and_versions(svc, game):
    a = svc.create_branch(game.id, game.root_id, "A")
    r1 = svc.simulate(game.id, a.id)
    svc.set_overrides(game.id, a.id, {"masters_premium_pct": 0.2})
    assert svc.is_stale(game, r1)
    r2 = svc.simulate(game.id, a.id)
    assert r2.assumptions_version != r1.assumptions_version and r2.version == 2
    assert svc.latest(game.id, a.id, version=1) is r1          # never overwritten


def test_fixed_seed_replay(svc, game):
    a = svc.create_branch(game.id, game.root_id, "A")
    svc.add_decision(game.id, a.id, "masters", 0, {"annual_tuition_chf": 4000})
    svc.simulate(game.id, a.id)
    rep = svc.replay(game.id, a.id)
    assert rep["identical"] is True


def test_fresh_seed_is_labelled_and_not_comparable(svc, game):
    a = svc.create_branch(game.id, game.root_id, "A")
    svc.simulate(game.id, game.root_id)
    fresh = svc.simulate(game.id, a.id, fresh_seed=game.seed + 1)
    assert fresh.run_kind == "fresh_seed"
    with pytest.raises(ResultError):
        svc.compare(game.id, game.root_id, a.id)


def test_compare_requires_fresh_compatible_results(svc, game):
    a = svc.create_branch(game.id, game.root_id, "A")
    with pytest.raises(Conflict):
        svc.compare(game.id, game.root_id, a.id)
    svc.add_decision(game.id, a.id, "masters", 0, {})
    svc.simulate(game.id, game.root_id)
    svc.simulate(game.id, a.id)
    out = svc.compare(game.id, game.root_id, a.id)
    d = out["differences"]["net_worth_real.final.p50"]
    assert d["difference_b_minus_a"] == pytest.approx(d["b"] - d["a"], abs=0.01)
    assert d["direction"] in ("b_higher", "b_lower", "equal")
    svc.add_decision(game.id, a.id, "one_off", 5, {"amount_chf": 1})
    with pytest.raises(Conflict, match="stale"):
        svc.compare(game.id, game.root_id, a.id)


def test_not_found(svc, game):
    with pytest.raises(NotFound):
        svc.repo.branch(game.id, "br_missing")
    with pytest.raises(NotFound):
        svc.repo.game("game_missing")


def test_break_even_bracketed_and_unbracketed(svc, player_raw):
    g = svc.start(player_raw, SimulationConfig(horizon_months=120, n_paths=200), seed=11)
    a = svc.create_branch(g.id, g.root_id, "A")
    svc.add_decision(g.id, a.id, "masters", 0, {})
    hit = svc.break_even(g.id, a.id, g.root_id, "masters_premium_pct", 0.0, 1.0)
    assert hit["status"] == "converged" and 0.0 < hit["value"] < 1.0
    assert hit["f_lower"] < 0 < hit["f_upper"]
    miss = svc.break_even(g.id, a.id, g.root_id, "masters_premium_pct", 0.0, 0.05)
    assert miss["status"] == "no_break_even_in_range" and miss["value"] is None
