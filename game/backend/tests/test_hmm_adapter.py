"""Adapter between town actions and the regime engine. Town routes stay as they are."""
import ast
import inspect
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.engine.game import Action
from app.engine.hmm_adapter import (
    advance_year, branch_view, commit, commit_net_flows, life_distribution, sample_future,
    start_snapshot, town_action_interventions,
)
from app.main import app

ENGINE = Path(__file__).resolve().parents[1] / "app" / "engine"


def test_import_lock_and_unimplemented_seam():
    import app.engine.decisions as decisions
    import app.engine.game as game
    import app.engine.hmm_adapter as adapter
    import app.engine.simulate as simulate

    assert adapter.advance_year and decisions and game and simulate
    assert "step_month(" not in inspect.getsource(adapter)
    for name in ("simulate.py", "game.py", "decisions.py"):
        tree = ast.parse((ENGINE / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all("twin" not in alias.name for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                assert node.module is None or "twin" not in node.module
    with pytest.raises(NotImplementedError):
        commit_net_flows(None)


def test_master_ft_maps_and_other_town_actions_are_rejected():
    from twin.twin_session import DECISIONS, decision_interventions

    ivs, offs = town_action_interventions(Action("life_choice", "master_ft"), 0)
    expected, expected_offs = decision_interventions("masters", 0, {})
    assert [(i.kind, i.state) for i in ivs] == [("force_state", "student"), ("set_attribute", None)]
    assert ivs[1].attribute == ("has_masters", True)
    assert [i.kind for i in ivs] == [i.kind for i in expected]
    assert ivs[0].state == expected[0].state
    assert len(offs) == len(expected_offs)
    assert "masters" in DECISIONS
    for action in (
        Action("life_choice", "master_pt"),
        Action("buy_home", "studio"),
        Action("life_choice", "university"),
    ):
        with pytest.raises(ValueError):
            town_action_interventions(action, 0)


def test_year_advance_is_one_twelve_month_simulate(monkeypatch):
    from twin.data import load_profile
    from twin.regime_engine import RegimeEngine
    from twin.types import DecisionSnapshot

    calls = []
    real = RegimeEngine.simulate

    def wrapped(self, profile, horizon_months=60, n_paths=5000, interventions=None,
                assumption_overrides=None, initial_belief=None, seed=0):
        calls.append(horizon_months)
        return real(self, profile, horizon_months, n_paths, interventions,
                    assumption_overrides, initial_belief, seed)

    monkeypatch.setattr(RegimeEngine, "simulate", wrapped)
    engine = RegimeEngine.stub()
    profile = load_profile("P00001")
    snap = DecisionSnapshot(
        month=0, profile=profile, belief=engine.filter(profile, [1.0]),
        observed_net_flows=[1.0], interventions=[], one_offs=[],
        assumption_overrides={}, seed=0, parent_id=None, snapshot_id="root",
    )
    advance_year(engine, snap, "start_working", n_paths=2)
    assert calls == [12]


def test_branch_does_not_grow_the_observation_list():
    from twin.data import load_profile
    from twin.regime_engine import RegimeEngine
    from twin.types import DecisionSnapshot
    from app.engine.hmm_adapter import snapshot_to_dict

    engine = RegimeEngine.stub()
    profile = load_profile("P00001")
    flows = [1.0, 2.0, 3.0]
    snap = DecisionSnapshot(
        month=0, profile=profile, belief=engine.filter(profile, flows),
        observed_net_flows=list(flows), interventions=[], one_offs=[],
        assumption_overrides={}, seed=0, parent_id=None, snapshot_id="root",
    )
    raw = snapshot_to_dict(snap, label="Start")
    view = branch_view(raw, ["masters", "start_working"], n_paths=2, engine=engine)
    assert len(raw["observed_net_flows"]) == 3
    assert view["committed"] is False
    assert {f["decision"] for f in view["forks"]} == {"masters", "start_working"}
    assert all(f["horizon_months"] == 12 for f in view["forks"])
    assert all("mortgage" not in f for f in view["forks"])


def test_commit_injected_flows_then_distribution_and_jump():
    """Hand-built flows go through the seam. The production reduction stays unimplemented."""
    from twin import metrics
    from twin.data import load_profile
    from app.engine.hmm_adapter import get_engine

    client = TestClient(app)
    started = client.post("/hmm/start", json={"persona_id": "P00001"})
    assert started.status_code == 200, started.text
    root = started.json()["snapshot"]
    before = root["belief"]["employed"]
    n = len(root["observed_net_flows"])

    branched = client.post("/hmm/branch", json={
        "snapshot": root, "decisions": ["masters", "start_working"], "n_paths": 4,
    })
    assert branched.status_code == 200, branched.text
    body = branched.json()
    assert body["committed"] is False
    assert len(root["observed_net_flows"]) == n
    labels = {f["label"] for f in body["forks"]}
    assert "Master's pin (HMM)" in labels and "Start working (HMM)" in labels

    alexia = load_profile("P00001")
    employed_net = get_engine().income_table.params["entry_salary_chf"] - alexia.rent * 1.75
    committed = client.post("/hmm/commit", json={
        "snapshot": root, "flows": [employed_net] * 6, "decision": "start_working", "label": "Work",
    })
    assert committed.status_code == 200, committed.text
    child = committed.json()
    assert child["parent_id"] == "root"
    assert len(child["observed_net_flows"]) == n + 6
    assert child["belief"]["employed"] > before
    assert "mortgage" not in child

    jumped = client.post("/hmm/restore", json={"snapshot": root})
    assert jumped.status_code == 200
    assert jumped.json()["matches"] is True
    assert jumped.json()["snapshot_id"] == "root"

    dist = client.post("/hmm/distribution", json={"snapshot": child, "horizon_months": 8, "n_paths": 4})
    assert dist.status_code == 200, dist.text
    payload = dist.json()
    assert len(payload["p50"]) == 8
    assert payload["disclaimer"] == metrics.DISCLAIMER
    assert len(payload["debt"]) == 8 and len(payload["investments"]) == 8
    assert "mortgage" not in payload

    direct = life_distribution(child, horizon_months=8, n_paths=4)
    assert len(direct["p50"]) == 8
    kept = commit(root, [employed_net] * 6, decision="start_working")
    assert kept["belief"]["employed"] > before


def test_sample_future_continues_lived_totals_without_town_coins():
    from twin.regime_engine import RegimeEngine

    lived = {
        "cash": 12345.0, "debt": 800.0, "investments": 500.0,
        "status": "working", "program": None, "program_end": None,
        "has_master": False, "work_share": 1.0,
        "home_value": 450_000.0, "mortgage": 300_000.0, "home_id": "studio",
        "retiree": False,
    }
    drawn = sample_future("P00001", month=12, horizon=18, lived=lived, seed=3, n_paths=2,
                          engine=RegimeEngine.stub())
    assert drawn["cash"].shape == (2, 6)
    assert drawn["events"][0]["kind"] == "hmm"
    assert "seed 3" in drawn["events"][0]["text"]
    text = drawn["events"][0]["text"].lower()
    assert "market" not in text and "rent rise" not in text
    # Opening cash is the lived balance, and the house is equity, not a new mortgage.
    assert drawn["debt"][0, 0] == pytest.approx(800.0)
    assert drawn["investments"][0, 0] >= 500.0
    equity = 450_000.0 - 300_000.0
    assert drawn["net_worth"][0, 0] == pytest.approx(
        drawn["cash"][0, 0] + drawn["investments"][0, 0] - drawn["debt"][0, 0] + equity
    )
    assert "make_worlds(" not in inspect.getsource(sample_future)
