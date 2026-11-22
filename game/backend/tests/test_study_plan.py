"""A starting student plans when they expect to finish; at that month they are asked how it went."""
import pytest
from fastapi.testclient import TestClient

from app.engine.game import Action as A, InvalidAction, Turn as T, run_turns
from app.main import app
from tests.conftest import make_params

client = TestClient(app)
PLAN = lambda n: A("plan_study", None, float(n))  # noqa: E731
WORK = A("life_choice", "start_work")


def _status(rows, months):
    return [rows[m]["status"] for m in months]


def test_without_a_plan_nothing_changes():
    p = make_params()
    rows = run_turns(p, []).rows
    assert set(r["status"] for r in rows) == {"studying"}


def test_no_answer_means_finished_as_planned_then_work():
    p = make_params()
    run = run_turns(p, [T(0, (PLAN(18),))])
    assert _status(run.rows, (17, 18, 30)) == ["studying", "working", "working"]
    assert run.rows[18]["income"] == pytest.approx(p["salary_work"])
    kinds = {(x["month"], x["kind"]) for x in run.letters}
    assert (0, "study_planned") in kinds and (18, "degree") in kinds
    assert run.rows[18]["education"] == "bachelor"


def test_keep_studying_then_asked_again_and_no_tuition():
    p = make_params()
    run = run_turns(p, [T(0, (PLAN(12),)), T(12, (A("life_choice", "extend_study", 6.0),))])
    assert _status(run.rows, (12, 17, 18)) == ["studying", "studying", "working"]
    assert all(r["one_offs"] == 0 for r in run.rows[:24])


def test_answers_at_the_planned_end():
    p = make_params()
    done = run_turns(p, [T(0, (PLAN(12),)), T(12, (A("life_choice", "finish_study"),))]).rows
    assert done[12]["status"] == "working" and done[12]["education"] == "bachelor"
    out = run_turns(p, [T(0, (PLAN(12),)), T(12, (A("life_choice", "quit_school"),))]).rows
    assert out[12]["status"] == "unemployed" and out[12]["education"] == "none"
    master = run_turns(p, [T(0, (PLAN(12),)), T(12, (A("life_choice", "master_ft"),))]).rows
    assert master[12]["status"] == "studying" and master[12]["education"] == "master_studying"
    straight = run_turns(p, [T(0, (PLAN(12), A("life_choice", "master_ft")))])
    assert straight.final_state.study_end is None and straight.rows[12]["status"] == "studying"


def test_a_plan_can_be_changed_before_the_end():
    p = make_params()
    rows = run_turns(p, [T(0, (PLAN(12),)), T(6, (PLAN(24),))]).rows
    assert _status(rows, (12, 29, 30)) == ["studying", "studying", "working"]


@pytest.mark.parametrize("turns, match", [
    ([T(0, (WORK,)), T(3, (PLAN(6),))], "Only while you study"),
    ([T(0, (A("life_choice", "master_ft"),)), T(3, (PLAN(6),))], "master's already has a planned end"),
    ([T(0, (PLAN(0),))], "whole months"),
    ([T(0, (PLAN(12),)), T(12, (PLAN(6),))], "say how it went"),
])
def test_invalid_plans(turns, match):
    with pytest.raises(InvalidAction, match=match):
        run_turns(make_params(), turns)


def test_api_asks_checks_in_and_feeds_the_twin():
    base = {"world": 9, "seed": 42}
    s = client.post("/game/state", json={**base, "turns": [], "current_month": 0}).json()
    assert s["study_plan"]["planned_end"] is None
    assert any(a["type"] == "plan_study" for a in s["actions"]["university"])
    plan = [{"month": 0, "actions": [{"type": "plan_study", "value": 18}]}]
    s = client.post("/game/state", json={**base, "turns": plan, "current_month": 18}).json()
    assert s["study_checkin"]["program"] == "bachelor" and s["study_checkin"]["planned_end"] == 18
    ids = [a["id"] for a in s["actions"]["university"]]
    assert "finish_study" in ids and "extend_study" in ids and "quit_school" in ids
    twin = client.post("/twin/fan", json={**base, "turns": plan, "current_month": 6, "n": 10}).json()
    assert twin["assumed_degree_after_months"] == 12
