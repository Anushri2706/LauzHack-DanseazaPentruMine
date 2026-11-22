"""Study plan and check-in: the player says when they expect to finish a master's; at that date they
are asked whether they finished, want to keep studying, or drop out."""
import pytest
from fastapi.testclient import TestClient

from app.engine.game import Action as A, InvalidAction, Turn as T, available_actions, run_turns, state_at
from app.main import app
from tests.conftest import make_params

client = TestClient(app)


def ft(months=None):
    return A("life_choice", "master_ft", None if months is None else float(months))


def tuition(run):
    return sum(r["one_offs"] for r in run.rows)  # tuition is the only one-off in these turns


def test_default_plan_is_unchanged():
    p = make_params()
    a = run_turns(p, [T(0, (ft(),))])
    b = run_turns(p, [T(0, (ft(p["master_ft_months"]),))])
    assert [r["net_worth"] for r in a.rows] == pytest.approx([r["net_worth"] for r in b.rows])


def test_planned_length_sets_end_date_and_tuition():
    p = make_params()
    dur = int(p["master_ft_months"])
    s = state_at(p, [T(0, (ft(18),))], 0)
    assert s.program_end == 18
    run = run_turns(p, [T(0, (ft(18),))])
    assert tuition(run) == pytest.approx(p["tuition_annual"] * dur / 12 * 18 / dur)
    with pytest.raises(InvalidAction, match="Planned length"):
        run_turns(p, [T(0, (ft(100),))])


def test_checkin_waits_for_an_answer_and_defaults_to_finished():
    p = make_params()
    due = state_at(p, [T(0, (ft(18),))], 18)
    assert due.program == "master_ft" and due.program_end == 18
    ids = {a["id"]: a for a in available_actions(due, 18, p)["university"]}
    assert ids["finish_study"]["enabled"] and ids["quit_school"]["enabled"]
    assert any(a["id"] == "extend_study" and a["enabled"] for a in available_actions(due, 18, p)["university"])
    after = state_at(p, [T(0, (ft(18),))], 19)  # no answer: finished as planned, starts a job
    assert after.has_master and after.status == "working"


def test_finished_answer():
    p = make_params()
    s = state_at(p, [T(0, (ft(18),)), T(18, (A("life_choice", "finish_study"),))], 19)
    assert s.has_master and s.status == "working" and s.salary == pytest.approx(p["salary_master"])


def test_keep_studying_moves_the_date_and_asks_again():
    p = make_params()
    turns = [T(0, (ft(18),)), T(18, (A("life_choice", "extend_study", 6.0),))]
    s = state_at(p, turns, 18)
    assert s.program_end == 24 and not s.has_master
    run = run_turns(p, turns)
    base = run_turns(p, [T(0, (ft(18),))])
    assert tuition(run) - tuition(base) == pytest.approx(p["tuition_annual"] / 12 * 6)
    assert state_at(p, turns, 24).program_end == 24  # asked again at the new date
    assert state_at(p, turns, 25).has_master
    with pytest.raises(InvalidAction, match="planned end"):  # only at the check-in
        run_turns(p, [T(0, (ft(18),)), T(10, (A("life_choice", "extend_study", 6.0),))])


def test_drop_out_at_checkin_keeps_bachelor():
    p = make_params()
    s = state_at(p, [T(0, (ft(18),)), T(18, (A("life_choice", "quit_school"),))], 19)
    assert not s.has_master and not s.no_degree and s.status == "unemployed"


def test_other_big_move_counts_as_finished():
    p = make_params()
    s = state_at(p, [T(0, (ft(18),)), T(18, (A("life_choice", "move_abroad"),))], 19)
    assert s.has_master and s.status == "abroad"


def test_api_reports_the_checkin():
    req = {"persona_id": "P00001", "turns": [{"month": 0, "actions": [{"type": "life_choice", "id": "master_ft", "value": 12}]}],
           "current_month": 12}
    st = client.post("/game/state", json=req).json()
    assert st["study_checkin"]["planned_end"] == 12 and not st["study_checkin"]["answered"]
    assert st["study_checkin"]["months_studied"] == 12
    req["turns"].append({"month": 12, "actions": [{"type": "life_choice", "id": "finish_study"}]})
    assert client.post("/game/state", json=req).json()["study_checkin"]["answered"]
    req["current_month"] = 5
    req["turns"] = req["turns"][:1]
    assert client.post("/game/state", json=req).json()["study_checkin"] is None
