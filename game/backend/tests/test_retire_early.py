"""Retiring early: the life choice (engine) and the "retire early at age X" goal (what-if replay)."""
import pytest
from fastapi.testclient import TestClient

from app.engine.game import Action as A, InvalidAction, Turn as T, run_turns
from app.engine.metrics import check_goals, retire_goal_on_path
from app.main import app
from tests.conftest import make_params

client = TestClient(app)
WORK, RETIRE = A("life_choice", "start_work"), A("life_choice", "retire_early")
LIFE = dict(horizon_months=1200)


def goal(age, until=None, gid=None, start_age=20):
    return {"goal_id": gid or f"retire-{age}", "name": f"Retire at {age}", "target": 0,
            "month": int((age - start_age) * 12), "priority": "high", "kind": "retire_early", "until_age": until}


def test_retire_early_stops_income_until_the_pension():
    p = make_params(**LIFE)
    run = run_turns(p, [T(0, (WORK,)), T(420, (RETIRE,))])   # stop at 55
    before, at, pension = run.rows[419], run.rows[420], run.rows[540]
    assert before["status"] == "working" and at["status"] == "retired"
    assert at["income"] == 0 and run.rows[539]["income"] == 0
    # Pension from 65, based on the full salary at 55, 10 years × the cut lower.
    salary = run_turns(p, [T(0, (WORK,))], until=420).final_state.salary
    expect = salary * p["pension_replacement_rate"] * (1 - 10 * p["early_pension_cut_per_year"])
    assert pension["income"] == pytest.approx(expect, rel=1e-9)
    assert [x["kind"] for x in run.letters if x["kind"] in ("retired_early", "pension_starts")] == \
        ["retired_early", "pension_starts"]
    assert not any(x["kind"] == "retired" for x in run.letters)   # no second, normal retirement


def test_retire_early_lowers_living_costs_like_retirement():
    p = make_params(**LIFE, inflation_annual=0)
    run = run_turns(p, [T(0, (WORK,)), T(420, (RETIRE,))])
    assert run.rows[420]["expenses"] == pytest.approx(
        p["rent"] + (run.rows[419]["expenses"] - p["rent"]) * p["retirement_spending_factor"])


def test_retire_early_not_allowed_while_studying_or_twice():
    p = make_params(**LIFE)
    with pytest.raises(InvalidAction, match="school"):
        run_turns(p, [T(0, (RETIRE,))])                     # Alexia starts as a student
    with pytest.raises(InvalidAction, match="already retired"):
        run_turns(p, [T(0, (WORK,)), T(300, (RETIRE,)), T(400, (RETIRE,))])


def test_goal_on_a_path_that_keeps_working_is_not_met():
    rows = run_turns(make_params(**LIFE), [T(0, (WORK,))]).rows
    g = check_goals([goal(55)], rows)[0]
    assert g["status"] == "not_met" and g["reason"].startswith("Still working")


def test_goal_on_a_lived_early_retirement():
    rows = run_turns(make_params(**LIFE), [T(0, (WORK,)), T(420, (RETIRE,))]).rows
    rich = retire_goal_on_path(goal(55, until=60), rows)
    assert rich["status"] == "met" and rich["runs_out_age"] is None
    poor = retire_goal_on_path(goal(55, until=119), rows)
    assert poor["status"] == "not_met" and 60 < poor["runs_out_age"] < 119


def test_savings_goals_unchanged_by_retirement_goals():
    rows = run_turns(make_params(**LIFE), [T(0, (WORK,))]).rows
    saving = {"goal_id": "s", "name": "Fund", "target": 10_000, "month": 60, "priority": "high"}
    assert check_goals([saving], rows) == check_goals([saving, goal(55)], rows)[:1]


def _projection(goals, current=12, n=40):
    body = {"world": 9, "seed": 42, "current_month": current, "n": n, "goals": goals,
            "turns": [{"month": 0, "actions": [{"type": "life_choice", "id": "start_work"}]}]}
    r = client.post("/game/projection", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_projection_answers_what_if_and_earliest_age():
    out = _projection([goal(45, gid="early"), goal(60, gid="late")])
    by = {g["goal_id"]: g for g in out["goals"]}
    early, late = by["early"], by["late"]
    assert early["what_if"] and early["status"] == "not_met" and early["runs_out_age"] is not None
    assert late["what_if"] and late["status"] == "met"
    e = early["earliest_age"]
    assert e is not None and 45 < e <= 60 and late["earliest_age"] == e
    # The earliest age works and the year before doesn't (same calm path).
    at = {g["goal_id"]: g for g in _projection([goal(e, gid="e"), goal(e - 1, gid="e1")])["goals"]}
    assert at["e"]["status"] == "met" and at["e1"]["status"] == "not_met"
    share = out["goal_met_share"]
    assert 0 <= share["early"] <= share["late"] <= 1


def test_a_goal_is_a_check_not_a_choice():
    with_goal = _projection([goal(55)])
    without = _projection([])
    assert with_goal["bands"] == without["bands"] and with_goal["no_surprises"] == without["no_surprises"]


def test_compare_reports_retirement_goals():
    turns = [{"month": 0, "actions": [{"type": "life_choice", "id": "start_work"}]}]
    req = {"world": 9, "seed": 42, "current_month": 12, "turns": turns, "goals": [goal(58)]}
    r = client.post("/compare", json={"a": req, "b": req, "name_a": "A", "name_b": "B"}).json()
    row = next(x for x in r["table"] if x["key"] == "goals_met")
    assert row["a"] == row["b"]


def test_until_age_is_validated():
    body = {"world": 9, "current_month": 0, "turns": [], "goals": [goal(55, until=20)]}
    assert client.post("/game/projection", json=body).status_code == 422
