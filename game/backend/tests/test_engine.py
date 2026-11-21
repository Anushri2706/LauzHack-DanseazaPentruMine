"""Validation tests from evaluation/technical-validation.md (numbering follows BUILD_PROMPT §9)."""
import pytest

from app.engine.decisions import InvalidDecision
from app.engine.finance import annuity_payment, monthly_compound_rate
from app.engine.metrics import check_goals, summarize
from app.engine.simulate import initial_state, run_branch, step_month
from app.engine.state import Decision
from tests.conftest import PLAIN, make_params

D = Decision


# 1. Cash identity --------------------------------------------------------------------------
def test_cash_identity_single_month():
    # 8,000 opening + 1,200 income − 1,800 expenses = 7,400
    p = make_params(**PLAIN, rent=900, living_studying=900, student_income=1200)
    _, row = step_month(initial_state(p), p)
    assert row["income"] == 1200 and row["expenses"] == 1800
    assert row["cash"] == pytest.approx(7400)


def test_cash_identity_every_month(params):
    run = run_branch(params, [D(0, "master_ft"), D(24, "move_abroad")])
    prev = params["opening_cash"]
    for r in run.rows:
        expected = (prev + r["income"] - r["expenses"] - r["one_offs"] - r["debt_payment"]
                    - r["invested"] + r["sold_investments"])
        assert r["cash"] == pytest.approx(expected, abs=1e-6)
        prev = r["cash"]


# 2. Net worth: a transfer cash → investments leaves net worth unchanged -----------------------
def test_transfer_does_not_change_net_worth(plain):
    saving = run_branch({**plain, "savings_rate": 1.0}, [D(0, "start_work")])
    not_saving = run_branch(plain, [D(0, "start_work")])
    assert sum(r["invested"] for r in saving.rows) > 0
    for a, b in zip(saving.rows, not_saving.rows):
        assert a["net_worth"] == pytest.approx(b["net_worth"])
        assert a["net_worth"] == pytest.approx(a["cash"] + a["investments"] - a["debt"])


# 3. Loans -------------------------------------------------------------------------------------
def test_loan_disbursement_and_amortization(plain):
    p = {**plain, "loan_rate": 0.06, "loan_term_months": 24}
    base = run_branch(p, [D(0, "start_work")])
    loan = run_branch(p, [D(0, "start_work", loan_amount=10_000)])
    pay = annuity_payment(10_000, 0.06, 24)
    r0 = loan.rows[0]
    assert r0["interest"] == pytest.approx(10_000 * 0.06 / 12)
    assert r0["principal"] == pytest.approx(pay - r0["interest"])
    assert r0["debt"] == pytest.approx(10_000 - r0["principal"])
    # Net worth differs from the no-loan branch exactly by cumulative interest paid.
    cum_interest = 0.0
    for a, b in zip(loan.rows, base.rows):
        cum_interest += a["interest"]
        assert a["net_worth"] == pytest.approx(b["net_worth"] - cum_interest, abs=1e-6)
        assert a["debt"] >= 0
    assert loan.rows[23]["debt"] == 0  # fully repaid at end of term
    assert loan.rows[30]["debt_payment"] == 0
    assert sum(r["principal"] for r in loan.rows) == pytest.approx(10_000)


def test_repay_debt_is_a_transfer(plain):
    p = {**plain, "opening_debt": 5000, "opening_debt_rate": 0.05}
    keep = run_branch(p, [])
    repay = run_branch(p, [D(0, "start_work", repay_debt=True)])
    assert repay.rows[0]["debt"] == 0
    # Month 0: same opening NW; repaying early avoids interest, so NW ends higher.
    assert repay.rows[-1]["net_worth"] > run_branch(p, [D(0, "start_work")]).rows[-1]["net_worth"]
    assert keep.rows[0]["interest"] > 0


def test_cannot_repay_without_cash(plain):
    p = {**plain, "opening_debt": 50_000, "opening_debt_rate": 0.05}
    with pytest.raises(InvalidDecision):
        run_branch(p, [D(0, "start_work", repay_debt=True)])


# 4. Determinism and rollback -------------------------------------------------------------------
def test_same_inputs_identical_output(params):
    ds = [D(0, "master_ft"), D(24, "move_abroad"), D(36, "return_home", savings_rate=0.8)]
    assert run_branch(params, ds).rows == run_branch(params, ds).rows


def test_rollback_replay_is_exact(params):
    full = [D(0, "master_ft"), D(12, "continue"), D(24, "start_work"), D(36, "move_abroad")]
    a = run_branch(params, full)
    # Roll back to the node after 2 decisions, then choose differently.
    b = run_branch(params, full[:2])
    assert b.next_decision_month == 24
    assert b.rows[:24] == a.rows[:24]
    assert b.states_at_points[24] == a.states_at_points[24]
    c = run_branch(params, full[:2] + [D(24, "move_abroad")])
    assert c.rows[:24] == a.rows[:24] and c.rows[24] != a.rows[24]


# 5. Scenario isolation --------------------------------------------------------------------------
def test_relocation_cost_only_affects_moving_branches(plain):
    hi = {**plain, "relocation_cost": 9000}
    assert run_branch(plain, [D(0, "start_work")]).rows == run_branch(hi, [D(0, "start_work")]).rows
    a = run_branch(plain, [D(0, "move_abroad")]).rows
    b = run_branch(hi, [D(0, "move_abroad")]).rows
    for ra, rb in zip(a, b):
        assert ra["cash"] - rb["cash"] == pytest.approx(9000 - plain["relocation_cost"])
        assert ra["income"] == rb["income"] and ra["expenses"] == rb["expenses"]


def test_tuition_does_not_touch_work_branch(params):
    hi = {**params, "tuition_annual": 20_000}
    assert run_branch(params, [D(0, "start_work")]).rows == run_branch(hi, [D(0, "start_work")]).rows


# 6. Monotonicity ---------------------------------------------------------------------------------
@pytest.mark.parametrize("choice", ["master_ft", "master_pt"])
def test_higher_tuition_never_more_money(params, choice):
    finals, mins = [], []
    for tuition in [0, 4000, 8000, 12_000, 20_000]:
        rows = run_branch({**params, "tuition_annual": tuition}, [D(0, choice)]).rows
        finals.append(rows[-1]["net_worth"])
        mins.append(min(r["cash"] + r["investments"] for r in rows))
    assert finals == sorted(finals, reverse=True)
    assert mins == sorted(mins, reverse=True)


# 7. Edge cases -------------------------------------------------------------------------------------
def test_zero_income_flags_financing_need(plain):
    p = {**plain, "student_income": 0}
    rows = run_branch(p, []).rows  # stays a student on zero income
    assert rows[1]["cash"] < rows[0]["cash"]
    flagged = [r for r in rows if "financing_needed" in r["flags"]]
    assert flagged and all(r["cash"] < 0 for r in flagged)
    assert summarize(rows)["first_negative_month"] == flagged[0]["t"]


def test_zero_interest_rate_loan():
    assert annuity_payment(12_000, 0.0, 12) == 1000
    p = make_params(**PLAIN, loan_rate=0.0, loan_term_months=12)
    rows = run_branch(p, [D(0, "start_work", loan_amount=12_000)]).rows
    assert all(r["interest"] == 0 for r in rows)
    assert rows[11]["debt"] == 0


def test_negative_returns_shrink_investments(plain):
    p = {**plain, "opening_investments": 10_000, "return_annual": -0.2}
    rows = run_branch(p, [D(0, "start_work")]).rows
    assert rows[0]["investment_return"] < 0
    # With no further flows the portfolio falls by exactly 20% per year.
    only = run_branch({**p, "salary_work": 0, "living_working": 0, "rent": 0, "opening_cash": 0},
                      [D(0, "start_work")]).rows
    assert only[11]["investments"] == pytest.approx(10_000 * 0.8)


def test_monthly_annual_conversion_consistent():
    for r in [-0.3, 0.0, 0.01, 0.05, 0.12]:
        assert (1 + monthly_compound_rate(r)) ** 12 == pytest.approx(1 + r)


def test_inflation_applied_annually_consistent(params):
    rows = run_branch(params, [D(0, "start_work")]).rows
    assert rows[12]["expenses"] == pytest.approx(rows[0]["expenses"] * (1 + params["inflation_annual"]))


def test_negative_cash_sells_investments_first(plain):
    p = {**plain, "opening_cash": 0, "opening_investments": 5000, "student_income": 0}
    rows = run_branch(p, []).rows
    assert rows[0]["sold_investments"] > 0 and rows[0]["cash"] == pytest.approx(0)
    assert "financing_needed" not in rows[0]["flags"]


# Action masking ---------------------------------------------------------------------------------------
def test_invalid_choice_rejected(params):
    with pytest.raises(InvalidDecision):
        run_branch(params, [D(0, "return_home")])
    with pytest.raises(InvalidDecision):
        run_branch(params, [D(5, "start_work")])  # not a decision point
    with pytest.raises(InvalidDecision):
        run_branch(params, [D(0, "master_ft"), D(12, "master_ft")])  # already enrolled


def test_degree_completion_is_a_decision_point(params):
    run = run_branch(params, [D(0, "start_work"), D(12, "master_ft")])
    pt = next(x for x in run.points if x["month"] == 36)
    assert "degree_finished" in pt["reasons"]
    assert run.states_at_points[36].has_master


def test_next_decision_point(params):
    assert run_branch(params, []).next_decision_month == 0
    assert run_branch(params, [D(0, "start_work")]).next_decision_month == 12
    full = [D(m, "continue") for m in range(12, 120, 12)]
    assert run_branch(params, [D(0, "start_work")] + full).next_decision_month is None


# Goals ---------------------------------------------------------------------------------------------------
def test_goals_checked_in_priority_order_without_double_counting():
    rows = [{"t": t, "cash": 10_000, "investments": 0} for t in range(24)]
    goals = [{"goal_id": "a", "name": "A", "target": 8000, "month": 5, "priority": "high"},
             {"goal_id": "b", "name": "B", "target": 5000, "month": 10, "priority": "medium"},
             {"goal_id": "c", "name": "C", "target": 1, "month": 50, "priority": "high"}]
    out = {g["goal_id"]: g["status"] for g in check_goals(goals, rows)}
    assert out == {"a": "met", "b": "not_met", "c": "beyond_horizon"}
