"""Negative cash is an automatic credit line, never free money (rubric: "do not reward financially
impossible behaviour"). The shortfall at the start of a month pays overdraft_rate / 12 that month."""
import pytest

from app.engine.metrics import net_worth_bridge, summarize
from app.engine.simulate import initial_state, run_branch, step_month
from app.engine.state import Decision as D
from tests.conftest import PLAIN, make_params


def test_interest_is_charged_on_the_opening_shortfall():
    p = make_params(**{**PLAIN, "overdraft_rate": 0.12, "opening_cash": -10_000})
    _, row = step_month(initial_state(p), p)
    assert row["overdraft_interest"] == pytest.approx(10_000 * 0.12 / 12)  # 100 CHF
    # cash identity still holds with the new term
    assert row["cash"] == pytest.approx(-10_000 + row["income"] - row["expenses"] - 100)


def test_no_interest_when_cash_is_positive():
    p = make_params(**{**PLAIN, "overdraft_rate": 0.12})
    rows = run_branch(p, [D(0, "start_work")]).rows
    assert min(r["cash"] for r in rows) > 0
    assert all(r["overdraft_interest"] == 0 for r in rows)


def test_negative_cash_is_never_free():
    """A master's that runs out of cash ends with less money once financing costs something."""
    free = make_params(overdraft_rate=0)
    paid = make_params(overdraft_rate=0.10)
    a = run_branch(free, [D(0, "master_ft")]).rows
    b = run_branch(paid, [D(0, "master_ft")]).rows
    assert min(r["cash"] for r in a) < 0
    assert b[-1]["net_worth"] < a[-1]["net_worth"]
    assert summarize(b)["total_overdraft_interest"] > 0


def test_bridge_stays_exact_with_overdraft():
    p = make_params(overdraft_rate=0.10)
    rows = run_branch(p, [D(0, "master_ft"), D(24, "move_abroad")]).rows
    br = net_worth_bridge(rows, p["opening_cash"])
    total = br["opening"] + sum(v for k, v in br.items() if k not in ("opening", "closing"))
    assert total == pytest.approx(br["closing"], abs=1e-6)


def test_overdraft_rate_is_isolated():
    """Changing the overdraft rate never touches a reality that keeps cash above zero."""
    p = make_params()
    hi = {**p, "overdraft_rate": 0.25}
    assert run_branch(p, [D(0, "start_work")]).rows == run_branch(hi, [D(0, "start_work")]).rows


def test_interest_is_capped_at_the_credit_limit_and_flagged():
    p = make_params(**{**PLAIN, "overdraft_rate": 0.12, "overdraft_limit": 30_000, "opening_cash": -50_000})
    _, row = step_month(initial_state(p), p)
    assert row["overdraft_interest"] == pytest.approx(30_000 * 0.12 / 12)  # only the financeable part
    assert "beyond_credit" in row["flags"] and "financing_needed" in row["flags"]


def test_within_limit_is_not_flagged_beyond_credit():
    p = make_params(**{**PLAIN, "overdraft_rate": 0.12, "opening_cash": -5_000})
    _, row = step_month(initial_state(p), p)
    assert "beyond_credit" not in row["flags"]


def test_compare_at_age_cuts_both_realities_at_the_same_age():
    from fastapi.testclient import TestClient
    from app.main import app
    c = TestClient(app)
    ra = {"world": 9, "turns": [{"month": 0, "actions": [{"type": "life_choice", "id": "master_ft"}]}]}
    rb = {"world": 9, "turns": [{"month": 0, "actions": [{"type": "life_choice", "id": "start_work"}]}]}
    d = c.post("/compare", json={"a": ra, "b": rb, "at_age": 30}).json()
    assert d["at_age"] == 30 and len(d["net_worth"]["a"]) == len(d["net_worth"]["b"]) == 120
