"""Tests for the game engine extensions (GAME_BUILD_PROMPT §4.7): monthly turns, habits,
purchases, housing, joy, masking and the shared seeded world."""
import pytest

from app.engine.game import (Action as A, InvalidAction, Turn, available_actions, decisions_to_turns,
                             run_turns, state_at)
from app.engine.metrics import net_worth_bridge
from app.engine.montecarlo import make_worlds
from app.engine.simulate import initial_state, run_branch
from app.engine.state import Decision as D
from tests.conftest import PLAIN, make_params

WORK = A("life_choice", "start_work")


def T(month, *actions):
    return Turn(month, tuple(actions))


def nw_of(s):
    return s.cash + s.investments + s.home_value - s.debt - s.mortgage


# ----------------------------------------------------------------------------- habits
def test_habit_cost_appears_exactly_in_lifestyle(plain):
    base = run_turns(plain, [T(0, WORK)]).rows
    habit = run_turns(plain, [T(0, WORK, A("set_habit", "eat_out", 2))]).rows
    for t, (a, b) in enumerate(zip(base, habit)):
        assert b["lifestyle"] == pytest.approx(340) and a["lifestyle"] == 0
        assert a["cash"] - b["cash"] == pytest.approx(340 * (t + 1))
        assert b["expenses"] == a["expenses"]  # rent + living untouched


def test_habit_cost_inflates_like_other_costs(params):
    rows = run_turns(params, [T(0, WORK, A("set_habit", "eat_out", 1))]).rows
    assert rows[24]["lifestyle"] == pytest.approx(170 * 1.01 ** 2)


def test_habit_price_change_isolated(plain):
    pricier = {**plain, "habit_eat_out_2_cost": 500}
    shop = [T(0, WORK, A("set_habit", "shopping", 2))]
    eat = [T(0, WORK, A("set_habit", "eat_out", 2))]
    assert run_turns(plain, shop).rows == run_turns(pricier, shop).rows
    assert run_turns(plain, eat).rows[-1]["cash"] > run_turns(pricier, eat).rows[-1]["cash"]


def test_habit_back_to_zero(plain):
    rows = run_turns(plain, [T(0, WORK, A("set_habit", "gym", 1)), T(5, A("set_habit", "gym", 0))]).rows
    assert [r["lifestyle"] for r in rows[:7]] == [60] * 5 + [0, 0]


# ----------------------------------------------------------------------------- purchases
def test_purchase_reduces_cash_and_net_worth_by_price(plain):
    base = run_turns(plain, [T(0, WORK)]).rows
    bought = run_turns(plain, [T(0, WORK), T(3, A("buy_item", "laptop"))]).rows
    assert bought[3]["purchases"] == 1500 and bought[3]["one_offs"] == 0
    for t in range(120):
        d = 1500 if t >= 3 else 0
        assert base[t]["cash"] - bought[t]["cash"] == pytest.approx(d)
        assert base[t]["net_worth"] - bought[t]["net_worth"] == pytest.approx(d)
        assert bought[t]["investments"] == base[t]["investments"] and bought[t]["home_value"] == 0


def test_trip_is_consumption_and_gives_joy(params):
    rows = run_turns(params, [T(0, A("buy_item", "weekend_trip"))]).rows
    assert rows[0]["purchases"] == 450
    assert rows[0]["joy"] == params["joy_studying"] + params["item_weekend_trip_joy"]


# ----------------------------------------------------------------------------- housing
RICH = dict(PLAIN, opening_cash=300_000, rent=0, mortgage_rate=0, maintenance_annual=0, home_growth_annual=0)


def test_buy_home_changes_net_worth_only_by_fees():
    p = make_params(**RICH)
    base = run_turns(p, [T(0, WORK)]).rows
    own = run_turns(p, [T(0, WORK, A("buy_home", "studio"))]).rows
    fees = 450_000 * 0.02
    assert own[0]["home_fees"] == fees
    for a, b in zip(base, own):
        assert a["net_worth"] - b["net_worth"] == pytest.approx(fees)


def test_buy_home_balance_sheet():
    p = make_params(**RICH)
    s0 = state_at(p, [T(0, WORK)], 0)
    s1 = state_at(p, [T(0, WORK, A("buy_home", "studio"))], 0)
    assert s1.home_value == 450_000 and s1.mortgage == pytest.approx(360_000)
    assert s0.cash - s1.cash == pytest.approx(90_000)       # down payment now; fees paid in the month
    assert nw_of(s1) == pytest.approx(nw_of(s0))


def test_amortization_is_a_transfer_and_mortgage_never_negative():
    p = make_params(**{**RICH, "amortization_annual": 0.1, "opening_cash": 1_000_000})
    rows = run_turns(p, [T(0, WORK, A("buy_home", "studio"))]).rows
    amort = 450_000 * 0.1 / 12
    assert rows[0]["amortization"] == pytest.approx(amort)
    assert all(r["mortgage"] >= 0 for r in rows)
    assert rows[-1]["mortgage"] == 0 and sum(r["amortization"] for r in rows) == pytest.approx(360_000)
    # Net worth only moves with income − costs: amortization never shows up.
    for prev, r in zip(rows, rows[1:]):
        assert r["net_worth"] - prev["net_worth"] == pytest.approx(r["income"] - r["expenses"])


def test_mortgage_interest_maintenance_and_growth(params):
    p = {**params, "opening_cash": 300_000}
    rows = run_turns(p, [T(0, WORK, A("buy_home", "studio"))]).rows
    assert rows[0]["mortgage_interest"] == pytest.approx(360_000 * 0.02 / 12)
    assert rows[0]["maintenance"] == pytest.approx(450_000 * 0.01 / 12)
    assert rows[0]["rent"] == 0
    assert rows[11]["home_value"] == pytest.approx(450_000 * 1.015)


def test_sell_home_clears_mortgage_and_back_to_rent():
    p = make_params(**{**RICH, "rent": 900})
    run = run_turns(p, [T(0, WORK, A("buy_home", "flat")), T(24, A("sell_home"))])
    r = run.rows[24]
    assert r["mortgage"] == 0 and r["home_value"] == 0 and r["rent"] == 900
    assert r["home_fees"] == pytest.approx(750_000 * 0.03)


def test_cannot_buy_without_down_payment(params):
    with pytest.raises(InvalidAction, match="down payment"):
        run_turns(params, [T(0, A("buy_home", "studio"))])


def test_cannot_move_abroad_while_owning():
    p = make_params(**RICH)
    with pytest.raises(InvalidAction, match="Sell your home"):
        run_turns(p, [T(0, WORK, A("buy_home", "studio")), T(12, A("life_choice", "move_abroad"))])


def test_bridge_exact_with_housing_habits_purchases_and_loans(params):
    p = {**params, "opening_cash": 250_000, "opening_debt": 3000, "opening_debt_rate": 0.05}
    turns = [T(0, WORK, A("set_habit", "eat_out", 2), A("buy_item", "bike")),
             T(6, A("change_flat", "nicer"), A("buy_home", "studio")),
             T(30, A("buy_item", "weekend_trip"), A("set_habit", "shopping", 3)),
             T(60, A("sell_home")), T(61, A("take_loan", value=10_000)), T(70, A("change_flat", "cheaper"))]
    rows = run_turns(p, turns).rows
    b = net_worth_bridge(rows, 250_000 - 3000)
    total = b["opening"] + sum(v for k, v in b.items() if k not in ("opening", "closing"))
    assert total == pytest.approx(b["closing"], abs=1e-6)
    assert b["home_fees"] < 0 and b["lifestyle"] < 0 and b["purchases"] < 0 and b["home_appreciation"] > 0


def test_change_flat_rent_factor_and_moving_cost(plain):
    rows = run_turns(plain, [T(0, WORK), T(2, A("change_flat", "cheaper"))]).rows
    assert rows[1]["rent"] == 900 and rows[2]["rent"] == pytest.approx(720)
    assert rows[2]["one_offs"] == plain["flat_moving_cost"]


# ----------------------------------------------------------------------------- turns
def test_rollback_replays_bit_for_bit(params):
    full = [T(0, A("life_choice", "master_ft"), A("set_habit", "eat_out", 1)), T(5, A("buy_item", "phone")),
            T(24, A("life_choice", "move_abroad")), T(40, A("buy_item", "flight_home"))]
    w = make_worlds(params, 8, 42)[7]
    a = run_turns(params, full, world=w)
    b = run_turns(params, full[:2], world=w)
    assert a.rows[:24] == b.rows[:24]
    assert run_turns(params, full, world=w).rows == a.rows


@pytest.mark.parametrize("decisions", [
    [D(0, "start_work")], [D(0, "master_ft")], [D(0, "move_abroad")],
    [D(0, "master_ft"), D(12, "continue"), D(24, "move_abroad")],
    [D(0, "start_work", loan_amount=10_000), D(12, "master_pt", savings_rate=0.2)],
    [], [D(0, "stay_student"), D(12, "master_ft"), D(24, "drop_out", repay_debt=False)],
])
def test_legacy_decisions_as_turns_reproduce_old_engine(params, decisions):
    old = run_branch(params, decisions).rows
    new = run_turns(params, decisions_to_turns(params, decisions)).rows
    assert [r.pop("regime") for r in new]  # the game adds the HMM regime of each month
    assert new == old


def test_one_life_choice_per_month(params):
    with pytest.raises(InvalidAction, match="one big life move"):
        run_turns(params, [T(0, A("life_choice", "master_ft"), A("life_choice", "move_abroad"))])


def test_cooldown_blocks_second_move(params):
    with pytest.raises(InvalidAction, match="12 months"):
        run_turns(params, [T(0, A("life_choice", "move_abroad")), T(6, A("life_choice", "return_home"))])
    run_turns(params, [T(0, A("life_choice", "move_abroad")), T(12, A("life_choice", "return_home"))])


def test_degree_completes_and_graduate_starts_work(params):
    rows = run_turns(params, [T(0, A("life_choice", "master_ft"))]).rows
    assert rows[23]["status"] == "studying" and rows[24]["status"] == "working"
    assert rows[24]["income"] == pytest.approx(params["salary_master"])


def test_turn_ordering_rejected(params):
    with pytest.raises(InvalidAction):
        run_turns(params, [T(5, A("buy_item", "bike")), T(2, A("buy_item", "bike"))])


# ----------------------------------------------------------------------------- masking
def _by_id(actions, place):
    return {(a["type"], a.get("id"), a.get("value")): a for a in actions[place]}


def _retiree():
    from app.data.loader import build_persona, load_dataset
    snap = load_dataset().snapshot
    pid = snap[snap.employment_status == "retired"].profile_id.iloc[0]
    per = build_persona(pid)
    return per.params, initial_state(per.params, per.status, per.start_salary, per.start_living)


def test_retiree_can_study_full_time_but_not_take_jobs():
    p, s = _retiree()
    acts = available_actions(s, 0, p)
    uni = _by_id(acts, "university")
    assert uni[("life_choice", "master_ft", None)]["enabled"]
    assert not uni[("life_choice", "master_pt", None)]["enabled"]
    assert not any(a["enabled"] for a in acts["office"] if a["type"] in ("life_choice", "ask_raise"))


def test_retiree_master_keeps_pension_and_returns_to_retirement():
    p, s0 = _retiree()
    dur = int(p["master_ft_months"])
    run = run_turns(p, [T(0, A("life_choice", "master_ft"))], start=s0, keep_states=True)
    base = run_turns(p, [], start=s0)
    for t in range(dur):
        assert run.rows[t]["income"] == pytest.approx(base.rows[t]["income"])  # pension unchanged
    end = state_at(p, [T(0, A("life_choice", "master_ft"))], dur, start=s0)
    assert end.status == "retired" and end.has_master and not end.retiree
    assert any(l["kind"] == "degree" and "retirement" in l["text"] for l in run.letters)
    extra = sum(r["one_offs"] for r in run.rows) - sum(r["one_offs"] for r in base.rows)
    assert extra == pytest.approx(p["tuition_annual"] * dur / 12)  # tuition is the only extra cost


def test_retiree_drop_out_returns_to_retirement():
    p, s0 = _retiree()
    turns = [T(0, A("life_choice", "master_ft")), T(12, A("life_choice", "drop_out"))]
    s = state_at(p, turns, 12, start=s0)
    assert s.status == "retired" and not s.has_master and not s.retiree
    assert not any(o.kind == "tuition" and o.month >= 12 for o in s.scheduled)


def test_student_can_stay_student_at_any_age(params):
    from dataclasses import replace
    s = replace(initial_state(params), start_age=31.0)
    from app.engine.decisions import options
    assert "stay_student" in {c["id"] for c in options(s, 0, params)["choices"]}


def test_student_cannot_ask_for_raise_and_masking_reasons(params):
    s = state_at(params, [], 0)
    office = _by_id(available_actions(s, 0, params), "office")
    assert not office[("ask_raise", None, None)]["enabled"]
    assert office[("life_choice", "start_work", None)]["enabled"]
    s = state_at(params, [T(0, A("life_choice", "move_abroad"))], 6)
    station = _by_id(available_actions(s, 6, params), "station")
    assert "12 months" in station[("life_choice", "return_home", None)]["reason"]


def test_previews_come_from_the_engine(params):
    acts = available_actions(state_at(params, [], 0), 0, params)
    cafe = _by_id(acts, "cafe")
    assert cafe[("set_habit", "eat_out", 2)]["preview"]["monthly"] == 340
    shop = _by_id(acts, "shop")
    assert shop[("buy_item", "laptop", None)]["preview"]["one_off"] == 1500
    estate = _by_id(acts, "realestate")
    studio = estate[("buy_home", "studio", None)]
    assert not studio["enabled"] and "down payment" in studio["reason"]


def test_ask_raise_needs_24_months_and_is_reproducible(params):
    turns = [T(0, WORK), T(24, A("ask_raise"))]
    with pytest.raises(InvalidAction):
        run_turns(params, [T(0, WORK), T(12, A("ask_raise"))])
    no_ask = run_turns(params, [T(0, WORK)]).rows[24]["income"]
    outcomes = set()
    for world in range(12):
        r1 = run_turns(params, turns, luck=(42, world))
        r2 = run_turns(params, turns, luck=(42, world))
        assert r1.rows == r2.rows
        got = r1.rows[24]["income"]
        assert got == pytest.approx(no_ask) or got == pytest.approx(no_ask * 1.05)
        outcomes.add(got > no_ask * 1.01)
    assert outcomes == {True, False}  # both outcomes occur across worlds


# ----------------------------------------------------------------------------- joy
def test_joy_points_and_switch_off(params):
    rows = run_turns(params, [T(0, A("set_habit", "eat_out", 3))]).rows
    assert rows[0]["joy"] == params["joy_studying"] + params["habit_eat_out_3_joy"]
    off = run_turns({**params, "joy_enabled": 0}, [T(0, A("set_habit", "eat_out", 3))]).rows
    assert all(r["joy"] == 0 for r in off)
    assert [r["cash"] for r in off] == [r["cash"] for r in rows]  # joy is not money


# ----------------------------------------------------------------------------- world & letters
def test_same_world_same_events_across_realities(params):
    w = make_worlds(params, 138, 42)[137]
    a = run_turns(params, [T(0, WORK)], world=w)
    b = run_turns(params, [T(0, A("life_choice", "master_ft"))], world=w)
    assert a.world_events == b.world_events == w["events"]
    rent_a = [x["month"] for x in a.letters if x["kind"] == "rent_rise"]
    rent_b = [x["month"] for x in b.letters if x["kind"] == "rent_rise"]
    assert rent_a == rent_b


def test_negative_cash_sends_financing_letter(params):
    run = run_turns({**params, "tuition_annual": 40_000}, [T(0, A("life_choice", "master_ft"))])
    assert any(x["kind"] == "financing_needed" for x in run.letters)
    assert any("financing_needed" in r["flags"] for r in run.rows)


def test_freedom_runway_same_definition_in_hud_and_final_scores(params):
    from app.engine.metrics import scores, scores_now
    p = {**params, "opening_cash": 300_000}
    run = run_turns(p, [T(0, WORK, A("buy_home", "studio"), A("set_habit", "eat_out", 2))])
    final = scores(run.rows, [], [])["freedom"]["runway_months"]
    now = scores_now(run.rows, [], run.final_state, 120)["freedom"]["runway_months"]
    assert final == pytest.approx(now)


# ----------------------------------------------------------------------------- sell investments
def _invested(plain, **kw):
    return {**plain, "opening_cash": 20_000, "opening_investments": 30_000, "savings_rate": 0.0, **kw}


def test_sell_investments_to_cash_is_a_transfer(plain):
    p = _invested(plain)
    before = state_at(p, [T(0, WORK)], 0)
    after = state_at(p, [T(0, WORK, A("sell_investments", "cash", 12_000))], 0)
    assert after.cash - before.cash == pytest.approx(12_000)
    assert before.investments - after.investments == pytest.approx(12_000)
    assert nw_of(after) == pytest.approx(nw_of(before))


def test_sell_investments_to_debt_cuts_debt_and_keeps_net_worth(plain):
    p = _invested(plain, opening_debt=10_000, opening_debt_rate=0.05)
    before = state_at(p, [T(0, WORK)], 0)
    part = state_at(p, [T(0, WORK, A("sell_investments", "debt", 4_000))], 0)
    assert before.debt - part.debt == pytest.approx(4_000) and part.cash == before.cash
    assert part.debt_payment == before.debt_payment          # same payment, loan ends sooner
    assert nw_of(part) == pytest.approx(nw_of(before))
    full = state_at(p, [T(0, WORK, A("sell_investments", "debt", 10_000))], 0)
    assert full.debt == 0 and full.debt_payment == 0 and full.investments == pytest.approx(20_000)
    # Less debt ⇒ less interest paid over the horizon
    base = run_turns(p, [T(0, WORK)]).rows
    paid = run_turns(p, [T(0, WORK, A("sell_investments", "debt", 4_000))]).rows
    assert sum(r["interest"] for r in paid) < sum(r["interest"] for r in base)


def test_sell_investments_validation(plain):
    p = _invested(plain)
    with pytest.raises(InvalidAction, match="at most"):
        run_turns(p, [T(0, WORK, A("sell_investments", "cash", 30_001))])
    with pytest.raises(InvalidAction, match="no debt"):
        run_turns(p, [T(0, WORK, A("sell_investments", "debt", 1_000))])
    with pytest.raises(InvalidAction, match="no investments"):
        run_turns({**p, "opening_investments": 0}, [T(0, WORK, A("sell_investments", "cash", 1))])
    with pytest.raises(InvalidAction, match="'cash' or 'debt'"):
        run_turns(p, [T(0, WORK, A("sell_investments", "house", 1))])
    debt = {**p, "opening_debt": 5_000, "opening_debt_rate": 0.05}
    with pytest.raises(InvalidAction, match="smaller of investments and debt"):
        run_turns(debt, [T(0, WORK, A("sell_investments", "debt", 6_000))])


def test_sell_investments_capped_in_other_worlds(plain):
    p = _invested(plain)
    s = state_at(p, [T(0, WORK)], 0)
    run = run_turns(p, [T(0, WORK, A("sell_investments", "cash", 99_999))], strict=False, until=0)
    assert run.final_state.investments == 0 and run.final_state.cash == pytest.approx(s.cash + s.investments)


def test_bank_offers_sell_investments(plain):
    p = _invested(plain, opening_debt=8_000, opening_debt_rate=0.05)
    s = state_at(p, [T(0, WORK)], 0)
    bank = available_actions(s, 0, p)["bank"]
    cash = [a for a in bank if a["type"] == "sell_investments" and a["id"] == "cash"]
    debt = [a for a in bank if a["type"] == "sell_investments" and a["id"] == "debt"]
    assert [a["value"] for a in cash] == pytest.approx([7_500, 15_000, 30_000]) and all(a["enabled"] for a in cash)
    assert [a["value"] for a in debt] == pytest.approx([4_000, 8_000]) and all(a["enabled"] for a in debt)


# ----------------------------------------------------------------------------- safety = risk, not history
def _row(t, cash, buffer=1000.0):
    return {"t": t, "cash": cash, "investments": 0.0, "debt": 0.0, "net_worth": cash, "status": "working",
            "flags": (["financing_needed"] if cash < 0 else []) + (["below_buffer"] if cash < buffer else []),
            "buffer_target": buffer, "expenses": 500.0, "income": 0.0, "one_offs": 0.0, "interest": 0.0,
            "investment_return": 0.0, "joy": 0.0}


def test_recovered_dip_is_not_a_lasting_shortfall():
    from types import SimpleNamespace
    from app.engine.metrics import scores, scores_now
    rows = [_row(0, -500), _row(1, -200), _row(2, 3000)]   # in the red while studying, then recovered
    now = scores_now(rows, [], SimpleNamespace(cash=3000, investments=0, home_value=0, debt=0, mortgage=0,
                                               has_master=False, program=None), 3)["safety"]
    assert now["level"] == "steady" and now["recovered"] and now["months_negative_cash"] == 2
    full = scores(rows, [], [])["safety"]
    assert full["level"] == "tight" and full["recovered"]
    assert scores(rows + [_row(3, -10)], [], [])["safety"]["level"] == "shortfall"   # still short at the end


def test_cash_risk_counts_only_the_future():
    import numpy as np
    from app.engine.metrics import cash_risk
    cash = np.array([[100, 50, -10] + [200] * 20, [500] * 23, [500] * 15 + [-1] * 8, [500] * 23])
    r = cash_risk(cash, cash < 300)
    assert r["below_zero_12m"] == 0.25 and r["below_zero_ahead"] == 0.5 and r["level"] == "high"
    calm = cash_risk(np.full((10, 24), 800.0), np.zeros((10, 24), bool))
    assert calm["below_zero_ahead"] == 0 and calm["level"] == "low"


def test_projection_reports_forward_risk():
    from app.service import game_projection
    from app.models import GameProjectionRequest
    out = game_projection(GameProjectionRequest(turns=[], current_month=0, n=20))
    assert 0 <= out["risk"]["below_zero_12m"] <= out["risk"]["below_zero_ahead"] <= 1
    assert out["share_worlds_shortfall"] == out["risk"]["below_zero_ahead"]
