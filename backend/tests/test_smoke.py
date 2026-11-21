"""Smoke tests: the stub pipeline runs end to end (regime_engine -> cash_engine -> metrics -> session -> agent tools)."""
import json

import numpy as np
import pytest

from regime import RegimeEngine, TwinSession, load_profile, load_transition_priors
from regime import cash_engine, validation
from regime.agent import TOOL_SCHEMAS, dispatch_tool, run_agent
from regime.types import STATE_INDEX, Intervention


@pytest.fixture(scope="module")
def alexia():
    return load_profile("P00001")


def test_profile_loads(alexia):
    assert alexia.name == "Alexia" and alexia.initial_state == "student"
    assert alexia.opening_cash == 8000 and alexia.mean_savings_rate < 0


def test_stub_simulate_contract(alexia):
    engine = RegimeEngine.stub()
    paths = engine.simulate(alexia, horizon_months=60, n_paths=50,
                            interventions=[Intervention("force_state", 0, 24, state="student"),
                                           Intervention("set_attribute", 24, attribute=("has_masters", True))])
    for arr in (paths.states, paths.income_chf, paths.savings_rate, paths.net_flow_chf):
        assert arr.shape == (50, 60)
    assert paths.occupancy.shape == (60, 5) and paths.transition_draws.shape == (50, 5, 5)
    assert np.allclose(paths.net_flow_chf, paths.savings_rate * paths.income_chf)
    assert (paths.states[:, :24] == STATE_INDEX["student"]).all()
    assert (paths.states[:, 24:] == STATE_INDEX["employed"]).all()
    table = engine.income_table.params
    assert paths.income_chf[0, 0] == alexia.ref_income
    assert paths.gross_income_chf[0, 24] == pytest.approx(table["entry_salary_chf"] * (1 + table["masters_premium_pct"]))
    assert paths.income_chf[0, 24] < paths.gross_income_chf[0, 24]  # net of contributions and tax
    belief = engine.filter(alexia, [-700.0] * 6)
    assert pytest.approx(sum(belief.values())) == 1.0 and max(belief, key=belief.get) == "student"


def test_cash_engine_reconciles(alexia):
    paths = RegimeEngine.stub().simulate(alexia, horizon_months=24, n_paths=3)
    cp = cash_engine.run(alexia, paths, [cash_engine.OneOff(0, 2500.0)])
    assert cash_engine.reconciles(cp)


def test_deterministic_equivalence_with_starter_kit():
    rows = validation.deterministic_equivalence()
    assert len(rows) == 3 and all(r["ok"] for r in rows), rows


def test_transition_priors_sane():
    priors = load_transition_priors()
    chk = validation.check_transition_matrix(priors)
    assert chk["rows_sum_to_one"] and chk["non_negative"] and chk["draws_rows_sum_to_one"]
    pi0 = np.eye(5)[STATE_INDEX["employed"]]
    assert validation.check_occupancy(priors, pi0)["rows_sum_to_one"]
    A = priors.matrix({"job_loss_hazard": 0.01})
    assert A[STATE_INDEX["employed"], STATE_INDEX["job_search"]] == pytest.approx(0.01)


def test_session_end_to_end():
    s = TwinSession.from_profile_id("P00001", n_paths=200)
    s.create_scenario("masters", "Master's degree")
    s.apply_decision("masters", "masters", 0)
    s.create_scenario("work", "Start working")
    s.apply_decision("work", "start_working", 0)
    s.set_assumption("work", "entry_salary_chf", 4500)
    base = s.simulate("baseline")
    m = s.simulate("masters")
    assert base["closing_cash_chf"]["p50"] < 0          # Alexia's baseline is already infeasible
    assert m["expected_months_in_state"]["student"] == 24
    cmp_ = s.compare("baseline", "work")
    assert cmp_["delta_b_minus_a"]["closing_cash_p50_chf"] > 0
    sens = s.sensitivity("masters", n_paths=50)
    assert sens["ranking"] and sens["ranking"][0]["swing_closing_p50_chf"] >= sens["ranking"][-1]["swing_closing_p50_chf"]
    sp = s.sample_paths("masters", n=3)
    assert len(sp["paths"]) == 3 and sp["paths"][0]["regimes"][0]["regime"] == "student"
    json.dumps(s.get_state(), default=str)
    json.dumps(m)


def test_agent_tools_dispatch_and_loop():
    s = TwinSession.from_profile_id("P00001", n_paths=100)
    names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    assert {"get_state", "apply_decision", "set_assumption", "simulate", "compare", "sensitivity", "sample_paths"} <= names
    assert "error" in dispatch_tool(s, "simulate", {"scenario_id": "nope"})

    class _Call:
        def __init__(self, i, name, args):
            self.id, self.function = i, type("F", (), {"name": name, "arguments": json.dumps(args)})()

    class _Msg:
        def __init__(self, content, calls=None):
            self.content, self.tool_calls = content, calls

    class FakeClient:
        """Scripted LLM: one tool call, then a final answer."""
        def __init__(self):
            self.step = 0
            self.chat = type("C", (), {"completions": self})()

        def create(self, **kw):
            self.step += 1
            msg = (_Msg("", [_Call("c1", "simulate", {"scenario_id": "baseline"})]) if self.step == 1
                   else _Msg("Done."))
            return type("R", (), {"choices": [type("Ch", (), {"message": msg})()]})()

    text, history, trace = run_agent(s, "What happens if I do nothing?", client=FakeClient())
    assert text == "Done." and trace[0]["tool"] == "simulate" and "closing_cash_chf" in trace[0]["result"]


def _wealth_session(**kw):
    return TwinSession.from_profile_id("P00001", n_paths=60, wealth=True, **kw)


def test_wealth_layer_reconciles_and_keeps_levers():
    from regime import cash_engine
    s = _wealth_session(keep_factor=0.25)
    s.apply_decision("baseline", "start_working", 0)
    summ = s.simulate("baseline")
    res = s.results["baseline"]
    assert cash_engine.reconciles(res.cash)
    w = res.wealth
    surplus = w.net() > 0
    assert (w.flows(0.25)[surplus] < w.flows(1.0)[surplus]).all()
    assert (w.flows(0.25)[~surplus] == w.flows(1.0)[~surplus]).all()
    gained = w.cash(side=300.0)[:, -1] - w.cash()[:, -1]
    assert np.allclose(gained, 300.0 * w.price.sum(axis=1))
    assert summ["net_worth_real_by_year"][0]["p50"] == round(w.opening + w.port0 - w.debt)
    assert summ["keep_factor"] == 0.25


def test_default_session_is_unchanged_by_the_wealth_option():
    a = TwinSession.from_profile_id("P00001", n_paths=50).simulate("baseline")
    assert "net_worth_real_chf" not in a


def test_new_decisions_change_the_right_things():
    s = _wealth_session()
    s.apply_decision("baseline", "start_working", 0)  # Alexia is a student: give her a job first
    for sid, kind, t0, params in (("move", "relocate", 6, {"rent_factor": 1.5}),
                                  ("child", "family", 12, {}),
                                  ("se", "self_employed", 0, {})):
        s.create_scenario(sid, copy_from="baseline")
        s.apply_decision(sid, kind, t0, **params)
    base = s.simulate("baseline")
    assert s.simulate("move")["net_worth_real_chf"]["p50"] < base["net_worth_real_chf"]["p50"]
    child = s.simulate("child")
    assert child["net_worth_real_chf"]["p50"] < base["net_worth_real_chf"]["p50"]
    assert s.scenarios["se"].profile_changes == {"self_employed": True}
    s.simulate("se")
    assert s.results["se"].wealth.oneoff[0] == 10000.0


def test_explore_tool_times_decisions():
    from regime.explore import dominates
    s = _wealth_session(keep_factor=0.3)
    s.apply_decision("baseline", "start_working", 0)
    res = dispatch_tool(s, "explore", {"max_start_month": 12})
    assert "error" not in res
    objs = [(-m["wealth_median"], m["months_below_buffer"], m["sacrifice_chf_month"]) for m in res["front"]]
    assert all(not any(dominates(b, a) for b in objs) for a in objs)
    assert [t["start_month"] for t in res["timing"]["masters"]] == [0, 6, 12]
    assert res["settings"]["simulations"] <= 4 + 3 * 2   # cached per (decision, start month)
    json.dumps(res)
    assert "error" in dispatch_tool(TwinSession.from_profile_id("P00001", n_paths=20), "explore", {})
