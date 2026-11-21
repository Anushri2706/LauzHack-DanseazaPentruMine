"""Decision catalogue (mapping preserved from TwinSession), assumption registry, sensitivity, break-even."""
import math

import numpy as np
import pytest

from app.engine.assumptions import AssumptionError
from app.engine.decisions import CATALOGUE, DecisionError, to_engine, validate_decision
from app.engine.montecarlo import summarise
from app.engine.sensitivity import bisect, one_at_a_time, perturbed_values
from app.engine.simulate import SimulationConfig, run_monte_carlo

from .regression_cases import EQUIV
from .test_regression import REF, plain


def _v(A, kind, t0=0, initial="employed", **p):
    return validate_decision(kind, t0, p, initial_state=initial, horizon_months=120, assumptions=A, current_canton="VD")


@pytest.mark.parametrize("i", range(len(EQUIV)))
def test_mapping_matches_twin_session(assumptions, i):
    """Same interventions and one-offs as the original TwinSession (frozen in legacy_reference.json)."""
    kind, params, _, _ = EQUIV[i]
    initial = "student" if kind == "start_working" else "employed"
    new = to_engine([_v(assumptions, kind, 5, initial, **params)], 120)
    ref = REF["decision_mapping"][i]
    as_list = lambda iv: plain([iv.kind, iv.start_month, iv.duration, iv.state, iv.transition, iv.factor, iv.attribute])
    assert [as_list(iv) for iv in new.interventions] == ref["interventions"]
    assert [[o.month, o.amount_chf] for o in new.one_offs] == ref["one_offs"]


def test_relocate_mapping(assumptions):
    new = to_engine([_v(assumptions, "relocate", 2, to_canton="GE")], 120, canton_rent_ratio=lambda c: 1.3)
    assert [(i.kind, i.attribute) for i in new.interventions] == [("set_attribute", ("rent_factor_move", 1.3)),
                                                                   ("set_attribute", ("location", "GE"))]
    assert new.one_offs[0].amount_chf == 5000.0


def test_defaults_come_from_registry(assumptions):
    d = _v(assumptions, "masters", 0)
    assert d.params["annual_tuition_chf"] == assumptions.value("masters_annual_tuition_chf")
    assert d.params["duration_months"] == 24
    assert set(d.defaults_used) == {"annual_tuition_chf", "duration_months"}


@pytest.mark.parametrize("kind,start,params,initial,code", [
    ("teleport", 0, {}, "employed", "unknown_decision"),
    ("masters", 0, {"duration_months": 2}, "employed", "out_of_range"),
    ("masters", 0, {"duration_months": "two"}, "employed", "invalid_type"),
    ("masters", 0, {"vibes": 1}, "employed", "unknown_param"),
    ("masters", 0, {}, "retired", "not_eligible"),
    ("masters", 500, {}, "employed", "out_of_range"),
    ("relocate", 0, {}, "employed", "missing_param"),
    ("relocate", 0, {"to_canton": "XX"}, "employed", "invalid_choice"),
    ("relocate", 0, {"to_canton": "VD"}, "employed", "no_change"),
    ("one_off", 0, {"amount_chf": -5}, "employed", "out_of_range"),
    ("continue_employment", 0, {}, "student", "not_eligible"),
])
def test_invalid_decisions_rejected(assumptions, kind, start, params, initial, code):
    with pytest.raises(DecisionError) as e:
        validate_decision(kind, start, params, initial_state=initial, horizon_months=120, assumptions=assumptions,
                          current_canton="VD")
    assert code in [x["code"] for x in e.value.errors]


def test_no_raw_interventions_exposed():
    assert not {"force_state", "scale_hazard", "set_attribute"} & set(CATALOGUE)


# ----------------------------------------------------------------------------- registry

def test_registry_provenance(assumptions):
    rows = assumptions.registry.rows
    assert all(a.source_url is None and a.last_verified is None for a in rows.values())
    assert {a.source_status for a in rows.values()} <= {"cited", "estimated", "illustrative", "placeholder"}
    assert rows["tax_rate_at_80000"].source_status == "estimated" and rows["tax_rate_at_80000"].geography == "CH-VD"
    assert rows["portfolio_return"].source_status == "illustrative"
    assert rows["event_major_expense_cost_chf"].source_status == "placeholder"
    assert rows["ahv_iv_eo_rate"].source_status == "cited"      # cited in the CSV, not called "official"
    assert rows["job_loss_hazard"].kind == "probability"
    assert rows["apply_income_tax"].kind == "flag" and not rows["apply_income_tax"].sensitivity_eligible


def test_registry_matches_engine_params(assumptions, hmm_engine):
    params = hmm_engine.income_table.params
    for a in assumptions.registry.rows.values():
        if a.table in ("income", "expense", "tax", "market"):
            assert params[a.id] == a.value, a.id
        if a.table == "transitions":
            pri = hmm_engine.priors
            v = pri.age_names[a.id].hazard if a.id in pri.age_names else pri.hazards[pri.names[a.id]]
            assert v == a.value, a.id


def test_overrides_versioned_and_validated(assumptions):
    assert assumptions.version == "1"
    b = assumptions.with_overrides({"masters_premium_pct": 0.2})
    assert b.version.startswith("1+") and b.version != assumptions.version
    assert assumptions.with_overrides({"masters_premium_pct": 0.2}).version == b.version
    assert assumptions.value("masters_premium_pct") == 0.1          # original unchanged
    for bad in ({"nope": 1}, {"job_loss_hazard": 1.5}, {"apply_income_tax": 0.5}, {"entry_salary_chf": -1},
                {"inflation_stay_low": 0.9}, {"job_search_exit_rate": 0.99}):
        with pytest.raises(AssumptionError):
            assumptions.with_overrides(bad)


# ----------------------------------------------------------------------------- sensitivity / break-even

def test_sensitivity_rules(assumptions, engine_profile, hmm_engine):
    cfg = SimulationConfig(60, 100)
    evaluate = lambda A: summarise(run_monte_carlo(engine_profile, [], A, cfg, 100, 3, hmm_engine))
    out = one_at_a_time(assumptions, evaluate, ["entry_salary_chf", "apply_income_tax", "job_loss_hazard", "ui_max_months"])
    ids = [r["assumption_id"] for r in out["rows"]]
    assert "apply_income_tax" not in ids and "ui_max_months" not in ids           # categorical / discrete skipped
    assert {s["assumption_id"] for s in out["skipped"]} == {"apply_income_tax", "ui_max_months"}
    row = next(r for r in out["rows"] if r["assumption_id"] == "job_loss_hazard")
    assert row["low"]["value"] == pytest.approx(0.005 * 0.9) and row["high"]["value"] == pytest.approx(0.005 * 1.1)
    assert all("source_status" in r for r in out["rows"])


def test_probability_perturbation_clamped(assumptions):
    A = assumptions.with_overrides({"inflation_stay_low": 0.0})  # keep the inflation row stochastic for the test
    lo, hi, w = perturbed_values(A.with_overrides({"masters_premium_pct": 0.95}), "masters_premium_pct", 0.10)
    assert hi == 1.0 and w


def test_bisect_finds_bracketed_root():
    r = bisect(lambda x: x ** 3 - 2, 0, 3, tol=1e-6)
    assert r.status == "converged" and r.value == pytest.approx(2 ** (1 / 3), abs=1e-5)


def test_bisect_no_root():
    r = bisect(lambda x: x * x + 1, -2, 2)
    assert r.status == "no_break_even_in_range" and r.value is None


def test_bisect_undefined():
    r = bisect(lambda x: math.nan, 0, 1)
    assert r.status == "undefined" and r.value is None
