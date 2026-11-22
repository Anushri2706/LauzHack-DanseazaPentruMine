"""HMM regime engine: emissions, transitions, interventions, filter, held-out checks."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from twin import RegimeEngine, cash_engine, load_profile, load_transition_priors
from twin.cash_engine import CashPaths
from twin.data import (
    ASSUMPTIONS_DIR, DATA_DIR, liability_annual_rate, load_cash_balances, load_goals,
    load_profiles, scenario_annual_tuition,
)
from twin.income import IncomeTable
from twin.net_pay import gross_from_net, net_from_gross
from twin.metrics import goal_hit_rate
from twin.regime_engine import _schedule, emission_loglik, fit_emissions, forward_filter, sample_yearly_lump
from twin.types import STATE_INDEX, STATES, Intervention, Profile
from twin import validation

E, P, J, S, R = (STATE_INDEX[s] for s in STATES)


@pytest.fixture(scope="module")
def fitted():
    return fit_emissions(load_cash_balances(), load_profiles())


@pytest.fixture(scope="module")
def engine(fitted):
    return RegimeEngine(fitted, load_transition_priors())


def _worker(age=35, employment="employed", income=7000.0, rent=1800.0, household="single"):
    return Profile(age=age, employment=employment, ref_income=income, rent=rent,
                   attributes={"household_type": household, "has_masters": False, "abroad": False},
                   start_month_of_year=1)


def test_assumptions_replace_placeholders():
    for name in ("transitions.csv", "income_assumptions.csv", "expense_assumptions.csv"):
        assert "PLACEHOLDER" not in (ASSUMPTIONS_DIR / name).read_text()
    table = IncomeTable.from_csv().params
    assert table["ui_replacement_rate"] == pytest.approx(0.70)
    assert table["ui_replacement_with_dependents"] == pytest.approx(0.80)
    assert table["ui_insured_earnings_cap_chf"] == pytest.approx(12350)
    assert table["retired_income_fraction"] == pytest.approx(0.475)
    assert table["retirement_age"] == pytest.approx(65)
    assert table["abroad_income_multiplier"] == pytest.approx(0.51)
    assert table["abroad_cost_multiplier"] == pytest.approx(0.63)
    assert table["zurich_housing_multiplier"] == pytest.approx(1.20)
    assert table["zurich_move_cost_chf"] == pytest.approx(4000)
    assert table["germany_move_cost_chf"] == pytest.approx(5000)
    assert table["student_rent_chf"] == pytest.approx(900)
    assert table["employed_rent_chf"] == pytest.approx(1546)
    assert table["entry_salary_chf"] * (1 + table["masters_premium_pct"]) * 12 == pytest.approx(80900, rel=1e-4)
    assert table["phd_premium_pct"] == pytest.approx(0.1)
    assert table["long_term_unemployment_multiplier"] == pytest.approx(0.5)
    assert table["rent_step_factor"] == pytest.approx(1.4)
    assert table["purchase_deposit_fraction"] == pytest.approx(0.15)
    expense = (ASSUMPTIONS_DIR / "expense_assumptions.csv").read_text()
    assert "rent_step_factor,1.4" in expense and "purchase_deposit_fraction,0.15" in expense
    assert expense.split("rent_step_factor")[1].split("\n")[0].split(",")[4] == "prior"
    assert expense.split("purchase_deposit_fraction")[1].split("\n")[0].split(",")[4] == "prior"
    assert table["parental_leave_replacement"] == pytest.approx(0.80)
    assert table["parental_leave_months"] == pytest.approx(12)
    assumptions = (ASSUMPTIONS_DIR / "income_assumptions.csv").read_text()
    assert assumptions.count("EO is more complicated") == 1
    phd_line = next(line for line in assumptions.splitlines() if line.startswith("phd_premium_pct,"))
    assert "Guess" in phd_line and "guess" in phd_line.lower()

    priors = load_transition_priors()
    assert priors.hazards[E, J] == pytest.approx(0.005)
    assert priors.matrix(age=22)[E, J] == pytest.approx(0.010)
    assert priors.matrix(age=30)[E, J] == pytest.approx(0.006)
    assert priors.matrix(age=40)[E, J] == pytest.approx(0.0045)
    assert priors.matrix(age=60)[E, J] == pytest.approx(0.0033)
    assert priors.matrix({"job_loss_hazard": 0.01}, age=22)[E, J] == pytest.approx(0.020)
    assert priors.hazards[E, P] == pytest.approx(0.004)
    assert priors.hazards[P, E] == pytest.approx(0.01)
    for age, total in ((None, 0.11), (22, 0.20), (30, 0.13), (40, 0.11), (60, 0.08)):
        A = priors.matrix(age=age)
        assert A[J, E] + A[J, P] == pytest.approx(total)
        assert np.allclose(A.sum(1), 1.0) and (A >= -1e-12).all()


def test_income_rules_match_the_sources():
    # Pay rules are defined on gross wages; tax is off so the worker's income is the wage.
    table = IncomeTable.from_csv().with_overrides({"apply_income_tax": 0})
    employed = _worker(age=40, income=10000, rent=2000, household="single")
    states = np.full((1, 24), J)
    inc = table.income_paths(states, employed, experience=False)[0]
    assert inc[0] == pytest.approx(7000)                 # 70% of 10,000
    assert inc[17] == pytest.approx(7000)                # still inside 18.4 months
    assert inc[19] == pytest.approx(0)                   # spell exhausted, floor is 0

    capped = _worker(age=40, income=20000, rent=2000)
    assert table.income_paths(np.full((1, 1), J), capped, experience=False)[0, 0] == pytest.approx(0.70 * 12350)

    family = _worker(age=40, income=10000, household="family")
    assert table.income_paths(np.full((1, 1), J), family, experience=False)[0, 0] == pytest.approx(8000)

    low = _worker(age=40, income=3000)
    assert table.income_paths(np.full((1, 1), J), low, experience=False)[0, 0] == pytest.approx(2400)  # 80%

    young = _worker(age=22, income=10000)
    young_inc = table.income_paths(np.full((1, 12), J), young, experience=False)[0]
    assert young_inc[0] == pytest.approx(7000)
    assert young_inc[8] == pytest.approx(7000)           # counter 8 is still inside 9.2 months
    assert young_inc[10] == pytest.approx(0)

    student = _worker(age=20, employment="student", income=1200, rent=900)
    grad = table.income_paths(np.full((1, 12), J), student, experience=False)[0]
    assert grad[0] == pytest.approx(0)
    assert grad[6] == pytest.approx(1330)
    assert grad[10] == pytest.approx(0)

    unemployed = _worker(age=30, employment="unemployed", income=533, rent=1800)
    assert table.income_paths(np.full((1, 3), J), unemployed, experience=False)[0, 2] == pytest.approx(533)

    retired = table.income_paths(np.full((1, 1), R), employed, experience=False)[0, 0]
    assert retired == pytest.approx(10000 * 0.475)

    grown = table.income_paths(np.full((1, 13), E), _worker(age=22, income=5000), experience=True)[0]
    assert grown[0] == pytest.approx(5000)
    assert grown[12] == pytest.approx(5000 * 1.068)

    older = table.income_paths(np.full((1, 13), E), _worker(age=45, income=5000), experience=True)[0]
    assert older[12] == pytest.approx(5000 * 1.028)

    abroad = table.income_paths(np.full((1, 1), E), employed,
                                [Intervention("set_attribute", 0, attribute=("abroad", True))])[0, 0]
    assert abroad == pytest.approx(10000 * 0.51)

    rent = table.rent_paths(np.array([[S, S, E, E, J]]), student)[0]
    assert rent[0] == pytest.approx(900) and rent[1] == pytest.approx(900)
    assert rent[2] == pytest.approx(1546) and rent[4] == pytest.approx(1546)
    away = table.rent_paths(np.full((1, 1), E), employed,
                            [Intervention("set_attribute", 0, attribute=("abroad", True))])[0, 0]
    assert away == pytest.approx(2000 * 0.63)
    assert table.params["scaled_spending"] == pytest.approx(0)


def test_masters_field_scales_the_master_entry():
    table = IncomeTable.from_csv()
    notes = (ASSUMPTIONS_DIR / "income_assumptions.csv").read_text()
    assert "The ±15% field gap is a prior, not a survey figure." in notes
    assert table.params["masters_field_same_multiplier"] == pytest.approx(1.0)
    assert table.params["masters_field_higher_multiplier"] == pytest.approx(1.15)
    assert table.params["masters_field_lower_multiplier"] == pytest.approx(0.85)

    alexia = load_profile("P00001")
    entry = table.params["entry_salary_chf"]
    master = entry * (1.0 + table.params["masters_premium_pct"])
    states = np.full((1, 25), S)
    states[0, 24] = E
    bare = [Intervention("set_attribute", 24, attribute=("has_masters", True))]
    got = table.income_paths(states, alexia, bare, experience=False)[0]
    assert got[23] == pytest.approx(alexia.ref_income)
    assert got[24] == pytest.approx(master)

    for field, factor in (("same", 1.0), ("higher", 1.15), ("lower", 0.85)):
        iv = bare + [Intervention("set_attribute", 0, attribute=("masters_field", field))]
        wage = table.full_time_salary(alexia, {"has_masters": True, "masters_field": field})
        assert wage == pytest.approx(master * factor)
        assert table.income_paths(states, alexia, iv, experience=False)[0, 24] == pytest.approx(wage)
        # The field is chosen at the start. It does not change the stipend before the degree.
        early = table.income_paths(
            states, alexia,
            [Intervention("set_attribute", 0, attribute=("masters_field", field))],
            experience=False,
        )[0]
        assert early[0] == pytest.approx(alexia.ref_income)
        assert early[24] == pytest.approx(entry)

    with pytest.raises(ValueError):
        table.full_time_salary(alexia, {"has_masters": True, "masters_field": "medicine"})

    scaled = table.with_overrides({"scaled_spending": 1})
    higher_iv = [
        Intervention("set_attribute", 0, attribute=("masters_field", "higher")),
        Intervention("set_attribute", 24, attribute=("has_masters", True)),
    ]
    lower_iv = [
        Intervention("set_attribute", 0, attribute=("masters_field", "lower")),
        Intervention("set_attribute", 24, attribute=("has_masters", True)),
    ]
    housing_h, _other = scaled.consumption_paths(states, alexia, higher_iv, experience=False)
    housing_l, _other = scaled.consumption_paths(states, alexia, lower_iv, experience=False)
    assert housing_h[0, 23] == pytest.approx(900) and housing_l[0, 23] == pytest.approx(900)
    assert housing_h[0, 24] > housing_l[0, 24]
    high_offer = master * 1.15
    assert housing_h[0, 24] == pytest.approx(
        float(scaled.rent_from_earnings(high_offer)) + scaled.params["rent_charges_chf"]
    )


def test_zurich_scales_housing_only_and_germany_ui_stays_swiss():
    df = pd.read_csv(ASSUMPTIONS_DIR / "income_assumptions.csv")
    for name in ("zurich_housing_multiplier", "zurich_move_cost_chf", "germany_move_cost_chf"):
        source = str(df.loc[df["parameter"] == name, "source"].iloc[0])
        assert source.strip().lower() == "prior"

    table = IncomeTable.from_csv().with_overrides({"apply_income_tax": 0})  # rules on gross wages
    worker = _worker(age=30, income=8000, rent=2000)
    moved = table.income_paths(
        np.array([[E, E, J, J, J, J, E]]),
        worker,
        [Intervention("set_attribute", 2, attribute=("abroad", True))],
        experience=False,
    )[0]
    assert moved[1] == pytest.approx(8000)
    assert moved[2] == pytest.approx(5600)          # 70% of the Swiss wage, not of the German wage
    assert moved[5] == pytest.approx(5600)
    assert moved[6] == pytest.approx(8000 * 0.51)

    zurich = [Intervention("set_attribute", 1, attribute=("zurich", True))]
    pay = table.income_paths(np.array([[E, E, E]]), worker, zurich, experience=False)[0]
    assert pay == pytest.approx([8000, 8000, 8000])
    rent = table.rent_paths(np.array([[E, E, E]]), worker, zurich)[0]
    assert rent[0] == pytest.approx(2000)
    assert rent[1] == pytest.approx(2000 * 1.20)
    assert rent[2] == pytest.approx(rent[1])

    scaled = table.with_overrides({"scaled_spending": 1})
    housing, other = scaled.consumption_paths(np.array([[E, E, E]]), worker, zurich, experience=False)
    assert housing[0, 1] == pytest.approx(housing[0, 0] * 1.20)
    assert other[0, 1] == pytest.approx(other[0, 0])

    away = table.rent_paths(
        np.array([[E, E]]), worker,
        [Intervention("set_attribute", 1, attribute=("abroad", True))],
    )[0]
    assert away[0] == pytest.approx(2000)
    assert away[1] == pytest.approx(2000 * 0.63)


def test_bachelor_age_wage_and_phd_premium():
    """Year-5 switch drops the master's premium. A PhD adds another 1.10, as a guess."""
    table = IncomeTable.from_csv()
    alexia = load_profile("P00001")
    entry = table.params["entry_salary_chf"]
    g = table.params["annual_experience_growth"]
    grown = entry * (1.0 + g) ** 3
    assert table.aged_bachelor_salary(alexia, 24, 69) == pytest.approx(grown)

    T = 110
    worked = np.full((1, T), E)
    worked[0, :24] = S
    masters = [Intervention("set_attribute", 24, attribute=("has_masters", True))]
    stay = table.income_paths(worked, alexia, masters)[0]
    assert stay[24] == pytest.approx(entry * 1.10)
    assert stay[60] == pytest.approx(grown * 1.10)
    assert stay[69] == pytest.approx(grown * 1.10)

    gappy = worked.copy()
    gappy[0, 35] = J
    switch = masters + [
        Intervention("set_attribute", 69, attribute=("bachelor_age_wage", 24)),
        Intervention("set_attribute", 69, attribute=("has_masters", False)),
    ]
    switched = table.income_paths(gappy, alexia, switch)[0]
    missed = table.income_paths(gappy, alexia, masters)[0, 69] / 1.10
    assert switched[69] == pytest.approx(grown)
    assert missed == pytest.approx(entry * (1.0 + g) ** 2)
    assert switched[69] == pytest.approx(missed * (1.0 + g))

    phd_states = worked.copy()
    phd_states[0, 60:102] = S
    phd = masters + [Intervention("set_attribute", 102, attribute=("has_phd", True))]
    phd_pay = table.income_paths(phd_states, alexia, phd)[0]
    assert np.allclose(phd_pay[60:102], 1200.0)
    assert phd_pay[102] == pytest.approx(stay[60] * 1.10)
    assert phd_pay[102] == pytest.approx(entry * 1.10 * 1.10 * (1.0 + g) ** 3)

    scaled = table.with_overrides({"scaled_spending": 1})
    housing, other = scaled.consumption_paths(phd_states, alexia, phd)
    assert np.allclose(housing[0, 60:102] + other[0, 60:102], scaled.params["student_living_chf"])


def test_scaled_spending_uses_the_expense_priors_recipe():
    table = IncomeTable.from_csv().with_overrides({"scaled_spending": 1, "apply_income_tax": 0})
    p = table.params
    student = load_profile("P00001")
    states = np.array([[S, S, E, E]])
    rent = table.rent_paths(states, student)[0]
    assert rent[0] == pytest.approx(900) and rent[1] == pytest.approx(900)
    entry_net = float(table.rent_from_earnings(p["entry_salary_chf"]))
    assert p["rent_floor_chf"] < entry_net < p["rent_cap_chf"]
    assert rent[2] == pytest.approx(entry_net + p["rent_charges_chf"])
    assert rent[3] == pytest.approx(rent[2])

    # The lease is signed at the starting wage and does not follow later raises.
    seq = np.full((1, 14), E)
    seq[0, 13] = J
    grown = table.rent_paths(seq, _worker(age=22, income=5000))[0]
    assert grown[12] == pytest.approx(grown[0])
    assert grown[13] == pytest.approx(grown[0])

    rich = float(table.rent_paths(np.full((1, 1), E), _worker(income=80000))[0, 0])
    assert rich == pytest.approx(p["rent_cap_chf"] + p["rent_charges_chf"])

    anchor = _worker(income=p["rent_income_anchor_chf"])
    housing, other = table.consumption_paths(np.full((1, 1), E), anchor)
    assert housing[0, 0] == pytest.approx(p["rent_anchor_chf"] + p["rent_charges_chf"])
    assert other[0, 0] == pytest.approx(p["nonhousing_base_chf"] + p["nonhousing_rate"] * p["rent_income_anchor_chf"])

    # Student months use the EPFL all-in budget, not 0.25 x gross and not 1.75 x rent.
    sh, so = table.consumption_paths(np.full((1, 1), S), student)
    assert sh[0, 0] == pytest.approx(900)
    assert sh[0, 0] + so[0, 0] == pytest.approx(p["student_living_chf"])

    shocks = sample_yearly_lump(np.random.default_rng(0), 20000, 12, p).sum(axis=1)
    assert shocks.mean() == pytest.approx(0.0, abs=150)


def test_masters_after_work_signs_the_lease_on_the_graduation_wage():
    """A job before the master's must not freeze the lease on the entry salary.

    The premium and the new lease both use the wage at graduation: entry grown
    by the months she worked, then times 1.10. The no-history offer is only
    the wage when she has not worked.
    """
    table = IncomeTable.from_csv().with_overrides({"scaled_spending": 1})
    alexia = load_profile("P00001")
    p = table.params
    entry = p["entry_salary_chf"]
    premium = 1.0 + p["masters_premium_pct"]

    def wage_at_graduation(work_months: int) -> float:
        salary = entry
        for t in range(work_months):
            if (t + 1) % 12 == 0:
                salary *= 1.0 + table._growth_rate(alexia.age + (t + 1) / 12.0)
        return salary * premium

    for work in (0, 24, 36):
        grad = work + 24
        ivs = []
        if work:
            ivs.append(Intervention("force_state", 0, work, state="employed"))
        ivs.append(Intervention("force_state", work, 24, state="student"))
        ivs.append(Intervention("set_attribute", grad, attribute=("has_masters", True)))
        forced, _prev = _schedule(grad + 2, ivs)
        if work:
            assert (forced[:work] == E).all()
        assert (forced[work:grad] == S).all()
        assert forced[grad] == E

        states = np.full((1, grad + 4), E)
        states[0, :grad] = S
        if work:
            states[0, :work] = E
        income = table.income_paths(states, alexia, ivs)[0]
        housing, _other = table.consumption_paths(states, alexia, ivs)
        wage = wage_at_graduation(work)
        assert income[grad] == pytest.approx(wage)
        assert table.lease_signed_on[0, grad] == pytest.approx(wage)
        assert housing[0, grad] == pytest.approx(table.rent_from_earnings(wage) + p["rent_charges_chf"])
        assert housing[0, grad + 1] == pytest.approx(housing[0, grad])
        assert housing[0, grad - 1] == pytest.approx(alexia.rent)
        assert table.lease_signed_on[0, grad - 1] == pytest.approx(0.0)
        if work == 0:
            assert wage == pytest.approx(entry * premium)
        else:
            assert wage > entry * premium
            assert table.lease_signed_on[0, 0] == pytest.approx(entry)
            assert abs(housing[0, grad] - housing[0, 0]) > 1.0


def test_rent_step_lifts_the_two_room_cap_and_purchase_keeps_the_bill():
    table = IncomeTable.from_csv().with_overrides({"scaled_spending": 1})
    p = table.params
    rich = _worker(age=30, income=80000)
    states = np.full((1, 80), E)
    keep = table.rent_paths(states, rich)[0]
    assert keep[0] == pytest.approx(p["rent_cap_chf"] + p["rent_charges_chf"])
    stepped = table.rent_paths(
        states, rich, [Intervention("set_attribute", 72, attribute=("rent_step", True))],
    )[0]
    assert stepped[:72] == pytest.approx(keep[:72])
    stepped_net = p["rent_cap_chf"] * p["rent_step_factor"]
    assert stepped_net > p["rent_cap_chf"]
    assert stepped[72] == pytest.approx(stepped_net + p["rent_charges_chf"])
    assert stepped[79] == pytest.approx(stepped[72])
    # Unemployment does not cut the stepped lease.
    gap = states.copy()
    gap[0, 75] = J
    stepped_gap = table.rent_paths(
        gap, rich, [Intervention("set_attribute", 72, attribute=("rent_step", True))],
    )[0]
    assert stepped_gap[75] == pytest.approx(stepped[72])

    owned = table.rent_paths(
        states, rich, [Intervention("set_attribute", 72, attribute=("owns_home", True))],
    )[0]
    assert owned == pytest.approx(keep)

    alexia = load_profile("P00001")
    seq = np.full((1, 80), E)
    seq[0, :24] = S
    masters = [
        Intervention("force_state", 0, 24, state="student"),
        Intervention("set_attribute", 24, attribute=("has_masters", True)),
    ]
    base = table.rent_paths(seq, alexia, masters)[0]
    step = table.rent_paths(
        seq, alexia, masters + [Intervention("set_attribute", 72, attribute=("rent_step", True))],
    )[0]
    bought = table.rent_paths(
        seq, alexia, masters + [Intervention("set_attribute", 72, attribute=("owns_home", True))],
    )[0]
    assert base[:24] == pytest.approx(alexia.rent)
    assert np.allclose(base[24:], base[24])
    net = base[24] - p["rent_charges_chf"]
    assert p["rent_floor_chf"] < net < p["rent_cap_chf"]
    assert net * p["rent_step_factor"] > p["rent_cap_chf"]
    assert step[:72] == pytest.approx(base[:72])
    assert step[72] == pytest.approx(net * p["rent_step_factor"] + p["rent_charges_chf"])
    assert bought == pytest.approx(base)


def test_purchase_deposit_is_a_fraction_of_cash_that_month():
    cash = np.array([[1000.0, 2500.0, 4000.0], [-100.0, -50.0, 10.0]])
    net = np.array([[1000.0, 1500.0, 1500.0], [-100.0, 50.0, 60.0]])
    cp = CashPaths(0.0, cash, {"net_flow_chf": net, "one_off_costs_chf": np.zeros_like(cash)})
    assert cash_engine.reconciles(cp)
    out, deposit = cash_engine.apply_fractional_drop(cp, 1, 0.15)
    assert deposit == pytest.approx([375.0, 0.0])
    assert out.cash[:, 0] == pytest.approx(cash[:, 0])
    assert out.cash[:, 1] == pytest.approx([2125.0, -50.0])
    assert out.cash[:, 2] == pytest.approx([3625.0, 10.0])
    assert cash_engine.reconciles(out)
    assert cash_engine.reconciles(cp)


def test_forward_filter_matches_a_hand_calculation():
    A = np.array([[0.7, 0.3], [0.4, 0.6]])
    log_lik = np.log([[0.5, 0.5], [0.2, 0.8]])
    filtered, total = forward_filter(log_lik, A, np.array([0.8, 0.2]))
    assert filtered[0] == pytest.approx([0.8, 0.2])
    assert filtered[1] == pytest.approx([0.128 / 0.416, 0.288 / 0.416])
    assert total == pytest.approx(np.log(0.5) + np.log(0.416))
    assert filtered.shape == (2, 2)


def test_emission_fit_is_heavy_tailed_and_shared(fitted):
    assert fitted.family == "student_t"
    medians = []
    for state in STATES:
        p = fitted.params[state]
        medians.append(fitted.diagnostics["by_state"][state]["median_spend_ratio"])
        assert 2.0 < p["df"] < 30
        assert 1.4 < p["loc"] < 2.2
        block = fitted.diagnostics["by_state"][state]
        # Student and job_search holdouts are a handful of profiles, so the KS moves around.
        assert block["ks_stat"] is not None and block["ks_stat"] < 0.25
        assert len(block["qq_holdout"]) == len(block["qq_model"]) == 10
    assert all(1.6 <= m <= 2.0 for m in medians)
    ratio = fitted.diagnostics["self_employed_scale_ratio"]
    within = fitted.diagnostics["self_employed_within_sd_ratio"]
    assert 2.3 < ratio < 3.5
    assert 2.5 < within < 3.6
    shift = fitted.household_shift["employed"]
    assert shift["shared_housing"] > shift["couple"] + 0.3
    assert shift["single"] > shift["couple"]
    assert 0.2 < fitted.december_bonus_quantiles["employed"][50] < 0.45
    assert set(fitted.diagnostics["holdout_ids"]) == set(STATES)
    assert len(fitted.diagnostics["holdout_ids"]["job_search"]) == 7


def test_emission_loglik_floors_and_scores_december(fitted):
    ratios = np.array([1.75, 50.0])
    ll = emission_loglik(fitted, ratios, np.array([6, 12]), household="single")
    assert ll.shape == (2, 5)
    assert np.isfinite(ll).all()
    assert ll[1].min() == pytest.approx(np.log(1e-6))
    # A December observation equal to the body level is more likely once the employed
    # offset (the bonus) is removed than the raw figure would be. The offset is negative.
    assert fitted.december_offset["employed"] < -0.5


def test_simulate_contract_interventions_and_heterogeneity(engine):
    alexia = load_profile("P00001")
    paths = engine.simulate(
        alexia, horizon_months=36, n_paths=80, seed=0,
        interventions=[Intervention("force_state", 0, 24, state="student"),
                       Intervention("set_attribute", 24, attribute=("has_masters", True))],
        assumption_overrides={"dirichlet": 0},
    )
    assert paths.states.shape == (80, 36)
    assert paths.income_chf.shape == paths.savings_rate.shape == paths.net_flow_chf.shape == (80, 36)
    assert paths.occupancy.shape == (36, 5) and paths.transition_draws.shape == (80, 5, 5)
    ok = np.abs(paths.income_chf) > 1
    assert np.allclose(paths.net_flow_chf[ok], paths.savings_rate[ok] * paths.income_chf[ok])
    assert (paths.states[:, :24] == S).all()
    assert (paths.states[:, 24] == E).all()
    table = engine.income_table.params
    assert paths.income_chf[0, 0] == pytest.approx(alexia.ref_income)
    offer = table["entry_salary_chf"] * (1 + table["masters_premium_pct"])
    assert paths.gross_income_chf[0, 24] == pytest.approx(offer)
    # What lands in the account is the offer after contributions and tax.
    assert paths.income_chf[0, 24] == pytest.approx(float(net_from_gross(offer, alexia.age + 2, "wage", table)))
    assert paths.income_chf[0, 24] < 0.85 * offer

    worker = _worker()
    base = engine.simulate(worker, horizon_months=24, n_paths=600, seed=1, assumption_overrides={"dirichlet": 0})
    month = (worker.start_month_of_year - 1 + np.arange(24)) % 12 + 1
    ratio = (base.income_chf - base.net_flow_chf) / worker.rent
    body = ratio[:, month != 12]
    # Single households sit a bit above the 1.75 grand median.
    assert 1.6 < np.median(body) < 2.3
    dec = base.net_flow_chf[:, month == 12].mean()
    rest = base.net_flow_chf[:, month != 12].mean()
    assert dec > rest + 500

    def mad(x):
        x = np.asarray(x)
        return np.median(np.abs(x - np.median(x))) * 1.4826

    se = engine.simulate(_worker(employment="self_employed"), horizon_months=24, n_paths=600, seed=1,
                         assumption_overrides={"dirichlet": 0})
    se_ratio = ((se.income_chf - se.net_flow_chf) / 1800)[:, month != 12]
    assert mad(se_ratio) > 2.2 * mad(body)

    couple = engine.simulate(_worker(household="couple"), horizon_months=12, n_paths=500, seed=2,
                             assumption_overrides={"dirichlet": 0})
    shared = engine.simulate(_worker(household="shared_housing"), horizon_months=12, n_paths=500, seed=2,
                             assumption_overrides={"dirichlet": 0})
    # January only, so the bonus is not in the comparison. Same seed, same shocks would
    # differ by household; compare medians.
    c = np.median((couple.income_chf[:, 0] - couple.net_flow_chf[:, 0]) / 1800)
    h = np.median((shared.income_chf[:, 0] - shared.net_flow_chf[:, 0]) / 1800)
    assert h > c + 0.3

    pinned = engine.simulate(worker, horizon_months=8, n_paths=30, seed=0,
                             interventions=[Intervention("force_state", 0, 3, state="part_time")],
                             assumption_overrides={"dirichlet": 0})
    assert (pinned.states[:, :3] == P).all()
    assert (pinned.states[:, 3] == E).all()

    jump = engine.simulate(_worker(employment="student", income=1200, rent=900), horizon_months=1, n_paths=10,
                           interventions=[Intervention("force_state", 0, None, state="employed")])
    assert (jump.states[:, 0] == E).all()


def test_age_hazards_dirichlet_duration_and_occupancy(engine):
    young = engine.simulate(_worker(age=22), horizon_months=24, n_paths=2000, seed=0,
                            assumption_overrides={"dirichlet": 0})
    old = engine.simulate(_worker(age=55), horizon_months=24, n_paths=2000, seed=0,
                          assumption_overrides={"dirichlet": 0})
    # Spell lengths run the other way, so compare entries, not average months in search.
    assert (young.states == J).any(1).mean() > (old.states == J).any(1).mean() + 0.05

    calm = engine.simulate(_worker(), horizon_months=60, n_paths=1200, seed=0, assumption_overrides={"dirichlet": 0})
    storm = engine.simulate(
        _worker(), horizon_months=60, n_paths=1200, seed=0, assumption_overrides={"dirichlet": 0},
        interventions=[Intervention("scale_hazard", 0, None, transition=("employed", "job_search"), factor=5)],
    )
    assert (storm.states == J).mean() > (calm.states == J).mean() + 0.02

    seeker = _worker(age=35, employment="unemployed", income=2000, rent=1500)
    short = engine.simulate(seeker, horizon_months=36, n_paths=800, seed=0,
                            assumption_overrides={"dirichlet": 0, "long_term_unemployment_multiplier": 1})
    long = engine.simulate(seeker, horizon_months=36, n_paths=800, seed=0,
                           assumption_overrides={"dirichlet": 0, "long_term_unemployment_multiplier": 0.05})
    assert (long.states == J).mean() > (short.states == J).mean()

    noisy = engine.simulate(_worker(), horizon_months=12, n_paths=400, seed=0)
    fixed = engine.simulate(_worker(), horizon_months=12, n_paths=400, seed=0,
                            assumption_overrides={"dirichlet": 0})
    assert noisy.transition_draws[:, E, J].std() > 0.001
    assert fixed.transition_draws[:, E, J].std() == pytest.approx(0.0)

    mid = _worker(age=40)
    paths = engine.simulate(mid, horizon_months=36, n_paths=3000, seed=1, assumption_overrides={"dirichlet": 0})
    pi0 = np.eye(5)[E]
    gap = validation.simulated_vs_analytic_occupancy(paths, pi0, engine.priors.matrix(age=40))
    assert gap < 0.05
    assert np.allclose(paths.occupancy.sum(1), 1.0)

    retired = engine.simulate(
        _worker(age=64), horizon_months=18, n_paths=15, seed=0,
        assumption_overrides={"dirichlet": 0, "job_loss_hazard": 0, "employed_to_part_time": 0},
    )
    assert (retired.states[:, 11] == E).all()
    assert (retired.states[:, 12] == R).all()


def test_filter_on_alexia_and_a_shift(engine):
    from twin.data import observed_net_flows

    alexia = load_profile("P00001")
    flows = observed_net_flows("P00001")
    belief = engine.filter(alexia, flows)
    assert belief["student"] > 0.9
    assert sum(belief.values()) == pytest.approx(1.0)

    p = engine.income_table.params
    take_home = float(net_from_gross(p["entry_salary_chf"], alexia.age, "wage", p))
    employed_net = take_home - alexia.rent * 1.75  # a ledger shows net pay
    shocked = list(flows)
    shocked[-6:] = [employed_net] * 6
    moved = engine.filter(alexia, shocked)
    assert moved["employed"] > belief["employed"] + 0.2


def test_parental_leave_prior_pays_eighty_percent_and_holds_housing():
    table = IncomeTable.from_csv().with_overrides({"apply_income_tax": 0})  # rules on gross wages
    worker = _worker(age=22, income=5000)
    states = np.full((1, 30), E)
    leave = [Intervention("parental_leave", 14)]
    inc = table.income_paths(states, worker, leave)[0]
    paid = 5000 * (1 + table.params["annual_experience_growth"])
    assert inc[13] == pytest.approx(paid)
    assert inc[14:26] == pytest.approx(0.80 * paid)
    # The anniversary inside the leave does not raise the wage she returns to.
    assert inc[26] == pytest.approx(paid)
    scaled = table.with_overrides({"scaled_spending": 1})
    housing, other = scaled.consumption_paths(states, worker, leave)
    assert np.allclose(housing[0, 14:26], housing[0, 13])
    assert np.allclose(other[0, 14:26], other[0, 13])
    assert np.allclose(housing[0, 26], housing[0, 13])


def test_stub_parental_leave_returns_to_employed():
    engine = RegimeEngine.stub()
    worker = _worker(age=30, income=8000)
    paths = engine.simulate(
        worker, horizon_months=8, n_paths=3,
        interventions=[Intervention("parental_leave", 2, 3)],
        assumption_overrides={"apply_income_tax": 0},  # rules on gross wages
    )
    assert (paths.states[:, 2:5] == E).all()
    assert (paths.states[:, 5:] == E).all()
    assert paths.income_chf[:, 2:5] == pytest.approx(0.80 * 8000)
    assert paths.income_chf[:, 5] == pytest.approx(8000)


def test_parental_leave_pins_employed_and_skips_the_bonus(engine):
    worker = _worker(age=35, income=10000, rent=1800)
    iv = [Intervention("parental_leave", 0, 12)]
    paths = engine.simulate(
        worker, horizon_months=14, n_paths=25, seed=0, interventions=iv,
        assumption_overrides={
            "dirichlet": 0, "scaled_spending": 1,
            "job_loss_hazard": 0, "employed_to_part_time": 0, "apply_income_tax": 0,
        },
    )
    assert (paths.states == E).all()
    assert paths.income_chf[:, :12] == pytest.approx(8000)
    assert paths.income_chf[:, 12] == pytest.approx(10000)
    table = engine.income_table.with_overrides({"scaled_spending": 1, "apply_income_tax": 0})
    housing, other = table.consumption_paths(paths.states, worker, iv)
    assert np.allclose(housing, housing[:, :1])
    expected = paths.income_chf - housing - other - paths.lump_chf
    assert np.allclose(paths.net_flow_chf, expected)


def test_held_out_emission_report_and_calibration(fitted):
    report = validation.emission_fit_report(model=fitted)
    assert report["by_state"]["employed"]["ks_stat"] < 0.15
    back = validation.calibration_backtest(model=fitted, n_paths=80, seed=0)
    assert back["n_profiles"] == 240
    assert back["filter_accuracy"] > 0.8
    assert 0.7 <= back["coverage"] <= 0.99
    assert back["median_band_width_chf"] > 0


def test_first_job_after_masters_pins_and_pays_the_graduate_wait():
    """The three exits at month 24. Part-time needs its own employed pin.

    A part-time phase returns to the regime before the window. After 24 student
    months that regime is student, so full-time is a one-month employed pin.
    Job search exits to employed by itself. The six search months are the
    never-worked wait (pay 0). The flat under-25 benefit is the seventh month,
    which this decision does not spend in search.
    """
    from run_alexia_first_job_20y import (
        DECISIONS, FULL_TIME_MONTH, GRADUATION_MONTH, HIRED_MONTH, SEARCH_MONTHS, _waiting_rules,
    )
    from twin.regime_engine import _schedule

    alexia = load_profile("P00001")
    table = IncomeTable.from_csv()
    assert alexia.initial_state == "student"
    assert table.params["part_time_fraction"] == pytest.approx(0.60)
    assert table.params["graduate_waiting_months"] == pytest.approx(5.5)
    offer = table.params["entry_salary_chf"] * (1 + table.params["masters_premium_pct"])
    by_name = {name: ivs for name, ivs, _colour in DECISIONS}
    horizon = 60

    forced, _prev = _schedule(horizon, by_name["Employed immediately"])
    assert (forced[:GRADUATION_MONTH] == S).all()
    assert forced[GRADUATION_MONTH] == E
    assert (forced[GRADUATION_MONTH + 1:] == -1).all()

    forced, prev = _schedule(horizon, by_name["Part-time, then full-time"])
    assert (forced[:GRADUATION_MONTH] == S).all()
    assert (forced[GRADUATION_MONTH:FULL_TIME_MONTH] == P).all()
    assert forced[FULL_TIME_MONTH] == E
    assert FULL_TIME_MONTH not in prev

    # Without the employed pin the part-time phase would hand month 48 back to study.
    bare = [iv for iv in by_name["Part-time, then full-time"]
            if not (iv.kind == "force_state" and iv.state == "employed")]
    bare_forced, bare_prev = _schedule(horizon, bare)
    assert bare_forced[FULL_TIME_MONTH] == -2
    assert bare_prev[FULL_TIME_MONTH] == GRADUATION_MONTH

    forced, _prev = _schedule(horizon, by_name["Search 6 months, then work"])
    assert (forced[GRADUATION_MONTH:HIRED_MONTH] == J).all()
    assert forced[HIRED_MONTH] == E

    states = np.full((1, horizon), E)
    states[0, :GRADUATION_MONTH] = S
    immediate = table.income_paths(states, alexia, by_name["Employed immediately"])[0]
    assert immediate[0] == pytest.approx(alexia.ref_income)
    assert immediate[GRADUATION_MONTH:GRADUATION_MONTH + 6] == pytest.approx(offer)

    states = np.full((1, horizon), E)
    states[0, :GRADUATION_MONTH] = S
    states[0, GRADUATION_MONTH:FULL_TIME_MONTH] = P
    part = table.income_paths(states, alexia, by_name["Part-time, then full-time"])[0]
    assert part[GRADUATION_MONTH:GRADUATION_MONTH + 6] == pytest.approx(offer * 0.60)
    # Two work anniversaries fall inside the part-time window (months 35 and 47).
    # The raise is booked after that month, and she is still under 30.
    g = table.params["annual_experience_growth"]
    assert part[35] == pytest.approx(offer * 0.60)
    assert part[36] == pytest.approx(offer * (1 + g) * 0.60)
    assert part[FULL_TIME_MONTH] == pytest.approx(offer * (1 + g) ** 2)

    states = np.full((1, horizon), E)
    states[0, :GRADUATION_MONTH] = S
    states[0, GRADUATION_MONTH:HIRED_MONTH] = J
    search = table.income_paths(states, alexia, by_name["Search 6 months, then work"])[0]
    assert search[GRADUATION_MONTH:HIRED_MONTH] == pytest.approx(0)
    assert search[HIRED_MONTH] == pytest.approx(offer)
    assert _waiting_rules(table, alexia) == ["graduate_waiting"] * SEARCH_MONTHS

    stayed = states.copy()
    stayed[0, HIRED_MONTH] = J
    seventh = table.income_paths(stayed, alexia, by_name["Search 6 months, then work"])[0, HIRED_MONTH]
    assert alexia.age + HIRED_MONTH / 12.0 < 25
    assert seventh == pytest.approx(table.params["graduate_benefit_under_25_chf"])


def test_goal_hit_rate_on_a_known_fan_and_the_csv_row():
    """Hit rate is the share at the goal month. The crossing month is where the median stays above.

    Quantiles are NumPy's linear method. A goal past the horizon is scored on the
    last simulated month, and that choice is reported.
    """
    cash = np.array([
        [0.0, 10.0, 10.0, 40.0, 40.0],
        [0.0, 30.0, 10.0, 60.0, 80.0],
        [0.0, 50.0, 10.0, 80.0, 120.0],
    ])
    # Cross-sectional median: 0, 30, 10, 60, 80. Target 50, so it stays from month 3.
    goal = {"goal_name": "Buffer", "target_amount_chf": 50.0, "target_date": "2025-04-30", "priority": "high"}
    scored = goal_hit_rate(cash, "2025-01", goal)
    assert scored["goal_month"] == 3 and scored["evaluated_month"] == 3
    assert scored["evaluated_at"] == "goal_month" and scored["evaluated_label"] == "2025-04"
    assert scored["hit_rate"] == pytest.approx(2 / 3)
    assert scored["median_cross_month"] == 3 and scored["median_cross_label"] == "2025-04"
    assert scored["median_cash_at_goal"] == pytest.approx(60)
    assert scored["p10_cash_at_goal"] == pytest.approx(44)
    assert scored["p90_cash_at_goal"] == pytest.approx(76)
    assert scored["target_amount_chf"] == pytest.approx(50) and scored["goal_name"] == "Buffer"

    # A one-month blip does not count. A later dip cancels an earlier crossing.
    blip = np.array([[0.0, 80.0, 0.0, 80.0, 10.0]])
    blipped = goal_hit_rate(blip, date(2025, 1, 1), (50.0, "2025-02"))
    assert blipped["goal_month"] == 1 and blipped["hit_rate"] == pytest.approx(1.0)
    assert blipped["median_cross_month"] is None and blipped["median_cross_label"] is None
    held = goal_hit_rate(np.array([[49.0, 50.0, 50.0]]), "2025-01", (50.0, date(2025, 3, 31)))
    assert held["median_cross_month"] == 1
    assert held["hit_rate"] == pytest.approx(1.0)
    assert held["median_cash_at_goal"] == pytest.approx(50)

    short = cash[:, :2]
    clipped = goal_hit_rate(CashPaths(0.0, short, {}), "2025-01", goal)
    assert clipped["goal_month"] == 3 and clipped["evaluated_month"] == 1
    assert clipped["evaluated_at"] == "last_simulated_month"
    assert clipped["evaluated_label"] == "2025-02"
    # Last month is 10, 30, 50. Only one path meets 50.
    assert clipped["hit_rate"] == pytest.approx(1 / 3)
    assert clipped["median_cash_at_goal"] == pytest.approx(30)
    assert clipped["median_cross_month"] is None
    early = goal_hit_rate(cash, "2025-01", {"amount_chf": 50, "date": "2024-06-01"})
    assert early["goal_month"] == -7 and early["evaluated_at"] == "before_start"
    assert early["evaluated_month"] == 0 and early["hit_rate"] == pytest.approx(0.0)

    goals = load_goals()
    alexia = goals.loc[goals["profile_id"] == "P00001"]
    row = alexia.iloc[0]
    wide = np.full((4, 60), 0.0)
    wide[:, 59] = [0.0, 12800.0, 12800.0, 20000.0]
    from_row = goal_hit_rate(wide, "2025-01", row)
    from_tuple = goal_hit_rate(wide, date(2025, 1, 1), next(alexia.itertuples(index=False)))
    assert row["goal_name"] == "Emergency savings" and row["target_amount_chf"] == pytest.approx(12800)
    assert from_row["target_date"] == "2029-12-31" and from_row["priority"] == "high"
    assert from_row["goal_month"] == 59 and from_row["evaluated_at"] == "goal_month"
    assert from_row["evaluated_label"] == "2029-12"
    assert from_row["hit_rate"] == pytest.approx(0.75)
    assert from_row["median_cash_at_goal"] == pytest.approx(12800)
    assert from_tuple["hit_rate"] == from_row["hit_rate"]
    assert from_tuple["goal_month"] == from_row["goal_month"]
    with pytest.raises(TypeError):
        goal_hit_rate(wide, "2025-01", alexia)


def _rolled_debt(months: int, accrual: float, monthly_rate: float) -> list[float]:
    debt = 0.0
    out = []
    for _ in range(months):
        debt = debt * (1.0 + monthly_rate) + accrual
        out.append(debt)
    return out


def test_masters_pin_accrues_tuition_debt_and_other_paths_do_not(engine):
    """A 24-month master's increases the debt total. A path without that pin does not.

    The window is force_state student for 24 months with has_masters set at month 24.
    Debt is last month's total grown by the monthly loan rate, plus that month's
    tuition/12. Living costs stay in net_flow_chf.
    """
    assert STATES == ["employed", "part_time", "job_search", "student", "retired"]
    alexia = load_profile("P00001")
    tuition = scenario_annual_tuition("P00001")
    assert tuition == pytest.approx(8000)
    assert liability_annual_rate("P00001") is None
    assert alexia.use_existing_loan_rate is False
    annual_rate = float(engine.income_table.params["tuition_loan_annual_rate"])
    assert annual_rate == pytest.approx(0.0)
    notes = (ASSUMPTIONS_DIR / "income_assumptions.csv").read_text()
    rate_line = next(line for line in notes.splitlines() if line.startswith("tuition_loan_annual_rate,"))
    assert "https://www.fedlex.admin.ch/eli/cc/2023/473/de" in rate_line
    assert "ETH Zurich's own student-loan rate, not a national SBFI rate" in rate_line
    assert "21 Nov 2026" in rate_line

    pin = [
        Intervention("force_state", 0, 24, state="student"),
        Intervention("set_attribute", 24, attribute=("has_masters", True)),
    ]
    paths = engine.simulate(
        alexia, horizon_months=36, n_paths=4, seed=0, interventions=pin,
        assumption_overrides={"dirichlet": 0},
    )
    accrual = tuition / 12.0
    monthly = annual_rate / 12.0
    rolled = _rolled_debt(24, accrual, monthly)
    assert paths.debt_chf.shape == paths.net_flow_chf.shape == (4, 36)
    assert paths.states.shape == paths.income_chf.shape == paths.savings_rate.shape
    assert np.allclose(paths.debt_chf, paths.debt_chf[0])
    assert paths.debt_chf[0, :24] == pytest.approx(rolled)
    assert paths.debt_chf[0, 23] == pytest.approx(16000)
    assert paths.debt_chf[0, 24:] == pytest.approx(rolled[-1])
    assert paths.tuition_accrual_chf[0, :24] == pytest.approx(accrual)
    assert paths.tuition_accrual_chf[0, 24:] == pytest.approx(0)
    assert (paths.states[:, :24] == S).all()
    assert (paths.states[:, 24] == E).all()
    ok = np.abs(paths.income_chf) > 1
    assert np.allclose(paths.net_flow_chf[ok], paths.savings_rate[ok] * paths.income_chf[ok])
    cash = cash_engine.run(alexia, paths)
    assert cash_engine.reconciles(cash)
    expected_cash = alexia.opening_cash + np.cumsum(paths.net_flow_chf + paths.tuition_accrual_chf, axis=1)
    assert cash.cash == pytest.approx(expected_cash)

    stub = RegimeEngine.stub().simulate(alexia, horizon_months=36, n_paths=2, interventions=pin)
    assert stub.debt_chf[0] == pytest.approx(paths.debt_chf[0])

    no_masters = engine.simulate(
        alexia, horizon_months=36, n_paths=4, seed=0, assumption_overrides={"dirichlet": 0},
    )
    assert np.all(no_masters.debt_chf == 0)
    assert np.all(no_masters.tuition_accrual_chf == 0)
    working = engine.simulate(
        _worker(), horizon_months=36, n_paths=4, seed=1,
        interventions=[Intervention("force_state", 0, None, state="employed")],
        assumption_overrides={"dirichlet": 0},
    )
    assert np.all(working.debt_chf == 0)
    student_only = engine.simulate(
        alexia, horizon_months=36, n_paths=2, seed=0,
        interventions=[Intervention("force_state", 0, 24, state="student")],
        assumption_overrides={"dirichlet": 0},
    )
    assert np.all(student_only.debt_chf == 0)

    later = pin + [Intervention("force_state", 30, 6, state="student")]
    again = engine.simulate(
        alexia, horizon_months=40, n_paths=2, seed=0, interventions=later,
        assumption_overrides={"dirichlet": 0},
    )
    assert again.debt_chf[0, 23] == pytest.approx(rolled[-1])
    assert again.debt_chf[0, 24:] == pytest.approx(rolled[-1])
    assert again.tuition_accrual_chf[0, 30:36] == pytest.approx(0)

    priced = engine.simulate(
        alexia, horizon_months=24, n_paths=1, seed=0, interventions=pin,
        assumption_overrides={"dirichlet": 0, "tuition_loan_annual_rate": 0.12},
    )
    assert priced.debt_chf[0, -1] == pytest.approx(_rolled_debt(24, accrual, 0.12 / 12.0)[-1])
    assert float(priced.debt_chf[0, -1]) > 16000

    other_tuition = scenario_annual_tuition("P00002")
    assert other_tuition == pytest.approx(1200)
    other = engine.simulate(
        load_profile("P00002"), horizon_months=1, n_paths=1, seed=0, interventions=pin,
        assumption_overrides={"dirichlet": 0},
    )
    assert other.debt_chf[0, 0] == pytest.approx(other_tuition / 12.0)

    own_rate = liability_annual_rate("P00013")
    assert own_rate == pytest.approx(0.0593)
    borrower = load_profile("P00013")
    assert borrower.use_existing_loan_rate is False
    ignored = engine.simulate(
        borrower, horizon_months=2, n_paths=1, seed=0, interventions=pin,
        assumption_overrides={"dirichlet": 0},
    )
    own_accrual = scenario_annual_tuition("P00013") / 12.0
    # False ignores the personal-loan rate, so two months are principal only.
    assert ignored.debt_chf[0, 1] == pytest.approx(own_accrual * 2.0)
    personal = own_accrual * (2.0 + own_rate / 12.0)
    assert abs(float(ignored.debt_chf[0, 1]) - personal) > 0.01


def test_use_existing_loan_rate_is_read_when_the_masters_pin_starts(engine):
    """True uses the profile loan rate. False ignores it. An override still wins.

    The flag is the profile field, unless a set_attribute of the same name is already
    in force at the first month of the master's window.
    """
    pin = [
        Intervention("force_state", 0, 24, state="student"),
        Intervention("set_attribute", 24, attribute=("has_masters", True)),
    ]
    own_rate = liability_annual_rate("P00013")
    accrual = scenario_annual_tuition("P00013") / 12.0
    personal = accrual * (2.0 + own_rate / 12.0)

    opted_in = load_profile("P00013")
    opted_in.use_existing_loan_rate = True
    borrowed = engine.simulate(
        opted_in, horizon_months=2, n_paths=1, seed=0, interventions=pin,
        assumption_overrides={"dirichlet": 0},
    )
    assert borrowed.debt_chf[0, 1] == pytest.approx(personal)

    no_loan = load_profile("P00001")
    no_loan.use_existing_loan_rate = True
    assert liability_annual_rate("P00001") is None
    still_zero = engine.simulate(
        no_loan, horizon_months=2, n_paths=1, seed=0, interventions=pin,
        assumption_overrides={"dirichlet": 0},
    )
    alexia_accrual = scenario_annual_tuition("P00001") / 12.0
    assert still_zero.debt_chf[0, 1] == pytest.approx(alexia_accrual * 2.0)

    overridden = engine.simulate(
        opted_in, horizon_months=2, n_paths=1, seed=0, interventions=pin,
        assumption_overrides={"dirichlet": 0, "tuition_loan_annual_rate": 0.12},
    )
    assert overridden.debt_chf[0, 1] == pytest.approx(accrual * (2.0 + 0.12 / 12.0))

    # The pin starts at month 2. A flag set before that month is the one that counts.
    delayed = [
        Intervention("set_attribute", 0, attribute=("use_existing_loan_rate", True)),
        Intervention("force_state", 2, 2, state="student"),
        Intervention("set_attribute", 4, attribute=("has_masters", True)),
    ]
    held = load_profile("P00013")
    assert held.use_existing_loan_rate is False
    from_attribute = engine.simulate(
        held, horizon_months=4, n_paths=1, seed=0, interventions=delayed,
        assumption_overrides={"dirichlet": 0},
    )
    assert from_attribute.debt_chf[0, :2] == pytest.approx(0)
    assert from_attribute.debt_chf[0, 3] == pytest.approx(personal)

    # Written after the pin has started: the choice was already false.
    late = pin + [Intervention("set_attribute", 1, attribute=("use_existing_loan_rate", True))]
    missed = engine.simulate(
        held, horizon_months=2, n_paths=1, seed=0, interventions=late,
        assumption_overrides={"dirichlet": 0},
    )
    assert missed.debt_chf[0, 1] == pytest.approx(accrual * 2.0)


def test_investments_total_tracks_positive_net_flow_and_is_not_a_mortgage(engine):
    """Investments sit beside cash and debt: a path total, not a regime and not a mortgage."""
    alexia = load_profile("P00001")
    paths = engine.simulate(
        alexia, horizon_months=6, n_paths=3, seed=0, assumption_overrides={"dirichlet": 0},
    )
    assert paths.investments_chf.shape == paths.net_flow_chf.shape == paths.debt_chf.shape
    assert np.allclose(paths.investments_chf, np.cumsum(np.maximum(paths.net_flow_chf, 0.0), axis=1))
    assert "mortgage" not in paths.__dict__
    priced = engine.simulate(
        alexia, horizon_months=2, n_paths=2, seed=0,
        assumption_overrides={"dirichlet": 0, "investment_annual_rate": 0.12},
    )
    net = priced.net_flow_chf
    monthly = 0.12 / 12.0
    first = np.maximum(net[:, 0], 0.0)
    second = first * (1.0 + monthly) + np.maximum(net[:, 1], 0.0)
    assert priced.investments_chf[:, 0] == pytest.approx(first)
    assert priced.investments_chf[:, 1] == pytest.approx(second)


def test_simulate_until_next_decision_matches_simulate_and_rejects_unknowns(engine):
    import inspect

    from twin import branch_decisions
    from twin.data import observed_net_flows
    from twin.twin_session import decision_interventions
    from twin.types import DecisionSnapshot

    sim = inspect.signature(RegimeEngine.simulate)
    filt = inspect.signature(RegimeEngine.filter)
    assert list(sim.parameters) == [
        "self", "profile", "horizon_months", "n_paths", "interventions",
        "assumption_overrides", "initial_belief", "seed",
    ]
    assert list(filt.parameters) == ["self", "profile", "observed_net_flows", "assumption_overrides"]
    assert sim.parameters["horizon_months"].default == 60
    assert sim.parameters["n_paths"].default == 5000
    assert sim.parameters["interventions"].default is None
    assert sim.parameters["initial_belief"].default is None
    assert sim.parameters["seed"].default == 0
    assert filt.parameters["assumption_overrides"].default is None

    alexia = load_profile("P00001")
    wrapped = engine.simulate_until_next_decision(
        alexia, "masters", from_month=0, until_month=24, n_paths=6, seed=0,
        assumption_overrides={"dirichlet": 0},
    )
    ivs, _offs = decision_interventions("masters", 0, {})
    direct = engine.simulate(
        alexia, horizon_months=24, n_paths=6, seed=0, interventions=ivs,
        assumption_overrides={"dirichlet": 0},
    )
    assert np.array_equal(wrapped.states, direct.states)
    assert np.allclose(wrapped.income_chf, direct.income_chf)
    assert np.allclose(wrapped.net_flow_chf, direct.net_flow_chf)
    assert (wrapped.states[:, :24] == S).all()
    assert wrapped.horizon == 24

    with pytest.raises(ValueError):
        engine.simulate_until_next_decision(alexia, "university", from_month=0, until_month=12)
    with pytest.raises(ValueError):
        engine.simulate_until_next_decision(alexia, "buy_home", from_month=0, until_month=12)
    with pytest.raises(ValueError):
        engine.simulate_until_next_decision(alexia, "masters", from_month=5, until_month=5)

    home = engine.simulate_until_next_decision(
        alexia, "owns_home", from_month=0, until_month=2, n_paths=2, seed=0,
        assumption_overrides={"dirichlet": 0},
    )
    home_iv, home_off = decision_interventions("owns_home", 0, {})
    assert len(home_iv) == 1 and home_iv[0].kind == "set_attribute"
    assert home_iv[0].attribute == ("owns_home", True) and home_off == []
    assert "mortgage" not in home.__dict__

    later = engine.simulate_until_next_decision(
        alexia, "masters", from_month=12, until_month=24, n_paths=3, seed=0,
        assumption_overrides={"dirichlet": 0},
    )
    assert later.horizon == 12 and (later.states == S).all()

    flows = observed_net_flows("P00001")
    overrides = {"dirichlet": 0}
    belief = engine.filter(alexia, flows, overrides)
    assert belief["student"] > 0.9
    attrs = set(engine.__dict__)
    snap = DecisionSnapshot(
        month=0, profile=alexia, belief=belief, observed_net_flows=list(flows),
        interventions=[], one_offs=[], assumption_overrides=overrides,
        seed=1, parent_id=None, snapshot_id="root",
    )
    assert engine.filter(snap.profile, snap.observed_net_flows, snap.assumption_overrides) == snap.belief
    assert set(engine.__dict__) == attrs
    before = list(snap.observed_net_flows)
    calls = {"n": 0}
    original = engine.filter

    def _counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    engine.filter = _counting
    try:
        first = branch_decisions(engine, snap, ["masters", "start_working"], until_month=12, n_paths=4)
        second = branch_decisions(engine, snap, ["masters", "start_working"], until_month=12, n_paths=4)
    finally:
        del engine.filter
    assert calls["n"] == 0
    assert snap.observed_net_flows == before
    assert set(engine.__dict__) == attrs
    assert first["masters"].n_paths == first["start_working"].n_paths == 4
    assert np.array_equal(first["masters"].states, second["masters"].states)
    assert np.array_equal(first["masters"].income_chf, second["masters"].income_chf)
    assert np.array_equal(first["masters"].net_flow_chf, second["masters"].net_flow_chf)
    assert not np.array_equal(first["masters"].states, first["start_working"].states)
