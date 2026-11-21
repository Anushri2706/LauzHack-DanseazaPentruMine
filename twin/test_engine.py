"""Sanity tests: hand-checkable cases the engine must reproduce.

    python3 -m unittest twin.test_engine -v

Deterministic runs switch off wage growth, inflation, returns and noise, so results can be
checked with a calculator. Each test states the arithmetic it expects.
"""
import statistics as st
import unittest

from .assumptions import A, override
from .chat import unusual
from .profile import canton_rent_index, profile_from_inputs
from .scenarios import TEMPLATES
from .sim import make_shocks, run_path
from .trajectory import peer_context

H = 120


def person(**kw):
    base = {"cash": 2000, "income": 6000, "rent": 2000, "other_spend": 2000, "portfolio": 0,
            "has_debt": False, "age": 30, "employment": "employed", "canton": "VD"}
    return profile_from_inputs({**base, **kw})


def det(p, scenario="baseline", **params):
    """End wealth in a flat world: no wage growth, inflation, returns or noise."""
    flat = ([0.0] * H, [0.0] * H, [0.0] * H)
    with override(wage_growth_annual=0.0):
        return run_path(p, TEMPLATES[scenario](p, **params), flat)


class Accounting(unittest.TestCase):
    def test_saving_accumulates_exactly(self):
        # 25,000 in - 4,000 out = 21,000 a month; 2,000 + 21,000 x 120 = 2,522,000
        r = det(person(income=25000))
        self.assertAlmostEqual(r["final_nw_real"], 2_522_000, places=2)

    def test_deficit_falls_exactly_and_no_loan_is_invented(self):
        # 3,000 in - 3,500 out = -500 a month; 1,000 - 500 x 120 = -59,000 (no interest added)
        r = det(person(cash=1000, income=3000, rent=2000, other_spend=1500))
        self.assertAlmostEqual(r["final_nw_real"], -59_000, places=2)
        self.assertEqual(r["short_months"], 118)  # cash is 500, 0, then negative from month 3

    def test_inflation_lowers_real_wealth(self):
        p = person()
        with override(wage_growth_annual=0.0):
            plan = TEMPLATES["baseline"](p)
            flat = run_path(p, plan, ([0.0] * H, [0.0] * H, [0.0] * H))["final_nw_real"]
            infl = run_path(p, plan, ([0.03] * H, [0.0] * H, [0.0] * H))["final_nw_real"]
        self.assertLess(infl, flat)


class Scenarios(unittest.TestCase):
    def test_masters_payoff_matches_formula(self):
        # Working person, income I = 6,000: study 24 months at 50% (-3,000 x 24 = -72,000),
        # tuition 2 x 8,000 = -16,000, then +15% for 96 months (+900 x 96 = +86,400). Net -1,600.
        p = person()
        diff = det(p, "masters")["final_nw_real"] - det(p)["final_nw_real"]
        self.assertAlmostEqual(diff, -72_000 - 16_000 + 86_400, places=2)

    def test_masters_never_better_without_a_premium(self):
        for emp, inc in (("employed", 6000), ("student", 1500), ("student", 25000)):
            p = person(employment=emp, income=inc)
            with override(degree_premium=0.0):
                self.assertLessEqual(det(p, "masters")["final_nw_real"], det(p)["final_nw_real"] + 1e-6,
                                     f"{emp} earning {inc}")

    def test_student_baseline_never_below_current_income(self):
        p = person(employment="student", income=25000)
        self.assertEqual(TEMPLATES["baseline"](p)["income"][0], 25000)
        q = person(employment="student", income=1200)
        self.assertEqual(TEMPLATES["baseline"](q)["income"][0], A("working_income_student"))

    def test_job_loss_cost(self):
        # 6 months at 70%: loses 30% x 6,000 x 6 = 10,800
        p = person()
        diff = det(p, "job_loss", months=6)["final_nw_real"] - det(p)["final_nw_real"]
        self.assertAlmostEqual(diff, -0.3 * 6000 * 6, places=2)

    def test_relocation_cost(self):
        # extra rent for 120 months + 5,000 moving cost
        p = person()
        ratio = canton_rent_index()["ZH"] / canton_rent_index()["VD"]
        diff = det(p, "relocate", to_canton="ZH")["final_nw_real"] - det(p)["final_nw_real"]
        self.assertAlmostEqual(diff, -(2000 * (ratio - 1) * 120 + 5000), places=2)


class Uncertainty(unittest.TestCase):
    def test_noise_is_in_francs_not_a_share_of_income(self):
        rich, poor = person(income=25000), person(income=2500)
        sd = lambda p: st.pstdev(x for s in make_shocks(p, 50) for x in s[2])
        self.assertAlmostEqual(sd(rich), sd(poor), delta=1.0)
        self.assertAlmostEqual(sd(rich), A("noise_chf"), delta=0.1 * A("noise_chf"))

    def test_monte_carlo_median_matches_flat_world_without_inflation(self):
        p = person(income=8000)
        with override(regime_inflation={"low": 0.0, "base": 0.0, "high": 0.0},
                      portfolio_return=0.0, portfolio_vol=0.0, wage_growth_annual=0.0):
            plan = TEMPLATES["baseline"](p)
            ends = [run_path(p, plan, s)["final_nw_real"] for s in make_shocks(p, 300)]
        self.assertAlmostEqual(st.median(ends) / det(p)["final_nw_real"], 1.0, delta=0.01)


class Guards(unittest.TestCase):
    def test_peer_context_does_not_calibrate(self):
        p = peer_context(person())
        self.assertNotIn("calibration_rate", p)
        self.assertIn("trajectory", p)

    def test_unusual_inputs_are_flagged(self):
        self.assertEqual([f for f, _ in unusual({"income": 25000})], ["income"])
        self.assertEqual([f for f, _ in unusual({"income": 3000, "rent": 3500})], ["rent"])
        self.assertEqual(unusual({"income": 6000, "rent": 2000, "cash": 5000}), [])


def calibrated(keep, **kw):
    p = person(**kw)
    p["calibration_rate"] = 1 - keep
    return p


class SavingCalibration(unittest.TestCase):
    def test_keeps_only_the_calibrated_share_of_surplus(self):
        # surplus 6,000 - 4,000 = 2,000; keep 20% -> 400 a month; 2,000 + 400 x 120 = 50,000
        self.assertAlmostEqual(det(calibrated(0.2))["final_nw_real"], 50_000, places=2)

    def test_deficit_months_are_untouched(self):
        # same as the deficit test above: -59,000 whatever the keep factor
        r = det(calibrated(0.2, cash=1000, income=3000, rent=2000, other_spend=1500))
        self.assertAlmostEqual(r["final_nw_real"], -59_000, places=2)

    def test_deliberate_saving_is_kept_in_full(self):
        # a side income of 500 chosen on purpose is not calibrated away: +500 x 120 = +60,000
        from .explore import apply_levers
        p = calibrated(0.2)
        with override(wage_growth_annual=0.0):
            plan = apply_levers(TEMPLATES["baseline"](p), 0.0, 500.0)
            r = run_path(p, plan, ([0.0] * H, [0.0] * H, [0.0] * H))
        self.assertAlmostEqual(r["final_nw_real"] - det(p)["final_nw_real"], 60_000, places=2)

    def test_fitted_table_is_sane(self):
        from .calibration import factors, keep_factor
        table = factors()
        self.assertTrue(all(0.0 <= k <= 1.0 for _, _, k in table))
        for age in (16, 20, 30, 50, 70, 90):
            self.assertIsNotNone(keep_factor(age))

    def test_apply_sets_rate_and_tag(self):
        from . import calibration
        p = calibration.apply(person(age=30))
        self.assertAlmostEqual(p["calibration_rate"], 1 - calibration.keep_factor(30))
        self.assertIn("saving calibration", p["tags"])


class Decisions(unittest.TestCase):
    def test_part_time_cost(self):
        # 40% of 6,000 less for 120 months = -288,000
        p = person()
        diff = det(p, "part_time")["final_nw_real"] - det(p)["final_nw_real"]
        self.assertAlmostEqual(diff, -0.4 * 6000 * 120, places=2)

    def test_self_employed_formula(self):
        # 12 months at 50% (-3,000 x 12 = -36,000), then +10% for 108 months (+600 x 108 = +64,800),
        # setup -10,000. Net +18,800.
        p = person()
        diff = det(p, "self_employed")["final_nw_real"] - det(p)["final_nw_real"]
        self.assertAlmostEqual(diff, -36_000 + 64_800 - 10_000, places=2)
        self.assertEqual(TEMPLATES["self_employed"](p)["noise_scale"], A("self_employed_noise"))

    def test_child_formula(self):
        # from month 12: rent +25% (+500) and +1,300 child costs for 108 months = -194,400;
        # 80% income for 24 months: -1,200 x 24 = -28,800. Net -223,200.
        p = person()
        diff = det(p, "family")["final_nw_real"] - det(p)["final_nw_real"]
        self.assertAlmostEqual(diff, -194_400 - 28_800, places=2)


class Explore(unittest.TestCase):
    def test_front_is_non_dominated_and_complete(self):
        from .explore import frontier, dominates
        p = calibrated(0.2)
        front, ref, per = frontier(p, pop=16, gens=5, paths=10)
        objs = [(-m["wealth_median"], m["months_below_buffer"], m["sacrifice_chf_month"]) for m in front]
        for a in objs:
            self.assertFalse(any(dominates(b, a) for b in objs))
        self.assertEqual(ref["sacrifice_chf_month"], 0)
        self.assertIn("baseline", per)
        self.assertTrue(all(r["no_sacrifice"]["sacrifice_chf_month"] == 0 for r in per.values()))


class Language(unittest.TestCase):
    def test_router_knows_the_new_decisions(self):
        from .router import route
        for q, key in (("What if I work part-time?", "part_time"), ("What if I go freelance?", "self_employed"),
                       ("What if we have a baby?", "family"), ("Explore all my options", "explore")):
            self.assertEqual(route(q)[0], key, q)

    def test_llm_output_validation_accepts_new_decisions(self):
        from .agents import validate
        for sc in ("part_time", "self_employed", "family", "explore"):
            self.assertEqual(validate({"inputs": {}, "scenario": sc})[1], sc)


if __name__ == "__main__":
    unittest.main()
