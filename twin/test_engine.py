"""Sanity tests: hand-checkable cases the engine must reproduce.

    python3 -m unittest twin.test_engine -v

Deterministic runs switch off wage growth, inflation, returns and noise, so results can be
checked with a calculator. Each test states the arithmetic it expects.
"""
import statistics as st
import unittest

from .assumptions import A, override
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
        try:
            from .chat import unusual  # the chat runs on the regime engine (numpy, pandas, scipy)
        except ImportError as e:
            self.skipTest(f"regime engine unavailable: {e}")
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

    def test_fitted_tables_are_sane(self):
        from .calibration import K_MAX, factors, keep_factor
        old = factors("sim")  # sim.py engine: one factor per age band
        self.assertTrue(all(0.0 <= k <= 1.0 for _, _, k in old))
        new = factors()       # regime engine: age band x income tertile
        self.assertTrue(all(len(c) == 5 and 0.0 <= c[-1] <= K_MAX for c in new))
        for age in (16, 20, 30, 50, 70, 90):
            for income in (1000, 6000, 20000):
                self.assertIsNotNone(keep_factor(age, new, income))
        # every age band has its own cells: 70-year-olds must not fall back to the youngest band
        self.assertEqual(next(c for c in new if c[0] <= 70 < c[1])[0], 65)

    def test_apply_sets_rate_and_tag(self):
        from . import calibration
        p = calibration.apply(person(age=30, income=6000))
        self.assertAlmostEqual(p["calibration_rate"], 1 - calibration.keep_factor(30, income=6000))
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


class Language(unittest.TestCase):
    def test_router_knows_the_new_decisions(self):
        from .router import route
        for q, key in (("What if I work part-time?", "part_time"), ("What if I go freelance?", "self_employed"),
                       ("What if we have a baby?", "family"), ("Explore all my options", "explore")):
            self.assertEqual(route(q)[0], key, q)

    def test_when_questions_go_to_the_search(self):
        from .router import route
        self.assertEqual(route("When should I do a master's?")[:2], ("explore", {"decision": "masters"}))
        self.assertEqual(route("What if I do a master's?")[0], "masters")

    def test_llm_start_month_and_focus_are_validated(self):
        from .agents import validate
        _, sc, params, *_ = validate({"scenario": "masters", "params": {"start_in_months": 24}})
        self.assertEqual((sc, params["start_month"]), ("masters", 24))
        _, sc, params, *_ = validate({"scenario": "explore", "params": {"decision": "masters"}})
        self.assertEqual((sc, params), ("explore", {"decision": "masters"}))
        _, _, params, *_ = validate({"scenario": "explore", "params": {"decision": "rob a bank"}})
        self.assertEqual(params, {})

    def test_llm_output_validation_accepts_new_decisions(self):
        from .agents import validate
        for sc in ("part_time", "self_employed", "family", "explore"):
            self.assertEqual(validate({"inputs": {}, "scenario": sc})[1], sc)


class RegimeBridge(unittest.TestCase):
    """twin/engine.py: the regime engine (backend/regime) plus cash, investments, debt and inflation."""

    @classmethod
    def setUpClass(cls):
        try:
            from . import engine
        except ImportError as e:  # numpy / pandas / scipy (backend/requirements.txt)
            raise unittest.SkipTest(f"regime engine unavailable: {e}")
        cls.engine = engine
        from .profile import load_profile
        cls.p = load_profile("P00001")  # Alexia, a student

    def test_keep_factor_scales_only_surplus_months(self):
        r = self.engine.run(self.p, years=2, n_paths=20)
        full, half = r.flows(1.0), r.flows(0.5)
        surplus = r.net() > 0
        self.assertTrue((full[~surplus] == half[~surplus]).all())   # deficit months untouched
        self.assertTrue((half[surplus] < full[surplus]).all())      # surplus months shrink

    def test_levers_are_kept_in_full(self):
        # a side income chosen on purpose is not calibrated away: +500 x price level every month
        r = self.engine.run(self.p, years=2, n_paths=10)
        gained = r.cash(0.2, side=500.0)[:, -1] - r.cash(0.2)[:, -1]
        self.assertTrue(abs(gained - 500.0 * r.price.sum(axis=1)).max() < 1e-6)

    def test_cash_reconciles(self):
        r = self.engine.run(self.p, years=3, n_paths=10)
        c = r.cash(0.3)
        self.assertTrue(abs(c[:, -1] - (r.opening + r.flows(0.3).sum(axis=1))).max() < 1e-6)

    def test_student_baseline_starts_working(self):
        from regime.types import STATE_INDEX
        r = self.engine.run(self.p, years=1, n_paths=50)
        self.assertGreater((r.states[:, 0] == STATE_INDEX["employed"]).mean(), 0.9)

    def test_decisions_share_futures(self):
        a = self.engine.run(self.p, "baseline", years=2, n_paths=10)
        b = self.engine.run(self.p, "relocate", {"to_canton": "GE"}, years=2, n_paths=10)
        self.assertTrue((a.price == b.price).all())
        self.assertTrue((a.port == b.port).all())

    def test_year_zero_is_todays_net_worth(self):
        r = self.engine.run(self.p, years=1, n_paths=5)
        self.assertAlmostEqual(r.net_worth_years()[0, 0], self.p["cash"] + self.p["portfolio"]
                               - sum(x for x, _ in self.p["loans"]))

    def test_whatif_goes_through_the_session_tools(self):
        from .run import whatif
        res = whatif(dict(self.p, keep_factor=0.3), "masters", {"start_month": 12}, paths=30)
        base, m = res["rows"]
        self.assertEqual(base["months_in_state"]["student"], 0)
        self.assertEqual(m["months_in_state"]["student"], 24)
        self.assertEqual(res["median_end_wealth_difference"], m["nw_p50"] - base["nw_p50"])
        self.assertIn("baseline", res["session"].results)

    def test_search_front_is_non_dominated_and_times_the_decision(self):
        from regime.explore import dominates
        from .explore import frontier
        res = frontier(dict(self.p, keep_factor=0.3), only="masters", n_paths=20, gens=4, pop=12)
        objs = [(-m["wealth_median"], m["months_below_buffer"], m["sacrifice_chf_month"]) for m in res["front"]]
        for o in objs:
            self.assertFalse(any(dominates(b, o) for b in objs))
        self.assertEqual(res["as_is"]["sacrifice_chf_month"], 0)
        self.assertEqual([t["start_month"] for t in res["timing"]["masters"]], res["settings"]["start_months"])
        self.assertEqual(set(res["per_decision"]), {"baseline", "masters"})

    def test_a_child_never_arrives_at_month_zero(self):
        from .explore import frontier
        res = frontier(dict(self.p, keep_factor=0.3), only="family", n_paths=10, gens=2, pop=8)
        self.assertTrue(all(t["start_month"] >= 12 for t in res["timing"]["family"]))


if __name__ == "__main__":
    unittest.main()
