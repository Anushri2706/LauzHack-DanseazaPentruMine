"""Trade-off search: TwinSession.explore (NSGA-II in backend/regime/explore.py) for a twin profile.

A plan = one decision + when it starts + two levers:
    decision       baseline | masters | part_time | self_employed | family | move to the cheapest canton
    start month    0, 6, ..., 36 (for decisions whose timing matters: "when should I do the master's?")
    expense cut    0-20% less spending every month
    side income    CHF 0-800/month
Three objectives on the SAME simulated futures: median real net worth at the horizon (up), median months
with cash below the 3-month buffer (down), CHF/month given up on purpose (down). The levers are
deliberate, so the saving calibration does not shrink them. NSGA-II returns the Pareto front; it does not
pick a winner.

    python3 -m twin.explore --profile P00001                 # all decisions, calibrated
    python3 -m twin.explore --profile P00001 --only masters  # when should I do the master's?
    python3 -m twin.explore --profile P00001 --raw           # no saving calibration
"""
import argparse

from .engine import explore_options, session_for


def frontier(p, only=None, max_start_month=36, n_paths=100, gens=30, pop=40):
    """TwinSession.explore for profile p. only: restrict to baseline + one decision kind."""
    import numpy as np
    s = session_for(p, n_paths=n_paths)
    res = s.explore(explore_options(p, only), max_start_month=max_start_month, n_paths=n_paths,
                    generations=gens, population=pop)
    s.simulate("baseline")
    raw = s.results["baseline"].wealth.wealth(1.0)  # same futures, keep factor 100%
    res["as_is_uncalibrated_median"] = round(float(np.median(raw)))
    return res


def banner(p, res):
    """Code-written statement of whether the search is calibrated (the front is scored calibrated)."""
    cal, k = p.get("calibration") or {}, p.get("keep_factor")
    years = res["settings"]["horizon_months"] // 12
    if k is None or not cal.get("fitted", True):
        why = "no keep factor fits your group" if k is not None else "calibration is off"
        return (f"UNCALIBRATED search ({why}): plans are scored on the engine's saving as-is. As-is median net worth "
                f"after {years} years: CHF {res['as_is']['wealth_median']:,.0f}.")
    return (f"CALIBRATED search: every plan is scored with the engine's monthly surplus scaled to {k:.0%} "
            f"(a group average for {cal.get('group', 'your group')}, not your own path); expense cuts and side "
            f"income are kept in full. As-is median net worth after {years} years: CHF "
            f"{res['as_is']['wealth_median']:,.0f} calibrated / CHF {res['as_is_uncalibrated_median']:,.0f} uncalibrated.")


def highlights(res):
    """A few representative plans from the front: most wealth, least risk, least sacrifice."""
    front = res["front"]
    return {"most_wealth": max(front, key=lambda m: m["wealth_median"]),
            "least_risk": min(front, key=lambda m: (m["months_below_buffer"], -m["wealth_median"])),
            "least_sacrifice": min(front, key=lambda m: (m["sacrifice_chf_month"], -m["wealth_median"]))}


def _when(m):
    return "-" if m["start_month"] is None else f"m{m['start_month']}"


def print_timing(res):
    if not res["timing"]:
        return
    print("\nWhen to start (no levers), median real net worth at the horizon by start month:")
    for d, rows in res["timing"].items():
        print(f"  {d:<20}" + "  ".join(f"m{r['start_month']}: {r['wealth_median']:,.0f}" for r in rows))


def print_per_decision(res):
    print("\nEach decision (best start, no sacrifice | best found with sacrifice), median real wealth:")
    print(f"{'decision':<20}{'start':>6}{'no sacrifice':>14}{'mo<buffer':>11}{'cash<0':>8}{'best':>11}"
          f"{'at CHF/mo':>11}{'on front':>10}")
    for d, r in res["per_decision"].items():
        a, b = r["no_sacrifice_best_start"], r["most_wealth"]
        print(f"{d:<20}{_when(a):>6}{a['wealth_median']:>14,.0f}{a['months_below_buffer']:>11.0f}"
              f"{a['chance_cash_negative']:>8.0%}{b['wealth_median']:>11,.0f}{b['sacrifice_chf_month']:>11,.0f}"
              f"{r['on_overall_front']:>10}")


def print_front(res, limit=25):
    front, ref = res["front"], res["as_is"]
    print(f"Pareto front: {len(front)} plans where no goal improves without another getting worse "
          f"({res['settings']['simulations']} simulations, {res['settings']['plans_scored']} plans scored).\n")
    print(f"{'decision':<22}{'start':>6}{'cut':>5}{'side/mo':>9}{'sacrifice':>11}{'wealth P10':>12}{'median':>10}"
          f"{'P90':>10}{'mo<buffer':>11}{'cash<0':>8}")
    rows = front if len(front) <= limit else [front[round(i * (len(front) - 1) / (limit - 1))]
                                               for i in range(limit)]
    for m in [ref] + rows:
        tag = " (as-is)" if m is ref else ""
        print(f"{(m['decision'] + tag)[:21]:<22}{_when(m):>6}{m['expense_cut']:>5.0%}"
              f"{m['side_income_chf_month']:>9,.0f}{m['sacrifice_chf_month']:>11,.0f}{m['wealth_p10']:>12,.0f}"
              f"{m['wealth_median']:>10,.0f}{m['wealth_p90']:>10,.0f}{m['months_below_buffer']:>11.0f}"
              f"{m['chance_cash_negative']:>8.0%}")
    if len(front) > limit:
        print(f"... {limit} of {len(front)} plans shown, spread evenly by wealth")


if __name__ == "__main__":
    from . import calibration
    from .profile import load_profile
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="P00001")
    ap.add_argument("--only", help="one decision kind, e.g. masters (search its timing and levers)")
    ap.add_argument("--raw", action="store_true", help="no saving calibration")
    a = ap.parse_args()
    prof = load_profile(a.profile)
    if not a.raw:
        calibration.apply(prof)
    print(f"{prof['id']} {prof['name']}, {prof['age']}, {prof['employment']}, {prof['canton']}\n")
    r = frontier(prof, only=a.only)
    print(banner(prof, r) + "\n")
    print_front(r, limit=15)
    print_timing(r)
    print_per_decision(r)
    from .limits import FOOTER
    print("\n" + FOOTER.replace("Type 'limits'", "Run `python3 -m twin.limits`"))
