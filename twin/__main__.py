"""python -m twin "What if I move to Zurich?" --profile P00001 --paths 500"""
import argparse
import json
import sys

from . import calibration
from .assumptions import ASSUMPTIONS, A
from .peers import bands, compare, find_peers
from .profile import load_profile, profile_from_inputs
from .router import route
from .run import whatif
from .scenarios import working_income
from .trajectory import peer_context

LIMITS = ("Synthetic data and illustrative assumptions. Not financial advice, not a forecast, "
          "not representative of the Swiss population. Results are conditional on the assumptions listed.")


def fmt(x):
    return f"{x:,.0f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--profile", default="P00001")
    ap.add_argument("--paths", type=int, default=500)
    ap.add_argument("--assumptions", action="store_true", help="print all assumptions")
    ap.add_argument("--inputs", help="JSON file with a new user's inputs (instead of --profile)")
    ap.add_argument("--raw", action="store_true",
                    help="switch off the saving calibration (use the ledger's saving as-is)")
    ap.add_argument("--no-peers", action="store_true", help="skip the peer benchmark")
    a = ap.parse_args(argv)

    if a.inputs:
        with open(a.inputs) as f:
            p = profile_from_inputs(json.load(f))
    else:
        p = load_profile(a.profile)
    peer_context(p)
    if not a.raw:
        calibration.apply(p)
    key, params, conf = route(a.question)
    if key is None:
        print("I couldn't tell which decision you mean. Try: master's degree, part-time, self-employed, "
              "having a child, job loss, moving (e.g. 'What if I move to Zurich?'), or 'explore my options'.")
        return 1
    if key == "explore":
        from .explore import frontier, print_front, print_per_decision, print_timing
        print(f"Question : {a.question}\nProfile  : {p['id']} {p['name']}, {p['age']}, {p['employment']}\n")
        res = frontier(p, only=params.get("decision"))
        print_front(res, limit=15)
        print_timing(res)
        print_per_decision(res)
        from .explore import banner
        print("\n" + banner(p, res))
        print(f"\nLimits: {LIMITS}\n")
        from .limits import print_limits
        print_limits()
        return 0
    print(f"Question : {a.question}")
    print(f"Profile  : {p['id']} {p['name']}, {p['age']}, {p['city']} ({p['canton']}), {p['employment']}, {p['household']}")
    print(f"Routed to: {key} {params or ''}  (confidence {conf:.0%})")
    print()
    print("Inputs and provenance:")
    for k, v in p["tags"].items():
        print(f"  - {k}: {v}")
    print(f"  Opening cash CHF {fmt(p['cash'])}, income {fmt(p['income'])}/mo, rent {fmt(p['rent'])}/mo\n")

    res = whatif(p, key, params, a.paths)
    rows = res["rows"]
    for line in res["banner"]:
        print(line)
    print()
    print(f"Results over {p['horizon']} months, {a.paths} simulated futures "
          f"(life regimes, inflation regimes, market returns), regime engine via TwinSession:\n")
    for s in rows:
        print(f"== {s['scenario']}")
        tag = "uncalibrated" if a.raw else "calibrated  "
        print(f"   Net worth at end, {tag} (today's CHF)  P10 {fmt(s['nw_p10'])} | median {fmt(s['nw_p50'])} | P90 {fmt(s['nw_p90'])}")
        if not a.raw:
            u = s["uncalibrated"]
            print(f"   Net worth at end, uncalibrated (today's CHF)  P10 {fmt(u['nw_p10'])} | median {fmt(u['nw_p50'])} | P90 {fmt(u['nw_p90'])}")
        print(f"   Chance of needing financing (cash < 0): {s['p_shortfall']:.0%}"
              f"  (median {s['median_short_months']:.0f} months; bad case peak CHF {fmt(s['p90_peak_shortfall'])})")
        print(f"   Median months below {A('buffer_months')}-month buffer: {s['median_buffer_months']:.0f}")
        for g, pr in s["goals"].items():
            print(f"   Goal '{g}' reached on time in {pr:.0%} of futures")
        lives = ", ".join(f"{k} {v:.0f}" for k, v in s["months_in_state"].items() if v >= 0.5)
        print(f"   Average months per life state: {lives}")
        for n in s["notes"]:
            print(f"   note: {n}")
        print()
    for w in res["warnings"]:
        print(f"WARNING: {w}")
    if res["warnings"]:
        print()
    if p.get("trajectory"):
        pts = {q["year"]: q for q in p["trajectory"]["points"]}
        print(f"Trajectory: median net wealth by year (today's CHF). Peer path = the "
              f"{p['trajectory']['percentile']:.0%} percentile of people at each age in the data.")
        head = "".join(f"{s['scenario'][:22]:>24}" for s in rows)
        print(f"   {'age':>4}{'peer path (low-mid-high)':>32}{head}")
        for y in range(len(rows[0]["years"])):
            q = pts.get(y)
            peer = f"{fmt(q['low'])} / {fmt(q['mid'])} / {fmt(q['high'])}" if q else "-"
            vals = "".join(f"{fmt(s['years'][y]['p50']):>24}" for s in rows)
            print(f"   {p['age'] + y:>4}{peer:>32}{vals}")
        print()
    if len(rows) == 2:
        print(f"Trade-off: median end net worth differs by CHF {fmt(res['median_end_wealth_difference'])} versus "
              f"baseline; financing risk {rows[0]['p_shortfall']:.0%} -> {rows[1]['p_shortfall']:.0%}.")
    if not a.no_peers:
        years = p["horizon"] // 12
        pr = find_peers(p, years=years, ref_income=working_income(p))
        if len(pr["peers"]) >= 5:
            b = bands(pr["peers"])
            ages = sorted(q["age"] for q in pr["peers"])
            print(f"\nPeer benchmark (not a forecast): {len(pr['peers'])} synthetic profiles aged "
                  f"{ages[0]}-{ages[-1]} (target age {pr['center_age']} +/-{pr['window']}), matched on "
                  f"canton, household, income and rent, not on age. Mean match distance {pr['mean_distance']:.2f}.")
            print(f"   {'':<12}{'P25':>10}{'median':>10}{'P75':>10}")
            for key, lbl in (("wealth", "net wealth"), ("cash", "cash"), ("portfolio", "portfolio"),
                             ("goal_total", "goal total")):
                print(f"   {lbl:<12}{fmt(b[key]['p25']):>10}{fmt(b[key]['p50']):>10}{fmt(b[key]['p75']):>10}")
            for s in rows:
                c = compare(s["nw_samples"], b["wealth"])
                print(f"   {s['scenario'][:44]:<44} reaches peer P25 in {c['p25']:.0%}, median in {c['p50']:.0%}, "
                      f"P75 in {c['p75']:.0%} of futures")
            if any(s["nw_p50"] > 2 * b["wealth"]["p75"] for s in rows):
                print("   WARNING: simulated wealth is far above the peer band. Compare scenarios with each "
                      "other; do not read the absolute levels or the peer-band percentages literally"
                      + (" (saving calibration is off: --raw)." if a.raw else "."))
            if len(pr["peers"]) < 15 or pr["window"] > 3:
                print(f"   note: few close peers; age window widened to +/-{pr['window']} years.")
        else:
            print("\nPeer benchmark skipped: too few similar older profiles.")
    if a.assumptions:
        print("\nAssumptions:")
        for k, (v, why) in ASSUMPTIONS.items():
            print(f"  {k} = {v}  -- {why}")
    print(f"\nLimits: {LIMITS}\n")
    from .limits import print_limits
    print_limits()
    return 0


if __name__ == "__main__":
    sys.exit(main())
