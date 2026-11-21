"""Step 1: calibrate saving so a simulated decade matches how wealth differs between age groups.

The problem: the 24-month ledger says a typical person saves ~CHF 3,000/month (~CHF 360k per decade),
but in the snapshot, similar people 10 years older hold only ~CHF 5-65k more. Both cannot be true.

The fix (one documented number per age band):
  keep factor k = share of the ledger's monthly surplus a person actually keeps.
  The rest (1 - k) is spending the ledger does not show. Deficit months are untouched.

How k is fitted, per age band:
  for every synthetic profile in the band
    target  = median net wealth of its 25 most similar profiles ~10 years older
              (matched on canton, household, income, rent; age is NOT a matching feature)
    sim(k)  = simulated net wealth after 10 years, in an average future (base inflation,
              mean investment return, no noise), with keep factor k
  choose k so that the MEDIAN over the band of sim(k) - target is zero (bisection on 0..1).

The month-to-month pattern of the ledger (size of the swings) is kept unchanged: only the average
level of saving is scaled. Profiles 55+ are matched to peers who are mostly retired while the engine
has no retirement yet, so their factor also absorbs that gap.

    python3 -m twin.calibration          # fit on all profiles, write twin/calibration.json, print report
"""
import json
import statistics as st
from functools import lru_cache
from pathlib import Path

from .assumptions import A
from .peers import find_peers
from .profile import _rows, load_profile

BANDS = [(15, 25), (25, 35), (35, 45), (45, 55), (55, 65), (65, 200)]
YEARS, K_PEERS = 10, 25
FILE = Path(__file__).with_name("calibration.json")


def band_of(age):
    return next(i for i, (lo, hi) in enumerate(BANDS) if lo <= age < hi) if age >= BANDS[0][0] else 0


def _average_future(p):
    H = p["horizon"]
    return ([A("regime_inflation")["base"]] * H, [A("portfolio_return") / 12] * H, [0.0] * H)


def sim_wealth(p, keep):
    """Real net worth after the horizon, baseline plan, average future, keep factor `keep`."""
    from .scenarios import TEMPLATES
    from .sim import run_path
    q = dict(p, calibration_rate=1.0 - keep)
    return run_path(q, TEMPLATES["baseline"](q), _average_future(q))["final_nw_real"]


def peer_target(p):
    from .scenarios import working_income
    res = find_peers(p, years=YEARS, k=K_PEERS, ref_income=working_income(p))
    return st.median(q["wealth"] for q in res["peers"]) if res["peers"] else None


def fit(ids=None):
    """Fit one keep factor per age band. ids: restrict to these profile ids (for held-out checks)."""
    pids = [r["profile_id"] for r in _rows("synthetic-profiles.csv")]
    if ids is not None:
        ids = set(ids)
        pids = [i for i in pids if i in ids]
    rows = {b: [] for b in range(len(BANDS))}
    for pid in pids:
        p = load_profile(pid)
        t = peer_target(p)
        if t is not None:
            rows[band_of(p["age"])].append((p, t))
    out = []
    for b, items in rows.items():
        lo_age, hi_age = BANDS[b]
        if not items:
            out.append({"band": [lo_age, hi_age], "keep": 1.0, "n": 0})
            continue
        gap = lambda k: st.median(sim_wealth(p, k) - t for p, t in items)
        lo, hi = 0.0, 1.0
        if gap(lo) >= 0:
            k = lo
        elif gap(hi) <= 0:
            k = hi
        else:
            for _ in range(25):  # wealth rises with k: bisection
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if gap(mid) < 0 else (lo, mid)
            k = (lo + hi) / 2
        out.append({"band": [lo_age, hi_age], "keep": round(k, 4), "n": len(items),
                    "median_target_chf": round(st.median(t for _, t in items)),
                    "median_sim_raw_chf": round(st.median(sim_wealth(p, 1.0) for p, _ in items)),
                    "median_sim_calibrated_chf": round(st.median(sim_wealth(p, k) for p, _ in items)),
                    "median_ledger_surplus_chf_month": round(st.median(
                        p["income"] - p["rent"] - p["other_spend"] for p, _ in items))})
    return {"method": "keep factor per age band; median simulated 10-year wealth = median wealth of "
                      "background-matched peers 10 years older", "bands": out}


@lru_cache(maxsize=1)
def factors():
    """Fitted bands from calibration.json (fits and writes it on first use if missing)."""
    if not FILE.exists():
        FILE.write_text(json.dumps(fit(), indent=1))
    return tuple((b["band"][0], b["band"][1], b["keep"]) for b in json.loads(FILE.read_text())["bands"])


def _band(age, table=None):
    """(lo, hi, keep) of the band containing `age`; the youngest band for anyone younger."""
    table = table or factors()
    return next((b for b in table if b[0] <= age < b[1]), table[0])


def keep_factor(age, table=None):
    return _band(age, table)[2]


def apply(p, table=None):
    """Calibrate profile p in place (sets calibration_rate) and tag it. Returns p."""
    lo, hi, k = _band(p["age"], table)
    p["calibration_rate"] = 1.0 - k
    p["keep_factor"] = k
    ages = f"{lo}-{hi - 1}" if hi < 200 else f"{lo}+"
    p["tags"]["saving calibration"] = (f"you keep {k:.0%} of any monthly surplus (fitted for ages {ages} "
                                       f"so a simulated decade matches similar people 10 years older)")
    return p


if __name__ == "__main__":
    res = fit()
    FILE.write_text(json.dumps(res, indent=1))
    print("Saving calibration (median over each age band, real CHF after 10 years)\n")
    print(f"{'ages':<8}{'n':>5}{'keep':>7}{'surplus/mo':>12}{'peer target':>13}{'raw sim':>11}{'calibrated':>12}")
    for b in res["bands"]:
        lo, hi = b["band"]
        if not b["n"]:
            continue
        print(f"{f'{lo}-{hi - 1}' if hi < 200 else f'{lo}+':<8}{b['n']:>5}{b['keep']:>7.0%}"
              f"{b['median_ledger_surplus_chf_month']:>12,}{b['median_target_chf']:>13,}"
              f"{b['median_sim_raw_chf']:>11,}{b['median_sim_calibrated_chf']:>12,}")
    print(f"\nwrote {FILE}")
