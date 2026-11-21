"""Peer check of the regime engine: each decision against similar people already in that decision's state.

The data records no decisions, so a decision is mapped to a PRESENT STATE of older people:
    part-time       -> employment = part_time
    self-employed   -> employment = self_employed
    have a child    -> household  = family
    move to canton  -> canton     = <target>
  compared with the BASE state: employed, same household and canton as the person today.
  (Master's degree and job loss have no observable state in the data: not checkable.)

Two rules keep the comparison fair:
  1. Matching ignores whatever the decision changes (part-time: income; child: household and rent;
     move: canton and rent). Otherwise the filter and the match fight each other.
  2. Both groups (decision state and base state) are matched the same way, YEARS older.

Three checks, at 10 and at 20 years, on profiles NOT used to fit the saving calibration
(held out: odd profile numbers; the calibration is fitted at 10 and 20 years on the even ones):
  A. Sign:    does the engine's change vs baseline have the same sign as the peer-group difference?
  B. Ranking: do engine and peers order the decisions the same way? (Spearman rho)
  C. Range:   is the engine's median end wealth inside the peer group's middle 50% (P25-P75)?
              ~50% of profiles = well calibrated; near 0% = systematic bias.
Engine = the regime engine (backend/regime) through twin/engine.py, all decisions of a person on the
same simulated futures. Peers are different people who chose their situation for their own reasons:
this is a plausibility check with tolerances, not proof.

    python3 -m twin.validate_decisions            # ~1 minute
    python3 -m twin.validate_decisions --quick    # fewer profiles and futures
"""
import argparse
import statistics as st

import numpy as np

from . import calibration
from .engine import run as engine_run
from .peers import find_peers, population, quantile
from .profile import load_profile
from .scenarios import working_income

HORIZONS, K, MIN_GROUP = (10, 20), 20, 8
MOVE_TO = {"expensive": "GE", "cheap": "SG"}  # highest and lowest median rent in the synthetic data

# decision key -> (engine decision, params, state change, features ignored when matching)
DECISIONS = {
    "part_time": ("part_time", {}, {"employment": "part_time"}, ("income",)),
    "self_employed": ("self_employed", {}, {"employment": "self_employed"}, ("income",)),
    "family": ("family", {}, {"household": "family"}, ("household", "rent")),
    "move_expensive": ("relocate", {"to_canton": MOVE_TO["expensive"]}, {"canton": MOVE_TO["expensive"]},
                       ("canton", "rent")),
    "move_cheap": ("relocate", {"to_canton": MOVE_TO["cheap"]}, {"canton": MOVE_TO["cheap"]},
                   ("canton", "rent")),
}


def peer_group(p, state, ignore, pool, years):
    """Wealth of the K most similar people ~years older who are in `state` (None if too few)."""
    peers = find_peers(p, years=years, k=K, ref_income=working_income(p), pool=pool, state=state,
                       ignore=ignore)["peers"]
    return [q["wealth"] for q in peers] if len(peers) >= MIN_GROUP else None


def base_state(p, ignore):
    s = {"employment": "employed"}
    for f in ("household", "canton"):
        if f in ignore:
            s[f] = p[f]  # the decision changes this field; the base keeps today's value
    return s


def spearman(x, y):
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return st.correlation(x, y, method="ranked")


def _empty():
    return {"rec": {d: {"eng": [], "peer": []} for d in DECISIONS},
            "base_in": {"raw": [], "calibrated": []}, "dec_in": {d: [] for d in DECISIONS}, "rho": []}


def run(max_age=40, paths=60, limit=None):
    pool = population()
    train = [q["id"] for q in pool if int(q["id"][1:]) % 2 == 0]
    table = calibration.table_of(calibration.fit(train))
    subjects = [q for q in pool if int(q["id"][1:]) % 2 == 1 and q["age"] <= max_age
                and q["employment"] in ("employed", "student")]
    subjects = subjects[:limit] if limit else subjects
    H = max(HORIZONS)
    out = {y: _empty() for y in HORIZONS}
    for q in subjects:
        p = load_profile(q["id"])
        k = calibration.keep_factor(p["age"], table, working_income(p))
        runs = {"baseline": engine_run(p, "baseline", years=H, n_paths=paths)}
        for key, params, *_ in DECISIONS.values():
            name = key + str(params)
            if name not in runs:
                runs[name] = engine_run(p, key, params, years=H, n_paths=paths)
        med = lambda r, y, keep=k: float(np.median(r.wealth(keep, y)))
        for y, res in out.items():
            eng_base = med(runs["baseline"], y)
            g0 = peer_group(p, base_state(p, ()), (), pool, y)
            if g0:  # C. range for the baseline, before and after calibration
                lo, hi = quantile(g0, .25), quantile(g0, .75)
                res["base_in"]["raw"].append(lo <= med(runs["baseline"], y, 1.0) <= hi)
                res["base_in"]["calibrated"].append(lo <= eng_base <= hi)
            xs, ys = [], []
            for d, (key, params, state, ignore) in DECISIONS.items():
                if any(p[f] == v for f, v in state.items()):
                    continue  # already in that state (e.g. already lives in GE)
                gd = peer_group(p, dict(base_state(p, ignore), **state), ignore, pool, y)
                gb = peer_group(p, base_state(p, ignore), ignore, pool, y)
                if not gd or not gb:
                    continue
                e = med(runs[key + str(params)], y)
                res["rec"][d]["peer"].append(st.median(gd) - st.median(gb))
                res["rec"][d]["eng"].append(e - eng_base)
                xs.append(e - eng_base)
                ys.append(res["rec"][d]["peer"][-1])
                res["dec_in"][d].append(quantile(gd, .25) <= e <= quantile(gd, .75))
            r = spearman(xs, ys)
            if r is not None:
                res["rho"].append(r)
    return {"subjects": len(subjects), "table": table, "by_years": out}


def sign(x):
    return (x > 0) - (x < 0)


def _share(v):
    return f"{sum(v) / len(v):>6.0%}  (n={len(v)})" if v else "   n/a"


def report(res):
    print(f"Held-out profiles checked: {res['subjects']} (aged <= 40, employed or student; saving calibration "
          f"fitted at 10 and 20 years on the other half)")
    ks = [c[4] for c in res["table"] if c[5]]
    print(f"Keep factor (held-out fit, {len(ks)} cells): median {st.median(ks):.0%}, "
          f"range {min(ks):.0%}-{max(ks):.0%}")
    for y, r in res["by_years"].items():
        rec = r["rec"]
        print(f"\n=== {y} years ===")
        print(f"A. Sign: median change in end wealth vs baseline (today's CHF)")
        print(f"   {'decision':<16}{'n':>5}{'peers':>11}{'engine':>11}{'agree':>7}{'per-person':>12}")
        agg_e, agg_p = [], []
        for d, v in rec.items():
            if not v["peer"]:
                print(f"   {d:<16}  no comparable peer groups")
                continue
            mp, me = st.median(v["peer"]), st.median(v["eng"])
            pp = sum(sign(a) == sign(b) for a, b in zip(v["eng"], v["peer"])) / len(v["peer"])
            agg_p.append(mp)
            agg_e.append(me)
            print(f"   {d:<16}{len(v['peer']):>5}{mp:>11,.0f}{me:>11,.0f}"
                  f"{'yes' if sign(me) == sign(mp) else 'NO':>7}{pp:>12.0%}")
        overall, per = spearman(agg_e, agg_p), r["rho"]
        print("B. Ranking of decisions, engine vs peers (Spearman rho, 1 = same order)")
        if overall is not None and per:
            print(f"   across decisions {overall:+.2f}   |  per person, median {st.median(per):+.2f} (n={len(per)})")
        else:
            print("   not enough data")
        print("C. Range: engine median end wealth inside the peer group's middle 50% (target ~50%)")
        for k, v in r["base_in"].items():
            print(f"   baseline, {k:<11}{_share(v)}")
        for d, v in r["dec_in"].items():
            if v:
                print(f"   {d:<21}{_share(v)}")
    print("\n(masters, job_loss: no matching present state in the data, not checkable)")
    print("Plausibility check with tolerances, not proof: peers are different people who chose their "
          "situation for their own reasons.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    report(run(paths=30, limit=60) if a.quick else run())
