"""Step 3: check each decision against similar people who are already in that decision's state.

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
  2. Both groups (decision state and base state) are matched the same way, ~10 years older.

Three checks, on profiles NOT used to fit the saving calibration (held out: odd profile numbers):
  A. Sign:    does the engine's change vs baseline have the same sign as the peer-group difference?
  B. Ranking: do engine and peers order the decisions the same way? (Spearman rho)
  C. Range:   is the engine's median end wealth inside the peer group's middle 50% (P25-P75)?
              ~50% of profiles = well calibrated; near 0% = systematic bias.
Peers are different people who chose their situation for their own reasons: this is a plausibility
check with tolerances, not proof. Code correctness is covered by test_engine.py.

    python3 -m twin.validate_decisions            # ~1 minute
    python3 -m twin.validate_decisions --quick    # fewer profiles and futures
"""
import argparse
import statistics as st

from . import calibration
from .peers import find_peers, population, quantile
from .profile import load_profile
from .scenarios import TEMPLATES, working_income
from .sim import make_shocks, run_path, summarize

YEARS, K, MIN_GROUP = 10, 20, 8
MOVE_TO = {"expensive": "GE", "cheap": "SG"}  # highest and lowest median rent in the synthetic data

# decision key -> (scenario, params, state change, features ignored when matching)
DECISIONS = {
    "part_time": ("part_time", {}, {"employment": "part_time"}, ("income",)),
    "self_employed": ("self_employed", {}, {"employment": "self_employed"}, ("income",)),
    "family": ("family", {}, {"household": "family"}, ("household", "rent")),
    "move_expensive": ("relocate", {"to_canton": MOVE_TO["expensive"]}, {"canton": MOVE_TO["expensive"]},
                       ("canton", "rent")),
    "move_cheap": ("relocate", {"to_canton": MOVE_TO["cheap"]}, {"canton": MOVE_TO["cheap"]},
                   ("canton", "rent")),
}


def peer_group(p, state, ignore, pool):
    """Wealth of the K most similar people ~YEARS older who are in `state` (None if too few)."""
    peers = find_peers(p, years=YEARS, k=K, ref_income=working_income(p), pool=pool, state=state,
                       ignore=ignore)["peers"]
    return [q["wealth"] for q in peers] if len(peers) >= MIN_GROUP else None


def base_state(p, ignore):
    s = {"employment": "employed"}
    for f in ("household", "canton"):
        if f in ignore:
            s[f] = p[f]  # the decision changes this field; the base keeps today's value
    return s


def engine(p, key, params, shocks):
    s = summarize([run_path(p, TEMPLATES[key](p, **params), sh) for sh in shocks], p)
    return s["nw_p50"]


def spearman(x, y):
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return st.correlation(x, y, method="ranked")


def run(max_age=40, paths=60, limit=None):
    pool = population()
    train = [q["id"] for q in pool if int(q["id"][1:]) % 2 == 0]
    table = tuple((b["band"][0], b["band"][1], b["keep"]) for b in calibration.fit(train)["bands"])
    subjects = [q for q in pool if int(q["id"][1:]) % 2 == 1 and q["age"] <= max_age
                and q["employment"] in ("employed", "student")]
    subjects = subjects[:limit] if limit else subjects
    rec = {d: {"eng": [], "peer": []} for d in DECISIONS}
    base_in = {"raw": [], "calibrated": []}
    dec_in = {d: [] for d in DECISIONS}
    per_person_rho = []
    for q in subjects:
        raw = load_profile(q["id"])
        shocks = make_shocks(raw, paths)
        cal = calibration.apply(load_profile(q["id"]), table)
        eng_base = engine(cal, "baseline", {}, shocks)
        # C. range for the baseline, before and after calibration
        g0 = peer_group(raw, base_state(raw, ()), (), pool)
        if g0:
            lo, hi = quantile(g0, .25), quantile(g0, .75)
            base_in["raw"].append(lo <= engine(raw, "baseline", {}, shocks) <= hi)
            base_in["calibrated"].append(lo <= eng_base <= hi)
        xs, ys = [], []
        for d, (key, params, state, ignore) in DECISIONS.items():
            if any(raw[k] == v for k, v in state.items()):
                continue  # already in that state (e.g. already lives in GE)
            gd = peer_group(raw, dict(base_state(raw, ignore), **state), ignore, pool)
            gb = peer_group(raw, base_state(raw, ignore), ignore, pool)
            if not gd or not gb:
                continue
            e = engine(cal, key, params, shocks)
            rec[d]["peer"].append(st.median(gd) - st.median(gb))
            rec[d]["eng"].append(e - eng_base)
            xs.append(e - eng_base)
            ys.append(rec[d]["peer"][-1])
            dec_in[d].append(quantile(gd, .25) <= e <= quantile(gd, .75))
        r = spearman(xs, ys)
        if r is not None:
            per_person_rho.append(r)
    return {"subjects": len(subjects), "rec": rec, "base_in": base_in, "dec_in": dec_in, "rho": per_person_rho}


def sign(x):
    return (x > 0) - (x < 0)


def report(res):
    rec = res["rec"]
    print(f"Held-out profiles checked: {res['subjects']} (aged <= 40, employed or student; calibration fitted "
          f"on the other half)\n")
    print("A. Sign: median change in end wealth after 10 years vs baseline (real CHF)")
    print(f"   {'decision':<16}{'n':>5}{'peers':>11}{'engine':>11}{'agree':>7}{'per-person':>12}")
    agg_e, agg_p = [], []
    for d, r in rec.items():
        if not r["peer"]:
            print(f"   {d:<16}  no comparable peer groups")
            continue
        mp, me = st.median(r["peer"]), st.median(r["eng"])
        pp = sum(sign(a) == sign(b) for a, b in zip(r["eng"], r["peer"])) / len(r["peer"])
        agg_p.append(mp)
        agg_e.append(me)
        print(f"   {d:<16}{len(r['peer']):>5}{mp:>11,.0f}{me:>11,.0f}{'yes' if sign(me) == sign(mp) else 'NO':>7}"
              f"{pp:>12.0%}")
    print("   (masters, job_loss: no matching present state in the data, not checkable)\n")
    print("B. Ranking of decisions, engine vs peers (Spearman rho, 1 = same order)")
    overall, per = spearman(agg_e, agg_p), res["rho"]
    if overall is not None and per:
        print(f"   across decisions {overall:+.2f}   |  per person, median {st.median(per):+.2f} (n={len(per)})")
    else:
        print("   not enough data")
    print("\nC. Range: engine median end wealth inside the peer group's middle 50% (target ~50%)")
    for k, v in res["base_in"].items():
        print(f"   baseline, {k:<11}{sum(v) / len(v):>6.0%}  (n={len(v)})")
    for d, v in res["dec_in"].items():
        if v:
            print(f"   {d:<20}{sum(v) / len(v):>6.0%}  (n={len(v)})")
    print("\nPlausibility check with tolerances, not proof: peers are different people who chose their "
          "situation for their own reasons.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    report(run(paths=30, limit=60) if a.quick else run())
