"""Step 4: explore trade-offs with NSGA-II on top of the twin engine.

A plan = one decision + two levers:
    decision       baseline | masters | part_time | self_employed | relocate (cheapest-rent canton)
    expense_cut    0-20% less non-loan spending every month
    extra_income   CHF 0-800/month from a side job
Three objectives, all evaluated on the SAME simulated futures (common random numbers):
    wealth     median real net worth after 10 years            (higher is better)
    risk       median months with cash below the 3-month buffer (lower is better)
    sacrifice  CHF/month given up: expense cut + extra work     (lower is better)
NSGA-II returns the Pareto front: plans where no objective can improve without another getting worse.
It does not pick a winner; the person chooses what they care about.

    python3 -m twin.explore --profile P00001          # calibrated (default)
    python3 -m twin.explore --profile P00001 --raw    # ledger saving as-is
"""
import argparse
import random

from .profile import canton_rent_index
from .scenarios import TEMPLATES
from .sim import make_shocks, run_path, summarize

CUT_MAX, EXTRA_MAX = 0.20, 800.0


def default_decisions(p):
    idx = canton_rent_index()
    cheapest = min((c for c in idx if c != p["canton"]), key=idx.get)
    out = [("baseline", {}), ("masters", {}), ("part_time", {}), ("self_employed", {})]
    if idx[cheapest] < idx[p["canton"]]:
        out.append(("relocate", {"to_canton": cheapest}))
    return out


def label(dec):
    key, params = dec
    return key if not params else f"{key} to {params['to_canton']}"


def apply_levers(plan, cut, extra):
    """Levers are deliberate: the CHF cut or earned on purpose is kept in full (sim.py 'deliberate'),
    not shrunk by the saving calibration, which models spending the ledger does not show."""
    plan = dict(plan)
    plan["deliberate"] = [cut * s + extra for s in plan["spend"]]
    return plan


def evaluate(p, dec, cut, extra, shocks, cache):
    cut, extra = round(cut, 3), round(extra, 0)  # evaluate at the precision used for caching
    k = (label(dec), cut, extra)
    if k in cache:
        return cache[k]
    key, params = dec
    base = TEMPLATES[key](p, **params)
    plan = apply_levers(base, cut, extra)
    s = summarize([run_path(p, plan, sh) for sh in shocks], p)
    sacrifice = cut * base["spend"][0] + extra
    m = {"decision": label(dec), "scenario_name": base["name"], "expense_cut": cut, "extra_income": extra,
         "wealth_p10": s["nw_p10"], "wealth_median": s["nw_p50"], "wealth_p90": s["nw_p90"],
         "months_below_buffer": s["median_buffer_months"], "chance_cash_negative": s["p_shortfall"],
         "sacrifice_chf_month": sacrifice}
    cache[k] = ((-s["nw_p50"], s["median_buffer_months"], sacrifice), m)
    return cache[k]


def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def fast_sort(objs):
    n = len(objs)
    S, cnt, fronts = [[] for _ in range(n)], [0] * n, [[]]
    for i in range(n):
        for j in range(n):
            if dominates(objs[i], objs[j]):
                S[i].append(j)
            elif dominates(objs[j], objs[i]):
                cnt[i] += 1
        if cnt[i] == 0:
            fronts[0].append(i)
    f = 0
    while fronts[f]:
        nxt = []
        for i in fronts[f]:
            for j in S[i]:
                cnt[j] -= 1
                if cnt[j] == 0:
                    nxt.append(j)
        f += 1
        fronts.append(nxt)
    return fronts[:-1]


def crowding(front, objs):
    d = {i: 0.0 for i in front}
    for k in range(len(objs[0])):
        order = sorted(front, key=lambda i: objs[i][k])
        d[order[0]] = d[order[-1]] = float("inf")
        span = (objs[order[-1]][k] - objs[order[0]][k]) or 1.0
        for j in range(1, len(order) - 1):
            d[order[j]] += (objs[order[j + 1]][k] - objs[order[j - 1]][k]) / span
    return d


def frontier(p, decisions=None, pop=48, gens=40, paths=60, seed=11):
    """Run NSGA-II. Returns (front: list of plan dicts sorted by wealth, reference: as-is baseline)."""
    rng = random.Random(seed)
    decisions = decisions or default_decisions(p)
    shocks = make_shocks(p, paths)
    cache = {}
    bounds = [(0, len(decisions) - 1e-9), (0.0, CUT_MAX), (0.0, EXTRA_MAX)]

    def ev(g):
        return evaluate(p, decisions[int(g[0])], g[1], g[2], shocks, cache)

    def child(a, b):
        c = [a[i] if rng.random() < 0.5 else b[i] for i in range(3)]
        for i, (lo, hi) in enumerate(bounds):
            if rng.random() < 0.3:
                c[i] += rng.gauss(0, (hi - lo) * 0.15)
            c[i] = min(hi, max(lo, c[i]))
        return c

    # seed every decision with no levers so each one is represented from the start
    popn = [[i + 0.5, 0.0, 0.0] for i in range(len(decisions))]
    popn += [[rng.uniform(lo, hi) for lo, hi in bounds] for _ in range(pop - len(popn))]
    for _ in range(gens):
        allp = popn + [child(rng.choice(popn), rng.choice(popn)) for _ in range(pop)]
        objs = [ev(g)[0] for g in allp]
        nxt = []
        for f in fast_sort(objs):
            if len(nxt) + len(f) <= pop:
                nxt += f
            else:
                cd = crowding(f, objs)
                nxt += sorted(f, key=lambda i: -cd[i])[:pop - len(nxt)]
                break
        popn = [allp[i] for i in nxt]
    for dec in decisions:  # a small grid per decision, so every decision can be compared
        for cut in (0.0, 0.1, 0.2):
            for extra in (0.0, 400.0, 800.0):
                evaluate(p, dec, cut, extra, shocks, cache)
    # final front over everything evaluated, de-duplicated at display precision
    items = list(cache.values())
    objs = [o for o, _ in items]
    seen, front = set(), []
    for i in fast_sort(objs)[0]:
        m = items[i][1]
        key = (m["decision"], round(m["expense_cut"], 2), round(m["extra_income"], -1))
        if key not in seen:
            seen.add(key)
            front.append(m)
    front.sort(key=lambda m: -m["wealth_median"])
    reference = evaluate(p, ("baseline", {}), 0.0, 0.0, shocks, cache)[1]
    per_decision = {}
    for dec in decisions:
        own = [m for _, m in items if m["decision"] == label(dec)]
        per_decision[label(dec)] = {
            "no_sacrifice": next(m for m in own if m["sacrifice_chf_month"] == 0),
            "most_wealth": max(own, key=lambda m: m["wealth_median"]),
            "on_overall_front": sum(m["decision"] == label(dec) for m in front)}
    return front, reference, per_decision


def highlights(front):
    """A few representative plans from the front (for the explanation): most wealth, least risk,
    least sacrifice, and the best plan of each decision with no sacrifice."""
    out = {"most_wealth": max(front, key=lambda m: m["wealth_median"]),
           "least_risk": min(front, key=lambda m: (m["months_below_buffer"], -m["wealth_median"])),
           "least_sacrifice": min(front, key=lambda m: (m["sacrifice_chf_month"], -m["wealth_median"]))}
    for m in front:
        if m["sacrifice_chf_month"] < 1:
            key = f"no_sacrifice_{m['decision']}"
            if key not in out or m["wealth_median"] > out[key]["wealth_median"]:
                out[key] = m
    return out


def print_per_decision(per):
    print("\nEach decision (no sacrifice | best found with sacrifice), median real wealth after 10 years:")
    print(f"{'decision':<22}{'no sacrifice':>14}{'mo<buffer':>11}{'cash<0':>8}{'best':>11}{'at CHF/mo':>11}"
          f"{'on front':>10}")
    for d, r in per.items():
        a, b = r["no_sacrifice"], r["most_wealth"]
        print(f"{d:<22}{a['wealth_median']:>14,.0f}{a['months_below_buffer']:>11.0f}{a['chance_cash_negative']:>8.0%}"
              f"{b['wealth_median']:>11,.0f}{b['sacrifice_chf_month']:>11,.0f}{r['on_overall_front']:>10}")


def print_front(front, reference, limit=25):
    print(f"Pareto front: {len(front)} plans where no goal improves without another getting worse.\n")
    print(f"{'decision':<26}{'cut':>5}{'extra/mo':>10}{'sacrifice':>11}{'wealth P10':>12}{'median':>10}"
          f"{'P90':>10}{'mo<buffer':>11}{'cash<0':>8}")
    rows = front if len(front) <= limit else [front[round(i * (len(front) - 1) / (limit - 1))]
                                               for i in range(limit)]
    for m in [reference] + rows:
        tag = " (as-is)" if m is reference else ""
        print(f"{(m['decision'] + tag)[:25]:<26}{m['expense_cut']:>5.0%}{m['extra_income']:>10,.0f}"
              f"{m['sacrifice_chf_month']:>11,.0f}{m['wealth_p10']:>12,.0f}{m['wealth_median']:>10,.0f}"
              f"{m['wealth_p90']:>10,.0f}{m['months_below_buffer']:>11.0f}{m['chance_cash_negative']:>8.0%}")
    if len(front) > limit:
        print(f"... {limit} of {len(front)} plans shown, spread evenly by wealth")


if __name__ == "__main__":
    from . import calibration
    from .profile import load_profile
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="P00001")
    ap.add_argument("--raw", action="store_true", help="no saving calibration")
    a = ap.parse_args()
    prof = load_profile(a.profile)
    if not a.raw:
        calibration.apply(prof)
    calib = "" if a.raw else f" · keeps {prof['keep_factor']:.0%} of surplus (calibrated)"
    print(f"{prof['id']} {prof['name']}, {prof['age']}, {prof['employment']}, {prof['canton']}{calib}\n")
    f, ref, per = frontier(prof)
    print_front(f, ref, limit=15)
    print_per_decision(per)
