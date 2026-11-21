"""Which traits of a young person move WITH (positive) or AGAINST (negative) the finances of
older people who share their background?

For every profile aged <= YOUNG_MAX, find its k most similar profiles ~`years` older (same
matching as twin.peers: canton, household, income, rent — age is not a matching feature),
take the peer-group median of each outcome, then compute Spearman rank correlations between
the young person's features and those peer medians.

A permutation test (shuffling which young person gets which peer group) gives p-values.
Caveat: income and rent are matching features, so their correlations are partly by design.
Cross-sectional synthetic data: this is association, not a forecast or causal effect.

python3 -m twin.peer_correlations            # prints the correlation tables
"""
import random
import statistics as st
from collections import defaultdict

from .peers import find_peers, population
from .profile import _rows

YOUNG_MAX = 30
HORIZONS = (10, 20, 30)
K = 25
FEATURES = ("income", "rent", "rent_share", "cash", "portfolio", "debt", "wealth",
            "goal_total", "net_flow")
OUTCOMES = ("income", "cash", "portfolio", "debt_share", "wealth", "goal_total", "net_flow")


def peer_stat(peers, o):
    """Median for most outcomes; debt and portfolio are zero for most people, so use the share
    of peers holding any debt and the mean portfolio value instead (a median would be constant)."""
    if o == "debt_share":
        return sum(q["debt"] > 0 for q in peers) / len(peers)
    if o == "portfolio":
        return st.mean(q["portfolio"] for q in peers)
    return st.median(q[o] for q in peers)


def _net_flow():
    """Mean monthly net cash flow (CHF) per profile from the cash-balance ledger."""
    acc = defaultdict(list)
    for r in _rows("synthetic-cash-balances.csv"):
        acc[r["profile_id"]].append(float(r["net_transactions_chf"]))
    return {k: st.mean(v) for k, v in acc.items()}


def enriched_population():
    flow = _net_flow()
    pop = []
    for q in population():
        q = dict(q)
        q["net_flow"] = flow.get(q["id"], 0.0)
        q["rent_share"] = q["rent"] / q["income"]
        pop.append(q)
    return pop


def spearman(x, y):
    return st.correlation(x, y, method="ranked")


def perm_p(x, y, r_obs, rng, n=2000):
    y = list(y)
    hits = 0
    for _ in range(n):
        rng.shuffle(y)
        hits += abs(spearman(x, y)) >= abs(r_obs)
    return (hits + 1) / (n + 1)


def run(seed=7):
    pop = enriched_population()
    young = [p for p in pop if p["age"] <= YOUNG_MAX]
    rng = random.Random(seed)
    out = {"n_young": len(young), "k": K, "young_max_age": YOUNG_MAX, "horizons": {}}
    for h in HORIZONS:
        medians = {o: [] for o in OUTCOMES}
        for p in young:
            peers = find_peers(p, years=h, k=K, pool=pop)["peers"]
            for o in OUTCOMES:
                medians[o].append(peer_stat(peers, o))
        cells = []
        for f in FEATURES:
            x = [p[f] for p in young]
            for o in OUTCOMES:
                r = spearman(x, medians[o])
                cells.append({"feature": f, "outcome": o, "rho": round(r, 3),
                              "p": round(perm_p(x, medians[o], r, rng, n=500), 4),
                              "by_design": f in ("income", "rent", "rent_share") or o == "income"})
        out["horizons"][h] = {"cells": cells}
    return out


if __name__ == "__main__":
    res = run()
    print(f"young profiles (age <= {YOUNG_MAX}): {res['n_young']}, peers per person: {K}")
    for h, d in res["horizons"].items():
        print(f"\n=== peers ~{h} years older ===  (Spearman rho, * = p<0.01, † = partly by design)")
        print(f"{'young feature':<12}" + "".join(f"{o:>11}" for o in OUTCOMES))
        for f in FEATURES:
            row = [c for c in d["cells"] if c["feature"] == f]
            print(f"{f:<12}" + "".join(
                f"{c['rho']:>+9.2f}{'*' if c['p'] < .01 else ' '}{'†' if c['by_design'] else ' '}"
                for c in row))
