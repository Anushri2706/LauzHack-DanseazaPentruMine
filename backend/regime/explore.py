"""Trade-off search (NSGA-II) over plans, scored by TwinSession with the wealth layer.

A plan = one decision + when it starts + two levers:
    decision      an option {"kind", "params", "label", "min_start"} (TwinSession decision kinds;
                  "baseline" = none; min_start = earliest allowed start month, e.g. 12 for a child)
    start_month   0, start_step, ..., max_start_month (only for decisions whose timing matters)
    expense_cut   0-20% less spending every month          (deliberate: kept in full, not calibrated)
    side_income   CHF 0-800/month                          (deliberate)
Objectives, all on the SAME simulated futures (same seed for every plan):
    wealth     median real net worth at the horizon                (higher is better)
    risk       median months with cash below the buffer            (lower is better)
    sacrifice  CHF/month given up on purpose (cut + side income)    (lower is better)
NSGA-II returns the Pareto front: plans where no objective improves without another getting worse. It
does not pick a winner. Each (decision, start month) is simulated once and cached; the levers are pure
accounting on the cached paths, so the search costs a few dozen simulations.

"When should I do X?" is answered by the start-month gene: ``timing`` lists, per decision, the result of
each start month with no levers, and the front shows which start months survive.
"""
from __future__ import annotations

import random

import numpy as np

CUT_MAX, SIDE_MAX = 0.20, 800.0
TIMED = {"masters", "part_time", "relocate", "family", "job_loss", "move_abroad", "start_working"}
DEFAULT_OPTIONS = [{"kind": "baseline"}, {"kind": "masters"}, {"kind": "part_time"}, {"kind": "self_employed"}]


def _label(opt):
    return opt.get("label") or opt["kind"]


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


class _Scorer:
    def __init__(self, session, options, starts, n_paths):
        self.s, self.options, self.starts, self.n = session, options, starts, n_paths
        self.sims, self.cache = {}, {}

    def starts_for(self, i):
        opt = self.options[i]
        if opt["kind"] not in TIMED:
            return [0]
        lo = int(opt.get("min_start", 0))
        return [m for m in self.starts if m >= lo] or [lo]

    def wealth(self, i, start):
        from .twin_session import Scenario
        opt = self.options[i]
        ok = self.starts_for(i)
        start = min(ok, key=lambda m: abs(m - start))  # nearest allowed start month
        key = (i, start)
        if key not in self.sims:
            base = self.s.scenarios["baseline"]
            sc = Scenario(f"explore:{_label(opt)}@{start}", _label(opt), list(base.interventions),
                          list(base.one_offs), dict(base.assumptions), list(base.decisions),
                          dict(base.profile_changes))
            if opt["kind"] != "baseline":
                ivs, offs = self.s._decision_to_interventions(opt["kind"], start, dict(opt.get("params") or {}))
                sc.interventions += ivs
                sc.one_offs += offs
                if opt["kind"] == "self_employed":
                    sc.profile_changes["self_employed"] = True
            self.sims[key] = self.s._run(sc, self.n, self.s.seed).wealth
        return self.sims[key], start

    def score(self, i, start, cut, side):
        w, start = self.wealth(i, start)
        cut, side = round(cut, 3), round(side, -1)
        key = (i, start, cut, side)
        if key not in self.cache:
            nw = w.wealth(None, None, cut, side)
            cash = w.cash(None, cut, side)
            buf = float(np.median(w.months_below_buffer(None, cut, side)))
            sac = round(w.sacrifice(cut, side))  # scored as displayed, so the shown front is consistent
            opt = self.options[i]
            plan = {"decision": _label(opt), "kind": opt["kind"],
                    "start_month": start if opt["kind"] in TIMED else None,
                    "expense_cut": cut, "side_income_chf_month": side, "sacrifice_chf_month": sac,
                    "wealth_p10": round(float(np.quantile(nw, .1))), "wealth_median": round(float(np.median(nw))),
                    "wealth_p90": round(float(np.quantile(nw, .9))), "months_below_buffer": buf,
                    "chance_cash_negative": round(float((cash < 0).any(axis=1).mean()), 3)}
            self.cache[key] = ((-plan["wealth_median"], buf, sac), plan)
        return self.cache[key]


def frontier(session, options=None, max_start_month=36, start_step=6, n_paths=100, gens=30, pop=40, seed=11):
    """Run NSGA-II for a TwinSession (wealth=True). Returns a JSON-ready dict (see module docstring)."""
    options = [dict(o) for o in (options or DEFAULT_OPTIONS)]
    horizon = session.horizon_months
    starts = [m for m in range(0, max(0, int(max_start_month)) + 1, max(1, int(start_step))) if m < horizon]
    sc = _Scorer(session, options, starts, n_paths)
    rng = random.Random(seed)
    bounds = [(0, len(options) - 1e-9), (0, len(starts) - 1e-9), (0.0, CUT_MAX), (0.0, SIDE_MAX)]

    def ev(g):
        return sc.score(int(g[0]), starts[int(g[1])], g[2], g[3])

    def child(a, b):
        c = [a[i] if rng.random() < 0.5 else b[i] for i in range(4)]
        for i, (lo, hi) in enumerate(bounds):
            if rng.random() < 0.3:
                c[i] += rng.gauss(0, (hi - lo) * 0.15)
            c[i] = min(hi, max(lo, c[i]))
        return c

    # every decision at every start month, no levers, so each is represented and timing is complete
    seeds = [[i + 0.5, starts.index(m) + 0.5 if m in starts else 0.5, 0.0, 0.0] for i in range(len(options))
             for m in sc.starts_for(i)]
    for g in seeds:
        ev(g)
    popn = seeds[:pop] + [[rng.uniform(lo, hi) for lo, hi in bounds] for _ in range(max(0, pop - len(seeds)))]
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
    for i, o in enumerate(options):  # a small lever grid per decision at its best start, for comparison
        best_start = max((sc.score(i, m, 0, 0)[1] for m in sc.starts_for(i)),
                         key=lambda p: p["wealth_median"])["start_month"] or 0
        for cut in (0.0, 0.1, 0.2):
            for side in (0.0, 400.0, 800.0):
                sc.score(i, best_start, cut, side)
    items = list(sc.cache.values())
    objs = [o for o, _ in items]
    seen, front = set(), []
    for i in fast_sort(objs)[0]:
        m = items[i][1]
        key = (m["decision"], m["start_month"], round(m["expense_cut"], 2), round(m["side_income_chf_month"], -1))
        if key not in seen:
            seen.add(key)
            front.append(m)
    front.sort(key=lambda m: -m["wealth_median"])
    base_i = next((i for i, o in enumerate(options) if o["kind"] == "baseline"), 0)
    reference = sc.score(base_i, 0, 0.0, 0.0)[1]
    per_decision, timing = {}, {}
    for i, o in enumerate(options):
        own = [m for _, m in items if m["decision"] == _label(o)]
        free = [m for m in own if m["sacrifice_chf_month"] == 0]
        per_decision[_label(o)] = {
            "no_sacrifice_best_start": max(free, key=lambda m: m["wealth_median"]),
            "most_wealth": max(own, key=lambda m: m["wealth_median"]),
            "on_overall_front": sum(m["decision"] == _label(o) for m in front)}
        if o["kind"] in TIMED:
            timing[_label(o)] = [{k: m[k] for k in ("start_month", "wealth_p10", "wealth_median", "wealth_p90",
                                                    "months_below_buffer", "chance_cash_negative")}
                                 for m in sorted(free, key=lambda m: m["start_month"])]
    return {"front": front, "as_is": reference, "per_decision": per_decision, "timing": timing,
            "settings": {"horizon_months": horizon, "start_months": starts,
                         "min_start": {_label(o): o["min_start"] for o in options if "min_start" in o}, "n_paths": n_paths,
                         "generations": gens, "population": pop, "simulations": len(sc.sims),
                         "plans_scored": len(sc.cache), "keep_factor": session.keep_factor},
            "note": "Simulated under stated assumptions; the front shows trade-offs, not a recommendation."}
