"""Saving calibration: check (and if needed fix) the regime engine against older people in the data.

The question: does a simulated decade match how wealth differs between age groups in the snapshot?
If the engine's saving is right, the answer needs no fix and the keep factor is 100%.

  keep factor k = share of the engine's monthly surplus a person actually keeps.
  k < 1: spending the engine does not show.  k > 1: saving it does not show.  Deficit months untouched.

How k is fitted, per cell (age band x income tertile, 18 cells):
  for every synthetic profile in the cell and h = 10 and 20 years
    target_h  = median net wealth of its 25 most similar profiles ~h years older
                (matched on canton, household, income, rent; age is NOT a matching feature)
    sim_h(k)  = median over simulated futures of net wealth after h years (today's CHF), keep factor k
  choose k so that the MEDIAN over the cell of all sim_h(k) - target_h is zero (bisection on 0..K_MAX).
Income cells are needed because in this data wealth barely rises with income (validate_peers.py), while
the engine's saving does: one factor per age band left the high earners too rich and the low ones too poor.

When the synthetic data and the Swiss statistics in the engine disagree, the data wins (it is what the
twin is checked against), and the factor says by how much, openly: it is printed here, written to
calibration.json, shown in every profile's tags and in the chat's warnings.

    python3 -m twin.calibration          # fit on all profiles, write twin/calibration.json, print report

The old engine (twin/sim.py, still used by the CLI, chat and explorer until they move to the regime
engine) keeps its own factors in calibration_sim.json: factors("sim").
"""
import json
import statistics as st
from functools import lru_cache
from pathlib import Path

from .peers import find_peers
from .profile import _rows, load_profile

BANDS = [(15, 25), (25, 35), (35, 45), (45, 55), (55, 65), (65, 200)]
YEARS, HORIZONS, K_PEERS, PATHS, K_MAX = 10, (10, 20), 25, 100, 2.0
FILES = {"regime": Path(__file__).with_name("calibration.json"),
         "sim": Path(__file__).with_name("calibration_sim.json")}
FILE = FILES["regime"]


INCOME_GROUPS = 3  # income tertiles inside each age band (monthly working income, today's CHF)


@lru_cache(maxsize=1)
def income_cuts():
    """Tertile cut points of working income over all synthetic profiles."""
    from .scenarios import working_income
    xs = sorted(working_income(load_profile(r["profile_id"])) for r in _rows("synthetic-profiles.csv"))
    return tuple(round(xs[int(len(xs) * i / INCOME_GROUPS)]) for i in range(1, INCOME_GROUPS))


def band_of(age):
    return next(i for i, (lo, hi) in enumerate(BANDS) if lo <= age < hi) if age >= BANDS[0][0] else 0


def group_of(income):
    return sum(income >= c for c in income_cuts())


def cell_of(p):
    from .scenarios import working_income
    return band_of(p["age"]), group_of(working_income(p))


def simulate(p, years=YEARS, paths=PATHS):
    """Baseline run of the regime engine; .wealth(k, year) gives real net worth for any keep factor."""
    from .engine import run
    return run(p, "baseline", years=years, n_paths=paths)


def sim_wealth(r, keep, year=YEARS):
    """Median over futures of real net worth after `year` years with keep factor `keep`."""
    import numpy as np
    return float(np.median(r.wealth(keep, year)))


def peer_target(p, years=YEARS):
    from .scenarios import working_income
    res = find_peers(p, years=years, k=K_PEERS, ref_income=working_income(p))
    return st.median(q["wealth"] for q in res["peers"]) if res["peers"] else None


def _solve(gap):
    """k in [0, K_MAX] with gap(k) = 0; gap rises with k. Clamped at the ends."""
    lo, hi = 0.0, K_MAX
    if gap(lo) >= 0:
        return lo
    if gap(hi) <= 0:
        return hi
    for _ in range(25):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gap(mid) < 0 else (lo, mid)
    return (lo + hi) / 2


def fit(ids=None, horizons=HORIZONS):
    """Fit one keep factor per age band x income tertile. ids: restrict to these profiles (held-out checks).

    Each profile gives one gap per horizon (simulated wealth after h years - median wealth of peers h years
    older); k makes the median of all gaps in the cell zero, so it is fitted to 10 and 20 years at once."""
    pids = [r["profile_id"] for r in _rows("synthetic-profiles.csv")]
    if ids is not None:
        ids = set(ids)
        pids = [i for i in pids if i in ids]
    cuts = (0,) + income_cuts() + (10 ** 9,)
    rows = {(b, g): [] for b in range(len(BANDS)) for g in range(INCOME_GROUPS)}
    for pid in pids:
        p = load_profile(pid)
        targets = {h: t for h in horizons if (t := peer_target(p, h)) is not None}
        if targets:  # 65+ has no peers 20 years older: fitted on 10 years only
            rows[cell_of(p)].append((p, simulate(p, max(horizons)), targets))
    gaps = lambda items, k: [sim_wealth(r, k, h) - t for _, r, ts in items for h, t in ts.items()]
    out = []
    for (b, g), items in rows.items():
        lo_age, hi_age = BANDS[b]
        cell = {"band": [lo_age, hi_age], "income": [cuts[g], cuts[g + 1]]}
        if not items:
            out.append({**cell, "keep": 1.0, "n": 0})
            continue
        k = _solve(lambda k: st.median(gaps(items, k)))
        surplus = st.median(float((r.net() > 0).mean()) for _, r, _ in items)
        row = {**cell, "keep": round(k, 4), "fitted": True, "n": len(items),
               "share_months_in_surplus": round(surplus, 2),
               "median_engine_net_flow_chf_month": round(st.median(
                   float(r.flows(1.0)[:, :12].mean()) for _, r, _ in items))}
        if k <= 1e-3 or k >= K_MAX - 1e-3:
            # The factor hit a bound: no keep factor in 0..K_MAX matches the peers. Scaling surplus months
            # cannot fix a gap made in deficit months, so the cell is left uncalibrated and says why.
            gap = st.median(gaps(items, 1.0))
            row.update(keep=1.0, fitted=False, keep_at_bound=round(k, 4),
                       reason=(f"the engine is in surplus only {surplus:.0%} of months here and its "
                               f"median wealth is CHF {abs(gap):,.0f} {'below' if gap < 0 else 'above'} similar older "
                               f"people; scaling surplus (even x{K_MAX:g}) cannot close a gap made in deficit "
                               f"months (mostly retirement on 47.5% of the last wage with unchanged spending)"))
            k = 1.0
        for h in horizons:
            if not all(h in ts for *_, ts in items):
                continue
            row[f"y{h}"] = {"median_target_chf": round(st.median(ts[h] for *_, ts in items)),
                            "median_sim_raw_chf": round(st.median(sim_wealth(r, 1.0, h) for _, r, _ in items)),
                            "median_sim_calibrated_chf": round(st.median(sim_wealth(r, k, h) for _, r, _ in items))}
        out.append(row)
    hs = " and ".join(map(str, horizons))
    return {"engine": "regime (backend/regime via twin/engine.py)",
            "method": f"keep factor per age band x income tertile; median of (simulated wealth after h years - "
                      f"median wealth of background-matched peers h years older) = 0, h = {hs}",
            "bands": out}


def table_of(res):
    """(age lo, age hi, keep) [sim] or (age lo, age hi, income lo, income hi, keep, fitted) [regime] per cell."""
    return tuple((b["band"][0], b["band"][1], b["keep"]) if "income" not in b else
                 (b["band"][0], b["band"][1], *b["income"], b["keep"], b.get("fitted", True))
                 for b in res["bands"])


def _keep(cell):
    return cell[4] if len(cell) == 6 else cell[-1]


def _fitted(cell):
    return cell[5] if len(cell) == 6 else True


@lru_cache(maxsize=None)
def reasons(engine="regime"):
    """{(age lo, income lo): reason} for cells left uncalibrated."""
    f = FILES[engine]
    if not f.exists():
        return {}
    return {(b["band"][0], b.get("income", [0])[0]): b["reason"] for b in json.loads(f.read_text())["bands"]
            if b.get("fitted") is False}


@lru_cache(maxsize=None)
def factors(engine="regime"):
    """Fitted bands (lo, hi, keep). The regime file is fitted and written on first use if missing."""
    f = FILES[engine]
    if not f.exists():
        if engine != "regime":
            raise FileNotFoundError(f)
        f.write_text(json.dumps(fit(), indent=1))
    return table_of(json.loads(f.read_text()))


def _band(age, table=None, income=None):
    """Cell (lo, hi, [inc lo, inc hi,] keep) containing `age` (and `income` when the table has income
    groups). Ages below the first band use the youngest band."""
    table = table or factors()
    age = max(age, table[0][0])
    ok = lambda b: b[0] <= age < b[1] and (len(b) == 3 or income is None or b[2] <= income < b[3])
    return next((b for b in table if ok(b)), table[0])


def keep_factor(age, table=None, income=None):
    return _keep(_band(age, table, income))


def apply(p, table=None):
    """Calibrate profile p in place (sets calibration_rate and keep_factor) and tag it. Returns p."""
    from .scenarios import working_income
    cell = _band(p["age"], table, working_income(p))
    lo, hi, k = cell[0], cell[1], _keep(cell)
    p["calibration_rate"] = 1.0 - k
    p["keep_factor"] = k
    ages = f"{lo}-{hi - 1}" if hi < 200 else f"{lo}+"
    if len(cell) == 6:
        ages += f", income CHF {cell[2]:,}-{cell[3]:,}/mo" if cell[3] < 10 ** 9 else f", income CHF {cell[2]:,}+/mo"
    p["calibration"] = {"keep": k, "fitted": _fitted(cell), "group": ages,
                        "reason": None if _fitted(cell) else reasons().get((cell[0], cell[2]))}
    if not _fitted(cell):
        p["tags"]["saving calibration"] = f"none for your group (ages {ages}): " + (
            p["calibration"]["reason"] or "the keep factor could not be fitted")
        return p
    what = "of any monthly surplus" if k <= 1 else "times the modelled monthly surplus"
    p["tags"]["saving calibration"] = (f"you keep {k:.0%} {what} (fitted for ages {ages} so simulated wealth "
                                       f"after 10 and 20 years matches similar people that much older)")
    return p


def report(res):
    h = [int(k[1:]) for k in next(b for b in res["bands"] if b["n"]) if k.startswith("y")]
    print(f"Saving calibration, {res['engine']}\n(medians per cell, today's CHF; target = peers h years older)\n")
    head = "".join(f"{f'target {y}y':>13}{f'raw {y}y':>12}{f'calib. {y}y':>12}" for y in h)
    print(f"{'ages':<7}{'income/mo':>14}{'n':>5}{'keep':>7}{'flow/mo':>9}{head}")
    for b in res["bands"]:
        lo, hi = b["band"]
        if not b["n"]:
            continue
        il, ih = b.get("income", (0, 0))
        inc = f"{il:,}-{ih:,}" if ih < 10 ** 9 else f"{il:,}+"
        cols = "".join(f"{b[f'y{y}']['median_target_chf']:>13,}{b[f'y{y}']['median_sim_raw_chf']:>12,}"
                       f"{b[f'y{y}']['median_sim_calibrated_chf']:>12,}" if f"y{y}" in b else f"{'-':>13}{'-':>12}{'-':>12}"
                       for y in h)
        keep = f"{b['keep']:.0%}" if b.get("fitted", True) else "none"
        print(f"{f'{lo}-{hi - 1}' if hi < 200 else f'{lo}+':<7}{inc:>14}{b['n']:>5}{keep:>7}"
              f"{b['median_engine_net_flow_chf_month']:>9,}{cols}")
    ks = [b["keep"] for b in res["bands"] if b["n"] and b.get("fitted", True)]
    print(f"\nkeep factor (fitted cells): median {st.median(ks):.0%}, range {min(ks):.0%}-{max(ks):.0%} "
          f"(100% = engine needs no correction)")
    for b in res["bands"]:
        if b.get("fitted") is False:
            print(f"ages {b['band'][0]}+, income from CHF {b['income'][0]:,}: {b['reason']}")


if __name__ == "__main__":
    res = fit()
    FILE.write_text(json.dumps(res, indent=1))
    report(res)
    print(f"\nwrote {FILE}")
