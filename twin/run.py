"""Run one what-if against the baseline and return plain data (shared by the CLI and the chat)."""
from .scenarios import TEMPLATES
from .sim import make_shocks, run_path, summarize


def warnings_for(p, key):
    w = []
    if key == "job_loss" and p["employment"] == "retired":
        w.append("Profile is retired: 'job loss' only models a temporary cut in pension income.")
    if key == "masters" and p["age"] > 45:
        w.append("Engine has no retirement yet: a degree premium is applied until the end of the "
                 "10-year horizon, which overstates the payoff at this age.")
    if p["age"] + p["horizon"] // 12 > 65 and p["employment"] != "retired":
        w.append("Engine does not model retirement at 65 yet: income continues unchanged.")
    if p.get("keep_factor") is not None:
        w.append(f"Saving is calibrated: you are assumed to keep {p['keep_factor']:.0%} of any monthly surplus, "
                 f"because similar people 10 years older in the data hold far less than full saving would give. "
                 f"The rest is treated as spending your inputs do not show.")
    elif p["id"] != "USER":
        w.append("Synthetic profile: its spending comes from a ledger whose cash flows imply far more saving "
                 "than the wealth levels in the data show, so its end wealth is likely overstated.")
    if str(p["tags"].get("other_spend", "")).startswith("estimated"):
        w.append(f"Your non-rent spending was not given; CHF {p['other_spend']:,.0f}/month (typical for "
                 f"similar profiles) was used. If you spend more, every scenario saves less.")
    return w


def whatif(p, key, params=None, paths=300):
    params = params or {}
    shocks = make_shocks(p, paths)
    rows = []
    for k in (("baseline",) if key == "baseline" else ("baseline", key)):
        plan = TEMPLATES[k](p, **params)
        s = summarize([run_path(p, plan, sh) for sh in shocks], p)
        rows.append({"scenario": plan["name"], "notes": plan["notes"], **s})
    out = {"rows": rows, "warnings": warnings_for(p, key), "paths": paths, "horizon_months": p["horizon"]}
    if p.get("trajectory"):
        out["peer_trajectory"] = p["trajectory"]["points"]
        out["peer_percentile"] = p["trajectory"]["percentile"]
    if len(rows) == 2:
        out["median_end_wealth_difference"] = rows[1]["nw_p50"] - rows[0]["nw_p50"]
    return out
