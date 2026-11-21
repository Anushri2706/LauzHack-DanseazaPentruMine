"""Run one what-if against the baseline through TwinSession's tools (shared by the CLI and the chat)."""
import numpy as np

from .engine import scenario


def warnings_for(p, key):
    w = []
    if key == "job_loss" and p["employment"] == "retired":
        w.append("Profile is retired: a 'job loss' changes little, since pension income continues.")
    if key == "masters" and p["age"] > 45:
        w.append("A degree late in a career has few working years left to pay off before retirement at 65.")
    k = p.get("keep_factor")
    cal = p.get("calibration") or {}
    if cal and not cal.get("fitted", True):
        w.append(f"Saving is NOT calibrated for your group (ages {cal['group']}): {cal.get('reason') or 'no factor fits'}. "
                 "Results use the engine's saving as-is and are likely too low for this group.")
    elif k is not None:
        what = (f"you are assumed to keep {k:.0%} of any monthly surplus" if k <= 1 else
                f"your modelled monthly surplus is scaled by {k:.0%}")
        w.append(f"Saving is calibrated: {what}, because similar people 10 and 20 years older in the "
                 f"data hold {'far less' if k < 1 else 'more'} than the engine's saving would give. "
                 f"Deficit months, expense cuts and side income are not scaled.")
    elif p["id"] != "USER":
        w.append("Synthetic profile without saving calibration: its spending comes from a ledger whose cash "
                 "flows imply far more saving than the wealth levels in the data show.")
    if str(p["tags"].get("other_spend", "")).startswith("estimated"):
        w.append("Your non-rent spending was not given; the engine's typical spending for your situation "
                 "(fitted on the synthetic ledgers) was used. If you spend more, every scenario saves less.")
    return w


def _row(summary, notes):
    """A TwinSession.simulate summary -> the row the explanation and the tables read."""
    nw = summary["net_worth_real_chf"]
    return {"scenario": summary["name"], "notes": notes,
            "nw_p10": nw["p10"], "nw_p50": nw["p50"], "nw_p90": nw["p90"],
            "p_shortfall": summary["share_paths_cash_negative"],
            "median_short_months": summary["median_months_cash_negative"],
            "p90_peak_shortfall": summary["p90_peak_shortfall_chf"],
            "median_buffer_months": summary["median_months_below_buffer"],
            "goals": {g["name"]: g["p_met"] for g in summary["goals"]},
            "years": summary["net_worth_real_by_year"],
            "months_in_state": summary["expected_months_in_state"]}


def calibration_banner(p, rows):
    """Code-written (not LLM) statement of whether results are calibrated, with both values side by side."""
    cal, k = p.get("calibration") or {}, p.get("keep_factor")
    years = rows[0]["years"][-1]["year"]
    both = "; ".join(f"{r['scenario']} CHF {r['nw_p50']:,.0f} calibrated / CHF {r['uncalibrated']['nw_p50']:,.0f} "
                     f"uncalibrated" for r in rows)
    if k is None:
        return [f"UNCALIBRATED results: the engine's saving is used as-is. Median net worth after {years} years: "
                + "; ".join(f"{r['scenario']} CHF {r['nw_p50']:,.0f}" for r in rows) + "."]
    if not cal.get("fitted", True):
        return [f"NOT CALIBRATED for your group (ages {cal['group']}): no keep factor fits it, so results are the engine "
                f"as-is. Median net worth after {years} years: "
                + "; ".join(f"{r['scenario']} CHF {r['nw_p50']:,.0f}" for r in rows) + "."]
    return [f"CALIBRATED results: the engine's monthly surplus is scaled to {k:.0%} so that people like you "
            f"(ages {cal.get('group', 'your age and income group')}) match the wealth of people 10-20 years older in the "
            "data. This corrects a group average, not your own path.",
            f"Median net worth after {years} years: {both}."]


def whatif(p, key, params=None, paths=300):
    """Baseline and what-if via TwinSession.simulate / compare. Returns plain data."""
    s, sid, notes = scenario(p, key, params, n_paths=paths)
    from .engine import baseline_note
    rows = [_row(s.simulate("baseline"), [baseline_note(p)])]
    out = {"rows": rows, "warnings": warnings_for(p, key), "paths": paths, "horizon_months": s.horizon_months,
           "keep_factor": s.keep_factor}
    ids = ["baseline"]
    if sid != "baseline":
        rows.append(_row(s.simulate(sid), notes))
        ids.append(sid)
    for row, i in zip(rows, ids):
        w = s.results[i].wealth
        row["nw_samples"] = w.wealth().tolist()
        raw = w.wealth(1.0)  # same futures, keep factor 100%
        row["uncalibrated"] = {q: round(float(np.quantile(raw, v))) for q, v in
                               (("nw_p10", .1), ("nw_p50", .5), ("nw_p90", .9))}
    out["banner"] = calibration_banner(p, rows)
    if sid != "baseline":
        out["compare"] = s.compare("baseline", sid)["delta_b_minus_a"]
        out["median_end_wealth_difference"] = out["compare"]["net_worth_real_p50_chf"]
    if p.get("trajectory"):
        out["peer_trajectory"] = p["trajectory"]["points"]
        out["peer_percentile"] = p["trajectory"]["percentile"]
    out["session"] = s  # for follow-up tool calls (sample_paths, sensitivity); not sent to the LLM
    return out
