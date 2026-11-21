"""Run one what-if against the baseline through TwinSession's tools (shared by the CLI and the chat)."""
from .engine import scenario


def warnings_for(p, key):
    w = []
    if key == "job_loss" and p["employment"] == "retired":
        w.append("Profile is retired: a 'job loss' changes little, since pension income continues.")
    if key == "masters" and p["age"] > 45:
        w.append("A degree late in a career has few working years left to pay off before retirement at 65.")
    k = p.get("keep_factor")
    if k is not None:
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


def whatif(p, key, params=None, paths=300):
    """Baseline and what-if via TwinSession.simulate / compare. Returns plain data."""
    s, sid, notes = scenario(p, key, params, n_paths=paths)
    from .engine import baseline_note
    rows = [_row(s.simulate("baseline"), [baseline_note(p)])]
    out = {"rows": rows, "warnings": warnings_for(p, key), "paths": paths, "horizon_months": s.horizon_months,
           "keep_factor": s.keep_factor}
    rows[0]["nw_samples"] = s.results["baseline"].wealth.wealth().tolist()
    if sid != "baseline":
        rows.append(_row(s.simulate(sid), notes))
        rows[1]["nw_samples"] = s.results[sid].wealth.wealth().tolist()
        out["compare"] = s.compare("baseline", sid)["delta_b_minus_a"]
        out["median_end_wealth_difference"] = out["compare"]["net_worth_real_p50_chf"]
    if p.get("trajectory"):
        out["peer_trajectory"] = p["trajectory"]["points"]
        out["peer_percentile"] = p["trajectory"]["percentile"]
    out["session"] = s  # for follow-up tool calls (sample_paths, sensitivity); not sent to the LLM
    return out
