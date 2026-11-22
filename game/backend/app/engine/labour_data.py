"""Chances of losing and finding a job in Switzerland, by age, education and time without a job.

The numbers are the teammate's sourced assumptions in the repo-root `backend/assumptions/` (the same data
as the regime HMM), read straight from their CSVs so both stay in sync:

  transitions.csv         employed -> job_search by age band (BFS SAKE ILO unemployment by age, 2025)
                          job_search -> employed / part_time by age band (SECO durations by age, Dec 2025)
  income_assumptions.csv  masters_hazard_factor (BFS EHA 2023: graduates of a university master's lose
                          their job less often), long-term unemployment: after 12 months the chance of
                          finding work is halved (ILO long-term share)

Finding work = the band's exit to a full-time job + the exit to a part-time job (2%/month), so the total
equals the ILO exit rate for that age (the teammate's notes).

The Eurostat snapshot (scripts/fetch_job_loss_data.py, lfsi_long_q, all ages) is kept as a cross-check of
the all-ages level: see swiss_flows().
"""
from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
TEAM = REPO / "backend" / "assumptions"
SNAPSHOT = Path(__file__).resolve().parents[1] / "data" / "eurostat_lfsi_long_q_CH.csv"
BANDS = ("18_24", "25_34", "35_49", "50_64")


def band(age: float) -> str:
    """Age band of the teammate's table (under 18 uses the youngest, 65+ the oldest)."""
    a = int(age)
    return "18_24" if a < 25 else "25_34" if a < 35 else "35_49" if a < 50 else "50_64"


@lru_cache(maxsize=1)
def team_rates() -> dict:
    """{key: (value, source note)} for every labour-market assumption the game uses."""
    tr = {r["assumption_name"]: r for r in csv.DictReader(open(TEAM / "transitions.csv"))}
    inc = {r["parameter"]: r for r in csv.DictReader(open(TEAM / "income_assumptions.csv"))}
    pt = float(tr["job_search_to_part_time"]["monthly_hazard"])
    out: dict[str, tuple[float, str]] = {}
    for b in BANDS:
        loss = tr[f"job_loss_{b}"]
        out[f"job_loss_{b}"] = (float(loss["monthly_hazard"]), f"{loss['source']}. {loss['notes']}")
        exit_ = tr[f"job_search_exit_{b}"]
        out[f"job_find_{b}"] = (round(float(exit_["monthly_hazard"]) + pt, 6),
                                f"{exit_['source']}: {exit_['notes']} Includes {pt:.0%}/month into part-time work.")
    for key, name in (("masters_job_loss_factor", "masters_hazard_factor"),
                      ("long_term_find_factor", "long_term_unemployment_multiplier"),
                      ("long_term_months", "long_term_unemployment_months")):
        r = inc[name]
        out[key] = (float(r["value"]), f"{r['source']}. {r['notes']}")
    return out


@lru_cache(maxsize=1)
def swiss_flows(window_from: str = "2015-Q1") -> dict:
    """All-ages Swiss flows from the Eurostat snapshot (cross-check)."""
    with open(SNAPSHOT) as f:
        rows = [r for r in csv.DictReader(f) if r["quarter"] >= window_from]
    eu = sum(float(r["emp_to_une_pct"]) for r in rows) / len(rows)
    ue = sum(float(r["une_to_emp_pct"]) for r in rows) / len(rows)
    meta = json.loads(SNAPSHOT.with_suffix(".json").read_text())
    m = lambda q: 1 - (1 - q / 100) ** (1 / 3)  # noqa: E731
    return {"job_loss_monthly": m(eu), "job_find_monthly": m(ue), "from": rows[0]["quarter"],
            "to": rows[-1]["quarter"], "retrieved": meta.get("retrieved")}
