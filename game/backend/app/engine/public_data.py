"""Defaults and cross-checks from the public datasets listed in the challenge resources.

Snapshots are written by scripts/fetch_public_data.py into app/data/public/ (provenance in sources.json).
The engine never goes online: it reads those files. Each derived default says how it was computed, so
the Settings panel can show it and anyone can recompute it from the CSV.

  inflation_annual    World Bank FP.CPI.TOTL.ZG: geometric mean of Swiss CPI inflation over the last
                      INFLATION_YEARS years (a span about as long as a game life).
  mortgage_rate       SNB zikrepro: mean rate on new 10-year fixed mortgages over the whole series
                      (Dec 2007 on). The game's mortgage has one rate for its whole life, so a fixed rate fits.
  home_growth_annual  BIS WS_SPP: long-run REAL growth of Swiss residential property prices since 1970,
                      plus the inflation default above. The nominal series since 1970 would bring in the
                      high inflation of the 1970s-80s; the real series does not.

Cross-checks only (they do not set a default): World Bank unemployment against the unemployment the
job rates imply, the SNB policy rate against the mortgage rate, EUR/CHF moves for life abroad, and the
years with falling GDP.
"""
from __future__ import annotations

import csv
import json
import math
from functools import lru_cache
from pathlib import Path

DIR = Path(__file__).resolve().parents[1] / "data" / "public"
INFLATION_YEARS = 35
PROPERTY_FROM = "1970-Q1"


def _rows(name: str) -> list[dict]:
    with open(DIR / f"{name}.csv") as f:
        return list(csv.DictReader(f))


@lru_cache(maxsize=1)
def sources() -> dict:
    return json.loads((DIR / "sources.json").read_text())


def _cite(name: str) -> str:
    s = sources()[name]
    return f"{s['provider']} {s['code']}, {s['period']}, retrieved {s['retrieved']}"


def _geo_mean_pct(values: list[float]) -> float:
    return math.prod(1 + v / 100 for v in values) ** (1 / len(values)) - 1


def _quarters(a: str, b: str) -> int:
    return (int(b[:4]) - int(a[:4])) * 4 + int(b[-1]) - int(a[-1])


@lru_cache(maxsize=1)
def public_defaults() -> dict[str, tuple[float, str]]:
    """{assumption key: (default, note)} for the engine numbers set from public data."""
    cpi = [(int(r["year"]), float(r["value"])) for r in _rows("wb_inflation")]
    last = cpi[-INFLATION_YEARS:]
    infl = round(_geo_mean_pct([v for _, v in last]), 4)

    mort = [float(r["fixed_10y"]) for r in _rows("snb_mortgage_rates") if r["fixed_10y"]]
    mort_rows = _rows("snb_mortgage_rates")
    mortgage = round(sum(mort) / len(mort) / 100, 4)

    prop = {r["quarter"]: float(r["real_index"]) for r in _rows("bis_property_prices")}
    q_last = max(prop)
    real = (prop[q_last] / prop[PROPERTY_FROM]) ** (4 / _quarters(PROPERTY_FROM, q_last)) - 1
    home = round((1 + real) * (1 + infl) - 1, 4)

    return {
        "inflation_annual": (infl, (
            f"Geometric mean of Swiss CPI inflation {last[0][0]}-{last[-1][0]} "
            f"({_cite('wb_inflation')}). Latest year {last[-1][1]:.1f}%.")),
        "mortgage_rate": (mortgage, (
            f"Mean rate on new 10-year fixed mortgages, {mort_rows[0]['month']} to {mort_rows[-1]['month']} "
            f"({_cite('snb_mortgage_rates')}). Latest month {mort[-1]:.2f}%.")),
        "home_growth_annual": (home, (
            f"Real Swiss home prices grew {real:.2%}/year from {PROPERTY_FROM} to {q_last} "
            f"({_cite('bis_property_prices')}); plus the {infl:.2%} inflation default = {home:.2%}/year nominal.")),
    }


@lru_cache(maxsize=1)
def context() -> dict:
    """Public-data figures used for cross-checks and notes (not engine inputs)."""
    une = [(int(r["year"]), float(r["value"])) for r in _rows("wb_unemployment")]
    u10 = une[-10:]
    pol = _rows("bis_policy_rate")
    fx = [(r["month"], float(r["chf_per_eur"])) for r in _rows("fred_eur_chf")]
    fx00 = [v for m, v in fx if m >= "2000-01"]
    logs = [math.log(b / a) for a, b in zip(fx00, fx00[1:])]
    mean = sum(logs) / len(logs)
    vol = math.sqrt(sum((x - mean) ** 2 for x in logs) / len(logs) * 12)
    gdp = [(int(r["year"]), float(r["value"])) for r in _rows("wb_gdp_growth") if int(r["year"]) >= 1980]
    return {
        "unemployment_mean_pct": sum(v for _, v in u10) / len(u10), "unemployment_years": (u10[0][0], u10[-1][0]),
        "policy_rate_last_pct": float(pol[-1]["rate"]), "policy_rate_month": pol[-1]["month"],
        "policy_rate_mean_pct": sum(float(r["rate"]) for r in pol) / len(pol), "policy_rate_from": pol[0]["month"],
        "eur_chf_last": fx[-1][1], "eur_chf_month": fx[-1][0],
        "eur_chf_vol_annual": vol, "eur_chf_drift_annual": math.exp(mean * 12) - 1,
        "gdp_down_years": [y for y, v in gdp if v < 0], "gdp_years": (gdp[0][0], gdp[-1][0]),
    }


def implied_unemployment(p: dict, bands: tuple[str, ...]) -> float:
    """Steady-state unemployment u = loss / (loss + find), averaged over the age bands."""
    us = [p[f"job_loss_{b}"] / (p[f"job_loss_{b}"] + p[f"job_find_{b}"]) for b in bands]
    return sum(us) / len(us)
