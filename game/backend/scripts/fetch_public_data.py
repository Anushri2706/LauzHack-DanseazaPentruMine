"""Download the public datasets listed in the challenge resources into dated snapshots the game reads.

Only the Swiss (or CHF) series that fit a part of the engine are kept
(challenge-instructions-and-data/resources/data/*.md):

  World Bank  FP.CPI.TOTL.ZG    Swiss consumer price inflation, annual %       -> inflation_annual
  World Bank  SL.UEM.TOTL.ZS    Swiss unemployment, % of labour force (ILO)    -> cross-check of job rates
  World Bank  NY.GDP.MKTP.KD.ZG Swiss real GDP growth, annual %                -> context (recession years)
  SNB         zikrepro          Rates on new mortgages, mean, monthly          -> mortgage_rate
  BIS         WS_SPP Q.CH.N/R   Swiss residential property prices, nominal and real, quarterly -> home_growth_annual
  BIS         WS_CBPOL M.CH     SNB policy rate, monthly                       -> cross-check of mortgage_rate
  FRED        DEXSZUS, DEXUSEU  CHF per USD and USD per EUR, daily             -> EUR/CHF context for moving abroad

The US-only series in the same lists (DGS10, FEDFUNDS, CPIAUCSL, UNRATE, USSTHPI, SCF) are not used: the
game models a person living in Switzerland, and the lists themselves say they are not Swiss figures.

Run from game/backend (needs internet; the engine itself never goes online):

    .venv/bin/python scripts/fetch_public_data.py

Writes app/data/public/<series>.csv plus app/data/public/sources.json (provider, code, unit, URL, dataset
date, retrieval date, terms), following the challenge citation template.
"""
from __future__ import annotations

import csv
import io
import json
import sys
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "app" / "data" / "public"
UA = {"User-Agent": "many-lives-game/1.0"}
WB = "https://api.worldbank.org/v2/country/CHE/indicator/{}?format=json&per_page=200"
SNB = "https://data.snb.ch/api/cube/zikrepro/data/csv/en"
BIS = "https://stats.bis.org/api/v1/data/{}/all?format=csv"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

TERMS = {
    "World Bank": "CC BY 4.0 (World Bank Open Data terms of use)",
    "SNB": "Free reuse with source citation (SNB data portal terms, data.snb.ch)",
    "BIS": "Reuse with attribution under the BIS terms of use (bis.org/terms_statistics.htm)",
    "FRED": "Federal Reserve Board H.10 data via FRED; cite FRED (fred.stlouisfed.org/legal)",
}


def get(url: str, tries: int = 3) -> bytes:
    # FRED drops connections from unknown user agents; a plain curl agent string is accepted.
    headers = {"User-Agent": "curl/8.7.1"} if "fred.stlouisfed.org" in url else UA
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as r:
                return r.read()
        except OSError:
            if i == tries - 1:
                raise
    raise AssertionError("unreachable")


def write(name: str, header: list[str], rows: list[list]) -> None:
    with open(OUT / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def world_bank(code: str) -> tuple[list[list], str]:
    meta, data = json.loads(get(WB.format(code)))
    rows = sorted([int(r["date"]), round(float(r["value"]), 6)] for r in data if r["value"] is not None)
    return rows, meta.get("lastupdated", "")


def snb_mortgages() -> tuple[list[list], str]:
    """Mean rate on new mortgages: variable, 5-year fixed, 10-year fixed (% per year, monthly)."""
    text = get(SNB).decode("utf-8-sig")
    published = next((ln.split(";")[1].strip('"') for ln in text.splitlines() if ln.startswith('"PublishingDate"')), "")
    want = {"MV": "variable", "50": "fixed_5y", "10": "fixed_10y"}
    by_month: dict[str, dict] = defaultdict(dict)
    for ln in text.splitlines():
        parts = [x.strip('"') for x in ln.split(";")]
        if len(parts) == 4 and parts[1] == "M" and parts[2] in want and parts[3]:
            by_month[parts[0]][want[parts[2]]] = round(float(parts[3]), 6)
    rows = [[m, v.get("variable", ""), v.get("fixed_5y", ""), v.get("fixed_10y", "")]
            for m, v in sorted(by_month.items()) if "fixed_10y" in v]
    return rows, published


def bis(key: str) -> list[tuple[str, float]]:
    out = []
    for r in csv.DictReader(io.StringIO(get(BIS.format(key)).decode())):
        if r["OBS_VALUE"]:
            out.append((r["TIME_PERIOD"], float(r["OBS_VALUE"])))
    return out


def eur_chf_monthly() -> list[list]:
    """Monthly mean of CHF per EUR = (CHF per USD) × (USD per EUR), from daily FRED quotes."""
    def daily(sid: str) -> dict[str, float]:
        rd = csv.DictReader(io.StringIO(get(FRED.format(sid)).decode()))
        return {r["observation_date"]: float(r[sid]) for r in rd if r[sid] not in ("", ".")}
    chf_per_usd, usd_per_eur = daily("DEXSZUS"), daily("DEXUSEU")
    months: dict[str, list[float]] = defaultdict(list)
    for d, v in chf_per_usd.items():
        if d in usd_per_eur:
            months[d[:7]].append(v * usd_per_eur[d])
    return [[m, round(sum(v) / len(v), 6), len(v)] for m, v in sorted(months.items())]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    sources: dict[str, dict] = {}

    for name, code, title, unit in (
        ("wb_inflation", "FP.CPI.TOTL.ZG", "Inflation, consumer prices", "% per year"),
        ("wb_unemployment", "SL.UEM.TOTL.ZS", "Unemployment, total (modeled ILO estimate)", "% of labour force"),
        ("wb_gdp_growth", "NY.GDP.MKTP.KD.ZG", "GDP growth", "% per year, constant prices"),
    ):
        rows, updated = world_bank(code)
        write(name, ["year", "value"], rows)
        sources[name] = {"provider": "World Bank", "code": code, "title": title, "geo": "CH", "frequency": "annual",
                         "unit": unit, "period": f"{rows[0][0]}-{rows[-1][0]}", "dataset_updated": updated,
                         "url": WB.format(code)}

    rows, published = snb_mortgages()
    write("snb_mortgage_rates", ["month", "variable", "fixed_5y", "fixed_10y"], rows)
    sources["snb_mortgage_rates"] = {
        "provider": "SNB", "code": "zikrepro",
        "title": "Interest rates on new mortgage loans, mean value (D0=M; D1=MV, 50, 10)", "geo": "CH", "frequency": "monthly",
        "unit": "% per year", "period": f"{rows[0][0]}..{rows[-1][0]}", "dataset_updated": published, "url": SNB}

    nominal, real = dict(bis("WS_SPP/Q.CH.N.628")), dict(bis("WS_SPP/Q.CH.R.628"))
    q = sorted(set(nominal) & set(real))
    write("bis_property_prices", ["quarter", "nominal_index", "real_index"], [[k, nominal[k], real[k]] for k in q])
    sources["bis_property_prices"] = {
        "provider": "BIS", "code": "WS_SPP Q.CH.R.628",
        "title": "Residential property prices, Switzerland, nominal and real (CPI-deflated), index 2010 = 100",
        "geo": "CH", "frequency": "quarterly", "unit": "index", "period": f"{q[0]}..{q[-1]}",
        "url": f"{BIS.format('WS_SPP/Q.CH.N.628')} ; {BIS.format('WS_SPP/Q.CH.R.628')}"}

    pol = [[m, v] for m, v in bis("WS_CBPOL/M.CH") if m >= "2000-01"]
    write("bis_policy_rate", ["month", "rate"], pol)
    sources["bis_policy_rate"] = {
        "provider": "BIS", "code": "WS_CBPOL M.CH", "title": "Central bank policy rate, Switzerland (SNB)",
        "geo": "CH", "frequency": "monthly, end of period", "unit": "% per year",
        "period": f"{pol[0][0]}..{pol[-1][0]}", "url": BIS.format("WS_CBPOL/M.CH")}

    fx = eur_chf_monthly()
    write("fred_eur_chf", ["month", "chf_per_eur", "days"], fx)
    sources["fred_eur_chf"] = {
        "provider": "FRED", "code": "DEXSZUS × DEXUSEU",
        "title": "CHF per EUR, monthly mean of daily noon buying rates (Federal Reserve H.10)",
        "geo": "CH/EA", "frequency": "monthly (from daily)", "unit": "CHF per EUR",
        "period": f"{fx[0][0]}..{fx[-1][0]}", "url": f"{FRED.format('DEXSZUS')} ; {FRED.format('DEXUSEU')}"}

    for s in sources.values():
        s["retrieved"] = today
        s["terms"] = TERMS[s["provider"]]
    (OUT / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    for k, s in sources.items():
        print(f"{k:22} {s['period']:>20}  {s['provider']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
