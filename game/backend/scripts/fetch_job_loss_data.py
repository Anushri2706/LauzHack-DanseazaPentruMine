"""Download Swiss labour-market flows from Eurostat (open data) into a dated snapshot the game reads.

Source: Eurostat, "Labour market transitions - quarterly data" (lfsi_long_q), DOI 10.2908/LFSI_LONG_Q,
built from the Swiss Labour Force Survey (BFS SAKE / ESPA). Free to reuse under the Eurostat copyright
notice (CC BY 4.0). For each quarter it gives the share of people who were

  * employed last quarter and are unemployed now  (EMP_UNE, % of employment)
  * unemployed last quarter and are employed now  (UNE_EMP, % of unemployment)

We keep the seasonally adjusted series for both sexes together. Run from game/backend:

    .venv/bin/python scripts/fetch_job_loss_data.py

It writes app/data/eurostat_lfsi_long_q_CH.csv and a .json with the provenance (URL, dataset update
date, retrieval date). The engine never goes online: it reads that snapshot.
"""
from __future__ import annotations

import csv
import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

URL = ("https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/lfsi_long_q"
       "?geo=CH&sex=T&s_adj=SA&format=JSON&lang=en")
OUT = Path(__file__).resolve().parents[1] / "app" / "data"
SERIES = {"emp_to_une_pct": ("EMP_UNE", "PC_EMP"), "une_to_emp_pct": ("UNE_EMP", "PC_UNE")}


def fetch(url: str = URL) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "many-lives-game/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def rows_from(d: dict) -> list[dict]:
    ids, size, dims = d["id"], d["size"], d["dimension"]

    def flat(sel: dict) -> int:
        i = 0
        for k, n in zip(ids, size):
            cat = dims[k]["category"]["index"]
            i = i * n + cat[sel[k]] if k in sel else i * n
        return i

    times = sorted(dims["time"]["category"]["index"], key=lambda t: dims["time"]["category"]["index"][t])
    fixed = {k: next(iter(dims[k]["category"]["index"])) for k in ids if len(dims[k]["category"]["index"]) == 1}
    out = []
    for t in times:
        row = {"quarter": t}
        for col, (ws, unit) in SERIES.items():
            v = d["value"].get(str(flat({**fixed, "wstatus": ws, "unit": unit, "time": t})))
            row[col] = v
        if all(row[c] is not None for c in SERIES):
            out.append(row)
    return out


def main() -> int:
    d = fetch()
    rows = rows_from(d)
    if not rows:
        print("No rows returned", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "eurostat_lfsi_long_q_CH.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["quarter", *SERIES])
        w.writeheader()
        w.writerows(rows)
    meta = {"dataset": "lfsi_long_q", "title": d.get("label"), "doi": "10.2908/LFSI_LONG_Q",
            "publisher": "Eurostat (from the Swiss Labour Force Survey, BFS SAKE)",
            "license": "Eurostat copyright notice, CC BY 4.0", "url": URL,
            "dataset_updated": d.get("updated"), "retrieved": date.today().isoformat(),
            "geo": "CH", "sex": "T", "s_adj": "SA",
            "columns": {"emp_to_une_pct": "employed -> unemployed, % of employment, per quarter",
                        "une_to_emp_pct": "unemployed -> employed, % of unemployment, per quarter"}}
    (OUT / "eurostat_lfsi_long_q_CH.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{len(rows)} quarters {rows[0]['quarter']}..{rows[-1]['quarter']} -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
