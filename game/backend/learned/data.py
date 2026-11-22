"""Monthly ledger of the synthetic people, aligned with the locked peers split (peers/artifacts/split.json)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from peers.model import DATA, load_people

SPLIT = Path(__file__).resolve().parents[1] / "peers" / "artifacts" / "split.json"


@dataclass(frozen=True)
class Ledger:
    people: pd.DataFrame          # one row per person (peers.model.load_people), index = profile_id
    flows: np.ndarray             # [n, 24] monthly net cash flow, CHF
    opening: np.ndarray           # [n] opening cash
    month_of_year: np.ndarray     # [24] 1..12
    split: dict                   # train / validation / test id lists

    def rows(self, part: str) -> np.ndarray:
        ids = set(self.split[part])
        return np.array([i for i, pid in enumerate(self.people.index) if pid in ids])


@lru_cache(maxsize=1)
def ledger() -> Ledger:
    people = load_people()
    cb = pd.read_csv(DATA / "synthetic-cash-balances.csv").sort_values(["profile_id", "month"])
    wide = cb.pivot(index="profile_id", columns="month", values="net_transactions_chf").loc[people.index]
    months = list(wide.columns)
    return Ledger(people=people, flows=wide.values.astype(float),
                  opening=people.opening_cash.values.astype(float),
                  month_of_year=np.array([int(m[-2:]) for m in months]),
                  split=json.loads(SPLIT.read_text()))
