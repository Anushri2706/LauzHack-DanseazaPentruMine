"""People like you: a k-nearest-neighbours model on a normalised map of people.

Every person in Data/ is a point in a multi-dimensional space with normalised dimensions:

  age                     z-score of age
  income                  z-score of log(1 + monthly net income)
  rent                    z-score of log(1 + monthly rent) and of rent as a share of income
  wealth                  z-scores of log(1 + opening cash), log(1 + investments), log(1 + debt)
  household               one-hot (single, couple, family, shared_housing)
  canton                  one-hot (10 cantons)

Each group has a weight (grid-searched). The life decision is the job status (employed, part_time,
self_employed, student, unemployed, retired): neighbours are looked up only among people who made that
decision ("people like you who chose X"), or, as a grid option, status is one more weighted dimension.

Target: the person's average monthly net cash flow over the 24 months in the data (CHF/month). Their cash
after 24 months is opening cash + 24 x that flow. The prediction is the (optionally distance-weighted)
mean of the neighbours' flows; the neighbours' P10-P90 is the range.

There is no sex column in the data; we do not infer it from first names.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[3] / "Data"
STATUSES = ("employed", "part_time", "self_employed", "student", "unemployed", "retired")
HOUSEHOLDS = ("single", "couple", "family", "shared_housing")
GROUPS = ("age", "income", "rent", "wealth", "household", "canton")


def load_people() -> pd.DataFrame:
    """One row per person: profile features + target (average monthly net flow over 24 months)."""
    p = pd.read_csv(DATA / "synthetic-profiles.csv")
    a = pd.read_csv(DATA / "synthetic-assets.csv")
    li = pd.read_csv(DATA / "synthetic-liabilities.csv")
    cb = pd.read_csv(DATA / "synthetic-cash-balances.csv")
    inv = a[a.asset_type == "investment_portfolio"].groupby("profile_id").value_chf.sum()
    debt = li.groupby("profile_id").opening_principal_chf.sum()
    flow = cb.groupby("profile_id").net_transactions_chf.mean()
    months = cb.groupby("profile_id").size()
    df = p.set_index("profile_id")
    out = pd.DataFrame({
        "name": df.display_name, "age": df.age_at_start.astype(float), "status": df.employment_status,
        "household": df.household_type, "canton": df.canton,
        "income": df.reference_monthly_net_income_chf.astype(float),
        "rent": df.reference_monthly_rent_chf.astype(float),
        "opening_cash": df.opening_cash_chf.astype(float),
        "investments": inv.reindex(df.index).fillna(0.0), "debt": debt.reindex(df.index).fillna(0.0),
        "flow": flow.reindex(df.index), "months": months.reindex(df.index),
    })
    out["rent_share"] = out.rent / out.income.clip(lower=1.0)
    return out


@dataclass
class Scaler:
    """Normalisation fitted on the training people only (no information from validation/test)."""
    mean: dict = field(default_factory=dict)
    std: dict = field(default_factory=dict)
    cantons: tuple = ()

    @staticmethod
    def raw(df: pd.DataFrame) -> dict[str, np.ndarray]:
        return {"age": df.age.values, "income": np.log1p(df.income.clip(lower=0).values),
                "rent": np.log1p(df.rent.values), "rent_share": np.clip(df.rent_share.values, 0, 5),
                "cash": np.log1p(df.opening_cash.clip(lower=0).values),
                "inv": np.log1p(df.investments.values), "debt": np.log1p(df.debt.values)}

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "Scaler":
        r = cls.raw(df)
        return cls({k: float(v.mean()) for k, v in r.items()}, {k: float(v.std() or 1.0) for k, v in r.items()},
                   tuple(sorted(df.canton.unique())))

    def groups(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        r = self.raw(df)
        z = {k: (v - self.mean[k]) / self.std[k] for k, v in r.items()}
        onehot = lambda col, cats: np.stack([(np.asarray(df[col], dtype=object) == c).astype(float) for c in cats], 1)  # noqa: E731
        return {"age": z["age"][:, None], "income": z["income"][:, None],
                "rent": np.stack([z["rent"], z["rent_share"]], 1),
                "wealth": np.stack([z["cash"], z["inv"], z["debt"]], 1),
                "household": onehot("household", HOUSEHOLDS), "canton": onehot("canton", self.cantons),
                "status": onehot("status", STATUSES)}

    def matrix(self, df: pd.DataFrame, weights: dict[str, float], status_weight: float = 0.0) -> np.ndarray:
        g = self.groups(df)
        parts = [g[k] * np.sqrt(weights.get(k, 0.0)) for k in GROUPS]
        parts.append(g["status"] * np.sqrt(status_weight))
        return np.concatenate(parts, 1)


def sq_dist(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    return ((q[:, None, :] - x[None, :, :]) ** 2).sum(-1)


def weighted_quantile(v: np.ndarray, w: np.ndarray, q: float) -> float:
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = np.cumsum(w) - 0.5 * w
    return float(np.interp(q * w.sum(), c, v))


@dataclass
class PeerModel:
    """Hyper-parameters chosen by grid search on the validation set."""
    k: int
    weights: dict
    decision: str            # "filter" = neighbours must share the job status; "feature" = weighted dimension
    status_weight: float = 0.0
    distance_weighted: bool = False

    def predict(self, scaler: Scaler, pool: pd.DataFrame, query: pd.DataFrame) -> pd.DataFrame:
        """Prediction, P10, P90 of the monthly net flow, and the neighbours used, for each query person."""
        sw = self.status_weight if self.decision == "feature" else 0.0
        X, Q = scaler.matrix(pool, self.weights, sw), scaler.matrix(query, self.weights, sw)
        D = sq_dist(Q, X)
        if self.decision == "filter":
            D = np.where(np.asarray(query.status, dtype=object)[:, None] == np.asarray(pool.status, dtype=object)[None, :], D, np.inf)
        y = pool.flow.values
        rows = []
        for i in range(len(query)):
            idx = np.argsort(D[i])[: self.k]
            idx = idx[np.isfinite(D[i, idx])]
            d = np.sqrt(D[i, idx])
            w = 1.0 / (d + 1e-3) if self.distance_weighted else np.ones(len(idx))
            v = y[idx]
            rows.append({"pred": float((w * v).sum() / w.sum()), "p10": weighted_quantile(v, w, 0.10),
                         "p90": weighted_quantile(v, w, 0.90), "neighbours": list(pool.index[idx])})
        return pd.DataFrame(rows, index=query.index)


def metrics(y: np.ndarray, pred: np.ndarray, p10: np.ndarray | None = None, p90: np.ndarray | None = None) -> dict:
    err = pred - y
    ss_res, ss_tot = float((err ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    out = {"mae_flow_chf_per_month": float(np.abs(err).mean()),
           "mae_cash_after_24m_chf": float(np.abs(24 * err).mean()),
           "median_abs_error_chf_per_month": float(np.median(np.abs(err))),
           "r2": 1 - ss_res / ss_tot if ss_tot else float("nan"), "n": int(len(y))}
    if p10 is not None:
        out["range_hit_rate_p10_p90"] = float(((y >= p10) & (y <= p90)).mean())
    return out
