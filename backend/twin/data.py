"""Loaders for the synthetic challenge data in Data/ and the assumption CSVs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .types import Profile

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
DATA_DIR = REPO_ROOT / "Data"
ASSUMPTIONS_DIR = BACKEND_DIR / "assumptions"
STARTER_KIT_DIR = REPO_ROOT / "starter-kit"


def load_profiles(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(data_dir / "synthetic-profiles.csv")


def load_cash_balances(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(data_dir / "synthetic-cash-balances.csv")


def load_goals(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(data_dir / "synthetic-goals.csv")


def person_months(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """One row per profile-month with the emission variable.

    Columns added: ``ratio`` = net_transactions_chf / reference_monthly_net_income_chf,
    ``is_december``, ``state`` (regime from employment_status), ``t`` (0..23).
    """
    from .types import state_of

    cb = load_cash_balances(data_dir)
    pr = load_profiles(data_dir)[
        ["profile_id", "employment_status", "age_at_start", "canton", "household_type",
         "reference_monthly_net_income_chf", "reference_monthly_rent_chf"]
    ]
    df = cb.merge(pr, on="profile_id", how="left")
    df["ratio"] = df["net_transactions_chf"] / df["reference_monthly_net_income_chf"]
    df["is_december"] = df["month"].str.endswith("-12")
    df["state"] = df["employment_status"].map(state_of)
    df["t"] = df.groupby("profile_id").cumcount()
    return df


def load_profile(profile_id: str = "P00001", data_dir: Path = DATA_DIR) -> Profile:
    """Build a ``Profile`` from the synthetic data (default: Alexia, P00001)."""
    pr = load_profiles(data_dir)
    row = pr.loc[pr["profile_id"] == profile_id]
    if row.empty:
        raise KeyError(f"Unknown profile_id {profile_id}")
    r = row.iloc[0]
    cb = load_cash_balances(data_dir)
    flows = cb.loc[cb["profile_id"] == profile_id, "net_transactions_chf"]
    ref_income = float(r["reference_monthly_net_income_chf"])
    mean_rate = float(flows.mean() / ref_income) if len(flows) and ref_income > 0 else None
    return Profile(
        age=int(r["age_at_start"]),
        employment=str(r["employment_status"]),
        ref_income=ref_income,
        rent=float(r["reference_monthly_rent_chf"]),
        attributes={"has_masters": False, "abroad": False, "canton": r["canton"],
                    "household_type": r["household_type"]},
        profile_id=profile_id,
        name=str(r["display_name"]),
        opening_cash=float(r["opening_cash_chf"]),
        mean_savings_rate=mean_rate,
    )


def observed_net_flows(profile_id: str, data_dir: Path = DATA_DIR) -> list[float]:
    cb = load_cash_balances(data_dir)
    return cb.loc[cb["profile_id"] == profile_id, "net_transactions_chf"].astype(float).tolist()
