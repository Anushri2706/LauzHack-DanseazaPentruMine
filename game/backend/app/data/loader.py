"""Load the synthetic CSVs (read-only) and build personas with labelled default assumptions.

We use the OPENING SNAPSHOT (assets + liabilities at 2025-01-01), not synthetic-cash-balances.csv.
Liabilities are snapshots with monthly_payment_chf = 0, so the engine amortizes them over
`loan_term_months` (our labelled assumption).
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from ..engine.catalogue import CANTON_PRICE_LEVEL, HOMES, home_price_key
from ..engine.assumptions import (ALEXIA_DEFAULTS, COST_SHARE_BY_HOUSEHOLD, LIVING_BY_HOUSEHOLD,
                                  LIVING_SHARE_OF_INCOME)

DATA_DIR = Path(__file__).resolve().parents[4] / "Data"
AS_OF = pd.Timestamp("2025-01-01")

STATUS_MAP = {
    "student": "studying", "employed": "working", "self_employed": "working",
    "part_time": "working", "unemployed": "unemployed", "retired": "retired",
}


@dataclass(frozen=True)
class Persona:
    profile_id: str
    name: str
    age: int
    city: str
    canton: str
    employment_status: str   # as in the data
    status: str              # engine status
    household_type: str
    opening_cash: float
    opening_investments: float
    opening_debt: float
    reference_income: float
    reference_rent: float
    goals: tuple[dict, ...]
    params: dict             # default assumption values
    tags: dict               # source tag per assumption
    start_salary: float
    start_living: float

    def card(self) -> dict:
        return {
            "profile_id": self.profile_id, "name": self.name, "age": self.age, "city": self.city,
            "canton": self.canton, "employment_status": self.employment_status, "status": self.status,
            "household_type": self.household_type, "opening_cash": self.opening_cash,
            "opening_investments": self.opening_investments, "opening_debt": self.opening_debt,
            "reference_income": self.reference_income, "reference_rent": self.reference_rent,
            "net_worth": self.opening_cash + self.opening_investments - self.opening_debt,
            "goals": list(self.goals), "synthetic": True,
        }


@dataclass(frozen=True)
class Dataset:
    profiles: pd.DataFrame
    assets: pd.DataFrame
    liabilities: pd.DataFrame
    goals: pd.DataFrame
    scenario_assumptions: pd.DataFrame
    snapshot: pd.DataFrame   # one row per profile: cash, investments, debt, net_worth + profile cols


def _month_index(date_str: str) -> int:
    """Row index of the month containing `date_str` (row 0 = Jan 2025)."""
    d = pd.Timestamp(date_str)
    return (d.year - AS_OF.year) * 12 + (d.month - AS_OF.month)


@lru_cache(maxsize=1)
def load_dataset(data_dir: str | None = None) -> Dataset:
    root = Path(data_dir) if data_dir else DATA_DIR
    profiles = pd.read_csv(root / "synthetic-profiles.csv")
    assets = pd.read_csv(root / "synthetic-assets.csv")
    liabilities = pd.read_csv(root / "synthetic-liabilities.csv")
    goals = pd.read_csv(root / "synthetic-goals.csv")
    scen = pd.read_csv(root / "scenario-assumptions.csv")

    cash = assets[assets.asset_type == "cash"].groupby("profile_id").value_chf.sum()
    inv = assets[assets.asset_type == "investment_portfolio"].groupby("profile_id").value_chf.sum()
    debt = liabilities.groupby("profile_id").opening_principal_chf.sum()
    snap = profiles.set_index("profile_id").copy()
    snap["cash"] = cash.reindex(snap.index).fillna(0.0)
    snap["investments"] = inv.reindex(snap.index).fillna(0.0)
    snap["debt"] = debt.reindex(snap.index).fillna(0.0)
    snap["net_worth"] = snap.cash + snap.investments - snap.debt
    return Dataset(profiles, assets, liabilities, goals, scen, snap.reset_index())


def _peer_salary(ds: Dataset, age: int, household: str) -> float:
    """Median net income of EMPLOYED people of similar age (±5y) in the data: a peer reference,
    used as the salary someone could earn when (re)starting work."""
    prof = ds.profiles
    q = prof[(prof.employment_status == "employed") & ((prof.age_at_start - age).abs() <= 5)]
    if len(q) < 10:
        q = prof[prof.employment_status == "employed"]
    return float(q.reference_monthly_net_income_chf.median())


def _goals_for(ds: Dataset, pid: str) -> tuple[dict, ...]:
    g = ds.goals[ds.goals.profile_id == pid]
    return tuple({"goal_id": r.goal_id, "name": r.goal_name, "target": float(r.target_amount_chf),
                  "target_date": r.target_date, "month": _month_index(r.target_date),
                  "priority": r.priority} for r in g.itertuples())


def build_persona(pid: str, ds: Dataset | None = None) -> Persona:
    ds = ds or load_dataset()
    rows = ds.snapshot[ds.snapshot.profile_id == pid]
    if rows.empty:
        raise KeyError(pid)
    r = rows.iloc[0]
    lia = ds.liabilities[ds.liabilities.profile_id == pid]
    debt = float(lia.opening_principal_chf.sum())
    debt_rate = float((lia.opening_principal_chf * lia.annual_interest_rate).sum() / debt) if debt > 0 else 0.0
    status = STATUS_MAP[r.employment_status]
    income, rent = float(r.reference_monthly_net_income_chf), float(r.reference_monthly_rent_chf)

    if pid == "P00001":
        params = {k: float(v) for k, (v, _) in ALEXIA_DEFAULTS.items()}
        tags = {k: tag for k, (_, tag) in ALEXIA_DEFAULTS.items()}
    else:
        params = {k: float(v) for k, (v, _) in ALEXIA_DEFAULTS.items()}
        tags = {k: ("ours" if tag == "profile" else tag) for k, (_, tag) in ALEXIA_DEFAULTS.items()}
        share = COST_SHARE_BY_HOUSEHOLD.get(r.household_type, 1.0)
        floor = float(LIVING_BY_HOUSEHOLD.get(r.household_type, 1600)) * share
        set_ = lambda k, v, tag: (params.__setitem__(k, float(v)), tags.__setitem__(k, tag))  # noqa: E731
        set_("rent", rent * share, "profile" if share == 1.0 else "ours")
        set_("abroad_rent", rent * share * 1100 / 900, "ours")
        if status in ("studying", "unemployed"):
            work = _peer_salary(ds, int(r.age_at_start), r.household_type)
            set_("salary_work", work, "peer")
        else:
            work = income
            set_("salary_work", income, "profile")
        set_("salary_master", work * 4800 / 4200, "ours")
        set_("abroad_income", work * 3700 / 4200, "ours")
        living_work = max(floor, LIVING_SHARE_OF_INCOME * work)
        set_("living_working", living_work, "ours")
        set_("living_working_master", living_work * 2200 / 2100, "ours")
        set_("abroad_living", living_work * 1800 / 2100, "ours")
        set_("living_studying", floor * 0.75, "ours")
        if status == "studying":
            set_("student_income", income, "profile")
        elif status == "retired":
            set_("pension_income", income, "profile")
            # Retirees: living costs scale with the pension (editable as "living while working").
            set_("living_working", max(floor, LIVING_SHARE_OF_INCOME * income), "ours")
    params["opening_cash"], tags["opening_cash"] = float(r.cash), "profile"
    params["opening_investments"], tags["opening_investments"] = float(r.investments), "profile"
    params["opening_debt"], tags["opening_debt"] = debt, "profile"
    params["opening_debt_rate"], tags["opening_debt_rate"] = debt_rate, "profile"
    params["start_age"], tags["start_age"] = float(r.age_at_start), "profile"

    level = CANTON_PRICE_LEVEL.get(str(r.canton), 1.0)
    for hid, home in HOMES.items():  # Lausanne prices scaled by a rough canton price level (ours)
        params[home_price_key(hid)] = home.price * level

    sa = ds.scenario_assumptions[ds.scenario_assumptions.profile_id == pid]
    for row in sa.itertuples():
        if row.parameter == "annual_tuition_chf":
            params["tuition_annual"], tags["tuition_annual"] = float(row.value), "scenario_csv"
        elif row.parameter == "study_duration_years":
            params["master_ft_months"], tags["master_ft_months"] = float(row.value) * 12, "scenario_csv"

    start_salary = {"studying": params["student_income"], "working": params["salary_work"],
                    "unemployed": income, "retired": params["pension_income"]}[status]
    start_living = {"studying": params["living_studying"], "working": params["living_working"],
                    "unemployed": params["living_studying"],  # leaner budget between jobs (ours)
                    "retired": params["living_working"]}[status]
    return Persona(
        profile_id=pid, name=str(r.display_name), age=int(r.age_at_start), city=str(r.city),
        canton=str(r.canton), employment_status=str(r.employment_status), status=status,
        household_type=str(r.household_type), opening_cash=float(r.cash),
        opening_investments=float(r.investments), opening_debt=debt, reference_income=income,
        reference_rent=rent, goals=_goals_for(ds, pid), params=params, tags=tags,
        start_salary=start_salary, start_living=start_living,
    )
