"""Income table: absolute monthly net income as a function of regime and path history.

income_chf_t = f(state_t, age_t, attributes_t, years_employed_t, last_salary_t)

Income is absolute, not a multiplier on today's income (a student's current income is a
useless base for her post-graduation salary). Parameters live in
assumptions/income_assumptions.csv so every number has a source/notes column.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ASSUMPTIONS_DIR
from .types import STATE_INDEX, Intervention, Profile

E, P, J, S, R = (STATE_INDEX[s] for s in ["employed", "part_time", "job_search", "student", "retired"])


@dataclass
class IncomeTable:
    params: dict[str, float]
    table: pd.DataFrame | None = None
    baseline_levels: pd.DataFrame | None = None   # optional: median income by employment x age band

    @classmethod
    def from_csv(cls, path: str | Path = ASSUMPTIONS_DIR / "income_assumptions.csv") -> "IncomeTable":
        df = pd.read_csv(path)
        return cls(dict(zip(df["parameter"], df["value"].astype(float))), df)

    def with_overrides(self, overrides: dict | None) -> "IncomeTable":
        if not overrides:
            return self
        p = dict(self.params)
        p.update({k: float(v) for k, v in overrides.items() if k in p})
        return IncomeTable(p, self.table, self.baseline_levels)

    def full_time_salary(self, profile: Profile, attrs: dict) -> float:
        """Starting full-time salary for this person when entering 'employed'."""
        p = self.params
        if profile.initial_state in ("employed", "part_time") and profile.ref_income > 0:
            base = profile.ref_income if profile.initial_state == "employed" else profile.ref_income / p["part_time_fraction"]
        else:
            base = p["entry_salary_chf"]
        if attrs.get("has_masters"):
            base *= 1.0 + p["masters_premium_pct"]
        if attrs.get("abroad"):
            base *= p["abroad_income_multiplier"]
        return base

    def income_paths(self, states: np.ndarray, profile: Profile,
                     interventions: list[Intervention] | None = None) -> np.ndarray:
        """Vectorised over paths: [n_paths, T] income in CHF for sampled regime paths.

        Tracks per-path months employed (experience growth) and last full-time salary
        (unemployment insurance and retirement are fractions of it).
        """
        p = self.params
        n, T = states.shape
        ivs = interventions or []
        attrs = dict(profile.attributes)
        salary = np.full(n, self.full_time_salary(profile, attrs))
        last_salary = salary.copy()
        months_in_ui = np.zeros(n)
        out = np.empty((n, T))
        prev_attrs = dict(attrs)
        for t in range(T):
            attrs = dict(profile.attributes)
            for iv in ivs:
                if iv.kind == "set_attribute" and t >= iv.start_month:
                    attrs[iv.attribute[0]] = iv.attribute[1]
            if attrs != prev_attrs:
                # Attribute change (master's completed, moved abroad) rescales the salary level
                # but keeps accumulated experience growth.
                salary = salary * (self.full_time_salary(profile, attrs) / self.full_time_salary(profile, prev_attrs))
                prev_attrs = attrs
            s = states[:, t]
            inc = np.zeros(n)
            inc[s == E] = salary[s == E]
            inc[s == P] = salary[s == P] * p["part_time_fraction"]
            in_ui = (s == J) & (months_in_ui < p["ui_max_months"])
            inc[in_ui] = last_salary[in_ui] * p["ui_replacement_rate"]
            inc[(s == J) & ~in_ui] = p["job_search_floor_chf"]
            inc[s == S] = p["student_income_chf"]
            if profile.initial_state == "retired":
                inc[s == R] = profile.ref_income
            else:
                inc[s == R] = last_salary[s == R] * p["retired_income_fraction"]
            out[:, t] = inc

            working = (s == E) | (s == P)
            last_salary = np.where(working, salary, last_salary)
            months_in_ui = np.where(s == J, months_in_ui + 1, 0.0)
            if (t + 1) % 12 == 0:
                salary = np.where(working, salary * (1.0 + p["annual_experience_growth"]), salary)
        return out


def fit_baseline_levels(profiles: pd.DataFrame) -> pd.DataFrame:
    """Median and IQR of reference net income by employment status and age band.

    TODO(team): decide whether entry_salary_chf should come from this table (synthetic data)
    or from BFS LSE (public, citable). The synthetic table is fine for spread; cite BFS for levels.
    """
    bands = pd.cut(profiles["age_at_start"], [17, 25, 35, 50, 65, 120],
                   labels=["18-25", "26-35", "36-50", "51-65", "65+"])
    g = profiles.groupby([profiles["employment_status"], bands], observed=True)["reference_monthly_net_income_chf"]
    return g.describe(percentiles=[0.25, 0.5, 0.75])[["count", "25%", "50%", "75%"]].reset_index()
