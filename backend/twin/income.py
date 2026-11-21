"""Income table: absolute monthly net income as a function of regime and path history.

income_chf_t = f(state_t, age_t, attributes_t, years_employed_t, last_salary_t)

Income is absolute, not a multiplier on today's income (a student's current income is a
useless base for her post-graduation salary). Someone who is already in a regime keeps
their own reference income while they stay there. Parameters live in
assumptions/income_assumptions.csv; every number has a source.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import AGE_BINS, AGE_LABELS, ASSUMPTIONS_DIR
from .types import STATE_INDEX, Intervention, Profile

E, P, J, S, R = (STATE_INDEX[s] for s in ["employed", "part_time", "job_search", "student", "retired"])


def attributes_at(profile: Profile, interventions: list[Intervention] | None, t: int) -> dict:
    attrs = dict(profile.attributes)
    for iv in interventions or []:
        if iv.kind == "set_attribute" and iv.attribute is not None and t >= iv.start_month:
            attrs[iv.attribute[0]] = iv.attribute[1]
    return attrs


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

    def _growth_rate(self, age: float) -> float:
        p = self.params
        if age < 30:
            return p["annual_experience_growth"]
        if age < 40:
            return p["experience_growth_30_40"]
        if age < 50:
            return p["experience_growth_40_50"]
        return p["experience_growth_50_plus"]

    def _replacement_rate(self, last_salary: np.ndarray, household: str | None) -> np.ndarray:
        p = self.params
        high = p["ui_replacement_with_dependents"]
        rate = np.full(last_salary.shape, high if household == "family" else p["ui_replacement_rate"])
        return np.where(last_salary <= p["ui_low_earnings_threshold_chf"], high, rate)

    def student_pay(self, profile: Profile) -> float:
        if profile.initial_state == "student" and profile.ref_income > 0:
            return float(profile.ref_income)
        return float(self.params["student_income_chf"])

    def income_paths(self, states: np.ndarray, profile: Profile,
                     interventions: list[Intervention] | None = None, *, experience: bool = True) -> np.ndarray:
        """[n_paths, T] income in CHF.

        Tracks months worked (experience growth, unemployment insurance, the graduate
        allowance) and the last wage. ``experience=False`` keeps the reference level
        flat, which is what the filter wants for a two-year ledger with no raises.
        """
        p = self.params
        n, T = states.shape
        ivs = interventions or []
        # Salary at t=0 is the level BEFORE this month's attribute interventions; the loop
        # rescales on the first month if a decision already applies.
        salary = np.full(n, self.full_time_salary(profile, dict(profile.attributes)))
        if profile.initial_state == "employed":
            last_salary = salary.copy()
        elif profile.initial_state == "part_time":
            last_salary = salary * p["part_time_fraction"]
        else:
            last_salary = np.zeros(n)
        has_worked = np.full(n, profile.initial_state in ("employed", "part_time"))
        anchor_unemployed = profile.initial_state == "job_search" and profile.ref_income > 0
        months_in_js = np.zeros(n)
        student_income = self.student_pay(profile)
        out = np.empty((n, T))
        prev_attrs = dict(profile.attributes)
        for t in range(T):
            attrs = attributes_at(profile, ivs, t)
            if attrs != prev_attrs:
                new = self.full_time_salary(profile, attrs)
                old = self.full_time_salary(profile, prev_attrs)
                if old > 0:
                    salary = salary * (new / old)
                    last_salary = last_salary * (new / old)
                prev_attrs = attrs
            age = profile.age + t / 12.0
            s = states[:, t]
            inc = np.zeros(n)
            inc[s == E] = salary[s == E]
            inc[s == P] = salary[s == P] * p["part_time_fraction"]

            js = s == J
            if np.any(js):
                max_m = p["ui_max_months_under_25"] if age < 25 else p["ui_max_months"]
                rate = self._replacement_rate(last_salary, attrs.get("household_type"))
                benefit = rate * np.minimum(last_salary, p["ui_insured_earnings_cap_chf"])
                on_ui = js & has_worked & (months_in_js < max_m)
                inc[on_ui] = benefit[on_ui]
                expired = js & has_worked & ~on_ui
                inc[expired] = p["job_search_floor_chf"]
                never = js & ~has_worked
                if anchor_unemployed:
                    inc[never] = profile.ref_income
                else:
                    wait = p["graduate_waiting_months"]
                    span = p["graduate_benefit_months"]
                    on_grad = never & (months_in_js >= wait) & (months_in_js < wait + span)
                    grad_pay = p["graduate_benefit_under_25_chf"] if age < 25 else p["graduate_benefit_chf"]
                    inc[on_grad] = grad_pay

            inc[s == S] = student_income
            if profile.initial_state == "retired":
                inc[s == R] = profile.ref_income
            else:
                inc[s == R] = last_salary[s == R] * p["retired_income_fraction"]
            out[:, t] = inc

            working = (s == E) | (s == P)
            earned = np.where(s == P, salary * p["part_time_fraction"], salary)
            last_salary = np.where(working, earned, last_salary)
            has_worked = has_worked | working
            months_in_js = np.where(js, months_in_js + 1.0, 0.0)
            if experience and (t + 1) % 12 == 0:
                g = self._growth_rate(profile.age + (t + 1) / 12.0)
                salary = np.where(working, salary * (1.0 + g), salary)
        return out

    def rent_paths(self, states: np.ndarray, profile: Profile,
                   interventions: list[Intervention] | None = None) -> np.ndarray:
        """[n_paths, T] rent in CHF.

        A student keeps their own rent until the first job, then moves to the employed
        rent. Anyone else keeps their own rent, and a month spent as a student uses the
        student rent. Abroad multiplies the result by the cost factor.
        """
        p = self.params
        n, T = states.shape
        ivs = interventions or []
        if profile.initial_state == "student" and profile.rent > 0:
            student_level = float(profile.rent)
            working_level = float(p["employed_rent_chf"])
        else:
            student_level = float(p["student_rent_chf"])
            working_level = float(profile.rent) if profile.rent > 0 else float(p["employed_rent_chf"])
        on_student_budget = np.full(n, profile.initial_state == "student")
        out = np.empty((n, T))
        for t in range(T):
            attrs = attributes_at(profile, ivs, t)
            s = states[:, t]
            working = (s == E) | (s == P)
            on_student_budget = np.where(working, False, on_student_budget)
            level = np.where((s == S) | on_student_budget, student_level, working_level)
            if attrs.get("abroad"):
                level = level * p["abroad_cost_multiplier"]
            out[:, t] = level
        return out


def fit_baseline_levels(profiles: pd.DataFrame) -> pd.DataFrame:
    """Median and IQR of reference net income by employment status and age band.

    The synthetic table is the spread. Cited levels (entry salary, UI, pensions, rents)
    live in assumptions/income_assumptions.csv.
    """
    bands = pd.cut(profiles["age_at_start"], AGE_BINS, labels=AGE_LABELS)
    g = profiles.groupby([profiles["employment_status"], bands], observed=True)["reference_monthly_net_income_chf"]
    return g.describe(percentiles=[0.25, 0.5, 0.75])[["count", "25%", "50%", "75%"]].reset_index()
