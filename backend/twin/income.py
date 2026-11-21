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
        params = dict(zip(df["parameter"], df["value"].astype(float)))
        expense = ASSUMPTIONS_DIR / "expense_assumptions.csv"
        if expense.exists():
            extra = pd.read_csv(expense)
            params.update(dict(zip(extra["parameter"], extra["value"].astype(float))))
        return cls(params, df)

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

    def _apply_attribute_change(self, profile: Profile, prev_attrs: dict, attrs: dict,
                                salary: np.ndarray, last_salary: np.ndarray):
        """Rescale the next job's wage when attributes change.

        ``salary`` is the full-time wage she would earn if employed. ``last_salary``
        is the wage she has already been paid, which is what unemployment insurance
        uses. Turning ``abroad`` on or off changes the next job only. It does not
        rewrite insured earnings, so a move that starts in job search still pays
        UI on the last Swiss wage. A Zurich move does not change either wage.
        """
        if attrs == prev_attrs:
            return salary, last_salary, prev_attrs
        old = self.full_time_salary(profile, prev_attrs)
        new = self.full_time_salary(profile, attrs)
        if old > 0:
            ratio = new / old
            salary = salary * ratio
            was_abroad = bool(prev_attrs.get("abroad"))
            now_abroad = bool(attrs.get("abroad"))
            if was_abroad != now_abroad:
                factor = float(self.params["abroad_income_multiplier"])
                abroad_ratio = factor if now_abroad else 1.0 / factor
                ratio = ratio / abroad_ratio
            last_salary = last_salary * ratio
        return salary, last_salary, dict(attrs)

    def _housing_location_factor(self, attrs: dict) -> float:
        """Housing-bill prior for where she lives. Abroad and Zurich are alternatives."""
        p = self.params
        if attrs.get("abroad"):
            return float(p["abroad_cost_multiplier"])
        if attrs.get("zurich"):
            return float(p["zurich_housing_multiplier"])
        return 1.0

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
            salary, last_salary, prev_attrs = self._apply_attribute_change(
                profile, prev_attrs, attrs, salary, last_salary)
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

    def scales_spending(self) -> bool:
        """The fixed 1,546 rent is the default. ``scaled_spending`` turns on income-scaled rent."""
        return float(self.params.get("scaled_spending", 0.0)) >= 0.5

    def _capacity_earnings(self, states: np.ndarray, profile: Profile,
                           interventions: list[Intervention] | None, experience: bool) -> np.ndarray:
        """Pay that rent follows: this month's earnings while working, otherwise the last wage.

        Unemployment insurance does not reset the lease. A month with no wage yet stays at 0
        and the caller keeps the student rent.
        """
        p = self.params
        n, T = states.shape
        ivs = interventions or []
        salary = np.full(n, self.full_time_salary(profile, dict(profile.attributes)))
        if profile.initial_state == "employed":
            last_salary = salary.copy()
        elif profile.initial_state == "part_time":
            last_salary = salary * p["part_time_fraction"]
        else:
            last_salary = np.zeros(n)
        out = np.empty((n, T))
        prev_attrs = dict(profile.attributes)
        for t in range(T):
            attrs = attributes_at(profile, ivs, t)
            salary, last_salary, prev_attrs = self._apply_attribute_change(
                profile, prev_attrs, attrs, salary, last_salary)
            s = states[:, t]
            working = (s == E) | (s == P)
            earned = np.where(s == P, salary * p["part_time_fraction"], salary)
            out[:, t] = np.where(working, earned, last_salary)
            last_salary = np.where(working, earned, last_salary)
            if experience and (t + 1) % 12 == 0:
                g = self._growth_rate(profile.age + (t + 1) / 12.0)
                salary = np.where(working, salary * (1.0 + g), salary)
        return out

    def rent_from_earnings(self, earnings: np.ndarray) -> np.ndarray:
        """Net rent on a new lease: clip(floor, cap, anchor * (gross / anchor_income) ** elasticity)."""
        p = self.params
        raw = p["rent_anchor_chf"] * np.power(
            np.maximum(np.asarray(earnings, dtype=float), 1.0) / p["rent_income_anchor_chf"],
            p["rent_income_elasticity"],
        )
        return np.clip(raw, p["rent_floor_chf"], p["rent_cap_chf"])

    def consumption_paths(self, states: np.ndarray, profile: Profile,
                          interventions: list[Intervention] | None = None, *,
                          experience: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Housing and non-housing spend under the expense-priors recipe. Both are [n, T].

        While studying, housing is the student rent and the rest of the EPFL living budget
        makes up non-housing. The 0.25 rate is not used. After school a lease is signed once,
        at the wage on offer if she has no payslip yet, and that net rent stays put through
        raises and unemployment. Charges sit on top of the net rent. Non-housing is then
        ``base + rate * gross``, with gross the current wage, or the last wage, or the wage
        the lease was underwritten on.
        """
        p = self.params
        n, T = states.shape
        ivs = interventions or []
        if profile.initial_state == "student" and profile.rent > 0:
            student_level = float(profile.rent)
        else:
            student_level = float(p["student_rent_chf"])
        living = float(p["student_living_chf"])
        student_other = max(0.0, living - student_level)
        charges = float(p["rent_charges_chf"])
        base = float(p["nonhousing_base_chf"])
        rate = float(p["nonhousing_rate"])
        earnings = self._capacity_earnings(states, profile, ivs, experience)
        signed = np.zeros(n, dtype=bool)
        lease_net = np.zeros(n)
        housing = np.empty((n, T))
        other = np.empty((n, T))
        for t in range(T):
            attrs = attributes_at(profile, ivs, t)
            s = states[:, t]
            studentish = s == S
            offer = self.full_time_salary(profile, attrs)
            gross = np.where(earnings[:, t] > 1.0, earnings[:, t], offer)
            # Leaving school signs the lease, including a month of job search. It is not
            # recomputed when the wage later grows, and unemployment does not cut it.
            need = ~studentish & ~signed
            if np.any(need):
                lease_net = np.where(need, self.rent_from_earnings(gross), lease_net)
                signed = signed | need
            housing[:, t] = np.where(studentish, student_level, lease_net + charges)
            other[:, t] = np.where(studentish, student_other, base + rate * gross)
            housing[:, t] = housing[:, t] * self._housing_location_factor(attrs)
        return housing, other

    def rent_paths(self, states: np.ndarray, profile: Profile,
                   interventions: list[Intervention] | None = None, *, experience: bool = True) -> np.ndarray:
        """[n_paths, T] housing bill in CHF.

        Default (``scaled_spending`` off): a student keeps their own rent until the first job,
        then moves to the fixed employed rent. Anyone else keeps their own rent, and a month
        spent as a student uses the student rent.

        ``scaled_spending`` on: the expense-priors housing bill (student rent, or the sticky
        lease plus charges). Abroad multiplies the result by the Germany factor, and a
        Zurich move multiplies it by ``zurich_housing_multiplier``. The two do not stack.
        """
        if self.scales_spending():
            housing, _other = self.consumption_paths(states, profile, interventions, experience=experience)
            return housing
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
            level = level * self._housing_location_factor(attrs)
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
