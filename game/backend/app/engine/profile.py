"""Player profile: the user's facts, their provenance, and the conversion to the engine's ``Profile``.

All money is CHF. The engine's ``Profile.ref_income`` is NET monthly pay (that is what the synthetic
ledger records), so a gross or yearly income is converted here, in Python, with the migrated tax
module. Nothing is filled in silently: optional facts that are missing stay ``None`` and the result
carries a warning saying what the engine did instead.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

from . import tax
from .assumptions import AssumptionSet
from .decisions import CANTON_CHOICES
from .types import Profile, state_of

EMPLOYMENT = ("employed", "part_time", "self_employed", "student", "unemployed", "retired")
HOUSEHOLD = ("single", "couple", "family", "shared_housing")
PERIODS = ("monthly", "yearly")
BASES = ("gross", "net")
WAGE_STATUSES = ("employed", "part_time", "self_employed")


class ProfileError(ValueError):
    def __init__(self, errors: list[dict]):
        super().__init__("; ".join(e["message"] for e in errors))
        self.errors = errors


@dataclass(frozen=True)
class PlayerProfile:
    age: int
    employment: str
    income_amount_chf: float
    income_period: str            # monthly | yearly
    income_basis: str             # gross | net
    rent_chf_month: float
    cash_chf: float
    investments_chf: float = 0.0
    debt_chf: float = 0.0
    debt_annual_rate: float | None = None     # fraction, e.g. 0.05; required when debt_chf > 0
    other_expenses_chf_month: float | None = None
    canton: str | None = None
    household: str | None = None
    emergency_reserve_chf: float | None = None
    goals: tuple = ()
    sources: dict = field(default_factory=dict)  # field -> "user" | "confirmed_user" | "default" | "derived"

    def fingerprint(self) -> str:
        d = asdict(self)
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


def validate_profile(raw: dict) -> PlayerProfile:
    """Strict validation. Unknown fields, wrong types, missing required facts and out-of-range values are
    rejected with structured errors; nothing is corrected silently."""
    errors: list[dict] = []
    allowed = set(PlayerProfile.__dataclass_fields__)
    for k in raw:
        if k not in allowed:
            errors.append({"code": "unknown_field", "field": k, "message": f"unknown profile field {k!r}"})

    def num(name, lo, hi, required=True, integer=False):
        v = raw.get(name)
        if v is None:
            if required:
                errors.append({"code": "missing", "field": name, "message": f"{name} is required"})
            return None
        if isinstance(v, bool):
            errors.append({"code": "invalid_type", "field": name, "message": f"{name} must be a number"})
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            errors.append({"code": "invalid_type", "field": name, "message": f"{name} must be a number"})
            return None
        if integer and f != int(f):
            errors.append({"code": "invalid_type", "field": name, "message": f"{name} must be a whole number"})
            return None
        if not lo <= f <= hi:
            errors.append({"code": "out_of_range", "field": name, "message": f"{name}={f:g} outside [{lo:g}, {hi:g}]"})
            return None
        return int(f) if integer else f

    def choice(name, options, required=True):
        v = raw.get(name)
        if v is None:
            if required:
                errors.append({"code": "missing", "field": name, "message": f"{name} is required"})
            return None
        if v not in options:
            errors.append({"code": "invalid_choice", "field": name, "message": f"{name} must be one of {list(options)}"})
            return None
        return v

    age = num("age", 18, 100, integer=True)
    employment = choice("employment", EMPLOYMENT)
    income = num("income_amount_chf", 0, 10_000_000)
    period = choice("income_period", PERIODS)
    basis = choice("income_basis", BASES)
    rent = num("rent_chf_month", 0, 50_000)
    cash = num("cash_chf", -10_000_000, 100_000_000)
    inv = num("investments_chf", 0, 100_000_000, required=False) or 0.0
    debt = num("debt_chf", 0, 100_000_000, required=False) or 0.0
    rate = num("debt_annual_rate", 0, 0.5, required=False)
    other = num("other_expenses_chf_month", 0, 100_000, required=False)
    reserve = num("emergency_reserve_chf", 0, 10_000_000, required=False)
    canton = choice("canton", CANTON_CHOICES, required=False)
    household = choice("household", HOUSEHOLD, required=False)
    if debt and rate is None:
        errors.append({"code": "missing", "field": "debt_annual_rate",
                       "message": "debt_annual_rate is required when debt_chf > 0 (no rate is assumed)"})
    if basis == "gross" and employment is not None and employment not in WAGE_STATUSES:
        errors.append({"code": "unsupported", "field": "income_basis",
                       "message": f"gross-to-net conversion is only modelled for wages; give net income for {employment}"})
    if income is not None and period is not None:
        monthly = income / 12 if period == "yearly" else income
        if monthly > 100_000:
            errors.append({"code": "implausible", "field": "income_amount_chf",
                           "message": f"CHF {monthly:,.0f}/month is outside the supported range; please confirm the period"})
    if errors:
        raise ProfileError(errors)
    sources = dict(raw.get("sources") or {})
    for k in raw:
        if k != "sources" and raw[k] is not None:
            sources.setdefault(k, "user")
    return PlayerProfile(age=age, employment=employment, income_amount_chf=income, income_period=period,
                         income_basis=basis, rent_chf_month=rent, cash_chf=cash, investments_chf=inv,
                         debt_chf=debt, debt_annual_rate=rate, other_expenses_chf_month=other,
                         canton=canton, household=household, emergency_reserve_chf=reserve,
                         goals=tuple(raw.get("goals") or ()), sources=sources)


@dataclass
class EngineProfile:
    profile: Profile
    net_monthly_income_chf: float
    reserve_chf: float
    derived: dict          # derived values with how they were obtained
    warnings: list[str]
    assumption_ids: list[str]


def to_engine_profile(p: PlayerProfile, assumptions: AssumptionSet) -> EngineProfile:
    warnings: list[str] = []
    derived: dict = {}
    used: list[str] = []
    monthly = p.income_amount_chf / 12.0 if p.income_period == "yearly" else p.income_amount_chf
    if p.income_period == "yearly":
        warnings.append("Yearly income was divided by 12. The engine also samples a December bonus fitted on the "
                        "synthetic ledger, so a yearly figure that already includes a 13th salary may be overstated.")
    if p.income_basis == "gross":
        params = {r.id: assumptions.value(r.id) for r in assumptions.registry.rows.values() if r.table == "tax"}
        net = float(tax.net_from_gross(monthly, p.age, "wage", params))
        derived["net_monthly_income_chf"] = {"value": net, "method": "engine/tax.py net_from_gross (wage)",
                                             "from_gross_monthly_chf": monthly}
        used += [r.id for r in assumptions.registry.rows.values() if r.table == "tax"]
        warnings.append("Gross pay converted to net with an estimated single-person Lausanne (VD) tax curve "
                        "and Swiss social contributions; it is not your actual payslip.")
        if p.employment == "self_employed":
            warnings.append("Self-employed contributions differ from employee contributions; the wage conversion is an approximation.")
    else:
        net = monthly
    if p.canton and p.canton != "VD":
        warnings.append(f"Income tax uses the Lausanne (VD) estimate, not canton {p.canton}.")
    if p.other_expenses_chf_month is None:
        warnings.append("Other monthly expenses not given: spending follows the population spend/rent fit, not your own budget.")
    buffer = assumptions.value("buffer_months")
    if p.emergency_reserve_chf is not None:
        reserve = float(p.emergency_reserve_chf)
    else:
        base = p.rent_chf_month + (p.other_expenses_chf_month or 0.0)
        reserve = buffer * base
        used.append("buffer_months")
        derived["emergency_reserve_chf"] = {"value": reserve, "method": f"buffer_months ({buffer:g}) x (rent + other expenses given)"}
        if p.other_expenses_chf_month is None:
            warnings.append("Emergency reserve derived from rent only (other expenses unknown).")
    loans = [(float(p.debt_chf), float(p.debt_annual_rate))] if p.debt_chf else []
    if loans:
        warnings.append("Debt is modelled interest-only: the principal is never repaid in the simulation.")
    if p.employment == "student":
        warnings.append("As a student with no decision, the model keeps you studying for the whole horizon "
                        "(student exit hazards are 0); add 'start working' or a master's decision.")
    prof = Profile(
        age=int(p.age), employment=p.employment, ref_income=float(net), rent=float(p.rent_chf_month),
        attributes={"has_masters": False, "abroad": False, "canton": p.canton, "household_type": p.household},
        opening_cash=float(p.cash_chf), portfolio_chf=float(p.investments_chf), loans=loans)
    state_of(p.employment)  # raises on an unknown status
    return EngineProfile(prof, float(net), float(reserve), derived, warnings, used)
