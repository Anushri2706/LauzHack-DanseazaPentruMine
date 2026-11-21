"""Assumption catalogue: every number the engine uses, with unit and source tag.

Source tags (shown in the UI as honesty tags):
  profile      🟢 taken from the synthetic profile CSVs
  starter_kit  🟡 taken from the challenge starter kit (starter-kit/scenario_config.json)
  scenario_csv 🟡 taken from Data/scenario-assumptions.csv
  ours         🟡 our own labelled assumption
Simulated outputs are 🔵 and peer references are ⚪ (not assumptions, so not listed here).

Salary and income figures from the starter kit are already NET; tax_rate and social_rate
therefore default to 0 so nothing is taxed twice.
"""
from dataclasses import dataclass

from .catalogue import catalogue_assumptions


@dataclass(frozen=True)
class AssumptionMeta:
    label: str
    unit: str
    group: str
    min: float | None = None
    max: float | None = None
    note: str = ""


META: dict[str, AssumptionMeta] = {
    # Starting position
    "opening_cash": AssumptionMeta("Starting cash", "CHF", "Starting position"),
    "opening_investments": AssumptionMeta("Starting investments", "CHF", "Starting position", 0),
    "opening_debt": AssumptionMeta("Starting debt (principal)", "CHF", "Starting position", 0),
    "opening_debt_rate": AssumptionMeta("Starting debt interest", "%/year", "Starting position", 0, 0.3),
    "start_age": AssumptionMeta("Age at start", "years", "Starting position", 16, 90),
    "horizon_months": AssumptionMeta("Horizon", "months", "Starting position", 12, 240),
    # Income
    "student_income": AssumptionMeta("Student side-job income", "CHF/month net", "Income", 0),
    "salary_work": AssumptionMeta("Salary if working now", "CHF/month net", "Income", 0),
    "salary_master": AssumptionMeta("Salary after a master's", "CHF/month net", "Income", 0),
    "abroad_income": AssumptionMeta("Income abroad (in CHF)", "CHF/month net", "Income", 0,
                                    note="Converted at a fixed, stated FX rate; no FX risk modelled."),
    "pension_income": AssumptionMeta("Pension income", "CHF/month net", "Income", 0),
    "unemployment_benefit_rate": AssumptionMeta("Unemployment benefit", "% of last salary", "Income", 0, 1),
    "part_time_fraction": AssumptionMeta("Work share during part-time master's", "% of full salary", "Income", 0, 1),
    "raise_annual": AssumptionMeta("Annual raise", "%/year", "Income", -0.1, 0.2),
    "tax_rate": AssumptionMeta("Income tax rate (flat)", "% of salary", "Income", 0, 0.6,
                               note="0 because salary figures are already net."),
    "social_rate": AssumptionMeta("Social contributions (flat)", "% of salary", "Income", 0, 0.3,
                                  note="0 because salary figures are already net."),
    # Costs
    "rent": AssumptionMeta("Rent in Switzerland", "CHF/month", "Costs", 0),
    "living_studying": AssumptionMeta("Other living costs while studying", "CHF/month", "Costs", 0),
    "living_working": AssumptionMeta("Other living costs while working", "CHF/month", "Costs", 0),
    "living_working_master": AssumptionMeta("Other living costs, working after master's", "CHF/month", "Costs", 0),
    "abroad_rent": AssumptionMeta("Rent abroad (in CHF)", "CHF/month", "Costs", 0),
    "abroad_living": AssumptionMeta("Other living costs abroad (in CHF)", "CHF/month", "Costs", 0),
    "inflation_annual": AssumptionMeta("Inflation on costs", "%/year", "Costs", -0.05, 0.2),
    # Education
    "tuition_annual": AssumptionMeta("Master's tuition", "CHF/year", "Education", 0),
    "tuition_payments_per_year": AssumptionMeta("Tuition instalments", "per year", "Education", 1, 2),
    "master_ft_months": AssumptionMeta("Full-time master's duration", "months", "Education", 12, 48),
    "master_pt_months": AssumptionMeta("Part-time master's duration", "months", "Education", 12, 72),
    # Moves
    "relocation_cost": AssumptionMeta("Relocation one-off cost", "CHF", "Moves", 0),
    # Saving & investing
    "savings_rate": AssumptionMeta("Share of cash above buffer invested each month", "%", "Saving & investing", 0, 1),
    "buffer_months": AssumptionMeta("Cash buffer kept before investing", "months of costs", "Saving & investing", 0, 24),
    "return_annual": AssumptionMeta("Investment return", "%/year", "Saving & investing", -0.5, 0.3),
    # Borrowing
    "loan_rate": AssumptionMeta("Interest on new loans", "%/year", "Borrowing", 0, 0.3),
    "loan_term_months": AssumptionMeta("Loan term (new and existing debt)", "months", "Borrowing", 1, 360,
                                       note="Existing debt in the data has no payment schedule; we amortize it over this term."),
    "loan_max": AssumptionMeta("Largest loan you can take", "CHF", "Borrowing", 0),
    # Uncertainty (Monte Carlo worlds; ranges under our assumptions, not real probabilities)
    "return_volatility": AssumptionMeta("Investment volatility", "%/year (std. dev.)", "Uncertainty", 0, 0.6),
    "crash_prob_annual": AssumptionMeta("Chance of a market drop per year", "%/year", "Uncertainty", 0, 1),
    "crash_size": AssumptionMeta("Size of a market drop", "% of portfolio", "Uncertainty", 0, 0.9),
    "job_loss_prob_annual": AssumptionMeta("Chance of losing income per year", "%/year", "Uncertainty", 0, 1,
                                           note="While working or abroad; lasts 3-9 months on unemployment benefit."),
    "rent_shock_prob_annual": AssumptionMeta("Chance of a rent rise per year", "%/year", "Uncertainty", 0, 1),
    "rent_shock_size": AssumptionMeta("Size of a rent rise", "% of rent", "Uncertainty", 0, 1),
    # Rules
    "switch_cooldown_months": AssumptionMeta("Minimum months between big moves", "months", "Rules", 0, 60),
}

for _key, _default, _label, _unit, _group, _lo, _hi, _note in catalogue_assumptions():
    META[_key] = AssumptionMeta(_label, _unit, _group, _lo, _hi, _note)

# Alexia (P00001): values from the starter kit and the scenario CSV.
# Starter-kit "monthly_expenses_chf" are TOTAL costs including rent, so we split them as
# rent (profile, 900) + other living costs, keeping the totals identical:
#   studying 2,100 = 900 + 1,200 | working 3,000 = 900 + 2,100 | after master 3,100 = 900 + 2,200
#   abroad 2,900 = 1,100 (our split) + 1,800
ALEXIA_DEFAULTS: dict[str, tuple[float, str]] = {
    "opening_cash": (8000, "profile"),
    "opening_investments": (0, "profile"),
    "opening_debt": (0, "profile"),
    "opening_debt_rate": (0.0, "profile"),
    "start_age": (20, "profile"),
    "horizon_months": (120, "starter_kit"),
    "student_income": (1200, "profile"),
    "salary_work": (4200, "starter_kit"),
    "salary_master": (4800, "starter_kit"),
    "abroad_income": (3700, "starter_kit"),
    "pension_income": (0, "ours"),
    "unemployment_benefit_rate": (0.7, "ours"),
    "part_time_fraction": (0.6, "ours"),
    "raise_annual": (0.02, "ours"),
    "tax_rate": (0.0, "ours"),
    "social_rate": (0.0, "ours"),
    "rent": (900, "profile"),
    "living_studying": (1200, "starter_kit"),
    "living_working": (2100, "starter_kit"),
    "living_working_master": (2200, "starter_kit"),
    "abroad_rent": (1100, "ours"),
    "abroad_living": (1800, "starter_kit"),
    "inflation_annual": (0.01, "ours"),
    "tuition_annual": (8000, "scenario_csv"),
    "tuition_payments_per_year": (2, "ours"),
    "master_ft_months": (24, "scenario_csv"),
    "master_pt_months": (36, "ours"),
    "relocation_cost": (6500, "starter_kit"),
    "savings_rate": (0.5, "ours"),
    "buffer_months": (3, "ours"),
    "return_annual": (0.05, "ours"),
    "loan_rate": (0.06, "ours"),
    "loan_term_months": (60, "ours"),
    "loan_max": (30000, "ours"),
    "switch_cooldown_months": (12, "ours"),
    "return_volatility": (0.15, "ours"),
    "crash_prob_annual": (0.05, "ours"),
    "crash_size": (0.2, "ours"),
    "job_loss_prob_annual": (0.05, "ours"),
    "rent_shock_prob_annual": (0.1, "ours"),
    "rent_shock_size": (0.1, "ours"),
}
# Game catalogue (habits, items, homes, flats, housing rules, joy table): all our assumptions.
for _key, _default, *_rest in catalogue_assumptions():
    ALEXIA_DEFAULTS[_key] = (_default, "ours")

# Rough "other living costs" (excluding rent) by household type for personas that have no
# starter-kit figures. These are our labelled assumptions, not Swiss statistics.
LIVING_BY_HOUSEHOLD = {"single": 1600, "shared_housing": 1400, "couple": 2600, "family": 3600}

# The data gives an INDIVIDUAL net income but a HOUSEHOLD rent. For couples and families we assume
# the persona pays this share of household rent and living costs (a partner covers the rest).
# Other living costs (excluding rent) rise with income: we assume 50% of net income, never below the
# household minimum above. With median rent ≈ 26% of income this leaves ≈ 20-25% saved.
LIVING_SHARE_OF_INCOME = 0.5

COST_SHARE_BY_HOUSEHOLD = {"single": 1.0, "shared_housing": 1.0, "couple": 0.5, "family": 0.6}


def assumption_table(values: dict[str, float], tags: dict[str, str]) -> list[dict]:
    """Rows for the Assumptions panel: key, label, value, unit, tag, group, bounds, note."""
    rows = []
    for key, meta in META.items():
        rows.append({
            "key": key, "label": meta.label, "value": values[key], "unit": meta.unit,
            "tag": tags.get(key, "ours"), "group": meta.group,
            "min": meta.min, "max": meta.max, "note": meta.note,
        })
    return rows


def validate_params(params: dict[str, float]) -> None:
    missing = [k for k in META if k not in params]
    if missing:
        raise ValueError(f"Missing assumptions: {missing}")
    unknown = [k for k in params if k not in META]
    if unknown:
        raise ValueError(f"Unknown assumptions: {unknown}")
    for key, meta in META.items():
        v = params[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or abs(v) == float("inf"):
            raise ValueError(f"{key} must be a finite number")
        if meta.min is not None and v < meta.min:
            raise ValueError(f"{key}={v} is below the minimum {meta.min}")
        if meta.max is not None and v > meta.max:
            raise ValueError(f"{key}={v} is above the maximum {meta.max}")
