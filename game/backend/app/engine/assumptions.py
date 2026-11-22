"""Assumption catalogue: every number the engine uses, with unit and source tag.

Source tags (shown in the UI as honesty tags):
  profile      🟢 taken from the synthetic profile CSVs
  starter_kit  🟡 taken from the challenge starter kit (starter-kit/scenario_config.json)
  scenario_csv 🟡 taken from Data/scenario-assumptions.csv
  ours         🟡 our own labelled assumption
  official_est 🟢 estimated from official Swiss statistics (BFS, SECO, Eurostat; teammate's backend/assumptions)
Simulated outputs are 🔵 and peer references are ⚪ (not assumptions, so not listed here).

Salary and income figures from the starter kit are already NET of social contributions; tax_rate and
social_rate therefore default to 0 so nothing is deducted twice. Swiss income tax, however, is NOT
taken out of net pay, so it is a separate cost (income_tax_rate above income_tax_allowance).
Realism layers (all ours, all editable): income tax, investment costs, lifestyle creep on real
raises, lower living costs after retiring. Calibrated so a typical persona saves ≈ 15% of net
income, and Alexia's "start working" path is ≈ 90-110k at 30 (the challenge slide shows ≈ 93k).
"""
from dataclasses import dataclass

from .catalogue import catalogue_assumptions
from .labour_data import BANDS, team_rates


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
    "horizon_months": AssumptionMeta("Horizon", "months", "Starting position", 12, 1440),
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
    "investment_costs_annual": AssumptionMeta("Investment costs (fees + wealth tax)", "%/year of investments",
                                              "Saving & investing", 0, 0.05,
                                              note="Taken off the return every month. Fund fees ≈ 0.3-1%, Swiss wealth tax ≈ 0.1-0.5%."),
    "income_tax_rate": AssumptionMeta("Income tax on net pay (rough)", "% of net income above the allowance",
                                      "Income", 0, 0.5,
                                      note="Swiss income tax is not taken out of net pay: it is paid separately. "
                                           "Rough single-person estimate for Vaud; 0 = no income tax."),
    "income_tax_allowance": AssumptionMeta("Tax-free part of monthly income", "CHF/month (today's prices)", "Income", 0),
    "retirement_spending_factor": AssumptionMeta("Living costs after retiring", "% of living costs before", "Costs",
                                                 0.3, 1.5,
                                                 note="Other living costs (not rent) change once at retirement."),
    "lifestyle_creep": AssumptionMeta("Share of each pay rise spent", "% of the raise", "Costs", 0, 1,
                                      note="Living costs rise by this share of every raise (after inflation), "
                                           "as people's spending follows their income. 0 = fixed lifestyle forever."),
    # Borrowing
    "loan_rate": AssumptionMeta("Interest on new loans", "%/year", "Borrowing", 0, 0.3),
    "loan_term_months": AssumptionMeta("Loan term (new and existing debt)", "months", "Borrowing", 1, 360,
                                       note="Existing debt in the data has no payment schedule; we amortize it over this term."),
    "loan_max": AssumptionMeta("Largest loan you can take", "CHF", "Borrowing", 0),
    # Uncertainty (Monte Carlo worlds; ranges under our assumptions, not real probabilities)
    "return_volatility": AssumptionMeta("Investment volatility", "%/year (std. dev.)", "Uncertainty", 0, 0.6),
    "crash_prob_annual": AssumptionMeta("Chance of a market drop per year", "%/year", "Uncertainty", 0, 1),
    "crash_size": AssumptionMeta("Size of a market drop", "% of portfolio", "Uncertainty", 0, 0.9),
    "retirement_age": AssumptionMeta("Retirement age", "years", "Life course", 55, 75,
                                     note="BSV AHV 21 (backend/assumptions/income_assumptions.csv): working, between "
                                          "jobs and abroad all move to retired at this age."),
    "pension_replacement_rate": AssumptionMeta("Pension (share of last full salary)", "%", "Life course", 0, 1,
                                               note="OECD Pensions at a Glance 2025 net replacement rate of mandatory "
                                                    "pensions for an average earner (retired_income_fraction)."),
    "no_degree_wage_factor": AssumptionMeta("Pay without a degree (× salary if working now)", "×", "Income", 0, 1.5,
                                            note="Our assumption, roughly in line with the BFS wage structure "
                                                 "survey (LSE) gap between upper-secondary and tertiary education."),
    "ahv_max_pension": AssumptionMeta("Maximum AHV pension (2025)", "CHF/month", "Life course", 0, 10000,
                                      note="AHV/AVS 2025: single full pension 1,260-2,520 CHF/month. The AHV part of "
                                           "your pension is capped here; the rest is BVG."),
    "wage_index_annual": AssumptionMeta("Swiss nominal wage growth", "%/year", "Life course", -0.05, 0.1,
                                        note="BFS Swiss Wage Index, long-run nominal growth of about 1%/year. Half of "
                                             "the AHV mixed index."),
    "ahv_13th_pension": AssumptionMeta("13th AHV pension (from Dec 2026)", "1 = on, 0 = off", "Life course", 0, 1,
                                       note="Accepted by vote in March 2024: one extra monthly AHV pension each "
                                            "December, first paid in December 2026."),
    "rent_shock_prob_annual": AssumptionMeta("Chance of a rent rise per year", "%/year", "Uncertainty", 0, 1),
    "rent_shock_size": AssumptionMeta("Size of a rent rise", "% of rent", "Uncertainty", 0, 1),
    # Rules
    "switch_cooldown_months": AssumptionMeta("Minimum months between big moves", "months", "Rules", 0, 60),
}

for _key, _default, _label, _unit, _group, _lo, _hi, _note in catalogue_assumptions():
    META[_key] = AssumptionMeta(_label, _unit, _group, _lo, _hi, _note)

# Losing and finding a job: by age band, education and time without a job (labour_data.py).
_JOB = "Losing your job"
for _b in BANDS:
    _ages = _b.replace("_", "-")
    META[f"job_loss_{_b}"] = AssumptionMeta(f"Chance of losing your job per month, age {_ages}", "%/month", _JOB,
                                            0, 0.2, team_rates()[f"job_loss_{_b}"][1])
    META[f"job_find_{_b}"] = AssumptionMeta(f"Chance of finding work per month, age {_ages}", "%/month", _JOB,
                                            0, 1, team_rates()[f"job_find_{_b}"][1])
META["masters_job_loss_factor"] = AssumptionMeta("With a master's: × chance of losing your job", "×", _JOB, 0, 2,
                                                 team_rates()["masters_job_loss_factor"][1])
META["long_term_find_factor"] = AssumptionMeta("After long without a job: × chance of finding work", "×", _JOB,
                                               0, 1, team_rates()["long_term_find_factor"][1])
META["long_term_months"] = AssumptionMeta("Long without a job means more than", "months", _JOB, 1, 60,
                                          team_rates()["long_term_months"][1])

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
    "investment_costs_annual": (0.01, "ours"),
    "lifestyle_creep": (0.5, "ours"),
    "retirement_spending_factor": (0.8, "ours"),
    "income_tax_rate": (0.18, "ours"),
    "income_tax_allowance": (1000, "ours"),
    "loan_rate": (0.06, "ours"),
    "loan_term_months": (60, "ours"),
    "loan_max": (30000, "ours"),
    "switch_cooldown_months": (12, "ours"),
    "return_volatility": (0.15, "ours"),
    "crash_prob_annual": (0.05, "ours"),
    "crash_size": (0.2, "ours"),
    "retirement_age": (65, "ours"),
    "pension_replacement_rate": (0.475, "ours"),
    "no_degree_wage_factor": (0.8, "ours"),
    "ahv_max_pension": (2520, "ours"),
    "wage_index_annual": (0.01, "ours"),
    "ahv_13th_pension": (1, "ours"),
    "rent_shock_prob_annual": (0.1, "ours"),
    "rent_shock_size": (0.1, "ours"),
}
for _k, (_v, _src) in team_rates().items():
    ALEXIA_DEFAULTS[_k] = (_v, "official_est")
# Game catalogue (habits, items, homes, flats, housing rules, joy table): all our assumptions.
for _key, _default, *_rest in catalogue_assumptions():
    ALEXIA_DEFAULTS[_key] = (_default, "ours")

# Rough "other living costs" (excluding rent) by household type for personas that have no
# starter-kit figures. These are our labelled assumptions, not Swiss statistics.
LIVING_BY_HOUSEHOLD = {"single": 1600, "shared_housing": 1400, "couple": 2600, "family": 3600}

# The data gives an INDIVIDUAL net income but a HOUSEHOLD rent. For couples and families we assume
# the persona pays this share of household rent and living costs (a partner covers the rest).
# Other living costs (excluding rent) rise with income: we assume 45% of net income, never below the
# household minimum above. With median rent ≈ 26% and income tax ≈ 13% this leaves ≈ 15% saved.
LIVING_SHARE_OF_INCOME = 0.45

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
