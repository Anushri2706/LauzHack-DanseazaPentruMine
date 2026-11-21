"""Every modelling assumption in one place. All are illustrative, not official statistics."""
from contextlib import contextmanager

ASSUMPTIONS = {
    "wage_growth_annual":   (0.01, "Nominal income growth per year. Illustrative."),
    "regime_inflation":     ({"low": 0.005, "base": 0.015, "high": 0.04},
                             "Annual inflation on spending, by regime. Illustrative; replace with a licensed series."),
    "regime_transitions":   ([[0.80, 0.18, 0.02], [0.10, 0.80, 0.10], [0.05, 0.35, 0.60]],
                             "Yearly regime switching matrix (low, base, high). Hand-set, not fitted."),
    "portfolio_return":     (0.04, "Nominal annual mean return on existing investments."),
    "portfolio_vol":        (0.12, "Annual volatility of investment returns."),
    "noise_chf":            (300.0, "Monthly random swing in CHF for a new user (no history). "
                                    "Fixed in francs, NOT a share of income."),
    "buffer_months":        (3, "Comfortable buffer = this many months of spending."),
    "working_income_student": (4200.0, "Monthly net income if a student starts working (starter-kit value). "
                                       "Never below the income they already have."),
    "study_income_factor":  (0.5, "Share of current income kept while studying, for someone who is working."),
    "master_default_years": (2, "Master's duration if not given."),
    "master_default_tuition": (8000.0, "Annual tuition if not given (S001 value)."),
    "degree_premium":       (0.15, "Income uplift after completing a master's. Illustrative."),
    "job_loss_benefit":     (0.70, "Share of income received while unemployed."),
    "relocation_one_off":   (5000.0, "One-off moving cost in CHF."),
    "part_time_share":      (0.6, "Share of working income earned when working part-time (60% workload)."),
    "self_employed_rampup_months": (12, "Months a new business earns only part of the income."),
    "self_employed_rampup_share":  (0.5, "Share of working income earned during the ramp-up."),
    "self_employed_premium": (0.10, "Income uplift once the business is established. Illustrative."),
    "self_employed_setup":  (10000.0, "One-off cost to set up a business (equipment, registration)."),
    "self_employed_noise":  (2.0, "Monthly cash swings are this many times larger when self-employed."),
    "child_start_month":    (12, "Month the child arrives in the 'have a child' decision."),
    "child_cost_monthly":   (1300.0, "Extra monthly spending for a child (care, food, insurance). Illustrative."),
    "child_rent_factor":    (1.25, "Rent multiplier for a larger home once the child arrives."),
    "child_income_share":   (0.8, "Share of income kept during reduced work after the birth."),
    "child_reduced_months": (24, "Months of reduced work after the birth."),
}


def A(name):
    return ASSUMPTIONS[name][0]


@contextmanager
def override(**values):
    """Temporarily change assumptions (used by tests): with override(wage_growth_annual=0): ..."""
    old = {k: ASSUMPTIONS[k] for k in values}
    try:
        for k, v in values.items():
            ASSUMPTIONS[k] = (v, ASSUMPTIONS[k][1])
        yield
    finally:
        ASSUMPTIONS.update(old)
