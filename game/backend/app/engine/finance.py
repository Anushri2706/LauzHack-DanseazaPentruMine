"""Small, auditable financial helper formulas. All amounts CHF, all rates as decimals."""


def monthly_compound_rate(annual_rate: float) -> float:
    """Convert an effective annual rate to the equivalent monthly rate.

    (1 + r_month) ** 12 == 1 + annual_rate, so monthly and annual views stay consistent.
    Used for investment returns. Negative returns are allowed (must be > -100%).
    """
    if annual_rate <= -1:
        raise ValueError("annual_rate must be greater than -100%")
    return (1.0 + annual_rate) ** (1.0 / 12.0) - 1.0


def inflation_factor(annual_inflation: float, month: int) -> float:
    """Price level after `month` months relative to month 0: (1 + i) ** (t / 12)."""
    return (1.0 + annual_inflation) ** (month / 12.0)


def annuity_payment(principal: float, annual_rate: float, term_months: int) -> float:
    """Fixed monthly payment that fully amortizes `principal` over `term_months`.

    Loans use the nominal-rate convention: monthly rate = annual_rate / 12.
    A 0% rate is handled explicitly (no division by zero): payment = principal / term.
    """
    if principal <= 0:
        return 0.0
    if term_months < 1:
        raise ValueError("term_months must be >= 1")
    r = annual_rate / 12.0
    if r == 0:
        return principal / term_months
    return principal * r / (1.0 - (1.0 + r) ** (-term_months))
