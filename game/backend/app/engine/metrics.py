"""Metric definitions for the Monte Carlo result contract (computed in ``montecarlo.summarise``).

Every metric reported by the game is listed here with its unit and interpretation. Percentiles are
quantiles of the simulated distribution (numpy 'linear' method) across paths: P10 means 10% of
simulated paths are at or below the value. They are NOT confidence intervals. Every path runs to the
horizon (no incomplete paths), so every probability's denominator is n_paths.
"""

PERCENTILE_NOTE = ("Quantile of the simulated distribution across paths (numpy linear interpolation); "
                   "not a confidence interval and not a forecast.")

METRICS: dict[str, dict] = {
    "cash": {"name": "Cash balance", "unit": "CHF", "basis": "nominal", "time_basis": "end of month",
             "stat": "P10/P50/P90 per month and at the horizon", "interpretation": PERCENTILE_NOTE},
    "net_worth_nominal": {"name": "Net worth (nominal)", "unit": "CHF", "basis": "nominal",
                          "time_basis": "end of month", "stat": "P10/P50/P90",
                          "definition": "cash + portfolio - debt", "interpretation": PERCENTILE_NOTE},
    "net_worth_real": {"name": "Net worth (real, today's CHF)", "unit": "CHF (today's prices)", "basis": "real",
                       "time_basis": "end of month", "stat": "P10/P50/P90",
                       "definition": "(cash + portfolio - debt) / price level", "interpretation": PERCENTILE_NOTE},
    "p_cash_below_zero_by_month": {"name": "Share of paths with negative cash", "unit": "probability",
                                   "basis": "nominal", "time_basis": "each month",
                                   "denominator": "all simulated paths (n_paths)"},
    "p_cash_ever_below_zero": {"name": "Share of paths with negative cash in any month", "unit": "probability",
                               "basis": "nominal", "time_basis": "any month up to the horizon",
                               "denominator": "all simulated paths (n_paths)"},
    "p_below_reserve_by_month": {"name": "Share of paths below the emergency reserve", "unit": "probability",
                                 "basis": "nominal CHF reserve, fixed", "time_basis": "each month",
                                 "denominator": "all simulated paths (n_paths)"},
    "p_ever_below_reserve": {"name": "Share of paths below the emergency reserve in any month",
                             "unit": "probability", "basis": "nominal CHF reserve, fixed",
                             "time_basis": "any month up to the horizon", "denominator": "all simulated paths (n_paths)"},
    "months_below_reserve": {"name": "Months below the emergency reserve", "unit": "months",
                             "basis": "count per path", "time_basis": "over the horizon",
                             "stat": "P10/P50/P90 and mean across paths", "interpretation": PERCENTILE_NOTE},
    "expected_months_in_state": {"name": "Mean months in each life regime", "unit": "months",
                                 "basis": "mean across paths", "time_basis": "over the horizon"},
}
