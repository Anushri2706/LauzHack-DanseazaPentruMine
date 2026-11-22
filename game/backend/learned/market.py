"""Inflation regimes and major-expense events, from feature/combined-engine (engine/wealth.py, engine/events.py,
data/assumptions/market_assumptions.csv, game_assumptions.csv). Values are the teammates' (illustrative / hand-set /
placeholder, as they label them). Used only if they improve the backtest (learned/backtest.py)."""
from __future__ import annotations

import numpy as np

MARKET = {"inflation_low": 0.005, "inflation_base": 0.015, "inflation_high": 0.04, "inflation_stay_low": 0.80,
          "inflation_low_to_base": 0.18, "inflation_base_to_low": 0.10, "inflation_base_to_high": 0.10,
          "inflation_high_to_low": 0.05, "inflation_high_to_base": 0.35}
MAJOR_EXPENSE = {"annual_prob": 0.10, "cost_chf": 3000.0}


def price_level(n: int, horizon: int, seed: int, p: dict = MARKET) -> np.ndarray:
    """[n, T] price level: a yearly low / base / high inflation regime chain (combined-engine sample_market)."""
    A = np.array([
        [p["inflation_stay_low"], p["inflation_low_to_base"], 1 - p["inflation_stay_low"] - p["inflation_low_to_base"]],
        [p["inflation_base_to_low"], 1 - p["inflation_base_to_low"] - p["inflation_base_to_high"], p["inflation_base_to_high"]],
        [p["inflation_high_to_low"], p["inflation_high_to_base"], 1 - p["inflation_high_to_low"] - p["inflation_high_to_base"]]])
    rates = np.array([p["inflation_low"], p["inflation_base"], p["inflation_high"]])
    rng = np.random.default_rng(10_000 + seed)
    state, infl = np.full(n, 1), np.empty((n, horizon))
    for t in range(horizon):
        if t and t % 12 == 0:
            state = (rng.random(n)[:, None] > np.cumsum(A[state], axis=1)).sum(axis=1).clip(0, 2)
        infl[:, t] = rates[state]
    return np.cumprod(1 + infl / 12, axis=1)


def major_expenses(n: int, horizon: int, seed: int, p: dict = MAJOR_EXPENSE) -> np.ndarray:
    """[n, T] one-off costs: each month with prob 1-(1-p_year)^(1/12) (combined-engine events.py)."""
    pm = 1 - (1 - p["annual_prob"]) ** (1 / 12)
    return (np.random.default_rng([seed, 0xE7E7]).random((n, horizon)) < pm) * p["cost_chf"]
