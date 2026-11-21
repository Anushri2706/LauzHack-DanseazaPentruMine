"""Wealth layer: from the regime engine's net cash flow to real net worth.

The regime engine produces monthly net income and spending along sampled life regimes. This layer adds
what it does not model, as separate named flows so cash still reconciles:

  kept flow   = keep x net flow in surplus months, net flow in deficit months   (saving calibration)
  deliberate  = (expense cut x spending + side income) x price level            (levers, kept in full)
  spending    = engine spending x price level  (engine wages already grow in nominal terms)
  one-offs    = cost in today's CHF x price level
  cash        = opening cash + cumsum(kept flow + deliberate - one-offs - loan interest)
  portfolio   = existing investments x random monthly return
  net worth   = cash + portfolio - debt; "real" = divided by the price level (today's CHF)

Debt is interest-only (principal constant). Negative cash is never clamped: it is a financing need.
Inflation regimes and returns come from assumptions/market_assumptions.csv, drawn from ``seed`` so every
scenario of one person shares the same futures. Off by default in TwinSession (wealth=False).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cash_engine import OneOff
from .types import Profile, RegimePaths


def inflation_matrix(params: dict) -> np.ndarray:
    p = params
    return np.array([
        [p["inflation_stay_low"], p["inflation_low_to_base"], 1 - p["inflation_stay_low"] - p["inflation_low_to_base"]],
        [p["inflation_base_to_low"], 1 - p["inflation_base_to_low"] - p["inflation_base_to_high"], p["inflation_base_to_high"]],
        [p["inflation_high_to_low"], p["inflation_high_to_base"], 1 - p["inflation_high_to_low"] - p["inflation_high_to_base"]],
    ])


def sample_market(params: dict, n: int, horizon: int, seed: int, inflation: bool = True):
    """(price level [n, T], monthly portfolio return [n, T]). Yearly inflation regime switches (low/base/high)."""
    rng = np.random.default_rng(10_000 + seed)
    rates = np.array([params["inflation_low"], params["inflation_base"], params["inflation_high"]])
    A = inflation_matrix(params)
    state = np.full(n, 1)
    infl = np.empty((n, horizon))
    for t in range(horizon):
        if t and t % 12 == 0:
            u = rng.random(n)
            cum = np.cumsum(A[state], axis=1)
            state = (u[:, None] > cum).sum(axis=1).clip(0, 2)
        infl[:, t] = rates[state]
    if not inflation:
        infl[:] = 0.0
    ret = rng.normal(params["portfolio_return"] / 12, params["portfolio_vol"] / np.sqrt(12), (n, horizon))
    return np.cumprod(1 + infl / 12, axis=1), ret


@dataclass
class Wealth:
    """One scenario, many futures. Any keep factor or lever can be scored without re-simulating."""
    income: np.ndarray        # [n, T] net income, nominal CHF
    spend: np.ndarray         # [n, T] engine spending, today's CHF
    oneoff: np.ndarray        # [T] one-off costs, today's CHF
    price: np.ndarray         # [n, T] price level
    port: np.ndarray          # [n, T] portfolio value, nominal
    opening: float
    port0: float
    debt: float
    loan_interest: float      # CHF/month
    states: np.ndarray        # [n, T] regime index
    keep: float = 1.0         # default keep factor for this person
    buffer_months: float = 3.0

    def _k(self, keep):
        return self.keep if keep is None else keep

    def net(self):
        """Engine net flow with spending inflated, before calibration and levers, nominal CHF."""
        return self.income - self.spend * self.price

    def flows(self, keep=None, cut=0.0, side=0.0):
        net = self.net()
        kept = np.where(net > 0, self._k(keep) * net, net)
        deliberate = (cut * self.spend + side) * self.price
        return kept + deliberate - self.oneoff[None, :] * self.price - self.loan_interest

    def cash(self, keep=None, cut=0.0, side=0.0):
        return self.opening + np.cumsum(self.flows(keep, cut, side), axis=1)

    def net_worth_years(self, keep=None, cut=0.0, side=0.0):
        """[n, years+1] real net worth at the end of each year; column 0 = today."""
        nw = (self.cash(keep, cut, side) + self.port - self.debt) / self.price
        start = np.full((nw.shape[0], 1), self.opening + self.port0 - self.debt)
        return np.hstack([start, nw[:, 11::12]])

    def wealth(self, keep=None, year=None, cut=0.0, side=0.0):
        """[n] real net worth after `year` years (default: the horizon)."""
        y = self.net_worth_years(keep, cut, side)
        return y[:, -1 if year is None else year]

    def months_below_buffer(self, keep=None, cut=0.0, side=0.0):
        """[n] months with cash below buffer_months of (lever-reduced) spending."""
        buf = self.buffer_months * (1 - cut) * self.spend * self.price
        return (self.cash(keep, cut, side) < buf).sum(axis=1)

    def sacrifice(self, cut=0.0, side=0.0):
        """CHF/month given up on purpose today: expense cut on month-0 spending + side income."""
        return float(cut * np.median(self.spend[:, 0]) + side)

    def summary(self, keep=None, cut=0.0, side=0.0) -> dict:
        y = self.net_worth_years(keep, cut, side)
        cash = self.cash(keep, cut, side)
        neg = cash < 0
        q = lambda a, p: float(np.quantile(a, p))
        return {
            "net_worth_real_chf": {"p10": q(y[:, -1], .1), "p50": q(y[:, -1], .5), "p90": q(y[:, -1], .9)},
            "net_worth_real_by_year": [{"year": t, "p10": q(y[:, t], .1), "p50": q(y[:, t], .5),
                                        "p90": q(y[:, t], .9)} for t in range(y.shape[1])],
            "share_paths_cash_negative": float(neg.any(axis=1).mean()),
            "median_months_cash_negative": float(np.median(neg.sum(axis=1))),
            "p90_peak_shortfall_chf": q(np.maximum(-cash.min(axis=1), 0.0), .9),
            "median_months_below_buffer": float(np.median(self.months_below_buffer(keep, cut, side))),
            "keep_factor": self._k(keep),
        }


def build(profile: Profile, paths: RegimePaths, one_offs: list[OneOff] | None, params: dict, seed: int,
          keep: float = 1.0, inflation: bool = True) -> Wealth:
    n, T = paths.income_chf.shape
    oneoff = np.zeros(T)
    for o in one_offs or []:
        if 0 <= o.month < T:
            oneoff[o.month] += o.amount_chf
    price, ret = sample_market(params, n, T, seed, inflation)
    port0 = float(profile.portfolio_chf)
    port = port0 * np.cumprod(1 + ret, axis=1)
    loans = list(profile.loans or [])
    return Wealth(paths.income_chf, paths.income_chf - paths.net_flow_chf, oneoff, price, port,
                  float(profile.opening_cash), port0, float(sum(a for a, _ in loans)),
                  float(sum(a * r for a, r in loans) / 12), paths.states, float(keep),
                  float(params.get("buffer_months", 3)))
