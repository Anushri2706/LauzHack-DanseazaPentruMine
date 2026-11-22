"""The learned engine: how people's monthly cash actually moves, learned from the training people.

For a person in life state s with income I_t and rent R in calendar month m:

    flow_t = I_t - spend_s(I_t, R)                 learned spending (GMM regression per data status)
           + a_s[m] * I_t + c_s[m] * R             learned seasons (December bonus, holiday months)
           + offset                                learned from the person's own history (shrunk, kappa)
           + e_t,   e_t = phi_s e_{t-1} + sd_s(I) z   learned month-to-month noise (AR(1), sd = alpha + beta I)

Everything is estimated on the TRAINING people only; kappa (how much to trust a person's own history) and
the noise scale are chosen on the VALIDATION people (learned/backtest.py).

Life states come from the regime HMM. HMM state -> data status: employed -> employed (self_employed when
the person is self-employed), part_time -> part_time, job_search -> unemployed, student -> student,
retired -> retired. Spending learned from people already in a status is applied when the HMM moves a person
into it (features are clamped to the range seen for that status).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .gmmreg import GMMRegressor

DATA_STATUSES = ("employed", "part_time", "self_employed", "student", "unemployed", "retired")
HMM_STATES = ("employed", "part_time", "job_search", "student", "retired")
HMM_TO_STATUS = {"employed": "employed", "part_time": "part_time", "job_search": "unemployed",
                 "student": "student", "retired": "retired"}


def status_of_state(state_idx: np.ndarray, self_employed: bool) -> np.ndarray:
    names = np.array([HMM_TO_STATUS[s] for s in HMM_STATES], dtype=object)
    out = names[state_idx]
    if self_employed:
        out = np.where(out == "employed", "self_employed", out)
    return out


def _x(income, rent):
    return np.stack(np.broadcast_arrays(np.log1p(np.maximum(rent, 0)), np.log1p(np.maximum(income, 0))), -1)


@dataclass
class Behaviour:
    n_components: int = 3
    covariance_type: str = "full"
    kappa: float = 12.0          # history weight = n / (n + kappa)
    noise_scale: float = 1.0
    spend: dict = field(default_factory=dict)
    season_a: dict = field(default_factory=dict)   # status -> [13] (index = month 1..12)
    season_c: dict = field(default_factory=dict)
    sd_ab: dict = field(default_factory=dict)      # status -> (alpha, beta)
    phi: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ fit (training people)
    def fit(self, people, flows: np.ndarray, moy: np.ndarray) -> "Behaviour":
        status = np.asarray(people.status, dtype=object)
        income, rent = people.income.values, people.rent.values
        for s in DATA_STATUSES:
            m = status == s
            F, I, R = flows[m], income[m], rent[m]
            mu = F.mean(1)
            self.spend[s] = GMMRegressor(self.n_components, self.covariance_type).fit(_x(I, R), I - mu)
            dev = F - mu[:, None]
            a, c = np.zeros(13), np.zeros(13)
            for month in range(1, 13):
                cols = moy == month
                y = dev[:, cols].ravel()
                A = np.stack([np.repeat(I, cols.sum()), np.repeat(R, cols.sum())], 1)
                a[month], c[month] = np.linalg.lstsq(A, y, rcond=None)[0]
            self.season_a[s], self.season_c[s] = a, c
            e = dev - (a[moy][None, :] * I[:, None] + c[moy][None, :] * R[:, None])
            sd = e.std(1)
            beta, alpha = np.polyfit(I, sd, 1) if len(I) > 2 else (0.0, float(sd.mean()))
            self.sd_ab[s] = (float(alpha), float(beta))
            self.phi[s] = float(np.clip(np.corrcoef(e[:, :-1].ravel(), e[:, 1:].ravel())[0, 1], -0.5, 0.5))
        return self

    # ------------------------------------------------------------------ pieces
    def spending(self, status: np.ndarray, income: np.ndarray, rent) -> np.ndarray:
        out = np.empty(np.shape(income))
        X = _x(income, rent)
        for s in DATA_STATUSES:
            m = status == s
            if m.any():
                out[m] = self.spend[s].predict(X[m])
        return out

    def mean_flow(self, status: np.ndarray, income: np.ndarray, rent) -> np.ndarray:
        return income - self.spending(status, income, rent)

    def season(self, status: np.ndarray, month: np.ndarray, income: np.ndarray, rent) -> np.ndarray:
        out = np.zeros(np.shape(income))
        rent = np.broadcast_to(rent, np.shape(income))
        for s in DATA_STATUSES:
            m = status == s
            if m.any():
                out[m] = self.season_a[s][month[m]] * income[m] + self.season_c[s][month[m]] * rent[m]
        return out

    def noise_sd(self, status: np.ndarray, income: np.ndarray) -> np.ndarray:
        out = np.empty(np.shape(income))
        for s in DATA_STATUSES:
            m = status == s
            if m.any():
                a, b = self.sd_ab[s]
                out[m] = np.maximum(a + b * income[m], 80.0)
        return out * self.noise_scale

    def history_offset(self, status: str, income: float, rent: float, history: np.ndarray, months: np.ndarray):
        """(offset, last residual): how this person differs from people like them, shrunk by kappa."""
        st = np.array([status] * len(history), dtype=object)
        inc = np.full(len(history), income)
        expected = self.mean_flow(st, inc, rent) + self.season(st, months, inc, rent)
        resid = history - expected
        w = len(history) / (len(history) + self.kappa) if np.isfinite(self.kappa) else 0.0
        offset = w * resid.mean()
        return float(offset), float(resid[-1] - offset)

    # ------------------------------------------------------------------ simulate
    def flows(self, states: np.ndarray, income: np.ndarray, rent, first_month: int, self_employed: bool,
              rng: np.random.Generator, offset: float = 0.0, home_state: int | None = None,
              e_last: float = 0.0, keep: float = 1.0, price: np.ndarray | None = None,
              events: np.ndarray | None = None) -> np.ndarray:
        """Monthly flows [n, T] for HMM state paths and incomes (rent: scalar or [n, T]).
        `offset` applies while in `home_state`."""
        n, T = states.shape
        rent = np.broadcast_to(np.asarray(rent, dtype=float), (n, T))
        status = status_of_state(states, self_employed)
        months = ((first_month - 1 + np.arange(T)) % 12 + 1)[None, :].repeat(n, 0)
        # Optional layers from feature/combined-engine's wealth layer, each kept only if it helped on validation
        # (learned/backtest.py): `price` scales learned spending (inflation regimes), `keep` keeps that share of
        # surplus months, `events` are independent one-off costs.
        spend = self.spending(status, income, rent)
        if price is not None:
            spend = spend * price
        mean = income - spend + self.season(status, months, income, rent)
        if home_state is not None:
            mean = mean + np.where(states == home_state, offset, 0.0)
        sd = self.noise_sd(status, income)
        phi = np.vectorize(self.phi.get)(status) if n * T else np.zeros((n, T))
        z = rng.standard_normal((n, T))
        e = np.empty((n, T))
        prev = np.full(n, e_last)
        for t in range(T):
            prev = phi[:, t] * prev + sd[:, t] * np.sqrt(1 - phi[:, t] ** 2) * z[:, t]
            e[:, t] = prev
        out = mean + e
        if keep != 1.0:
            out = np.where(out > 0, keep * out, out)
        if events is not None:
            out = out - events
        return out
