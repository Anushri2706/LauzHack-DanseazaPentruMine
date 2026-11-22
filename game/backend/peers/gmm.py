"""People like you, as a Gaussian mixture: Gaussian mixture regression per life decision (job status).

For each job status, a Gaussian mixture is fitted on the joint space [normalised profile features, monthly
net flow] of the training people. Each component is a "kind of person". Given a new profile x, the
prediction for the flow y is the conditional mixture

    p(y | x) = sum_j w_j(x) N(y; m_j(x), v_j)
    w_j(x)  ~ pi_j N(x; mu_x_j, S_xx_j)                      (how much x looks like component j)
    m_j(x)  = mu_y_j + S_yx_j S_xx_j^-1 (x - mu_x_j)          (that component's linear trend)
    v_j     = S_yy_j - S_yx_j S_xx_j^-1 S_xy_j

The point prediction is the mixture mean; P10 / P90 come from the mixture's distribution, so the range is
a real predictive distribution, not just the spread of a few neighbours.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

from .model import Scaler

FEATURE_SETS = {
    "income": ("income",),
    "income+rent": ("income", "rent", "rent_share"),
    "income+rent+age": ("income", "rent", "rent_share", "age"),
    "all": ("age", "income", "rent", "rent_share", "cash", "inv", "debt"),
}


def features(scaler: Scaler, df: pd.DataFrame, names: tuple[str, ...]) -> np.ndarray:
    r = scaler.raw(df)
    return np.stack([(r[k] - scaler.mean[k]) / scaler.std[k] for k in names], 1)


def _mixture_quantile(w: np.ndarray, m: np.ndarray, s: np.ndarray, q: float) -> float:
    lo, hi = float((m - 6 * s).min()), float((m + 6 * s).max())
    for _ in range(60):  # bisection on the mixture CDF
        mid = (lo + hi) / 2
        if (w * norm.cdf((mid - m) / s)).sum() < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@dataclass
class GMMModel:
    feature_set: str
    n_components: int
    covariance_type: str          # "full" (each component has its own trend) or "diag" (soft clusters)
    reg_covar: float = 1e-3
    fits: dict = field(default_factory=dict)       # status -> fitted GaussianMixture
    y_mean: float = 0.0
    y_std: float = 1.0

    def fit(self, scaler: Scaler, train: pd.DataFrame) -> "GMMModel":
        names = FEATURE_SETS[self.feature_set]
        self.y_mean, self.y_std = float(train.flow.mean()), float(train.flow.std())
        for status, g in train.groupby("status"):
            Z = np.hstack([features(scaler, g, names), ((g.flow.values - self.y_mean) / self.y_std)[:, None]])
            c = max(1, min(self.n_components, len(g) // 10))   # small groups get fewer components
            self.fits[status] = GaussianMixture(c, covariance_type=self.covariance_type, reg_covar=self.reg_covar,
                                                n_init=3, random_state=0).fit(Z)
        return self

    def predict(self, scaler: Scaler, query: pd.DataFrame) -> pd.DataFrame:
        names = FEATURE_SETS[self.feature_set]
        X = features(scaler, query, names)
        d = X.shape[1]
        rows = []
        for x, status in zip(X, np.asarray(query.status, dtype=object)):
            gm = self.fits[status]
            covs = gm.covariances_
            if self.covariance_type == "diag":
                covs = np.array([np.diag(c) for c in covs])
            logw, means, sds = [], [], []
            for j in range(gm.n_components):
                mu, S = gm.means_[j], covs[j]
                Sxx, Sxy, Syy = S[:d, :d], S[:d, d], S[d, d]
                inv = np.linalg.inv(Sxx)
                diff = x - mu[:d]
                _, logdet = np.linalg.slogdet(Sxx)
                logw.append(np.log(gm.weights_[j]) - 0.5 * (diff @ inv @ diff + logdet))
                means.append(mu[d] + Sxy @ inv @ diff)
                sds.append(np.sqrt(max(Syy - Sxy @ inv @ Sxy, 1e-9)))
            logw = np.array(logw)
            w = np.exp(logw - logw.max())
            w /= w.sum()
            m, s = np.array(means), np.array(sds)
            to_chf = lambda v: v * self.y_std + self.y_mean  # noqa: E731
            rows.append({"pred": to_chf(float((w * m).sum())),
                         "p10": to_chf(_mixture_quantile(w, m, s, 0.10)),
                         "p90": to_chf(_mixture_quantile(w, m, s, 0.90)),
                         "components": int(gm.n_components)})
        return pd.DataFrame(rows, index=query.index)
