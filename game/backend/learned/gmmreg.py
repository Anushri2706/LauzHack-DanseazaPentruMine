"""Gaussian mixture regression, vectorised: fit a GMM on [x, y], predict y | x for many x at once."""
from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture


class GMMRegressor:
    def __init__(self, n_components: int = 3, covariance_type: str = "full", reg_covar: float = 1e-3, seed: int = 0):
        self.n_components, self.covariance_type, self.reg_covar, self.seed = n_components, covariance_type, reg_covar, seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GMMRegressor":
        self.x_mean, self.x_std = X.mean(0), X.std(0) + 1e-9
        self.y_mean, self.y_std = float(y.mean()), float(y.std() + 1e-9)
        self.x_lo, self.x_hi = X.min(0), X.max(0)       # predictions clamp x to the range seen in training
        Z = np.hstack([(X - self.x_mean) / self.x_std, ((y - self.y_mean) / self.y_std)[:, None]])
        c = max(1, min(self.n_components, len(X) // 10))
        self.gm = GaussianMixture(c, covariance_type=self.covariance_type, reg_covar=self.reg_covar,
                                  n_init=3, random_state=self.seed).fit(Z)
        d = X.shape[1]
        covs = self.gm.covariances_
        if self.covariance_type == "diag":
            covs = np.array([np.diag(v) for v in covs])
        self._parts = []
        for j in range(self.gm.n_components):
            S = covs[j]
            inv = np.linalg.inv(S[:d, :d])
            beta = S[d, :d] @ inv
            self._parts.append((np.log(self.gm.weights_[j]) - 0.5 * np.linalg.slogdet(S[:d, :d])[1],
                                self.gm.means_[j][:d], inv, self.gm.means_[j][d], beta,
                                max(S[d, d] - beta @ S[:d, d], 1e-9)))
        return self

    def predict(self, X: np.ndarray, return_std: bool = False):
        """Conditional mean (and std) of y given each row of X. X may be any shape [..., d]."""
        shape = X.shape[:-1]
        Xz = (np.clip(X.reshape(-1, X.shape[-1]), self.x_lo, self.x_hi) - self.x_mean) / self.x_std
        logw, means, vars_ = [], [], []
        for logc, mx, inv, my, beta, v in self._parts:
            diff = Xz - mx
            logw.append(logc - 0.5 * np.einsum("ni,ij,nj->n", diff, inv, diff))
            means.append(my + diff @ beta)
            vars_.append(np.full(len(Xz), v))
        logw = np.array(logw)
        w = np.exp(logw - logw.max(0))
        w /= w.sum(0)
        m, v = np.array(means), np.array(vars_)
        mean = (w * m).sum(0)
        out = (mean * self.y_std + self.y_mean).reshape(shape)
        if not return_std:
            return out
        var = (w * (v + m ** 2)).sum(0) - mean ** 2
        return out, (np.sqrt(np.maximum(var, 0)) * self.y_std).reshape(shape)
