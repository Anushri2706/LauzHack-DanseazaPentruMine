"""Transition priors: an editable, sourced table of monthly hazards plus Dirichlet uncertainty.

Transitions are assumptions, not learned: the synthetic ledger contains no regime switches
(see docs/hmm-digital-twin.md §1). Each off-diagonal cell of A is a named assumption that
sliders / the LLM can override by ``assumption_name``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ASSUMPTIONS_DIR
from .types import K, STATE_INDEX, STATES, Intervention


@dataclass
class TransitionPriors:
    hazards: np.ndarray                      # [K, K] off-diagonal monthly hazards; diagonal = 1 - row sum
    concentration: np.ndarray                # [K] Dirichlet total concentration per row (higher = more certain)
    names: dict[str, tuple[int, int]]        # assumption_name -> (i, j)
    age_bounds: dict[tuple[int, int], tuple[float, float]] = field(default_factory=dict)
    table: pd.DataFrame | None = None        # raw CSV for the assumptions panel

    def matrix(self, overrides: dict | None = None, age: float | None = None,
               interventions: list[Intervention] | None = None, t: int | None = None) -> np.ndarray:
        """Mean monthly transition matrix A (rows sum to 1) after overrides and active hazard scalings."""
        h = self.hazards.copy()
        for name, value in (overrides or {}).items():
            if name in self.names:
                h[self.names[name]] = float(value)
        if age is not None:
            for (i, j), (lo, hi) in self.age_bounds.items():
                if not (lo <= age <= hi):
                    h[i, j] = 0.0
        for iv in interventions or []:
            if iv.kind == "scale_hazard" and (t is None or iv.active(t)):
                a, b = iv.transition
                h[STATE_INDEX[a], STATE_INDEX[b]] *= float(iv.factor)
        np.fill_diagonal(h, 0.0)
        row = h.sum(axis=1)
        if np.any(row > 1.0 + 1e-12):
            raise ValueError("Hazards in a row sum to more than 1")
        A = h + np.diag(1.0 - row)
        return A

    def sample(self, rng: np.random.Generator, n: int, overrides: dict | None = None) -> np.ndarray:
        """Draw n transition matrices from row-wise Dirichlet(kappa_i * A_i). Returns [n, K, K]."""
        A = self.matrix(overrides)
        out = np.empty((n, K, K))
        eps = 1e-6
        for i in range(K):
            alpha = np.maximum(self.concentration[i] * A[i], eps)
            out[:, i, :] = rng.dirichlet(alpha, size=n)
            # Keep structural zeros exactly zero (e.g. absorbing retired, scheduled graduation).
            zero = A[i] == 0.0
            if zero.any():
                out[:, i, zero] = 0.0
                out[:, i, :] /= out[:, i, :].sum(axis=1, keepdims=True)
        return out


def load_transition_priors(path: str | Path = ASSUMPTIONS_DIR / "transitions.csv") -> TransitionPriors:
    df = pd.read_csv(path)
    hazards = np.zeros((K, K))
    concentration = np.full(K, 100.0)
    names: dict[str, tuple[int, int]] = {}
    age_bounds: dict[tuple[int, int], tuple[float, float]] = {}
    for r in df.itertuples(index=False):
        i, j = STATE_INDEX[r.from_state], STATE_INDEX[r.to_state]
        concentration[i] = float(r.row_concentration)
        if i == j:
            continue
        hazards[i, j] = float(r.monthly_hazard)
        names[str(r.assumption_name)] = (i, j)
        if not (pd.isna(r.age_min) and pd.isna(r.age_max)):
            lo = -np.inf if pd.isna(r.age_min) else float(r.age_min)
            hi = np.inf if pd.isna(r.age_max) else float(r.age_max)
            age_bounds[(i, j)] = (lo, hi)
    return TransitionPriors(hazards, concentration, names, age_bounds, df)


def describe(priors: TransitionPriors) -> list[dict]:
    """Assumption rows for the UI / LLM: name, from, to, hazard, implied mean dwell, source."""
    rows = []
    for name, (i, j) in priors.names.items():
        h = priors.hazards[i, j]
        rows.append({"name": name, "from": STATES[i], "to": STATES[j], "monthly_hazard": float(h),
                     "mean_months_until": (1.0 / h) if h > 0 else None})
    return rows
