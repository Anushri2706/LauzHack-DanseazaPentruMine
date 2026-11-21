"""Transition priors: an editable, sourced table of monthly hazards plus Dirichlet uncertainty.

Transitions are assumptions, not learned: the synthetic ledger contains no regime switches
(see docs/hmm-digital-twin.md). Each off-diagonal cell of A is a named assumption that
sliders can override by ``assumption_name``.

Rows with ``age_min``/``age_max`` replace that cell while ``floor(age)`` is inside the
band. They do not overwrite the all-ages hazard used when ``age`` is omitted. Overriding
the all-ages name scales every band in proportion. Overriding the band's own name sets
that band's hazard directly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ASSUMPTIONS_DIR
from .types import K, STATE_INDEX, STATES, Intervention


@dataclass
class AgeHazard:
    name: str
    i: int
    j: int
    lo: float
    hi: float
    hazard: float


@dataclass
class TransitionPriors:
    hazards: np.ndarray                      # [K, K] all-ages off-diagonal monthly hazards
    concentration: np.ndarray                # [K] Dirichlet total concentration per row
    names: dict[str, tuple[int, int]]        # all-ages assumption_name -> (i, j)
    age_bounds: dict[tuple[int, int], tuple[float, float]] = field(default_factory=dict)
    table: pd.DataFrame | None = None
    age_hazards: list[AgeHazard] = field(default_factory=list)
    age_names: dict[str, AgeHazard] = field(default_factory=dict)

    def age_hazard(self, name: str) -> float:
        return float(self.age_names[name].hazard)

    def _base_hazards(self, overrides: dict | None) -> np.ndarray:
        h = self.hazards.copy()
        for name, value in (overrides or {}).items():
            if name in self.names:
                h[self.names[name]] = float(value)
        return h

    def matrix(self, overrides: dict | None = None, age: float | None = None,
               interventions: list[Intervention] | None = None, t: int | None = None) -> np.ndarray:
        """Mean monthly transition matrix A (rows sum to 1) after overrides, age and hazard scalings."""
        h = self._base_hazards(overrides)
        if age is not None:
            year = math.floor(age)
            for ah in self.age_hazards:
                if not (ah.lo <= year <= ah.hi):
                    continue
                if overrides and ah.name in overrides:
                    h[ah.i, ah.j] = float(overrides[ah.name])
                elif self.hazards[ah.i, ah.j] > 0:
                    h[ah.i, ah.j] = ah.hazard * (h[ah.i, ah.j] / self.hazards[ah.i, ah.j])
                else:
                    h[ah.i, ah.j] = ah.hazard
            for (i, j), (lo, hi) in self.age_bounds.items():
                if not (lo <= age <= hi):
                    h[i, j] = 0.0
        for iv in interventions or []:
            if iv.kind == "scale_hazard" and iv.transition and iv.factor is not None and (t is None or iv.active(t)):
                a, b = iv.transition
                h[STATE_INDEX[a], STATE_INDEX[b]] *= float(iv.factor)
        np.fill_diagonal(h, 0.0)
        row = h.sum(axis=1)
        if np.any(row > 1.0 + 1e-9):
            h = h / np.maximum(row, 1e-12)[:, None]
            row = h.sum(axis=1)
        return h + np.diag(1.0 - row)

    def sample(self, rng: np.random.Generator, n: int, overrides: dict | None = None) -> np.ndarray:
        """Draw n transition matrices from row-wise Dirichlet(kappa_i * A_i). Returns [n, K, K].

        The draw is around the all-ages matrix. The sampler rescales it by age at each month.
        """
        A = self.matrix(overrides)
        out = np.empty((n, K, K))
        eps = 1e-6
        for i in range(K):
            alpha = np.maximum(self.concentration[i] * A[i], eps)
            out[:, i, :] = rng.dirichlet(alpha, size=n)
            zero = A[i] <= 0.0
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
    age_hazards: list[AgeHazard] = []
    age_names: dict[str, AgeHazard] = {}
    for r in df.itertuples(index=False):
        i, j = STATE_INDEX[r.from_state], STATE_INDEX[r.to_state]
        concentration[i] = float(r.row_concentration)
        if i == j:
            continue
        has_age = not (pd.isna(r.age_min) and pd.isna(r.age_max))
        if has_age:
            lo = -np.inf if pd.isna(r.age_min) else float(r.age_min)
            hi = np.inf if pd.isna(r.age_max) else float(r.age_max)
            # A band is an age-specific level when the same cell also has an all-ages row.
            # A band on its own (no all-ages row yet, and none later) is recorded after the pass.
            ah = AgeHazard(str(r.assumption_name), i, j, lo, hi, float(r.monthly_hazard))
            age_hazards.append(ah)
            age_names[ah.name] = ah
            continue
        hazards[i, j] = float(r.monthly_hazard)
        names[str(r.assumption_name)] = (i, j)
    # A band whose cell has no all-ages hazard is a window: the hazard applies inside the
    # ages and is zero outside. Fold it into age_bounds plus the base cell.
    for ah in list(age_hazards):
        if (ah.i, ah.j) not in {v for v in names.values()}:
            hazards[ah.i, ah.j] = ah.hazard
            names[ah.name] = (ah.i, ah.j)
            age_bounds[(ah.i, ah.j)] = (ah.lo, ah.hi)
            age_hazards.remove(ah)
            age_names.pop(ah.name, None)
    return TransitionPriors(hazards, concentration, names, age_bounds, df, age_hazards, age_names)


def describe(priors: TransitionPriors) -> list[dict]:
    """Assumption rows for the UI: name, from, to, hazard, implied mean dwell."""
    rows = []
    for name, (i, j) in priors.names.items():
        h = float(priors.hazards[i, j])
        rows.append({"name": name, "from": STATES[i], "to": STATES[j], "monthly_hazard": h,
                     "mean_months_until": (1.0 / h) if h > 0 else None, "age": None})
    for ah in priors.age_hazards:
        rows.append({"name": ah.name, "from": STATES[ah.i], "to": STATES[ah.j],
                     "monthly_hazard": ah.hazard, "mean_months_until": (1.0 / ah.hazard) if ah.hazard > 0 else None,
                     "age": [ah.lo, ah.hi]})
    return rows
