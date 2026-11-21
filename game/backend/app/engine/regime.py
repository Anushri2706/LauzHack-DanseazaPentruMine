"""The regime HMM as the backbone of a life, stepped one month at a time inside the game.

Each month of a life is one regime of the teammate's chain (repo-root `backend/twin`):
`employed`, `part_time`, `job_search`, `student` or `retired`. The monthly transition matrix comes
from `backend/assumptions/transitions.csv` (sourced, age-banded hazards: Eurostat lfsi_long_q, BFS
SAKE, SECO), with Dirichlet parameter uncertainty: every world carries its own draw of the matrix.

Conditioned on the player's life (this is what the old decision-independent masks could not do):
  * the chain runs from the player's CURRENT regime and age, month by month;
  * decisions pin regimes: studying = `student`, retired = `retired`, quitting = `job_search`;
  * a master's degree scales the job-loss hazard by `masters_hazard_factor` (BFS EHA);
  * job-search exits halve after `long_term_unemployment_months` (duration dependence);
  * at `retirement_age` everyone still in the labour market retires (AHV 21 age rule).

Common random numbers: world k holds one uniform per month (`regime_u`) and one matrix draw. Every
reality replays the same uniforms, so two realities differ because of choices, not luck. World k
is identical whether 1 or 500 worlds are drawn.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

STATES = ("employed", "part_time", "job_search", "student", "retired")
IDX = {s: i for i, s in enumerate(STATES)}
E, PT, J, ST, R = (IDX[s] for s in STATES)
LABOUR = (E, PT, J)


@lru_cache(maxsize=1)
def _model():
    from .. import hmm  # lazy: puts backend/ on sys.path and fits the engine once
    eng = hmm.engine()
    params = eng.income_table.params
    return eng.priors, {
        "masters_hazard_factor": float(params.get("masters_hazard_factor", 1.0)),
        "long_term_multiplier": float(params.get("long_term_unemployment_multiplier", 0.5)),
        "long_term_months": float(params.get("long_term_unemployment_months", 12)),
    }


@lru_cache(maxsize=256)
def _scale(age_year: int) -> np.ndarray:
    """Age rescaling of the all-ages hazards (target / base off-diagonal), as in sample_regime_paths."""
    priors, _ = _model()
    base = priors.matrix()
    target = priors.matrix(age=float(age_year))
    with np.errstate(divide="ignore", invalid="ignore"):
        sc = np.where(base > 1e-12, target / np.maximum(base, 1e-12), 0.0)
    np.fill_diagonal(sc, 0.0)
    return sc


def world_draw(seed: int, k: int, horizon: int) -> dict:
    """Regime luck of world k: one Dirichlet draw of the matrix and one uniform per month."""
    priors, _ = _model()
    rng = np.random.default_rng([int(seed), int(k), 0x2E61])
    A = priors.sample(rng, 1)[0]
    return {"regime_A": ((0, A),), "regime_u": rng.random(horizon)}


def matrix_at(world: dict, t: int) -> np.ndarray | None:
    segs = world.get("regime_A")
    if not segs:
        return None
    A = None
    for start, M in segs:
        if start <= t:
            A = M
    return A


def step(prev: int, age: float, has_master: bool, dwell: int, A: np.ndarray, u: float) -> int:
    """Next regime from `prev` with the world's matrix draw, rescaled to this age."""
    _, k = _model()
    row = A[prev] * _scale(int(age))[prev]
    if prev == E and has_master:
        row[J] *= k["masters_hazard_factor"]
    if prev == J and dwell >= k["long_term_months"]:
        row = row * k["long_term_multiplier"]
    row[prev] = 0.0
    off = row.sum()
    if off > 1.0:
        row, off = row / off, 1.0
    row[prev] = 1.0 - off
    return int(np.searchsorted(np.cumsum(row), u, side="left").clip(0, len(STATES) - 1))


def regime_for_status(status: str) -> int:
    return {"studying": ST, "retired": R, "unemployed": J}.get(status, E)


def label(idx: int, work_share: float) -> str:
    """Displayed regime: hours are a player choice, so the chain's part_time shows as employed
    unless the player works reduced hours."""
    if idx in (E, PT):
        return "part_time" if work_share < 1 else "employed"
    return STATES[idx]
