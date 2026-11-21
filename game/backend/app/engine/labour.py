"""Labour-market luck from the regime HMM (repo-root `backend/twin`).

The Monte Carlo worlds need one thing from the labour market: in which months would a working
person be out of a job? The flat model draws that from `job_loss_prob_annual`. This module
draws it from the teammate's regime chain instead, whose monthly hazards are sourced
(Eurostat/BFS/SECO, see backend/assumptions/transitions.csv): age-banded job-loss hazards,
age-banded exit rates, long-term-unemployment duration dependence, Dirichlet parameter
uncertainty on the transition matrix.

Only the regime STATES are used. Income, spending and cash stay in our engine, so there is one
accounting (net salaries, investments, debt, inflation) and no gross/net mix-up.

Scenario isolation: the chain starts in `employed` with no interventions, so a world's spells
depend only on (start age, horizon, seed, world index), never on the player's decisions. Every
branch replays the same spells; they only bite while the player is working or abroad.

Paths are sampled in fixed blocks of BLOCK worlds, each with its own seed, so world k is the
same whether 1, 300 or 500 worlds are requested.

Limitations: decision-dependent hazards (e.g. the master's lowers job-loss risk by
masters_hazard_factor) are not applied, because that would break scenario isolation.
Part-time months in the chain count as full pay: in the game, hours are a player choice.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

BLOCK = 100
JOB_SEARCH = "job_search"


@lru_cache(maxsize=1)
def _engine():
    from .. import hmm  # lazy: puts backend/ on sys.path and fits the engine once
    return hmm.engine()


@lru_cache(maxsize=256)
def _block(start_age: int, horizon: int, seed: int, block: int) -> np.ndarray:
    """[BLOCK, horizon] bool: True where the chain is in job search."""
    engine = _engine()  # first: this import puts backend/ on sys.path
    from twin.types import STATE_INDEX, Profile
    profile = Profile(age=start_age, employment="employed", ref_income=0.0, rent=0.0)
    sub_seed = int(np.random.SeedSequence([seed, block, 0x1AB0]).generate_state(1)[0])
    paths = engine.simulate(profile, horizon_months=horizon, n_paths=BLOCK, seed=sub_seed)
    return paths.states == STATE_INDEX[JOB_SEARCH]


def job_search_masks(start_age: float, horizon: int, n: int, seed: int) -> np.ndarray:
    """[n, horizon] bool job-search months for worlds 0..n-1 (decision-independent)."""
    blocks = [_block(int(start_age), int(horizon), int(seed), b) for b in range((n + BLOCK - 1) // BLOCK)]
    return np.concatenate(blocks, axis=0)[:n]


def spells(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start month, length) of each run of True in a 1-D mask."""
    out, t, T = [], 0, len(mask)
    while t < T:
        if mask[t]:
            s = t
            while t < T and mask[t]:
                t += 1
            out.append((s, t - s))
        else:
            t += 1
    return out
