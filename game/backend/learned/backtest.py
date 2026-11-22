"""Backtest the combined simulator on the locked people split.

    cd game/backend && .venv/bin/python -m learned.backtest

Task: each person's first 12 months (2025) are known; predict their cash over the next 12 months (2026).
All models are fitted on the 958 TRAINING people. Settings are chosen on the 121 VALIDATION people.
The 121 TEST people (the same locked set as the kNN / GMM models) are evaluated once, at the end.

Models
  persistence        cash keeps moving by the person's own 2025 average (+- their monthly spread)
  hmm                the teammates' regime HMM on its own (its emissions fitted on training people, its
                     personal savings rate and its filtered starting belief from the 2025 months)
  hmm+gmm            HMM life states and incomes + the learned engine (GMM spending, seasons, noise),
                     no personal history
  hmm+gmm+history    the full combined simulator: + the person's own 2025 months, shrunk (kappa), and the
                     last month's surprise carried over (AR)
Metrics on cash at the end of 2026: mean absolute error (CHF), R2 across people, and how often the true
value falls inside the predicted P10-P90 range (target 80%); also month-by-month range coverage.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np

from .behaviour import Behaviour
from .data import ledger
from .hmm import STATE_INDEX, Intervention, engine, profile_for, run_cash

ART = Path(__file__).resolve().parent / "artifacts"
N_PATHS = 400
SEED = 7
H = 12   # months known / months predicted
COMPONENTS = (1, 2, 3, 4)
KAPPAS = (0.0, 1.0, 3.0, 6.0, 12.0, 24.0, 48.0, float("inf"))
NOISE = (0.6, 0.8, 1.0, 1.2, 1.5)


def _summary(truth_end: np.ndarray, truth_path: np.ndarray, sims: list[np.ndarray]) -> dict:
    """sims[i] = [n_paths, H] predicted cash paths for person i."""
    end = np.array([s[:, -1] for s in sims])
    med, p10, p90 = np.median(end, 1), np.percentile(end, 10, 1), np.percentile(end, 90, 1)
    err = med - truth_end
    ss = ((truth_end - truth_end.mean()) ** 2).sum()
    lo = np.array([np.percentile(s, 10, 0) for s in sims])
    hi = np.array([np.percentile(s, 90, 0) for s in sims])
    return {"mae_cash_end_chf": float(np.abs(err).mean()), "median_abs_error_chf": float(np.median(np.abs(err))),
            "r2_cash_end": float(1 - (err ** 2).sum() / ss),
            "range_hit_rate_p10_p90": float(((truth_end >= p10) & (truth_end <= p90)).mean()),
            "monthly_range_coverage": float(((truth_path >= lo) & (truth_path <= hi)).mean()),
            "mean_range_width_chf": float((p90 - p10).mean()), "n": int(len(truth_end))}


class Case:
    """One person in the backtest: what is known (2025) and what happened (2026)."""

    def __init__(self, L, i: int, eng, start: str = "known", life_events: bool = True):
        self.row = L.people.iloc[i]
        self.hist, self.future = L.flows[i, :H], L.flows[i, H:2 * H]
        self.cash12 = L.opening[i] + self.hist.sum()
        self.truth_path = self.cash12 + np.cumsum(self.future)
        self.months = L.month_of_year[:H]
        age = self.row.age + 1
        if not life_events and self.row.status != "retired":
            age = min(age, 64)   # held state: the HMM's income table pays a pension from 65 even when pinned
        prof = profile_for(self.row, age=age, opening_cash=self.cash12, history=self.hist)
        self.profile = prof
        # "known": the data states the person's life state, so the chain starts there (one-hot).
        # "filtered": the HMM infers it from the 2025 flows (its own filter), as it must when the state is unknown.
        belief = eng.filter(prof, list(self.hist)) if start == "filtered" else None
        # life_events=False holds the person in their known state for the 12 months: the synthetic data never
        # changes anyone's state (nobody retires at 65 or finds a job), so the fair test of the learned money
        # behaviour pins the state. life_events=True is how the game runs the chain.
        pins = [] if life_events else [Intervention("force_state", 0, H, state=prof.initial_state)]
        self.paths = eng.simulate(prof, horizon_months=H, n_paths=N_PATHS, initial_belief=belief,
                                  interventions=pins, seed=SEED + i)
        self.hmm_cash = run_cash(prof, self.paths).cash
        self.home = STATE_INDEX[prof.initial_state]

    def persistence(self, rng) -> np.ndarray:
        m, s = self.hist.mean(), self.hist.std()
        return self.cash12 + np.cumsum(m + s * rng.standard_normal((N_PATHS, H)), 1)

    def twin(self, beh: Behaviour, history: bool, rng) -> np.ndarray:
        offset, e_last = (beh.history_offset(self.row.status, self.row.income, self.row.rent, self.hist, self.months)
                          if history else (0.0, 0.0))
        f = beh.flows(self.paths.states, self.paths.income_chf, self.row.rent, first_month=1,
                      self_employed=self.row.status == "self_employed", rng=rng, offset=offset,
                      home_state=self.home, e_last=e_last)
        return self.cash12 + np.cumsum(f, 1)


def evaluate(cases: list[Case], model: str, beh: Behaviour | None = None) -> dict:
    rng = np.random.default_rng(SEED)
    sims = []
    for c in cases:
        if model == "persistence":
            sims.append(c.persistence(rng))
        elif model == "hmm":
            sims.append(c.hmm_cash)
        else:
            sims.append(c.twin(beh, model == "hmm+gmm+history", rng))
    truth_end = np.array([c.truth_path[-1] for c in cases])
    truth_path = np.array([c.truth_path for c in cases])
    return _summary(truth_end, truth_path, sims)


MODELS = ("persistence", "hmm", "hmm+gmm", "hmm+gmm+history")


def main() -> None:
    t0 = time.time()
    ART.mkdir(exist_ok=True)
    L = ledger()
    tr, va, te = (L.rows(k) for k in ("train", "validation", "test"))
    eng = engine(tuple(L.people.index[tr]))           # HMM emissions from training people only
    val = [Case(L, i, eng, life_events=False) for i in va]
    fitted = {c: Behaviour(n_components=c).fit(L.people.iloc[tr], L.flows[tr], L.month_of_year) for c in COMPONENTS}

    # ---- choose on validation (states held, the fair test): components and kappa by error, noise by calibration
    grid = []
    for c, k in itertools.product(COMPONENTS, KAPPAS):
        b = fitted[c]
        b.kappa, b.noise_scale = k, 1.0
        grid.append({"n_components": c, "kappa": k, **evaluate(val, "hmm+gmm+history", b)})
    grid.sort(key=lambda r: r["mae_cash_end_chf"])
    best = dict(grid[0])
    noise = []
    for s_ in NOISE:
        b = fitted[best["n_components"]]
        b.kappa, b.noise_scale = best["kappa"], s_
        noise.append({"noise_scale": s_, **evaluate(val, "hmm+gmm+history", b)})
    best_noise = min(noise, key=lambda r: (abs(r["range_hit_rate_p10_p90"] - 0.8), r["mae_cash_end_chf"]))
    beh = fitted[best["n_components"]]
    beh.kappa, beh.noise_scale = best["kappa"], best_noise["noise_scale"]

    report = {"task": "known: 12 months of 2025; predicted: cash over the 12 months of 2026",
              "sizes": {"train": len(tr), "validation": len(va), "test": len(te)},
              "chosen_on_validation": {"n_components": best["n_components"], "kappa": best["kappa"],
                                       "noise_scale": best_noise["noise_scale"]},
              "validation": {m: evaluate(val, m, beh) for m in MODELS},
              "grid_top5": grid[:5], "noise_grid": noise}

    # ---- the locked test people, once: states held (fair test) and life events on (as in the game)
    test_held = [Case(L, i, eng, life_events=False) for i in te]
    test_live = [Case(L, i, eng, life_events=True) for i in te]
    report["test"] = {m: evaluate(test_held, m, beh) for m in MODELS}
    report["test_life_events_on"] = {m: evaluate(test_live, m, beh) for m in MODELS}
    report["learned"] = {"phi": beh.phi, "noise_sd_alpha_beta": beh.sd_ab,
                         "december_income_share": {s: round(float(beh.season_a[s][12]), 3) for s in beh.season_a},
                         "february_rent_share": {s: round(float(beh.season_c[s][2]), 3) for s in beh.season_c},
                         "august_rent_share": {s: round(float(beh.season_c[s][8]), 3) for s in beh.season_c}}
    report["seconds"] = round(time.time() - t0, 1)
    (ART / "backtest.json").write_text(json.dumps(report, indent=1, default=float))
    r = lambda d: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}  # noqa: E731
    print("chosen on validation:", report["chosen_on_validation"])
    for part in ("validation", "test", "test_life_events_on"):
        for m, v in report[part].items():
            print(f"{part:20s} {m:18s}", r(v))
    print("seconds", report["seconds"])


if __name__ == "__main__":
    main()
