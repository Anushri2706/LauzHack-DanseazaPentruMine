"""Regime engine: a regime-switching (input-output HMM) scenario generator.

Answers one question: given who this person is, what they decide and our assumptions, what
life regime are they likely in each month, and what monthly net cash flow does that produce?
It does no cash accounting, tax, investments, goals, UI or narration (see cash_engine, metrics).

Two modes, same ``simulate`` / ``filter`` signature:
- ``RegimeEngine.stub()``: deterministic regimes, constant per-state savings rate.
- ``RegimeEngine(emissions, priors)``: the HMM. Emissions are a heavy-tailed spend/rent
  ratio (about 1.75 in every regime, so the regime acts through income). Household type
  shifts the level. Self-employed scale is about 3x the employee scale. The December
  bonus is an extra draw in income units, not a shift of the spend ratio.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp

from .income import IncomeTable, attributes_at, parental_leave_duration, parental_leave_mask
from .transitions import TransitionPriors, load_transition_priors
from .types import K, STATE_INDEX, STATES, Intervention, Profile, RegimePaths

# Where a scheduled (force_state with duration) phase exits to.
EXIT_STATE: dict[str, str] = {"student": "employed", "job_search": "employed"}

E = STATE_INDEX["employed"]
J = STATE_INDEX["job_search"]
_IDX = np.arange(K)
_LOG_FLOOR = math.log(1e-6)


# ---------------------------------------------------------------------------
# Emissions
# ---------------------------------------------------------------------------

def _is_self_employed(profile: Profile | None) -> bool:
    if profile is None:
        return False
    status = profile.employment.replace("-", "_")
    return status == "self_employed" or bool(profile.attributes.get("self_employed"))


@dataclass
class EmissionModel:
    """Per-regime distribution of ``spend_ratio = spend / rent``.

    ``params[state]`` is a Student-t ``{"df", "loc", "scale"}`` or a 2-GMM
    ``{"w", "mu", "sigma"}``. ``loc`` is the level before the household shift.
    ``december_offset`` is the December drop in the spend ratio (the bonus shows
    up as lower spending). The bonus itself is drawn from ``december_bonus_quantiles``.
    """
    family: str                                   # "student_t" | "gmm2" | "constant"
    params: dict[str, dict]
    december_offset: dict[str, float] = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    household_shift: dict = field(default_factory=dict)          # state -> household -> additive shift
    self_employed_params: dict | None = None
    december_bonus_quantiles: dict = field(default_factory=dict)  # employment -> quantile grid in income units
    within_sd: dict = field(default_factory=dict)

    def component(self, state: str, self_employed: bool = False) -> dict:
        if self_employed and state == "employed" and self.self_employed_params:
            return self.self_employed_params
        return self.params[state]

    def household_offset(self, state: str, household: str | None) -> float:
        if not household:
            return 0.0
        return float(self.household_shift.get(state, {}).get(household, 0.0))

    def location(self, state: str, household: str | None = None, self_employed: bool = False) -> float:
        return float(self.component(state, self_employed)["loc"]) + self.household_offset(state, household)

    def mean(self, state: str) -> float:
        p = self.params[state]
        if "loc" in p:
            return float(p["loc"])
        return float(np.dot(p["w"], p["mu"]))

    def personalize(self, state: str, observations: np.ndarray, household: str | None = None,
                    kappa: float = 6.0, self_employed: bool = False) -> "EmissionModel":
        """Shrink the state's location toward this person's non-December spend ratios.

        The scale becomes the within-person residual (shrunk toward the regime's
        within-person sd). Regimes the person has not been observed in are unchanged.
        """
        x = np.asarray(observations, dtype=float)
        x = x[np.isfinite(x)]
        params = {s: dict(v) for s, v in self.params.items()}
        se = dict(self.self_employed_params) if self.self_employed_params else None
        use_se = self_employed and state == "employed" and se is not None
        target = se if use_se else params[state]
        shift = self.household_offset(state, household)
        center = float(target["loc"]) + shift
        n = int(x.size)
        if n:
            center = (n * float(np.mean(x)) + kappa * center) / (n + kappa)
            pool_key = "self_employed" if use_se else state
            pool = float(self.within_sd.get(pool_key, target.get("scale", 0.3)))
            if n >= 3:
                s_i = float(np.std(x, ddof=1))
                scale = math.sqrt((n * s_i * s_i + kappa * pool * pool) / (n + kappa))
            else:
                scale = pool
            target["scale"] = max(scale, 0.05)
            target["df"] = max(float(target.get("df", 30.0)), 30.0)
            target["loc"] = center - shift
        if use_se:
            se = target
        else:
            params[state] = target
        return EmissionModel(self.family, params, dict(self.december_offset), self.diagnostics,
                             self.household_shift, se, self.december_bonus_quantiles, dict(self.within_sd))

    @classmethod
    def placeholder(cls) -> "EmissionModel":
        """Constant savings rates for the stub. Not a spend-ratio model."""
        means = {"employed": 0.52, "part_time": 0.11, "job_search": -0.5,
                 "student": -0.75, "retired": 0.04}
        return cls("constant", {s: {"loc": m, "scale": 0.0, "df": np.inf} for s, m in means.items()},
                   {"employed": 0.35, "part_time": 0.35})


def _fit_t(y: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    lo, hi = np.quantile(y, [0.005, 0.995])
    yw = np.clip(y, lo, hi)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df, loc, scale = stats.t.fit(yw)
    return {"df": float(max(df, 2.05)), "loc": float(loc), "scale": float(max(scale, 1e-3)),
            "winsor_lo": float(lo), "winsor_hi": float(hi)}


def _adjust(df: pd.DataFrame, shift: dict[str, dict[str, float]]) -> np.ndarray:
    if df.empty:
        return np.array([], dtype=float)
    out = np.empty(len(df), dtype=float)
    ratios = df["spend_ratio"].to_numpy(dtype=float)
    states = df["state"].to_numpy()
    houses = df["household_type"].to_numpy()
    for i, (s, h) in enumerate(zip(states, houses)):
        out[i] = ratios[i] - shift.get(s, {}).get(h, 0.0)
    return out


def _within_sd(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    sd = df.groupby("profile_id")["spend_ratio"].std()
    sd = sd[np.isfinite(sd.to_numpy())]
    return float(sd.median()) if len(sd) else float("nan")


def _ks_block(y: np.ndarray, params: dict) -> dict:
    y = y[np.isfinite(y)]
    qs = np.round(np.linspace(0.05, 0.95, 10), 2)
    if len(y) < 8:
        return {"n": int(len(y)), "ks_stat": None, "ks_pvalue": None,
                "qq_probabilities": qs.tolist(), "qq_holdout": [], "qq_model": []}
    ks = stats.kstest(y, "t", args=(params["df"], params["loc"], params["scale"]))
    theo = stats.t.ppf(qs, params["df"], loc=params["loc"], scale=params["scale"])
    emp = np.quantile(y, qs)
    return {"n": int(len(y)), "ks_stat": float(ks.statistic), "ks_pvalue": float(ks.pvalue),
            "qq_probabilities": qs.tolist(),
            "qq_holdout": [float(v) for v in emp], "qq_model": [float(v) for v in theo]}


def fit_emissions(cash_balances: pd.DataFrame | None = None, profiles: pd.DataFrame | None = None,
                  family: str = "student_t", holdout_frac: float = 0.2, seed: int = 0,
                  prepared=None) -> EmissionModel:
    """Fit per-regime Student-t distributions of ``spend_ratio`` on the training profiles.

    The frames are the challenge tables. The split, outlier flags and December bonus
    come from ``data.prepare_data`` (20% of profiles, stratified by regime, seed 0),
    so the holdout ids match ``backend/artifacts/holdout_profiles.json``.

    Self-employed rows are not pooled into the employee fit: they get their own scale.
    Household shifts are removed before the fit and added back when sampling.
    December is not in the body; its bonus is an empirical quantile distribution.
    """
    del cash_balances, profiles  # the seeded prep below is the canonical table
    if family not in ("student_t", "gmm2", "constant"):
        raise ValueError("family must be 'student_t', 'gmm2' or 'constant'")
    if prepared is None:
        from .data import prepare_data
        prepared = prepare_data(holdout_frac=holdout_frac, seed=seed)
    prep = prepared
    body = prep.train_body
    raw_median = {s: float(body.loc[body["state"] == s, "spend_ratio"].median()) for s in STATES}

    # Household shift: per (regime, household) when the cell is large, otherwise the
    # sample-size-weighted average of the regime-level shifts.
    household_shift: dict[str, dict[str, float]] = {s: {} for s in STATES}
    pooled: dict[str, list[tuple[int, float]]] = {}
    for (s, hh), g in body.groupby(["state", "household_type"]):
        delta = float(g["spend_ratio"].median() - raw_median[s])
        pooled.setdefault(hh, []).append((len(g), delta))
        if len(g) >= 80:
            household_shift[s][hh] = delta
    pooled_shift = {hh: float(np.average([d for _, d in pairs], weights=[n for n, _ in pairs]))
                    for hh, pairs in pooled.items()}
    for s in STATES:
        for hh, delta in pooled_shift.items():
            household_shift[s].setdefault(hh, delta)

    params: dict[str, dict] = {}
    by_state: dict[str, dict] = {}
    hold = prep.holdout
    hold_body = hold.loc[~hold["is_outlier"] & ~hold["is_december"]]
    for s in STATES:
        g = body.loc[body["state"] == s]
        if s == "employed":
            g = g.loc[g["employment_status"] == "employed"]
        params[s] = _fit_t(_adjust(g, household_shift))
        h = hold_body.loc[hold_body["state"] == s]
        if s == "employed":
            h = h.loc[h["employment_status"] == "employed"]
        block = _ks_block(_adjust(h, household_shift), params[s])
        block.update({"median_spend_ratio": raw_median[s], "loc": params[s]["loc"],
                      "scale": params[s]["scale"], "df": params[s]["df"],
                      "n_train": int(len(g))})
        by_state[s] = block

    se_rows = body.loc[body["employment_status"] == "self_employed"]
    se_params = _fit_t(_adjust(se_rows, household_shift))
    se_hold = hold_body.loc[hold_body["employment_status"] == "self_employed"]
    se_ks = _ks_block(_adjust(se_hold, household_shift), se_params)
    scale_ratio = float(se_params["scale"] / params["employed"]["scale"]) if params["employed"]["scale"] else float("nan")

    within = {s: _within_sd(body.loc[(body["state"] == s) & (body["employment_status"] != "self_employed")]
                            if s == "employed" else body.loc[body["state"] == s])
              for s in STATES}
    within["self_employed"] = _within_sd(se_rows)
    for s in STATES:
        by_state[s]["within_sd"] = within[s]

    dec = prep.train_december
    quantiles: dict[str, list[float]] = {}
    bonus_median: dict[str, float] = {}
    grid = np.linspace(0.0, 1.0, 101)
    for status in ("employed", "self_employed"):
        x = dec.loc[dec["employment_status"] == status, "dec_bonus_income"].dropna().to_numpy(dtype=float)
        quantiles[status] = [float(v) for v in np.quantile(x, grid)] if len(x) else [0.0]
        bonus_median[status] = float(np.median(x)) if len(x) else 0.0

    diagnostics = {
        "family": family,
        "emission": "spend_ratio = (reference_income - net) / rent",
        "holdout_ids": prep.holdout_ids,
        "by_state": by_state,
        "self_employed": {**se_ks, "loc": se_params["loc"], "scale": se_params["scale"],
                          "df": se_params["df"], "n_train": int(len(se_rows)),
                          "median_spend_ratio": float(se_rows["spend_ratio"].median()) if len(se_rows) else None},
        "self_employed_scale_ratio": scale_ratio,
        "self_employed_within_sd_ratio": float(within["self_employed"] / within["employed"])
        if within["employed"] else None,
        "household_shift": household_shift,
        "december_bonus_median": bonus_median,
        "december_offset_spend_ratio": {s: float(v) for s, v in prep.december_offset.items()},
        "note": "Regimes share a spend/rent level near 1.75. They differ by income, not by how hard they spend relative to rent.",
    }
    return EmissionModel(family, params, {s: float(v) for s, v in prep.december_offset.items()},
                         diagnostics, household_shift, se_params, quantiles, within)


def _gmm_logpdf(x: np.ndarray, params: dict, loc_shift: float) -> np.ndarray:
    w = np.asarray(params["w"], dtype=float)
    mu = np.asarray(params["mu"], dtype=float) + loc_shift
    sig = np.asarray(params["sigma"], dtype=float)
    parts = np.stack([np.log(max(wi, 1e-12)) + stats.norm.logpdf(x, loc=mi, scale=max(si, 1e-6))
                      for wi, mi, si in zip(w, mu, sig)], axis=0)
    return logsumexp(parts, axis=0)


def emission_loglik(model: EmissionModel, ratios: np.ndarray, month_of_year: np.ndarray,
                    household: str | None = None, self_employed: bool = False,
                    floor: float = 1e-6) -> np.ndarray:
    """log p(spend_ratio_t | s_t = k). Returns [T, K].

    ``ratios`` is [T] (the same observation scored under every regime) or [T, K]
    (a regime-specific implied spend ratio, which is what the filter passes).
    December observations have ``december_offset[k]`` removed first. The density
    is floored at ``log(floor)`` so one outlier cannot zero out a regime.
    """
    ratios = np.asarray(ratios, dtype=float)
    month = np.atleast_1d(np.asarray(month_of_year))
    if ratios.ndim == 1:
        ratios = np.repeat(ratios[:, None], K, axis=1)
    if ratios.shape[1] != K:
        raise ValueError(f"ratios must have {K} columns, got {ratios.shape}")
    T = ratios.shape[0]
    if len(month) != T:
        raise ValueError("month_of_year must have one entry per row of ratios")
    ll = np.empty((T, K))
    dec = month == 12
    floor_ll = math.log(floor)
    for k, name in enumerate(STATES):
        p = model.component(name, self_employed and name == "employed")
        x = ratios[:, k].copy()
        if dec.any():
            x[dec] = x[dec] - float(model.december_offset.get(name, 0.0))
        loc = float(p.get("loc", 0.0)) + model.household_offset(name, household)
        scale = max(float(p.get("scale", 1e-3)), 1e-6)
        if model.family == "gmm2" or "w" in p:
            ll[:, k] = _gmm_logpdf(x, p, model.household_offset(name, household))
        elif model.family == "constant" and scale == 0.0:
            ll[:, k] = np.where(np.abs(x - loc) < 1e-6, 0.0, floor_ll)
        else:
            df = float(p.get("df", 1e6))
            if not np.isfinite(df) or df > 1e5:
                ll[:, k] = stats.norm.logpdf(x, loc=loc, scale=scale)
            else:
                ll[:, k] = stats.t.logpdf(x, df, loc=loc, scale=scale)
    return np.maximum(ll, floor_ll)


# ---------------------------------------------------------------------------
# Filtering and sampling
# ---------------------------------------------------------------------------

def forward_filter(log_lik: np.ndarray, A: np.ndarray, pi0: np.ndarray) -> tuple[np.ndarray, float]:
    """Scaled forward algorithm in log space.

    Returns ``(filtered [T, K], total log-likelihood)``. ``A`` is ``[K, K]`` or
    time-varying ``[T, K, K]``, and ``A[t]`` is ``P(s_t | s_{t-1})`` (``A[0]`` is unused).
    """
    log_lik = np.asarray(log_lik, dtype=float)
    T, k = log_lik.shape
    A = np.asarray(A, dtype=float)
    pi = np.asarray(pi0, dtype=float)
    pi = np.maximum(pi, 0.0)
    pi = pi / pi.sum()
    filtered = np.empty((T, k))
    log_alpha = np.log(np.maximum(pi, 1e-300)) + log_lik[0]
    log_norm = float(logsumexp(log_alpha))
    log_alpha = log_alpha - log_norm
    total = log_norm
    filtered[0] = np.exp(log_alpha)
    varying = A.ndim == 3
    for t in range(1, T):
        At = A[t] if varying else A
        log_pred = logsumexp(log_alpha[:, None] + np.log(np.maximum(At, 1e-300)), axis=0)
        log_alpha = log_pred + log_lik[t]
        log_norm = float(logsumexp(log_alpha))
        log_alpha = log_alpha - log_norm
        total += log_norm
        filtered[t] = np.exp(log_alpha)
    filtered = np.clip(filtered, 0.0, None)
    filtered /= filtered.sum(axis=1, keepdims=True)
    return filtered, float(total)


def _draw_family(rng: np.random.Generator, family: str, params: dict, loc: float, n: int) -> np.ndarray:
    if family == "gmm2" or "w" in params:
        w = np.asarray(params["w"], dtype=float)
        mu = np.asarray(params["mu"], dtype=float)
        sig = np.asarray(params["sigma"], dtype=float)
        # `loc` replaces the mixture mean: shift every component by loc - mean.
        shift = loc - float(np.dot(w, mu))
        comp = rng.choice(len(w), size=n, p=w / w.sum())
        return mu[comp] + shift + sig[comp] * rng.standard_normal(n)
    scale = float(params.get("scale", 0.0))
    df = float(params.get("df", 1e6))
    if family == "constant" or scale == 0.0:
        return np.full(n, loc)
    if not np.isfinite(df) or df > 1e5:
        return loc + scale * rng.standard_normal(n)
    return loc + scale * rng.standard_t(df, size=n)


def sample_emissions(rng: np.random.Generator, model: EmissionModel, states: np.ndarray,
                     month_of_year: np.ndarray, overrides: dict | None = None,
                     profile: Profile | None = None) -> np.ndarray:
    """Sample ``spend_ratio`` with shape ``[n_paths, T]``.

    December is not added here. The bonus is drawn by ``sample_december_bonus``.
    ``spend_ratio_<state>`` overrides replace that state's location.
    """
    overrides = overrides or {}
    n, T = states.shape
    del month_of_year  # the bonus, not the ratio, depends on the month
    household = profile.attributes.get("household_type") if profile is not None else None
    se = _is_self_employed(profile)
    out = np.empty((n, T))
    for k, name in enumerate(STATES):
        mask = states == k
        count = int(mask.sum())
        if count == 0:
            continue
        p = model.component(name, se and name == "employed")
        key = f"spend_ratio_{name}"
        loc = float(overrides[key]) if key in overrides else model.location(name, household, se and name == "employed")
        out[mask] = _draw_family(rng, model.family, p, loc, count)
    return out


def sample_december_bonus(rng: np.random.Generator, model: EmissionModel, states: np.ndarray,
                          month_of_year: np.ndarray, profile: Profile | None = None) -> np.ndarray:
    """December bump in income units, shape ``[n_paths, T]``. Zero outside December and outside employment.

    Employed and self-employed each have their own empirical quantile function, fit on
    the training Decembers. Other regimes have no 13th-salary bonus in the ledger.
    """
    n, T = states.shape
    bonus = np.zeros((n, T))
    dec = np.asarray(month_of_year) == 12
    if not np.any(dec):
        return bonus
    key = "self_employed" if _is_self_employed(profile) else "employed"
    qs = model.december_bonus_quantiles.get(key)
    if not qs:
        return bonus
    qs_arr = np.asarray(qs, dtype=float)
    grid = np.linspace(0.0, 1.0, len(qs_arr))
    n_dec = int(dec.sum())
    draws = np.interp(rng.random((n, n_dec)), grid, qs_arr)
    employed = states[:, dec] == E
    block = np.zeros((n, n_dec))
    block[employed] = draws[employed]
    bonus[:, dec] = block
    return bonus


def sample_yearly_lump(rng: np.random.Generator, n: int, horizon: int, params: dict) -> np.ndarray:
    """Shock around the irregular-spend mean, charged in one random month per year.

    ``M ~ Lognormal(mean 1, sigma)``, ``L = mean * M``, and the cash hit is ``L - mean``.
    The mean itself sits inside the 0.25 non-housing rate. Off unless ``scaled_spending`` is set.
    """
    lump = np.zeros((n, horizon))
    if float(params.get("scaled_spending", 0.0)) < 0.5:
        return lump
    mean = float(params["lump_mean_chf"])
    sigma = float(params["lump_sigma"])
    mu = -0.5 * sigma * sigma
    year = 0
    while year * 12 < horizon:
        t0 = year * 12
        width = min(12, horizon - t0)
        shock = mean * (rng.lognormal(mu, sigma, size=n) - 1.0)
        month = rng.integers(0, width, size=n)
        lump[np.arange(n), t0 + month] = shock
        year += 1
    return lump


def _schedule(T: int, interventions: list[Intervention],
              params: dict | None = None) -> tuple[np.ndarray, dict[int, int]]:
    """Forced months and, for exits back to the previous regime, the month to copy from.

    ``forced[t] >= 0`` pins that state. ``forced[t] == -2`` means "the regime just
    before this phase". ``forced[t] == -1`` is free.

    ``parental_leave`` pins employed for its duration and for the following month.
    Pay for the duration is the replacement prior; that is applied by the income table.
    """
    params = params or {}
    forced = np.full(T, -1, dtype=int)
    exits: list[tuple[int, int | None, int]] = []
    kind_rank = {"force_state": 0, "parental_leave": 1}
    pins = [iv for iv in interventions if iv.kind in kind_rank]
    for iv in sorted(pins, key=lambda i: (i.start_month, kind_rank[i.kind])):
        t0 = int(iv.start_month)
        if t0 >= T or t0 < 0:
            continue
        if iv.kind == "parental_leave":
            dur = parental_leave_duration(iv, params)
            t1 = min(T, t0 + dur)
            forced[t0:t1] = E
            # The month after the pin is employed. A later force_state can still overwrite it.
            if t1 < T and forced[t1] < 0:
                forced[t1] = E
            continue
        if iv.state not in STATE_INDEX:
            raise ValueError(f"Unknown force_state {iv.state!r}")
        sid = STATE_INDEX[iv.state]
        if iv.duration is None:
            forced[t0] = sid
            continue
        t1 = min(T, t0 + int(iv.duration))
        forced[t0:t1] = sid
        if t1 < T:
            exit_name = EXIT_STATE.get(iv.state)
            exits.append((t1, STATE_INDEX[exit_name] if exit_name else None, t0))
    prev_from: dict[int, int] = {}
    for t1, sid, t0 in exits:
        if forced[t1] >= 0:
            continue
        if sid is None:
            forced[t1] = -2
            prev_from[t1] = t0
        else:
            forced[t1] = sid
    return forced, prev_from


def _rebuild_diagonal(P: np.ndarray) -> np.ndarray:
    """P is [n, K, K] with zeros on the diagonal. Put the residual probability there."""
    off = P.sum(axis=2)
    over = off > 1.0
    if np.any(over):
        P[over] /= off[over, None, None]
        off = P.sum(axis=2)
    P[:, _IDX, _IDX] = np.clip(1.0 - off, 0.0, 1.0)
    P /= P.sum(axis=2, keepdims=True)
    return P


def sample_regime_paths(rng: np.random.Generator, pi0: np.ndarray, A_draws: np.ndarray, horizon: int,
                        profile: Profile, priors: TransitionPriors, interventions: list[Intervention],
                        overrides: dict, retirement_age: float = 65.0,
                        params: dict | None = None) -> np.ndarray:
    """Sample ``[n_paths, T]`` state indices.

    One Dirichlet draw per path (``A_draws``) carries parameter uncertainty. Each month
    rescales that draw to the age-specific hazards, applies ``scale_hazard``, and halves
    the job-search exit after 12 months. ``force_state`` with a duration is a pinned
    HSMM phase; without a duration it is a one-month jump and the chain runs after that.
    """
    params = params or {}
    n = A_draws.shape[0]
    pi = np.asarray(pi0, dtype=float)
    pi = np.maximum(pi, 0.0)
    pi = pi / pi.sum()
    forced, prev_from = _schedule(horizon, interventions, params)
    leave = parental_leave_mask(horizon, interventions, params)
    states = np.empty((n, horizon), dtype=int)
    snapshots: dict[int, np.ndarray] = {}
    for t_exit, t_start in prev_from.items():
        if t_start <= 0:
            snapshots[t_exit] = np.full(n, STATE_INDEX[profile.initial_state])
    base = priors.matrix(overrides)
    decay = float(params.get("long_term_unemployment_multiplier", 0.5))
    after = float(params.get("long_term_unemployment_months", 12))
    masters_f = float(params.get("masters_hazard_factor", 1.0))
    dwell = np.zeros(n)
    cdf0 = np.cumsum(pi)
    for t in range(horizon):
        if t > 0:
            for t_exit, t_start in prev_from.items():
                if t_start == t:
                    snapshots[t_exit] = states[:, t - 1].copy()
        code = int(forced[t])
        if code >= 0:
            states[:, t] = code
        elif code == -2:
            states[:, t] = snapshots[t]
        elif t == 0:
            u = rng.random(n)
            states[:, t] = np.argmax(cdf0 >= u[:, None], axis=1)
        else:
            age = profile.age + t / 12.0
            target = priors.matrix(overrides, age=age)
            with np.errstate(divide="ignore", invalid="ignore"):
                scale = np.where(base > 1e-12, target / np.maximum(base, 1e-12), 0.0)
            np.fill_diagonal(scale, 0.0)
            P = A_draws * scale[None, :, :]
            attrs = attributes_at(profile, interventions, t)
            if attrs.get("has_masters") and masters_f != 1.0:
                P[:, E, J] *= masters_f
            for iv in interventions:
                if iv.kind == "scale_hazard" and iv.active(t) and iv.transition and iv.factor is not None:
                    a, b = iv.transition
                    P[:, STATE_INDEX[a], STATE_INDEX[b]] *= float(iv.factor)
            long = dwell >= after
            if decay != 1.0 and np.any(long):
                P[long, J, :] *= decay
            P = _rebuild_diagonal(P)
            row = P[np.arange(n), states[:, t - 1]]
            u = rng.random(n)
            states[:, t] = np.argmax(np.cumsum(row, axis=1) >= u[:, None], axis=1)
        age_now = profile.age + t / 12.0
        if age_now >= retirement_age and not bool(leave[t]):
            working = (states[:, t] == E) | (states[:, t] == STATE_INDEX["part_time"])
            states[working, t] = STATE_INDEX["retired"]
        dwell = np.where(states[:, t] == J, dwell + 1.0, 0.0)
    return states


def analytic_occupancy(pi0: np.ndarray, A: np.ndarray, horizon: int) -> np.ndarray:
    """P(s_t = k) = pi0 A^t for t = 0..T-1. ``A`` is [K, K] or [T, K, K]. Returns [T, K]."""
    occ = np.empty((horizon, len(pi0)))
    p = np.asarray(pi0, dtype=float)
    for t in range(horizon):
        occ[t] = p
        At = A[t] if A.ndim == 3 else A
        p = p @ At
    return occ


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def _one_hot(state: str) -> np.ndarray:
    v = np.zeros(K)
    v[STATE_INDEX[state]] = 1.0
    return v


def _initial_belief(profile: Profile, initial_belief: dict | None) -> np.ndarray:
    if initial_belief:
        pi = np.array([float(initial_belief.get(s, 0.0)) for s in STATES])
        if pi.sum() <= 0:
            pi = _one_hot(profile.initial_state)
        return pi / pi.sum()
    # One-hot mixed with 5% uniform, so a regime change can still be detected.
    return 0.95 * _one_hot(profile.initial_state) + 0.05 / K


def scheduled_states(profile: Profile, horizon: int, interventions: list[Intervention],
                     retirement_age: float, params: dict | None = None) -> np.ndarray:
    """Deterministic state sequence [T] for the stub: initial state, force_state, retirement.

    A jump with no duration persists, because the stub has no transition matrix.
    Parental leave is employed for its duration and the stub then stays employed,
    because it has no chain to run. A later pin still overwrites that tail.
    """
    params = params or {}
    base = STATE_INDEX[profile.initial_state]
    seq = np.full(horizon, base)
    kind_rank = {"force_state": 0, "parental_leave": 1}
    pins = [iv for iv in interventions if iv.kind in kind_rank]
    for iv in sorted(pins, key=lambda i: (i.start_month, kind_rank[i.kind])):
        if iv.start_month >= horizon or iv.start_month < 0:
            continue
        if iv.kind == "parental_leave":
            dur = parental_leave_duration(iv, params)
            end = min(horizon, iv.start_month + dur)
            seq[iv.start_month:end] = E
            if end < horizon:
                seq[end:] = E
            continue
        end = horizon if iv.duration is None else min(horizon, iv.start_month + iv.duration)
        prev = seq[iv.start_month] if iv.start_month < horizon else base
        seq[iv.start_month:end] = STATE_INDEX[iv.state]
        if iv.duration is not None and end < horizon:
            exit_state = EXIT_STATE.get(iv.state)
            seq[end:] = STATE_INDEX[exit_state] if exit_state else prev
    ages = profile.age + np.arange(horizon) / 12.0
    leave = parental_leave_mask(horizon, interventions, params)
    working = np.isin(seq, [E, STATE_INDEX["part_time"]]) & ~leave
    seq[working & (ages >= retirement_age)] = STATE_INDEX["retired"]
    return seq


class RegimeEngine:
    def __init__(self, emissions: EmissionModel, priors: TransitionPriors,
                 income_table: IncomeTable | None = None, mode: str = "hmm"):
        if mode not in ("hmm", "stub"):
            raise ValueError("mode must be 'hmm' or 'stub'")
        self.emissions = emissions
        self.priors = priors
        self.income_table = income_table or IncomeTable.from_csv()
        self.mode = mode

    @classmethod
    def stub(cls, priors: TransitionPriors | None = None, income_table: IncomeTable | None = None) -> "RegimeEngine":
        return cls(EmissionModel.placeholder(), priors or load_transition_priors(), income_table, mode="stub")

    def filter(self, profile: Profile, observed_net_flows: list[float],
               assumption_overrides: dict | None = None) -> dict[str, float]:
        """Belief over the current regime after observing monthly net flows (CHF)."""
        if self.mode == "stub":
            belief = {s: 0.1 / (K - 1) for s in STATES}
            belief[profile.initial_state] = 0.9
            return belief
        flows = np.asarray(list(observed_net_flows), dtype=float)
        if flows.ndim != 1 or len(flows) == 0:
            raise ValueError("observed_net_flows must be a non-empty sequence")
        T = len(flows)
        overrides = dict(assumption_overrides or {})
        table = self.income_table.with_overrides(overrides)
        seqs = np.stack([np.full(T, k, dtype=int) for k in range(K)])
        table.income_paths(seqs, profile, experience=False)
        incomes = table.net_income  # [K, T], net: the ledger flows are net
        rent = float(profile.rent) if profile.rent > 0 else float(table.params["employed_rent_chf"])
        ratios = ((incomes - flows[None, :]) / rent).T  # [T, K]
        month = (profile.start_month_of_year - 1 + np.arange(T)) % 12 + 1
        ll = emission_loglik(self.emissions, ratios, month,
                             household=profile.attributes.get("household_type"),
                             self_employed=_is_self_employed(profile))
        ages = profile.age + np.arange(T) / 12.0
        A = np.stack([self.priors.matrix(overrides, age=float(a)) for a in ages])
        filtered, _ = forward_filter(ll, A, _initial_belief(profile, None))
        return {s: float(filtered[-1, i]) for i, s in enumerate(STATES)}

    def simulate(self, profile: Profile, horizon_months: int = 60, n_paths: int = 5000,
                 interventions: list[Intervention] | None = None, assumption_overrides: dict | None = None,
                 initial_belief: dict | None = None, seed: int = 0) -> RegimePaths:
        interventions = list(interventions or [])
        overrides = dict(assumption_overrides or {})
        income_table = self.income_table.with_overrides(overrides)
        if self.mode == "stub":
            return self._simulate_stub(profile, horizon_months, n_paths, interventions, overrides, income_table)
        return self._simulate_hmm(profile, horizon_months, n_paths, interventions, overrides,
                                  initial_belief, seed, income_table)

    def savings_rate_for(self, state: str, profile: Profile, overrides: dict) -> float:
        key = f"savings_rate_{state}"
        if key in overrides:
            return float(overrides[key])
        if self.mode == "stub":
            if state == profile.initial_state and profile.mean_savings_rate is not None:
                return float(profile.mean_savings_rate)
            return self.emissions.mean(state)
        inc, rent = self._income_rent_level(state, profile)
        loc = self.emissions.location(state, profile.attributes.get("household_type"),
                                      _is_self_employed(profile) and state == "employed")
        if inc <= 1.0:
            return 0.0
        return float((inc - rent * loc) / inc)

    def _income_rent_level(self, state: str, profile: Profile) -> tuple[float, float]:
        seq = np.full((1, 1), STATE_INDEX[state], dtype=int)
        self.income_table.income_paths(seq, profile, experience=False)
        inc = float(self.income_table.net_income[0, 0])
        rent = float(self.income_table.rent_paths(seq, profile)[0, 0])
        return inc, rent

    def _simulate_stub(self, profile, T, n, interventions, overrides, income_table) -> RegimePaths:
        seq = scheduled_states(profile, T, interventions, income_table.params["retirement_age"],
                               income_table.params)
        states = np.repeat(seq[None, :], n, axis=0)
        gross = np.repeat(income_table.income_paths(seq[None, :], profile, interventions), n, axis=0)
        income = np.repeat(income_table.net_income, n, axis=0)
        rate_by_state = np.array([self.savings_rate_for(s, profile, overrides) for s in STATES])
        savings_rate = rate_by_state[states]
        occupancy = np.zeros((T, K))
        occupancy[np.arange(T), seq] = 1.0
        A = self.priors.matrix(overrides)
        return RegimePaths(
            states=states,
            income_chf=income,
            savings_rate=savings_rate,
            net_flow_chf=savings_rate * income,
            occupancy=occupancy,
            transition_draws=np.broadcast_to(A, (n, K, K)).copy(),
            gross_income_chf=gross,
        )

    def _simulate_hmm(self, profile, T, n, interventions, overrides, initial_belief, seed, income_table) -> RegimePaths:
        rng = np.random.default_rng(seed)
        # A passed-in belief is used as-is (the filter already mixed in 5% uniform).
        # With no belief, the path starts in the profile's regime.
        if initial_belief:
            pi0 = np.array([float(initial_belief.get(s, 0.0)) for s in STATES])
            pi0 = np.maximum(pi0, 0.0)
            pi0 = pi0 / pi0.sum() if pi0.sum() > 0 else _one_hot(profile.initial_state)
        else:
            pi0 = _one_hot(profile.initial_state)
        if float(overrides.get("dirichlet", 1.0)) == 0.0:
            A_draws = np.broadcast_to(self.priors.matrix(overrides), (n, K, K)).copy()
        else:
            A_draws = self.priors.sample(rng, n, overrides)
        states = sample_regime_paths(rng, pi0, A_draws, T, profile, self.priors, interventions, overrides,
                                     income_table.params["retirement_age"], income_table.params)
        month_of_year = (profile.start_month_of_year - 1 + np.arange(T)) % 12 + 1
        spend = sample_emissions(rng, self.emissions, states, month_of_year, overrides, profile)
        bonus_units = sample_december_bonus(rng, self.emissions, states, month_of_year, profile)
        # The replacement prior is the whole paycheck. Do not also draw a 13th salary.
        leave = parental_leave_mask(T, interventions, income_table.params)
        bonus_units[:, leave] = 0.0
        gross = income_table.income_paths(states, profile, interventions)
        income = income_table.net_income  # what lands in the account; rent rules use gross
        lump = sample_yearly_lump(rng, n, T, income_table.params)
        if income_table.scales_spending():
            # The expense-priors recipe replaces spend_ratio x rent. Do not add both.
            housing, nonhousing = income_table.consumption_paths(states, profile, interventions)
            net = income - housing - nonhousing + bonus_units * income - lump
        else:
            rent = income_table.rent_paths(states, profile, interventions)
            net = income - rent * spend + bonus_units * income
            household = profile.attributes.get("household_type")
            se = _is_self_employed(profile)
            for k, name in enumerate(STATES):
                key = f"savings_rate_{name}"
                if key not in overrides:
                    continue
                loc = self.emissions.location(name, household, se and name == "employed")
                center = np.zeros(income.shape)
                ok = np.abs(income) > 1.0
                center[ok] = (income[ok] - rent[ok] * loc) / income[ok]
                shift = np.where(states == k, (float(overrides[key]) - center) * income, 0.0)
                net = net + shift
        savings = np.zeros_like(net)
        ok = np.abs(income) > 1.0
        savings[ok] = net[ok] / income[ok]
        occupancy = np.stack([(states == k).mean(axis=0) for k in range(K)], axis=1)
        return RegimePaths(states, income, savings, net, occupancy, A_draws, lump, gross)
