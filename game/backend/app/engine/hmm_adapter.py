"""Translator between a town action and one HMM query.

This is the only game module that imports ``twin``. ``step_month`` stays the town's
month stepper and is not called here. ``commit_net_flows`` stays unimplemented:
the reduction of a path ensemble to one number per month is still an open question.
Callers pass the list they want appended.
"""
from __future__ import annotations

import sys
import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np

_BACKEND = Path(__file__).resolve().parents[4] / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from twin import cash_engine, metrics  # noqa: E402
from twin.cash_engine import OneOff  # noqa: E402
from twin.data import load_cash_balances, load_profile, load_profiles, observed_net_flows  # noqa: E402
from twin.regime_engine import RegimeEngine, branch_decisions, fit_emissions  # noqa: E402
from twin.transitions import load_transition_priors  # noqa: E402
from twin.twin_session import DECISIONS, decision_interventions  # noqa: E402
from twin.types import STATES, DecisionSnapshot, Intervention  # noqa: E402

# Interactive sample. ``simulate``'s own default stays 5000.
PHONE_N_PATHS = 40

# Accessible names must not contain the town button "Full-time master's".
DECISION_LABELS = {
    "masters": "Master's pin (HMM)",
    "start_working": "Start working (HMM)",
    "move_abroad": "Move abroad (HMM)",
    "job_loss": "Job search (HMM)",
    "parental_leave": "Parental leave (HMM)",
    "part_time": "Part-time regime (HMM)",
    "downturn": "Downturn (HMM)",
    "rent_step": "Step up the rent (HMM)",
    "owns_home": "Own the home (HMM)",
}

_MENU = (
    "masters", "start_working", "move_abroad", "job_loss", "parental_leave",
    "part_time", "downturn", "rent_step", "owns_home",
)


def commit_net_flows(paths) -> list[float]:
    """Open question: which reduction of ``net_flow_chf`` becomes the next observation.

    The body stays unimplemented. ``commit`` appends a list the caller supplies.
    """
    raise NotImplementedError(
        "Which reduction of RegimePaths.net_flow_chf is appended is undecided."
    )


def decision_menu() -> list[dict]:
    return [{"decision": name, "label": DECISION_LABELS[name]} for name in _MENU if name in DECISION_LABELS]


def town_action_interventions(action, month: int = 0):
    """Map one town action onto HMM interventions.

    ``master_ft`` is the one control that lines up with ``DECISIONS['masters']``.
    Anything else is rejected, including ``master_pt``, ``buy_home``, and a
    made-up university id. The master's duration stays the HMM default of 24.
    """
    kind = getattr(action, "type", None)
    ident = getattr(action, "id", None)
    if kind is None and isinstance(action, tuple) and len(action) == 2:
        kind, ident = action
    if kind == "life_choice" and ident == "master_ft":
        return decision_interventions("masters", int(month), {})
    raise ValueError(f"No HMM decision for {kind}:{ident}")


def advance_year(engine: RegimeEngine, snapshot: DecisionSnapshot, decision: str, *, n_paths: int = 5000):
    """One 12-month ``simulate_until_next_decision``. Does not call ``step_month``."""
    return engine.simulate_until_next_decision(
        snapshot.profile,
        decision,
        from_month=snapshot.month,
        until_month=int(snapshot.month) + 12,
        n_paths=n_paths,
        interventions=list(snapshot.interventions),
        assumption_overrides=dict(snapshot.assumption_overrides),
        initial_belief=dict(snapshot.belief) if snapshot.belief else None,
        seed=snapshot.seed,
    )


@lru_cache(maxsize=1)
def get_engine() -> RegimeEngine:
    return RegimeEngine(fit_emissions(load_cash_balances(), load_profiles()), load_transition_priors())


def _engine(engine: RegimeEngine | None) -> RegimeEngine:
    return engine if engine is not None else get_engine()


def intervention_to_dict(iv: Intervention) -> dict:
    return {
        "kind": iv.kind,
        "start_month": int(iv.start_month),
        "duration": None if iv.duration is None else int(iv.duration),
        "state": iv.state,
        "transition": list(iv.transition) if iv.transition else None,
        "factor": None if iv.factor is None else float(iv.factor),
        "attribute": list(iv.attribute) if iv.attribute else None,
    }


def intervention_from_dict(raw: dict) -> Intervention:
    transition = raw.get("transition")
    attribute = raw.get("attribute")
    duration = raw.get("duration")
    factor = raw.get("factor")
    return Intervention(
        kind=str(raw["kind"]),
        start_month=int(raw["start_month"]),
        duration=None if duration is None else int(duration),
        state=raw.get("state"),
        transition=tuple(transition) if transition else None,
        factor=None if factor is None else float(factor),
        attribute=(attribute[0], attribute[1]) if attribute else None,
    )


def one_off_to_dict(item: OneOff) -> dict:
    return {"month": int(item.month), "amount_chf": float(item.amount_chf), "label": item.label}


def one_off_from_dict(raw: dict) -> OneOff:
    return OneOff(int(raw["month"]), float(raw["amount_chf"]), str(raw.get("label", "")))


def snapshot_to_dict(snap: DecisionSnapshot, *, label: str = "", cash=None, debt=None, investments=None) -> dict:
    """JSON for the caller. Cash, debt, and investments are display readings, not snapshot fields."""
    return {
        "snapshot_id": snap.snapshot_id,
        "parent_id": snap.parent_id,
        "month": int(snap.month),
        "label": label,
        "profile_id": snap.profile.profile_id,
        "belief": {k: float(v) for k, v in snap.belief.items()},
        "observed_net_flows": [float(x) for x in snap.observed_net_flows],
        "interventions": [intervention_to_dict(iv) for iv in snap.interventions],
        "one_offs": [one_off_to_dict(item) for item in snap.one_offs],
        "assumption_overrides": dict(snap.assumption_overrides),
        "seed": int(snap.seed),
        "cash": None if cash is None else float(cash),
        "debt": None if debt is None else float(debt),
        "investments": None if investments is None else float(investments),
    }


def snapshot_from_dict(raw: dict) -> DecisionSnapshot:
    profile = load_profile(str(raw["profile_id"]))
    return DecisionSnapshot(
        month=int(raw["month"]),
        profile=profile,
        belief={str(k): float(v) for k, v in dict(raw.get("belief") or {}).items()},
        observed_net_flows=[float(x) for x in raw.get("observed_net_flows") or []],
        interventions=[intervention_from_dict(iv) for iv in raw.get("interventions") or []],
        one_offs=[one_off_from_dict(item) for item in raw.get("one_offs") or []],
        assumption_overrides=dict(raw.get("assumption_overrides") or {}),
        seed=int(raw.get("seed", 0)),
        parent_id=raw.get("parent_id"),
        snapshot_id=str(raw["snapshot_id"]),
    )


def _shift_intervention(iv: Intervention, origin: int) -> Intervention:
    return Intervention(
        iv.kind, int(iv.start_month) - int(origin), iv.duration,
        state=iv.state, transition=iv.transition, factor=iv.factor, attribute=iv.attribute,
    )


def block_readings(profile, paths, one_offs: list, origin: int) -> dict:
    """End-of-block cash, debt, and investments. Debt and investments are path totals."""
    kept = []
    for item in one_offs:
        month = int(item.month) - int(origin)
        if 0 <= month < paths.horizon:
            kept.append(OneOff(month, float(item.amount_chf), item.label))
    cash = cash_engine.run(profile, paths, kept)
    debt = float(paths.debt_chf[0, -1]) if paths.debt_chf is not None else 0.0
    investments = float(np.median(paths.investments_chf[:, -1])) if paths.investments_chf is not None else 0.0
    return {
        "cash": float(np.median(cash.cash[:, -1])),
        "debt": debt,
        "investments": investments,
    }


def start_snapshot(persona_id: str = "P00001", *, seed: int = 0, engine: RegimeEngine | None = None) -> dict:
    """Root of the branch tree. Month 0, belief from the profile's observed ledger."""
    profile = load_profile(persona_id)
    flows = observed_net_flows(persona_id)
    if not flows:
        raise ValueError(f"No observed net flows for {persona_id}")
    hmm = _engine(engine)
    overrides = {"dirichlet": 0.0}
    belief = hmm.filter(profile, flows, overrides)
    snap = DecisionSnapshot(
        month=0,
        profile=profile,
        belief=belief,
        observed_net_flows=list(flows),
        interventions=[],
        one_offs=[],
        assumption_overrides=overrides,
        seed=seed,
        parent_id=None,
        snapshot_id="root",
    )
    return {"snapshot": snapshot_to_dict(snap, label="Start"), "decisions": decision_menu()}


def branch_view(raw_snapshot: dict, decisions: list[str], *, until_month: int | None = None,
                n_paths: int = PHONE_N_PATHS, engine: RegimeEngine | None = None) -> dict:
    """Side-by-side forks. Does not write a snapshot and does not append observations."""
    snap = snapshot_from_dict(raw_snapshot)
    before = list(snap.observed_net_flows)
    end = int(snap.month) + 12 if until_month is None else int(until_month)
    paths = branch_decisions(_engine(engine), snap, list(decisions), until_month=end, n_paths=n_paths)
    if snap.observed_net_flows != before:
        raise RuntimeError("branch_decisions appended observations")
    forks = []
    for name, block in paths.items():
        _ivs, offs = decision_interventions(name, snap.month, {})
        readings = block_readings(snap.profile, block, list(snap.one_offs) + offs, snap.month)
        forks.append({
            "decision": name,
            "label": DECISION_LABELS.get(name, name),
            "horizon_months": int(block.horizon),
            "cash": readings["cash"],
            "debt": readings["debt"],
            "investments": readings["investments"],
            # One seeded path the screen may post. Not the commit statistic.
            "seeded_net_flows": [float(x) for x in block.net_flow_chf[0]],
        })
    return {
        "snapshot_id": snap.snapshot_id,
        "from_month": int(snap.month),
        "until_month": end,
        "forks": forks,
        "committed": False,
    }


def commit(raw_snapshot: dict, flows: list[float], *, decision: str | None = None, label: str = "",
           cash=None, debt=None, investments=None, engine: RegimeEngine | None = None) -> dict:
    """Append ``flows`` and filter. Does not call ``commit_net_flows``."""
    if not list(flows):
        raise ValueError("flows must be a non-empty list")
    snap = snapshot_from_dict(raw_snapshot)
    observed = list(snap.observed_net_flows) + [float(x) for x in flows]
    belief = _engine(engine).filter(snap.profile, observed, snap.assumption_overrides)
    interventions = list(snap.interventions)
    one_offs = list(snap.one_offs)
    if decision:
        if decision not in DECISIONS and decision not in ("rent_step", "owns_home"):
            raise ValueError(f"Unknown decision kind {decision!r}; choose from {sorted(DECISIONS)}")
        extra, offs = decision_interventions(decision, snap.month, {})
        interventions.extend(extra)
        one_offs.extend(offs)
    child = DecisionSnapshot(
        month=int(snap.month) + len(list(flows)),
        profile=snap.profile,
        belief=belief,
        observed_net_flows=observed,
        interventions=interventions,
        one_offs=one_offs,
        assumption_overrides=dict(snap.assumption_overrides),
        seed=int(snap.seed),
        parent_id=snap.snapshot_id,
        snapshot_id=uuid.uuid4().hex[:12],
    )
    text = label or DECISION_LABELS.get(decision or "", decision or "Path")
    return snapshot_to_dict(child, label=text, cash=cash, debt=debt, investments=investments)


def restore(raw_snapshot: dict, *, engine: RegimeEngine | None = None) -> dict:
    """Replay ``filter`` on the snapshot. The next simulate uses the returned belief."""
    snap = snapshot_from_dict(raw_snapshot)
    belief = _engine(engine).filter(snap.profile, snap.observed_net_flows, snap.assumption_overrides)
    matches = all(abs(float(belief.get(name, 0.0)) - float(snap.belief.get(name, 0.0))) < 1e-6 for name in STATES)
    return {
        "snapshot_id": snap.snapshot_id,
        "month": int(snap.month),
        "belief": {name: float(belief[name]) for name in STATES},
        "matches": matches,
        "cash": raw_snapshot.get("cash"),
        "debt": raw_snapshot.get("debt"),
        "investments": raw_snapshot.get("investments"),
        "label": raw_snapshot.get("label") or "Path",
    }


def life_distribution(raw_snapshot: dict, *, horizon_months: int = 60, n_paths: int = PHONE_N_PATHS,
                      engine: RegimeEngine | None = None) -> dict:
    """Cash fan over the remaining horizon, plus the debt and investments totals."""
    snap = snapshot_from_dict(raw_snapshot)
    horizon = int(horizon_months)
    if horizon <= 0:
        raise ValueError("horizon_months must be positive")
    origin = int(snap.month)
    shifted = [_shift_intervention(iv, origin) for iv in snap.interventions]
    one_offs = []
    for item in snap.one_offs:
        month = int(item.month) - origin
        if 0 <= month < horizon:
            one_offs.append(OneOff(month, float(item.amount_chf), item.label))
    paths = _engine(engine).simulate(
        snap.profile,
        horizon_months=horizon,
        n_paths=n_paths,
        interventions=shifted,
        assumption_overrides=dict(snap.assumption_overrides),
        initial_belief=dict(snap.belief) if snap.belief else None,
        seed=int(snap.seed),
    )
    cash = cash_engine.run(snap.profile, paths, one_offs)
    fan = metrics.fan_quantiles(cash)
    debt = paths.debt_chf[0].tolist() if paths.debt_chf is not None else [0.0] * horizon
    investments = np.median(paths.investments_chf, axis=0).tolist() if paths.investments_chf is not None else [0.0] * horizon
    return {
        "p05": fan["p05"].tolist(),
        "p25": fan["p25"].tolist(),
        "p50": fan["p50"].tolist(),
        "p75": fan["p75"].tolist(),
        "p95": fan["p95"].tolist(),
        "debt": [float(x) for x in debt],
        "investments": [float(x) for x in investments],
        "occupancy": paths.occupancy.tolist(),
        "disclaimer": metrics.DISCLAIMER,
        "horizon_months": horizon,
    }
