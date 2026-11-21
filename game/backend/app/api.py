"""HTTP endpoints. Thin: parse with Pydantic, call GameService / ai, return JSON. No financial maths here.

Simulations run synchronously in the request (about 0.05-0.5 s for 500 paths x 120 months; sensitivity and
break-even run several simulations and can take a few seconds). There is no background job queue.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from .ai.companion import explain_branches, respond
from .ai.extract import extract
from .dependencies import get_service
from .engine.events import CATALOGUE as EVENTS, ENGINE_INTERNAL_SHOCKS, HMM_OWNED_RISKS
from .engine.metrics import METRICS
from .engine.simulate import SimulationConfig
from .game_service import GameService
from .schemas import (BreakEvenIn, ChatIn, CompareIn, CreateBranchIn, DecisionIn, ExplainIn, ExtractIn, OverridesIn,
                      ReplayIn, SensitivityIn, SimulateIn, StartGameIn, UpdateBranchIn)
from .storage import Conflict

router = APIRouter(prefix="/api")
Svc = Depends(get_service)


@router.get("/health")
def health(svc: GameService = Svc):
    return {"status": "ok", "storage": "in-memory (lost on restart)",
            "assumption_registry_version": svc.base_assumptions.version}


@router.get("/assumptions")
def assumptions(svc: GameService = Svc):
    return {"version": svc.base_assumptions.version,
            "assumptions": [a.to_dict() for a in svc.base_assumptions.registry.rows.values()]}


@router.get("/catalogue/events")
def events():
    return {"events": [e.to_dict() for e in EVENTS.values()], "hmm_owned_risks": HMM_OWNED_RISKS,
            "engine_internal_shocks": ENGINE_INTERNAL_SHOCKS}


@router.get("/catalogue/metrics")
def metrics():
    return METRICS


@router.post("/game/start", status_code=201)
def start_game(body: StartGameIn, svc: GameService = Svc):
    cfg = SimulationConfig(horizon_months=body.config.horizon_months, n_paths=body.config.n_paths,
                           inflation=body.config.inflation, enabled_events=tuple(body.config.enabled_events))
    game = svc.start(body.profile.model_dump(), cfg, body.seed)
    return svc.game_view(game)


@router.get("/game/{game_id}")
def get_game(game_id: str, svc: GameService = Svc):
    return svc.game_view(svc.repo.game(game_id))


@router.get("/game/{game_id}/decisions")
def decision_catalogue(game_id: str, svc: GameService = Svc):
    return {"decisions": svc.decision_catalogue(svc.repo.game(game_id))}


@router.get("/game/{game_id}/branches")
def list_branches(game_id: str, svc: GameService = Svc):
    g = svc.repo.game(game_id)
    return {"branches": [svc.branch_view(g, b) for b in g.branches.values()]}


@router.post("/game/{game_id}/branches", status_code=201)
def create_branch(game_id: str, body: CreateBranchIn, svc: GameService = Svc):
    b = svc.create_branch(game_id, body.parent_id, body.name, body.fork_month)
    return svc.branch_view(svc.repo.game(game_id), b)


@router.get("/game/{game_id}/branches/{branch_id}")
def get_branch(game_id: str, branch_id: str, svc: GameService = Svc):
    return svc.branch_view(svc.repo.game(game_id), svc.repo.branch(game_id, branch_id))


@router.patch("/game/{game_id}/branches/{branch_id}")
def update_branch(game_id: str, branch_id: str, body: UpdateBranchIn, svc: GameService = Svc):
    b = svc.update_branch(game_id, branch_id, body.name, body.pinned)
    return svc.branch_view(svc.repo.game(game_id), b)


@router.post("/game/{game_id}/branches/{branch_id}/decisions", status_code=201)
def add_decision(game_id: str, branch_id: str, body: DecisionIn, svc: GameService = Svc):
    d = svc.add_decision(game_id, branch_id, body.decision_id, body.start_month, body.params)
    return {"decision": d.to_dict(), "branch": svc.branch_view(svc.repo.game(game_id), svc.repo.branch(game_id, branch_id))}


@router.post("/game/{game_id}/branches/{branch_id}/assumptions")
def set_assumptions(game_id: str, branch_id: str, body: OverridesIn, svc: GameService = Svc):
    b = svc.set_overrides(game_id, branch_id, body.overrides)
    return svc.branch_view(svc.repo.game(game_id), b)


@router.post("/game/{game_id}/branches/{branch_id}/simulate")
def simulate(game_id: str, branch_id: str, body: SimulateIn | None = None, svc: GameService = Svc):
    res = svc.simulate(game_id, branch_id, body.fresh_seed if body else None)
    return svc.result_view(svc.repo.game(game_id), res)


@router.get("/game/{game_id}/branches/{branch_id}/results")
def results(game_id: str, branch_id: str, version: int | None = None, svc: GameService = Svc):
    g = svc.repo.game(game_id)
    res = svc.latest(game_id, branch_id, version)
    if res is None:
        raise Conflict("no result for this branch (simulate first)" if version is None else f"no version {version}")
    out = svc.result_view(g, res)
    out["versions"] = [{"version": r.version, "result_id": r.result_id, "run_kind": r.run_kind, "seed": r.seed,
                        "created_at": r.created_at, "stale": svc.is_stale(g, r)} for r in svc.repo.results(game_id, branch_id)]
    return out


@router.get("/game/{game_id}/branches/{branch_id}/trace")
def trace(game_id: str, branch_id: str, path: int = 0, svc: GameService = Svc):
    """Month-by-month record of one future (single-path replay with the game seed)."""
    from .engine.simulate import run_branch
    g = svc.repo.game(game_id)
    A = svc.effective_assumptions(g, branch_id)
    ep = svc._engine_profile(g, A)
    run = run_branch(ep, list(svc.effective_decisions(g, branch_id)), A, g.config, g.seed + int(path),
                     svc._engine(g, ep), svc._canton_ratio(g))
    return {"seed": g.seed + int(path), "trace": run.trace(0)}


@router.post("/game/{game_id}/branches/{branch_id}/replay")
def replay(game_id: str, branch_id: str, body: ReplayIn | None = None, svc: GameService = Svc):
    return svc.replay(game_id, branch_id, body.result_id if body else None)


@router.post("/game/{game_id}/compare")
def compare(game_id: str, body: CompareIn, svc: GameService = Svc):
    return svc.compare(game_id, body.a, body.b)


@router.post("/game/{game_id}/sensitivity")
def sensitivity(game_id: str, body: SensitivityIn, svc: GameService = Svc):
    out = svc.sensitivity(game_id, body.branch_id, body.assumption_ids, body.metric, body.rel_change, body.n_paths)
    latest = svc.latest(game_id, body.branch_id)
    out["result_id"] = latest.result_id if latest else None
    return out


@router.post("/game/{game_id}/break-even")
def break_even(game_id: str, body: BreakEvenIn, svc: GameService = Svc):
    return svc.break_even(game_id, body.a, body.b, body.assumption_id, body.lower, body.upper, body.metric)


@router.get("/game/{game_id}/peers")
def peers(game_id: str, years: int = 10, svc: GameService = Svc):
    """Synthetic peer band (cross-sectional, not a forecast). Shown separately from simulated results."""
    from .peers.bands import peer_band
    g = svc.repo.game(game_id)
    ep = svc._engine_profile(g, svc.base_assumptions)
    if not 1 <= years <= 40:
        raise ValueError("years must be in [1, 40]")
    return peer_band({"age": g.profile.age, "canton": g.profile.canton, "household": g.profile.household,
                      "income": ep.net_monthly_income_chf, "rent": g.profile.rent_chf_month}, years)


@router.post("/ai/extract")
def ai_extract(body: ExtractIn):
    return extract(body.message, use_llm=body.use_llm)


@router.post("/ai/explain")
def ai_explain(body: ExplainIn, svc: GameService = Svc):
    out = explain_branches(svc, body.game_id, body.branch_id, body.compare_with, body.question, body.use_llm, body.draft)
    if out is None:
        raise Conflict("no fresh result to explain: simulate the branch(es) first")
    return out


@router.post("/ai/chat")
def ai_chat(body: ChatIn, svc: GameService = Svc):
    return respond(body.message, svc, body.game_id, body.branch_id, body.compare_with, body.use_llm)
