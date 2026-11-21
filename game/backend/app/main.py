"""FastAPI app. All maths lives in app.engine; this layer only translates JSON."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from . import hmm, service
from .data.loader import build_persona, load_dataset
from .engine.assumptions import assumption_table
from .engine.decisions import InvalidDecision
from .models import (BranchRequest, BranchResult, CompareRequest, GameProjectionRequest, GameRequest,
                     GameRollRequest, MonteCarloRequest, WorldRequest)
from .validation import run_checks

DISCLAIMER = "Simulation, not a prediction or financial advice. Synthetic data only."

app = FastAPI(title="Many Lives — Financial Twin engine", description=DISCLAIMER)
app.add_middleware(CORSMiddleware, allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
                   allow_methods=["*"], allow_headers=["*"])


def _guard(fn, *a):
    try:
        return fn(*a)
    except KeyError as e:
        raise HTTPException(404, f"Unknown persona {e}")
    except (InvalidDecision, ValueError) as e:
        raise HTTPException(422, str(e))


@app.get("/health")
def health() -> dict:
    return {"ok": True, "profiles": len(load_dataset().profiles), "disclaimer": DISCLAIMER}


@app.get("/personas")
def personas(status: str | None = None, min_age: int = 0, max_age: int = 200, limit: int = 60,
             random: bool = False, seed: int = 0) -> list[dict]:
    snap = load_dataset().snapshot
    q = snap[(snap.age_at_start >= min_age) & (snap.age_at_start <= max_age)]
    if status:
        q = q[q.employment_status == status]
    ids = q.profile_id.sample(min(limit, len(q)), random_state=seed).tolist() if random else q.profile_id.head(limit).tolist()
    return [build_persona(pid).card() for pid in ids]


@app.get("/personas/{pid}")
def persona(pid: str) -> dict:
    return _guard(lambda: build_persona(pid).card())


@app.get("/assumptions/defaults/{pid}")
def assumption_defaults(pid: str) -> list[dict]:
    def f():
        p = build_persona(pid)
        return assumption_table(p.params, p.tags)
    return _guard(f)


@app.post("/branch/options")
def branch_options(req: BranchRequest) -> dict:
    return _guard(service.options_for, req)


@app.post("/branch/simulate", response_model=BranchResult)
def branch_simulate(req: BranchRequest) -> dict:
    return _guard(service.simulate, req)


@app.post("/branch/montecarlo")
def branch_montecarlo(req: MonteCarloRequest) -> dict:
    return _guard(service.montecarlo, req)


@app.post("/branch/world")
def branch_world(req: WorldRequest) -> dict:
    return _guard(service.world, req)


@app.post("/compare")
def compare(req: CompareRequest) -> dict:
    return _guard(service.compare_branches, req)


@app.post("/game/state")
def game_state(req: GameRequest) -> dict:
    """Current month, state, HUD scores, actions per place (masked, with previews), letters, statement."""
    return _guard(service.game_state, req)


@app.post("/game/projection")
def game_projection(req: GameProjectionRequest) -> dict:
    """'No surprises' path from now, P10–P90 bands over the seeded worlds, stress-test ghosts, bridge."""
    return _guard(service.game_projection, req)


@app.post("/game/roll")
def game_roll(req: GameRollRequest) -> dict:
    """Same choices, a different world from now on."""
    return _guard(service.game_roll, req)


@app.get("/catalogue/{pid}")
def catalogue(pid: str) -> dict:
    return _guard(service.catalogue, pid)


@app.get("/validation")
def validation() -> list[dict]:
    return run_checks()


# ============================================================== regime HMM twin (teammate's backend/twin)
class HmmFanRequest(BaseModel):
    persona_id: str
    decisions: list[str] = Field(default_factory=lambda: ["stay"])
    horizon_months: int = 240
    n_paths: int = 2000
    seed: int = 0
    start_month: int = 0


@app.get("/hmm/decisions/{pid}")
def hmm_decisions(pid: str) -> dict:
    """Big life decisions the regime HMM can simulate for this persona (do() on the regime chain)."""
    return _guard(hmm.decisions_for, pid)


@app.post("/hmm/fan")
def hmm_fan(req: HmmFanRequest) -> dict:
    """Cash fans (P10/P50/P90), regime occupancy and goal hit rates per decision, same seed for all."""
    def f():
        goals = list(build_persona(req.persona_id).goals)
        return hmm.fan(req.persona_id, req.decisions, req.horizon_months, req.n_paths, req.seed,
                       req.start_month, goals)
    return _guard(f)
