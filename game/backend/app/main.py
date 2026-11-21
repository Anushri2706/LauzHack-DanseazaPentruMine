"""FastAPI app. All maths lives in app.engine; this layer only translates JSON."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import service
from .data.loader import build_persona, load_dataset
from .engine.assumptions import assumption_table
from .engine.decisions import InvalidDecision
from .models import (BranchRequest, BranchResult, CompareRequest, GameProjectionRequest, GameRequest,
                     GameRollRequest, HmmBranchRequest, HmmCommitRequest, HmmDistributionRequest,
                     HmmRestoreRequest, HmmStartRequest, MonteCarloRequest, WorldRequest)
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


@app.post("/hmm/start")
def hmm_start(req: HmmStartRequest) -> dict:
    """Root rollback snapshot for the phone branch tree."""
    return _guard(service.hmm_start, req.persona_id)


@app.post("/hmm/branch")
def hmm_branch(req: HmmBranchRequest) -> dict:
    """Side-by-side forks. Does not commit a snapshot."""
    return _guard(service.hmm_branch, req)


@app.post("/hmm/commit")
def hmm_commit(req: HmmCommitRequest) -> dict:
    """Append the caller's flows, filter, and return the new snapshot."""
    return _guard(service.hmm_commit, req)


@app.post("/hmm/restore")
def hmm_restore(req: HmmRestoreRequest) -> dict:
    """Replay filter on a rollback snapshot so the phone can jump to it."""
    return _guard(service.hmm_restore, req)


@app.post("/hmm/distribution")
def hmm_distribution(req: HmmDistributionRequest) -> dict:
    """Final life distribution: cash fan, debt, investments, disclaimer."""
    return _guard(service.hmm_distribution, req)


@app.get("/validation")
def validation() -> list[dict]:
    return run_checks()
