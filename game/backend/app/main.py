"""FastAPI app. All maths lives in app.engine; this layer only translates JSON."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from . import service
from .data.loader import build_persona, load_dataset
from .engine.assumptions import assumption_table
from .engine.decisions import InvalidDecision
from .models import (BranchRequest, BranchResult, CompareRequest, GameProjectionRequest, GameRequest,
                     GameRollRequest, MonteCarloRequest, SensitivityRequest, WorldRequest)
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
    from .ai.outcome import with_compare_story
    return _guard(lambda: with_compare_story(service.compare_branches(req), req))


@app.post("/game/state")
def game_state(req: GameRequest) -> dict:
    """Current month, state, HUD scores, actions per place (masked, with previews), letters, statement."""
    return _guard(service.game_state, req)


class AskRequest(BaseModel):
    request: GameRequest
    message: str = Field(min_length=1, max_length=500)


@app.post("/ai/ask")
def ai_ask(req: AskRequest) -> dict:
    """The companion: answers with facts the engines computed; may propose one allowed action."""
    from .ai.companion import respond
    return _guard(lambda: respond(req.message, req.request))


@app.post("/ai/outcome")
def ai_outcome(req: GameProjectionRequest) -> dict:
    """One reality as a short plain-language story (LLM-written, guarded, rule fallback)."""
    from .ai.outcome import outcome_story
    return _guard(outcome_story, req)


@app.post("/ai/compare")
def ai_compare(req: CompareRequest) -> dict:
    """Two realities side by side in plain language (same guard and fallback as /ai/outcome)."""
    from .ai.outcome import compare_story
    return _guard(compare_story, req)


@app.post("/ai/timeline")
def ai_timeline(req: GameProjectionRequest) -> dict:
    """Insights pinned to moments on the life timeline: the code picks the moments, the AI explains them."""
    from .ai.timeline import cached
    return _guard(cached, req.model_dump_json())


@app.post("/ai/twin")
def ai_twin(req: GameProjectionRequest) -> dict:
    """The learned twin in plain words, and why it can differ from your own timeline (AI-written, guarded, rule fallback)."""
    from .ai.twin import cached
    return _guard(cached, req.model_dump_json())


@app.post("/twin/fan")
def twin_fan(req: GameProjectionRequest) -> dict:
    """The learned twin from the player's state now: HMM life states x money behaviour learned from data."""
    return _guard(service.twin_fan, req)


@app.get("/twin/model")
def twin_model() -> dict:
    """Model card: backtest on the locked test people (12 months known, next 12 predicted)."""
    return _guard(service.twin_model)


@app.post("/game/projection")
def game_projection(req: GameProjectionRequest) -> dict:
    """'No surprises' path from now, P10–P90 bands over the seeded worlds, stress-test ghosts, bridge."""
    from .ai.outcome import with_outcome_story
    return _guard(lambda: with_outcome_story(service.game_projection(req), req))


@app.post("/game/sensitivity")
def game_sensitivity(req: SensitivityRequest) -> dict:
    """Tornado: which assumptions move net worth at the target age most (one at a time, ±swing, no luck)."""
    return _guard(service.sensitivity, req)


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
