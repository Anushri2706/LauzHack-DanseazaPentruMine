"""FastAPI app. All maths lives in app.engine; this layer only translates JSON."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .data.loader import build_persona, load_dataset
from .engine.assumptions import assumption_table

app = FastAPI(title="Many Lives — Financial Twin engine",
              description="Simulation, not a prediction or financial advice. Synthetic data only.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health() -> dict:
    return {"ok": True, "profiles": len(load_dataset().profiles)}


@app.get("/personas")
def personas(status: str | None = None, min_age: int = 0, max_age: int = 200, limit: int = 50) -> list[dict]:
    snap = load_dataset().snapshot
    q = snap[(snap.age_at_start >= min_age) & (snap.age_at_start <= max_age)]
    if status:
        q = q[q.employment_status == status]
    return [build_persona(pid).card() for pid in q.profile_id.head(limit)]


@app.get("/personas/{pid}")
def persona(pid: str) -> dict:
    try:
        return build_persona(pid).card()
    except KeyError:
        raise HTTPException(404, f"Unknown persona {pid}")


@app.get("/assumptions/defaults/{pid}")
def assumption_defaults(pid: str) -> list[dict]:
    try:
        p = build_persona(pid)
    except KeyError:
        raise HTTPException(404, f"Unknown persona {pid}")
    return assumption_table(p.params, p.tags)
