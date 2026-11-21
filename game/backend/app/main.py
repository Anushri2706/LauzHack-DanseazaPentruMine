"""FastAPI application: API routes, structured errors, CORS for a separate frontend.

Backend only: no frontend is served. Run (from game/backend):
    ../../.venv/bin/uvicorn app.main:app --port 8000
The separate frontend (default http://localhost:5173) calls the JSON API under /api.
"""
from __future__ import annotations

import logging
import os

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api import router
from .engine.assumptions import AssumptionError
from .engine.decisions import DecisionError
from .engine.montecarlo import ResultError
from .engine.profile import ProfileError
from .engine.simulate import SimulationError
from .storage import Conflict, NotFound

DEV_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"

logging.basicConfig(level=os.environ.get("GAME_LOG_LEVEL", "INFO"), format="%(levelname)s %(name)s %(message)s")


def _err(status: int, error: str, errors: list[dict] | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": error, "errors": errors or []})


def create_app() -> FastAPI:
    app = FastAPI(title="Danseaza Pentru Mine — financial life simulation", version="0.1.0")
    origins = [o.strip() for o in os.environ.get("GAME_CORS_ORIGINS", DEV_ORIGINS).split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST", "PATCH"],
                       allow_headers=["Content-Type"])
    app.include_router(router)

    @app.exception_handler(ProfileError)
    @app.exception_handler(DecisionError)
    async def _validation(_: Request, exc):
        return _err(422, "validation_failed", exc.errors)

    @app.exception_handler(AssumptionError)
    async def _assumption(_: Request, exc: AssumptionError):
        return _err(422, "invalid_assumption", [exc.to_dict()])

    @app.exception_handler(RequestValidationError)
    async def _request(_: Request, exc: RequestValidationError):
        errs = [{"code": e.get("type", "invalid"), "field": ".".join(str(x) for x in e.get("loc", [])[1:]),
                 "message": e.get("msg", "")} for e in exc.errors()]
        return _err(422, "invalid_request", errs)

    @app.exception_handler(NotFound)
    async def _nf(_: Request, exc: NotFound):
        return _err(404, "not_found", [{"code": "not_found", "field": exc.what, "message": str(exc.args[0])}])

    @app.exception_handler(Conflict)
    @app.exception_handler(ResultError)
    async def _conflict(_: Request, exc):
        return _err(409, "conflict", [{"code": "conflict", "field": None, "message": str(exc)}])

    @app.exception_handler(SimulationError)
    @app.exception_handler(ValueError)
    async def _value(_: Request, exc):
        return _err(422, "invalid_input", [{"code": "invalid_input", "field": None, "message": str(exc)}])

    return app


app = create_app()
