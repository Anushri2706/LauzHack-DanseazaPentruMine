# Danseaza Pentru Mine — AI financial twin (LauzHack 2026)

This repository holds the **backend** for the financial life simulation. The frontend lives in a separate
repository and talks to this backend over HTTP.

| Folder | What it is |
|---|---|
| [`game/backend/`](game/backend) | **The backend.** FastAPI JSON API under `/api`: HMM life-regime engine, income, tax, cash, wealth, decisions, branches, Monte Carlo, sensitivity, AI companion with claim guard |
| [`Data/`](Data) | Synthetic challenge data (CSV) the engine is fitted on |
| `docs/`, `evaluation/`, `examples/`, `faq/`, `resources/`, `starter-kit/` | Challenge material from the organisers (not code) |

Setup, API reference, architecture and limitations: **[game/README.md](game/README.md)**.

```bash
cd game/backend
../../.venv/bin/uvicorn app.main:app --port 8000     # API at http://localhost:8000/api, docs at /docs
../../.venv/bin/python -m pytest -q                   # tests
```

History: the engine was built from two earlier prototypes, `backend/` (HMM regime engine) and `twin/`
(calibration, peer checks, trade-off search, chat CLI). They were merged, moved into `game/backend`, and
removed after commit `33470ff`; check out that commit to see them.
