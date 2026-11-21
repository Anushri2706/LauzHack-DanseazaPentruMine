# Danseaza Pentru Mine — financial life simulation backend

> **Backend only.** This directory contains no frontend. A separate frontend (default
> `http://localhost:5173`) connects to the FastAPI JSON API under `/api` described below.

Enter a financial profile, branch your life into alternative decisions (master's, move canton, part-time,
child, ...), simulate each branch as a Monte Carlo distribution, compare branches, inspect every assumption
and ask an AI companion to explain the results — without the AI being able to invent or change any number.

The engine is the `feature/combined-engine` HMM regime engine (formerly `backend/regime`), migrated into
`game/backend/app/engine/` and regression-tested to produce **bit-identical** output for the same seed.
The original `backend/` and `twin/` folders were removed after commit `33470ff`; their outputs are frozen in
`tests/fixtures/legacy_reference.json`. See [docs/AUDIT.md](docs/AUDIT.md) for the
Stage 0 audit and migration map.

> Simulated outcomes are conditional on stated assumptions and a seed. They are not predictions,
> guarantees or financial advice. **Storage is in memory: every game is lost when the server restarts.**

## Setup and run

Requires Python ≥ 3.11 (tested on 3.15.2).

```bash
cd game/backend
python -m venv .venv && .venv/bin/pip install -r requirements.txt    # or reuse the repo's .venv
.venv/bin/uvicorn app.main:app --port 8000
```

The API is at http://localhost:8000/api (interactive docs at http://localhost:8000/docs). The synthetic data is read from `<repo>/Data` (override with `GAME_DATA_DIR`).

LLM (optional): the AI layer reads `LLM_PROVIDER` and `GEMINI_API_KEY` / `GROQ_API_KEY` from the environment
or the repo's git-ignored `.env` (same convention as the former `twin/`). Without a key, or with `GAME_DISABLE_LLM=1`,
the companion uses the deterministic rule-based extractor and template explanations. Calls are capped per
process (`LLM_MAX_CALLS`, default 40). Only the message text (extraction) or evidence facts (explanation) are
sent — no profile identifiers.

CORS allows `http://localhost:5173` and `http://127.0.0.1:5173` only — the separate frontend's dev origin
(override with a comma-separated `GAME_CORS_ORIGINS`).

## Tests

```bash
cd game/backend
../../.venv/bin/python -m pytest -q
```

Result on 2026-11-22: see the "Verification" section at the bottom. Tests never call a real LLM
(`tests/conftest.py` sets `GAME_DISABLE_LLM=1`).

| File | What it checks |
|---|---|
| `test_regression.py` | Engine == the original `backend/regime` + `twin` outputs frozen in `fixtures/legacy_reference.json` (emission fit, HMM paths for 7 intervention sets × 3 starting regimes, stub, net pay, wealth layer, full `TwinSession` pipeline, peer distances). `regression_cases.py` holds the shared inputs; `fixtures/make_legacy_reference.py` regenerates the file from commit `33470ff` |
| `test_engine.py` | Same seed identical; seeds differ; no caller mutation; cash and net-worth reconciliation month by month; debt/portfolio constraints; stochastic transition matrices; horizon; percentile order; job loss + one-off in the same month; event time-basis conversion, eligibility, windows; events cannot re-declare HMM risks; enabling events does not change regimes |
| `test_decisions_assumptions.py` | Decision → intervention mapping identical to `TwinSession`; validation errors; registry provenance and engine consistency; override versioning and stochastic constraints; sensitivity eligibility, ±10%, clamping; bisection root / no-root / undefined |
| `test_branches.py` | Inheritance; parent and sibling isolation; staleness of descendants only; versioned results never overwritten; fixed-seed replay; fresh seed labelled and not comparable; break-even bracketed and unbracketed |
| `test_ai.py` | The 10 adversarial cases from the brief, hostile LLM replies (invented values, unknown fields, unsupported decisions, fabricated claims, certainty and causal text), claim-guard semantic checks |
| `test_acceptance.py` | The 14-step acceptance scenario through HTTP; structured API errors; sensitivity endpoint; peer band |

## Architecture

```
game/
├── backend/app/
│   ├── main.py, api.py, schemas.py, dependencies.py     FastAPI (thin; no maths)
│   ├── game_service.py, storage.py                      branches, results, replay, compare (no HTTP)
│   ├── engine/            (imports no FastAPI / HTTP / LLM code)
│   │   ├── types.py        Profile, Intervention, RegimePaths           (migrated unchanged)
│   │   ├── transitions.py  sourced monthly hazards + Dirichlet          (migrated)
│   │   ├── regime.py       HMM emissions, filter, path sampler          (migrated)
│   │   ├── income.py, tax.py   gross pay / UI / housing, gross→net      (migrated)
│   │   ├── accounting.py, wealth.py   cash, inflation, portfolio, debt  (migrated; +optional event flow)
│   │   ├── metrics.py      metric definitions (unit, basis, denominator)     (new)
│   │   ├── assumptions.py  provenance registry, versioned overrides     (new)
│   │   ├── profile.py      player facts → engine Profile                (new)
│   │   ├── decisions.py    typed decision catalogue + validation        (new; mapping from TwinSession)
│   │   ├── events.py       independent events, time-basis conversion    (new)
│   │   ├── simulate.py     orchestration, monthly order, trace          (new)
│   │   ├── montecarlo.py   P10/P50/P90 result contract, comparison      (new)
│   │   └── sensitivity.py  ±10% one-at-a-time, bisection break-even     (new)
│   ├── data/               synthetic loaders, canton rent index, assumptions/*.csv (registry v1)
│   ├── peers/bands.py      synthetic peer band (from the former twin/peers.py)
│   └── ai/                 llm, prompts, extract, validate, explain, claim_guard, companion
└── docs/AUDIT.md           Stage 0 audit and migration map
```

### Monthly order (as computed)

1. decisions dated this month take effect (pins, attribute changes, one-offs);
2. regime: drawn from the previous month's row of the age-specific transition matrix unless pinned; retirement rule;
3. independent events eligible in the new regime (own random stream);
4. income for the regime (gross → net), raises at year ends, UI clock;
5. recurring spending (sampled spend/rent ratio × rent) and December bonus;
6. market: price level (yearly inflation-regime switch), portfolio return;
7. cash flow = kept net flow − one-offs × price − event costs × price − loan interest;
8. cash, portfolio (debt constant: interest-only);
9. net worth = cash + portfolio − debt; real = ÷ price level.

The migrated engine is vectorised; a per-month `step_month` API was **not** added because it would be a
second engine. `run_branch` (one path) and `run_monte_carlo` (n paths) exist, and `SimulationRun.trace(i)`
gives any path's month-by-month record (also `GET /api/game/{id}/branches/{bid}/trace`).

### Randomness and common random numbers

Explicit generators only: HMM `default_rng(seed)`, market `default_rng(10_000 + seed)` (both unchanged from the
original), events `SeedSequence([seed, 0xE7E7, i])`. All branches of a game use the game seed, so they share
the Dirichlet transition draws, the market stream and the event stream exactly. The HMM month stream shares
the seed but realigns only statistically after a decision pins months (pinned months consume no draws).
Comparisons are therefore *differences of distribution statistics* (e.g. difference of medians), not
path-wise effects. A `fresh_seed` run is labelled and refused by `compare`.

### Branches

A branch stores parent id, name, fork month, its own append-only decisions, its own assumption overrides and a
pin flag. Its effective history is its ancestors' decisions then its own. Every result records seed, config,
assumptions version, the decisions it used and an input fingerprint; a result is **stale** when the branch's
current fingerprint differs (a decision added to a parent makes descendants stale, never siblings). Results
are versioned and never overwritten. `replay` re-runs a stored result's exact inputs and compares output
fingerprints.

### Assumptions and provenance

`GET /api/assumptions` lists all 113 registry rows. Status is derived from the original CSV text by fixed rules:
`cited` (a named publication/law is cited in the CSV, value **not re-verified** — never called "official"),
`estimated` ([EST] / internal estimation notes), `illustrative` (prior, hand-set, modelling choice, synthetic),
`placeholder`. `source_url` and `last_verified` are `null` everywhere: no URL or verification was invented.
Every result lists the assumptions it actually used (a decision default is listed only when the default was used)
with the value used and `user_override` where overridden.

### AI layer

* **Extraction** (`ai/extract.py`, `ai/validate.py`): the LLM (or the rule extractor) proposes facts with
  field, value, currency, period, gross/net, subject and span. Validation rejects unknown fields, wrong types,
  out-of-range values, facts about other people and **numbers that do not appear in the message**; asks one
  concise question when period / gross-net / currency is missing; asks for confirmation of unusual values.
  Accepted facts are only *copied into the profile form*; the profile changes when the user submits it.
* **Decisions**: the AI proposes a decision id + params; the backend validates (exists, schema, units,
  eligibility, start month); the user must click "Confirm" which calls the normal decisions endpoint.
* **Explanations** (`ai/claim_guard.py`): evidence = facts with stable ids from completed results, the exact
  comparison, sensitivity rows and flagged assumptions. The LLM may only return claims citing fact ids; the guard
  rejects fabricated ids, wrong claim type, wrong direction, unit/basis/horizon/value mismatches, numbers in
  free text, certainty and causal language. Accepted claims are rendered from templates with backend values.
  If nothing passes (or no LLM), a deterministic template explanation is used. Rejections are logged by index,
  fact id and reason only (no raw conversation).

## API for the separate frontend

All routes are under `/api`; errors are `{"error": str, "errors": [{code, field, message}]}`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness |
| GET | `/assumptions` | registry rows with provenance status |
| GET | `/catalogue/events`, `/catalogue/metrics` | independent events; metric definitions |
| POST | `/game/start` | validate a profile, create a game with a root branch |
| GET | `/game/{id}`, `/game/{id}/decisions` | game state; decision catalogue |
| GET / POST | `/game/{id}/branches` | list branches; fork a child (parent, name, fork month) |
| GET / PATCH | `/game/{id}/branches/{bid}` | one branch; rename / pin |
| POST | `/game/{id}/branches/{bid}/decisions` | append a decision (validated) |
| POST | `/game/{id}/branches/{bid}/assumptions` | versioned assumption override |
| POST | `/game/{id}/branches/{bid}/simulate` | Monte Carlo run (P10/P50/P90, risk metrics) |
| GET | `/game/{id}/branches/{bid}/results`, `/trace` | stored results (with staleness); one path month by month |
| POST | `/game/{id}/branches/{bid}/replay` | re-run a stored result from its seed and config |
| POST | `/game/{id}/compare` | difference of distribution statistics between two branches |
| POST | `/game/{id}/sensitivity`, `/break-even` | ±10% one-at-a-time; bisection break-even |
| GET | `/game/{id}/peers` | synthetic peer band |
| POST | `/ai/extract`, `/ai/explain`, `/ai/chat` | AI companion (extraction validated, claims guarded) |

Full request/response schemas: `/docs` (OpenAPI) or `app/schemas.py`. The demo flow (profile → branches →
decisions → simulate → compare → AI → replay → fork at month 30) is exercised over HTTP in
`tests/test_acceptance.py`.

## Brief checklist — status

| Item | Status |
|---|---|
| `game/` architecture | done |
| Combined-engine functionality reused and regression-tested | done (bit-identical tests) |
| Engine runs without FastAPI or an LLM | done (`engine/` imports neither) |
| Decisions and independent events separated | done; HMM-owned risks cannot be re-declared as events |
| Reproducible from recorded seed/config | done (replay endpoint + tests) |
| Branches immutable and isolated | done (frozen values, append-only decisions, staleness) |
| Monte Carlo documented distributions and risk metrics | done (`engine/metrics.METRICS`) |
| Sensitivity and break-even with validity / convergence | done |
| Assumptions with provenance and visible status | done (`GET /api/assumptions`; status per row) |
| AI cannot silently fill missing values / change state | done (tests) |
| Explanations checked against result metrics; unsupported claims rejected | done (claim guard + fallback) |
| Frontend | removed: backend only; a separate frontend uses the API (every number comes from the backend) |
| End-to-end acceptance scenario | done (`test_acceptance.py`) |

## Known limitations

* **In-memory storage**; max 200 games per process, oldest evicted.
* `step_month` per-month API not provided (see above). NSGA-II trade-off search (`backend/regime/explore.py`),
  offline validation reports (`backend/regime/validation.py`) and the twin calibration / chat CLI were not
  migrated; they exist only in git history (commit `33470ff`).
* Income tax is one estimated effective-rate curve for a single person in Lausanne (VD), used for every canton
  and household (warned on every result). Abroad keeps Swiss tax and CHF (no FX assumption).
* The synthetic ledger has no regime switches: transition hazards are assumptions.
* Debt is interest-only; savings stay in cash (no contributions to the portfolio).
* `keep_factor` defaults to 1.0 (no saving calibration). The twin calibration found ≈0.16 against the synthetic
  wealth table; with 1.0 simulated wealth is likely high relative to that table.
* `self_employed` applies wider spending swings from month 0 (original engine behaviour); no income premium.
* `family` changes housing and leave only; child costs and household type are not modelled.
* The `unexpected_major_expense` event uses placeholder probability and cost and is off by default.
* Reserve risk uses a fixed nominal CHF reserve (it erodes in real terms with inflation).
* The emergency reserve default is `buffer_months` × (rent + other expenses given).
* Simulations run synchronously in the request (≈0.05–0.5 s for 500 × 120; sensitivity / break-even take longer).
* The claim guard checks structure, values and wording; it cannot rule out every misleading framing an LLM could
  write, which is why free LLM text is limited to one number-free framing sentence.

## Verification (2026-11-22)

| Suite | Command | Result |
|---|---|---|
| game | `cd game/backend && ../../.venv/bin/python -m pytest -q` | 114 passed, 0 failed, 0 skipped after the frontend was removed (1 Starlette deprecation warning about httpx in TestClient) |

Engine import boundary: `grep` finds no FastAPI / Starlette / Pydantic / HTTP / AI imports under `app/engine/`.
Before the frontend was removed, a manual browser check of the full flow worked against the live server; the
same flow is now covered over HTTP by `tests/test_acceptance.py`.
One live LLM extraction call (Gemini, synthetic sentence) went through the same validation and rejected a
partner's salary.

Observation from the demo profile: the synthetic peer band at age 36 (net worth P25 ≈ CHF 40k, P50 ≈ CHF 67k,
P75 ≈ CHF 130k in the synthetic table) is far below the simulated median at the 10-year horizon (≈ CHF 374k
real) — the same gap the twin calibration found. It is
shown, not hidden; lowering `keep_factor` is the documented lever.
