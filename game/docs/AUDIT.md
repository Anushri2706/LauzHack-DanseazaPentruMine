# Stage 0 — Repository audit and migration plan

> Historical: this audit describes `backend/` and `twin/` as they were before the migration. Both folders
> were removed after commit `33470ff`; `game/backend` is now the only backend.

Audit date: 2026-11-22. Inspected on branch `feature/combined-engine` (HEAD `54c3024`) with the
working-tree changes present at the time (11 modified files under `twin/`, `backend/requirements.txt`,
untracked `twin/limits.py`). Nothing in `backend/` or `twin/` was modified by this work.

## 1. Where the two named sources live

| Name in brief | What it actually is |
|---|---|
| `feature-combined-engine` | The current branch `feature/combined-engine`. Two packages: `backend/regime/` (HMM regime engine, importable as `regime`) and `twin/` (profile tagging, calibration, peers, NSGA-II explore front-end, LLM chat). `twin/engine.py` puts `backend/` on `sys.path` and routes everything through `regime.TwinSession`. |
| `feature-hmm-where` | Remote branch `origin/feature/hmm-where` (tip `4dfeae2`). `git merge-base --is-ancestor` confirms it is **already an ancestor of HEAD** (merged via `feature/hmm-decisions`). Its content (Zurich/abroad housing multipliers, `run_alexia_decisions_20y_where.py`) is present in `backend/regime/income.py`. On that branch the package was still called `backend/twin/`; the rename to `backend/regime/` happened on the combined branch. No separate source needs porting. |

## 2. Existing test results (executed, not assumed)

| Suite | Command (from repo root unless noted) | Result |
|---|---|---|
| backend | `cd backend && ../.venv/bin/python -m pytest -q -p no:cacheprovider` | **46 collected, 46 passed**, 0 failed, 0 skipped (6.8 s) |
| twin | `.venv/bin/python -m pytest -q -p no:cacheprovider twin/test_engine.py` | **35 collected, 35 passed**, 0 failed, 0 skipped (2.7 s) |

Environment: `.venv` Python 3.15.2, numpy 2.5.5, pandas 3.1.1, scipy 1.18.1, pytest 9.2.0. FastAPI,
uvicorn and httpx were **not installed**; they were added to `.venv` for the new API (fastapi 0.165.0,
pydantic 2.14.4). Node.js/npm are **not installed** on this machine, so the frontend is written as
build-free ES modules served by FastAPI (no bundler).

## 3. Actual execution path (combined engine)

```
twin/__main__.py / twin/chat.py  ──►  twin/agents.py (LLM understand/explain, regex number check)
            │                                  │
            ▼                                  ▼
       twin/engine.py ── builds Profile, personalises emissions, maps twin decision keys
            │
            ▼
 backend/regime/twin_session.py  TwinSession (scenarios, apply_decision, simulate, compare, sensitivity, explore)
            │
            ├─► regime_engine.RegimeEngine.simulate
            │      ├─ transitions.TransitionPriors (sourced monthly hazards, Dirichlet draws, age bands)
            │      ├─ sample_regime_paths (pins, scale_hazard, long-term-unemployment decay, retirement at 65)
            │      ├─ income.IncomeTable.income_paths (gross; UI; experience growth) ─► net_pay (AHV/ALV/NBU/BVG/tax)
            │      ├─ sample_emissions (Student-t spend/rent ratio) + December bonus
            │      └─ RegimePaths(states, net income, net flow, ...)
            ├─► wealth.build (wealth=True)  inflation regimes, portfolio returns, loan interest, keep factor
            │   or cash_engine.run (wealth=False)
            └─► metrics.compute_metrics (fan quantiles, reserve risk, goals)

backend/regime/agent.py — a SECOND agent pipeline: OpenAI function-calling loop over TwinSession tools.
```

**Duplicated execution paths found**

1. Two LLM pipelines: `backend/regime/agent.py` (OpenAI tool-calling, model narrates tool JSON) and
   `twin/agents.py` + `twin/chat.py` + `twin/llm.py` (Gemini/Groq/Ollama via OpenAI-compatible HTTP,
   structured `understand` → code `validate`, `explain` → regex `check_numbers`). Only the second is
   used by the CLI. → The game keeps **one** pipeline (`app/ai/`), reusing the `twin/` approach
   (structured extraction + code validation, stdlib HTTP client) and retiring the tool-calling loop.
2. Two decision vocabularies: `TwinSession.DECISIONS` (14 kinds incl. raw interventions) and
   `twin/engine.add_decision` (6 keys with defaults from `twin/assumptions.py`). → Unified into one
   typed catalogue `engine/decisions.py`; raw interventions (`force_state`, `scale_hazard`,
   `set_attribute`) are **not** exposed to players or the LLM.
3. Two sensitivity implementations: `TwinSession.sensitivity` (±25%, ranks by closing cash, no
   probability validity checks) and none in `twin/`. → Replaced by `engine/sensitivity.py` (±10%,
   eligibility, probability clamps, provenance).
4. Legacy engine `twin/sim.py` still present for its own tests and `validate_learned.py` only. Not migrated.
5. Assumptions in three places: `backend/assumptions/*.csv` (sourced, with notes), `twin/assumptions.py`
   (illustrative dict), `twin/calibration.json` (fitted keep factors). → One registry
   (`engine/assumptions.py`) loaded from versioned copies of the CSVs plus the few `twin/assumptions.py`
   decision defaults actually used.

## 4. Migration inventory

| Source | Destination | Action | Notes |
|---|---|---|---|
| `backend/regime/types.py` | `engine/types.py` | reuse unchanged (+ new domain types) | `Profile`, `Intervention`, `RegimePaths`, `STATES` byte-for-byte semantics |
| `backend/regime/regime_engine.py` | `engine/regime.py` | move | HMM emissions, forward filter, path sampling. Only import paths changed |
| `backend/regime/transitions.py` | `engine/transitions.py` | move | kept separate from `regime.py` for cohesion (prior table vs sampler) |
| `backend/regime/income.py` | `engine/income.py` | move | gross pay, UI, experience growth, housing |
| `backend/regime/net_pay.py` | `engine/tax.py` | move | unchanged maths; warnings ([EST]) surface via the registry |
| `backend/regime/cash_engine.py` | `engine/accounting.py` | move | `CashPaths`, `OneOff`, `reconciles` |
| `backend/regime/wealth.py` | `engine/wealth.py` | move + extend | adds an optional per-path `event_costs` flow (zero by default ⇒ identical output) |
| `backend/regime/metrics.py` | `engine/metrics.py` | move + extend | adds metric metadata (units, basis, denominators) and P10/P50/P90 result contract |
| `backend/regime/twin_session.py` | `engine/decisions.py` + `engine/simulate.py` + `app/branches.py` | refactor | decision → intervention mapping preserved exactly; mutable scenario dict replaced by immutable branches |
| `backend/regime/data.py` | `app/data/synthetic.py` + `app/data/assumptions/*.csv` | move | CSVs copied as registry version `1`; originals untouched |
| `backend/regime/explore.py` (NSGA-II) | — | **not migrated (documented)** | It is tested only through the twin CLI and is optimisation, not sensitivity. Original remains usable in `backend/regime`. |
| `backend/regime/validation.py` | — | not migrated | Offline validation reports (starter-kit equivalence, KS). Original remains. |
| `backend/regime/agent.py` | — | retired for the game | Competing agent pipeline (see §3). Original untouched. |
| `twin/peers.py` | `app/peers/` | move (read-only adapter) | Peer bands from the synthetic population, labelled synthetic |
| `twin/agents.py` | `app/ai/extract.py`, `validate.py`, `explain.py` | refactor | Extraction schema now carries unit/period/gross-net/source/span/status; regex check replaced by claim guard |
| `twin/llm.py` | `app/ai/llm.py` | reuse | stdlib OpenAI-compatible client; key from env |
| `twin/calibration.py` keep factor | registry assumption `keep_factor` (default 1.0) | replaced | Calibration fit not re-run; default = no calibration, labelled illustrative |
| `twin/sim.py`, `twin/learned.py`, `twin/scenarios.py` | — | retire | legacy engine, not on the current path |
| `starter-kit/` | — | not used | reference accounting only |

## 5. Known model limitations (carried into the game)

* The synthetic ledger has **no regime switches**; transition hazards are assumptions (sourced or [EST]), not learned.
* Income tax is one effective rate curve for **a single person in Lausanne (VD)**, marked `[EST]`. It is applied
  regardless of the canton the user enters; the registry carries geography `CH-VD` and the API returns a warning.
* Self-employment has no sourced income premium: only wider spending swings and a setup cost.
* The `family` decision does not change household type in the emission model (twin audit, step 4 partial).
* Debt is interest-only (principal never amortises). Portfolio has no contributions; savings stay in cash.
* `keep_factor` calibration: the twin found median keep ≈16% to match the synthetic wealth table; the game
  defaults to 1.0 (no calibration) and exposes it as an illustrative assumption.
* Inflation and portfolio return assumptions are illustrative (not a licensed series).

## 6. Staged plan

0. Audit (this file).
1. Engine package: move modules, keep outputs **bit-identical** to `backend/regime` for the same seed (regression tests import both).
2. Decisions (typed catalogue + validation) and events (separate from HMM; HMM-owned risks cannot be re-declared).
3. Monte Carlo result contract (P10/P50/P90 per month and final), metric metadata, sensitivity (±10%), break-even (bisection with bracketing).
4. Immutable branch model + in-memory repository (fork, decision, simulate, replay, compare, staleness).
5. FastAPI app with typed schemas and structured errors.
6. AI layer: extraction schema, validation, decision mapping, claim guard with templated fallbacks, adversarial tests.
7. Frontend: build-free ES modules (profile, decisions, timeline, dashboard, AI companion). **Removed 2026-11-22: `game/` is now backend only; a separate frontend uses the API.**
8. End-to-end acceptance test and README.
