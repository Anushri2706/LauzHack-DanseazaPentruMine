# Build Prompt — "Many Lives": a branching Financial Twin game

> Paste everything below this line into a new Claude Code chat opened in `/Users/parmida/Desktop/LauzHack/LauzHack-DanseazaPentruMine`.

---

You are building our entry for the **UBS × LauzHack 2026 "AI Financial Twin" challenge** (24-hour hackathon). Read `TASK.md` in the repo root first; it summarises the challenge. The source materials are in `docs/` (incl. the challenge PDF), `evaluation/`, `faq/`, `starter-kit/` and `Data/`. Skim `evaluation/technical-validation.md` and `evaluation/responsible-ai-and-data.md` before coding; judges will test against them.

## 1. What the challenge wants (non-negotiable)

Motto: **"Don't predict the future. Make it explorable."** This is conditional what-if exploration, NOT forecasting and NOT financial advice.

MUST:
1. A fictional client profile
2. One working scenario (a decision that recomputes the financial state)
3. A before/after (baseline vs alternative) comparison
4. Transparent assumptions

BONUS: conversational AI; sensitivity and uncertainty.

Draft judging rubric (100 pts): financial modelling & correctness 25, scenario & counterfactual exploration 20, technical execution & validation 15, UX & communication 15, innovation 15, responsible design & transparency 10. An LLM earns nothing by itself. Higher net worth must NOT be presented as the "best" life.

Known gaps in the challenge materials (design around them, don't depend on them):
- `synthetic-transactions.csv`, `life-events.csv` and `alexia-profile.json` are listed in the README but missing. `data-dictionary.md` is a copy of the README.
- `scenario-assumptions.csv` covers only S001 (master's: duration + tuition).
- Liabilities are snapshots with `monthly_payment_chf = 0` (not amortized); we must add amortization ourselves.
- Slide 14 of the PDF (Master's ≈112k > Abroad ≈95k > Work ≈93k at 30) contradicts the starter kit's own numbers (Work ≈152k > Master's ≈145k > Abroad ≈97k). Use this in the pitch: outcomes depend on assumptions. That's why we show sensitivity and break-even points.
- The rubric is a draft and the deadline/demo length are TBC. Keep a list of open questions in the README.

## 2. The product

A browser game in which the player picks a **persona** and lives their next ~10 years through **decision points**. At any point they can **roll back** and choose differently, creating **parallel realities** (a timeline tree). They compare branches side by side. Behind it runs a transparent, tested **mathematical simulation**. Uncertainty is shown throughout, so it always feels like "one of many possible futures," never a prediction.

Default persona: **Alexia** (profile P00001): 20, Lausanne (VD), single, student, CHF 8,000 cash, CHF 1,200/month income, CHF 900 rent, horizon ages 20–30. Her question: *"What if I start working, do a master's, or move abroad?"* Other personas are loaded from `Data/synthetic-profiles.csv` (1,200 fictional profiles). Their decision options depend on age and status (e.g. a retiree gets different choices than a student).

## 3. Tech stack

- **Backend: Python 3.11+ with FastAPI.** All the maths lives here: engine, Monte Carlo (NumPy), sensitivity, kNN peers. It reads the CSVs in `Data/` directly with pandas at startup. Do not modify the original CSVs.
- **Frontend: TypeScript + React + Vite**, with Recharts (or visx) for charts. UI only: no financial maths in the frontend.
- Folders: `backend/` and `frontend/` (in a new `game/` folder).
- Tests: **pytest** for the backend (the core of the validation), and a few Vitest/Playwright smoke tests for the frontend if time allows.
- Dependencies: `fastapi`, `uvicorn`, `pydantic`, `numpy`, `pandas`, `pytest`. Pin versions in `requirements.txt`, and use a `.venv`.
- Optional LLM narrator as a backend endpoint behind a feature flag and env key (`.env`, git-ignored; never commit keys). The game must fully work without it.

## 4. Architecture

```
game/
  backend/
    app/
      main.py          FastAPI app, CORS for the Vite dev server
      api.py           endpoints (below)
      models.py        Pydantic schemas: State, Decision, Event, Assumptions, BranchRequest, BranchResult
      engine/          pure simulation — no FastAPI imports
        simulate.py      step_month(), run_until_next_decision(), run_branch()
        decisions.py     decision catalogue + validity rules per persona/state
        events.py        random life events with probabilities and impacts
        montecarlo.py    N runs (NumPy) → P10/P50/P90 per month
        sensitivity.py   ±10% tornado ranking + break-even finder (bisection)
        assumptions.py   defaults + source tags
      data/            CSV loading, persona building
      peers/           kNN "people like you" bands
    tests/             pytest
    requirements.txt
  frontend/
    src/
      api/             typed client for the backend
      timeline/        branch tree, rollback, pin/compare (UI state only)
      ui/              screens and components
```

**Stateless backend, deterministic replay.** A branch is defined entirely by `(persona_id, assumptions, seed, decisions: [(decision_point, choice), ...])`. The frontend holds the timeline tree (each node = that decision list). To roll back, it truncates the list and sends a new branch request; the backend replays from month 0. The engine is a pure function, so same inputs + same seed ⇒ identical output, and rollback is exact by construction. No server-side sessions or database needed. Cache results with an LRU cache keyed on the request hash if speed matters.

**Endpoints (initial):**

| Method | Path | Purpose |
|---|---|---|
| GET | `/personas`, `/personas/{id}` | List/load personas from the synthetic CSVs (🟢 starting state, goals) |
| GET | `/assumptions/defaults/{persona_id}` | Default assumptions with value, unit, source tag |
| POST | `/branch/options` | Valid decisions at the current decision point (action masking) |
| POST | `/branch/simulate` | Run a branch → monthly path, flags, goal status, next decision point |
| POST | `/branch/montecarlo` | P10/P50/P90 bands + ghost scenarios for a branch |
| POST | `/compare` | Two branches → metric table + data for a plain-language explanation of the differences |
| POST | `/sensitivity` | Tornado ranking + break-even values |
| GET | `/peers` | kNN peer band by age/status for a persona |
| GET | `/validation` | Runs the live consistency checks for the demo screen |
| POST | `/narrate` (flagged) | LLM narration from engine JSON only |

## 5. The mathematical model

**Two clocks.** Money moves **monthly**. The player decides only at **decision points**: start, each birthday/year end, and when an event fires (degree finished, job loss, offer received). Expect ~10–15 decisions per 10-year run. Use action masking: only valid choices are shown, and there are cooldowns (no starting and quitting a master's every month).

**State** (per month): `t, age, cash, investments, debt, status (studying|working|abroad|unemployed|retired), grossSalary, rent, livingCosts, savingsRate, goals[], flags`.

**Monthly update, in this order:**
1. Apply the pending decision/event effects (status, salary, rent, one-off costs such as tuition or relocation).
2. `netIncome = grossSalary × (1 − taxRate − socialRate)`, with flat, labelled rates (avoid gross/net confusion and double-counted tax).
3. `expenses = (rent + livingCosts) × (1 + inflation)^(t/12)`
4. Debt: `interest = debt × rDebt/12` is an expense; `principal = payment − interest`; `debt = max(0, debt − principal)`.
5. `cash += netIncome − expenses − oneOffs − debtPayment`
6. Saving rule: cash above the buffer target (e.g. 3 months of expenses) × savingsRate moves to investments. This is a **transfer**: net worth unchanged.
7. `investments ×= (1 + r_month)`
8. Outputs: `netWorth = cash + investments − debt`, buffer months = `cash / expenses`, goal checks, and `cash < 0` ⇒ flag "financing needed" (a visible warning, not a game over).

**Decision catalogue (initial):**

| Decision | Effects |
|---|---|
| Start working | status working, salary = S_work, Lausanne rent |
| Full-time master's (24 months) | studying, part-time income, tuition per semester; afterwards salary = S_master |
| Part-time master's | reduced salary + tuition, longer duration |
| Move abroad | relocation one-off; salary/rent from that country's assumptions; fixed, stated FX rate |
| Take loan / repay loan | cash ± L, debt ± L, payment schedule |
| Change savings rate | savingsRate |
| (later) Buy home, change job, reduce hours | as defined in assumptions |

**Alexia default assumptions** (all editable in an Assumptions panel, each with a source tag):

| Assumption | Value | Tag |
|---|---|---|
| Starting cash / income / rent | 8,000 / 1,200 / 900 | 🟢 profile (synthetic data) |
| Net salary, work now | 4,200/month | 🟡 starter kit |
| Net salary after master's | 4,800/month | 🟡 starter kit |
| Living costs | 2,100 studying / 3,000 working | 🟡 starter kit |
| Tuition | 8,000/year | 🟡 `scenario-assumptions.csv` |
| Relocation | 6,500 | 🟡 starter kit |
| Abroad | 3,700 income / 2,900 expenses | 🟡 starter kit |
| Return / inflation / raise | 5% / 1% / 2% per year | 🟡 our assumption |

Where the starter kit uses net figures, keep them net and set tax to 0 for those; don't double-tax.

## 6. Uncertainty ("it's only a simulation — here's what could be")

- **Random life events** with stated probabilities and impacts (job loss for 3–9 months, rent rise, market drop, scholarship, salary shock), all listed in the Assumptions panel.
- **Monte Carlo:** each branch runs N = 500 times on the backend (vectorize across runs with NumPy). Show P10/P50/P90 fan bands, labelled "range under our assumptions", never "probability". Use a fixed seed set so that **all branches see the same random worlds** (scenario isolation: differences come from decisions, not luck).
- **"What could be" ghost lines:** faded alternative paths such as "if the market drops 20% at 25" or "if unemployed 6 months". Clicking one turns it into a real branch.
- **"What drives this" panel:** a tornado chart (each assumption ±10%, ranked by effect on net worth at 30) plus **break-even statements**, e.g. "If your post-master salary is below CHF X/month, working now ends higher." Sliders recompute live.
- **Honesty tags on every number:** 🟢 from profile · 🟡 assumption · 🔵 simulated · ⚪ peer reference.
- A persistent badge, "Simulation, not a prediction or financial advice", and an end-of-run screen: "This is one of many possible futures."

## 7. Peer reference bands (kNN "people like you")

Data reality (already checked):
- Strict matches are too few: VD + single + employed + ages 25–30 gives only 4 people.
- Relaxed matching works: any canton, single, employed/part-time, 25–30 gives 38; adding shared housing and ages 24–32 gives 66.
- Their net worth (cash + investments − debt) at 25–30 ranges roughly −15k (P10) to 67k (P90), median ≈34k; ~30% have debt.

Implementation:
- `findPeers(persona, age, status, k=30)` uses a weighted distance: `w_age·|Δage|/5 + w_emp·[emp≠] + w_hh·[household≠] + w_region·[region≠] + w_inc·|Δlog income|`. Canton gets the lowest weight; it's nearly uniform in the data, so group cantons into cost regions. Show the weights in the Assumptions panel.
- For each age 21–30, match peers of that age (±1) in **the status the current branch is in** (working branch → employed peers, study branch → students). Plot the P10/P50/P90 net worth band as a faded ⚪ layer behind the player's path.
- Use the **opening snapshot** (`synthetic-assets` + `synthetic-liabilities` at 2025-01-01), NOT `synthetic-cash-balances.csv`. Its cash grows implausibly fast (~25k → ~110k in 24 months) while investments stay flat.
- Caveats to show in the UI:
  - This is cross-sectional: "where similar fictional people are today, not where you will be."
  - Alexia is an outlier: the other students hold 22–54k vs her 8k.
  - The data is synthetic.
  - Correlation is not causation.
- Use peers as a **sanity check** on our simulation. If a branch ends far above peer P90, prompt the player to inspect the assumptions.
- Do NOT train a model that "predicts" Alexia's future from other people's outcomes. The data has no life histories, it's synthetic, and that would contradict the challenge.

## 8. Game UX

1. **Persona select:** Alexia first; "random persona" or browse by age/status. Each card shows 🟢 starting data.
2. **Life view:** a timeline with the current age, a chart of cash/net worth with uncertainty bands, the peer band, goal markers, and short narrative cards at each decision point. ("Alexia, 22. Your master's is done. An offer in Zurich pays 5,100 but rent is 1,900…")
3. **Decision card:** 2–4 valid choices, each with a one-line preview of the immediate effect. No preview of the "best" option.
4. **Timeline tree / multiverse map:** every branch is visible, abandoned ones faded. Clicking any past node rolls back. The player can pin two branches.
5. **Compare view (the MUST before/after):** two branches side by side with net worth, cash, minimum cash, months below buffer, months with negative cash, goals reached, and a plain-language explanation of *why* they differ ("the master's cost ~CHF 20k of buffer by 22; it catches up around 28 because…"). Explanations are generated from engine numbers.
6. **Scoring:** no single net-worth score. Show a multi-dimensional profile (safety/buffer, goals met, flexibility/free time, wealth). The player can choose what matters to them at the start; show how that would change which branch "fits" them.
7. **Assumptions drawer:** every assumption with value, unit, source tag and an edit control. Editing re-runs all branches.
8. **Optional LLM narrator** (flagged): it only receives engine JSON and must never invent numbers. Add a check that every number in its text appears in the engine output.

## 9. Validation (write these tests FIRST, pytest)

1. Cash identity: 8,000 + 1,200 − 1,800 = 7,400 for a single month.
2. A cash→investment transfer leaves net worth unchanged.
3. Loan: principal repayment reduces cash and debt; interest is an expense; debt never goes negative.
4. Same seed ⇒ identical branch; rollback + replay ⇒ an exactly identical state.
5. Scenario isolation: changing one assumption changes only the expected outputs.
6. Monotonicity: higher tuition never yields higher cash.
7. Edge cases: zero income (cash falls, negative-cash flag fires), 0% interest (no divide-by-zero), negative returns, monthly↔annual conversion consistent.
8. Starter-kit cross-check: with uncertainty off and the starter-kit inputs, our engine reproduces the final cash from `starter-kit/financial_twin.py` for all 3 scenarios. Import it directly in the test; it's stdlib Python.
9. Peers: kNN returns k results; widening filters never shrinks the set.
10. API: the `/branch/simulate` response round-trips through the Pydantic schemas, and replaying a truncated decision list equals the earlier node's state.

Also expose a "Validation" screen in the app that runs a few checks live (useful for the demo's "show one consistency check" step).

## 10. Build order (stop and show me a running demo after each milestone)

1. Scaffold `game/backend` (FastAPI + data loading) and the engine + pytest tests (deterministic, no randomness). Starter-kit cross-check passing. The engine is usable from a CLI/notebook before any UI exists.
2. `game/frontend` + endpoints. Alexia: 3–4 decision points, timeline tree, rollback, compare view. **This alone satisfies all four MUSTs, so make it solid.**
3. Seeded random events + Monte Carlo bands + ghost lines.
4. Assumptions drawer, tornado/sensitivity, break-even statements, honesty tags.
5. kNN peer bands.
6. More personas, multi-dimensional scoring, polish, optional LLM narrator.

## 11. Deliverables in the repo

- `game/README.md`: setup (backend: `python -m venv .venv && pip install -r requirements.txt && uvicorn app.main:app --reload`, `pytest`; frontend: `npm install && npm run dev`), a one-command start script if easy, model description with equations, full assumption table with sources, how to reproduce the Alexia scenario, limitations, open questions for organizers, data attribution.
- A demo script (`game/DEMO.md`) following the suggested demo flow: person + decision → initial state → intervention → recompute → baseline vs alternative → trade-offs/sensitivity/limits → one live validation check.

## 12. Rules

- Synthetic data only; no real personal data; no committed secrets.
- Never present outputs as predictions, probabilities of real life, or advice.
- Keep the engine small, readable and auditable. Prefer correctness over feature count: one polished branching story beats ten shallow ones.
- Don't build a copy of any existing commercial game; this is an original design.
- Ask me before big scope changes. Otherwise make sensible choices, note them in the README, and keep going.
