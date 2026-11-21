# Many Lives — a branching Financial Twin game

UBS × LauzHack 2026 "AI Financial Twin". *Don't predict the future. Make it explorable.*
**This is a simulation, not a prediction or financial advice. All data is synthetic.**

## Progress (for the team)

The plan is in [`../BUILD_PROMPT.md`](../BUILD_PROMPT.md) (6 milestones), and the challenge summary is in [`../TASK.md`](../TASK.md).

| Milestone | Status |
|---|---|
| 1. Backend scaffold + deterministic engine + tests | ✅ done |
| 2. React frontend, branch endpoints, timeline tree, rollback, compare view | ⏳ next (**no web UI yet**) |
| 3. Random events, Monte Carlo bands, ghost lines | — |
| 4. Assumptions drawer, sensitivity, break-even | — |
| 5. kNN peer bands | — |
| 6. More personas, scoring, LLM narrator | — |

### What exists now (Milestone 1)

```
game/backend/
  app/engine/finance.py      monthly↔annual rates, inflation factor, loan annuity (0% safe)
  app/engine/assumptions.py  every assumption: label, unit, bounds, source tag; Alexia defaults
  app/engine/state.py        immutable State + Decision
  app/engine/decisions.py    decision catalogue, valid choices per state (action masking), effects
  app/engine/simulate.py     step_month() + run_branch(): pure replay from month 0
  app/engine/metrics.py      summary metrics + goal checks
  app/data/loader.py         reads Data/*.csv (read-only), builds any of the 1,200 personas
  app/main.py                FastAPI: /health, /personas, /personas/{id}, /assumptions/defaults/{id}
  app/cli.py                 prints Alexia's branches in the terminal
  tests/                     29 pytest checks
```

The **tests** cover cash identity (8,000 + 1,200 − 1,800 = 7,400), transfers leaving net worth unchanged, loan amortization, determinism and exact rollback, scenario isolation, tuition monotonicity, edge cases (zero income, 0% interest, negative returns, monthly/annual conversion), and invalid decisions. They also replay **`starter-kit/financial_twin.py` month by month for all 3 scenarios**, and the cash matches to the cent.

Alexia's results with default assumptions (`python -m app.cli`):

| Branch | Net worth at 30 | Min cash | Months cash < 0 |
|---|---:|---:|---:|
| Start working | 226'651 | 9'100 | 0 |
| Full-time master's | 169'638 | −30'084 | 38 |
| Move abroad | 148'806 | 2'300 | 0 |
| Master's, then abroad at 22 | 115'515 | −35'313 | 48 |

With the CSV tuition (8,000/year) the master's needs about 30k of financing, which makes a good demo moment. The values are higher than the starter kit's because we add a 5% return and 2% raises. This is the "outcomes depend on assumptions" point for the pitch.

## Setup

```bash
cd game/backend
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pytest -q                       # validation suite
.venv/bin/python -m app.cli               # Alexia's branches in the terminal
.venv/bin/uvicorn app.main:app --reload   # API on http://127.0.0.1:8000 (docs at /docs)
```

## Model (monthly, in this order)

| Step | Equation |
|---|---|
| 1. Decision effects | status, salary, rent, living costs, scheduled one-offs (tuition, relocation), loans |
| 2. Income | `income = salary × (1 − tax_rate − social_rate)`; raise of `raise_annual` every 12 months in the same job |
| 3. Expenses | `expenses = (rent + living) × (1 + inflation)^(t/12)` |
| 4. Debt | `interest = debt × r/12`, `principal = payment − interest`, `debt = max(0, debt − principal)`; annuity payment, 0% handled |
| 5. Cash | `cash += income − expenses − one_offs − debt_payment`; if `cash < 0`, sell investments to cover (a transfer) |
| 6. Saving | if `cash > buffer_months × expenses`, move `savings_rate × (cash − buffer)` to investments (a transfer) |
| 7. Returns | `investments × (1 + r_month)`, with `(1 + r_month)^12 = 1 + return_annual` |
| 8. Outputs | `net_worth = cash + investments − debt`; buffer months; `cash < 0` ⇒ flag **financing needed** (not game over) |

**Branches** are `(persona, assumptions, decisions[])`. The engine replays from month 0, so it is a pure function and rollback is exact. **Decision points** fall at month 0, every birthday (12 months), and when a degree finishes. Only valid choices are accepted (action masking, with a 12-month cooldown on big moves).

## Design choices (made without asking; easy to change)

- Starter-kit expenses are **totals including rent**. We split them into rent (900, from the profile) plus other living costs, so the totals stay the same (studying 2,100, working 3,000, after a master's 3,100, abroad 2,900, where the 1,100/1,800 split is ours).
- Starter-kit salaries are **net**, so tax and social rates default to 0 to avoid double taxation.
- Tuition defaults to **8,000/year** from `scenario-assumptions.csv`, paid per semester. The starter kit uses 2 × 2,500, and the cross-check test uses those values. A part-time master's costs the same total tuition, spread over 36 months.
- A master's raises the abroad salary by the same ratio as in Switzerland (4,800/4,200).
- Goals are checked in priority order against cash + investments at their target date. Money for a goal already met is earmarked, so the same francs are never counted twice.
- Existing debt (a snapshot with no payment) is amortized over `loan_term_months` (60).
- Other personas: rent and income come from the profile; living costs by household type are **our assumptions** (single 1,600, shared 1,400, couple 2,600, family 3,600).

## Open questions for organizers

- `synthetic-transactions.csv`, `life-events.csv` and `alexia-profile.json` are listed but missing.
- Slide 14 (Master's ≈112k > Abroad ≈95k > Work ≈93k) contradicts the starter kit (Work > Master's > Abroad). Which assumptions produced slide 14?
- Tuition: is it 8,000/year (CSV) or 2 × 2,500 (starter kit)?
- Final rubric, deadline and demo length.
