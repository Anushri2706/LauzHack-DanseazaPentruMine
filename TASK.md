# Task Brief — UBS × LauzHack 2026: AI Financial Twin

> My working notes. Sources: `docs/Lauzhack 2026 - AI Financial Twin Challenge*.pdf`, `faq/`, `evaluation/`, `starter-kit/`, `Data/`.

## One-liner
Build, in **24 hours**, a prototype "Financial Twin": a fictional person's financial state + a model that
recomputes it when a **life decision / event / assumption** changes, so the user can **compare futures and understand trade-offs**.

**Motto: "Don't predict the future. Make it explorable."** — conditional what-if, NOT forecasting, NOT financial advice.

## Headline use case (from the PDF)
**Alexia**, 20, student in Lausanne, CHF 8,000 starting buffer, values financial flexibility, horizon **ages 20–30**.
She asks: *"What if I start working, do a master's, or move abroad?"*
Income and living costs are **scenario-dependent** (we must choose and disclose them).
Slide 14 shows the target experience: 3 savings paths over 10 years (illustrative end values: Master's ~112k, Abroad ~95k, Work ~93k) with labelled events (first salary, study costs, relocation costs, savings recover).
Other suggested life events: buy a home, retire earlier, market shock (scenario catalog has 120 ideas; we do NOT need all).

## Requirements (slide 17 "What success looks like")
| | Item |
|---|---|
| MUST | Fictional client profile |
| MUST | One working scenario |
| MUST | Before / after (baseline vs alternative) comparison |
| MUST | Transparent assumptions |
| BONUS | Conversational AI |
| BONUS | Sensitivity and uncertainty |

"Success is helping someone understand a decision — not just displaying numbers."

## Suggested architecture (slide 16, not mandatory)
1. **Synthetic profile** – income, spending, assets, goals
2. **Scenario engine** – cash flow, investment assumptions, life events (deterministic maths)
3. **AI assistant** – interprets questions, explains calculations (*"let AI handle language; let a financial model handle the maths"*)
4. **Visualization** – baseline vs alternatives vs uncertainty

Four building blocks: **Data** (what's true today?) → **Model** (how does it behave?) → **Scenarios** (what if?) → **Explain** (what does it mean?).
Any tech allowed: sim, Monte Carlo, LLM, game, RL, agents, optimization, 3D...

## Judging (draft rubric, 100 pts)
| Criterion | Weight |
|---|---:|
| Financial modelling & correctness | 25 |
| Scenario & counterfactual exploration | 20 |
| Technical execution & validation | 15 |
| UX & communication | 15 |
| Innovation & conceptual depth | 15 |
| Responsible design & transparency | 10 |

Using an LLM gives no points by itself; demonstrated, correct capability does.
Judges will likely ask: Which assumptions matter most? How does a decision change state? How did you verify? What if a constraint is violated? What next?

## Validation the model must respect (`evaluation/technical-validation.md`)
- Cash identity: `C[t+1] = C[t] + income − expenses − one-offs` (e.g. 8000 + 1200 − 1800 = 7400)
- Net worth: `NW = Assets − Liabilities`; moving cash→investments does **not** change NW (before fees/returns)
- Loan: principal repayment reduces cash **and** debt; interest is an expense; principal never goes negative
- Scenario isolation: change one assumption, only the expected outputs move
- Edge cases: zero income, 0% rate, negative returns, higher tuition never lowers cost, **negative cash → flag financing need**, monthly/annual conversion consistent, missing input → labelled assumption
- Avoid: gross/net confusion, double-counted tax, investments counted as spending, snapshot ≠ transaction history
- If stochastic: document distributions, seeds, sample count; percentiles ≠ real probabilities
- LLM text must match engine output

## Demo flow (suggested)
1. Person + decision → 2. Initial state → 3. Intervention → 4. Recompute → 5. Baseline vs alternative →
6. Trade-offs / sensitivities / limits → 7. One reproducibility / consistency check live.
Repo: README with setup, model description, assumptions, reproducible scenario, limitations.
Deadline / platform / demo length: **TBC by organizers**.

## Responsible-AI rules
- Synthetic / licensed public data only, no real client data, no committed API keys.
- Label clearly: synthetic inputs vs assumptions vs model outputs vs uncertainty.
- No guarantees, no personalised investment advice; higher net worth ≠ "best" life choice.

## What we have in `Data/`
| File | Rows | Notes |
|---|---:|---|
| `synthetic-profiles.csv` | 1200 | P00001 = Alexia. Age, city/canton, employment, household, opening cash, ref. monthly net income & rent |
| `synthetic-accounts.csv` | 1200 | One cash account per profile |
| `synthetic-assets.csv` | 2399 | 1200 cash + 1199 investment portfolios (P00001 has none). Snapshot at 2025-01-01. Checked: no profile is missing cash |
| `synthetic-liabilities.csv` | 283 | Personal loans only; snapshot, `monthly_payment_chf = 0`, not amortized |
| `synthetic-cash-balances.csv` | 28800 | 24 months (2025-01 → 2026-12) per profile; 772 months flagged `negative_balance_requires_financing` |
| `synthetic-goals.csv` | 1500 | Emergency savings, housing deposit, education, retirement, etc. with target & date |
| `scenario-assumptions.csv` | 240 | Only S001 (full-time master's): study duration + annual tuition for 120 profiles |
| `scenario-catalog.csv` | 120 | Idea list: baseline vs intervention, by category |

Listed in README but **missing**: `synthetic-transactions.csv`, `life-events.csv`, `alexia-profile.json`.
Also: `docs/scenario-library-120.csv`, `starter-kit/` (stdlib Python: 120-month cash-only sim of Alexia's 3 scenarios, with tests).

## Starter kit recap
`starter-kit/financial_twin.py`: monthly deterministic cash sim, phases `[start,end)` covering 120 months,
compares final cash & minimum cash, flags negative months. Cash only: no investments, debt, tax, inflation, FX, uncertainty.
Run: `python starter-kit/run_example.py`, tests: `python -m unittest discover -s starter-kit -p "test_*.py"`.
→ Natural extension points: tax, investments/returns, loans, inflation, Monte Carlo bands, LLM explainer, interactive UI.

## Minimum credible demo (our floor)
Alexia profile → 3 scenarios (work / master's / abroad) over 10 years → chart of cash + net worth per path →
assumptions panel editable → explanation of differences → negative-cash / goal-feasibility flags → 1 passing accounting test.
Then bonuses: LLM chat that calls the engine, sensitivity sliders / Monte Carlo bands.
