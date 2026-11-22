# Demo script (about 5 minutes)

Open **http://localhost:5173/?fresh&world=9**. `?world=9` fixes the seeded world, so every number below is
reproducible; `?fresh` clears any saved game. Start with `./start.sh`.

All figures are engine outputs under the default assumptions in world #9 (checked against the engine on
2026-11-22, `npm run smoke` replays the same path). They are a simulation, not a prediction.

The demo follows the judges' sequence: person → initial state → intervention → recompute → comparison →
trade-offs and sensitivities → validation.

## 1. Who and what (≈20 s)

- Pick **Alexia, 20, student in Lausanne**: 🟢 CHF 8'000 cash, 🟢 1'200 income, 🟢 900 rent (synthetic
  profile). Her question: *"What if I start working, do a master's, or move abroad?"*
- In the creator, click **🎲 Surprise me** once ("looks never change any number"), then **Walk into town**.
- The student advisor asks *"When do you expect to finish your studies?"*. Point out that the future follows
  the answer (and that she'll be asked again at that date), then click **Not sure yet**: this demo goes
  straight into a master's, so every number below stays the same.
- Point at the HUD badge **"Simulation · not a prediction or advice"** and at **World #9**: "Her luck comes
  from one seeded world. Every reality we try lives in the same world, so differences come from choices."

## 2. Choices with visible prices (≈40 s)

- Click the **University** and choose **Full-time master's**. The preview comes from the engine: *"24 months
  on CHF 1'200/month, tuition CHF 16'000 in total; afterwards salary CHF 4'800/month."* Note the honesty tags
  🟡/🔵. The other big moves grey out: *"You already made one big life move this month: sleep first."*
- Go to the **Café** and set **eating out 2× per week**: *"CHF 340/month · +2 joy/month"*.
- Optional: in the **Homes 4 Sale** agency, the studio is greyed out: *"You need CHF 99'000 for the down
  payment and fees (20% + 2%); you have CHF 8'000…"*.
- At **Home**: **Sleep → next month**. The HUD moves to **Feb 2025** and cash drops to **CHF 2'724**
  (8'000 + 1'200 income − 36 income tax − 900 rent − 1'200 living − 340 habit − 4'000 first tuition instalment).

## 3. A year passes; the world happens; negative cash costs money (≈40 s)

- **Sleep until next birthday** twice.
- Letters arrive as toasts and in the mailbox: **⚠️ Financing needed** (April 2025), then **📬 Rent rise**
  (September 2026, an event from world #9).
- The HUD shows cash in red: **−15'982** in Jan 2026, **−42'682** in Jan 2027, with "⚠ financing".
- Go to the **Bank** statement: income, income tax, rent, living costs, the habit, and **"Interest on negative
  cash −250"**. *"Negative cash is never free: the shortfall is an automatic credit line at 10%/year (🟡, near
  the Swiss consumer-credit cap). Past CHF 30'000 the HUD says 🔴 'beyond any credit line: not financeable as
  planned'. The engine flags it instead of inventing money."*
- Offer **Borrow CHF 10'000**: *"CHF 193/month for 60 months at 6.0%"*. A loan raises cash and debt
  equally (net worth unchanged), and it is cheaper than the 10% overdraft it replaces.

## 4. Another reality (≈40 s)

- Press **P** → **Multiverse**. Tap the first square ("Master's +1") → **Go back** to Jan 2025.
- At the **Office**, choose **Start working in Switzerland**. Sleep until the next birthday ten times.
- At **30 (Jan 2035)** the **Homes 4 Sale** agency unlocks the **Studio**: *"CHF 567'664: pay CHF 113'533 down
  + 11'353 fees now. Each month: interest 863 + amortization 473 + maintenance 473 = 1'809, and no rent.
  Net worth changes only by the fees."* Buy it and sleep: **🏠 You own a home**.

## 5. Compare at 30 and 40 (≈50 s)

- Phone → **Multiverse** → pin **A = Master's**, **B = Work → Buy home**. The compare view opens **at 30**
  (the challenge's 10-year horizon). Chips switch to 40 / 65 / end.
- Same world, same assumptions:

  | | A: Master's + eating out | B: Work → studio |
  |---|---:|---:|
  | Net worth at 30 (world #9) | 7'712 | 164'796 |
  | Range at 30 (P10–P90, other worlds) | ≈ −17k – 8k | ≈ 64k – 124k |
  | Net worth at 40 (world #9) | 97'298 | 398'160 |
  | Range at 40 (P10–P90) | ≈ 57k – 136k | ≈ 376k – 498k |
  | Lowest cash | −42'316 | 0 |
  | Months with negative cash (to 30) | 107 | 0 |
  | Home equity at 30 | 0 | 115'106 |
  | Joy points to 30 (not money) | 266 | 1 |

- Read **"why they differ"** at 30: *the gap comes mostly from investment growth, the eating-out habit and
  income earned; the master's needs financing for 107 months; "Master's" collects more joy points: a game
  preference score, not money.*
- Tap ❤️ on the score tiles (e.g. Joy and Safety) and show **"fit with what matters to you"**: no single
  score wins.
- Bonus insight: world #9 is a *lucky* world for B. Its 165k at 30 lies above the P90 (124k) of other
  worlds. Same choices, different luck.

## 6. What matters most? (≈40 s) — the judges' first question

- Phone → **🌪️ Matters** on reality A, at **40**. Each assumption is moved ±20% on its own, the choices replayed
  with no luck. Top bars:
  - **Salary after a master's**: net worth at 40 from **−113k to +416k** (default 137k).
  - **Living costs once working after the master's**: 294k to −27k.
  - **Rent in Switzerland**: 212k to 60k.
- Switch to reality B: **salary if working now** (96k – 604k), **living costs while working** and **income
  tax** lead.
- Point at the line **"✅ No effect on this reality: Master's tuition, …"** on B: scenario isolation, live.
- *"This is sensitivity, not probability: it tells you which guess to check first, not how likely anything is."*

## 7. Same choices, different luck, and the long run (≈30 s)

- **Money** app on A: the headline range is **at 65**; the 120 line says "no mortality model: if you live
  that long". The **Safety** tile reads **🧓 savings run out after 65 in 89% of worlds (typically at 81)** and
  🔴 beyond the credit line in every world during the degree. On B: savings run out in 17% of worlds.
  B bought the studio at 30 at 568k. Swiss home prices grow 2.35%/year (BIS data), so the mortgage,
  amortization and maintenance are sized to that price, and pension years are tighter.
  *"Eating out twice a week for life plus a later start is joy now against safety later. The model shows
  the trade-off and doesn't pick for you."*
- Toggle **Stress tests** (markets −20%, 6 months without a job, rent +20%, one year from now) and **🎲 Roll
  another world**.
- Press **💱 today's CHF** (timeline or Money app): every value is divided by the engine's price index
  (0.96%/year, the World Bank's Swiss CPI over 1991–2025), so the late decades stop looking like lottery wins.

## 8. Change an assumption (≈20 s)

- **Settings** → *Saving & investing* → **Investment return** 5% → **2%**. Every reality recomputes from
  month 0. B's range at 40 moves from ≈ 376k–498k to ≈ 348k–411k.
- Switch off **Market swings and drops**: the band collapses; that part of the spread came from the market.

## 9. Is the maths right? (≈20 s)

- **Checks** app → **Run live checks**: **13/13**, including
  - cash identity (8'000 + 1'200 − 1'800 = 7'400)
  - saving is a transfer
  - rollback replays exactly
  - **buying a home changes net worth only by fees (exactly 9'000 in every month)**
  - habit price change is isolated
  - the net-worth bridge with housing and habits is exact
  - loans never go below zero
  - **negative cash is never free** (−10'000 at 12% costs exactly 100.00 that month)
  - **job rates match World Bank unemployment** (they imply 4.3%; World Bank 2016–2025 mean 4.6%)
  - **the mortgage costs more than the SNB policy rate** (2.28% vs a 0.53% mean since 2000)
  - **matches the official starter kit** to the cent, for all three scenarios
- Mention: 191 pytest tests for the game engine + 46 for the regime twin, and `npm run smoke` plays this
  whole script headlessly.

Close on the postcard: *"This is one of many possible futures."*

---

## Judge questions: 15-second answers

| Question (scorecard) | Answer | Show |
|---|---|---|
| Which assumptions most affect the result? | For a master's: the salary after it, then living costs and rent. For working: salary, living costs, income tax. Returns matter less before 40. | 🌪️ Matters |
| How does a decision change the state? | Each action edits the state (status, salary, scheduled tuition, habits, home, debt); the month is then simulated in a fixed order; every franc shows up on the statement and in the exact net-worth bridge. | Bank statement, Money → bridge |
| How did you verify the calculations? | 13 live checks incl. a to-the-cent match with the official starter kit and a cross-check against World Bank unemployment; 191 + 46 pytest tests; a headless run of this script. | ✅ Checks |
| What if a constraint is violated? | Impossible actions are refused with the engine's reason (greyed out). Negative cash is allowed but costs 10% interest, raises "financing needed", and beyond CHF 30k is flagged "not financeable as planned". | Bank, HUD |
| What would you improve next? | A mortality model and Swiss supplementary benefits (EL) for old-age shortfalls; a bank affordability test for mortgages; tax by canton; the missing `synthetic-transactions.csv`. | |
| What's fictional vs external? | 🟢 synthetic profile, 🟢 official Swiss rates (BFS, SECO, Eurostat), 🟢 public data from the challenge list (World Bank inflation, SNB mortgage rates, BIS home prices), 🟡 our assumptions, 🔵 engine output. Every assumption is tagged in Settings. | ⚙️ Settings |
| What stops the AI inventing numbers? | The LLM may only write sentences with `{fact_id}` placeholders; a guard rejects its own digits, unknown ids and certainty language; code fills in engine numbers. | 💬 Ask |
| Why do 🧠 Learned twin and 📐 Rules differ? | The dotted line is your own path either way (what you'll live, and what the ✨ notes say). The fans answer different questions: the learned twin predicts **liquid savings** from behaviour learned on 958 synthetic people (backtested, no investment returns, as in the data); Rules projects **net worth** under the stated assumptions, with returns and housing. | Timeline switch |
