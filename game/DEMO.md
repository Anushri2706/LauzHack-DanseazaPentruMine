# Demo script (about 4 minutes)

Open **http://localhost:5173/?fresh&world=9**. `?world=9` fixes the seeded world, so every number below is
reproducible; `?fresh` clears any saved game. Start with `./start.sh`.

All figures are engine outputs under the default assumptions in world #9. They are a simulation, not a
prediction.

## 1. Who and what (≈20 s)

- Pick **Alexia, 20, student in Lausanne**: 🟢 CHF 8'000 cash, 🟢 1'200 income, 🟢 900 rent (synthetic
  profile). Her question: *"What if I start working, do a master's, or move abroad?"*
- In the creator, click **🎲 Surprise me** once ("looks never change any number"), then **Walk into town**.
- Point at the HUD badge **"Simulation · not a prediction or advice"** and at **World #9**: "Her luck comes
  from one seeded world. Every reality we try lives in the same world, so differences come from choices."

## 2. Choices with visible prices (≈40 s)

- Click the **University** and choose **Full-time master's**. The preview comes from the engine: *"24 months
  on CHF 1'200/month, tuition CHF 16'000 in total; afterwards salary CHF 4'800/month."* Note the honesty tags
  🟡/🔵. The other big moves grey out: *"You already made one big life move this month: sleep first."*
- Go to the **Café** and set **eating out 2× per week**: *"CHF 340/month · +2 joy/month"*. A 🎓☕ icon now
  floats above Alexia.
- Optional: in the **Homes 4 Sale** agency, the studio is greyed out with *"You need CHF 99'000 for the down
  payment and fees…"*.
- At **Home**: **Sleep → next month**. The HUD moves to **Feb 2025** and cash drops to **CHF 2'760**
  (8'000 + 1'200 − 900 rent − 1'200 living − 340 habit − 4'000 tuition instalment).

## 3. A year passes; the world happens (≈40 s)

- **Sleep until next birthday** twice.
- Letters arrive as toasts and in the mailbox: **⚠️ Financing needed** (cash went below zero in April 2025),
  then **📬 Rent rise** in March 2026. Your landlord raised the rent 10%: an event from world #9.
- The HUD shows cash in red (−15'014 in Jan 2026, −39'236 in Jan 2027) with "⚠ financing". It's a warning,
  not game over.
- Go to the **Bank**: the monthly statement lists income, rent, living costs, the habit, tuition and
  transfers. Offer **Borrow CHF 10'000**: *"CHF 193/month for 60 months at 6.0%"*. A loan raises cash and
  debt equally, so net worth doesn't change.

## 4. Another reality (≈40 s)

- Press **P** → **Multiverse**. Tap the first square ("Master's +1") → **Go back** to Jan 2025.
- At the **Office**, choose **Start working in Switzerland**. Sleep until the next birthday several times
  (≈6×).
- At **26 (Jan 2031)** the **Homes 4 Sale** agency unlocks the **Studio**: *"CHF 492'049: pay 98'410 down
  + 9'841 fees now; each month interest + amortization + maintenance… and no rent. Net worth changes only by
  the fees."* Buy it and sleep: **🏠 You own a home**.

## 5. Compare (≈40 s)

- Phone → **Multiverse** → pin **A = Master's**, **B = Work → Buy home**.
- Same world, same assumptions:

  | | A: Master's | B: Work → studio |
  |---|---:|---:|
  | Net worth at 30 (world #9) | 143'503 | 256'879 |
  | Range at 30 (P10–P90, other worlds) | ≈ 75k – 122k | ≈ 211k – 245k |
  | Lowest cash | −39'236 | 0 |
  | Months with negative cash | 52 | 0 |
  | Home equity | 0 | 148'286 |
  | Joy points (not money) | 264 | 48 |

- Read **"why they differ"**: *the gap comes mostly from habits, living costs and mortgage interest; the
  master's needs financing for 52 months; "Master's" collects more joy points: a game preference score, not
  money.*
- Tap ❤️ on the score tiles (e.g. Joy and Safety) and show **"fit with what matters to you"**: no single
  score wins.
- Bonus insight: world #9 happens to be a *lucky* world for the master's path. Its 143k lies above the P90
  of other worlds. Same choices, different luck.

## 6. Same choices, different luck (≈20 s)

- **Money** app: toggle **Stress tests** (ghost lines: markets −20%, 6 months without a job, rent +20%,
  one year from now) and press **🎲 Roll another world**. The orange line and its list of events show one
  other future with the same choices.

## 7. What drives it (≈20 s)

- **Settings** → *Saving & investing* → change **Investment return** from 5% to **2%**. Every reality
  recomputes from month 0: B's range at 30 moves from ≈ 209k–246k to ≈ 199k–226k.
- Switch off **Market swings and drops**: the band collapses; that part of the spread came from the market.
- Point out the honesty tags: 🟢 profile data, 🟡 assumptions (incl. every habit, item and home price), 🔵
  simulated.

## 8. Is the maths right? (≈20 s)

- **Checks** app → **Run live checks**: 10/10, including
  - cash identity (8'000 + 1'200 − 1'800 = 7'400)
  - saving is a transfer
  - rollback replays exactly
  - **buying a home changes net worth only by fees (exactly 9'000 in every month)**
  - habit price change is isolated
  - monthly turns reproduce the original engine
  - the net-worth bridge with housing and habits is exact
  - loans never go below zero
  - **matches the official starter kit** to the cent
- Mention: 80 pytest tests, and `npm run smoke` plays this whole script headlessly.

Close on the postcard (sleep to 30): *"This is one of many possible futures."*
