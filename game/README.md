# Many Lives: a lakeside town Financial Twin

UBS × LauzHack 2026 "AI Financial Twin". *Don't predict the future. Make it explorable.*
**This is a simulation, not a prediction or financial advice. All people and data are synthetic.**

You pick a fictional person (Alexia is featured), choose how they look, and walk around a small pastel
lakeside town inspired by Lausanne. You visit the university, the office, the café, the shop, the bank, the
real-estate agency and the train station, and you make life and money choices there. A **life timeline**
along the bottom runs from today to **age 120**: click or drag to live forward or go back, click a choice to
return to it or undo it. The **regime HMM** is the backbone: every month of a life is *working*, *job search*,
*studying* or *retired*, sampled from sourced, age-banded hazards and conditioned on your choices. Right of
the playhead, the timeline fans out the possible futures from where you are. Your **in-game phone** shows your
scores, the uncertainty ranges, and a **multiverse** of every reality you've lived.

All maths happens in the Python engine. The client never computes money: it only sends the list of choices
and shows what the engine returns.

## Run it

```bash
cd game && ./start.sh        # engine on :8000 + town on :5173, opens the browser
```

Or manually, in two terminals:

```bash
cd game/backend && python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --reload          # API on :8000, docs at /docs
cd game/client && npm install && npm run dev     # http://localhost:5173
```

URL options: `?world=9` fixes the seeded world (reproducible demos; see [`DEMO.md`](DEMO.md)), and `?fresh`
clears the saved game.

### Checks

```bash
cd game/backend && .venv/bin/pytest -q           # 95 tests
cd game/client && npx tsc -b                     # type-check
cd game/client && npm run smoke                  # headless walk through the whole demo (needs both servers
                                                 # and Google Chrome; BASE=http://localhost:5173 by default)
```

The smoke test walks through the whole demo story in headless Chrome. It picks Alexia, queues a full-time
master's and eating out, then sleeps a month and on to the next birthday. It then checks the bank statement
and opens every phone app, including a roll of another world. The live checks must all pass (10/10). After
that it goes back to the start in the Multiverse and starts working instead. It buys the studio as soon as
the engine allows it, compares the two realities, and fast-forwards to the 10-year postcard.

## The life timeline (bottom of the screen)

- It is a **drawer**: only a small *▲ Life timeline* tab shows. Point at it and the timeline slides up; move
  the mouse away and it hides. It stays open while you drag, have a choice open, or time is moving. Click the
  tab, press 📌, or press **T** to pin it open.
- **Left of 📍** is the life you lived: net worth (line) and the regime of every month (coloured strip), as
  the HMM played it out in your world. Life events (job ended, back at work, degree, home, retirement) are
  small icons on the strip.
- **Right of 📍** is what the regime HMM fans out from your current state: the P10–P90 net-worth band and the
  median (dashed) over 60 simulated lives, and per month the share of lives in each regime. Hover anywhere
  for the numbers ("Of 60 simulated lives: Working 91% · Job search 4% · Retired 5%").
- **Move freely:** click or drag on the timeline, or use ⏮ −1y −1m +1m +1y +10y 🌅65 🏁120. Moving forward
  lives the months (this month's plans happen first); moving back rewinds exactly (deterministic replay).
- **Your choices are markers.** Click one to *go back to this moment* or to *undo* one choice.
  - Going back keeps that choice and later ones **ahead of you as faded plans**: moving forward re-lives them
    (redo). Make a different choice there and you branch into a new reality; the old one stays in the
    Multiverse.
  - Undo builds a new reality without that one choice and replays the later choices on top. If a later choice
    depends on it (for example, a home bought with that job's savings), the engine says why and nothing
    changes.

## How to play

| | |
|---|---|
| Walk | **WASD** / arrow keys, or **click** a tile to walk there (A* pathfinding) |
| Enter a building | stand on its door and press **E** / **Enter**, or just **click the building** |
| Quick travel | the place buttons in the bottom-right corner walk you to that door |
| Phone | **P** or the 📱 button · **Esc** closes dialogs and the phone |

Everything works with the mouse alone.

- **Plans, then sleep.** Every action you confirm in a building is queued for *this month* and shows as an
  icon above your head. The engine accepts or refuses it immediately: refused actions are greyed out with
  the engine's reason, for example "You need 12 months since your last big move (6 to go)". When you sleep
  at **Home** (*Sleep → next month* or *Sleep until next birthday*), the month is simulated.
- **Letters** arrive in the mailbox and as toasts. They cover rent rises, job losses and market drops from
  your seeded world, plus "financing needed", "degree finished", "you own a home", raises and loans paid off.
- **Places:**

  | Place | Actions |
  |---|---|
  | Home | sleep 1 month / until birthday · mailbox · change flat (cheaper / similar / nicer) |
  | University | full-time master's · part-time master's · 🎓 get your degree · 🚪 drop out without a degree (both only while enrolled; see below) |
  | Office | start working · ask for a raise (after 24 months) · reduce to 80% / back to 100% · quit |
  | Café | habit: eat out never / 1× / 2× / 4× a week · one-off: treat your friends |
  | Shop | bike, laptop, phone, clothes, furniture · habits: shopping budget, gym |
  | Bank | monthly statement · savings rule (0–100%) · borrow · repay |
  | Real-estate | buy a studio / flat / small house · sell |
  | Train station | move abroad / back to Switzerland · weekend trip · flight home (when abroad) |

  **School.** *Get your degree* and *Drop out without a degree* are only possible while you are enrolled
  (Alexia starts enrolled; a master's enrols you). Getting the degree ends school and starts a job search: the
  regime HMM decides when you find a job (about 20%/month at 20), at the graduate salary. During a master's
  the degree comes by itself when the programme ends, so that button is greyed out. Dropping out cancels the
  remaining tuition and starts a job search. Out of your first degree you have **no degree**: jobs pay
  `no_degree_wage_factor` × the graduate salary (80%, our assumption) and a master's is closed. Out of a
  master's you keep your bachelor's. A retiree who drops out goes back to retirement.
  The engine masks what a persona can do. A retiree gets no job choices but can do a full-time master's: the pension continues, tuition is paid, and they return to retirement when they graduate or drop out. There is no age limit on studying. A student can't ask
  for a raise.
- **Phone apps:**
  - **Money:** six score tiles with ranges, and the life chart (lived path solid, "no surprises" dotted,
    P10–P90 band from now on, stress-test ghosts, 🎲 roll another world, fog over the future). Below the
    chart: the net-worth bridge and last month's statement.
  - **Multiverse:** the reality tree. Tap a square to go back to just before that choice, tap the end of a
    path to continue that reality, and pin A/B to compare. The compare view shows the metric table, ranges,
    "why they differ" from the exact bridge, and "fit with what matters to you" (♥ on the tiles).
  - **Goals:** the persona's goals from the data, plus your own.
  - **Settings:** every assumption with its tag, including the game catalogue, plus toggles for each source
    of uncertainty and for joy.
  - **Checks:** live validation.
  - **Mail:** every letter you've received.
- **At age 120** a postcard shows this reality's six scores. Its ranges come from the same choices
  replayed in other worlds. It reminds you that this is one of many possible futures.

### The six scores (deliberately no single "best" number)

- 💰 **Money:** net worth now. In the phone, the P10–P90 range at the end.
- 🛡️ **Safety:** steady / tight / shortfall (months below the buffer, months with cash < 0), and the share
  of worlds that ever run short of cash.
- 🎯 **Goals:** goals met (priority order, no double counting) and the share of worlds where each is met.
- 🕊️ **Freedom:** months of costs covered by liquid savings; debt-free or not.
- 🎓 **Experience:** degree, months studying, years abroad.
- 🌸 **Joy:** a **game preference score**: points from habits, trips and your life situation. It is not
  money and not a measure of a good life. It can be switched off in Settings. It's there to make the
  trade-offs visible: more joy can mean less wealth.

## Architecture

```
game/
  backend/                 Python 3.11 + FastAPI (stateless)
    app/engine/            PURE engine (no FastAPI imports)
      finance.py           rate conversion, inflation, annuity (0% safe)
      assumptions.py       every assumption: label, unit, bounds, source tag (+ the catalogue)
      catalogue.py         NEW habits, items, homes, flats, housing/work rules, joy table
      state.py             immutable State (+ habits, flat, home, mortgage, work share)
      decisions.py         original decision catalogue + masking (reused by the game)
      simulate.py          step_month() (extended: lifestyle, purchases, housing, joy) + run_branch()
      game.py              NEW monthly turns, actions, masking with reasons, previews, letters
      metrics.py           summaries, goals, scores, scores_now, exact net-worth bridge
      montecarlo.py        seeded worlds, stress tests
      compare.py           "why they differ" (now with habits, housing and joy)
    app/service.py         glue: persona + overrides → engine; /game/* responses
    tests/                 80 pytest tests
  client/                  Vite + React 19 + TypeScript + Phaser 3.90
    public/assets/         Kenney CC0 packs (each with its License.txt) + OFL fonts
    src/api/               typed API client (from the old frontend, extended)
    src/game/              Phaser: TownScene, ASCII map, tile indices, PhaserHost
    src/ui/                HUD, place dialogs, phone + apps, creator, picker, postcard
    src/store/             game store: persona, look, multiverse tree of turns, this month's queue
    src/bridge.ts          mitt event bus between Phaser and React
    scripts/smoke.mjs      headless smoke test (playwright-core + local Chrome)
```

**Stateless, deterministic replay.** A reality is `(persona_id, assumptions, seed, world, turns,
current_month)`, where `turns = [{month, actions: [{type, id?, value?}]}]`. The client holds the multiverse
tree. Every request replays from month 0, so going back in time is exact by construction: a test checks it
bit for bit. A turn dated at `current_month` is this month's queue. The town shows the state after the queue
but before the month is simulated.

### API

| Method | Path | Purpose |
|---|---|---|
| POST | `/game/state` | Month, date, age, state (cash, investments, home, mortgage, debt), HUD scores, `actions` per place (enabled, reason, engine preview, tags), letters, last month's statement, warnings |
| POST | `/game/projection` | "No surprises" path from now, P10–P90 bands over 500 worlds from now, stress ghosts, goal status, net-worth bridge. At the end: ranges of the same choices in all worlds |
| POST | `/game/roll` | The same choices in another world from now on |
| POST | `/compare` | Two realities (turns) in the same world: table, ranges, bridge, explanation, scores. Still accepts the old decision branches |
| GET | `/catalogue/{id}` | Habits, items, homes (canton-scaled), flats, rules, joy table |
| GET | `/personas`, `/personas/{id}`, `/assumptions/defaults/{id}`, `/validation` | unchanged (validation has 4 new checks) |

The original `/branch/*` endpoints still exist and are still tested.

## Model (monthly, in this order)

| Step | Equation |
|---|---|
| 0. Turn | a master's ending this month completes. The player's actions are applied in order. A graduate with no other big move this month starts a job at the post-master salary (the old engine's default) |
| 1. Income | `income = salary × (1 − tax − social)`; +`raise_annual` every 12 months in the same job; × the world's income multiplier during a job-loss spell |
| 2. Rent & living | `expenses = (rent × flat_factor × rent_shock + living) × (1+i)^(t/12)`, rent = 0 while you own your home |
| 3. Habits | `lifestyle = Σ habit_cost(level) × (1+i)^(t/12)` (a separate line) |
| 4. One-offs | tuition, relocation, moving (`one_offs`); items and trips (`purchases`, **consumption**); home fees (`home_fees`) |
| 5. Loans | `interest = debt × r/12` (expense); `principal = payment − interest` (transfer); debt never < 0 |
| 6. Housing | `mortgage_interest = mortgage × r_m/12` and `maintenance = 1% × price/12` are expenses; `amortization = min(mortgage, 1% × price/12)` is a **transfer**; `home_value ×= (1+g)^(1/12)` |
| 7. Cash | `cash += income − expenses − lifestyle − purchases − one_offs − home_fees − loan payment − mortgage interest − maintenance − amortization`; if `cash < 0`, investments are sold to cover it (transfer) |
| 8. Saving | cash above `buffer_months × expenses` × `savings_rate` → investments (transfer) |
| 9. Returns | `investments × (1 + r_m)`, with `(1 + r_m)^12 = 1 + return_annual` (or the world's draw) |
| 10. Outputs | `net_worth = cash + investments + home_value − debt − mortgage`; `cash < 0` ⇒ "financing needed" (warning + letter, not game over); joy points |

**Buying a home** (price grows at the stated home growth rate from month 0): you need
`cash + investments ≥ 20% down + 2% fees`. Then `cash −= down` (investments are sold if needed),
`home_value = price`, `mortgage = price − down`. Net worth changes **only by the fees**, which a test checks
exactly. **Selling:** `cash += home_value − mortgage`, then 3% fees, the mortgage is cleared, and you rent a
"similar" flat again. Housing costs above ⅓ of income show a 🟡 warning; they aren't blocked.

**Net-worth bridge (exact):**
`NW_T = NW_0 + income − expenses − lifestyle − purchases − one-offs − interest − mortgage interest −
maintenance − home fees + investment returns + home appreciation`. Transfers never appear in it: saving,
loans, principal, amortization, and the down payment against the mortgage.

### New assumptions (all 🟡 ours, editable in Settings → each group)

| Group | Assumption | Default |
|---|---|---|
| Habits (CHF/month, on top of base living costs) | Eating out 1× / 2× / 4× per week | 170 / 340 / 680 |
| | Shopping budget low / medium / high | 80 / 200 / 450 |
| | Gym | 60 |
| Purchases & trips (CHF once) | bike 800 · laptop 1,500 · phone 900 · clothes 200 · furniture 2,000 · treat friends 120 · weekend trip 450 · flight home 250 | |
| Housing | Home prices studio / flat / small house (Lausanne) | 450k / 750k / 1.1M |
| | Canton price level (× Lausanne): ZH 1.25, GE 1.2, VD 1.0, BS/LU 0.95, BE/TI 0.8, FR/SG 0.75, NE 0.65 | |
| | Mortgage rate · down payment · purchase fees · selling fees | 2% · 20% · 2% · 3% |
| | Amortization · maintenance (of price per year) · home value growth | 1% · 1% · 1.5%/year |
| | Rent factor cheaper / similar / nicer · moving cost | 0.8 / 1.0 / 1.3 · 1,500 |
| Work | Raise when asking works · chance · months in job first | +5% · 50% · 24 |
| | Reduced hours · income after quitting | 80% · 0% |
| Joy (points/month) | studying +1 · working 0 · abroad +2 · between jobs −2 · retired +1 · home owner +1 · reduced hours +1 | |
| Joy (habits / items) | eat out 1/2/3 · shopping 0.5/1/1.5 · gym 1 · bike 6 · laptop 4 · phone 3 · clothes 2 · furniture 5 · treat 3 · trip 6 · flight home 5 | |

The original assumptions are unchanged (starter-kit incomes and costs, CSV tuition, 5% return, 1%
inflation, 2% raises, the 50% savings rule, a 3-month buffer and so on). Their table and sources are in
Settings. With no habits, purchases or homes, the game engine reproduces the original engine exactly. That
includes the starter-kit cross-check, to the cent, for all three scenarios.

### Uncertainty and your world

- The playthrough lives in **one seeded world** (`World #N` in the HUD, from 0 to 499, rolled at the start
  or set with `?world=`). Its market drops, job-loss spells and rent rises actually happen to you, and
  arrive as letters, but only when they matter to you. For example, there's no rent letter if you own your
  home.
- **All realities use the same world**, so comparisons come from choices, not luck. Tests check this.
- The phone fans out **500 worlds from now on**, given what you have lived (`P10–P90`, labelled "ranges
  under our assumptions", never probabilities). The dotted line is "if nothing surprising happens from now".
  The ghosts are stress tests one year from now. 🎲 shows one other world.
- Distributions: monthly return ~ N(μ, σ/√12) with σ = 15%/year; a 20% market drop with
  5%/year chance; a 10% rent rise with 10%/year chance. Seed 42.
- **The regime HMM is the backbone of a life** (`labour_market_hmm = 1`, default; `app/engine/regime.py`,
  stepped inside `game.run_turns`). Every month is one regime of the teammate's chain (`backend/twin`):
  `employed`, `part_time`, `job_search`, `student`, `retired`. Hazards are sourced and age-banded (Eurostat
  `lfsi_long_q`, BFS SAKE, SECO; `backend/assumptions/transitions.csv`); each world carries its own Dirichlet
  draw of the transition matrix (parameter uncertainty). The chain is **conditioned on your life**:
  - it runs month by month from your *current* regime and age;
  - your choices pin regimes: studying = `student`, quitting = `job_search` (and the chain decides when you
    find a job), working / abroad = the free labour-market chain;
  - a master's scales the job-loss hazard by `masters_hazard_factor` (0.81, BFS EHA); job-search exits halve
    after 12 months (duration dependence);
  - at `retirement_age` (65, AHV 21) everyone in the labour market retires on `pension_replacement_rate`
    (47.5% of the last full salary, OECD net replacement rate). The pension is split into **AHV** (up to the
    maximum AHV pension, 2,520 CHF/month in 2025) and **BVG** (the rest). The AHV part rises every two years
    (January 2027, 2029, …) by the **mixed index**, the average of Swiss nominal wage growth (1%/year, BFS Swiss
    Wage Index) and inflation (AHVG Art. 33ter), and a **13th AHV pension** is paid each December from
    December 2026 (2024 vote). The BVG part is not indexed: the law leaves it to each pension fund. All four
    numbers can be changed under Settings → Life course.
  In `job_search` months, job income is × `unemployment_benefit_rate` (70%, AVIG Art. 22). Money stays in this
  engine, so there is one set of accounts.
  - **Common random numbers:** world *k* holds one uniform per month and one matrix draw, identical whether 1
    or 500 worlds are drawn. All realities replay the same draws, so they differ because of choices.
  - `labour_market_hmm = 0` restores the flat model (3–9 months with 5%/year chance).
  - Tests: `tests/test_regime_backbone.py` (spells, retirement at 65, student pin, common random numbers,
    master's lowers job-search months, quitting ends when the chain finds a job, regime shares in the
    projection). HMM part-time months count as full pay, because hours are a player choice.
- Asking for a raise uses one draw from your world, `rng(seed, world, month)`, so it is reproducible.

## Design choices (made without asking; easy to change)

- **Habit level 0 = nothing on top of the profile's living costs.** The starter-kit living costs already
  include a basic lifestyle, so habits are extras. That's why shopping also has a free "basics only" level.
- **Affordability is checked only for this month's plans.** If you edit an assumption later, earlier
  purchases stay lived and show up as negative cash instead of disappearing. Structural rules, such as a
  longer cooldown, can still invalidate an earlier choice; you then get the engine's reason.
- **Monte Carlo runs of other worlds** replay the same choices without affordability checks. A purchase
  that a poorer world couldn't afford shows up as "financing needed" in that world.
- Students who never choose stay students. The old engine made them start work at 24 automatically; the
  game leaves it to you. Graduates do start working automatically (the old default).
- The down payment may come from investments. You can't buy a home while abroad, or move abroad while you
  own one. Moving to or from abroad resets the flat factor (new country, new flat).
- Item prices inflate like other costs; home prices grow at the home growth rate.
- Quitting a job gives no income by default (benefit rate editable).
- `?world=9` is the recommended demo world. Defaults were **not** tuned to make the demo work. In world #137,
  for example, the studio only becomes affordable at about 26¾.
- Pixelify Sans draws 2 and 5 like 8 and S, which is risky for a money game, so all digits use VT323 (also
  OFL) through a `unicode-range` font face.
- Seasons are a cheap CSS wash over the canvas, plus autumn tree tiles and CSS snow. Abroad uses a warmer
  palette and different signs. A second map is a stretch goal.

## Assets and credits

All art and sound are **CC0** by [Kenney](https://kenney.nl). Each pack's `License.txt` sits next to its
files in `client/public/assets/`, and each was checked before use.

- **Tiny Town**: grass, paths, trees, roofs, walls, doors
- **RPG Urban Pack**: lake water, pier planks, fountain, benches, lamps, mailbox, walking townsfolk
- **Roguelike Characters**: the layered, customizable player (skin, hair, top, trousers)
- **Pixel UI Pack**: inlay panel texture
- **Interface Sounds**: clicks, confirmations, letters, phone

The fonts are **Pixelify Sans** and **VT323**, both under the SIL Open Font License; their licenses are in
`client/public/assets/fonts/`. Tile indices were read from the sheets (rendered with index overlays) and are
recorded in `client/src/game/tiles.ts`. The mountains and the little sailing boat are simple shapes drawn in
code.

## Limitations

- Net salaries come from the starter kit (tax = 0), so there is no tax model. Mortgage, home and habit
  figures are rough labelled assumptions, not Swiss statistics.
- Home prices follow one fixed growth rate. There's no housing-market randomness, no rate changes over the
  mortgage's life, no affordability check by the bank (only the ⅓ warning) and no imputed rental value tax.
- Abroad is one generic country at a fixed FX rate. The town is re-tinted rather than a second map.
- Joy points are invented preference weights. They're there to show trade-offs and mean nothing about
  real wellbeing.
- Percentiles describe our model's worlds, not real-world probabilities.
- Peer (kNN) bands from the original plan are not in the game.
- The timeline runs to age 120 with no mortality model, and amounts are nominal. A century of 5% returns
  compounds to large numbers (median net worth in the millions at 120); read the late decades as "under these
  assumptions", not as wealth anyone should expect.
- The future fan on the timeline uses 60 simulated lives (P10–P90 is rough at that size), to keep a projection
  over a whole life to 1–3 seconds.

## Open questions for organizers (unchanged)

- `synthetic-transactions.csv`, `life-events.csv` and `alexia-profile.json` are listed but missing.
- Slide 14 (Master's ≈112k > Abroad ≈95k > Work ≈93k) contradicts the starter kit (Work > Master's > Abroad).
- Tuition: 8,000/year (CSV) or 2 × 2,500 (starter kit)?
