# Game Build Prompt — "Many Lives": pixel-art life-sim Financial Twin

> Paste everything below this line into a new Claude Code chat opened in `/Users/parmida/Desktop/LauzHack/LauzHack-DanseazaPentruMine`.

---

You are turning our hackathon entry into a **top-down, pixel-art life-simulation game**. The challenge is the UBS × LauzHack 2026 "AI Financial Twin", and its motto is *"Don't predict the future. Make it explorable."*

The player picks a fictional person, customizes their look, and walks around a cozy pastel town. They visit places (university, office, café, shop, bank, real-estate agency, train station) and make life and money choices. They press **"Sleep → next month"** at home to let a month pass. At any time they can open an **in-game phone** to see their scores, uncertainty ranges and a **multiverse** of the realities they've lived, roll back, and compare.

This is an **original game**. Use the life-sim *genre* (walk around, visit places, spend, buy a home), but do not copy any existing game's characters, UI, icons, names, art or music (e.g. no Sims-style plumbob or needs bars).

Budget: **8–16 hours** in total. Stop and show me a running build after each milestone (§9).

## 0. Read first

1. `TASK.md`: challenge summary, rubric and validation rules.
2. `game/README.md`: what already exists (engine, API, tests, current web UI).
3. `evaluation/technical-validation.md` and `evaluation/responsible-ai-and-data.md`. Judges test against these.
4. `BUILD_PROMPT.md`: the original plan. The model rules in §5–7 there still apply.

### What already exists (keep it; it's tested)

- `game/backend/`: Python 3.11 + FastAPI. A **pure, deterministic monthly engine** (`app/engine/`) with cash, investments, debt, inflation, raises, a savings rule and loan amortization. It has a decision catalogue with action masking, 500-world Monte Carlo with shared seeded worlds, stress-test ghost lines, scores, a compare/"why they differ" module with an exact net-worth bridge, narratives, live validation checks, and **36 pytest tests**. One test matches `starter-kit/financial_twin.py` month by month.
- Run it with: `cd game/backend && .venv/bin/pytest -q` and `.venv/bin/uvicorn app.main:app --reload`.
- `game/frontend/`: the current React dashboard UI (HUD tiles, life chart with P10–P90 band, multiverse tree, compare, goals, assumptions drawer, validation). **Reuse its components and API client inside the phone (§6).** Delete this folder only once the new client has every feature.

Rules: all maths stays in the Python engine; the client never computes money. Don't modify the CSVs in `Data/`. Every existing test must keep passing (update a test only if its API shape changed, never to hide a regression).

## 1. Decisions already made (don't re-ask)

| Topic | Decision |
|---|---|
| View | 2D **top-down pixel art**, 16×16 tiles rendered at ×3 |
| Mood | **Bright, cozy pastel**, an original lakeside town inspired by Lausanne (lake to the south, mountains on the horizon) |
| Art | **Free CC0 packs from kenney.nl** only (§7). No hand-made art needed |
| Time | **Free roam while time is paused.** Actions taken this month queue up. **"Sleep → next month"** at home simulates one month; "Sleep until next birthday" simulates up to 12 |
| Places | Home, University, Office, Café/Restaurant, Shop, Bank, Real-estate agency, Train station |
| Spending | **Habits** (monthly recurring, e.g. "eat out 2×/week") + **one-off purchases**. The cost is always shown before confirming |
| Housing | **Simple mortgage**: pick 1 of 3 homes, 20% down payment, interest + amortization, home value grows at a stated rate |
| Character | Pick a persona from the data (Alexia featured), then **customize the look** (skin, hair, outfit colours) |
| Existing features | Move into an **in-game pixel phone** with apps (§6), plus a slim HUD bar on screen |
| AI | **No LLM.** All text is generated from engine numbers |
| Tech | **Phaser 3** (world) + **React** (phone, HUD, dialogs) + the **existing Python engine** |

## 2. Tech and folders

```
game/
  backend/            existing; extend the engine (§4) and API (§5)
  client/             NEW: Vite + React + TypeScript + Phaser 3
    public/assets/    Kenney packs (each with its License.txt) + font
    src/
      api/            copy and extend the typed client from game/frontend/src/api
      game/           Phaser: BootScene, TownScene, map builder, player, NPCs, doors
      ui/             React: HudBar, PlaceDialog, Phone + apps, CharacterCreator, PersonaPicker, EndScreen
      store/          game state (persona, look, multiverse tree, current turn's actions)
      bridge.ts       event bus between Phaser and React (e.g. mitt or a tiny emitter)
  start.sh            update so it starts backend + client
```

- Phaser renders into a `<div>` that React owns, with a React overlay on top (HUD, dialogs, phone). Phaser → React: "player entered door X". React → Phaser: "freeze input while a dialog is open", "change season tint".
- Dependencies: `phaser@3`, `react`, `recharts` (phone charts), `easystarjs` (click-to-move pathfinding, optional), `mitt`. Pin versions.
- Controls: WASD/arrows to walk, click/tap a tile to walk there, **E/Enter** or click a door to enter, **P** for the phone, **Esc** to close. Must be fully playable with the mouse alone (demo laptop).

## 3. The world

### Map

A ~40×28 tile town built **in code from an ASCII layout** (versionable, no Tiled dependency):

```
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM   M mountains/backdrop
TTTT..........TTTT.........TTTT.......TT   T trees  . grass
T..[UNIV]......=.....[OFFICE]....=....TT   [XXXX] building footprint + door
T..........=========================..TT   = road / path
T..[HOME]..=.[CAFE].....[SHOP]......=.TT
T..........=..............==========..TT
T..[BANK]..=..[REALESTATE].....fountain   (park in the middle)
T..........=====================[STATION]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   ~ lake (animated water if the pack has it)
```

Treat this as a sketch: make it look good, with paths, flowers, benches, lamp posts and a little pier on the lake. Each building has a **door trigger tile** and a **pixel sign** with its name.

- **Ambient life:** 4–6 NPCs from the character pack wandering on paths (pure decoration, no logic), and birds or boats on the lake if the assets exist.
- **Seasons:** tint the scene by month (spring green, summer warm, autumn orange, winter cool, with snow overlay tiles if available). It's cheap and shows time passing.
- **Moving abroad:** when the persona lives abroad, the town gets a different tint palette and signage ("Abroad"), and the station sign reads "Back to Switzerland". (A second map is a stretch goal.)

### Places → actions (each action maps to an engine action, §4)

Every dialog shows an NPC portrait, a one-line greeting, a list of actions with **the cost/effect preview from the engine**, and the honesty tags (🟢 data, 🟡 assumption, 🔵 simulated). Unavailable actions are shown greyed out with the engine's reason (e.g. "You need 12 months since your last big move").

| Place | Actions |
|---|---|
| **Home** | Sleep → next month · Sleep until next birthday · Check mailbox (event letters, §4.6) · Change flat (3 rental options: cheaper/same/nicer → rent) |
| **University** | Full-time master's · Part-time master's · Drop out (after 12 months) · Short evening course (one-off, small salary bump: stretch) |
| **Office** | Take a job / start working · Ask for a raise (only after ≥ 24 months in the job: +X% with a stated chance; uses the seeded world) · Reduce to 80% · Quit (→ between jobs) |
| **Café/Restaurant** | Habit: eat out *never / 1× / 2× / 4× per week* · One-off: treat friends (one-off) |
| **Shop** | One-offs: bike, laptop, phone, clothes, furniture · Habit: "shopping budget" *low / medium / high* |
| **Bank** | Savings rate (share of cash above buffer invested) · Take loan · Repay debt · See monthly statement (income, costs, habits, one-offs, interest, returns) |
| **Real-estate agency** | Buy home A/B/C (studio, flat, small house; price, down payment, monthly cost shown) · Sell home |
| **Train station** | Move abroad · Move back · Weekend trip (one-off + joy) |

What a persona sees is masked by the engine (a retiree has no "take a job" or master's; a student has no "ask for a raise").

## 4. Engine extensions (Python, test first)

Keep `app/engine/` pure (no FastAPI imports). Add new modules rather than bloating `decisions.py`. Write the new pytest tests **before** the code.

### 4.1 Monthly turns

- Replace "decision points every 12 months" with **a turn every month**. A branch becomes `(persona_id, assumptions, seed, world, turns: [{month, actions: [Action, …]}])`.
- `Action = {type, id?, value?}`, where `type ∈ {life_choice, set_habit, buy_item, change_flat, buy_home, sell_home, take_loan, repay_debt, set_savings_rate, ask_raise, trip}`. Existing life choices (`start_work`, `master_ft`, `master_pt`, `move_abroad`, `return_home`, `drop_out`, plus new `quit_job`, `reduce_hours`) keep their validity rules and cooldowns.
- A month with no actions = "nothing changed". The multiverse only shows months with actions.
- Keep backwards compatibility for the tests: a helper converts the old `Decision(month, choice)` into a one-action turn, so the starter-kit cross-check still passes unchanged.
- Action masking: `available_actions(state, month, params)` returns, **per place**, every action with `enabled`, `reason_if_disabled` and `preview` (immediate monthly/one-off effect, computed by the engine).

### 4.2 Habits (recurring lifestyle costs)

- `state.habits: {habit_id: level}`. Each level has a monthly cost (in month-0 prices, inflated like other costs) and **joy points** per month.
- Example defaults, all 🟡 our assumptions in the catalogue:
  - eat out: 0 / 170 / 340 / 680 CHF per month
  - shopping: 80 / 200 / 450
  - gym (shop or office perk): 0 / 60
- Habit costs are a separate expense line `lifestyle` in each monthly row, so the statement and the "why they differ" bridge can name them.

### 4.3 One-off purchases and trips

- An item catalogue with price and joy. Example: bike 800, laptop 1,500, phone 900, clothes 200, furniture 2,000, treat friends 120, weekend trip 450, abroad flight home 250.
- Paid in the month bought, recorded as `OneOff(kind="purchase" | "trip")`.
- Purchases are **consumption, not assets**: they reduce net worth. State this in the UI so nobody thinks a laptop is an investment.

### 4.4 Housing

- **Rental options** (Change flat): cheaper = rent × 0.8, same, nicer = rent × 1.3, with a one-off moving cost. All tagged 🟡.
- **Buy home** (3 options scaled to the persona's region: studio 450k, flat 750k, small house 1.1M for Lausanne; 🟡):
  - Requires `cash ≥ 20% down payment + 2% purchase fees`.
  - At purchase: `cash −= down + fees`, `home_value += price`, `mortgage += 80% × price`. **Net worth changes only by the fees** (test this).
  - Each month: `mortgage interest = mortgage × r_mortgage / 12` (r_mortgage = 2%, 🟡) is an expense; `amortization = 1% of price per year / 12` is a transfer (reduces cash and mortgage, net worth unchanged); `maintenance = 1% of price per year / 12` is an expense; **rent → 0**; `home_value ×= (1 + g_home)^(1/12)` with g_home = 1.5%/year (🟡).
  - **Sell:** `cash += home_value − mortgage − 3% fees`, the mortgage is cleared, and you go back to renting at the "same" flat rent.
- `net_worth = cash + investments + home_value − debt − mortgage`. Update `metrics.net_worth_bridge` with new terms (lifestyle, purchases, mortgage interest, maintenance, home appreciation, fees) and keep it **exact** (test).
- Affordability is shown as a warning, not blocked: if housing costs exceed ⅓ of income, show "🟡 above the common one-third guideline".

### 4.5 New score: Joy

- Add a sixth dimension, **Joy**: the sum of declared joy points from habits, purchases, trips and status (e.g. studying +1/month, abroad +2/month, unemployed −2/month). All values come from an editable table in the assumptions.
- It's a **game preference score**, explicitly "not money, not a measure of a good life". The player can set it to zero in Settings.
- It lets the compare view show trade-offs (more joy, less wealth) and reinforces "higher net worth ≠ best life".

### 4.6 Your world and events

- The playthrough lives in **one seeded world** (`world` index rolled at game start, shown as "World #137"). Market drops, income-loss spells and rent rises from `montecarlo.py` actually happen to you, and arrive as **letters in the home mailbox** plus a toast ("📬 Your landlord raised the rent 10%").
- The phone still shows the P10–P90 band over all 500 worlds, and the "no surprises" path as a dotted line.
- All realities in the multiverse use the **same world**, so comparisons stay fair.
- Optional (stretch): "ask for a raise" success uses a seeded draw, so it is reproducible.

### 4.7 Tests to add (pytest)

- Habit cost appears exactly in `lifestyle`; changing one habit's price changes only branches that use it (isolation).
- A purchase reduces cash and net worth by exactly its price; it is never counted as an asset.
- Buying a home changes net worth only by fees; amortization is a transfer; the mortgage never goes negative; selling clears the mortgage; the net-worth bridge stays exact with housing.
- Cannot buy without the down payment (InvalidAction with a reason).
- Monthly turns: rollback to any month replays bit-for-bit. The old year-based decisions converted to turns reproduce the old results (all 36 existing tests pass).
- Masking: a retiree has no job/university actions; the cooldown blocks a second move within 12 months.
- Same world ⇒ the same events across realities.

## 5. API changes

Keep it stateless; the client holds the multiverse tree.

| Method | Path | Purpose |
|---|---|---|
| POST | `/game/state` | `{persona_id, look, assumptions, seed, world, turns}` → current month/age/date, state (cash, investments, home, debt, mortgage), HUD scores, `available_actions` per place, new letters this month, monthly statement |
| POST | `/game/projection` | Same input → full 10-year projection if nothing else changes ("no surprises") + P10–P90 bands (500 worlds) + ghosts |
| POST | `/compare` | Extend to turns and the new metrics (Joy, home equity) |
| GET | `/catalogue/{persona_id}` | Habits, items, homes, flats, joy table, with sources/tags (for Settings) |
| existing | `/personas`, `/assumptions/defaults/{id}`, `/validation` | Keep; add new checks to `/validation` (home purchase NW, habit isolation) |

## 6. UI (React overlay, pixel-styled)

- **Font:** a free pixel font (e.g. Kenney Fonts or Google's "Pixelify Sans", OFL). Use crisp pixel borders from the Kenney Pixel UI pack. Set `image-rendering: pixelated` everywhere.
- **Persona picker:** Alexia featured ("What if I start working, do a master's, or move abroad?"), filters by status, random. Cards show 🟢 data.
- **Character creator:** layered sprite from Roguelike Characters (body/skin, hair style + colour, top, bottom). Live preview walking. Saved with the game; it affects looks only, never maths.
- **HUD bar (top, always visible):** 📅 *Month Year · age* · 💰 cash · 6 small score icons (Money, Safety, Goals, Freedom, Experience, Joy) with tooltips · World # · badge **"Simulation · not a prediction or advice"**.
- **Place dialog:** NPC portrait, greeting, action list with previews and tags, "Confirm" (adds to this month's actions) or "Leave". Queued actions show as little icons above the player until the month is slept.
- **Phone** (press P or click the phone icon; a pixel phone frame sliding up). Apps:
  - **Money:** six score tiles with ranges (reuse `Hud.tsx`), and the life chart (reuse `LifeChart.tsx`) with lived path (solid), "no surprises" (dotted), P10–P90 band, stress-test ghosts, **"🎲 roll another world"**, and the fog over the future labelled "not lived yet: one of many possible futures". It also shows the monthly statement and net-worth bridge.
  - **Multiverse:** the reality tree over age (reuse `Multiverse.tsx`). Tap a node → "Go back to {month/age}?" → the town reloads at that month. Pin A/B → compare (reuse `ComparePanel.tsx`), including "fit with what matters to you" (♥ on score tiles).
  - **Goals:** select persona goals, add your own (reuse `GoalsPanel.tsx`).
  - **Settings:** the assumptions drawer (reuse), the catalogue (habit/item/home prices, joy table), and the uncertainty toggles.
  - **Checks:** live validation (reuse `ValidationPanel.tsx`).
  - **Mail:** event letters history.
- **Toasts** for events and milestones ("🎓 Degree finished!", "🏠 You own a home").
- **End screen at 10 years:** a postcard-style summary of this reality (six scores, ranges), "This is one of many possible futures", buttons for "Explore another reality" (opens Multiverse) and "Compare".
- **Responsible design everywhere:** honesty tags; never "best choice"; percentiles called "ranges under our assumptions", never probabilities; the disclaimer badge always visible.

## 7. Art and sound (CC0, from kenney.nl)

Download into `game/client/public/assets/<pack>/`, keep each pack's `License.txt`, and verify it says **CC0** before use. If a URL has moved, search kenney.nl for the pack name.

| Use | Pack (16×16 unless noted) |
|---|---|
| Town tiles, houses, trees, water | **Tiny Town** (kenney.nl/assets/tiny-town) |
| Roads, shops, city props | **RPG Urban Pack** (kenney.nl/assets/rpg-urban-pack) |
| Player + NPCs, layered for customization | **Roguelike Characters** (kenney.nl/assets/roguelike-characters) |
| Phone/dialog frames, buttons | **Pixel UI Pack** (kenney.nl/assets/pixel-ui-pack) |
| Click/confirm/notification sounds (stretch) | **Interface Sounds**, **RPG Audio** |

- Inspect each spritesheet (open the preview PNGs) and record tile indices in `src/game/tiles.ts`. Don't guess indices.
- **Pastel palette** (apply as UI colours and gentle scene tints): grass `#9fd8a5`, water `#8fd3f4`, path `#f3e3c3`, roof accents `#f7a8b8` / `#ffd59e`, UI cream `#fff6e9`, ink `#4a4063`, accent lilac `#b8a4e3`, good `#6cc6a0`, warn `#ffc46b`, bad `#ff8a8a`.
- Credit the packs in `game/README.md` (CC0 doesn't require it, but it's good practice).
- The user may add reference screenshots in `docs/style/`. If that folder exists, look at them first for mood only.

## 8. Data and modelling rules (unchanged, still enforced)

- Synthetic data only. No real personal data and no keys in git.
- Salaries in the starter kit are net, so tax = 0. Use the opening snapshot (assets + liabilities), not `synthetic-cash-balances.csv`.
- Every new number (habit prices, items, homes, mortgage rate, joy points) lives in the assumptions catalogue with a unit, bounds and a 🟡 tag, and is editable in Settings.
- Negative cash ⇒ "financing needed" warning (red cash in the HUD and a letter), not game over.
- Never present outputs as predictions, real-world probabilities or advice.

## 9. Milestones (show me a running build after each)

1. **Engine (2–3 h):** monthly turns, habits, purchases, housing, Joy, your-world events, the new tests, and the `/game/*` endpoints. All tests green, including the old 36.
2. **Walkable town (3–4 h):** `game/client` scaffold, Kenney assets loaded, ASCII map rendered, player walking (keyboard + click), 8 doors opening place dialogs that call `/game/state`, confirming actions, "Sleep → next month" advancing the month, HUD bar live. Persona picker + character creator.
3. **Phone (2–3 h):** Money, Multiverse (rollback reloads the town), Goals, Settings, Checks and Mail apps, reusing the old components restyled.
4. **Juice (1–2 h):** letters and toasts, season tints, ambient NPCs, sounds, end-of-10-years postcard, abroad tint.
5. **Ship (1 h):** update `game/README.md` (how to play, controls, model additions with equations, new assumptions table, asset credits, limitations) and write `game/DEMO.md`. Update `start.sh`. Delete `game/frontend` once every feature lives in the phone.

## 10. Demo script (for `game/DEMO.md`, about 4 minutes)

1. Pick **Alexia** and customize her look (10 s). Point at the badge "Simulation · not a prediction or advice".
2. Walk to the **University** and choose a full-time master's, which shows the tuition preview. Then go to the **Café** and set eating out 2×/week, which adds the habit cost. Sleep → next month and show the HUD changing.
3. "Sleep until next birthday" a few times. A **letter** arrives (rent rise in this world). Cash dips below zero, the "financing needed" warning appears, and she goes to the **Bank** for a loan.
4. Open the **phone → Multiverse**, go back to age 20, choose **Office → start working**, and fast-forward. Then **Real-estate → buy the studio** at 26.
5. **Compare** the two realities: money range, safety, goals, joy, and "why they differ" (net-worth bridge). Note that no single score wins.
6. In the **Money app**, roll another world and toggle the stress tests: same choices, different luck.
7. Open **Settings**, change the investment return, and watch everything recompute. Point out the main drivers.
8. **Checks app:** run the live validation (cash identity, home purchase changes net worth only by fees, starter-kit match).

## 11. Rules for you (the builder)

- Correctness over features: one polished, honest loop beats many shallow places.
- Ask me before big scope changes. Otherwise make sensible choices, note them in the README, and keep going.
- Don't commit or push unless I ask. Never commit `.env`, `.venv` or `node_modules`.
- After each milestone, run `pytest`, the client type-check (`tsc -b`) and a quick headless smoke test of the main flow, and report the results honestly.
