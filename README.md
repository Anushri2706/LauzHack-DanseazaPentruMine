# Many Lives: an AI Financial Twin you can walk around in

Our entry for the **UBS × LauzHack 2026 "AI Financial Twin"** challenge.
Motto: *Don't predict the future. Make it explorable.*

> **This is a simulation, not a prediction or financial advice. All people and data are synthetic.**

## Why we built this

We think it's unfair that your future depends so much on your family background. The OECD estimates that,
on average across its member countries, it can take **around five generations** for a child from a
low-income family to reach the average income. In some countries it takes nine generations or more
([OECD, *A Broken Social Elevator? How to Promote Social Mobility*, 2018](https://www.oecd.org/social-issues-migration-health/broken-elevator-how-to-promote-social-mobility_9789264301085-en)).

Part of that gap is knowledge: what a master's really costs, what inflation does to savings, why a loan
costs more than its monthly payment. Children who don't hear these things at home start out behind. We
believe everyone should have access to this knowledge. That's why we built a game: children from any
background can try different life decisions in a safe world, see how they could play out, and learn the
money basics along the way (the phone's 📚 Learn app).

**If we win this challenge, we would like the prize money to go to [UNICEF](https://www.unicef.org/about-us).**
We believe it is the most impactful organisation for children's wellbeing and education. UNICEF, the
United Nations agency for children, works in more than 190 countries and territories to protect the rights
of every child, especially the most disadvantaged and those hardest to reach
([UNICEF, *Who we are*](https://www.unicef.org/who-we-are)). It works to provide quality learning
opportunities in 147 countries, and in many humanitarian emergencies it is the largest provider of
educational support ([UNICEF, *Education*](https://www.unicef.org/education)).

## What we built

You pick a fictional person (Alexia, 20, a student in Lausanne, is the featured one) and walk around a small
pixel-art lakeside town. In its buildings you make life and money choices: study a master's, start working,
eat out, buy a flat, move abroad, borrow, save. Each choice re-runs a Python engine, which shows how that
person's finances could unfold month by month up to age 120, with uncertainty ranges and not a single
"right answer".

The repo has three parts:

| Part | Folder | What it is |
|---|---|---|
| **The game** | [`game/`](game/) | FastAPI engine (`game/backend`) + React/Vite pixel-art town (`game/client`). This is the demo. |
| **Regime twin** | [`backend/`](backend/) | A regime-switching HMM: each month a person is *employed*, *part-time*, *job search*, *student* or *retired*, with sourced Swiss hazards, income by state, gross-to-net pay, and a vectorised cash engine. The game uses it for life states. |
| **Challenge materials** | [`challenge-instructions-and-data/`](challenge-instructions-and-data/) | Everything the organizers gave us, unchanged: the challenge PDF and docs, synthetic data (`Data/`), starter kit, FAQ, evaluation criteria, examples, resources. |

### Main features

- **Life timeline to age 120.** Drag forward to live months, drag back to rewind exactly. Right of "now" the
  timeline fans out the P10–P90 net-worth band and the share of simulated lives in each state.
- **Multiverse.** Every choice branches a new reality. Go back to any choice, undo it, or pin two realities
  side by side to see why they differ.
- **Six scores, no single "best" number:** net worth, cash safety, goals hit and more, each with a range.
- **Learned twin.** The regime HMM (life states) is combined with models learned from the synthetic data: a GMM
  for spending by state, December bonus, holiday months, month-to-month noise, and "people like you" (kNN/GMM
  on a normalised profile map).
- **💬 Ask companion.** An optional LLM agent that answers only with numbers the engine produced and can propose
  one allowed action that you confirm. Without an API key the game falls back to rule-based answers.
- **🌪️ What matters most?** A tornado chart per reality: each assumption moved ±20% on its own, ranked by how
  much it moves net worth at 30 / 40 / 65. Assumptions with no effect are listed (scenario isolation).
- **Negative cash is never free.** A shortfall is an automatic credit line at 10%/year up to CHF 30k; beyond that
  the engine says the plan is not financeable instead of inventing money. The Safety tile also reports how often
  savings run out after 65.
- **💱 Today's CHF.** One toggle shows every chart in January-2025 francs (the engine's own price index), so decades
  of inflation don't look like wealth.
- **Transparent assumptions.** Every assumption is tagged with where it came from and can be edited in the
  phone's Settings app. The Checks app runs 11 validation checks live, including a to-the-cent cross-check against the
  official starter kit.

The full gameplay guide, architecture, model and limitations are in [`game/README.md`](game/README.md). There's
a scripted demo walkthrough in [`game/DEMO.md`](game/DEMO.md).

## How to run it

Requirements: **Python 3.11+**, **Node.js 18+** and npm.

### The game (one command)

```bash
cd game
./start.sh
```

This creates `game/backend/.venv`, installs dependencies, starts the engine on http://localhost:8000 (API docs at
`/docs`) and opens the town at http://localhost:5173.

Or start them by hand in two terminals:

```bash
# terminal 1: engine
cd game/backend
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --reload --port 8000

# terminal 2: town
cd game/client
npm install
npm run dev
```

URL options: `?world=9` fixes the random world (for reproducible demos), and `?fresh` clears the saved game.

Controls: **WASD**/arrows or click to walk, **E**/Enter or click a building to enter, **P** for the phone,
**T** to pin the life timeline, **Esc** to close.

### Optional: the 💬 Ask companion (LLM)

Create a `.env` file in the repo root (it is git-ignored) with one provider:

```bash
LLM_PROVIDER=gemini        # or groq, or ollama (local, no key)
GEMINI_API_KEY=...         # or GROQ_API_KEY=...
```

Set `GAME_DISABLE_LLM=1` to turn it off completely.

### The regime twin on its own

```bash
pip install -r backend/requirements.txt
python backend/run_demo.py              # Alexia: baseline vs start working / master's / move abroad
```

The `backend/run_alexia_*.py` scripts explore specific decisions (when to do a master's, breaks, where to
live, first job, housing).

### Tests

```bash
cd game/backend && .venv/bin/pytest -q          # game engine (162 tests)
game/backend/.venv/bin/python -m pytest backend -q   # regime twin (46 tests), run from the repo root
cd game/client && npx tsc -b                    # client type-check
cd game/client && npm run smoke                 # headless walk through the demo (needs both servers + Chrome)
```

## Repo layout

```
.
├── game/                              the playable demo
│   ├── backend/                       FastAPI engine: app/ (engine, API, AI), learned/, peers/, tests/
│   ├── client/                        React + Vite pixel-art town
│   ├── start.sh                       one-command launcher
│   ├── README.md                      full game documentation
│   └── DEMO.md                        demo script
├── backend/                           regime-switching HMM twin
│   ├── twin/                          engine, transitions, income, net pay, cash engine, metrics, agent
│   ├── assumptions/                   editable, sourced assumption CSVs
│   └── run_*.py                       scenario scripts
└── challenge-instructions-and-data/   organizer materials (PDF, docs, Data/, starter-kit/, faq/, ...)
```

## Data and responsibility

All profiles and transactions come from the organizers' synthetic dataset in
`challenge-instructions-and-data/Data/`. We use no real personal data. The engine reports ranges under stated
assumptions and never says which life is "best". The game always shows that it is a simulation and not advice.
