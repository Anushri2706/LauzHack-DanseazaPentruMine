# Minimal Financial Twin

```
python3 -m twin.chat                                   # chat: describe yourself, ask what-ifs
python3 -m twin.chat --profile P00001                  # chat with a synthetic profile
python3 -m twin "What if I do a master's degree?" --profile P00001     # one-shot report, no LLM
python3 -m twin "What if I lose my job for 6 months?" --inputs twin/example_user.json
python3 -m unittest twin.test_engine -v                # sanity tests (must pass)
python3 -m twin.limits --all                           # model simplifications + every placeholder / estimate
python3 -m twin.calibration                            # refit the saving calibration (writes calibration.json, ~1 min)
python3 -m twin.validate_decisions                     # check decisions against peers in that state (~1 min)
python3 -m twin.explore --profile P00001               # NSGA-II trade-off front: decision x start month x levers
python3 -m twin.explore --profile P00001 --only masters  # "when should I do the master's?"
python3 -m twin "explore all my options" --profile P00001
python3 -m twin "When should I do a master's?" --profile P00001
```

## Pipeline

1. **Saving calibration** (`calibration.py`), now fitted on the **regime engine** (`backend/regime`,
   run through `engine.py`, which adds cash, investments, debt and inflation). The engine's saving
   comes from spending fitted on the 24-month ledgers: a median surplus of ~CHF 2,000-6,000/month for
   working ages. Similar people 10 and 20 years older in the data hold far less. Swiss statistics don't
   close the gap either: with HABE spending (own rent + 540 + 25% of gross pay) the factor was still
   11-29%. So the synthetic wealth table disagrees with both, the data wins, and the factor is shown
   openly. One keep factor (share of any monthly surplus kept) per age band x income tertile is fitted so
   the median simulated wealth after 10 and 20 years matches peers 10 and 20 years older. Fitted
   values: median 15% (fitted cells), 4-15% for working ages with mid/high income, higher for low income;
   the 55-64 and 65+ low-income cells cannot be fitted and are left uncalibrated (see Known limits). Income cells are
   needed because in this data wealth barely rises with income. Deficit months are untouched. The old
   engine (`sim.py`, still behind the CLI, chat and explorer) keeps its own factors in `calibration_sim.json`.
2. **Decisions** (`scenarios.py`) change income, spending or one-off costs on the calibrated baseline:
   master's, part-time, self-employed, have a child, move canton, job loss. Parameters are hand-set in
   `assumptions.py` (deliberately NOT taken from peers, so step 3 is not circular).
3. **Validation** (`validate_decisions.py`, regime engine): each decision mapped to a present state of
   older peers (part_time, self_employed, family, canton) vs employed peers; matching ignores what the
   decision changes; calibration fitted on the other half of profiles. Checks sign, ranking and range at
   10 and 20 years. Held-out baseline inside the peers' middle 50%: 48% at 10 and 20 years (1-2%
   uncalibrated, 46% after the move into TwinSession); decisions 37-56%. Moving canton and (at 20 years) having a child disagree in sign with
   peers, which pulls the ranking check negative.
4. **Trade-offs** (`explore.py` -> `TwinSession.explore`, NSGA-II in `backend/regime/explore.py`): a plan is
   decision x start month (0-36, step 6) x expense cut (0-20%) x side income (CHF 0-800). Objectives: median
   real net worth at the horizon, months with cash below a 3-month buffer, CHF/month sacrificed. Each
   (decision, start month) is simulated once; the levers are scored on the cached paths (~30 simulations,
   ~900 plans, a few seconds). `timing` lists each decision's result by start month, so "when should I do
   the master's?" is answered by the search. A child cannot start before month 12.
5. **Language** (`chat.py`, `agents.py`): Gemini extracts the situation, the decision and its start month (or
   "explore", optionally focused on one decision for a "when" question); the chat calls TwinSession's
   tools (`simulate`, `compare`, `explore`) through `engine.py`; Gemini explains the tool output and
   `check_numbers` flags any number not in it. Directions (rises/falls) of timing effects are written by
   code, because the number check cannot catch a wrong direction.

## One engine

`engine.py` builds a `TwinSession` (backend/regime) with `wealth=True`: the regime engine for life states
(employed, part-time, job search, student, retired; Swiss pay, tax and unemployment insurance) and the
wealth layer (`backend/regime/wealth.py`: keep factor, inflation regimes on spending, investments, loan
interest, real net worth). Spending in today's regime is shrunk toward the person's own ledger (or the
non-rent spending a new user gave). A student's baseline is "start working now". The old engine below
(`sim.py`) remains only for its tests and `validate_learned.py`.

## The old engine (`sim.py`): one accounting rule

Every month, in each of N simulated futures:

```
cash += income - spending - one-off costs - loan interest + noise
```

| Term | Comes from |
|---|---|
| income | your input, changed by the scenario (`scenarios.py`), +1%/yr wage growth |
| spending | your rent + non-rent spending (yours, or a stated typical value), x inflation |
| one-off costs | scenario: tuition, moving cost |
| loan interest | your debt x its rate (principal kept constant) |
| noise | a random swing in CHF: your own 24-month history for synthetic profiles, CHF 300 for new users. Fixed in francs, never a share of income |

Uncertainty across futures: inflation regime (low/base/high, switching yearly), noise, and returns on
existing investments. Negative cash is reported as an unfunded shortfall; no loan or interest is invented
(starter-kit rule). Every hand-set number is in `assumptions.py` (`--assumptions` prints them).

Scenario rules (`scenarios.py`):
- **Baseline:** nothing changes. A student's baseline is "start working now" at CHF 4,200/month, or their
  current income if higher.
- **Master's:** a worker keeps 50% of income while studying; a student keeps their current income.
  Tuition yearly, then +15% income. Studying in another canton adds a CHF 5,000 moving cost.
- **Job loss:** 70% of income for N months. **Move:** rent x canton ratio (synthetic medians) + CHF 5,000.

### Sanity tests (`test_engine.py`)
Hand-checkable cases with growth, inflation, returns and noise switched off, e.g.
earn 25,000 and spend 4,000 -> exactly 2,522,000 after 10 years; spend 500 more than you earn -> exactly
-59,000 and no invented interest; a master's payoff equals its formula; a master's never wins without a
salary premium; noise is the same in CHF for a rich and a poor person; the Monte Carlo median matches the
flat-world result within 1%. Run them after any engine change.

## Chat with an LLM (`chat.py`, `agents.py`)
Per message: the LLM extracts inputs and the what-if as JSON -> code validates (ranges, yearly -> monthly,
"5k", "1'600", cities -> cantons) -> unusual values are confirmed with the user (income > CHF 20k/month,
rent >= income, cash > CHF 2M) -> employment is never guessed (a master's needs it) -> engine ->
the LLM explains -> `check_numbers` flags any number or percentage not in the engine output.
The LLM never computes. Without an LLM: keyword router and a plain table.

LLM setup in `.env` (git-ignored; template `.env.example`): `LLM_PROVIDER` = gemini | groq | ollama,
`LLM_MODEL` (comma-separated = fallback order), `LLM_MAX_CALLS` (default 40). Default Gemini models:
`gemini-flash-lite-latest`, then `gemini-flash-latest`. Free tier: prompts may be used by the provider,
so use fictional details only.

## Context, not part of the engine
- **Peer path** (`trajectory.py`): your net-wealth percentile among people your age in the data, read at
  each later age. Shown next to results as context; it does not change the simulation. Older peers are
  different people, not your future.
- **Learned estimates** (`learned.py`): typical non-rent spending, cash and portfolio for inputs the user
  did not give, each tagged "estimated" and named in the warnings. Seasonality and income-scaled
  volatility were removed from the engine: scaling noise by income gave a CHF 25k earner CHF 8.5k of
  random swing per month.
- **Checks:** `validate_learned.py`, `validate_peers.py` (background predicts wealth no better than age),
  `validate_decisions.py` (step 3). An alternative calibration (hidden spending as a fixed CHF amount
  that does not shrink when income falls) was tested and dropped: it gave part-time -CHF 335k.
- **Analysis:** `peer_correlations.py` (which young traits move with older peers' finances).

## Known limits

`limits.py` holds the list shown by `python3 -m twin.limits`, the chat's `limits` command and the CLI; every
chat answer ends with a one-line summary of it.
- **Calibrated vs uncalibrated is always visible.** A code-written banner above every result says whether it
  is calibrated, the keep factor and the group it was fitted on, and gives the uncalibrated median next to
  the calibrated one (the search's banner does the same for the as-is plan).
- **The calibration is a group correction, not your path.** Older peers are different people; one factor
  per age band x income third makes medians match, not individuals (held-out: ~46% inside the peers'
  middle 50%).
- **Groups it cannot fit are left uncalibrated.** Low-income people aged 55+ are in surplus only ~15% of
  simulated months (retirement on 47.5% of the last wage, spending unchanged), so no factor up to 200%
  closes the gap; those cells used to sit at the 200% cap. They now keep 100%, say "NOT CALIBRATED" and
  give the reason (calibration.json `fitted: false`).
- **Wealth engine simplifications:** interest-only debt (principal constant), normal monthly returns,
  three inflation regimes with hand-set switching, no individual cash flows for investments or debts, no
  pillar 2/3 capital. Nothing shows these return or inflation assumptions predict real outcomes.
- **Assumption register:** 111 assumptions across `backend/assumptions/*.csv` and the decision costs in
  `assumptions.py`: 42 sourced (law or official statistics), 30 estimates (incl. the Vaud income tax
  curve), 36 placeholders (incl. all market assumptions), 3 switches. `--all` lists the 66 to validate.
- Synthetic data; not advice or a forecast. Moving canton changes housing cost only, not pay.
