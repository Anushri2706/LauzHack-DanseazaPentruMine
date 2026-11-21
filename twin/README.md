# Minimal Financial Twin

```
python3 -m twin.chat                                   # chat: describe yourself, ask what-ifs
python3 -m twin.chat --profile P00001                  # chat with a synthetic profile
python3 -m twin "What if I do a master's degree?" --profile P00001     # one-shot report, no LLM
python3 -m twin "What if I lose my job for 6 months?" --inputs twin/example_user.json
python3 -m unittest twin.test_engine -v                # sanity tests (must pass)
python3 -m twin.calibration                            # refit the saving calibration (writes calibration.json)
python3 -m twin.validate_decisions                     # check decisions against peers in that state (~20 s)
python3 -m twin.explore --profile P00001               # NSGA-II trade-off front across all decisions
python3 -m twin "explore all my options" --profile P00001
```

## Pipeline

1. **Saving calibration** (`calibration.py`, on by default; `--raw` switches it off). The ledger implies
   ~CHF 3,000/month saved, but similar people 10 years older hold only CHF 5-65k more. One keep factor
   per age band (share of any monthly surplus actually kept) is fitted so the median simulated decade
   matches the median wealth of background-matched peers 10 years older. Fitted values: 15-24 8%,
   25-34 18%, 35-44 6%, 45-54 6%, 55-64 2%, 65+ 29%. Deficit months are untouched; deliberate cuts and
   side income (explore levers) are kept in full. The size of monthly swings still comes from the ledger.
2. **Decisions** (`scenarios.py`) change income, spending or one-off costs on the calibrated baseline:
   master's, part-time, self-employed, have a child, move canton, job loss. Parameters are hand-set in
   `assumptions.py` (deliberately NOT taken from peers, so step 3 is not circular).
3. **Validation** (`validate_decisions.py`): each decision mapped to a present state of older peers
   (part_time, self_employed, family, canton) vs employed peers, matching ignores what the decision
   changes, calibration fitted on the other half of profiles. Checks sign, ranking, range.
4. **Trade-offs** (`explore.py`): NSGA-II over decision x expense cut (0-20%) x side income
   (CHF 0-800), objectives median wealth / months below buffer / CHF sacrificed per month.
5. **Language** (`chat.py`, `agents.py`): Gemini extracts the situation and decision (or "explore"),
   explains results; `check_numbers` flags any number not in the engine output.

## The engine (`sim.py`): one accounting rule

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
- Synthetic data and illustrative assumptions; not advice or a forecast.
- No retirement yet: income continues unchanged past 65 (a warning is shown).
- The ledger implies far more saving than the data's wealth levels; the saving calibration fixes the
  average level, not individual profiles (`--raw` shows the uncalibrated result).
- Decision parameters are illustrative; a master's in another canton is the only combined what-if.
- Moving canton disagrees with peers (people in expensive cantons are richer in the data): the engine
  does not assume a move raises pay.
- Rent by canton comes from synthetic medians, not official data and not student-specific.
