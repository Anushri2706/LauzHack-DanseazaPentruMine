# LauzHack-DanseazaPentruMine

## Backend: regime-switching financial twin (`backend/`)

A stochastic what-if simulator for a fictional person (default: Alexia, `P00001`). It is not a forecast.

```
regime_engine (HMM: life regime per month -> income_chf, savings_rate, net_flow_chf)
  -> cash_engine (starter-kit accounting, vectorised over paths)
  -> metrics (fan quantiles, P(cash < reserve), P(goal met), months per regime)
TwinSession wraps the pipeline per scenario; its methods are the LLM tools used by agent.py.
```

| Module | Role |
|---|---|
| `backend/regime/types.py` | `STATES`, `Profile`, `Intervention`, `RegimePaths` (interface contract) |
| `backend/regime/regime_engine.py` | `RegimeEngine.simulate/filter`; working **stub** mode; real HMM functions marked `TODO` (`fit_emissions`, `forward_filter`, `sample_regime_paths`, `sample_emissions`) |
| `backend/regime/transitions.py` + `backend/assumptions/transitions.csv` | Monthly hazards (sourced, editable assumptions) and Dirichlet uncertainty |
| `backend/regime/income.py` + `backend/assumptions/income_assumptions.csv` | Absolute income by regime, age/experience, attributes (master's, abroad), unemployment insurance |
| `backend/regime/net_pay.py` + `backend/assumptions/tax_assumptions.csv` | Gross pay to what lands in the account: AHV/IV/EO, ALV, accident insurance, BVG employee share by age, income tax (Vaud, `[EST]`). `income_chf` is net, `gross_income_chf` is gross. Switch: `apply_income_tax` |
| `backend/regime/cash_engine.py` | `C_{t+1} = C_t + net_flow - one_offs`; no clamping of negative cash |
| `backend/regime/metrics.py` | Risk and goal metrics |
| `backend/regime/twin_session.py` | Scenarios, decisions, assumptions; tools: `get_state`, `create_scenario`, `apply_decision`, `set_assumption`, `simulate`, `compare`, `sensitivity`, `sample_paths` |
| `backend/regime/agent.py` | Function-calling loop + tool schemas (key from `OPENAI_API_KEY`, never committed) |
| `backend/regime/validation.py` | Starter-kit deterministic equivalence, transition/occupancy checks; emission fit and calibration backtest `TODO` |

Run:

```bash
pip install -r backend/requirements.txt
python -m pytest backend -q        # smoke tests (stub pipeline end to end)
python backend/run_demo.py         # Alexia: baseline vs start working / master's / move abroad
```

All numbers in `backend/assumptions/*.csv` marked `PLACEHOLDER` must be replaced with cited values before the demo. Transition probabilities are assumptions, not learned: the synthetic data contains no regime switches.

## Regime engine

A regime HMM (`backend/regime/regime_engine.py`) for one person, default Alexia `P00001`. Each month of each path is one of `employed`, `part_time`, `job_search`, `student`, `retired` (`STATES` in `backend/regime/types.py`). A decision is an `Intervention` on that chain. The engine returns monthly income and net flow. `backend/regime/cash_engine.py` keeps the balance: cash starts at opening cash and adds `net_flow_chf` each month, minus one-off costs. That net flow is income minus spending, so cash is savings after spending.

From `backend/`, use this machine's interpreter:

```bash
/opt/anaconda3/bin/python run_alexia_goal_hit.py
```

`run_alexia_goal_hit.py` builds the HMM, calls `RegimeEngine.simulate`, rolls cash with `regime.cash_engine.run`, and scores the goal with `goal_hit_rate` in `backend/regime/metrics.py`. The other `backend/run_alexia_*.py` scripts on this branch call the same `simulate`. `backend/run_demo.py` is the `TwinSession` printout.

```python
from regime import Intervention, RegimeEngine, fit_emissions, load_transition_priors

engine = RegimeEngine(fit_emissions(), load_transition_priors())
paths = engine.simulate(
    profile,
    horizon_months=60,          # default 60
    n_paths=4000,               # default 5000
    interventions=[
        Intervention("force_state", 0, 24, state="student"),
        Intervention("set_attribute", 24, attribute=("has_masters", True)),
    ],
    assumption_overrides={"scaled_spending": 1},
    seed=0,
)
```

`simulate(profile, horizon_months=60, n_paths=5000, interventions=None, assumption_overrides=None, initial_belief=None, seed=0)` returns a `RegimePaths`: `states`, `income_chf`, `savings_rate`, `net_flow_chf`, `occupancy`, `transition_draws`, and, in hmm mode, `lump_chf` (zeros unless `scaled_spending` is on). `RegimeEngine(...)` runs in mode `"hmm"`. `RegimeEngine.stub()` is the deterministic mode.

`Intervention.kind` is `force_state`, `set_attribute`, `scale_hazard`, or `parental_leave`. `force_state` pins `state` at `start_month`. With `duration`, those months stay pinned. When the pin ends inside the horizon, `student` and `job_search` take one further month of `employed`, and any other state takes one further month of the regime from just before the pin. With `duration` omitted, only `start_month` is pinned. `set_attribute` writes `attribute` `(name, value)` from `start_month` through the rest of the horizon. `("has_masters", True)` is the graduate wage. Pass `scaled_spending` in `assumption_overrides` (also the row in `backend/assumptions/expense_assumptions.csv`, default `0`). At `1`, `IncomeTable.scales_spending` is on (the flag counts as on at `0.5` or above) and the bill is the expense-priors housing and non-housing recipe, plus a December bonus in income units, minus the yearly lump shock. That recipe replaces `spend_ratio * rent`. At `0`, net flow is income minus `spend_ratio * rent`, plus the December bonus.

`fit_emissions` learns a Student-t of `spend_ratio` (spend / rent) per regime on the training profiles. Household type shifts the location. Self-employed rows get their own Student-t and are used when the profile is self-employed and the state is `employed`. The December bonus is a separate empirical quantile draw from training Decembers. Transitions, income, and the consumption recipe are assumptions: monthly hazards in `backend/assumptions/transitions.csv` (the ledger has no regime switches), the income table in `backend/assumptions/income_assumptions.csv`, and the recipe in `backend/assumptions/expense_assumptions.csv`.

`goal_hit_rate(cash, start, goal)` is the share of paths whose cash is at least the target in the goal's month. `cash` has shape `(n_paths, n_months)`, and a `CashPaths` is accepted. Month 0 is `start` (`"2025-01"` or a date). `goal` is one row of `Data/synthetic-goals.csv` (`target_amount_chf` and `target_date`; `amount` and `date` are accepted names) or an `(amount, date)` pair. The index is `(Y_goal - Y_start) * 12 + (M_goal - M_start)`, and the day is ignored. A month inside the horizon is used as-is (`evaluated_at` `"goal_month"`). A goal past the horizon is scored on the last month (`"last_simulated_month"`). A goal before `start` is scored on month 0 (`"before_start"`). `median_cross_month` is the first month from which the cross-sectional median stays at or above the target through the end, and it is null when the last median is still short. `median_cash_at_goal`, `p10_cash_at_goal`, and `p90_cash_at_goal` are the linear quantiles in that same month. `run_alexia_goal_hit.py` scores Alexia's Emergency savings row (12,800 CHF by 2029-12-31) on the master's-now path: student for 24 months, then `has_masters`, `scaled_spending` 1, 4,000 paths, 60 months from 2025-01, seed 0. The printed `hit_rate` is that share of simulated paths. The return carries `disclaimer`: simulated under the stated assumptions.

This note is on `feature/hmm-decisions`. `main` was not updated.
