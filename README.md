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
| `backend/twin/types.py` | `STATES`, `Profile`, `Intervention`, `RegimePaths` (interface contract) |
| `backend/twin/regime_engine.py` | `RegimeEngine.simulate/filter`; working **stub** mode; real HMM functions marked `TODO` (`fit_emissions`, `forward_filter`, `sample_regime_paths`, `sample_emissions`) |
| `backend/twin/transitions.py` + `backend/assumptions/transitions.csv` | Monthly hazards (sourced, editable assumptions) and Dirichlet uncertainty |
| `backend/twin/income.py` + `backend/assumptions/income_assumptions.csv` | Absolute income by regime, age/experience, attributes (master's, abroad), unemployment insurance |
| `backend/twin/cash_engine.py` | `C_{t+1} = C_t + net_flow - one_offs`; no clamping of negative cash |
| `backend/twin/metrics.py` | Risk and goal metrics |
| `backend/twin/twin_session.py` | Scenarios, decisions, assumptions; tools: `get_state`, `create_scenario`, `apply_decision`, `set_assumption`, `simulate`, `compare`, `sensitivity`, `sample_paths` |
| `backend/twin/agent.py` | Function-calling loop + tool schemas (key from `OPENAI_API_KEY`, never committed) |
| `backend/twin/validation.py` | Starter-kit deterministic equivalence, transition/occupancy checks; emission fit and calibration backtest `TODO` |

Run:

```bash
pip install -r backend/requirements.txt
python -m pytest backend -q        # smoke tests (stub pipeline end to end)
python backend/run_demo.py         # Alexia: baseline vs start working / master's / move abroad
```

All numbers in `backend/assumptions/*.csv` marked `PLACEHOLDER` must be replaced with cited values before the demo. Transition probabilities are assumptions, not learned: the synthetic data contains no regime switches.
