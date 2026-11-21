# Regime engine (HMM) — interface contract

The HMM module is a **stochastic scenario generator**. It answers one question:

> "Given who this person is, what they decide, and our assumptions, what life regime are they likely in each month, and what monthly net cash flow does that regime produce?"

It does **not** do cash accounting, taxes, investments, goals, UI or LLM narration. Those consume its output.

Related: [HMM digital twin plan](/cursor/stores/bc-e3d1c389-9d7a-4206-a8d2-334e3178188b/docs/hmm-digital-twin.md), [Challenge requirements](/cursor/stores/bc-e3d1c389-9d7a-4206-a8d2-334e3178188b/docs/challenge-requirements.md).

## Ownership split

| HMM owner (Victor) | Rest of team |
|---|---|
| Regime definitions, emission fitting, transition assumptions table | Cash/financial engine F: cash, investments, debt, tax, one-off costs |
| Filtering (current-regime belief), path sampling, interventions | Metrics: P(cash < reserve), P(goal met), sensitivity/tornado |
| Validation of emissions and calibration | UI (Streamlit/Plotly), LLM narration, pitch |

## Types

```python
STATES = ["employed", "part_time", "job_search", "student", "retired"]

@dataclass
class Profile:
    age: int
    employment: str            # initial regime hint
    ref_income: float          # CHF/month
    rent: float
    attributes: dict           # e.g. {"has_masters": False, "abroad": False}
    use_existing_loan_rate: bool = False  # new tuition debt: False = ETH 0%, True = this profile's personal loan

@dataclass
class Intervention:            # decisions = do() on the chain
    kind: str                  # "force_state" | "scale_hazard" | "set_attribute"
    start_month: int
    duration: int | None = None
    state: str | None = None   # for force_state
    transition: tuple[str, str] | None = None  # for scale_hazard, e.g. ("employed","job_search")
    factor: float | None = None
    attribute: tuple[str, object] | None = None

@dataclass
class RegimePaths:
    states: np.ndarray           # [n_paths, T] int, index into STATES
    income_chf: np.ndarray       # [n_paths, T] absolute monthly net income in CHF, from the income table
    savings_rate: np.ndarray     # [n_paths, T] net flow / income_chf, sampled from the state's fitted emission
    net_flow_chf: np.ndarray     # [n_paths, T] savings_rate * income_chf (team adds one-offs on top)
    occupancy: np.ndarray        # [T, K] P(state_t = k), analytic
    transition_draws: np.ndarray # [n_paths, K, K] sampled A (parameter uncertainty)
    debt_chf: np.ndarray            # [n_paths, T] extension: end-of-month tuition-debt total
    tuition_accrual_chf: np.ndarray # [n_paths, T] extension: loan principal booked that month
```

## Income table (how income is known at time t)

Income is absolute, not a multiplier on today's income (a student's current income is a useless base):

`income_chf_t = f(state_t, age_t, attributes, years_employed_t)`

- Baseline levels: median and spread of reference net income from `synthetic-profiles.csv`, by employment status, age band and canton.
- Assumptions (sourced, editable CSV): master's premium on entry salary, annual experience growth, unemployment insurance at 70–80% of last salary, abroad income/cost levels, student income.
- Example, Alexia on the master's path: CHF 1,200 while a student for 24 months, then the entry salary for "employed, master's, age 22", growing yearly, about 75% of the last salary during job-search months.

## Functions

```python
# Offline, once
fit_emissions(cash_balances: pd.DataFrame, profiles: pd.DataFrame) -> EmissionModel
load_transition_priors(path="assumptions/transitions.csv") -> TransitionPriors  # editable, sourced

engine = RegimeEngine(emissions, priors)

# "Twin syncs with reality"
engine.filter(profile, observed_net_flows: list[float]) -> dict[str, float]
#   -> {"employed": 0.93, "job_search": 0.05, ...}

# Main call the team uses
engine.simulate(
    profile: Profile,
    horizon_months: int = 60,
    n_paths: int = 5000,
    interventions: list[Intervention] = [],
    assumption_overrides: dict = {},   # slider values, e.g. {"job_loss_hazard": 0.006}
    initial_belief: dict | None = None,
    seed: int = 0,
) -> RegimePaths
```

## Extension: master's tuition debt

The five regimes stay. Debt is not a sixth state and it is not bucketed. `simulate` still returns every field it returned before. Two fields are added.

On the existing master's path — `force_state` `student` with a duration, and `set_attribute` `has_masters=True` at `start_month + duration` — each month of that pin books a tuition draw. The chain does not infer the total. It is a running total on the path, the same on every path, for a later adapter to read. A student month outside that pin, including someone who is already a student and a later return to study that does not end on `has_masters`, books nothing.

Inputs, in the order the code checks them:

- Annual tuition. `assumption_overrides["annual_tuition_chf"]` if set. Otherwise the profile's S001 `annual_tuition_chf` in `Data/scenario-assumptions.csv` (Alexia `P00001` is 8000; the file says `synthetic_assumption`, "Illustrative not verified Swiss tuition"). If that row is missing, `annual_tuition_chf` in `assumptions/income_assumptions.csv` (8000, the same P00001 figure).
- Annual loan rate, chosen once at the first month of the master's window (when the pin starts). `assumption_overrides["tuition_loan_annual_rate"]` if set: that override wins over both of the choices below. Otherwise `Profile.use_existing_loan_rate`, or a `set_attribute` of that same name already in force at the pin start. The profile field defaults to **false**. `load_profile` sets false. A `set_attribute` that lands after the pin has started does not change the rate. When the flag is **true**, the profile's `annual_interest_rate` in `Data/synthetic-liabilities.csv` (principal-weighted if there is more than one). A true flag with no liability row falls through to the assumptions rate. When the flag is **false**, that personal-loan rate is ignored. The rate is then `tuition_loan_annual_rate` in the assumptions CSV: **0**. That 0 is ETH Zurich's own student-loan rate, [SR 414.156](https://www.fedlex.admin.ch/eli/cc/2023/473/de) Art. 18 para. 3 ("Es werden keine Zinsen erhoben"), read **21 Nov 2026**. It is ETH's own loan rate, not a national SBFI rate. Alexia has no liability row and the flag is false, so she uses 0.

The monthly accrual is `annual_tuition_chf / 12`. The scenario file has one tuition per profile, so the draw is that number, not a new noise model. Let `m` be the annual rate divided by 12. Inside the window, with `D` equal to 0 before the first month:

`D_t = D_{t-1} * (1 + m) + accrual_t`

Outside the window, `D_t = D_{t-1}`. No further accrual and no further growth. Interest in the window is on the previous total only. The new draw does not earn interest in the month it is booked.

`net_flow_chf` is unchanged. For a student that is still the existing emission (income minus the spend-ratio rent bill, the weak net flow). Living costs stay there. The cash engine adds the accrual on top:

`C_t = C_{t-1} + net_flow_chf_t + tuition_accrual_chf_t - one_offs_t`

The accrual is loan proceeds added to cash and to debt. Interest raises `debt_chf` only. It is not a cash inflow.

For Alexia at the default rate of 0, 24 months at 8000/year ends at **16000** CHF of debt (the principal only). A path with no such pin stays at 0.

Open questions, not settled by this extension:

- The monthly rate is the annual rate divided by 12. Whether that should be the effective monthly rate `(1 + r) ** (1/12) - 1` is open.
- Interest stops when the pin stops. There is no repayment. Opening `opening_principal_chf` is not the start of this total.
- `study_duration_years` is 2 on every S001 row. The window length is the pin's duration, not that column.
- `TwinSession`'s master's decision still posts the annual tuition as a cash one-off. That payment is separate from this loan draw.

## How the team consumes it

```python
paths = engine.simulate(alexia, interventions=[Intervention("force_state", 0, 24, state="student")])
cash = cash_engine.run(alexia, paths)          # C += net_flow_chf + tuition_accrual_chf - one_offs
metrics = compute_metrics(cash, paths)         # fan quantiles, P(breach), P(goal)
```

## What the HMM learns vs assumes

- **Learned (from 28.8k person-months):** per-regime distribution of net flow / income (Student-t or 2-GMM), December 13th-salary offset.
- **Assumed (documented, editable):** transition hazards (job loss, job-search exit, graduation), Dirichlet concentration, income multipliers per regime, effect of attributes (e.g. master's raises income, lowers hazard).
- **Not predicted:** cash balances, point forecasts. Output is a distribution of plausible regime paths.

## What the HMM owner validates

1. Emission fit: simulated vs held-out net-flow distributions per regime (QQ / KS, 20% profiles held out).
2. Calibration backtest: filter on 2025, simulate 2026; real 2026 values should land in the P5–P95 band about 90% of the time.
3. Analytic occupancy (π₀Aᵗ) matches simulated occupancy.
4. Degenerate case: noise off, A = I, must reproduce the starter-kit deterministic output once plugged into the cash engine.

## Stub for parallel work

Until the real engine exists, ship a stub with the same signature that returns fixed regimes, constant `income_chf` and a constant `savings_rate` (the profile mean). The team builds the cash engine, metrics and UI against it.
