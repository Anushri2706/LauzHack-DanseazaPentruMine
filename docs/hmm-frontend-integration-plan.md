# HMM and frontend integration plan

Plan only. Implementation comes later, on this branch (`feature/hmm-and-frontend`). Do not merge to `main`. Do not change `RegimeEngine.simulate` or `RegimeEngine.filter`. Do not restyle the town.

The contract to extend is the Project-store document `hmm-regime-engine-api.md` (this repo does not contain that file; `backend/twin/types.py` points at it). A later edit adds the sections in [Contract additions](#contract-additions). It leaves the current signatures in place. The game does not import `backend/twin` today (`README.md`, Regime engine section).

## Current simulate / filter contract

The HMM is a stochastic scenario generator. It does not do cash accounting, taxes, investments, goals, UI, or narration. Cash is `backend/twin/cash_engine.py`. Metrics are `backend/twin/metrics.py`.

From the API doc, the calls the rest of the team uses:

```python
engine.filter(profile, observed_net_flows: list[float]) -> dict[str, float]
#   -> {"employed": 0.93, "job_search": 0.05, ...}

engine.simulate(
    profile: Profile,
    horizon_months: int = 60,
    n_paths: int = 5000,
    interventions: list[Intervention] = [],
    assumption_overrides: dict = {},
    initial_belief: dict | None = None,
    seed: int = 0,
) -> RegimePaths
```

`RegimePaths` in that doc is `states`, `income_chf`, `savings_rate`, `net_flow_chf`, `occupancy`, `transition_draws`. Consumption is `cash_engine.run(profile, paths)` then `metrics.compute_metrics(cash, paths)`.

`Intervention.kind` in that doc is `force_state`, `scale_hazard`, or `set_attribute`.

The code already implements that contract, with these additions that the later API edit must keep when it extends the doc (they are not part of this plan's new work):

- `RegimeEngine.filter(profile, observed_net_flows, assumption_overrides=None)` still returns `dict[str, float]`, one weight per name in `STATES`, summing to 1. It returns only the last row of `forward_filter`. It does not accept `initial_belief`. In HMM mode the prior inside the filter is `_initial_belief`: 0.95 on `profile.initial_state` and `0.05 / K` on the rest. An empty flow list raises `ValueError`. Stub mode ignores the flows and returns 0.9 on the initial state and `0.1 / (K - 1)` on the others (`test_filter_on_alexia_and_a_shift`).
- `RegimeEngine.simulate` matches the signature above (`interventions=None` is treated as `[]`). With `initial_belief` set, that vector is the path prior as given, with no extra 0.05 mix. With `initial_belief=None`, month 0 is one-hot on `profile.initial_state`. HMM mode also fills `RegimePaths.lump_chf` (zeros unless `scaled_spending` is on). `RegimeEngine.stub()` is the deterministic mode (`README.md`).
- `Intervention.kind` also includes `parental_leave`.
- `Profile` also has `profile_id`, `name`, `opening_cash`, `mean_savings_rate`, `start_month_of_year`.
- `STATES` is `employed`, `part_time`, `job_search`, `student`, `retired`.

`simulate` and `filter` are pure. `RegimeEngine` stores emissions, priors, and the income table. It does not store a person, a belief, or a balance sheet.

## Where decisions live today

HMM named decisions are `TwinSession.DECISIONS` in `backend/twin/twin_session.py`. `apply_decision` rejects any other kind. `_decision_to_interventions` is the only map from a name to `Intervention`s plus `cash_engine.OneOff`s:

| `DECISIONS` key | What it appends |
|---|---|
| `masters` | `force_state` `student` for `duration` (default 24), then `set_attribute` `("has_masters", True)` at `t0 + duration`. Tuition `OneOff`s, default 8000 CHF per year. |
| `start_working` | `force_state` `employed` at `t0`, `duration` omitted (one pinned month, then the chain). |
| `move_abroad` | `set_attribute` `("abroad", True)`. Optional `force_state` `job_search` for `job_search_months`. Relocation `OneOff`, default 6500. |
| `job_loss` | `force_state` `job_search`. `duration` omitted means the exit hazard runs. |
| `parental_leave` | `parental_leave` at `t0`. |
| `part_time` | `force_state` `part_time`. |
| `downturn` | `scale_hazard` on `("employed", "job_search")`, `factor` default 2. |
| `one_off` | no intervention; one `OneOff`. |
| `force_state`, `scale_hazard`, `set_attribute` | the raw intervention. |

`TwinSession.create_scenario(..., copy_from=)` copies interventions, one-offs, assumptions, and the decision log. It does not copy a filter belief or observations. `TwinSession.compare(a, b)` subtracts two full-horizon summaries from `simulate` + `cash_engine.run`. Both use the session seed. There is no snapshot of hidden state and no "stop at the next decision".

Other HMM attributes that are not `DECISIONS` keys, applied only as `set_attribute` inside `IncomeTable`:

- `has_masters`, `masters_field` (`same` / `higher` / `lower`), `has_phd`, `abroad`, `zurich`, `bachelor_age_wage` (`income.py`).
- `rent_step` and `owns_home` (`IncomeTable.consumption_paths` / `rent_paths`). They change the housing bill only when `scales_spending()` is true (`scaled_spending` ≥ 0.5). `owns_home` locks the bill at the rent already being paid. The purchase is not an asset: `expense_assumptions.csv` `purchase_deposit_fraction` (0.15) is applied by `cash_engine.apply_fractional_drop`, which subtracts that fraction of that month's cash on each path. `rent_paths` with `scaled_spending` off does not read `owns_home` or `rent_step`.

`bachelor_age_wage` replaces the running wage with `aged_bachelor_salary`. It is a wage reset on a job change (`run_alexia_20y_later.py`). It is not an enrollment decision.

Game decisions are separate and deterministic. `game/backend/app/engine/decisions.py` `options` / `apply_decision` / `Decision`. `decision_reasons` fires at month 0, every 12th month, and when `program_end == t`. `game/backend/app/engine/game.py` `ACTION_TYPES` adds `buy_home`, `sell_home`, `change_flat`, `quit_job`, `reduce_hours`, `full_hours`, `ask_raise`, habits, loans. `LIFE_PLACE` puts `master_ft`, `master_pt`, and `drop_out` in the `university` building. `State` holds `cash`, `investments`, `debt`, `mortgage`, `home_value`, `home_id`. Rollback in the game is replay from month 0 (`run_turns` / `run_branch`), not a stored HMM belief.

The client calls `api.state` → `POST /game/state`, `api.projection` → `POST /game/projection`, `api.compare` → `POST /compare` (`game/client/src/api/client.ts`). Those handlers call `run_turns`, `step_month`, `make_worlds`, and `compare_games`. None of them call `RegimeEngine`.

## Decision map

Map only symbols that exist. Do not invent an intervention kind for a gap.

| Area | HMM (use these) | Game (already on screen) | Not in the HMM |
|---|---|---|---|
| University | Regime `student`. `IncomeTable.student_pay`. `bachelor_age_wage` is a later wage, not a degree start. `transitions.csv` says graduation is a pinned `force_state`, not a monthly hazard. | Place id `university`. No bachelor-enrollment action. Opening status can be `studying`. | No `DECISIONS` key for starting university. |
| Master's | `DECISIONS["masters"]` only. Field and PhD stay `set_attribute` (`masters_field`, `has_phd`), which `full_time_salary` already applies. | `master_ft`, `master_pt`, `drop_out`, `stay_student`, `continue`. Smoke test clicks "Full-time master's" at the university door (`game/client/scripts/smoke.mjs`). | No part-time pin, no drop-out, no "stay a student one more year". |
| Job | `start_working`, `job_loss`, `part_time`, `parental_leave`, `downturn`, and the job-search months on `move_abroad`. | `start_work`, `quit_job`, `reduce_hours` (`work_share` 0.8), `full_hours`, `ask_raise`, `move_abroad`, `return_home`. | Quit, 80%, raise, and return-home are not `DECISIONS` keys. `part_time` is a regime pin, not `work_share`. |
| House | `set_attribute` `("rent_step", True)` and `("owns_home", True)`, plus `cash_engine.apply_fractional_drop` with `purchase_deposit_fraction`. | `buy_home`, `sell_home`, `change_flat`. Balance sheet is `State.home_id`, `home_price`, `home_value`, `mortgage`. | No `DECISIONS` key. No home equity, mortgage principal, or sale inside `cash_engine` (`CashPaths` is opening cash and `cash[n_paths, T]`). `cash_engine` says investments, debt, tax, and inflation are still TODO. |

`masters` is the one named decision that lines up with a control the town already shows (`master_ft`). The adapter may translate that one action as soon as workstream 2 starts. University, part-time master's, and the game's house purchase stay unmapped until the open questions at the end are answered.

## Contract additions

Add these beside `simulate` and `filter`. Do not change those two signatures, their defaults, or the meaning of `RegimePaths`.

### `DecisionSnapshot`

A frozen record the caller stores. `RegimeEngine` does not hold it.

Fields the current types can fill:

- `month: int`
- `profile: Profile`
- `belief: dict[str, float]` — the dict `filter` returns (last month only)
- `observed_net_flows: list[float]` — the list passed to `filter`. The engine does not keep it.
- `interventions: list[Intervention]`
- `one_offs: list[cash_engine.OneOff]`
- `assumption_overrides: dict`
- `seed: int`
- `parent_id: str | None` and `snapshot_id: str` — the branch tree. This is not `TwinSession.scenarios` (flat dict, no belief) and not the client `TNode` (turns only).

Restore is a pure replay, the same idea as git checking out a commit, and the same idea as the game replaying turns:

1. `filter(snapshot.profile, snapshot.observed_net_flows, snapshot.assumption_overrides)` must equal `snapshot.belief`. `filter` cannot be warm-started; the whole list is the history.
2. The next `simulate` receives `initial_belief=snapshot.belief`, `interventions=snapshot.interventions`, and `seed=snapshot.seed`.

There is no balance-sheet field on `RegimeEngine`, on `Profile` beyond `opening_cash`, or on the filter return. `CashPaths` is a path ensemble, not one restorable statement. What a snapshot stores for "balance sheet" is an open question. Do not add investments, debt, or a house onto the HMM inside this work.

`forward_filter` already returns `filtered[T, K]`, and `filter` throws that array away. Whether a snapshot also stores that array is an open question. The last-month dict plus the observation list is enough to reproduce today's `filter` return.

### `simulate_until_next_decision`

```python
engine.simulate_until_next_decision(
    profile: Profile,
    decision: str,
    *,
    from_month: int,
    until_month: int,
    n_paths: int = 5000,
    interventions: list[Intervention] | None = None,
    assumption_overrides: dict | None = None,
    initial_belief: dict | None = None,
    seed: int = 0,
) -> RegimePaths
```

Behavior that is fixed:

- `decision` must be a key of `TwinSession.DECISIONS`, or the name `rent_step` / `owns_home` for the two housing attributes that already exist. Anything else, including a university enrollment name, raises `ValueError` the same way `apply_decision` does.
- Resolve `decision` with the same intervention list `_decision_to_interventions` builds for that kind (`rent_step` and `owns_home` are a single `set_attribute` at `from_month`). Append it to `interventions`.
- Call `simulate` once. Do not sample inside the new method.
- `horizon_months` passed to `simulate` is `until_month - from_month`. Require `until_month > from_month`. The caller supplies `until_month`. This method does not invent a birthday calendar or a pin-end calendar (see open questions).
- Return that `RegimePaths` unchanged. The outcome distribution is the existing arrays, summarized later with `metrics.fan_quantiles`, `metrics.compute_metrics`, and `metrics.goal_hit_rate`. Do not add a second path type.

`EXIT_STATE` (`student` → `employed`, `job_search` → `employed`) stays inside `sample_regime_paths`. A master's roll whose `until_month` equals `from_month + duration` does not include the exit month; a roll one month longer does, because that is how `force_state` with a duration already works (`test_simulate_contract_interventions_and_heterogeneity`). The wrapper does not special-case it.

### `branch_decisions`

```python
branch_decisions(
    engine: RegimeEngine,
    snapshot: DecisionSnapshot,
    decisions: list[str],
    *,
    until_month: int,
    n_paths: int = 5000,
) -> dict[str, RegimePaths]
```

One `simulate_until_next_decision` per name, from the same snapshot: same `profile`, same `initial_belief`, same `seed`, same overrides, same already-committed interventions. The only difference is the extra interventions of that option. Return the dict keyed by decision name, side by side.

This is the short-horizon form of `TwinSession.compare` and of `compare_games` forcing both realities into the same world. It does not replace either function. It does not write a snapshot and it does not append observations. Commit is separate, so a fork can be shown before it is kept.

### What the later API-doc edit adds

In `hmm-regime-engine-api.md`, after the current `simulate` / `filter` block:

- The code additions already listed above (`assumption_overrides` on `filter`, `parental_leave`, `lump_chf`, stub mode, last-row-only filter).
- `DecisionSnapshot`, `simulate_until_next_decision`, and `branch_decisions`, marked as wrappers over `simulate` and `filter`.
- The decision map table, including the rows that have no HMM name.
- One sentence that cash, the deposit drop, and metrics stay outside the HMM, as the ownership table already says.

Do not rewrite the question the HMM answers, and do not delete the stub paragraph.

## Workstream 1 — HMM (`backend/`)

Order inside this workstream:

1. `decision_interventions(kind, start_month, params) -> (list[Intervention], list[OneOff])` in `backend/twin/twin_session.py`, extracted from `_decision_to_interventions` so the session and the wrapper share one map. Behavior of `apply_decision` stays the same.
2. `DecisionSnapshot` in `backend/twin/types.py`. No balance-sheet fields until that open question is answered. Include `opening_cash` only via the `Profile` already on the snapshot.
3. `simulate_until_next_decision` and `branch_decisions` on `RegimeEngine` in `backend/twin/regime_engine.py`. Both call `simulate`.
4. Export the new names from `backend/twin/__init__.py`.

Do not change `sample_regime_paths`, `forward_filter`, `IncomeTable`, or `cash_engine.run`.

### Tests (workstream 1)

Add cases to `backend/tests/test_regime_engine.py`. Existing `test_simulate_contract_interventions_and_heterogeneity` and `test_filter_on_alexia_and_a_shift` stay as they are and must still pass.

- Signature lock: `inspect.signature` of `simulate` and `filter` is unchanged. `simulate_until_next_decision("masters", ...)` with `dirichlet: 0` and a fixed `until_month` equals `simulate` called with the master's interventions and `horizon_months=until_month - from_month` (same `seed`, same `n_paths`). Compare `states`, `income_chf`, and `net_flow_chf`.
- `masters` at month 0 with `until_month=24` matches today's pin: `states[:, :24]` are `student` (`STATE_INDEX["student"]`). That is the existing contract, not a new exit rule.
- Unknown kind `"university"` raises `ValueError`. `"buy_home"` raises `ValueError`. `"owns_home"` returns a single `set_attribute` and does not create a mortgage array.
- `branch_decisions` on `["masters", "start_working"]` returns two `RegimePaths` with the same `n_paths`. Same seed, same snapshot, called twice, gives identical arrays. The two `states` arrays differ.
- Snapshot restore: `filter(profile, snapshot.observed_net_flows, snapshot.assumption_overrides) == snapshot.belief` for Alexia's ledger (`observed_net_flows("P00001")`, belief `student > 0.9`). Building a snapshot does not add attributes to the engine instance.
- `branch_decisions` does not append to `observed_net_flows` and does not call `filter`.

## Workstream 2 — adapter (`game/backend/app/engine/`)

New module `game/backend/app/engine/hmm_adapter.py`. It is the only game module that imports `backend/twin`. `simulate.py`, `decisions.py`, `game.py`, `finance.py`, `montecarlo.py`, and `state.py` stay free of that import. `step_month` stays the town's month stepper.

The adapter is a translator:

| Town input | HMM call |
|---|---|
| `Action("life_choice", "master_ft")` at month `t` | `decision="masters"`, `from_month=t`, params `duration` from the game's `master_ft_months` only if that number is passed through as the master's `duration`. Default 24 is the HMM default. Do not also emit `master_pt`. |
| A year advance | one `simulate_until_next_decision` with `until_month = from_month + 12` |
| Side-by-side options before commit | `branch_decisions` |
| After a block is committed | append the committed net flows, then `filter` |
| Life finished, or the player asks for the final plot | one `simulate` over the remaining horizon with `initial_belief` from that `filter` and the committed interventions, then `cash_engine.run` and `metrics.fan_quantiles` |

"Sleep for a year" is that 12-month call, once per year, not `sleep(1)` twelve times and not a 60- or 120-month `simulate`. The town control that can already span a year is the home button "Sleep until next birthday" (`sleepUntilBirthday` in `game/client/src/store/actions.ts`), and it sleeps 12 months only when `months_to_birthday == 12`. It currently commits a `TNode` and posts `/game/state`. The adapter does not replace that path until the frontend workstream wires a year action at it. `sleep(1)` ("Sleep → next month") is unchanged in this plan.

Commit: when the block returned by `simulate_until_next_decision` ends, the adapter appends a `list[float]` of monthly net flows to `observed_net_flows` and calls `filter`. The function that turns a `RegimePaths` into that list is a seam, `commit_net_flows(paths) -> list[float]`, with the length of the block. Which reduction is "the mean/convergent simulation" is an open question. The seam is called in one place so the answer can be filled in without touching `filter`.

The final plot payload is the dict `metrics.fan_quantiles` already returns (`p05`, `p25`, `p50`, `p75`, `p95`, each length `T`) on the cash from `cash_engine.run`, plus `metrics.DISCLAIMER` and `paths.occupancy`. That is "the final simulated-life distribution". It does not replace `game_projection`, which remains the town Monte Carlo (`make_worlds`, P10/P50/P90 of net worth and cash) until a screen is pointed at the new payload.

JSON for the client goes through new handlers next to the existing ones in `game/backend/app/main.py` and `service.py`, the same shape as `POST /game/projection`: the engine stays pure, the route only translates JSON. Suggested paths, all new:

- `POST /hmm/branch` → `branch_decisions`
- `POST /hmm/commit` → append flows, `filter`, return the new `DecisionSnapshot` as JSON (no arrays of paths)
- `POST /hmm/distribution` → the fan above

`POST /game/state` and `POST /game/projection` stay. An adapter that is not called leaves the town identical.

### Tests (workstream 2)

New `game/backend/tests/test_hmm_adapter.py`. Run the existing `game/backend/tests/test_game.py` and `test_api.py` as well; they must pass with no assertion changes.

- Import lock: `hmm_adapter` is importable, and `app.engine.simulate`, `app.engine.game`, `app.engine.decisions` do not import `twin`.
- `master_ft` at month 0 produces the same intervention kinds as `DECISIONS["masters"]`: `force_state` / `student`, then `set_attribute` / `has_masters`. `master_pt`, `buy_home`, and a made-up university id are rejected, not guessed.
- Year advance: one call, `horizon_months == 12`, one `simulate`. It does not call `step_month`.
- `branch_decisions` from a snapshot leaves `observed_net_flows` the same length.
- Commit seam: pass a hand-built `list[float]` through the seam (do not assert a mean or a median). After commit, `filter` on the longer list matches `test_filter_on_alexia_and_a_shift`'s direction when the appended months are the employed net used in that test (belief on `employed` rises). The production body of `commit_net_flows` stays unimplemented until the open question is answered; the test injects the list.
- Final fan: after that commit, the distribution endpoint's `p50` has length equal to the remaining horizon, and the payload includes `metrics.DISCLAIMER`.

## Workstream 3 — frontend (`game/client/`)

Style stays. Reuse `btn`, `phone`, `phone-nav`, `mv-node`, `cmp-table`, and `MoneyChart`'s `ComposedChart`. No new typeface, palette, or layout.

What is already on screen, and must keep working:

- HUD button `📱 Phone` (`HudBar.tsx`, title "Open your phone (P)") and the `P` key (`GameScreen.tsx`) set `phone: 'money'`.
- `Phone.tsx` apps: Money, Multiverse, Goals, Settings, Checks, Mail (`AppId` in `store/game.ts`).
- `MultiverseApp` draws the client `TNode` tree, `goBack` / `continueReality`, and pin A/B. Compare posts `/compare` (`compare_games`), which is the town engine in one shared world. Nodes store turns, not an HMM belief.
- `MoneyChart` plots the lived path, `no_surprises`, and P10–P90 from `/game/projection`.
- University door already offers "Full-time master's". Do not add a second button with that accessible name. `smoke.mjs` clicks it.

Add UI only for HMM decision kinds the town does not already show, and only after the map says the kind is real. `parental_leave` is a real `DECISIONS` key with no town button. A control for it uses `available_actions` / `_entry` (place, label, preview, enabled, reason), the same row shape as `master_ft`. Which building it sits in is an open question; until that is answered, do not put it on the university door next to the master's.

There is no control labeled "check your phone". The phone menu that does exist is the HUD button and `Phone.tsx` above. Branch browsing and fork compare already exist for town turns, not for `DecisionSnapshot`s. The new button's label and which existing surface it opens are an open question. Do not add a second phone, and do not add the button until that question is answered.

What the frontend does once those questions are answered, and not before:

- A year action calls `POST /hmm/branch` or the year advance, then `POST /hmm/commit` only after the player keeps a fork. Compare-before-commit reads the `/hmm/branch` payload and does not call `/hmm/commit`.
- The final distribution is a `MoneyChart`-style chart bound to `fan_quantiles` from `POST /hmm/distribution`. It is an added series or an added phone section. It does not restyle the lived town line.
- `commitSleep`, `rollbackTo`, and `TNode` stay the town tree. An HMM snapshot id may be stored on the node later; this plan does not make `TNode` the HMM snapshot. The client store comment still holds for money: the backend replays, the store holds choices.

### Tests (workstream 3)

No component test runner in `game/client`. Checks:

- `npm run typecheck` in `game/client`.
- `game/backend/tests/test_api.py` still accepts a `master_ft` turn on `/game/state`.
- `node scripts/smoke.mjs` still queues "Full-time master's" and still opens the phone. Run it only when the town backend and Vite are up, as the script's header says. New copy must not steal the accessible name `/Full-time master's/`.
- When the phone-button question is answered, extend `smoke.mjs` with one step that opens that control, sees two fork labels from a fixture `/hmm/branch` response, and does not commit. Do not add that step before the control exists.

## Work order

1. This document. No code, no API-doc edit, no PR.
2. Workstream 1 and its tests. `game/` untouched.
3. Later, separate change: extend the store's `hmm-regime-engine-api.md` as specified above. Still no game code.
4. Workstream 2 and its tests. Town routes unchanged. New `/hmm/*` routes only.
5. Workstream 3, and only the pieces whose open questions are closed. `parental_leave`'s building and the phone button wait. `master_ft` needs no new town control.
6. Smoke the town. Do not merge to `main`.

## Open questions

No answer is implied by the code. Do not pick one while implementing the steps above.

1. University. There is no HMM decision for starting a bachelor's or for "go to university" as distinct from the `student` regime and from `masters`. `bachelor_age_wage` resets a wage. What, if anything, a university choice should emit is undecided.
2. Part-time master's, drop-out, and staying a student for one more year exist only on the game (`master_pt`, `drop_out`, `stay_student`). The HMM master's is a full-time `student` pin plus `has_masters`. Whether any of those three become `DECISIONS` keys is undecided.
3. Job controls that exist only on the game: `quit_job`, `reduce_hours` / `full_hours` (`work_share`), `ask_raise`, `return_home`. `part_time` is not `work_share`. Whether each maps onto an existing kind or stays on `step_month` is undecided.
4. House. The HMM has a housing bill (`rent_step`, `owns_home`) and a cash drop (`apply_fractional_drop`). The game has a balance sheet (`buy_home`, mortgage, `home_value`). Which one a house choice uses, and whether `sell_home` and `change_flat` have an HMM form at all, is undecided.
5. Balance sheet on `DecisionSnapshot`. The HMM has `Profile.opening_cash` and `CashPaths.cash`. It has no investments, debt, tax, home equity, or mortgage. The game `State` has those. What the snapshot stores under "balance sheet", and how a restore writes it back, is undecided. Do not add a balance sheet to `RegimeEngine` in order to close this.
6. Hidden distribution beyond the last month. `filter` returns one dict. `forward_filter` returns `[T, K]` and the engine discards it. Whether the snapshot keeps that array is undecided. Replay of `observed_net_flows` already reproduces the dict.
7. Who sets `until_month`. The wrapper requires the caller to pass it. The game's next decision point is `decision_reasons` (start, birthday, `program_end`). A master's pin has its own duration (default 24), which is not the birthday. A year advance is +12. Which of these wins when they disagree is undecided.
8. Commit statistic. "Mean/convergent simulation" does not name a function. `RegimePaths.net_flow_chf` is `[n_paths, T]`. `fan_quantiles` and `occupancy` are summaries. `filter` wants one scalar per month. Which reduction is appended (cross-sectional mean, median, one seeded path, or something else) is undecided. The seam stays empty until then.
9. Does `sleep(1)` move the HMM at all, or only the year advance? The year advance is specified as one 12-month `simulate`. Month sleep is still `step_month`. Whether a month of town time is also an HMM observation is undecided.
10. "Check your phone". A phone menu exists (`📱 Phone`, `Phone.tsx`, and the Multiverse tree of town turns). A button with that label does not, and nothing in the phone reads `DecisionSnapshot` or `/hmm/branch`. Whether the new entry is a new HUD label, a new `AppId`, or the existing Multiverse pointed at HMM forks is undecided. Fork compare before commit and the final fan chart wait on this, because both need a place to render.
11. Where a new HMM-only action sits in the town. `parental_leave` has no `LIFE_PLACE`. Putting it on `university` would sit beside the master's the smoke test already clicks. The building is undecided.
12. `master_ft_months` in the game assumptions versus the HMM default duration of 24. Whether the adapter passes the game number through as `duration` is undecided. Until then, `masters` uses the HMM default.
