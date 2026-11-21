# Run from backend/: /opt/anaconda3/bin/python run_alexia_goal_hit.py
"""Alexia's emergency-savings goal on the master's-now path.

Student for 24 months, then employed with has_masters. Expense-priors recipe
(scaled_spending=1). 60 months covers December 2029. The goal row is the CSV
row, not a retyped amount.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from regime import RegimeEngine, fit_emissions, load_profile, load_transition_priors
from regime.cash_engine import run as run_cash
from regime.data import load_goals
from regime.metrics import goal_hit_rate
from regime.types import STATE_INDEX, Intervention

HORIZON = 60
N_PATHS = 4000
SEED = 0
START = date(2025, 1, 1)
OVERRIDES = {"scaled_spending": 1}
STUDY = 24
ARTIFACT = ROOT / "artifacts" / "alexia-goal-hit.png"
STORE_PNG = Path(
    "/Users/victorracu/Library/Application Support/Cursor/AgentStores/"
    "cursor_agent_stores/bc-e3d1c389-9d7a-4206-a8d2-334e3178188b/files/media/alexia-goal-hit.png"
)


def _masters_now() -> list[Intervention]:
    return [
        Intervention("force_state", 0, STUDY, state="student"),
        Intervention("set_attribute", STUDY, attribute=("has_masters", True)),
    ]


def main() -> None:
    alexia = load_profile("P00001")
    goals = load_goals()
    row = goals.loc[(goals["profile_id"] == alexia.profile_id) & (goals["goal_name"] == "Emergency savings")].iloc[0]
    engine = RegimeEngine(fit_emissions(), load_transition_priors())
    if engine.mode != "hmm":
        raise RuntimeError(f"expected the HMM engine, got mode={engine.mode!r}")
    interventions = _masters_now()
    paths = engine.simulate(
        alexia,
        horizon_months=HORIZON,
        n_paths=N_PATHS,
        interventions=interventions,
        assumption_overrides=OVERRIDES,
        seed=SEED,
    )
    cash = run_cash(alexia, paths).cash
    _check(alexia, engine, paths, interventions)
    scored = goal_hit_rate(cash, START, row)
    if scored["evaluated_at"] != "goal_month" or scored["goal_month"] != 59:
        raise RuntimeError(f"60 months should cover 2029-12, got {scored['evaluated_at']} month {scored['goal_month']}")
    _print(alexia, scored)
    _plot(cash, scored)


def _check(alexia, engine, paths, interventions) -> None:
    student = STATE_INDEX["student"]
    employed = STATE_INDEX["employed"]
    if paths.states.shape != (N_PATHS, HORIZON):
        raise RuntimeError(f"expected {(N_PATHS, HORIZON)} paths, got {paths.states.shape}")
    if not (paths.states[:, :STUDY] == student).all():
        raise RuntimeError("the master's was not pinned student")
    if not (paths.states[:, STUDY] == employed).all():
        raise RuntimeError("graduation month was not employed")
    table = engine.income_table.with_overrides(OVERRIDES)
    offer = table.params["entry_salary_chf"] * (1.0 + table.params["masters_premium_pct"])
    if not np.allclose(paths.income_chf[:, STUDY], offer):
        raise RuntimeError("has_masters did not set the graduate wage")
    if float(table.params["scaled_spending"]) != 1.0:
        raise RuntimeError("scaled_spending was not on")
    if alexia.opening_cash != 8000:
        raise RuntimeError(f"expected Alexia's opening cash of 8000, got {alexia.opening_cash}")


def _print(alexia, scored: dict) -> None:
    cross = scored["median_cross_month"]
    cross_text = "never" if cross is None else f"month {cross} ({scored['median_cross_label']})"
    print(
        f"{alexia.name} {alexia.profile_id}: {scored['goal_name']}, "
        f"{scored['target_amount_chf']:,.0f} CHF by {scored['target_date']} ({scored['priority']})"
    )
    print(
        f"master's now, scaled_spending=1, {scored['n_paths']} paths, "
        f"{scored['horizon_months']} months from {scored['start']}, seed {SEED}"
    )
    print(
        f"scored at month {scored['evaluated_month']} ({scored['evaluated_label']}), "
        f"which is the {scored['evaluated_at'].replace('_', ' ')}"
    )
    print(f"hit_rate {scored['hit_rate']:.4f}")
    print(f"median_cross_month {cross_text}")
    print(
        f"cash at {scored['evaluated_label']}: "
        f"median {scored['median_cash_at_goal']:,.0f}  "
        f"p10 {scored['p10_cash_at_goal']:,.0f}  "
        f"p90 {scored['p90_cash_at_goal']:,.0f}"
    )


def _plot(cash: np.ndarray, scored: dict) -> None:
    months = np.arange(cash.shape[1])
    median = np.median(cash, axis=0)
    amount = scored["target_amount_chf"]
    goal_month = scored["evaluated_month"]
    fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=140)
    ax.plot(months, median, color="#4C78A8", linewidth=2.2, label="Median cash")
    ax.axhline(amount, color="#E45756", linewidth=1.4, label=f"Goal {amount:,.0f} CHF")
    ax.axvline(goal_month, color="#F58518", linewidth=1.4, linestyle="--", label=scored["evaluated_label"])
    ax.set_xlim(-1, cash.shape[1] - 1)
    cross = scored["median_cross_month"]
    cross_text = "median never stays above the goal" if cross is None else f"median stays above from month {cross}"
    ax.set_title(f"Alexia: emergency savings hit rate {scored['hit_rate']:.1%}\n{cross_text}")
    ax.set_xlabel("Month from 2025-01")
    ax.set_ylabel("Cash (CHF)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:,.0f}"))
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    hline = [line for line in ax.lines if len(line.get_ydata()) == 2 and np.allclose(line.get_ydata(), amount)]
    vline = [line for line in ax.lines if len(line.get_xdata()) == 2 and np.allclose(line.get_xdata(), goal_month)]
    plotted = [line for line in ax.lines if len(line.get_ydata()) == cash.shape[1]]
    if len(hline) != 1 or len(vline) != 1 or len(plotted) != 1:
        raise RuntimeError("chart is missing the median, the goal line, or the goal date")
    if not np.allclose(plotted[0].get_ydata(), median):
        raise RuntimeError("plotted median does not match the cash paths")
    for path in (ARTIFACT, STORE_PNG):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    plt.close(fig)
    for path in (ARTIFACT, STORE_PNG):
        data = path.read_bytes()
        if data[:8] != b"\x89PNG\r\n\x1a\n" or len(data) < 8_000:
            raise RuntimeError(f"{path} is not a real PNG ({len(data)} bytes)")
        image = plt.imread(path)
        if float(np.std(image)) < 0.01:
            raise RuntimeError(f"{path} looks blank")
        print(f"wrote {path} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
