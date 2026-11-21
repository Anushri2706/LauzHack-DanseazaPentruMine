# Run from backend/: /opt/anaconda3/bin/python run_alexia_paths.py
"""Cash fans for Alexia under three decisions, from the HMM (not the stub).

Stay a student is the chain with no intervention (student hazards are zero, so she stays).
A 24-month master's is ``force_state student`` for 24 months; that phase exits to employed,
and ``has_masters`` turns on in the same month. Leaving to look for work is a one-month
``force_state job_search`` (duration None): she is out of study immediately, and the
job-search row of the chain decides how long the search lasts.
"""
from __future__ import annotations

import sys
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
from regime.types import STATE_INDEX, Intervention

HORIZON = 60
N_PATHS = 4000
SEED = 0
ARTIFACT = ROOT / "artifacts" / "alexia-decision-fan.png"
STORE_PNG = Path(
    "/Users/victorracu/Library/Application Support/Cursor/AgentStores/"
    "cursor_agent_stores/bc-e3d1c389-9d7a-4206-a8d2-334e3178188b/files/media/alexia-decision-fan.png"
)

# Name, interventions, colour. Same seed in each simulate() call, so each fan is reproducible.
DECISIONS = [
    ("Stay a student", [], "#4C78A8"),
    (
        "Master's, then work",
        [
            Intervention("force_state", 0, 24, state="student"),
            Intervention("set_attribute", 24, attribute=("has_masters", True)),
        ],
        "#F58518",
    ),
    (
        "Leave and look for work",
        [Intervention("force_state", 0, None, state="job_search")],
        "#54A24B",
    ),
]


def main() -> None:
    alexia = load_profile("P00001")
    engine = RegimeEngine(fit_emissions(), load_transition_priors())
    if engine.mode != "hmm":
        raise RuntimeError(f"expected the HMM engine, got mode={engine.mode!r}")
    print(
        f"{alexia.name} {alexia.profile_id}: age {alexia.age}, {alexia.initial_state}, "
        f"Lausanne, opening {alexia.opening_cash:.0f} CHF, income {alexia.ref_income:.0f}, "
        f"rent {alexia.rent:.0f}"
    )
    print(f"engine mode {engine.mode}, {N_PATHS} paths, {HORIZON} months, seed {SEED}")

    fans = []
    for name, interventions, colour in DECISIONS:
        paths = engine.simulate(
            alexia,
            horizon_months=HORIZON,
            n_paths=N_PATHS,
            interventions=interventions,
            seed=SEED,
        )
        cash = run_cash(alexia, paths).cash
        fans.append((name, colour, paths, cash))

    _check(fans)
    _print_table(fans)
    _plot(fans)


def _check(fans) -> None:
    stay, masters, search = (item[2] for item in fans)
    student = STATE_INDEX["student"]
    employed = STATE_INDEX["employed"]
    job_search = STATE_INDEX["job_search"]
    if (stay.states != student).any():
        raise RuntimeError("stay-a-student left the student regime")
    if not (masters.states[:, :24] == student).all():
        raise RuntimeError("master's path was not a student for the first 24 months")
    if (masters.states[:, 24] == employed).mean() < 0.95:
        raise RuntimeError("master's path did not enter work at month 24")
    if not (search.states[:, 0] == job_search).all():
        raise RuntimeError("leave-studies path did not start in job search")
    if (search.states == student).mean() > 0.05:
        raise RuntimeError("leave-studies path fell back into study")


def _print_table(fans) -> None:
    header = f"{'decision':<28} {'median_60':>12} {'p10_60':>12} {'p90_60':>12} {'share_ever_negative':>20}"
    print(header)
    print("-" * len(header))
    for name, _colour, _paths, cash in fans:
        closing = cash[:, -1]
        ever = (cash < 0).any(axis=1).mean()
        print(
            f"{name:<28} {np.median(closing):>12,.0f} {np.quantile(closing, 0.10):>12,.0f} "
            f"{np.quantile(closing, 0.90):>12,.0f} {ever:>20.3f}"
        )


def _plot(fans) -> None:
    months = np.arange(1, HORIZON + 1)
    fig, ax = plt.subplots(figsize=(10.2, 5.8), dpi=140)
    for name, colour, _paths, cash in fans:
        low, mid, high = np.quantile(cash, [0.10, 0.50, 0.90], axis=0)
        ax.fill_between(months, low, high, color=colour, alpha=0.22, linewidth=0)
        ax.plot(months, mid, color=colour, linewidth=2.2, label=name)
    ax.axhline(0, color="#444444", linewidth=0.8, zorder=0)
    ax.set_xlim(1, HORIZON)
    ax.set_title("Alexia: cash under three decisions")
    ax.set_xlabel("Month")
    ax.set_ylabel("Cash (CHF)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:,.0f}"))
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for path in (ARTIFACT, STORE_PNG):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    plt.close(fig)
    for path in (ARTIFACT, STORE_PNG):
        _assert_png(path)
        print(f"wrote {path}")


def _assert_png(path: Path) -> None:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n" or len(data) < 8_000:
        raise RuntimeError(f"{path} is not a real PNG ({len(data)} bytes)")


if __name__ == "__main__":
    main()
