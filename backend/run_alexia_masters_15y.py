# Run from backend/: /opt/anaconda3/bin/python run_alexia_masters_15y.py
"""15-year cash fan for Alexia's master's, from the HMM (not the stub).

Same profile and decision as the master's series in run_alexia_paths.py: force_state
student for 24 months, then that phase exits to employed and has_masters turns on.
Month 0 is the start. Month 24 is the first month of work.
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

HORIZON = 180
N_PATHS = 4000
SEED = 0
GRADUATION_MONTH = 24
COLOR = "#F58518"
ARTIFACT = ROOT / "artifacts" / "alexia-masters-15y.png"
STORE_PNG = Path(
    "/Users/victorracu/Library/Application Support/Cursor/AgentStores/"
    "cursor_agent_stores/bc-e3d1c389-9d7a-4206-a8d2-334e3178188b/files/media/alexia-masters-15y.png"
)
MASTERS = [
    Intervention("force_state", 0, GRADUATION_MONTH, state="student"),
    Intervention("set_attribute", GRADUATION_MONTH, attribute=("has_masters", True)),
]


def main() -> None:
    alexia = load_profile("P00001")
    engine = RegimeEngine(fit_emissions(), load_transition_priors())
    if engine.mode != "hmm":
        raise RuntimeError(f"expected the HMM engine, got mode={engine.mode!r}")
    paths = engine.simulate(
        alexia,
        horizon_months=HORIZON,
        n_paths=N_PATHS,
        interventions=MASTERS,
        seed=SEED,
    )
    _check(paths)
    cash = run_cash(alexia, paths).cash
    print(
        f"{alexia.name} {alexia.profile_id}: age {alexia.age}, {alexia.initial_state}, "
        f"opening {alexia.opening_cash:.0f} CHF, income {alexia.ref_income:.0f}, rent {alexia.rent:.0f}"
    )
    print(f"engine mode {engine.mode}, master's only, {N_PATHS} paths, {HORIZON} months, seed {SEED}")
    _print_table(cash)
    _plot(cash)


def _check(paths) -> None:
    student = STATE_INDEX["student"]
    employed = STATE_INDEX["employed"]
    if paths.states.shape != (N_PATHS, HORIZON):
        raise RuntimeError(f"unexpected shape {paths.states.shape}")
    if not (paths.states[:, :GRADUATION_MONTH] == student).all():
        raise RuntimeError("path was not a student for the first 24 months")
    if (paths.states[:, GRADUATION_MONTH] == employed).mean() < 0.95:
        raise RuntimeError("path did not enter work at month 24")


def _print_table(cash: np.ndarray) -> None:
    header = f"{'horizon':<12} {'median':>12} {'p10':>12} {'p90':>12}"
    print(header)
    print("-" * len(header))
    for year in (5, 10, 15):
        closing = cash[:, year * 12 - 1]
        print(
            f"{'year ' + str(year):<12} {np.median(closing):>12,.0f} "
            f"{np.quantile(closing, 0.10):>12,.0f} {np.quantile(closing, 0.90):>12,.0f}"
        )
    ever = (cash < 0).any(axis=1).mean()
    still = (cash[:, -1] < 0).mean()
    print(f"share_ever_negative {ever:.3f}")
    print(f"share_negative_at_year_15 {still:.3f}")


def _plot(cash: np.ndarray) -> None:
    months = np.arange(HORIZON)
    low, mid, high = np.quantile(cash, [0.10, 0.50, 0.90], axis=0)
    fig, ax = plt.subplots(figsize=(10.2, 5.8), dpi=140)
    ax.fill_between(months, low, high, color=COLOR, alpha=0.22, linewidth=0)
    ax.plot(months, mid, color=COLOR, linewidth=2.2, label="Master's, then work")
    ax.axhline(0, color="#444444", linewidth=0.8, zorder=0)
    ax.axvline(0, color="#333333", linewidth=1.0, linestyle="--")
    ax.axvline(GRADUATION_MONTH, color="#333333", linewidth=1.0, linestyle="--")
    ymin = min(0.0, float(low.min()))
    span = float(high.max()) - ymin
    label_y = ymin + 0.08 * span
    ax.text(2, label_y, "month 0\nstart", ha="left", va="bottom", fontsize=9, color="#333333")
    ax.text(GRADUATION_MONTH + 2, label_y, "month 24\ngraduation", ha="left", va="bottom", fontsize=9, color="#333333")
    ax.set_xlim(-6, HORIZON - 1)
    ax.set_title("Alexia: master's now, cash over 15 years")
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
