# Run from backend/: /opt/anaconda3/bin/python run_alexia_decisions_15y_consumption.py
"""15-year fans under the expense-priors recipe. Same three decisions as run_alexia_paths.py.

While studying, spend is the EPFL living budget (rent 900 inside it). After school the
lease is clip(1007, 1376, 1084 * (gross/6413)**0.27) plus 200 charges, signed once.
Non-housing is 540 + 0.25 * gross. The yearly cash hit is only (L - 3300).
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

from twin import RegimeEngine, fit_emissions, load_profile, load_transition_priors
from twin.cash_engine import run as run_cash
from twin.types import STATE_INDEX, Intervention

HORIZON = 180
N_PATHS = 4000
SEED = 0
GRADUATION_MONTH = 24
OVERRIDES = {"scaled_spending": 1}
ARTIFACT = ROOT / "artifacts" / "alexia-decisions-15y-consumption.png"
STORE_PNG = Path(
    "/Users/victorracu/Library/Application Support/Cursor/AgentStores/"
    "cursor_agent_stores/bc-e3d1c389-9d7a-4206-a8d2-334e3178188b/files/media/alexia-decisions-15y-consumption.png"
)
DECISIONS = [
    ("Stay a student", [], "#4C78A8"),
    (
        "Master's, then work",
        [
            Intervention("force_state", 0, GRADUATION_MONTH, state="student"),
            Intervention("set_attribute", GRADUATION_MONTH, attribute=("has_masters", True)),
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
    print(f"engine mode {engine.mode}, expense-priors recipe, {N_PATHS} paths, {HORIZON} months, seed {SEED}")
    table = engine.income_table.with_overrides(OVERRIDES)

    fans = []
    for name, interventions, colour in DECISIONS:
        paths = engine.simulate(
            alexia,
            horizon_months=HORIZON,
            n_paths=N_PATHS,
            interventions=interventions,
            assumption_overrides=OVERRIDES,
            seed=SEED,
        )
        cash = run_cash(alexia, paths).cash
        fans.append((name, colour, paths, cash, interventions))
    _check(fans, table, alexia)
    _print_table(fans)
    _breakdown(alexia, table, fans[1])
    _plot(fans)


def _check(fans, table, alexia) -> None:
    stay, masters, search = (item[2] for item in fans)
    student = STATE_INDEX["student"]
    employed = STATE_INDEX["employed"]
    job_search = STATE_INDEX["job_search"]
    if (stay.states != student).any():
        raise RuntimeError("stay-a-student left the student regime")
    if not (masters.states[:, :GRADUATION_MONTH] == student).all():
        raise RuntimeError("master's path was not a student for the first 24 months")
    if (masters.states[:, GRADUATION_MONTH] == employed).mean() < 0.95:
        raise RuntimeError("master's path did not enter work at month 24")
    if not (search.states[:, 0] == job_search).all():
        raise RuntimeError("leave-studies path did not start in job search")
    housing, _other = table.consumption_paths(masters.states, alexia, fans[1][4])
    if not np.allclose(housing[:, :GRADUATION_MONTH], alexia.rent):
        raise RuntimeError("student rent was not held at 900")
    signed = housing[0, GRADUATION_MONTH]
    if not np.allclose(housing[:, GRADUATION_MONTH:], signed):
        raise RuntimeError("the post-school lease was recomputed")


def _print_table(fans) -> None:
    header = (
        f"{'decision':<28} {'year':>6} {'median':>12} {'p10':>12} {'p90':>12} "
        f"{'ever_neg':>10} {'neg_y15':>10}"
    )
    print(header)
    print("-" * len(header))
    for name, _colour, _paths, cash, _iv in fans:
        ever = (cash < 0).any(axis=1).mean()
        still = (cash[:, -1] < 0).mean()
        for year in (5, 10, 15):
            closing = cash[:, year * 12 - 1]
            print(
                f"{name:<28} {year:>6} {np.median(closing):>12,.0f} "
                f"{np.quantile(closing, 0.10):>12,.0f} {np.quantile(closing, 0.90):>12,.0f} "
                f"{ever:>10.3f} {still:>10.3f}"
            )


def _breakdown(alexia, table, fan) -> None:
    _name, _colour, paths, _cash, interventions = fan
    housing, other = table.consumption_paths(paths.states, alexia, interventions)
    not_dec = (np.arange(paths.horizon) % 12) != 11
    mask = (paths.states == STATE_INDEX["employed"]) & not_dec
    inc = float(paths.income_chf[mask].mean())
    ren = float(housing[mask].mean())
    non = float(other[mask].mean())
    lump = float(paths.lump_chf[mask].mean())
    saved = inc - ren - non - lump
    print(
        "typical working month (master's, employed, not December, means): "
        f"income {inc:,.0f}, rent {ren:,.0f}, non-housing {non:,.0f}, "
        f"lump {lump:,.0f}, saved {saved:,.0f}"
    )


def _plot(fans) -> None:
    months = np.arange(HORIZON)
    fig, ax = plt.subplots(figsize=(10.2, 5.8), dpi=140)
    for name, colour, _paths, cash, _iv in fans:
        low, mid, high = np.quantile(cash, [0.10, 0.50, 0.90], axis=0)
        ax.fill_between(months, low, high, color=colour, alpha=0.20, linewidth=0)
        ax.plot(months, mid, color=colour, linewidth=2.2, label=name)
    ax.axhline(0, color="#444444", linewidth=0.8, zorder=0)
    ax.axvline(0, color="#333333", linewidth=1.0, linestyle="--")
    ax.axvline(GRADUATION_MONTH, color="#333333", linewidth=1.0, linestyle="--")
    lows = [float(np.quantile(cash, 0.10, axis=0).min()) for _n, _c, _p, cash, _i in fans]
    highs = [float(np.quantile(cash, 0.90, axis=0).max()) for _n, _c, _p, cash, _i in fans]
    ymin = min(0.0, min(lows))
    span = max(highs) - ymin
    label_y = ymin + 0.08 * span
    ax.text(2, label_y, "month 0\nstart", ha="left", va="bottom", fontsize=9, color="#333333")
    ax.text(
        GRADUATION_MONTH + 2, label_y, "month 24\ngraduation",
        ha="left", va="bottom", fontsize=9, color="#333333",
    )
    ax.set_xlim(-6, HORIZON - 1)
    ax.set_title("Alexia: 15 years, consumption recipe")
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
        data = path.read_bytes()
        if data[:8] != b"\x89PNG\r\n\x1a\n" or len(data) < 8_000:
            raise RuntimeError(f"{path} is not a real PNG ({len(data)} bytes)")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
