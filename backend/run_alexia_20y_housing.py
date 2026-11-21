# Run from backend/: /opt/anaconda3/bin/python run_alexia_20y_housing.py
"""20-year fans. Master's started now, lease signed at graduation, then a housing choice.

All three paths are the master's path (student for 24 months, has_masters after that)
under the expense-priors recipe (scaled_spending=1). The lease signed at graduation
stays put. At month 72 the paths split:

1. Keep that lease.
2. Net rent times rent_step_factor (prior 1.4). The 2-room cap is not reapplied on
   this path. Charges stay.
3. Buy. Cash drops by purchase_deposit_fraction (prior 0.15) of the end-of-month
   balance at month 72, before the drop. The monthly housing bill stays equal to
   the rent she was paying.
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
from twin.cash_engine import apply_fractional_drop, run as run_cash
from twin.types import STATE_INDEX, Intervention

HORIZON = 240
N_PATHS = 4000
SEED = 0
GRADUATION_MONTH = 24
CHOICE_MONTH = 72
OVERRIDES = {"scaled_spending": 1}
ARTIFACT = ROOT / "artifacts" / "alexia-20y-housing.png"
STORE_PNG = Path(
    "/Users/victorracu/Library/Application Support/Cursor/AgentStores/"
    "cursor_agent_stores/bc-e3d1c389-9d7a-4206-a8d2-334e3178188b/files/media/alexia-20y-housing.png"
)
MASTERS = [
    Intervention("force_state", 0, GRADUATION_MONTH, state="student"),
    Intervention("set_attribute", GRADUATION_MONTH, attribute=("has_masters", True)),
]
DECISIONS = [
    ("Keep the lease", list(MASTERS), "#4C78A8"),
    (
        "Rent x1.4 at month 72",
        MASTERS + [Intervention("set_attribute", CHOICE_MONTH, attribute=("rent_step", True))],
        "#F58518",
    ),
    (
        "Buy at month 72",
        MASTERS + [Intervention("set_attribute", CHOICE_MONTH, attribute=("owns_home", True))],
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
    table = engine.income_table.with_overrides(OVERRIDES)
    step_factor = float(table.params["rent_step_factor"])
    deposit_fraction = float(table.params["purchase_deposit_fraction"])
    if step_factor != 1.4 or deposit_fraction != 0.15:
        raise RuntimeError(f"priors are {step_factor} and {deposit_fraction}, expected 1.4 and 0.15")
    if float(table.params["scaled_spending"]) != 1.0:
        raise RuntimeError("scaled_spending baseline is not 1")
    print(
        f"engine mode {engine.mode}, expense-priors recipe, {N_PATHS} paths, "
        f"{HORIZON} months, seed {SEED}"
    )
    print(
        f"master's from month 0, lease at month {GRADUATION_MONTH}, choice at month {CHOICE_MONTH}: "
        f"keep the lease; net rent x{step_factor:g} with the 2-room cap lifted; "
        f"buy with a {deposit_fraction:.0%} deposit and the same monthly housing bill"
    )

    fans = []
    deposits = None
    for name, interventions, colour in DECISIONS:
        paths = engine.simulate(
            alexia,
            horizon_months=HORIZON,
            n_paths=N_PATHS,
            interventions=interventions,
            assumption_overrides=OVERRIDES,
            seed=SEED,
        )
        cash_paths = run_cash(alexia, paths)
        if name.startswith("Buy"):
            before = cash_paths.cash.copy()
            cash_paths, deposit = apply_fractional_drop(cash_paths, CHOICE_MONTH, deposit_fraction)
            deposits = (before, deposit, cash_paths.cash)
        fans.append((name, colour, paths, cash_paths.cash, interventions))

    _check(fans, table, alexia, step_factor, deposit_fraction, deposits)
    _print_table(fans)
    _print_deposit(deposits, deposit_fraction)
    _plot(fans)


def _check(fans, table, alexia, step_factor: float, deposit_fraction: float, deposits) -> None:
    keep, step, buy = fans
    student = STATE_INDEX["student"]
    employed = STATE_INDEX["employed"]
    if not (keep[2].states[:, :GRADUATION_MONTH] == student).all():
        raise RuntimeError("the master's path was not a student for the first 24 months")
    if (keep[2].states[:, GRADUATION_MONTH] == employed).mean() < 0.95:
        raise RuntimeError("the master's path did not enter work at month 24")
    if not np.array_equal(keep[2].states, step[2].states) or not np.array_equal(keep[2].states, buy[2].states):
        raise RuntimeError("the housing choice changed the regime path")
    if not np.array_equal(keep[2].income_chf, step[2].income_chf):
        raise RuntimeError("the rent step changed income")
    if not np.array_equal(keep[2].income_chf, buy[2].income_chf):
        raise RuntimeError("the purchase changed income")

    keep_h, _ = table.consumption_paths(keep[2].states, alexia, keep[4])
    step_h, step_other = table.consumption_paths(step[2].states, alexia, step[4])
    buy_h, buy_other = table.consumption_paths(buy[2].states, alexia, buy[4])
    _, keep_other = table.consumption_paths(keep[2].states, alexia, keep[4])
    if not np.allclose(keep_h[:, :GRADUATION_MONTH], alexia.rent):
        raise RuntimeError("student rent was not held at 900")
    signed = keep_h[:, GRADUATION_MONTH]
    if not np.allclose(keep_h[:, GRADUATION_MONTH:], signed[:, None]):
        raise RuntimeError("the kept lease was recomputed after graduation")
    if not np.allclose(step_h[:, :CHOICE_MONTH], keep_h[:, :CHOICE_MONTH]):
        raise RuntimeError("the stepped path changed rent before month 72")
    if not np.allclose(buy_h, keep_h):
        raise RuntimeError("buying changed the monthly housing bill")
    if not np.allclose(step_other, keep_other) or not np.allclose(buy_other, keep_other):
        raise RuntimeError("housing choice changed non-housing spend")

    charges = float(table.params["rent_charges_chf"])
    cap = float(table.params["rent_cap_chf"])
    net = signed - charges
    stepped_net = net * step_factor
    if not np.all(stepped_net > cap):
        raise RuntimeError("the 1.4 rent did not clear the 2-room cap, so lifting it was not tested")
    if not np.allclose(step_h[:, CHOICE_MONTH:], (stepped_net + charges)[:, None]):
        raise RuntimeError("month 72 rent is not the uncapped 1.4 net rent plus the same charges")
    clipped = np.clip(stepped_net, float(table.params["rent_floor_chf"]), cap) + charges
    if np.allclose(step_h[:, CHOICE_MONTH], clipped):
        raise RuntimeError("the 2-room cap was reapplied to the stepped rent")

    if not np.allclose(keep[2].net_flow_chf, buy[2].net_flow_chf):
        raise RuntimeError("buying changed net flow; only the cash deposit should differ")
    before, deposit, after = deposits
    if not np.allclose(before, keep[3]):
        raise RuntimeError("cash before the deposit does not match the kept lease")
    if not np.allclose(deposit, deposit_fraction * np.maximum(before[:, CHOICE_MONTH], 0.0)):
        raise RuntimeError("the deposit is not 15% of cash at month 72")
    if not np.allclose(after[:, :CHOICE_MONTH], before[:, :CHOICE_MONTH]):
        raise RuntimeError("the deposit changed cash before month 72")
    if not np.allclose(after[:, CHOICE_MONTH:], before[:, CHOICE_MONTH:] - deposit[:, None]):
        raise RuntimeError("the deposit was not held for the rest of the path")
    nonpositive = int(np.sum(before[:, CHOICE_MONTH] <= 0))
    print(
        f"lease net rent median {np.median(net):,.0f}; "
        f"stepped net rent median {np.median(stepped_net):,.0f} "
        f"(2-room cap {cap:,.0f}, charges {charges:,.0f}); "
        f"paths with non-positive cash at month 72: {nonpositive}"
    )


def _print_table(fans) -> None:
    header = f"{'decision':<24} {'year':>6} {'median':>14} {'p10':>14} {'p90':>14}"
    print(header)
    print("-" * len(header))
    for name, _colour, _paths, cash, _iv in fans:
        for year in (10, 20):
            closing = cash[:, year * 12 - 1]
            print(
                f"{name:<24} {year:>6} {np.median(closing):>14,.0f} "
                f"{np.quantile(closing, 0.10):>14,.0f} {np.quantile(closing, 0.90):>14,.0f}"
            )


def _print_deposit(deposits, fraction: float) -> None:
    before, deposit, after = deposits
    closing = after[:, 20 * 12 - 1]
    median_cash = float(np.median(closing))
    idx = int(np.argmin(np.abs(closing - median_cash)))
    pre = float(before[idx, CHOICE_MONTH])
    print(
        f"deposit on the median path: {deposit[idx]:,.2f} CHF "
        f"(path {idx}, year-20 cash {closing[idx]:,.0f}, "
        f"month-72 cash before the drop {pre:,.2f}, {fraction:.0%} of that)"
    )
    print(
        f"15% of the median month-72 cash before the drop: "
        f"{fraction * float(np.median(before[:, CHOICE_MONTH])):,.2f} CHF"
    )


def _plot(fans) -> None:
    months = np.arange(HORIZON)
    fig, ax = plt.subplots(figsize=(10.2, 5.8), dpi=140)
    for name, colour, _paths, cash, _iv in fans:
        low, mid, high = np.quantile(cash, [0.10, 0.50, 0.90], axis=0)
        ax.fill_between(months, low, high, color=colour, alpha=0.20, linewidth=0)
        ax.plot(months, mid, color=colour, linewidth=2.2, label=name)
    ax.axhline(0, color="#444444", linewidth=0.8, zorder=0)
    for month in (0, GRADUATION_MONTH, CHOICE_MONTH):
        ax.axvline(month, color="#333333", linewidth=1.0, linestyle="--")
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
    ax.text(
        CHOICE_MONTH + 2, label_y, "month 72\nhousing choice",
        ha="left", va="bottom", fontsize=9, color="#333333",
    )
    ax.set_xlim(-6, HORIZON - 1)
    ax.set_title("Alexia: 20 years, housing after the master's")
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
