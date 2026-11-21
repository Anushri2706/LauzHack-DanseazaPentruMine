# Run from backend/: /opt/anaconda3/bin/python run_alexia_field_20y.py
"""20-year fans for the field of Alexia's master's.

She starts the degree now: student for 24 months, then employed with has_masters.
The three paths differ only by masters_field. Same field keeps the 1.10 premium.
A higher-pay field multiplies that master entry by 1.15 (about +26.5% versus the
bachelor). A lower-pay field multiplies it by 0.85. Those two factors are a prior
in income_assumptions.csv, not a survey figure. Spending is the expense-priors
recipe (scaled_spending=1), so the post-school lease and the 0.25 non-housing
rate follow the field wage. The regime path does not.
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

HORIZON = 240
N_PATHS = 4000
SEED = 0
GRADUATION_MONTH = 24
OVERRIDES = {"scaled_spending": 1}
ARTIFACT = ROOT / "artifacts" / "alexia-20y-field.png"
STORE_PNG = Path(
    "/Users/victorracu/Library/Application Support/Cursor/AgentStores/"
    "cursor_agent_stores/bc-e3d1c389-9d7a-4206-a8d2-334e3178188b/files/media/alexia-20y-field.png"
)
# name, masters_field attribute, colour. Same seed, so the regime draws match.
FIELDS = (
    ("Same field", "same", "#F58518"),
    ("Higher-pay field", "higher", "#54A24B"),
    ("Lower-pay field", "lower", "#E45756"),
)


def _interventions(field: str) -> list[Intervention]:
    return [
        Intervention("force_state", 0, GRADUATION_MONTH, state="student"),
        Intervention("set_attribute", 0, attribute=("masters_field", field)),
        Intervention("set_attribute", GRADUATION_MONTH, attribute=("has_masters", True)),
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
    print(
        f"engine mode {engine.mode}, expense-priors recipe, {N_PATHS} paths, "
        f"{HORIZON} months, seed {SEED}"
    )
    table = engine.income_table.with_overrides(OVERRIDES)
    entry = float(table.params["entry_salary_chf"])
    premium = 1.0 + float(table.params["masters_premium_pct"])
    factors = {
        "same": float(table.params["masters_field_same_multiplier"]),
        "higher": float(table.params["masters_field_higher_multiplier"]),
        "lower": float(table.params["masters_field_lower_multiplier"]),
    }
    if factors != {"same": 1.0, "higher": 1.15, "lower": 0.85}:
        raise RuntimeError(f"field multipliers changed: {factors}")
    if abs(premium - 1.10) > 1e-9:
        raise RuntimeError(f"master's premium is {premium}, expected 1.10")

    fans = []
    for name, field, colour in FIELDS:
        wage = entry * premium * factors[field]
        interventions = _interventions(field)
        paths = engine.simulate(
            alexia,
            horizon_months=HORIZON,
            n_paths=N_PATHS,
            interventions=interventions,
            assumption_overrides=OVERRIDES,
            seed=SEED,
        )
        cash = run_cash(alexia, paths).cash
        versus_bachelor = premium * factors[field]
        shown = f"{versus_bachelor:.3f}".rstrip("0").rstrip(".")
        if "." not in shown:
            shown += ".00"
        elif len(shown.split(".")[1]) == 1:
            shown += "0"
        label = f"{name} (×{shown})"
        fans.append((label, colour, paths, cash, interventions, wage, field))
    _check(fans, table, alexia)
    _print_wages(fans)
    _print_table(fans)
    _plot(fans)


def _check(fans, table, alexia) -> None:
    student = STATE_INDEX["student"]
    employed = STATE_INDEX["employed"]
    states = [item[2].states for item in fans]
    if not (states[0] == states[1]).all() or not (states[0] == states[2]).all():
        raise RuntimeError("the field changed the regime path")
    for label, _colour, paths, _cash, interventions, wage, _field in fans:
        if not (paths.states[:, :GRADUATION_MONTH] == student).all():
            raise RuntimeError(f"{label} was not a student for the first 24 months")
        if not (paths.states[:, GRADUATION_MONTH] == employed).all():
            raise RuntimeError(f"{label} did not enter work at month 24")
        if not np.allclose(paths.income_chf[:, :GRADUATION_MONTH], alexia.ref_income):
            raise RuntimeError(f"{label} changed the student stipend")
        if not np.allclose(paths.income_chf[:, GRADUATION_MONTH], wage):
            raise RuntimeError(
                f"{label} starting wage {paths.income_chf[0, GRADUATION_MONTH]:.2f} != {wage:.2f}"
            )
        housing, _other = table.consumption_paths(paths.states, alexia, interventions)
        if not np.allclose(housing[:, :GRADUATION_MONTH], alexia.rent):
            raise RuntimeError(f"{label} did not keep the 900 student rent")
        signed = float(table.rent_from_earnings(wage)) + float(table.params["rent_charges_chf"])
        if not np.allclose(housing[:, GRADUATION_MONTH], signed):
            raise RuntimeError(f"{label} lease was not signed at the field wage")
        if not np.allclose(housing[:, GRADUATION_MONTH:], housing[:, GRADUATION_MONTH : GRADUATION_MONTH + 1]):
            raise RuntimeError(f"{label} recomputed the post-school lease")
    wages = [item[5] for item in fans]
    if not (wages[1] > wages[0] > wages[2]):
        raise RuntimeError(f"starting wages are not ordered higher > same > lower: {wages}")


def _print_wages(fans) -> None:
    print(f"{'path':<32} {'starting_wage':>14} {'vs_bachelor':>12}")
    for label, _colour, _paths, _cash, _iv, wage, _field in fans:
        print(f"{label:<32} {wage:>14,.2f} {wage / fans[0][5] * 1.10:>12.3f}")


def _print_table(fans) -> None:
    header = f"{'path':<32} {'year':>6} {'median':>12} {'p10':>12} {'p90':>12}"
    print(header)
    print("-" * len(header))
    for label, _colour, _paths, cash, _iv, _wage, _field in fans:
        for year in (10, 20):
            closing = cash[:, year * 12 - 1]
            print(
                f"{label:<32} {year:>6} {np.median(closing):>12,.0f} "
                f"{np.quantile(closing, 0.10):>12,.0f} {np.quantile(closing, 0.90):>12,.0f}"
            )


def _plot(fans) -> None:
    months = np.arange(HORIZON)
    fig, ax = plt.subplots(figsize=(10.2, 5.8), dpi=140)
    for label, colour, _paths, cash, _iv, _wage, _field in fans:
        low, mid, high = np.quantile(cash, [0.10, 0.50, 0.90], axis=0)
        ax.fill_between(months, low, high, color=colour, alpha=0.18, linewidth=0)
        ax.plot(months, mid, color=colour, linewidth=2.2, label=label)
    ax.axhline(0, color="#444444", linewidth=0.8, zorder=0)
    ax.axvline(0, color="#333333", linewidth=1.0, linestyle="--")
    ax.axvline(GRADUATION_MONTH, color="#333333", linewidth=1.0, linestyle="--")
    lows = [float(np.quantile(cash, 0.10, axis=0).min()) for _l, _c, _p, cash, _i, _w, _f in fans]
    highs = [float(np.quantile(cash, 0.90, axis=0).max()) for _l, _c, _p, cash, _i, _w, _f in fans]
    ymin = min(0.0, min(lows))
    span = max(highs) - ymin
    label_y = ymin + 0.08 * span
    ax.text(2, label_y, "month 0\nstart", ha="left", va="bottom", fontsize=9, color="#333333")
    ax.text(
        GRADUATION_MONTH + 2, label_y, "month 24\ngraduation",
        ha="left", va="bottom", fontsize=9, color="#333333",
    )
    ax.set_xlim(-6, HORIZON - 1)
    ax.set_title("Alexia: 20 years, field of the master's")
    ax.set_xlabel("Month")
    ax.set_ylabel("Cash (CHF)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:,.0f}"))
    ax.legend(frameon=False, loc="upper left", title="Master's field")
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
