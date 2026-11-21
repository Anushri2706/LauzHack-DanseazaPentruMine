# Run from backend/: /opt/anaconda3/bin/python run_alexia_decisions_20y_where.py
"""20-year fans: where Alexia lives after a master's she starts now.

All three paths study for 24 months and then work, with the master's premium on.
The decision is the city after that:

1. Stay in Lausanne.
2. At month 48, move inside Switzerland. Housing is multiplied by the Zurich
   prior (1.20) from then on. The wage is unchanged. A one-off of 4,000 is
   charged that month.
3. At month 60, move to Germany. The wage is multiplied by 0.51 and housing by
   0.63. A one-off of 5,000 is charged that month, and she is in job search for
   four months. Unemployment insurance is 70% of the last Swiss wage, not of
   the German wage.

``scaled_spending`` is 1, so the expense-priors recipe is on. The Zurich factor
and both one-off amounts are priors in assumptions/income_assumptions.csv.
Non-housing is not multiplied by either housing factor.
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
from regime.cash_engine import OneOff
from regime.cash_engine import run as run_cash
from regime.types import STATE_INDEX, Intervention

HORIZON = 240
N_PATHS = 4000
SEED = 0
GRADUATION_MONTH = 24
ZURICH_MONTH = 48
GERMANY_MONTH = 60
SEARCH_MONTHS = 4
OVERRIDES = {"scaled_spending": 1}
ZURICH_FACTOR = 1.20
GERMANY_WAGE = 0.51
GERMANY_HOUSING = 0.63
ZURICH_COST = 4000.0
GERMANY_COST = 5000.0
ARTIFACT = ROOT / "artifacts" / "alexia-20y-where.png"
STORE_PNG = Path(
    "/Users/victorracu/Library/Application Support/Cursor/AgentStores/"
    "cursor_agent_stores/bc-e3d1c389-9d7a-4206-a8d2-334e3178188b/files/media/alexia-20y-where.png"
)
MASTERS = [
    Intervention("force_state", 0, GRADUATION_MONTH, state="student"),
    Intervention("set_attribute", GRADUATION_MONTH, attribute=("has_masters", True)),
]
DECISIONS = [
    ("Lausanne", list(MASTERS), [], "#4C78A8"),
    (
        "Zurich",
        MASTERS + [Intervention("set_attribute", ZURICH_MONTH, attribute=("zurich", True))],
        [OneOff(ZURICH_MONTH, ZURICH_COST, "move to Zurich")],
        "#F58518",
    ),
    (
        "Germany",
        MASTERS + [
            Intervention("set_attribute", GERMANY_MONTH, attribute=("abroad", True)),
            Intervention("force_state", GERMANY_MONTH, SEARCH_MONTHS, state="job_search"),
        ],
        [OneOff(GERMANY_MONTH, GERMANY_COST, "move to Germany")],
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
    print(
        f"engine mode {engine.mode}, scaled_spending=1, {N_PATHS} paths, "
        f"{HORIZON} months, seed {SEED}"
    )
    table = engine.income_table.with_overrides(OVERRIDES)
    _check_priors(table)

    fans = []
    for name, interventions, one_offs, colour in DECISIONS:
        paths = engine.simulate(
            alexia,
            horizon_months=HORIZON,
            n_paths=N_PATHS,
            interventions=interventions,
            assumption_overrides=OVERRIDES,
            seed=SEED,
        )
        cash = run_cash(alexia, paths, one_offs).cash
        fans.append((name, colour, paths, cash, interventions, one_offs))
    _check(fans, table, alexia)
    _print_table(fans)
    _plot(fans)


def _check_priors(table) -> None:
    p = table.params
    if float(p["scaled_spending"]) < 0.5:
        raise RuntimeError("scaled_spending did not turn the expense-priors recipe on")
    expected = {
        "zurich_housing_multiplier": ZURICH_FACTOR,
        "abroad_income_multiplier": GERMANY_WAGE,
        "abroad_cost_multiplier": GERMANY_HOUSING,
        "zurich_move_cost_chf": ZURICH_COST,
        "germany_move_cost_chf": GERMANY_COST,
    }
    for name, value in expected.items():
        if float(p[name]) != value:
            raise RuntimeError(f"{name} is {p[name]}, expected {value}")
    if float(p["zurich_move_cost_chf"]) != DECISIONS[1][2][0].amount_chf:
        raise RuntimeError("the Zurich one-off is not the prior")
    if float(p["germany_move_cost_chf"]) != DECISIONS[2][2][0].amount_chf:
        raise RuntimeError("the Germany one-off is not the prior")


def _check(fans, table, alexia) -> None:
    lausanne, zurich, germany = (item[2] for item in fans)
    student = STATE_INDEX["student"]
    employed = STATE_INDEX["employed"]
    job_search = STATE_INDEX["job_search"]
    if not (lausanne.states[:, :GRADUATION_MONTH] == student).all():
        raise RuntimeError("Lausanne was not a student for the first 24 months")
    if (lausanne.states[:, GRADUATION_MONTH] == employed).mean() < 0.95:
        raise RuntimeError("Lausanne did not enter work at month 24")
    for name, paths in (("Zurich", zurich), ("Germany", germany)):
        if not (paths.states[:, :GRADUATION_MONTH] == student).all():
            raise RuntimeError(f"{name} was not a student for the first 24 months")
        if (paths.states[:, GRADUATION_MONTH] == employed).mean() < 0.95:
            raise RuntimeError(f"{name} did not enter work at month 24")

    if not np.array_equal(zurich.states, lausanne.states):
        raise RuntimeError("the Zurich move changed the regime path")
    if not np.allclose(zurich.income_chf, lausanne.income_chf):
        raise RuntimeError("the Zurich move changed the wage")
    if not np.array_equal(germany.states[:, :GERMANY_MONTH], lausanne.states[:, :GERMANY_MONTH]):
        raise RuntimeError("Germany diverged before the move")
    if not np.allclose(germany.income_chf[:, :GERMANY_MONTH], lausanne.income_chf[:, :GERMANY_MONTH]):
        raise RuntimeError("the wage changed before the Germany move")
    spell = slice(GERMANY_MONTH, GERMANY_MONTH + SEARCH_MONTHS)
    if not (germany.states[:, spell] == job_search).all():
        raise RuntimeError("Germany was not in job search for the four move months")
    if not (germany.states[:, GERMANY_MONTH + SEARCH_MONTHS] == employed).all():
        raise RuntimeError("the job-search phase did not exit to employed")

    lz_h, lz_o = table.consumption_paths(lausanne.states, alexia, fans[0][4])
    zh_h, zh_o = table.consumption_paths(zurich.states, alexia, fans[1][4])
    de_h, de_o = table.consumption_paths(germany.states, alexia, fans[2][4])
    if not np.allclose(lz_h[:, :GRADUATION_MONTH], alexia.rent):
        raise RuntimeError("student rent was not held at 900")
    if not np.allclose(zh_h[:, :ZURICH_MONTH], lz_h[:, :ZURICH_MONTH]):
        raise RuntimeError("Zurich housing changed before month 48")
    if not np.allclose(zh_h[:, ZURICH_MONTH:], lz_h[:, ZURICH_MONTH:] * ZURICH_FACTOR):
        raise RuntimeError("Zurich housing was not 1.20 times Lausanne from month 48")
    if not np.allclose(zh_o, lz_o):
        raise RuntimeError("Zurich changed non-housing")
    if not np.allclose(de_h[:, :GERMANY_MONTH], lz_h[:, :GERMANY_MONTH]):
        raise RuntimeError("Germany housing changed before month 60")
    if not np.allclose(de_h[:, GERMANY_MONTH:], lz_h[:, GERMANY_MONTH - 1][:, None] * GERMANY_HOUSING):
        raise RuntimeError("Germany housing was not 0.63 times the Lausanne lease from month 60")

    at_work = lausanne.states[:, GERMANY_MONTH - 1] == employed
    if int(at_work.sum()) < 1000:
        raise RuntimeError("too few paths were employed in the month before the Germany move")
    last_swiss = lausanne.income_chf[at_work, GERMANY_MONTH - 1]
    rate = np.where(
        last_swiss <= table.params["ui_low_earnings_threshold_chf"],
        table.params["ui_replacement_with_dependents"],
        table.params["ui_replacement_rate"],
    )
    benefit = rate * np.minimum(last_swiss, table.params["ui_insured_earnings_cap_chf"])
    ui = germany.income_chf[at_work, GERMANY_MONTH:GERMANY_MONTH + SEARCH_MONTHS]
    if not np.allclose(ui, benefit[:, None]):
        raise RuntimeError("unemployment insurance was not the last Swiss wage")
    if np.allclose(ui, benefit[:, None] * GERMANY_WAGE):
        raise RuntimeError("unemployment insurance was scaled by the Germany wage factor")
    g = table._growth_rate(alexia.age + GERMANY_MONTH / 12.0)
    german_wage = germany.income_chf[at_work, GERMANY_MONTH + SEARCH_MONTHS]
    if not np.allclose(german_wage, last_swiss * (1.0 + g) * GERMANY_WAGE):
        raise RuntimeError("the wage after the search was not the raised Swiss wage times 0.51")
    print(
        f"month {GERMANY_MONTH - 1} employed on {int(at_work.sum())} paths; "
        f"UI median {np.median(benefit):,.0f} on last Swiss wage median {np.median(last_swiss):,.0f}; "
        f"German wage at month {GERMANY_MONTH + SEARCH_MONTHS} median {np.median(german_wage):,.0f}"
    )

    # The one-off is outside net flow. Same paths, with and without it, differ by that amount.
    for label, paths, one_offs, month, amount in (
        ("Zurich", zurich, fans[1][5], ZURICH_MONTH, ZURICH_COST),
        ("Germany", germany, fans[2][5], GERMANY_MONTH, GERMANY_COST),
    ):
        with_cost = run_cash(alexia, paths, one_offs).cash
        without = run_cash(alexia, paths, []).cash
        if not np.allclose(with_cost[:, :month], without[:, :month]):
            raise RuntimeError(f"{label} one-off changed cash before the move month")
        if not np.allclose(with_cost[:, month:], without[:, month:] - amount):
            raise RuntimeError(f"{label} one-off was not {amount:.0f} in the move month")

    zurich_cash, lausanne_cash = fans[1][3], fans[0][3]
    if not np.allclose(zurich_cash[:, :ZURICH_MONTH], lausanne_cash[:, :ZURICH_MONTH]):
        raise RuntimeError("Zurich cash diverged before month 48")
    extra = (ZURICH_FACTOR - 1.0) * lz_h[:, ZURICH_MONTH]
    gap = lausanne_cash[:, ZURICH_MONTH] - zurich_cash[:, ZURICH_MONTH]
    if not np.allclose(gap, extra + ZURICH_COST):
        raise RuntimeError("the Zurich cash gap is not the extra housing plus 4,000")


def _print_table(fans) -> None:
    header = (
        f"{'where':<16} {'year':>6} {'median':>12} {'p10':>12} {'p90':>12} "
        f"{'ever_neg':>10} {'neg_end':>10}"
    )
    print(header)
    print("-" * len(header))
    for name, _colour, _paths, cash, _iv, _off in fans:
        ever = (cash < 0).any(axis=1).mean()
        still = (cash[:, -1] < 0).mean()
        for year in (10, 20):
            closing = cash[:, year * 12 - 1]
            print(
                f"{name:<16} {year:>6} {np.median(closing):>12,.0f} "
                f"{np.quantile(closing, 0.10):>12,.0f} {np.quantile(closing, 0.90):>12,.0f} "
                f"{ever:>10.3f} {still:>10.3f}"
            )


def _plot(fans) -> None:
    months = np.arange(HORIZON)
    fig, ax = plt.subplots(figsize=(10.2, 5.8), dpi=140)
    for name, colour, _paths, cash, _iv, _off in fans:
        low, mid, high = np.quantile(cash, [0.10, 0.50, 0.90], axis=0)
        ax.fill_between(months, low, high, color=colour, alpha=0.20, linewidth=0)
        ax.plot(months, mid, color=colour, linewidth=2.2, label=name)
    ax.axhline(0, color="#444444", linewidth=0.8, zorder=0)
    for month, colour in (
        (0, "#333333"),
        (GRADUATION_MONTH, "#333333"),
        (ZURICH_MONTH, "#F58518"),
        (GERMANY_MONTH, "#54A24B"),
    ):
        ax.axvline(month, color=colour, linewidth=1.0, linestyle="--")
    lows = [float(np.quantile(cash, 0.10, axis=0).min()) for _n, _c, _p, cash, _i, _o in fans]
    highs = [float(np.quantile(cash, 0.90, axis=0).max()) for _n, _c, _p, cash, _i, _o in fans]
    ymin = min(0.0, min(lows))
    span = max(highs) - ymin
    label_y = ymin + 0.08 * span
    ax.text(2, label_y, "month 0\nmaster's", ha="left", va="bottom", fontsize=9, color="#333333")
    ax.text(
        GRADUATION_MONTH + 2, label_y, "month 24\ngraduation",
        ha="left", va="bottom", fontsize=9, color="#333333",
    )
    ax.text(
        ZURICH_MONTH - 2, label_y + 0.16 * span, "month 48\nZurich",
        ha="right", va="bottom", fontsize=9, color="#F58518",
    )
    ax.text(
        GERMANY_MONTH + 2, label_y + 0.16 * span, "month 60\nGermany",
        ha="left", va="bottom", fontsize=9, color="#54A24B",
    )
    ax.set_xlim(-6, HORIZON - 1)
    ax.set_title("Alexia: 20 years, where she lives after the master's")
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
