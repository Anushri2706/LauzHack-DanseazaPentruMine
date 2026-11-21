# Run from backend/: /opt/anaconda3/bin/python run_alexia_decisions_15y_move_abroad.py
"""15-year consumption fans. The master's path moves abroad at month 60.

Same expense-priors recipe as run_alexia_decisions_15y_consumption.py
(scaled_spending=1). The only change is one attribute on the master's path:
abroad=True from month 60, five years in and three years after the month-24
graduation. The engine already applies the Germany factors when that attribute
is set: the grown wage is rescaled by abroad_income_multiplier (0.51) and the
housing bill by abroad_cost_multiplier (0.63). Non-housing stays
base + rate * gross, so the wage-linked part falls with the wage and the flat
base is not multiplied by 0.63. No relocation bill and no extra job-search spell.
Stay-a-student and leave-and-look-for-work stay in Switzerland.
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
MOVE_MONTH = 60
OVERRIDES = {"scaled_spending": 1}
INCOME_FACTOR = 0.51
HOUSING_FACTOR = 0.63
ARTIFACT = ROOT / "artifacts" / "alexia-15y-move-abroad.png"
STORE_PNG = Path(
    "/Users/victorracu/Library/Application Support/Cursor/AgentStores/"
    "cursor_agent_stores/bc-e3d1c389-9d7a-4206-a8d2-334e3178188b/files/media/alexia-15y-move-abroad.png"
)
MASTERS_SWISS = [
    Intervention("force_state", 0, GRADUATION_MONTH, state="student"),
    Intervention("set_attribute", GRADUATION_MONTH, attribute=("has_masters", True)),
]
MASTERS_MOVE = MASTERS_SWISS + [
    Intervention("set_attribute", MOVE_MONTH, attribute=("abroad", True)),
]
DECISIONS = [
    ("Stay a student", [], "#4C78A8"),
    ("Master's, move abroad", MASTERS_MOVE, "#F58518"),
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
    print(
        f"engine mode {engine.mode}, expense-priors recipe, {N_PATHS} paths, "
        f"{HORIZON} months, seed {SEED}"
    )
    table = engine.income_table.with_overrides(OVERRIDES)
    income_factor = float(table.params["abroad_income_multiplier"])
    housing_factor = float(table.params["abroad_cost_multiplier"])
    if income_factor != INCOME_FACTOR or housing_factor != HOUSING_FACTOR:
        raise RuntimeError(
            f"Germany factors are {income_factor} and {housing_factor}, "
            f"expected {INCOME_FACTOR} and {HOUSING_FACTOR}"
        )
    print(
        f"move: set abroad=True at month {MOVE_MONTH} on the master's path only; "
        f"wage x{income_factor}, housing x{housing_factor}; "
        "other two decisions stay in Switzerland; no job-search spell"
    )

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

    swiss = engine.simulate(
        alexia,
        horizon_months=HORIZON,
        n_paths=N_PATHS,
        interventions=MASTERS_SWISS,
        assumption_overrides=OVERRIDES,
        seed=SEED,
    )
    _check(fans, table, alexia, swiss, income_factor, housing_factor)
    _print_table(fans)
    _breakdown(alexia, table, fans[1], income_factor, housing_factor)
    _plot(fans)


def _check(fans, table, alexia, swiss, income_factor: float, housing_factor: float) -> None:
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
    if not np.array_equal(masters.states, swiss.states):
        raise RuntimeError("abroad changed the master's regime path")

    housing, other = table.consumption_paths(masters.states, alexia, fans[1][4])
    swiss_housing, swiss_other = table.consumption_paths(swiss.states, alexia, MASTERS_SWISS)
    if not np.allclose(housing[:, :GRADUATION_MONTH], alexia.rent):
        raise RuntimeError("student rent was not held at 900")
    signed = housing[:, GRADUATION_MONTH]
    if not np.allclose(housing[:, GRADUATION_MONTH:MOVE_MONTH], signed[:, None]):
        raise RuntimeError("the Swiss lease moved before month 60")
    if not np.allclose(housing[:, MOVE_MONTH:], signed[:, None] * housing_factor):
        raise RuntimeError("housing was not multiplied by the Germany cost factor from month 60")
    if not np.allclose(masters.income_chf[:, :MOVE_MONTH], swiss.income_chf[:, :MOVE_MONTH]):
        raise RuntimeError("income changed before the move")
    # The salary itself is scaled by 0.51. Unemployment insurance then applies the
    # insured-earnings cap, so a capped Swiss benefit does not fall by the full 0.51.
    earning = (masters.states[:, MOVE_MONTH:] == employed) | (masters.states[:, MOVE_MONTH:] == STATE_INDEX["part_time"])
    moved_pay = masters.income_chf[:, MOVE_MONTH:]
    swiss_pay = swiss.income_chf[:, MOVE_MONTH:]
    if not np.allclose(moved_pay[earning], swiss_pay[earning] * income_factor):
        raise RuntimeError("the wage was not multiplied by the Germany income factor from month 60")
    if not np.allclose(housing[:, :MOVE_MONTH], swiss_housing[:, :MOVE_MONTH]):
        raise RuntimeError("housing changed before the move")
    if not np.allclose(housing[:, MOVE_MONTH:], swiss_housing[:, MOVE_MONTH:] * housing_factor):
        raise RuntimeError("housing did not match the Switzerland lease times 0.63")
    at_move = masters.states[:, MOVE_MONTH] == employed
    wage_ratio = masters.income_chf[at_move, MOVE_MONTH] / swiss.income_chf[at_move, MOVE_MONTH]
    rent_ratio = housing[:, MOVE_MONTH] / swiss_housing[:, MOVE_MONTH]
    print(
        f"against the same master's paths kept in Switzerland, month {MOVE_MONTH}: "
        f"employed wage ratio median {np.median(wage_ratio):.4f} "
        f"on {int(at_move.sum())} paths, housing ratio median {np.median(rent_ratio):.4f}"
    )

    # Non-housing is base + rate * gross. The wage-linked part follows the 0.51 wage;
    # the flat base is not multiplied by 0.63.
    base = float(table.params["nonhousing_base_chf"])
    working = masters.states[:, MOVE_MONTH:] == employed
    if working.sum() < 1000:
        raise RuntimeError("too few employed months after the move to check non-housing")
    linked = base + income_factor * (swiss_other[:, MOVE_MONTH:] - base)
    if not np.allclose(other[:, MOVE_MONTH:][working], linked[working]):
        raise RuntimeError("non-housing did not follow base + rate * the reduced wage")
    if np.allclose(other[:, MOVE_MONTH:][working], swiss_other[:, MOVE_MONTH:][working] * housing_factor):
        raise RuntimeError("non-housing was scaled by the rent factor; the recipe does not do that")

    moved_cash = fans[1][3]
    swiss_cash = run_cash(alexia, swiss).cash
    if not np.allclose(moved_cash[:, :MOVE_MONTH], swiss_cash[:, :MOVE_MONTH]):
        raise RuntimeError("cash diverged before month 60")
    if np.allclose(moved_cash[:, MOVE_MONTH], swiss_cash[:, MOVE_MONTH]):
        raise RuntimeError("the move did not change cash at month 60")

    for name, paths, interventions in (
        ("Stay a student", stay, fans[0][4]),
        ("Leave and look for work", search, fans[2][4]),
    ):
        home, _rest = table.consumption_paths(paths.states, alexia, interventions)
        if not np.allclose(home[:, MOVE_MONTH:], home[:, MOVE_MONTH - 1][:, None]):
            raise RuntimeError(f"{name} housing changed at the move month")


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


def _breakdown(alexia, table, fan, income_factor: float, housing_factor: float) -> None:
    _name, _colour, paths, _cash, interventions = fan
    housing, other = table.consumption_paths(paths.states, alexia, interventions)
    not_dec = (np.arange(paths.horizon) % 12) != 11
    before = (paths.states == STATE_INDEX["employed"]) & not_dec & (np.arange(paths.horizon) < MOVE_MONTH)
    after = (paths.states == STATE_INDEX["employed"]) & not_dec & (np.arange(paths.horizon) >= MOVE_MONTH)
    before = before & (np.arange(paths.horizon) >= GRADUATION_MONTH)
    for label, mask in (("before the move", before), ("after the move", after)):
        inc = float(paths.income_chf[mask].mean())
        ren = float(housing[mask].mean())
        non = float(other[mask].mean())
        print(
            f"typical working month (master's, employed, not December, {label}): "
            f"income {inc:,.0f}, rent {ren:,.0f}, non-housing {non:,.0f}"
        )
    print(
        f"Germany factors applied by the engine from month {MOVE_MONTH}: "
        f"wage x{income_factor}, housing x{housing_factor}. "
        "Month 59 also takes the anniversary raise, so the adjacent-month wage ratio "
        "is that raise times 0.51."
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
    ax.axvline(MOVE_MONTH, color="#E45756", linewidth=1.2, linestyle="--")
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
        MOVE_MONTH + 2, label_y, "month 60\nmove abroad",
        ha="left", va="bottom", fontsize=9, color="#E45756",
    )
    ax.set_xlim(-6, HORIZON - 1)
    ax.set_title("Alexia: 15 years, master's path moves abroad at month 60")
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
