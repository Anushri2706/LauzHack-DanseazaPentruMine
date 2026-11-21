# Run from backend/: /opt/anaconda3/bin/python run_alexia_20y_during_degree.py
"""20-year cash fans for working during Alexia's master's. The degree starts now.

All three paths finish the degree, then enter employed with has_masters. They stay
in the student regime while studying, so the side job and the half-time wage do
not sign the lease, do not start the experience clock, and do not set the
unemployment-insurance wage. ``scaled_spending=1`` is the expense-priors recipe.
The yearly lump is left as the engine drew it.

(1) No side job, 24 months. Income 1,200. Spend 2,393 (EPFL living budget, no tuition).
(2) Small side job, 24 months. Income 1,680 = 1,200 + 480. Spend 2,523 = 2,393 + 130.
(3) Half-time, 30 months. Income 1,200 + 0.5 x 6,129 during study. Spend 2,523.
    6,129 is the entry salary rounded from entry_salary_chf (6,128.79), before the
    master's premium. 130/month is EPFL tuition of 780 per semester.
"""
from __future__ import annotations

import sys
from dataclasses import replace
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
from twin.types import STATE_INDEX, Intervention, RegimePaths

HORIZON = 240
N_PATHS = 4000
SEED = 0
OVERRIDES = {"scaled_spending": 1}
STIPEND_CHF = 1200.0
SIDE_JOB_CHF = 480.0
TUITION_PER_SEMESTER_CHF = 780.0
TUITION_MONTHLY_CHF = TUITION_PER_SEMESTER_CHF / 6.0  # 130
STUDENT_LIVING_CHF = 2393.0
ENTRY_SALARY_ROUNDED_CHF = 6129.0
HALF_TIME_CHF = 0.5 * ENTRY_SALARY_ROUNDED_CHF  # 3,064.5
SHORT_MONTHS = 24
LONG_MONTHS = 30
ARTIFACT = ROOT / "artifacts" / "alexia-20y-during-degree.png"
STORE_PNG = Path(
    "/Users/victorracu/Library/Application Support/Cursor/AgentStores/"
    "cursor_agent_stores/bc-e3d1c389-9d7a-4206-a8d2-334e3178188b/files/media/"
    "alexia-20y-during-degree.png"
)
# name, study months, study income, study spend, colour
PATHS = [
    ("No side job", SHORT_MONTHS, STIPEND_CHF, STUDENT_LIVING_CHF, "#4C78A8"),
    (
        "Small side job",
        SHORT_MONTHS,
        STIPEND_CHF + SIDE_JOB_CHF,
        STUDENT_LIVING_CHF + TUITION_MONTHLY_CHF,
        "#F58518",
    ),
    (
        "Half-time",
        LONG_MONTHS,
        STIPEND_CHF + HALF_TIME_CHF,
        STUDENT_LIVING_CHF + TUITION_MONTHLY_CHF,
        "#54A24B",
    ),
]


def _chf(value: float) -> str:
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"


def _interventions(months: int) -> list[Intervention]:
    return [
        Intervention("force_state", 0, months, state="student"),
        Intervention("set_attribute", months, attribute=("has_masters", True)),
    ]


def _apply_study_cash(paths: RegimePaths, months: int, income_chf: float, spend_chf: float) -> RegimePaths:
    """Set study-month income and the living bill. The yearly lump stays on the net flow."""
    if paths.lump_chf is None:
        raise RuntimeError("scaled spending did not attach the yearly lump")
    income = np.array(paths.income_chf, copy=True)
    net = np.array(paths.net_flow_chf, copy=True)
    savings = np.array(paths.savings_rate, copy=True)
    lump = paths.lump_chf[:, :months]
    income[:, :months] = income_chf
    net[:, :months] = income_chf - spend_chf - lump
    window_income = income[:, :months]
    window_net = net[:, :months]
    ok = np.abs(window_income) > 1.0
    savings[:, :months] = np.where(
        ok, window_net / np.where(ok, window_income, 1.0), savings[:, :months]
    )
    return replace(paths, income_chf=income, net_flow_chf=net, savings_rate=savings)


def main() -> None:
    if TUITION_MONTHLY_CHF != 130.0:
        raise RuntimeError(f"expected tuition 130/month, got {TUITION_MONTHLY_CHF}")
    if PATHS[1][2] != 1680.0 or PATHS[1][3] != 2523.0:
        raise RuntimeError("small side job is not income 1,680 and spend 2,523")
    if PATHS[2][2] != 4264.5 or PATHS[2][3] != 2523.0:
        raise RuntimeError("half-time is not income 4,264.5 and spend 2,523")
    alexia = load_profile("P00001")
    engine = RegimeEngine(fit_emissions(), load_transition_priors())
    if engine.mode != "hmm":
        raise RuntimeError(f"expected the HMM engine, got mode={engine.mode!r}")
    table = engine.income_table.with_overrides(OVERRIDES)
    entry = float(table.params["entry_salary_chf"])
    if round(entry) != ENTRY_SALARY_ROUNDED_CHF:
        raise RuntimeError(f"entry salary {entry} does not round to {ENTRY_SALARY_ROUNDED_CHF}")
    if float(table.params["student_living_chf"]) != STUDENT_LIVING_CHF:
        raise RuntimeError("student cost of living is not 2,393")
    if float(alexia.ref_income) != STIPEND_CHF or float(table.params["student_income_chf"]) != STIPEND_CHF:
        raise RuntimeError("student stipend is not 1,200")
    if float(table.params["scaled_spending"]) < 0.5:
        raise RuntimeError("scaled_spending did not turn on the expense-priors recipe")
    offer_attrs = dict(alexia.attributes)
    offer_attrs["has_masters"] = True
    offer = table.full_time_salary(alexia, offer_attrs)
    print(
        f"{alexia.name} {alexia.profile_id}: age {alexia.age}, {alexia.initial_state}, "
        f"Lausanne, opening {alexia.opening_cash:.0f} CHF, income {alexia.ref_income:.0f}, "
        f"rent {alexia.rent:.0f}"
    )
    print(
        f"engine mode {engine.mode}, scaled_spending=1, {N_PATHS} paths, "
        f"{HORIZON} months, seed {SEED}"
    )
    print(
        f"no side job: {SHORT_MONTHS} months, income {STIPEND_CHF:,.0f}, spend {STUDENT_LIVING_CHF:,.0f}. "
        f"small side job: {SHORT_MONTHS} months, income {STIPEND_CHF + SIDE_JOB_CHF:,.0f} "
        f"= {STIPEND_CHF:,.0f}+{SIDE_JOB_CHF:,.0f}, spend {STUDENT_LIVING_CHF + TUITION_MONTHLY_CHF:,.0f} "
        f"= {STUDENT_LIVING_CHF:,.0f}+{TUITION_MONTHLY_CHF:,.0f}. "
        f"half-time: {LONG_MONTHS} months, income {STIPEND_CHF + HALF_TIME_CHF:,.2f} "
        f"= {STIPEND_CHF:,.0f}+0.5x{ENTRY_SALARY_ROUNDED_CHF:,.0f}, spend "
        f"{STUDENT_LIVING_CHF + TUITION_MONTHLY_CHF:,.0f}. "
        f"Then employed at {offer:,.2f} (entry {entry:,.2f} x 1.10)."
    )

    raw_by_months: dict[int, tuple[list[Intervention], RegimePaths]] = {}
    for months in (SHORT_MONTHS, LONG_MONTHS):
        interventions = _interventions(months)
        paths = engine.simulate(
            alexia,
            horizon_months=HORIZON,
            n_paths=N_PATHS,
            interventions=interventions,
            assumption_overrides=OVERRIDES,
            seed=SEED,
        )
        raw_by_months[months] = (interventions, paths)

    fans = []
    for name, months, income_chf, spend_chf, colour in PATHS:
        interventions, raw = raw_by_months[months]
        paths = _apply_study_cash(raw, months, income_chf, spend_chf)
        _check_path(name, raw, paths, months, income_chf, spend_chf, table, alexia, interventions, offer)
        cash = run_cash(alexia, paths).cash
        fans.append((name, colour, paths, cash, months, income_chf, spend_chf))
    _check_gaps(fans)
    _print_table(fans)
    _plot(fans)


def _check_path(name, raw, paths, months, income_chf, spend_chf, table, alexia, interventions, offer) -> None:
    student = STATE_INDEX["student"]
    employed = STATE_INDEX["employed"]
    if paths.states.shape != (N_PATHS, HORIZON):
        raise RuntimeError(f"{name}: unexpected shape {paths.states.shape}")
    if not (raw.states[:, :months] == student).all():
        raise RuntimeError(f"{name}: not a student for the first {months} months")
    if not (raw.states[:, months] == employed).all():
        raise RuntimeError(f"{name}: did not enter employed at month {months}")
    if not np.allclose(raw.income_chf[:, :months], STIPEND_CHF):
        raise RuntimeError(f"{name}: stipend was not 1,200 before the overlay")
    if raw.lump_chf is None or not np.any(raw.lump_chf):
        raise RuntimeError(f"{name}: yearly lump was not sampled")
    paid = raw.income_chf[:, :months] - raw.net_flow_chf[:, :months] - raw.lump_chf[:, :months]
    if not np.allclose(paid, STUDENT_LIVING_CHF):
        raise RuntimeError(f"{name}: student months did not pay the 2,393 living budget")
    if not np.allclose(paths.income_chf[:, :months], income_chf):
        raise RuntimeError(f"{name}: study income was not {income_chf}")
    charged = income_chf - spend_chf - raw.lump_chf[:, :months]
    if not np.allclose(paths.net_flow_chf[:, :months], charged):
        raise RuntimeError(f"{name}: study spend was not {spend_chf} plus the unchanged lump")
    if not np.allclose(paths.income_chf[:, months:], raw.income_chf[:, months:]):
        raise RuntimeError(f"{name}: study terms changed income after graduation")
    if not np.allclose(paths.net_flow_chf[:, months:], raw.net_flow_chf[:, months:]):
        raise RuntimeError(f"{name}: study terms changed cash flow after graduation")
    if not np.array_equal(paths.states, raw.states):
        raise RuntimeError(f"{name}: study terms changed the regime path")
    ok = np.abs(paths.income_chf) > 1.0
    if not np.allclose(paths.net_flow_chf[ok], paths.savings_rate[ok] * paths.income_chf[ok]):
        raise RuntimeError(f"{name}: savings rate does not match net flow / income")
    if not np.allclose(raw.income_chf[:, months], offer):
        raise RuntimeError(f"{name}: graduation wage was not the master's entry salary")
    housing, other = table.consumption_paths(raw.states, alexia, interventions)
    if not np.allclose(housing[:, :months], alexia.rent):
        raise RuntimeError(f"{name}: student rent was not held at 900")
    if not np.allclose(housing[:, :months] + other[:, :months], STUDENT_LIVING_CHF):
        raise RuntimeError(f"{name}: student living budget was not 2,393")
    signed = table.rent_from_earnings(np.full(N_PATHS, offer)) + float(table.params["rent_charges_chf"])
    if not np.allclose(housing[:, months], signed):
        raise RuntimeError(f"{name}: the lease was not signed at the master's wage")
    if not np.allclose(housing[:, months:], signed[:, None]):
        raise RuntimeError(f"{name}: the post-school lease was recomputed")
    # Half-time pay must not be what the lease was underwritten on.
    half_lease = float(table.rent_from_earnings(STIPEND_CHF + HALF_TIME_CHF))
    if np.allclose(housing[0, months] - float(table.params["rent_charges_chf"]), half_lease):
        raise RuntimeError(f"{name}: the lease was signed on the half-time wage")
    gap = (income_chf - STIPEND_CHF) - (spend_chf - STUDENT_LIVING_CHF)
    extra = gap * np.minimum(np.arange(1, HORIZON + 1), months)
    if not np.allclose(run_cash(alexia, paths).cash - run_cash(alexia, raw).cash, extra):
        raise RuntimeError(f"{name}: cash did not move by {gap} CHF per study month and then stay flat")


def _check_gaps(fans) -> None:
    none, side, half = fans
    if none[4] != SHORT_MONTHS or side[4] != SHORT_MONTHS or half[4] != LONG_MONTHS:
        raise RuntimeError("study lengths were reordered")
    if not np.array_equal(none[2].states, side[2].states):
        raise RuntimeError("the side job changed the regime path relative to no side job")
    side_gap = (SIDE_JOB_CHF - TUITION_MONTHLY_CHF) * np.minimum(np.arange(1, HORIZON + 1), SHORT_MONTHS)
    if not np.allclose(side[3] - none[3], side_gap):
        raise RuntimeError("small side job cash is not 350 CHF per study month ahead of no side job")
    if np.array_equal(half[2].states[:, :LONG_MONTHS + 1], none[2].states[:, :LONG_MONTHS + 1]):
        raise RuntimeError("half-time used the 24-month degree")
    if not (half[2].states[:, SHORT_MONTHS:LONG_MONTHS] == STATE_INDEX["student"]).all():
        raise RuntimeError("half-time left school before month 30")


def _print_table(fans) -> None:
    header = (
        f"{'path':<16} {'year':>6} {'median':>14} {'p10':>14} {'p90':>14} "
        f"{'ever_neg':>10} {'neg_end':>10}"
    )
    print(header)
    print("-" * len(header))
    for name, _colour, _paths, cash, _months, _income, _spend in fans:
        ever = (cash < 0).any(axis=1).mean()
        still = (cash[:, -1] < 0).mean()
        for year in (10, 20):
            closing = cash[:, year * 12 - 1]
            print(
                f"{name:<16} {year:>6} {np.median(closing):>14,.2f} "
                f"{np.quantile(closing, 0.10):>14,.2f} {np.quantile(closing, 0.90):>14,.2f} "
                f"{ever:>10.3f} {still:>10.3f}"
            )


def _plot(fans) -> None:
    months = np.arange(HORIZON)
    fig, ax = plt.subplots(figsize=(10.2, 5.8), dpi=140)
    for name, colour, _paths, cash, study_months, income_chf, spend_chf in fans:
        low, mid, high = np.quantile(cash, [0.10, 0.50, 0.90], axis=0)
        label = (
            f"{name}, {study_months} mo "
            f"({_chf(income_chf)} / {_chf(spend_chf)})"
        )
        ax.fill_between(months, low, high, color=colour, alpha=0.18, linewidth=0)
        ax.plot(months, mid, color=colour, linewidth=2.2, label=label)
    ax.axhline(0, color="#444444", linewidth=0.8, zorder=0)
    ax.axvline(SHORT_MONTHS, color="#333333", linewidth=1.0, linestyle="--", zorder=0)
    ax.axvline(LONG_MONTHS, color="#333333", linewidth=1.0, linestyle=":", zorder=0)
    ax.set_xlim(-6, HORIZON - 1)
    ax.set_title("Alexia: working during the master's, cash over 20 years")
    ax.set_xlabel("Month")
    ax.set_ylabel("Cash (CHF)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:,.0f}"))
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.16, 1.0))
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(
        0.5, 0.012,
        "Degree starts now, then employed with a master's. Dashed line: month 24. Dotted line: month 30.\n"
        "No side job: 24 months, income 1,200, spend 2,393.   "
        "Small side job: 24 months, income 1,680 (1,200+480), spend 2,523 (2,393+130).\n"
        "Half-time: 30 months, income 4,264.50 (1,200 + 0.5×6,129), spend 2,523.   "
        "Tuition 130/month is EPFL 780 per semester. Bands are P10–P90.",
        ha="center", va="bottom", fontsize=7.5, color="#444444",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
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
