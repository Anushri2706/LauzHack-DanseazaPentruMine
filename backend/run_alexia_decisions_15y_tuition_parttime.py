# Run from backend/: /opt/anaconda3/bin/python run_alexia_decisions_15y_tuition_parttime.py
"""15-year fans, consumption recipe. Only the master's decision changes.

During her 24 student months she keeps the 1,200 stipend and the 2,393 student
cost of living, plus a side job of 0.40 x 1,200 and EPFL Swiss tuition of 780
per semester (130 per month). She stays in the student regime: the side job is
not the part-time state, and it does not reprice the lease. Stay-a-student and
leave-and-look-for-work are the consumption baseline.
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

HORIZON = 180
N_PATHS = 4000
SEED = 0
GRADUATION_MONTH = 24
OVERRIDES = {"scaled_spending": 1}
STIPEND_CHF = 1200.0
SIDE_JOB_FRACTION = 0.40
SIDE_JOB_CHF = STIPEND_CHF * 40.0 / 100.0  # 480, on top of the stipend
TUITION_PER_SEMESTER_CHF = 780.0
TUITION_MONTHLY_CHF = TUITION_PER_SEMESTER_CHF / 6.0  # 130
STUDENT_LIVING_CHF = 2393.0
MASTERS_STUDENT_INCOME_CHF = STIPEND_CHF + SIDE_JOB_CHF  # 1,680
MASTERS_STUDENT_SPEND_CHF = STUDENT_LIVING_CHF + TUITION_MONTHLY_CHF  # 2,523
MASTERS_NAME = "Master's, then work"
ARTIFACT = ROOT / "artifacts" / "alexia-15y-tuition-parttime.png"
STORE_PNG = Path(
    "/Users/victorracu/Library/Application Support/Cursor/AgentStores/"
    "cursor_agent_stores/bc-e3d1c389-9d7a-4206-a8d2-334e3178188b/files/media/"
    "alexia-15y-tuition-parttime.png"
)
DECISIONS = [
    ("Stay a student", [], "#4C78A8"),
    (
        MASTERS_NAME,
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


def _apply_masters_student_terms(paths: RegimePaths) -> RegimePaths:
    """Add the side job and the monthly tuition on the 24 student months only."""
    income = np.array(paths.income_chf, copy=True)
    net = np.array(paths.net_flow_chf, copy=True)
    savings = np.array(paths.savings_rate, copy=True)
    income[:, :GRADUATION_MONTH] += SIDE_JOB_CHF
    net[:, :GRADUATION_MONTH] += SIDE_JOB_CHF - TUITION_MONTHLY_CHF
    window_income = income[:, :GRADUATION_MONTH]
    window_net = net[:, :GRADUATION_MONTH]
    ok = np.abs(window_income) > 1.0
    savings[:, :GRADUATION_MONTH] = np.where(
        ok, window_net / np.where(ok, window_income, 1.0), savings[:, :GRADUATION_MONTH]
    )
    return replace(paths, income_chf=income, net_flow_chf=net, savings_rate=savings)


def main() -> None:
    if SIDE_JOB_CHF != 480.0 or TUITION_MONTHLY_CHF != 130.0:
        raise RuntimeError(
            f"expected side job 480 and tuition 130, got {SIDE_JOB_CHF} and {TUITION_MONTHLY_CHF}"
        )
    if MASTERS_STUDENT_INCOME_CHF != 1680.0 or MASTERS_STUDENT_SPEND_CHF != 2523.0:
        raise RuntimeError(
            f"expected income 1680 and spend 2523, got {MASTERS_STUDENT_INCOME_CHF} and {MASTERS_STUDENT_SPEND_CHF}"
        )
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
        "master's student months only: "
        f"income {MASTERS_STUDENT_INCOME_CHF:,.0f} = {STIPEND_CHF:,.0f} stipend + "
        f"{SIDE_JOB_CHF:,.0f} side job ({SIDE_JOB_FRACTION:.2f} x {STIPEND_CHF:,.0f}); "
        f"spend {MASTERS_STUDENT_SPEND_CHF:,.0f} = {STUDENT_LIVING_CHF:,.0f} student cost of living + "
        f"{TUITION_MONTHLY_CHF:,.0f} tuition ({TUITION_PER_SEMESTER_CHF:,.0f} CHF per semester / 6). "
        "Other decisions unchanged."
    )
    print(f"engine mode {engine.mode}, expense-priors recipe, {N_PATHS} paths, {HORIZON} months, seed {SEED}")
    table = engine.income_table.with_overrides(OVERRIDES)
    if float(table.params["student_living_chf"]) != STUDENT_LIVING_CHF:
        raise RuntimeError("student cost of living is not 2,393")
    if float(alexia.ref_income) != STIPEND_CHF or float(table.params["student_income_chf"]) != STIPEND_CHF:
        raise RuntimeError("student stipend is not 1,200")

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
        if name == MASTERS_NAME:
            raw = paths
            paths = _apply_masters_student_terms(raw)
            _check_masters_terms(raw, paths, table, alexia, interventions)
        fans.append((name, colour, paths, run_cash(alexia, paths).cash, interventions))
    _check(fans, table, alexia)
    _print_table(fans)
    _plot(fans)


def _check_masters_terms(raw: RegimePaths, paths: RegimePaths, table, alexia, interventions) -> None:
    student = STATE_INDEX["student"]
    if not (raw.states[:, :GRADUATION_MONTH] == student).all():
        raise RuntimeError("master's path was not a student for the first 24 months")
    if not np.allclose(raw.income_chf[:, :GRADUATION_MONTH], STIPEND_CHF):
        raise RuntimeError("master's stipend was not 1,200 before the side job")
    housing, other = table.consumption_paths(raw.states, alexia, interventions)
    living = housing[:, :GRADUATION_MONTH] + other[:, :GRADUATION_MONTH]
    if not np.allclose(living, STUDENT_LIVING_CHF):
        raise RuntimeError("master's student months did not pay the 2,393 cost of living")
    if not np.allclose(housing[:, :GRADUATION_MONTH], alexia.rent):
        raise RuntimeError("student rent was not held at 900")
    if raw.lump_chf is None:
        raise RuntimeError("consumption recipe did not sample the yearly lump")
    stipend_net = STIPEND_CHF - STUDENT_LIVING_CHF - raw.lump_chf[:, :GRADUATION_MONTH]
    if not np.allclose(raw.net_flow_chf[:, :GRADUATION_MONTH], stipend_net):
        raise RuntimeError("student net flow was not stipend minus living minus the lump")
    if not np.allclose(paths.income_chf[:, :GRADUATION_MONTH], MASTERS_STUDENT_INCOME_CHF):
        raise RuntimeError("master's student income was not 1,680")
    charged = MASTERS_STUDENT_INCOME_CHF - MASTERS_STUDENT_SPEND_CHF - raw.lump_chf[:, :GRADUATION_MONTH]
    if not np.allclose(paths.net_flow_chf[:, :GRADUATION_MONTH], charged):
        raise RuntimeError("master's student spend was not 2,523 plus the unchanged lump")
    if not np.allclose(paths.income_chf[:, GRADUATION_MONTH:], raw.income_chf[:, GRADUATION_MONTH:]):
        raise RuntimeError("the side job changed income after graduation")
    if not np.allclose(paths.net_flow_chf[:, GRADUATION_MONTH:], raw.net_flow_chf[:, GRADUATION_MONTH:]):
        raise RuntimeError("tuition or the side job changed cash flow after graduation")
    if not np.shares_memory(paths.states, raw.states) and not np.array_equal(paths.states, raw.states):
        raise RuntimeError("the side job changed the regime path")
    extra = (SIDE_JOB_CHF - TUITION_MONTHLY_CHF) * np.minimum(np.arange(1, HORIZON + 1), GRADUATION_MONTH)
    raw_cash = run_cash(alexia, raw).cash
    new_cash = run_cash(alexia, paths).cash
    if not np.allclose(new_cash - raw_cash, extra):
        raise RuntimeError("cash did not rise by 350 CHF per student month and then stay flat")


def _check(fans, table, alexia) -> None:
    stay, masters, search = (item[2] for item in fans)
    student = STATE_INDEX["student"]
    employed = STATE_INDEX["employed"]
    job_search = STATE_INDEX["job_search"]
    if (stay.states != student).any():
        raise RuntimeError("stay-a-student left the student regime")
    if not np.allclose(stay.income_chf, STIPEND_CHF):
        raise RuntimeError("stay-a-student income was changed")
    if not (masters.states[:, :GRADUATION_MONTH] == student).all():
        raise RuntimeError("master's path was not a student for the first 24 months")
    if (masters.states[:, GRADUATION_MONTH] == employed).mean() < 0.95:
        raise RuntimeError("master's path did not enter work at month 24")
    if not (search.states[:, 0] == job_search).all():
        raise RuntimeError("leave-studies path did not start in job search")
    if np.allclose(search.income_chf[:, 0], MASTERS_STUDENT_INCOME_CHF):
        raise RuntimeError("the side job was applied to the leave-studies path")
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
    ax.set_title("Alexia: 15 years, master's tuition and side job")
    ax.set_xlabel("Month")
    ax.set_ylabel("Cash (CHF)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:,.0f}"))
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(
        0.5, 0.005,
        "Master's months 0–23 only: income 1,680 CHF (1,200 + 480 side job), "
        "spend 2,523 CHF (2,393 living + 130 tuition). Other decisions unchanged.",
        ha="center", va="bottom", fontsize=8, color="#444444",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
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
