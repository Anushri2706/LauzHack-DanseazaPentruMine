# Run from backend/: /opt/anaconda3/bin/python run_alexia_decisions_20y_breaks.py
"""20-year fans for a break after Alexia starts a master's now.

All three paths study for 24 months, then work with the master's premium.
Baseline spending is the expense-priors recipe (scaled_spending=1).

1. No break.
2. force_state job_search for 6 months at month 60. Pay is the existing
   unemployment-insurance rule: 70% of the last wage, 80% when that wage is
   at or below the low-earnings threshold, capped at insured earnings.
3. Parental leave for parental_leave_months (12) at month 84. Pay is
   parental_leave_replacement (0.80) times the last wage, housing unchanged,
   then employed. EO is more complicated; the prior is the flat year in
   assumptions/income_assumptions.csv.
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

HORIZON = 240
N_PATHS = 4000
SEED = 0
GRADUATION_MONTH = 24
JOB_SEARCH_START = 60
JOB_SEARCH_MONTHS = 6
PARENTAL_START = 84
OVERRIDES = {"scaled_spending": 1}
ARTIFACT = ROOT / "artifacts" / "alexia-20y-breaks.png"
STORE_PNG = Path(
    "/Users/victorracu/Library/Application Support/Cursor/AgentStores/"
    "cursor_agent_stores/bc-e3d1c389-9d7a-4206-a8d2-334e3178188b/files/media/alexia-20y-breaks.png"
)
MASTERS = [
    Intervention("force_state", 0, GRADUATION_MONTH, state="student"),
    Intervention("set_attribute", GRADUATION_MONTH, attribute=("has_masters", True)),
]
DECISIONS = [
    ("No break", list(MASTERS), "#4C78A8"),
    (
        "Job search, 6 months",
        list(MASTERS) + [
            Intervention("force_state", JOB_SEARCH_START, JOB_SEARCH_MONTHS, state="job_search"),
        ],
        "#F58518",
    ),
    (
        "Parental leave, 12 months",
        list(MASTERS) + [Intervention("parental_leave", PARENTAL_START)],
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
        f"rent {alexia.rent:.0f}, household {alexia.attributes.get('household_type')}"
    )
    table = engine.income_table.with_overrides(OVERRIDES)
    if not table.scales_spending():
        raise RuntimeError("scaled_spending did not turn on the consumption recipe")
    rate = float(table.params["parental_leave_replacement"])
    months = int(table.params["parental_leave_months"])
    if abs(rate - 0.80) > 1e-12 or months != 12:
        raise RuntimeError(f"parental-leave prior is {rate} for {months} months, expected 0.80 for 12")
    parental_end = PARENTAL_START + months
    if parental_end >= HORIZON:
        raise RuntimeError("parental leave does not finish inside the horizon")
    print(
        f"engine mode {engine.mode}, scaled_spending=1, {N_PATHS} paths, "
        f"{HORIZON} months, seed {SEED}"
    )
    print(
        f"job search months {JOB_SEARCH_START}..{JOB_SEARCH_START + JOB_SEARCH_MONTHS - 1}; "
        f"parental leave months {PARENTAL_START}..{parental_end - 1} at {rate:.0%} of the last wage"
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
    _check(fans, table, alexia, months, rate)
    _print_cash(fans)
    _print_breaks(fans, table, alexia, months, rate)
    _plot(fans, parental_end)


def _check(fans, table, alexia, parental_months: int, rate: float) -> None:
    quiet, search, leave = fans
    student = STATE_INDEX["student"]
    employed = STATE_INDEX["employed"]
    job_search = STATE_INDEX["job_search"]
    spell = slice(JOB_SEARCH_START, JOB_SEARCH_START + JOB_SEARCH_MONTHS)
    back = JOB_SEARCH_START + JOB_SEARCH_MONTHS
    leave_at = slice(PARENTAL_START, PARENTAL_START + parental_months)
    leave_back = PARENTAL_START + parental_months
    for name, _colour, paths, _cash, _iv in fans:
        if not (paths.states[:, :GRADUATION_MONTH] == student).all():
            raise RuntimeError(f"{name} was not a student for the first 24 months")
        if not (paths.states[:, GRADUATION_MONTH] == employed).all():
            raise RuntimeError(f"{name} did not enter employed at month 24")
        rebuilt = table.income_paths(paths.states, alexia, _iv)
        if not np.allclose(rebuilt, paths.income_chf):
            raise RuntimeError(f"{name} income does not match the income table")
    if not np.array_equal(quiet[2].states[:, :JOB_SEARCH_START], search[2].states[:, :JOB_SEARCH_START]):
        raise RuntimeError("job-search paths diverged before month 60")
    if not np.array_equal(quiet[2].states[:, :PARENTAL_START], leave[2].states[:, :PARENTAL_START]):
        raise RuntimeError("parental-leave paths diverged before month 84")
    if not (search[2].states[:, spell] == job_search).all():
        raise RuntimeError("job-search path was not in job search for all 6 months")
    if not (search[2].states[:, back] == employed).all():
        raise RuntimeError(f"job-search path did not return to employed at month {back}")
    if (quiet[2].states[:, JOB_SEARCH_START] == job_search).mean() > 0.5:
        raise RuntimeError("the no-break path was pinned into job search")
    if not (leave[2].states[:, leave_at] == employed).all():
        raise RuntimeError("parental leave was not pinned to employed")
    if not (leave[2].states[:, leave_back] == employed).all():
        raise RuntimeError(f"parental leave did not return to employed at month {leave_back}")

    for name, _colour, paths, _cash, interventions in fans:
        housing, _other = table.consumption_paths(paths.states, alexia, interventions)
        if not np.allclose(housing[:, :GRADUATION_MONTH], alexia.rent):
            raise RuntimeError(f"{name} did not keep the student rent at 900")
        signed = housing[:, GRADUATION_MONTH:GRADUATION_MONTH + 1]
        if not np.allclose(housing[:, GRADUATION_MONTH:], signed):
            raise RuntimeError(f"{name} recomputed the post-school lease")

    search_income = search[2].income_chf[:, spell]
    expected = _ui_benefit(search[2].states, search[2].income_chf, table, alexia)
    if not np.allclose(search_income, expected, atol=1e-6):
        gap = float(np.max(np.abs(search_income - expected)))
        raise RuntimeError(f"job-search income does not match the UI rule (max gap {gap:.4f})")
    fresh = _working(search[2].states[:, JOB_SEARCH_START - 1])
    if fresh.any() and not np.allclose(search_income[fresh], search_income[fresh, :1]):
        raise RuntimeError("a fresh job-search spell did not pay a constant benefit")

    leave_income = leave[2].income_chf[:, leave_at]
    if not np.allclose(leave_income, leave_income[:, :1]):
        raise RuntimeError("parental-leave income was not constant across the 12 months")
    working = _working(leave[2].states[:, PARENTAL_START - 1])
    wage = leave[2].income_chf[:, PARENTAL_START - 1]
    if not np.allclose(leave_income[working], rate * wage[working, None]):
        raise RuntimeError("parental leave is not 0.80 times the last wage")
    if not (leave_income[~working] > 0).all():
        raise RuntimeError("a parental-leave path had no last wage")
    housing, _other = table.consumption_paths(leave[2].states, alexia, leave[4])
    if not np.allclose(housing[:, leave_at], housing[:, PARENTAL_START - 1:PARENTAL_START]):
        raise RuntimeError("housing moved during parental leave")


def _working(states_t: np.ndarray) -> np.ndarray:
    employed = STATE_INDEX["employed"]
    part_time = STATE_INDEX["part_time"]
    return (states_t == employed) | (states_t == part_time)


def _last_wage(states: np.ndarray, income: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wage = np.zeros(states.shape[0])
    has = np.zeros(states.shape[0], dtype=bool)
    for t in range(JOB_SEARCH_START):
        working = _working(states[:, t])
        wage = np.where(working, income[:, t], wage)
        has = has | working
    return has, wage


def _ui_benefit(states, income, table, alexia) -> np.ndarray:
    """Existing UI rule on the pinned spell, from the last wage and the spell counter."""
    p = table.params
    has_worked, last_wage = _last_wage(states, income)
    high = p["ui_replacement_with_dependents"]
    household = alexia.attributes.get("household_type")
    rate = np.full(states.shape[0], high if household == "family" else p["ui_replacement_rate"])
    rate = np.where(last_wage <= p["ui_low_earnings_threshold_chf"], high, rate)
    benefit = rate * np.minimum(last_wage, p["ui_insured_earnings_cap_chf"])
    trailing = np.zeros(states.shape[0])
    alive = np.ones(states.shape[0], dtype=bool)
    job_search = STATE_INDEX["job_search"]
    for t in range(JOB_SEARCH_START - 1, -1, -1):
        alive &= states[:, t] == job_search
        trailing += alive
    out = np.empty((states.shape[0], JOB_SEARCH_MONTHS))
    for i, t in enumerate(range(JOB_SEARCH_START, JOB_SEARCH_START + JOB_SEARCH_MONTHS)):
        age = alexia.age + t / 12.0
        max_m = p["ui_max_months_under_25"] if age < 25 else p["ui_max_months"]
        months_in = trailing + i
        on_ui = has_worked & (months_in < max_m)
        inc = np.where(on_ui, benefit, 0.0)
        inc = np.where(has_worked & ~on_ui, p["job_search_floor_chf"], inc)
        never = ~has_worked
        wait = p["graduate_waiting_months"]
        span = p["graduate_benefit_months"]
        on_grad = never & (months_in >= wait) & (months_in < wait + span)
        grad_pay = p["graduate_benefit_under_25_chf"] if age < 25 else p["graduate_benefit_chf"]
        inc = np.where(on_grad, grad_pay, inc)
        out[:, i] = inc
    return out


def _print_cash(fans) -> None:
    header = (
        f"{'decision':<28} {'year':>6} {'median':>12} {'p10':>12} {'p90':>12} "
        f"{'ever_neg':>10}"
    )
    print(header)
    print("-" * len(header))
    for name, _colour, _paths, cash, _iv in fans:
        ever = (cash < 0).any(axis=1).mean()
        for year in (10, 20):
            closing = cash[:, year * 12 - 1]
            print(
                f"{name:<28} {year:>6} {np.median(closing):>12,.0f} "
                f"{np.quantile(closing, 0.10):>12,.0f} {np.quantile(closing, 0.90):>12,.0f} "
                f"{ever:>10.3f}"
            )


def _quant(x: np.ndarray) -> str:
    return (
        f"median {np.median(x):,.2f}, "
        f"p10 {np.quantile(x, 0.10):,.2f}, "
        f"p90 {np.quantile(x, 0.90):,.2f}"
    )


def _print_breaks(fans, table, alexia, parental_months: int, parental_rate: float) -> None:
    p = table.params
    _quiet, search, leave = fans
    spell = slice(JOB_SEARCH_START, JOB_SEARCH_START + JOB_SEARCH_MONTHS)
    fresh = _working(search[2].states[:, JOB_SEARCH_START - 1])
    _has, last_wage = _last_wage(search[2].states, search[2].income_chf)
    got = search[2].income_chf[:, spell]
    high = p["ui_replacement_with_dependents"]
    household = alexia.attributes.get("household_type")
    rate = np.full(got.shape[0], high if household == "family" else p["ui_replacement_rate"])
    rate = np.where(last_wage <= p["ui_low_earnings_threshold_chf"], high, rate)
    housing, _other = table.consumption_paths(search[2].states, alexia, search[4])
    print(
        f"job search income, months {JOB_SEARCH_START}..{JOB_SEARCH_START + JOB_SEARCH_MONTHS - 1}, "
        f"age {alexia.age + JOB_SEARCH_START / 12:.2f}, household {household}"
    )
    changing = np.ptp(got, axis=1) > 1e-6
    print(f"  all paths, month {JOB_SEARCH_START}: {_quant(got[:, 0])}")
    print(f"  paths whose benefit changes during the spell: {changing.mean():.3f}")
    if changing.any():
        print(f"  those paths, month {JOB_SEARCH_START + JOB_SEARCH_MONTHS - 1}: {_quant(got[changing, -1])}")
    if fresh.any():
        paid = got[fresh, 0]
        wage = last_wage[fresh]
        print(f"  working at month {JOB_SEARCH_START - 1}: {fresh.mean():.3f} of paths")
        print(f"  last wage: {_quant(wage)}")
        print(f"  income during the spell: {_quant(paid)}")
        print(
            f"  share at {p['ui_replacement_rate']:.0%} {np.isclose(rate[fresh], p['ui_replacement_rate']).mean():.3f}; "
            f"share at {high:.0%} {np.isclose(rate[fresh], high).mean():.3f}; "
            f"share above the cap {(wage > p['ui_insured_earnings_cap_chf']).mean():.3f}"
        )
    print(f"  not working the month before: {(~fresh).mean():.3f}")
    print(f"  housing during the spell: {_quant(housing[fresh, spell][:, 0]) if fresh.any() else 'n/a'}")

    leave_at = slice(PARENTAL_START, PARENTAL_START + parental_months)
    working = _working(leave[2].states[:, PARENTAL_START - 1])
    pay = leave[2].income_chf[:, leave_at][:, 0]
    wage = leave[2].income_chf[:, PARENTAL_START - 1]
    housing_l, _other_l = table.consumption_paths(leave[2].states, alexia, leave[4])
    print(
        f"parental leave income, months {PARENTAL_START}..{PARENTAL_START + parental_months - 1}, "
        f"age {alexia.age + PARENTAL_START / 12:.2f}, prior {parental_rate:.0%} for {parental_months} months"
    )
    print(f"  all paths, each month: {_quant(pay)}")
    print(f"  working at month {PARENTAL_START - 1}: {working.mean():.3f} of paths")
    if working.any():
        print(f"  last wage: {_quant(wage[working])}")
        print(f"  income during the leave: {_quant(pay[working])}")
        print(f"  income / last wage: {_quant(pay[working] / wage[working])}")
    if (~working).any():
        print(f"  not working the month before: {(~working).mean():.3f}; income {_quant(pay[~working])}")
    print(f"  housing during the leave: {_quant(housing_l[:, leave_at][:, 0])}")
    before = housing_l[:, PARENTAL_START - 1]
    if not np.allclose(housing_l[:, leave_at], before[:, None]):
        raise RuntimeError("reported housing does not match the month before the leave")


def _plot(fans, parental_end: int) -> None:
    months = np.arange(HORIZON)
    fig, ax = plt.subplots(figsize=(10.2, 5.8), dpi=140)
    ax.axvspan(
        JOB_SEARCH_START, JOB_SEARCH_START + JOB_SEARCH_MONTHS,
        color="#F58518", alpha=0.16, zorder=0,
    )
    ax.axvspan(PARENTAL_START, parental_end, color="#54A24B", alpha=0.16, zorder=0)
    for name, colour, _paths, cash, _iv in fans:
        low, mid, high = np.quantile(cash, [0.10, 0.50, 0.90], axis=0)
        ax.fill_between(months, low, high, color=colour, alpha=0.20, linewidth=0)
        ax.plot(months, mid, color=colour, linewidth=2.2, label=name)
    ax.axhline(0, color="#444444", linewidth=0.8, zorder=1)
    for at in (0, GRADUATION_MONTH, JOB_SEARCH_START, PARENTAL_START):
        ax.axvline(at, color="#333333", linewidth=1.0, linestyle="--", zorder=1)
    lows = [float(np.quantile(cash, 0.10, axis=0).min()) for _n, _c, _p, cash, _i in fans]
    highs = [float(np.quantile(cash, 0.90, axis=0).max()) for _n, _c, _p, cash, _i in fans]
    ymin = min(0.0, min(lows))
    span = max(highs) - ymin
    label_y = ymin + 0.06 * span
    ax.text(2, label_y, "month 0\nstart", ha="left", va="bottom", fontsize=8, color="#333333")
    ax.text(GRADUATION_MONTH + 2, label_y, "month 24\ngraduation", ha="left", va="bottom", fontsize=8, color="#333333")
    ax.text(JOB_SEARCH_START + 2, label_y, "month 60\njob search", ha="left", va="bottom", fontsize=8, color="#333333")
    ax.text(PARENTAL_START + 2, label_y, "month 84\nparental leave", ha="left", va="bottom", fontsize=8, color="#333333")
    ax.set_xlim(-6, HORIZON - 1)
    ax.set_title("Alexia: 20 years, a break after the master's")
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
        img = plt.imread(path)
        if img.ndim != 3 or img.shape[0] < 400 or img.shape[1] < 700:
            raise RuntimeError(f"{path} decoded to an unexpected shape {img.shape}")
        print(f"wrote {path} ({len(data)} bytes, {img.shape[1]}x{img.shape[0]})")


if __name__ == "__main__":
    main()
