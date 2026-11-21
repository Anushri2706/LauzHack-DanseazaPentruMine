# Run from backend/: /opt/anaconda3/bin/python run_alexia_decisions_15y_unemployment.py
"""Consumption-recipe fans, plus one shock: six months of job search at year 4.

Same three decisions and the same expense-priors recipe as
run_alexia_decisions_15y_consumption.py (scaled_spending=1). On both working
decisions a job_search phase is pinned for months 48..53. The engine's own
unemployment-insurance rule pays that spell: 70% of the last wage, 80% when
the last wage is at or below the low-earnings threshold, capped at insured
earnings. The lease signed on leaving school is not rewritten. Staying a
student is untouched.
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
SPELL_START = 48
SPELL_MONTHS = 6
OVERRIDES = {"scaled_spending": 1}
ARTIFACT = ROOT / "artifacts" / "alexia-15y-unemployment.png"
STORE_PNG = Path(
    "/Users/victorracu/Library/Application Support/Cursor/AgentStores/"
    "cursor_agent_stores/bc-e3d1c389-9d7a-4206-a8d2-334e3178188b/files/media/alexia-15y-unemployment.png"
)
UNEMPLOYMENT = Intervention("force_state", SPELL_START, SPELL_MONTHS, state="job_search")
DECISIONS = [
    ("Stay a student", [], "#4C78A8"),
    (
        "Master's, then work",
        [
            Intervention("force_state", 0, GRADUATION_MONTH, state="student"),
            Intervention("set_attribute", GRADUATION_MONTH, attribute=("has_masters", True)),
            UNEMPLOYMENT,
        ],
        "#F58518",
    ),
    (
        "Leave and look for work",
        [
            Intervention("force_state", 0, None, state="job_search"),
            UNEMPLOYMENT,
        ],
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
    print(
        f"engine mode {engine.mode}, expense-priors recipe, {N_PATHS} paths, "
        f"{HORIZON} months, seed {SEED}, job_search months {SPELL_START}..{SPELL_START + SPELL_MONTHS - 1}"
    )
    table = engine.income_table.with_overrides(OVERRIDES)
    if not table.scales_spending():
        raise RuntimeError("scaled_spending did not turn on the consumption recipe")

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
    _print_spell_income(fans, table, alexia)
    _plot(fans)


def _check(fans, table, alexia) -> None:
    stay, masters, search = (item[2] for item in fans)
    student = STATE_INDEX["student"]
    employed = STATE_INDEX["employed"]
    job_search = STATE_INDEX["job_search"]
    spell = slice(SPELL_START, SPELL_START + SPELL_MONTHS)
    back = SPELL_START + SPELL_MONTHS
    if (stay.states != student).any():
        raise RuntimeError("stay-a-student left the student regime")
    if (stay.states == job_search).any():
        raise RuntimeError("stay-a-student entered job search")
    if not (masters.states[:, :GRADUATION_MONTH] == student).all():
        raise RuntimeError("master's path was not a student for the first 24 months")
    if (masters.states[:, GRADUATION_MONTH] == employed).mean() < 0.95:
        raise RuntimeError("master's path did not enter work at month 24")
    if not (search.states[:, 0] == job_search).all():
        raise RuntimeError("leave-studies path did not start in job search")
    for name, paths in (("Master's, then work", masters), ("Leave and look for work", search)):
        if not (paths.states[:, spell] == job_search).all():
            raise RuntimeError(f"{name} was not in job search for all 6 months")
        if not (paths.states[:, back] == employed).all():
            raise RuntimeError(f"{name} did not return to employed at month {back}")

    housing_m, _other_m = table.consumption_paths(masters.states, alexia, fans[1][4])
    if not np.allclose(housing_m[:, :GRADUATION_MONTH], alexia.rent):
        raise RuntimeError("student rent was not held at 900")
    signed = housing_m[0, GRADUATION_MONTH]
    if not np.allclose(housing_m[:, GRADUATION_MONTH:], signed):
        raise RuntimeError("the post-school lease was recomputed on the master's path")
    housing_s, _other_s = table.consumption_paths(search.states, alexia, fans[2][4])
    if not np.allclose(housing_s, housing_s[:, :1]):
        raise RuntimeError("the leave-studies lease was recomputed")
    if not np.allclose(housing_m[:, spell], signed):
        raise RuntimeError("master's rent moved during unemployment")
    if not np.allclose(housing_s[:, spell], housing_s[:, :1]):
        raise RuntimeError("leave-studies rent moved during unemployment")

    for name, paths, interventions in (
        ("Master's, then work", masters, fans[1][4]),
        ("Leave and look for work", search, fans[2][4]),
    ):
        expected = _expected_spell_income(paths.states, paths.income_chf, table, alexia)
        got = paths.income_chf[:, spell]
        if not np.allclose(got, expected, atol=1e-6):
            gap = np.max(np.abs(got - expected))
            raise RuntimeError(f"{name} spell income does not match the UI rule (max gap {gap:.4f})")


def _trailing_job_search(states: np.ndarray) -> np.ndarray:
    job_search = STATE_INDEX["job_search"]
    count = np.zeros(states.shape[0], dtype=float)
    alive = np.ones(states.shape[0], dtype=bool)
    for t in range(SPELL_START - 1, -1, -1):
        alive &= states[:, t] == job_search
        count += alive
    return count


def _last_wage(states: np.ndarray, income: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    employed = STATE_INDEX["employed"]
    part_time = STATE_INDEX["part_time"]
    worked = (states[:, :SPELL_START] == employed) | (states[:, :SPELL_START] == part_time)
    has_worked = worked.any(axis=1)
    offset = worked[:, ::-1].argmax(axis=1)
    last_idx = SPELL_START - 1 - offset
    wage = income[np.arange(states.shape[0]), last_idx]
    return has_worked, np.where(has_worked, wage, 0.0)


def _expected_spell_income(states, income, table, alexia) -> np.ndarray:
    """What income_paths pays on months 48..53, from the states and the last wage."""
    p = table.params
    n = states.shape[0]
    has_worked, last_wage = _last_wage(states, income)
    high = p["ui_replacement_with_dependents"]
    household = alexia.attributes.get("household_type")
    rate = np.full(n, high if household == "family" else p["ui_replacement_rate"])
    rate = np.where(last_wage <= p["ui_low_earnings_threshold_chf"], high, rate)
    benefit = rate * np.minimum(last_wage, p["ui_insured_earnings_cap_chf"])
    trailing = _trailing_job_search(states)
    out = np.empty((n, SPELL_MONTHS))
    for i, t in enumerate(range(SPELL_START, SPELL_START + SPELL_MONTHS)):
        age = alexia.age + t / 12.0
        max_m = p["ui_max_months_under_25"] if age < 25 else p["ui_max_months"]
        months_in = trailing + i
        on_ui = has_worked & (months_in < max_m)
        inc = np.where(on_ui, benefit, 0.0)
        expired = has_worked & ~on_ui
        inc = np.where(expired, p["job_search_floor_chf"], inc)
        never = ~has_worked
        wait = p["graduate_waiting_months"]
        span = p["graduate_benefit_months"]
        on_grad = never & (months_in >= wait) & (months_in < wait + span)
        grad_pay = p["graduate_benefit_under_25_chf"] if age < 25 else p["graduate_benefit_chf"]
        inc = np.where(on_grad, grad_pay, inc)
        out[:, i] = inc
    return out


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


def _print_spell_income(fans, table, alexia) -> None:
    p = table.params
    spell = slice(SPELL_START, SPELL_START + SPELL_MONTHS)
    age0 = alexia.age + SPELL_START / 12.0
    print(
        f"spell income, months {SPELL_START}..{SPELL_START + SPELL_MONTHS - 1}, "
        f"age {age0:.2f} (rate {p['ui_replacement_rate']:.0%} of last wage, "
        f"{p['ui_replacement_with_dependents']:.0%} at or below {p['ui_low_earnings_threshold_chf']:,.0f}, "
        f"cap {p['ui_insured_earnings_cap_chf']:,.0f}, "
        f"under-25 allowance {p['ui_max_months_under_25']} months, "
        f"household {alexia.attributes.get('household_type')})"
    )
    employed = STATE_INDEX["employed"]
    part_time = STATE_INDEX["part_time"]
    for name, _colour, paths, _cash, interventions in fans[1:]:
        has_worked, last_wage = _last_wage(paths.states, paths.income_chf)
        got = paths.income_chf[:, spell]
        high = p["ui_replacement_with_dependents"]
        household = alexia.attributes.get("household_type")
        rate = np.full(got.shape[0], high if household == "family" else p["ui_replacement_rate"])
        rate = np.where(last_wage <= p["ui_low_earnings_threshold_chf"], high, rate)
        fresh = (paths.states[:, SPELL_START - 1] == employed) | (paths.states[:, SPELL_START - 1] == part_time)
        if fresh.any() and not np.allclose(got[fresh], got[fresh, :1]):
            raise RuntimeError(f"{name} fresh spell did not pay a constant benefit")
        benefit = rate * np.minimum(last_wage, p["ui_insured_earnings_cap_chf"])
        if fresh.any() and not np.allclose(got[fresh, 0], benefit[fresh]):
            raise RuntimeError(f"{name} fresh spell is not the capped replacement of the last wage")
        housing, _other = table.consumption_paths(paths.states, alexia, interventions)
        rent = housing[:, spell]
        if not np.allclose(rent, rent[:, :1]):
            raise RuntimeError(f"{name} rent changed during the spell")
        mean_income = got.mean(axis=1)
        print(f"{name}")
        print(
            f"  all paths, mean income over the 6 months: "
            f"median {np.median(mean_income):,.2f}, "
            f"p10 {np.quantile(mean_income, 0.10):,.2f}, "
            f"p90 {np.quantile(mean_income, 0.90):,.2f}"
        )
        for i in range(SPELL_MONTHS):
            col = got[:, i]
            print(
                f"  month {SPELL_START + i}: median {np.median(col):,.2f}, "
                f"p10 {np.quantile(col, 0.10):,.2f}, p90 {np.quantile(col, 0.90):,.2f}"
            )
        paid = got[fresh, 0]
        wage = last_wage[fresh]
        used = rate[fresh]
        print(
            f"  fresh spell (working at month {SPELL_START - 1}): {fresh.mean():.3f} of paths; "
            f"last wage median {np.median(wage):,.2f} "
            f"(p10 {np.quantile(wage, 0.10):,.2f}, p90 {np.quantile(wage, 0.90):,.2f}); "
            f"income each of the 6 months median {np.median(paid):,.2f} "
            f"(p10 {np.quantile(paid, 0.10):,.2f}, p90 {np.quantile(paid, 0.90):,.2f}); "
            f"share at 70% {np.isclose(used, p['ui_replacement_rate']).mean():.3f}; "
            f"share at 80% {np.isclose(used, high).mean():.3f}; "
            f"share above the cap {(wage > p['ui_insured_earnings_cap_chf']).mean():.3f}; "
            f"rent {float(rent[0, 0]):,.2f}"
        )
        print(
            f"  not a fresh spell: {(~fresh).mean():.3f}; "
            f"never worked before month {SPELL_START}: {(~has_worked).mean():.3f}"
        )


def _plot(fans) -> None:
    months = np.arange(HORIZON)
    fig, ax = plt.subplots(figsize=(10.2, 5.8), dpi=140)
    ax.axvspan(
        SPELL_START, SPELL_START + SPELL_MONTHS, color="#bbbbbb", alpha=0.35,
        zorder=0, label="6-month job search",
    )
    for name, colour, _paths, cash, _iv in fans:
        low, mid, high = np.quantile(cash, [0.10, 0.50, 0.90], axis=0)
        ax.fill_between(months, low, high, color=colour, alpha=0.20, linewidth=0)
        ax.plot(months, mid, color=colour, linewidth=2.2, label=name)
    ax.axhline(0, color="#444444", linewidth=0.8, zorder=1)
    ax.axvline(0, color="#333333", linewidth=1.0, linestyle="--", zorder=1)
    ax.axvline(GRADUATION_MONTH, color="#333333", linewidth=1.0, linestyle="--", zorder=1)
    ax.axvline(SPELL_START, color="#333333", linewidth=1.0, linestyle="--", zorder=1)
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
        SPELL_START + 2, label_y, "month 48\n6 months out",
        ha="left", va="bottom", fontsize=9, color="#333333",
    )
    ax.set_xlim(-6, HORIZON - 1)
    ax.set_title("Alexia: 15 years, 6 months unemployed at year 4")
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
