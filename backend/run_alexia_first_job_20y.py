# Run from backend/: /opt/anaconda3/bin/python run_alexia_first_job_20y.py
"""20-year fans for the first job after a master's, expense-priors recipe.

Alexia starts the master's now: student for 24 months, then has_masters. From
month 24 the three paths are employed immediately, part-time for 24 months at
the existing part_time_fraction (0.60) then full-time, and job search for 6
months then employed. scaled_spending stays 1.

A part-time phase is not in EXIT_STATE, so it would return to the regime
before the window (student). The full-time step is therefore its own one-month
employed pin, the same shape as the student phase's exit, and the chain runs
after that. Job search is in EXIT_STATE, so the 6-month pin exits to employed
on its own.

She has never worked. Study does not set has_worked, and she does not start
unemployed, so the spell is the graduate-waiting branch in income_paths, not
the 70% wage replacement. The six pinned months are counters 0..5, all inside
the 5.5-month wait, so the pay is 0. The flat benefit would start on the next
search month.
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
from twin.regime_engine import _schedule
from twin.types import STATE_INDEX, Intervention

HORIZON = 240
N_PATHS = 4000
SEED = 0
GRADUATION_MONTH = 24
PART_TIME_MONTHS = 24
SEARCH_MONTHS = 6
OVERRIDES = {"scaled_spending": 1}
ARTIFACT = ROOT / "artifacts" / "alexia-20y-first-job.png"
STORE_PNG = Path(
    "/Users/victorracu/Library/Application Support/Cursor/AgentStores/"
    "cursor_agent_stores/bc-e3d1c389-9d7a-4206-a8d2-334e3178188b/files/media/alexia-20y-first-job.png"
)
FULL_TIME_MONTH = GRADUATION_MONTH + PART_TIME_MONTHS
HIRED_MONTH = GRADUATION_MONTH + SEARCH_MONTHS

# Shared master's. The attribute turns the entry salary into the master's level
# in the same month the phase ends.
MASTERS = [
    Intervention("force_state", 0, GRADUATION_MONTH, state="student"),
    Intervention("set_attribute", GRADUATION_MONTH, attribute=("has_masters", True)),
]
DECISIONS = [
    (
        "Employed immediately",
        list(MASTERS),
        "#F58518",
    ),
    (
        "Part-time, then full-time",
        [
            *MASTERS,
            Intervention("force_state", GRADUATION_MONTH, PART_TIME_MONTHS, state="part_time"),
            Intervention("force_state", FULL_TIME_MONTH, None, state="employed"),
        ],
        "#4C78A8",
    ),
    (
        "Search 6 months, then work",
        [
            *MASTERS,
            Intervention("force_state", GRADUATION_MONTH, SEARCH_MONTHS, state="job_search"),
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
        f"engine mode {engine.mode}, scaled_spending=1, {N_PATHS} paths, "
        f"{HORIZON} months, seed {SEED}"
    )
    table = engine.income_table.with_overrides(OVERRIDES)
    if not table.scales_spending():
        raise RuntimeError("scaled_spending did not turn on the expense-priors recipe")
    fraction = float(table.params["part_time_fraction"])
    if fraction != 0.60:
        raise RuntimeError(f"part_time_fraction is {fraction}, expected 0.60")

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
        if paths.lump_chf is None or not np.any(paths.lump_chf):
            raise RuntimeError(f"{name} did not sample a yearly lump")
        cash = run_cash(alexia, paths).cash
        fans.append((name, colour, paths, cash, interventions))
    _check(fans, table, alexia)
    _print_table(fans)
    _print_pay(fans, table, alexia)
    _plot(fans)


def _check(fans, table, alexia) -> None:
    student = STATE_INDEX["student"]
    employed = STATE_INDEX["employed"]
    part_time = STATE_INDEX["part_time"]
    job_search = STATE_INDEX["job_search"]
    offer = _masters_salary(table)
    fraction = float(table.params["part_time_fraction"])
    for name, _colour, paths, _cash, interventions in fans:
        forced, prev_from = _schedule(paths.horizon, interventions)
        if not (paths.states[:, :GRADUATION_MONTH] == student).all():
            raise RuntimeError(f"{name} was not a student for the first 24 months")
        if not np.allclose(paths.income_chf[:, :GRADUATION_MONTH], alexia.ref_income):
            raise RuntimeError(f"{name} did not keep the student stipend through the master's")
        pinned = forced >= 0
        for t in range(paths.horizon):
            if not pinned[t]:
                continue
            if not (paths.states[:, t] == forced[t]).all():
                raise RuntimeError(f"{name} left the pinned state at month {t}")
        if FULL_TIME_MONTH in prev_from or HIRED_MONTH in prev_from:
            raise RuntimeError(f"{name} took a return-to-previous exit inside a later pin")

    immediate, part, search = fans
    if not (immediate[2].states[:, GRADUATION_MONTH] == employed).all():
        raise RuntimeError("employed-immediately was not employed at month 24")
    if not np.allclose(immediate[2].income_chf[:, GRADUATION_MONTH], offer):
        raise RuntimeError("employed-immediately month 24 is not the master's salary")

    part_states = part[2].states
    part_window = slice(GRADUATION_MONTH, FULL_TIME_MONTH)
    if not (part_states[:, part_window] == part_time).all():
        raise RuntimeError("part-time path was not part-time for months 24..47")
    if not (part_states[:, FULL_TIME_MONTH] == employed).all():
        raise RuntimeError("part-time path did not enter full-time at month 48")
    if (part_states[:, FULL_TIME_MONTH] == student).any():
        raise RuntimeError("part-time path returned to study at month 48")
    part_pay = part[2].income_chf[:, GRADUATION_MONTH:GRADUATION_MONTH + 6]
    if not np.allclose(part_pay, offer * fraction):
        raise RuntimeError("part-time pay over the first six months is not 0.60 of the master's salary")

    search_states = search[2].states
    spell = slice(GRADUATION_MONTH, HIRED_MONTH)
    if not (search_states[:, spell] == job_search).all():
        raise RuntimeError("search path was not in job search for months 24..29")
    if not (search_states[:, HIRED_MONTH] == employed).all():
        raise RuntimeError("search path did not enter employed at month 30")
    worked = (search_states[:, :GRADUATION_MONTH] == employed) | (search_states[:, :GRADUATION_MONTH] == part_time)
    if worked.any():
        raise RuntimeError("search path worked before graduation, so the graduate rule would not apply")
    if alexia.initial_state in ("employed", "part_time"):
        raise RuntimeError("profile already counts as having worked")
    if alexia.initial_state == "job_search":
        raise RuntimeError("profile starts unemployed, so anchor income would replace the graduate rule")
    if not np.allclose(search[2].income_chf[:, spell], 0.0):
        raise RuntimeError("search path was paid during the six-month wait")
    rules = _waiting_rules(table, alexia)
    if rules != ["graduate_waiting"] * SEARCH_MONTHS:
        raise RuntimeError(f"expected six months of graduate waiting, got {rules}")
    if not np.allclose(search[2].income_chf[:, HIRED_MONTH], offer):
        raise RuntimeError("month 30 on the search path is not the master's salary")


def _masters_salary(table) -> float:
    p = table.params
    return float(p["entry_salary_chf"] * (1.0 + p["masters_premium_pct"]))


def _waiting_rules(table, alexia) -> list[str]:
    """Which never-worked branch income_paths uses on search months 24..29.

    The counter at the start of a month is the number of job-search months
    already completed. Waiting is months_in_js < graduate_waiting_months.
    """
    p = table.params
    rules = []
    for i in range(SEARCH_MONTHS):
        age = alexia.age + (GRADUATION_MONTH + i) / 12.0
        months_in = float(i)
        wait = p["graduate_waiting_months"]
        span = p["graduate_benefit_months"]
        if months_in < wait:
            rules.append("graduate_waiting")
        elif months_in < wait + span:
            rules.append("graduate_benefit_under_25" if age < 25 else "graduate_benefit")
        else:
            rules.append("job_search_floor")
    return rules


def _print_table(fans) -> None:
    header = f"{'decision':<32} {'year':>6} {'median':>12} {'p10':>12} {'p90':>12} {'ever_neg':>10} {'neg_end':>10}"
    print(header)
    print("-" * len(header))
    for name, _colour, _paths, cash, _iv in fans:
        ever = (cash < 0).any(axis=1).mean()
        still = (cash[:, -1] < 0).mean()
        for year in (10, 20):
            closing = cash[:, year * 12 - 1]
            print(
                f"{name:<32} {year:>6} {np.median(closing):>12,.0f} "
                f"{np.quantile(closing, 0.10):>12,.0f} {np.quantile(closing, 0.90):>12,.0f} "
                f"{ever:>10.3f} {still:>10.3f}"
            )


def _print_pay(fans, table, alexia) -> None:
    p = table.params
    offer = _masters_salary(table)
    fraction = float(p["part_time_fraction"])
    part_pay = offer * fraction
    print(
        f"months 0-5 are the master's on every path, stipend {alexia.ref_income:,.2f}. "
        f"The six months below are months {GRADUATION_MONTH}-{GRADUATION_MONTH + 5}, "
        f"the first six of each path."
    )
    print(
        f"master's full-time salary {offer:,.2f} = {p['entry_salary_chf']:,.2f} * "
        f"{1 + p['masters_premium_pct']:.2f}. Part-time is that times {fraction:.2f} = {part_pay:,.2f}."
    )
    employed = STATE_INDEX["employed"]
    part_time = STATE_INDEX["part_time"]
    job_search = STATE_INDEX["job_search"]
    for name, _colour, paths, _cash, _iv in fans:
        print(name)
        block = paths.income_chf[:, GRADUATION_MONTH:GRADUATION_MONTH + 6]
        for i in range(6):
            month = GRADUATION_MONTH + i
            col = block[:, i]
            share_e = (paths.states[:, month] == employed).mean()
            share_p = (paths.states[:, month] == part_time).mean()
            share_j = (paths.states[:, month] == job_search).mean()
            if np.ptp(col) < 0.01:
                paid = f"income {col[0]:,.2f} on every path"
            else:
                paid = (
                    f"income median {np.median(col):,.2f}, "
                    f"p10 {np.quantile(col, 0.10):,.2f}, p90 {np.quantile(col, 0.90):,.2f}"
                )
            print(
                f"  month {month}: {paid}; "
                f"employed {share_e:.3f}, part-time {share_p:.3f}, job_search {share_j:.3f}"
            )
    search = fans[2][2]
    age30 = alexia.age + HIRED_MONTH / 12.0
    benefit = p["graduate_benefit_under_25_chf"] if age30 < 25 else p["graduate_benefit_chf"]
    benefit_name = "graduate_benefit_under_25_chf" if age30 < 25 else "graduate_benefit_chf"
    print(
        "search rule: graduate_waiting on all six months. "
        "income_paths pays this from `never = job_search & ~has_worked` when the profile "
        "did not start in job_search. has_worked is false because the initial state is "
        f"{alexia.initial_state} and no employed or part-time month comes first. "
        f"Spell counters are 0,1,2,3,4,5, all below graduate_waiting_months "
        f"{p['graduate_waiting_months']}, so the pay is 0. "
        f"The flat benefit ({benefit_name} {benefit:,.0f}, age at month {HIRED_MONTH} is {age30:.2f}) "
        f"would start once the counter reaches {p['graduate_waiting_months']}, the seventh search month. "
        f"Month {HIRED_MONTH} is employed at {search.income_chf[0, HIRED_MONTH]:,.2f} instead, "
        "so the benefit is not paid."
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
        GRADUATION_MONTH + 2, label_y, "month 24\nfirst job",
        ha="left", va="bottom", fontsize=9, color="#333333",
    )
    ax.set_xlim(-6, HORIZON - 1)
    ax.set_title("Alexia: first job after a master's, 20 years")
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
