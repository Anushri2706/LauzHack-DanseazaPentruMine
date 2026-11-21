# Run from backend/: /opt/anaconda3/bin/python run_alexia_20y_later.py
"""20-year fans for what Alexia does by year 5, after a master's started now.

All three paths start the master's at month 0 (student for 24 months, then
has_masters). The choice is at month 60. Baseline is the expense-priors recipe
(scaled_spending=1). The irregular-spend shock stays on; it is not part of the
2,393 student budget.

1. Stay in the first career. The master's wage keeps its experience raises.
2. At month 60, job search for 9 months, then employed at the bachelor entry
   salary grown to that age. The master's premium does not carry: the wage is
   entry_salary compounded on the engine's anniversaries since graduation
   (months 36, 48 and 60), not her realised wage divided by 1.10. Unemployment
   insurance during the nine months still uses the master's wage she just left.
3. At month 60, student for 42 months at income 1,200 and spend 2,393, then
   employed at an extra 1.10 on the master's wage she had at month 60. That
   PhD factor is phd_premium_pct in the assumptions CSV, marked as a guess.
   The 42 student months do not earn experience raises.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory

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
DECISION_MONTH = 60
SEARCH_MONTHS = 9
REHIRE_MONTH = DECISION_MONTH + SEARCH_MONTHS  # 69
PHD_MONTHS = 42
PHD_DONE = DECISION_MONTH + PHD_MONTHS  # 102
STIPEND_CHF = 1200.0
STUDENT_LIVING_CHF = 2393.0
OVERRIDES = {"scaled_spending": 1}
ARTIFACT = ROOT / "artifacts" / "alexia-20y-later.png"
STORE_PNG = Path(
    "/Users/victorracu/Library/Application Support/Cursor/AgentStores/"
    "cursor_agent_stores/bc-e3d1c389-9d7a-4206-a8d2-334e3178188b/files/media/alexia-20y-later.png"
)
MASTERS = [
    Intervention("force_state", 0, GRADUATION_MONTH, state="student"),
    Intervention("set_attribute", GRADUATION_MONTH, attribute=("has_masters", True)),
]
DECISIONS = [
    ("Stay in the first career", list(MASTERS), "#F58518"),
    (
        "Switch, bachelor wage",
        MASTERS + [
            Intervention("force_state", DECISION_MONTH, SEARCH_MONTHS, state="job_search"),
            Intervention("set_attribute", REHIRE_MONTH, attribute=("bachelor_age_wage", GRADUATION_MONTH)),
            Intervention("set_attribute", REHIRE_MONTH, attribute=("has_masters", False)),
        ],
        "#54A24B",
    ),
    (
        "PhD, then work",
        MASTERS + [
            Intervention("force_state", DECISION_MONTH, PHD_MONTHS, state="student"),
            Intervention("set_attribute", PHD_DONE, attribute=("has_phd", True)),
        ],
        "#4C78A8",
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
    if not table.scales_spending():
        raise RuntimeError("scaled_spending did not turn on the consumption recipe")
    if float(table.params["phd_premium_pct"]) != 0.1:
        raise RuntimeError("PhD prior is not the 1.10 guess")
    if float(table.params["student_income_chf"]) != STIPEND_CHF:
        raise RuntimeError("student income is not 1,200")
    if float(table.params["student_living_chf"]) != STUDENT_LIVING_CHF:
        raise RuntimeError("student spend is not 2,393")

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
    _plot(fans)


def _check(fans, table, alexia) -> None:
    stay, switch, phd = (item[2] for item in fans)
    student = STATE_INDEX["student"]
    employed = STATE_INDEX["employed"]
    job_search = STATE_INDEX["job_search"]
    if not (stay.states[:, :GRADUATION_MONTH] == student).all():
        raise RuntimeError("the master's was not a student spell")
    if not (stay.states[:, GRADUATION_MONTH] == employed).all():
        raise RuntimeError("graduation did not enter employment")
    if not np.array_equal(stay.states[:, :DECISION_MONTH], switch.states[:, :DECISION_MONTH]):
        raise RuntimeError("the switch path diverged before month 60")
    if not np.array_equal(stay.states[:, :DECISION_MONTH], phd.states[:, :DECISION_MONTH]):
        raise RuntimeError("the PhD path diverged before month 60")
    if not np.allclose(stay.income_chf[:, :DECISION_MONTH], switch.income_chf[:, :DECISION_MONTH]):
        raise RuntimeError("income diverged before the career switch")
    if not np.allclose(stay.income_chf[:, :DECISION_MONTH], phd.income_chf[:, :DECISION_MONTH]):
        raise RuntimeError("income diverged before the PhD")

    search = slice(DECISION_MONTH, REHIRE_MONTH)
    if not (switch.states[:, search] == job_search).all():
        raise RuntimeError("the switch was not 9 months of job search")
    if not (switch.states[:, REHIRE_MONTH] == employed).all():
        raise RuntimeError("the switch did not return to employed at month 69")
    bachelor = table.aged_bachelor_salary(alexia, GRADUATION_MONTH, REHIRE_MONTH)
    if not np.allclose(switch.income_chf[:, REHIRE_MONTH], bachelor):
        raise RuntimeError("the new job is not the bachelor entry salary grown to that age")
    on_master_wage = stay.states[:, DECISION_MONTH - 1] == employed
    if on_master_wage.mean() < 0.5:
        raise RuntimeError("too few paths were employed in the month before the decision")
    ui = table.params["ui_replacement_rate"] * stay.income_chf[on_master_wage, DECISION_MONTH - 1]
    if not np.allclose(switch.income_chf[on_master_wage, DECISION_MONTH], ui):
        raise RuntimeError("job-search pay is not 70% of the master's wage she left")

    phd_spell = slice(DECISION_MONTH, PHD_DONE)
    if not (phd.states[:, phd_spell] == student).all():
        raise RuntimeError("the PhD was not 42 student months")
    if not (phd.states[:, PHD_DONE] == employed).all():
        raise RuntimeError("the PhD did not return to employed at month 102")
    if not np.allclose(phd.income_chf[:, phd_spell], STIPEND_CHF):
        raise RuntimeError("PhD income was not 1,200")
    housing, other = table.consumption_paths(phd.states, alexia, fans[2][4])
    if not np.allclose(housing[:, phd_spell] + other[:, phd_spell], STUDENT_LIVING_CHF):
        raise RuntimeError("PhD spend was not 2,393")
    back = stay.states[:, DECISION_MONTH] == employed
    if not np.allclose(phd.income_chf[back, PHD_DONE], stay.income_chf[back, DECISION_MONTH] * 1.10):
        raise RuntimeError("the PhD wage is not an extra 1.10 on the master's wage at month 60")
    print(
        f"bachelor wage at month {REHIRE_MONTH}: {bachelor:,.2f} CHF "
        f"(entry {table.params['entry_salary_chf']:,.2f} grown since month {GRADUATION_MONTH})"
    )
    print(f"PhD wage is 1.10 times the month-{DECISION_MONTH} master's wage, phd_premium_pct is a guess")


def _print_table(fans) -> None:
    header = (
        f"{'decision':<28} {'year':>6} {'median':>12} {'p10':>12} {'p90':>12} "
        f"{'ever_neg':>10} {'neg_end':>10}"
    )
    print(header)
    print("-" * len(header))
    for name, _colour, _paths, cash, _iv in fans:
        ever = (cash < 0).any(axis=1).mean()
        still = (cash[:, -1] < 0).mean()
        for year in (10, 20):
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
    for month in (0, GRADUATION_MONTH, DECISION_MONTH):
        ax.axvline(month, color="#333333", linewidth=1.0, linestyle="--")
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(2, 0.16, "month 0\nstart", transform=trans, ha="left", va="bottom", fontsize=9, color="#333333")
    ax.text(
        GRADUATION_MONTH + 2, 0.16, "month 24\ngraduation",
        transform=trans, ha="left", va="bottom", fontsize=9, color="#333333",
    )
    ax.text(
        DECISION_MONTH + 2, 0.28, "month 60\nyear 5",
        transform=trans, ha="left", va="bottom", fontsize=9, color="#333333",
    )
    ax.set_xlim(-6, HORIZON - 1)
    ax.set_title("Alexia: year-5 choice after a master's, cash over 20 years")
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
