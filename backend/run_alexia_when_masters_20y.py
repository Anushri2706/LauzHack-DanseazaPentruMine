# Run from backend/: /opt/anaconda3/bin/python run_alexia_when_masters_20y.py
"""20-year fans: when Alexia starts the master's. Expense-priors recipe.

Three paths, each pinned with force_state and has_masters turned on at graduation.
The lease signed that month, and the master's premium, use the wage in force then
(entry grown by the months she has worked, then times 1.10). A job before the
degree does not keep the lease that was signed on the entry salary.
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
STUDY = 24
OVERRIDES = {"scaled_spending": 1}
ARTIFACT = ROOT / "artifacts" / "alexia-20y-when-masters.png"
STORE_PNG = Path(
    "/Users/victorracu/Library/Application Support/Cursor/AgentStores/"
    "cursor_agent_stores/bc-e3d1c389-9d7a-4206-a8d2-334e3178188b/files/media/alexia-20y-when-masters.png"
)


def _interventions(work_months: int) -> list[Intervention]:
    grad = work_months + STUDY
    ivs = []
    if work_months:
        ivs.append(Intervention("force_state", 0, work_months, state="employed"))
    ivs.append(Intervention("force_state", work_months, STUDY, state="student"))
    ivs.append(Intervention("set_attribute", grad, attribute=("has_masters", True)))
    return ivs


DECISIONS = [
    ("Master's now", 0, _interventions(0), "#4C78A8"),
    ("Work 24 months, then master's", 24, _interventions(24), "#F58518"),
    ("Work 36 months, then master's", 36, _interventions(36), "#54A24B"),
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
        f"engine mode {engine.mode}, expense-priors recipe (scaled_spending=1), "
        f"{N_PATHS} paths, {HORIZON} months, seed {SEED}"
    )
    table = engine.income_table.with_overrides(OVERRIDES)

    fans = []
    for name, work, interventions, colour in DECISIONS:
        paths = engine.simulate(
            alexia,
            horizon_months=HORIZON,
            n_paths=N_PATHS,
            interventions=interventions,
            assumption_overrides=OVERRIDES,
            seed=SEED,
        )
        cash = run_cash(alexia, paths).cash
        fans.append((name, work, colour, paths, cash, interventions))
    wages = _check(fans, table, alexia)
    _print_table(fans, wages)
    _plot(fans)


def _wage_at_graduation(table, alexia, work_months: int) -> float:
    p = table.params
    salary = p["entry_salary_chf"]
    for t in range(work_months):
        if (t + 1) % 12 == 0:
            salary *= 1.0 + table._growth_rate(alexia.age + (t + 1) / 12.0)
    return salary * (1.0 + p["masters_premium_pct"])


def _check(fans, table, alexia) -> list[float]:
    student = STATE_INDEX["student"]
    employed = STATE_INDEX["employed"]
    p = table.params
    entry = p["entry_salary_chf"]
    wages = []
    for name, work, _colour, paths, _cash, ivs in fans:
        grad = work + STUDY
        if work and not (paths.states[:, :work] == employed).all():
            raise RuntimeError(f"{name}: the work spell was not pinned employed")
        if not (paths.states[:, work:grad] == student).all():
            raise RuntimeError(f"{name}: the master's was not pinned student")
        if not (paths.states[:, grad] == employed).all():
            raise RuntimeError(f"{name}: graduation month was not employed")
        if (paths.states[:, grad:] == student).any():
            raise RuntimeError(f"{name}: a path returned to study after the master's")
        housing, _other = table.consumption_paths(paths.states, alexia, ivs)
        if not np.allclose(housing[:, work:grad], alexia.rent):
            raise RuntimeError(f"{name}: student rent was not held at 900 during the master's")
        wage = _wage_at_graduation(table, alexia, work)
        signed = table.lease_signed_on[:, grad]
        if not np.allclose(signed, wage):
            raise RuntimeError(f"{name}: lease was not signed on the graduation wage {wage:.2f}")
        if not np.allclose(paths.income_chf[:, grad], wage):
            raise RuntimeError(f"{name}: the master's premium was not applied to the graduation wage")
        if work == 0 and not np.isclose(wage, entry * (1.0 + p["masters_premium_pct"])):
            raise RuntimeError(f"{name}: starting now should pay the master's offer")
        if work and not wage > entry * (1.0 + p["masters_premium_pct"]):
            raise RuntimeError(f"{name}: graduation wage ignored the months she worked")
        if work and abs(float(housing[0, grad]) - float(housing[0, 0])) < 1.0:
            raise RuntimeError(f"{name}: the post-master's lease reused the pre-work rent")
        if not np.allclose(housing[:, grad:], housing[:, grad:grad + 1]):
            raise RuntimeError(f"{name}: the post-school lease was recomputed")
        wages.append(float(signed[0]))
    return wages


def _print_table(fans, wages: list[float]) -> None:
    header = f"{'decision':<32} {'year':>6} {'median':>12} {'p10':>12} {'p90':>12}"
    print(header)
    print("-" * len(header))
    for (name, work, _colour, paths, cash, _iv), wage in zip(fans, wages):
        for year in (10, 20):
            closing = cash[:, year * 12 - 1]
            print(
                f"{name:<32} {year:>6} {np.median(closing):>12,.0f} "
                f"{np.quantile(closing, 0.10):>12,.0f} {np.quantile(closing, 0.90):>12,.0f}"
            )
        grad = work + STUDY
        print(
            f"  lease signed on {wage:,.2f} CHF at month {grad} "
            f"(income that month {paths.income_chf[0, grad]:,.2f})"
        )


def _plot(fans) -> None:
    months = np.arange(HORIZON)
    fig, ax = plt.subplots(figsize=(10.8, 6.0), dpi=140)
    for name, _work, colour, _paths, cash, _iv in fans:
        low, mid, high = np.quantile(cash, [0.10, 0.50, 0.90], axis=0)
        ax.fill_between(months, low, high, color=colour, alpha=0.20, linewidth=0)
        ax.plot(months, mid, color=colour, linewidth=2.2, label=name)
    ax.axhline(0, color="#444444", linewidth=0.8, zorder=0)
    lows = [float(np.quantile(cash, 0.10, axis=0).min()) for _n, _w, _c, _p, cash, _i in fans]
    highs = [float(np.quantile(cash, 0.90, axis=0).max()) for _n, _w, _c, _p, cash, _i in fans]
    ymin = min(0.0, min(lows))
    span = max(highs) - ymin
    base = ymin + 0.06 * span
    for grad, text, colour, dy in (
        (24, "month 24\nmaster's now", "#4C78A8", 0.00),
        (48, "month 48\nafter 2 years", "#F58518", 0.10),
        (60, "month 60\nafter 3 years", "#54A24B", 0.20),
    ):
        ax.axvline(grad, color=colour, linewidth=1.0, linestyle="--", alpha=0.85)
        ax.text(grad + 2, base + dy * span, text, ha="left", va="bottom", fontsize=8, color=colour)
    ax.set_xlim(-6, HORIZON - 1)
    ax.set_title("Alexia: 20 years, when she starts the master's")
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
