"""Live consistency checks for the in-app Validation screen (a subset of the pytest suite)."""
import json
import sys
from pathlib import Path

from .engine.assumptions import ALEXIA_DEFAULTS
from .engine.game import Action as A, Turn, decisions_to_turns, run_turns
from .engine.metrics import net_worth_bridge
from .engine.simulate import initial_state, run_branch, step_month
from .engine.state import Decision as D

# The realism layers (income tax, investment costs, lifestyle creep, overdraft interest) are off too: the starter kit has none.
PLAIN = dict(inflation_annual=0, raise_annual=0, return_annual=0, savings_rate=0,
             income_tax_rate=0, investment_costs_annual=0, lifestyle_creep=0,
             overdraft_rate=0)
KIT = Path(__file__).resolve().parents[2].parent / "challenge-instructions-and-data" / "starter-kit"


def _p(**kw):
    p = {k: float(v) for k, (v, _) in ALEXIA_DEFAULTS.items()}
    p.update({k: float(v) for k, v in kw.items()})
    return p


def _check(name, passed, detail):
    return {"name": name, "passed": bool(passed), "detail": detail}


def run_checks() -> list[dict]:
    out = []
    p = _p(**PLAIN, rent=900, living_studying=900)
    _, row = step_month(initial_state(p), p)
    out.append(_check("Cash identity", abs(row["cash"] - 7400) < 1e-9,
                      f"8'000 + 1'200 income − 1'800 expenses = {row['cash']:,.2f}"))

    p = _p(**PLAIN)
    a = run_branch({**p, "savings_rate": 1.0}, [D(0, "start_work")]).rows
    b = run_branch(p, [D(0, "start_work")]).rows
    moved = sum(r["invested"] for r in a)
    same = all(abs(x["net_worth"] - y["net_worth"]) < 1e-6 for x, y in zip(a, b))
    out.append(_check("Saving is a transfer", same,
                      f"CHF {moved:,.0f} moved cash → investments; net worth unchanged in all 120 months"))

    p = _p()
    full = [D(0, "master_ft"), D(12, "continue"), D(24, "move_abroad")]
    x, y = run_branch(p, full), run_branch(p, full[:2])
    out.append(_check("Rollback replays exactly", x.rows[:24] == y.rows[:24],
                      "Branch truncated to 2 decisions reproduces the first 24 months bit-for-bit"))

    rows = run_branch(p, full).rows
    br = net_worth_bridge(rows, 8000)
    total = (br["opening"] + br["income"] + br["income_tax"] + br["expenses"] + br["one_offs"] + br["interest"]
             + br["investment_return"])
    out.append(_check("Net worth bridge", abs(total - br["closing"]) < 1e-6,
                      f"opening + income − costs − one-offs − interest + returns = {total:,.2f} = closing"))

    # Town game: home purchase, habits, legacy equivalence, full bridge with housing.
    rich = _p(**PLAIN, opening_cash=300_000, rent=0, mortgage_rate=0, maintenance_annual=0, home_growth_annual=0)
    work = A("life_choice", "start_work")
    base = run_turns(rich, [Turn(0, (work,))]).rows
    own = run_turns(rich, [Turn(0, (work, A("buy_home", "studio")))]).rows
    gap = {round(a["net_worth"] - b["net_worth"], 6) for a, b in zip(base, own)}
    out.append(_check("Buying a home changes net worth only by fees", gap == {9000.0},
                      f"Studio 450'000: 90'000 down + 360'000 mortgage; net worth lower by exactly "
                      f"{', '.join(f'{g:,.0f}' for g in gap)} (2% fees) in every month"))

    hp = _p(**PLAIN)
    shop = [Turn(0, (work, A("set_habit", "shopping", 2)))]
    eat = [Turn(0, (work, A("set_habit", "eat_out", 2)))]
    pricier = {**hp, "habit_eat_out_2_cost": 500}
    iso = run_turns(hp, shop).rows == run_turns(pricier, shop).rows
    moved = run_turns(hp, eat).rows[-1]["cash"] - run_turns(pricier, eat).rows[-1]["cash"]
    out.append(_check("Habit price change is isolated", iso and abs(moved - 160 * 120) < 1e-6,
                      f"Eating-out price +160/month: shopping-only reality unchanged; eating-out reality "
                      f"{moved:,.0f} lower after 120 months (= 160 × 120)"))

    legacy = [D(0, "master_ft"), D(12, "continue"), D(24, "move_abroad")]
    same = run_branch(p, legacy).rows == [{k: v for k, v in r.items() if k != "regime"}
                                          for r in run_turns(p, decisions_to_turns(p, legacy)).rows]
    out.append(_check("Monthly turns reproduce the original engine", same,
                      "Master's then abroad, as decision points vs as monthly turns: identical in all 120 months"))

    gp = {**p, "opening_cash": 250_000}
    game = [Turn(0, (work, A("set_habit", "eat_out", 2), A("buy_item", "bike"))),
            Turn(6, (A("buy_home", "studio"),)), Turn(60, (A("sell_home"),))]
    grows = run_turns(gp, game).rows
    gb = net_worth_bridge(grows, 250_000)
    gt = gb["opening"] + sum(v for k, v in gb.items() if k not in ("opening", "closing"))
    out.append(_check("Net worth bridge with housing and habits", abs(gt - gb["closing"]) < 1e-6,
                      f"+ habits, purchases, mortgage interest, maintenance, fees, home growth = {gt:,.2f} = closing"))

    loan = run_branch(_p(**PLAIN, loan_term_months=24), [D(0, "start_work", loan_amount=10_000)]).rows
    out.append(_check("Loan never below zero", min(r["debt"] for r in loan) == 0 and all(r["debt"] >= 0 for r in loan),
                      f"10'000 loan fully repaid after 24 months; principal sum {sum(r['principal'] for r in loan):,.2f}"))

    od = _p(**{**PLAIN, "overdraft_rate": 0.12, "opening_cash": -10_000})
    _, orow = step_month(initial_state(od), od)
    out.append(_check("Negative cash is never free", abs(orow["overdraft_interest"] - 100) < 1e-9,
                      f"Start at −10'000: the credit line costs 12%/12 × 10'000 = "
                      f"{orow['overdraft_interest']:,.2f} interest this month (an expense)"))

    sys.path.insert(0, str(KIT))
    try:
        import financial_twin
        prof = json.loads((KIT / "sample_profile.json").read_text())
        scen = {s["name"]: s for s in json.loads((KIT / "scenario_config.json").read_text())["scenarios"]}
        kp = _p(**PLAIN, tuition_annual=2500, tuition_payments_per_year=1)
        diffs = []
        for name, ds in {"Start working": [D(0, "start_work")], "Master's degree": [D(0, "master_ft")],
                         "Move abroad": [D(0, "move_abroad")]}.items():
            kit = financial_twin.simulate(prof, scen[name])
            ours = run_branch(kp, ds).rows[-1]["cash"]
            diffs.append(f"{name}: {ours:,.0f} vs kit {kit.closing_cash_chf:,.0f}")
            ok = abs(ours - kit.closing_cash_chf) < 0.01
            if not ok:
                break
        out.append(_check("Matches official starter kit", ok, "; ".join(diffs)))
    except Exception as e:  # pragma: no cover
        out.append(_check("Matches official starter kit", False, f"Could not run: {e}"))
    return out
