"""Run Alexia's three headline branches from the terminal: python -m app.cli"""
from .data.loader import build_persona
from .engine.metrics import check_goals, summarize
from .engine.simulate import initial_state, run_branch
from .engine.state import Decision

BRANCHES = {
    "Start working": [Decision(0, "start_work")],
    "Full-time master's": [Decision(0, "master_ft")],
    "Move abroad": [Decision(0, "move_abroad")],
    "Master's, then abroad at 22": [Decision(0, "master_ft"), Decision(24, "move_abroad")],
}


def chf(x: float) -> str:
    return f"{x:>11,.0f}".replace(",", "'")


def main() -> None:
    a = build_persona("P00001")
    start = initial_state(a.params, a.status, a.start_salary, a.start_living)
    print(f"{a.name}, {a.age}, {a.city} — synthetic profile. Simulation, not a prediction or advice.")
    print(f"Start: cash {a.opening_cash:,.0f} CHF. Horizon {int(a.params['horizon_months'])} months.\n")
    head = f"{'Branch':<30}{'NW at 30':>11}{'Cash':>11}{'Invest.':>11}{'Min cash':>11}{'<buffer':>9}{'<0':>5}  goals met"
    print(head + "\n" + "-" * len(head))
    for name, ds in BRANCHES.items():
        rows = run_branch(a.params, ds, start=start).rows
        m = summarize(rows)
        goals = check_goals(list(a.goals), rows)
        met = sum(g["status"] == "met" for g in goals)
        print(f"{name:<30}{chf(m['final_net_worth'])}{chf(m['final_cash'])}{chf(m['final_investments'])}"
              f"{chf(m['min_cash'])}{m['months_below_buffer']:>9}{m['months_negative_cash']:>5}  {met}/{len(goals)}")
    print("\nAssumptions: starter-kit incomes/costs, 5% return, 1% inflation, 2% raises, 50% of cash above "
          "a 3-month buffer invested. Edit them in app/engine/assumptions.py.")


if __name__ == "__main__":
    main()
