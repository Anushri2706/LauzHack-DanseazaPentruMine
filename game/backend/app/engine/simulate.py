"""Pure monthly simulation. Same inputs ⇒ same outputs (no hidden state, no clock, no I/O).

Monthly update order (see README for the equations):
  1. decision effects (already applied to the state before step_month is called)
  2. income       = salary × (1 − tax_rate − social_rate)      (+ raise every 12 months in a job)
  3. expenses     = (rent + living) × (1 + inflation) ** (t / 12)
  4. debt         interest = debt × r / 12 ; principal = payment − interest ; debt −= principal
  5. cash        += income − expenses − one_offs − debt_payment
     if cash < 0 and investments > 0: sell investments to cover (transfer, net worth unchanged)
  6. if cash > buffer: move savings_rate × (cash − buffer) to investments (transfer)
  7. investments ×= 1 + r_month
  8. outputs: net worth = cash + investments − debt, buffer months, flags
"""
from dataclasses import dataclass, replace

from .assumptions import validate_params
from .decisions import (InvalidDecision, apply_decision, complete_programs, decision_reasons,
                        default_decision, options)
from .finance import annuity_payment, inflation_factor, monthly_compound_rate
from .state import Decision, State


def initial_state(p: dict, status: str = "studying", salary: float | None = None,
                  living: float | None = None) -> State:
    """Opening state from assumptions. Existing debt is amortized over loan_term_months."""
    debt = float(p["opening_debt"])
    if salary is None:
        salary = p["student_income"] if status == "studying" else p["salary_work"]
    if living is None:
        living = p["living_studying"] if status == "studying" else p["living_working"]
    return State(
        t=0, start_age=float(p["start_age"]), cash=float(p["opening_cash"]),
        investments=float(p["opening_investments"]), debt=debt, debt_rate=float(p["opening_debt_rate"]),
        debt_payment=annuity_payment(debt, p["opening_debt_rate"], int(p["loan_term_months"])),
        status=status, salary=float(salary), salary_start=0,
        rent=float(p["rent"]), living=float(living),
        savings_rate=float(p["savings_rate"]),
    )


def step_month(s: State, p: dict) -> tuple[State, dict]:
    """Advance one month from opening state `s` (decisions already applied). Returns (next, row)."""
    t = s.t
    salary = s.salary
    # Raise every full year in the same job (working or abroad).
    if s.status in ("working", "abroad") and t > s.salary_start and (t - s.salary_start) % 12 == 0:
        salary = salary * (1.0 + p["raise_annual"])

    income = salary * (1.0 - p["tax_rate"] - p["social_rate"])
    expenses = (s.rent + s.living) * inflation_factor(p["inflation_annual"], t)

    due = [o for o in s.scheduled if o.month == t]
    one_offs = sum(o.amount for o in due)

    interest = s.debt * s.debt_rate / 12.0
    payment = min(s.debt_payment, s.debt + interest) if s.debt > 0 else 0.0
    principal = payment - interest
    debt = s.debt - principal
    if debt < 0.005:  # paid off (guard against float residue); never negative
        debt = 0.0
    debt_payment = s.debt_payment if debt > 0 else 0.0

    cash = s.cash + income - expenses - one_offs - payment
    investments = s.investments

    sold = 0.0
    if cash < 0 and investments > 0:
        sold = min(investments, -cash)
        cash += sold
        investments -= sold

    buffer_target = p["buffer_months"] * expenses
    invested = 0.0
    if cash > buffer_target and s.savings_rate > 0:
        invested = (cash - buffer_target) * s.savings_rate
        cash -= invested
        investments += invested

    ret = investments * monthly_compound_rate(p["return_annual"])
    investments += ret

    flags = []
    if cash < 0:
        flags.append("financing_needed")
    if cash < buffer_target:
        flags.append("below_buffer")

    nxt = replace(s, t=t + 1, cash=cash, investments=investments, debt=debt, debt_payment=debt_payment,
                  salary=salary, scheduled=tuple(o for o in s.scheduled if o.month > t))
    row = {
        "t": t, "age": round(s.age, 4), "status": s.status, "program": s.program,
        "income": income, "expenses": expenses, "one_offs": one_offs,
        "one_off_labels": [o.label for o in due],
        "interest": interest, "principal": principal, "debt_payment": payment,
        "invested": invested, "sold_investments": sold, "investment_return": ret,
        "cash": cash, "investments": investments, "debt": debt,
        "net_worth": cash + investments - debt,
        "buffer_target": buffer_target,
        "buffer_months": (cash / expenses) if expenses > 0 else None,
        "flags": flags,
    }
    return nxt, row


@dataclass(frozen=True)
class BranchRun:
    rows: list[dict]                 # full horizon; points after the player's last choice use defaults
    points: list[dict]               # every decision point: month, age, reasons, options, chosen, by
    states_at_points: dict[int, State]
    next_decision_month: int | None  # first point awaiting the player (None = branch complete)
    final_state: State


def run_branch(p: dict, decisions: list[Decision], start: State | None = None) -> BranchRun:
    """Replay a branch from month 0. Deterministic by construction (no randomness yet)."""
    validate_params(p)
    horizon = int(p["horizon_months"])
    months = [d.month for d in decisions]
    if months != sorted(months) or len(set(months)) != len(months):
        raise InvalidDecision("Decisions must be in strictly increasing month order")
    by_month = {d.month: d for d in decisions}
    last_player_month = max(months) if months else None

    s = start or initial_state(p)
    rows, points, states = [], [], {}
    next_decision = None
    for t in range(horizon):
        reasons = decision_reasons(s, t)
        if reasons:
            s = complete_programs(s, t)
            states[t] = s
            opts = options(s, t, p)
            d = by_month.pop(t, None)
            by = "player"
            if d is None:
                by = "default"
                d = default_decision(s, t, p)
                if next_decision is None and (last_player_month is None or t > last_player_month):
                    next_decision = t
            s = apply_decision(s, d, p)
            points.append({"month": t, "age": round(states[t].age, 2), "reasons": reasons,
                           "options": opts, "chosen": d.choice, "by": by,
                           "savings_rate": d.savings_rate, "loan_amount": d.loan_amount,
                           "repay_debt": d.repay_debt})
        elif t in by_month:
            raise InvalidDecision(f"Month {t} is not a decision point")
        s, row = step_month(s, p)
        rows.append(row)
    if by_month:
        raise InvalidDecision(f"Decisions outside the horizon or not at decision points: {sorted(by_month)}")
    return BranchRun(rows=rows, points=points, states_at_points=states,
                     next_decision_month=next_decision, final_state=s)
