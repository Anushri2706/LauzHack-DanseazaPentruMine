"""Pure monthly simulation. Same inputs ⇒ same outputs (no hidden state, no clock, no I/O).

Monthly update order (see README for the equations):
  1. decision effects (already applied to the state before step_month is called)
  2. income       = salary × (1 − tax_rate − social_rate)      (+ raise every 12 months in a job)
  3. expenses     = (rent × flat_factor + living) × (1 + inflation) ** (t / 12)   (rent = 0 when owning)
     lifestyle    = Σ habit costs × (1 + inflation) ** (t / 12)
     purchases    = one-off items/trips due this month (consumption, never an asset)
     housing      mortgage interest = mortgage × r_m/12 and maintenance are expenses;
                  amortization reduces cash and mortgage equally (transfer); home value × (1+g)^(1/12)
  4. debt         interest = debt × r / 12 ; principal = payment − interest ; debt −= principal
  5. cash        += income − expenses − one_offs − debt_payment
     if cash < 0 and investments > 0: sell investments to cover (transfer, net worth unchanged)
  6. if cash > buffer: move savings_rate × (cash − buffer) to investments (transfer)
  7. investments ×= 1 + r_month
  8. outputs: net worth = cash + investments + home value − debt − mortgage, buffer months, flags, joy
"""
from dataclasses import dataclass, replace

from .assumptions import validate_params
from .decisions import (InvalidDecision, apply_decision, complete_programs, decision_reasons,
                        default_decision, options)
from .finance import annuity_payment, inflation_factor, monthly_compound_rate
from .state import Decision, State, education, fast_replace

CONSUMPTION_KINDS = ("purchase", "trip")


def habit_cost(habits: tuple, p: dict) -> float:
    """Monthly habit cost in month-0 prices (level 0 = nothing on top of base living costs)."""
    return sum(p[f"habit_{hid}_{lv}_cost"] for hid, lv in habits if lv > 0)


def month_joy(s: State, due: list, p: dict) -> float:
    """Joy points this month: status + habits + one-off purchases/trips. Not money."""
    if not p["joy_enabled"]:
        return 0.0
    joy = p[f"joy_{s.status}"] if f"joy_{s.status}" in p else 0.0
    joy += sum(p[f"habit_{hid}_{lv}_joy"] for hid, lv in s.habits if lv > 0)
    joy += sum(o.joy for o in due)
    if s.owner:
        joy += p["joy_home_owner"]
    if s.status == "working" and s.work_share < 1:
        joy += p["joy_reduced_hours"]
    return joy


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


def step_month(s: State, p: dict, ret_month: float | None = None, income_mult: float = 1.0,
               rent_mult: float = 1.0, income_extra: float = 0.0) -> tuple[State, dict]:
    """Advance one month from opening state `s` (decisions already applied). Returns (next, row).

    The optional shock arguments come from a Monte Carlo "world" (see montecarlo.py):
    ret_month overrides the expected monthly return, income_mult scales job income during an
    income-loss spell, rent_mult scales rent after a rent rise. Defaults = no shock.
    """
    t = s.t
    salary = s.salary
    # Raise every full year in the same job (working or abroad).
    if s.status in ("working", "abroad") and t > s.salary_start and (t - s.salary_start) % 12 == 0:
        salary = salary * (1.0 + p["raise_annual"])

    income = salary * (1.0 - p["tax_rate"] - p["social_rate"])
    if s.status in ("working", "abroad"):
        income *= income_mult
    income += income_extra * (1.0 - p["tax_rate"] - p["social_rate"])  # e.g. the 13th AHV pension
    infl = inflation_factor(p["inflation_annual"], t)
    rent_now = 0.0 if s.owner else s.rent * rent_mult * s.flat_factor
    expenses = (rent_now + s.living) * infl
    lifestyle = habit_cost(s.habits, p) * infl

    due = [o for o in s.scheduled if o.month == t]
    one_offs = sum(o.amount for o in due if o.kind not in CONSUMPTION_KINDS and o.kind != "home_fee")
    purchases = sum(o.amount for o in due if o.kind in CONSUMPTION_KINDS)
    home_fees = sum(o.amount for o in due if o.kind == "home_fee")

    # Housing (owner): interest and maintenance are expenses, amortization is a transfer.
    mortgage_interest = maintenance = amortization = appreciation = 0.0
    mortgage, home_value = s.mortgage, s.home_value
    if s.owner:
        mortgage_interest = mortgage * p["mortgage_rate"] / 12.0
        amortization = min(mortgage, s.home_price * p["amortization_annual"] / 12.0)
        maintenance = s.home_price * p["maintenance_annual"] / 12.0
        mortgage = mortgage - amortization
        if mortgage < 0.005:
            mortgage = 0.0

    interest = s.debt * s.debt_rate / 12.0
    payment = min(s.debt_payment, s.debt + interest) if s.debt > 0 else 0.0
    principal = payment - interest
    debt = s.debt - principal
    if debt < 0.005:  # paid off (guard against float residue); never negative
        debt = 0.0
    debt_payment = s.debt_payment if debt > 0 else 0.0

    cash = s.cash + income - expenses - one_offs - payment
    cash = cash - lifestyle - purchases - home_fees - mortgage_interest - maintenance - amortization
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

    r_m = monthly_compound_rate(p["return_annual"]) if ret_month is None else ret_month
    ret = investments * r_m
    investments += ret
    if s.owner:
        appreciation = home_value * monthly_compound_rate(p["home_growth_annual"])
        home_value += appreciation
    net_worth = cash + investments - debt + home_value - mortgage

    flags = []
    if cash < 0:
        flags.append("financing_needed")
    if cash < buffer_target:
        flags.append("below_buffer")

    nxt = fast_replace(s, t=t + 1, cash=cash, investments=investments, debt=debt, debt_payment=debt_payment,
                  salary=salary, scheduled=tuple(o for o in s.scheduled if o.month > t),
                  mortgage=mortgage, home_value=home_value)
    joy = month_joy(s, due, p)
    row = {
        "t": t, "age": round(s.age, 4), "status": s.status, "program": s.program, "education": education(s),
        "income": income, "expenses": expenses, "one_offs": one_offs,
        "one_off_labels": [o.label for o in due],
        "interest": interest, "principal": principal, "debt_payment": payment,
        "invested": invested, "sold_investments": sold, "investment_return": ret,
        "cash": cash, "investments": investments, "debt": debt,
        "net_worth": net_worth,
        "lifestyle": lifestyle, "purchases": purchases, "home_fees": home_fees,
        "purchase_labels": [o.label for o in due if o.kind in CONSUMPTION_KINDS],
        "mortgage_interest": mortgage_interest, "maintenance": maintenance, "amortization": amortization,
        "home_appreciation": appreciation, "home_value": home_value, "mortgage": mortgage,
        "rent": rent_now * infl, "joy": joy,
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


def run_branch(p: dict, decisions: list[Decision], start: State | None = None,
               world: dict | None = None) -> BranchRun:
    """Replay a branch from month 0. Pure: same inputs (incl. world) ⇒ identical output.

    `world` (optional) holds per-month shock arrays: ret, income_mult, rent_mult.
    Shocks never change which decisions are valid, so every world replays the same choices.
    """
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
        if world is None:
            s, row = step_month(s, p)
        else:
            s, row = step_month(s, p, float(world["ret"][t]), float(world["income_mult"][t]),
                                float(world["rent_mult"][t]))
        rows.append(row)
    if by_month:
        raise InvalidDecision(f"Decisions outside the horizon or not at decision points: {sorted(by_month)}")
    return BranchRun(rows=rows, points=points, states_at_points=states,
                     next_decision_month=next_decision, final_state=s)
