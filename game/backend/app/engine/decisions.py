"""Decision catalogue, validity rules (action masking) and decision effects.

Decision points happen at month 0, every 12 months (birthdays) and when a degree finishes.
Only the choices returned by `options()` are valid at a given point; anything else is rejected.
"""
from dataclasses import replace

from .finance import annuity_payment
from .state import Decision, OneOff, State


class InvalidDecision(ValueError):
    pass


CHOICE_LABELS = {
    "continue": "Keep going as you are",
    "stay_student": "Stay a student one more year",
    "start_work": "Start working in Switzerland",
    "master_ft": "Full-time master's",
    "master_pt": "Part-time master's while working",
    "move_abroad": "Move abroad",
    "return_home": "Move back to Switzerland",
    "drop_out": "Leave the master's and start working",
}


def decision_reasons(state: State, t: int) -> list[str]:
    reasons = []
    if t == 0:
        reasons.append("start")
    elif t % 12 == 0:
        reasons.append("birthday")
    if state.program is not None and state.program_end == t:
        reasons.append("degree_finished")
    return reasons


def complete_programs(state: State, t: int) -> State:
    """A master's that ends this month is marked complete before the decision is taken."""
    if state.program is not None and state.program_end == t:
        state = replace(state, program=None, program_start=None, program_end=None, has_master=True)
        if state.retiree:  # back to retirement; the pension never stopped
            state = replace(state, status="retired", retiree=False)
    return state


def _cooldown_ok(state: State, t: int, p: dict) -> bool:
    return state.last_switch is None or t - state.last_switch >= p["switch_cooldown_months"]


def _work_terms(state: State, p: dict) -> tuple[float, float]:
    """(salary, other living costs) for a Swiss job, depending on whether a master's is done."""
    if state.has_master:
        return p["salary_master"], p["living_working_master"]
    if state.no_degree:  # game only: left school without a degree
        return p["salary_work"] * p["no_degree_wage_factor"], p["living_working"]
    return p["salary_work"], p["living_working"]


def _abroad_salary(state: State, p: dict) -> float:
    # Assumption: a master's raises pay abroad by the same ratio as in Switzerland.
    if state.has_master and p["salary_work"] > 0:
        return p["abroad_income"] * p["salary_master"] / p["salary_work"]
    return p["abroad_income"]


def _choice(cid: str, preview: str, default: bool = False) -> dict:
    return {"id": cid, "label": CHOICE_LABELS[cid], "preview": preview, "default": default}


def _money(x: float) -> str:
    return f"CHF {x:,.0f}".replace(",", "'")


def _tuition_total(p: dict) -> float:
    return p["tuition_annual"] * p["master_ft_months"] / 12.0


def options(state: State, t: int, p: dict) -> dict:
    """Valid choices at decision point t. `state` must already have programs completed.

    Returns {"choices": [...], "extras": {...}}. Exactly one choice is flagged default; it is
    what "keep going" means when a branch is projected forward without further input.
    """
    choices: list[dict] = []
    cool = _cooldown_ok(state, t, p)
    work_salary, work_living = _work_terms(state, p)
    work_preview = (f"Income {_money(work_salary)}/month, costs about "
                    f"{_money(p['rent'] + work_living)}/month (today's prices).")
    abroad_preview = (f"Income {_money(_abroad_salary(state, p))}/month, costs about "
                      f"{_money(p['abroad_rent'] + p['abroad_living'])}/month, "
                      f"one-off move {_money(p['relocation_cost'])}.")
    ft_preview = (f"{int(p['master_ft_months'])} months on {_money(p['student_income'])}/month, "
                  f"tuition {_money(_tuition_total(p))} in total; afterwards salary {_money(p['salary_master'])}/month.")
    pt_preview = (f"{int(p['master_pt_months'])} months on {_money(p['salary_work'] * p['part_time_fraction'])}/month, "
                  f"same total tuition {_money(_tuition_total(p))}; afterwards salary {_money(p['salary_master'])}/month.")

    if state.status == "studying" and state.program is not None:
        choices.append(_choice("continue", "Carry on with the master's.", default=True))
        if t - (state.program_start or 0) >= 12:
            after = "you go back to retirement." if state.retiree else work_preview
            choices.append(_choice("drop_out", "Remaining tuition is cancelled; " + after))
    elif state.status == "studying":
        can_stay = not state.has_master
        choices.append(_choice("start_work", work_preview, default=not can_stay))
        if can_stay:
            choices.append(_choice("stay_student", f"Income {_money(state.salary)}/month as now.", default=True))
        if not state.has_master:
            choices.append(_choice("master_ft", ft_preview))
            choices.append(_choice("master_pt", pt_preview))
        if cool:
            choices.append(_choice("move_abroad", abroad_preview))
    elif state.status == "working":
        choices.append(_choice("continue", "Stay in your job.", default=True))
        if not state.has_master and cool:
            choices.append(_choice("master_ft", ft_preview))
            choices.append(_choice("master_pt", pt_preview))
        if cool:
            choices.append(_choice("move_abroad", abroad_preview))
    elif state.status == "abroad":
        choices.append(_choice("continue", "Stay abroad.", default=True))
        if cool:
            choices.append(_choice("return_home", f"One-off move {_money(p['relocation_cost'])}; " + work_preview))
    elif state.status == "unemployed":
        choices.append(_choice("start_work", work_preview, default=True))
        choices.append(_choice("continue", "Keep looking for the right job.", default=False))
        if not state.has_master:
            choices.append(_choice("master_ft", ft_preview))
    elif state.status == "retired":
        choices.append(_choice("continue", "Keep your current routine.", default=True))
        if not state.has_master and cool:
            choices.append(_choice("master_ft", (
                f"{int(p['master_ft_months'])} months of study; your pension of {_money(state.salary)}/month "
                f"continues, tuition {_money(_tuition_total(p))} in total; afterwards you return to retirement.")))
    else:  # pragma: no cover - guarded by STATUSES
        raise InvalidDecision(f"Unknown status {state.status}")

    extras = {
        "savings_rate": {"current": state.savings_rate, "min": 0.0, "max": 1.0},
        "loan": ({"max": p["loan_max"], "rate": p["loan_rate"], "term_months": int(p["loan_term_months"])}
                 if state.debt <= 0 and p["loan_max"] > 0 else None),
        "repay_debt": ({"amount": round(state.debt, 2), "affordable": state.cash >= state.debt}
                       if state.debt > 0 else None),
    }
    return {"choices": choices, "extras": extras}


def default_decision(state: State, t: int, p: dict) -> Decision:
    opts = options(state, t, p)["choices"]
    return Decision(month=t, choice=next(c["id"] for c in opts if c["default"]))


def _tuition_schedule(t: int, duration_months: int, p: dict, total: float | None = None,
                      label: str = "Tuition instalment") -> tuple[OneOff, ...]:
    """Same total tuition for full- and part-time; instalments spread over the duration.
    `total` overrides the programme's total (a planned length other than the default, or an extension)."""
    per_year = int(round(p["tuition_payments_per_year"]))
    interval = 12 // per_year
    count = max(1, round(duration_months / 12 * per_year))
    amount = (_tuition_total(p) if total is None else total) / count
    return tuple(OneOff(t + k * interval, amount, f"{label} {k + 1}/{count}", "tuition")
                 for k in range(count))


def _start_job(state: State, t: int, status: str, salary: float, rent: float, living: float) -> State:
    return replace(state, status=status, salary=salary, salary_start=t, rent=rent, living=living,
                   program=None, program_start=None, program_end=None, last_switch=t)


def apply_decision(state: State, d: Decision, p: dict) -> State:
    """Apply a validated decision at the start of month d.month. Returns the new state."""
    t = d.month
    opts = options(state, t, p)
    valid = {c["id"] for c in opts["choices"]}
    if d.choice not in valid:
        raise InvalidDecision(f"'{d.choice}' is not available at month {t} (valid: {sorted(valid)})")

    s = state
    if d.choice in ("continue", "stay_student"):
        pass
    elif d.choice == "start_work":
        sal, living = _work_terms(s, p)
        s = _start_job(s, t, "working", sal, p["rent"], living)
    elif d.choice == "master_ft" and s.status == "retired":
        dur = int(p["master_ft_months"])  # pension, rent and living costs carry on unchanged
        s = replace(s, status="studying", retiree=True, program="master_ft", program_start=t,
                    program_end=t + dur, last_switch=t,
                    scheduled=s.scheduled + _tuition_schedule(t, dur, p))
    elif d.choice == "master_ft":
        dur = int(p["master_ft_months"])
        s = replace(s, status="studying", salary=p["student_income"], salary_start=t, rent=p["rent"],
                    living=p["living_studying"], program="master_ft", program_start=t,
                    program_end=t + dur, last_switch=t,
                    scheduled=s.scheduled + _tuition_schedule(t, dur, p))
    elif d.choice == "master_pt":
        dur = int(p["master_pt_months"])
        s = replace(s, status="studying", salary=p["salary_work"] * p["part_time_fraction"], salary_start=t,
                    rent=p["rent"], living=p["living_studying"], program="master_pt", program_start=t,
                    program_end=t + dur, last_switch=t,
                    scheduled=s.scheduled + _tuition_schedule(t, dur, p))
    elif d.choice == "move_abroad":
        s = _start_job(s, t, "abroad", _abroad_salary(s, p), p["abroad_rent"], p["abroad_living"])
        s = replace(s, scheduled=s.scheduled + (OneOff(t, p["relocation_cost"], "Relocation abroad", "relocation"),))
    elif d.choice == "return_home":
        sal, living = _work_terms(s, p)
        s = _start_job(s, t, "working", sal, p["rent"], living)
        s = replace(s, scheduled=s.scheduled + (OneOff(t, p["relocation_cost"], "Relocation home", "relocation"),))
    elif d.choice == "drop_out":
        kept = tuple(o for o in s.scheduled if not (o.kind == "tuition" and o.month >= t))
        s = replace(s, scheduled=kept)
        if s.retiree:
            s = replace(s, status="retired", retiree=False, program=None, program_start=None,
                        program_end=None, last_switch=t)
        else:
            s = _start_job(s, t, "working", p["salary_work"], p["rent"], p["living_working"])
    else:  # pragma: no cover
        raise InvalidDecision(d.choice)

    # Optional extras, applied after the main choice.
    if d.savings_rate is not None:
        if not 0.0 <= d.savings_rate <= 1.0:
            raise InvalidDecision("savings_rate must be between 0 and 1")
        s = replace(s, savings_rate=float(d.savings_rate))
    if d.loan_amount is not None:
        if opts["extras"]["loan"] is None:
            raise InvalidDecision("A new loan is only available when you have no debt")
        if not 0 < d.loan_amount <= p["loan_max"]:
            raise InvalidDecision(f"loan_amount must be in (0, {p['loan_max']}]")
        s = take_loan(s, d.loan_amount, p)
    if d.repay_debt:
        if s.debt <= 0:
            raise InvalidDecision("There is no debt to repay")
        if s.cash < s.debt:
            raise InvalidDecision("Not enough cash to repay the debt in full")
        s = repay_all(s)
    return s


def take_loan(s: State, amount: float, p: dict) -> State:
    """Loan disbursement: cash and debt rise by the same amount, so net worth is unchanged."""
    return replace(s, cash=s.cash + amount, debt=s.debt + amount, debt_rate=p["loan_rate"],
                   debt_payment=annuity_payment(amount, p["loan_rate"], int(p["loan_term_months"])))


def repay_all(s: State) -> State:
    """Repayment: cash and debt fall by the same amount, so net worth is unchanged."""
    return replace(s, cash=s.cash - s.debt, debt=0.0, debt_payment=0.0)
