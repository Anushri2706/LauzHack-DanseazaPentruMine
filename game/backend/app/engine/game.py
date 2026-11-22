"""Monthly turns for the town game. Pure and deterministic, like simulate.py.

A reality is (persona, assumptions, seed, world, turns). A turn is the list of actions the player
queued in one month; months without a turn mean "nothing changed". The engine replays from month 0,
so rollback is exact. Order inside a month t:

  1. a master's that ends at t is completed (and a letter is sent)
  2. the player's actions for month t are validated and applied, in order
  3. a graduate who chose no big move this month starts a job at the post-master salary
     (the same default as the original decision-point engine)
  4. step_month(t) with the shocks of the player's seeded world
  5. letters for what happened (world events that actually bite, cash below zero, loan paid off)

Validity (action masking) is checked against the player's own world ("strict"). Monte Carlo runs of
other worlds replay the SAME choices with affordability checks off: a choice that would not be
affordable in a poorer world still happens and shows up as negative cash ("financing needed").
"""
from dataclasses import dataclass, field, replace

import numpy as np

from .assumptions import validate_params
from .catalogue import (FLATS, HABITS, HOMES, ITEMS, flat_factor_key, habit_cost_key, habit_joy_key,
                        home_price_key, item_joy_key, item_price_key)
from .decisions import (InvalidDecision, _start_job, _work_terms, apply_decision, complete_programs, options,
                        repay_all, take_loan)
from .finance import annuity_payment, inflation_factor
from .narrative import chf
from .simulate import initial_state, run_branch, step_month
from .state import Decision, OneOff, State, fast_replace


class InvalidAction(InvalidDecision):
    pass


ACTION_TYPES = ("life_choice", "set_habit", "buy_item", "change_flat", "buy_home", "sell_home",
                "take_loan", "repay_debt", "set_savings_rate", "ask_raise")
LEGACY_LIFE = ("start_work", "master_ft", "master_pt", "move_abroad", "return_home", "drop_out")
LIFE_LABELS = {
    "start_work": "Start working in Switzerland", "master_ft": "Full-time master's",
    "master_pt": "Part-time master's while working", "move_abroad": "Move abroad",
    "return_home": "Move back to Switzerland", "drop_out": "Leave the master's and start working",
    "quit_job": "Quit your job", "reduce_hours": "Reduce to 80%", "full_hours": "Back to full time",
    "graduate": "🎓 Get your degree", "quit_school": "🚪 Drop out without a degree",
}
LIFE_PLACE = {"start_work": "office", "quit_job": "office", "reduce_hours": "office", "full_hours": "office",
              "master_ft": "university", "master_pt": "university", "drop_out": "university",
              "graduate": "university", "quit_school": "university",
              "move_abroad": "station", "return_home": "station"}
PLACES = ("home", "university", "office", "cafe", "shop", "bank", "realestate", "station")
SAVINGS_PRESETS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class Action:
    type: str
    id: str | None = None
    value: float | None = None


@dataclass(frozen=True)
class Turn:
    month: int
    actions: tuple[Action, ...] = ()


@dataclass
class GameRun:
    rows: list[dict]
    letters: list[dict]
    world_events: list[dict]
    final_state: State
    opening: dict[int, State] = field(default_factory=dict)  # state at the start of t (before anything)
    applied: dict[int, State] = field(default_factory=dict)  # state at t after the player's actions


def _letter(t: int, kind: str, icon: str, title: str, text: str) -> dict:
    return {"month": t, "kind": kind, "icon": icon, "title": title, "text": text}


# ----------------------------------------------------------------------------- validity rules
def _cooldown_left(s: State, t: int, p: dict) -> int:
    if s.last_switch is None:
        return 0
    return max(0, int(p["switch_cooldown_months"]) - (t - s.last_switch))


def _cooldown_text(s: State, t: int, p: dict) -> str:
    return (f"You need {int(p['switch_cooldown_months'])} months since your last big move "
            f"({_cooldown_left(s, t, p)} to go).")


def life_reason(s: State, t: int, p: dict, cid: str) -> str | None:
    """None if life choice `cid` is allowed now, otherwise the reason (shown greyed out in the UI)."""
    if cid == "quit_job":
        return None if s.status == "working" else "You don't have a job here to quit."
    if cid == "reduce_hours":
        if s.status != "working":
            return "You need a job in Switzerland first."
        return None if s.work_share >= 1 else "You already work reduced hours."
    if cid == "full_hours":
        if s.status != "working":
            return "You need a job in Switzerland first."
        return None if s.work_share < 1 else "You already work full time."
    if cid in ("graduate", "quit_school"):
        if s.status != "studying":
            return "You're not enrolled in school."
        if cid == "graduate" and s.program is not None:
            return (f"Your master's finishes by itself in {max(0, (s.program_end or t) - t)} months: "
                    "keep studying, or drop out.")
        return None
    if cid in ("master_ft", "master_pt") and s.no_degree:
        return "You left school without a degree: a master's needs a bachelor's first."
    if cid not in LEGACY_LIFE:
        return f"Unknown life choice '{cid}'."
    valid = {c["id"] for c in options(s, t, p)["choices"]}
    if cid == "move_abroad" and s.owner and cid in valid:
        return "Sell your home first (Real-estate agency)."
    if cid in valid:
        return None
    enrolled = s.status == "studying" and s.program is not None
    if cid == "start_work":
        if s.status == "working":
            return "You already have a job."
        if s.status == "abroad":
            return "You work abroad: move back first (Train station)."
        if enrolled:
            return "Finish or leave your master's first (University)."
    if cid in ("master_ft", "master_pt"):
        if s.has_master:
            return "You already have a master's."
        if enrolled:
            return "You're already enrolled."
        if s.status == "abroad":
            return "Move back to Switzerland first."
        if s.status == "retired" and cid == "master_pt":
            return "A part-time master's runs alongside a job; the full-time one fits around retirement."
        if s.status == "unemployed" and cid == "master_pt":
            return "A part-time master's needs a job: start working first."
        if _cooldown_left(s, t, p):
            return _cooldown_text(s, t, p)
    if cid == "drop_out":
        if not enrolled:
            return "You're not enrolled in a master's."
        return f"Possible after 12 months of study ({12 - (t - (s.program_start or 0))} to go)."
    if cid == "move_abroad":
        if s.status == "abroad":
            return "You already live abroad."
        if enrolled:
            return "Finish or leave your master's first."
        if s.status == "unemployed":
            return "Find a job first."
        if _cooldown_left(s, t, p):
            return _cooldown_text(s, t, p)
    if cid == "return_home":
        if s.status != "abroad":
            return "You already live in Switzerland."
        if _cooldown_left(s, t, p):
            return _cooldown_text(s, t, p)
    if s.status == "retired":
        return "Not part of a retiree's choices in this game."
    return "Not available right now."


def _home_price(hid: str, t: int, p: dict) -> float:
    """Market price at month t: the month-0 price grown at the stated home growth rate."""
    return p[home_price_key(hid)] * (1 + p["home_growth_annual"]) ** (t / 12.0)


def _home_need(price: float, p: dict) -> tuple[float, float]:
    return price * p["down_payment_share"], price * p["purchase_fee_rate"]


def action_reason(s: State, a: Action, t: int, p: dict, strict: bool = True,
                  life_done: bool = False) -> str | None:
    """Why action `a` is not allowed in state `s` at month t (None = allowed)."""
    if a.type not in ACTION_TYPES:
        return f"Unknown action type '{a.type}'."
    if a.type == "life_choice":
        if life_done:
            return "You already made one big life move this month: sleep first."
        return life_reason(s, t, p, a.id or "")
    if a.type == "set_habit":
        h = HABITS.get(a.id or "")
        if h is None:
            return f"Unknown habit '{a.id}'."
        if a.value is None or int(a.value) != a.value or not 0 <= int(a.value) < len(h.levels):
            return f"Level must be 0-{len(h.levels) - 1}."
        return None
    if a.type == "buy_item":
        it = ITEMS.get(a.id or "")
        if it is None:
            return f"Unknown item '{a.id}'."
        if it.abroad_only and s.status != "abroad":
            return "Only when you live abroad."
        return None
    if a.type == "change_flat":
        if a.id not in FLATS:
            return f"Unknown flat option '{a.id}'."
        if s.owner:
            return "You own your home: sell it first to rent again."
        if abs(s.flat_factor - p[flat_factor_key(a.id)]) < 1e-9:
            return "That's the kind of flat you already live in."
        return None
    if a.type == "buy_home":
        if a.id not in HOMES:
            return f"Unknown home '{a.id}'."
        if s.owner:
            return "You already own a home: sell it first."
        if s.status == "abroad":
            return "These homes are in Switzerland: move back first."
        down, fee = _home_need(_home_price(a.id, t, p), p)
        liquid = s.cash + s.investments
        if strict and liquid < down + fee:
            return (f"You need {chf(down + fee)} for the down payment and fees "
                    f"({p['down_payment_share']:.0%} + {p['purchase_fee_rate']:.0%}); "
                    f"you have {chf(liquid)} in cash and investments.")
        return None
    if a.type == "sell_home":
        return None if s.owner else "You don't own a home."
    if a.type == "take_loan":
        if p["loan_max"] <= 0:
            return "No loans in these assumptions."
        if s.debt > 0:
            return "Repay your current loan first."
        if a.value is None or not 0 < a.value <= p["loan_max"]:
            return f"Loan amount must be between 0 and {chf(p['loan_max'])}."
        return None
    if a.type == "repay_debt":
        if s.debt <= 0:
            return "You have no debt."
        if strict and s.cash < s.debt:
            return f"Not enough cash to repay {chf(s.debt)} in full."
        return None
    if a.type == "set_savings_rate":
        if a.value is None or not 0 <= a.value <= 1:
            return "Savings rate must be between 0% and 100%."
        return None
    if a.type == "ask_raise":
        if s.status != "working":
            return "You need a job in Switzerland to ask for a raise."
        need = int(p["raise_ask_min_months"])
        months = t - s.salary_start
        if months < need:
            return f"Possible after {need} months in the job ({need - months} to go)."
        if s.last_raise_ask is not None and t - s.last_raise_ask < 12:
            return f"You asked {t - s.last_raise_ask} months ago: wait a year between asks."
        return None
    return None  # pragma: no cover


# ----------------------------------------------------------------------------- effects
def _raise_draw(luck: tuple[int, int], t: int) -> float:
    return float(np.random.default_rng([int(luck[0]), int(luck[1]), t, 7919]).random())


def _apply_life(s: State, t: int, p: dict, cid: str) -> State:
    if cid in LEGACY_LIFE:
        s = apply_decision(s, Decision(t, cid), p)
        s = replace(s, work_share=1.0)
        if cid in ("move_abroad", "return_home"):
            s = replace(s, flat_factor=1.0)  # new country, new flat
        return s
    if cid in ("graduate", "quit_school"):
        # Leave school: remaining tuition is cancelled. A retiree goes back to retirement; anyone else
        # starts looking for a job (the regime HMM decides when they find one).
        kept = tuple(o for o in s.scheduled if not (o.kind == "tuition" and o.month >= t))
        no_degree = s.no_degree or (cid == "quit_school" and s.program is None)
        s = replace(s, scheduled=kept, program=None, program_start=None, program_end=None, no_degree=no_degree)
        if s.retiree:
            return replace(s, status="retired", retiree=False)
        return replace(s, status="unemployed", salary=0.0, salary_start=t, living=p["living_studying"],
                       work_share=1.0)
    if cid == "quit_job":
        return replace(s, status="unemployed", salary=s.salary * p["quit_benefit_rate"], salary_start=t,
                       living=p["living_studying"], work_share=1.0, last_switch=t)
    if cid == "reduce_hours":
        share = p["reduced_hours_share"]
        return replace(s, salary=s.salary * share, work_share=share)
    if cid == "full_hours":
        return replace(s, salary=s.salary / s.work_share, work_share=1.0)
    raise InvalidAction(cid)  # pragma: no cover


def apply_action(s: State, a: Action, t: int, p: dict, luck: tuple[int, int] = (42, 0),
                 strict: bool = True, life_done: bool = False) -> tuple[State, list[dict]]:
    reason = action_reason(s, a, t, p, strict, life_done)
    if reason:
        raise InvalidAction(f"{a.type}{':' + a.id if a.id else ''} at month {t}: {reason}")
    infl = inflation_factor(p["inflation_annual"], t)
    letters: list[dict] = []
    if a.type == "life_choice":
        was_master = s.program is not None
        s = _apply_life(s, t, p, a.id)
        if a.id == "graduate":
            letters.append(_letter(t, "degree", "🎓", "You got your degree!",
                                   "Congratulations. Visit the Office to start working: "
                                   f"jobs pay about {chf(_work_terms(s, p)[0])}/month."))
        elif a.id == "quit_school":
            kept = "You keep your bachelor's. " if was_master else "You have no degree: a master's is closed to you. "
            letters.append(_letter(t, "dropped_out", "🚪", "You left school",
                                   kept + f"Jobs pay about {chf(_work_terms(s, p)[0])}/month: "
                                   "visit the Office to start working."))
    elif a.type == "set_habit":
        kept = tuple(x for x in s.habits if x[0] != a.id)
        lv = int(a.value)
        s = replace(s, habits=tuple(sorted(kept + (((a.id, lv),) if lv > 0 else ()))))
    elif a.type == "buy_item":
        it = ITEMS[a.id]
        s = replace(s, scheduled=s.scheduled + (OneOff(t, p[item_price_key(it.id)] * infl, it.label, it.kind,
                                                       joy=p[item_joy_key(it.id)]),))
    elif a.type == "change_flat":
        s = replace(s, flat_factor=p[flat_factor_key(a.id)],
                    scheduled=s.scheduled + (OneOff(t, p["flat_moving_cost"] * infl, "Moving flat", "moving"),))
    elif a.type == "buy_home":
        price = _home_price(a.id, t, p)
        down, fee = _home_need(price, p)
        cash, inv = s.cash - down, s.investments
        if cash < 0 and inv > 0:  # sell investments for the down payment (a transfer)
            sold = min(inv, -cash)
            cash, inv = cash + sold, inv - sold
        s = replace(s, cash=cash, investments=inv, home_id=a.id, home_price=price, home_value=price,
                    mortgage=price - down,
                    scheduled=s.scheduled + (OneOff(t, fee, f"Purchase fees: {HOMES[a.id].label}", "home_fee"),))
        letters.append(_letter(t, "home_bought", "🏠", "You own a home",
                               f"{HOMES[a.id].label} for {chf(price)}: {chf(down)} down, "
                               f"mortgage {chf(price - down)}, fees {chf(fee)}."))
    elif a.type == "sell_home":
        fee = s.home_value * p["sale_fee_rate"]
        proceeds = s.home_value - s.mortgage
        label = HOMES[s.home_id].label
        letters.append(_letter(t, "home_sold", "🔑", "Home sold",
                               f"{label} sold for {chf(s.home_value)}; after the mortgage you get "
                               f"{chf(proceeds)}, minus {chf(fee)} fees. Back to renting."))
        s = replace(s, cash=s.cash + proceeds, home_id=None, home_price=0.0, home_value=0.0, mortgage=0.0,
                    flat_factor=p[flat_factor_key("same")],
                    scheduled=s.scheduled + (OneOff(t, fee, f"Selling fees: {label}", "home_fee"),))
    elif a.type == "take_loan":
        s = take_loan(s, float(a.value), p)
    elif a.type == "repay_debt":
        s = repay_all(s)
    elif a.type == "set_savings_rate":
        s = replace(s, savings_rate=float(a.value))
    elif a.type == "ask_raise":
        ok = _raise_draw(luck, t) < p["raise_ask_chance"]
        if ok:
            s = replace(s, salary=s.salary * (1 + p["raise_ask_size"]), last_raise_ask=t)
            letters.append(_letter(t, "raise", "📈", "Raise approved",
                                   f"Your salary rises {p['raise_ask_size']:.0%} to {chf(s.salary)}/month."))
        else:
            s = replace(s, last_raise_ask=t)
            letters.append(_letter(t, "raise", "📄", "Not this year",
                                   "Your manager says there is no budget for a raise right now. "
                                   "You can ask again in 12 months."))
    return s, letters


# ----------------------------------------------------------------------------- replay
def _check_turns(turns: list[Turn], horizon: int) -> dict[int, Turn]:
    months = [x.month for x in turns]
    if months != sorted(months) or len(set(months)) != len(months):
        raise InvalidAction("Turns must be in strictly increasing month order")
    if months and (months[0] < 0 or months[-1] >= horizon):
        raise InvalidAction(f"Turn months must be within 0-{horizon - 1}")
    return {x.month: x for x in turns}


def _world_letters(t: int, events: list[dict], pre: State, row: dict, p: dict) -> list[dict]:
    out = []
    for e in events:
        if e["month"] != t:
            continue
        if e["kind"] == "market_drop":
            if pre.investments > 1:
                out.append(_letter(t, "market_drop", "📉", "Markets dropped",
                                   f"Markets fell {p['crash_size']:.0%} this month. Your investments are now "
                                   f"{chf(row['investments'])}. In this game nothing is sold in panic."))
        elif e["kind"] == "income_loss":
            spared = pre.has_master and e.get("keep", 0.0) >= p["masters_job_loss_factor"]
            if pre.status in ("working", "abroad") and not spared:
                out.append(_letter(t, "income_loss", "📮", "Your job ended",
                                   f"{e['text'].split(' from')[0]}: you get {p['unemployment_benefit_rate']:.0%} "
                                   f"of your income as benefits until you find work again."))
        elif e["kind"] == "rent_rise":
            if not pre.owner:
                out.append(_letter(t, "rent_rise", "📬", "Rent rise",
                                   f"Your landlord raised the rent {p['rent_shock_size']:.0%}. "
                                   f"Rent is now {chf(row['rent'])}/month."))
    return out


AHV_ADJUST_EVERY = 24   # months; month 0 = Jan 2025, so Jan 2027, Jan 2029, ... (AHVG Art. 33ter)
AHV_13TH_FIRST = 23     # December 2026


def mixed_index(p: dict) -> float:
    """Swiss AHV mixed index: the average of nominal wage growth and inflation (per year)."""
    return (p["wage_index_annual"] + p["inflation_annual"]) / 2.0


def ahv_max_at(t: int, p: dict) -> float:
    """Maximum AHV pension in month t: the 2025 value, raised by the mixed index at each adjustment."""
    return p["ahv_max_pension"] * (1 + mixed_index(p)) ** (2 * (t // AHV_ADJUST_EVERY))


def pension_month(s: State, t: int, p: dict) -> tuple[State, float, list[dict]]:
    """Swiss pension rules for a retiree in month t: AHV indexation every 2 years, 13th AHV pension in December.

    Returns (state, extra income this month, letters)."""
    letters: list[dict] = []
    ahv, bvg = s.pension_ahv, s.pension_bvg
    if ahv <= 0 and bvg <= 0 and s.salary > 0:  # retired from the start: split the pension once
        ahv = min(s.salary, ahv_max_at(t, p))
        bvg = s.salary - ahv
        s = fast_replace(s, pension_ahv=ahv, pension_bvg=bvg)
    if t > 0 and t % AHV_ADJUST_EVERY == 0 and ahv > 0:
        f = (1 + mixed_index(p)) ** 2
        ahv *= f
        s = fast_replace(s, pension_ahv=ahv, salary=ahv + bvg)
        letters.append(_letter(t, "pension_indexed", "📈", "AHV pension adjusted",
                               f"The Federal Council adjusts AHV pensions to the mixed index (+{f - 1:.1%} over two "
                               f"years). Your AHV pension is now {chf(ahv)}/month; your BVG pension of {chf(bvg)} "
                               "stays the same."))
    extra = ahv if p["ahv_13th_pension"] >= 0.5 and t >= AHV_13TH_FIRST and t % 12 == 11 else 0.0
    return s, extra, letters


def job_mult(s: State, world: dict | None, t: int, p: dict) -> float:
    """Income multiplier from the world's job-loss spells this month. With a master's, a spell only happens
    if its `keep` draw is below masters_job_loss_factor (fewer job losses, same random numbers)."""
    if world is None:
        return 1.0
    m = float(world["income_mult"][t])
    if m < 1.0 and s.has_master and "keep" in world and float(world["keep"][t]) >= p["masters_job_loss_factor"]:
        return 1.0
    return m


def life_status(s: State, mult: float) -> str:
    """What this month looks like on the timeline: employed, part_time, job_search, student or retired."""
    if s.status == "studying":
        return "student"
    if s.status == "retired":
        return "retired"
    if s.status == "unemployed":
        return "job_search"
    if mult < 1.0:
        return "job_search"  # a job-loss spell from the world while working
    return "part_time" if s.work_share < 1 else "employed"


def life_month(s: State, t: int, p: dict, world: dict | None) -> tuple[State, float, list[dict]]:
    """Age rules for month t (retirement, Swiss pension). Returns (state, extra income, letters).

    Runs after the player's actions and before step_month."""
    letters: list[dict] = []
    age = s.start_age + t / 12.0
    in_labour = s.status in ("working", "abroad", "unemployed") or (s.status == "studying" and s.program is None)
    if age >= p["retirement_age"] and in_labour:
        base = s.salary / s.work_share if s.status in ("working", "abroad") else _work_terms(s, p)[0]
        pension = max(p["pension_income"], base * p["pension_replacement_rate"])
        ahv = min(pension, ahv_max_at(t, p))
        letters.append(_letter(t, "retired", "🌅", "You retired",
                               f"At {int(age)} you leave the labour market (AHV age rule). Your pension is "
                               f"{chf(pension)}/month, {p['pension_replacement_rate']:.1%} of your last full salary: "
                               f"AHV {chf(ahv)} (adjusted to the mixed index every 2 years, plus a 13th AHV "
                               f"pension each December) and BVG {chf(pension - ahv)} (not indexed)."))
        s = fast_replace(s, status="retired", salary=pension, salary_start=t, work_share=1.0,
                         pension_ahv=ahv, pension_bvg=pension - ahv)
    extra = 0.0
    if s.status == "retired" or s.retiree:  # a retiree studying keeps the pension (and its rules)
        s, extra, ls = pension_month(s, t, p)
        letters.extend(ls)
    return s, extra, letters


def run_turns(p: dict, turns: list[Turn], start: State | None = None, world: dict | None = None,
              luck: tuple[int, int] = (42, 0), strict: bool = True, until: int | None = None,
              resume: tuple[State, int] | None = None, keep_states: bool = False,
              strict_from: int = 0) -> GameRun:
    """Replay a reality from month 0 (or from `resume`=(opening state, month)).

    `until`=m stops at month m after applying that month's actions (without simulating it); the
    resulting state is GameRun.final_state. That is the "pending" view the player sees in town.
    Affordability is checked from month `strict_from` on (so editing an assumption later never
    "un-lives" an earlier purchase: it shows up as negative cash instead).
    """
    validate_params(p)
    horizon = int(p["horizon_months"])
    by_month = _check_turns(turns, horizon)
    events = list(world.get("events", [])) if world else []
    s, t0 = resume if resume is not None else (start or initial_state(p), 0)
    rows, letters = [], []
    opening, applied = {}, {}
    if until is not None and not 0 <= until <= horizon:
        raise InvalidAction(f"Month {until} is outside the horizon")
    for t in range(t0, horizon if until is None else until + 1):
        if keep_states:
            opening[t] = s
        graduated = s.program is not None and s.program_end == t
        if graduated:
            s = complete_programs(s, t)
            letters.append(_letter(t, "degree", "🎓", "Degree finished!",
                                   "Your master's is done. You're back in retirement, pension unchanged."
                                   if s.status == "retired" else
                                   "Your master's is done. Unless you choose another move this month, "
                                   f"you start a job at {chf(p['salary_master'])}/month."))
        life_done = False
        for a in (by_month[t].actions if t in by_month else ()):
            s, ls = apply_action(s, a, t, p, luck, strict and t >= strict_from, life_done)
            letters.extend(ls)
            life_done = life_done or a.type == "life_choice"
        if t == until:
            break
        if graduated and not life_done and s.status == "studying" and s.program is None:
            s = _apply_life(s, t, p, "start_work")
        if keep_states:
            applied[t] = s
        s, extra, ls = life_month(s, t, p, world)
        letters.extend(ls)
        pre = s
        mult = job_mult(pre, world, t, p)
        if world is None:
            s, row = step_month(s, p, income_extra=extra)
        else:
            s, row = step_month(s, p, float(world["ret"][t]), mult, float(world["rent_mult"][t]), extra)
        row["regime"] = life_status(pre, mult)
        rows.append(row)
        letters.extend(_world_letters(t, events, pre, row, p))
        prev_cash = rows[-2]["cash"] if len(rows) > 1 else pre.cash
        if row["cash"] < 0 <= prev_cash:
            letters.append(_letter(t, "financing_needed", "⚠️", "Financing needed",
                                   f"Your cash is {chf(row['cash'])}. In real life you would need a loan, "
                                   "family help or to cut costs. Visit the Bank to see your options."))
        if pre.debt > 0 and row["debt"] == 0:
            letters.append(_letter(t, "loan_paid", "✅", "Loan paid off", "Your debt is fully repaid."))
        if pre.mortgage > 0 and row["mortgage"] == 0:
            letters.append(_letter(t, "mortgage_paid", "🏡", "Mortgage paid off", "Your home is fully yours."))
    return GameRun(rows=rows, letters=letters, world_events=events, final_state=s,
                   opening=opening, applied=applied)


def state_at(p: dict, turns: list[Turn], month: int, start: State | None = None,
             world: dict | None = None, luck: tuple[int, int] = (42, 0)) -> State:
    """Opening state of `month` after that month's queued actions (what the player sees in town)."""
    return run_turns(p, turns, start=start, world=world, luck=luck, until=month).final_state


def decisions_to_turns(p: dict, decisions: list[Decision], start: State | None = None) -> list[Turn]:
    """Convert an old decision-point branch into monthly turns (defaults made explicit)."""
    run = run_branch(p, decisions, start=start)
    turns = []
    for pt in run.points:
        acts = []
        if pt["chosen"] not in ("continue", "stay_student"):
            acts.append(Action("life_choice", pt["chosen"]))
        if pt["savings_rate"] is not None:
            acts.append(Action("set_savings_rate", value=pt["savings_rate"]))
        if pt["loan_amount"] is not None:
            acts.append(Action("take_loan", value=pt["loan_amount"]))
        if pt["repay_debt"]:
            acts.append(Action("repay_debt"))
        if acts:
            turns.append(Turn(pt["month"], tuple(acts)))
    return turns


# ----------------------------------------------------------------------------- masking + previews
def _preview(text: str, one_off: float | None = None, monthly: float | None = None,
             warning: str | None = None, delta: float | None = None, joy: float | None = None) -> dict:
    return {"text": text, "one_off": one_off, "monthly": monthly, "monthly_delta": delta,
            "joy": joy, "warning": warning}


def _entry(s, t, p, place, a: Action, label, preview, tags, life_done=False, current=False) -> dict:
    reason = action_reason(s, a, t, p, True, life_done)
    return {"place": place, "type": a.type, "id": a.id, "value": a.value, "label": label,
            "enabled": reason is None and not current, "reason": reason if not current else "This is what you have now.",
            "current": current, "preview": preview, "tags": tags}


def monthly_income(s: State, p: dict) -> float:
    return s.salary * (1 - p["tax_rate"] - p["social_rate"])


def available_actions(s: State, t: int, p: dict, queued: tuple[Action, ...] = ()) -> dict[str, list[dict]]:
    """Every action per place with enabled, reason_if_disabled and an engine-computed preview.

    `s` must be the opening state of month t after the actions already queued (`queued`)."""
    infl = inflation_factor(p["inflation_annual"], t)
    life_done = any(a.type == "life_choice" for a in queued)
    out: dict[str, list[dict]] = {k: [] for k in PLACES}
    legacy = {c["id"]: c["preview"] for c in options(s, t, p)["choices"]}
    gen_preview = {
        "quit_job": f"Income drops to {chf(s.salary * p['quit_benefit_rate'])}/month until you find work again.",
        "reduce_hours": f"Salary {chf(s.salary * p['reduced_hours_share'])}/month instead of {chf(s.salary)}; "
                        f"+{p['joy_reduced_hours']:g} joy/month.",
        "full_hours": f"Salary back to {chf(s.salary / s.work_share if s.work_share else s.salary)}/month.",
        "drop_out": "Remaining tuition is cancelled; you start working at the pre-master salary.",
        "graduate": "Leave school with your degree. Then start a job at the Office: "
                    f"{chf(p['salary_master'] if s.has_master else p['salary_work'])}/month.",
        "quit_school": ("Leave now. Remaining tuition is cancelled; you keep your bachelor's and look for a job."
                        if s.program is not None else
                        "Leave without a degree. Jobs at the Office pay about "
                        f"{chf(p['salary_work'] * p['no_degree_wage_factor'])}/month "
                        f"({p['no_degree_wage_factor']:.0%}) and a master's is closed to you."),
    }
    for cid in ("master_ft", "master_pt", "graduate", "quit_school", "start_work", "reduce_hours", "full_hours", "quit_job",
                "move_abroad", "return_home"):
        text = legacy.get(cid) or gen_preview.get(cid) or ""
        tags = ["assumption", "simulated"]
        out[LIFE_PLACE[cid]].append(_entry(s, t, p, LIFE_PLACE[cid], Action("life_choice", cid),
                                           LIFE_LABELS[cid], _preview(text), tags, life_done))
    # Office: ask for a raise
    out["office"].insert(len(out["office"]) - 1, _entry(
        s, t, p, "office", Action("ask_raise"), "Ask for a raise",
        _preview(f"{p['raise_ask_chance']:.0%} chance (drawn from your world) of +{p['raise_ask_size']:.0%}: "
                 f"{chf(s.salary * (1 + p['raise_ask_size']))}/month.", delta=s.salary * p["raise_ask_size"]),
        ["assumption", "simulated"]))

    # Habits (café: eat out; shop: shopping, gym)
    current = dict(s.habits)
    for h in HABITS.values():
        cur = current.get(h.id, 0)
        cur_cost = p[habit_cost_key(h.id, cur)] if cur else 0.0
        for i, lv in enumerate(h.levels):
            cost = p[habit_cost_key(h.id, i)] if i else 0.0
            joy = p[habit_joy_key(h.id, i)] if i else 0.0
            text = (f"{chf(cost * infl)}/month" if cost else "Nothing on top of your living costs") + \
                   (f" · +{joy:g} joy/month" if joy else "")
            out[h.place].append(_entry(s, t, p, h.place, Action("set_habit", h.id, i), f"{h.label}: {lv.label}",
                                       _preview(text, monthly=cost * infl, delta=(cost - cur_cost) * infl, joy=joy),
                                       ["assumption"], current=(i == cur)))
    # One-off items and trips
    for it in ITEMS.values():
        price = p[item_price_key(it.id)] * infl
        warn = "🔴 More than your cash: you'd need financing." if price > s.cash else None
        out[it.place].append(_entry(s, t, p, it.place, Action("buy_item", it.id), it.label,
                                    _preview(f"{chf(price)} once · +{p[item_joy_key(it.id)]:g} joy. "
                                             "Consumption, not an investment: net worth falls by the price.",
                                             one_off=price, warning=warn, joy=p[item_joy_key(it.id)]),
                                    ["assumption"]))
    # Home: change flat
    base_rent = s.rent
    for fid, (label, _) in FLATS.items():
        f = p[flat_factor_key(fid)]
        out["home"].append(_entry(s, t, p, "home", Action("change_flat", fid), label,
                                  _preview(f"Rent {chf(base_rent * f * infl)}/month (now "
                                           f"{chf(base_rent * s.flat_factor * infl)}), moving cost "
                                           f"{chf(p['flat_moving_cost'] * infl)} once.",
                                           one_off=p["flat_moving_cost"] * infl, monthly=base_rent * f * infl,
                                           delta=base_rent * (f - s.flat_factor) * infl),
                                  ["data" if fid == "same" else "assumption"]))
    # Bank
    for r in SAVINGS_PRESETS:
        out["bank"].append(_entry(s, t, p, "bank", Action("set_savings_rate", value=r), f"Invest {r:.0%} of spare cash",
                                  _preview(f"Each month {r:.0%} of cash above your {p['buffer_months']:g}-month "
                                           "buffer moves to investments. A transfer: net worth unchanged."),
                                  ["assumption"], current=abs(s.savings_rate - r) < 1e-9))
    for amount in sorted({5000.0, 10000.0, 20000.0, float(p["loan_max"])}):
        if amount > p["loan_max"] or amount <= 0:
            continue
        pay = annuity_payment(amount, p["loan_rate"], int(p["loan_term_months"]))
        out["bank"].append(_entry(s, t, p, "bank", Action("take_loan", value=amount), f"Borrow {chf(amount)}",
                                  _preview(f"{chf(pay)}/month for {int(p['loan_term_months'])} months at "
                                           f"{p['loan_rate']:.1%}; total interest {chf(pay * p['loan_term_months'] - amount)}.",
                                           monthly=pay),
                                  ["assumption"]))
    out["bank"].append(_entry(s, t, p, "bank", Action("repay_debt"), "Repay all debt",
                              _preview(f"Pay {chf(s.debt)} from cash. A transfer: net worth unchanged.",
                                       one_off=s.debt or None), ["simulated"]))
    # Real estate
    income = monthly_income(s, p)
    for hid, home in HOMES.items():
        price = _home_price(hid, t, p)
        down, fee = _home_need(price, p)
        mort = price - down
        interest = mort * p["mortgage_rate"] / 12
        amort = price * p["amortization_annual"] / 12
        maint = price * p["maintenance_annual"] / 12
        monthly = interest + amort + maint
        warn = None
        if income <= 0 or monthly > income / 3:
            warn = "🟡 Housing cost above the common one-third-of-income guideline."
        text = (f"{chf(price)}: pay {chf(down)} down + {chf(fee)} fees now. Each month: interest "
                f"{chf(interest)} + amortization {chf(amort)} + maintenance {chf(maint)} = {chf(monthly)}, "
                f"and no rent. Net worth changes only by the fees.")
        out["realestate"].append(_entry(s, t, p, "realestate", Action("buy_home", hid), f"Buy: {home.label}",
                                        _preview(text, one_off=down + fee, monthly=monthly, warning=warn),
                                        ["assumption"]))
    if s.owner:
        fee = s.home_value * p["sale_fee_rate"]
        text = f"Get {chf(s.home_value - s.mortgage - fee)} after the mortgage and {chf(fee)} fees; rent again."
    else:
        text = "You don't own a home."
    out["realestate"].append(_entry(s, t, p, "realestate", Action("sell_home"), "Sell your home",
                                    _preview(text), ["simulated"]))
    return out
