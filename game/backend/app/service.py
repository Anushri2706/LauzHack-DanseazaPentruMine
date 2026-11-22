"""Glue between the API schemas and the pure engine (persona + overrides → engine inputs)."""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .data.loader import Persona, build_persona
from .engine.assumptions import META, validate_params
from .engine.finance import inflation_factor
from .engine.compare import compare
from .engine.catalogue import (CANTON_PRICE_LEVEL, FLATS, HABITS, HOMES, ITEMS, RULES, flat_factor_key,
                               habit_cost_key, habit_joy_key, home_price_key, item_joy_key, item_price_key)
from .engine.decisions import CHOICE_LABELS
from .engine.game import (Action, GameRun, InvalidAction, Turn, available_actions, checkin_due, run_turns,
                          study_default_months)
from .engine.metrics import cash_risk, check_goals, net_worth_bridge, scores, scores_now, summarize
from .engine.montecarlo import make_worlds, run_montecarlo, sample_world, stress_world
from .engine.narrative import chf, decision_narrative
from .engine.simulate import initial_state, run_branch
from .engine.state import Decision, State
from .models import (N_WORLDS, BranchRequest, CompareRequest, GameProjectionRequest, GameRequest,
                     GameRollRequest, MonteCarloRequest, SensitivityRequest, WorldRequest)

ROW_FIELDS = ("t", "age", "status", "income", "expenses", "one_offs", "one_off_labels", "cash",
              "investments", "debt", "net_worth", "buffer_target", "flags")


@dataclass(frozen=True)
class Ctx:
    persona: Persona
    params: dict
    start: State
    decisions: list[Decision]
    goals: list[dict]


def context(req: BranchRequest) -> Ctx:
    persona = build_persona(req.persona_id)
    params = {**persona.params, **{k: float(v) for k, v in req.assumptions.items()}}
    validate_params(params)
    # Starting position follows the (possibly edited) opening assumptions.
    start = initial_state(params, persona.status, persona.start_salary, persona.start_living)
    decisions = [Decision(**d.model_dump()) for d in req.decisions]
    goals = ([g.model_dump() for g in req.goals] if req.goals is not None
             else [dict(g) for g in persona.goals])
    return Ctx(persona, params, start, decisions, goals)


def _key(req) -> str:
    return req.model_dump_json()


@lru_cache(maxsize=512)
def _simulate(key: str) -> dict:
    req = BranchRequest.model_validate_json(key)
    c = context(req)
    run = run_branch(c.params, c.decisions, start=c.start)
    rows = [{k: r[k] for k in ROW_FIELDS} for r in run.rows]
    goals = check_goals(c.goals, run.rows)
    horizon = len(rows)
    current = run.next_decision_month if run.next_decision_month is not None else horizon
    nxt = None
    if run.next_decision_month is not None:
        pt = next(x for x in run.points if x["month"] == run.next_decision_month)
        s = run.states_at_points[pt["month"]]
        nxt = {"month": pt["month"], "age": pt["age"], "reasons": pt["reasons"],
               "narrative": decision_narrative(c.persona.name, c.persona.city, s, pt["reasons"], c.params),
               "choices": pt["options"]["choices"], "extras": pt["options"]["extras"]}
    now_row = rows[current - 1] if current > 0 else None
    now = ({"age": c.start.age, "cash": c.start.cash, "investments": c.start.investments,
            "debt": c.start.debt, "net_worth": c.start.cash + c.start.investments - c.start.debt,
            "status": c.start.status} if now_row is None else
           {"age": round(c.start.age + current / 12, 2), "cash": now_row["cash"],
            "investments": now_row["investments"], "debt": now_row["debt"],
            "net_worth": now_row["net_worth"], "status": now_row["status"]})
    points = [{"month": x["month"], "age": x["age"], "reasons": x["reasons"], "chosen": x["chosen"],
               "label": CHOICE_LABELS.get(x["chosen"], x["chosen"]), "by": x["by"]} for x in run.points]
    return {"persona": c.persona.card(), "rows": rows, "points": points, "current_month": current,
            "next_decision": nxt, "summary": summarize(run.rows), "goals": goals,
            "scores": scores(run.rows, goals, run.points), "now": now}


def simulate(req: BranchRequest) -> dict:
    return _simulate(_key(req))


@lru_cache(maxsize=128)
def _montecarlo(key: str) -> dict:
    req = MonteCarloRequest.model_validate_json(key)
    c = context(req)
    out = run_montecarlo(c.params, c.decisions, c.start, c.goals, n=req.n, seed=req.seed)
    # Ghost lines: named stress tests one year after the current decision point.
    run = run_branch(c.params, c.decisions, start=c.start)
    at = min(int(c.params["horizon_months"]) - 1,
             (run.next_decision_month if run.next_decision_month is not None else 0) + 12)
    ghosts = []
    for kind in ("market_drop", "income_loss", "rent_rise"):
        w = stress_world(c.params, kind, at)
        rows = run_branch(c.params, c.decisions, start=c.start, world=w).rows
        ghosts.append({"kind": kind, "text": w["text"], "month": at,
                       "net_worth": [r["net_worth"] for r in rows], "cash": [r["cash"] for r in rows]})
    out["ghosts"] = ghosts
    return out


def montecarlo(req: MonteCarloRequest) -> dict:
    return _montecarlo(_key(req))


def world(req: WorldRequest) -> dict:
    c = context(req)
    return sample_world(c.params, c.decisions, c.start, req.seed, req.world)


def compare_branches(req: CompareRequest) -> dict:
    if isinstance(req.a, GameRequest) != isinstance(req.b, GameRequest):
        raise ValueError("Compare two realities of the same kind (both with turns, or both with decisions)")
    if isinstance(req.a, GameRequest):
        return compare_games(req.a, req.b, req.name_a, req.name_b, req.at_age)
    a, b = context(req.a), context(req.b)
    ra = run_branch(a.params, a.decisions, start=a.start).rows
    rb = run_branch(b.params, b.decisions, start=b.start).rows
    opening = a.start.cash + a.start.investments - a.start.debt
    out = compare(ra, rb, req.name_a, req.name_b, opening, a.start.start_age,
                  check_goals(a.goals, ra), check_goals(b.goals, rb))
    out["note"] = ("Both realities use the same assumptions and the same random worlds; "
                   "differences come from decisions. Simulation, not a prediction.")
    return out


def options_for(req: BranchRequest) -> dict:
    res = simulate(req)
    return res["next_decision"] or {"month": None, "choices": [], "extras": {}}



# ============================================================================ town game (monthly turns)
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
START_YEAR = 2025  # the data snapshot is 2025-01-01
GAME_ROW_FIELDS = ROW_FIELDS + ("lifestyle", "purchases", "purchase_labels", "home_fees", "mortgage_interest",
                                "maintenance", "amortization", "home_appreciation", "home_value", "mortgage",
                                "rent", "joy", "interest", "overdraft_interest", "principal", "debt_payment", "invested",
                                "sold_investments", "investment_return")


LIFE_END = 120
REGIMES = ("employed", "part_time", "job_search", "student", "retired")
REGIME_IDX = {r: i for i, r in enumerate(REGIMES)}


def life_horizon(start_age: float) -> float:
    return float(max(12, min(1440, round((LIFE_END - start_age) * 12))))


@dataclass(frozen=True)
class GameCtx:
    persona: Persona
    params: dict
    start: State
    turns: list[Turn]
    goals: list[dict]
    world: dict
    luck: tuple[int, int]
    current: int


def game_context(req: GameRequest) -> GameCtx:
    persona = build_persona(req.persona_id)
    params = {**persona.params, **{k: float(v) for k, v in req.assumptions.items()}}
    if "horizon_months" not in req.assumptions:  # a whole life: the timeline runs to age LIFE_END
        params["horizon_months"] = life_horizon(params["start_age"])
    validate_params(params)
    horizon = int(params["horizon_months"])
    if req.current_month > horizon:
        raise InvalidAction(f"current_month {req.current_month} is beyond the horizon ({horizon})")
    if any(t.month > req.current_month for t in req.turns):
        raise InvalidAction("A turn lies in the future (month > current_month)")
    start = initial_state(params, persona.status, persona.start_salary, persona.start_living)
    turns = [Turn(t.month, tuple(Action(a.type, a.id, a.value) for a in t.actions)) for t in req.turns if t.actions]
    goals = ([g.model_dump() for g in req.goals] if req.goals is not None else [dict(g) for g in persona.goals])
    world = make_worlds(params, N_WORLDS, req.seed)[req.world]
    return GameCtx(persona, params, start, turns, goals, world, (req.seed, req.world), req.current_month)


def _date(month: int) -> dict:
    y, m = divmod(month, 12)
    return {"month_name": MONTHS[m], "year": START_YEAR + y, "label": f"{MONTHS[m]} {START_YEAR + y}"}


def _state_dict(s: State, p: dict) -> dict:
    return {"cash": s.cash, "investments": s.investments, "debt": s.debt, "home_value": s.home_value,
            "mortgage": s.mortgage, "home_id": s.home_id,
            "net_worth": s.cash + s.investments + s.home_value - s.debt - s.mortgage,
            "status": s.status, "program": s.program, "program_end": s.program_end, "has_master": s.has_master,
            "no_degree": s.no_degree, "enrolled": s.status == "studying",
            "salary": s.salary, "work_share": s.work_share, "savings_rate": s.savings_rate,
            "rent_base": s.rent, "flat_factor": s.flat_factor, "habits": dict(s.habits),
            "abroad": s.status == "abroad", "last_switch": s.last_switch}


def _statement(row: dict | None) -> dict | None:
    """Monthly statement of the last lived month (everything the Bank shows)."""
    if row is None:
        return None
    lines = [
        ("Income", row["income"], "in"), ("Income tax (estimate)", -row.get("income_tax", 0.0), "cost"),
        ("Rent", -row["rent"], "cost"),
        ("Other living costs", -(row["expenses"] - row["rent"]), "cost"),
        ("Habits", -row["lifestyle"], "cost"), ("Purchases & trips", -row["purchases"], "cost"),
        ("One-offs (tuition, moves)", -row["one_offs"], "cost"),
        ("Loan interest", -row["interest"], "cost"),
        ("Interest on negative cash", -row.get("overdraft_interest", 0.0), "cost"),
        ("Mortgage interest", -row["mortgage_interest"], "cost"),
        ("Home maintenance", -row["maintenance"], "cost"), ("Home fees", -row["home_fees"], "cost"),
        ("Loan principal (transfer)", -row["principal"], "transfer"),
        ("Mortgage amortization (transfer)", -row["amortization"], "transfer"),
        ("Moved to investments (transfer)", -row["invested"], "transfer"),
        ("Sold investments (transfer)", row["sold_investments"], "transfer"),
        ("Investment return", row["investment_return"], "return"),
        ("Home value change", row["home_appreciation"], "return"),
    ]
    return {"month": row["t"], **_date(row["t"]),
            "lines": [{"label": a, "amount": b, "kind": c} for a, b, c in lines if abs(b) > 0.005],
            "labels": row["one_off_labels"] + row["purchase_labels"],
            "cash_after": row["cash"], "net_worth_after": row["net_worth"]}


def _study_checkin(now: State, pending: State, m: int, p: dict) -> dict | None:
    """The master's reaches the end date the player planned: ask whether they finished, keep studying or
    drop out. `answered` once this month's plans settle it (any big move also counts as finished)."""
    if not checkin_due(now, m):
        return None
    return {"program": now.program, "started": now.program_start, "planned_end": m,
            "months_studied": m - (m if now.program_start is None else now.program_start), "answered": not checkin_due(pending, m),
            "default_months": study_default_months(now.program, p), "retiree": now.retiree}


@lru_cache(maxsize=256)
def _game_state(key: str) -> dict:
    req = GameRequest.model_validate_json(key)
    c = game_context(req)
    p, m = c.params, c.current
    run = run_turns(p, c.turns, start=c.start, world=c.world, luck=c.luck, until=m, keep_states=True,
                    strict_from=m)
    rows = run.rows
    now = run.opening[m]          # before this month's queue
    pending = run.final_state     # after this month's queue
    queued = next((t.actions for t in c.turns if t.month == m), ())
    goals = check_goals(c.goals, rows) if rows else [{**g, "status": "not_met" if 0 <= g["month"] < int(p["horizon_months"]) else "beyond_horizon", "available": None} for g in c.goals]
    horizon = int(p["horizon_months"])
    warnings = []
    if now.cash < 0:
        warnings.append({"kind": "financing_needed", "text": f"Cash is {chf(now.cash)}: financing needed "
                         f"(credit line at {p['overdraft_rate']:.0%}/year)."})
    if now.cash < -p["overdraft_limit"]:
        warnings.append({"kind": "beyond_credit", "text": f"🔴 The shortfall is beyond any credit line "
                         f"({chf(p['overdraft_limit'])}): this life is not financeable as planned."})
    if rows and pending.owner:
        last = rows[-1]
        housing = last["mortgage_interest"] + last["amortization"] + last["maintenance"]
        if last["income"] <= 0 or housing > last["income"] / 3:
            warnings.append({"kind": "housing_third", "text": "🟡 Housing costs are above the common "
                                                              "one-third-of-income guideline."})
    letters = [x for x in run.letters if x["month"] < m]
    queued_letters = [x for x in run.letters if x["month"] == m]
    return {
        "persona": c.persona.card(), "look": req.look, "world": req.world, "seed": req.seed,
        "month": m, "horizon": horizon, "ended": m >= horizon,
        "age": round(c.start.start_age + m / 12, 2), "age_years": int(c.start.start_age + m / 12),
        "date": _date(m), "months_to_birthday": 12 - (m % 12),
        "state": _state_dict(now, p), "pending_state": _state_dict(pending, p),
        "queued": [{"type": a.type, "id": a.id, "value": a.value} for a in queued],
        "queued_letters": queued_letters,
        "scores": scores_now(rows, goals, now, m),
        "actions": available_actions(pending, m, p, queued) if m < horizon else {},
        "study_checkin": _study_checkin(now, pending, m, p),
        "letters": letters,
        "statement": _statement(rows[-1] if rows else None),
        "warnings": warnings,
        "goals": goals,
        "rows": [{k: r[k] for k in ("t", "age", "status", "regime", "cash", "investments", "net_worth",
                                    "home_value", "mortgage", "debt", "joy", "flags")} for r in rows],
        "life_end_age": LIFE_END,
        # Price index (month 0 = 1): divide a value after i months by price_level[i] to read it in
        # today's (Jan 2025) francs. Same inflation the engine applies to costs.
        "inflation_annual": p["inflation_annual"],
        "price_level": [round(inflation_factor(p["inflation_annual"], i), 8) for i in range(horizon + 1)],
        "disclaimer": "Simulation, not a prediction or financial advice. Synthetic data only.",
    }


def game_state(req: GameRequest) -> dict:
    return _game_state(req.model_dump_json())


def _splice(past: dict, future: dict, m: int, horizon: int) -> dict:
    """World whose months < m come from the player's world and months ≥ m from `future`.
    Rent rises are levels, so the future world's rises are applied on top of the lived level."""
    if m <= 0:
        return future
    ret = np.concatenate([past["ret"][:m], future["ret"][m:]])
    inc = np.concatenate([past["income_mult"][:m], future["income_mult"][m:]])
    base = past["rent_mult"][m - 1] / future["rent_mult"][m - 1]
    rent = np.concatenate([past["rent_mult"][:m], future["rent_mult"][m:] * base])
    events = [e for e in past.get("events", []) if e["month"] < m] + \
             [e for e in future.get("events", []) if e["month"] >= m]
    out = {"ret": ret, "income_mult": inc, "rent_mult": rent, "events": events}
    if "keep" in past and "keep" in future:
        out["keep"] = np.concatenate([past["keep"][:m], future["keep"][m:]])
    return out


def _expected_world(p: dict) -> dict:
    from .engine.finance import monthly_compound_rate
    h = int(p["horizon_months"])
    return {"ret": np.full(h, monthly_compound_rate(p["return_annual"])), "income_mult": np.ones(h),
            "rent_mult": np.ones(h), "events": []}


def _future_rows(c: GameCtx, prefix_state: State, world: dict) -> GameRun:
    return run_turns(c.params, c.turns, world=world, luck=c.luck, strict=False, resume=(prefix_state, c.current))


@lru_cache(maxsize=64)
def _game_projection(key: str) -> dict:
    req = GameProjectionRequest.model_validate_json(key)
    c = game_context(req)
    p, m = c.params, c.current
    horizon = int(p["horizon_months"])
    lived = run_turns(p, c.turns, start=c.start, world=c.world, luck=c.luck, keep_states=True, strict_from=m)
    past = lived.rows[:m]
    pick = lambda rows, k: [r[k] for r in rows]  # noqa: E731
    out = {"month": m, "horizon": horizon, "n": req.n, "seed": req.seed, "world": req.world,
           "lived": {"net_worth": pick(past, "net_worth"), "cash": pick(past, "cash")}}
    ended = m >= horizon
    if ended:
        # The lived path is complete. Ranges then answer: "the same choices, replayed in every world
        # from the start" (other luck), so the postcard can still show ranges instead of one number.
        full_calm = lived.rows
        out["no_surprises"] = None
        m = 0
        past = []
    else:
        calm = _future_rows(c, lived.opening[m], _splice(c.world, _expected_world(p), m, horizon)).rows
        full_calm = past + calm
        out["no_surprises"] = {"net_worth": pick(calm, "net_worth"), "cash": pick(calm, "cash"),
                               "regime": pick(calm, "regime"),
                               "joy_total": sum(r["joy"] for r in full_calm)}
    s0 = lived.opening[m]
    c = GameCtx(c.persona, c.params, c.start, c.turns, c.goals, c.world, c.luck, m)
    worlds = make_worlds(p, N_WORLDS, req.seed)[:req.n]
    nw = np.empty((req.n, horizon - m))
    cash = np.empty((req.n, horizon - m))
    below_buf = np.zeros((req.n, horizon - m), dtype=bool)
    goal_hits = {g["goal_id"]: 0 for g in c.goals}
    reg = np.zeros((len(REGIMES), horizon - m))
    runs_out = []      # per world: first age, at or after retirement, with cash below zero (None = never)
    beyond = 0         # worlds that ever go beyond the credit line
    for k, w in enumerate(worlds):
        rows = _future_rows(c, s0, _splice(c.world, w, m, horizon)).rows
        runs_out.append(next((r["age"] for r in rows if r["cash"] < 0 and r["age"] >= p["retirement_age"]), None))
        beyond += any("beyond_credit" in r["flags"] for r in rows)
        for i, r in enumerate(rows):
            reg[REGIME_IDX[r["regime"]], i] += 1
        nw[k] = pick(rows, "net_worth")
        cash[k] = pick(rows, "cash")
        below_buf[k] = ["below_buffer" in r["flags"] for r in rows]
        for g in check_goals(c.goals, past + rows):
            if g["status"] == "met":
                goal_hits[g["goal_id"]] += 1
    q = lambda a: np.percentile(a, [10, 50, 90], axis=0)  # noqa: E731
    nq, cq = q(nw), q(cash)
    out["bands"] = None if ended else {
        "net_worth": {"p10": nq[0].tolist(), "p50": nq[1].tolist(), "p90": nq[2].tolist()},
        "cash": {"p10": cq[0].tolist(), "p50": cq[1].tolist(), "p90": cq[2].tolist()}}
    final = nw[:, -1]
    out["final_net_worth"] = {"p10": float(np.percentile(final, 10)), "p50": float(np.median(final)),
                              "p90": float(np.percentile(final, 90))}
    # From now on only: a past dip that was recovered from is not counted as risk.
    out["risk"] = cash_risk(cash, below_buf)
    out["share_worlds_shortfall"] = out["risk"]["below_zero_ahead"]
    out["regimes"] = None if ended else {
        "states": list(REGIMES), "from_month": m,
        "share": {st: (reg[i] / req.n).round(3).tolist() for i, st in enumerate(REGIMES)}}
    out["goal_met_share"] = {gid: h / req.n for gid, h in goal_hits.items()}
    hit = [a for a in runs_out if a is not None]
    out["retirement_runs_out"] = {"share": len(hit) / req.n,
                                  "median_age": float(np.median(hit)) if hit else None,
                                  "retirement_age": p["retirement_age"]}
    out["share_worlds_beyond_credit"] = beyond / req.n
    at = min(horizon - 1, m + 12)
    ghosts = []
    for kind in (() if ended else ("market_drop", "income_loss", "rent_rise")):
        w = stress_world(p, kind, at)
        rows = _future_rows(c, s0, _splice(c.world, {**w, "events": []}, m, horizon)).rows
        ghosts.append({"kind": kind, "text": w["text"], "month": at,
                       "net_worth": pick(rows, "net_worth"), "cash": pick(rows, "cash")})
    out["ghosts"] = ghosts
    out["bridge"] = net_worth_bridge(full_calm, c.start.cash + c.start.investments - c.start.debt)
    out["goals"] = check_goals(c.goals, full_calm)
    out["final_scores"] = scores(full_calm, out["goals"], [])
    out["ended"] = ended
    out["note"] = (f"Ranges: 80% CI (P10–P90) of the same choices replayed in {req.n} seeded worlds from the start (other luck)."
                   if ended else
                   f"Bands: 80% CI (P10–P90) over {req.n} seeded worlds from {_date(m)['label']} on, given what you "
                   "have lived so far. Ranges under our assumptions, not probabilities.")
    return out


def game_projection(req: GameProjectionRequest) -> dict:
    return _game_projection(req.model_dump_json())


def game_roll(req: GameRollRequest) -> dict:
    """'🎲 Roll another world': the same choices, different luck from now on."""
    c = game_context(req)
    m, horizon = c.current, int(c.params["horizon_months"])
    lived = run_turns(c.params, c.turns, start=c.start, world=c.world, luck=c.luck, keep_states=True,
                      strict_from=m)
    if m >= horizon:
        return {"world": req.roll, "net_worth": [], "cash": [], "events": []}
    w = _splice(c.world, make_worlds(c.params, N_WORLDS, req.seed)[req.roll], m, horizon)
    rows = _future_rows(c, lived.opening[m], w).rows
    return {"world": req.roll, "from_month": m, "net_worth": [r["net_worth"] for r in rows],
            "cash": [r["cash"] for r in rows], "events": [e for e in w["events"] if e["month"] >= m]}


def game_rows(req: GameRequest) -> tuple[list[dict], GameCtx]:
    """Full-horizon path of a reality in the player's world (the future is 'if nothing else changes')."""
    c = game_context(req)
    return run_turns(c.params, c.turns, start=c.start, world=c.world, luck=c.luck,
                     strict_from=c.current).rows, c


def compare_games(a: GameRequest, b: GameRequest, name_a: str, name_b: str, at_age: float | None = None) -> dict:
    # Fairness: both realities are evaluated in the SAME world (A's seed and world).
    b = b.model_copy(update={"seed": a.seed, "world": a.world})
    if at_age is not None:
        # Cut both realities at the same age: rows, goals, bridge and ranges all stop there.
        start_age = game_context(a).params["start_age"]
        life = int(life_horizon(start_age))
        months = max(12, a.current_month, b.current_month, min(life, int(round((at_age - start_age) * 12))))
        a = a.model_copy(update={"assumptions": {**a.assumptions, "horizon_months": months}})
        b = b.model_copy(update={"assumptions": {**b.assumptions, "horizon_months": months}})
    ra, ca = game_rows(a)
    rb, cb = game_rows(b)
    opening = ca.start.cash + ca.start.investments - ca.start.debt
    out = compare(ra, rb, name_a, name_b, opening, ca.start.start_age,
                  check_goals(ca.goals, ra), check_goals(cb.goals, rb))
    ranges = {}
    for key, r in (("a", a), ("b", b)):
        proj = game_projection(GameProjectionRequest(**r.model_dump(), n=40))
        ranges[key] = proj["final_net_worth"]
    out["ranges"] = ranges
    out["scores"] = {"a": scores(ra, check_goals(ca.goals, ra), []), "b": scores(rb, check_goals(cb.goals, rb), [])}
    out["net_worth"] = {"a": [r["net_worth"] for r in ra], "b": [r["net_worth"] for r in rb]}
    out["at_age"] = round(ca.start.start_age + len(ra) / 12, 2)
    out["note"] = (f"Both realities live in the same world (#{a.world}) with the same assumptions; "
                   "differences come from choices, not luck. Simulation, not a prediction.")
    return out


def catalogue(pid: str) -> dict:
    persona = build_persona(pid)
    p = persona.params
    lvl = CANTON_PRICE_LEVEL.get(persona.canton, 1.0)
    return {
        "habits": [{"id": h.id, "label": h.label, "place": h.place,
                    "levels": [{"level": i, "label": lv.label,
                                "cost": p[habit_cost_key(h.id, i)] if i else 0.0,
                                "joy": p[habit_joy_key(h.id, i)] if i else 0.0,
                                "cost_key": habit_cost_key(h.id, i) if i else None,
                                "joy_key": habit_joy_key(h.id, i) if i else None} for i, lv in enumerate(h.levels)]}
                   for h in HABITS.values()],
        "items": [{"id": it.id, "label": it.label, "place": it.place, "kind": it.kind,
                   "price": p[item_price_key(it.id)], "joy": p[item_joy_key(it.id)],
                   "price_key": item_price_key(it.id), "joy_key": item_joy_key(it.id),
                   "abroad_only": it.abroad_only} for it in ITEMS.values()],
        "homes": [{"id": h.id, "label": h.label, "price": p[home_price_key(h.id)], "price_key": home_price_key(h.id)}
                  for h in HOMES.values()],
        "home_price_level": {"canton": persona.canton, "level": lvl},
        "flats": [{"id": fid, "label": label, "factor": p[flat_factor_key(fid)], "key": flat_factor_key(fid)}
                  for fid, (label, _) in FLATS.items()],
        "rules": [{"key": k, "value": p[k], "label": v[1], "unit": v[2], "group": v[3]} for k, v in RULES.items()],
        "tag": "ours",
        "note": "All catalogue numbers are our labelled assumptions (🟡), editable in Settings. "
                "Purchases are consumption, not assets. Joy is a game preference score, not money.",
    }


# ============================================================================ learned twin (HMM x learned engine)
def twin_inputs(req: GameRequest) -> dict:
    """The player's situation now, as the learned twin needs it (from the town engine's replay)."""
    from .engine.game import monthly_income
    c = game_context(req)
    p, m = c.params, c.current
    run = run_turns(p, c.turns, start=c.start, world=c.world, luck=c.luck, until=m, keep_states=True, strict_from=m)
    s = run.final_state
    infl = (1 + p["inflation_annual"]) ** (m / 12)
    rent = 0.0 if s.owner else s.rent * s.flat_factor * infl
    return {"persona_id": req.persona_id, "month": m, "age": c.start.start_age + m / 12, "status": s.status,
            "work_share": s.work_share, "income": monthly_income(s, p), "rent": rent,
            "liquid": s.cash + s.investments, "has_master": s.has_master, "abroad": s.status == "abroad",
            "program_months_left": (s.program_end - m) if s.program and s.program_end else None,
            "horizon_months": int(p["horizon_months"]) - m, "inflation": p["inflation_annual"]}


@lru_cache(maxsize=64)
def _twin_fan(key: str) -> dict:
    from learned.twin import fan, model_card
    req = GameProjectionRequest.model_validate_json(key)
    inp = twin_inputs(req)
    if inp["horizon_months"] <= 0:
        raise InvalidAction("The timeline has ended: go back to explore the future again.")
    out = fan(**inp, n_paths=max(50, min(400, req.n * 5)), seed=req.seed)
    out["inputs"] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in inp.items()}
    out["model"] = model_card()["test"]["hmm+gmm+history"]
    return out


def twin_fan(req: GameProjectionRequest) -> dict:
    return _twin_fan(req.model_dump_json())


def twin_model() -> dict:
    from learned.twin import model_card
    return model_card()


# ============================================================================ sensitivity ("tornado")
# Assumptions swung one at a time. Profile facts (opening cash, age) are not swung: they are data, not guesses.
SENSITIVITY_KEYS = ("return_annual", "inflation_annual", "raise_annual", "salary_work", "salary_master",
                    "student_income", "abroad_income", "rent", "living_studying", "living_working",
                    "living_working_master", "abroad_rent", "abroad_living", "tuition_annual", "income_tax_rate",
                    "investment_costs_annual", "lifestyle_creep", "savings_rate", "overdraft_rate",
                    "mortgage_rate", "home_growth_annual", "relocation_cost")


def _calm_net_worth(req: GameRequest, overrides: dict, months: int) -> tuple[float, float, int]:
    """Net worth (nominal and in month-0 prices) after `months`, replaying the reality's choices with no luck."""
    r = req.model_copy(update={"assumptions": {**req.assumptions, **overrides, "horizon_months": months}})
    c = game_context(r)
    rows = run_turns(c.params, c.turns, start=c.start, world=_expected_world(c.params), luck=c.luck,
                     strict=False).rows
    nominal = rows[-1]["net_worth"]
    neg = sum(1 for x in rows if x["cash"] < 0)
    return nominal, nominal / inflation_factor(c.params["inflation_annual"], months), neg


@lru_cache(maxsize=64)
def _sensitivity(key: str) -> dict:
    req = SensitivityRequest.model_validate_json(key)
    c = game_context(req)
    p, start_age = c.params, c.params["start_age"]
    life = int(life_horizon(start_age))
    target_age = req.target_age if req.target_age is not None else (30.0 if start_age < 29 else start_age + 10)
    months = int(round((target_age - start_age) * 12))
    months = max(12, req.current_month + 1, min(life, months))
    target_age = start_age + months / 12
    base, base_real, base_neg = _calm_net_worth(req, {}, months)
    bars = []
    for k in SENSITIVITY_KEYS:
        v = p.get(k)
        if k not in META or k in req.assumptions or not v:
            continue  # not a guess for this persona, edited by the player, or 0 (a ±% swing is meaningless)
        meta = META[k]
        lo, hi = v * (1 - req.swing), v * (1 + req.swing)
        lo = max(lo, meta.min) if meta.min is not None else lo
        hi = min(hi, meta.max) if meta.max is not None else hi
        nl, rl, negl = _calm_net_worth(req, {k: lo}, months)
        nh, rh, negh = _calm_net_worth(req, {k: hi}, months)
        bars.append({"key": k, "label": meta.label, "unit": meta.unit, "group": meta.group, "value": v,
                     "low": lo, "high": hi,
                     "net_worth_low": nl, "net_worth_high": nh, "real_low": rl, "real_high": rh,
                     "negative_months_low": negl, "negative_months_high": negh,
                     "spread": abs(nh - nl)})
    bars.sort(key=lambda b: -b["spread"])
    no_effect = [b["label"] for b in bars if b["spread"] < 0.5]
    return {"target_age": round(target_age, 2), "months": months, "swing": req.swing,
            "base": base, "base_real": base_real, "base_negative_months": base_neg,
            "bars": [b for b in bars if b["spread"] >= 0.5], "no_effect": no_effect,
            "note": (f"Each assumption moved ±{req.swing:.0%} on its own, everything else fixed, your choices "
                     "replayed with no luck (no market swings, job losses or rent rises). This is sensitivity, "
                     "not probability: it shows what the result leans on, not how likely anything is. "
                     "Assumptions with no effect on this reality are listed separately (scenario isolation).")}


def sensitivity(req: SensitivityRequest) -> dict:
    return _sensitivity(req.model_dump_json())
