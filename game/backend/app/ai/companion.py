"""The game's AI companion: answers questions about the player's life using only numbers the engines produced.

Adapted from the teammates' AI layer (feature/combined-engine, app/ai: claim_guard + companion). Same rules:

  1. The engines compute everything. This module turns their outputs into citable FACTS with stable ids
     (town engine: state, statement, projection; learned twin: HMM x learned engine; model card).
  2. The LLM (optional: Gemini / Groq / Ollama via llm.py and a git-ignored .env) may only write sentences
     with {fact_id} placeholders. The guard rejects sentences with numbers of their own, unknown fact ids,
     certainty or causal language. Numbers are filled in by code from the facts.
  3. The LLM may PROPOSE one town action. It is shown only when the engine says it is allowed now, and it
     happens only when the player presses the button (the normal queue + engine validation).
  4. Without an LLM (no key, quota, error) the same facts are answered by deterministic templates.
"""
from __future__ import annotations

import json
import re

from ..models import GameProjectionRequest, GameRequest
from .llm import LLMUnavailable, chat_json, describe

DISCLAIMER = ("Simulated outcomes under stated assumptions and synthetic data, not predictions or financial "
              "advice.")
CERTAINTY = re.compile(r"\b(guarantee\w*|definitely|certain(ly)?|for sure|will (be|have|make|earn|reach)|"
                       r"no doubt|always|never fail|risk[- ]free|proven)\b", re.I)
CAUSAL = re.compile(r"\b(because of|caus(e|es|ed|ing)|leads? to|results? in|due to)\b", re.I)
PLACEHOLDER = re.compile(r"\{([a-z0-9_.]+)\}")


# ----------------------------------------------------------------------------- formatting
def _chf(v: float) -> str:
    return ("CHF −" if v < 0 else "CHF ") + f"{abs(v):,.0f}".replace(",", "'")


def fmt(f: dict) -> str:
    v, u = f["value"], f["unit"]
    if v is None:
        return "n/a"
    if u == "chf":
        return _chf(v)
    if u == "chf_month":
        return _chf(v) + "/month"
    if u == "share":
        return f"{100 * v:.0f}%"
    if u == "months":
        return f"{v:.0f} months"
    if u == "age":
        return f"{v:.0f}"
    return str(v)


# ----------------------------------------------------------------------------- facts
def _f(facts: dict, fid: str, label: str, value, unit: str, source: str) -> None:
    if value is not None:
        facts[fid] = {"label": label, "value": value, "unit": unit, "source": source}


def build_facts(st: dict, proj: dict | None, twin: dict | None, card: dict | None) -> dict:
    facts: dict[str, dict] = {}
    s = st["state"]
    T = "town engine"
    _f(facts, "now.age", "your age", st["age_years"], "age", T)
    _f(facts, "now.cash", "cash now", s["cash"], "chf", T)
    _f(facts, "now.investments", "investments now", s["investments"], "chf", T)
    _f(facts, "now.net_worth", "net worth now", s["net_worth"], "chf", T)
    _f(facts, "now.debt", "debt now", s["debt"], "chf", T)
    stmt = st.get("statement")
    if stmt:
        lines = sorted(stmt["lines"], key=lambda x: x["amount"])
        costs = [x for x in lines if x["kind"] == "cost"]
        for i, x in enumerate(costs[:3]):
            _f(facts, f"month.cost{i + 1}", f"last month: {x['label'].lower()}", -x["amount"], "chf", T)
        inc = next((x for x in stmt["lines"] if x["label"] == "Income"), None)
        if inc:
            _f(facts, "month.income", "last month: income", inc["amount"], "chf", T)
        moved = next((x for x in stmt["lines"] if x["label"].startswith("Moved to investments")), None)
        if moved:
            _f(facts, "month.invested", "last month: moved from cash to investments (net worth unchanged)",
               -moved["amount"], "chf", T)
        rows = st.get("rows") or []
        if len(rows) >= 2:
            _f(facts, "month.cash_change", "change in cash last month", rows[-1]["cash"] - rows[-2]["cash"], "chf", T)
    if proj and proj.get("final_net_worth"):
        fn = proj["final_net_worth"]
        for k in ("p10", "p50", "p90"):
            _f(facts, f"rules.end_nw_{k}", f"town engine: net worth at the end of the timeline ({k.upper()})", fn[k], "chf", T)
        _f(facts, "rules.shortfall", "town engine: share of worlds where cash goes below zero",
           proj.get("share_worlds_shortfall"), "share", T)
    if twin:
        L = "learned twin"
        sm = twin["summary"]
        _f(facts, "twin.flow_now", "learned twin: typical monthly cash flow for people like you now", sm["flow_now_p50"], "chf_month", L)
        _f(facts, "twin.job_search", "learned twin: share of simulated lives with a job search before 65",
           sm.get("share_lives_with_job_search_before_65"), "share", L)
        _f(facts, "twin.job_search_months", "learned twin: typical months in job search before 65",
           sm.get("months_in_job_search_before_65_p50"), "months", L)
        _f(facts, "twin.retire_age", "learned twin: typical retirement age", sm.get("retirement_age_p50"), "age", L)
        if sm.get("wealth_at_65"):
            for k in ("p10", "p50", "p90"):
                _f(facts, f"twin.w65_{k}", f"learned twin: liquid savings at 65 ({k.upper()})", sm["wealth_at_65"][k], "chf", L)
        _f(facts, "twin.below_zero", "learned twin: share of lives that ever go below zero",
           sm.get("share_lives_ever_below_zero"), "share", L)
    if card:
        t = card["test"]
        M = "model card"
        _f(facts, "model.mae", "learned twin backtest: average error in cash after 12 months (locked test people)",
           t["hmm+gmm+history"]["mae_cash_end_chf"], "chf", M)
        _f(facts, "model.mae_hmm", "HMM alone backtest: average error", t["hmm"]["mae_cash_end_chf"], "chf", M)
        _f(facts, "model.mae_persist", "'keep your past average' backtest: average error",
           t["persistence"]["mae_cash_end_chf"], "chf", M)
        _f(facts, "model.coverage", "learned twin backtest: true value inside the 80% CI (P10–P90)",
           t["hmm+gmm+history"]["range_hit_rate_p10_p90"], "share", M)
        _f(facts, "model.n_test", "locked test people", t["hmm+gmm+history"]["n"], "count", M)
    return facts


# ----------------------------------------------------------------------------- intents and actions
TOPICS = {
    "cash": r"\bcash|money|drop|spend|cost|expens|statement|where.*go|lose\b",
    "future": r"futur|65|retire|later|end|old|pension|save|savings|rich|wealth|net worth",
    "risk": r"risk|job|unemploy|lose my|fired|safe|worst|bad",
    "model": r"model|accura|how do you|how does|trust|reliab|predict|hmm|gmm|learn",
    "debt": r"debt|loan|owe|borrow",
}
ACTION_WORDS = [
    (r"full.?time master|master", ("life_choice", "master_ft", None)),
    (r"start work|get a job|find a job|start a job|work\b", ("life_choice", "start_work", None)),
    (r"abroad|move to germany|emigrat", ("life_choice", "move_abroad", None)),
    (r"degree|graduat", ("life_choice", "graduate", None)),
    (r"drop out|quit school|leave school", ("life_choice", "quit_school", None)),
    (r"quit (my )?job", ("life_choice", "quit_job", None)),
    (r"80|part.?time|reduce", ("life_choice", "reduce_hours", None)),
    (r"studio", ("buy_home", "studio", None)),
    (r"buy (a )?(home|house|flat)", ("buy_home", "studio", None)),
]


def topics(text: str, default: bool = True) -> list[str]:
    t = text.lower()
    found = [k for k, rx in TOPICS.items() if re.search(rx, t)]
    return found or (["future"] if default else [])


def find_action(text: str, st: dict) -> dict | None:
    t = text.lower()
    allowed = {(a["type"], a["id"], a["value"]): a for acts in st["actions"].values() for a in acts}
    for rx, key in ACTION_WORDS:
        if re.search(rx, t):
            a = next((x for k, x in allowed.items() if k[0] == key[0] and k[1] == key[1]), None)
            if a:
                return {"type": a["type"], "id": a["id"], "value": a["value"], "label": a["label"],
                        "enabled": a["enabled"], "reason": a["reason"], "preview": a["preview"]["text"]}
    return None


# ----------------------------------------------------------------------------- answers
def render(sentence: str, facts: dict) -> str:
    return PLACEHOLDER.sub(lambda m: fmt(facts[m.group(1)]), sentence)


def guard(sentences: list, facts: dict) -> tuple[list[dict], list[str]]:
    ok, rejected = [], []
    for s in sentences:
        text = str(s.get("text", "") if isinstance(s, dict) else s)
        ids = PLACEHOLDER.findall(text)
        bare = PLACEHOLDER.sub("", text)
        if not text.strip():
            continue
        if re.search(r"\d", bare):
            rejected.append("number outside a fact")
        elif any(i not in facts for i in ids):
            rejected.append("unknown fact id")
        elif CERTAINTY.search(bare) or CAUSAL.search(bare):
            rejected.append("certainty or causal language")
        else:
            ok.append({"text": render(text, facts), "facts": ids})
    return ok, rejected


def rule_answer(text: str, facts: dict, action: dict | None) -> list[dict]:
    has = facts.__contains__
    out: list[str] = []
    for topic in topics(text, default=action is None):
        if topic == "cash":
            if has("month.cash_change"):
                out.append("Last month your cash changed by {month.cash_change}.")
            if has("month.income"):
                out.append("Income was {month.income}; the biggest costs were " + ", ".join(
                    f"{{{k}}} ({facts[k]['label'].split(': ', 1)[-1]})" for k in ("month.cost1", "month.cost2", "month.cost3") if has(k)) + ".")
            if has("month.invested"):
                out.append("Also {month.invested} moved from cash into investments: that lowers cash, not net worth.")
            if not out:
                out.append("You have {now.cash} in cash and {now.investments} invested. Live a month to see a statement.")
        elif topic == "future":
            if has("twin.w65_p50"):
                out.append("The learned twin, which follows people like you through simulated life events, has "
                           "liquid savings at sixty-five between {twin.w65_p10} and {twin.w65_p90}, typically {twin.w65_p50}.")
            if has("rules.end_nw_p50"):
                out.append("The town engine, with investment returns, ends your timeline with a net worth between "
                           "{rules.end_nw_p10} and {rules.end_nw_p90}.")
            if has("twin.retire_age"):
                out.append("In the twin's lives the typical retirement age is {twin.retire_age}.")
        elif topic == "risk":
            if has("twin.job_search"):
                out.append("In {twin.job_search} of the twin's simulated lives you go through a job search before sixty-five, "
                           "typically for {twin.job_search_months} in total.")
            if has("rules.shortfall"):
                out.append("In the town engine's worlds, cash goes below zero in {rules.shortfall} of them.")
        elif topic == "model":
            if has("model.mae"):
                out.append("Backtested on {model.n_test} locked test people (twelve months known, the next twelve "
                           "predicted), the learned twin is off by {model.mae} on average, against {model.mae_hmm} "
                           "for the HMM alone and {model.mae_persist} for keeping your past average; the truth falls "
                           "inside its range in {model.coverage} of cases.")
        elif topic == "debt":
            out.append("Your debt is {now.debt}." if has("now.debt") else "You have no debt.")
    if action:
        if has("if.twin.w65_p50") and has("twin.w65_p50"):
            out.append("If you choose “{action.label}” now, the learned twin's typical savings at sixty-five go from "
                       "{twin.w65_p50} to {if.twin.w65_p50}.")
        if has("if.rules.end_nw_p50") and has("rules.end_nw_p50"):
            out.append("In the town engine the typical net worth at the end of the timeline goes from "
                       "{rules.end_nw_p50} to {if.rules.end_nw_p50}.")
        if has("if.twin.job_search") and has("twin.job_search"):
            out.append("Lives with a job search before sixty-five: {twin.job_search} now, {if.twin.job_search} with it.")
        if action["enabled"]:
            out.append("You can choose “{action.label}” now; the engine's preview is below. Nothing happens until you confirm.")
        else:
            out.append("“{action.label}” is not possible right now: the engine's reason is below.")
    seen, sentences = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            sentences.append(s)
    return guard(sentences, facts)[0]


PROMPT = """You are the companion inside a financial life-simulation game (CHF, Switzerland, synthetic people).
Python engines computed every number. You only explain them.
Rules:
- Write two to four short sentences answering the player's question.
- Every number must be a placeholder {fact_id} from FACTS. Never write digits yourself.
- Only use fact ids that are in FACTS. Do not invent facts.
- No certainty words (guaranteed, definitely, will be) and no causal claims (because of, causes, leads to).
- Simulated outcomes are ranges under assumptions, not predictions. No personal financial advice; describe trade-offs.
- If the player asks to do something listed in ACTIONS, you may propose exactly that action; the player confirms.
Return ONLY JSON: {"sentences": [{"text": "..."}], "action": null or {"type": "...", "id": "..."}}"""


def llm_answer(text: str, facts: dict, actions: list[dict]) -> tuple[list[dict], dict | None, list[str]]:
    listing = "\n".join(f"{fid}: {f['label']} = {fmt(f)} ({f['source']})" for fid, f in facts.items())
    acts = "\n".join(f"{a['type']}:{a['id']} = {a['label']}" for a in actions[:40])
    reply = chat_json([{"role": "system", "content": PROMPT},
                       {"role": "user", "content": f"FACTS:\n{listing}\n\nACTIONS (allowed now):\n{acts}\n\n"
                                                   f"PLAYER: {text}"}])
    ok, rejected = guard(reply.get("sentences") or [], facts)
    prop = reply.get("action")
    action = None
    if isinstance(prop, dict):
        action = next((a for a in actions if a["type"] == prop.get("type") and a["id"] == prop.get("id")), None)
    return ok, action, rejected


def with_action(req: GameRequest, action: dict) -> GameRequest:
    """The same reality with `action` added to this month's plans."""
    turns = [t.model_copy(deep=True) for t in req.turns]
    item = {"type": action["type"], "id": action["id"], "value": action["value"]}
    cur = next((t for t in turns if t.month == req.current_month), None)
    data = req.model_dump()
    if cur is None:
        data["turns"] = [t.model_dump() for t in turns] + [{"month": req.current_month, "actions": [item]}]
    else:
        data["turns"] = [t.model_dump() if t is not cur else {"month": t.month, "actions": [*t.model_dump()["actions"], item]}
                         for t in turns]
    return GameRequest(**data)


def what_if(req: GameRequest, action: dict) -> dict:
    """Facts for the same reality with the action: both engines, same seed and world (differences = the choice)."""
    from .. import service
    out: dict[str, dict] = {}
    try:
        req2 = with_action(req, action)
        service.game_state(req2)  # the engine validates the plan
        preq = GameProjectionRequest(**req2.model_dump(), n=60)
        proj, twin = service.game_projection(preq), service.twin_fan(preq)
    except Exception:  # an action the engines refuse simply gives no what-if facts
        return out
    if proj.get("final_net_worth"):
        _f(out, "if.rules.end_nw_p50", "town engine with this decision: typical net worth at the end",
           proj["final_net_worth"]["p50"], "chf", "town engine (what-if)")
    sm = twin["summary"]
    if sm.get("wealth_at_65"):
        _f(out, "if.twin.w65_p50", "learned twin with this decision: typical liquid savings at 65",
           sm["wealth_at_65"]["p50"], "chf", "learned twin (what-if)")
    _f(out, "if.twin.job_search", "learned twin with this decision: lives with a job search before 65",
       sm.get("share_lives_with_job_search_before_65"), "share", "learned twin (what-if)")
    return out


def respond(message: str, req: GameRequest, *, use_llm: bool = True) -> dict:
    from .. import service
    message = (message or "").strip()[:500]
    st = service.game_state(req)
    preq = GameProjectionRequest(**req.model_dump(), n=60)
    proj = service.game_projection(preq) if not st["ended"] else None
    try:
        twin = service.twin_fan(preq) if not st["ended"] else None
    except Exception:  # the twin is optional evidence
        twin = None
    facts = build_facts(st, proj, twin, service.twin_model())
    action = find_action(message, st)
    if action:
        _f(facts, "action.label", "the decision you asked about", action["label"], "text", "town engine")
        if action["enabled"]:
            facts.update(what_if(req, action))
    allowed = [{"type": a["type"], "id": a["id"], "value": a["value"], "label": a["label"], "enabled": a["enabled"],
                "reason": a["reason"], "preview": a["preview"]["text"]}
               for acts in st["actions"].values() for a in acts if a["enabled"]]
    mode, rejected, answer = "rules", [], []
    if use_llm:
        try:
            answer, llm_action, rejected = llm_answer(message, facts, allowed)
            if answer:
                mode = "llm"
                action = llm_action or action
        except LLMUnavailable:
            answer = []
    if not answer:
        answer = rule_answer(message, facts, action)
    used = sorted({i for a in answer for i in a["facts"]})
    return {"answer": answer, "action": action, "mode": mode, "llm": describe(),
            "facts_used": {i: {**facts[i], "text": fmt(facts[i])} for i in used},
            "rejected_sentences": len(rejected), "disclaimer": DISCLAIMER}
