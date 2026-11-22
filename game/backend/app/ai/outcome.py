"""Plain-language outcome stories: turns the engines' end-of-timeline numbers into something a person can read.

Same safety rules as the companion (see companion.py), reusing its claim guard:

  1. The engines compute everything. Here they are turned into citable FACTS with stable ids.
  2. The LLM (optional, via llm.py and the git-ignored .env) may only write sentences with {fact_id}
     placeholders, in everyday words. The guard rejects sentences with digits of their own, unknown fact ids,
     certainty or causal language. Numbers are filled in by code.
  3. Without an LLM (no key, quota, error, or too few sentences accepted) deterministic templates tell the
     same story from the same facts, so the story is always there.

Two stories: one reality (`outcome_story`) and two realities side by side (`compare_story`). They are built
into the normal responses of /compare and (once the timeline has ended) /game/projection as a `story` field.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

from ..engine.metrics import summarize
from ..models import CompareRequest, GameProjectionRequest
from .companion import DISCLAIMER, _f, fmt, guard
from .llm import LLMUnavailable, chat_json, describe

TOWN = "town engine"


# ----------------------------------------------------------------------------- one reality: facts
def outcome_facts(req: GameProjectionRequest) -> tuple[dict, dict]:
    from .. import service
    st = service.game_state(req)
    proj = service.game_projection(req)
    rows, _ = service.game_rows(req)
    m = summarize(rows)
    sc = proj["final_scores"]
    facts: dict[str, dict] = {}
    end_age = rows[-1]["age"] if rows else st["age"]
    _f(facts, "person.name", "the player's first name", st["persona"]["name"], "text", TOWN)
    _f(facts, "end.age", "age at the end of the timeline", round(end_age), "age", TOWN)
    _f(facts, "end.net_worth", "everything owned minus everything owed, at the end (this one path)",
       m["final_net_worth"], "chf", TOWN)
    for k in ("p10", "p50", "p90"):
        _f(facts, f"end.range_{k}", f"same choices in many different luck worlds: net worth at the end ({k.upper()})",
           proj["final_net_worth"][k], "chf", TOWN)
    _f(facts, "end.cash", "cash at the end", m["final_cash"], "chf", TOWN)
    _f(facts, "end.investments", "investments at the end", m["final_investments"], "chf", TOWN)
    if m["final_debt"] > 1:
        _f(facts, "end.debt", "debt at the end", m["final_debt"], "chf", TOWN)
    if m["final_home_equity"] > 1:
        _f(facts, "end.home_equity", "value of the home minus its mortgage, at the end", m["final_home_equity"], "chf", TOWN)
    runway = sc["freedom"]["runway_months"]
    if runway is not None and runway <= 600:  # beyond fifty years the number stops meaning anything
        _f(facts, "end.runway", "how many months the savings would cover everyday costs", runway, "months", TOWN)
    _f(facts, "risk.shortfall", "share of luck worlds in which cash goes below zero at some point",
       proj["share_worlds_shortfall"], "share", TOWN)
    _f(facts, "risk.low_cash", "lowest cash balance on this path", m["min_cash"], "chf", TOWN)
    if rows:
        _f(facts, "risk.low_age", "age at the lowest cash balance", round(rows[m["min_cash_month"]]["age"]), "age", TOWN)
    if m["months_negative_cash"]:
        _f(facts, "risk.negative_months", "months with cash below zero on this path", m["months_negative_cash"], "months", TOWN)
    if m["months_below_buffer"]:
        _f(facts, "risk.thin_months", "months with less cash than the safety cushion", m["months_below_buffer"], "months", TOWN)
    g = sc["goals"]
    if g["total"]:
        _f(facts, "goals.met", "goals reached", g["met"], "count", TOWN)
        _f(facts, "goals.total", "goals set", g["total"], "count", TOWN)
    ex = sc["experience"]
    if ex["months_studying"]:
        _f(facts, "life.studying", "time spent studying", ex["months_studying"], "months", TOWN)
    if ex["months_abroad"]:
        _f(facts, "life.abroad", "time spent living abroad", ex["months_abroad"], "months", TOWN)
    br = proj["bridge"]
    _f(facts, "money.earned", "total income earned over the whole path", br["income"], "chf", TOWN)
    living = -(br["expenses"] + br["lifestyle"] + br["purchases"])
    _f(facts, "money.living", "total spent on rent, living costs, habits and purchases", living, "chf", TOWN)
    if br["investment_return"] > 1:
        _f(facts, "money.growth", "total growth of investments", br["investment_return"], "chf", TOWN)
    return facts, {"ended": proj["ended"], "degree": bool(ex["degree"]), "safety": sc["safety"]["level"],
                   "debt_free": sc["freedom"]["debt_free"]}


# ----------------------------------------------------------------------------- one reality: words
def outcome_rules(facts: dict, ctx: dict) -> tuple[str, list[str]]:
    has = facts.__contains__
    ended = ctx["ended"]
    out: list[str] = []
    if has("end.range_p50"):
        out.append(("By {end.age}, " if ended else "If things carry on like this, by {end.age} ")
                   + "you end up owning about {end.range_p50} once everything you owe is taken off. "
                   "With bad luck that could be nearer {end.range_p10}; with good luck nearer {end.range_p90}.")
    if has("money.earned"):
        out.append("Over the years you earn {money.earned} and spend {money.living} on living, "
                   "so most of what you keep comes from the gap between the two.")
    if has("money.growth"):
        out.append("Your investments add about {money.growth} on top.")
    if has("end.home_equity"):
        out.append("You own part of a home worth {end.home_equity} after the mortgage.")
    if has("end.debt"):
        out.append("You still owe {end.debt} at the end.")
    if has("end.runway"):
        out.append("Your savings could pay your everyday costs for about {end.runway}.")
    if ctx["safety"] == "shortfall" and has("risk.low_age"):
        out.append("There is a rough patch: around {risk.low_age} your cash drops to {risk.low_cash}, "
                   "so you would need to borrow or cut back for a while.")
    elif ctx["safety"] == "tight" and has("risk.thin_months"):
        out.append("Money is tight at times: in total {risk.thin_months} are spent with less cash than a comfortable cushion.")
    else:
        out.append("Cash stays comfortably above zero on this path.")
    if has("risk.shortfall"):
        out.append("When luck changes, cash runs out at some point in {risk.shortfall} of the simulated lives.")
    bits = []
    if has("life.studying"):
        bits.append("studying for {life.studying}")
    if has("life.abroad"):
        bits.append("living abroad for {life.abroad}")
    if bits:
        out.append("Along the way you spend time " + " and ".join(bits) + ".")
    if has("goals.total"):
        out.append("You reach {goals.met} of your {goals.total} goals.")
    headline = {"shortfall": "A path with a rough patch.", "tight": "A path that works, but is tight in places.",
                "steady": "A steady path."}[ctx["safety"]]
    return headline, out


STORY_PROMPT = """You are the narrator inside a financial life-simulation game (CHF, Switzerland, synthetic people).
Python engines computed every number. You only retell them in everyday words for someone with no finance background.
Rules:
- Write a short headline (under ten words) and four to six short sentences, like a friend explaining how this life turned out.
- Avoid jargon: say "everything you own minus what you owe" instead of net worth, "bad luck / good luck" instead of P10/P90 or percentiles, "your savings cover N months of costs" instead of runway or liquidity.
- Every number, age or duration must be a placeholder {fact_id} from FACTS. Never write digits yourself (write "a few years" style words only if no fact fits).
- Only use fact ids that are in FACTS. Do not invent facts or mention things that are not in FACTS.
- No certainty words (guaranteed, definitely, will be) and no causal claims (because of, causes, leads to, due to).
- A fact with unit "text" already holds the words to use (e.g. a reason or a name). Write the placeholder alone, never its label next to it.
- Skip facts that are zero, or the same for both realities. Never write "0 months", "2 and 2" or other empty comparisons. Prefer what is different and what it means for the person.
- Never repeat a phrase. Do not paste a fact's label into a sentence; rephrase it naturally.
- Simulated outcomes are ranges under assumptions, not predictions. No personal financial advice; describe trade-offs.
- Money is only one part of a life: do not call a higher number a better life.
Return ONLY JSON: {"headline": "...", "sentences": [{"text": "..."}]}"""

_REPEAT = re.compile(r"\b(\w+(?: \w+){0,3}) \1\b", re.I)


def _facts_listing(facts: dict) -> str:
    return "\n".join(f"{fid}: {f['label']} = {fmt(f)}" for fid, f in facts.items())


def _clean(sentences: list[dict]) -> list[dict]:
    """Drop sentences that stutter ("investment growth investment growth"): a sign the model pasted a label."""
    return [x for x in sentences if not _REPEAT.search(x["text"])]


def _narrate(facts: dict, headline_rule: str, rule_sentences: list[str], extra: str, use_llm: bool) -> dict:
    mode, rejected, sentences, headline = "rules", 0, [], headline_rule
    if use_llm:
        try:
            reply = chat_json([{"role": "system", "content": STORY_PROMPT},
                               {"role": "user", "content": f"FACTS:\n{_facts_listing(facts)}\n\n{extra}"}])
            ok, rej = guard(reply.get("sentences") or [], facts)
            ok = _clean(ok)
            rejected = len(rej)
            if len(ok) >= 3:  # a story with fewer than three accepted sentences is not worth showing
                sentences, mode = ok, "llm"
                head_ok, _ = guard([{"text": str(reply.get("headline", ""))[:120]}], facts)
                headline = head_ok[0]["text"] if head_ok else headline_rule
        except LLMUnavailable:
            pass
    if not sentences:
        sentences = guard(rule_sentences, facts)[0]
    sentences = [{**s, "text": s["text"][:1].upper() + s["text"][1:]} for s in sentences]
    used = sorted({i for s in sentences for i in s["facts"]})
    return {"headline": headline, "story": sentences, "mode": mode, "llm": describe(),
            "facts_used": {i: {**facts[i], "text": fmt(facts[i])} for i in used},
            "rejected_sentences": rejected, "disclaimer": DISCLAIMER}


def outcome_story(req: GameProjectionRequest, *, use_llm: bool = True) -> dict:
    facts, ctx = outcome_facts(req)
    headline, rules = outcome_rules(facts, ctx)
    extra = ("The timeline has ended: tell it in the past tense." if ctx["ended"]
             else "The timeline is still running: tell it as 'if things carry on like this'.")
    return _narrate(facts, headline, rules, extra, use_llm)


# ----------------------------------------------------------------------------- two realities
COMPARE_KEYS = ("final_net_worth", "min_cash", "months_negative_cash", "months_below_buffer",
                "final_debt", "final_home_equity", "goals_met")
DRIVER_NAMES = {"income": "income earned", "expenses": "living costs", "one_offs": "one-off costs like tuition and moves",
                "interest": "interest paid", "investment_return": "investment growth",
                "lifestyle": "habits like eating out and shopping", "purchases": "purchases and trips",
                "mortgage_interest": "mortgage interest", "maintenance": "home maintenance",
                "home_fees": "home buying and selling fees", "home_appreciation": "growth in the home's value"}
UNITS = {"final_net_worth": "chf", "min_cash": "chf", "final_debt": "chf", "final_home_equity": "chf",
         "months_negative_cash": "months", "months_below_buffer": "months", "goals_met": "count"}
LABELS = {"final_net_worth": "everything owned minus owed, at the end", "min_cash": "lowest cash balance",
          "final_debt": "debt at the end", "final_home_equity": "home value minus mortgage, at the end",
          "months_negative_cash": "months with cash below zero", "months_below_buffer": "months below the safety cushion",
          "goals_met": "goals reached"}


def compare_facts(res: dict, name_a: str, name_b: str) -> tuple[dict, dict]:
    facts: dict[str, dict] = {}
    _f(facts, "a.name", "name of the first reality", name_a, "text", TOWN)
    _f(facts, "b.name", "name of the second reality", name_b, "text", TOWN)
    table = {r["key"]: r for r in res["table"]}
    for k in COMPARE_KEYS:
        r = table.get(k)
        if not r or (r["a"] == r["b"] and k != "final_net_worth"):
            continue  # identical for both: nothing to tell
        if k in ("months_negative_cash", "months_below_buffer", "final_debt", "final_home_equity") and not (r["a"] or r["b"]):
            continue
        for side in ("a", "b"):
            v = r[side]
            _f(facts, f"{side}.{k}", f"{LABELS[k]} ({'first' if side == 'a' else 'second'} reality)",
               int(v) if UNITS[k] in ("count", "months") else v, UNITS[k], TOWN)
    d = table["final_net_worth"]["diff"]
    lead = None if abs(d) < 500 else ("b" if d > 0 else "a")
    _f(facts, "gap.net_worth", "difference in what is owned at the end", abs(d), "chf", TOWN)
    drivers = [x for x in sorted(res["bridge"], key=lambda x: -abs(x["diff"])) if abs(x["diff"]) >= 500][:2]
    for i, x in enumerate(drivers):
        _f(facts, f"driver{i + 1}.name", "main reason for the gap", DRIVER_NAMES[x["key"]], "text", TOWN)
        _f(facts, f"driver{i + 1}.amount", f"size of that reason ({DRIVER_NAMES[x['key']]})", abs(x["diff"]), "chf", TOWN)
        facts[f"driver{i + 1}.name"]["favours"] = "b" if x["diff"] > 0 else "a"
    if res.get("ranges") and res["ranges"].get("a") and res["ranges"].get("b"):
        for side in ("a", "b"):
            for k in ("p10", "p90"):
                _f(facts, f"{side}.range_{k}", f"net worth at the end over many luck worlds ({k.upper()}, "
                   f"{'first' if side == 'a' else 'second'} reality)", res["ranges"][side][k], "chf", TOWN)
    return facts, {"lead": lead, "drivers": len(drivers), "ranges": "a.range_p10" in facts}


def compare_rules(facts: dict, ctx: dict) -> tuple[str, list[str]]:
    has = facts.__contains__
    out: list[str] = []
    lead = ctx["lead"]
    if lead is None:
        out.append("{a.name} and {b.name} end up in about the same place: {a.final_net_worth} and {b.final_net_worth}.")
        headline = "Two paths, about the same result."
    else:
        hi, lo = lead, ("a" if lead == "b" else "b")
        out.append(f"{{{hi}.name}} ends up owning {{gap.net_worth}} more than {{{lo}.name}}: "
                   f"{{{hi}.final_net_worth}} against {{{lo}.final_net_worth}}.")
        headline = "One path leaves more money, but money is not the whole story."
    for i in range(1, ctx["drivers"] + 1):
        word = "helps" if facts[f"driver{i}.name"]["favours"] == "b" else "holds back"
        out.append(f"{{driver{i}.name}} is a big part of the gap, about {{driver{i}.amount}}, and it {word} {{b.name}}.")
    for side in ("a", "b"):
        if has(f"{side}.months_negative_cash") and facts[f"{side}.months_negative_cash"]["value"] > 0:
            out.append(f"{{{side}.name}} runs out of cash for {{{side}.months_negative_cash}}, so it needs borrowing or cutting back.")
    if has("a.min_cash") and has("b.min_cash"):
        thin = "a" if facts["a.min_cash"]["value"] < facts["b.min_cash"]["value"] else "b"
        out.append(f"{{{thin}.name}} has the thinner safety cushion: its cash bottoms out at {{{thin}.min_cash}}.")
    if has("a.goals_met") and has("b.goals_met"):
        out.append("Goals reached: {a.goals_met} in {a.name} against {b.goals_met} in {b.name}.")
    if ctx["ranges"]:
        out.append("Across many different luck worlds, {a.name} ends between {a.range_p10} and {a.range_p90}, "
                   "and {b.name} between {b.range_p10} and {b.range_p90}.")
    return headline, out


def compare_story(req: CompareRequest, *, use_llm: bool = True, res: dict | None = None) -> dict:
    from .. import service
    res = res or service.compare_branches(req)
    facts, ctx = compare_facts(res, req.name_a, req.name_b)
    headline, rules = compare_rules(facts, ctx)
    extra = ("Compare the two realities fairly: they live in the same world, so differences come from choices, "
             "not luck. Do not say which life is better.")
    return _narrate(facts, headline, rules, extra, use_llm)


# ----------------------------------------------------------------------------- built into the normal responses
# /compare and the end-of-timeline /game/projection carry a "story" automatically. It is cached per request
# (one LLM call per distinct reality, not one per refresh) and never breaks the response it rides on.
@lru_cache(maxsize=128)
def _cached_outcome(key: str) -> dict | None:
    try:
        return outcome_story(GameProjectionRequest.model_validate_json(key))
    except Exception:
        return None


@lru_cache(maxsize=128)
def _cached_compare(key: str, res_key: str) -> dict | None:
    try:
        return compare_story(CompareRequest.model_validate_json(key), res=json.loads(res_key))
    except Exception:
        return None


def with_outcome_story(proj: dict, req: GameProjectionRequest) -> dict:
    """Ended timelines only: mid-game projections refresh on every action and must stay cheap."""
    return {**proj, "story": _cached_outcome(req.model_dump_json())} if proj.get("ended") else proj


def with_compare_story(res: dict, req: CompareRequest) -> dict:
    slim = {k: res[k] for k in ("table", "bridge", "ranges") if k in res}
    return {**res, "story": _cached_compare(req.model_dump_json(), json.dumps(slim, sort_keys=True))}
