"""The learned twin in plain words, written by the AI from the twin's numbers.

The twin (HMM life states x money behaviour learned from data) answers "where do people like you usually end up?".
The town engine answers "where does the life you actually chose lead?". When the two disagree a lot, that gap is the
most useful thing to explain, so both sides become facts. As everywhere: the engines compute every number, the model
only writes sentences with {fact_id} placeholders, the claim guard checks them, and rule sentences are the fallback.
"""
from __future__ import annotations


from ..models import GameProjectionRequest
from .companion import DISCLAIMER, PLACEHOLDER, fmt, guard, _f
from .llm import LLMUnavailable, chat_json, describe
from .outcome import _clean, _facts_listing
from .timeline import _money

TWIN, TOWN = "learned twin", "town engine"
STATE = {"employed": "working", "part_time": "working reduced hours", "job_search": "looking for a job",
         "student": "studying", "retired": "retired"}


def twin_facts(req: GameProjectionRequest) -> tuple[dict, dict]:
    from .. import service
    fan = service.twin_fan(req)
    s = fan["summary"]
    facts: dict[str, dict] = {}
    _f(facts, "twin.start", "what you are doing now", STATE.get(fan["start_state"], fan["start_state"]), "text", TWIN)
    flow, off = s["flow_now_p50"], fan.get("offset_chf_per_month") or 0
    _f(facts, "twin.flow", "monthly amount for people like you right now (see twin.flow_way)", abs(flow), "chf", TWIN)
    _f(facts, "twin.flow_way", "whether that amount is saved or overspent each month",
       "put aside" if flow >= 0 else "spend more than they earn by", "text", TWIN)
    if abs(off) >= 1:
        _f(facts, "twin.offset", "how far your own monthly saving differs from people like you", abs(off), "chf", TWIN)
        _f(facts, "twin.offset_way", "do you save more or less per month than people like you", "more" if off > 0 else "less", "text", TWIN)
    if fan.get("assumed_degree_after_months"):
        _f(facts, "twin.degree_in", "how long until the twin assumes you finish studying and start working",
           fan["assumed_degree_after_months"], "months", TWIN)
    _f(facts, "twin.age", "the age the twin looks at", 65, "age", TWIN)
    if s.get("wealth_at_65"):
        for k, who in (("p10", "with bad luck"), ("p50", "typically"), ("p90", "with good luck")):
            _money(facts, f"twin.w65_{k}", f"savings at 65 for people like you, {who}", s["wealth_at_65"][k], TWIN)
    if s.get("share_lives_with_job_search_before_65") is not None:
        _f(facts, "twin.jobsearch", "share of lives with at least one job search before 65", s["share_lives_with_job_search_before_65"], "share", TWIN)
        if s.get("months_in_job_search_before_65_p50"):
            _f(facts, "twin.jobsearch_len", "typical total time spent looking for work before 65", s["months_in_job_search_before_65_p50"], "months", TWIN)
    if s.get("retirement_age_p50"):
        _f(facts, "twin.retire", "typical retirement age", round(s["retirement_age_p50"]), "age", TWIN)
    if s.get("share_lives_ever_below_zero"):
        _f(facts, "twin.below0", "share of lives where savings dip below zero at some point", s["share_lives_ever_below_zero"], "share", TWIN)

    # the town engine's path: the life as you actually chose it so far
    rows, _ = service.game_rows(req)
    m = req.current_month
    at65 = next((t for t in range(len(rows)) if rows[t]["age"] >= 65), len(rows) - 1)
    town = rows[at65]["cash"] + rows[at65]["investments"]
    _money(facts, "town.w65", "savings at 65 on the path you have chosen so far (if nothing else changes)", town, TOWN)
    studying = sum(1 for r in rows[m:at65] if r["status"] == "studying")
    ctx = {"twin65": (s.get("wealth_at_65") or {}).get("p50"), "town65": town, "stuck_studying": studying >= 12 * 15}
    if ctx["stuck_studying"]:
        _f(facts, "town.studying", "how long you would keep studying on your current path if you never choose to stop", studying, "months", TOWN)
    return facts, ctx


def _say(facts: dict, fid: str, debt_words: str = "in debt") -> str:
    return "{%s_debt} %s" % (fid, debt_words) if f"{fid}_debt" in facts else "{%s}" % fid


def twin_rules(facts: dict, ctx: dict) -> tuple[str, list[str]]:
    out = ["Right now you are {twin.start}, and people like you typically {twin.flow_way} {twin.flow} a month."]
    if "twin.degree_in" in facts:
        out.append("The twin assumes you finish studying in about {twin.degree_in} and then start working.")
    if "twin.w65_p50" in facts:
        out.append("By {twin.age}, people like you typically have about %s saved; with bad luck %s, with good luck %s."
                   % (_say(facts, "twin.w65_p50"), _say(facts, "twin.w65_p10"), _say(facts, "twin.w65_p90")))
    if "twin.jobsearch" in facts:
        out.append("About {twin.jobsearch} of these lives include a job search before {twin.age}"
                   + (", typically {twin.jobsearch_len} in total." if "twin.jobsearch_len" in facts else "."))
    if "town.studying" in facts:
        out.append("Your own timeline looks very different: you have not chosen to finish studying or start work, so it keeps "
                   "you studying for {town.studying} and ends at about %s by {twin.age}." % _say(facts, "town.w65"))
    elif "twin.w65_p50" in facts:
        out.append("On the path you have chosen so far, you would have about %s at {twin.age}." % _say(facts, "town.w65"))
    head = "Where people like you usually end up"
    return head, out


PROMPT = """You explain the "learned twin" in a financial life-simulation game (CHF, Switzerland, synthetic people).
The learned twin follows many lives of people like the player, learned from data: they finish studies, find jobs, lose
them, retire. The "town engine" path follows only the choices the player actually made. Python computed every number.
Your job is the wording: read the numbers, work out what they mean for this person, and say it like a friendly, clear person.
Rules:
- A headline (under ten words) and four to six short sentences.
- Every number, age or duration must be a placeholder {fact_id} from FACTS. The listing shows real values so you can
  understand them; you still only write placeholders. Never write digits, never invent facts.
- Money below zero means owing money: never show a negative amount. Use the matching {..._debt} placeholder with words
  like "in debt", "owe" or "short".
- If the town engine's path (town.*) is far from the twin's typical outcome, explain the difference plainly: the twin
  shows what usually happens to people like you, your timeline shows only what you chose so far. If a town.studying fact
  is present, say that your timeline keeps you studying because you have not chosen to finish or start work yet, e.g.
  "Your own timeline keeps you studying for {town.studying}, since you have not chosen to finish or start working yet."
- Ages are placeholders too: write {twin.age}, never the digits. twin.flow and twin.offset are about right now,
  not about age {twin.age}.
- twin.flow is always a positive amount: say it with twin.flow_way ("people like you typically {twin.flow_way} {twin.flow} a month"),
  and twin.offset with twin.offset_way ("you save about {twin.offset} {twin.offset_way} than them each month").
- No jargon (net worth, liquid, percentile, P10/P50/P90, HMM, GMM, regime): say "with bad luck", "typically", "with good luck".
- Never copy a fact's label; a fact with unit "text" is written as the placeholder alone.
- No certainty words (guaranteed, definitely, will be) and no causal claims (because of, causes, leads to, due to).
- These are possible outcomes under assumptions, not predictions. No personal financial advice.
Return ONLY JSON: {"headline": "...", "sentences": [{"text": "..."}]}"""


def twin_story(req: GameProjectionRequest, *, use_llm: bool = True) -> dict:
    facts, ctx = twin_facts(req)
    head_rule, rules = twin_rules(facts, ctx)
    mode, rejected, sentences, headline = "rules", 0, [], head_rule
    if use_llm:
        try:
            reply = chat_json([{"role": "system", "content": PROMPT},
                               {"role": "user", "content": f"FACTS:\n{_facts_listing(facts)}"}])
            raw = [x for x in (reply.get("sentences") or []) if isinstance(x, dict)]
            # a negative amount must be shown in its debt form: drop sentences that cite the signed one
            bad = [x for x in raw if any(f"{i}_debt" in facts for i in PLACEHOLDER.findall(str(x.get("text", ""))))]
            ok, rej = guard([x for x in raw if x not in bad], facts)
            ok = _clean(ok)
            rejected = len(rej) + len(bad)
            if len(ok) >= 3:
                sentences, mode = ok, "llm"
                h, _ = guard([{"text": str(reply.get("headline", ""))[:120]}], facts)
                headline = h[0]["text"] if h else head_rule
        except LLMUnavailable:
            pass
    if not sentences:
        sentences = guard(rules, facts)[0]
    sentences = [{**s, "text": s["text"][:1].upper() + s["text"][1:]} for s in sentences]
    used = sorted({i for s in sentences for i in s["facts"]})
    return {"headline": headline, "story": sentences, "mode": mode, "llm": describe(),
            "facts_used": {i: {**facts[i], "text": fmt(facts[i])} for i in used},
            "rejected_sentences": rejected, "disclaimer": DISCLAIMER}


_CACHE: dict[str, dict] = {}


def cached(key: str) -> dict:
    """Remembers AI-written results only: if the AI was briefly unavailable, the next request tries it again."""
    if key in _CACHE:
        return _CACHE[key]
    out = twin_story(GameProjectionRequest.model_validate_json(key))
    if out["mode"] == "llm":
        if len(_CACHE) >= 64:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = out
    return out


cached.cache_clear = _CACHE.clear  # type: ignore[attr-defined]
