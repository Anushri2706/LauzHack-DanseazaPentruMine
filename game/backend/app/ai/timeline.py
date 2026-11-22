"""AI insights anchored to moments on the life timeline.

The code, not the model, decides WHICH moments deserve a note, straight from engine data:
  * each big decision you made (with what it changed, measured against the same life without it),
  * the tightest point for cash, retirement, stress-test shocks, and where the possible futures fan out,
  * what the learned twin says about people like you at 65.
The model only writes the plain-language explanation of each moment. It may cite just that moment's own facts
(placeholders {d1.delta10} and so on), and the same claim guard as the companion applies: no digits of its own, no
unknown facts, no certainty or causal language. A moment the model gets wrong falls back to a rule template, so
every pin on the timeline always has an explanation, and every number in it comes from an engine.
"""
from __future__ import annotations


from ..engine.metrics import monthly_costs
from ..models import GameProjectionRequest, GameRequest
from .companion import DISCLAIMER, PLACEHOLDER, _f, fmt, guard
from .llm import LLMUnavailable, chat_json, describe
from .outcome import _REPEAT, _facts_listing

TOWN, TWIN = "town engine", "learned twin"
MAX_DECISIONS, MAX_PINS = 4, 7
BIG = ("life_choice", "buy_home", "sell_home", "take_loan", "set_savings_rate")
PHRASE = {"start_work": "start working", "master_ft": "start a full-time master's", "master_pt": "start a part-time master's",
          "move_abroad": "move abroad", "return_home": "come back home", "drop_out": "drop out",
          "graduate": "finish your degree", "quit_school": "leave school without a degree", "quit_job": "quit your job",
          "reduce_hours": "cut down to 80% hours", "full_hours": "go back to full hours"}
PHRASE_TYPE = {"buy_home": "buy a home", "sell_home": "sell your home", "take_loan": "take out a loan",
               "set_savings_rate": "change how much you save"}
SHOCKS = {"income_loss": "lose your income", "market_drop": "see the markets fall"}


def _money(facts: dict, fid: str, label: str, v: float, source: str = TOWN) -> bool:
    """A CHF amount that may be below zero. Below zero it also gets {fid_debt}: the same amount as a positive debt,
    so the model (and the rules) can say "about CHF 885'000 in debt" instead of "CHF -885'000". Returns True if a debt."""
    _f(facts, fid, label, v, "chf", source)
    if v < 0:
        _f(facts, f"{fid}_debt", f"{label}, as a debt (how much is owed)", -v, "chf", source)
    return v < 0


def _phrase(a) -> str:
    return PHRASE.get(a.id or "", "") if a.type == "life_choice" else PHRASE_TYPE.get(a.type, a.type)


def _without(req: GameRequest, month: int, idx: int) -> GameRequest:
    """The same reality with one action removed (later choices are replayed)."""
    data = req.model_dump()
    for t in data["turns"]:
        if t["month"] == month:
            t["actions"] = [a for i, a in enumerate(t["actions"]) if i != idx]
    data["turns"] = [t for t in data["turns"] if t["actions"]]
    return GameRequest(**{k: v for k, v in data.items() if k in GameRequest.model_fields})


# ----------------------------------------------------------------------------- anchors (moments) and their facts
def build(req: GameProjectionRequest) -> tuple[dict, list[dict], str]:
    """Facts, anchors and a fallback headline. An anchor is {id, kind, month, age, title, rule} (rule = template text)."""
    from .. import service
    rows, _ = service.game_rows(req)
    proj = service.game_projection(req)
    H, m = len(rows), req.current_month
    facts: dict[str, dict] = {}
    anchors: list[dict] = []
    age = lambda t: round(rows[max(0, min(t, H - 1))]["age"])  # noqa: E731
    nw = [r["net_worth"] for r in rows]

    # decisions you made, and what each one changed in this same world
    picks = [(t.month, i, a) for t in sorted(req.turns, key=lambda t: t.month) for i, a in enumerate(t.actions) if a.type in BIG]
    picks = sorted(sorted(picks, key=lambda x: (x[2].type != "life_choice", x[0]))[:MAX_DECISIONS])
    for n, (month, idx, a) in enumerate(picks, 1):
        aid = f"d{n}"
        _f(facts, f"{aid}.age", "your age at that choice", age(month), "age", TOWN)
        _f(facts, f"{aid}.label", "what you chose", _phrase(a), "text", TOWN)
        rule = "At {%s.age} you {%s.label}." % (aid, aid)
        try:
            alt, _ = service.game_rows(_without(req, month, idx))
            at10 = month + 120
            if at10 < H and abs(nw[at10] - alt[at10]["net_worth"]) >= 500:
                d = nw[at10] - alt[at10]["net_worth"]
                _f(facts, f"{aid}.delta10", "difference in what you own ten years later, with vs without this choice", abs(d), "chf", TOWN)
                _f(facts, f"{aid}.direction", "do you own more or less ten years later because of this choice", "more" if d > 0 else "less", "text", TOWN)
                rule += " Ten years on, you own about {%s.delta10} %s than you would have without it." % (aid, "more" if d > 0 else "less")
            else:
                d = nw[-1] - alt[-1]["net_worth"]
                if abs(d) >= 500:
                    _f(facts, f"{aid}.delta", "difference in what you own at the end, with vs without this choice", abs(d), "chf", TOWN)
                    _f(facts, f"{aid}.direction", "do you own more or less at the end because of this choice", "more" if d > 0 else "less", "text", TOWN)
                    rule += " By the end of the timeline you own about {%s.delta} %s than without it." % (aid, "more" if d > 0 else "less")
                else:
                    rule += " It makes little difference to your money."
        except Exception:  # a counterfactual the engine refuses simply gives no comparison
            pass
        anchors.append({"id": aid, "kind": "decision", "month": month, "age": age(month),
                        "title": f"You {_phrase(a)}", "rule": rule})

    # the tightest point for cash, only when cash really goes below zero (spare cash is invested, so "cash
    # near zero" alone is normal)
    lo = min(range(H), key=lambda t: rows[t]["cash"])
    tight = rows[lo]["cash"] < 0
    if tight:
        _f(facts, "low.age", "age at the lowest cash balance", age(lo), "age", TOWN)
        _money(facts, "low.cash", "the lowest cash balance", rows[lo]["cash"])
        anchors.append({"id": "low", "kind": "low", "month": lo, "age": age(lo), "title": "Cash runs out", "rule":
                        "Money is tightest around {low.age}, when you would be about {low.cash_debt} short on cash. "
                        "That gap has to be borrowed or saved somewhere else."})

    # retirement
    ret = next((t for t in range(H) if rows[t]["status"] == "retired"), None)
    if ret is not None:
        liquid, costs = rows[ret]["cash"] + rows[ret]["investments"], monthly_costs(rows[ret])
        _f(facts, "ret.age", "age at retirement", age(ret), "age", TOWN)
        if _money(facts, "ret.liquid", "savings and investments at retirement", liquid):
            rule = "Around {ret.age} you retire, but instead of savings you would be about {ret.liquid_debt} in debt."
        else:
            rule = "Around {ret.age} you retire, with about {ret.liquid} put aside."
        if costs > 0 and 0 < liquid / costs <= 600:
            _f(facts, "ret.runway", "how long those savings would cover everyday costs", liquid / costs, "months", TOWN)
            rule += " They could cover your everyday costs for about {ret.runway}."
        anchors.append({"id": "ret", "kind": "retire", "month": ret, "age": age(ret), "title": "Retirement", "rule": rule})

    if not proj["ended"]:
        calm = proj["no_surprises"]["net_worth"]
        for kind, who in (("income_loss", "shock"), ("market_drop", "mkt")):
            g = next((x for x in proj["ghosts"] if x["kind"] == kind), None)
            if g and abs(g["net_worth"][-1] - calm[-1]) >= 500:
                d = g["net_worth"][-1] - calm[-1]
                _f(facts, f"{who}.age", "age of the shock", age(g["month"]), "age", TOWN)
                _f(facts, f"{who}.delta", f"change in what you own at the end if you {SHOCKS[kind]}", abs(d), "chf", TOWN)
                _f(facts, f"{who}.direction", "does that leave you with more or less", "more" if d > 0 else "less", "text", TOWN)
                anchors.append({"id": who, "kind": "stress", "month": g["month"], "age": age(g["month"]),
                                "title": f"If you {SHOCKS[kind]}", "rule":
                                    "If you were to %s around {%s.age}, you would end about {%s.delta} %s." %
                                    (SHOCKS[kind], who, who, "lower" if d < 0 else "higher")})
        band = proj["bands"]["net_worth"]
        t65 = min(H - 1, ret if ret is not None else m + 12 * 20)
        i = max(0, min(t65 - m - 1, len(band["p50"]) - 1))
        if t65 > m:
            _f(facts, "fan.age", "age where the outcomes are compared", age(t65), "age", TOWN)
            names = {"p10": "with bad luck", "p50": "in a typical world", "p90": "with good luck"}
            debt = {k: _money(facts, f"fan.{k}", f"what you would own after debts there, {names[k]}", band[k][i]) for k in names}
            say = lambda k: "{fan.%s_debt} in debt" % k if debt[k] else "{fan.%s}" % k  # noqa: E731
            anchors.append({"id": "fan", "kind": "uncertainty", "month": t65, "age": age(t65), "title": "Where outcomes fan out",
                            "rule": "By {fan.age} luck matters a lot: with bad luck you might be at %s, with good luck at %s, "
                                    "and typically at %s." % (say("p10"), say("p90"), say("p50"))})
        try:
            twin = service.twin_fan(req)["summary"]
            if twin.get("wealth_at_65"):
                at = min(H - 1, round((65 - rows[0]["age"]) * 12))
                _f(facts, "twin.age", "age the learned twin looks at", 65, "age", TWIN)
                for k in ("p10", "p50", "p90"):
                    _money(facts, f"twin.{k}", "savings that people like you have at 65, " +
                           {"p10": "the less lucky ones", "p50": "the typical one", "p90": "the luckier ones"}[k], twin["wealth_at_65"][k], TWIN)
                anchors.append({"id": "twin", "kind": "twin", "month": at, "age": 65, "title": "What people like you tend to have",
                                "rule": "People with a life like yours typically have about {twin.p50} saved by {twin.age}; "
                                        "the less lucky around {twin.p10}, the luckier around {twin.p90}."})
        except Exception:  # the learned twin is optional evidence
            pass

    # the pins: decisions and the tightest moment first, then the rest, shown in time order
    rank = {"decision": 0, "low": 1, "retire": 2, "stress": 3, "uncertainty": 4, "twin": 5}
    anchors = sorted(sorted(anchors, key=lambda a: rank[a["kind"]])[:MAX_PINS], key=lambda a: a["month"])
    head = "Money gets tight around {low.age}." if tight else "No money crunch on this path."
    return facts, anchors, head


# ----------------------------------------------------------------------------- the model's part
PROMPT = """You are the narrator on a life timeline in a financial life-simulation game (CHF, Switzerland, synthetic people).
Python engines computed every number and picked the moments. Your job is the wording: look at each moment's numbers,
work out what they mean for this person, and say it the way a friendly, clear person would say it out loud.
Rules:
- For EACH moment (id) write one or two short sentences (under 35 words in total). Lead with what matters most.
- Every number, age or duration must be a placeholder {fact_id} from THAT moment's own facts (ids start with the
  moment's id and a dot). Never write digits yourself, never use another moment's facts, never invent facts.
- The listing shows each fact's real value so you can understand it; you still only write the placeholder.
- Money below zero means the person owes money. Never show a negative amount: use the matching {..._debt}
  placeholder and say it with words like "in debt", "owe" or "short". Example: "By {ret.age} you'd be about {ret.liquid_debt} in debt rather than having savings."
- A direction fact (more / less) means a gain or a loss: say plainly which, e.g. "Ten years on you'd own about {d1.delta10} {d1.direction} than without it." Never hide a loss.
  The direction placeholder is already the word "more" or "less": put it right after the amount and do not add
  "drop", "rise", "gain" or "lose" as well. Stress-test example: "If your income stopped around {shock.age}, you could end up with about {shock.delta} {shock.direction}."
- Write like a person, not a report: never copy a fact's label; use short everyday words ("put aside", "after paying off debts",
  "with bad luck", "typically"). No jargon (net worth, liquid, percentile, P10, P50, P90, simulation worlds).
- Vary your sentence openings: do not start every moment with "At age".
- A fact with unit "text" already holds the words: write the placeholder alone, never its label.
- No certainty words (guaranteed, definitely, will be) and no causal claims (because of, causes, leads to, due to).
- These are possible outcomes under assumptions, not predictions. No personal financial advice.
Also write a one-sentence headline (under 15 words) for the whole timeline, using facts of any moment.
Return ONLY JSON: {"headline": "...", "insights": [{"id": "...", "text": "..."}]}"""


def _ok_for(anchor_id: str, text: str, facts: dict) -> dict | None:
    ids = PLACEHOLDER.findall(text)
    if any(i.split(".")[0] != anchor_id for i in ids) or _REPEAT.search(PLACEHOLDER.sub("", text)):
        return None
    if any(f"{i}_debt" in facts for i in ids):   # a negative amount shown as "CHF -885'061": must use the debt form
        return None
        return None
    ok, _ = guard([{"text": text}], facts)
    return ok[0] if ok else None


def timeline_insights(req: GameProjectionRequest, *, use_llm: bool = True) -> dict:
    facts, anchors, head_rule = build(req)
    by: dict[str, dict] = {}
    headline, mode, rejected = head_rule, "rules", 0
    if use_llm and anchors:
        listing = "\n".join(
            f"MOMENT {a['id']} ({a['kind']}, age {a['age']}): {a['title']}\n" +
            "\n".join(f"  {fid}: {f['label']} = {fmt(f)}" for fid, f in facts.items() if fid.split('.')[0] == a["id"])
            for a in anchors)
        try:
            reply = chat_json([{"role": "system", "content": PROMPT},
                               {"role": "user", "content": f"MOMENTS AND FACTS:\n{listing}\n\nALL FACTS FOR THE HEADLINE:\n{_facts_listing(facts)}"}])
            for x in reply.get("insights") or []:
                aid = str(x.get("id", ""))
                got = _ok_for(aid, str(x.get("text", "")), facts) if any(a["id"] == aid for a in anchors) else None
                if got:
                    by[aid] = {**got, "by": "llm"}
                else:
                    rejected += 1
            h, _ = guard([{"text": str(reply.get("headline", ""))[:140]}], facts)
            if h and not _REPEAT.search(h[0]["text"]):
                headline = h[0]["text"]
            if by:
                mode = "llm"
        except LLMUnavailable:
            pass
    out = []
    for a in anchors:
        if a["id"] not in by:
            by[a["id"]] = {**guard([{"text": a["rule"]}], facts)[0][0], "by": "rules"}
        t = by[a["id"]]
        out.append({"id": a["id"], "kind": a["kind"], "month": a["month"], "age": a["age"], "title": a["title"],
                    "text": t["text"][:1].upper() + t["text"][1:], "by": t["by"], "facts": t["facts"]})
    if headline == head_rule:
        headline = guard([{"text": head_rule}], facts)[0][0]["text"]
    used = sorted({i for x in out for i in x["facts"]})
    return {"headline": headline, "insights": out, "mode": mode, "llm": describe(), "month": req.current_month,
            "facts_used": {i: {**facts[i], "text": fmt(facts[i])} for i in used},
            "rejected_sentences": rejected, "disclaimer": DISCLAIMER}


_CACHE: dict[str, dict] = {}


def cached(key: str) -> dict:
    """Remembers AI-written results only: if the AI was briefly unavailable, the next request tries it again."""
    if key in _CACHE:
        return _CACHE[key]
    out = timeline_insights(GameProjectionRequest.model_validate_json(key))
    if out["mode"] == "llm":
        if len(_CACHE) >= 64:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = out
    return out


cached.cache_clear = _CACHE.clear  # type: ignore[attr-defined]
