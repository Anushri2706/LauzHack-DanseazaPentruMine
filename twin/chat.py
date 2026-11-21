"""Interactive Financial Twin chat.

    python3 -m twin.chat                     # new user: describe yourself, then ask what-ifs
    python3 -m twin.chat --profile P00001    # use a synthetic profile, just ask what-ifs

Type 'state' to see what the twin knows, 'quit' to exit.
Flow per message: understand (LLM + validation) -> update state -> engine -> explain (LLM + number check).
"""
import argparse
import re

from . import calibration
from .agents import explain, explain_frontier, understand
from .explore import frontier, highlights, print_front, print_per_decision
from .llm import describe
from .profile import REQUIRED, canton_rent_index, load_profile, profile_from_inputs
from .run import whatif
from .trajectory import peer_context

ASK = {"cash": "how much cash or savings you have", "income": "your net income (monthly or yearly)",
       "rent": "your monthly rent or housing cost (or ask me to use a typical rent for your city)",
       "employment": "whether you are currently studying or working"}
NAMES = {"masters": "a master's degree", "job_loss": "a job loss", "relocate": "a move",
         "part_time": "working part-time", "self_employed": "becoming self-employed",
         "family": "having a child", "explore": "a trade-off search across all decisions"}


def fmt_table(results):
    lines = []
    for r in results["rows"]:
        goals = "; ".join(f"{g} {v:.0%}" for g, v in r["goals"].items()) or "no goals"
        lines.append(f"  {r['scenario']}\n    end wealth P10 {r['nw_p10']:,.0f} | median {r['nw_p50']:,.0f} | "
                     f"P90 {r['nw_p90']:,.0f}  ·  financing needed {r['p_shortfall']:.0%}  ·  goals: {goals}")
    return "\n".join(lines)


def describe_inputs(inputs):
    return ", ".join(f"{k} {v:,.0f}" if isinstance(v, (int, float)) else f"{k} {v}"
                     for k, v in inputs.items()
                     if k not in ("goals", "loans", "estimated") and not k.startswith("_")) + \
        ("" if "goals" not in inputs else f", goal {inputs['goals'][0]['name']} "
         f"CHF {inputs['goals'][0]['target']:,.0f} in {inputs['goals'][0]['month']} months") + \
        ("" if "loans" not in inputs else f", debt CHF {inputs['loans'][0][0]:,.0f}")


def unusual(state):
    """Inputs that are possible but unusual enough to confirm before simulating."""
    found = []
    inc, rent, cash = state.get("income"), state.get("rent"), state.get("cash")
    if inc is not None and inc > 20000:
        found.append(("income", f"your net income is CHF {inc:,.0f} a month (CHF {inc * 12:,.0f} a year)"))
    if inc is not None and rent is not None and rent >= inc:
        found.append(("rent", f"your rent (CHF {rent:,.0f}) is at least your whole income (CHF {inc:,.0f})"))
    if cash is not None and cash > 2_000_000:
        found.append(("cash", f"you have CHF {cash:,.0f} in savings"))
    return found


def _estimate_rent(state, sess, out):
    canton = sess.get("params", {}).get("to_canton") or state.get("canton")
    if not canton:
        out.append("To estimate rent I need to know which city or canton you'll live in.")
        return
    value = round(canton_rent_index()[canton])
    state["rent"] = value
    state.setdefault("estimated", {})["rent"] = (f"median rent in {canton} across synthetic profiles "
                                                 "(not official data, not student-specific)")
    out.append(f"Using CHF {value:,}/month as your rent: the median rent in {canton} in the synthetic data. "
               "This is not official data and not specific to students; tell me a number to replace it.")


def run_explore(p, message):
    """Trade-off search + explanation. Falls back to the plain tables without an LLM."""
    import contextlib
    import io
    from .run import warnings_for
    front, ref, per = frontier(p)
    text, flags = explain_frontier(p, front, ref, per, highlights(front), warnings_for(p, "explore"), message)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_front(front, ref, limit=8)
        print_per_decision(per)
    if text is None:
        return buf.getvalue() + f"(explanation unavailable: {flags[0]})"
    if flags:
        return (text + f"\n(check: these numbers in the explanation do not match the engine output: "
                f"{', '.join(flags)} - see the table below)\n" + buf.getvalue())
    return text


def handle(message, state, fixed_profile=None, paths=300):
    """One turn. Returns the reply text. `state` (known inputs + session) is updated in place."""
    sess = state.setdefault("_session", {})
    known = None if fixed_profile else {k: v for k, v in state.items() if not k.startswith("_")}
    current = sess.get("scenario") or (sess.get("last") or {}).get("scenario")
    if current:
        known = dict(known or {}, current_what_if=current,
                     **(sess.get("params") or (sess.get("last") or {}).get("params") or {}))
    u = understand(message, known=known, awaiting=sess.get("awaiting"))
    out = []

    # Remember the what-if until it can be run. After a run, keep it as "last" so follow-ups
    # ("I'd be studying in Zurich", "my rent is 1000") update it and re-run it.
    if not sess.get("scenario") and sess.get("last"):
        sess["scenario"], sess["params"] = sess["last"]["scenario"], dict(sess["last"]["params"])
        rerun_only = True  # becomes a real request only if something changes below
    else:
        rerun_only = False
    changed = False
    if u["scenario"] == "relocate" and sess.get("scenario") == "masters":
        new = dict(sess.get("params", {}), **u["params"])  # a move during a master's = study there
        changed = new != sess.get("params")
        sess["params"] = new
    elif u["scenario"] != "none":
        changed = (u["scenario"], u["params"]) != (sess.get("scenario"), sess.get("params"))
        sess["scenario"], sess["params"] = u["scenario"], u["params"]
    elif u["params"].get("to_canton") and sess.get("scenario") in ("masters", "relocate"):
        sess["params"] = dict(sess.get("params", {}), to_canton=u["params"]["to_canton"])
        changed = True

    # Code guard: "student" only if the message says so (models infer it from "I want a master's").
    if u["inputs"].get("employment") == "student" and not re.search(
            r"\b(student|studying|i study|enrolled|at (uni|university|school|college))\b", message, re.I):
        u["inputs"].pop("employment")
    if u["inputs"] and not fixed_profile:
        new = {k: v for k, v in u["inputs"].items() if state.get(k) != v}  # only what changed
        if "rent" in new:
            state.get("estimated", {}).pop("rent", None)
        state.update(new)
        if new:
            out.append(f"Noted: {describe_inputs(new)}.")
            changed = True
    # Only estimate when the user actually asked for a typical value (the model can be over-eager).
    asked = re.search(r"\b(average|typical|usual|normal|median|estimat\w*)\b", message, re.I)
    if "rent" in u.get("estimates", []) and asked and not fixed_profile:
        _estimate_rent(state, sess, out)
        changed = True
    if rerun_only and not changed:
        sess.pop("scenario", None)
        sess.pop("params", None)
    for prob in u["problems"]:
        out.append(f"(note: {prob})")

    # Confirm values that would dominate every result if they were a typo.
    if not fixed_profile:
        confirmed = sess.setdefault("confirmed", set())
        if sess.get("confirm") and re.search(r"\b(yes|yep|yeah|correct|right|confirm\w*|that'?s right)\b",
                                             message, re.I):
            confirmed.add(sess.pop("confirm"))
        elif sess.get("confirm") and sess["confirm"] in u["inputs"]:
            sess.pop("confirm")  # user gave a corrected value; it is checked again below
        for field, why in unusual(state):
            if field not in confirmed:
                sess["confirm"], sess["awaiting"] = field, [field]
                out.append(f"Just to check: {why}. Is that right? Reply yes, or give the correct amount.")
                return "\n".join(out)

    if fixed_profile:
        p = fixed_profile
    else:
        needed = list(REQUIRED) + (["employment"] if sess.get("scenario") == "masters" else [])
        missing = [k for k in needed if k not in state]
        sess["awaiting"] = missing
        if missing:
            if sess.get("scenario"):
                out.append(f"I'll simulate {NAMES[sess['scenario']]} once I know a bit more.")
            out.append("I still need " + " and ".join(ASK[k] for k in missing) + ".")
            return "\n".join(out)
        p = calibration.apply(peer_context(profile_from_inputs(
            {k: v for k, v in state.items() if not k.startswith("_")})))

    if not sess.get("scenario"):
        out.append("Ask me a what-if: a master's, part-time work, becoming self-employed, having a child, "
                   "losing your job for a while, moving to another canton - or ask me to explore all options.")
        return "\n".join(out)

    key, params = sess.pop("scenario"), sess.pop("params", {})
    sess["last"] = {"scenario": key, "params": params}
    if key == "explore":
        out.append("Searching plans across all decisions (NSGA-II, a few seconds)...")
        out.append(run_explore(p, message))
        return "\n".join(out)
    out.append(f"{'Re-running' if rerun_only else 'Simulating'} {NAMES[key]} {params or ''} against your baseline "
               f"over {paths} possible futures...")
    results = whatif(p, key, params, paths)
    text, flags = explain(p, results, message)
    if text is None:
        out.append(fmt_table(results))
        out.append(f"(explanation unavailable: {flags[0]})")
    else:
        out.append(text)
        if flags:
            out.append(f"(check: these numbers in the explanation do not match the engine output: "
                       f"{', '.join(flags)} - see the table below)")
            out.append(fmt_table(results))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", help="use a synthetic profile instead of describing yourself")
    ap.add_argument("--paths", type=int, default=300)
    a = ap.parse_args(argv)
    fixed = calibration.apply(peer_context(load_profile(a.profile))) if a.profile else None
    print(f"Financial Twin chat  ·  language model: {describe()}")
    print("Illustrative simulations on synthetic data. Not financial advice. Type 'quit' to exit.\n")
    if fixed:
        print(f"Using profile {fixed['id']} {fixed['name']}, {fixed['age']}, {fixed['employment']}. Ask a what-if.")
    else:
        print("Tell me about your situation: savings, income, rent, age, job... then ask a what-if.")
    state = {}
    while True:
        try:
            msg = input("\nyou> ").strip()
        except EOFError:
            break
        if not msg:
            continue
        if msg.lower() in ("quit", "exit"):
            break
        if msg.lower() == "state":
            print(f"twin> {describe_inputs(state) or 'nothing yet'}")
            continue
        print("twin> " + handle(msg, state, fixed, a.paths).replace("\n", "\n      "))


if __name__ == "__main__":
    main()
