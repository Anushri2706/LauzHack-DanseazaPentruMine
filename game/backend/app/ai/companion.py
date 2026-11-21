"""AI companion: routes one user message to extraction, decision proposal or explanation.

It never mutates game state. Proposed profile updates and decisions are returned for the user to confirm
in the UI, which then calls the ordinary (validated) game endpoints.
"""
from __future__ import annotations

from .claim_guard import build_evidence
from .explain import REFUSAL, explain
from .extract import extract, intents, propose_decision

OPTIONAL_PROFILE = ("other_expenses_chf_month", "canton", "household", "emergency_reserve_chf")


def _tax_reply(svc) -> str:
    reg = svc.base_assumptions.registry
    pts = [reg.get(a) for a in reg.rows if a.startswith("tax_rate_at_")]
    status = {a.source_status for a in pts}
    return ("There is no single 'average Swiss tax rate' in this model, and the AI cannot pick one. Income tax uses "
            f"one effective-rate curve ({len(pts)} points, ids tax_rate_at_*) for a single person without children in "
            f"Lausanne ({pts[0].geography}), labelled {', '.join(sorted(status))} in the assumption registry. "
            "If you live elsewhere or your household differs, the result carries that limitation. Which canton and "
            "household should apply? You can override curve points as your own illustrative assumptions.")


def respond(message: str, svc, game_id: str | None = None, branch_id: str | None = None,
            compare_with: str | None = None, use_llm: bool = True) -> dict:
    found = intents(message)
    ex = extract(message, use_llm=use_llm)
    reply: list[str] = []
    proposal = None
    explanation = None
    game = svc.repo.game(game_id) if game_id else None
    if "manipulation" in found and not (game is not None and branch_id):
        reply.append(REFUSAL.split(" Here is")[0])     # with a branch, the explanation carries the refusal
    if "assumption_request" in found and "tax" in message.lower():
        reply.append(_tax_reply(svc))
    if game is not None and branch_id:
        b = svc.repo.branch(game_id, branch_id)
        if ex.get("decision_candidate") and "manipulation" not in found:
            ep = svc._engine_profile(game, svc.effective_assumptions(game, branch_id))
            proposal = propose_decision(ex["decision_candidate"], initial_state=ep.profile.initial_state,
                                        horizon_months=game.config.horizon_months,
                                        assumptions=svc.effective_assumptions(game, branch_id),
                                        current_canton=game.profile.canton, min_start_month=b.fork_month)
        wants_explanation = bool({"explain_request", "certainty_request", "manipulation"} & set(found))
        if wants_explanation:
            explanation = explain_branches(svc, game_id, branch_id, compare_with, message, use_llm)
            if explanation is None:
                reply.append("Simulate this branch first; there is no completed result to explain yet.")
            else:
                reply.append(explanation["text"])
        if "missing_info" in found:
            missing = [f for f in OPTIONAL_PROFILE if getattr(game.profile, f) is None]
            reply.append("Not provided: " + (", ".join(missing) if missing else "nothing optional is missing") +
                         ". Fields left empty are not guessed; see the warnings on each result.")
    if ex["questions"]:
        reply.append(" ".join(ex["questions"]))
    if ex["proposed_profile_updates"]:
        reply.append("I read these values from your message: " +
                     ", ".join(f"{k} = {v}" for k, v in ex["proposed_profile_updates"].items()) +
                     ". They are not applied until you confirm them in the profile form.")
    rejected = [f for f in ex["facts"] if f["status"] == "rejected"]
    for f in rejected:
        reply.append(f"Not used: {f['field']} ({'; '.join(f['problems'])}).")
    if proposal:
        reply.append(f"Proposed decision: {proposal.get('name', proposal['decision_id'])} — {proposal.get('message', proposal['status'])}")
    if not reply:
        reply.append("I can read facts about you, propose a supported decision, or explain simulated results. "
                     "I don't calculate anything myself.")
    return {"reply": "\n\n".join(reply), "intents": found, "extraction": ex, "decision_proposal": proposal,
            "explanation": explanation, "labels": {"simulated_outcome": "simulated outcome",
                                                  "illustrative": "illustrative assumption",
                                                  "user": "user-provided value"}}


def explain_branches(svc, game_id: str, branch_id: str, compare_with: str | None, question: str,
                     use_llm: bool = True, draft: dict | None = None) -> dict | None:
    game = svc.repo.game(game_id)
    ids = [branch_id] + ([compare_with] if compare_with else [])
    results = []
    for bid in ids:
        r = svc.latest(game_id, bid)
        if r is None or svc.is_stale(game, r):
            return None
        results.append({"result_id": r.result_id, "branch_id": bid, "branch_name": game.branches[bid].name,
                        "summary": r.summary})
    comparison = None
    if compare_with:
        comparison = svc.compare(game_id, compare_with, branch_id)    # a = reference, b = selected branch
    sensitivity = None
    if "assumption" in (question or "").lower():
        sensitivity = svc.sensitivity(game_id, branch_id, None, "net_worth_real.final.p50", 0.10, 200)
        sensitivity["result_id"] = results[0]["result_id"]
    ev = build_evidence(results, comparison, sensitivity)
    out = explain(ev, question, draft=draft, use_llm=use_llm, horizon_months=results[0]["summary"]["horizon_months"])
    out["result_ids"] = [r["result_id"] for r in results]
    return out


__all__ = ["respond", "explain_branches"]
