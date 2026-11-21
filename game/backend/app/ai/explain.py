"""Explanations of completed results. The LLM may only pick and order claims; claim_guard renders them.

``explain`` never computes a financial number. Inputs are completed, versioned results (by result id),
an optional exact comparison and optional sensitivity rows, all produced by the engine.
"""
from __future__ import annotations

import logging
import re

from .claim_guard import DISCLAIMER, build_evidence, check_claims, check_text, fallback_claims, render
from .extract import intents
from .llm import LLMUnavailable, chat_json
from .prompts import EXPLAIN_PROMPT

log = logging.getLogger("game.ai.explain")

REFUSAL = ("I can't change assumptions or results to reach a particular conclusion, and I can't state a claim "
           "the calculations do not support. Here is what the simulation does show under its stated assumptions:")
CERTAINTY_NOTE = ("Nobody can say what your net worth will definitely be. The game simulates many possible futures "
                  "under stated assumptions; the figures below describe that simulated distribution, not a forecast.")


def _horizon_note(question: str, horizon_months: int | None) -> str | None:
    if not horizon_months:
        return None
    m = re.search(r"(\d+)\s*(year|yr|month)s?", question.lower())
    if not m:
        return None
    asked = int(m.group(1)) * (12 if m.group(2) in ("year", "yr") else 1)
    if asked > horizon_months:
        return (f"You asked about {m.group(0)}, but these results only simulate {horizon_months} months "
                f"({horizon_months / 12:g} years); nothing is simulated beyond that.")
    return None


def _llm_claims(evidence: dict, question: str) -> tuple[list | None, str | None]:
    import json
    compact = {fid: {k: v for k, v in f.items() if k in ("kind", "metric", "metric_key", "stat", "unit", "basis",
                                                          "horizon_months", "direction", "branch", "a_name", "b_name",
                                                          "source_status", "assumption_id")}
               for fid, f in evidence["facts"].items()}
    raw = chat_json([{"role": "system", "content": EXPLAIN_PROMPT},
                     {"role": "user", "content": json.dumps({"question": question, "evidence": compact})}])
    return raw.get("claims"), raw.get("framing")


def explain(evidence: dict, question: str = "", draft: dict | None = None, use_llm: bool = True,
            horizon_months: int | None = None) -> dict:
    found = intents(question or "")
    preface = []
    if "manipulation" in found:
        preface.append(REFUSAL)
    if "certainty_request" in found:
        preface.append(CERTAINTY_NOTE)
    note = _horizon_note(question or "", horizon_months)
    if note:
        preface.append(note)
    claims, framing, source = None, None, "deterministic"
    if draft is not None:
        claims, framing, source = draft.get("claims"), draft.get("framing"), "client_draft"
    elif use_llm and "manipulation" not in found:
        try:
            claims, framing = _llm_claims(evidence, question)
            source = "llm"
        except LLMUnavailable as e:
            log.info("LLM unavailable for explanation: %s", type(e).__name__)
    accepted, rejected = check_claims(claims, evidence) if claims is not None else ([], [])
    used_fallback = not accepted
    if used_fallback:
        accepted = fallback_claims(evidence)
    sentences = [render(c, evidence) for c in accepted]
    framing_ok = check_text(framing)
    return {"preface": preface, "framing": framing_ok, "sentences": sentences, "accepted_claims": accepted,
            "rejected_claims": rejected, "source": source, "used_deterministic_fallback": used_fallback,
            "disclaimer": DISCLAIMER,
            "text": " ".join(preface + ([framing_ok] if framing_ok else []) + sentences + [DISCLAIMER])}


__all__ = ["build_evidence", "explain"]
