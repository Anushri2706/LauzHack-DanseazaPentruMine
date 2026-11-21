"""LLM agent: a function-calling loop over TwinSession tools.

"Let AI handle language; let a financial model handle the maths." The model only chooses tools
and narrates their JSON outputs; every number in its answer must come from a tool result.

The API key is read from the environment (OPENAI_API_KEY); never commit keys. The client is
any object exposing the OpenAI ``chat.completions.create`` interface, so a fake client can be
injected for tests or another provider adapter can be swapped in.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable

from .twin_session import DECISIONS, TwinSession
from .types import STATES

SYSTEM_PROMPT = """You are the explanation layer of a financial twin (educational simulator, CHF, Swiss context).
Rules:
- Never compute, estimate or invent numbers. Every figure you state must come from a tool result.
- To answer a what-if: create or reuse a scenario, apply_decision / set_assumption, simulate, then compare against baseline.
- Results are simulations under stated assumptions, not forecasts or real-world probabilities. Say so.
- Do not give personalised financial advice or guarantees; describe trade-offs instead.
- When a fact is missing (e.g. tuition, salary), ask the user or state the labelled assumption you used.
- Mention which assumptions matter most when relevant (call sensitivity)."""

_SID = {"type": "string", "description": "Scenario id, e.g. 'baseline' or 'masters'."}

TOOL_SCHEMAS: list[dict] = [
    {"type": "function", "function": {
        "name": "get_state", "description": "Profile, goals, scenarios, settings, decision kinds and assumption names.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "create_scenario", "description": "Create a new scenario, optionally copying another.",
        "parameters": {"type": "object", "properties": {
            "scenario_id": _SID, "name": {"type": "string"}, "copy_from": {"type": "string"}},
            "required": ["scenario_id"]}}},
    {"type": "function", "function": {
        "name": "apply_decision",
        "description": "Apply a decision/life event to a scenario. Kinds: " + "; ".join(f"{k}: {v}" for k, v in DECISIONS.items()),
        "parameters": {"type": "object", "properties": {
            "scenario_id": _SID,
            "kind": {"type": "string", "enum": sorted(DECISIONS)},
            "start_month": {"type": "integer", "minimum": 0},
            "params": {"type": "object", "description": "Kind-specific parameters (see description).",
                       "additionalProperties": True}},
            "required": ["scenario_id", "kind", "start_month"]}}},
    {"type": "function", "function": {
        "name": "set_assumption", "description": "Override a named assumption (hazard, income parameter, savings_rate_<state>) for one scenario.",
        "parameters": {"type": "object", "properties": {
            "scenario_id": _SID, "name": {"type": "string"}, "value": {"type": "number"}},
            "required": ["scenario_id", "name", "value"]}}},
    {"type": "function", "function": {
        "name": "simulate", "description": "Run the scenario; returns a compact summary (cash quantiles, risk of breaching reserve, regime probabilities, goals).",
        "parameters": {"type": "object", "properties": {"scenario_id": _SID}, "required": ["scenario_id"]}}},
    {"type": "function", "function": {
        "name": "compare", "description": "Difference b minus a between two scenarios.",
        "parameters": {"type": "object", "properties": {"a": _SID, "b": _SID}, "required": ["a", "b"]}}},
    {"type": "function", "function": {
        "name": "sensitivity", "description": "Rank assumptions by their effect on median closing cash (one-at-a-time ±25%).",
        "parameters": {"type": "object", "properties": {
            "scenario_id": _SID, "names": {"type": "array", "items": {"type": "string"}}},
            "required": ["scenario_id"]}}},
    {"type": "function", "function": {
        "name": "sample_paths", "description": f"Representative simulated life paths as regime segments ({', '.join(STATES)}).",
        "parameters": {"type": "object", "properties": {
            "scenario_id": _SID, "n": {"type": "integer", "minimum": 1, "maximum": 5}},
            "required": ["scenario_id"]}}},
]


def dispatch_tool(session: TwinSession, name: str, args: dict) -> dict:
    """Execute one tool call against the session. Errors are returned to the model, not raised."""
    handlers: dict[str, Callable[..., Any]] = {
        "get_state": session.get_state,
        "create_scenario": session.create_scenario,
        "apply_decision": lambda scenario_id, kind, start_month=0, params=None:
            session.apply_decision(scenario_id, kind, start_month, **(params or {})),
        "set_assumption": session.set_assumption,
        "simulate": session.simulate,
        "compare": session.compare,
        "sensitivity": session.sensitivity,
        "sample_paths": session.sample_paths,
    }
    if name not in handlers:
        return {"error": f"Unknown tool {name}"}
    try:
        return handlers[name](**args)
    except (ValueError, KeyError, TypeError) as exc:
        return {"error": str(exc)}


def default_client():
    """OpenAI client from OPENAI_API_KEY. Imported lazily so the backend runs without it."""
    from openai import OpenAI  # noqa: PLC0415
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in the environment (never commit it).")
    return OpenAI()


def run_agent(session: TwinSession, user_message: str, history: list[dict] | None = None,
              client=None, model: str | None = None, max_steps: int = 8) -> tuple[str, list[dict], list[dict]]:
    """One user turn. Returns (assistant_text, updated_history, tool_trace).

    tool_trace records every call with arguments and result, for the judges' "calculation trace".
    """
    client = client or default_client()
    model = model or os.environ.get("TWIN_LLM_MODEL", "gpt-4o-mini")
    messages = history or [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": user_message})
    trace: list[dict] = []
    for _ in range(max_steps):
        resp = client.chat.completions.create(model=model, messages=messages, tools=TOOL_SCHEMAS)
        msg = resp.choices[0].message
        calls = getattr(msg, "tool_calls", None) or []
        messages.append({"role": "assistant", "content": msg.content or "",
                         **({"tool_calls": [c.model_dump() if hasattr(c, "model_dump") else c for c in calls]} if calls else {})})
        if not calls:
            return msg.content or "", messages, trace
        for call in calls:
            args = json.loads(call.function.arguments or "{}")
            result = dispatch_tool(session, call.function.name, args)
            trace.append({"tool": call.function.name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, default=str)})
    # TODO(team): on hitting max_steps, ask the model for a final answer without tools.
    return "Stopped after the maximum number of tool steps.", messages, trace
