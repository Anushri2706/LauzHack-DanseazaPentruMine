"""Minimal LLM client for any OpenAI-compatible chat endpoint (standard library only).

Copied unchanged from feature/combined-engine (game/backend/app/ai/llm.py), originally twin/llm.py. Keys come from environment variables or the repo .env
(git-ignored); they are never logged or stored. Calls are capped per process (LLM_MAX_CALLS).

Provider and model come from .env:
    LLM_PROVIDER = gemini | groq | ollama        (default gemini)
    LLM_MODEL    = optional override of the model name

The LLM only handles language. It never does the financial calculations.
"""
import json
import os
import re
import urllib.error
import urllib.request

from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[4] / ".env"


def load_env(path=ENV_FILE):
    """Read KEY=VALUE lines from the repo .env; values already in the environment win (twin/env.py)."""
    if os.environ.get("GAME_DISABLE_LLM") == "1" or not Path(path).exists():
        return
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip().strip('"').strip("'")
        if value and name not in os.environ:
            os.environ[name] = value

PROVIDERS = {
    "gemini": {"url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
               # tried in order; the next one is used if a model is overloaded (503) or unavailable (404)
               "key": "GEMINI_API_KEY", "model": "gemini-flash-lite-latest,gemini-flash-latest"},
    "groq": {"url": "https://api.groq.com/openai/v1/chat/completions",
             "key": "GROQ_API_KEY", "model": "openai/gpt-oss-20b"},
    "ollama": {"url": "http://localhost:11434/v1/chat/completions",
               "key": None, "model": "qwen2.5:3b"},
}


class LLMUnavailable(Exception):
    """No key, rate limit, network error or provider error. Callers fall back to rules."""


def config():
    if os.environ.get("GAME_DISABLE_LLM") == "1":
        raise LLMUnavailable("LLM disabled (GAME_DISABLE_LLM=1)")
    load_env()
    name = os.environ.get("LLM_PROVIDER", "gemini").strip().lower()
    if name not in PROVIDERS:
        raise LLMUnavailable(f"Unknown LLM_PROVIDER '{name}'")
    p = dict(PROVIDERS[name], name=name)
    p["model"] = os.environ.get("LLM_MODEL") or p["model"]
    if p["key"] and not os.environ.get(p["key"]):
        raise LLMUnavailable(f"{p['key']} is not set in .env")
    return p


_calls = 0


def chat(messages, json_mode=False, temperature=0.2, timeout=60):
    """Send a chat request. Returns the reply text. Raises LLMUnavailable on any failure.

    Safeguards: a hard cap on calls per run (LLM_MAX_CALLS, default 40), and no automatic
    retry on rate-limit or quota errors (HTTP 429): the next model in the list is tried once, never the same one.
    """
    p = config()
    models = [m.strip() for m in p["model"].split(",") if m.strip()]
    last = None
    for model in models:
        try:
            return _request(p, model, messages, json_mode, temperature, timeout)
        except _Overloaded as e:
            last = e  # try the next model in the list
    raise LLMUnavailable(f"All models busy or unavailable: {last}")


class _Overloaded(Exception):
    pass


def _request(p, model, messages, json_mode, temperature, timeout):
    global _calls
    cap = int(os.environ.get("LLM_MAX_CALLS", "40"))
    if _calls >= cap:
        raise LLMUnavailable(f"LLM call cap reached ({cap} per run; set LLM_MAX_CALLS to change)")
    _calls += 1
    body = {"model": model, "messages": messages, "temperature": temperature}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    headers = {"Content-Type": "application/json"}
    if p["key"]:
        headers["Authorization"] = f"Bearer {os.environ[p['key']]}"
    req = urllib.request.Request(p["url"], data=json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        detail = e.read()[:300].decode(errors="replace")
        if e.code == 429:  # this model's quota is used up: the next model in the list has its own (never the same one again)
            raise _Overloaded(f"{model} rate limit or free-tier quota reached") from None
        if e.code in (503, 404):
            raise _Overloaded(f"{model} HTTP {e.code}") from None
        if e.code == 400 and json_mode:  # provider may not support response_format: retry without it
            return _request(p, model, messages, False, temperature, timeout)
        raise LLMUnavailable(f"{p['name']} HTTP {e.code}: {detail}") from None
    except (urllib.error.URLError, TimeoutError) as e:
        raise LLMUnavailable(f"{p['name']} unreachable: {e}") from None
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError):
        raise LLMUnavailable(f"Unexpected reply from {p['name']}") from None


def chat_json(messages, retries=1):
    """Ask for JSON and parse it, tolerating code fences or extra text around the object."""
    for attempt in range(retries + 1):
        text = chat(messages, json_mode=True)
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        messages = messages + [{"role": "assistant", "content": text},
                               {"role": "user", "content": "That was not valid JSON. Reply with only the JSON object."}]
    raise LLMUnavailable("LLM did not return valid JSON")


def describe():
    try:
        p = config()
        return f"{p['name']} / {p['model']}"
    except LLMUnavailable as e:
        return f"unavailable ({e})"
