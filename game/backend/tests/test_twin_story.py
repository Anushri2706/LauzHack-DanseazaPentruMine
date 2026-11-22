"""The learned twin in plain words: rule fallback, debts never shown as negative amounts, LLM path with guard."""
import os

os.environ["GAME_DISABLE_LLM"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from app.ai import twin  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)
REQ = {"world": 137, "turns": [], "current_month": 0, "n": 20}


def test_rules_explain_the_twin_and_why_your_timeline_differs():
    d = client.post("/ai/twin", json=REQ).json()
    text = " ".join(s["text"] for s in d["story"])
    assert d["mode"] == "rules" and len(d["story"]) >= 3 and "{" not in text and "−" not in text
    assert "studying" in text  # no choice made yet: the timeline keeps you studying, the twin does not


def test_llm_sentences_with_negative_amounts_are_rejected(monkeypatch):
    twin.cached.cache_clear()
    facts, _ = twin.twin_facts(twin.GameProjectionRequest(**REQ))
    neg = next(k[:-5] for k in facts if k.endswith("_debt"))
    monkeypatch.setattr(twin, "chat_json", lambda m, **kw: {"headline": "x", "sentences": [
        {"text": "You end at {%s}." % neg}, {"text": "You are {twin.start}."},
        {"text": "Most people like you retire around {twin.retire}."}, {"text": "You would owe {%s_debt}." % neg}]})
    d = twin.twin_story(twin.GameProjectionRequest(**REQ))
    assert d["mode"] == "llm" and d["rejected_sentences"] == 1 and "−" not in " ".join(s["text"] for s in d["story"])
