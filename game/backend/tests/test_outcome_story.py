"""Plain-language outcome stories: rule fallback, LLM path, claim guard, and built into /compare + /game/projection."""
import os

os.environ["GAME_DISABLE_LLM"] = "1"   # tests never call a real LLM

import re  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from app.ai import outcome  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)
WORK = {"month": 0, "actions": [{"type": "life_choice", "id": "start_work"}]}
MASTER = {"month": 0, "actions": [{"type": "life_choice", "id": "master_ft"}]}
REQ = {"world": 137, "turns": [WORK], "n": 20}
CMP = {"a": {"world": 137, "turns": [WORK]}, "b": {"world": 137, "turns": [MASTER]},
       "name_a": "Work first", "name_b": "Master's first"}


def fake_llm(monkeypatch, reply):
    outcome._cached_outcome.cache_clear()
    monkeypatch.setattr(outcome, "chat_json", lambda messages, **kw: reply)


def test_outcome_story_without_llm_is_plain_language_from_facts():
    d = client.post("/ai/outcome", json=REQ).json()
    assert d["mode"] == "rules" and d["headline"] and len(d["story"]) >= 3
    assert d["facts_used"] and "Simulated" in d["disclaimer"]
    text = " ".join(s["text"] for s in d["story"])
    assert "{" not in text and "P10" not in text and "net worth" not in text.lower()
    assert all(s["facts"] and s["text"][0].isupper() for s in d["story"])


def test_compare_story_names_both_realities_and_skips_empty_comparisons():
    d = client.post("/ai/compare", json=CMP).json()
    text = " ".join(s["text"] for s in d["story"])
    assert "Work first" in text and "Master's first" in text and " 0 months" not in text


def test_llm_story_is_used_when_every_sentence_passes_the_guard(monkeypatch):
    fake_llm(monkeypatch, {"headline": "A calm life.", "sentences": [
        {"text": "By {end.age} you could own around {end.range_p50}."},
        {"text": "On a bad run of luck it might be closer to {end.range_p10}."},
        {"text": "On a good run, closer to {end.range_p90}."}]})
    d = client.post("/ai/outcome", json=REQ).json()
    assert d["mode"] == "llm" and d["headline"] == "A calm life." and d["rejected_sentences"] == 0
    assert "{" not in " ".join(s["text"] for s in d["story"])


def test_guard_rejects_invented_numbers_certainty_and_unknown_facts_then_falls_back(monkeypatch):
    fake_llm(monkeypatch, {"headline": "x", "sentences": [
        {"text": "You will have 5 million."},                      # own digits
        {"text": "You will be rich: {end.range_p50}."},             # certainty
        {"text": "You end with {no.such.fact}."},                   # unknown fact
        {"text": "It happens because of {end.age}."}]})            # causal
    d = client.post("/ai/outcome", json=REQ).json()
    assert d["mode"] == "rules" and d["rejected_sentences"] == 4 and len(d["story"]) >= 3


def test_stuttering_sentences_are_dropped(monkeypatch):
    fake_llm(monkeypatch, {"headline": "x", "sentences": [
        {"text": "Your investment growth investment growth adds {money.growth}."},
        {"text": "By {end.age} you own {end.range_p50}."}]})
    d = client.post("/ai/outcome", json=REQ).json()
    assert d["mode"] == "rules"          # only one clean sentence survived: not enough for an LLM story


def test_story_is_built_into_compare_and_the_ended_projection_but_not_mid_game():
    c = client.post("/compare", json=CMP).json()
    assert c["explanation"] and c["story"]["story"]          # the old technical text is still there
    mid = client.post("/game/projection", json={**REQ, "current_month": 12}).json()
    assert "story" not in mid
    end = client.post("/game/projection", json={**REQ, "current_month": 1200}).json()
    assert end["ended"] and end["story"]["headline"] and len(end["story"]["story"]) >= 3
