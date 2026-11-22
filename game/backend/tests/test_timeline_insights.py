"""AI timeline insights: the code picks the moments, the model only explains them (per-moment guard + fallback)."""
import os

os.environ["GAME_DISABLE_LLM"] = "1"   # tests never call a real LLM

from fastapi.testclient import TestClient  # noqa: E402

from app.ai import timeline  # noqa: E402
from app.main import app  # noqa: E402
from app.models import GameProjectionRequest  # noqa: E402

client = TestClient(app)
WORK = {"month": 0, "actions": [{"type": "life_choice", "id": "start_work"}]}
ABROAD = {"month": 12, "actions": [{"type": "life_choice", "id": "move_abroad"}]}
MASTER = {"month": 0, "actions": [{"type": "life_choice", "id": "master_ft"}]}
BODY = {"world": 137, "n": 20, "current_month": 24, "turns": [WORK, ABROAD]}


def test_rule_insights_are_pinned_to_engine_moments_in_time_order():
    r = client.post("/ai/timeline", json=BODY)
    assert r.status_code == 200, r.text
    d = r.json()
    months = [i["month"] for i in d["insights"]]
    assert months == sorted(months) and d["headline"] and d["mode"] == "rules"
    decisions = [i for i in d["insights"] if i["kind"] == "decision"]
    assert [i["title"] for i in decisions] == ["You start working", "You move abroad"]
    assert [i["month"] for i in decisions] == [0, 12]
    assert any(i["kind"] == "retire" for i in d["insights"]) and any(i["kind"] == "stress" for i in d["insights"])
    for i in d["insights"]:
        assert i["facts"] and "{" not in i["text"] and i["text"][0].isupper() and i["by"] == "rules"


def test_decisions_report_what_they_changed_against_the_same_life_without_them():
    d = client.post("/ai/timeline", json=BODY).json()
    first = next(i for i in d["insights"] if i["id"] == "d1")
    assert "more" in first["text"] or "less" in first["text"]            # a with/without comparison from the engine
    assert any(k.startswith("d1.delta") for k in d["facts_used"])


def test_cash_going_below_zero_gets_its_own_moment():
    d = client.post("/ai/timeline", json={**BODY, "turns": [MASTER], "current_month": 6}).json()
    low = [i for i in d["insights"] if i["kind"] == "low"]
    assert low and "borrow" in low[0]["text"] and "tight" in d["headline"].lower()


def fake_llm(monkeypatch, reply):
    monkeypatch.setattr(timeline, "chat_json", lambda messages, **kw: reply)


def req():
    return GameProjectionRequest(**BODY)


def test_llm_text_is_used_per_moment_and_bad_moments_fall_back(monkeypatch):
    fake_llm(monkeypatch, {"headline": "Your life, with a job at the start.", "insights": [
        {"id": "d1", "text": "At {d1.age} you {d1.label}, which shapes the years after."},          # fine
        {"id": "d2", "text": "You end with {d1.age}."},                                              # fact of another moment
        {"id": "ret", "text": "You will definitely retire with {ret.liquid}."},                      # certainty
        {"id": "nope", "text": "Invented moment."}]})                                                # unknown moment
    d = timeline.timeline_insights(req())
    by = {i["id"]: i for i in d["insights"]}
    assert by["d1"]["by"] == "llm" and "At 20 you start working" in by["d1"]["text"]
    assert by["d2"]["by"] == "rules" and by["ret"]["by"] == "rules"
    assert "nope" not in by and d["mode"] == "llm" and d["rejected_sentences"] == 3


def test_models_cannot_add_or_move_moments(monkeypatch):
    fake_llm(monkeypatch, {"headline": "x", "insights": [{"id": "zzz", "text": "A made-up moment."}]})
    d = timeline.timeline_insights(req())
    assert {i["id"] for i in d["insights"]} >= {"d1", "d2"} and "zzz" not in {i["id"] for i in d["insights"]}
    assert d["mode"] == "rules"
