"""'Which assumptions matter most?' — the tornado endpoint (one assumption at a time, no luck)."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
MASTER = [{"month": 0, "actions": [{"type": "life_choice", "id": "master_ft"}]}]
WORK = [{"month": 0, "actions": [{"type": "life_choice", "id": "start_work"}]}]


def _sens(turns, **kw):
    r = client.post("/game/sensitivity", json={"world": 9, "turns": turns, "current_month": 0, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def test_ranked_by_spread_and_reproducible():
    a, b = _sens(MASTER), _sens(MASTER)
    assert a == b
    spreads = [x["spread"] for x in a["bars"]]
    assert spreads == sorted(spreads, reverse=True) and spreads
    assert a["target_age"] == 30


def test_isolation_tuition_does_not_touch_work():
    s = _sens(WORK)
    assert "Master's tuition" in s["no_effect"]
    assert all(b["key"] != "tuition_annual" for b in s["bars"])


def test_higher_tuition_never_raises_net_worth():
    bar = next(b for b in _sens(MASTER)["bars"] if b["key"] == "tuition_annual")
    assert bar["high"] > bar["low"] and bar["net_worth_high"] <= bar["net_worth_low"]


def test_higher_salary_raises_net_worth():
    bar = next(b for b in _sens(WORK)["bars"] if b["key"] == "salary_work")
    assert bar["net_worth_high"] > bar["net_worth_low"]


def test_edited_assumptions_are_not_swung():
    s = _sens(WORK, assumptions={"return_annual": 0.03})
    assert all(b["key"] != "return_annual" for b in s["bars"] + [{"key": x} for x in s["no_effect"]])
