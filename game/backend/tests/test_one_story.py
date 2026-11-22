"""The graph, the AI notes and the companion tell one story: the path drawn ahead of you is exactly what you
then live, and the notes and the companion quote that same path."""
import pytest
from fastapi.testclient import TestClient

from app.ai.companion import build_facts
from app.ai.timeline import build
from app.main import app
from app.models import GameProjectionRequest

client = TestClient(app)
TURNS = [{"month": 0, "actions": [{"type": "life_choice", "id": "master_ft", "value": 24},
                                  {"type": "set_habit", "id": "eat_out", "value": 2}]}]


def _req(m, **kw):
    return {"world": 9, "seed": 42, "turns": TURNS, "current_month": m, "n": 20, **kw}


@pytest.mark.parametrize("m", [0, 24, 300])
def test_the_path_ahead_is_what_you_then_live(m):
    ahead = client.post("/game/projection", json=_req(m)).json()["your_world"]
    later = 900
    lived = client.post("/game/state", json=_req(later)).json()["rows"]
    for t in (m, m + 1, m + 120, later - 1):
        assert lived[t]["net_worth"] == pytest.approx(ahead["net_worth"][t - m])
        assert lived[t]["cash"] + lived[t]["investments"] == pytest.approx(ahead["liquid"][t - m])


def test_ai_notes_quote_the_drawn_path():
    req = _req(24)
    proj = client.post("/game/projection", json=req).json()
    past = client.post("/game/state", json=req).json()["rows"]
    cash = [r["cash"] for r in past] + proj["your_world"]["cash"]
    facts, anchors, _ = build(GameProjectionRequest(**req))
    assert any(a["kind"] == "low" for a in anchors)
    assert facts["low.cash"]["value"] == pytest.approx(min(cash))


def test_companion_leads_with_the_drawn_path():
    req = _req(24)
    st = client.post("/game/state", json=req).json()
    proj = client.post("/game/projection", json=req).json()
    facts = build_facts(st, proj, None, None)
    assert facts["yours.end_nw"]["value"] == pytest.approx(proj["your_world"]["net_worth"][-1])
    assert facts["yours.low_cash"]["value"] == pytest.approx(min(proj["your_world"]["cash"]))


def test_no_path_ahead_once_the_life_is_lived():
    h = client.post("/game/state", json=_req(0)).json()["horizon"]
    assert client.post("/game/projection", json=_req(h)).json()["your_world"] is None
