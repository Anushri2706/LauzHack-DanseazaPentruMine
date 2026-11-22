"""API tests for the town game endpoints (/game/*, /catalogue, /compare with turns)."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

MASTER = {"month": 0, "actions": [{"type": "life_choice", "id": "master_ft"},
                                  {"type": "set_habit", "id": "eat_out", "value": 2}]}
WORK = {"month": 0, "actions": [{"type": "life_choice", "id": "start_work"}]}


def state(**kw):
    r = client.post("/game/state", json={"world": 137, "turns": [], **kw})
    assert r.status_code == 200, r.text
    return r.json()


def test_state_has_places_actions_and_hud():
    s = state()
    assert s["month"] == 0 and s["age"] == 20 and s["date"]["label"] == "Jan 2025"
    assert set(s["actions"]) == {"home", "university", "office", "cafe", "shop", "bank", "realestate", "station"}
    assert set(s["scores"]) == {"money", "safety", "goals", "freedom", "experience", "joy"}
    office = {a["id"] or a["type"]: a for a in s["actions"]["office"]}
    assert office["start_work"]["enabled"] and not office["ask_raise"]["enabled"]
    assert "Simulation" in s["disclaimer"]


def test_queue_then_sleep():
    queued = state(turns=[MASTER], current_month=0)
    assert queued["state"]["status"] == "studying" and queued["pending_state"]["habits"] == {"eat_out": 2}
    assert len(queued["queued"]) == 2
    slept = state(turns=[MASTER], current_month=1)
    assert slept["month"] == 1 and slept["statement"]["month"] == 0
    labels = {x["label"]: x["amount"] for x in slept["statement"]["lines"]}
    assert labels["Habits"] == -340 and labels["One-offs (tuition, moves)"] == -4000


def test_invalid_and_future_turns_are_422():
    bad = {"month": 0, "actions": [{"type": "buy_home", "id": "studio"}]}
    r = client.post("/game/state", json={"turns": [bad]})
    assert r.status_code == 422 and "down payment" in r.json()["detail"]
    r = client.post("/game/state", json={"turns": [{**WORK, "month": 5}], "current_month": 2})
    assert r.status_code == 422


def test_rollback_is_exact_through_the_api():
    turns = [MASTER, {"month": 24, "actions": [{"type": "life_choice", "id": "move_abroad"}]},
             {"month": 40, "actions": [{"type": "buy_item", "id": "flight_home"}]}]
    late = state(turns=turns, current_month=60)
    back = state(turns=turns[:2], current_month=30)
    assert late["rows"][:30] == back["rows"]


def test_editing_assumptions_never_unlives_the_past():
    # Buy a laptop; then make tuition enormous: the past stays lived, cash just goes negative.
    turns = [MASTER, {"month": 2, "actions": [{"type": "buy_item", "id": "laptop"}]}]
    s = state(turns=turns, current_month=30, assumptions={"tuition_annual": 60000})
    assert s["state"]["cash"] < 0 and any(w["kind"] == "financing_needed" for w in s["warnings"])
    assert any(x["kind"] == "financing_needed" for x in s["letters"])


def test_projection_bands_ghosts_and_no_surprises():
    req = {"turns": [WORK], "current_month": 12, "world": 137, "n": 60}
    p = client.post("/game/projection", json=req).json()
    nw = p["bands"]["net_worth"]
    assert len(nw["p10"]) == 108 and len(p["lived"]["net_worth"]) == 12
    assert all(lo <= mid <= hi for lo, mid, hi in zip(nw["p10"], nw["p50"], nw["p90"]))
    assert len(p["no_surprises"]["net_worth"]) == 108 and len(p["ghosts"]) == 3
    assert {g["goal_id"] for g in p["goals"]} == {g["goal_id"] for g in client.get("/personas/P00001").json()["goals"]}
    # Fan opens from the lived present: the first future month is narrow compared with the last.
    assert nw["p90"][0] - nw["p10"][0] < nw["p90"][-1] - nw["p10"][-1]
    roll = client.post("/game/roll", json={**req, "roll": 3}).json()
    other = client.post("/game/roll", json={**req, "roll": 9}).json()
    assert len(roll["net_worth"]) == 108 and len(roll["cash"]) == 108
    assert roll["events"] and roll["events"][0]["kind"] == "hmm"
    assert "HMM draw" in roll["events"][0]["text"]
    assert roll["net_worth"] != other["net_worth"]
    assert "HMM paths" in p["note"]


def test_compare_realities_with_turns():
    a = {"turns": [MASTER], "current_month": 24, "world": 137}
    b = {"turns": [WORK, {"month": 6, "actions": [{"type": "buy_item", "id": "weekend_trip"}]}],
         "current_month": 24, "world": 1}  # B's world is overridden by A's for fairness
    c = client.post("/compare", json={"a": a, "b": b, "name_a": "Master", "name_b": "Work"})
    assert c.status_code == 200, c.text
    c = c.json()
    keys = {row["key"] for row in c["table"]}
    assert {"final_net_worth", "total_joy", "final_home_equity"} <= keys
    assert {"a", "b"} == set(c["ranges"]) and "#137" in c["note"]
    assert any(x["key"] == "lifestyle" for x in c["bridge"])


def test_catalogue():
    cat = client.get("/catalogue/P00001").json()
    assert {h["id"] for h in cat["habits"]} == {"eat_out", "shopping", "gym"}
    assert {h["id"]: h["price"] for h in cat["homes"]}["studio"] == 450_000
    assert client.get("/catalogue/NOPE").status_code == 404


def test_projection_at_the_end_still_gives_ranges():
    req = {"turns": [WORK], "current_month": 120, "world": 9, "n": 40}
    p = client.post("/game/projection", json=req)
    assert p.status_code == 200, p.text
    p = p.json()
    f = p["final_net_worth"]
    assert p["ended"] and p["bands"] is None and p["no_surprises"] is None
    assert f["p10"] <= f["p50"] <= f["p90"] and len(p["lived"]["net_worth"]) == 120
    assert p["bridge"]["closing"] == p["lived"]["net_worth"][-1]
