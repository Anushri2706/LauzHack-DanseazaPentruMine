"""The game API's bridge to the regime HMM twin (repo-root backend/twin)."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_decisions_listed_and_masked():
    alexia = client.get("/hmm/decisions/P00001").json()
    assert alexia["initial_state"] == "student"
    ids = {d["id"]: d for d in alexia["decisions"]}
    assert ids["masters_now"]["available"] and ids["stay"]["label"].startswith("Finish studies")
    rafael = client.get("/hmm/decisions/P00002").json()       # retired
    avail = {d["id"] for d in rafael["decisions"] if d["available"]}
    assert "masters_now" not in avail and "stay" in avail


def test_fan_shapes_and_same_seed_shared_start():
    r = client.post("/hmm/fan", json={"persona_id": "P00001", "decisions": ["stay", "masters_now"],
                                      "horizon_months": 60, "n_paths": 400})
    assert r.status_code == 200
    out = r.json()
    assert [d["id"] for d in out["decisions"]] == ["stay", "masters_now"]
    for d in out["decisions"]:
        assert len(d["cash"]["p10"]) == 60
        assert all(a <= b <= c for a, b, c in zip(d["cash"]["p10"], d["cash"]["p50"], d["cash"]["p90"]))
        assert abs(sum(v[0] for v in d["occupancy"].values()) - 1) < 1e-6
        assert {g["name"] for g in d["goals"]} >= {"Emergency savings"}
        assert all(0 <= g["hit_rate"] <= 1 for g in d["goals"])
    masters = out["decisions"][1]
    assert masters["months_in_state"]["student"] == 24      # the pinned degree


def test_fan_errors():
    assert client.post("/hmm/fan", json={"persona_id": "NOPE"}).status_code == 404
    bad = client.post("/hmm/fan", json={"persona_id": "P00002", "decisions": ["masters_now"]})
    assert bad.status_code == 422
    assert client.post("/hmm/fan", json={"persona_id": "P00001", "decisions": ["warp"]}).status_code == 422
