from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_personas_and_defaults():
    assert client.get("/health").json()["profiles"] == 1200
    alexia = client.get("/personas/P00001").json()
    assert alexia["name"] == "Alexia" and alexia["opening_cash"] == 8000
    rows = client.get("/assumptions/defaults/P00001").json()
    by_key = {r["key"]: r for r in rows}
    assert by_key["tuition_annual"]["value"] == 8000 and by_key["tuition_annual"]["tag"] == "scenario_csv"
    assert client.get("/personas/NOPE").status_code == 404
    assert len(client.get("/personas", params={"status": "retired", "limit": 5}).json()) == 5


def test_simulate_roundtrip_and_rollback():
    from app.models import BranchResult
    full = {"persona_id": "P00001", "decisions": [{"month": 0, "choice": "master_ft"},
                                                  {"month": 12, "choice": "continue"},
                                                  {"month": 24, "choice": "move_abroad"}]}
    r = client.post("/branch/simulate", json=full)
    assert r.status_code == 200
    res = BranchResult.model_validate(r.json())
    assert res.current_month == 36 and res.next_decision.month == 36
    trunc = client.post("/branch/simulate", json={**full, "decisions": full["decisions"][:2]}).json()
    assert trunc["current_month"] == 24
    assert trunc["rows"][:24] == r.json()["rows"][:24]
    assert "master" in trunc["next_decision"]["narrative"]


def test_invalid_decision_is_422():
    r = client.post("/branch/simulate", json={"decisions": [{"month": 0, "choice": "return_home"}]})
    assert r.status_code == 422


def test_montecarlo_bands_and_same_worlds():
    req = {"decisions": [{"month": 0, "choice": "start_work"}], "n": 50}
    a = client.post("/branch/montecarlo", json=req).json()
    nw = a["net_worth"]
    assert all(lo <= mid <= hi for lo, mid, hi in zip(nw["p10"], nw["p50"], nw["p90"]))
    assert len(a["ghosts"]) == 3
    b = client.post("/branch/montecarlo", json={**req, "decisions": [{"month": 0, "choice": "master_ft"}]}).json()
    # Scenario isolation: both branches see the same random worlds (same event list).
    assert a["samples"][0]["events"] == b["samples"][0]["events"]


def test_compare_and_validation():
    req = {"a": {"decisions": [{"month": 0, "choice": "start_work"}]},
           "b": {"decisions": [{"month": 0, "choice": "master_ft"}]}, "name_a": "Work", "name_b": "Master"}
    c = client.post("/compare", json=req).json()
    assert c["explanation"] and c["table"][0]["key"] == "final_net_worth"
    checks = client.get("/validation").json()
    assert checks and all(x["passed"] for x in checks), checks
