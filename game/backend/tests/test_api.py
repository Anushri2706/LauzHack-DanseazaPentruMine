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
