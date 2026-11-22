"""A persona the player builds: same starting facts as the synthetic people, carried inside its id."""
import pytest
from fastapi.testclient import TestClient

from app.data.loader import build_persona, decode_custom, encode_custom
from app.main import app

client = TestClient(app)
SPEC = {"name": "Sam", "age": 31, "city": "Zurich", "canton": "ZH", "employment_status": "employed",
        "household_type": "single", "cash": 22_000, "investments": 15_000, "debt": 8_000, "debt_rate": 0.065,
        "income": 6_400, "rent": 1_900,
        "goals": [{"name": "Home deposit", "target": 60_000, "target_date": "2032-12-31", "priority": "high"}]}


def test_round_trip_and_card():
    pid = encode_custom(SPEC)
    assert pid.startswith("custom-") and decode_custom(pid)["name"] == "Sam"
    card = client.get(f"/personas/{pid}").json()
    assert card["custom"] is True and card["synthetic"] is False
    assert (card["name"], card["age"], card["canton"], card["status"]) == ("Sam", 31, "ZH", "working")
    assert (card["opening_cash"], card["opening_investments"], card["opening_debt"]) == (22_000, 15_000, 8_000)
    assert card["goals"][0]["month"] == 95   # Dec 2032, row 0 = Jan 2025


def test_entered_facts_are_tagged_user_and_drive_the_engine():
    p = build_persona(encode_custom(SPEC))
    for k, v in (("opening_cash", 22_000), ("opening_investments", 15_000), ("opening_debt", 8_000),
                 ("opening_debt_rate", 0.065), ("start_age", 31), ("salary_work", 6_400), ("rent", 1_900)):
        assert p.params[k] == v and p.tags[k] == "user", k
    assert p.params["home_studio_price"] == pytest.approx(450_000 * 1.25)   # Zurich price level


def test_same_numbers_as_a_synthetic_twin():
    """A custom persona with exactly a data person's facts gets that person's defaults (except tags)."""
    data = build_persona("P00002")
    spec = {"name": data.name, "age": data.age, "city": data.city, "canton": data.canton,
            "employment_status": data.employment_status, "household_type": data.household_type,
            "cash": data.opening_cash, "investments": data.opening_investments, "debt": data.opening_debt,
            "debt_rate": data.params["opening_debt_rate"], "income": data.reference_income, "rent": data.reference_rent,
            "goals": [{"name": g["name"], "target": g["target"], "target_date": g["target_date"],
                       "priority": g["priority"]} for g in data.goals]}
    mine = build_persona(encode_custom(spec))
    differ = {k for k in data.params if mine.params[k] != pytest.approx(data.params[k])}
    assert differ <= {"tuition_annual", "master_ft_months"}   # scenario-CSV values exist only for data people
    assert [g["month"] for g in mine.goals] == [g["month"] for g in data.goals]


def test_whole_game_works_with_a_custom_persona():
    req = {"persona_id": encode_custom(SPEC), "world": 3, "current_month": 12,
           "turns": [{"month": 0, "actions": [{"type": "set_habit", "id": "gym", "value": 1}]}]}
    for path, extra in (("/game/state", {}), ("/game/projection", {"n": 20}), ("/game/sensitivity", {}),
                        ("/twin/fan", {"n": 10})):
        r = client.post(path, json={**req, **extra})
        assert r.status_code == 200, (path, r.text)
    assert client.get(f"/assumptions/defaults/{req['persona_id']}").status_code == 200
    assert client.get(f"/catalogue/{req['persona_id']}").status_code == 200
    twin = client.post("/twin/fan", json={**req, "n": 10}).json()
    assert "no history" in twin["note"]


@pytest.mark.parametrize("change, msg", [
    ({"canton": "XX"}, "canton"), ({"age": 12}, "age"), ({"name": ""}, "name"),
    ({"employment_status": "astronaut"}, "employment_status"), ({"investments": -5}, "investments"),
    ({"goals": [{"name": "Old", "target": 1, "target_date": "2020-01-01"}]}, "after the start"),
])
def test_invalid_specs_are_rejected_with_a_reason(change, msg):
    r = client.get(f"/personas/{encode_custom({**SPEC, **change})}")
    assert r.status_code == 422 and msg in r.json()["detail"]


def test_garbage_id_is_rejected():
    assert client.get("/personas/custom-%%%").status_code == 422
