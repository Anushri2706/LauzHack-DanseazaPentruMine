"""End-to-end acceptance scenario (brief section 13), entirely through the HTTP API, no mocked outputs."""
import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


PROFILE = dict(age=26, employment="employed", income_amount_chf=88000, income_period="yearly", income_basis="gross",
               rent_chf_month=1550, cash_chf=18000, investments_chf=6000, debt_chf=12000, debt_annual_rate=0.035,
               other_expenses_chf_month=1700, canton="VD", household="single")


def test_acceptance_scenario(client):
    # 1-2. profile -> game + root branch, baseline Monte Carlo
    r = client.post("/api/game/start", json={"profile": PROFILE, "config": {"horizon_months": 120, "n_paths": 400},
                                             "seed": 2026})
    assert r.status_code == 201, r.text
    game = r.json()
    gid, root = game["id"], game["root_id"]
    assert game["storage"].startswith("in-memory")
    base = client.post(f"/api/game/{gid}/branches/{root}/simulate").json()

    # 3. Branch A: master's with explicit assumptions
    a = client.post(f"/api/game/{gid}/branches", json={"parent_id": root, "name": "Master's"}).json()["id"]
    r = client.post(f"/api/game/{gid}/branches/{a}/decisions",
                    json={"decision_id": "masters", "start_month": 0,
                          "params": {"duration_months": 24, "annual_tuition_chf": 1460}})
    assert r.status_code == 201, r.text
    # 4. Branch B: continue employment
    b = client.post(f"/api/game/{gid}/branches", json={"parent_id": root, "name": "Keep working"}).json()["id"]
    assert client.post(f"/api/game/{gid}/branches/{b}/decisions",
                       json={"decision_id": "continue_employment", "start_month": 0}).status_code == 201

    # 5. run both with the game seed (common random numbers)
    ra = client.post(f"/api/game/{gid}/branches/{a}/simulate").json()
    rb = client.post(f"/api/game/{gid}/branches/{b}/simulate").json()
    assert ra["seed"] == rb["seed"] == 2026

    # 6-8. P10/P50/P90 trajectories, reserve risk, provenance and warnings
    for res in (ra, rb):
        s = res["summary"]
        assert len(s["monthly"]["cash"]["p10"]) == 120 and len(s["monthly"]["net_worth_real"]["p90"]) == 120
        assert s["final"]["net_worth_real"]["p10"] <= s["final"]["net_worth_real"]["p50"] <= s["final"]["net_worth_real"]["p90"]
        assert 0 <= s["risk"]["p_ever_below_reserve"] <= 1 and s["reserve_chf"] > 0
        ids = {x["id"]: x for x in s["assumptions_used"]}
        assert "tax_rate_at_80000" in ids and ids["tax_rate_at_80000"]["source_status"] == "estimated"
        assert all(x["source_url"] is None for x in s["assumptions_used"])
        assert s["warnings"] and s["reconciles"] is True
    assert {x["id"] for x in ra["summary"]["assumptions_used"]} >= {"masters_premium_pct", "student_income_chf"}

    # 9. deterministic comparison
    cmp = client.post(f"/api/game/{gid}/compare", json={"a": b, "b": a}).json()
    d = cmp["differences"]["net_worth_real.final.p50"]
    assert d["a"] == rb["summary"]["final"]["net_worth_real"]["p50"]
    assert d["b"] == ra["summary"]["final"]["net_worth_real"]["p50"]
    assert d["difference_b_minus_a"] == pytest.approx(d["b"] - d["a"], abs=0.01)
    assert cmp == client.post(f"/api/game/{gid}/compare", json={"a": b, "b": a}).json()

    # 10-11. AI explanation, verified against result ids and metrics
    ex = client.post("/api/ai/explain", json={"game_id": gid, "branch_id": a, "compare_with": b,
                                              "question": "Why do these branches differ?", "use_llm": False}).json()
    assert set(ex["result_ids"]) == {ra["result_id"], rb["result_id"]}
    assert ex["accepted_claims"] and not ex["rejected_claims"]
    p50_a = f"CHF {ra['summary']['final']['net_worth_real']['p50']:,.0f}"
    assert p50_a in ex["text"] and "Under the assumptions used" in ex["text"]

    # 12. an intentionally fabricated metric claim is rejected
    bad = client.post("/api/ai/explain", json={
        "game_id": gid, "branch_id": a, "compare_with": b, "use_llm": False,
        "draft": {"claims": [{"type": "level", "fact": f"{ra['result_id']}:lifetime_happiness.final.p50"}]}}).json()
    assert bad["rejected_claims"] and "does not exist" in bad["rejected_claims"][0]["reason"]
    assert bad["used_deterministic_fallback"] and "lifetime_happiness" not in bad["text"]

    # 13. child of A with another decision; root, A and B unchanged
    snap = {x: client.get(f"/api/game/{gid}/branches/{x}").json() for x in (root, a, b)}
    a1 = client.post(f"/api/game/{gid}/branches", json={"parent_id": a, "name": "Master's + move",
                                                         "fork_month": 30}).json()["id"]
    assert client.post(f"/api/game/{gid}/branches/{a1}/decisions",
                       json={"decision_id": "relocate", "start_month": 30, "params": {"to_canton": "ZH"}}).status_code == 201
    r1 = client.post(f"/api/game/{gid}/branches/{a1}/simulate").json()
    assert [d["decision_id"] for d in r1["decisions"]] == ["masters", "relocate"]
    for x in (root, a, b):
        now = client.get(f"/api/game/{gid}/branches/{x}").json()
        assert {k: v for k, v in now.items() if k != "children"} == {k: v for k, v in snap[x].items() if k != "children"}
        assert now["status"] == "simulated"
    for x, res in ((root, base), (a, ra), (b, rb)):
        assert client.get(f"/api/game/{gid}/branches/{x}/results").json()["summary"]["fingerprint"] == res["summary"]["fingerprint"]

    # 14. replay with the same seed -> identical
    rep = client.post(f"/api/game/{gid}/branches/{a}/replay", json={}).json()
    assert rep["identical"] is True and rep["original_fingerprint"] == ra["summary"]["fingerprint"]


def test_api_errors_are_structured(client):
    r = client.post("/api/game/start", json={"profile": {**PROFILE, "age": 7}})
    assert r.status_code == 422 and r.json()["errors"][0]["field"] == "age"
    r = client.post("/api/game/start", json={"profile": {**PROFILE, "salary_of_friend": 1}})
    assert r.status_code == 422
    r = client.post("/api/game/start", json={"profile": {**PROFILE, "debt_annual_rate": None}})
    assert r.status_code == 422 and r.json()["errors"][0]["field"] == "debt_annual_rate"
    gid = client.post("/api/game/start", json={"profile": PROFILE, "config": {"n_paths": 50}}).json()["id"]
    assert client.get(f"/api/game/{gid}/branches/br_nope").status_code == 404
    root = client.get(f"/api/game/{gid}").json()["root_id"]
    r = client.post(f"/api/game/{gid}/branches/{root}/assumptions", json={"overrides": {"job_loss_hazard": 2}})
    assert r.status_code == 422 and r.json()["error"] == "invalid_assumption"
    r = client.post(f"/api/game/{gid}/compare", json={"a": root, "b": root})
    assert r.status_code == 409
    assert client.get("/api/health").json()["status"] == "ok"
    assert len(client.get("/api/assumptions").json()["assumptions"]) > 100


def test_sensitivity_endpoint(client):
    g = client.post("/api/game/start", json={"profile": PROFILE, "config": {"n_paths": 100, "horizon_months": 60},
                                             "seed": 1}).json()
    out = client.post(f"/api/game/{g['id']}/sensitivity",
                      json={"branch_id": g["root_id"], "assumption_ids": ["entry_salary_chf", "job_loss_hazard",
                                                                          "apply_income_tax"], "n_paths": 100}).json()
    assert {r["assumption_id"] for r in out["rows"]} == {"entry_salary_chf", "job_loss_hazard"}
    assert out["skipped"][0]["assumption_id"] == "apply_income_tax"


def test_peer_band_is_labelled_synthetic(client):
    gid = client.post("/api/game/start", json={"profile": PROFILE, "config": {"n_paths": 50}}).json()["id"]
    out = client.get(f"/api/game/{gid}/peers?years=10").json()
    assert out["data_status"] == "synthetic" and out["n_peers"] > 0
    assert out["wealth_chf"]["p25"] <= out["wealth_chf"]["p50"] <= out["wealth_chf"]["p75"]
    assert any("not a forecast" in x for x in out["limitations"])


def test_backend_only_no_frontend_served(client):
    """Backend-only: the root path serves nothing; the API lives under /api."""
    assert client.get("/").status_code == 404
    assert client.get("/api/health").status_code == 200


def test_cors_allows_separate_frontend_origin(client):
    r = client.options("/api/health", headers={"Origin": "http://localhost:5173",
                                               "Access-Control-Request-Method": "GET"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
    r = client.options("/api/health", headers={"Origin": "http://evil.example",
                                               "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in r.headers
