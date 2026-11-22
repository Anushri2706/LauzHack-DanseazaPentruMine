"""A whole life to age 120: retirement, Swiss pensions, school choices and the life status per month."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.engine.game import Action as A, InvalidAction, Turn as T, run_turns, state_at
from app.engine.montecarlo import make_worlds
from app.main import app
from tests.conftest import make_params

client = TestClient(app)
LIFE = 1200  # 20 -> 120


def life(**kw):
    return make_params(**{"horizon_months": LIFE, **kw})


WORK = (A("life_choice", "start_work"),)


def test_job_loss_is_the_flat_model():
    p = life()
    ws = make_worlds(p, 40, 2)
    assert all("regime_u" not in w for w in ws)
    spells = [w["income_mult"] < 1 for w in ws]
    assert any(m.any() for m in spells)
    assert all(np.all(w["income_mult"][m] == p["unemployment_benefit_rate"]) for w, m in zip(ws, spells))
    calm = make_worlds(make_params(**{f'job_loss_{b}': 0 for b in ('18_24', '25_34', '35_49', '50_64')}), 20, 2)
    assert all(np.all(w["income_mult"] == 1.0) for w in calm)


def test_job_rates_are_the_teammates_sourced_csv():
    import csv
    from app.engine.assumptions import ALEXIA_DEFAULTS
    from app.engine.labour_data import TEAM, swiss_flows
    tr = {r["assumption_name"]: float(r["monthly_hazard"]) for r in csv.DictReader(open(TEAM / "transitions.csv"))}
    p = make_params()
    for b in ("18_24", "25_34", "35_49", "50_64"):
        assert p[f"job_loss_{b}"] == tr[f"job_loss_{b}"]
        assert p[f"job_find_{b}"] == pytest.approx(tr[f"job_search_exit_{b}"] + tr["job_search_to_part_time"])
        assert ALEXIA_DEFAULTS[f"job_loss_{b}"][1] == "official_est"
    assert p["masters_job_loss_factor"] == 0.81 and p["long_term_find_factor"] == 0.5 and p["long_term_months"] == 12
    # Younger people lose jobs more often but find work faster.
    assert p["job_loss_18_24"] > p["job_loss_25_34"] > p["job_loss_35_49"] > p["job_loss_50_64"]
    assert p["job_find_18_24"] > p["job_find_50_64"]
    # Cross-check with the official all-ages Eurostat flows we pulled: same order of magnitude.
    eu = swiss_flows()["job_loss_monthly"]
    assert p["job_loss_50_64"] < eu < p["job_loss_18_24"]


def test_spells_follow_age_and_duration():
    from app.engine.labour_data import band
    from app.engine.montecarlo import _job_spells
    p = make_params(horizon_months=540)
    by = {}
    for k in range(400):
        for start, d, _ in _job_spells(p, 540, 7, k):
            by.setdefault(band(p["start_age"] + start / 12), []).append(d)
    years = {"18_24": 5, "25_34": 10, "35_49": 15, "50_64": 15}
    rate = {b: len(v) / 400 / (years[b] * 12) for b, v in by.items()}
    assert rate["18_24"] > rate["35_49"] > rate["50_64"]
    assert np.mean(by["18_24"]) < np.mean(by["50_64"])        # older people stay out longer
    assert max(max(v) for v in by.values()) > 24               # long-term spells exist (exit halves after 12)


def test_a_masters_spares_some_job_losses_with_the_same_luck():
    p = life(horizon_months=480)
    ws = make_worlds(p, 120, 4)
    work = [T(0, WORK)]
    master = [T(0, (A("life_choice", "master_ft"),))]
    js = lambda turns: sum(sum(r["regime"] == "job_search" for r in run_turns(p, turns, world=w).rows[24:])  # noqa: E731
                           for w in ws)
    assert js(master) < js(work)
    p0 = {**p, "masters_job_loss_factor": 1.0}
    lost = lambda q: sum(sum(r["regime"] == "job_search" for r in run_turns(q, master, world=w).rows[24:])  # noqa: E731
                         for w in ws)
    assert lost(p) <= lost(p0)


def test_life_status_per_month_and_retirement_at_65():
    p = life()
    w = make_worlds(p, 10, 1)[3]
    rows = run_turns(p, [T(0, WORK)], world=w).rows
    r65 = 45 * 12
    assert {r["regime"] for r in rows[:r65]} <= {"employed", "job_search"}
    assert set(r["regime"] for r in rows[r65:]) == {"retired"}
    assert all(r["regime"] == "job_search" for r, m in zip(rows, w["income_mult"]) if m < 1 and r["t"] < r65)
    rows = run_turns(p, [T(0, (A("life_choice", "master_ft"),))]).rows
    assert set(r["regime"] for r in rows[:24]) == {"student"}


def test_swiss_pension_rises_with_the_mixed_index_and_pays_a_13th_ahv():
    p = life()
    run = run_turns(p, [T(0, WORK)])
    rows = run.rows
    r65 = 45 * 12
    assert next(x for x in run.letters if x["kind"] == "retired")["month"] == r65
    mixed = (p["wage_index_annual"] + p["inflation_annual"]) / 2
    ahv_max = p["ahv_max_pension"] * (1 + mixed) ** (2 * (r65 // 24))
    jan, jan2 = rows[r65 + 1]["income"], rows[r65 + 25]["income"]
    assert jan2 - jan == pytest.approx(ahv_max * ((1 + mixed) ** 2 - 1), rel=1e-9)  # BVG part stays flat
    dec = r65 + 11
    assert rows[dec]["income"] - rows[dec - 1]["income"] == pytest.approx(ahv_max, rel=1e-9)
    off = run_turns({**p, "ahv_13th_pension": 0}, [T(0, WORK)]).rows
    assert off[dec]["income"] == pytest.approx(off[dec - 1]["income"])


def test_degree_and_drop_out_only_while_enrolled():
    p = life(horizon_months=240)
    for cid in ("graduate", "quit_school"):
        with pytest.raises(InvalidAction, match="not enrolled"):
            run_turns(p, [T(0, WORK), T(1, (A("life_choice", cid),))])
    # Degree: school ends, no job until you start one at the Office.
    s = state_at(p, [T(0, (A("life_choice", "graduate"),))], 0)
    assert s.status == "unemployed" and not s.no_degree and s.salary == 0
    rows = run_turns(p, [T(0, (A("life_choice", "graduate"),))]).rows
    assert all(r["status"] == "unemployed" and r["regime"] == "job_search" for r in rows[:60])
    rows = run_turns(p, [T(0, (A("life_choice", "graduate"),)), T(3, WORK)]).rows
    assert rows[3]["income"] == pytest.approx(p["salary_work"])
    # Drop out of the first degree: no degree, lower pay, no master's.
    rows = run_turns(p, [T(0, (A("life_choice", "quit_school"),)), T(2, WORK)]).rows
    assert rows[2]["income"] == pytest.approx(p["salary_work"] * p["no_degree_wage_factor"])
    with pytest.raises(InvalidAction, match="bachelor"):
        run_turns(p, [T(0, (A("life_choice", "quit_school"),)), T(1, (A("life_choice", "master_ft"),))])
    # In a master's: the degree comes by itself; dropping out keeps the bachelor's, cancels tuition.
    with pytest.raises(InvalidAction, match="finishes by itself"):
        run_turns(p, [T(0, (A("life_choice", "master_ft"),)), T(3, (A("life_choice", "graduate"),))])
    s = state_at(p, [T(0, (A("life_choice", "master_ft"),)), T(3, (A("life_choice", "quit_school"),))], 3)
    assert s.status == "unemployed" and not s.no_degree and not s.has_master
    assert not any(o.kind == "tuition" and o.month >= 3 for o in s.scheduled)


def test_game_runs_to_age_120_and_projection_has_status_shares():
    W = {"month": 0, "actions": [{"type": "life_choice", "id": "start_work"}]}
    s = client.post("/game/state", json={"persona_id": "P00001", "turns": [W], "current_month": 600}).json()
    assert s["horizon"] == LIFE and s["life_end_age"] == 120 and s["rows"][-1]["regime"]
    p = client.post("/game/projection", json={"persona_id": "P00001", "turns": [W], "current_month": 600,
                                              "n": 20}).json()
    share = p["regimes"]["share"]
    n = LIFE - 600
    for i in (0, n // 2, n - 1):
        assert abs(sum(v[i] for v in share.values()) - 1) < 1e-6
    assert share["retired"][-1] == 1.0
    assert client.get("/hmm/decisions/P00001").status_code == 404
