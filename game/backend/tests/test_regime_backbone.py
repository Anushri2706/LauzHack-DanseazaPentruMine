"""The regime HMM as the backbone of a life (engine/regime.py, stepped inside game.run_turns)."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.engine.game import Action as A, Turn as T, run_turns
from app.engine.labour import job_search_masks
from app.engine.montecarlo import make_worlds
from app.main import app
from tests.conftest import make_params

client = TestClient(app)
LIFE = 1200  # 20 -> 120


def life(**kw):
    return make_params(**{"horizon_months": LIFE, **kw})


def regimes(rows):
    return [r["regime"] for r in rows]


def test_worlds_carry_regime_luck_and_world_k_is_stable():
    p = life()
    a, b = make_worlds(p, 1, 5)[0], make_worlds(p, 120, 5)[0]
    assert np.array_equal(a["regime_u"], b["regime_u"]) and len(a["regime_u"]) == LIFE
    assert np.array_equal(a["regime_A"][0][1], b["regime_A"][0][1])
    # Dirichlet parameter uncertainty: worlds have different transition matrices.
    w = make_worlds(p, 3, 5)
    assert not np.array_equal(w[0]["regime_A"][0][1], w[1]["regime_A"][0][1])
    # No precomputed spells any more: the chain runs inside the life.
    assert all(np.all(x["income_mult"] == 1.0) for x in w)


def test_working_life_has_spells_then_retires_at_65():
    p = life()
    worlds = make_worlds(p, 60, 1)
    any_spell = False
    for w in worlds:
        rows = run_turns(p, [T(0, (A("life_choice", "start_work"),))], world=w).rows
        reg = regimes(rows)
        any_spell |= "job_search" in reg[: 45 * 12]
        assert set(reg[45 * 12:]) == {"retired"}          # 65 and over
        assert "retired" not in reg[: 45 * 12]
        r65 = rows[45 * 12]
        assert r65["status"] == "retired" and r65["income"] > 0
        # Spells only happen while in the labour market and pay the benefit rate.
        for r in rows:
            if r["regime"] == "job_search":
                assert r["status"] == "working"
    assert any_spell


def test_studying_pins_the_student_regime():
    p = life()
    w = make_worlds(p, 1, 3)[0]
    rows = run_turns(p, [T(0, (A("life_choice", "master_ft"),))], world=w).rows
    assert set(regimes(rows[:24])) == {"student"}


def test_common_random_numbers_same_past_same_luck():
    # Two realities that only differ at month 120 are identical before it.
    p = life()
    w = make_worlds(p, 10, 2)[7]
    base = [T(0, (A("life_choice", "start_work"),))]
    a = run_turns(p, base, world=w).rows
    b = run_turns(p, base + [T(120, (A("set_habit", "eat_out", 2),))], world=w).rows
    assert a[:120] == b[:120]


def test_masters_lowers_job_search_months_on_average():
    # Decision-conditioned hazards: masters_hazard_factor scales employed -> job_search.
    p = life(horizon_months=480)
    worlds = make_worlds(p, 200, 4)
    work = [T(0, (A("life_choice", "start_work"),))]
    master = [T(0, (A("life_choice", "master_ft"),))]
    js = lambda turns: np.mean([sum(r["regime"] == "job_search" for r in run_turns(p, turns, world=w).rows[24:])  # noqa: E731
                                for w in worlds])
    assert js(master) < js(work)


def test_quitting_ends_when_the_chain_finds_a_job():
    p = life(horizon_months=240)
    found = 0
    for w in make_worlds(p, 40, 8):
        run = run_turns(p, [T(0, (A("life_choice", "start_work"),)), T(24, (A("life_choice", "quit_job"),))], world=w)
        found += any(x["kind"] == "found_job" for x in run.letters)
        assert run.rows[24]["regime"] == "job_search"
    assert found >= 30  # exits of ~10-18%/month: almost everyone finds work within 18 years


def test_job_search_share_is_plausible():
    m = job_search_masks(20, 240, 300, 0)
    assert 0.01 < m.mean() < 0.12


def test_hmm_off_restores_the_flat_model():
    p = make_params(labour_market_hmm=0, job_loss_prob_annual=0)
    ws = make_worlds(p, 30, 2)
    assert all("regime_u" not in w and np.all(w["income_mult"] == 1.0) for w in ws)


def test_game_runs_to_age_120_and_projection_has_regime_shares():
    WORK = {"month": 0, "actions": [{"type": "life_choice", "id": "start_work"}]}
    s = client.post("/game/state", json={"persona_id": "P00001", "turns": [WORK], "current_month": 600}).json()
    assert s["horizon"] == LIFE and s["life_end_age"] == 120 and s["rows"][-1]["regime"]
    p = client.post("/game/projection", json={"persona_id": "P00001", "turns": [WORK], "current_month": 600,
                                              "n": 20}).json()
    share = p["regimes"]["share"]
    n = LIFE - 600
    assert all(len(v) == n for v in share.values())
    for i in (0, n // 2, n - 1):
        assert abs(sum(v[i] for v in share.values()) - 1) < 1e-6
    assert share["retired"][-1] == 1.0


def test_swiss_pension_rises_with_the_mixed_index_and_pays_a_13th_ahv():
    p = life()
    run = run_turns(p, [T(0, (A("life_choice", "start_work"),))])
    rows = run.rows
    r65 = 45 * 12                                    # month 540 = Jan 2070, an adjustment month
    ahv0 = next(l for l in run.letters if l["kind"] == "retired")
    assert ahv0["month"] == r65
    mixed = (p["wage_index_annual"] + p["inflation_annual"]) / 2
    # Two years later only the AHV part has grown by the mixed index; BVG is flat.
    jan, jan2 = rows[r65 + 1]["income"], rows[r65 + 25]["income"]
    assert jan2 > jan
    ahv_max = p["ahv_max_pension"] * (1 + mixed) ** (2 * (r65 // 24))
    assert jan2 - jan == pytest.approx(ahv_max * ((1 + mixed) ** 2 - 1), rel=1e-9)
    # 13th AHV pension: December income = monthly pension + the AHV part.
    dec = r65 + 11
    assert rows[dec]["income"] - rows[dec - 1]["income"] == pytest.approx(ahv_max, rel=1e-9)
    assert any(l["kind"] == "pension_indexed" for l in run.letters)
    # Switching the 13th off removes the December bump.
    off = run_turns({**p, "ahv_13th_pension": 0}, [T(0, (A("life_choice", "start_work"),))]).rows
    assert off[dec]["income"] == pytest.approx(off[dec - 1]["income"])


def test_degree_and_drop_out_only_while_enrolled():
    from app.engine.game import InvalidAction, state_at
    p = life(horizon_months=240)
    # Not enrolled: both refused with a reason.
    for cid in ("graduate", "quit_school"):
        with pytest.raises(InvalidAction, match="not enrolled"):
            run_turns(p, [T(0, (A("life_choice", "start_work"),)), T(1, (A("life_choice", cid),))])
    # Enrolled (Alexia starts as a student): get the degree -> job search, then the HMM finds a job.
    s = state_at(p, [T(0, (A("life_choice", "graduate"),))], 0)
    assert s.status == "unemployed" and not s.no_degree and s.salary == 0
    found = 0
    for w in make_worlds(p, 30, 3):
        run = run_turns(p, [T(0, (A("life_choice", "graduate"),))], world=w)
        assert run.rows[0]["regime"] == "job_search"
        job = next((l for l in run.letters if l["kind"] == "found_job"), None)
        if job:
            found += 1
            assert run.rows[job["month"]]["income"] == pytest.approx(p["salary_work"])
    assert found >= 25
    # Drop out of the bachelor's: no degree, lower pay, no master's.
    w = make_worlds(p, 1, 3)[0]
    run = run_turns(p, [T(0, (A("life_choice", "quit_school"),))], world=w)
    job = next(l for l in run.letters if l["kind"] == "found_job")
    assert run.rows[job["month"]]["income"] == pytest.approx(p["salary_work"] * p["no_degree_wage_factor"])
    with pytest.raises(InvalidAction, match="bachelor"):
        run_turns(p, [T(0, (A("life_choice", "quit_school"),)), T(1, (A("life_choice", "master_ft"),))])
    # In a master's: the degree comes by itself (button disabled); dropping out keeps the bachelor's
    # and cancels the remaining tuition.
    with pytest.raises(InvalidAction, match="finishes by itself"):
        run_turns(p, [T(0, (A("life_choice", "master_ft"),)), T(3, (A("life_choice", "graduate"),))])
    s = state_at(p, [T(0, (A("life_choice", "master_ft"),)), T(3, (A("life_choice", "quit_school"),))], 3)
    assert s.status == "unemployed" and not s.no_degree and not s.has_master
    assert not any(o.kind == "tuition" and o.month >= 3 for o in s.scheduled)
