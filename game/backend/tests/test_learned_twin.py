"""Learned twin (HMM x learned engine) and the companion."""
import json
import os

import numpy as np
from fastapi.testclient import TestClient

os.environ["GAME_DISABLE_LLM"] = "1"   # tests never call a real LLM

from app.ai.companion import build_facts, guard, rule_answer  # noqa: E402
from app.main import app  # noqa: E402
from learned.behaviour import Behaviour  # noqa: E402
from learned.data import ledger  # noqa: E402
from learned.twin import ART, model_card  # noqa: E402

client = TestClient(app)
WORK = {"month": 0, "actions": [{"type": "life_choice", "id": "start_work"}]}


def test_split_is_the_locked_peers_split_and_learning_uses_train_only():
    L = ledger()
    tr, va, te = (set(L.people.index[L.rows(k)]) for k in ("train", "validation", "test"))
    assert len(tr) == 958 and len(va) == 121 and len(te) == 121 and not (tr & te) and not (va & te)


def test_learned_seasons_and_noise_make_sense():
    L = ledger()
    tr = L.rows("train")
    b = Behaviour(3).fit(L.people.iloc[tr], L.flows[tr], L.month_of_year)
    assert b.season_a["employed"][12] > 0.2          # December bonus for employees (13th salary)
    assert b.season_c["employed"][2] < 0 and b.season_c["employed"][8] < 0   # February / August spending
    assert all(abs(p) < 0.3 for p in b.phi.values())   # month-to-month surprises barely carry over
    st = np.array(["employed"] * 3, dtype=object)
    inc = np.array([4000.0, 4000.0, 4000.0])
    low, high = b.mean_flow(st, inc, np.array([800.0, 800.0, 800.0])), b.mean_flow(st, inc, np.array([2500.0] * 3))
    assert (low > high).all()                          # higher rent, less saved


def test_backtest_learned_twin_beats_hmm_alone_and_persistence_on_the_locked_test():
    rep = json.loads((ART / "backtest.json").read_text())
    t = rep["test"]
    assert t["hmm+gmm+history"]["n"] == 121
    assert t["hmm+gmm+history"]["mae_cash_end_chf"] < t["hmm"]["mae_cash_end_chf"]
    assert t["hmm+gmm+history"]["mae_cash_end_chf"] < t["persistence"]["mae_cash_end_chf"]
    assert 0.7 <= t["hmm+gmm+history"]["range_hit_rate_p10_p90"] <= 0.9
    assert set(rep["chosen_on_validation"]) == {"n_components", "kappa", "noise_scale", "layers"}
    assert model_card()["sizes"] == {"train": 958, "validation": 121, "test": 121}


def test_twin_fan_from_the_players_state():
    req = {"persona_id": "P00001", "turns": [WORK], "current_month": 12, "n": 20, "seed": 42, "world": 9}
    r = client.post("/twin/fan", json=req)
    assert r.status_code == 200, r.text
    f = r.json()
    assert f["start_state"] == "employed" and f["inputs"]["income"] > 0
    n = len(f["wealth"]["p50"])
    assert all(len(v) == n for v in f["states"].values())
    assert f["states"]["retired"][-1] == 1.0 and f["summary"]["retirement_age_p50"] == 65
    w = f["wealth"]
    assert all(a <= b <= c for a, b, c in zip(w["p10"], w["p50"], w["p90"]))
    assert client.post("/twin/fan", json=req).json() == f            # same seed, same lives
    s = client.post("/twin/fan", json={**req, "turns": [], "current_month": 0}).json()
    assert s["start_state"] == "student" and s["assumed_degree_after_months"] == 12
    assert client.get("/twin/model").status_code == 200


def test_guard_blocks_invented_numbers_unknown_facts_and_certainty():
    facts = {"now.cash": {"label": "cash now", "value": 1234.0, "unit": "chf", "source": "town engine"}}
    ok, rejected = guard([{"text": "You have {now.cash}."}, {"text": "You will have 5000 soon."},
                          {"text": "That is {made.up}."}, {"text": "Your savings are guaranteed to grow."}], facts)
    assert [o["text"] for o in ok] == ["You have CHF 1'234."]
    assert len(rejected) == 3


def test_companion_answers_from_engine_facts_and_only_proposes_allowed_actions():
    req = {"persona_id": "P00001", "turns": [WORK], "current_month": 14, "seed": 42, "world": 9}
    r = client.post("/ai/ask", json={"request": req, "message": "What if I do a full-time master's?"}).json()
    assert r["mode"] == "rules" and r["answer"]
    assert r["action"]["id"] == "master_ft" and r["action"]["enabled"] is True
    assert all(fid in r["facts_used"] for a in r["answer"] for fid in a["facts"])
    r = client.post("/ai/ask", json={"request": req, "message": "Should I get my degree?"}).json()
    assert r["action"] is None or r["action"]["enabled"] is False   # not enrolled any more: not offered as doable
    r = client.post("/ai/ask", json={"request": req, "message": "How accurate is your model?"}).json()
    assert any("model.mae" in a["facts"] for a in r["answer"])


def test_combined_engine_layers_are_judged_on_validation():
    rep = json.loads((ART / "backtest.json").read_text())
    abl = rep["ablation_on_validation"]
    assert {"baseline", "keep 0.95", "inflation regimes", "major expenses"} <= set(abl)
    for name, v in abl.items():
        if name != "baseline":   # adopted only when it lowered the validation error
            assert v["adopted"] == (v["validation_gain"] >= 0.005 and v["adopted"])
    assert rep["chosen_on_validation"]["layers"] == rep["adopted_layers"]
    tc = rep["tax_check_on_validation"]
    assert set(tc) == {"with_gross_to_net", "without_gross_to_net"}


def test_hmm_income_is_net_of_swiss_contributions_and_tax():
    from learned.twin import hmm
    from learned.hmm import profile_for
    row = ledger().people.loc["P00001"]
    p = profile_for(row, age=40, opening_cash=0, employment="employed", ref_income=8000, rent=1500)
    paths = hmm().simulate(p, horizon_months=24, n_paths=20, seed=0)
    assert (paths.gross_income_chf >= paths.income_chf - 1e-6).all() and (paths.gross_income_chf > paths.income_chf).any()
