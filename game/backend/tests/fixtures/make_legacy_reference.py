"""Write legacy_reference.json: outputs of the original engines, frozen for the regression tests.

Generated once from commit 33470ff, the last commit that still had <repo>/backend/regime and
<repo>/twin. To regenerate, check out that commit's backend/ and twin/ folders, then run from
game/backend:
    ../../.venv/bin/python -m tests.fixtures.make_legacy_reference
Large arrays are stored as SHA-256 digests (dtype, shape and bytes), so a match means bit-identical.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from tests.regression_cases import EQUIV_LEGACY, INTERVENTION_SETS, peer_query, profile_kwargs

REPO = Path(__file__).resolve().parents[4]
OUT = Path(__file__).with_name("legacy_reference.json")


def digest(a) -> str:
    a = np.ascontiguousarray(a)
    return hashlib.sha256(f"{a.dtype.str}{a.shape}".encode() + a.tobytes()).hexdigest()


def main() -> None:
    sys.path[:0] = [str(REPO / "backend"), str(REPO)]
    import regime
    import regime.types as T
    from regime import net_pay
    from regime import wealth as old_wealth
    from regime.cash_engine import OneOff
    from twin import peers

    ivs = lambda spec: [T.Intervention(k, t0, d, **kw) for k, t0, d, kw in spec]
    prof = lambda emp="employed": T.Profile(**profile_kwargs(emp))
    eng = regime.RegimeEngine(regime.fit_emissions(), regime.load_transition_priors())
    em = eng.emissions
    ref: dict = {"source_commit": "33470ff",
                 "emissions": {"params": em.params, "household_shift": em.household_shift,
                               "december_bonus_quantiles": em.december_bonus_quantiles}}

    ref["hmm_paths"] = {}
    for name, spec in INTERVENTION_SETS.items():
        for emp in ("employed", "student", "part_time"):
            p = eng.simulate(prof(emp), 96, 300, ivs(spec), {"job_loss_hazard": 0.006}, seed=5)
            ref["hmm_paths"][f"{name}/{emp}"] = {f: digest(getattr(p, f)) for f in
                                                 ("states", "income_chf", "net_flow_chf", "gross_income_chf",
                                                  "occupancy", "transition_draws")}

    stub = regime.RegimeEngine.stub().simulate(prof(), 60, 3, ivs(INTERVENTION_SETS["masters"]))
    ref["stub_net_flow"] = digest(stub.net_flow_chf)

    params = eng.income_table.params
    g = np.linspace(0, 30000, 61)
    ref["net_pay"] = {k: digest(net_pay.net_from_gross(g, 33, k, params)) for k in ("wage", "ui")}
    ref["net_pay"]["gross_from_net_5000_30"] = net_pay.gross_from_net(5000, 30, params)

    pa = eng.simulate(prof(), 120, 200, ivs(INTERVENTION_SETS["masters"]), seed=9)
    w = old_wealth.build(prof(), pa, [OneOff(0, 8000.0), OneOff(12, 8000.0)], params, 9, 0.7)
    ref["wealth"] = {"cash": digest(w.cash()), "net_worth_years": digest(w.net_worth_years()),
                     "summary": w.summary()}

    s = regime.TwinSession(prof(), engine=eng, horizon_months=120, n_paths=250, seed=3, wealth=True)
    s.create_scenario("m", copy_from="baseline")
    s.apply_decision("m", "masters", 6, duration=24, annual_tuition_chf=9000.0)
    wo = s._run(s.scenarios["m"], 250, 3).wealth
    ref["pipeline"] = {"cash": digest(wo.cash()), "net_worth_years": wo.net_worth_years()[:, 1:].tolist()}

    ref["decision_mapping"] = []
    for old_kind, old_params in EQUIV_LEGACY:
        sess = regime.TwinSession(T.Profile(age=30, employment="employed", ref_income=5000, rent=1500),
                                  horizon_months=120)
        o_ivs, o_offs = sess._decision_to_interventions(old_kind, 5, old_params)
        ref["decision_mapping"].append({
            "interventions": [[i.kind, i.start_month, i.duration, i.state, i.transition, i.factor, i.attribute]
                              for i in o_ivs],
            "one_offs": [[o.month, o.amount_chf] for o in o_offs]})

    pop = peers.population()[:50]
    me = peer_query()
    ref["peers"] = [{"id": q["id"], "wealth": q["wealth"], "income": q["income"], "rent": q["rent"],
                     "distance": peers.distance({**me, "id": None}, q)} for q in pop]

    OUT.write_text(json.dumps(ref, indent=1, default=lambda o: list(o) if isinstance(o, tuple) else float(o)) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
