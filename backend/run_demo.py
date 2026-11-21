"""Print Alexia's three decisions vs baseline using the current engine (stub until the HMM lands).

Usage: python backend/run_demo.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from regime import TwinSession  # noqa: E402

s = TwinSession.from_profile_id("P00001", horizon_months=120, n_paths=1000)
for sid, kind in [("work", "start_working"), ("masters", "masters"), ("abroad", "move_abroad")]:
    s.create_scenario(sid)
    if sid == "abroad":
        s.apply_decision(sid, "start_working", 0)
    s.apply_decision(sid, kind, 0)
for sid in ["baseline", "work", "masters", "abroad"]:
    r = s.simulate(sid)
    print(f"{sid:9s} closing p50 {r['closing_cash_chf']['p50']:>9,} CHF | min p05 {r['minimum_cash_chf']['p05']:>9,} "
          f"| P(ever < reserve) {r['p_ever_below_reserve']:.2f} | first negative month {r['median_first_negative_month']}")
print(json.dumps(s.compare("work", "masters")["delta_b_minus_a"], indent=2))
