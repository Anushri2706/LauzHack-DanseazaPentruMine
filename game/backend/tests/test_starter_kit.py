"""§9.8 Cross-check against the official starter kit (starter-kit/financial_twin.py, stdlib).

With uncertainty, inflation, raises, returns and saving switched off, and the starter kit's
own inputs, our engine must reproduce its monthly cash path for all three scenarios.
"""
import json
import sys
from pathlib import Path

import pytest

from app.engine.simulate import run_branch
from app.engine.state import Decision
from tests.conftest import PLAIN, make_params

KIT = Path(__file__).resolve().parents[3] / "challenge-instructions-and-data" / "starter-kit"
sys.path.insert(0, str(KIT))
import financial_twin  # noqa: E402

PROFILE = json.loads((KIT / "sample_profile.json").read_text())
SCENARIOS = {s["name"]: s for s in json.loads((KIT / "scenario_config.json").read_text())["scenarios"]}

# Starter kit's master's: 2 annual tuition payments of 2,500 (months 0 and 12).
KIT_PARAMS = make_params(**PLAIN, tuition_annual=2500, tuition_payments_per_year=1)

CASES = {
    "Start working": [Decision(0, "start_work")],
    "Master's degree": [Decision(0, "master_ft")],
    "Move abroad": [Decision(0, "move_abroad")],
}


@pytest.mark.parametrize("name", list(CASES))
def test_matches_starter_kit(name):
    kit = financial_twin.simulate(PROFILE, SCENARIOS[name])
    ours = run_branch(KIT_PARAMS, CASES[name]).rows
    assert len(ours) == len(kit.cash_by_month_chf) == 120
    for t, (r, k) in enumerate(zip(ours, kit.cash_by_month_chf)):
        assert r["cash"] == pytest.approx(k, abs=0.01), f"month {t}"
    assert ours[-1]["cash"] == pytest.approx(kit.closing_cash_chf, abs=0.01)
    assert min([8000.0] + [r["cash"] for r in ours]) == pytest.approx(kit.minimum_cash_chf, abs=0.01)
