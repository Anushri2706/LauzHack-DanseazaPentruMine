import os
import sys
from pathlib import Path

import pytest

GAME_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GAME_BACKEND))
os.environ["GAME_DISABLE_LLM"] = "1"     # tests never call a real LLM


@pytest.fixture(scope="session")
def assumptions():
    from app.engine.assumptions import AssumptionSet
    return AssumptionSet.default()


@pytest.fixture(scope="session")
def hmm_engine():
    from app.engine.simulate import base_engine
    return base_engine("hmm")


@pytest.fixture
def player_raw():
    return dict(age=27, employment="employed", income_amount_chf=95000, income_period="yearly",
                income_basis="gross", rent_chf_month=1600, cash_chf=20000, investments_chf=5000,
                debt_chf=10000, debt_annual_rate=0.04, other_expenses_chf_month=1800, canton="VD",
                household="single")


@pytest.fixture
def engine_profile(player_raw, assumptions):
    from app.engine.profile import to_engine_profile, validate_profile
    return to_engine_profile(validate_profile(player_raw), assumptions)
