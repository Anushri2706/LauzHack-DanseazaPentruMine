import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.engine.assumptions import ALEXIA_DEFAULTS  # noqa: E402


def make_params(**overrides) -> dict:
    p = {k: float(v) for k, (v, _) in ALEXIA_DEFAULTS.items()}
    p.update({k: float(v) for k, v in overrides.items()})
    return p


# "Plain" world: no inflation, raises, returns or saving, so cash maths is easy to check by hand.
PLAIN = dict(inflation_annual=0, raise_annual=0, return_annual=0, savings_rate=0)


@pytest.fixture
def params():
    return make_params()


@pytest.fixture
def plain():
    return make_params(**PLAIN)
