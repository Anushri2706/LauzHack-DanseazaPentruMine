"""Gross -> net pay. Each case states the arithmetic it expects (assumptions/tax_assumptions.csv)."""
import numpy as np
import pytest

from regime import RegimeEngine
from regime.income import IncomeTable
from regime.net_pay import gross_from_net, net_from_gross
from regime.types import Profile

P = IncomeTable.from_csv().params


def test_wage_at_30_hand_checked():
    # 80,000/yr at 30: AHV/IV/EO 5.3% + accident 1.2% = 5,200; ALV 1.1% = 880;
    # BVG half of 7% on (80,000 - 26,460) = 1,873.90; tax 11.5% = 9,200.
    # Net 62,846.10/yr = 5,237.175/month.
    assert float(net_from_gross(80000 / 12, 30, "wage", P)) == pytest.approx(62846.10 / 12)


def test_no_pension_fund_under_25():
    # Same wage at 22: no BVG. 80,000 - 5,200 - 880 - 9,200 = 64,720.
    assert float(net_from_gross(80000 / 12, 22, "wage", P)) == pytest.approx(64720 / 12)


def test_unemployment_benefit_hand_checked():
    # 4,900/month = 58,800/yr: AHV/IV/EO + accident 6.5% = 3,822; no ALV, no BVG;
    # tax interpolated 6% + 18,800/20,000 x 3% = 8.82% -> 5,186.16. Net 49,791.84.
    assert float(net_from_gross(4900, 30, "ui", P)) == pytest.approx(49791.84 / 12)


def test_alv_stops_at_the_ceiling():
    rich = float(net_from_gross(200000 / 12, 22, "wage", P)) * 12
    # 200,000: 6.5% = 13,000; ALV only on 148,200 = 1,630.20; tax 20% = 40,000.
    assert rich == pytest.approx(200000 - 13000 - 1630.20 - 40000)


def test_gross_from_net_inverts():
    for age, net in ((22, 4000.0), (30, 5237.175), (50, 9000.0)):
        gross = gross_from_net(net, age, P)
        assert float(net_from_gross(gross, age, "wage", P)) == pytest.approx(net, abs=0.01)
        assert gross > net


def test_switch_off_is_identity():
    off = IncomeTable.from_csv().with_overrides({"apply_income_tax": 0}).params
    assert float(net_from_gross(7000, 30, "wage", off)) == 7000
    assert gross_from_net(7000, 30, off) == 7000


def test_data_net_income_round_trips_through_the_engine():
    # A worker's reference income in the data is net: the engine converts it up to gross
    # and back, so what lands in the account is that same reference income.
    worker = Profile(age=30, employment="employed", ref_income=5000.0, rent=1500.0)
    paths = RegimeEngine.stub().simulate(worker, horizon_months=3, n_paths=2)
    assert paths.income_chf[0, 0] == pytest.approx(5000.0, abs=0.01)
    assert paths.gross_income_chf[0, 0] > 6000


def test_student_stipend_is_not_taxed_again():
    student = Profile(age=20, employment="student", ref_income=1200.0, rent=900.0)
    paths = RegimeEngine.stub().simulate(student, horizon_months=2, n_paths=1)
    assert paths.income_chf[0, 0] == pytest.approx(1200.0)
    assert np.allclose(paths.income_chf, paths.gross_income_chf)
