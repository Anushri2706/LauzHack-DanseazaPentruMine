"""Public datasets from the challenge resources (World Bank, SNB, BIS, FRED): snapshots, defaults, cross-checks."""
import csv
import math

import pytest

from app.engine.assumptions import ALEXIA_DEFAULTS, META
from app.engine.labour_data import BANDS
from app.engine.public_data import DIR, context, implied_unemployment, public_defaults, sources

KEYS = ("inflation_annual", "mortgage_rate", "home_growth_annual")


def test_every_snapshot_has_provenance():
    src = sources()
    for name, s in src.items():
        assert (DIR / f"{name}.csv").exists()
        for field in ("provider", "code", "unit", "period", "url", "retrieved", "terms"):
            assert s[field], f"{name}.{field}"
    assert {p["provider"] for p in src.values()} == {"World Bank", "SNB", "BIS", "FRED"}


def test_public_defaults_are_tagged_and_explained():
    for k in KEYS:
        v, tag = ALEXIA_DEFAULTS[k]
        assert tag == "public_data" and v == public_defaults()[k][0]
        assert META[k].min <= v <= META[k].max
        assert "retrieved" in META[k].note


def test_inflation_recomputes_from_the_csv():
    rows = list(csv.DictReader(open(DIR / "wb_inflation.csv")))[-35:]
    geo = math.prod(1 + float(r["value"]) / 100 for r in rows) ** (1 / 35) - 1
    assert public_defaults()["inflation_annual"][0] == pytest.approx(geo, abs=5e-5)
    assert 0 <= geo < 0.03  # SNB price stability is 0-2%


def test_mortgage_is_the_mean_10y_fixed_rate_and_above_the_policy_rate():
    rates = [float(r["fixed_10y"]) for r in csv.DictReader(open(DIR / "snb_mortgage_rates.csv"))]
    m = public_defaults()["mortgage_rate"][0]
    assert m == pytest.approx(sum(rates) / len(rates) / 100, abs=5e-5)
    assert m > context()["policy_rate_mean_pct"] / 100  # a mortgage costs more than the SNB policy rate


def test_home_growth_is_real_growth_plus_inflation():
    infl = public_defaults()["inflation_annual"][0]
    home = public_defaults()["home_growth_annual"][0]
    real = (1 + home) / (1 + infl) - 1
    assert 0 < real < 0.03


def test_job_rates_match_world_bank_unemployment():
    p = {k: v for k, (v, _) in ALEXIA_DEFAULTS.items()}
    implied = implied_unemployment(p, BANDS)
    assert implied * 100 == pytest.approx(context()["unemployment_mean_pct"], abs=1.0)


def test_fx_context_is_sane():
    c = context()
    assert 0.8 < c["eur_chf_last"] < 1.8 and 0 < c["eur_chf_vol_annual"] < 0.2
    assert 2009 in c["gdp_down_years"] and 2020 in c["gdp_down_years"]
