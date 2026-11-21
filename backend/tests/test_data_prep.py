"""Data prep stage (backend plan §2): person-month table, holdout split, outliers, December."""
import numpy as np
import pytest

from twin import data
from twin.types import STATES


@pytest.fixture(scope="module")
def prep():
    return data.prepare_data()


def test_person_month_columns(prep):
    pm = prep.person_months
    assert len(pm) == 28800 and pm["profile_id"].nunique() == 1200
    assert np.allclose(pm["spend"], pm["reference_monthly_net_income_chf"] - pm["net_transactions_chf"])
    assert np.allclose(pm["spend_ratio"], pm["spend"] / pm["reference_monthly_rent_chf"])
    assert set(pm["month_of_year"]) == set(range(1, 13))
    assert (pm["is_december"] == (pm["month_of_year"] == 12)).all()
    assert pm["age_band"].notna().all() and pm["state"].isin(STATES).all()
    assert (pm.loc[pm["t"] == 12, "age"].values == pm.loc[pm["t"] == 0, "age"].values + 1).all()


def test_holdout_split_by_profile_stratified(prep):
    pm = prep.person_months
    assert pm.groupby("profile_id")["split"].nunique().max() == 1
    n_per_state = pm.drop_duplicates("profile_id")["state"].value_counts()
    for s in STATES:
        assert len(prep.holdout_ids[s]) == max(1, round(0.2 * n_per_state[s]))
        assert set(pm.loc[pm["profile_id"].isin(prep.holdout_ids[s]), "state"]) == {s}
        assert (prep.train_body["state"] == s).any()
    assert len(prep.holdout_ids["job_search"]) == 7
    assert data.holdout_split(pm, 0.2, 0) == prep.holdout_ids
    assert data.holdout_split(pm, 0.2, 1) != prep.holdout_ids


def test_outliers_exclude_december_and_body(prep):
    pm = prep.person_months
    out = pm.loc[pm["is_outlier"]]
    assert not out["is_december"].any()
    assert (out["robust_z"].abs() > 6).all()
    assert len(out) < 0.01 * len(pm)
    body = prep.train_body
    assert not body["is_outlier"].any() and not body["is_december"].any()
    assert (body["split"] == "train").all()


def test_december_bonus(prep):
    assert 0.2 < prep.december_bonus["employed"] < 0.4
    for s in ["part_time", "retired", "student", "job_search"]:
        assert abs(prep.december_bonus[s]) < 0.1
    pm = prep.person_months
    nd = ~pm["is_december"]
    assert pm.loc[nd, "dec_bonus_income"].isna().all() and pm.loc[~nd, "dec_bonus_income"].notna().all()
    assert np.allclose(pm.loc[nd, "spend_ratio_adj"], pm.loc[nd, "spend_ratio"])


def test_summaries(prep):
    rs = prep.regime_stats
    assert list(rs["state"]) == STATES
    assert (rs["body_rows"] > 500).all()
    assert rs["spend_ratio_median"].between(1.5, 2.1).all()
    assert (rs["spend_ratio_within_sd"] < rs["spend_ratio_std"]).all()
    assert rs["acf1_median"].abs().max() < 0.2
    b = prep.baseline_levels
    reg = b[(b["level"] == "regime") & (b["age_band"] == "all")]
    assert set(reg["group"]) == set(STATES) and set(reg["variable"]) == {"income", "rent"}
    assert reg.loc[(reg["group"] == "student") & (reg["variable"] == "income"), "median"].item() == 1200
    assert set(prep.spend_regression["state"]) == set(STATES)


def test_write_artifacts_roundtrip(prep, tmp_path):
    paths = data.write_artifacts(prep, tmp_path)
    assert all(p.exists() for p in paths.values())
    assert data.load_holdout_ids(paths["holdout_profiles"]) == prep.holdout_ids
