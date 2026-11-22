"""People-like-you kNN: the split and the saved artifacts."""
import json
from pathlib import Path

from peers.model import STATUSES, load_people
from peers.train import split

ART = Path(__file__).resolve().parents[1] / "peers" / "artifacts"


def test_split_is_by_person_80_10_10_and_stratified():
    df = load_people()
    parts = split(df)
    ids = [set(v) for v in parts.values()]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert sum(len(v) for v in ids) == len(df) == 1200
    assert [len(parts[k]) for k in ("train", "validation", "test")] == [958, 121, 121]
    for s in STATUSES:
        n = (df.status == s).sum()
        assert abs(sum(df.loc[parts["test"]].status == s) - 0.1 * n) <= 1
    assert split(df) == parts  # fixed seed: the locked test set never moves


def test_saved_split_matches_and_report_has_test_metrics():
    saved = json.loads((ART / "split.json").read_text())
    assert saved["test"] == split(load_people())["test"]
    rep = json.loads((ART / "report.json").read_text())
    knn = rep["test"]["kNN (people like you)"]
    assert knn["n"] == 121 and knn["mae_flow_chf_per_month"] < rep["test"]["same decision (job status) average"]["mae_flow_chf_per_month"]


def test_gmm_report_uses_the_same_locked_test_set_and_picks_on_validation():
    rep = json.loads((ART / "report_gmm.json").read_text())
    assert rep["test"]["GMM"]["n"] == 121 and rep["grid_size"] == 48
    v = rep["validation"]
    assert rep["chosen_on_validation"] == ("GMM" if v["GMM"]["mae_flow_chf_per_month"]
                                           < v["kNN"]["mae_flow_chf_per_month"] else "kNN")
    assert 0.7 < rep["test"]["GMM"]["range_hit_rate_p10_p90"] < 0.95   # ranges roughly calibrated
