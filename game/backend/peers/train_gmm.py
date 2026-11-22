"""GMM version of "people like you": grid search on validation, one evaluation on the locked test set.

    cd game/backend && .venv/bin/python -m peers.train_gmm

Uses exactly the same split (peers/artifacts/split.json, by person, stratified, seed 42) and the same
normalisation (fitted on the 958 training people) as the kNN model, and compares with its saved report.
The winner between kNN and GMM is chosen on VALIDATION, never on test.
"""
from __future__ import annotations

import itertools
import json
import time
import warnings

import numpy as np

from .model import STATUSES, Scaler, load_people, metrics
from .gmm import FEATURE_SETS, GMMModel
from .train import ART

COMPONENTS = (1, 2, 3, 4, 6, 8)
COVARIANCES = ("full", "diag")


def main() -> None:
    warnings.filterwarnings("ignore")  # sklearn convergence chatter on tiny groups
    t0 = time.time()
    df = load_people()
    split = json.loads((ART / "split.json").read_text())
    train, val, test = (df.loc[split[k]] for k in ("train", "validation", "test"))
    scaler = Scaler.fit(train)

    grid = []
    for fs, c, cov in itertools.product(FEATURE_SETS, COMPONENTS, COVARIANCES):
        m = GMMModel(fs, c, cov).fit(scaler, train)
        pv = m.predict(scaler, val)
        grid.append({"feature_set": fs, "n_components": c, "covariance_type": cov,
                     **metrics(val.flow.values, pv.pred.values, pv.p10.values, pv.p90.values)})
    grid.sort(key=lambda r: r["mae_flow_chf_per_month"])
    best = grid[0]
    model = GMMModel(best["feature_set"], best["n_components"], best["covariance_type"]).fit(scaler, train)

    knn = json.loads((ART / "report.json").read_text())
    report = {"grid_size": len(grid), "best": best, "top5": grid[:5],
              "validation": {"GMM": metrics(val.flow.values, *[model.predict(scaler, val)[c].values
                                                              for c in ("pred", "p10", "p90")]),
                             "kNN": knn["validation"]["kNN (people like you)"]}}
    pt = model.predict(scaler, test)   # the locked test set: one evaluation, everything fixed
    report["test"] = {"GMM": metrics(test.flow.values, pt.pred.values, pt.p10.values, pt.p90.values),
                      "kNN": knn["test"]["kNN (people like you)"],
                      **{k: v for k, v in knn["test"].items() if k != "kNN (people like you)"}}
    report["test_by_status_gmm"] = {
        s: metrics(test.flow.values[mk], pt.pred.values[mk], pt.p10.values[mk], pt.p90.values[mk])
        for s in STATUSES if (mk := (np.asarray(test.status, dtype=object) == s)).any()}
    report["chosen_on_validation"] = ("GMM" if report["validation"]["GMM"]["mae_flow_chf_per_month"]
                                      < report["validation"]["kNN"]["mae_flow_chf_per_month"] else "kNN")
    test[["name", "status", "income", "flow"]].assign(pred=pt.pred, p10=pt.p10, p90=pt.p90,
                                                      components=pt.components).to_csv(ART / "test_predictions_gmm.csv")
    report["seconds"] = round(time.time() - t0, 1)
    (ART / "report_gmm.json").write_text(json.dumps(report, indent=1, default=float))
    r = lambda d: {k: round(v, 3) if isinstance(v, float) else v for k, v in d.items()}  # noqa: E731
    print("best:", r(best))
    for split_name in ("validation", "test"):
        for k, v in report[split_name].items():
            print(f"{split_name:10s} {k:38s}", r(v))
    print("by status (GMM, test):", {s: round(v["mae_flow_chf_per_month"]) for s, v in report["test_by_status_gmm"].items()})
    print("chosen on validation:", report["chosen_on_validation"], "| seconds:", report["seconds"])
    print("top5:", [(g["feature_set"], g["n_components"], g["covariance_type"], round(g["mae_flow_chf_per_month"]),
                     round(g["range_hit_rate_p10_p90"], 2)) for g in grid[:5]])


if __name__ == "__main__":
    main()
