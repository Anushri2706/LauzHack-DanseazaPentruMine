"""Split, grid search on validation, then one final evaluation on the locked test set.

    cd game/backend && .venv/bin/python -m peers.train

Split: by person, stratified by job status, 80% train / 10% validation / 10% test, seed 42, saved to
peers/artifacts/split.json. The test people are not touched until the model is chosen.
Neighbour pool: the training people only (also for the test evaluation).
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .model import GROUPS, STATUSES, PeerModel, Scaler, load_people, metrics

ART = Path(__file__).resolve().parent / "artifacts"
SEED = 42
K_GRID = (1, 2, 3, 5, 8, 12, 20, 30, 50, 80)
W_GRID = (0.0, 0.5, 1.0, 2.0)
STATUS_FEATURE_WEIGHTS = (1.0, 4.0, 16.0)


def split(df: pd.DataFrame, seed: int = SEED) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    parts = {"train": [], "validation": [], "test": []}
    for status in STATUSES:
        ids = np.array(sorted(df.index[df.status == status]))
        rng.shuffle(ids)
        n = len(ids)
        n_test, n_val = round(0.1 * n), round(0.1 * n)
        parts["test"] += ids[:n_test].tolist()
        parts["validation"] += ids[n_test:n_test + n_val].tolist()
        parts["train"] += ids[n_test + n_val:].tolist()
    return {k: sorted(v) for k, v in parts.items()}


# ----------------------------------------------------------------------------- baselines
def baseline(train: pd.DataFrame, query: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Mean (and P10/P90) of the training people in the same group; falls back to everyone."""
    out = []
    for _, r in query.iterrows():
        g = train
        for k in keys:
            sub = g[g[k] == r[k]]
            if len(sub) == 0:
                break
            g = sub
        v = g.flow.values
        out.append({"pred": v.mean(), "p10": np.percentile(v, 10), "p90": np.percentile(v, 90)})
    return pd.DataFrame(out, index=query.index)


BASELINES = {"everyone's average": [], "same decision (job status) average": ["status"],
             "same decision + household average": ["status", "household"]}


# ----------------------------------------------------------------------------- fast grid search
def grid_search(scaler: Scaler, train: pd.DataFrame, val: pd.DataFrame) -> tuple[list[dict], dict]:
    gt, gv = scaler.groups(train), scaler.groups(val)
    Dg = {g: ((gv[g][:, None, :] - gt[g][None, :, :]) ** 2).sum(-1) for g in GROUPS}
    Ds = ((gv["status"][:, None, :] - gt["status"][None, :, :]) ** 2).sum(-1)   # 0 same status, 2 different
    same = np.asarray(val.status, dtype=object)[:, None] == np.asarray(train.status, dtype=object)[None, :]
    y, yv = train.flow.values, val.flow.values
    kmax = max(K_GRID)
    results = []
    decisions = [("filter", 0.0)] + [("feature", w) for w in STATUS_FEATURE_WEIGHTS]
    for ws in itertools.product(W_GRID, repeat=len(GROUPS)):
        if not any(ws):
            continue
        D = sum(w * Dg[g] for w, g in zip(ws, GROUPS) if w)
        for decision, sw in decisions:
            DD = np.where(same, D, np.inf) if decision == "filter" else D + sw * Ds
            idx = np.argpartition(DD, kmax, axis=1)[:, :kmax]
            order = np.take_along_axis(DD, idx, 1).argsort(1)
            idx = np.take_along_axis(idx, order, 1)
            d = np.sqrt(np.take_along_axis(DD, idx, 1))
            v = y[idx]
            finite = np.isfinite(d)
            for dw in (False, True):
                w = np.where(finite, 1.0 / (d + 1e-3) if dw else 1.0, 0.0)
                cw, cv = np.cumsum(w, 1), np.cumsum(w * v, 1)
                for k in K_GRID:
                    pred = cv[:, k - 1] / cw[:, k - 1]
                    results.append({"weights": dict(zip(GROUPS, ws)), "decision": decision, "status_weight": sw,
                                    "distance_weighted": dw, "k": k, "val_mae": float(np.abs(pred - yv).mean())})
    results.sort(key=lambda r: r["val_mae"])
    return results, results[0]


def main() -> None:
    t0 = time.time()
    ART.mkdir(exist_ok=True)
    df = load_people()
    assert (df.months == 24).all(), "every person should have 24 months"
    parts = split(df)
    (ART / "split.json").write_text(json.dumps({"seed": SEED, "stratified_by": "employment_status",
                                                **{k: v for k, v in parts.items()}}, indent=1))
    train, val, test = (df.loc[parts[k]] for k in ("train", "validation", "test"))
    scaler = Scaler.fit(train)   # normalisation from the training people only

    report = {"sizes": {k: len(v) for k, v in parts.items()},
              "target": "average monthly net cash flow over 24 months (CHF/month); cash after 24 months = "
                        "opening cash + 24 x flow", "validation": {}, "test": {}}
    for name, keys in BASELINES.items():
        b = baseline(train, val, keys)
        report["validation"][name] = metrics(val.flow.values, b.pred.values, b.p10.values, b.p90.values)

    results, best = grid_search(scaler, train, val)
    model = PeerModel(k=best["k"], weights=best["weights"], decision=best["decision"],
                      status_weight=best["status_weight"], distance_weighted=best["distance_weighted"])
    pv = model.predict(scaler, train, val)
    report["validation"]["kNN (people like you)"] = metrics(val.flow.values, pv.pred.values, pv.p10.values,
                                                            pv.p90.values)
    report["grid"] = {"combinations": len(results), "best": best, "top5": results[:5],
                      "k_curve_for_best_weights": {
                          r["k"]: r["val_mae"] for r in results
                          if r["weights"] == best["weights"] and r["decision"] == best["decision"]
                          and r["status_weight"] == best["status_weight"]
                          and r["distance_weighted"] == best["distance_weighted"]}}

    # ---- the test set is opened exactly once, with everything fixed
    pt = model.predict(scaler, train, test)
    report["test"]["kNN (people like you)"] = metrics(test.flow.values, pt.pred.values, pt.p10.values,
                                                      pt.p90.values)
    for name, keys in BASELINES.items():
        b = baseline(train, test, keys)
        report["test"][name] = metrics(test.flow.values, b.pred.values, b.p10.values, b.p90.values)
    report["test_by_status"] = {
        s: metrics(test.flow.values[m], pt.pred.values[m], pt.p10.values[m], pt.p90.values[m])
        for s in STATUSES if (m := (test.status.values == s)).any()}
    out = test[["name", "status", "household", "age", "income", "flow"]].assign(
        pred=pt.pred, p10=pt.p10, p90=pt.p90, neighbours=pt.neighbours.map(lambda x: " ".join(x[:5])))
    out.to_csv(ART / "test_predictions.csv")
    (ART / "model.json").write_text(json.dumps({"model": model.__dict__, "scaler": scaler.__dict__}, indent=1))
    report["seconds"] = round(time.time() - t0, 1)
    (ART / "report.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: report[k] for k in ("sizes", "validation", "test", "test_by_status")}, indent=1,
                     default=lambda x: round(x, 3)))
    print("best:", json.dumps(best), "| combinations:", len(results), "| seconds:", report["seconds"])


if __name__ == "__main__":
    main()
