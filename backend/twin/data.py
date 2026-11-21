"""Loaders for the synthetic challenge data in Data/ and the assumption CSVs, plus the data prep
stage for emission fitting (backend plan §2).

Emission variable: ``spend_ratio = (reference income - net flow) / rent``. It is stable across
regimes, unlike net / income, which blows up where income is tiny (student, job search).
``savings_rate`` is derived downstream as ``net_flow_chf / income_chf``.

Run ``python prepare_data.py`` (from backend/) to write the artifacts to backend/artifacts/.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .types import STATES, Profile

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
DATA_DIR = REPO_ROOT / "Data"
ASSUMPTIONS_DIR = BACKEND_DIR / "assumptions"
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"
STARTER_KIT_DIR = REPO_ROOT / "starter-kit"

AGE_BINS = [17, 25, 35, 50, 65, 120]
AGE_LABELS = ["18-25", "26-35", "36-50", "51-65", "65+"]
MAD_TO_SD = 1.4826


def load_profiles(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(data_dir / "synthetic-profiles.csv")


def load_cash_balances(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(data_dir / "synthetic-cash-balances.csv")


def load_goals(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(data_dir / "synthetic-goals.csv")


def age_band(age) -> pd.Series:
    return pd.cut(pd.Series(age), AGE_BINS, labels=AGE_LABELS)


def person_months(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """One row per profile-month.

    Columns added: ``ratio`` = net / reference income (legacy), ``spend`` = reference income -
    net, ``spend_ratio`` = spend / rent (the emission), ``is_december``, ``month_of_year``,
    ``state`` (regime from employment_status), ``t`` (0..23), ``age`` (age in that month) and
    ``age_band``.
    """
    from .types import state_of

    cb = load_cash_balances(data_dir)
    pr = load_profiles(data_dir)[
        ["profile_id", "employment_status", "age_at_start", "canton", "household_type",
         "reference_monthly_net_income_chf", "reference_monthly_rent_chf"]
    ]
    df = cb.merge(pr, on="profile_id", how="left")
    income = df["reference_monthly_net_income_chf"]
    df["ratio"] = df["net_transactions_chf"] / income
    df["spend"] = income - df["net_transactions_chf"]
    df["spend_ratio"] = df["spend"] / df["reference_monthly_rent_chf"]
    df["month_of_year"] = df["month"].str[-2:].astype(int)
    df["is_december"] = df["month_of_year"] == 12
    df["state"] = df["employment_status"].map(state_of)
    df["t"] = df.groupby("profile_id").cumcount()
    df["age"] = df["age_at_start"] + df["t"] // 12
    df["age_band"] = age_band(df["age"]).values
    return df


# ---------------------------------------------------------------------------
# Data prep stage (plan §2)
# ---------------------------------------------------------------------------

def holdout_split(pm: pd.DataFrame, frac: float = 0.2, seed: int = 0) -> dict[str, list[str]]:
    """Hold out ``frac`` of *profiles* per regime (stratified, seeded); at least one per regime."""
    rng = np.random.default_rng(seed)
    per_profile = pm.drop_duplicates("profile_id")[["profile_id", "state"]]
    out: dict[str, list[str]] = {}
    for s in STATES:
        ids = np.sort(per_profile.loc[per_profile["state"] == s, "profile_id"].to_numpy())
        if len(ids) == 0:
            out[s] = []
            continue
        n_hold = max(1, int(round(frac * len(ids))))
        out[s] = sorted(rng.choice(ids, size=n_hold, replace=False).tolist())
    return out


def robust_z(pm: pd.DataFrame, col: str = "spend_ratio") -> pd.Series:
    """Within-profile robust z: (x - median) / (1.4826 * MAD), with median and MAD taken over the
    profile's non-December months (December carries the bonus, see ``december_bonus``); 0 where
    MAD is 0."""
    ref = pm.loc[~pm["is_december"]]
    med = ref.groupby("profile_id")[col].median()
    mad = (ref[col] - ref["profile_id"].map(med)).abs().groupby(ref["profile_id"]).median() * MAD_TO_SD
    z = (pm[col] - pm["profile_id"].map(med)) / pm["profile_id"].map(mad).replace(0.0, np.nan)
    return z.fillna(0.0)


def december_offsets(pm: pd.DataFrame, col: str = "spend_ratio") -> dict[str, float]:
    """Per regime: median(col | December) - median(col | other months)."""
    out = {}
    for s, g in pm.groupby("state"):
        dec, rest = g.loc[g["is_december"], col], g.loc[~g["is_december"], col]
        out[s] = float(dec.median() - rest.median()) if len(dec) and len(rest) else 0.0
    return out


def december_bonus(pm: pd.DataFrame) -> pd.Series:
    """December bump in income units: (net_Dec - profile median non-December net) / ref income.

    NaN outside December. Employed / self-employed get a bonus spread roughly uniformly over
    0..1.1x income, uncorrelated between the two Decembers of a profile; other regimes ~0.
    """
    med = pm.loc[~pm["is_december"]].groupby("profile_id")["net_transactions_chf"].median()
    bump = (pm["net_transactions_chf"] - pm["profile_id"].map(med)) / pm["reference_monthly_net_income_chf"]
    return bump.where(pm["is_december"])


def december_bonus_stats(pm: pd.DataFrame, by: str = "state") -> pd.DataFrame:
    d = pm.loc[pm["is_december"]]
    return (d.groupby(by)["dec_bonus_income"]
            .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
            .rename(columns=lambda c: f"p{c[:-1].zfill(2)}" if c.endswith("%") else c)
            .reset_index())


def lag1_autocorr(pm: pd.DataFrame, col: str = "spend_ratio_adj") -> pd.DataFrame:
    """Lag-1 autocorrelation of ``col`` per profile (rows ordered by t)."""
    def acf1(x: np.ndarray) -> float:
        x = x - x.mean()
        den = float(np.dot(x, x))
        return float(np.dot(x[:-1], x[1:]) / den) if den > 0 else 0.0

    pm = pm.sort_values(["profile_id", "t"])
    rows = [(pid, g["state"].iat[0], acf1(g[col].to_numpy()), len(g))
            for pid, g in pm.groupby("profile_id")]
    return pd.DataFrame(rows, columns=["profile_id", "state", "acf1", "n"])


def spend_regression(pm: pd.DataFrame) -> pd.DataFrame:
    """Per regime, least squares ``spend = a*rent + b*income`` (no intercept) vs rent only.

    ``income_share`` = b*median(income) / median fitted spend: how much of spending scales with
    income rather than rent. Used to decide whether the emission needs an income term.
    """
    rows = []
    for s in STATES:
        g = pm.loc[pm["state"] == s]
        if g.empty:
            continue
        y = g["spend"].to_numpy()
        rent = g["reference_monthly_rent_chf"].to_numpy()
        inc = g["reference_monthly_net_income_chf"].to_numpy()
        X = np.column_stack([rent, inc])
        (a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
        a_only = float(np.dot(rent, y) / np.dot(rent, rent))
        sst = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - float(np.sum((y - X @ [a, b]) ** 2)) / sst
        r2_rent = 1 - float(np.sum((y - a_only * rent) ** 2)) / sst
        med_fit = a * np.median(rent) + b * np.median(inc)
        rows.append({"state": s, "n": len(g), "a_rent": a, "b_income": b, "r2": r2,
                     "a_rent_only": a_only, "r2_rent_only": r2_rent,
                     "income_share": b * np.median(inc) / med_fit if med_fit else np.nan})
    return pd.DataFrame(rows)


def _q(x: pd.Series, q: float) -> float:
    return float(x.quantile(q)) if len(x) else np.nan


def group_summary(pm: pd.DataFrame, by: str | list[str] = "state",
                  acf: pd.DataFrame | None = None) -> pd.DataFrame:
    """Summary stats per group. Counts use all rows; distribution stats use the emission body
    (training split, non-December, non-outlier). ``within_sd`` is the median per-profile sd of
    the body, i.e. the scale a personalised emission would use (plan §3.1)."""
    by = [by] if isinstance(by, str) else list(by)
    body_mask = (pm["split"] == "train") & ~pm["is_outlier"] & ~pm["is_december"]
    rows = []
    for key, g_all in pm.groupby(by, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        g = g_all.loc[body_mask.loc[g_all.index]]
        prof = g_all.drop_duplicates("profile_id")
        dec = g_all.loc[g_all["is_december"] & (g_all["split"] == "train"), "dec_bonus_income"]
        row = dict(zip(by, key))
        row.update({
            "profiles": len(prof),
            "train_profiles": int((prof["split"] == "train").sum()),
            "holdout_profiles": int((prof["split"] == "holdout").sum()),
            "rows": len(g_all),
            "body_rows": len(g),
            "outlier_rows": int(g_all["is_outlier"].sum()),
            "median_income": float(prof["reference_monthly_net_income_chf"].median()),
            "median_rent": float(prof["reference_monthly_rent_chf"].median()),
            "dec_bonus_income_median": _q(dec, 0.5),
            "dec_bonus_income_mean": float(dec.mean()) if len(dec) else np.nan,
            "dec_bonus_income_p90": _q(dec, 0.9),
        })
        for col, tag in [("spend_ratio", "spend_ratio"), ("ratio", "net_over_income")]:
            x = g[col]
            row.update({f"{tag}_mean": float(x.mean()), f"{tag}_std": float(x.std()),
                        f"{tag}_p05": _q(x, 0.05), f"{tag}_median": _q(x, 0.5), f"{tag}_p95": _q(x, 0.95)})
        row["spend_ratio_within_sd"] = float(g.groupby("profile_id")["spend_ratio"].std().median())
        if acf is not None:
            a = acf.loc[acf["profile_id"].isin(prof["profile_id"]), "acf1"]
            row["acf1_median"] = float(a.median()) if len(a) else np.nan
            row["acf1_frac_sig"] = float((a.abs() > 2 / np.sqrt(24)).mean()) if len(a) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    if by == ["state"]:
        out = out.set_index("state").reindex([s for s in STATES if s in set(out["state"])]).reset_index()
    return out


def baseline_levels(profiles: pd.DataFrame) -> pd.DataFrame:
    """Income and rent baselines from the profiles: count / P25 / median / P75 by regime and by
    raw employment status, each per age band and overall (age_band = "all")."""
    from .types import state_of

    pr = profiles.assign(state=profiles["employment_status"].map(state_of),
                         age_band=age_band(profiles["age_at_start"]).values)
    var_cols = {"income": "reference_monthly_net_income_chf", "rent": "reference_monthly_rent_chf"}
    out = []
    for level, key in [("regime", "state"), ("employment_status", "employment_status")]:
        for by_age in (True, False):
            keys = [key, "age_band"] if by_age else [key]
            for var, col in var_cols.items():
                d = (pr.groupby(keys, observed=True)[col]
                     .describe(percentiles=[0.25, 0.5, 0.75])[["count", "25%", "50%", "75%"]]
                     .reset_index()
                     .rename(columns={key: "group", "25%": "p25", "50%": "median", "75%": "p75"}))
                if not by_age:
                    d["age_band"] = "all"
                d["level"], d["variable"] = level, var
                out.append(d)
    df = pd.concat(out, ignore_index=True)
    df["age_band"] = df["age_band"].astype(str)
    df["count"] = df["count"].astype(int)
    return df[["level", "group", "age_band", "variable", "count", "p25", "median", "p75"]]


@dataclass
class PreparedData:
    """Output of ``prepare_data``.

    ``person_months`` gains: ``split`` (train/holdout), ``robust_z`` and ``is_outlier`` (non-December
    rows only), ``dec_bonus_income`` (December rows), ``spend_ratio_adj`` (December rows with the
    regime's median bonus added back to spending; equal to spend_ratio elsewhere).
    """
    person_months: pd.DataFrame
    holdout_ids: dict[str, list[str]]
    december_bonus: dict[str, float]      # median December bump per regime, in income units
    december_offset: dict[str, float]     # plan §2 variant: median Dec - non-Dec spend_ratio (train)
    regime_stats: pd.DataFrame
    employment_stats: pd.DataFrame
    household_stats: pd.DataFrame
    december_stats: pd.DataFrame
    spend_regression: pd.DataFrame
    baseline_levels: pd.DataFrame
    acf: pd.DataFrame
    settings: dict = field(default_factory=dict)

    @property
    def train(self) -> pd.DataFrame:
        return self.person_months.loc[self.person_months["split"] == "train"]

    @property
    def train_body(self) -> pd.DataFrame:
        """Rows for the emission body fit: training profiles, non-December, outliers removed."""
        pm = self.person_months
        return pm.loc[(pm["split"] == "train") & ~pm["is_outlier"] & ~pm["is_december"]]

    @property
    def train_december(self) -> pd.DataFrame:
        """Training December rows, for fitting the bonus distribution (``dec_bonus_income``)."""
        pm = self.person_months
        return pm.loc[(pm["split"] == "train") & pm["is_december"]]

    @property
    def holdout(self) -> pd.DataFrame:
        return self.person_months.loc[self.person_months["split"] == "holdout"]

    @property
    def outliers(self) -> pd.DataFrame:
        """Shock rows (all splits), kept for an optional shock component."""
        return self.person_months.loc[self.person_months["is_outlier"]]

    def summary(self) -> dict:
        pm = self.person_months
        nd = pm.loc[~pm["is_december"]]
        return {
            "settings": self.settings,
            "rows": len(pm),
            "profiles": int(pm["profile_id"].nunique()),
            "holdout_profiles": {s: len(v) for s, v in self.holdout_ids.items()},
            "december_bonus_income_median": self.december_bonus,
            "december_offset_spend_ratio": self.december_offset,
            "outliers": {"rows": int(pm["is_outlier"].sum()),
                         "profiles_with_outlier": int(pm.loc[pm["is_outlier"], "profile_id"].nunique()),
                         "by_state": pm.loc[pm["is_outlier"], "state"].value_counts().to_dict(),
                         "positive_z": int((pm["is_outlier"] & (pm["robust_z"] > 0)).sum()),
                         "max_abs_robust_z": float(nd["robust_z"].abs().max()),
                         "median_profile_max_abs_robust_z":
                             float(nd.groupby("profile_id")["robust_z"].apply(lambda z: z.abs().max()).median())},
            "acf1_median_all": float(self.acf["acf1"].median()),
            "negative_spend_rows": int((pm["spend"] < 0).sum()),
        }


def prepare_data(data_dir: Path = DATA_DIR, holdout_frac: float = 0.2, seed: int = 0,
                 outlier_z: float = 6.0) -> PreparedData:
    """Build the person-month table and everything emission fitting needs (plan §2)."""
    pm = person_months(data_dir)
    holdout = holdout_split(pm, holdout_frac, seed)
    hold_set = {pid for ids in holdout.values() for pid in ids}
    pm["split"] = np.where(pm["profile_id"].isin(hold_set), "holdout", "train")
    pm["robust_z"] = robust_z(pm, "spend_ratio")
    pm["is_outlier"] = (pm["robust_z"].abs() > outlier_z) & ~pm["is_december"]
    pm["dec_bonus_income"] = december_bonus(pm)

    train = pm.loc[pm["split"] == "train"]
    bonus = {s: float(g["dec_bonus_income"].median())
             for s, g in train.loc[train["is_december"]].groupby("state")}
    dec_offset = december_offsets(train.loc[~train["is_outlier"]], "spend_ratio")
    income_over_rent = pm["reference_monthly_net_income_chf"] / pm["reference_monthly_rent_chf"]
    pm["spend_ratio_adj"] = pm["spend_ratio"] + (
        pm["is_december"] * pm["state"].map(bonus).fillna(0.0) * income_over_rent)

    acf = lag1_autocorr(pm.loc[~pm["is_outlier"]], "spend_ratio_adj")
    reg = spend_regression(pm.loc[(pm["split"] == "train") & ~pm["is_outlier"] & ~pm["is_december"]])
    settings = {"holdout_frac": holdout_frac, "seed": seed, "outlier_z": outlier_z,
                "emission": "spend_ratio = (reference_income - net) / rent",
                "body": "train split, non-December, |robust z| <= outlier_z",
                "december": "dec_bonus_income = (net_Dec - median non-Dec net) / reference_income"}
    return PreparedData(
        person_months=pm, holdout_ids=holdout, december_bonus=bonus, december_offset=dec_offset,
        regime_stats=group_summary(pm, "state", acf),
        employment_stats=group_summary(pm, "employment_status", acf),
        household_stats=group_summary(pm, ["state", "household_type"]),
        december_stats=december_bonus_stats(train, "employment_status"),
        spend_regression=reg, baseline_levels=baseline_levels(load_profiles(data_dir)),
        acf=acf, settings=settings)


ARTIFACT_FILES = {
    "person_months": "person_months.csv.gz",      # gitignored (large)
    "regime_stats": "regime_stats.csv",
    "employment_stats": "employment_stats.csv",
    "household_stats": "household_stats.csv",
    "december_stats": "december_bonus_stats.csv",
    "spend_regression": "spend_regression.csv",
    "baseline_levels": "baseline_levels.csv",
    "holdout_profiles": "holdout_profiles.json",
    "summary": "data_prep_summary.json",
}


def write_artifacts(prep: PreparedData, out_dir: Path = ARTIFACTS_DIR) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {k: out_dir / v for k, v in ARTIFACT_FILES.items()}
    prep.person_months.to_csv(paths["person_months"], index=False)
    for k in ["regime_stats", "employment_stats", "household_stats", "december_stats", "spend_regression"]:
        getattr(prep, k).round(4).to_csv(paths[k], index=False)
    prep.baseline_levels.round(2).to_csv(paths["baseline_levels"], index=False)
    paths["holdout_profiles"].write_text(json.dumps(
        {"settings": prep.settings, "holdout_ids": prep.holdout_ids}, indent=1) + "\n")
    paths["summary"].write_text(json.dumps(prep.summary(), indent=1) + "\n")
    return paths


def load_holdout_ids(path: Path = ARTIFACTS_DIR / ARTIFACT_FILES["holdout_profiles"]) -> dict[str, list[str]]:
    return json.loads(Path(path).read_text())["holdout_ids"]


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Data prep for the HMM regime engine (plan §2).")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=ARTIFACTS_DIR)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outlier-z", type=float, default=6.0)
    args = ap.parse_args(argv)

    prep = prepare_data(args.data_dir, args.holdout_frac, args.seed, args.outlier_z)
    paths = write_artifacts(prep, args.out_dir)
    cols = ["profiles", "holdout_profiles", "body_rows", "outlier_rows", "median_income", "median_rent",
            "spend_ratio_median", "spend_ratio_p05", "spend_ratio_p95", "spend_ratio_std",
            "spend_ratio_within_sd", "dec_bonus_income_median"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        for name, df, key in [("regime", prep.regime_stats, ["state"]),
                              ("employment status", prep.employment_stats, ["employment_status"])]:
            print(f"--- per {name} ---")
            print(df[key + cols + ["acf1_median"]].round(3).to_string(index=False))
        print("--- spend_ratio median by regime x household ---")
        print(prep.household_stats.pivot(index="state", columns="household_type",
                                         values="spend_ratio_median").round(2).to_string())
        print("--- spend = a*rent + b*income (train body) ---")
        print(prep.spend_regression.round(4).to_string(index=False))
        print("--- baseline income by regime (profiles) ---")
        b = prep.baseline_levels
        print(b[(b["level"] == "regime") & (b["age_band"] == "all")].to_string(index=False))
    print(json.dumps(prep.summary(), indent=1))
    for k, p in paths.items():
        print(f"wrote {k}: {p}")


def load_profile(profile_id: str = "P00001", data_dir: Path = DATA_DIR) -> Profile:
    """Build a ``Profile`` from the synthetic data (default: Alexia, P00001)."""
    pr = load_profiles(data_dir)
    row = pr.loc[pr["profile_id"] == profile_id]
    if row.empty:
        raise KeyError(f"Unknown profile_id {profile_id}")
    r = row.iloc[0]
    cb = load_cash_balances(data_dir)
    flows = cb.loc[cb["profile_id"] == profile_id, "net_transactions_chf"]
    ref_income = float(r["reference_monthly_net_income_chf"])
    mean_rate = float(flows.mean() / ref_income) if len(flows) and ref_income > 0 else None
    return Profile(
        age=int(r["age_at_start"]),
        employment=str(r["employment_status"]),
        ref_income=ref_income,
        rent=float(r["reference_monthly_rent_chf"]),
        attributes={"has_masters": False, "abroad": False, "canton": r["canton"],
                    "household_type": r["household_type"]},
        profile_id=profile_id,
        name=str(r["display_name"]),
        opening_cash=float(r["opening_cash_chf"]),
        mean_savings_rate=mean_rate,
    )


def observed_net_flows(profile_id: str, data_dir: Path = DATA_DIR) -> list[float]:
    cb = load_cash_balances(data_dir)
    return cb.loc[cb["profile_id"] == profile_id, "net_transactions_chf"].astype(float).tolist()


if __name__ == "__main__":
    main()
