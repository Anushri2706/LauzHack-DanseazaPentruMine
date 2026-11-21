"""Check the learned layer against the data it was learned from.

1. Backtest: replay 24 months for every profile with its real income and spending, and compare the
   simulated share of months with negative cash against the ledger, by employment type.
   Done twice: with the learned noise (own swings minus the seasonal pattern) and the raw ledger swings.
2. New-user backtest: only cash, income, rent, age, employment, household given.
3. Imputation: how well do learned estimates recover a profile's own cash, portfolio and spending,
   compared with using one overall median.

python3 -m twin.validate_learned
"""
import csv
import statistics as st
from collections import defaultdict

from .learned import estimate, fit
from .profile import DATA, load_profile
from .sim import make_shocks, run_path

def backtest(paths=20, profile_ids=None):
    f = fit()
    with open(DATA / "synthetic-profiles.csv", newline="") as fh:
        ids = profile_ids or [r["profile_id"] for r in csv.DictReader(fh)]
    raw_sd = defaultdict(list)
    with open(DATA / "synthetic-cash-balances.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            raw_sd[r["profile_id"]].append(float(r["net_transactions_chf"]))
    learned, old, n = defaultdict(float), defaultdict(float), defaultdict(int)
    for pid in ids:
        p = load_profile(pid)
        p["horizon"], p["loans"] = 24, []  # the ledger does not contain loan payments
        plan = {"income": [p["income"]] * 24, "spend": [p["rent"] + p["other_spend"]] * 24,
                "oneoff": [0.0] * 24}
        shocks = make_shocks(p, paths, seed=11)
        neg_l = st.mean(run_path(p, plan, s)["short_months"] for s in shocks) / 24
        q = dict(p, net_sd=st.pstdev(raw_sd[pid]))
        shocks_old = make_shocks(q, paths, seed=11)
        neg_o = st.mean(run_path(q, plan, s)["short_months"] for s in shocks_old) / 24
        learned[p["employment"]] += neg_l
        old[p["employment"]] += neg_o
        n[p["employment"]] += 1
    print(f"\n1) Backtest over 24 months, {len(ids)} profiles, {paths} futures each: "
          "share of months with negative cash")
    print(f"   {'employment':<14}{'ledger':>9}{'learned':>10}{'raw sd':>8}")
    err_l = err_o = 0.0
    for emp in sorted(n):
        led = f["financing_rate"][emp]
        a, b = learned[emp] / n[emp], old[emp] / n[emp]
        err_l += abs(a - led) * n[emp]
        err_o += abs(b - led) * n[emp]
        print(f"   {emp:<14}{led:>9.1%}{a:>10.1%}{b:>8.1%}")
    tot = sum(n.values())
    print(f"   weighted abs error: learned {err_l / tot:.2%}, raw sd {err_o / tot:.2%}")


def imputation(sample=300):
    from .peers import population
    pop = population()[:sample]
    allp = population()
    glob = {"cash": st.median(q["cash"] for q in allp), "portfolio": st.median(q["portfolio"] for q in allp)}
    print(f"\n3) Imputation on {len(pop)} profiles (mean abs error, CHF). "
          "Note: group medians include the profile itself, so this is slightly optimistic.")
    for field in ("cash", "portfolio"):
        e_l = st.mean(abs(estimate(field, q["age"], q["employment"], q["household"])[0] - q[field]) for q in pop)
        e_g = st.mean(abs(glob[field] - q[field]) for q in pop)
        print(f"   {field:<12} learned {e_l:>9,.0f}   overall median {e_g:>9,.0f}   "
              f"improvement {1 - e_l / e_g:+.0%}")
    f = fit()
    eh, e = f["spend"]
    all_sp = [x for xs in e.values() for x in xs]
    g = st.median(all_sp)
    # other_spend per profile, in the same order as the fit
    sp = []
    for q in pop:
        p = load_profile(q["id"])
        sp.append((estimate("other_spend", q["age"], q["employment"], q["household"])[0], p["other_spend"]))
    e_l = st.mean(abs(a - b) for a, b in sp)
    e_g = st.mean(abs(g - b) for _, b in sp)
    print(f"   {'other_spend':<12} learned {e_l:>9,.0f}   overall median {e_g:>9,.0f}   "
          f"improvement {1 - e_l / e_g:+.0%}")


def backtest_new_user(paths=20):
    """Pretend every profile is a NEW user who gives only cash, income, rent, age, employment,
    household. Spending comes from the learned layer, noise is the fixed CHF default.
    Compare with a naive new-user setup (overall median spending, noise 10% of income)."""
    from statistics import median
    from .profile import _rows, profile_from_inputs
    f = fit()
    all_spend = median(x for xs in f["spend"][1].values() for x in xs)
    res = {"learned": defaultdict(float), "naive": defaultdict(float)}
    n = defaultdict(int)
    for r in _rows("synthetic-profiles.csv"):
        inputs = {"cash": r["opening_cash_chf"], "income": r["reference_monthly_net_income_chf"],
                  "rent": r["reference_monthly_rent_chf"], "age": r["age_at_start"],
                  "employment": r["employment_status"], "household": r["household_type"], "has_debt": False}
        q = profile_from_inputs(inputs)
        q["horizon"] = 24
        naive = dict(q, other_spend=all_spend, net_sd=0.10 * q["income"])
        for label, prof in (("learned", q), ("naive", naive)):
            plan = {"income": [prof["income"]] * 24, "spend": [prof["rent"] + prof["other_spend"]] * 24,
                    "oneoff": [0.0] * 24}
            shocks = make_shocks(prof, paths, seed=11)
            res[label][q["employment"]] += st.mean(run_path(prof, plan, s)["short_months"] for s in shocks) / 24
        n[q["employment"]] += 1
    print(f"\n2) New-user backtest: only cash, income, rent, age, employment, household given "
          f"({sum(n.values())} profiles)")
    print(f"   {'employment':<14}{'ledger':>9}{'learned':>10}{'naive':>8}")
    err = {"learned": 0.0, "naive": 0.0}
    for emp in sorted(n):
        led = f["financing_rate"][emp]
        a, b = res["learned"][emp] / n[emp], res["naive"][emp] / n[emp]
        err["learned"] += abs(a - led) * n[emp]
        err["naive"] += abs(b - led) * n[emp]
        print(f"   {emp:<14}{led:>9.1%}{a:>10.1%}{b:>8.1%}")
    tot = sum(n.values())
    print(f"   weighted abs error: learned {err['learned'] / tot:.2%}, naive {err['naive'] / tot:.2%}")


if __name__ == "__main__":
    backtest()
    backtest_new_user()
    imputation()
