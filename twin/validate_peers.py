"""Does matching on background (not age) predict a profile's wealth better than age alone?

For each sampled profile, hide its wealth and predict it from (a) the median of its k most similar
profiles of about the same age, (b) the median of ALL profiles of about the same age.
python3 -m twin.validate_peers
"""
import random
import statistics as st

from .peers import distance, population


def run(n=300, k=15, age_window=3, seed=3):
    pop = population()
    rng = random.Random(seed)
    err_match, err_age, err_global = [], [], []
    gmed = st.median(q["wealth"] for q in pop)
    for p in rng.sample(pop, n):
        cohort = [q for q in pop if q["id"] != p["id"] and abs(q["age"] - p["age"]) <= age_window]
        if len(cohort) < k:
            continue
        matched = sorted(cohort, key=lambda q: distance(p, q))[:k]
        err_match.append(abs(p["wealth"] - st.median(q["wealth"] for q in matched)))
        err_age.append(abs(p["wealth"] - st.median(q["wealth"] for q in cohort)))
        err_global.append(abs(p["wealth"] - gmed))
    m = lambda xs: st.mean(xs)
    print(f"profiles tested: {len(err_match)}  (k={k}, same-age window +/-{age_window})")
    print(f"mean abs error, predict wealth with:")
    print(f"  overall median (ignore age and background): CHF {m(err_global):>9,.0f}")
    print(f"  same-age cohort median (age only):          CHF {m(err_age):>9,.0f}")
    print(f"  matched peers at same age (age + background): CHF {m(err_match):>7,.0f}")
    print(f"background adds: {1 - m(err_match) / m(err_age):+.1%} error change vs age-only; "
          f"age adds {1 - m(err_age) / m(err_global):+.1%} vs overall median")


if __name__ == "__main__":
    run()
