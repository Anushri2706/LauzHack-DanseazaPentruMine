"""Peer trajectory: a younger person's path mapped from older people in the data.

Method (synthetic cohort with rank persistence):
  1. Find the person's percentile of net wealth among profiles of about the same age.
  2. For each later age, read the same percentile among profiles of that age.
  3. The band is the path if their position shifts by +/- BAND percentile points.
Assumption: people tend to keep their relative position among peers as they age.
Cross-sectional data: older profiles are different people, not this person later. Not a forecast.

Background (income, household...) is deliberately not used: in this synthetic data it predicts
wealth no better than age alone (see validate_peers.py), so age windows are the honest choice.
"""
from .peers import population, quantile

WINDOW, WIDE, MIN_COHORT = 3, 5, 40
BAND = 0.15


def cohort(age, exclude_id=None):
    for w in (WINDOW, WIDE):
        c = [q["wealth"] for q in population() if abs(q["age"] - age) <= w and q["id"] != exclude_id]
        if len(c) >= MIN_COHORT:
            return c, w
    return c, WIDE


def percentile_of(wealth, age, exclude_id=None):
    c, _ = cohort(age, exclude_id)
    below = sum(x < wealth for x in c) + 0.5 * sum(x == wealth for x in c)
    return min(0.98, max(0.02, below / len(c))), len(c)


def peer_trajectory(p, years=10):
    """Year-by-year wealth path for profile p. Returns dict with percentile and points."""
    wealth = p["cash"] + p["portfolio"] - sum(x for x, _ in p["loans"])
    pct, n_now = percentile_of(wealth, p["age"], p.get("id"))
    max_age = max(q["age"] for q in population())
    points = []
    for t in range(years + 1):
        age = p["age"] + t
        if age > max_age - WINDOW:
            break  # not enough older people in the data
        c, w = cohort(age, p.get("id"))
        points.append({"year": t, "age": age, "cohort": len(c), "window": w,
                       "low": quantile(c, max(0.0, pct - BAND)), "mid": quantile(c, pct),
                       "high": quantile(c, min(1.0, pct + BAND))})
    return {"wealth_now": wealth, "percentile": pct, "cohort_now": n_now, "points": points}


def peer_context(p):
    """Attach the peer path as CONTEXT only (no effect on the simulation). Returns p."""
    p["trajectory"] = peer_trajectory(p, years=p["horizon"] // 12)
    return p

