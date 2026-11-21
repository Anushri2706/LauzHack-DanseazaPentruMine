"""Canton rent index from the synthetic profiles (median reference rent per canton).

These are medians of SYNTHETIC challenge data, not official rent statistics. Used only for the ratio
applied by the ``relocate`` decision (same approach as twin/profile.canton_rent_index).
"""
from __future__ import annotations

from functools import lru_cache

from .synthetic import load_profiles


@lru_cache(maxsize=1)
def canton_rent_index() -> dict[str, float]:
    pr = load_profiles()
    return {str(k): float(v) for k, v in pr.groupby("canton")["reference_monthly_rent_chf"].median().items()}


def rent_ratio(from_canton: str, to_canton: str) -> float:
    idx = canton_rent_index()
    if from_canton not in idx or to_canton not in idx:
        raise KeyError(f"no synthetic rent data for {from_canton!r} or {to_canton!r}")
    return idx[to_canton] / idx[from_canton]
