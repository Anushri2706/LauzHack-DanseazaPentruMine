"""Bridge to the teammates' regime HMM (repo-root backend/twin, the feature/hmm-and-frontend version).

The HMM supplies the life-state paths (employed / part-time / job search / student / retired) and the income
in each state (wage, unemployment insurance, pension, master's premium, raises). The learned engine turns those
into monthly cash flows. For the honest backtest the HMM's own emissions are fitted on the training people only.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

_TWIN = Path(__file__).resolve().parents[3] / "backend"
if str(_TWIN) not in sys.path:
    sys.path.insert(0, str(_TWIN))

from twin import RegimeEngine, fit_emissions, load_transition_priors  # noqa: E402
from twin.cash_engine import run as run_cash  # noqa: E402
from twin.data import load_cash_balances, load_profiles  # noqa: E402
from twin.types import STATE_INDEX, STATES, Intervention, Profile  # noqa: E402

__all__ = ["engine", "profile_for", "run_cash", "Intervention", "Profile", "STATES", "STATE_INDEX"]


@lru_cache(maxsize=4)
def engine(train_ids: tuple[str, ...] | None = None) -> RegimeEngine:
    cb, pr = load_cash_balances(), load_profiles()
    if train_ids is not None:
        keep = set(train_ids)
        cb, pr = cb[cb.profile_id.isin(keep)], pr[pr.profile_id.isin(keep)]
    return RegimeEngine(fit_emissions(cb, pr), load_transition_priors())


def profile_for(row, *, age: float, opening_cash: float, history=None, start_month_of_year: int = 1,
                attributes: dict | None = None, employment: str | None = None, ref_income: float | None = None,
                rent: float | None = None) -> Profile:
    """An HMM Profile for one person. `history` (monthly flows) sets the HMM's own personal savings rate."""
    income = float(row.income if ref_income is None else ref_income)
    mean_rate = float(sum(history) / len(history) / income) if history is not None and len(history) and income > 0 else None
    return Profile(age=int(age), employment=str(employment or row.status), ref_income=income,
                   rent=float(row.rent if rent is None else rent),
                   attributes={"has_masters": False, "abroad": False, "canton": row.canton,
                               "household_type": row.household, **(attributes or {})},
                   profile_id=row.name if isinstance(row.name, str) else None, opening_cash=float(opening_cash),
                   mean_savings_rate=mean_rate, start_month_of_year=start_month_of_year)
