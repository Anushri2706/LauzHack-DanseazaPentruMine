"""Summary metrics and goal checks computed from a branch's monthly rows."""
import numpy as np

from .state import education


def summarize(rows: list[dict]) -> dict:
    last = rows[-1]
    negative = [r["t"] for r in rows if r["cash"] < 0]
    return {
        "final_cash": last["cash"],
        "final_investments": last["investments"],
        "final_debt": last["debt"],
        "final_net_worth": last["net_worth"],
        "min_cash": min(r["cash"] for r in rows),
        "min_cash_month": min(rows, key=lambda r: r["cash"])["t"],
        "months_below_buffer": sum(1 for r in rows if "below_buffer" in r["flags"]),
        "months_negative_cash": len(negative),
        "first_negative_month": negative[0] if negative else None,
        "total_income": sum(r["income"] for r in rows),
        "total_income_tax": sum(r.get("income_tax", 0.0) for r in rows),
        "total_expenses": sum(r["expenses"] for r in rows),
        "total_one_offs": sum(r["one_offs"] for r in rows),
        # Loan interest + interest on negative cash (the automatic credit line).
        "total_interest": sum(r["interest"] + r.get("overdraft_interest", 0.0) for r in rows),
        "total_overdraft_interest": sum(r.get("overdraft_interest", 0.0) for r in rows),
        "total_investment_return": sum(r["investment_return"] for r in rows),
        "total_lifestyle": sum(r.get("lifestyle", 0.0) for r in rows),
        "total_purchases": sum(r.get("purchases", 0.0) for r in rows),
        "total_mortgage_interest": sum(r.get("mortgage_interest", 0.0) for r in rows),
        "total_maintenance": sum(r.get("maintenance", 0.0) for r in rows),
        "total_home_fees": sum(r.get("home_fees", 0.0) for r in rows),
        "total_home_appreciation": sum(r.get("home_appreciation", 0.0) for r in rows),
        "final_home_equity": last.get("home_value", 0.0) - last.get("mortgage", 0.0),
        "total_joy": sum(r.get("joy", 0.0) for r in rows),
        "months_studying": sum(1 for r in rows if r["status"] == "studying"),
        "months_abroad": sum(1 for r in rows if r["status"] == "abroad"),
    }


# Risk bands for "cash below zero within the next 12 months" across simulated worlds. Our assumption.
RISK_HIGH, RISK_SOME = 0.20, 0.05


def cash_risk(cash, below_buffer, months: int = 12) -> dict:
    """Forward-looking risk from future paths only (worlds × months arrays, month 0 = next month).
    The past never counts: a dip you already recovered from is history, not risk."""
    cash, below_buffer = np.asarray(cash), np.asarray(below_buffer)
    if cash.size == 0:
        return {"months": months, "below_zero_12m": 0.0, "below_zero_ahead": 0.0, "below_buffer_12m": 0.0,
                "level": "low"}
    soon = (cash[:, :months] < 0).any(axis=1).mean()
    ahead = (cash < 0).any(axis=1).mean()
    buf = below_buffer[:, :months].any(axis=1).mean()
    level = "high" if soon >= RISK_HIGH else ("some" if soon >= RISK_SOME or ahead >= RISK_HIGH else "low")
    return {"months": months, "below_zero_12m": float(soon), "below_zero_ahead": float(ahead),
            "below_buffer_12m": float(buf), "level": level}


PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def check_goals(goals: list[dict], rows: list[dict]) -> list[dict]:
    """Each goal: {goal_id, name, target, month, priority}. `month` = row index of the target date.

    Goals are checked in priority order, then date. A goal is met if liquid assets
    (cash + investments) at its date, minus money already earmarked for goals met before it,
    reach the target. This avoids counting the same francs twice across goals.
    """
    horizon = len(rows)
    earmarked = 0.0
    out = []
    for g in sorted(goals, key=lambda g: (PRIORITY_ORDER.get(g["priority"], 9), g["month"])):
        if g["month"] >= horizon or g["month"] < 0:
            out.append({**g, "status": "beyond_horizon", "available": None})
            continue
        r = rows[g["month"]]
        available = r["cash"] + r["investments"] - earmarked
        met = available >= g["target"]
        if met:
            earmarked += g["target"]
        out.append({**g, "status": "met" if met else "not_met", "available": available})
    return out


def monthly_costs(row: dict) -> float:
    """Recurring monthly costs for the Freedom runway: rent + living + habits + mortgage interest + maintenance."""
    return (row["expenses"] + row.get("lifestyle", 0.0) + row.get("mortgage_interest", 0.0)
            + row.get("maintenance", 0.0))


def scores(rows: list[dict], goals_checked: list[dict], points: list[dict]) -> dict:
    """Multi-dimensional profile. Deliberately no single 'best' score."""
    m = summarize(rows)
    last = rows[-1]
    liquid = last["cash"] + last["investments"]
    costs = monthly_costs(last)
    runway = liquid / costs if costs > 0 else None
    # A dip below zero that the path recovers from is "tight", not a lasting shortfall.
    if last["cash"] < 0:
        safety = "shortfall"
    elif m["months_negative_cash"] > 0 or m["months_below_buffer"] > 12:
        safety = "tight"
    else:
        safety = "steady"
    counted = [g for g in goals_checked if g["status"] != "beyond_horizon"]
    edu = last.get("education", "bachelor")
    degree = edu == "master" or any("degree_finished" in p["reasons"] for p in points)
    return {
        "wealth": {"net_worth": last["net_worth"], "cash": last["cash"], "investments": last["investments"],
                   "debt": last["debt"]},
        "safety": {"level": safety, "months_below_buffer": m["months_below_buffer"],
                   "months_negative_cash": m["months_negative_cash"], "min_cash": m["min_cash"],
                   "recovered": m["months_negative_cash"] > 0 and last["cash"] >= 0},
        "goals": {"met": sum(g["status"] == "met" for g in counted), "total": len(counted),
                  "beyond_horizon": len(goals_checked) - len(counted)},
        "freedom": {"runway_months": runway, "debt_free": last["debt"] <= 0},
        "experience": {"degree": degree, "education": edu, "months_abroad": m["months_abroad"],
                       "months_studying": m["months_studying"]},
        "joy": {"total": m["total_joy"]},
    }


def scores_now(rows: list[dict], goals_checked: list[dict], state, current: int) -> dict:
    """HUD scores for the lived months only (rows = months 0..current-1). No single 'best' score."""
    nw = state.cash + state.investments + state.home_value - state.debt - state.mortgage
    neg = sum(1 for r in rows if r["cash"] < 0)
    below = sum(1 for r in rows if "below_buffer" in r["flags"])
    # Safety is where you stand now, not a permanent mark from an earlier dip (e.g. while studying).
    buffer_now = rows[-1]["buffer_target"] if rows else 0.0
    safety = "shortfall" if state.cash < 0 else ("tight" if state.cash < buffer_now else "steady")
    last_costs = monthly_costs(rows[-1]) if rows else None
    liquid = state.cash + state.investments
    due = [g for g in goals_checked if g["status"] != "beyond_horizon" and g["month"] < current]
    return {
        "money": {"net_worth": nw, "cash": state.cash, "investments": state.investments,
                  "home_equity": state.home_value - state.mortgage, "debt": state.debt + state.mortgage},
        "safety": {"level": safety, "months_below_buffer": below, "months_negative_cash": neg,
                   "recovered": neg > 0 and state.cash >= 0},
        "goals": {"met": sum(g["status"] == "met" for g in due), "due": len(due),
                  "total": sum(g["status"] != "beyond_horizon" for g in goals_checked)},
        "freedom": {"runway_months": (liquid / last_costs) if last_costs else None,
                    "debt_free": state.debt <= 0 and state.mortgage <= 0},
        "experience": {"degree": state.has_master, "studying": state.program is not None,
                       "education": education(state),
                       "months_abroad": sum(1 for r in rows if r["status"] == "abroad"),
                       "months_studying": sum(1 for r in rows if r["status"] == "studying")},
        "joy": {"total": sum(r["joy"] for r in rows)},
    }


def net_worth_bridge(rows: list[dict], opening_net_worth: float) -> dict:
    """Exact decomposition: NW_T = NW_0 + income − income tax − expenses − lifestyle − purchases − one-offs
    − interest − mortgage interest − maintenance − home fees + investment returns + home appreciation.

    Transfers (saving, selling investments, loans, principal repayment, amortization, the down payment
    and the mortgage itself) cancel out of net worth.
    """
    s = summarize(rows)
    return {
        "opening": opening_net_worth, "income": s["total_income"], "income_tax": -s["total_income_tax"],
        "expenses": -s["total_expenses"],
        "one_offs": -s["total_one_offs"], "interest": -s["total_interest"],
        "investment_return": s["total_investment_return"],
        "lifestyle": -s["total_lifestyle"], "purchases": -s["total_purchases"],
        "mortgage_interest": -s["total_mortgage_interest"], "maintenance": -s["total_maintenance"],
        "home_fees": -s["total_home_fees"], "home_appreciation": s["total_home_appreciation"],
        "closing": rows[-1]["net_worth"],
    }
