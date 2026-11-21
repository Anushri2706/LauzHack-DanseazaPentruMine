"""Summary metrics and goal checks computed from a branch's monthly rows."""


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
        "total_expenses": sum(r["expenses"] for r in rows),
        "total_one_offs": sum(r["one_offs"] for r in rows),
        "total_interest": sum(r["interest"] for r in rows),
        "total_investment_return": sum(r["investment_return"] for r in rows),
        "months_studying": sum(1 for r in rows if r["status"] == "studying"),
        "months_abroad": sum(1 for r in rows if r["status"] == "abroad"),
    }


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
