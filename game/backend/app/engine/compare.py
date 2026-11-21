"""Before/after comparison of two branches, with explanations generated from engine numbers."""
from .metrics import net_worth_bridge, summarize
from .narrative import chf

METRICS = [
    ("final_net_worth", "Net worth at the end"), ("final_cash", "Cash at the end"),
    ("final_investments", "Investments at the end"), ("final_debt", "Debt at the end"),
    ("min_cash", "Lowest cash"), ("months_below_buffer", "Months below buffer"),
    ("months_negative_cash", "Months with negative cash"),
    ("final_home_equity", "Home equity at the end"), ("total_joy", "Joy points (not money)"),
]
BRIDGE_KEYS = ("income", "expenses", "lifestyle", "purchases", "one_offs", "interest", "mortgage_interest",
               "maintenance", "home_fees", "investment_return", "home_appreciation")


def _age(start_age: float, t: int) -> float:
    return round(start_age + (t + 1) / 12, 1)


def compare(rows_a: list[dict], rows_b: list[dict], name_a: str, name_b: str, opening_nw: float,
            start_age: float, goals_a: list[dict], goals_b: list[dict]) -> dict:
    sa, sb = summarize(rows_a), summarize(rows_b)
    table = [{"key": k, "label": label, "a": sa[k], "b": sb[k], "diff": sb[k] - sa[k]} for k, label in METRICS]
    met_a = sum(g["status"] == "met" for g in goals_a)
    met_b = sum(g["status"] == "met" for g in goals_b)
    table.append({"key": "goals_met", "label": "Goals met", "a": met_a, "b": met_b, "diff": met_b - met_a})

    ba, bb = net_worth_bridge(rows_a, opening_nw), net_worth_bridge(rows_b, opening_nw)
    bridge = [{"key": k, "a": ba[k], "b": bb[k], "diff": bb[k] - ba[k]}
              for k in BRIDGE_KEYS]

    expl = []
    d = sb["final_net_worth"] - sa["final_net_worth"]
    end_age = _age(start_age, len(rows_a) - 1)
    if abs(d) < 500:
        expl.append(f"Both realities end at a similar net worth by {end_age:.0f} (within {chf(500)}).")
    else:
        hi, lo = (name_b, name_a) if d > 0 else (name_a, name_b)
        expl.append(f"Under these assumptions, “{hi}” ends {chf(abs(d))} higher in net worth than "
                    f"“{lo}” by age {end_age:.0f}. Higher net worth is only one dimension, not a verdict.")
    drivers = sorted(bridge, key=lambda x: -abs(x["diff"]))
    names = {"income": "income earned", "expenses": "living costs", "one_offs": "one-off costs (tuition, moves)",
             "interest": "interest paid", "investment_return": "investment growth",
             "lifestyle": "habits (eating out, shopping, gym)", "purchases": "purchases and trips",
             "mortgage_interest": "mortgage interest", "maintenance": "home maintenance",
             "home_fees": "home buying/selling fees", "home_appreciation": "home value growth"}
    top = [x for x in drivers if abs(x["diff"]) >= 500][:3]
    if top:
        bits = [f"{names[x['key']]} ({'+' if x['diff'] > 0 else '−'}{chf(abs(x['diff']))[4:]} for “{name_b}”)"
                for x in top]
        expl.append("The gap comes mostly from " + ", ".join(bits) + ".")
    # Crossover: last month the lead changes hands.
    diffs = [b["net_worth"] - a["net_worth"] for a, b in zip(rows_a, rows_b)]
    cross = None
    for t in range(1, len(diffs)):
        if (diffs[t - 1] < 0) != (diffs[t] < 0):
            cross = t
    if cross is not None:
        leader = name_b if diffs[-1] > 0 else name_a
        expl.append(f"“{leader}” takes the lead in net worth around age {_age(start_age, cross):.0f}.")
    if sa["min_cash"] != sb["min_cash"]:
        lower = name_a if sa["min_cash"] < sb["min_cash"] else name_b
        low_val = min(sa["min_cash"], sb["min_cash"])
        low_t = sa["min_cash_month"] if lower == name_a else sb["min_cash_month"]
        expl.append(f"“{lower}” has the thinner cushion: cash bottoms out at {chf(low_val)} "
                    f"around age {_age(start_age, low_t):.0f}.")
    for name, s in ((name_a, sa), (name_b, sb)):
        if s["months_negative_cash"]:
            expl.append(f"“{name}” needs financing for {s['months_negative_cash']} months "
                        f"(cash below zero).")
    if met_a != met_b:
        expl.append(f"Goals met: {met_a} in “{name_a}” vs {met_b} in “{name_b}”.")
    ja, jb = sa["total_joy"], sb["total_joy"]
    if abs(ja - jb) >= 1:
        more = name_a if ja > jb else name_b
        expl.append(f"“{more}” collects more joy points ({max(ja, jb):.0f} vs {min(ja, jb):.0f}): "
                    "a game preference score, not money and not a measure of a good life.")
    return {"table": table, "bridge": bridge, "explanation": expl,
            "net_worth_diff_by_month": diffs}
