"""Short narrative cards at decision points. Built only from engine numbers (no invention)."""
from .state import State

STATUS_TEXT = {
    "studying": "studying", "working": "working", "abroad": "living and working abroad",
    "unemployed": "between jobs", "retired": "retired",
}


def chf(x: float) -> str:
    return f"CHF {x:,.0f}".replace(",", "'")


def decision_narrative(name: str, city: str, s: State, reasons: list[str], p: dict) -> str:
    age = int(s.age)
    where = city if s.status != "abroad" else "abroad"
    money = f"You have {chf(s.cash)} in cash"
    if s.investments > 1:
        money += f" and {chf(s.investments)} invested"
    if s.debt > 1:
        money += f", and owe {chf(s.debt)}"
    money += "."
    parts = []
    if "start" in reasons:
        doing = STATUS_TEXT[s.status] + (f" in {where}" if s.status != "abroad" else "")
        parts.append(f"{name}, {age}. You're {doing}. {money} What do you do next?")
    elif "degree_finished" in reasons and s.status == "retired":
        parts.append(f"{name}, {age}. Your master's is done and you're back in retirement. {money}")
    elif "degree_finished" in reasons:
        parts.append(f"{name}, {age}. Your master's is done. A Swiss job would pay "
                     f"{chf(p['salary_master'])}/month, with costs around "
                     f"{chf(p['rent'] + p['living_working_master'])}/month (today's prices). {money}")
    else:
        parts.append(f"{name}, {age}. Another year has passed and you're {STATUS_TEXT[s.status]}. {money}")
        if s.program:
            left = (s.program_end or 0) - s.t
            parts.append(f"{left} months of your master's are left.")
    if s.cash < 0:
        parts.append(f"Your cash is below zero: this path needs {chf(-s.cash)} of financing so far.")
    return " ".join(parts)
