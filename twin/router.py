"""Rule-based question router (swap for an LLM classifier later). Returns scenario, params, confidence."""
import re

CITY_TO_CANTON = {"zurich": "ZH", "zürich": "ZH", "geneva": "GE", "lausanne": "VD", "bern": "BE",
                  "basel": "BS", "lugano": "TI", "fribourg": "FR", "neuchatel": "NE", "neuchâtel": "NE",
                  "lucerne": "LU", "st. gallen": "SG", "st gallen": "SG"}
CANTONS = {"zh", "ge", "vd", "be", "bs", "ti", "fr", "ne", "lu", "sg"}

RULES = {
    "masters": ["master", "degree", "study", "university", "tuition", "phd"],
    "job_loss": ["lose my job", "laid off", "unemployed", "job loss", "fired", "redundan", "lose job"],
    "relocate": ["move", "relocat", "moving"],
    "part_time": ["part-time", "part time", "parttime", "reduce my hours", "work less", "fewer hours",
                  "reduced hours"],
    "self_employed": ["self-employed", "self employed", "freelanc", "own business", "start a business",
                      "startup", "start-up", "my own company"],
    "family": ["child", "baby", "kid", "start a family", "parent"],
    "explore": ["trade-off", "tradeoff", "trade off", "explore", "all options", "all my options",
                "best combination", "which decision", "compare everything"],
}


def route(question: str):
    q = question.lower()
    scores = {k: sum(w in q for w in ws) for k, ws in RULES.items()}
    best, hit = max(scores.items(), key=lambda kv: kv[1])
    total = sum(scores.values())
    if hit == 0:
        return None, {}, 0.0
    params = {}
    if best == "job_loss":
        m = re.search(r"(\d+)\s*month", q)
        y = re.search(r"(\d+)\s*year", q)
        params["months"] = int(m.group(1)) if m else (int(y.group(1)) * 12 if y else 6)
    if best == "relocate":
        for city, c in CITY_TO_CANTON.items():
            if city in q:
                params["to_canton"] = c
                break
        else:
            for tok in re.findall(r"\b[a-z]{2}\b", q):
                if tok in CANTONS:
                    params["to_canton"] = tok.upper()
                    break
    if best == "masters":
        y = re.search(r"(\d+)\s*(?:year|yr)", q)
        if y:
            params["years"] = int(y.group(1))
    if re.search(r"\bwhen (should|could|would|do|is)\b|\b(now or later|later or now)\b", q) and best in (
            "masters", "relocate", "part_time", "family"):
        return "explore", {"decision": best}, hit / total  # timing question: the search tries start months
    return best, params, hit / total
