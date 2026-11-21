"""What the model simplifies, and which of its numbers are not yet validated.

    python3 -m twin.limits            # model limits + assumption register (counts per status)
    python3 -m twin.limits --all      # also list every placeholder and estimate by name

Used by the CLI and the chat ('limits' command, and a one-line footer under every result).
"""
import argparse
import csv
from pathlib import Path

ASSUMPTIONS_DIR = Path(__file__).resolve().parent.parent / "backend" / "assumptions"

MODEL_LIMITS = {
    "Saving calibration": [
        "The keep factor scales the engine's monthly surplus so that the median simulated wealth after 10 and "
        "20 years matches the median wealth of similar people 10 and 20 years older in the synthetic data.",
        "Older peers are different people, not you later: the calibration corrects group averages (per age band "
        "and income third), not your individual path.",
        "It only scales surplus months. Where the engine runs deficits (low-income people near or in retirement), "
        "it cannot close the gap; those groups are left uncalibrated and say so.",
        "Every result shows the uncalibrated value next to the calibrated one.",
    ],
    "Wealth engine": [
        "Debt is interest-only: the principal never changes and there are no repayments.",
        "Investments have normally distributed monthly returns (4% a year on average, 12% volatility); no fat "
        "tails, no asset mix, no contributions or withdrawals, no fees, no tax on returns.",
        "Inflation is one of three regimes (0.5%, 1.5%, 4% a year) switching once a year with hand-set "
        "probabilities.",
        "Investments and liabilities have no individual cash flows (dividends, amortisation, pension capital).",
        "Retirement pays 47.5% of the last wage (OECD mandatory replacement) while spending stays at its "
        "working-age level; pillar 2 and 3 capital is not modelled.",
        "Nothing here shows that these return or inflation assumptions predict real future outcomes.",
    ],
    "Data and scope": [
        "All profiles and ledgers are synthetic; results are not representative of the Swiss population.",
        "Income tax is an estimate for Vaud (federal + cantonal + Lausanne communal), used for every canton.",
        "Moving canton changes housing cost only (synthetic canton medians), not pay.",
        "Simulated percentages are shares of simulated futures under these assumptions, not real-world "
        "probabilities. Not financial advice, not a forecast.",
    ],
}

FOOTER = ("Simplified model: interest-only debt, normal returns, 3 inflation regimes, estimated Vaud tax, "
          "synthetic data. Type 'limits' for the full list.")

PLACEHOLDER = ("guess", "prior", "modelling assumption", "illustrative", "hand-set", "convention",
               "synthetic profiles")

# twin/assumptions.py values the regime engine still uses (decision costs and timing): all illustrative
TWIN_USED = ("master_default_years", "master_default_tuition", "relocation_one_off", "self_employed_setup",
             "child_rent_factor", "child_start_month", "working_income_student")


def status(source, notes=""):
    s, n = source.strip().lower(), notes.lower()
    if s == "modelling switch":
        return "switch"
    if any(s.startswith(w) for w in PLACEHOLDER):
        return "placeholder"
    if s.startswith("est") or "[est" in n or "verify" in n:
        return "estimate"
    return "sourced"


def register():
    """[{file, name, value, source, status}] for every assumption the regime engine and the twin use."""
    rows = []
    for f in sorted(ASSUMPTIONS_DIR.glob("*.csv")):
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                name = r.get("parameter") or r.get("assumption_name")
                value = r.get("value") or r.get("monthly_hazard")
                rows.append({"file": f"backend/assumptions/{f.name}", "name": name, "value": value,
                             "source": r["source"], "status": status(r["source"], r.get("notes", ""))})
    from .assumptions import ASSUMPTIONS
    for k in TWIN_USED:
        v, why = ASSUMPTIONS[k]
        rows.append({"file": "twin/assumptions.py", "name": k, "value": v, "source": why,
                     "status": "placeholder"})
    return rows


def summary():
    """{status: count}, in a fixed order."""
    counts = {"sourced": 0, "estimate": 0, "placeholder": 0, "switch": 0}
    for r in register():
        counts[r["status"]] += 1
    return counts


def print_limits():
    for head, items in MODEL_LIMITS.items():
        print(f"{head}:")
        for x in items:
            print(f"  - {x}")
    c = summary()
    print(f"\nAssumption register: {sum(c.values())} assumptions - {c['sourced']} sourced (law or official "
          f"statistics), {c['estimate']} estimates derived from sources, {c['placeholder']} placeholders "
          f"(priors, guesses, illustrative), {c['switch']} modelling switches. "
          f"`python3 -m twin.limits --all` lists them.")


def limits_text():
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_limits()
    return buf.getvalue().rstrip()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="list every placeholder and estimate")
    a = ap.parse_args()
    print_limits()
    if a.all:
        rows = register()
        for st_ in ("placeholder", "estimate"):
            print(f"\n{st_.upper()}S (need replacing or validating):")
            for r in rows:
                if r["status"] == st_:
                    print(f"  {r['name']:<34}{str(r['value']):>10}  {r['file']:<40} {r['source'][:60]}")
