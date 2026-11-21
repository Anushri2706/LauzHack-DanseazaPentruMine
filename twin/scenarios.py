"""Scenario templates. Each returns monthly plan arrays in today's CHF (before inflation/wage growth)."""
from .assumptions import A
from .profile import canton_rent_index


def working_income(p):
    """A student's baseline is 'start working now' (as in the scenario catalog); others keep their income.
    A student's assumed starting salary never undercuts income they already have."""
    if p["employment"] == "student":
        return max(A("working_income_student"), p["income"])
    return p["income"]


def _base(p):
    H = p["horizon"]
    return {"income": [working_income(p)] * H, "spend": [p["rent"] + p["other_spend"]] * H,
            "oneoff": [0.0] * H, "notes": []}


def baseline(p, **_):
    plan = _base(p)
    if p["employment"] == "student":
        plan["name"] = "Baseline: start working now"
        if working_income(p) > p["income"]:
            plan["notes"].append(f"Working income CHF {working_income(p):,.0f}/mo is an assumption (starter-kit value).")
        else:
            plan["notes"].append(f"Working income kept at your current CHF {p['income']:,.0f}/mo.")
    else:
        plan["name"] = "Baseline (nothing changes)"
    return plan


def masters(p, years=None, tuition=None, to_canton=None, **_):
    H = p["horizon"]
    mp = p["master_params"]
    years = years or mp.get("study_duration_years", A("master_default_years"))
    tuition = tuition or mp.get("annual_tuition_chf", A("master_default_tuition"))
    plan = _base(p)
    work = working_income(p)
    # A current student keeps their current income while studying; a worker drops to part-time.
    student = p["employment"] == "student"
    study_income = p["income"] if student else work * A("study_income_factor")
    n = int(years * 12)
    for m in range(H):
        if m < n:
            plan["income"][m] = study_income
            if m % 12 == 0:
                plan["oneoff"][m] += tuition
        else:
            plan["income"][m] = work * (1 + A("degree_premium"))
    where = ""
    if to_canton and to_canton != p.get("canton"):
        plan["oneoff"][0] += A("relocation_one_off")
        where = f", in {to_canton}"
        plan["notes"].append(f"Includes a one-off moving cost of CHF {A('relocation_one_off'):,.0f}; "
                             "the rent you gave is used for the whole period.")
    plan["name"] = f"Master's degree ({years:g} yrs, CHF {tuition:,.0f}/yr tuition{where})"
    if student:
        plan["notes"].append(f"Current income CHF {study_income:,.0f}/mo kept while studying; afterwards "
                             f"assumed working income +{A('degree_premium'):.0%}.")
    else:
        plan["notes"].append(f"Income {A('study_income_factor'):.0%} of working income while studying, "
                             f"+{A('degree_premium'):.0%} after.")
    return plan


def job_loss(p, months=6, **_):
    plan = _base(p)
    for m in range(min(months, p["horizon"])):
        plan["income"][m] = working_income(p) * A("job_loss_benefit")
    plan["name"] = f"Job loss for {months} months ({A('job_loss_benefit'):.0%} income support)"
    return plan


def relocate(p, to_canton=None, **_):
    idx = canton_rent_index()
    to_canton = to_canton or p["canton"]
    ratio = idx[to_canton] / idx[p["canton"]]
    plan = _base(p)
    new_rent = p["rent"] * ratio
    plan["spend"] = [new_rent + p["other_spend"]] * p["horizon"]
    plan["oneoff"][0] += A("relocation_one_off")
    plan["name"] = f"Move {p['canton']} -> {to_canton} (rent x{ratio:.2f})"
    plan["notes"].append("Rent ratio uses synthetic canton medians, not official data.")
    return plan


def part_time(p, **_):
    """Work at a reduced workload for the whole horizon."""
    plan = _base(p)
    share = A("part_time_share")
    plan["income"] = [working_income(p) * share] * p["horizon"]
    plan["name"] = f"Work part-time ({share:.0%} of working income)"
    plan["notes"].append(f"Income {share:.0%} of CHF {working_income(p):,.0f}/mo for the whole period; "
                         "spending unchanged.")
    return plan


def self_employed(p, **_):
    """Start a business: setup cost, a ramp-up year, then a premium, with larger monthly swings."""
    plan = _base(p)
    work, n = working_income(p), A("self_employed_rampup_months")
    for m in range(p["horizon"]):
        plan["income"][m] = work * (A("self_employed_rampup_share") if m < n else 1 + A("self_employed_premium"))
    plan["oneoff"][0] += A("self_employed_setup")
    plan["noise_scale"] = A("self_employed_noise")
    plan["name"] = "Become self-employed"
    plan["notes"].append(f"CHF {A('self_employed_setup'):,.0f} setup cost; {A('self_employed_rampup_share'):.0%} "
                         f"of income for {n} months, then +{A('self_employed_premium'):.0%}; monthly swings "
                         f"x{A('self_employed_noise'):g}. All illustrative.")
    return plan


def family(p, **_):
    """Have a child after a year: bigger home, child costs, reduced work for a while."""
    plan = _base(p)
    start, H = A("child_start_month"), p["horizon"]
    extra_rent = p["rent"] * (A("child_rent_factor") - 1)
    for m in range(start, H):
        plan["spend"][m] += extra_rent + A("child_cost_monthly")
        if m < start + A("child_reduced_months"):
            plan["income"][m] *= A("child_income_share")
    plan["name"] = "Have a child"
    plan["notes"].append(f"From month {start}: rent x{A('child_rent_factor'):g}, +CHF {A('child_cost_monthly'):,.0f}"
                         f"/mo child costs, {A('child_income_share'):.0%} of income for "
                         f"{A('child_reduced_months')} months. All illustrative.")
    return plan


TEMPLATES = {"baseline": baseline, "masters": masters, "job_loss": job_loss, "relocate": relocate,
             "part_time": part_time, "self_employed": self_employed, "family": family}
