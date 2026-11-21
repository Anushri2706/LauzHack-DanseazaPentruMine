"""Inputs shared by the regression tests and the script that froze the legacy outputs."""

INTERVENTION_SETS = {
    "baseline": [],
    "masters": [("force_state", 0, 24, {"state": "student"}), ("set_attribute", 24, None, {"attribute": ("has_masters", True)})],
    "job_search": [("force_state", 6, 9, {"state": "job_search"})],
    "abroad": [("set_attribute", 3, None, {"attribute": ("abroad", True)}), ("force_state", 3, 4, {"state": "job_search"})],
    "parental": [("parental_leave", 12, None, {})],
    "downturn": [("scale_hazard", 0, 36, {"transition": ("employed", "job_search"), "factor": 2.0})],
    "part_time": [("force_state", 5, None, {"state": "part_time"})],
}

# Game decision (kind, params) -> the original TwinSession (kind, params) with the same explicit values.
EQUIV = [
    ("masters", {"duration_months": 18, "annual_tuition_chf": 5000}, "masters", {"duration": 18, "annual_tuition_chf": 5000.0}),
    ("start_working", {}, "start_working", {}),
    ("part_time", {"duration_months": 12}, "part_time", {"duration": 12}),
    ("part_time", {}, "part_time", {}),
    ("job_search", {"duration_months": 6}, "job_loss", {"duration": 6}),
    ("move_abroad", {"relocation_cost_chf": 7000, "job_search_months": 3}, "move_abroad", {"relocation_cost_chf": 7000.0, "job_search_months": 3}),
    ("parental_leave", {"duration_months": 4}, "parental_leave", {"duration": 4}),
    ("family", {"rent_factor": 1.3, "leave_months": 6}, "family", {"rent_factor": 1.3, "leave_months": 6}),
    ("self_employed", {"setup_cost_chf": 12000}, "self_employed", {"setup_cost_chf": 12000.0}),
    ("one_off", {"amount_chf": 999}, "one_off", {"amount_chf": 999.0}),
    ("downturn", {"factor": 3.0, "duration_months": 24}, "downturn", {"factor": 3.0, "duration": 24}),
]
EQUIV_LEGACY = [(k, p) for _, _, k, p in EQUIV]


def profile_kwargs(employment="employed", ref=6100.0):
    return dict(age=27, employment=employment, ref_income=ref, rent=1600.0,
                attributes={"has_masters": False, "abroad": False, "canton": "VD", "household_type": "single"},
                opening_cash=20000.0, portfolio_chf=5000.0, loans=[(10000.0, 0.04)])


def peer_query():
    return {"canton": "VD", "household": "single", "income": 6000.0, "rent": 1500.0}
