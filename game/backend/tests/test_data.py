"""Data loading: read-only CSVs → personas with labelled defaults."""
from app.data.loader import build_persona, load_dataset
from app.engine.assumptions import META, validate_params
from app.engine.simulate import initial_state, run_branch


def test_alexia_profile():
    a = build_persona("P00001")
    assert (a.name, a.age, a.city, a.status) == ("Alexia", 20, "Lausanne", "studying")
    assert a.opening_cash == 8000 and a.opening_investments == 0 and a.opening_debt == 0
    assert a.params["student_income"] == 1200 and a.params["rent"] == 900
    assert a.tags["opening_cash"] == "profile" and a.tags["tuition_annual"] == "scenario_csv"
    assert {g["name"] for g in a.goals} == {"Emergency savings", "Career transition"}


def test_snapshot_uses_assets_and_liabilities():
    ds = load_dataset()
    assert len(ds.snapshot) == 1200
    row = ds.snapshot.set_index("profile_id").loc["P00013"]
    assert row.debt > 0
    assert row.net_worth == row.cash + row.investments - row.debt


def test_every_persona_builds_and_runs():
    ds = load_dataset()
    for pid in ds.profiles.profile_id.sample(60, random_state=1).tolist() + ["P00001", "P00002", "P00013"]:
        p = build_persona(pid, ds)
        validate_params(p.params)
        assert set(p.params) == set(META)
        start = initial_state(p.params, p.status, p.start_salary, p.start_living)
        run = run_branch(p.params, [], start=start)
        assert len(run.rows) == 120


def test_persona_defaults_are_plausible():
    """Calibration sanity: costs scale with income and household, so typical workers neither
    get rich implausibly fast nor go broke; the engine must not be steered by bad defaults."""
    import statistics
    ds = load_dataset()
    gains = []
    for pid in ds.profiles[ds.profiles.employment_status == "employed"].profile_id.head(80):
        p = build_persona(pid, ds)
        start = initial_state(p.params, p.status, p.start_salary, p.start_living)
        rows = run_branch({**p.params, "return_annual": 0.0}, [], start=start).rows
        gains.append((rows[-1]["net_worth"] - p.card()["net_worth"]) / 120 / p.reference_income)
    saving_rate = statistics.median(gains)
    assert 0.1 < saving_rate < 0.4, saving_rate   # share of net income saved, before returns
    p = build_persona("P00762", ds)  # unemployed family member: restarts at a peer salary
    assert p.tags["salary_work"] == "peer" and p.params["salary_work"] > p.reference_income
