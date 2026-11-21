"""The twin's door to the one engine: backend/regime's TwinSession, with its wealth layer on.

TwinSession runs the regime engine (life regimes -> net income and spending) and, with wealth=True,
the wealth layer (backend/regime/wealth.py): keep factor on surplus months, inflation on spending,
investments, loan interest, real net worth. Its methods are the tools the chat calls:
simulate, compare, explore.

This module only translates:
  twin profile dict -> regime Profile (+ goals, keep factor from the saving calibration)
  twin decision keys -> TwinSession decisions (apply_decision)
and personalizes spending: a synthetic profile's spending level in today's regime is shrunk toward its
own 24-month ledger; a new user's "other spending", when given, sets it.

Every scenario of one session shares the same seed, so differences between decisions are not noise.
"""
import sys
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from .assumptions import A
from .profile import canton_rent_index

BACKEND = Path(__file__).resolve().parent.parent / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from regime import RegimeEngine, TwinSession, fit_emissions, load_transition_priors  # noqa: E402
from regime.types import Profile  # noqa: E402

SEED = 7
TITLES = {"masters": "Master's degree", "job_loss": "Job loss", "relocate": "Move canton", "part_time": "Part-time",
          "self_employed": "Self-employed", "family": "Have a child"}


@lru_cache(maxsize=1)
def engine():
    """The fitted regime engine (emissions fitted on the training profiles, sourced transitions)."""
    return RegimeEngine(fit_emissions(), load_transition_priors())


@lru_cache(maxsize=1)
def _ledger_spend_ratios():
    """profile id -> its non-December spend/rent ratios from the 24-month ledger."""
    from regime.data import person_months
    pm = person_months()
    pm = pm.loc[~pm["is_december"]]
    return {pid: g["spend_ratio"].to_numpy(dtype=float) for pid, g in pm.groupby("profile_id")}


def _observed_spend_ratios(p):
    """The person's own spend/rent ratios: their ledger, or (new user) the non-rent spending they gave."""
    obs = _ledger_spend_ratios().get(p.get("id"))
    if obs is not None and len(obs):
        return obs
    if str(p.get("tags", {}).get("other_spend", "")).startswith("provided") and p["rent"] > 0:
        return [(p["rent"] + p["other_spend"]) / p["rent"]] * 24  # stated, so it dominates the prior
    return None


def engine_for(p, prof):
    """Engine whose spending level in today's regime is shrunk toward the person's own spending
    (EmissionModel.personalize; other regimes keep the population fit)."""
    base = engine()
    obs = _observed_spend_ratios(p)
    if obs is None:
        return base
    em = base.emissions.personalize(prof.initial_state, obs, prof.attributes.get("household_type"),
                                    self_employed=prof.employment.replace("-", "_") == "self_employed")
    return RegimeEngine(em, base.priors, base.income_table)


def backend_profile(p):
    """Twin profile dict -> regime Profile."""
    return Profile(age=int(p["age"]), employment=p["employment"], ref_income=float(p["income"]),
                   rent=float(p["rent"]),
                   attributes={"has_masters": False, "abroad": False, "canton": p.get("canton"),
                               "household_type": p.get("household")},
                   profile_id=p.get("id"), name=p.get("name"), opening_cash=float(p["cash"]),
                   portfolio_chf=float(p.get("portfolio", 0.0)),
                   loans=[(float(a), float(r)) for a, r in p.get("loans", [])])


def baseline_note(p):
    if p["employment"] == "student":
        return ("Baseline: you start working now, at the engine's entry salary for a first job "
                "(BFS, before tax), then follow the job-loss and job-finding rates.")
    return "Baseline: nothing changes on purpose; job loss, job finding and retirement follow the engine's rates."


def session_for(p, n_paths=300, years=None, seed=SEED, keep=None):
    """A TwinSession for profile p. keep: keep factor (default p["keep_factor"] set by calibration.apply,
    else 1.0 = no calibration). A student's baseline is 'start working now' (as in twin/scenarios.py);
    without it the engine keeps a student studying for the whole horizon."""
    prof = backend_profile(p)
    goals = [{"name": g["name"], "amount_chf": float(g["target"]), "month": int(g["month"])}
             for g in p.get("goals", [])]
    s = TwinSession(prof, engine=engine_for(p, prof), horizon_months=int(12 * (years or p.get("horizon", 120) // 12)),
                    n_paths=n_paths, seed=seed, goals=goals, wealth=True,
                    keep_factor=p.get("keep_factor", 1.0) if keep is None else keep)
    if p["employment"] == "student":
        s.apply_decision("baseline", "start_working", 0)
    s.scenarios["baseline"].name = "Start working now" if p["employment"] == "student" else "Baseline (nothing changes)"
    return s


def _rent_ratio(p, to_canton):
    idx = canton_rent_index()
    return idx[to_canton] / idx[p["canton"]]


def add_decision(s, sid, p, key, params=None, start_month=0):
    """Apply twin decision `key` to scenario `sid` of session s. Returns plain-words notes."""
    params = dict(params or {})
    t0 = int(params.pop("start_month", start_month) or 0)
    if key == "baseline":
        return []
    if key == "masters":
        mp = p.get("master_params", {})
        years = params.get("years") or mp.get("study_duration_years", A("master_default_years"))
        tuition = params.get("tuition") or mp.get("annual_tuition_chf", A("master_default_tuition"))
        s.apply_decision(sid, "masters", t0, duration=int(round(years * 12)), annual_tuition_chf=float(tuition))
        notes = [f"Master's of {years:g} years from month {t0}: you are a student (stipend income), pay CHF "
                 f"{tuition:,.0f} tuition a year, then look for work with the master's premium on the entry salary."]
        to = params.get("to_canton")
        if to and to != p.get("canton"):
            notes += add_decision(s, sid, p, "relocate", {"to_canton": to}, t0)
        return notes
    if key == "relocate":
        to = params.get("to_canton") or p["canton"]
        ratio = _rent_ratio(p, to)
        s.apply_decision(sid, "relocate", t0, rent_factor=ratio, label=to,
                         relocation_cost_chf=A("relocation_one_off"))
        return [f"Move {p['canton']} -> {to} in month {t0}: housing x{ratio:.2f} (synthetic canton medians, "
                f"not official data), CHF {A('relocation_one_off'):,.0f} moving cost; pay does not change."]
    if key == "part_time":
        s.apply_decision(sid, "part_time", t0)
        return [f"Part-time from month {t0} (60% of full-time pay); afterwards the engine's rates apply."]
    if key == "self_employed":
        s.apply_decision(sid, "self_employed", t0, setup_cost_chf=A("self_employed_setup"))
        return [f"Self-employed: CHF {A('self_employed_setup'):,.0f} setup cost, monthly spending swings about 3x "
                "wider (fitted on self-employed ledgers); income level unchanged."]
    if key == "family":
        t0 = t0 or A("child_start_month")
        s.apply_decision(sid, "family", t0, rent_factor=A("child_rent_factor"))
        return [f"Child in month {t0}: paid parental leave at 80% of the wage, then a home x{A('child_rent_factor'):g}."]
    if key == "job_loss":
        months = int(params.get("months", 6))
        s.apply_decision(sid, "job_loss", t0, duration=months)
        return [f"Job search for {months} months from month {t0}: unemployment insurance (70-80% of the last "
                "wage, AVIG), then back to the engine's rates."]
    raise ValueError(f"Unknown decision {key!r}")


def scenario(p, key, params=None, n_paths=300, years=None, seed=SEED, keep=None):
    """(session, scenario id, notes) with baseline and, unless key is baseline, a 'whatif' scenario."""
    s = session_for(p, n_paths, years, seed, keep)
    if key == "baseline":
        return s, "baseline", [baseline_note(p)]
    s.create_scenario("whatif", TITLES.get(key, key), copy_from="baseline")
    return s, "whatif", add_decision(s, "whatif", p, key, params)


def run(p, decision="baseline", params=None, years=10, n_paths=200, seed=SEED):
    """Wealth layer of one scenario (keep factor applied later: .wealth(keep, year)). Used by calibration."""
    s, sid, _ = scenario(p, decision, params, n_paths, years, seed, keep=1.0)
    return s._run(s.scenarios[sid], n_paths, seed).wealth


def explore_options(p, only=None):
    """Options for TwinSession.explore: every decision, the move to the cheapest-rent canton if cheaper."""
    opts = [{"kind": "baseline", "label": "baseline"}]
    mp = p.get("master_params", {})
    opts.append({"kind": "masters", "label": "masters",
                 "params": {"duration": int(round(mp.get("study_duration_years", A("master_default_years")) * 12)),
                            "annual_tuition_chf": float(mp.get("annual_tuition_chf", A("master_default_tuition")))}})
    opts += [{"kind": "part_time", "label": "part_time"},
             {"kind": "self_employed", "label": "self_employed",
              "params": {"setup_cost_chf": A("self_employed_setup")}},
             {"kind": "family", "label": "family", "params": {"rent_factor": A("child_rent_factor")},
              "min_start": A("child_start_month")}]  # a child cannot arrive at month 0
    idx = canton_rent_index()
    cheapest = min((c for c in idx if c != p["canton"]), key=idx.get)
    if idx[cheapest] < idx[p["canton"]]:
        opts.append({"kind": "relocate", "label": f"move to {cheapest}",
                     "params": {"rent_factor": _rent_ratio(p, cheapest), "label": cheapest,
                                "relocation_cost_chf": A("relocation_one_off")}})
    if only:
        opts = [o for o in opts if o["kind"] in ("baseline", only)]
    return opts
