"""Decision catalogue: what a player may choose, how it is validated and how it maps onto the engine.

Only decisions whose effects the migrated engine already models are listed. The mapping to
``Intervention`` / ``OneOff`` objects is the one in ``backend/regime/twin_session.py``
(``TwinSession._decision_to_interventions``), preserved exactly; ``tests/test_decisions.py`` checks that.
Raw engine interventions (``force_state``, ``scale_hazard``, ``set_attribute``) are NOT exposed.

Decisions are never reversible in place: a branch's decision history is append-only, and trying an
alternative means creating a new branch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from .accounting import OneOff
from .assumptions import AssumptionSet
from .types import STATES, Intervention

CANTON_CHOICES = ("ZH", "GE", "VD", "BE", "BS", "TI", "FR", "NE", "LU", "SG")


class DecisionError(ValueError):
    def __init__(self, errors: list[dict]):
        super().__init__("; ".join(e["message"] for e in errors))
        self.errors = errors


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: str                         # int | float | enum
    unit: str
    required: bool = False
    default_assumption: str | None = None   # registry id providing the default
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()
    description: str = ""

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass(frozen=True)
class DecisionDefinition:
    id: str
    name: str
    description: str
    params: tuple[ParamSpec, ...]
    eligible_states: tuple[str, ...]      # initial regimes for which the decision is allowed
    regime_effects: str
    financial_effects: str
    required_assumptions: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    reversible: bool = False
    requires_new_branch: bool = True
    min_start_month: int = 0

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "params"}
        d["params"] = [p.to_dict() for p in self.params]
        return d


ALL = tuple(STATES)
WORKING_AGE = ("employed", "part_time", "job_search", "student")

CATALOGUE: dict[str, DecisionDefinition] = {d.id: d for d in [
    DecisionDefinition(
        "masters", "Full-time master's degree",
        "Study full time, then look for work with the master's premium on the entry salary.",
        (ParamSpec("duration_months", "int", "months", default_assumption="masters_duration_months",
                   minimum=6, maximum=48, description="Length of study"),
         ParamSpec("annual_tuition_chf", "float", "CHF/year", default_assumption="masters_annual_tuition_chf",
                   minimum=0, maximum=100_000, description="Tuition paid at the start of each study year")),
        WORKING_AGE,
        "Pinned in 'student' for the duration, then 'employed' with has_masters=True.",
        "Student stipend instead of wages; tuition as one-off costs; master's premium on the next job.",
        ("student_income_chf", "entry_salary_chf", "masters_premium_pct", "masters_hazard_factor",
         "masters_annual_tuition_chf"),
        ("The master's premium is an average from survey data, not a promise for one person.",)),
    DecisionDefinition(
        "continue_employment", "Continue current employment",
        "Record that nothing changes on purpose. The baseline dynamics (job loss, job finding, raises) "
        "already model this; no intervention is added.",
        (), ("employed", "part_time"),
        "None (the HMM keeps running under the transition assumptions).",
        "None beyond the baseline.",
        ("job_loss_hazard", "job_search_exit_rate", "annual_experience_growth")),
    DecisionDefinition(
        "start_working", "Start working full time",
        "Enter employment at the start month, then evolve under the transition assumptions.",
        (), ("student", "job_search", "part_time"),
        "Jump to 'employed' at start_month (one month), then the HMM runs.",
        "Entry salary (or own reference wage) from that month.",
        ("entry_salary_chf", "job_loss_hazard")),
    DecisionDefinition(
        "part_time", "Switch to part-time",
        "Work part time at the model's part-time fraction of full-time pay.",
        (ParamSpec("duration_months", "int", "months", minimum=1, maximum=600,
                   description="Optional. Omitted = a one-month jump, after which the HMM may move back."),),
        ("employed", "part_time", "job_search"),
        "Pinned in 'part_time' for the duration (or a one-month jump if no duration).",
        "Pay = part_time_fraction x full-time wage.",
        ("part_time_fraction", "part_time_to_employed"),
        ("The working percentage is the registry's part_time_fraction; it is not a free parameter.",)),
    DecisionDefinition(
        "job_search", "Leave job / job-search period",
        "Spend a period in job search (unemployment insurance if eligible).",
        (ParamSpec("duration_months", "int", "months", minimum=1, maximum=60,
                   description="Optional. Omitted = exit via job_search_exit_rate."),),
        ("employed", "part_time"),
        "Pinned in 'job_search' for the duration, then back to the previous regime path.",
        "Unemployment insurance (AVIG replacement rate, capped) while eligible.",
        ("ui_replacement_rate", "ui_insured_earnings_cap_chf", "ui_max_months", "job_search_exit_rate"),
        ("This is a deliberate choice. The random job-loss risk is owned by the HMM and is not added twice.",)),
    DecisionDefinition(
        "relocate", "Move canton",
        "Move to another canton: housing changes by the ratio of synthetic canton median rents.",
        (ParamSpec("to_canton", "enum", "canton", required=True, choices=CANTON_CHOICES),
         ParamSpec("relocation_cost_chf", "float", "CHF", default_assumption="relocate_cost_chf",
                   minimum=0, maximum=100_000)),
        WORKING_AGE + ("retired",),
        "None.",
        "Housing bill x rent ratio from start_month; one-off relocation cost. Pay does not change.",
        ("relocate_cost_chf",),
        ("Canton rent ratios come from medians of the synthetic profiles, not official statistics.",
         "Income tax stays on the Lausanne (VD) estimate; the move does not change tax.")),
    DecisionDefinition(
        "move_abroad", "Relocate abroad (Germany factors)",
        "Live and work abroad with the model's income and cost multipliers.",
        (ParamSpec("relocation_cost_chf", "float", "CHF", default_assumption="move_abroad_relocation_cost_chf",
                   minimum=0, maximum=100_000),
         ParamSpec("job_search_months", "int", "months", minimum=0, maximum=24,
                   description="Months of job search after the move (default 0)")),
        WORKING_AGE,
        "Optional pinned job search after the move.",
        "abroad_income_multiplier on the next job, abroad_cost_multiplier on housing; one-off cost.",
        ("abroad_income_multiplier", "abroad_cost_multiplier", "move_abroad_relocation_cost_chf"),
        ("Values stay in CHF: there is no FX assumption, so no currency conversion is modelled.",
         "Swiss tax and social contributions are still applied abroad (model limitation).")),
    DecisionDefinition(
        "parental_leave", "Parental leave",
        "Paid leave at the replacement rate; the month after is employed.",
        (ParamSpec("duration_months", "int", "months", default_assumption="parental_leave_months",
                   minimum=1, maximum=24),),
        ("employed", "part_time"),
        "Pinned 'employed' during leave and the month after.",
        "Income = parental_leave_replacement x last wage during the leave.",
        ("parental_leave_replacement", "parental_leave_months"),
        ("Swiss EO rules are more complex than this single replacement rate.",)),
    DecisionDefinition(
        "family", "Have a child",
        "Parental leave plus a larger home from the start month.",
        (ParamSpec("rent_factor", "float", "fraction", default_assumption="family_rent_factor",
                   minimum=1.0, maximum=3.0),
         ParamSpec("leave_months", "int", "months", default_assumption="parental_leave_months",
                   minimum=0, maximum=24)),
        ("employed", "part_time"),
        "Parental leave pin (employed).",
        "Leave pay; housing x rent_factor from start_month.",
        ("family_rent_factor", "parental_leave_replacement", "parental_leave_months"),
        ("Child-care and other child costs are not modelled; only housing changes.",
         "Household type in the spending model does not change (known limitation).")),
    DecisionDefinition(
        "self_employed", "Become self-employed",
        "Self-employed spending pattern (wider swings) and a setup cost.",
        (ParamSpec("setup_cost_chf", "float", "CHF", default_assumption="self_employed_setup_cost_chf",
                   minimum=0, maximum=500_000),),
        ("employed", "part_time", "job_search"),
        "None.",
        "Setup cost at start_month; self-employed spending swings apply from month 0 (as in the original engine).",
        ("self_employed_setup_cost_chf",),
        ("No sourced self-employed income premium: income level is unchanged.",
         "The wider spending swings apply from month 0 even if start_month is later (original engine behaviour).")),
    DecisionDefinition(
        "one_off", "One-off expense",
        "A single planned cost (e.g. a car, a wedding).",
        (ParamSpec("amount_chf", "float", "CHF", required=True, minimum=0, maximum=10_000_000),),
        ALL, "None.", "Cost in that month, in today's CHF (inflated by the price level).", ()),
    DecisionDefinition(
        "downturn", "Economic downturn stress test",
        "Scale the employed -> job_search hazard for a period.",
        (ParamSpec("factor", "float", "multiplier", default_assumption="downturn_job_loss_factor",
                   minimum=0.0, maximum=10.0),
         ParamSpec("duration_months", "int", "months", minimum=1, maximum=600)),
        ("employed", "part_time"),
        "Multiplies the HMM job-loss hazard (it does not add a second job-loss process).",
        "Indirect, through more months in job search.",
        ("job_loss_hazard", "downturn_job_loss_factor")),
]}


@dataclass(frozen=True)
class Decision:
    """One validated decision instance. Params are stored with defaults resolved (explicit, traceable)."""
    decision_id: str
    start_month: int
    params: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    defaults_used: tuple[str, ...] = ()     # param names filled from registry defaults

    def to_dict(self) -> dict:
        return {"decision_id": self.decision_id, "start_month": self.start_month, "params": dict(self.params),
                "defaults_used": list(self.defaults_used)}


def _coerce(spec: ParamSpec, value, errors: list[dict]):
    field_name = f"params.{spec.name}"
    if spec.type == "enum":
        if value not in spec.choices:
            errors.append({"code": "invalid_choice", "field": field_name,
                           "message": f"{spec.name} must be one of {list(spec.choices)}"})
            return None
        return value
    if isinstance(value, bool):
        errors.append({"code": "invalid_type", "field": field_name, "message": f"{spec.name} must be a number"})
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        errors.append({"code": "invalid_type", "field": field_name, "message": f"{spec.name} must be a number"})
        return None
    if spec.type == "int":
        if v != int(v):
            errors.append({"code": "invalid_type", "field": field_name, "message": f"{spec.name} must be a whole number"})
            return None
        v = int(v)
    if spec.minimum is not None and v < spec.minimum or spec.maximum is not None and v > spec.maximum:
        errors.append({"code": "out_of_range", "field": field_name,
                       "message": f"{spec.name}={v} {spec.unit} outside [{spec.minimum}, {spec.maximum}]"})
        return None
    return v


def validate_decision(decision_id: str, start_month, params: Mapping | None, *, initial_state: str,
                      horizon_months: int, assumptions: AssumptionSet, current_canton: str | None = None,
                      min_start_month: int = 0) -> Decision:
    """Validate a proposed decision against the catalogue, the player's state and the horizon.

    Raises DecisionError with a list of structured errors. Defaults come from the registry (by id), never
    from the caller or the LLM.
    """
    errors: list[dict] = []
    if decision_id not in CATALOGUE:
        raise DecisionError([{"code": "unknown_decision", "field": "decision_id",
                              "message": f"Unsupported decision {decision_id!r}; supported: {sorted(CATALOGUE)}"}])
    d = CATALOGUE[decision_id]
    params = dict(params or {})
    try:
        t0 = int(start_month)
        if t0 != float(start_month):
            raise ValueError
    except (TypeError, ValueError):
        errors.append({"code": "invalid_type", "field": "start_month", "message": "start_month must be a whole number"})
        t0 = -1
    if errors == [] and not (max(min_start_month, d.min_start_month) <= t0 < horizon_months):
        errors.append({"code": "out_of_range", "field": "start_month",
                       "message": f"start_month must be in [{max(min_start_month, d.min_start_month)}, {horizon_months - 1}]"})
    if initial_state not in d.eligible_states:
        errors.append({"code": "not_eligible", "field": "decision_id",
                       "message": f"{d.name} is not available when starting as {initial_state!r}"})
    known = {p.name for p in d.params}
    for k in params:
        if k not in known:
            errors.append({"code": "unknown_param", "field": f"params.{k}",
                           "message": f"{decision_id} has no parameter {k!r}"})
    out: dict[str, Any] = {}
    defaults_used = []
    for spec in d.params:
        if spec.name in params and params[spec.name] is not None:
            v = _coerce(spec, params[spec.name], errors)
            if v is not None:
                out[spec.name] = v
        elif spec.required:
            errors.append({"code": "missing_param", "field": f"params.{spec.name}",
                           "message": f"{spec.name} ({spec.unit}) is required"})
        elif spec.default_assumption:
            v = assumptions.value(spec.default_assumption)
            out[spec.name] = int(v) if spec.type == "int" else float(v)
            defaults_used.append(spec.name)
    if decision_id == "relocate" and out.get("to_canton") and current_canton == out["to_canton"]:
        errors.append({"code": "no_change", "field": "params.to_canton", "message": "already living in that canton"})
    if decision_id == "relocate" and not current_canton:
        errors.append({"code": "missing_profile_fact", "field": "profile.canton",
                       "message": "moving canton needs the current canton in the profile"})
    if errors:
        raise DecisionError(errors)
    return Decision(decision_id, t0, MappingProxyType(out), tuple(defaults_used))


@dataclass
class EngineInputs:
    interventions: list[Intervention]
    one_offs: list[OneOff]
    profile_changes: dict
    notes: list[str]


def to_engine(decisions: list[Decision], horizon_months: int, *, canton_rent_ratio=None) -> EngineInputs:
    """Translate validated decisions into engine interventions, one-offs and profile changes.

    Mirrors TwinSession._decision_to_interventions. ``canton_rent_ratio(to_canton) -> float`` supplies
    the rent factor for ``relocate`` (synthetic canton medians).
    """
    ivs: list[Intervention] = []
    offs: list[OneOff] = []
    changes: dict = {}
    notes: list[str] = []
    for d in decisions:
        p, t0 = dict(d.params), d.start_month
        kind = d.decision_id
        if kind == "masters":
            dur, tuition = int(p["duration_months"]), float(p["annual_tuition_chf"])
            offs += [OneOff(t0 + 12 * y, tuition, "tuition") for y in range((dur + 11) // 12)
                     if t0 + 12 * y < horizon_months]
            ivs += [Intervention("force_state", t0, dur, state="student"),
                    Intervention("set_attribute", t0 + dur, attribute=("has_masters", True))]
        elif kind == "continue_employment":
            notes.append("continue_employment adds no intervention")
        elif kind == "start_working":
            ivs.append(Intervention("force_state", t0, None, state="employed"))
        elif kind == "part_time":
            ivs.append(Intervention("force_state", t0, p.get("duration_months"), state="part_time"))
        elif kind == "job_search":
            ivs.append(Intervention("force_state", t0, p.get("duration_months"), state="job_search"))
        elif kind == "relocate":
            if canton_rent_ratio is None:
                raise ValueError("relocate needs canton_rent_ratio")
            ratio = float(canton_rent_ratio(p["to_canton"]))
            ivs += [Intervention("set_attribute", t0, attribute=("rent_factor_move", ratio)),
                    Intervention("set_attribute", t0, attribute=("location", str(p["to_canton"])))]
            offs.append(OneOff(t0, float(p["relocation_cost_chf"]), "relocation"))
            notes.append(f"relocate: housing x{ratio:.3f} (synthetic canton medians)")
        elif kind == "move_abroad":
            ivs.append(Intervention("set_attribute", t0, attribute=("abroad", True)))
            js = int(p.get("job_search_months", 0) or 0)
            if js > 0:
                ivs.append(Intervention("force_state", t0, js, state="job_search"))
            offs.append(OneOff(t0, float(p["relocation_cost_chf"]), "relocation"))
        elif kind == "parental_leave":
            ivs.append(Intervention("parental_leave", t0, p.get("duration_months")))
        elif kind == "family":
            leave = p.get("leave_months")
            ivs += [Intervention("parental_leave", t0, None if leave is None else int(leave)),
                    Intervention("set_attribute", t0, attribute=("rent_factor_family", float(p["rent_factor"])))]
        elif kind == "self_employed":
            changes["self_employed"] = True
            offs.append(OneOff(t0, float(p["setup_cost_chf"]), "business setup"))
        elif kind == "one_off":
            offs.append(OneOff(t0, float(p["amount_chf"]), "one-off"))
        elif kind == "downturn":
            ivs.append(Intervention("scale_hazard", t0, p.get("duration_months"),
                                    transition=("employed", "job_search"), factor=float(p["factor"])))
        else:  # pragma: no cover - catalogue and mapping must stay in sync
            raise ValueError(kind)
    return EngineInputs(ivs, offs, changes, notes)


def catalogue_for(initial_state: str) -> list[dict]:
    out = []
    for d in CATALOGUE.values():
        row = d.to_dict()
        row["eligible"] = initial_state in d.eligible_states
        out.append(row)
    return out
