"""TwinSession: stateful twin for one profile; its public methods are the LLM tools.

Pipeline per scenario: regime_engine.simulate -> cash_engine.run -> metrics.compute_metrics.
Tool methods return compact JSON-serialisable summaries (never raw arrays); the full arrays
stay in ``self.results`` for the UI. The LLM never computes numbers, it only calls these.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from . import cash_engine, metrics
from .cash_engine import OneOff
from .data import load_goals, load_profile
from .regime_engine import RegimeEngine
from .types import STATES, Intervention, Profile

CHECKPOINTS = (12, 24, 36, 60, 120)
SIM_START = date(2025, 1, 1)   # as-of date of the synthetic data

DECISIONS: dict[str, str] = {
    "masters": "Full-time master's: pinned in 'student' for duration months (default 24), then employed "
               "with has_masters=True. Params: duration, annual_tuition_chf (default 8000, Data S001).",
    "start_working": "Enter 'employed' at start_month, then evolve under the transition assumptions.",
    "move_abroad": "Set abroad=True (income/cost multipliers). Params: relocation_cost_chf (default 6500), "
                   "job_search_months (default 0).",
    "job_loss": "Stress test: enter 'job_search' at start_month. Param duration pins it for that many months; "
                "omitted = exit via job_search_exit_rate.",
    "part_time": "Enter 'part_time' at start_month. Optional duration.",
    "downturn": "Scale the job-loss hazard. Params: factor (default 2.0), duration (months, optional).",
    "one_off": "One-off cost. Params: amount_chf, label.",
    "force_state": "Raw intervention. Params: state, duration.",
    "scale_hazard": "Raw intervention. Params: from_state, to_state, factor, duration.",
    "set_attribute": "Raw intervention. Params: name, value.",
}


def _r(x, nd=0):
    if x is None:
        return None
    v = round(float(x), nd)
    return int(v) if nd == 0 else v


@dataclass
class Scenario:
    scenario_id: str
    name: str
    interventions: list[Intervention] = field(default_factory=list)
    one_offs: list[OneOff] = field(default_factory=list)
    assumptions: dict = field(default_factory=dict)
    decisions: list[dict] = field(default_factory=list)   # human-readable log of apply_decision calls


@dataclass
class ScenarioResult:
    paths: object
    cash: cash_engine.CashPaths
    metrics: dict
    summary: dict


class TwinSession:
    def __init__(self, profile: Profile, engine: RegimeEngine | None = None, horizon_months: int = 120,
                 n_paths: int = 2000, seed: int = 0, reserve_chf: float = 0.0, goals: list[dict] | None = None):
        self.profile = profile
        self.engine = engine or RegimeEngine.stub()
        self.horizon_months = horizon_months
        self.n_paths = n_paths
        self.seed = seed
        self.reserve_chf = reserve_chf
        self.goals = goals or []
        self.scenarios: dict[str, Scenario] = {"baseline": Scenario("baseline", "Baseline (no decision)")}
        self.results: dict[str, ScenarioResult] = {}

    @classmethod
    def from_profile_id(cls, profile_id: str = "P00001", **kw) -> "TwinSession":
        profile = load_profile(profile_id)
        g = load_goals()
        goals = []
        for r in g.loc[g["profile_id"] == profile_id].itertuples(index=False):
            d = date.fromisoformat(r.target_date)
            month = (d.year - SIM_START.year) * 12 + (d.month - SIM_START.month)
            goals.append({"name": r.goal_name, "amount_chf": float(r.target_amount_chf), "month": month,
                          "priority": r.priority})
        return cls(profile, goals=goals, **kw)

    # ----------------------------------------------------------------- tools

    def get_state(self) -> dict:
        """Profile, scenarios (decisions + overridden assumptions), settings, available levers."""
        p = self.profile
        return {
            "profile": {"profile_id": p.profile_id, "name": p.name, "age": p.age, "employment": p.employment,
                        "initial_regime": p.initial_state, "ref_income_chf": p.ref_income, "rent_chf": p.rent,
                        "opening_cash_chf": p.opening_cash, "attributes": p.attributes},
            "goals": self.goals,
            "settings": {"horizon_months": self.horizon_months, "n_paths": self.n_paths, "seed": self.seed,
                         "reserve_chf": self.reserve_chf, "engine_mode": self.engine.mode},
            "scenarios": {sid: {"name": s.name, "decisions": s.decisions, "assumptions": s.assumptions,
                                "simulated": sid in self.results} for sid, s in self.scenarios.items()},
            "decision_kinds": DECISIONS,
            "assumption_names": sorted(self.assumption_names()),
        }

    def create_scenario(self, scenario_id: str, name: str | None = None, copy_from: str | None = None) -> dict:
        if scenario_id in self.scenarios:
            raise ValueError(f"Scenario {scenario_id!r} already exists")
        s = Scenario(scenario_id, name or scenario_id)
        if copy_from:
            src = self._scenario(copy_from)
            s.interventions, s.one_offs = list(src.interventions), list(src.one_offs)
            s.assumptions, s.decisions = dict(src.assumptions), list(src.decisions)
        self.scenarios[scenario_id] = s
        return {"created": scenario_id, "name": s.name}

    def apply_decision(self, scenario_id: str, kind: str, start_month: int = 0, **params) -> dict:
        s = self._scenario(scenario_id)
        if kind not in DECISIONS:
            raise ValueError(f"Unknown decision kind {kind!r}; choose from {sorted(DECISIONS)}")
        ivs, offs = self._decision_to_interventions(kind, int(start_month), params)
        s.interventions.extend(ivs)
        s.one_offs.extend(offs)
        s.decisions.append({"kind": kind, "start_month": int(start_month), **params})
        self.results.pop(scenario_id, None)
        return {"scenario_id": scenario_id, "applied": kind, "start_month": int(start_month),
                "interventions": len(ivs), "one_off_costs_chf": [o.amount_chf for o in offs]}

    def set_assumption(self, scenario_id: str, name: str, value: float) -> dict:
        s = self._scenario(scenario_id)
        if name not in self.assumption_names():
            raise ValueError(f"Unknown assumption {name!r}")
        old = self._base_value(name, s)
        s.assumptions[name] = float(value)
        self.results.pop(scenario_id, None)
        return {"scenario_id": scenario_id, "name": name, "old": old, "new": float(value)}

    def simulate(self, scenario_id: str) -> dict:
        s = self._scenario(scenario_id)
        res = self._run(s, self.n_paths, self.seed)
        self.results[scenario_id] = res
        return res.summary

    def compare(self, a: str, b: str) -> dict:
        """b minus a, on matching summary fields."""
        ra = self.results.get(a) or self._store(a)
        rb = self.results.get(b) or self._store(b)
        sa, sb = ra.summary, rb.summary
        delta = {
            "closing_cash_p50_chf": sb["closing_cash_chf"]["p50"] - sa["closing_cash_chf"]["p50"],
            "closing_cash_p05_chf": sb["closing_cash_chf"]["p05"] - sa["closing_cash_chf"]["p05"],
            "minimum_cash_p05_chf": sb["minimum_cash_chf"]["p05"] - sa["minimum_cash_chf"]["p05"],
            "p_ever_below_reserve": round(sb["p_ever_below_reserve"] - sa["p_ever_below_reserve"], 3),
            "months_in_state": {k: round(sb["expected_months_in_state"][k] - sa["expected_months_in_state"][k], 1)
                                for k in STATES},
        }
        return {"a": a, "b": b, "delta_b_minus_a": delta, "a_summary": sa, "b_summary": sb,
                "note": "Modelled difference under the stated assumptions; not a proven causal effect."}

    def sensitivity(self, scenario_id: str, names: list[str] | None = None, rel_change: float = 0.25,
                    n_paths: int = 500) -> dict:
        """One-at-a-time ±rel_change perturbation; ranks assumptions by swing in median closing cash."""
        s = self._scenario(scenario_id)
        names = names or self._default_sensitivity_names(s)
        base = self._run(s, n_paths, self.seed).summary
        rows = []
        for name in names:
            v0 = self._base_value(name, s)
            if v0 is None:
                continue
            out = {}
            for label, v in (("low", v0 * (1 - rel_change)), ("high", v0 * (1 + rel_change))):
                tmp = Scenario(s.scenario_id, s.name, s.interventions, s.one_offs, {**s.assumptions, name: v})
                summ = self._run(tmp, n_paths, self.seed).summary
                out[label] = {"value": v, "closing_cash_p50_chf": summ["closing_cash_chf"]["p50"],
                              "p_ever_below_reserve": summ["p_ever_below_reserve"]}
            swing = abs(out["high"]["closing_cash_p50_chf"] - out["low"]["closing_cash_p50_chf"])
            rows.append({"name": name, "base_value": v0, **out, "swing_closing_p50_chf": _r(swing)})
        rows.sort(key=lambda r: r["swing_closing_p50_chf"], reverse=True)
        return {"scenario_id": scenario_id, "rel_change": rel_change,
                "base_closing_cash_p50_chf": base["closing_cash_chf"]["p50"], "ranking": rows,
                "note": metrics.DISCLAIMER}

    def sample_paths(self, scenario_id: str, n: int = 3) -> dict:
        """n representative paths (spread over closing-cash quantiles), as regime segments."""
        res = self.results.get(scenario_id) or self._store(scenario_id)
        closing = res.cash.closing
        order = np.argsort(closing)
        picks = order[np.linspace(0, len(order) - 1, num=max(1, n)).round().astype(int)]
        out = []
        for idx in picks:
            seq = res.paths.states[idx]
            segments, start = [], 0
            for t in range(1, len(seq) + 1):
                if t == len(seq) or seq[t] != seq[start]:
                    segments.append({"regime": STATES[seq[start]], "start_month": start, "end_month": t})
                    start = t
            cash = res.cash.cash[idx]
            out.append({"path_index": int(idx), "regimes": segments, "closing_cash_chf": _r(cash[-1]),
                        "minimum_cash_chf": _r(min(cash.min(), res.cash.opening_cash_chf)),
                        "minimum_cash_month": int(cash.argmin()),
                        "first_negative_month": int(res.cash.first_negative_month[idx])})
        return {"scenario_id": scenario_id, "paths": out, "note": metrics.DISCLAIMER}

    # ------------------------------------------------------------- internals

    def assumption_names(self) -> set[str]:
        pri = self.engine.priors
        names = set(pri.names) | set(getattr(pri, "age_names", {})) | set(self.engine.income_table.params)
        names |= {f"savings_rate_{s}" for s in STATES}
        return names

    def _scenario(self, sid: str) -> Scenario:
        if sid not in self.scenarios:
            raise KeyError(f"Unknown scenario {sid!r}; create it first")
        return self.scenarios[sid]

    def _store(self, sid: str) -> ScenarioResult:
        self.simulate(sid)
        return self.results[sid]

    def _base_value(self, name: str, s: Scenario) -> float | None:
        if name in s.assumptions:
            return s.assumptions[name]
        pri = self.engine.priors
        if name in getattr(pri, "age_names", {}):
            return float(pri.age_hazard(name))
        if name in pri.names:
            return float(pri.hazards[pri.names[name]])
        if name in self.engine.income_table.params:
            return float(self.engine.income_table.params[name])
        if name.startswith("savings_rate_"):
            return self.engine.savings_rate_for(name.removeprefix("savings_rate_"), self.profile, {})
        return None

    def _default_sensitivity_names(self, s: Scenario) -> list[str]:
        visited = set()
        if s.scenario_id in self.results:
            st = self.results[s.scenario_id].paths.states
            visited = {STATES[k] for k in np.unique(st)}
        else:
            visited = {self.profile.initial_state} | {iv.state for iv in s.interventions if iv.state}
        return (["job_loss_hazard", "job_search_exit_rate", "entry_salary_chf", "annual_experience_growth",
                 "ui_replacement_rate", "masters_premium_pct"]
                + [f"savings_rate_{v}" for v in sorted(visited)])

    def _run(self, s: Scenario, n_paths: int, seed: int) -> ScenarioResult:
        paths = self.engine.simulate(self.profile, horizon_months=self.horizon_months, n_paths=n_paths,
                                     interventions=s.interventions, assumption_overrides=s.assumptions, seed=seed)
        cash = cash_engine.run(self.profile, paths, s.one_offs)
        m = metrics.compute_metrics(cash, paths, reserve_chf=self.reserve_chf, goals=self.goals)
        return ScenarioResult(paths, cash, m, self._summarise(s, m))

    def _summarise(self, s: Scenario, m: dict) -> dict:
        T = self.horizon_months
        cps = [c for c in CHECKPOINTS if c <= T]
        return {
            "scenario_id": s.scenario_id, "name": s.name, "decisions": s.decisions, "assumptions": s.assumptions,
            "closing_cash_chf": {k: _r(v) for k, v in m["closing"].items()},
            "minimum_cash_chf": {"p05": _r(m["minimum_cash_p05"]), "p50": _r(m["minimum_cash_p50"])},
            "cash_p50_at_month": {c: _r(m["fan"]["p50"][c - 1]) for c in cps},
            "reserve_chf": self.reserve_chf,
            "p_ever_below_reserve": round(m["p_ever_below_reserve"], 3),
            "share_paths_ever_negative": round(m["share_paths_ever_negative"], 3),
            "median_first_negative_month": m["median_first_negative_month"],
            "expected_months_in_state": {k: round(v, 1) for k, v in m["expected_months_in_state"].items()},
            "regime_probability_at_month": {c: {STATES[k]: round(float(m["occupancy"][c - 1, k]), 3)
                                                for k in range(len(STATES)) if m["occupancy"][c - 1, k] > 0}
                                            for c in cps},
            "goals": [{"name": g["name"], "amount_chf": g["amount_chf"], "month": g["month"],
                       "p_met": round(g["p_met"], 3)} for g in m["goals"]],
            "engine_mode": self.engine.mode, "n_paths": m["n_paths"], "seed": self.seed,
            "disclaimer": m["disclaimer"],
        }

    def _decision_to_interventions(self, kind: str, t0: int, p: dict) -> tuple[list[Intervention], list[OneOff]]:
        dur = p.get("duration")
        dur = None if dur is None else int(dur)
        if kind == "masters":
            d = int(p.get("duration", 24))
            tuition = float(p.get("annual_tuition_chf", 8000.0))
            offs = [OneOff(t0 + 12 * y, tuition, "tuition") for y in range((d + 11) // 12)
                    if t0 + 12 * y < self.horizon_months]
            return ([Intervention("force_state", t0, d, state="student"),
                     Intervention("set_attribute", t0 + d, attribute=("has_masters", True))], offs)
        if kind == "start_working":
            return [Intervention("force_state", t0, None, state="employed")], []
        if kind == "move_abroad":
            ivs = [Intervention("set_attribute", t0, attribute=("abroad", True))]
            js = int(p.get("job_search_months", 0))
            if js > 0:
                ivs.append(Intervention("force_state", t0, js, state="job_search"))
            return ivs, [OneOff(t0, float(p.get("relocation_cost_chf", 6500.0)), "relocation")]
        if kind == "job_loss":
            return [Intervention("force_state", t0, dur, state="job_search")], []
        if kind == "part_time":
            return [Intervention("force_state", t0, dur, state="part_time")], []
        if kind == "downturn":
            return [Intervention("scale_hazard", t0, dur, transition=("employed", "job_search"),
                                 factor=float(p.get("factor", 2.0)))], []
        if kind == "one_off":
            return [], [OneOff(t0, float(p["amount_chf"]), str(p.get("label", "one-off")))]
        if kind == "force_state":
            return [Intervention("force_state", t0, dur, state=p["state"])], []
        if kind == "scale_hazard":
            return [Intervention("scale_hazard", t0, dur, transition=(p["from_state"], p["to_state"]),
                                 factor=float(p["factor"]))], []
        if kind == "set_attribute":
            return [Intervention("set_attribute", t0, attribute=(p["name"], p["value"]))], []
        raise ValueError(kind)
