"""Central assumption registry with provenance.

Every number the engine uses comes from a CSV in ``app/data/assumptions/`` (registry version ``1`` =
verbatim copies of ``backend/assumptions/*.csv`` plus ``game_assumptions.csv``). This module only reads
them and classifies their provenance. It never invents sources:

* ``source_name`` is the text in the CSV ``source`` column, unchanged.
* ``source_url`` is always ``None``: the original CSVs carry no URLs and none were added.
* ``source_status`` is derived from the CSV text by fixed rules (``_status``):
    - ``estimated``    the CSV marks it [EST] / EST, or it comes from an internal estimation note
    - ``illustrative`` hand-set, prior, guess, modelling switch/assumption, convention, synthetic data
    - ``placeholder``  the CSV says Placeholder
    - ``cited``        a named publication or law (BFS, AVIG, BVG, OECD, ...) is cited in the CSV but the
                       value was NOT re-verified during this migration. We do not call it ``official``.
* ``last_verified`` is ``None`` for every row: nothing was re-verified against a primary source.

An ``AssumptionSet`` is the registry plus validated overrides. Its ``version`` changes whenever the
overrides change, so results record exactly which values they used.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ..data.synthetic import ASSUMPTIONS_DIR

REGISTRY_VERSION = "1"

PARAM_FILES = {
    "income": "income_assumptions.csv",
    "expense": "expense_assumptions.csv",
    "tax": "tax_assumptions.csv",
    "market": "market_assumptions.csv",
    "game": "game_assumptions.csv",
}

# Kinds. Only CONTINUOUS_KINDS are eligible for +/-10% sensitivity.
CONTINUOUS_KINDS = {"amount", "rate", "probability", "multiplier", "continuous"}


class AssumptionError(ValueError):
    """Unknown assumption, or a value outside its valid range."""

    def __init__(self, code: str, assumption_id: str, message: str):
        super().__init__(message)
        self.code = code
        self.assumption_id = assumption_id

    def to_dict(self) -> dict:
        return {"code": self.code, "field": self.assumption_id, "message": str(self)}


@dataclass(frozen=True)
class Assumption:
    id: str
    name: str
    value: float
    unit: str
    kind: str                       # amount | rate | probability | multiplier | continuous | flag | duration | age
    lower: float | None
    upper: float | None
    geography: str | None
    effective_date: str | None
    reference_year: int | None      # year written in the CSV citation, if any (not a verified effective date)
    source_name: str | None
    source_url: str | None
    source_status: str              # cited | estimated | illustrative | placeholder
    methodology: str
    last_verified: str | None
    version: str
    table: str                      # income | expense | tax | market | game | transitions
    applies_to: str

    @property
    def sensitivity_eligible(self) -> bool:
        return self.kind in CONTINUOUS_KINDS and self.value != 0.0

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["sensitivity_eligible"] = self.sensitivity_eligible
        return d


def _status(source: str, notes: str) -> str:
    s, n = source.strip(), notes.strip()
    low = s.lower()
    if low.startswith("placeholder"):
        return "placeholder"
    if s.startswith("EST") or n.startswith("[EST") or low.startswith("internal/"):
        return "estimated"
    if any(w in low for w in ("illustrative", "hand-set", "guess", "prior", "modelling", "modeling",
                              "convention", "synthetic")):
        return "illustrative"
    return "cited"


def _year(source: str) -> int | None:
    m = re.findall(r"\b(20\d\d)\b", source)
    return int(m[-1]) if m else None


def _kind(pid: str, unit: str, value: float, table: str) -> tuple[str, float | None, float | None]:
    u = unit.lower()
    if table == "transitions" or "probability" in u:
        return "probability", 0.0, 1.0
    if u == "flag":
        return "flag", 0.0, 1.0
    if u in ("months", "years") or u.startswith("months"):
        return ("age" if pid.endswith("_age") else "duration"), 0.0, None
    if "chf" in u and "per chf" not in u:
        return "amount", 0.0, None
    if "fraction" in u or "per chf" in u:
        if "multiplier" in pid or "factor" in pid or value > 1.0:
            return "multiplier", 0.0, None
        return "rate", 0.0, 1.0
    return "continuous", 0.0, None


def _geography(pid: str, table: str) -> str | None:
    if pid.startswith("tax_rate_at_"):
        return "CH-VD"        # effective rate curve for a single person in Lausanne (CSV notes)
    if table in ("market", "game"):
        return None           # illustrative / game defaults: no geography claimed
    if pid.startswith("abroad_"):
        return "DE"           # Germany factor (CSV notes)
    if pid.startswith("zurich_"):
        return "CH-ZH"
    return "CH"


def _row(pid, value, unit, applies_to, source, notes, table) -> Assumption:
    kind, lo, hi = _kind(pid, unit, value, table)
    return Assumption(
        id=pid, name=pid.replace("_", " "), value=float(value), unit=unit, kind=kind, lower=lo, upper=hi,
        geography=_geography(pid, table), effective_date=None, reference_year=_year(source),
        source_name=source or None, source_url=None, source_status=_status(source, notes),
        methodology=notes, last_verified=None, version=REGISTRY_VERSION, table=table, applies_to=applies_to)


def load_registry(directory: Path = ASSUMPTIONS_DIR) -> "Registry":
    rows: dict[str, Assumption] = {}
    for table, name in PARAM_FILES.items():
        with open(directory / name, newline="") as f:
            for r in csv.DictReader(f):
                a = _row(r["parameter"], float(r["value"]), r["unit"], r["applies_to"], r["source"],
                         r["notes"], table)
                if a.id in rows:
                    raise ValueError(f"duplicate assumption id {a.id}")
                rows[a.id] = a
    with open(directory / "transitions.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["from_state"] == r["to_state"]:
                continue
            ages = ""
            if r["age_min"] or r["age_max"]:
                ages = f" ages {r['age_min'] or '-'}..{r['age_max'] or '-'}"
            a = _row(r["assumption_name"], float(r["monthly_hazard"]), "probability/month",
                     f"{r['from_state']} -> {r['to_state']}{ages}", r["source"], r["notes"], "transitions")
            rows[a.id] = a
    return Registry(MappingProxyType(rows))


@dataclass(frozen=True)
class Registry:
    rows: Mapping[str, Assumption]

    def __contains__(self, aid: str) -> bool:
        return aid in self.rows

    def get(self, aid: str) -> Assumption:
        if aid not in self.rows:
            raise AssumptionError("unknown_assumption", aid, f"Unknown assumption {aid!r}")
        return self.rows[aid]

    def value(self, aid: str) -> float:
        return self.get(aid).value

    def validate_value(self, aid: str, value) -> float:
        a = self.get(aid)
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise AssumptionError("invalid_type", aid, f"{aid} must be a number") from None
        if not math.isfinite(v):
            raise AssumptionError("invalid_value", aid, f"{aid} must be finite")
        if a.lower is not None and v < a.lower:
            raise AssumptionError("out_of_range", aid, f"{aid}={v} is below {a.lower} ({a.unit})")
        if a.upper is not None and v > a.upper:
            raise AssumptionError("out_of_range", aid, f"{aid}={v} is above {a.upper} ({a.unit})")
        if a.kind == "flag" and v not in (0.0, 1.0):
            raise AssumptionError("invalid_value", aid, f"{aid} is a 0/1 switch")
        return v


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:10]


@dataclass(frozen=True)
class AssumptionSet:
    """Registry + validated overrides. Immutable; ``with_overrides`` returns a new set."""
    registry: Registry
    overrides: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))

    @classmethod
    def default(cls, registry: Registry | None = None) -> "AssumptionSet":
        return cls(registry or load_registry())

    def with_overrides(self, changes: Mapping[str, float]) -> "AssumptionSet":
        merged = dict(self.overrides)
        for aid, v in changes.items():
            merged[aid] = self.registry.validate_value(aid, v)
        _check_joint_validity(self.registry, merged)
        return replace(self, overrides=MappingProxyType(dict(sorted(merged.items()))))

    @property
    def version(self) -> str:
        if not self.overrides:
            return REGISTRY_VERSION
        return f"{REGISTRY_VERSION}+{_digest(dict(self.overrides))}"

    def value(self, aid: str) -> float:
        if aid in self.overrides:
            return float(self.overrides[aid])
        return self.registry.value(aid)

    def engine_overrides(self) -> dict[str, float]:
        """The overrides the regime engine understands (transition names and income/tax/market params)."""
        return {k: float(v) for k, v in self.overrides.items() if self.registry.get(k).table != "game"}

    def used(self, ids) -> list[dict]:
        """Registry rows (with the value actually used) for a result's provenance block."""
        out = []
        for aid in sorted(set(ids)):
            a = self.registry.get(aid)
            d = a.to_dict()
            d["value_used"] = self.value(aid)
            d["overridden"] = aid in self.overrides
            if d["overridden"]:
                d["source_status"] = "user_override"
            out.append(d)
        return out


def _check_joint_validity(registry: Registry, values: Mapping[str, float]) -> None:
    """Constraints across rows: yearly inflation switching rows must leave a non-negative stay probability,
    and every row of the monthly transition matrix must keep its off-diagonal sum <= 1."""
    def v(aid):
        return float(values.get(aid, registry.value(aid)))

    for a, b in (("inflation_stay_low", "inflation_low_to_base"),
                 ("inflation_base_to_low", "inflation_base_to_high"),
                 ("inflation_high_to_low", "inflation_high_to_base")):
        if v(a) + v(b) > 1.0 + 1e-12:
            raise AssumptionError("not_stochastic", a, f"{a} + {b} exceeds 1; the inflation regime matrix would not be stochastic")
    by_from: dict[str, float] = {}
    for a in registry.rows.values():
        if a.table == "transitions" and " ages " not in a.applies_to:
            frm = a.applies_to.split(" -> ")[0]
            by_from[frm] = by_from.get(frm, 0.0) + v(a.id)
    for frm, total in by_from.items():
        if total > 1.0 + 1e-12:
            raise AssumptionError("not_stochastic", frm, f"monthly exit hazards from {frm} sum to {total:.3f} > 1")
