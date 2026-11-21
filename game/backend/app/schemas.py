"""Pydantic request/response schemas (HTTP layer only; the engine uses its own dataclasses)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Strict = ConfigDict(extra="forbid")


class ProfileIn(BaseModel):
    model_config = Strict
    age: int
    employment: Literal["employed", "part_time", "self_employed", "student", "unemployed", "retired"]
    income_amount_chf: float
    income_period: Literal["monthly", "yearly"]
    income_basis: Literal["gross", "net"]
    rent_chf_month: float
    cash_chf: float
    investments_chf: float = 0.0
    debt_chf: float = 0.0
    debt_annual_rate: float | None = None
    other_expenses_chf_month: float | None = None
    canton: str | None = None
    household: Literal["single", "couple", "family", "shared_housing"] | None = None
    emergency_reserve_chf: float | None = None
    sources: dict[str, Literal["user", "confirmed_user", "default", "derived"]] = Field(default_factory=dict)


class ConfigIn(BaseModel):
    model_config = Strict
    horizon_months: int = Field(120, ge=12, le=600)
    n_paths: int = Field(500, ge=20, le=5000)
    inflation: bool = True
    enabled_events: list[str] = Field(default_factory=list)


class StartGameIn(BaseModel):
    model_config = Strict
    profile: ProfileIn
    config: ConfigIn = Field(default_factory=ConfigIn)
    seed: int | None = Field(None, ge=0, le=2**31 - 1)


class CreateBranchIn(BaseModel):
    model_config = Strict
    parent_id: str
    name: str = Field("Branch", max_length=80)
    fork_month: int | None = Field(None, ge=0)


class UpdateBranchIn(BaseModel):
    model_config = Strict
    name: str | None = Field(None, max_length=80)
    pinned: bool | None = None


class DecisionIn(BaseModel):
    model_config = Strict
    decision_id: str
    start_month: int
    params: dict[str, float | int | str | None] = Field(default_factory=dict)


class OverridesIn(BaseModel):
    model_config = Strict
    overrides: dict[str, float]


class SimulateIn(BaseModel):
    model_config = Strict
    fresh_seed: int | None = Field(None, ge=0, le=2**31 - 1)


class CompareIn(BaseModel):
    model_config = Strict
    a: str
    b: str


class SensitivityIn(BaseModel):
    model_config = Strict
    branch_id: str
    assumption_ids: list[str] | None = None
    metric: str = "net_worth_real.final.p50"
    rel_change: float = Field(0.10, gt=0, lt=0.5)
    n_paths: int | None = Field(None, ge=20, le=1000)


class BreakEvenIn(BaseModel):
    model_config = Strict
    a: str
    b: str
    assumption_id: str
    lower: float
    upper: float
    metric: str = "net_worth_real.final.p50"


class ReplayIn(BaseModel):
    model_config = Strict
    result_id: str | None = None


class ExtractIn(BaseModel):
    model_config = Strict
    message: str = Field(..., min_length=1, max_length=2000)
    use_llm: bool = True


class ExplainIn(BaseModel):
    model_config = Strict
    game_id: str
    branch_id: str
    compare_with: str | None = None
    question: str = Field("", max_length=1000)
    use_llm: bool = True
    draft: dict[str, Any] | None = None     # pre-written claims to check (e.g. from another client)


class ChatIn(BaseModel):
    model_config = Strict
    message: str = Field(..., min_length=1, max_length=2000)
    game_id: str | None = None
    branch_id: str | None = None
    compare_with: str | None = None
    use_llm: bool = True

