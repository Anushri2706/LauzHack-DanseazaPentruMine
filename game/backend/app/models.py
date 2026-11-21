"""Pydantic request/response schemas for the API."""
from pydantic import BaseModel, Field


class DecisionIn(BaseModel):
    month: int = Field(ge=0)
    choice: str
    savings_rate: float | None = Field(default=None, ge=0, le=1)
    loan_amount: float | None = Field(default=None, gt=0)
    repay_debt: bool = False


class GoalIn(BaseModel):
    goal_id: str
    name: str
    target: float = Field(ge=0)
    month: int  # row index of the target date (month 0 = first simulated month)
    priority: str = "high"
    custom: bool = False


class BranchRequest(BaseModel):
    persona_id: str = "P00001"
    assumptions: dict[str, float] = Field(default_factory=dict)  # overrides of the defaults
    decisions: list[DecisionIn] = Field(default_factory=list)
    goals: list[GoalIn] | None = None  # None = the persona's goals from the data
    seed: int = 42


class MonteCarloRequest(BranchRequest):
    n: int = Field(default=500, ge=10, le=2000)


class WorldRequest(BranchRequest):
    world: int = Field(ge=0, le=100_000)


class CompareRequest(BaseModel):
    """Old decision-point branches, or (since the town game) realities with monthly turns."""
    a: "GameRequest | BranchRequest" = Field(union_mode="left_to_right")
    b: "GameRequest | BranchRequest" = Field(union_mode="left_to_right")
    name_a: str = "Reality A"
    name_b: str = "Reality B"


class Row(BaseModel):
    t: int
    age: float
    status: str
    income: float
    expenses: float
    one_offs: float
    one_off_labels: list[str]
    cash: float
    investments: float
    debt: float
    net_worth: float
    buffer_target: float
    flags: list[str]


class Choice(BaseModel):
    id: str
    label: str
    preview: str
    default: bool


class NextDecision(BaseModel):
    month: int
    age: float
    reasons: list[str]
    narrative: str
    choices: list[Choice]
    extras: dict


class Point(BaseModel):
    month: int
    age: float
    reasons: list[str]
    chosen: str
    label: str
    by: str


class BranchResult(BaseModel):
    persona: dict
    rows: list[Row]
    points: list[Point]
    current_month: int            # rows before this month are "lived"; the rest is projection
    next_decision: NextDecision | None
    summary: dict
    goals: list[dict]
    scores: dict
    now: dict


# ----------------------------------------------------------------------------- town game
N_WORLDS = 500


class ActionIn(BaseModel):
    type: str
    id: str | None = None
    value: float | None = None


class TurnIn(BaseModel):
    month: int = Field(ge=0)
    actions: list[ActionIn] = Field(default_factory=list)


class GameRequest(BaseModel):
    """One reality. `turns` with month < current_month are lived; a turn AT current_month is this
    month's queue (applied when the player sleeps). The client holds the multiverse tree."""
    persona_id: str = "P00001"
    look: dict = Field(default_factory=dict)          # cosmetic only; never used in maths
    assumptions: dict[str, float] = Field(default_factory=dict)
    seed: int = 42
    world: int = Field(default=0, ge=0, lt=N_WORLDS)
    turns: list[TurnIn]                               # required (distinguishes it from BranchRequest)
    current_month: int = Field(default=0, ge=0)
    goals: list[GoalIn] | None = None


class GameProjectionRequest(GameRequest):
    n: int = Field(default=N_WORLDS, ge=10, le=N_WORLDS)


class GameRollRequest(GameRequest):
    roll: int = Field(ge=0, lt=N_WORLDS)


class HmmStartRequest(BaseModel):
    persona_id: str = "P00001"


class HmmBranchRequest(BaseModel):
    snapshot: dict
    decisions: list[str]
    until_month: int | None = None
    n_paths: int = Field(default=40, ge=2, le=5000)


class HmmCommitRequest(BaseModel):
    snapshot: dict
    flows: list[float]
    decision: str | None = None
    label: str = ""
    cash: float | None = None
    debt: float | None = None
    investments: float | None = None


class HmmRestoreRequest(BaseModel):
    snapshot: dict


class HmmDistributionRequest(BaseModel):
    snapshot: dict
    horizon_months: int = Field(default=60, ge=1, le=240)
    n_paths: int = Field(default=40, ge=2, le=5000)


CompareRequest.model_rebuild()
