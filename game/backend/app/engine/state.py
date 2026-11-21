"""Immutable simulation state and decision record."""
from dataclasses import dataclass, field

STATUSES = ("studying", "working", "abroad", "unemployed", "retired")


@dataclass(frozen=True)
class OneOff:
    month: int
    amount: float  # positive = cash out
    label: str
    kind: str  # "tuition" | "relocation" | "moving" | "purchase" | "trip" | "home_fee" | "other"
    joy: float = 0.0


@dataclass(frozen=True)
class State:
    t: int                      # month index; the state is the opening state of month t
    start_age: float
    cash: float
    investments: float
    debt: float
    debt_rate: float            # annual nominal rate on outstanding debt
    debt_payment: float         # scheduled monthly payment (interest + principal)
    status: str
    salary: float               # monthly salary (before the flat tax/social rates)
    salary_start: int           # month the current salary began (raises every 12 months after)
    rent: float                 # base rent in month-0 prices
    living: float               # other living costs in month-0 prices
    savings_rate: float
    program: str | None = None  # "master_ft" | "master_pt" while studying for a master's
    program_start: int | None = None
    program_end: int | None = None
    has_master: bool = False
    last_switch: int | None = None  # month of the last big move (cooldown rule)
    scheduled: tuple[OneOff, ...] = field(default_factory=tuple)
    # Game extensions (defaults reproduce the original engine exactly).
    habits: tuple[tuple[str, int], ...] = ()   # ((habit_id, level), ...), level 0 never stored
    flat_factor: float = 1.0                    # rent factor from "Change flat"
    home_id: str | None = None
    home_price: float = 0.0                     # purchase price (basis for amortization/maintenance)
    home_value: float = 0.0
    mortgage: float = 0.0
    work_share: float = 1.0                     # 0.8 after "reduce to 80%"
    last_raise_ask: int | None = None
    retiree: bool = False                       # studying after retirement: pension continues, returns to retired

    @property
    def owner(self) -> bool:
        return self.home_id is not None

    @property
    def age(self) -> float:
        return self.start_age + self.t / 12.0


@dataclass(frozen=True)
class Decision:
    """One choice at one decision point. A branch is just an ordered list of these."""
    month: int
    choice: str
    savings_rate: float | None = None   # optional: change savings rate at the same time
    loan_amount: float | None = None    # optional: take a loan of this size
    repay_debt: bool = False            # optional: repay all debt from cash
