"""Branch domain model and the in-memory repository.

STORAGE IS IN MEMORY ONLY. Every game, branch and result is lost when the server restarts. Nothing is
written to disk or to a database.

Immutability rules
* ``Branch`` and ``StoredResult`` are frozen dataclasses. "Changing" a branch (rename, pin, append a
  decision) stores a new value under the same id; the parent and siblings are never touched.
* A branch's own decision list is append-only. Its effective history is its ancestors' decisions
  (root first) followed by its own, resolved by reference, so a decision appended to a parent makes
  that parent's descendants' results stale (their input fingerprint changes) and leaves siblings alone.
* Results are never overwritten: each simulation appends a new version.
"""
from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Mapping

from .engine.decisions import Decision
from .engine.profile import PlayerProfile
from .engine.simulate import SimulationConfig

MAX_GAMES = 200


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class NotFound(KeyError):
    def __init__(self, what: str, ident: str):
        super().__init__(f"{what} {ident!r} not found")
        self.what, self.ident = what, ident


class Conflict(ValueError):
    pass


@dataclass(frozen=True)
class Branch:
    id: str
    game_id: str
    parent_id: str | None
    name: str
    fork_month: int                                # decisions on this branch must start at or after this month
    own_decisions: tuple[Decision, ...] = ()
    own_overrides: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))
    pinned: bool = False
    created_at: str = field(default_factory=now)


@dataclass(frozen=True)
class StoredResult:
    result_id: str
    branch_id: str
    version: int
    run_kind: str                  # standard | fresh_seed | replay
    seed: int
    config: SimulationConfig
    assumptions_version: str
    overrides: Mapping[str, float]
    decisions: tuple[Decision, ...]
    inputs_fingerprint: str
    summary: dict
    created_at: str = field(default_factory=now)


@dataclass
class Game:
    id: str
    profile: PlayerProfile
    seed: int
    config: SimulationConfig
    root_id: str
    created_at: str = field(default_factory=now)
    branches: dict[str, Branch] = field(default_factory=dict)
    results: dict[str, list[StoredResult]] = field(default_factory=dict)


class InMemoryRepository:
    """Thread-safe, process-local. Oldest games are evicted beyond MAX_GAMES."""

    persistent = False

    def __init__(self):
        self._games: OrderedDict[str, Game] = OrderedDict()
        self.lock = threading.RLock()

    def add_game(self, game: Game) -> None:
        with self.lock:
            self._games[game.id] = game
            while len(self._games) > MAX_GAMES:
                self._games.popitem(last=False)

    def game(self, game_id: str) -> Game:
        with self.lock:
            if game_id not in self._games:
                raise NotFound("game", game_id)
            return self._games[game_id]

    def branch(self, game_id: str, branch_id: str) -> Branch:
        g = self.game(game_id)
        if branch_id not in g.branches:
            raise NotFound("branch", branch_id)
        return g.branches[branch_id]

    def put_branch(self, branch: Branch) -> None:
        with self.lock:
            self.game(branch.game_id).branches[branch.id] = branch

    def add_result(self, game_id: str, result: StoredResult) -> None:
        with self.lock:
            self.game(game_id).results.setdefault(result.branch_id, []).append(result)

    def results(self, game_id: str, branch_id: str) -> list[StoredResult]:
        self.branch(game_id, branch_id)
        return list(self.game(game_id).results.get(branch_id, []))
