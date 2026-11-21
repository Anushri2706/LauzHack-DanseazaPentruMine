"""Process-wide singletons for the API. Storage is in memory (see storage.py)."""
from __future__ import annotations

from functools import lru_cache

from .game_service import GameService


@lru_cache(maxsize=1)
def get_service() -> GameService:
    return GameService()
