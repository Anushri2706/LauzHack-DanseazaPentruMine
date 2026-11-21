"""Game service: branch operations on top of the engine. No HTTP here (FastAPI lives in api.py).

Common random numbers
---------------------
Every branch of a game is simulated with the game's seed (unless a fresh run is asked for), so paired
comparisons share: the Dirichlet transition draws, the market stream (inflation regimes and returns,
default_rng(10_000 + seed)) and the independent-event stream (SeedSequence([seed, ...])), exactly.
The HMM month-by-month stream shares the seed too, but a decision that pins months consumes fewer
draws, so HMM paths realign only statistically after the first pinned month. Comparisons are therefore
differences of distribution statistics, not path-by-path effects.

Replay vs fresh run: ``replay`` re-runs a stored result's exact inputs (seed, config, decisions,
assumption overrides) and checks the output fingerprint. ``simulate(fresh_seed=...)`` is a NEW Monte
Carlo draw with another seed; it is labelled ``fresh_seed`` and is not comparable with branches run on
the game seed.
"""
from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import replace
from functools import lru_cache
from types import MappingProxyType

from .data.canton import rent_ratio
from .engine.assumptions import AssumptionSet
from .engine.decisions import Decision, catalogue_for, validate_decision
from .engine.montecarlo import compare_summaries, scalar_facts, summarise
from .engine.profile import to_engine_profile, validate_profile
from .engine.sensitivity import bisect, one_at_a_time
from .engine.simulate import SimulationConfig, engine_for, run_monte_carlo
from .storage import Branch, Conflict, Game, InMemoryRepository, StoredResult, new_id


@lru_cache(maxsize=1)
def default_assumptions() -> AssumptionSet:
    return AssumptionSet.default()


class GameService:
    def __init__(self, repo: InMemoryRepository | None = None, assumptions: AssumptionSet | None = None):
        self.repo = repo or InMemoryRepository()
        self.base_assumptions = assumptions or default_assumptions()
        self._engines: dict[str, object] = {}

    # ------------------------------------------------------------------ helpers
    def _engine_profile(self, game: Game, assumptions: AssumptionSet):
        return to_engine_profile(game.profile, assumptions)

    def _engine(self, game: Game, ep):
        key = (game.id, game.config.engine_mode)
        if key not in self._engines:
            self._engines[key] = engine_for(ep, game.profile.other_expenses_chf_month, game.config.engine_mode)
        return self._engines[key]

    def lineage(self, game: Game, branch_id: str) -> list[Branch]:
        out, cur = [], game.branches[branch_id]
        while cur is not None:
            out.append(cur)
            cur = game.branches.get(cur.parent_id) if cur.parent_id else None
        return list(reversed(out))

    def effective_decisions(self, game: Game, branch_id: str) -> tuple[Decision, ...]:
        return tuple(d for b in self.lineage(game, branch_id) for d in b.own_decisions)

    def effective_assumptions(self, game: Game, branch_id: str) -> AssumptionSet:
        merged: dict = {}
        for b in self.lineage(game, branch_id):
            merged.update(b.own_overrides)
        return self.base_assumptions.with_overrides(merged) if merged else self.base_assumptions

    def inputs_fingerprint(self, game: Game, decisions, assumptions: AssumptionSet, config: SimulationConfig,
                           seed: int) -> str:
        payload = {"profile": game.profile.fingerprint(), "decisions": [d.to_dict() for d in decisions],
                   "assumptions": assumptions.version, "config": config.to_dict(), "seed": seed}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def _canton_ratio(self, game: Game):
        canton = game.profile.canton
        return (lambda c: rent_ratio(canton, c)) if canton else None

    # ------------------------------------------------------------------ game
    def start(self, profile_raw: dict, config: SimulationConfig | None = None, seed: int | None = None) -> Game:
        profile = validate_profile(profile_raw)
        config = config or SimulationConfig()
        config.validate()
        seed = secrets.randbelow(2 ** 31) if seed is None else int(seed)
        if seed < 0:
            raise ValueError("seed must be non-negative")
        gid, rid = new_id("game"), new_id("br")
        game = Game(gid, profile, seed, config, rid)
        game.branches[rid] = Branch(rid, gid, None, "Root (today)", 0)
        to_engine_profile(profile, self.base_assumptions)  # fail fast on conversion problems
        self.repo.add_game(game)
        return game

    def decision_catalogue(self, game: Game) -> list[dict]:
        ep = self._engine_profile(game, self.base_assumptions)
        return catalogue_for(ep.profile.initial_state)

    # ------------------------------------------------------------------ branches
    def create_branch(self, game_id: str, parent_id: str, name: str, fork_month: int | None = None) -> Branch:
        with self.repo.lock:
            game = self.repo.game(game_id)
            parent = self.repo.branch(game_id, parent_id)
            inherited = self.effective_decisions(game, parent_id)
            floor = max([parent.fork_month] + [d.start_month for d in inherited])
            fm = floor if fork_month is None else int(fork_month)
            if fm < floor or fm >= game.config.horizon_months:
                raise Conflict(f"fork_month must be in [{floor}, {game.config.horizon_months - 1}]")
            b = Branch(new_id("br"), game_id, parent_id, name.strip() or "Branch", fm)
            self.repo.put_branch(b)
            return b

    def update_branch(self, game_id: str, branch_id: str, name: str | None = None, pinned: bool | None = None) -> Branch:
        with self.repo.lock:
            b = self.repo.branch(game_id, branch_id)
            b2 = replace(b, name=name.strip() if name else b.name, pinned=b.pinned if pinned is None else bool(pinned))
            self.repo.put_branch(b2)
            return b2

    def add_decision(self, game_id: str, branch_id: str, decision_id: str, start_month, params: dict | None) -> Decision:
        with self.repo.lock:
            game = self.repo.game(game_id)
            b = self.repo.branch(game_id, branch_id)
            A = self.effective_assumptions(game, branch_id)
            ep = self._engine_profile(game, A)
            d = validate_decision(decision_id, start_month, params, initial_state=ep.profile.initial_state,
                                  horizon_months=game.config.horizon_months, assumptions=A,
                                  current_canton=game.profile.canton, min_start_month=b.fork_month)
            self.repo.put_branch(replace(b, own_decisions=b.own_decisions + (d,)))
            return d

    def set_overrides(self, game_id: str, branch_id: str, overrides: dict[str, float]) -> Branch:
        with self.repo.lock:
            game = self.repo.game(game_id)
            b = self.repo.branch(game_id, branch_id)
            merged = {**b.own_overrides, **overrides}
            parent = self.effective_assumptions(game, b.parent_id) if b.parent_id else self.base_assumptions
            parent.with_overrides(merged)      # validates ranges and joint stochastic constraints
            b2 = replace(b, own_overrides=MappingProxyType(dict(sorted(merged.items()))))
            self.repo.put_branch(b2)
            return b2

    # ------------------------------------------------------------------ simulation
    def _run(self, game: Game, decisions, A: AssumptionSet, config: SimulationConfig, seed: int) -> dict:
        ep = self._engine_profile(game, A)
        run = run_monte_carlo(ep, list(decisions), A, config, config.n_paths, seed, self._engine(game, ep),
                              self._canton_ratio(game))
        s = summarise(run)
        s["assumptions_used"] = A.used(run.assumption_ids)
        s["profile_derived"] = ep.derived
        return s

    def simulate(self, game_id: str, branch_id: str, fresh_seed: int | None = None) -> StoredResult:
        game = self.repo.game(game_id)
        self.repo.branch(game_id, branch_id)
        decisions = self.effective_decisions(game, branch_id)
        A = self.effective_assumptions(game, branch_id)
        seed = game.seed if fresh_seed is None else int(fresh_seed)
        summary = self._run(game, decisions, A, game.config, seed)
        with self.repo.lock:
            version = len(self.repo.results(game_id, branch_id)) + 1
            res = StoredResult(new_id("res"), branch_id, version,
                               "standard" if fresh_seed is None or seed == game.seed else "fresh_seed",
                               seed, game.config, A.version, MappingProxyType(dict(A.overrides)), decisions,
                               self.inputs_fingerprint(game, decisions, A, game.config, seed), summary)
            self.repo.add_result(game_id, res)
            return res

    def is_stale(self, game: Game, res: StoredResult) -> bool:
        decisions = self.effective_decisions(game, res.branch_id)
        A = self.effective_assumptions(game, res.branch_id)
        return res.inputs_fingerprint != self.inputs_fingerprint(game, decisions, A, res.config, res.seed)

    def latest(self, game_id: str, branch_id: str, version: int | None = None) -> StoredResult | None:
        rs = self.repo.results(game_id, branch_id)
        if version is not None:
            return next((r for r in rs if r.version == version), None)
        return rs[-1] if rs else None

    def find_result(self, game_id: str, result_id: str) -> StoredResult | None:
        game = self.repo.game(game_id)
        for rs in game.results.values():
            for r in rs:
                if r.result_id == result_id:
                    return r
        return None

    def replay(self, game_id: str, branch_id: str, result_id: str | None = None) -> dict:
        game = self.repo.game(game_id)
        res = self.find_result(game_id, result_id) if result_id else self.latest(game_id, branch_id)
        if res is None or res.branch_id != branch_id:
            raise Conflict("no stored result to replay for this branch")
        A = self.base_assumptions.with_overrides(dict(res.overrides)) if res.overrides else self.base_assumptions
        if A.version != res.assumptions_version:
            raise Conflict("assumption registry changed since this result; cannot replay exactly")
        again = self._run(game, res.decisions, A, res.config, res.seed)
        return {"result_id": res.result_id, "seed": res.seed, "assumptions_version": res.assumptions_version,
                "original_fingerprint": res.summary["fingerprint"], "replay_fingerprint": again["fingerprint"],
                "identical": again["fingerprint"] == res.summary["fingerprint"] and again["final"] == res.summary["final"]}

    def compare(self, game_id: str, a: str, b: str) -> dict:
        game = self.repo.game(game_id)
        ra, rb = self.latest(game_id, a), self.latest(game_id, b)
        missing = [x for x, r in ((a, ra), (b, rb)) if r is None]
        if missing:
            raise Conflict(f"simulate first: no result for {missing}")
        stale = [x for x, r in ((a, ra), (b, rb)) if self.is_stale(game, r)]
        if stale:
            raise Conflict(f"stale result for {stale}; re-simulate before comparing")
        out = compare_summaries(ra.summary, rb.summary)
        out.update({"a": {"branch_id": a, "result_id": ra.result_id, "version": ra.version},
                    "b": {"branch_id": b, "result_id": rb.result_id, "version": rb.version},
                    "seed": ra.seed, "horizon_months": ra.summary["horizon_months"],
                    "noise_control": "common random numbers (same seed); see game_service docstring"})
        return out

    def sensitivity(self, game_id: str, branch_id: str, ids: list[str] | None, metric: str, rel: float = 0.10,
                    n_paths: int | None = None) -> dict:
        game = self.repo.game(game_id)
        decisions = self.effective_decisions(game, branch_id)
        A = self.effective_assumptions(game, branch_id)
        cfg = replace(game.config, n_paths=int(n_paths or min(game.config.n_paths, 300)))
        out = one_at_a_time(A, lambda X: self._run(game, decisions, X, cfg, game.seed), ids, rel, metric)
        out.update({"branch_id": branch_id, "seed": game.seed, "n_paths": cfg.n_paths})
        return out

    def break_even(self, game_id: str, a: str, b: str, assumption_id: str, lower: float, upper: float,
                   metric: str = "net_worth_real.final.p50", tol: float | None = None) -> dict:
        """x in [lower, upper] where metric(A; x) == metric(B; x). x overrides the assumption in both branches.
        Fixed game seed (common random numbers), so the objective is deterministic in x."""
        game = self.repo.game(game_id)
        A0 = self.base_assumptions
        A0.registry.get(assumption_id)
        A0.registry.validate_value(assumption_id, lower)
        A0.registry.validate_value(assumption_id, upper)
        cfg = replace(game.config, n_paths=min(game.config.n_paths, 300))
        da, db = self.effective_decisions(game, a), self.effective_decisions(game, b)

        def f(x):
            va = scalar_facts(self._run(game, da, self.effective_assumptions(game, a).with_overrides({assumption_id: x}), cfg, game.seed))
            vb = scalar_facts(self._run(game, db, self.effective_assumptions(game, b).with_overrides({assumption_id: x}), cfg, game.seed))
            return va[metric]["value"] - vb[metric]["value"]

        r = bisect(f, float(lower), float(upper), tol=tol or (upper - lower) * 1e-3, max_iter=30)
        return {**r.to_dict(), "assumption_id": assumption_id, "metric": metric, "a": a, "b": b,
                "objective": f"{metric}(a) - {metric}(b)", "seed": game.seed, "n_paths": cfg.n_paths,
                "noise_control": "fixed seed for every evaluation (common random numbers)",
                "caveat": "Simulated quantities under the stated assumptions; the function may not be monotone."}

    # ------------------------------------------------------------------ views
    def branch_view(self, game: Game, b: Branch) -> dict:
        latest = self.latest(game.id, b.id)
        status = "not_simulated" if latest is None else ("stale" if self.is_stale(game, latest) else "simulated")
        return {"id": b.id, "parent_id": b.parent_id, "name": b.name, "fork_month": b.fork_month,
                "pinned": b.pinned, "created_at": b.created_at,
                "own_decisions": [d.to_dict() for d in b.own_decisions],
                "inherited_decisions": [d.to_dict() for d in (self.effective_decisions(game, b.parent_id) if b.parent_id else ())],
                "own_overrides": dict(b.own_overrides),
                "assumptions_version": self.effective_assumptions(game, b.id).version,
                "status": status,
                "latest_result": None if latest is None else {"result_id": latest.result_id, "version": latest.version,
                                                              "run_kind": latest.run_kind, "seed": latest.seed},
                "children": [c.id for c in game.branches.values() if c.parent_id == b.id]}

    def game_view(self, game: Game) -> dict:
        from dataclasses import asdict
        return {"id": game.id, "seed": game.seed, "config": game.config.to_dict(), "root_id": game.root_id,
                "created_at": game.created_at, "profile": asdict(game.profile),
                "storage": "in-memory: lost when the server restarts",
                "branches": [self.branch_view(game, b) for b in game.branches.values()]}

    def result_view(self, game: Game, res: StoredResult) -> dict:
        return {"result_id": res.result_id, "branch_id": res.branch_id, "version": res.version,
                "run_kind": res.run_kind, "seed": res.seed, "assumptions_version": res.assumptions_version,
                "created_at": res.created_at, "stale": self.is_stale(game, res),
                "decisions": [d.to_dict() for d in res.decisions], "summary": res.summary}
