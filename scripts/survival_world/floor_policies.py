"""The B1 floor policies: what picks the substrate proposal in a floor arm (docs/experiments/b1_exp60_floor_prereg.md).

Each is a ``maxim.runtime.substrate_proposal.SubstratePolicy``: it sees this tick's ``available_tools``, the active
clusters and the encoder's own world values (``observation``), and nothing of NAc. Arms live in the harness (they
move to ``src/`` only when a second harness consumes them, docs/plans/baseline_floors.md "No shared arm registry").

- :class:`RandomPolicy` — uniform over the available tools (the RANDOM arm).
- :class:`TabularQPolicy` — tabular Q-learning (Q_EXPLORE, Q_EXPLORE_PRIOR, Q_EXPLORE_RAW), following the prereg's
  §Frozen implementation rules: keys, the time-integrated oxygen cost, the update order, ties broken by the seeded
  RNG, explicit episode ends, credit only for picks that executed.
- :class:`ArmedPolicy` — idle unless armed (the harness arms just before the teleport and disarms at rescue). It never
  raises inside the loop: a params violation or an inner exception is RECORDED and the pick dropped; the harness
  refuses the arm-seed when ``violations`` is non-empty after a window (a raise would end the loop, which the trial
  would read as a censored window).

The pure parts (keys, cost, the update rule, tie-breaking) are unit-tested in tests/unit/test_b1_floor_policies.py.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

# Frozen constants (prereg §The tabular learner + §Frozen implementation rules).
ALPHA = 0.5
GAMMA = 0.9
EPSILON_TRAIN = 0.2
EPSILON_PROBE = 0.0
OXYGEN_REST = 20.0  # the oxygen drive's set point; the cost is the deficit below it
COST_TICK_S = 0.5  # the cost is in 0.5 s units (one proposal interval)
PAIN_TERMINAL = -1.0  # the saturating oxygen pain, as a terminal reward
RAW_OXYGEN_BINS = (5.0, 10.0, 15.0)  # [0,5) [5,10) [10,15) [15,40]
PRIOR_SUFFIXES = ("_flee", "_escape_water")  # the `threat` affinity's tools on this body (asserted exactly two)
TERMINAL_REASONS = ("surfaced", "pain", "cap", "dirty", "never_submerged")


def oxygen_deficit(observation: Mapping[str, float]) -> float:
    """clip((20 − oxygen)/20, 0, 1) while in water (eye height), else 0."""
    if float(observation.get("is_in_water", 0.0) or 0.0) < 0.5:
        return 0.0
    oxygen = float(observation.get("oxygen", OXYGEN_REST) or 0.0)
    return min(1.0, max(0.0, (OXYGEN_REST - oxygen) / OXYGEN_REST))


def interval_cost(deficit_a: float, deficit_b: float, seconds: float) -> float:
    """The trapezoid of the deficit over an interval, in 0.5 s units (a blocking action pays for its time)."""
    return 0.5 * (deficit_a + deficit_b) * max(0.0, seconds) / COST_TICK_S


def raw_key(observation: Mapping[str, float]) -> tuple[Any, ...]:
    """(in water, oxygen bin) — the Q_EXPLORE_RAW key."""
    in_water = float(observation.get("is_in_water", 0.0) or 0.0) >= 0.5
    oxygen = float(observation.get("oxygen", OXYGEN_REST) or 0.0)
    return ("raw", in_water, sum(1 for edge in RAW_OXYGEN_BINS if oxygen >= edge))


def cluster_key(clusters: Mapping[str, str] | None) -> tuple[Any, ...]:
    """The tuple of the active modality clusters (Wire-4's keys)."""
    return ("clusters", tuple(sorted((clusters or {}).items())))


def argmax_ties_random(values: Mapping[str, float], rng: random.Random) -> str:
    """The best action, ties broken uniformly by ``rng`` (never by name; the candidates are sorted first so the draw
    is reproducible from the seed alone)."""
    best = max(values.values())
    candidates = sorted(a for a, v in values.items() if v == best)
    return candidates[0] if len(candidates) == 1 else rng.choice(candidates)


class RandomPolicy:
    """Uniform over the available tools, ``params={}``."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.picks: list[dict[str, Any]] = []

    def select(
        self,
        *,
        agent_id: str,
        available_tools: Sequence[str],
        clusters: Mapping[str, str] | None,
        observation: Mapping[str, float],
    ) -> dict[str, Any] | None:
        tool = self.rng.choice(sorted(available_tools))
        self.picks.append({"t": time.monotonic(), "action": tool})
        return {"tool_name": tool, "params": {}, "confidence": 1.0, "reasoning": "uniform"}

    def end_episode(self, reason: str, *, now: float | None = None) -> None:
        """RANDOM keeps no state across ticks."""


class TabularQPolicy:
    """Tabular Q-learning over the stamped action set.

    ``was_executed(action, since)`` answers whether the loop actually executed ``action`` after monotonic time
    ``since`` (the harness reads its executor spy); a pick that never executed (same-tool cap, pending at stop) is not
    credited and is recorded in ``drops``. ``pain_since()`` returns the summed intensity of the allow-listed pain
    published since its previous call (the harness's pain record, read with a cursor), so each pain counts once.
    """

    def __init__(
        self,
        *,
        seed: int,
        key_mode: str = "clusters",
        prior_only: bool = False,
        was_executed: Callable[[str, float], bool],
        pain_since: Callable[[], float],
        clock: Callable[[], float] = time.monotonic,
        alpha: float = ALPHA,
        gamma: float = GAMMA,
    ) -> None:
        if key_mode not in ("clusters", "raw"):
            raise ValueError(f"key_mode must be 'clusters' or 'raw', got {key_mode!r}")
        self.rng = random.Random(seed)
        self.key_mode, self.prior_only = key_mode, prior_only
        self.was_executed, self.pain_since, self.clock = was_executed, pain_since, clock
        self.alpha, self.gamma = alpha, gamma
        self.epsilon = EPSILON_TRAIN
        self.q: dict[tuple[Any, ...], dict[str, float]] = {}
        self._pending: dict[str, Any] | None = None  # {key, action, t, deficit}
        self._lock = threading.Lock()
        self.picks: list[dict[str, Any]] = []
        self.updates: list[dict[str, Any]] = []
        self.drops: list[dict[str, Any]] = []
        self.actions: tuple[str, ...] | None = None
        self.violations: list[str] = []

    # ── the pure pieces ──────────────────────────────────────────────────────────────────────────────
    def _key(self, clusters: Mapping[str, str] | None, observation: Mapping[str, float]) -> tuple[Any, ...]:
        return raw_key(observation) if self.key_mode == "raw" else cluster_key(clusters)

    def _row(self, key: tuple[Any, ...]) -> dict[str, float]:
        assert self.actions is not None
        return self.q.setdefault(key, {a: 0.0 for a in self.actions})

    def _credit(self, pending: dict[str, Any], reward: float, next_key: tuple[Any, ...] | None, now: float) -> None:
        """One Q update for ``pending``'s action; ``next_key`` None is terminal. Skipped if the pick never executed."""
        if not self.was_executed(pending["action"], pending["t"]):
            self.drops.append({"t": now, "key": pending["key"], "action": pending["action"]})
            return
        row = self._row(pending["key"])
        future = 0.0 if next_key is None else max(self._row(next_key).values())
        target = reward + self.gamma * future
        before = row[pending["action"]]
        row[pending["action"]] = before + self.alpha * (target - before)
        self.updates.append(
            {"t": now, "key": pending["key"], "action": pending["action"], "reward": reward, "target": target,
             "terminal": next_key is None, "q_before": before, "q_after": row[pending["action"]]}
        )  # fmt: skip

    # ── the SubstratePolicy surface (loop thread) ────────────────────────────────────────────────────
    def select(
        self,
        *,
        agent_id: str,
        available_tools: Sequence[str],
        clusters: Mapping[str, str] | None,
        observation: Mapping[str, float],
    ) -> dict[str, Any] | None:
        with self._lock:
            now = self.clock()
            if self.actions is None:
                tools = sorted(available_tools)
                if self.prior_only:
                    tools = [t for t in tools if t.endswith(PRIOR_SUFFIXES)]
                    if len(tools) != 2:
                        self.violations.append(f"prior action set is {tools}, not exactly two")
                self.actions = tuple(tools)
            key = self._key(clusters, observation)
            deficit = oxygen_deficit(observation)
            if self._pending is not None:
                p = self._pending
                reward = -interval_cost(p["deficit"], deficit, now - p["t"]) - self.pain_since()
                self._credit(p, reward, key, now)
            row = self._row(key)
            explored = self.rng.random() < self.epsilon
            action = self.rng.choice(sorted(row)) if explored else argmax_ties_random(row, self.rng)
            self._pending = {"key": key, "action": action, "t": now, "deficit": deficit}
            self.picks.append({"t": now, "key": key, "action": action, "explored": explored, "q": dict(row)})
            return {"tool_name": action, "params": {}, "confidence": 1.0, "reasoning": f"q={row[action]:.3f}"}

    # ── the harness surface (harness thread) ─────────────────────────────────────────────────────────
    def end_episode(self, reason: str, *, now: float | None = None) -> None:
        """Close the pending (s, a): surfaced → terminal 0; pain → terminal −1.0; cap/dirty/never_submerged →
        truncated (no update). The final interval's cost is the last deficit held to ``now``."""
        if reason not in TERMINAL_REASONS:
            raise ValueError(f"unknown episode end {reason!r}")
        with self._lock:
            p, self._pending = self._pending, None
            if p is None or reason not in ("surfaced", "pain"):
                return
            t = self.clock() if now is None else now
            reward = -interval_cost(p["deficit"], p["deficit"], t - p["t"]) - self.pain_since()
            if reason == "pain":
                reward += PAIN_TERMINAL
            self._credit(p, reward, None, t)

    def clear_pending(self) -> None:
        """Arming clears the pending (s, a) (prereg rule 4): nothing bootstraps across a rescue."""
        with self._lock:
            self._pending = None
            self.pain_since()  # discard pain published while disarmed

    def set_probe_mode(self, probe: bool) -> None:
        with self._lock:
            self.epsilon = EPSILON_PROBE if probe else EPSILON_TRAIN


class ArmedPolicy:
    """Idle unless armed; records (never raises) a params violation or an inner exception, for the harness to
    refuse on after the window."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self._armed = threading.Event()
        self.violations: list[str] = []
        self.t_arm: float | None = None

    def arm(self) -> float:
        if hasattr(self.inner, "clear_pending"):
            self.inner.clear_pending()
        self.t_arm = time.monotonic()
        self._armed.set()
        return self.t_arm

    def disarm(self) -> None:
        self._armed.clear()

    @property
    def armed(self) -> bool:
        return self._armed.is_set()

    def all_violations(self) -> list[str]:
        return self.violations + list(getattr(self.inner, "violations", []))

    def select(
        self,
        *,
        agent_id: str,
        available_tools: Sequence[str],
        clusters: Mapping[str, str] | None,
        observation: Mapping[str, float],
    ) -> Mapping[str, Any] | None:
        if not self._armed.is_set():
            return None
        try:
            pick = self.inner.select(
                agent_id=agent_id, available_tools=available_tools, clusters=clusters, observation=observation
            )
        except Exception as exc:  # noqa: BLE001 — recorded and refused by the harness; a raise would end the loop
            self.violations.append(f"policy raised {type(exc).__name__}: {exc}")
            return None
        if pick is not None and (pick.get("params") or {}):
            self.violations.append(f"non-empty params {pick.get('params')!r} for {pick.get('tool_name')!r}")
            return None
        return pick
