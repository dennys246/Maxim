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
GAMMA = 1.0  # undiscounted: an episodic cost-to-go that always terminates (owner decision OD5, 2026-10-11)
EPSILON_TRAIN = 0.2
EPSILON_PROBE = 0.0
OXYGEN_REST = 20.0  # the oxygen drive's set point; the cost is the deficit below it
COST_TICK_S = 0.5  # the cost is in 0.5 s units (one proposal interval)
PAIN_TERMINAL = -1.0  # the saturating oxygen pain, as a terminal reward
SATURATING_PAIN = 1.0  # that pain's own intensity, which the interval reward also charges
# Q₀: the pain components of a pain episode (the terminal plus the saturating pain's intensity; the full return, with
# its oxygen cost, is ≈ −4.4). Any Q₀ from −1 to −10 works once γ = 1 (rationale/b1-exp60-floor/confounding-v4.md);
# with Q₀ = 0 every untried action outranks a tried escape, so the floor explored instead of learning
# (rationale/b1-exp60-floor/learner_sim.py; owner decision OD5, 2026-10-11).
Q_INIT = PAIN_TERMINAL - SATURATING_PAIN
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


def in_water(observation: Mapping[str, float]) -> bool:
    return float(observation.get("is_in_water", 0.0) or 0.0) >= 0.5


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
        self.phase: str | None = None  # the harness's window label, stamped on each pick

    def select(
        self,
        *,
        agent_id: str,
        available_tools: Sequence[str],
        clusters: Mapping[str, str] | None,
        observation: Mapping[str, float],
    ) -> dict[str, Any] | None:
        tool = self.rng.choice(sorted(available_tools))
        self.picks.append(
            {"t": time.monotonic(), "action": tool, "in_water": in_water(observation), "phase": self.phase}
        )
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
        self.phase: str | None = None  # the harness's window label, stamped on each pick

    # ── the pure pieces ──────────────────────────────────────────────────────────────────────────────
    def _key(self, clusters: Mapping[str, str] | None, observation: Mapping[str, float]) -> tuple[Any, ...]:
        return raw_key(observation) if self.key_mode == "raw" else cluster_key(clusters)

    def _row(self, key: tuple[Any, ...]) -> dict[str, float]:
        assert self.actions is not None
        return self.q.setdefault(key, {a: Q_INIT for a in self.actions})

    def _credit(self, pending: dict[str, Any], reward: float, next_key: tuple[Any, ...] | None, now: float) -> None:
        """One Q update for ``pending``'s action; ``next_key`` None is terminal. Skipped if the pick never executed."""
        if not self.was_executed(pending["action"], pending["t"]):
            self.drops.append({"t": now, "key": pending["key"], "action": pending["action"], "why": "not_executed"})
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

    def _close(self, p: dict[str, Any], *, deficit: float, next_key: tuple[Any, ...] | None, now: float) -> None:
        """Credit the pending pick at the next select. Only a pick made IN WATER is credited (on the shore there is no
        cost and no terminal; a shore pick before the teleport did not move the body). In B1's placements and training
        the episode is closed by the harness's ``end_episode("surfaced")`` (water-only arming means this policy never
        sees a dry select there); as a backstop, a select that sees air after a water pick is terminal 0 (``next_key``
        None), so a pick made after the head cleared is never credited with the surfacing."""
        if not p["in_water"]:
            self.drops.append({"t": now, "key": p["key"], "action": p["action"], "why": "made_out_of_water"})
            return
        reward = -interval_cost(p["deficit"], deficit, now - p["t"]) - self.pain_since()
        self._credit(p, reward, next_key, now)

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
            water = in_water(observation)
            if self._pending is not None:
                self._close(self._pending, deficit=deficit, next_key=key if water else None, now=now)
            row = self._row(key)
            explored = self.rng.random() < self.epsilon
            action = self.rng.choice(sorted(row)) if explored else argmax_ties_random(row, self.rng)
            self._pending = {"key": key, "action": action, "t": now, "deficit": deficit, "in_water": water}
            self.picks.append(
                {"t": now, "key": key, "action": action, "explored": explored, "q": dict(row),
                 "in_water": water, "phase": self.phase}
            )  # fmt: skip
            return {"tool_name": action, "params": {}, "confidence": 1.0, "reasoning": f"q={row[action]:.3f}"}

    # ── the harness surface (harness thread) ─────────────────────────────────────────────────────────
    def end_episode(self, reason: str, *, now: float | None = None) -> None:
        """Close the pending (s, a): surfaced → terminal 0; pain → terminal −1.0; cap/dirty/never_submerged →
        truncated (no update). The final interval's cost is the last deficit held to ``now``. A pending pick made
        out of water (after the surfacing the policy already credited at its select) is truncated."""
        if reason not in TERMINAL_REASONS:
            raise ValueError(f"unknown episode end {reason!r}")
        with self._lock:
            p, self._pending = self._pending, None
            if p is None or reason not in ("surfaced", "pain") or not p["in_water"]:
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
        self.tool_sets: list[tuple[str, ...]] = []  # every distinct available-tool set the loop offered (stamped)
        self.water_only = False
        # selects skipped while armed water-only and the observation read dry: before the first water read of the
        # window (the teleport landing) and after it (the surfacing)
        self.idle_dry_before = 0
        self.idle_dry_after = 0
        self._seen_water = False

    def arm(self, phase: str | None = None, *, water_only: bool = False) -> float:
        """Arm for a window. ``water_only`` (placements and training episodes): idle until the policy's own
        observation reads water, so a pick made on the shore before the teleport lands never executes in the
        water window (it would act on what it has not seen, and the learner could not credit what it caused)."""
        if hasattr(self.inner, "clear_pending"):
            self.inner.clear_pending()
        if hasattr(self.inner, "phase"):
            self.inner.phase = phase
        self.water_only = water_only
        self.idle_dry_before = self.idle_dry_after = 0
        self._seen_water = False
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
        tools = tuple(sorted(available_tools))
        if tools not in self.tool_sets:
            self.tool_sets.append(tools)
        missing = [k for k in ("is_in_water", "oxygen") if k not in observation]
        if missing and not any("observation lacks" in v for v in self.violations):
            # the learner's cost, its RAW key and the water gate all read these; a default would read as "no cost"
            self.violations.append(f"observation lacks {missing}")
        if not self._armed.is_set():
            return None
        if in_water(observation):
            self._seen_water = True
        elif self.water_only:
            if self._seen_water:
                self.idle_dry_after += 1
            else:
                self.idle_dry_before += 1
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
        if pick is not None and pick.get("tool_name") not in available_tools:
            # the hook would raise SubstratePolicyError and end the loop; recorded instead, so a loop that ends early
            # without a recorded violation is the instrument's
            self.violations.append(f"pick {pick.get('tool_name')!r} is not among this tick's available tools")
            return None
        return pick
