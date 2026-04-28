"""Significance heuristics for short-term memory staging.

Determines which moments are worth encoding into short-term memory
based on NAc RPE magnitude, user interaction, novelty, and other
signals. Heuristic weights learn from long-term utility feedback.

Phase 3a of the consolidation expansion plan.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


def sigmoid(x: float) -> float:
    """Standard sigmoid, clamped to avoid overflow."""
    x = max(-10.0, min(10.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient. Returns 0.0 if degenerate."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    sx = statistics.stdev(xs)
    sy = statistics.stdev(ys)
    if sx < 1e-9 or sy < 1e-9:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / ((n - 1) * sx * sy)


@dataclass
class CycleContext:
    """Context available at the end of each agent cycle for significance scoring."""

    rpe_raw: float = 0.0  # |predicted - actual| from NAc
    has_user_input: bool = False  # CLI or transcript present
    novelty: float = 0.5  # Perception novelty score
    is_plan_boundary: bool = False  # Phase start/complete/fail
    energy_crossed_threshold: bool = False  # Crossed low/critical
    outcome_valence: float = 0.5  # 0=bad, 1=good, 0.5=neutral


@dataclass
class SignificanceHeuristic:
    """A single heuristic for evaluating moment significance."""

    name: str
    weight: float  # Contribution to final score
    init_range: tuple[float, float] = (0.1, 0.2)  # For random init
    evaluate: Callable[[CycleContext], float] | None = None  # Returns 0-1


# ── Baseline heuristic evaluation functions ──────────────────────────────────


def _eval_rpe(ctx: CycleContext) -> float:
    """RPE magnitude — normalized externally by SignificanceWeightLearner."""
    return ctx.rpe_raw  # Already normalized to 0-1 by caller


def _eval_user_interaction(ctx: CycleContext) -> float:
    return 1.0 if ctx.has_user_input else 0.0


def _eval_novelty(ctx: CycleContext) -> float:
    return ctx.novelty


def _eval_plan_boundary(ctx: CycleContext) -> float:
    return 1.0 if ctx.is_plan_boundary else 0.0


def _eval_energy_change(ctx: CycleContext) -> float:
    return 1.0 if ctx.energy_crossed_threshold else 0.0


def _eval_valence_extremity(ctx: CycleContext) -> float:
    return abs(ctx.outcome_valence - 0.5) * 2.0


BASELINE_HEURISTICS = [
    SignificanceHeuristic("rpe_magnitude", 0.35, (0.30, 0.50), _eval_rpe),
    SignificanceHeuristic("user_interaction", 0.20, (0.15, 0.25), _eval_user_interaction),
    SignificanceHeuristic("novelty", 0.15, (0.10, 0.20), _eval_novelty),
    SignificanceHeuristic("plan_phase_boundary", 0.10, (0.05, 0.15), _eval_plan_boundary),
    SignificanceHeuristic("energy_state_change", 0.05, (0.02, 0.08), _eval_energy_change),
    SignificanceHeuristic("outcome_valence_extremity", 0.10, (0.05, 0.15), _eval_valence_extremity),
]


@dataclass(frozen=True)
class SignificanceConfig:
    """Configuration for the significance evaluation system."""

    heuristics: list[SignificanceHeuristic] = field(default_factory=lambda: list(BASELINE_HEURISTICS))
    staging_threshold: float = 0.5  # Stage if weighted score exceeds this
    rpe_window: int = 100  # Rolling window for RPE normalization


# ── Learnable weights ────────────────────────────────────────────────────────


@dataclass
class LearnableWeight:
    """A heuristic weight that learns from memory utility."""

    name: str
    weight: float  # Current learned weight
    init_range: tuple[float, float]  # Confidence interval for random init
    update_count: int = 0  # How many updates applied
    utility_history: deque = field(default_factory=lambda: deque(maxlen=50))
    weight_snapshots: deque = field(default_factory=lambda: deque(maxlen=50))


@dataclass
class PromotionRecord:
    """Tracks a promoted memory for graph-based utility measurement."""

    memory_id: str
    heuristic_scores: dict[str, float]  # Frozen at staging time
    promoted_at: float  # Timestamp of promotion
    edges_at_promotion: int  # Associative edge count at promotion time


class SignificanceWeightLearner:
    """Manages persistent learnable weights for significance heuristics.

    Learning signal: associative graph integration, NOT consolidation.
    A memory is "useful" if it becomes well-connected in the hippocampal
    associative graph — meaning it keeps co-activating with other memories
    during recall, forming new edges.
    """

    def __init__(self, weights_path: str, hippocampus: Any) -> None:
        self.weights_path = weights_path
        self.hippocampus = hippocampus
        self.weights: dict[str, LearnableWeight] = {}
        self.rpe_running_mean: float = 0.0
        self.rpe_running_std: float = 1.0
        self.rpe_window: deque[float] = deque(maxlen=100)
        self.tracked_memories: dict[str, PromotionRecord] = {}
        self._load_or_init()

    def _load_or_init(self) -> None:
        """Load persisted weights or initialize randomly."""
        path = Path(self.weights_path)
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                import logging

                from maxim.utils.format_version import check_format_version

                check_format_version(data, "significance_weights", log=logging.getLogger(__name__))
                for name, w in data.get("weights", {}).items():
                    # Handle deque fields from JSON lists
                    uh = w.pop("utility_history", [])
                    ws = w.pop("weight_snapshots", [])
                    lw = LearnableWeight(**w)
                    lw.utility_history = deque(uh, maxlen=50)
                    lw.weight_snapshots = deque(ws, maxlen=50)
                    self.weights[name] = lw
                self.rpe_running_mean = data.get("rpe_running_mean", 0.0)
                self.rpe_running_std = data.get("rpe_running_std", 1.0)
                self.rpe_window = deque(data.get("rpe_window", []), maxlen=100)
                for mid, rec in data.get("tracked_memories", {}).items():
                    self.tracked_memories[mid] = PromotionRecord(**rec)
            except (json.JSONDecodeError, TypeError, KeyError):
                self._random_init()
        else:
            self._random_init()

    def _random_init(self) -> None:
        """Initialize weights randomly within confidence intervals, then normalize."""
        for h in BASELINE_HEURISTICS:
            lo, hi = h.init_range
            raw = random.uniform(lo, hi)
            self.weights[h.name] = LearnableWeight(name=h.name, weight=raw, init_range=(lo, hi))
        self._normalize_weights()
        self.rpe_running_mean = 0.0
        self.rpe_running_std = 1.0
        self.rpe_window = deque(maxlen=100)

    def evaluate(self, ctx: CycleContext, config: SignificanceConfig) -> tuple[float, dict[str, float]]:
        """Evaluate significance of a cycle. Returns (score, heuristic_scores)."""
        # Normalize RPE
        normalized_rpe = 0.0
        if ctx.rpe_raw > 0:
            normalized_rpe = (ctx.rpe_raw - self.rpe_running_mean) / max(self.rpe_running_std, 1e-6)
            ctx = CycleContext(
                rpe_raw=sigmoid(normalized_rpe),
                has_user_input=ctx.has_user_input,
                novelty=ctx.novelty,
                is_plan_boundary=ctx.is_plan_boundary,
                energy_crossed_threshold=ctx.energy_crossed_threshold,
                outcome_valence=ctx.outcome_valence,
            )

        scores: dict[str, float] = {}
        weighted_sum = 0.0
        for h in config.heuristics:
            if h.evaluate is not None:
                raw = h.evaluate(ctx)
            else:
                raw = 0.0
            scores[h.name] = raw
            w = self.weights.get(h.name)
            weight = w.weight if w else h.weight
            weighted_sum += weight * raw

        return weighted_sum, scores

    def update_rpe_stats(self, raw_rpe: float) -> None:
        """Update running RPE mean/std from new observation."""
        self.rpe_window.append(raw_rpe)
        if len(self.rpe_window) >= 2:
            self.rpe_running_mean = statistics.mean(self.rpe_window)
            self.rpe_running_std = max(statistics.stdev(self.rpe_window), 1e-6)

    def track_promotion(self, memory_id: str, heuristic_scores: dict[str, float]) -> None:
        """Start tracking a newly promoted memory for graph-based utility."""
        edges = self.hippocampus.graph.get_associated(memory_id)
        self.tracked_memories[memory_id] = PromotionRecord(
            memory_id=memory_id,
            heuristic_scores=heuristic_scores,
            promoted_at=time.time(),
            edges_at_promotion=len(edges),
        )

    def harvest_utility(self, min_age_hours: float = 24.0) -> None:
        """Harvest utility signals from tracked memories using associative graph."""
        now = time.time()
        to_remove = []

        for mid, rec in self.tracked_memories.items():
            age_hours = (now - rec.promoted_at) / 3600
            if age_hours < min_age_hours:
                continue

            edges = self.hippocampus.graph.get_associated(mid)

            if edges or age_hours > min_age_hours * 3:
                edge_growth = len(edges) - rec.edges_at_promotion
                mean_weight = statistics.mean(w for _, w in edges) if edges else 0.0
                utility = sigmoid(edge_growth / 3.0) * 0.6 + mean_weight * 0.4

                if not edges and age_hours > min_age_hours * 3:
                    utility = 0.0

                for name, score in rec.heuristic_scores.items():
                    if name in self.weights:
                        w = self.weights[name]
                        w.utility_history.append((score, utility))
                        w.update_count += 1

                        if len(w.utility_history) >= 10:
                            scores_list = [s for s, _ in w.utility_history]
                            utilities = [u for _, u in w.utility_history]
                            correlation = _pearson(scores_list, utilities)

                            learning_rate = 0.05
                            multiplier = 1.0 + learning_rate * correlation
                            w.weight = max(0.01, w.weight * multiplier)
                            w.weight_snapshots.append(w.weight)

                to_remove.append(mid)

        for mid in to_remove:
            del self.tracked_memories[mid]

        if to_remove:
            self._normalize_weights()

    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0."""
        total = sum(w.weight for w in self.weights.values())
        if total > 0:
            for w in self.weights.values():
                w.weight /= total

    def detect_stuck(self, window: int = 30, threshold: float = 0.05) -> bool:
        """Detect if learning is stuck by comparing weight snapshots."""
        has_enough = any(len(w.weight_snapshots) >= window for w in self.weights.values())
        if not has_enough:
            return False

        max_delta = 0.0
        for w in self.weights.values():
            if len(w.weight_snapshots) < window:
                continue
            old_weight = w.weight_snapshots[-window]
            delta = abs(w.weight - old_weight)
            max_delta = max(max_delta, delta)

        return max_delta < threshold

    def restart_from_scratch(self) -> None:
        """Re-initialize all weights randomly. Called when stuck."""
        self._random_init()

    def on_memory_deleted(self, memory_id: str) -> None:
        """Callback from hippocampus — clean up tracked memory."""
        self.tracked_memories.pop(memory_id, None)

    def load_weights_into_heuristics(self, heuristics: list[SignificanceHeuristic]) -> None:
        """Apply learned weights to heuristic list."""
        for h in heuristics:
            if h.name in self.weights:
                h.weight = self.weights[h.name].weight

    def save(self) -> None:
        """Persist weights + RPE stats + tracked memories to disk."""
        data = {
            "version": "2.0",
            "weights": {},
            "rpe_running_mean": self.rpe_running_mean,
            "rpe_running_std": self.rpe_running_std,
            "rpe_window": list(self.rpe_window),
            "tracked_memories": {mid: asdict(rec) for mid, rec in self.tracked_memories.items()},
        }
        # Serialize LearnableWeight with deque → list conversion
        for name, w in self.weights.items():
            d = asdict(w)
            d["utility_history"] = list(w.utility_history)
            d["weight_snapshots"] = list(w.weight_snapshots)
            data["weights"][name] = d

        from maxim.utils.atomic_io import atomic_write_json
        from maxim.utils.format_version import with_format_version

        atomic_write_json(self.weights_path, with_format_version(data))


__all__ = [
    "BASELINE_HEURISTICS",
    "CycleContext",
    "LearnableWeight",
    "PromotionRecord",
    "SignificanceConfig",
    "SignificanceHeuristic",
    "SignificanceWeightLearner",
    "sigmoid",
]
