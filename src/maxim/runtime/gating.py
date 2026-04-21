"""Shared gating abstractions — scoring protocols + adaptive thresholds.

Extracted from ``default_network/gate.py`` (G0) so that multiple
consumers (ThalamicGate, BioEnrichmentPipeline, ImaginationTrigger)
can share scoring protocols and threshold adaptation without depending
on the Default Network.

Architecture:
- ``SalienceScorer`` protocol: modality-specific scoring (Vision, Text, Audio)
- ``AdaptiveThresholdController``: learns optimal thresholds from outcomes
- ``GateScore`` / ``GateDecision``: uniform data types for logging/provenance
- ``GatingContext``: shared context for scoring (goal, energy, load)

Each gating site composes a scorer + controller with its own decision
logic.  The protocol is the shared interface; the decision structure
remains site-specific (cascade, budget, refractory, threshold).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GateScore:
    """Result of scoring an input for novelty and salience.

    All fields are 0.0–1.0.  ``combined`` is typically
    ``novelty * salience`` but scorers may use a different formula.
    """

    novelty: float
    salience: float
    goal_relevance: float = 0.0
    combined: float = 0.0


@dataclass(frozen=True, slots=True)
class GateDecision:
    """Uniform gating decision record for logging and provenance.

    Every gating site produces this so debugging/tracing is consistent
    across modalities.
    """

    passed: bool
    score: GateScore
    reason: str = ""
    threshold_used: float = 0.0


@dataclass(frozen=True, slots=True)
class GatingContext:
    """Shared context available to all scorers.

    Carries goal, energy, processing load — information that influences
    gating decisions across all modalities.
    """

    active_goal: str | None = None
    goal_keywords: tuple[str, ...] = ()
    energy: float = 1.0  # 0=exhausted, 1=fully energized (from SCN)
    processing_load: int = 0  # LLM queue depth or similar
    recent_input_count: int = 0  # inputs in last window


# ---------------------------------------------------------------------------
# Scoring protocol
# ---------------------------------------------------------------------------


class SalienceScorer(Protocol):
    """Score an input's novelty and salience for gating decisions.

    Each modality implements this independently:
    - Vision: track-ID novelty + COCO class salience + spatial relevance
    - Text: EC pattern separation + goal-keyword overlap + NAc familiarity
    - Audio: volume spike + speech detection + directional novelty
    - Pain: intensity as salience, refractory recency as inverse novelty

    Implementations may be stateful (caching recent inputs for novelty
    computation) but must be thread-safe.
    """

    def score(self, input: Any, context: GatingContext) -> GateScore:
        """Score an input. Returns GateScore with novelty/salience/combined."""
        ...


# ---------------------------------------------------------------------------
# AdaptiveThresholdController (extracted from default_network/gate.py)
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveThresholdConfig:
    """Configuration for adaptive threshold adjustment.

    Parameters tune how aggressively the controller adapts and what
    escalation rate it targets.  Persistence is optional — omit
    ``persist_path`` for ephemeral (per-session) thresholds.
    """

    base_novelty_threshold: float = 0.7
    base_salience_threshold: float = 0.6
    min_threshold: float = 0.3
    max_threshold: float = 0.95
    adaptation_rate: float = 0.1
    escalation_rate_target: float = 0.05  # Target 5% escalation rate
    window_seconds: float = 60.0
    adaptation_interval: float = 5.0  # How often to adapt (seconds)
    # Persistence (optional — set to "" for ephemeral)
    persist_path: str = ""
    auto_save_interval: float = 60.0


class AdaptiveThresholdController:
    """Dynamically adjusts escalation thresholds based on feedback.

    Tracks escalation history and outcomes to tune thresholds:
    - Too many escalations → raise threshold
    - Too few escalations → lower threshold
    - Escalations leading to actions → good, maintain
    - Escalations being ignored → bad, raise threshold
    - High processing load → raise threshold (reduce LLM pressure)
    - High urgency/fear → lower threshold (be more vigilant)

    Thread-safe via internal lock.  Each gating site owns its own
    instance with domain-appropriate parameters.
    """

    def __init__(self, config: AdaptiveThresholdConfig | None = None) -> None:
        self.config = config or AdaptiveThresholdConfig()
        self._novelty_threshold = self.config.base_novelty_threshold
        self._salience_threshold = self.config.base_salience_threshold

        # History tracking
        self._escalation_history: deque[tuple[float, bool]] = deque(maxlen=1000)
        self._outcome_history: deque[tuple[float, str]] = deque(maxlen=100)
        self._last_adaptation = time.time()
        self._last_save_time = time.time()
        self._lock = threading.Lock()

        # Auto-load on init if path exists
        if self.config.persist_path and os.path.exists(self.config.persist_path):
            self.load(self.config.persist_path)

    @property
    def novelty_threshold(self) -> float:
        """Current novelty threshold."""
        return self._novelty_threshold

    @property
    def salience_threshold(self) -> float:
        """Current salience threshold."""
        return self._salience_threshold

    @property
    def combined_threshold(self) -> float:
        """Combined novelty * salience threshold for escalation."""
        return self._novelty_threshold * self._salience_threshold

    def record_escalation(self, escalated: bool) -> None:
        """Record whether an input passed or was filtered."""
        with self._lock:
            self._escalation_history.append((time.time(), escalated))

    def record_outcome(self, outcome: str) -> None:
        """Record outcome of a passed input.

        Args:
            outcome: One of 'action_taken', 'ignored', 'goal_created'.
        """
        with self._lock:
            self._outcome_history.append((time.time(), outcome))

    def maybe_adapt(
        self,
        processing_load: int = 0,
        urgency: float = 0.0,
    ) -> bool:
        """Adjust thresholds if enough time has passed.

        Args:
            processing_load: Current load (LLM queue depth, enrichment backlog).
            urgency: Current urgency/fear level (0-1). Higher → lower threshold.

        Returns:
            True if adaptation was performed.
        """
        now = time.time()
        if now - self._last_adaptation < self.config.adaptation_interval:
            return False

        with self._lock:
            self._last_adaptation = now
            self._adapt(processing_load, urgency)

        # Auto-save periodically
        self._maybe_auto_save()
        return True

    def _adapt(self, processing_load: int, urgency: float) -> None:
        """Internal adaptation logic (called with lock held)."""
        now = time.time()
        window_start = now - self.config.window_seconds

        # Calculate recent escalation rate
        recent = [(t, e) for t, e in self._escalation_history if t > window_start]
        if len(recent) < 10:
            return  # Not enough data

        escalation_count = sum(1 for _, e in recent if e)
        escalation_rate = escalation_count / len(recent)

        # Calculate outcome quality (what % of escalations led to actions)
        recent_outcomes = [o for t, o in self._outcome_history if t > window_start]
        if recent_outcomes:
            useful_outcomes = sum(1 for o in recent_outcomes if o in ("action_taken", "goal_created"))
            outcome_quality = useful_outcomes / len(recent_outcomes)
        else:
            outcome_quality = 0.5  # Neutral if no data

        # Determine adjustment direction
        target_rate = self.config.escalation_rate_target
        rate_error = escalation_rate - target_rate

        adjustment = 0.0

        # Rate too high → raise threshold
        if rate_error > 0.02:
            adjustment += self.config.adaptation_rate * rate_error * 2

        # Rate too low → lower threshold
        elif rate_error < -0.02:
            adjustment += self.config.adaptation_rate * rate_error * 2

        # Outcomes being ignored → raise threshold
        if outcome_quality < 0.3 and recent_outcomes:
            adjustment += self.config.adaptation_rate * 0.5

        # High processing load → raise threshold (reduce pressure)
        if processing_load > 3:
            adjustment += self.config.adaptation_rate * 0.3

        # High urgency → lower threshold (be more vigilant)
        if urgency > 0.5:
            adjustment -= self.config.adaptation_rate * urgency * 0.5

        # Apply adjustment to both thresholds
        self._novelty_threshold = max(
            self.config.min_threshold,
            min(self.config.max_threshold, self._novelty_threshold + adjustment),
        )
        self._salience_threshold = max(
            self.config.min_threshold,
            min(self.config.max_threshold, self._salience_threshold + adjustment),
        )

        logger.debug(
            "Adapted thresholds: novelty=%.3f, salience=%.3f (rate=%.3f, quality=%.3f)",
            self._novelty_threshold,
            self._salience_threshold,
            escalation_rate,
            outcome_quality,
        )

    # ──────��──────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────��─────────────────────────────���────────────────────────

    def save(self, path: str | None = None) -> bool:
        """Save learned thresholds to disk.

        Returns True if save succeeded.
        """
        save_path = path or self.config.persist_path
        if not save_path:
            return False

        try:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            with self._lock:
                escalation_data = list(self._escalation_history)[-200:]
                outcome_data = list(self._outcome_history)[-100:]

                data = {
                    "version": 1,
                    "saved_at": time.time(),
                    "thresholds": {
                        "novelty": self._novelty_threshold,
                        "salience": self._salience_threshold,
                    },
                    "escalation_history": escalation_data,
                    "outcome_history": outcome_data,
                    "config": {
                        "base_novelty_threshold": self.config.base_novelty_threshold,
                        "base_salience_threshold": self.config.base_salience_threshold,
                    },
                }

            with open(save_path, "w") as f:
                json.dump(data, f, indent=2)

            self._last_save_time = time.time()
            logger.debug("AdaptiveThresholdController saved to %s", save_path)
            return True

        except Exception as e:
            logger.warning("Failed to save AdaptiveThresholdController: %s", e)
            return False

    def load(self, path: str | None = None) -> bool:
        """Load learned thresholds from disk.

        Returns True if load succeeded.
        """
        load_path = path or self.config.persist_path
        if not load_path or not os.path.exists(load_path):
            return False

        try:
            with open(load_path) as f:
                data = json.load(f)

            with self._lock:
                thresholds = data.get("thresholds", {})
                self._novelty_threshold = thresholds.get("novelty", self.config.base_novelty_threshold)
                self._salience_threshold = thresholds.get("salience", self.config.base_salience_threshold)

                for ts, escalated in data.get("escalation_history", []):
                    self._escalation_history.append((ts, escalated))
                for ts, outcome in data.get("outcome_history", []):
                    self._outcome_history.append((ts, outcome))

            logger.info(
                "Loaded adaptive thresholds (novelty=%.3f, salience=%.3f) from %s",
                self._novelty_threshold,
                self._salience_threshold,
                load_path,
            )
            return True

        except Exception as e:
            logger.warning("Failed to load AdaptiveThresholdController: %s", e)
            return False

    def _maybe_auto_save(self) -> None:
        """Auto-save if enough time has passed."""
        if not self.config.persist_path:
            return
        now = time.time()
        if now - self._last_save_time >= self.config.auto_save_interval:
            self.save()

    def reset(self) -> None:
        """Reset thresholds to base values."""
        with self._lock:
            self._novelty_threshold = self.config.base_novelty_threshold
            self._salience_threshold = self.config.base_salience_threshold
            self._escalation_history.clear()
            self._outcome_history.clear()


# ---------------------------------------------------------------------------
# TextSalienceScorer (G1) — text-specific scoring for bio-enrichment
# ---------------------------------------------------------------------------


class TextSalienceScorer:
    """Score text inputs for novelty and salience.

    Novelty: How different is this text from recent inputs? Uses keyword
    overlap with a sliding window of recent texts as a lightweight proxy.
    When EC is available, uses pattern_separate_or_complete for true
    semantic novelty.

    Salience: How relevant is this text to the current goal? Uses keyword
    overlap with goal keywords + NAc reward history for recognized concepts.

    Thread-safe via internal lock on the recent-texts window.
    """

    def __init__(
        self,
        *,
        ec: Any = None,
        nac: Any = None,
        recent_window_size: int = 20,
    ) -> None:
        self._ec = ec
        self._nac = nac
        self._recent_texts: deque[set[str]] = deque(maxlen=recent_window_size)
        self._lock = threading.Lock()

    def score(self, input: Any, context: GatingContext) -> GateScore:
        """Score text for novelty and salience."""
        if not isinstance(input, str) or not input.strip():
            return GateScore(novelty=0.0, salience=0.0, combined=0.0)

        text = input.strip().lower()
        words = set(text.replace("_", " ").split())

        novelty = self._compute_novelty(words, text)
        salience = self._compute_salience(words, context)
        goal_relevance = self._compute_goal_relevance(words, context)
        combined = novelty * max(salience, goal_relevance)

        # Record this text in the recent window
        with self._lock:
            self._recent_texts.append(words)

        return GateScore(
            novelty=novelty,
            salience=salience,
            goal_relevance=goal_relevance,
            combined=combined,
        )

    def _compute_novelty(self, words: set[str], text: str) -> float:
        """How different is this from recent texts?

        EC path: true semantic novelty via pattern separation.
        Fallback: keyword overlap with recent window (1.0 = completely new).
        """
        # Try EC semantic novelty if available
        if self._ec is not None:
            try:
                results = self._ec.find_semantic(text, k=1, threshold=0.5)
                if results:
                    # High similarity to existing memory = low novelty
                    _, similarity = results[0]
                    return 1.0 - similarity
                return 1.0  # Nothing similar found = completely novel
            except Exception:
                pass  # Fall through to keyword-based

        # Fallback: keyword overlap with recent window
        with self._lock:
            if not self._recent_texts:
                return 1.0  # First input is always novel
            # Average overlap with recent texts
            overlaps = []
            for recent_words in self._recent_texts:
                if not recent_words:
                    continue
                overlap = len(words & recent_words) / max(len(words | recent_words), 1)
                overlaps.append(overlap)
            if not overlaps:
                return 1.0
            avg_overlap = sum(overlaps) / len(overlaps)
            return 1.0 - avg_overlap  # High overlap = low novelty

    def _compute_salience(self, words: set[str], context: GatingContext) -> float:
        """How salient is this text? NAc reward history for recognized words."""
        if self._nac is None:
            return 0.5  # Neutral if no NAc

        # Check if any words match known rewarding event signatures
        salience_signals = []
        for word in words:
            try:
                links = self._nac.get_links_for_event(word)
                if links:
                    best = max(links, key=lambda lk: lk.confidence)
                    salience_signals.append(best.confidence)
            except Exception:
                continue

        if not salience_signals:
            return 0.5
        return min(1.0, 0.5 + max(salience_signals) * 0.5)

    def _compute_goal_relevance(self, words: set[str], context: GatingContext) -> float:
        """How relevant is this text to the current goal?"""
        if not context.goal_keywords:
            return 0.0
        goal_words = set(context.goal_keywords)
        if not goal_words or not words:
            return 0.0
        overlap = len(words & goal_words)
        return min(1.0, overlap / max(len(goal_words), 1))

    def reset(self) -> None:
        """Clear recent text history."""
        with self._lock:
            self._recent_texts.clear()
