"""EscalationLearningBridge - Learns when to escalate based on outcomes.

Connects: Hippocampus <-> SCN <-> NAc <-> ThalamicGate

Problem: ThalamicGate uses fixed thresholds (novelty=0.7, salience=0.6).
No learning from escalation outcomes.

Solution: Learn optimal thresholds per goal type and temporal context.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from maxim.decisions.nac import NAc
    from maxim.memory.hippocampus import Hippocampus
    from maxim.time.scn import SCN

logger = logging.getLogger(__name__)

# Default persistence path
DEFAULT_ESCALATION_PERSIST_PATH = ""  # resolved via __post_init__


@dataclass
class EscalationRecord:
    """Record of escalation decisions and outcomes."""

    pattern_hash: str
    goal_type: str
    hour_bin: int
    escalated: bool
    outcome_positive: bool
    timestamp: float
    threshold_at_decision: float


@dataclass
class LearnedThreshold:
    """Learned threshold for a goal type / temporal context."""

    goal_type: str
    hour_bin: int  # -1 means all hours
    base_threshold: float
    adjustment: float  # Learned adjustment (-0.3 to +0.3)
    samples: int
    successes: int
    last_updated: float


@dataclass
class EscalationLearningBridge:
    """Learns when to escalate based on historical outcomes.

    Escalation decisions determine when the robot asks for human help
    vs. acting autonomously. This bridge learns from outcomes to
    optimize the threshold between escalation and autonomous action.

    Features:
    - Per-goal thresholds: Different thresholds for different goal types
    - Temporal adjustment: Different thresholds at different times
    - Outcome learning: Lower thresholds after escalation helped, raise after unnecessary

    Example:
        bridge = EscalationLearningBridge(
            hippocampus=hippocampus,
            scn=scn,
            nac=nac,
        )

        # Get learned threshold for current context
        threshold = bridge.get_threshold(
            goal="find mug",
            novelty=0.6,
            salience=0.7,
        )

        # Record escalation outcome
        bridge.record_outcome(
            goal="find mug",
            escalated=True,
            success=True,  # Human help was useful
        )
    """

    hippocampus: "Hippocampus"
    scn: "SCN"
    nac: "NAc"

    # Learned thresholds: (goal_type, hour_bin) -> LearnedThreshold
    _thresholds: dict[tuple[str, int], LearnedThreshold] = field(default_factory=dict)

    # Recent escalation records for learning
    _recent_records: list[EscalationRecord] = field(default_factory=list)
    _max_records: int = 1000

    # Configuration
    default_novelty_threshold: float = 0.7
    default_salience_threshold: float = 0.6
    learning_rate: float = 0.1
    min_samples_for_adjustment: int = 5
    max_adjustment: float = 0.3

    # Persistence
    persist_path: str = ""  # resolved via maxim.utils.paths at runtime
    auto_save_interval: float = 60.0  # Save every 60 seconds

    # Health tracking
    _healthy: bool = True
    _error_count: int = 0
    _max_errors: int = 5
    _last_save_time: float = 0.0

    def __post_init__(self) -> None:
        """Initialize default factory fields."""
        if not self.persist_path:
            from maxim.utils.paths import resolve_user_state
            self.persist_path = str(resolve_user_state("util/escalation_learning.json"))
        if not hasattr(self, "_thresholds") or self._thresholds is None:
            self._thresholds = {}
        if not hasattr(self, "_recent_records") or self._recent_records is None:
            self._recent_records = []
        self._last_save_time = time.time()

        # Auto-load on init if path exists
        if self.persist_path and os.path.exists(self.persist_path):
            self.load(self.persist_path)

    # ─────────────────────────────────────────────────────────────────────────
    # Session Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def on_session_start(self) -> int:
        """Load learned thresholds from persistence.

        Returns:
            Number of threshold patterns loaded
        """
        try:
            loaded = self.load(self.persist_path)
            logger.info("EscalationLearningBridge loaded %d threshold patterns", loaded)
            return loaded

        except Exception as e:
            self._record_error(e)
            return 0

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str | None = None) -> bool:
        """Save learned thresholds to disk.

        Args:
            path: Path to save to. Uses persist_path if None.

        Returns:
            True if save succeeded.
        """
        save_path = path or self.persist_path
        if not save_path:
            return False

        try:
            # Ensure directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            # Serialize thresholds
            thresholds_data = {}
            for (goal_type, hour_bin), threshold in self._thresholds.items():
                key = f"{goal_type}:{hour_bin}"
                thresholds_data[key] = {
                    "goal_type": threshold.goal_type,
                    "hour_bin": threshold.hour_bin,
                    "base_threshold": threshold.base_threshold,
                    "adjustment": threshold.adjustment,
                    "samples": threshold.samples,
                    "successes": threshold.successes,
                    "last_updated": threshold.last_updated,
                }

            # Serialize recent records (keep last 100 for persistence)
            records_data = []
            for record in self._recent_records[-100:]:
                records_data.append(
                    {
                        "pattern_hash": record.pattern_hash,
                        "goal_type": record.goal_type,
                        "hour_bin": record.hour_bin,
                        "escalated": record.escalated,
                        "outcome_positive": record.outcome_positive,
                        "timestamp": record.timestamp,
                        "threshold_at_decision": record.threshold_at_decision,
                    }
                )

            data = {
                "version": 1,
                "saved_at": time.time(),
                "thresholds": thresholds_data,
                "records": records_data,
                "config": {
                    "default_novelty_threshold": self.default_novelty_threshold,
                    "default_salience_threshold": self.default_salience_threshold,
                    "learning_rate": self.learning_rate,
                },
            }

            with open(save_path, "w") as f:
                json.dump(data, f, indent=2)

            self._last_save_time = time.time()
            logger.debug("EscalationLearningBridge saved to %s", save_path)
            return True

        except Exception as e:
            logger.warning("Failed to save EscalationLearningBridge: %s", e)
            return False

    def load(self, path: str | None = None) -> int:
        """Load learned thresholds from disk.

        Args:
            path: Path to load from. Uses persist_path if None.

        Returns:
            Number of threshold patterns loaded.
        """
        load_path = path or self.persist_path
        if not load_path or not os.path.exists(load_path):
            return 0

        try:
            with open(load_path) as f:
                data = json.load(f)

            # Load thresholds
            thresholds_data = data.get("thresholds", {})
            for key, thresh_data in thresholds_data.items():
                parts = key.split(":", 1)
                if len(parts) == 2:
                    goal_type = parts[0]
                    hour_bin = int(parts[1])
                    self._thresholds[(goal_type, hour_bin)] = LearnedThreshold(
                        goal_type=thresh_data.get("goal_type", goal_type),
                        hour_bin=thresh_data.get("hour_bin", hour_bin),
                        base_threshold=thresh_data.get("base_threshold", 0.65),
                        adjustment=thresh_data.get("adjustment", 0.0),
                        samples=thresh_data.get("samples", 0),
                        successes=thresh_data.get("successes", 0),
                        last_updated=thresh_data.get("last_updated", 0.0),
                    )

            # Load recent records
            records_data = data.get("records", [])
            for rec_data in records_data:
                self._recent_records.append(
                    EscalationRecord(
                        pattern_hash=rec_data.get("pattern_hash", ""),
                        goal_type=rec_data.get("goal_type", "unknown"),
                        hour_bin=rec_data.get("hour_bin", -1),
                        escalated=rec_data.get("escalated", False),
                        outcome_positive=rec_data.get("outcome_positive", False),
                        timestamp=rec_data.get("timestamp", 0.0),
                        threshold_at_decision=rec_data.get("threshold_at_decision", 0.5),
                    )
                )

            logger.info(
                "Loaded %d escalation thresholds, %d records from %s",
                len(self._thresholds),
                len(self._recent_records),
                load_path,
            )
            return len(self._thresholds)

        except Exception as e:
            logger.warning("Failed to load EscalationLearningBridge: %s", e)
            return 0

    def _maybe_auto_save(self) -> None:
        """Auto-save if enough time has passed."""
        if not self.persist_path:
            return
        now = time.time()
        if now - self._last_save_time >= self.auto_save_interval:
            self.save()

    # ─────────────────────────────────────────────────────────────────────────
    # Threshold Queries
    # ─────────────────────────────────────────────────────────────────────────

    def get_threshold(
        self,
        goal: str | None = None,
        novelty: float = 0.5,
        salience: float = 0.5,
        seed_memory_ids: list[str] | None = None,
    ) -> float:
        """Get learned threshold for current context.

        The threshold represents the combined novelty+salience level
        above which we should escalate to human.

        When seed_memory_ids are provided, also queries the associative
        graph for related memories to inform the threshold with broader
        contextual history.

        Args:
            goal: Current goal (used to determine goal type)
            novelty: Current novelty level
            salience: Current salience level
            seed_memory_ids: Optional memory IDs to query associations from

        Returns:
            Threshold value (0-1). Escalate if novelty+salience > threshold
        """
        if not self._healthy:
            return self._get_default_threshold(novelty, salience)

        try:
            # Determine goal type
            goal_type = self._classify_goal(goal) if goal else "unknown"

            # Get current hour bin from SCN
            hour_bin = -1
            if self.scn:
                from maxim.time.temporal_signature import TemporalSignature

                sig = TemporalSignature.now()
                hour_bin = int(sig.circadian_phase * 24) % 24

            # Look up learned threshold
            key = (goal_type, hour_bin)
            if key in self._thresholds:
                threshold = self._thresholds[key]
                base = (self.default_novelty_threshold + self.default_salience_threshold) / 2
                result = base + threshold.adjustment
            elif (goal_type, -1) in self._thresholds:
                # Try goal-type without hour specificity
                threshold = self._thresholds[(goal_type, -1)]
                base = (self.default_novelty_threshold + self.default_salience_threshold) / 2
                temporal_adj = self._get_temporal_adjustment(hour_bin)
                result = base + threshold.adjustment * temporal_adj
            else:
                result = self._get_default_threshold(novelty, salience)

            # Enrich with associative graph context if seeds provided
            if seed_memory_ids and self.hippocampus:
                try:
                    associated = self.hippocampus.recall_associated(seed_memory_ids, limit=10)
                    # If associated memories show high failure rates, lower threshold
                    # (escalate more readily in contexts with historically bad outcomes)
                    successes = 0
                    failures = 0
                    for mem, _score in associated:
                        success_val = (
                            mem.success
                            if hasattr(mem, "success")
                            else mem.outcome.success
                            if hasattr(mem, "outcome")
                            else None
                        )
                        if success_val is True:
                            successes += 1
                        elif success_val is False:
                            failures += 1

                    if successes + failures >= 3:
                        failure_rate = failures / (successes + failures)
                        # High failure rate in associated memories -> lower threshold
                        if failure_rate > 0.5:
                            result -= 0.05 * (failure_rate - 0.5) * 2
                        # High success rate -> slightly raise threshold
                        elif failure_rate < 0.2:
                            result += 0.02
                except Exception:
                    pass  # Associative recall is best-effort

            return result

        except Exception as e:
            self._record_error(e)
            return self._get_default_threshold(novelty, salience)

    def should_escalate(
        self,
        goal: str | None = None,
        novelty: float = 0.5,
        salience: float = 0.5,
    ) -> tuple[bool, str]:
        """Determine if we should escalate to human.

        Args:
            goal: Current goal
            novelty: Current novelty level
            salience: Current salience level

        Returns:
            (should_escalate, reason) tuple
        """
        try:
            threshold = self.get_threshold(goal, novelty, salience)
            combined = (novelty + salience) / 2

            if combined > threshold:
                return True, f"combined_score={combined:.2f} > threshold={threshold:.2f}"
            else:
                return False, f"combined_score={combined:.2f} <= threshold={threshold:.2f}"

        except Exception as e:
            self._record_error(e)
            # Default to escalating on error (safer)
            return True, f"error: {e}"

    def _get_default_threshold(self, novelty: float, salience: float) -> float:
        """Get default threshold without learned adjustments."""
        return (self.default_novelty_threshold + self.default_salience_threshold) / 2

    def _get_temporal_adjustment(self, hour_bin: int) -> float:
        """Get temporal adjustment factor based on hour.

        Some hours may have more/less reliable escalation patterns.
        """
        if not self.scn:
            return 1.0

        # Query SCN for memory count at this hour
        # Fewer memories = less confidence = higher threshold
        try:
            hour_memories = self.scn.query_hour(hour_bin)
            count = len(hour_memories)

            if count < 5:
                return 1.2  # Raise threshold when less data
            elif count > 50:
                return 0.9  # Lower threshold when more data
            return 1.0

        except Exception:
            return 1.0

    # ─────────────────────────────────────────────────────────────────────────
    # Outcome Recording
    # ─────────────────────────────────────────────────────────────────────────

    def record_outcome(
        self,
        goal: str | None,
        escalated: bool,
        success: bool,
        novelty: float = 0.5,
        salience: float = 0.5,
    ) -> None:
        """Record the outcome of an escalation decision.

        Learning rules:
        - Escalated + success: Escalation was helpful -> lower threshold
        - Escalated + failure: Escalation didn't help -> raise threshold
        - Not escalated + success: Autonomous action worked -> raise threshold
        - Not escalated + failure: Should have escalated -> lower threshold

        Args:
            goal: Goal that was being pursued
            escalated: Whether we escalated to human
            success: Whether the overall outcome was successful
            novelty: Novelty level at decision time
            salience: Salience level at decision time
        """
        if not self._healthy:
            return

        try:
            goal_type = self._classify_goal(goal) if goal else "unknown"

            # Get hour bin
            hour_bin = -1
            if self.scn:
                from maxim.time.temporal_signature import TemporalSignature

                sig = TemporalSignature.now()
                hour_bin = int(sig.circadian_phase * 24) % 24

            # Compute pattern hash
            pattern_hash = hashlib.sha256(f"{goal_type}:{hour_bin}".encode()).hexdigest()[:12]

            # Record the outcome
            record = EscalationRecord(
                pattern_hash=pattern_hash,
                goal_type=goal_type,
                hour_bin=hour_bin,
                escalated=escalated,
                outcome_positive=success,
                timestamp=time.time(),
                threshold_at_decision=(novelty + salience) / 2,
            )

            self._recent_records.append(record)
            if len(self._recent_records) > self._max_records:
                self._recent_records = self._recent_records[-self._max_records :]

            # Update learned threshold
            self._update_threshold(record)

            # Auto-save periodically
            self._maybe_auto_save()

        except Exception as e:
            self._record_error(e)

    def _update_threshold(self, record: EscalationRecord) -> None:
        """Update learned threshold based on outcome."""
        key = (record.goal_type, record.hour_bin)

        # Get or create threshold entry
        if key not in self._thresholds:
            self._thresholds[key] = LearnedThreshold(
                goal_type=record.goal_type,
                hour_bin=record.hour_bin,
                base_threshold=self._get_default_threshold(0.5, 0.5),
                adjustment=0.0,
                samples=0,
                successes=0,
                last_updated=time.time(),
            )

        threshold = self._thresholds[key]
        threshold.samples += 1
        if record.outcome_positive:
            threshold.successes += 1
        threshold.last_updated = time.time()

        # Only adjust after sufficient samples
        if threshold.samples < self.min_samples_for_adjustment:
            return

        # Learning rule
        delta = 0.0
        if record.escalated and record.outcome_positive:
            # Escalation helped -> lower threshold (escalate more)
            delta = -self.learning_rate
        elif record.escalated and not record.outcome_positive:
            # Escalation didn't help -> raise threshold (escalate less)
            delta = self.learning_rate
        elif not record.escalated and record.outcome_positive:
            # Autonomous action worked -> raise threshold (escalate less)
            delta = self.learning_rate * 0.5  # Smaller adjustment
        else:
            # Autonomous action failed -> lower threshold (escalate more)
            delta = -self.learning_rate * 1.5  # Larger adjustment (safety)

        # Apply bounded adjustment
        threshold.adjustment = max(
            -self.max_adjustment,
            min(self.max_adjustment, threshold.adjustment + delta),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _classify_goal(self, goal: str) -> str:
        """Classify a goal into a type for threshold lookup.

        Simple keyword-based classification.
        """
        if not goal:
            return "unknown"

        goal_lower = goal.lower()

        # Movement/navigation goals
        if any(w in goal_lower for w in ["go to", "move to", "navigate", "walk"]):
            return "navigation"

        # Object manipulation
        if any(w in goal_lower for w in ["pick", "grasp", "grab", "put", "place"]):
            return "manipulation"

        # Search/find goals
        if any(w in goal_lower for w in ["find", "search", "look for", "locate"]):
            return "search"

        # Communication goals
        if any(w in goal_lower for w in ["say", "tell", "speak", "respond"]):
            return "communication"

        # Observation goals
        if any(w in goal_lower for w in ["look", "observe", "watch", "see"]):
            return "observation"

        return "general"

    def _record_error(self, error: Exception) -> None:
        """Record an error and potentially disable the bridge."""
        self._error_count += 1
        logger.warning(
            "EscalationLearningBridge error (%d/%d): %s",
            self._error_count,
            self._max_errors,
            error,
        )

        if self._error_count >= self._max_errors:
            self._healthy = False
            logger.error("EscalationLearningBridge disabled after %d errors", self._error_count)

    @property
    def is_healthy(self) -> bool:
        """Check if bridge is operational."""
        return self._healthy

    def stats(self) -> dict[str, Any]:
        """Return bridge statistics."""
        total_samples = sum(t.samples for t in self._thresholds.values())
        total_successes = sum(t.successes for t in self._thresholds.values())

        return {
            "healthy": self._healthy,
            "error_count": self._error_count,
            "learned_thresholds": len(self._thresholds),
            "total_samples": total_samples,
            "overall_success_rate": (total_successes / total_samples if total_samples > 0 else 0.5),
            "recent_records": len(self._recent_records),
        }


__all__ = ["EscalationLearningBridge", "EscalationRecord", "LearnedThreshold"]
