"""FearCircuitBridge - Memory-informed safety assessment.

Connects: Hippocampus <-> FearAgent <-> NAc

Problem: FearAgent uses static risk thresholds without learning from outcomes.
No memory of which actions were actually dangerous vs. false positives.

Solution: Learn risk patterns from historical outcomes:
- Track which code patterns led to actual problems
- Adjust risk thresholds based on context and history
- Remember false positive patterns to reduce unnecessary blocks
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from maxim.agents.fear_agent import DangerCategory, RiskLevel
    from maxim.decisions.nac import NAc
    from maxim.memory.hippocampus import Hippocampus
    from maxim.similarity.ec import EntorhinalCortex

logger = logging.getLogger(__name__)

# Default persistence path
DEFAULT_FEAR_PERSIST_PATH = "data/util/fear_learning.json"


@dataclass
class RiskEvent:
    """Record of a risk assessment and its outcome."""

    pattern_hash: str
    category: str  # DangerCategory value
    severity: str  # RiskLevel value
    action_type: str  # "code_review" or "action_review"
    was_blocked: bool
    actual_harm: bool | None  # None if unknown, True if caused harm, False if safe
    timestamp: float
    context: str = ""  # Additional context (file, agent, etc.)


@dataclass
class LearnedRiskAdjustment:
    """Learned adjustment for a risk pattern."""

    category: str
    pattern_type: str  # Type of pattern (e.g., "subprocess", "eval")
    base_severity: str
    adjustment: float  # -0.3 to +0.3
    samples: int
    false_positives: int
    true_positives: int
    last_updated: float


@dataclass
class FearCircuitBridge:
    """Memory-informed safety assessment bridge.

    Learns from historical outcomes to improve risk assessment accuracy:
    - Reduces false positives by learning safe patterns
    - Increases sensitivity to actually dangerous patterns
    - Provides context-aware risk adjustments

    Example:
        bridge = FearCircuitBridge(
            hippocampus=hippocampus,
            nac=nac,
            ec=ec,
        )

        # Get learned risk adjustment
        adjustment = bridge.get_risk_adjustment(
            category="code_execution",
            pattern="subprocess",
            context="trusted_source",
        )

        # Record outcome
        bridge.record_outcome(
            category="code_execution",
            pattern="subprocess",
            was_blocked=True,
            actual_harm=False,  # Was a false positive
        )
    """

    hippocampus: "Hippocampus"
    nac: "NAc"
    ec: "EntorhinalCortex"

    # Learned risk adjustments: (category, pattern_type) -> LearnedRiskAdjustment
    _adjustments: dict[tuple[str, str], LearnedRiskAdjustment] = field(
        default_factory=dict
    )

    # Recent risk events for learning
    _recent_events: list[RiskEvent] = field(default_factory=list)
    _max_events: int = 500

    # Category statistics
    _category_stats: dict[str, dict[str, int]] = field(default_factory=dict)

    # Configuration
    learning_rate: float = 0.05
    min_samples_for_adjustment: int = 3
    max_adjustment: float = 0.3
    false_positive_threshold: float = 0.7  # High FP rate triggers adjustment

    # Persistence
    persist_path: str = DEFAULT_FEAR_PERSIST_PATH
    auto_save_interval: float = 60.0  # Save every 60 seconds

    # Health tracking
    _healthy: bool = True
    _error_count: int = 0
    _max_errors: int = 5
    _last_save_time: float = 0.0

    def __post_init__(self) -> None:
        """Initialize default factory fields."""
        if not hasattr(self, "_adjustments") or self._adjustments is None:
            self._adjustments = {}
        if not hasattr(self, "_recent_events") or self._recent_events is None:
            self._recent_events = []
        if not hasattr(self, "_category_stats") or self._category_stats is None:
            self._category_stats = {}
        self._last_save_time = time.time()

        # Auto-load on init if path exists
        if self.persist_path and os.path.exists(self.persist_path):
            self.load(self.persist_path)

    # ─────────────────────────────────────────────────────────────────────────
    # Session Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def on_session_start(self) -> int:
        """Load learned risk adjustments from persistence.

        Returns:
            Number of adjustment patterns loaded
        """
        try:
            loaded = self.load(self.persist_path)
            logger.info(
                "FearCircuitBridge loaded %d adjustment patterns", loaded
            )
            return loaded

        except Exception as e:
            self._record_error(e)
            return 0

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str | None = None) -> bool:
        """Save learned risk adjustments to disk.

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

            # Serialize adjustments
            adjustments_data = {}
            for (category, pattern_type), adj in self._adjustments.items():
                key = f"{category}:{pattern_type}"
                adjustments_data[key] = {
                    "category": adj.category,
                    "pattern_type": adj.pattern_type,
                    "base_severity": adj.base_severity,
                    "adjustment": adj.adjustment,
                    "samples": adj.samples,
                    "false_positives": adj.false_positives,
                    "true_positives": adj.true_positives,
                    "last_updated": adj.last_updated,
                }

            # Serialize recent events (keep last 100 for persistence)
            events_data = []
            for event in self._recent_events[-100:]:
                events_data.append({
                    "pattern_hash": event.pattern_hash,
                    "category": event.category,
                    "severity": event.severity,
                    "action_type": event.action_type,
                    "was_blocked": event.was_blocked,
                    "actual_harm": event.actual_harm,
                    "timestamp": event.timestamp,
                    "context": event.context,
                })

            data = {
                "version": 1,
                "saved_at": time.time(),
                "adjustments": adjustments_data,
                "events": events_data,
                "category_stats": self._category_stats,
                "config": {
                    "learning_rate": self.learning_rate,
                    "min_samples_for_adjustment": self.min_samples_for_adjustment,
                    "max_adjustment": self.max_adjustment,
                },
            }

            with open(save_path, "w") as f:
                json.dump(data, f, indent=2)

            self._last_save_time = time.time()
            logger.debug("FearCircuitBridge saved to %s", save_path)
            return True

        except Exception as e:
            logger.warning("Failed to save FearCircuitBridge: %s", e)
            return False

    def load(self, path: str | None = None) -> int:
        """Load learned risk adjustments from disk.

        Args:
            path: Path to load from. Uses persist_path if None.

        Returns:
            Number of adjustment patterns loaded.
        """
        load_path = path or self.persist_path
        if not load_path or not os.path.exists(load_path):
            return 0

        try:
            with open(load_path) as f:
                data = json.load(f)

            # Load adjustments
            adjustments_data = data.get("adjustments", {})
            for key, adj_data in adjustments_data.items():
                parts = key.split(":", 1)
                if len(parts) == 2:
                    category = parts[0]
                    pattern_type = parts[1]
                    self._adjustments[(category, pattern_type)] = LearnedRiskAdjustment(
                        category=adj_data.get("category", category),
                        pattern_type=adj_data.get("pattern_type", pattern_type),
                        base_severity=adj_data.get("base_severity", "medium"),
                        adjustment=adj_data.get("adjustment", 0.0),
                        samples=adj_data.get("samples", 0),
                        false_positives=adj_data.get("false_positives", 0),
                        true_positives=adj_data.get("true_positives", 0),
                        last_updated=adj_data.get("last_updated", 0.0),
                    )

            # Load recent events
            events_data = data.get("events", [])
            for ev_data in events_data:
                self._recent_events.append(RiskEvent(
                    pattern_hash=ev_data.get("pattern_hash", ""),
                    category=ev_data.get("category", "unknown"),
                    severity=ev_data.get("severity", "medium"),
                    action_type=ev_data.get("action_type", "code_review"),
                    was_blocked=ev_data.get("was_blocked", False),
                    actual_harm=ev_data.get("actual_harm"),
                    timestamp=ev_data.get("timestamp", 0.0),
                    context=ev_data.get("context", ""),
                ))

            # Load category stats
            self._category_stats = data.get("category_stats", {})

            logger.info(
                "Loaded %d fear adjustments, %d events from %s",
                len(self._adjustments),
                len(self._recent_events),
                load_path,
            )
            return len(self._adjustments)

        except Exception as e:
            logger.warning("Failed to load FearCircuitBridge: %s", e)
            return 0

    def _maybe_auto_save(self) -> None:
        """Auto-save if enough time has passed."""
        if not self.persist_path:
            return
        now = time.time()
        if now - self._last_save_time >= self.auto_save_interval:
            self.save()

    # ─────────────────────────────────────────────────────────────────────────
    # Risk Adjustment Queries
    # ─────────────────────────────────────────────────────────────────────────

    def get_risk_adjustment(
        self,
        category: str,
        pattern: str,
        context: str = "",
        seed_memory_ids: list[str] | None = None,
    ) -> float:
        """Get learned risk adjustment for a pattern.

        Returns a value between -0.3 and +0.3:
        - Negative: Lower risk than default (many false positives)
        - Zero: Use default risk assessment
        - Positive: Higher risk than default (many true positives)

        When seed_memory_ids are provided, also queries the associative
        graph for related memories that involved similar risk patterns,
        enriching the adjustment with contextual history.

        Args:
            category: DangerCategory value
            pattern: Specific pattern (e.g., "subprocess", "eval")
            context: Additional context
            seed_memory_ids: Optional memory IDs to query associations from

        Returns:
            Risk adjustment (-0.3 to +0.3)
        """
        if not self._healthy:
            return 0.0

        try:
            adjustment = 0.0

            key = (category, pattern)
            if key in self._adjustments:
                adj = self._adjustments[key]
                adjustment = adj.adjustment
            else:
                # Check for category-level adjustment
                key_any = (category, "*")
                if key_any in self._adjustments:
                    adj = self._adjustments[key_any]
                    adjustment = adj.adjustment * 0.5  # Reduced weight for generic

            # Enrich with associative graph context if seeds provided
            if seed_memory_ids and self.hippocampus:
                try:
                    associated = self.hippocampus.recall_associated(
                        seed_memory_ids, limit=10
                    )
                    # Check if associated memories had risk-relevant outcomes
                    risk_signals = 0
                    safe_signals = 0
                    for mem, _score in associated:
                        success = (
                            mem.success if hasattr(mem, "success")
                            else mem.outcome.success if hasattr(mem, "outcome")
                            else None
                        )
                        if success is True:
                            safe_signals += 1
                        elif success is False:
                            risk_signals += 1

                    if risk_signals + safe_signals > 0:
                        # Blend in associative context (capped contribution)
                        assoc_factor = (risk_signals - safe_signals) / (
                            risk_signals + safe_signals
                        )
                        adjustment += assoc_factor * 0.1  # Max ±0.1 from associations
                        adjustment = max(-self.max_adjustment, min(self.max_adjustment, adjustment))
                except Exception:
                    pass  # Associative recall is best-effort

            return adjustment

        except Exception as e:
            self._record_error(e)
            return 0.0

    def should_block(
        self,
        category: str,
        severity: str,
        pattern: str = "",
        context: str = "",
    ) -> tuple[bool, str]:
        """Determine if action should be blocked with memory-informed adjustment.

        Args:
            category: DangerCategory value
            severity: RiskLevel value
            pattern: Specific pattern detected
            context: Additional context

        Returns:
            (should_block, reason) tuple
        """
        try:
            adjustment = self.get_risk_adjustment(category, pattern, context)

            # Convert severity to numeric score
            severity_scores = {
                "low": 0.25,
                "medium": 0.5,
                "high": 0.75,
                "critical": 1.0,
            }
            base_score = severity_scores.get(severity.lower(), 0.5)

            # Apply adjustment
            adjusted_score = base_score + adjustment

            # Check against NAc predictions if available
            nac_factor = self._get_nac_risk_factor(category, pattern)
            final_score = adjusted_score * nac_factor

            # Block if score exceeds threshold
            threshold = 0.65  # Adjustable based on policy
            should_block = final_score >= threshold

            reason = (
                f"score={final_score:.2f} (base={base_score:.2f}, "
                f"adj={adjustment:+.2f}, nac={nac_factor:.2f})"
            )

            return should_block, reason

        except Exception as e:
            self._record_error(e)
            # Default to blocking on error (safer)
            return True, f"error: {e}"

    def _get_nac_risk_factor(self, category: str, pattern: str) -> float:
        """Query NAc for learned risk factor.

        Uses causal inference to predict actual harm probability.
        """
        if not self.nac:
            return 1.0

        try:
            # Query NAc for outcomes of similar events
            event_key = f"fear:{category}:{pattern}"
            prediction = self.nac.predict_outcome(event_key)

            if prediction and hasattr(prediction, "probability"):
                # Invert: high harm probability = higher risk factor
                return 0.5 + (prediction.probability * 0.5)

            return 1.0

        except Exception:
            return 1.0

    # ─────────────────────────────────────────────────────────────────────────
    # Outcome Recording
    # ─────────────────────────────────────────────────────────────────────────

    def record_outcome(
        self,
        category: str,
        pattern: str,
        was_blocked: bool,
        actual_harm: bool | None = None,
        severity: str = "medium",
        action_type: str = "code_review",
        context: str = "",
    ) -> None:
        """Record the outcome of a risk assessment.

        Learning rules:
        - Blocked + No harm (false positive): Lower threshold
        - Blocked + Harm (true positive): Maintain/raise threshold
        - Not blocked + No harm: Maintain threshold
        - Not blocked + Harm (miss): Raise threshold significantly

        Args:
            category: DangerCategory value
            pattern: Specific pattern detected
            was_blocked: Whether action was blocked
            actual_harm: Whether actual harm occurred (None if unknown)
            severity: RiskLevel value
            action_type: Type of review
            context: Additional context
        """
        if not self._healthy:
            return

        try:
            # Create pattern hash
            pattern_hash = hashlib.sha256(
                f"{category}:{pattern}:{context[:50]}".encode()
            ).hexdigest()[:12]

            # Record the event
            event = RiskEvent(
                pattern_hash=pattern_hash,
                category=category,
                severity=severity,
                action_type=action_type,
                was_blocked=was_blocked,
                actual_harm=actual_harm,
                timestamp=time.time(),
                context=context[:100],
            )

            self._recent_events.append(event)
            if len(self._recent_events) > self._max_events:
                self._recent_events = self._recent_events[-self._max_events:]

            # Update statistics
            self._update_stats(event)

            # Update learned adjustment
            if actual_harm is not None:
                self._update_adjustment(event)

            # Report to NAc for causal learning
            self._report_to_nac(event)

            # Auto-save periodically
            self._maybe_auto_save()

        except Exception as e:
            self._record_error(e)

    def _update_stats(self, event: RiskEvent) -> None:
        """Update category statistics."""
        category = event.category

        if category not in self._category_stats:
            self._category_stats[category] = {
                "total": 0,
                "blocked": 0,
                "false_positives": 0,
                "true_positives": 0,
                "misses": 0,
            }

        stats = self._category_stats[category]
        stats["total"] += 1

        if event.was_blocked:
            stats["blocked"] += 1
            if event.actual_harm is False:
                stats["false_positives"] += 1
            elif event.actual_harm is True:
                stats["true_positives"] += 1
        else:
            if event.actual_harm is True:
                stats["misses"] += 1

    def _update_adjustment(self, event: RiskEvent) -> None:
        """Update learned adjustment based on outcome."""
        key = (event.category, event.pattern_hash[:8])

        # Get or create adjustment entry
        if key not in self._adjustments:
            self._adjustments[key] = LearnedRiskAdjustment(
                category=event.category,
                pattern_type=event.pattern_hash[:8],
                base_severity=event.severity,
                adjustment=0.0,
                samples=0,
                false_positives=0,
                true_positives=0,
                last_updated=time.time(),
            )

        adj = self._adjustments[key]
        adj.samples += 1
        adj.last_updated = time.time()

        if event.was_blocked:
            if event.actual_harm is False:
                adj.false_positives += 1
            elif event.actual_harm is True:
                adj.true_positives += 1

        # Only adjust after sufficient samples
        if adj.samples < self.min_samples_for_adjustment:
            return

        # Calculate adjustment
        delta = 0.0
        if event.was_blocked and event.actual_harm is False:
            # False positive: lower the threshold
            delta = -self.learning_rate
        elif event.was_blocked and event.actual_harm is True:
            # True positive: maintain/slightly raise
            delta = self.learning_rate * 0.3
        elif not event.was_blocked and event.actual_harm is True:
            # Miss (dangerous action not blocked): raise significantly
            delta = self.learning_rate * 2.0
        # Not blocked + no harm: no adjustment needed

        # Apply bounded adjustment
        adj.adjustment = max(
            -self.max_adjustment,
            min(self.max_adjustment, adj.adjustment + delta),
        )

    def _report_to_nac(self, event: RiskEvent) -> None:
        """Report event to NAc for causal learning."""
        if not self.nac:
            return

        try:
            event_key = f"fear:{event.category}:{event.pattern_hash[:8]}"
            outcome = 1.0 if event.actual_harm else 0.0

            # Only report if outcome is known
            if event.actual_harm is not None:
                self.nac.record_event(
                    event_key,
                    outcome,
                    metadata={
                        "was_blocked": event.was_blocked,
                        "severity": event.severity,
                    },
                )
        except Exception:
            pass  # Non-critical, don't record as error

    # ─────────────────────────────────────────────────────────────────────────
    # Analysis and Reporting
    # ─────────────────────────────────────────────────────────────────────────

    def get_false_positive_rate(self, category: str | None = None) -> float:
        """Calculate false positive rate for a category or overall.

        Args:
            category: Specific category, or None for overall

        Returns:
            False positive rate (0-1)
        """
        if category:
            stats = self._category_stats.get(category, {})
            blocked = stats.get("blocked", 0)
            fps = stats.get("false_positives", 0)
            return fps / blocked if blocked > 0 else 0.0

        # Overall
        total_blocked = sum(s.get("blocked", 0) for s in self._category_stats.values())
        total_fps = sum(
            s.get("false_positives", 0) for s in self._category_stats.values()
        )
        return total_fps / total_blocked if total_blocked > 0 else 0.0

    def get_patterns_to_review(self, fp_threshold: float = 0.5) -> list[str]:
        """Get patterns with high false positive rates for review.

        These patterns might need rule adjustments or whitelisting.

        Args:
            fp_threshold: Minimum FP rate to flag

        Returns:
            List of pattern descriptions
        """
        patterns = []

        for key, adj in self._adjustments.items():
            if adj.samples >= self.min_samples_for_adjustment:
                fp_rate = (
                    adj.false_positives / adj.samples if adj.samples > 0 else 0.0
                )
                if fp_rate >= fp_threshold:
                    patterns.append(
                        f"{key[0]}:{key[1]} (FP rate: {fp_rate:.1%}, "
                        f"samples: {adj.samples})"
                    )

        return patterns

    # ─────────────────────────────────────────────────────────────────────────
    # Health and Statistics
    # ─────────────────────────────────────────────────────────────────────────

    def _record_error(self, error: Exception) -> None:
        """Record an error and potentially disable the bridge."""
        self._error_count += 1
        logger.warning(
            "FearCircuitBridge error (%d/%d): %s",
            self._error_count,
            self._max_errors,
            error,
        )

        if self._error_count >= self._max_errors:
            self._healthy = False
            logger.error(
                "FearCircuitBridge disabled after %d errors", self._error_count
            )

    @property
    def is_healthy(self) -> bool:
        """Check if bridge is operational."""
        return self._healthy

    def stats(self) -> dict[str, Any]:
        """Return bridge statistics."""
        return {
            "healthy": self._healthy,
            "error_count": self._error_count,
            "learned_adjustments": len(self._adjustments),
            "recent_events": len(self._recent_events),
            "category_stats": dict(self._category_stats),
            "overall_fp_rate": self.get_false_positive_rate(),
            "patterns_to_review": len(self.get_patterns_to_review()),
        }


__all__ = ["FearCircuitBridge", "RiskEvent", "LearnedRiskAdjustment"]