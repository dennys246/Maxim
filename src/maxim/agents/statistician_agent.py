"""StatisticianAgent — pattern-emergence detector with IPS→AG escalation.

Uses the dual number system (IPS fast/approximate + Angular Gyrus slow/precise)
to distinguish random noise from forming patterns in operational metrics.
Publishes StatisticalInsight and StatisticalSummary to the agent bus.

Brain mapping:
  The posterior parietal cortex (IPS + angular gyrus) feeds quantitative
  analysis into the prefrontal cortex (ExecAgent) for executive decisions.
  The StatisticianAgent bridges this — turning raw event streams into
  statistical context.  IPS handles fast randomness assessment; AG is
  reserved for uncertain cases requiring algebraic precision and pattern
  memory recall.

SCN integration:
  When the SCN oscillator is available, temporal context enriches pattern
  interpretation.  A metric anomaly during a temporally unusual period is
  weighted differently than one during an established rhythm.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from maxim.agents.base import Agent
from maxim.agents.bus import (
    AgentBus,
    AnalysisSuggestion,
    GoalCompleted,
    StatisticalInsight,
    StatisticalSummary,
    ToolResult,
)
from maxim.math.ips import IPS
from maxim.math.math_types import MathMemory
from maxim.math.types import (
    MathCategory,
    RandomnessResult,
    TrendDirection,
)
from maxim.memory.semantic_promoter import PromotionCandidate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PatternDetector FSM
# ---------------------------------------------------------------------------


class PatternState(Enum):
    """State of per-metric pattern detection."""

    OBSERVING = auto()
    PATTERN_FORMING = auto()
    ESCALATED_TO_AG = auto()
    CONFIRMED_PATTERN = auto()
    CONFIRMED_RANDOM = auto()


class MetricDataType(Enum):
    """Inferred data type for a metric series."""

    BINARY = "binary"          # 0/1 success/fail metrics
    CONTINUOUS = "continuous"   # Unbounded float values
    RATE = "rate"              # 0.0-1.0 bounded rates
    LATENCY = "latency"        # Time-based measurements


@dataclass
class PatternDetector:
    """Per-metric pattern detection state machine."""

    metric_name: str
    state: PatternState = PatternState.OBSERVING
    observations: int = 0
    confidence_history: list[float] = field(default_factory=list)
    last_randomness: RandomnessResult | None = None

    # Angular Gyrus memory reference
    ag_memory_id: str | None = None
    ag_r_squared: float | None = None

    # Temporal context (from SCN)
    temporal_anomaly: float = 0.0

    # Transition thresholds
    min_observations: int = 15
    forming_threshold: float = 0.4
    confirm_threshold: float = 0.65
    escalation_patience: int = 8
    random_patience: int = 50
    confirm_window: int = 5

    # Internal counters
    _forming_steps: int = 0
    _random_steps: int = 0


# ---------------------------------------------------------------------------
# StatisticianAgent
# ---------------------------------------------------------------------------


class StatisticianAgent(Agent):
    """Pattern-emergence detector with IPS→AG escalation + SCN temporal context.

    Subscribes to ToolResult and GoalCompleted events on the bus,
    maintains per-metric time series, and uses a PatternDetector FSM
    to classify each metric as random or patterned.

    When IPS is uncertain (pattern_confidence between 0.3-0.65 for
    escalation_patience steps), Angular Gyrus is invoked for precise
    analysis and pattern memory recall/storage.
    """

    agent_name = "statistician"

    def __init__(
        self,
        bus: AgentBus,
        ips: IPS,
        angular_gyrus: Any | None = None,
        scn: Any | None = None,
        *,
        name: str | None = None,
        enabled: bool = True,
        window_size: int = 100,
        analysis_interval: float = 5.0,
        persistence_path: str | None = None,
    ) -> None:
        super().__init__(name=name, enabled=enabled)
        self._bus = bus
        self._ips = ips
        self._ag = angular_gyrus
        self._scn = scn

        self._window_size = window_size
        self._analysis_interval = analysis_interval
        self.persistence_path = persistence_path

        # Per-metric data
        self._metric_series: dict[str, deque[float]] = {}
        self._detectors: dict[str, PatternDetector] = {}

        # Timing
        self._last_analysis: float = 0.0
        self._last_summary: float = 0.0
        self._summary_interval: float = 30.0

        # Subscribe
        self._bus.subscribe(ToolResult, self._on_tool_result)
        self._bus.subscribe(GoalCompleted, self._on_goal_completed)

    # ------------------------------------------------------------------
    # Bus handlers
    # ------------------------------------------------------------------

    def _on_tool_result(self, result: ToolResult) -> None:
        """Record tool outcome as a metric series point."""
        metric = f"tool:{result.tool_name}:success"
        value = 1.0 if result.success else 0.0
        self._record_metric(metric, value)

    def _on_goal_completed(self, completed: GoalCompleted) -> None:
        """Record goal outcome as a metric series point."""
        metric = f"goal:{completed.goal_id}:success"
        value = 1.0 if completed.success else 0.0
        self._record_metric(metric, value)

    def _record_metric(self, metric: str, value: float) -> None:
        """Append value to the metric's time series."""
        if metric not in self._metric_series:
            self._metric_series[metric] = deque(maxlen=self._window_size)
            self._detectors[metric] = PatternDetector(metric_name=metric)
        self._metric_series[metric].append(value)

    # ------------------------------------------------------------------
    # Agent lifecycle
    # ------------------------------------------------------------------

    def on_start(self, **kwargs: Any) -> None:
        if self.persistence_path:
            self.load(self.persistence_path)

    def on_stop(self, **kwargs: Any) -> None:
        if self.persistence_path:
            self.save(self.persistence_path)
        self._bus.unsubscribe(ToolResult, self._on_tool_result)
        self._bus.unsubscribe(GoalCompleted, self._on_goal_completed)

    def on_step(self, **kwargs: Any) -> None:
        """Periodic analysis — step each PatternDetector FSM."""
        now = time.time()
        if now - self._last_analysis < self._analysis_interval:
            return

        for metric_name, detector in list(self._detectors.items()):
            series = self._metric_series.get(metric_name)
            if not series or len(series) < detector.min_observations:
                continue

            # Reduce monitoring frequency for confirmed random
            if detector.state == PatternState.CONFIRMED_RANDOM:
                if detector.observations % 10 != 0:
                    detector.observations += 1
                    continue

            # --- IPS fast path ---
            values = list(series)
            randomness = self._ips.assess_randomness(values[-self._window_size:])
            detector.last_randomness = randomness
            detector.confidence_history.append(randomness.pattern_confidence)
            # Keep confidence history bounded
            if len(detector.confidence_history) > 50:
                detector.confidence_history = detector.confidence_history[-50:]
            detector.observations += 1

            # Temporal context (if SCN available)
            if self._scn is not None:
                self._assess_with_temporal_context(detector)

            # State machine transitions
            old_state = detector.state
            self._transition(detector, randomness)

            # --- AG slow path (triggered only on uncertainty) ---
            if detector.state == PatternState.ESCALATED_TO_AG:
                self._escalate_to_angular_gyrus(detector, values)

            # Publish insight on meaningful transitions
            if (
                detector.state == PatternState.CONFIRMED_PATTERN
                and old_state != PatternState.CONFIRMED_PATTERN
            ):
                insight = self._build_insight(detector, randomness, values)
                self._bus.publish(insight)

        # Periodic summary
        if now - self._last_summary >= self._summary_interval:
            summary = self.summarize_for_context()
            if summary:
                try:
                    suggestions = self._generate_suggestions()
                    data_type_breakdown = self._compute_data_type_breakdown()
                except Exception:
                    suggestions = []
                    data_type_breakdown = {}
                self._bus.publish(
                    StatisticalSummary(
                        timestamp=now,
                        summary=summary,
                        active_patterns=self._count_active_patterns(),
                        metrics_monitored=len(self._detectors),
                        suggestions=suggestions,
                        data_type_breakdown=data_type_breakdown,
                    )
                )
            self._last_summary = now

        self._last_analysis = now

    # ------------------------------------------------------------------
    # State machine transitions
    # ------------------------------------------------------------------

    def _transition(self, detector: PatternDetector, randomness: RandomnessResult) -> None:
        """Advance the PatternDetector FSM based on IPS assessment."""
        if detector.state == PatternState.OBSERVING:
            if randomness.pattern_confidence > detector.forming_threshold:
                detector.state = PatternState.PATTERN_FORMING
                detector._forming_steps = 0

        elif detector.state == PatternState.PATTERN_FORMING:
            detector._forming_steps += 1

            # IPS confident enough → direct confirmation
            recent = detector.confidence_history[-detector.confirm_window:]
            if (
                len(recent) >= detector.confirm_window
                and all(c > detector.confirm_threshold for c in recent)
            ):
                detector.state = PatternState.CONFIRMED_PATTERN

            # IPS still uncertain after patience exhausted → escalate to AG
            elif detector._forming_steps >= detector.escalation_patience:
                if 0.3 < randomness.pattern_confidence < detector.confirm_threshold:
                    detector.state = PatternState.ESCALATED_TO_AG

            # Confidence dropped back → return to OBSERVING
            elif randomness.pattern_confidence < 0.2:
                detector.state = PatternState.OBSERVING
                detector._forming_steps = 0

        elif detector.state == PatternState.CONFIRMED_RANDOM:
            # Allow re-entry if new data shows pattern
            if randomness.pattern_confidence > detector.forming_threshold:
                detector.state = PatternState.OBSERVING
                detector._random_steps = 0

    # ------------------------------------------------------------------
    # IPS → Angular Gyrus escalation
    # ------------------------------------------------------------------

    def _escalate_to_angular_gyrus(
        self, detector: PatternDetector, series: list[float]
    ) -> None:
        """IPS uncertain → invoke AG for precise determination."""
        if self._ag is None:
            # No AG available — fall back to IPS thresholds
            if detector.last_randomness and detector.last_randomness.pattern_confidence > 0.5:
                detector.state = PatternState.CONFIRMED_PATTERN
            else:
                detector.state = PatternState.PATTERN_FORMING
            return

        try:
            # 1. Check AG memory first — have we seen this pattern before?
            prior = self._ag.recall(
                limit=1,
                name=f"stat:{detector.metric_name}",
                category="PATTERN",
                domain="operational_statistics",
            )

            if prior:
                record = prior[0]
                # Pattern previously learned — verify it still holds
                current_trend = self._ips.detect_trend(series)
                stored_slope = 0.0
                if isinstance(record, MathMemory) and isinstance(record.inputs, dict):
                    stored_slope = record.inputs.get("slope", 0)

                if (
                    (stored_slope > 0 and current_trend.direction == TrendDirection.INCREASING)
                    or (stored_slope < 0 and current_trend.direction == TrendDirection.DECREASING)
                    or current_trend.direction == TrendDirection.OSCILLATING
                ):
                    # Pattern persists — reinforce memory
                    record.observation_count += 1
                    record.confidence = min(1.0, record.confidence + 0.05)
                    detector.state = PatternState.CONFIRMED_PATTERN
                    detector.ag_memory_id = record.id
                    detector.ag_r_squared = record.confidence
                    return
                else:
                    # Pattern broke — reduce memory confidence
                    record.confidence *= 0.7
                    detector.state = PatternState.PATTERN_FORMING
                    detector._forming_steps = 0
                    return

            # 2. No prior memory — compute precisely
            analysis = self._ag.analyze(series, "linear")
            r_squared = analysis.confidence

            detector.ag_r_squared = r_squared

            # 3. Decision
            if r_squared > 0.5:
                # Confirmed trend — store as AG PATTERN memory
                now = time.time()
                record_id = f"stat_{detector.metric_name}_{int(now)}"
                direction = analysis.parameters.get("direction", "trend")
                slope = analysis.parameters.get("slope", 0)

                self._ag.store(
                    MathMemory(
                        id=record_id,
                        timestamp=now,
                        name=f"stat:{detector.metric_name}",
                        category=MathCategory.PATTERN,
                        domain="operational_statistics",
                        verbal=(
                            f"{detector.metric_name} shows {direction} "
                            f"(R²={r_squared:.2f}, slope={slope:.4f})"
                        ),
                        code=f"linear_fit('{detector.metric_name}', R2={r_squared:.2f})",
                        inputs={"slope": slope, "r_squared": r_squared},
                        source="learned",
                        confidence=r_squared,
                        observation_count=1,
                    )
                )
                detector.state = PatternState.CONFIRMED_PATTERN
                detector.ag_memory_id = record_id
            elif r_squared < 0.2:
                detector.state = PatternState.CONFIRMED_RANDOM
            else:
                # Still uncertain — return to PATTERN_FORMING
                detector.state = PatternState.PATTERN_FORMING
                detector._forming_steps = 0

        except Exception as e:
            logger.debug("AG escalation failed for %s: %s", detector.metric_name, e)
            # On error, fall back to PATTERN_FORMING
            detector.state = PatternState.PATTERN_FORMING
            detector._forming_steps = 0

    # ------------------------------------------------------------------
    # SCN temporal context
    # ------------------------------------------------------------------

    def _assess_with_temporal_context(self, detector: PatternDetector) -> None:
        """Weight pattern confidence by temporal normality."""
        try:
            from maxim.time.temporal_signature import TemporalSignature

            sig = TemporalSignature.now()
            osc = getattr(self._scn, "oscillator", None)
            if osc is not None:
                anomaly = self._scn.temporal_anomaly_score(sig)
                if anomaly is not None:
                    detector.temporal_anomaly = anomaly

                    # If temporal_anomaly is HIGH → this time itself is unusual,
                    # so a "pattern" here might be temporal novelty
                    if anomaly > 0.7 and detector.confidence_history:
                        detector.confidence_history[-1] *= 0.6
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Insight construction
    # ------------------------------------------------------------------

    def _build_insight(
        self,
        detector: PatternDetector,
        randomness: RandomnessResult,
        series: list[float],
    ) -> StatisticalInsight:
        """Build a StatisticalInsight for a confirmed pattern."""
        # Determine insight type
        pattern_type = randomness.pattern_type.name if randomness else "UNKNOWN"

        # Build description
        recent_mean = sum(series[-10:]) / max(len(series[-10:]), 1)
        desc_parts = [f"{detector.metric_name}: non-random pattern detected"]
        desc_parts.append(f"(type={pattern_type}, confidence={randomness.pattern_confidence:.2f})")
        desc_parts.append(f"recent mean={recent_mean:.2f}")

        if detector.ag_r_squared is not None:
            desc_parts.append(f"R²={detector.ag_r_squared:.2f}")

        # Temporal context
        temporal_ctx = "no temporal context"
        if detector.temporal_anomaly > 0.5:
            temporal_ctx = "unusual for this time period"
        elif detector.temporal_anomaly < 0.2:
            temporal_ctx = "normal temporal context"

        # Severity: higher for non-random patterns with high confidence
        severity = min(1.0, randomness.pattern_confidence * 0.8 + 0.2)

        return StatisticalInsight(
            timestamp=time.time(),
            insight_type="pattern",
            metric=detector.metric_name,
            description=" ".join(desc_parts),
            severity=severity,
            pattern_type=pattern_type,
            temporal_context=temporal_ctx,
            data={
                "runs_z_score": randomness.runs_z_score,
                "autocorrelation": randomness.autocorrelation,
                "observations": detector.observations,
                "ag_r_squared": detector.ag_r_squared,
                "ag_memory_id": detector.ag_memory_id,
            },
        )

    # ------------------------------------------------------------------
    # Summary for context
    # ------------------------------------------------------------------

    def summarize_for_context(self) -> str:
        """Actionable natural language for LLM context.

        Only includes CONFIRMED patterns and active anomalies.
        Enriched with Angular Gyrus pattern memory for historical context.
        """
        insights: list[tuple[float, str]] = []

        for metric_name, detector in self._detectors.items():
            if detector.state == PatternState.CONFIRMED_PATTERN:
                parts = [f"[PATTERN] {metric_name}"]

                if detector.last_randomness:
                    parts.append(f"type={detector.last_randomness.pattern_type.name}")
                    parts.append(f"confidence={detector.last_randomness.pattern_confidence:.2f}")

                if detector.ag_r_squared is not None:
                    parts.append(f"R²={detector.ag_r_squared:.2f}")

                # AG memory enrichment
                if self._ag and detector.ag_memory_id:
                    try:
                        record = self._ag.get(detector.ag_memory_id)
                        if record:
                            parts.append(
                                f"observed {record.observation_count} times "
                                f"(memory confidence={record.confidence:.2f})"
                            )
                    except Exception:
                        pass

                # Temporal context
                if detector.temporal_anomaly > 0.5:
                    parts.append("[TEMPORAL] unusual for this time")

                severity = detector.last_randomness.pattern_confidence if detector.last_randomness else 0.5
                insights.append((severity, " | ".join(parts)))

        if not insights:
            return ""

        # Sort by severity (highest first), limit to top 3
        insights.sort(key=lambda x: x[0], reverse=True)
        lines = [text for _, text in insights[:3]]

        # Add oscillator coherence if available
        if self._scn is not None:
            try:
                coherence = self._scn.phase_coherence()
                if coherence is not None:
                    lines.append(f"[TEMPORAL] Oscillator coherence: {coherence:.2f}")
            except Exception:
                pass

        return "\n".join(lines)

    def _count_active_patterns(self) -> int:
        """Count metrics in CONFIRMED_PATTERN state."""
        return sum(
            1
            for d in self._detectors.values()
            if d.state == PatternState.CONFIRMED_PATTERN
        )

    # ------------------------------------------------------------------
    # Data type inference + suggestion engine
    # ------------------------------------------------------------------

    def _infer_data_type(self, metric_name: str, series: deque[float]) -> MetricDataType:
        """Infer the data type from metric naming convention or value distribution."""
        name_lower = metric_name.lower()

        # Naming convention heuristics (fast path)
        if name_lower.endswith(":success") or name_lower.endswith(":fail"):
            return MetricDataType.BINARY
        if "latency" in name_lower or "duration" in name_lower or "time_ms" in name_lower:
            return MetricDataType.LATENCY
        if "rate" in name_lower or "ratio" in name_lower:
            return MetricDataType.RATE

        # Fall back to value distribution analysis
        values = list(series)
        if not values:
            return MetricDataType.CONTINUOUS

        unique = set(values)
        if unique <= {0.0, 1.0}:
            return MetricDataType.BINARY
        if all(0.0 <= v <= 1.0 for v in values):
            return MetricDataType.RATE
        if all(v >= 0.0 for v in values) and max(values) > 1.0:
            return MetricDataType.LATENCY

        return MetricDataType.CONTINUOUS

    def _generate_suggestions(self) -> list[AnalysisSuggestion]:
        """Generate ranked analysis suggestions from all active detectors."""
        suggestions: list[AnalysisSuggestion] = []

        for metric_name, detector in self._detectors.items():
            series = self._metric_series.get(metric_name)
            if not series or len(series) < detector.min_observations:
                continue

            data_type = self._infer_data_type(metric_name, series)
            suggestion = self._suggest_for_state(detector, data_type, series)
            if suggestion is not None:
                suggestions.append(suggestion)

        # Sort by priority (highest first), return top 5
        suggestions.sort(key=lambda s: s.priority, reverse=True)
        return suggestions[:5]

    def _suggest_for_state(
        self,
        detector: PatternDetector,
        data_type: MetricDataType,
        series: deque[float],
    ) -> AnalysisSuggestion | None:
        """Decision matrix: FSM state × data type → specific analysis suggestion."""
        state = detector.state

        if state == PatternState.OBSERVING or state == PatternState.CONFIRMED_RANDOM:
            return None

        # Base priority by state
        priority = {
            PatternState.PATTERN_FORMING: 0.5,
            PatternState.ESCALATED_TO_AG: 0.8,
            PatternState.CONFIRMED_PATTERN: 0.6,
        }.get(state, 0.3)

        # Temporal context boost
        if detector.temporal_anomaly > 0.5:
            priority = min(1.0, priority + 0.15)

        # Build recent stats for rationale
        recent = list(series)[-20:]
        recent_mean = sum(recent) / len(recent) if recent else 0.0

        if state == PatternState.PATTERN_FORMING:
            if data_type in (MetricDataType.BINARY, MetricDataType.RATE):
                return AnalysisSuggestion(
                    metric=detector.metric_name,
                    tool_call="math",
                    operation="assess_randomness",
                    rationale=(
                        f"{data_type.value} metric showing emerging pattern "
                        f"(mean={recent_mean:.2f}, obs={detector.observations})"
                    ),
                    priority=priority,
                    data_type=data_type.value,
                    fsm_state=state.name,
                )
            elif data_type == MetricDataType.LATENCY:
                return AnalysisSuggestion(
                    metric=detector.metric_name,
                    tool_call="math",
                    operation="anomaly",
                    rationale=(
                        f"Latency metric with forming pattern "
                        f"(mean={recent_mean:.2f}ms, obs={detector.observations})"
                    ),
                    priority=priority,
                    data_type=data_type.value,
                    fsm_state=state.name,
                )
            else:  # CONTINUOUS
                return AnalysisSuggestion(
                    metric=detector.metric_name,
                    tool_call="math",
                    operation="trend",
                    rationale=(
                        f"Continuous metric with forming pattern "
                        f"(mean={recent_mean:.2f}, obs={detector.observations})"
                    ),
                    priority=priority,
                    data_type=data_type.value,
                    fsm_state=state.name,
                )

        elif state == PatternState.ESCALATED_TO_AG:
            return AnalysisSuggestion(
                metric=detector.metric_name,
                tool_call="math",
                operation="analyze linear",
                rationale=(
                    f"IPS uncertain after {detector._forming_steps} steps — "
                    f"AG precision needed (mean={recent_mean:.2f})"
                ),
                priority=priority,
                data_type=data_type.value,
                fsm_state=state.name,
            )

        elif state == PatternState.CONFIRMED_PATTERN:
            if detector.ag_memory_id:
                return AnalysisSuggestion(
                    metric=detector.metric_name,
                    tool_call="math",
                    operation="recall_memory",
                    rationale=(
                        f"Confirmed pattern with AG memory — "
                        f"recall for characterization (R²={detector.ag_r_squared or 0:.2f})"
                    ),
                    priority=priority,
                    data_type=data_type.value,
                    fsm_state=state.name,
                )
            else:
                return AnalysisSuggestion(
                    metric=detector.metric_name,
                    tool_call="math",
                    operation="analyze",
                    rationale=(
                        f"Confirmed pattern without AG memory — "
                        f"characterize trend (mean={recent_mean:.2f})"
                    ),
                    priority=priority,
                    data_type=data_type.value,
                    fsm_state=state.name,
                )

        return None

    def _compute_data_type_breakdown(self) -> dict[str, int]:
        """Count metrics by inferred data type."""
        breakdown: dict[str, int] = {}
        for metric_name, series in self._metric_series.items():
            if not series:
                continue
            dt = self._infer_data_type(metric_name, series)
            breakdown[dt.value] = breakdown.get(dt.value, 0) + 1
        return breakdown

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist metric series and pattern states."""
        data: dict[str, Any] = {
            "version": "1.0",
            "saved_at": time.time(),
            "metric_series": {
                name: list(series)
                for name, series in self._metric_series.items()
            },
            "pattern_states": {},
        }

        for name, detector in self._detectors.items():
            data["pattern_states"][name] = {
                "state": detector.state.name,
                "observations": detector.observations,
                "confidence_history": detector.confidence_history[-20:],
                "ag_memory_id": detector.ag_memory_id,
                "ag_r_squared": detector.ag_r_squared,
                "temporal_anomaly": detector.temporal_anomaly,
            }

        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            tmp_path = path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, path)
            logger.debug(
                "StatisticianAgent saved %d metrics to %s",
                len(self._metric_series),
                path,
            )
        except OSError as e:
            logger.warning("Failed to save statistician state: %s", e)

    def load(self, path: str) -> None:
        """Restore from persisted state."""
        if not os.path.exists(path):
            return

        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.debug("Failed to load statistician state: %s", e)
            return

        # Restore metric series
        for name, values in data.get("metric_series", {}).items():
            series: deque[float] = deque(maxlen=self._window_size)
            series.extend(values[-self._window_size:])
            self._metric_series[name] = series

        # Restore pattern states
        for name, state_data in data.get("pattern_states", {}).items():
            if name not in self._detectors:
                self._detectors[name] = PatternDetector(metric_name=name)

            detector = self._detectors[name]
            try:
                detector.state = PatternState[state_data.get("state", "OBSERVING")]
            except KeyError:
                detector.state = PatternState.OBSERVING
            detector.observations = state_data.get("observations", 0)
            detector.confidence_history = state_data.get("confidence_history", [])
            detector.ag_memory_id = state_data.get("ag_memory_id")
            detector.ag_r_squared = state_data.get("ag_r_squared")
            detector.temporal_anomaly = state_data.get("temporal_anomaly", 0.0)

        logger.debug(
            "StatisticianAgent loaded %d metrics from %s",
            len(self._metric_series),
            path,
        )

    def stats(self) -> dict[str, Any]:
        """Return current statistics."""
        state_counts: dict[str, int] = {}
        for detector in self._detectors.values():
            state_name = detector.state.name
            state_counts[state_name] = state_counts.get(state_name, 0) + 1

        return {
            "metrics_monitored": len(self._detectors),
            "total_observations": sum(
                len(s) for s in self._metric_series.values()
            ),
            "state_distribution": state_counts,
            "active_patterns": self._count_active_patterns(),
        }

    # ------------------------------------------------------------------
    # PromotionSource protocol
    # ------------------------------------------------------------------

    def get_promotion_candidates(
        self,
        min_confidence: float = 0.6,
        min_observations: int = 3,
    ) -> list[PromotionCandidate]:
        """Surface confirmed operational patterns for ATL promotion.

        Implements PromotionSource protocol. Only returns metrics in
        CONFIRMED_PATTERN state with sufficient confidence and observations.
        """
        candidates: list[PromotionCandidate] = []
        for metric_name, detector in self._detectors.items():
            if detector.state != PatternState.CONFIRMED_PATTERN:
                continue
            if detector.observations < min_observations:
                continue

            # Get confidence from AG memory if available, else from IPS
            confidence = detector.ag_r_squared or (
                detector.last_randomness.pattern_confidence
                if detector.last_randomness
                else 0.0
            )
            if confidence < min_confidence:
                continue

            metadata: dict[str, Any] = {
                "metric_name": metric_name,
                "pattern_type": (
                    detector.last_randomness.pattern_type.name
                    if detector.last_randomness
                    else "UNKNOWN"
                ),
                "observations": detector.observations,
                "ag_memory_id": detector.ag_memory_id,
                "r_squared": detector.ag_r_squared,
            }

            # Add SCN temporal context if available
            if detector.temporal_anomaly > 0.0:
                metadata["temporal_anomaly"] = detector.temporal_anomaly
            if self._scn is not None:
                try:
                    coherence = self._scn.phase_coherence()
                    if coherence is not None:
                        metadata["phase_coherence"] = coherence
                except Exception:
                    pass

            candidates.append(
                PromotionCandidate(
                    pattern_name=f"stat:{metric_name}",
                    category="operational_pattern",
                    confidence=confidence,
                    source_memory_ids=[],  # StatAgent tracks metrics, not episode IDs
                    metadata=metadata,
                )
            )
        return candidates
