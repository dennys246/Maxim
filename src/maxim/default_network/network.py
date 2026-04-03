"""DefaultNetwork - Main coordinator for reactive behaviors.

The DefaultNetwork is the central component that:
1. Subscribes to YOLO detection stream
2. Runs behavior modules in parallel
3. Arbitrates between competing behaviors
4. Respects inhibition signals from deliberative layer
5. Publishes winning actions to motor queue
6. Acts as Thalamic Gate, filtering percepts for escalation
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from maxim.default_network.arbiter import ArbiterConfig, PriorityArbiter
from maxim.attention import AttentionConfig, AttentionNetwork
from maxim.attention import SalienceMap, SalienceMapConfig
from maxim.attention import GazeController, GazeControllerConfig, GazeCommand, GazeState
from maxim.attention import SceneContextDetector, SceneContextConfig
from maxim.default_network.behaviors.base import Behavior, BehaviorState
from maxim.default_network.behaviors.idle import ReturnToCenter
from maxim.default_network.behaviors.turn_around import TurnAround
from maxim.default_network.gate import (
    AdaptiveThresholdConfig,
    GateConfig,
    ThalamicGate,
)
from maxim.attention import GazeHistory, GazeHistoryConfig
from maxim.default_network.messages import (
    ActionProposal,
    DNActionProposal,
    FilteredPercept,
)
from maxim.default_network.background_tasks import (
    BackgroundTaskConfig,
    BackgroundTaskManager,
    ThrottleConfig,
    create_throttlers,
)
from maxim.default_network.movement_utils import (
    MovementConfig,
    compute_dynamic_duration,
    compute_opposite_position,
)
from maxim.salience import ThreadSafeNoveltyTracker
from maxim.salience import SalienceConfig, SalienceNetwork
from maxim.salience import MovementConfig as SalienceMovementConfig, MovementDetector
from maxim.spatial import SpatialMap, SpatialMapConfig, WorkspaceBoundsConfig, WorkspaceBoundsLearner
from maxim.proprioception import PainConfig, PainDetector, FocusLearner, FocusLearnerConfig
from maxim.bridges.pain_bridge import PainCircuitBridge, PainBridgeConfig

if TYPE_CHECKING:
    from maxim.agents.bus import AgentBus, Percept
    from maxim.agents.fear_agent import FearAgent
    from maxim.decisions.nac import NAc

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DefaultNetworkConfig:
    """Configuration for the Default Network.

    Attributes:
        enabled: Whether DN is active.
        update_hz: Target update rate in Hz.
        auto_release_timeout: Max inhibition duration before auto-release.
        publish_actions: Whether to publish DNActionProposal to bus.
        fear_gate_enabled: Whether to gate actions through FearAgent.
    """
    enabled: bool = True
    update_hz: float = 30.0
    auto_release_timeout: float = 5.0
    publish_actions: bool = True
    fear_gate_enabled: bool = True

    # Sub-component configs
    arbiter: ArbiterConfig = field(default_factory=ArbiterConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    adaptive_threshold: AdaptiveThresholdConfig = field(
        default_factory=AdaptiveThresholdConfig
    )
    gaze_history: GazeHistoryConfig = field(default_factory=GazeHistoryConfig)

    # Gaze salience gating - prevents excessive movements
    gaze_salience_enabled: bool = True

    # Idle exploration - random movements when scene is boring
    idle_exploration_enabled: bool = True
    idle_exploration_min_seconds: float = 3.0  # Min time before exploring
    idle_exploration_max_seconds: float = 8.0  # Max time before exploring
    idle_novelty_threshold: float = 0.3  # Below this = not interesting
    # UV range for random exploration targets. Expanded to (0.15, 0.85) to cover
    # more of the visual field while staying within safe servo limits.
    # Coordinates outside this range are clamped by selfy.look_at_image().
    idle_exploration_range: tuple[float, float] = (0.15, 0.85)

    # Spatial map - learns reachable positions and remembers interesting locations
    spatial_map_enabled: bool = True
    spatial_map: SpatialMapConfig = field(default_factory=SpatialMapConfig)

    # WorkspaceBoundsLearner - learns actual joint-space movement limits
    bounds_learning_enabled: bool = True
    bounds: WorkspaceBoundsConfig = field(default_factory=WorkspaceBoundsConfig)

    # AttentionNetwork - DataFrame-backed spatial attention management
    attention_network_enabled: bool = True
    attention: AttentionConfig = field(default_factory=AttentionConfig)

    # SalienceNetwork - DataFrame-backed object salience tracking
    salience_network_enabled: bool = True
    salience: SalienceConfig = field(default_factory=SalienceConfig)

    # MovementDetector - tracks object motion for salience boosting
    movement_detection_enabled: bool = True
    movement_detection: SalienceMovementConfig = field(default_factory=SalienceMovementConfig)

    # SalienceMap - unified spatial salience combining all attention factors
    salience_map_enabled: bool = True
    salience_map_config: SalienceMapConfig = field(default_factory=SalienceMapConfig)

    # GazeController - human-like saccade-fixate dynamics
    gaze_controller_enabled: bool = True
    gaze_controller: GazeControllerConfig = field(default_factory=GazeControllerConfig)

    # SceneContextDetector - detect major scene changes for exploration
    scene_context_enabled: bool = True
    scene_context: SceneContextConfig = field(default_factory=SceneContextConfig)

    # Movement dynamics - duration, jitter, reachability
    movement: MovementConfig = field(default_factory=MovementConfig)
    reachability_check_enabled: bool = True
    min_reachability_threshold: float = 0.3  # Below this = don't attempt movement

    # Throttling - reduce update rates for expensive operations
    throttle: ThrottleConfig = field(default_factory=ThrottleConfig)

    # Background tasks - cleanup, stats collection
    background_tasks: BackgroundTaskConfig = field(default_factory=BackgroundTaskConfig)

    # Pain detection - aversive learning from movement
    pain_detection_enabled: bool = True
    pain_angular_velocity_threshold: float = 100.0  # deg/sec triggers pain
    pain_translation_velocity_threshold: float = 50.0  # mm/sec triggers pain
    pain_prediction_threshold: float = 0.4  # Confidence for pain predictions

    # Focus learning - adaptive movement gain from focus feedback
    focus_learning_enabled: bool = True
    focus_learner: FocusLearnerConfig = field(default_factory=FocusLearnerConfig)


class DefaultNetwork:
    """Reactive behavior layer that generates actions from raw perception.

    The DefaultNetwork continuously processes YOLO detections and generates
    reactive movements without LLM involvement. It also acts as a Thalamic
    Gate, filtering which percepts are escalated to the deliberative layer.

    Example:
        dn = DefaultNetwork(
            maxim=maxim_instance,
            bus=agent_bus,
            behaviors=[OrientingResponse(), SocialAttention()],
        )
        dn.start()

        # Later, when deliberative layer needs focus:
        dn.inhibit(duration=2.0)

        # Cleanup
        dn.stop()
    """

    def __init__(
        self,
        maxim: Any,  # Maxim instance for motor control
        bus: "AgentBus | None" = None,
        behaviors: list[Behavior] | None = None,
        config: DefaultNetworkConfig | None = None,
        fear_agent: "FearAgent | None" = None,
        novelty_tracker: ThreadSafeNoveltyTracker | None = None,
        nac: "NAc | None" = None,
    ) -> None:
        """Initialize the Default Network.

        Args:
            maxim: Maxim instance for executing movements.
            bus: AgentBus for publishing messages.
            behaviors: List of behavior modules. If None, uses defaults.
            config: Network configuration.
            fear_agent: FearAgent for action gating.
            novelty_tracker: Shared novelty tracker. If None, creates new one.
            nac: NAc instance for causal learning (used by pain detection).
        """
        self._maxim = maxim
        self._bus = bus
        self._config = config or DefaultNetworkConfig()
        self._fear_agent = fear_agent

        # Create or use shared novelty tracker
        self._novelty_tracker = novelty_tracker or ThreadSafeNoveltyTracker()

        # Initialize behaviors (empty list if none provided)
        self._behaviors = behaviors or []

        # Initialize arbiter
        self._arbiter = PriorityArbiter(self._config.arbiter)

        # Initialize thalamic gate
        self._gate = ThalamicGate(
            config=self._config.gate,
            adaptive_config=self._config.adaptive_threshold,
        )

        # Initialize gaze history for salience-based movement filtering
        self._gaze_history = GazeHistory(self._config.gaze_history)

        # Initialize spatial map for learned reachability and spatial memory
        self._spatial_map: SpatialMap | None = None
        if self._config.spatial_map_enabled:
            self._spatial_map = SpatialMap(self._config.spatial_map)

        # Initialize WorkspaceBoundsLearner for learned joint-space limits
        self._bounds_learner: WorkspaceBoundsLearner | None = None
        if self._config.bounds_learning_enabled:
            self._bounds_learner = WorkspaceBoundsLearner(self._config.bounds)

        # Initialize AttentionNetwork for DataFrame-backed spatial attention
        self._attention_network: AttentionNetwork | None = None
        if self._config.attention_network_enabled:
            self._attention_network = AttentionNetwork(self._config.attention)

        # Initialize SalienceNetwork for DataFrame-backed object salience
        self._salience_network: SalienceNetwork | None = None
        if self._config.salience_network_enabled:
            self._salience_network = SalienceNetwork(self._config.salience)

        # Initialize MovementDetector for motion-based salience
        self._movement_detector: MovementDetector | None = None
        if self._config.movement_detection_enabled:
            self._movement_detector = MovementDetector(self._config.movement_detection)

        # Initialize SalienceMap for unified spatial attention
        self._salience_map_unified: SalienceMap | None = None
        if self._config.salience_map_enabled:
            self._salience_map_unified = SalienceMap(self._config.salience_map_config)

        # Initialize GazeController for human-like saccade-fixate dynamics
        self._gaze_controller: GazeController | None = None
        if self._config.gaze_controller_enabled:
            self._gaze_controller = GazeController(
                salience_map=self._salience_map_unified,
                config=self._config.gaze_controller,
            )

        # Initialize SceneContextDetector for scene change detection
        self._scene_context: SceneContextDetector | None = None
        if self._config.scene_context_enabled:
            self._scene_context = SceneContextDetector(self._config.scene_context)

        # Initialize operation throttlers to reduce CPU load
        self._throttlers = create_throttlers(self._config.throttle)

        # Initialize background task manager for cleanup/maintenance
        self._background_tasks = BackgroundTaskManager(
            config=self._config.background_tasks,
            spatial_map=self._spatial_map,
            attention_network=self._attention_network,
            salience_network=self._salience_network,
            novelty_tracker=self._novelty_tracker,
        )

        # Initialize PainCircuitBridge for aversive learning
        self._pain_bridge: PainCircuitBridge | None = None
        self._pain_bus = None
        if self._config.pain_detection_enabled and nac is not None:
            from maxim.proprioception.pain_bus import PainBus

            pain_config = PainConfig(
                angular_velocity_pain=self._config.pain_angular_velocity_threshold,
                translation_velocity_pain=self._config.pain_translation_velocity_threshold,
            )
            self._pain_bus = PainBus()
            pain_detector = PainDetector(config=pain_config, pain_bus=self._pain_bus)
            self._pain_bridge = PainCircuitBridge(
                nac=nac,
                pain_detector=pain_detector,
                config=PainBridgeConfig(
                    prediction_threshold=self._config.pain_prediction_threshold,
                ),
                pain_bus=self._pain_bus,
            )
            # Connect bounds learner for joint limit prediction
            if self._bounds_learner is not None:
                self._pain_bridge.set_bounds_learner(self._bounds_learner)
            logger.info("PainCircuitBridge initialized with angular_velocity_threshold=%.1f",
                        self._config.pain_angular_velocity_threshold)

        # Initialize FocusLearner for adaptive movement correction
        self._focus_learner: FocusLearner | None = None
        if self._config.focus_learning_enabled:
            self._focus_learner = FocusLearner(
                config=self._config.focus_learner,
                nac=nac,
                image_center=(320.0, 240.0),  # Standard frame center
            )
            logger.info(
                "FocusLearner initialized with initial_gain=%.2f",
                self._config.focus_learner.initial_gain,
            )

        # Idle exploration state
        self._last_interesting_time: float = time.time()
        self._next_exploration_time: float = self._schedule_next_exploration()

        # State
        self._running = False
        self._inhibited = False
        self._inhibit_until: float = 0.0
        self._behavior_overrides: dict[str, float] = {}

        # Detection buffer (latest from capture)
        self._latest_detections: list[dict] = []
        self._latest_percept: "Percept | None" = None
        self._detection_lock = threading.Lock()

        # Thread management
        self._thread: threading.Thread | None = None
        self._update_interval = 1.0 / self._config.update_hz

        # Current interests (class IDs to prioritize)
        self._interests: frozenset[int] = frozenset()

        # Callbacks for integration
        self._on_action_callbacks: list[Callable[[ActionProposal], None]] = []
        self._on_escalation_callbacks: list[Callable[[FilteredPercept], None]] = []

        # Statistics
        self._stats = {
            "ticks": 0,
            "actions_proposed": 0,
            "actions_executed": 0,
            "actions_blocked": 0,
            "actions_salience_blocked": 0,
            "actions_reachability_blocked": 0,
            "idle_explorations": 0,
            "percepts_received": 0,
            "percepts_escalated": 0,
            "percepts_filtered": 0,
        }
        self._stats_lock = threading.Lock()

        # Auto-subscribe to Percept messages if bus is provided
        self._subscribed_to_bus = False
        if self._bus is not None:
            try:
                from maxim.agents.bus import Percept
                self._bus.subscribe(Percept, self.on_percept)
                self._subscribed_to_bus = True
                logger.debug("DefaultNetwork subscribed to Percept messages")
            except Exception as e:
                logger.warning("Failed to subscribe to bus: %s", e)

    @property
    def novelty_tracker(self) -> ThreadSafeNoveltyTracker:
        """Get the novelty tracker (for sharing with other components)."""
        return self._novelty_tracker

    @property
    def gate(self) -> ThalamicGate:
        """Get the thalamic gate for configuration."""
        return self._gate

    @property
    def arbiter(self) -> PriorityArbiter:
        """Get the arbiter for configuration."""
        return self._arbiter

    @property
    def gaze_history(self) -> GazeHistory:
        """Get the gaze history for monitoring/configuration."""
        return self._gaze_history

    @property
    def spatial_map(self) -> SpatialMap | None:
        """Get the spatial map for monitoring (may be None if disabled)."""
        return self._spatial_map

    @property
    def attention_network(self) -> AttentionNetwork | None:
        """Get the AttentionNetwork for queries (may be None if disabled)."""
        return self._attention_network

    @property
    def salience_network(self) -> SalienceNetwork | None:
        """Get the SalienceNetwork for queries (may be None if disabled)."""
        return self._salience_network

    @property
    def movement_detector(self) -> MovementDetector | None:
        """Get the MovementDetector for motion tracking (may be None if disabled)."""
        return self._movement_detector

    @property
    def salience_map_unified(self) -> SalienceMap | None:
        """Get the unified SalienceMap for gaze decisions (may be None if disabled)."""
        return self._salience_map_unified

    @property
    def gaze_controller(self) -> GazeController | None:
        """Get the GazeController for saccade-fixate dynamics (may be None if disabled)."""
        return self._gaze_controller

    @property
    def scene_context(self) -> SceneContextDetector | None:
        """Get the SceneContextDetector (may be None if disabled)."""
        return self._scene_context

    @property
    def pain_bridge(self) -> PainCircuitBridge | None:
        """Get the PainCircuitBridge for pain detection/learning (may be None if disabled)."""
        return self._pain_bridge

    @property
    def focus_learner(self) -> FocusLearner | None:
        """Get the FocusLearner for adaptive movement correction (may be None if disabled)."""
        return self._focus_learner

    @property
    def is_running(self) -> bool:
        """Whether the DN processing loop is active."""
        return self._running

    @property
    def is_inhibited(self) -> bool:
        """Whether DN output is currently suppressed."""
        return self._inhibited

    def start(self) -> None:
        """Start the default network processing loop."""
        if self._running:
            logger.warning("DefaultNetwork already running")
            return

        if not self._config.enabled:
            logger.info("DefaultNetwork disabled in config")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            name="DefaultNetwork",
            daemon=True,
        )
        self._thread.start()

        # Start background maintenance tasks
        self._background_tasks.start()

        # Initial sync with hardware position to start with accurate tracking
        self._last_position_sync = 0.0  # Force immediate sync on first tick
        sync_fn = getattr(self._maxim, "sync_head_position", None)
        if sync_fn is not None and callable(sync_fn):
            try:
                if sync_fn():
                    logger.info(
                        "Initial position sync: yaw=%.1f°, pitch=%.1f°",
                        getattr(self._maxim, "yaw", 0),
                        getattr(self._maxim, "pitch", 0),
                    )
            except Exception as e:
                logger.debug("Initial position sync failed: %s", e)

        logger.info("DefaultNetwork started (%.1f Hz)", self._config.update_hz)

    def stop(self) -> None:
        """Stop the default network."""
        self._running = False

        # Stop background tasks first
        self._background_tasks.stop()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        # Unsubscribe from bus
        if self._subscribed_to_bus and self._bus is not None:
            try:
                from maxim.agents.bus import Percept
                self._bus.unsubscribe(Percept, self.on_percept)
                self._subscribed_to_bus = False
            except Exception:
                pass

        logger.info("DefaultNetwork stopped")

    def inhibit(self, duration: float | None = None) -> None:
        """Suppress all DN output for duration.

        Args:
            duration: Seconds to inhibit. None = until release() called.
                Max duration is auto_release_timeout from config.
        """
        self._inhibited = True
        if duration:
            effective_duration = min(duration, self._config.auto_release_timeout)
            self._inhibit_until = time.time() + effective_duration
        else:
            self._inhibit_until = time.time() + self._config.auto_release_timeout

        logger.debug("DefaultNetwork inhibited for %.1fs", duration or self._config.auto_release_timeout)

    def release(self) -> None:
        """Release inhibition."""
        self._inhibited = False
        self._inhibit_until = 0.0
        logger.debug("DefaultNetwork released")

    def boost_behavior(self, name: str, modifier: float) -> None:
        """Temporarily boost/suppress a specific behavior's priority.

        Args:
            name: Behavior name.
            modifier: Priority multiplier (>1 = boost, <1 = suppress).
        """
        self._behavior_overrides[name] = modifier

    def clear_behavior_overrides(self) -> None:
        """Clear all behavior priority overrides."""
        self._behavior_overrides.clear()

    def set_interests(self, class_ids: frozenset[int]) -> None:
        """Set interest class IDs for priority escalation.

        Args:
            class_ids: Set of COCO class IDs to prioritize.
        """
        self._interests = class_ids
        self._gate.set_interests(class_ids)

    def set_active_goal(self, goal: str | None) -> None:
        """Set active goal for relevance matching.

        Args:
            goal: Goal description or None.
        """
        self._gate.set_active_goal(goal)

    def on_percept(self, percept: "Percept") -> None:
        """Handler for incoming percepts from AgentBus.

        This method should be called when a new Percept is available.
        Thread-safe - can be called from any thread.

        Args:
            percept: The incoming percept.
        """
        with self._detection_lock:
            self._latest_percept = percept
            # Extract detections if available
            detections = getattr(percept, 'detections', [])
            self._latest_detections = list(detections)

        with self._stats_lock:
            self._stats["percepts_received"] += 1

    def on_detections(self, detections: list[dict]) -> None:
        """Handler for raw YOLO detections.

        Alternative to on_percept() for direct detection stream.

        Args:
            detections: List of detection dicts.
        """
        with self._detection_lock:
            self._latest_detections = list(detections)

    def add_action_callback(
        self,
        callback: Callable[[ActionProposal], None],
    ) -> None:
        """Add callback to be notified of executed actions.

        Args:
            callback: Function called with ActionProposal when action executes.
        """
        self._on_action_callbacks.append(callback)

    def add_escalation_callback(
        self,
        callback: Callable[[FilteredPercept], None],
    ) -> None:
        """Add callback to be notified of escalated percepts.

        Args:
            callback: Function called with FilteredPercept when escalating.
        """
        self._on_escalation_callbacks.append(callback)

    def get_stats(self) -> dict[str, Any]:
        """Get network statistics including sub-network stats."""
        with self._stats_lock:
            stats = dict(self._stats)

        # Include spatial map stats if available
        if self._spatial_map is not None:
            spatial_stats = self._spatial_map.get_stats()
            stats["spatial_map"] = spatial_stats

        # Include attention network stats
        if self._attention_network is not None:
            stats["attention_network"] = self._attention_network.get_stats()

        # Include salience network stats
        if self._salience_network is not None:
            stats["salience_network"] = self._salience_network.get_stats()

        # Include unified salience map stats
        if self._salience_map_unified is not None:
            stats["salience_map"] = self._salience_map_unified.get_stats()

        # Include movement detector stats
        if self._movement_detector is not None:
            stats["movement_detector"] = self._movement_detector.get_stats()

        # Include gaze controller stats
        if self._gaze_controller is not None:
            stats["gaze_controller"] = self._gaze_controller.get_stats()

        # Include scene context stats
        if self._scene_context is not None:
            stats["scene_context"] = self._scene_context.get_stats()

        return stats

    def to_context_str(self, max_attention_items: int = 5, max_salience_items: int = 5) -> str:
        """Generate combined context string for LLM consumption.

        Provides a concise summary of both attention and salience states
        that can be efficiently injected into LLM prompts.

        Args:
            max_attention_items: Max attention positions to include.
            max_salience_items: Max salient objects to include.

        Returns:
            Human-readable context string combining attention and salience.
        """
        lines = []

        # Attention state
        if self._attention_network is not None:
            lines.append(self._attention_network.to_context_str(max_items=max_attention_items))
        else:
            # Fallback to basic gaze history info
            lines.append("[ATTENTION STATE]")
            current = self._gaze_history.get_current_position()
            if current:
                lines.append(f"Focus: ({current[0]:.0f}, {current[1]:.0f})")
            else:
                lines.append("Focus: None")

        lines.append("")  # Blank line separator

        # Salience state
        if self._salience_network is not None:
            lines.append(self._salience_network.to_context_str(max_items=max_salience_items))
        else:
            lines.append("[SALIENCE STATE]")
            lines.append("Salience tracking not enabled")

        lines.append("")  # Blank line separator

        # Unified salience map state
        if self._salience_map_unified is not None:
            lines.append(self._salience_map_unified.to_context_str(max_targets=3))

        # Gaze controller state
        if self._gaze_controller is not None:
            lines.append("")
            lines.append("[GAZE DYNAMICS]")
            gc_stats = self._gaze_controller.get_stats()
            lines.append(f"State: {gc_stats.get('state', 'unknown')}")
            if gc_stats.get('current_target'):
                t = gc_stats['current_target']
                lines.append(f"Target: ({t[0]:.0f}, {t[1]:.0f})")
            if gc_stats.get('state') == 'fixating':
                lines.append(f"Fixation: {gc_stats.get('fixation_elapsed_ms', 0):.0f}ms / {gc_stats.get('fixation_duration_ms', 0):.0f}ms")

        # Scene context state
        if self._scene_context is not None:
            lines.append("")
            lines.append("[SCENE CONTEXT]")
            sc_stats = self._scene_context.get_stats()
            if sc_stats.get('is_scanning'):
                lines.append("Status: Scanning (scene recently changed)")
            else:
                age = sc_stats.get('scene_age_seconds', float('inf'))
                if age < 60:
                    lines.append(f"Status: Stable ({age:.0f}s since last change)")
                else:
                    lines.append("Status: Stable")

        # Add exploration hints
        lines.append("")
        lines.append("[EXPLORATION]")
        current = self._gaze_history.get_current_position()
        if current:
            opposite = compute_opposite_position(
                current,
                self._config.movement.image_width,
                self._config.movement.image_height,
            )
            lines.append(f"Current: ({current[0]:.0f}, {current[1]:.0f})")
            lines.append(f"Opposite side: ({opposite[0]:.0f}, {opposite[1]:.0f})")
            lines.append("Tip: Use look_opposite() to see the other side of the view")
        else:
            lines.append("No current focus position")

        return "\n".join(lines)

    def look_opposite(self, duration: float | None = None) -> bool:
        """Look at the opposite side of the current view.

        Useful for exploration when the agent wants to see what's behind them
        or check the other side of the scene.

        Args:
            duration: Movement duration in seconds. If None, uses dynamic calculation.

        Returns:
            True if movement was executed, False if blocked or no current position.
        """
        current = self._gaze_history.get_current_position()
        if current is None:
            logger.debug("look_opposite: no current position")
            return False

        target = compute_opposite_position(
            current,
            self._config.movement.image_width,
            self._config.movement.image_height,
        )

        # Check reachability
        if self._config.reachability_check_enabled:
            reachability = 1.0
            if self._attention_network is not None:
                reachability = self._attention_network.get_reachability(target)
            elif self._spatial_map is not None:
                reachability = self._spatial_map.get_reachability(target)

            if reachability < self._config.min_reachability_threshold:
                logger.debug("look_opposite: target unreachable (%.2f)", reachability)
                return False

        # Compute duration if not specified
        if duration is None:
            duration = compute_dynamic_duration(
                current,
                target,
                self._config.movement,
                add_jitter=True,
            )

        # Execute movement
        try:
            if hasattr(self._maxim, 'look_at_image'):
                self._maxim.look_at_image(target[0], target[1], duration=duration)
                self._gaze_history.record_gaze(target)

                if self._attention_network is not None:
                    self._attention_network.record_gaze(target, success=True)
                if self._spatial_map is not None:
                    self._spatial_map.record_movement(target, success=True)

                logger.debug("look_opposite: moved to (%.0f, %.0f)", target[0], target[1])
                return True
        except Exception as e:
            error_msg = str(e).lower()
            if "collision" in error_msg or "ik" in error_msg or "not achievable" in error_msg:
                if self._attention_network is not None:
                    self._attention_network.record_gaze(target, success=False)
                if self._spatial_map is not None:
                    self._spatial_map.record_movement(target, success=False)
            logger.warning("look_opposite failed: %s", e)

        return False

    def get_opposite_position(self) -> tuple[float, float] | None:
        """Get the opposite position from current gaze without moving.

        Useful for LLMs to know where the opposite side is.

        Returns:
            (u, v) coordinates of opposite position, or None if no current position.
        """
        current = self._gaze_history.get_current_position()
        if current is None:
            return None

        return compute_opposite_position(
            current,
            self._config.movement.image_width,
            self._config.movement.image_height,
        )

    def get_salience_target(
        self,
        mode: str = "peak",
        temperature: float = 1.0,
    ) -> tuple[float, float] | None:
        """Get a gaze target from the unified salience map.

        This provides access to the salience map's target selection for
        behaviors or LLM-directed gaze control.

        Args:
            mode: "peak" for most salient, "sample" for probabilistic.
            temperature: Sampling temperature (only for mode="sample").
                Higher = more random, Lower = more deterministic.

        Returns:
            (x, y) pixel coordinates, or None if salience map is disabled.
        """
        if self._salience_map_unified is None:
            return None

        if mode == "peak":
            return self._salience_map_unified.get_peak_target()
        elif mode == "sample":
            return self._salience_map_unified.sample_target(temperature=temperature)
        else:
            logger.warning("Unknown salience target mode: %s", mode)
            return self._salience_map_unified.get_peak_target()

    def get_salience_info(self, position: tuple[float, float]) -> dict[str, float] | None:
        """Get detailed salience breakdown for a position.

        Useful for understanding why certain positions are salient.

        Args:
            position: (x, y) pixel coordinates.

        Returns:
            Dict with component salience values, or None if map is disabled.
        """
        if self._salience_map_unified is None:
            return None

        return self._salience_map_unified.get_cell_info(position)

    def get_gaze_command(
        self,
        force_target: tuple[float, float] | None = None,
    ) -> GazeCommand | None:
        """Get a gaze command from the saccade-fixate controller.

        This provides human-like gaze dynamics by:
        1. Suggesting rapid saccades to salient targets
        2. Holding fixations for variable durations (200-800ms)
        3. Slow exploration when nothing is interesting

        Args:
            force_target: Force a saccade to this target if provided.

        Returns:
            GazeCommand if movement should occur, None to hold position.
        """
        if self._gaze_controller is None:
            return None

        current = self._gaze_history.get_current_position()
        return self._gaze_controller.update(
            current_gaze=current,
            force_target=force_target,
        )

    def is_scene_scanning(self) -> bool:
        """Check if currently in scene scanning mode after a scene change.

        Returns:
            True if in rapid scan mode, False otherwise.
        """
        if self._scene_context is None:
            return False
        return self._scene_context.is_scanning()

    def get_scene_age(self) -> float:
        """Get time since the last significant scene change.

        Returns:
            Seconds since scene changed, or inf if never changed.
        """
        if self._scene_context is None:
            return float('inf')
        return self._scene_context.get_scene_age()

    def force_scene_scan(self) -> None:
        """Force the system into scene scanning mode.

        Use when the robot has turned around or entered a new area.
        """
        if self._scene_context is not None:
            self._scene_context.force_scene_change()
            # Reset idle timer to trigger exploration
            self._last_interesting_time = 0.0
            self._next_exploration_time = time.time()
            logger.info("Forced scene scan triggered")

    def _schedule_next_exploration(self) -> float:
        """Schedule the next idle exploration at a random future time.

        Returns:
            Timestamp when next exploration should occur.
        """
        delay = random.uniform(
            self._config.idle_exploration_min_seconds,
            self._config.idle_exploration_max_seconds,
        )
        return time.time() + delay

    def _update_head_position_for_behaviors(self) -> None:
        """Update behaviors that need current head position.

        This enables ReturnToCenter to know when the head is at extreme
        positions and needs to drift back toward center. Without this,
        the robot can slowly drift to one side and get stuck at the limit.

        Also periodically syncs position from hardware to prevent drift
        between software tracking and actual head position.
        """
        # Periodically sync with hardware position (every 3 seconds)
        # This prevents accumulated drift between software tracking and reality
        now = time.time()
        sync_interval = 3.0  # seconds
        if not hasattr(self, "_last_position_sync"):
            self._last_position_sync = 0.0

        if (now - self._last_position_sync) >= sync_interval:
            sync_fn = getattr(self._maxim, "sync_head_position", None)
            if sync_fn is not None and callable(sync_fn):
                try:
                    synced = sync_fn()
                    if synced:
                        logger.debug(
                            "Periodic position sync: yaw=%.1f°, pitch=%.1f°",
                            getattr(self._maxim, "yaw", 0),
                            getattr(self._maxim, "pitch", 0),
                        )
                except Exception as e:
                    logger.debug("Periodic position sync failed: %s", e)
            self._last_position_sync = now

        # Get current head position from maxim
        yaw = getattr(self._maxim, "yaw", None)
        pitch = getattr(self._maxim, "pitch", None)

        if yaw is None or pitch is None:
            return

        try:
            yaw = float(yaw)
            pitch = float(pitch)
        except (TypeError, ValueError):
            return

        # Get full 6D position if available
        x = getattr(self._maxim, "x", 0.0) or 0.0
        y_trans = getattr(self._maxim, "y", 0.0) or 0.0
        z = getattr(self._maxim, "z", 0.0) or 0.0
        roll = getattr(self._maxim, "roll", 0.0) or 0.0

        try:
            x = float(x)
            y_trans = float(y_trans)
            z = float(z)
            roll = float(roll)
        except (TypeError, ValueError):
            x, y_trans, z, roll = 0.0, 0.0, 0.0, 0.0

        # Update behaviors that need current head position
        for behavior in self._behaviors:
            if isinstance(behavior, ReturnToCenter):
                try:
                    # Set full 6D position for translation-first approach
                    behavior.set_head_position_6d(x, y_trans, z, roll, pitch, yaw)
                except Exception as e:
                    logger.debug("Failed to update ReturnToCenter position: %s", e)
            elif isinstance(behavior, TurnAround):
                try:
                    behavior.set_head_yaw(yaw)
                    behavior.set_interests(self._interests)
                except Exception as e:
                    logger.debug("Failed to update TurnAround position: %s", e)

    def _has_interesting_content(self, detections: list[dict]) -> bool:
        """Check if any detections are interesting enough to reset idle timer.

        A detection is interesting if:
        - It's a person (class_id=0) - people are always interesting
        - It's in the interest set, OR
        - It has high novelty (not seen recently)

        Args:
            detections: List of detection dicts.

        Returns:
            True if at least one detection is interesting.
        """
        if not detections:
            return False

        # COCO class ID for person
        PERSON_CLASS_ID = 0

        for det in detections:
            class_id = det.get("class_id")

            # People are always interesting (social priority)
            if class_id == PERSON_CLASS_ID:
                return True

            # Check if in interests
            if class_id is not None and class_id in self._interests:
                return True

            # Check novelty (with class-level modulation)
            track_id = det.get("track_id")
            if track_id is not None:
                novelty = self._novelty_tracker.get_novelty(track_id, class_id=class_id)
                if novelty >= self._config.idle_novelty_threshold:
                    return True

        return False

    def _generate_exploration_target(self) -> tuple[float, float]:
        """Generate a safe target position for idle exploration.

        Uses the spatial map if available to select from known-reachable
        positions or conservatively-safe unexplored positions.

        Returns:
            (u, v) pixel coordinates for the target.
        """
        # Use spatial map for smart target selection if available
        if self._spatial_map is not None:
            target = self._spatial_map.get_safe_exploration_target(
                avoid_current=True,
                prefer_unexplored=True,
                prefer_interesting=True,
            )
            if target:
                return target

        # Fallback to random position in safe range
        u_min, u_max = self._config.idle_exploration_range
        v_min, v_max = self._config.idle_exploration_range

        u = random.uniform(u_min, u_max)
        v = random.uniform(v_min, v_max)

        # Convert to pixel coordinates (assuming 640x480 resolution)
        pixel_u = u * 640
        pixel_v = v * 480

        return (pixel_u, pixel_v)

    def _run_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            start = time.perf_counter()

            # Check auto-release of inhibition
            if self._inhibited and time.time() > self._inhibit_until:
                self.release()

            if not self._inhibited:
                self._process_tick()

            # Maintain target update rate
            elapsed = time.perf_counter() - start
            sleep_time = self._update_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            with self._stats_lock:
                self._stats["ticks"] += 1

    def _process_tick(self) -> None:
        """Single processing cycle."""
        # 1. Get latest detections (copy under lock)
        with self._detection_lock:
            detections = list(self._latest_detections)
            percept = self._latest_percept

        now = time.time()

        # 2. Check for interesting content and update idle timer
        has_interesting = self._has_interesting_content(detections) if detections else False
        if has_interesting:
            self._last_interesting_time = now
            self._next_exploration_time = self._schedule_next_exploration()

        # 3. Handle idle exploration if nothing interesting for a while
        if (
            self._config.idle_exploration_enabled
            and not has_interesting
            and now >= self._next_exploration_time
        ):
            # Generate random exploration movement
            target = self._generate_exploration_target()
            exploration_proposal = ActionProposal(
                behavior_name="idle_exploration",
                action_type="look_at",
                target=target,
                priority=0.3,  # Low priority - can be overridden by behaviors
                confidence=0.5,
                metadata={"reason": "idle_exploration"},
            )
            logger.debug(
                "Idle exploration triggered after %.1fs of no interesting content -> %s",
                now - self._last_interesting_time,
                target,
            )
            with self._stats_lock:
                self._stats["idle_explorations"] += 1
            self._execute_action(exploration_proposal)
            # Schedule next exploration
            self._next_exploration_time = self._schedule_next_exploration()
            return  # Skip normal behavior processing this tick

        if not detections:
            return

        # 4. Update novelty tracker with all track IDs (THROTTLED: 10Hz instead of 30Hz)
        if self._throttlers["novelty"].should_run():
            for d in detections:
                track_id = d.get("track_id")
                class_id = d.get("class_id")
                if track_id is not None and class_id is not None:
                    self._novelty_tracker.update_with_class(track_id, class_id)
                elif track_id is not None:
                    self._novelty_tracker.update(track_id)

        # 4b. Record observations in spatial map (THROTTLED: 5Hz instead of 30Hz)
        if self._spatial_map is not None and detections and self._throttlers["spatial_map"].should_run():
            # Get current gaze position (center of frame if unknown)
            current_pos = self._gaze_history.get_current_position()
            if current_pos is None:
                current_pos = (320.0, 240.0)  # Default to center

            # Get novelty scores for all detections (with class-level modulation)
            novelty_scores = {}
            for det in detections:
                track_id = det.get("track_id")
                if track_id is not None:
                    novelty_scores[track_id] = self._novelty_tracker.get_novelty(
                        track_id, class_id=det.get("class_id")
                    )

            self._spatial_map.record_observation(
                position=current_pos,
                detections=detections,
                novelty_scores=novelty_scores,
            )

        # 4c. Update SalienceNetwork with detections (THROTTLED: 10Hz instead of 30Hz)
        if self._salience_network is not None and detections and self._throttlers["salience"].should_run():
            self._salience_network.update_from_detections(detections, timestamp=now)

        # 4d. Update MovementDetector and inject movement scores into detections
        # This runs every frame (not throttled) for responsive motion detection
        if self._movement_detector is not None and detections:
            current_gaze = self._gaze_history.get_current_position()
            frame_center = current_gaze or (320.0, 240.0)
            movement_scores = self._movement_detector.update(detections, frame_center)

            # Inject movement scores into detections for behaviors to use
            for det in detections:
                track_id = det.get("track_id")
                if track_id is not None:
                    det["movement_score"] = movement_scores.get(track_id, 0.0)

        # 4e. Update unified SalienceMap (THROTTLED: 10Hz instead of 30Hz)
        # Combines novelty, social, movement, and spatial factors
        if self._salience_map_unified is not None and self._throttlers["salience"].should_run():
            self._salience_map_unified.update(
                detections=detections,
                gaze_history=self._gaze_history,
                novelty_tracker=self._novelty_tracker,
                movement_detector=self._movement_detector,
                current_time=now,
            )

        # 4f. Update SceneContextDetector for scene change detection
        # Triggers exploration when scene changes significantly
        scene_changed = False
        if self._scene_context is not None and detections:
            head_yaw = getattr(self._maxim, "yaw", None)
            head_pitch = getattr(self._maxim, "pitch", None)
            try:
                head_yaw = float(head_yaw) if head_yaw is not None else None
                head_pitch = float(head_pitch) if head_pitch is not None else None
            except (TypeError, ValueError):
                head_yaw, head_pitch = None, None

            scene_changed = self._scene_context.update(
                detections=detections,
                head_yaw=head_yaw,
                head_pitch=head_pitch,
            )

            if scene_changed:
                logger.info("Scene change detected - triggering exploration")
                # Reset idle timer to trigger immediate exploration
                self._last_interesting_time = 0.0
                self._next_exploration_time = now
                # Note activity in gaze controller
                if self._gaze_controller is not None:
                    self._gaze_controller.note_activity()

        # 4g. Update ReturnToCenter behavior with current head position
        # This enables drift correction by allowing the behavior to know when
        # the head is at extreme positions and needs to return toward center.
        self._update_head_position_for_behaviors()

        # 5. Build behavior state
        state = BehaviorState(
            inhibited_behaviors=frozenset(
                name for name, mod in self._behavior_overrides.items() if mod <= 0
            ),
            priority_modifiers=self._behavior_overrides,
            interests=self._interests,
            frame_timestamp=now,
            salience_map=self._salience_map_unified,
            focus_learner=self._focus_learner,
        )

        # 6. Run all behaviors (THROTTLED: 15Hz instead of 30Hz for full evaluation)
        proposals: list[ActionProposal] = []
        if self._throttlers["behaviors"].should_run():
            for behavior in self._behaviors:
                if not behavior.enabled:
                    continue
                if behavior.name in state.inhibited_behaviors:
                    continue

                try:
                    proposal = behavior.evaluate(detections, state)
                    if proposal:
                        # Apply priority overrides
                        modifier = self._behavior_overrides.get(behavior.name, 1.0)
                        if modifier != 1.0:
                            proposal = ActionProposal(
                                behavior_name=proposal.behavior_name,
                                action_type=proposal.action_type,
                                target=proposal.target,
                                priority=proposal.priority * modifier,
                                confidence=proposal.confidence,
                                metadata=proposal.metadata,
                            )
                        proposals.append(proposal)
                except Exception as e:
                    logger.warning("Behavior '%s' failed: %s", behavior.name, e)

        # 7. Arbitrate
        if proposals:
            # Debug: log all proposals
            if logger.isEnabledFor(logging.DEBUG):
                proposal_summary = [(p.behavior_name, p.priority, p.confidence, p.target) for p in proposals]
                logger.debug("Behavior proposals: %s", proposal_summary)

            winner = self._arbiter.select(proposals)
            if winner:
                logger.debug(
                    "Arbiter selected: %s (priority=%.2f, conf=%.2f) -> %s",
                    winner.behavior_name, winner.priority, winner.confidence, winner.target
                )
                with self._stats_lock:
                    self._stats["actions_proposed"] += 1
                self._execute_action(winner)

        # 8. Evaluate percept escalation (separate from behavior execution)
        if percept:
            self._evaluate_escalation(percept)

    def _execute_action(self, proposal: ActionProposal) -> None:
        """Execute the winning action proposal.

        Args:
            proposal: The action to execute.
        """
        # Gate through FearAgent if enabled
        if self._config.fear_gate_enabled and self._fear_agent:
            try:
                # Build action signature for pain prediction
                action_sig = ""
                if proposal.target:
                    # Estimate movement magnitude from target
                    current = self._gaze_history.get_current_position()
                    if current:
                        du = abs(proposal.target[0] - current[0])
                        dv = abs(proposal.target[1] - current[1])
                        # Rough conversion: pixels to degrees (~0.1 deg/pixel)
                        dyaw = du * 0.1
                        dpitch = dv * 0.1
                        action_sig = f"look_at:dy={dyaw:.0f}:dp={dpitch:.0f}"

                review = self._fear_agent.review_action(
                    action_type="dn_movement",
                    action_params={
                        "behavior": proposal.behavior_name,
                        "target": proposal.target,
                        "priority": proposal.priority,
                        "action_signature": action_sig,
                    },
                    agent_id="default_network",
                    pain_bridge=self._pain_bridge,
                )
                if not review.allow:
                    logger.debug(
                        "FearAgent blocked DN action: %s (risk=%s)",
                        proposal.behavior_name,
                        getattr(review, 'risk_level', 'unknown'),
                    )
                    with self._stats_lock:
                        self._stats["actions_blocked"] += 1
                    self._publish_action(proposal, executed=False)
                    return
            except Exception as e:
                logger.warning("FearAgent check failed: %s", e)

        # Gate through salience check if enabled (inhibition-of-return)
        if self._config.gaze_salience_enabled and proposal.target:
            should_move, salience, reason = self._gaze_history.should_move(proposal.target)
            if not should_move:
                logger.debug(
                    "Salience blocked DN action: %s -> %s (salience=%.2f, reason=%s)",
                    proposal.behavior_name,
                    proposal.target,
                    salience,
                    reason,
                )
                with self._stats_lock:
                    self._stats["actions_salience_blocked"] += 1
                self._publish_action(proposal, executed=False)
                return

        # Gate through reachability check if enabled
        if self._config.reachability_check_enabled and proposal.target:
            reachability = 1.0  # Default to reachable

            # Check AttentionNetwork first (has learned reachability)
            if self._attention_network is not None:
                reachability = self._attention_network.get_reachability(proposal.target)
            # Fall back to SpatialMap if available
            elif self._spatial_map is not None:
                reachability = self._spatial_map.get_reachability(proposal.target)

            if reachability < self._config.min_reachability_threshold:
                logger.debug(
                    "Reachability blocked DN action: %s -> %s (reach=%.2f < %.2f)",
                    proposal.behavior_name,
                    proposal.target,
                    reachability,
                    self._config.min_reachability_threshold,
                )
                with self._stats_lock:
                    self._stats["actions_reachability_blocked"] += 1
                self._publish_action(proposal, executed=False)
                return

        # Compute dynamic duration based on distance (+ random jitter)
        current_pos = self._gaze_history.get_current_position()
        if proposal.target:
            duration = compute_dynamic_duration(
                current_pos,
                proposal.target,
                self._config.movement,
                add_jitter=True,
            )
        else:
            duration = self._config.movement.base_duration

        # Execute the movement
        executed = False
        movement_failed = False  # Track IK/collision failures separately

        try:
            if proposal.action_type == "look_at" and proposal.target:
                u, v = proposal.target
                if hasattr(self._maxim, 'look_at_image'):
                    self._maxim.look_at_image(u, v, duration=duration)
                    executed = True
            elif proposal.action_type == "scan":
                # For exploration, prefer look_at_image if target is provided
                # This allows smooth movement to exploration waypoints
                if proposal.target and hasattr(self._maxim, 'look_at_image'):
                    u, v = proposal.target
                    # Use longer duration for leisurely exploration
                    explore_duration = max(0.5, duration * 0.8)
                    self._maxim.look_at_image(u, v, duration=explore_duration)
                    executed = True
                elif hasattr(self._maxim, 'move_relative'):
                    # Fallback to delta-based movement (for ReturnToCenter, etc.)
                    delta = proposal.metadata.get("delta", (0, 0))
                    self._maxim.move_relative(delta)
                    executed = True
            elif proposal.action_type == "track" and proposal.target:
                # Smooth tracking - use slightly shorter duration
                u, v = proposal.target
                if hasattr(self._maxim, 'look_at_image'):
                    track_duration = max(0.1, duration * 0.5)  # Half duration for tracking

                    # DEBUG: Log social tracking request
                    if proposal.behavior_name == "social":
                        cur_yaw = float(getattr(self._maxim, "yaw", 0.0) or 0.0)
                        # Target RIGHT of center (u > 320) → yaw decreases; LEFT → yaw increases
                        direction = "RIGHT" if u > 320 else "LEFT" if u < 320 else "CENTER"
                        logger.debug(
                            "SocialTrack: target=(%.0f,%.0f) %s, yaw=%.1f, track_id=%s",
                            u, v, direction, cur_yaw, proposal.metadata.get("track_id")
                        )

                    self._maxim.look_at_image(u, v, duration=track_duration)
                    executed = True
                    # Note: Movement is async (enqueued to motor worker), so we can't
                    # check post-movement position here - it hasn't happened yet.
            elif proposal.action_type == "turn_around":
                # Body rotation when head is at yaw limits
                turn_angle = proposal.metadata.get("turn_angle", 90.0)
                turn_duration = proposal.metadata.get("duration", 5.0)
                if hasattr(self._maxim, 'turn_around'):
                    self._maxim.turn_around(turn_angle, duration=turn_duration)
                    executed = True
                    logger.info(
                        "Executing turn_around: %.0f° over %.1fs",
                        turn_angle, turn_duration
                    )
        except Exception as e:
            error_msg = str(e).lower()
            # Check if this is an IK/collision error
            if "collision" in error_msg or "ik" in error_msg or "not achievable" in error_msg:
                movement_failed = True
                logger.debug("Movement to %s failed (IK/collision): %s", proposal.target, e)
            else:
                logger.warning("Failed to execute DN action: %s", e)

        # Record movement result in spatial map for learning
        if self._spatial_map is not None and proposal.target:
            if executed:
                self._spatial_map.record_movement(proposal.target, success=True)
            elif movement_failed:
                self._spatial_map.record_movement(proposal.target, success=False)

        # Record movement in AttentionNetwork for reachability tracking
        if self._attention_network is not None and proposal.target:
            if executed:
                self._attention_network.record_gaze(proposal.target, success=True)
            elif movement_failed:
                self._attention_network.record_gaze(proposal.target, success=False)

        if executed:
            with self._stats_lock:
                self._stats["actions_executed"] += 1

            # Mark the focused object as focused (affects novelty decay)
            track_id = proposal.metadata.get("track_id")
            if track_id is not None and self._novelty_tracker is not None:
                self._novelty_tracker.focus(track_id)

            # Record gaze for salience tracking (inhibition-of-return)
            if self._config.gaze_salience_enabled and proposal.target:
                self._gaze_history.record_gaze(proposal.target)

        # Notify callbacks and publish
        self._publish_action(proposal, executed=executed)

    def _publish_action(self, proposal: ActionProposal, executed: bool) -> None:
        """Publish action to bus and notify callbacks.

        Args:
            proposal: The action proposal.
            executed: Whether it was actually executed.
        """
        # Notify callbacks
        for callback in self._on_action_callbacks:
            try:
                callback(proposal)
            except Exception as e:
                logger.warning("Action callback failed: %s", e)

        # Publish to bus
        if self._config.publish_actions and self._bus:
            action_msg = DNActionProposal(
                behavior=proposal.behavior_name,
                action_type=proposal.action_type,
                target=proposal.target,
                timestamp=time.time(),
                confidence=proposal.confidence,
                was_executed=executed,
            )
            try:
                self._bus.publish(action_msg)
            except Exception as e:
                logger.debug("Failed to publish DNActionProposal: %s", e)

    def _evaluate_escalation(self, percept: "Percept") -> None:
        """Evaluate whether percept should be escalated.

        Args:
            percept: The percept to evaluate.
        """
        result = self._gate.evaluate(percept)

        if result.escalated:
            with self._stats_lock:
                self._stats["percepts_escalated"] += 1

            filtered = self._gate.create_filtered_percept(percept, result)

            # Notify callbacks
            for callback in self._on_escalation_callbacks:
                try:
                    callback(filtered)
                except Exception as e:
                    logger.warning("Escalation callback failed: %s", e)

            # Publish to bus
            if self._bus:
                try:
                    self._bus.publish(filtered)
                except Exception as e:
                    logger.debug("Failed to publish FilteredPercept: %s", e)
        else:
            with self._stats_lock:
                self._stats["percepts_filtered"] += 1

        # Periodically adapt thresholds
        self._gate.adapt_thresholds()


__all__ = [
    "DefaultNetwork",
    "DefaultNetworkConfig",
    # Re-exported for convenience
    "GazeCommand",
    "GazeState",
]
