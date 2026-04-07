"""Configuration system for the Default Network.

Provides dataclasses for configuring DN behaviors and a YAML loader
for runtime configuration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default config file location (data/util/default_network.yaml)
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "data" / "util" / "default_network.yaml"


@dataclass
class OrientingConfig:
    """Configuration for OrientingResponse behavior."""

    enabled: bool = True
    priority: float = 0.8
    novelty_threshold: float = 1.2
    min_confidence: float = 0.4
    cooldown_seconds: float = 0.5


@dataclass
class SocialConfig:
    """Configuration for SocialAttention behavior."""

    enabled: bool = True
    priority: float = 0.9
    prefer_faces: bool = True
    tracking_hysteresis: float = 0.1
    cooldown_seconds: float = 0.2


@dataclass
class MotionConfig:
    """Configuration for MotionTracking behavior."""

    enabled: bool = True
    priority: float = 0.7
    velocity_threshold: float = 50.0
    history_seconds: float = 0.5
    prediction_seconds: float = 0.1
    cooldown_seconds: float = 0.1


@dataclass
class StartleConfig:
    """Configuration for StartleResponse behavior."""

    enabled: bool = True
    priority: float = 0.95
    peripheral_threshold: float = 0.7
    appearance_window: float = 0.3
    min_confidence: float = 0.5
    cooldown_seconds: float = 2.0


@dataclass
class IdleScanConfig:
    """Configuration for IdleScan behavior."""

    enabled: bool = True
    priority: float = 0.2
    idle_timeout: float = 5.0
    scan_speed: float = 0.5
    scan_amplitude: float = 200.0
    cooldown_seconds: float = 0.5


@dataclass
class MicrosaccadesConfig:
    """Configuration for Microsaccades behavior."""

    enabled: bool = True
    priority: float = 0.1
    fixation_timeout: float = 2.0
    amplitude: float = 20.0
    cooldown_seconds: float = 0.3


@dataclass
class ReturnToCenterConfig:
    """Configuration for ReturnToCenter behavior.

    The threshold determines when the behavior activates (fraction of max range).
    Lower threshold = activates earlier, preventing drift from accumulating.
    """

    enabled: bool = True
    priority: float = 0.2  # Slightly higher to compete with idle behaviors
    threshold: float = 0.6  # Activate at 60% of max range (was 70%)
    return_speed: float = 0.35  # Slightly faster return
    cooldown_seconds: float = 0.8  # Slightly shorter cooldown


@dataclass
class TurnAroundConfig:
    """Configuration for TurnAround behavior.

    Rotates the body when the head is at its yaw limit and there's
    something interesting beyond what the head can see.
    """

    enabled: bool = True
    priority: float = 0.3  # Higher than idle, lower than tracking
    yaw_threshold: float = 0.85  # Trigger at 85% of yaw limit
    edge_threshold: float = 0.15  # Detection within 15% of frame edge
    turn_angle: float = 90.0  # Degrees to rotate body
    base_duration: float = 5.0  # Seconds for the turn
    duration_jitter: float = 1.0  # Random ± seconds
    cooldown_seconds: float = 10.0  # Don't turn too frequently


@dataclass
class ArbiterSettings:
    """Configuration for the PriorityArbiter."""

    hysteresis_bonus: float = 0.1
    min_switch_interval: float = 0.3
    score_threshold: float = 0.1


@dataclass
class GateSettings:
    """Configuration for the ThalamicGate."""

    novelty_threshold: float = 0.7
    salience_threshold: float = 0.6
    anomaly_threshold: float = 0.7
    adaptive: bool = True
    # Adaptive threshold settings
    min_threshold: float = 0.3
    max_threshold: float = 0.95
    adaptation_rate: float = 0.1
    escalation_rate_target: float = 0.05


@dataclass
class InhibitionSettings:
    """Configuration for inhibition behavior."""

    auto_release_timeout: float = 5.0
    inhibit_during_tool_execution: bool = False


@dataclass
class PainDetectionConfig:
    """Configuration for pain detection and aversive learning.

    Pain detection monitors movement velocity and acceleration,
    generating pain signals when thresholds are exceeded. These
    signals feed into NAc for causal learning.
    """

    enabled: bool = True
    angular_velocity_threshold: float = 100.0  # deg/sec triggers pain
    translation_velocity_threshold: float = 50.0  # mm/sec triggers pain
    angular_acceleration_threshold: float = 200.0  # deg/sec² triggers pain
    reversal_threshold: int = 3  # Direction reversals in window triggers pain
    pain_cooldown_seconds: float = 0.5  # Min time between pain signals
    prediction_threshold: float = 0.4  # Confidence for pain prediction


@dataclass
class GazeControllerSettings:
    """Configuration for human-like gaze controller.

    Controls saccade-fixate dynamics for natural eye movement patterns.
    """

    enabled: bool = True
    min_fixation_ms: float = 200.0
    max_fixation_ms: float = 800.0
    mean_fixation_ms: float = 350.0
    saccade_speed_multiplier: float = 2.0
    exploration_trigger_seconds: float = 2.0


@dataclass
class SceneContextSettings:
    """Configuration for scene context detection.

    Detects significant scene changes to trigger exploration behavior.
    """

    enabled: bool = True
    change_threshold: float = 0.4  # 40% change triggers scene scan
    scene_stability_seconds: float = 2.0
    position_change_threshold: float = 30.0  # degrees yaw change


@dataclass
class BehaviorsConfig:
    """Configuration for all behaviors."""

    orienting: OrientingConfig = field(default_factory=OrientingConfig)
    social: SocialConfig = field(default_factory=SocialConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    startle: StartleConfig = field(default_factory=StartleConfig)
    idle_scan: IdleScanConfig = field(default_factory=IdleScanConfig)
    microsaccades: MicrosaccadesConfig = field(default_factory=MicrosaccadesConfig)
    return_to_center: ReturnToCenterConfig = field(default_factory=ReturnToCenterConfig)
    turn_around: TurnAroundConfig = field(default_factory=TurnAroundConfig)


@dataclass
class DNConfig:
    """Complete configuration for the Default Network."""

    enabled: bool = True
    update_hz: float = 30.0
    publish_actions: bool = True
    fear_gate_enabled: bool = True

    behaviors: BehaviorsConfig = field(default_factory=BehaviorsConfig)
    arbiter: ArbiterSettings = field(default_factory=ArbiterSettings)
    gate: GateSettings = field(default_factory=GateSettings)
    inhibition: InhibitionSettings = field(default_factory=InhibitionSettings)
    pain: PainDetectionConfig = field(default_factory=PainDetectionConfig)
    gaze_controller: GazeControllerSettings = field(default_factory=GazeControllerSettings)
    scene_context: SceneContextSettings = field(default_factory=SceneContextSettings)


def _merge_dict_into_dataclass(data: dict[str, Any], dc_instance: Any) -> None:
    """Merge dictionary values into a dataclass instance.

    Args:
        data: Dictionary with values to merge.
        dc_instance: Dataclass instance to update.
    """
    for f in fields(dc_instance):
        if f.name in data:
            value = data[f.name]
            current = getattr(dc_instance, f.name)

            # If the field is itself a dataclass and value is a dict, recurse
            if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
                _merge_dict_into_dataclass(value, current)
            else:
                setattr(dc_instance, f.name, value)


def load_dn_config(path: Path | str | None = None) -> DNConfig:
    """Load Default Network configuration from YAML file.

    Args:
        path: Path to YAML config file. If None, uses default location.

    Returns:
        DNConfig instance with loaded values (or defaults if file not found).
    """
    config = DNConfig()

    if path is None:
        path = DEFAULT_CONFIG_PATH

    path = Path(path)

    if not path.exists():
        logger.debug("DN config file not found at %s, using defaults", path)
        return config

    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed, using default DN config")
        return config

    try:
        with open(path) as f:
            data = yaml.safe_load(f)

        if data and isinstance(data, dict):
            # Handle nested 'default_network' key
            if "default_network" in data:
                data = data["default_network"]

            _merge_dict_into_dataclass(data, config)
            logger.info("Loaded DN config from %s", path)

    except Exception as e:
        logger.warning("Failed to load DN config from %s: %s", path, e)

    return config


def save_dn_config(config: DNConfig, path: Path | str) -> None:
    """Save Default Network configuration to YAML file.

    Args:
        config: DNConfig instance to save.
        path: Path to write YAML file.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required to save config")

    from dataclasses import asdict

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {"default_network": asdict(config)}

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    logger.info("Saved DN config to %s", path)


def create_behaviors_from_config(
    config: BehaviorsConfig,
    novelty_tracker: Any = None,
    frame_size: tuple[int, int] = (640, 480),
    bounds_learner: Any = None,
) -> list:
    """Create behavior instances from configuration.

    Args:
        config: BehaviorsConfig with settings for each behavior.
        novelty_tracker: ThreadSafeNoveltyTracker for orienting behavior.
        frame_size: Video frame dimensions.
        bounds_learner: Optional WorkspaceBoundsLearner for learned limits.

    Returns:
        List of configured Behavior instances.
    """
    from maxim.default_network.behaviors import (
        OrientingResponse,
        SocialAttention,
        MotionTracking,
        StartleResponse,
        IdleScan,
        Microsaccades,
        ReturnToCenter,
    )
    from maxim.default_network.behaviors.turn_around import TurnAround

    behaviors = []

    # Orienting
    if config.orienting.enabled:
        b = OrientingResponse(
            novelty_tracker=novelty_tracker,
            novelty_threshold=config.orienting.novelty_threshold,
            min_confidence=config.orienting.min_confidence,
        )
        b.base_priority = config.orienting.priority
        b.cooldown_seconds = config.orienting.cooldown_seconds
        behaviors.append(b)

    # Social
    if config.social.enabled:
        b = SocialAttention(
            prefer_faces=config.social.prefer_faces,
            tracking_hysteresis=config.social.tracking_hysteresis,
        )
        b.base_priority = config.social.priority
        b.cooldown_seconds = config.social.cooldown_seconds
        behaviors.append(b)

    # Motion
    if config.motion.enabled:
        b = MotionTracking(
            velocity_threshold=config.motion.velocity_threshold,
            history_seconds=config.motion.history_seconds,
            prediction_seconds=config.motion.prediction_seconds,
        )
        b.base_priority = config.motion.priority
        b.cooldown_seconds = config.motion.cooldown_seconds
        behaviors.append(b)

    # Startle
    if config.startle.enabled:
        b = StartleResponse(
            peripheral_threshold=config.startle.peripheral_threshold,
            appearance_window=config.startle.appearance_window,
            min_confidence=config.startle.min_confidence,
            frame_size=frame_size,
        )
        b.base_priority = config.startle.priority
        b.cooldown_seconds = config.startle.cooldown_seconds
        behaviors.append(b)

    # Idle scan
    if config.idle_scan.enabled:
        b = IdleScan(
            idle_timeout=config.idle_scan.idle_timeout,
            # Note: scan_speed/scan_amplitude in config are not used by IdleScan
            # IdleScan uses zone-based exploration with randomized dwell times
        )
        b.base_priority = config.idle_scan.priority
        b.cooldown_seconds = config.idle_scan.cooldown_seconds
        behaviors.append(b)

    # Microsaccades
    if config.microsaccades.enabled:
        b = Microsaccades(
            fixation_timeout=config.microsaccades.fixation_timeout,
            amplitude=config.microsaccades.amplitude,
        )
        b.base_priority = config.microsaccades.priority
        b.cooldown_seconds = config.microsaccades.cooldown_seconds
        behaviors.append(b)

    # Return to center
    if config.return_to_center.enabled:
        b = ReturnToCenter(
            threshold=config.return_to_center.threshold,
            return_speed=config.return_to_center.return_speed,
            bounds_learner=bounds_learner,
        )
        b.base_priority = config.return_to_center.priority
        b.cooldown_seconds = config.return_to_center.cooldown_seconds
        behaviors.append(b)

    # Turn around (body rotation when head at yaw limit)
    if config.turn_around.enabled:
        b = TurnAround(
            novelty_tracker=novelty_tracker,
            yaw_threshold=config.turn_around.yaw_threshold,
            edge_threshold=config.turn_around.edge_threshold,
            turn_angle=config.turn_around.turn_angle,
            base_duration=config.turn_around.base_duration,
            duration_jitter=config.turn_around.duration_jitter,
        )
        b.base_priority = config.turn_around.priority
        b.cooldown_seconds = config.turn_around.cooldown_seconds
        behaviors.append(b)

    return behaviors


__all__ = [
    # Main config
    "DNConfig",
    "BehaviorsConfig",
    # Behavior configs
    "OrientingConfig",
    "SocialConfig",
    "MotionConfig",
    "StartleConfig",
    "IdleScanConfig",
    "MicrosaccadesConfig",
    "ReturnToCenterConfig",
    "TurnAroundConfig",
    # Component configs
    "ArbiterSettings",
    "GateSettings",
    "InhibitionSettings",
    "PainDetectionConfig",
    "GazeControllerSettings",
    "SceneContextSettings",
    # Functions
    "load_dn_config",
    "save_dn_config",
    "create_behaviors_from_config",
    "DEFAULT_CONFIG_PATH",
]
