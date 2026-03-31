"""Runtime capability detection for adaptive behavior."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class RuntimeCapabilities:
    """What hardware/software is available. Can change during runtime."""
    has_robot: bool = False
    has_gpu: bool = False
    has_vision: bool = False
    has_audio: bool = False
    has_motor: bool = False
    has_display: bool = False
    has_network: bool = False
    robot_type: str | None = None
    gpu_type: str | None = None
    connected_devices: list[str] = field(default_factory=list)
