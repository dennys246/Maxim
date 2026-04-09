"""Runtime capability detection for adaptive behavior."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class NodeLoad:
    """Live load metrics for a Maxim node (populated by heartbeat).

    Used by the Pecking Order Graph to route inference requests to the
    least-loaded node. All fields default to 0.0 so nodes that don't
    report load are treated as idle.
    """

    gpu_util_pct: float = 0.0  # GPU compute utilization (0-100)
    vram_used_gb: float = 0.0  # VRAM currently allocated
    queue_depth: int = 0  # Inference requests queued
    ram_pressure_pct: float = 0.0  # System RAM usage (0-100)
    thermal_throttle: bool = False  # GPU thermal throttling active
    timestamp: float = 0.0  # When this snapshot was taken (monotonic)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_util_pct": self.gpu_util_pct,
            "vram_used_gb": self.vram_used_gb,
            "queue_depth": self.queue_depth,
            "ram_pressure_pct": self.ram_pressure_pct,
            "thermal_throttle": self.thermal_throttle,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NodeLoad:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


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
    # Compute resources (populated by detect_compute_resources(), 0.0 when unknown).
    vram_gb: float = 0.0
    ram_gb: float = 0.0

    # Graph topology (POG-0 prep — wire format stability before v0.2.0)
    node_role: str = "solo"  # "solo" | "leader" | "peer" | "mother"

    # Live load metrics (POG-0a — populated by heartbeat, None when unknown)
    node_load: NodeLoad | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RuntimeCapabilities:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def detect_compute_resources() -> tuple[bool, str | None, float, float]:
    """Probe hardware and return (has_gpu, gpu_type, vram_gb, ram_gb).

    Pure function — does not mutate global state. Safe to call before or after
    any Blackwell CUDA-visibility decisions; respects CUDA_VISIBLE_DEVICES.
    Returns zeros/None on any detection failure.
    """
    has_gpu = False
    gpu_type: str | None = None
    vram_gb = 0.0
    ram_gb = 0.0

    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            has_gpu = True
            props = torch.cuda.get_device_properties(0)
            gpu_type = props.name
            vram_gb = props.total_memory / (1024**3)
        else:
            mps = getattr(getattr(torch, "backends", None), "mps", None)
            if mps is not None and getattr(mps, "is_available", lambda: False)():
                has_gpu = True
                gpu_type = "mps"
    except Exception:
        pass

    try:
        import psutil

        ram_gb = psutil.virtual_memory().total / (1024**3)
    except Exception:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        ram_gb = int(line.split()[1]) / (1024**2)
                        break
        except Exception:
            pass

    return has_gpu, gpu_type, vram_gb, ram_gb
