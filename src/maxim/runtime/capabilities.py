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


def _is_apple_silicon() -> bool:
    """Return True on macOS arm64 (M-series) machines.

    Used to gate unified-memory VRAM assumptions: on Apple Silicon the
    GPU addresses the same RAM pool as the CPU, so we can treat a
    fraction of total RAM as effective VRAM. On Intel Macs running
    legacy PyTorch MPS (with discrete AMD GPUs), this assumption is
    wrong — actual Metal-accessible VRAM is the discrete card's size,
    not system RAM.
    """
    import platform

    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _mps_effective_vram_gb(
    ram_gb: float,
    *,
    headroom_factor: float | None = None,
    ceiling_gb: float = 64.0,
) -> float:
    """Compute effective Metal-accessible VRAM on Apple Silicon.

    Apple's Metal documentation recommends leaving ~25% of unified
    memory free for the OS and other GPU clients, so the default
    headroom_factor of 0.75 yields ~18GB effective on a 24GB Mac.

    The ceiling caps absurd reports on huge-RAM Macs: a 192GB M2 Ultra
    shouldn't advertise 144GB of effective VRAM because no currently
    supported quantized model uses that much, and the number is
    misleading for doctor output and future tier additions.

    Environment override:
        MAXIM_MPS_VRAM_HEADROOM — float in [0.25, 0.85]. Defaults to
        0.75. Lower values reserve less for the OS (aggressive, good
        on idle machines); higher values reserve more (safe, good on
        machines with Chrome/Xcode/etc. running).

    Args:
        ram_gb: Total system RAM in gigabytes.
        headroom_factor: Override for the Metal headroom fraction.
            ``None`` reads from the env var with a 0.75 default.
        ceiling_gb: Maximum reported effective VRAM, regardless of
            RAM size. Defaults to 64 GB.

    Returns:
        Effective VRAM in gigabytes. Always non-negative. Returns 0.0
        on invalid inputs (ram_gb <= 0).
    """
    if ram_gb <= 0:
        return 0.0
    if headroom_factor is None:
        try:
            import os

            raw = os.environ.get("MAXIM_MPS_VRAM_HEADROOM", "0.75")
            headroom_factor = float(raw)
        except (ValueError, TypeError):
            headroom_factor = 0.75
    # Clamp to sane range. Below 0.25 risks starving the OS; above 0.85
    # is so conservative the user gets no benefit from unified memory.
    headroom_factor = max(0.25, min(headroom_factor, 0.85))
    effective = ram_gb * headroom_factor
    return min(effective, ceiling_gb)


def _detect_ram_gb() -> float:
    """Query total system RAM via psutil, falling back to /proc/meminfo."""
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024**2)
    except Exception:
        pass
    return 0.0


def detect_compute_resources() -> tuple[bool, str | None, float, float]:
    """Probe hardware and return (has_gpu, gpu_type, vram_gb, ram_gb).

    Pure function — does not mutate global state. Safe to call before or after
    any Blackwell CUDA-visibility decisions; respects CUDA_VISIBLE_DEVICES.
    Returns zeros/None on any detection failure.

    On Apple Silicon (MPS + arm64), ``vram_gb`` is set to a fraction
    of ``ram_gb`` via :func:`_mps_effective_vram_gb` because unified
    memory means the GPU can address most of the RAM pool. On Intel
    Macs running PyTorch MPS (legacy, deprecated), ``vram_gb`` stays
    at 0.0 — the unified-memory assumption doesn't hold there.
    """
    has_gpu = False
    gpu_type: str | None = None
    vram_gb = 0.0

    # Detect RAM first so the MPS branch below can derive effective
    # VRAM from unified memory.
    ram_gb = _detect_ram_gb()

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
                if _is_apple_silicon():
                    vram_gb = _mps_effective_vram_gb(ram_gb)
                # Intel Macs leave vram_gb at 0.0. PyTorch MPS on Intel
                # with discrete AMD GPUs is deprecated and we don't want
                # to guess at discrete-VRAM sizing.
    except Exception:
        pass

    return has_gpu, gpu_type, vram_gb, ram_gb


def derive_media_capabilities(robot: object) -> tuple[bool, bool]:
    """Derive ``(has_vision, has_audio)`` from a connected robot's ACTUAL media state.

    Capability truth (2026-08-01): the runtime used to hardcode both True
    for any connected robot. Under the Reachy SDK's
    ``media_backend: no_media`` the media manager exists with
    ``camera=None`` / ``audio=None``, and every capture-loop poll then logs
    the SDK's "Camera/Audio system is not initialized." WARNING — dozens
    per second across the CaptureManager, frame-observation, and
    audio-capture loops.

    Introspection is duck-typed through ``robot.mini.media`` (the Reachy
    controller's SDK handle): a media manager exposing ``camera`` /
    ``audio`` attributes answers truthfully; anything that cannot be
    introspected (third-party controllers, future SDK shapes) keeps the
    permissive ``True`` — this helper only ever DOWNGRADES on positive
    evidence of an absent device, so existing robots see no behavior
    change.

    Controllers WITHOUT an SDK handle (``mini`` absent/None) fall back to
    the :class:`~maxim.hardware.controller.RobotController` stream
    surface: a connected controller whose ``get_video_stream()`` /
    ``get_audio_stream()`` returns ``None`` has positively declared the
    device absent (the ABC defines those getters as THE media surface),
    so the flag downgrades — e.g. ``SimulatedController(video_enabled=
    False)`` in an audio-only scenario, where a permissive ``True`` would
    leave DN idle visual exploration fabricating gaze targets from a
    camera that does not exist (the 2026-08-01 capability-truth lesson,
    extended to the sim). A controller lacking the getters entirely keeps
    ``True`` (no evidence).
    """
    media = getattr(getattr(robot, "mini", None), "media", None)
    has_vision = True
    has_audio = True
    if media is not None:
        if hasattr(media, "camera"):
            has_vision = media.camera is not None
        if hasattr(media, "audio"):
            has_audio = media.audio is not None
        return has_vision, has_audio
    # Stream-surface fallback is only meaningful POST-connect (streams are
    # created in connect()); a pre-connect probe would read None and
    # downgrade on no evidence (review fold). Both production call sites
    # are post-connect; this guard makes the precondition structural.
    try:
        if not robot.is_connected():  # type: ignore[union-attr]
            return has_vision, has_audio
    except Exception:
        return has_vision, has_audio
    get_video = getattr(robot, "get_video_stream", None)
    if callable(get_video):
        try:
            has_vision = get_video() is not None
        except Exception:
            has_vision = True
    get_audio = getattr(robot, "get_audio_stream", None)
    if callable(get_audio):
        try:
            has_audio = get_audio() is not None
        except Exception:
            has_audio = True
    return has_vision, has_audio
