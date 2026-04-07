"""RTSPStreamingSkill — stream Reachy camera frames as RTSP.

An atomic skill that exposes the Reachy Mini's camera as an RTSP feed
via ffmpeg + MediaMTX. Reusable across any protocol that needs video
streaming (ShredderSegmenter, security monitoring, remote observation).

Prerequisites: ffmpeg installed. MediaMTX is auto-started if not
already running (set auto_start_mediamtx=False to disable).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Any

from maxim.skills.base import Skill, SkillConfig, SkillResult, SkillState
from maxim.tools.base import Tool, ToolOutput

__all__ = ["RTSPStreamingSkill", "RTSPStreamingConfig"]


@dataclass(frozen=True)
class RTSPStreamingConfig(SkillConfig):
    """Configuration for the RTSP streaming skill."""

    rtsp_url: str = "rtsp://localhost:8554/reachy"
    fps: int = 20
    preset: str = "ultrafast"
    tune: str = "zerolatency"
    gop_size: int = 30
    bitrate: str = "2M"


class RTSPStreamingSkill(Skill):
    """Streams Reachy camera frames to an RTSP endpoint."""

    def __init__(self, config: RTSPStreamingConfig | None = None):
        self._config = config or RTSPStreamingConfig()
        self._bridge: Any = None  # RTSPBridge instance (lazy)
        self._maxim: Any = None
        self._state = SkillState.IDLE
        self._last_result: SkillResult | None = None

    @property
    def name(self) -> str:
        return "rtsp_streaming"

    @property
    def description(self) -> str:
        return "Stream Reachy camera as RTSP feed via ffmpeg + MediaMTX"

    def tools(self) -> list[Tool]:
        """Provide start/stop tools for agentic control."""
        return [
            _StartStreamTool(self),
            _StopStreamTool(self),
        ]

    def can_activate(self, maxim: Any) -> tuple[bool, str]:
        """Check prerequisites: ffmpeg installed, robot connected."""
        if not shutil.which("ffmpeg"):
            return False, "ffmpeg not found in PATH (install with: apt install ffmpeg)"
        if not hasattr(maxim, "mini") or maxim.mini is None:
            return False, "No robot connection — cannot access camera"
        return True, ""

    def activate(self, maxim: Any, context: dict[str, Any] | None = None) -> SkillResult:
        import logging as _logging
        from maxim.tools.rtsp_bridge import RTSPBridge, RTSPBridgeConfig

        log = getattr(maxim, "log", _logging.getLogger(__name__))
        self._state = SkillState.ACTIVATING
        self._maxim = maxim
        bridge_config = RTSPBridgeConfig(
            rtsp_url=self._config.rtsp_url,
            fps=self._config.fps,
            preset=self._config.preset,
            tune=self._config.tune,
            gop_size=self._config.gop_size,
            bitrate=self._config.bitrate,
        )
        self._bridge = RTSPBridge(maxim, bridge_config)
        self._bridge.start(blocking=False)

        if self._bridge.is_running:
            self._state = SkillState.ACTIVE
            log.info(
                "RTSP stream started: %s at %d fps",
                self._config.rtsp_url,
                self._config.fps,
            )
            self._last_result = SkillResult(
                state=SkillState.ACTIVE,
                message=f"Streaming to {self._config.rtsp_url} at {self._config.fps} fps",
                metadata={"rtsp_url": self._config.rtsp_url, "fps": self._config.fps},
            )
            # Write to shared context so other skills can discover the stream URL
            if context is not None:
                context["rtsp_url"] = self._config.rtsp_url
                context["rtsp_fps"] = self._config.fps
        else:
            self._state = SkillState.FAILED
            log.error(
                "RTSP bridge failed to start (url=%s). Check that ffmpeg is installed and MediaMTX is reachable.",
                self._config.rtsp_url,
            )
            self._last_result = SkillResult(
                state=SkillState.FAILED,
                message="Bridge failed to start",
                error="ffmpeg process did not start. Is MediaMTX running?",
            )
            self._bridge = None

        return self._last_result

    def deactivate(self) -> SkillResult:
        self._state = SkillState.DEACTIVATING
        if self._bridge is not None:
            self._bridge.stop()
            self._bridge = None
        self._maxim = None
        self._state = SkillState.IDLE
        self._last_result = SkillResult(
            state=SkillState.IDLE,
            message="RTSP stream stopped",
        )
        return self._last_result

    @property
    def state(self) -> SkillState:
        # Detect runtime failure: bridge died after successful start
        if self._state == SkillState.ACTIVE and self._bridge is not None:
            if not self._bridge.is_running:
                self._state = SkillState.FAILED
                self._last_result = SkillResult(
                    state=SkillState.FAILED,
                    message="Bridge stopped unexpectedly",
                    error="ffmpeg process exited. MediaMTX may have stopped.",
                )
        return self._state

    def context_for_llm(self) -> str:
        if self.state == SkillState.FAILED:
            return super().context_for_llm()  # uses _last_result.error
        if self.state != SkillState.ACTIVE:
            return "RTSP streaming skill loaded but not active."
        return f"Streaming camera to {self._config.rtsp_url} at {self._config.fps} fps."

    def health(self) -> dict[str, Any]:
        base = super().health()
        if self._bridge is not None:
            base["rtsp_url"] = self._config.rtsp_url
            base["fps"] = self._config.fps
        return base


# --- Skill-owned tools ---


class _StartStreamTool(Tool):
    name = "start_rtsp_stream"
    description = "Start streaming the Reachy camera as RTSP. Requires MediaMTX running and ffmpeg installed."
    input_schema = {
        "rtsp_url": (str, "rtsp://localhost:8554/reachy"),
        "fps": (int, 20),
    }

    def __init__(self, skill: RTSPStreamingSkill) -> None:
        super().__init__()
        self._skill = skill

    def execute(self, **kwargs: Any) -> ToolOutput:
        if self._skill.is_active:
            return ToolOutput(
                success=True,
                output=f"Already streaming to {self._skill._config.rtsp_url}",
            )
        # Re-activate with updated config (frozen dataclass — use replace)
        import dataclasses as _dc

        url = kwargs.get("rtsp_url", self._skill._config.rtsp_url)
        fps = int(kwargs.get("fps", self._skill._config.fps))
        self._skill._config = _dc.replace(self._skill._config, rtsp_url=url, fps=fps)
        if self._skill._maxim is None:
            return ToolOutput(
                success=False,
                error="Cannot start stream: skill is not attached to Maxim. "
                "Activate the protocol first (e.g., 'run shredder segmenter protocol').",
            )
        result = self._skill.activate(self._skill._maxim)
        if not result.ok:
            return ToolOutput(
                success=False,
                error=result.error or "Failed to start RTSP stream. Check ffmpeg and MediaMTX.",
            )
        return ToolOutput(success=True, output=result.message or f"Streaming to {url} at {fps} fps.")


class _StopStreamTool(Tool):
    name = "stop_rtsp_stream"
    description = "Stop the RTSP camera stream."

    def __init__(self, skill: RTSPStreamingSkill) -> None:
        super().__init__()
        self._skill = skill

    def execute(self, **kwargs: Any) -> ToolOutput:
        if not self._skill.is_active:
            return ToolOutput(success=True, output="RTSP stream is not running.")
        self._skill.deactivate()
        return ToolOutput(success=True, output="RTSP stream stopped.")
