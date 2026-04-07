"""ShredderSegmenterProtocol — stream camera as RTSP for ski recording.

Composes:
  - RTSPStreamingSkill (camera → RTSP → ShredderSegmenter site agent)
  - Constrained workspace bounds (fixed gaze arc for ski slope)

Activated via: "Maxim run shredder segmenter protocol"
Deactivated via: "Maxim stop shredder segmenter protocol"
"""

from __future__ import annotations

import logging
from typing import Any

from maxim.skills.base import Skill, SkillResult
from maxim.skills.health_reporting import HealthReportingConfig, HealthReportingSkill
from maxim.skills.protocol import Protocol, WorkspaceBounds
from maxim.skills.rtsp_streaming import RTSPStreamingConfig, RTSPStreamingSkill
from maxim.skills.timed_protocol import TimedProtocolConfig, TimedProtocolSkill

__all__ = ["ShredderSegmenterProtocol"]

log = logging.getLogger(__name__)


class ShredderSegmenterProtocol(Protocol):
    def __init__(
        self,
        rtsp_url: str = "rtsp://localhost:8554/reachy",
        fps: int = 20,
        yaw_range: float = 30.0,  # ±30° horizontal arc
        pitch_range: float = 20.0,  # ±20° vertical arc
        # ShredderSegmenter API integration (optional)
        shredder_api_url: str | None = None,  # e.g., "http://localhost:8000"
        shredder_license_id: str | None = None,
        shredder_api_key: str | None = None,  # Bearer token for API auth
        shredder_camera_name: str = "reachy-maxim",
        shredder_site_id: str | None = None,  # Required for remote recording
        # Auto-stop after duration (0 = disabled)
        duration_minutes: float = 0.0,
        # Health reporting (empty endpoint = disabled)
        health_endpoint_url: str = "",
        health_interval_seconds: float = 30.0,
    ):
        super().__init__()
        self._rtsp_config = RTSPStreamingConfig(rtsp_url=rtsp_url, fps=fps)
        self._yaw_range = yaw_range
        self._pitch_range = pitch_range
        self._shredder_api_url = shredder_api_url
        self._shredder_license_id = shredder_license_id
        self._shredder_api_key = shredder_api_key
        self._shredder_camera_name = shredder_camera_name
        self._shredder_site_id = shredder_site_id
        self._registered_camera_id: str | None = None
        self._duration_minutes = duration_minutes
        self._health_endpoint_url = health_endpoint_url
        self._health_interval_seconds = health_interval_seconds

    @property
    def name(self) -> str:
        return "shredder_segmenter"

    @property
    def description(self) -> str:
        return "Stream Reachy camera as RTSP for ShredderSegmenter ski recording. Gaze constrained to a fixed arc."

    def skills(self) -> list[Skill]:
        skills: list[Skill] = [RTSPStreamingSkill(self._rtsp_config)]

        if self._health_endpoint_url:
            headers = {}
            if self._shredder_api_key:
                headers["Authorization"] = f"Bearer {self._shredder_api_key}"
            skills.append(
                HealthReportingSkill(
                    HealthReportingConfig(
                        endpoint_url=self._health_endpoint_url,
                        interval_seconds=self._health_interval_seconds,
                        headers=headers,
                    )
                )
            )

        # TimedProtocol must be last — it reads _protocol_name from context
        if self._duration_minutes > 0:
            skills.append(
                TimedProtocolSkill(
                    TimedProtocolConfig(
                        duration_minutes=self._duration_minutes,
                    )
                )
            )

        return skills

    def workspace_bounds(self) -> WorkspaceBounds:
        return WorkspaceBounds(
            yaw=self._yaw_range,
            pitch=self._pitch_range,
        )

    def phrases(self) -> list[str]:
        return [
            "run shredder segmenter protocol",
            "start shredder segmenter protocol",
            "run shredder segmenter",
            "start shredder segmenter",
            "run shredder protocol",
            "start shredder protocol",
            "start streaming for shredder",
        ]

    def stop_phrases(self) -> list[str]:
        return [
            "stop shredder segmenter protocol",
            "stop shredder segmenter",
            "stop shredder protocol",
            "stop streaming for shredder",
        ]

    def on_skill_failed(self, skill: Skill, result: SkillResult) -> str:
        """RTSP streaming is critical — abort. Timer and health are optional."""
        if skill.name == "rtsp_streaming":
            return "abort"
        # Health reporting and timed protocol are nice-to-have, not critical
        return "continue"

    def on_activate(self, maxim: Any) -> None:
        """Activate streaming, then optionally register camera with ShredderSegmenter API."""
        super().on_activate(maxim)

        # Auto-register with ShredderSegmenter central server if configured
        if self._shredder_api_url and self._shredder_license_id:
            self._register_with_shredder()

    def on_deactivate(self) -> None:
        """Deactivate streaming, then optionally unregister camera."""
        # Unregister camera from ShredderSegmenter if we registered it
        if self._registered_camera_id and self._shredder_api_url:
            self._unregister_from_shredder()
        super().on_deactivate()

    def _register_with_shredder(self) -> None:
        """Register Reachy's RTSP stream as a camera in ShredderSegmenter.

        Steps:
          1. POST /licenses/{license_id}/cameras — register the camera
          2. POST /sites/{site_id}/cameras — assign camera to site (required
             for the site agent to claim recording jobs for this camera)

        Retries up to 3 times with exponential backoff for transient failures.

        NOTE: This runs synchronously during on_activate() and can block up to
        ~18s worst case (3 × 5s timeout + 1s + 2s backoff). This is acceptable
        for a one-time activation operation on the agentic runtime thread.
        """
        import json
        import time
        import urllib.request
        from urllib.parse import urlparse

        api_url = self._shredder_api_url
        headers = {"Content-Type": "application/json"}
        if self._shredder_api_key:
            headers["Authorization"] = f"Bearer {self._shredder_api_key}"

        # --- Step 1: Register camera (with retry) ---
        register_url = f"{api_url}/licenses/{self._shredder_license_id}/cameras"
        parsed = urlparse(self._rtsp_config.rtsp_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 8554

        body = json.dumps(
            {
                "name": self._shredder_camera_name,
                "engine_type": "maxim",
                "host": host,
                "port": port,
                "rtsp_url": self._rtsp_config.rtsp_url,
            }
        ).encode()

        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    register_url,
                    data=body,
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read())
                    self._registered_camera_id = data.get("id")
                    log.info(
                        "Registered camera '%s' with ShredderSegmenter (id=%s)",
                        self._shredder_camera_name,
                        self._registered_camera_id,
                    )
                break
            except Exception as e:
                if attempt < 2:
                    delay = 2**attempt  # 1s, 2s
                    log.warning(
                        "Registration attempt %d failed: %s (retrying in %ds)",
                        attempt + 1,
                        e,
                        delay,
                        exc_info=True,
                    )
                    time.sleep(delay)
                else:
                    log.error(
                        "Failed to register with ShredderSegmenter after 3 attempts: %s",
                        e,
                        exc_info=True,
                    )
                    return

        # --- Step 2: Assign camera to site (required for recording) ---
        if self._registered_camera_id and self._shredder_site_id:
            try:
                assign_url = f"{api_url}/sites/{self._shredder_site_id}/cameras"
                assign_body = json.dumps(
                    {
                        "camera_id": self._registered_camera_id,
                    }
                ).encode()
                req = urllib.request.Request(
                    assign_url,
                    data=assign_body,
                    headers=headers,
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=5)
                log.info(
                    "Assigned camera %s to site %s",
                    self._registered_camera_id,
                    self._shredder_site_id,
                )
            except Exception as e:
                log.warning(
                    "Camera registered but site assignment failed: %s. "
                    "Recording jobs cannot be created until the camera is "
                    "manually assigned to a site via the dashboard.",
                    e,
                )
        elif self._registered_camera_id and not self._shredder_site_id:
            log.warning(
                "Camera registered but shredder_site_id not configured. "
                "Recording jobs cannot be created until the camera is "
                "assigned to a site (dashboard or API).",
            )

    def _unregister_from_shredder(self) -> None:
        """Remove the camera registration on protocol deactivation."""
        try:
            import urllib.request

            url = f"{self._shredder_api_url}/cameras/{self._registered_camera_id}"
            headers = {}
            if self._shredder_api_key:
                headers["Authorization"] = f"Bearer {self._shredder_api_key}"
            req = urllib.request.Request(url, method="DELETE", headers=headers)
            urllib.request.urlopen(req, timeout=5)
            log.info("Unregistered camera '%s' from ShredderSegmenter", self._registered_camera_id)
        except Exception as e:
            log.error(
                "Failed to unregister camera %s from ShredderSegmenter: %s. Manual cleanup may be required.",
                self._registered_camera_id,
                e,
                exc_info=True,
            )
        self._registered_camera_id = None
