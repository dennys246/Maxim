"""HealthReportingSkill — periodic health pings to an external endpoint.

Reads shared context (blackboard) to discover active stream URLs and
protocol state, then pushes periodic health reports via HTTP POST. Useful
for monitoring dashboards, ShredderSegmenter status tracking, or any
external system that needs to know Maxim's protocol health.

The skill is passive (no LLM tools) — it runs a background thread that
posts health data at a configurable interval.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from maxim.skills.base import Skill, SkillConfig, SkillResult, SkillState

if TYPE_CHECKING:
    from maxim.tools.base import Tool

__all__ = ["HealthReportingSkill", "HealthReportingConfig"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HealthReportingConfig(SkillConfig):
    """Configuration for health reporting."""
    endpoint_url: str = ""            # HTTP POST target (required)
    interval_seconds: float = 30.0    # How often to ping
    timeout_seconds: float = 5.0      # HTTP request timeout
    headers: dict[str, str] = field(default_factory=dict)  # Extra headers (auth, etc.)


class HealthReportingSkill(Skill):
    """Push periodic health reports to an external HTTP endpoint."""

    def __init__(self, config: HealthReportingConfig | None = None):
        self._config = config or HealthReportingConfig()
        self._state = SkillState.IDLE
        self._last_result: SkillResult | None = None
        self._maxim: Any = None
        self._context: dict[str, Any] | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()  # Protects _consecutive_failures and _state
        self._consecutive_failures: int = 0

    @property
    def name(self) -> str:
        return "health_reporting"

    @property
    def description(self) -> str:
        return f"Push health reports to {self._config.endpoint_url or '(not configured)'}"

    def tools(self) -> list[Tool]:
        return []  # Passive skill — no tools

    def can_activate(self, maxim: Any) -> tuple[bool, str]:
        if not self._config.endpoint_url:
            return False, "endpoint_url not configured"
        return True, ""

    def activate(self, maxim: Any, context: dict[str, Any] | None = None) -> SkillResult:
        self._state = SkillState.ACTIVATING
        self._maxim = maxim
        self._context = context
        self._stop_event.clear()
        self._consecutive_failures = 0

        self._thread = threading.Thread(
            target=self._report_loop, daemon=True, name="health-reporter",
        )
        self._thread.start()

        self._state = SkillState.ACTIVE
        self._last_result = SkillResult(
            state=SkillState.ACTIVE,
            message=f"Reporting every {self._config.interval_seconds}s to {self._config.endpoint_url}",
            metadata={
                "endpoint_url": self._config.endpoint_url,
                "interval_seconds": self._config.interval_seconds,
            },
        )
        return self._last_result

    def deactivate(self) -> SkillResult:
        self._state = SkillState.DEACTIVATING
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(
                timeout=self._config.interval_seconds + self._config.timeout_seconds + 2
            )
            self._thread = None
        self._maxim = None
        self._context = None
        self._state = SkillState.IDLE
        self._last_result = SkillResult(
            state=SkillState.IDLE,
            message="Health reporting stopped",
        )
        return self._last_result

    @property
    def state(self) -> SkillState:
        with self._lock:
            return self._state

    def _build_payload(self) -> dict[str, Any]:
        """Assemble the health report from available context."""
        payload: dict[str, Any] = {
            "status": "active",
            "timestamp": time.time(),
        }

        # Pull data from shared blackboard context
        ctx = self._context or {}
        if "rtsp_url" in ctx:
            payload["rtsp_url"] = ctx["rtsp_url"]
        if "rtsp_fps" in ctx:
            payload["rtsp_fps"] = ctx["rtsp_fps"]
        if "_protocol_name" in ctx:
            payload["protocol"] = ctx["_protocol_name"]

        # Collect health from sibling skills if protocol is available
        maxim = self._maxim
        if maxim is not None:
            registry = getattr(maxim, "_protocol_registry", None)
            if registry is not None:
                protocol_name = ctx.get("_protocol_name")
                protocol = registry._active.get(protocol_name) if protocol_name else None
                if protocol is not None:
                    skill_health = []
                    for skill in protocol._active_skills:
                        if skill is not self:
                            skill_health.append(skill.health())
                    payload["skills"] = skill_health

        return payload

    def _post_health(self, payload: dict[str, Any]) -> None:
        """Send the health report via HTTP POST."""
        headers = {"Content-Type": "application/json"}
        headers.update(self._config.headers)

        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            self._config.endpoint_url,
            data=body,
            headers=headers,
            method="POST",
        )
        urllib.request.urlopen(req, timeout=self._config.timeout_seconds)

    def _report_loop(self) -> None:
        """Background thread: post health at interval until stopped."""
        log = getattr(self._maxim, "log", logger) if self._maxim else logger

        while not self._stop_event.is_set():
            try:
                payload = self._build_payload()
                self._post_health(payload)
                with self._lock:
                    self._consecutive_failures = 0
            except Exception as e:
                with self._lock:
                    self._consecutive_failures += 1
                    failures = self._consecutive_failures
                if failures <= 3:
                    log.warning(
                        "Health report failed (%d consecutive): %s",
                        failures, e, exc_info=True,
                    )
                elif failures % 10 == 0:
                    log.error(
                        "Health reporting failing persistently (%d consecutive): %s",
                        failures, e,
                    )
                if failures >= 10:
                    with self._lock:
                        if self._state == SkillState.ACTIVE:
                            self._state = SkillState.FAILED
                            self._last_result = SkillResult(
                                state=SkillState.FAILED,
                                message="Health reporting failed",
                                error=f"10 consecutive failures posting to {self._config.endpoint_url}",
                            )

            self._stop_event.wait(self._config.interval_seconds)

    def context_for_llm(self) -> str:
        if self.state == SkillState.FAILED:
            return super().context_for_llm()
        if self.state == SkillState.ACTIVE:
            with self._lock:
                failures = self._consecutive_failures
            status = "healthy" if failures == 0 else f"{failures} consecutive failures"
            return f"Health reporting to {self._config.endpoint_url} ({status})"
        return ""

    def health(self) -> dict[str, Any]:
        base = super().health()
        base["endpoint_url"] = self._config.endpoint_url
        with self._lock:
            base["consecutive_failures"] = self._consecutive_failures
        if self._thread is not None:
            base["thread_alive"] = self._thread.is_alive()
        return base
