"""Skills & Protocols — composable capabilities for Maxim.

Skills are atomic, independently testable capabilities (e.g., streaming
video, tracking objects). Protocols compose skills with workspace
constraints, LLM context, and voice/CLI activation phrases into named
operational profiles.
"""

from __future__ import annotations

from maxim.skills.base import Skill, SkillConfig, SkillResult, SkillState
from maxim.skills.health_reporting import HealthReportingConfig, HealthReportingSkill
from maxim.skills.protocol import Protocol, WorkspaceBounds
from maxim.skills.timed_protocol import TimedProtocolConfig, TimedProtocolSkill

__all__ = [
    "HealthReportingConfig",
    "HealthReportingSkill",
    "Protocol",
    "Skill",
    "SkillConfig",
    "SkillResult",
    "SkillState",
    "TimedProtocolConfig",
    "TimedProtocolSkill",
    "WorkspaceBounds",
]
