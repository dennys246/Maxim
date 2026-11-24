"""ResponsePolicy — configurable user response simulation for sim mode.

When the AUT requests confirmation, plan approval, or timeout retry,
this policy determines the response instead of waiting for stdin.

Policies:
    AUTO_APPROVE  — Always approve (default for sim). Fast, no blocking.
    AUTO_REJECT   — Always reject. Tests refusal/cancellation paths.
    DELAYED       — Approve after a configurable delay. Tests timeout handling.

Every policy answers: a run with no policy passes ``response_policy=None``. ``ASK_ORCHESTRATOR`` was retired
(#1736): it had no route to the orchestrator, so the loop answered in its place.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class PolicyType(Enum):
    AUTO_APPROVE = "auto_approve"
    AUTO_REJECT = "auto_reject"
    DELAYED = "delayed"


@dataclass
class ResponsePolicy:
    """Determines how the AUT's confirmation/approval prompts are resolved in sim mode."""

    policy: PolicyType = PolicyType.AUTO_APPROVE
    delay_s: float = 3.0  # Only used for DELAYED policy

    def resolve_confirmation(self, confirmation: dict[str, Any]) -> str:
        """Resolve a pending confirmation prompt.

        Returns:
            "yes" to approve, "no" to reject.
        """
        tool_name = confirmation.get("tool_name", "unknown")

        if self.policy == PolicyType.AUTO_APPROVE:
            logger.info("ResponsePolicy: auto-approving %s", tool_name)
            return "yes"

        elif self.policy == PolicyType.AUTO_REJECT:
            logger.info("ResponsePolicy: auto-rejecting %s", tool_name)
            return "no"

        elif self.policy == PolicyType.DELAYED:
            logger.info("ResponsePolicy: delayed approval for %s (%.1fs)", tool_name, self.delay_s)
            time.sleep(self.delay_s)
            return "yes"

        return "yes"  # Fallback

    def resolve_plan_approval(self, plan_text: str | None) -> str:
        """Resolve a pending plan approval prompt.

        Returns:
            "yes" to approve, "no" to reject.
        """
        if self.policy == PolicyType.AUTO_APPROVE:
            logger.info("ResponsePolicy: auto-approving plan")
            return "yes"

        elif self.policy == PolicyType.AUTO_REJECT:
            logger.info("ResponsePolicy: auto-rejecting plan")
            return "no"

        elif self.policy == PolicyType.DELAYED:
            time.sleep(self.delay_s)
            return "yes"

        return "yes"

    def resolve_timeout_retry(self, timeout_s: float) -> str:
        """Resolve a timeout retry prompt.

        Returns:
            "yes" to retry, "no" to skip.
        """
        if self.policy == PolicyType.AUTO_APPROVE:
            logger.info("ResponsePolicy: auto-retrying with doubled timeout")
            return "yes"

        elif self.policy == PolicyType.AUTO_REJECT:
            logger.info("ResponsePolicy: declining timeout retry")
            return "no"

        elif self.policy == PolicyType.DELAYED:
            time.sleep(self.delay_s)
            return "yes"

        return "yes"


# Convenience constructors
def auto_approve() -> ResponsePolicy:
    return ResponsePolicy(PolicyType.AUTO_APPROVE)


def auto_reject() -> ResponsePolicy:
    return ResponsePolicy(PolicyType.AUTO_REJECT)


def delayed(delay_s: float = 3.0) -> ResponsePolicy:
    return ResponsePolicy(PolicyType.DELAYED, delay_s=delay_s)
