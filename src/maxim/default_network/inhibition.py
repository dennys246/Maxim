"""Inhibition mixin for DefaultNetwork.

Extracts deliberative-layer inhibition handling, behavior priority
overrides, and interest/goal management into a reusable mixin.

All methods access state via ``self._*`` attributes initialised in
``DefaultNetwork.__init__``.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


class InhibitionMixin:
    """Mixin providing inhibition and behavior-override helpers for DefaultNetwork.

    Assumes the host class exposes at least:
        _config, _inhibited, _inhibit_until, _behavior_overrides,
        _interests, _gate
    """

    # ------------------------------------------------------------------
    # Deliberative inhibition
    # ------------------------------------------------------------------

    def inhibit(self, duration: float | None = None) -> None:
        """Suppress all DN output for *duration* seconds.

        Args:
            duration: Seconds to inhibit.  ``None`` uses ``auto_release_timeout``.
                The effective duration is capped by ``auto_release_timeout``.
        """
        self._inhibited = True
        if duration:
            effective_duration = min(duration, self._config.auto_release_timeout)
            self._inhibit_until = time.time() + effective_duration
        else:
            self._inhibit_until = time.time() + self._config.auto_release_timeout

        logger.debug(
            "DefaultNetwork inhibited for %.1fs",
            duration or self._config.auto_release_timeout,
        )

    def release(self) -> None:
        """Release inhibition, re-enabling DN output."""
        self._inhibited = False
        self._inhibit_until = 0.0
        logger.debug("DefaultNetwork released")

    def _check_auto_release(self) -> None:
        """Auto-release inhibition if the timeout has elapsed.

        Intended to be called at the top of each tick in ``_run_loop``.
        """
        if self._inhibited and time.time() > self._inhibit_until:
            self.release()

    # ------------------------------------------------------------------
    # Behavior priority overrides
    # ------------------------------------------------------------------

    def boost_behavior(self, name: str, modifier: float) -> None:
        """Temporarily boost or suppress a specific behavior's priority.

        Args:
            name: Behaviour name.
            modifier: Priority multiplier (>1 = boost, <1 = suppress).
        """
        self._behavior_overrides[name] = modifier

    def clear_behavior_overrides(self) -> None:
        """Clear all behaviour priority overrides."""
        self._behavior_overrides.clear()

    # ------------------------------------------------------------------
    # Interest / goal management
    # ------------------------------------------------------------------

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
