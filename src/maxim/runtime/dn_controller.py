"""DefaultNetworkController: DN lifecycle management (Phase 9).

Extracts Default Network start/stop/configure logic out of the loop body
into a standalone class that can be tested in isolation.
"""

from __future__ import annotations

import logging
from typing import Any

from maxim.utils.structured_logging import log_agentic

logger = logging.getLogger(__name__)


class DefaultNetworkController:
    """Manages Default Network lifecycle and mode-based configuration."""

    def __init__(self, default_network: Any | None) -> None:
        self._dn = default_network
        self._last_mode: str | None = None

    @property
    def enabled(self) -> bool:
        return self._dn is not None

    def start(self) -> bool:
        """Start the DN. Returns True if started successfully."""
        if not self.enabled or self._dn is None:
            return False
        try:
            self._dn.start()
            log_agentic("default_network", "startup", {"status": "started"}, level="INFO")
            return True
        except Exception as e:
            logger.warning("Failed to start DefaultNetwork: %s", e)
            return False

    def stop(self) -> None:
        """Stop the DN if running."""
        if not self.enabled or self._dn is None:
            return
        try:
            self._dn.stop()
            log_agentic("default_network", "shutdown", {"status": "stopped"}, level="INFO")
        except Exception as e:
            logger.debug("Failed to stop DefaultNetwork: %s", e)

    def configure_for_mode(self, mode_name: str) -> None:
        """Apply mode-specific DN configuration (behavior priorities, thresholds)."""
        if not self.enabled or self._dn is None:
            return
        if mode_name == self._last_mode:
            return

        from maxim.modes.definitions import get_mode

        mode_def = get_mode(mode_name)
        if mode_def is None:
            return

        dn_config = mode_def.default_network

        if not dn_config.enabled:
            if self._dn.is_running:
                self._dn.stop()
                log_agentic("default_network", "dn_inhibited", {"reason": "mode_disabled", "mode": mode_name})
        else:
            if not self._dn.is_running:
                self._dn.start()
                log_agentic("default_network", "dn_released", {"reason": "mode_enabled", "mode": mode_name})

            self._dn.clear_behavior_overrides()
            for behavior_name, modifier in dn_config.behavior_priority_modifiers.items():
                self._dn.boost_behavior(behavior_name, modifier)

            if hasattr(self._dn, "gate") and hasattr(self._dn.gate, "_adaptive"):
                if self._dn.gate._adaptive:
                    self._dn.gate._adaptive._novelty_threshold = dn_config.escalation_threshold
                    self._dn.gate._adaptive._salience_threshold = dn_config.escalation_threshold

        self._last_mode = mode_name

    def inhibit_for_tool(self, mode_name: str) -> bool:
        """Check if DN should be inhibited during tool execution."""
        if not self.enabled or self._dn is None:
            return False
        from maxim.modes.definitions import get_mode

        mode_def = get_mode(mode_name)
        return bool(mode_def and mode_def.default_network.inhibit_during_tool_execution)
