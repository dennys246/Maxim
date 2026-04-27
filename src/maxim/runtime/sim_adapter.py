"""SimulationAdapter: Isolates simulation concerns from the core loop (Phase 4).

Replaces ~20 inline ``if percept_source is not None`` guards with a clean
adapter interface.  Production uses ``NullSimulationAdapter`` which no-ops.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from maxim.utils.structured_logging import log_agentic

logger = logging.getLogger(__name__)


class SimulationAdapter:
    """Wraps percept_source, action_sink, sim_logger, and response policy for sim mode."""

    def __init__(
        self,
        percept_source: Any,
        action_sink: Any | None = None,
        pain_bus: Any | None = None,
    ) -> None:
        self.percept_source = percept_source
        self.action_sink = action_sink
        self.pain_bus = pain_bus
        self._tool_registry: Any | None = None  # set by orchestrator for deregistered-tool filtering
        self._grace_deadline: float | None = None
        self._grace_action_count: int = 0

    @property
    def is_sim_mode(self) -> bool:
        return True

    def next_observation(self, environment: Any, default_network: Any | None = None) -> dict:
        """Get observation from percept source or empty dict."""
        sim_percept = self.percept_source.next_percept()
        if sim_percept is not None:
            # Route pain percepts through ReactionBus (Phase 2a: emit
            # Reaction directly instead of the old route_pain_percept detour).
            if sim_percept.source == "proprioception" and sim_percept.content == "pain_signal":
                try:
                    _pb = self.pain_bus
                    if _pb is None:
                        _pb = getattr(default_network, "pain_bus", None) if default_network else None
                    if _pb is not None:
                        from maxim.decisions.causal_link import Valence
                        from maxim.reactions.types import Reaction, ReactionContext

                        meta = sim_percept.metadata or {}
                        reaction = Reaction(
                            kind="pain",
                            intensity=meta.get("intensity", 0.5),
                            valence=Valence.NEGATIVE,
                            timestamp=sim_percept.timestamp,
                            source=f"sim_adapter:{meta.get('pain_type', 'external_signal')}",
                            context=ReactionContext(),
                        )
                        bus = getattr(_pb, "reaction_bus", _pb)
                        bus.publish(reaction)
                except Exception:
                    pass

            # Convert percept to observation dict
            _sim_cli = sim_percept.cli_input
            if not _sim_cli and sim_percept.transcript_chunk:
                _sim_cli = sim_percept.transcript_chunk
            if not _sim_cli and sim_percept.content and sim_percept.source != "proprioception":
                _sim_cli = sim_percept.content

            observation = {
                "source": sim_percept.source,
                "transcript": sim_percept.transcript_chunk,
                "cli_input": _sim_cli,
                "hard_override": sim_percept.hard_override,
                "raw_transcript_text": sim_percept.raw_transcript_text,
            }
        else:
            observation = {}

        if hasattr(self.percept_source, "advance_step"):
            self.percept_source.advance_step()
        return observation

    def check_exhaustion(self, pending_proposal: Any | None) -> bool:
        """Check if percept source is exhausted and grace period expired.

        Returns True if the loop should break.
        """
        if not self.percept_source.is_exhausted():
            return False

        if self._grace_deadline is None:
            self._grace_deadline = time.time() + 180.0
            self._grace_action_count = 0 if self.action_sink is None else len(self.action_sink.actions)
            log_agentic("agent_loop", "percept_source_exhausted", {"grace_seconds": 180})

        # Tighten grace if new actions appeared
        if (
            self.action_sink is not None
            and len(self.action_sink.actions) > self._grace_action_count
            and pending_proposal is None
        ):
            self._grace_action_count = len(self.action_sink.actions)
            self._grace_deadline = min(self._grace_deadline, time.time() + 15.0)
            log_agentic("agent_loop", "grace_tightened", {"actions": len(self.action_sink.actions)})

        if time.time() >= self._grace_deadline:
            log_agentic("agent_loop", "shutdown", {"reason": "percept_source_grace_expired"})
            return True

        return False

    def log(self, category: str, msg: str, data: dict | None = None) -> None:
        """Log a simulation event via sim_logger."""
        try:
            from maxim.simulation.sim_logger import sim_log

            sim_log(category, msg, data)
        except Exception:
            pass

    def should_skip_fallback_proposal(self, proposal: Any) -> bool:
        """In sim mode, skip fallback proposals — wait for real LLM.

        Also skips proposals for deregistered tools (e.g., 'respond' in
        embodied arcs where conversational tools are removed).
        """
        if getattr(proposal, "reasoning", "") == "llm_fallback":
            return True
        # If the proposed tool doesn't exist in the registry, skip it
        action = getattr(proposal, "action", None)
        if isinstance(action, dict):
            tool_name = action.get("tool_name", "")
            if tool_name and self._tool_registry is not None:
                try:
                    available = self._tool_registry.list_all()
                    if tool_name not in available and tool_name != "_llm_unavailable":
                        return True
                except Exception:
                    pass
        return False

    def resolve_confirmation(self, confirmation: dict[str, Any]) -> str | None:
        """Auto-resolve confirmation prompts using the bridge's response policy."""
        try:
            from maxim.simulation.bridge import SimulationBridge

            if isinstance(self.percept_source, SimulationBridge):
                return self.percept_source.response_policy.resolve_confirmation(confirmation)
            # For ConversationalSource (non-bridge sim), check if it has a policy
            bridge = getattr(self.percept_source, "_bridge", None)
            if bridge and hasattr(bridge, "response_policy"):
                return bridge.response_policy.resolve_confirmation(confirmation)
        except Exception:
            pass
        # Default: auto-approve in sim mode
        return "yes"

    def resolve_plan_approval(self, plan_text: str | None) -> str | None:
        """Auto-resolve plan approval prompts."""
        try:
            from maxim.simulation.bridge import SimulationBridge

            if isinstance(self.percept_source, SimulationBridge):
                return self.percept_source.response_policy.resolve_plan_approval(plan_text)
        except Exception:
            pass
        return "yes"

    def resolve_timeout_retry(self, timeout_s: float) -> str | None:
        """Auto-resolve timeout retry prompts."""
        try:
            from maxim.simulation.bridge import SimulationBridge

            if isinstance(self.percept_source, SimulationBridge):
                return self.percept_source.response_policy.resolve_timeout_retry(timeout_s)
        except Exception:
            pass
        return "yes"


class NullSimulationAdapter:
    """No-op adapter for production (non-simulation) mode."""

    @property
    def is_sim_mode(self) -> bool:
        return False

    def next_observation(self, environment: Any, default_network: Any | None = None) -> dict:
        return environment.observe()

    def check_exhaustion(self, pending_proposal: Any | None) -> bool:
        return False

    def log(self, category: str, msg: str, data: dict | None = None) -> None:
        pass

    def should_skip_fallback_proposal(self, proposal: Any) -> bool:
        return False

    def resolve_confirmation(self, confirmation: dict[str, Any]) -> str | None:
        return None  # No auto-resolve in production — wait for real user

    def resolve_plan_approval(self, plan_text: str | None) -> str | None:
        return None

    def resolve_timeout_retry(self, timeout_s: float) -> str | None:
        return None
