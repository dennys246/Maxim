"""SimulationAdapter: Isolates simulation concerns from the core loop (Phase 4).

Replaces ~20 inline ``if percept_source is not None`` guards with a clean
adapter interface.  Production uses ``NullSimulationAdapter`` which no-ops.

Naming note (CC8, 2026-04-29):
The flag ``is_sim_mode`` is more accurately read as "an external adapter
is driving percepts" — it gates behaviors that fire when ``percept_source``
is non-None, not behaviors specific to the simulation orchestrator.
A future Mineflayer/Minecraft adapter (per ``simulation/sources.py`` adapter
contract) will reuse this same path: it constructs a ``PerceptSource`` +
``ActionSink`` pair, hands them to ``run_agentic_loop``, and the loop
treats it as ``is_sim_mode=True``. The behaviors gated by the flag are:

- Skip Default Network startup (DN is robot-vision specific; external
  adapters provide their own world state).
- Use sim-specific structured logging (``sim_log`` calls).
- Skip LLM fallback proposals (wait for the real LLM rather than emit
  a placeholder action).
- **End the session in lightweight mode**:
  ``runtime/bio_integration.py::end_bio_session`` calls
  ``MemoryHub.on_session_end_lightweight()`` instead of
  ``on_session_end()``, skipping the blocking sleep/replay
  consolidation pass. The lightweight path still persists NAc decay,
  semantic embeddings, and subsystem state so learning is not lost,
  but it does not run the full consolidation. This matches sim
  expectations (short-lived AUT runs) but **may surprise long-running
  external adapters** (e.g. a Minecraft session lasting hours expects
  full consolidation). 1.1+ adapter integrations should either set
  ``is_sim_mode=False`` post-construction, or
  ``end_bio_session(is_sim_mode=False, ...)`` should be called
  directly. This trade-off is preserved in 1.0 to avoid changing
  existing sim behavior; revisit when the first non-sim adapter ships.

The field is kept as ``is_sim_mode`` for back-compat with existing
callers; do not rename without a deprecation cycle.

Sim-specific coupling lives in two places intentionally:
- ``resolve_confirmation`` / ``resolve_plan_approval`` /
  ``resolve_timeout_retry`` use ``isinstance(...,SimulationBridge)`` to
  reach the bridge's ``response_policy``. A non-sim adapter without a
  bridge falls through to the default ``return "yes"`` (auto-approve)
  or ``return None`` (no-resolve), both of which are sensible defaults
  for an external adapter that handles confirmations itself.
- ``_tool_registry`` is set by the orchestrator for deregistered-tool
  filtering; non-sim adapters leave it ``None`` and skip the filter.
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
        # Thalamic-relay side-channel (thalamus_relay_design_pass.md Decision 2):
        # the current tick's raw Percept, carried OFF the observation dict so it
        # never enters state.data (RuntimeState.update absorbs every dict key and
        # state.data is persisted — a Percept on the dict would leak as repr into
        # state.json + episode snapshots). Modality-preserving consumers read this
        # accessor; the observation dict stays scalar-only.
        self._current_percept: Any | None = None

    @property
    def is_sim_mode(self) -> bool:
        return True

    @property
    def current_percept(self) -> Any | None:
        """The raw ``Percept`` from the most recent ``next_observation`` tick.

        ``None`` on the idle path (no percept this tick) and before the first
        tick. Cleared every tick, so it never carries a stale percept. A
        non-``None`` percept may still have ``None`` ``modality``/``sensory``;
        consumers tolerate ``None`` at both levels.
        """
        return self._current_percept

    def next_observation(self, environment: Any, default_network: Any | None = None) -> dict:
        """Get observation from percept source or empty dict."""
        sim_percept = self.percept_source.next_percept()
        # Stash on the side-channel (None on the idle path — clears any stale
        # percept). Kept OFF the returned dict on purpose; see __init__.
        self._current_percept = sim_percept
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
        """Auto-resolve confirmation prompts using the bridge's response policy.

        CC8 audit (2026-04-29): the ``isinstance`` check below is the
        orchestrator-coupled escape hatch documented in the module
        docstring. A non-sim adapter (no ``SimulationBridge``) falls
        through to the default ``"yes"`` — it owns confirmation
        resolution itself and does not need to plug into the bridge's
        response policy. A previous draft probed
        ``self.percept_source._bridge`` as a fallback for a
        ``ConversationalSource``-with-policy shape that was never
        actually wired (no producer ever set ``_bridge`` on a percept
        source); the dead branch was removed during the CC8 fold round.
        """
        try:
            from maxim.simulation.bridge import SimulationBridge

            if isinstance(self.percept_source, SimulationBridge):
                return self.percept_source.response_policy.resolve_confirmation(confirmation)
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

    @property
    def current_percept(self) -> Any | None:
        """Production has no percept_source side-channel. On the non-sim path
        the observation returned by ``next_observation`` IS the percept (a bare
        ``Percept`` or a dict from ``environment.observe()``); read it there."""
        return None

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
