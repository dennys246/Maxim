"""Agent Pool — manages concurrent execution of multiple agents.

Provides orchestration for multi-agent simulations where each agent
has independent memory, tools, and execution.  The pool handles:
- Agent lifecycle (add, remove, shutdown)
- Turn execution (sequential or concurrent via threads)
- Inter-agent communication via LocalMessageBus
- Memory export for post-simulation analysis

Example::

    pool = AgentPool()
    pool.add(guard_instance)
    pool.add(merchant_instance)

    # Run one turn for each agent
    results = pool.run_round({
        "npc_guard": "The player draws a weapon.",
        "npc_merchant": "A customer approaches your stall.",
    })

    # Export all memories
    for agent_id, memories in pool.export_all_memories().items():
        print(f"{agent_id}: {memories['episodic_memories']} memories")
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class TurnResult:
    """Result of a single agent turn."""

    agent_id: str
    response: str | None = None
    actions: list[dict[str, Any]] = field(default_factory=list)
    timed_out: bool = False
    error: str | None = None
    duration_ms: float = 0.0
    percept: Any = None  # Typed Percept created from the input string (Phase 4)


class AgentPool:
    """Manages concurrent execution of multiple agent instances.

    Each agent in the pool has independent memory systems (Hippocampus,
    NAc, ATL) created by AgentFactory.  The pool orchestrates turn
    execution and inter-agent communication.
    """

    def __init__(
        self,
        bus: Any | None = None,
        max_workers: int = 4,
        llm_router: Any | None = None,
    ) -> None:
        from maxim.runtime.agent_factory import AgentInstance

        self._agents: dict[str, AgentInstance] = {}
        self._lock = threading.RLock()
        self._max_workers = max_workers
        self._executor: ThreadPoolExecutor | None = None
        self._llm_router = llm_router

        # Message bus for inter-agent communication
        if bus is not None:
            self._bus = bus
        else:
            try:
                from maxim.mesh.bus import LocalMessageBus

                self._bus = LocalMessageBus()
            except ImportError:
                self._bus = None

    # -- lifecycle ----------------------------------------------------------

    def add(self, instance: Any) -> None:
        """Register an agent instance.  Connects to message bus."""
        with self._lock:
            self._agents[instance.agent_id] = instance
            log.info("AgentPool: added '%s' (role=%s)", instance.agent_id, instance.role)

            # Register with message bus if available
            if self._bus is not None and instance.agent_id:
                try:
                    self._bus.register(
                        instance.agent_id, lambda msg, aid=instance.agent_id: self._handle_message(aid, msg)
                    )
                except Exception:
                    pass

    def remove(self, agent_id: str) -> None:
        """Stop and remove an agent from the pool."""
        with self._lock:
            instance = self._agents.pop(agent_id, None)
        if instance is not None:
            instance.shutdown()
            if self._bus is not None:
                try:
                    self._bus.unregister(agent_id)
                except Exception:
                    pass
            log.info("AgentPool: removed '%s'", agent_id)

    def get_agent(self, agent_id: str) -> Any:
        """Look up an agent by ID.  Raises KeyError if not found."""
        with self._lock:
            if agent_id not in self._agents:
                raise KeyError(f"Agent not found: '{agent_id}'")
            return self._agents[agent_id]

    @property
    def agent_ids(self) -> list[str]:
        """Return sorted list of all agent IDs in the pool."""
        with self._lock:
            return sorted(self._agents.keys())

    @property
    def size(self) -> int:
        """Number of agents in the pool."""
        with self._lock:
            return len(self._agents)

    # -- turn execution -----------------------------------------------------

    def run_turn(self, agent_id: str, percept: str, **kwargs: Any) -> TurnResult:
        """Run a single turn for one agent (synchronous).

        Delivers a percept to the agent and collects its response.
        This is a lightweight turn suitable for NPC agents — it does
        NOT run the full agent loop (LoopController/ContextPool/etc.).

        For NPC agents, the turn is:
        1. Store percept in hippocampus (if enabled)
        2. Generate response (via personality + percept context)
        3. Record any actions taken
        4. Return result

        Args:
            agent_id: The agent to run.
            percept: Text delivered to the agent (scene narrative, etc.).

        Returns:
            TurnResult with the agent's response and actions.
        """
        import time

        start = time.time()

        instance = self.get_agent(agent_id)

        try:
            # Phase 4: wrap the string stimulus in a typed Percept so both
            # runtimes share the same Percept surface. store_observation
            # still takes a string (the full MemoryAgent pipeline is a
            # bigger refactor for substrate P1), but the typed Percept is
            # stashed on TurnResult for any consumer that wants it.
            from maxim.agents.percept_factory import make_text_percept

            typed_percept = make_text_percept(
                percept[:500],
                source="npc_turn",
                agent_id=agent_id,
                channel="narrative",
            )

            # Store percept in hippocampus (lightweight observation capture)
            if instance.hippocampus is not None:
                try:
                    instance.hippocampus.store_observation(
                        text=percept[:500],
                        metadata={"agent_id": agent_id, "type": "npc_percept"},
                    )
                except Exception:
                    pass

            # For NPC agents, generate a simple response based on personality
            response = self._generate_npc_response(instance, percept, **kwargs)

            # Record in NAc if learning enabled
            if instance.nac is not None and response:
                try:
                    from maxim.decisions.nac import Valence

                    instance.nac.observe(
                        event_type="npc_respond",
                        event_signature=f"{agent_id}:respond",
                        outcome_type="response",
                        outcome_signature="generated",
                        outcome_valence=Valence.NEUTRAL,
                        delta_seconds=0.0,
                    )
                except Exception:
                    pass

            elapsed = (time.time() - start) * 1000
            return TurnResult(
                agent_id=agent_id,
                response=response,
                duration_ms=elapsed,
                percept=typed_percept,
            )

        except Exception as e:
            elapsed = (time.time() - start) * 1000
            log.warning("Agent '%s' turn failed: %s", agent_id, e)
            return TurnResult(
                agent_id=agent_id,
                error=str(e),
                duration_ms=elapsed,
            )

    def run_round(
        self,
        percepts: dict[str, str],
        concurrent: bool = True,
    ) -> dict[str, TurnResult]:
        """Run one turn for each agent (optionally concurrent).

        Args:
            percepts: Maps agent_id → percept text for this turn.
            concurrent: If True, run agents in parallel threads.

        Returns:
            Dict mapping agent_id → TurnResult.
        """
        results: dict[str, TurnResult] = {}

        if not concurrent or len(percepts) <= 1:
            # Sequential execution
            for agent_id, percept in percepts.items():
                if agent_id in self._agents:
                    results[agent_id] = self.run_turn(agent_id, percept)
            return results

        # Concurrent execution via persistent thread pool
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self._max_workers)

        futures = {
            self._executor.submit(self.run_turn, agent_id, percept): agent_id
            for agent_id, percept in percepts.items()
            if agent_id in self._agents
        }
        for future in as_completed(futures):
            agent_id = futures[future]
            try:
                results[agent_id] = future.result()
            except Exception as e:
                results[agent_id] = TurnResult(agent_id=agent_id, error=str(e))

        return results

    def broadcast_percept(self, percept: str) -> dict[str, TurnResult]:
        """Send the same percept to all agents and collect responses."""
        percepts = {aid: percept for aid in self.agent_ids}
        return self.run_round(percepts)

    # -- memory export ------------------------------------------------------

    def export_memories(self, agent_id: str) -> dict[str, Any]:
        """Export one agent's memory state for post-sim analysis."""
        instance = self.get_agent(agent_id)
        return instance.export_memories()

    def export_all_memories(self) -> dict[str, dict[str, Any]]:
        """Export all agents' memory states."""
        with self._lock:
            return {aid: inst.export_memories() for aid, inst in self._agents.items()}

    # -- messaging ----------------------------------------------------------

    def broadcast(self, message: Any) -> None:
        """Send a message to all agents via the bus."""
        if self._bus is not None:
            self._bus.send(message)

    def _handle_message(self, agent_id: str, message: Any) -> None:
        """Handle an incoming message for a specific agent."""
        log.debug("Agent '%s' received message: %s", agent_id, getattr(message, "type", "?"))

    # -- shutdown -----------------------------------------------------------

    # -- context manager -----------------------------------------------------

    def __enter__(self) -> "AgentPool":
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any) -> bool:
        self.shutdown()
        return False

    def shutdown(self) -> None:
        """Stop all agents, flush memories, cleanup."""
        with self._lock:
            agent_ids = list(self._agents.keys())

        for aid in agent_ids:
            try:
                self.remove(aid)
            except Exception as e:
                log.warning("Failed to remove agent '%s' during shutdown: %s", aid, e)

        # Shut down the persistent thread pool
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=False)
            except Exception:
                pass
            self._executor = None

        log.info("AgentPool shutdown complete")

    # -- private ------------------------------------------------------------

    def _generate_npc_response(
        self,
        instance: Any,
        percept: str,
        **kwargs: Any,
    ) -> str | None:
        """Generate a response for an NPC agent.

        Uses the LLM router (if available) to produce in-character dialogue
        based on the agent's personality and the scene percept.  Falls back
        to a personality-echo response when no LLM router is configured.
        """
        personality = instance.personality or ""
        if not personality:
            return None

        # Try LLM inference if router is available
        if self._llm_router is not None:
            try:
                prompt = (
                    f"You are an NPC in a narrative scenario.\n"
                    f"Your personality: {personality}\n\n"
                    f"{percept}\n\n"
                    f"Respond in character with 1-2 sentences of dialogue or action. "
                    f"Stay in character. Be concise."
                )
                result = self._llm_router.generate(
                    prompt,
                    function="npc_dialogue",
                    max_tokens=100,
                )
                if result and hasattr(result, "text") and result.text:
                    response = result.text.strip()
                    # Strip quotes if the LLM wrapped its response
                    if response.startswith('"') and response.endswith('"'):
                        response = response[1:-1]
                    log.debug("NPC '%s' LLM response: %s", instance.agent_id, response[:80])
                    return response
            except Exception as e:
                log.debug("NPC '%s' LLM inference failed, using fallback: %s", instance.agent_id, e)

        # Fallback: personality-based echo (no LLM available)
        # Extract a short phrase from the personality for a contextual stub
        persona_snippet = personality.split(".")[0].strip()
        if len(persona_snippet) > 60:
            persona_snippet = persona_snippet[:60] + "..."
        return f"*{persona_snippet}*"
