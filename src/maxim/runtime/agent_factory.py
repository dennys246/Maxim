"""Agent Factory — creates independent agent instances with isolated subsystems.

Each agent created by the factory gets its own:
- Hippocampus (separate episodic memory persistence)
- NAc (separate causal learning)
- ATL (separate semantic concepts)
- MemoryHub (independent coordinator)
- ToolRegistry (scoped to agent role)
- Executor (isolated tool execution)

The factory does NOT create the full agent loop infrastructure
(LoopController, ContextPool, etc.) — that's run_agentic_loop()'s job.
Instead, it creates the subsystems that the loop needs, suitable for
both full PC agents (via run_agentic_loop) and lightweight NPC agents
(via AgentPool.run_turn).

Example::

    factory = AgentFactory()
    guard = factory.create_npc_agent(
        npc_name="captain_aldric",
        entity_ref="npcs/guard",
        personality="A stern guard, loyal to the crown.",
    )
    # guard.hippocampus, guard.nac, guard.tool_registry are all independent
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    """Configuration for spawning an independent agent instance."""

    agent_id: str
    role: str = "npc"  # "pc", "npc", "companion"
    entity_spec: str | None = None  # Component ref for SEM body
    persistence_dir: str | None = None  # Auto-generated if None
    model_profile: str | None = None  # LLM profile override (NPCs use cheaper models)
    tool_whitelist: set[str] | None = None  # Restrict available tools
    personality: str | None = None  # System prompt overlay
    remembers: bool = True  # Enable hippocampus
    learns: bool = True  # Enable NAc


# ---------------------------------------------------------------------------
# Agent Instance
# ---------------------------------------------------------------------------


@dataclass
class AgentInstance:
    """A fully independent agent with its own subsystems.

    This is the runtime representation of one agent in a multi-agent
    simulation.  Each instance has isolated memory, tools, and state.
    """

    agent_id: str
    role: str
    config: AgentConfig

    # Memory subsystems (all per-agent, separate persistence)
    hippocampus: Any | None = None
    nac: Any | None = None
    atl: Any | None = None
    memory_hub: Any | None = None

    # Execution
    tool_registry: Any | None = None
    executor: Any | None = None

    # Embodiment
    entity: Any | None = None

    # Personality (injected into LLM system prompt)
    personality: str | None = None

    # Internal state
    _memories_exported: bool = field(default=False, repr=False)

    def export_memories(self) -> dict[str, Any]:
        """Export agent's memory state for post-sim analysis."""
        result: dict[str, Any] = {"agent_id": self.agent_id, "role": self.role}

        if self.hippocampus is not None:
            try:
                memories = list(self.hippocampus.memories.values())
                result["episodic_memories"] = len(memories)
                result["memory_summaries"] = [
                    {
                        "id": str(getattr(m, "id", "?")),
                        "tool": getattr(getattr(m, "action", None), "tool_name", "?"),
                        "valence": getattr(getattr(m, "outcome", None), "valence", 0),
                    }
                    for m in memories[:20]  # Cap at 20 for export
                ]
            except Exception:
                result["episodic_memories"] = 0

        if self.nac is not None:
            try:
                stats = self.nac.stats()
                result["causal_links"] = stats.get("total_links", 0)
                result["total_observations"] = stats.get("total_observations", 0)
            except Exception:
                result["causal_links"] = 0

        return result

    def shutdown(self) -> None:
        """Flush memories and clean up resources."""
        if self.memory_hub is not None:
            try:
                self.memory_hub.on_session_end()
            except Exception as e:
                log.warning("Agent %s: memory_hub shutdown failed: %s", self.agent_id, e)

        if self.hippocampus is not None:
            try:
                persist_path = getattr(getattr(self.hippocampus, "config", None), "persistence_path", None)
                if persist_path:
                    self.hippocampus.save(persist_path)
            except Exception as e:
                log.warning("Agent %s: hippocampus save failed: %s", self.agent_id, e)

        if self.nac is not None:
            try:
                nac_path = getattr(getattr(self.nac, "config", None), "persistence_path", None)
                if nac_path:
                    self.nac.save(nac_path)
            except Exception as e:
                log.warning("Agent %s: NAc save failed: %s", self.agent_id, e)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class AgentFactory:
    """Creates independent AgentInstance objects with isolated subsystems.

    Each agent gets its own memory systems, tool registry, and executor.
    The LLM Router is **shared** across all agents (expensive resource)
    but session cost tracking is per-agent.
    """

    def __init__(
        self,
        component_registry: Any | None = None,
        base_data_dir: Path | str | None = None,
    ) -> None:
        self._component_registry = component_registry
        if base_data_dir:
            self._base_data_dir = Path(base_data_dir)
        else:
            from maxim.utils.paths import data_home

            self._base_data_dir = data_home() / "agents"
        self._base_data_dir.mkdir(parents=True, exist_ok=True)

    def create_agent(self, config: AgentConfig, *, auto_load: bool = False) -> AgentInstance:
        """Create a fully independent agent with its own memory systems.

        Each agent gets:
        - Hippocampus (separate persistence path)
        - NAc (separate causal model)
        - ATL (separate concept layer)
        - MemoryHub (independent coordinator)
        - ToolRegistry (scoped to role via tool_whitelist)

        Args:
            config: Agent configuration.
            auto_load: If True, restore persisted state from the agent's
                persistence directory.  Used by ``maxim.load.agent()``.
                Default False = always start fresh (``maxim.create.agent()``).
        """
        agent_dir = self._resolve_persistence_dir(config)

        # Create memory subsystems
        hippocampus = None
        nac = None
        atl = None
        memory_hub = None

        if config.remembers:
            hippocampus = self._create_hippocampus(agent_dir, auto_load=auto_load)

        if config.learns:
            nac = self._create_nac(agent_dir, auto_load=auto_load)

        atl = self._create_atl(agent_dir)
        memory_hub = self._create_memory_hub(hippocampus, nac, atl, agent_dir=agent_dir)

        # Create tool registry (scoped by whitelist)
        tool_registry = self._create_tool_registry(config.tool_whitelist)

        # Create entity from component registry if spec provided
        entity = None
        if config.entity_spec and self._component_registry:
            try:
                entity = self._component_registry.instantiate(config.entity_spec)
            except Exception as e:
                log.warning("Agent %s: failed to instantiate entity '%s': %s", config.agent_id, config.entity_spec, e)

        instance = AgentInstance(
            agent_id=config.agent_id,
            role=config.role,
            config=config,
            hippocampus=hippocampus,
            nac=nac,
            atl=atl,
            memory_hub=memory_hub,
            tool_registry=tool_registry,
            entity=entity,
            personality=config.personality,
        )

        log.info(
            "Created agent '%s' (role=%s, remembers=%s, learns=%s)",
            config.agent_id,
            config.role,
            config.remembers,
            config.learns,
        )
        return instance

    def create_npc_agent(
        self,
        npc_name: str,
        entity_ref: str | None = None,
        personality: str = "",
        model_profile: str = "small",
        remembers: bool = True,
        learns: bool = True,
    ) -> AgentInstance:
        """Convenience: create a lightweight NPC agent.

        NPCs get:
        - Restricted tool set (speak, choose, memory_recall, sense)
        - Cheaper LLM tier (small by default)
        - Full bio-stack if remembers=True and learns=True
        - Personality injected into system prompt
        """
        config = AgentConfig(
            agent_id=f"npc_{npc_name}",
            role="npc",
            entity_spec=entity_ref,
            model_profile=model_profile,
            tool_whitelist={"speak", "choose", "memory_recall", "think"},
            personality=personality,
            remembers=remembers,
            learns=learns,
        )
        return self.create_agent(config)

    # -- private subsystem creation -----------------------------------------

    def _resolve_persistence_dir(self, config: AgentConfig) -> Path:
        """Resolve or create per-agent persistence directory."""
        if config.persistence_dir:
            p = Path(config.persistence_dir)
        else:
            p = self._base_data_dir / config.agent_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _create_hippocampus(self, agent_dir: Path, *, auto_load: bool = False) -> Any:
        """Create a Hippocampus with per-agent persistence.

        Args:
            agent_dir: Directory for persistence files.
            auto_load: If True, load existing state from disk (used by load.agent).
                       If False, always create fresh (used by create.agent).
        """
        try:
            from maxim.memory.hippocampus import Hippocampus, HippocampusConfig

            hippo_path = agent_dir / "hippocampus.json"
            hippo = Hippocampus(
                HippocampusConfig(
                    persistence_path=str(hippo_path),
                )
            )
            if auto_load and hippo_path.exists():
                try:
                    hippo.load(str(hippo_path))
                except Exception:
                    pass  # Start fresh if corrupt
            return hippo
        except Exception as e:
            log.warning("Failed to create Hippocampus: %s", e)
            return None

    def _create_nac(self, agent_dir: Path, *, auto_load: bool = False) -> Any:
        """Create a NAc with per-agent persistence.

        Args:
            agent_dir: Directory for persistence files.
            auto_load: If True, load existing state from disk.
        """
        try:
            from maxim.decisions.nac import NAc, NACConfig

            nac_path = str(agent_dir / "nac.json")
            nac = NAc(NACConfig(persistence_path=nac_path))
            if auto_load and (agent_dir / "nac.json").exists():
                nac.load_safe(nac_path)
            return nac
        except Exception as e:
            log.warning("Failed to create NAc: %s", e)
            return None

    def _create_atl(self, agent_dir: Path) -> Any:
        """Create an ATL with per-agent persistence."""
        try:
            from maxim.memory.atl import ATL, ATLConfig

            return ATL(
                ATLConfig(
                    persistence_path=str(agent_dir / "atl.json"),
                )
            )
        except Exception as e:
            log.warning("Failed to create ATL: %s", e)
            return None

    def _create_memory_hub(
        self,
        hippocampus: Any | None,
        nac: Any | None,
        atl: Any | None,
        agent_dir: Path | None = None,
    ) -> Any:
        """Create a MemoryHub coordinating the agent's memory systems."""
        try:
            from maxim.integration.memory_hub import MemoryHub
            from maxim.similarity.ec import EntorhinalCortex
            from maxim.time.scn import SCN

            # F0.5: SCN persistence path is bound at construction time, not
            # late-assigned. The old pattern
            #   scn = SCN(); scn._persistence_path = str(agent_dir / ...)
            # had a race window under concurrent agent construction where
            # two agents could overwrite each other's path binding. Passing
            # it to the constructor eliminates the window.
            scn_path_str: str | None = None
            if agent_dir is not None:
                scn_path_str = str(agent_dir / "scn.json")
            scn = SCN(persistence_path=scn_path_str)
            if scn_path_str is not None:
                scn_file = Path(scn_path_str)
                if scn_file.exists():
                    try:
                        scn.load(scn_path_str)
                    except Exception:
                        pass  # Start fresh if corrupt
            return MemoryHub(
                hippocampus=hippocampus,
                scn=scn,
                nac=nac,
                ec=EntorhinalCortex(),
                atl=atl,
            )
        except Exception as e:
            log.warning("Failed to create MemoryHub: %s", e)
            return None

    def _create_tool_registry(self, whitelist: set[str] | None) -> Any:
        """Create a ToolRegistry, optionally filtered by whitelist."""
        try:
            from maxim.tools.registry import ToolRegistry

            registry = ToolRegistry()

            # If whitelist is set, we'll filter tools at registration time.
            # For now, return empty registry — tools are registered by the
            # runtime (DM, generative runner, etc.) based on the encounter.
            if whitelist:
                registry._tool_whitelist = whitelist  # type: ignore[attr-defined]
            return registry
        except Exception as e:
            log.warning("Failed to create ToolRegistry: %s", e)
            return None
