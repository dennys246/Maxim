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
from typing import Any, Literal

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

    # F2: Full agent construction config (used by create_full_agent)
    with_bio_stack: bool = False  # Construct BioStack (pain_bus + reaction_bus etc.)
    with_executor: bool = False  # Construct Executor via build_executor
    with_pain_bridge: bool = True  # Subscribe ToolPainBridge to PainBus for out-of-band pain.
    # NOTE: bridge CONSTRUCTION is always gated on nac (not pain_bus).
    # with_pain_bridge only controls SUBSCRIPTION. When True + with_bio_stack=True,
    # both the bridge and create_pain_nac_subscriber are subscribed to the same
    # PainBus — correctness is load-bearing on the context-similarity mismatch
    # (see pain_bus_bridge_subscriber_unification.md). Set False when the agent's
    # tools don't produce SEM embodiment pain (e.g., orchestrator sim tools).
    with_fear_gate: bool = False  # Wrap executor with FearGatedExecutor
    embodiment_ref: str | None = None  # SEM component ref for embodiment (non-sim only)

    # Cross-session persistence (nac_cross_session_persistence.md): whether
    # build_bio_stack RESTORES persisted state (hippocampus/NAc/EC/cerebellum)
    # from persistence_dir at construction. Default True — a persistent agent
    # home is expected to remember. Set False for agents that must start each
    # run fresh while keeping their writes (the sim orchestrator NPC: its
    # ~/.maxim/orchestrator home accumulates state for the future "Phase 3"
    # cross-session-orchestration work, but reading months of unaudited
    # accumulation into every sim run would make the narrator/orchestrator a
    # cross-run confound — the narrator-state-confound class. Review fold,
    # Arch #2.) NOTE: this is deliberately separate from create_agent's
    # ``auto_load`` parameter, which governs only the skeleton subsystems
    # that create_full_agent discards when with_bio_stack=True.
    load_persisted: bool = True


# ---------------------------------------------------------------------------
# Agent Instance
# ---------------------------------------------------------------------------


def _maybe_wire_body_state(instance: "AgentInstance") -> None:
    """Exp 44 opt-in body_state wiring (arms B/C of the pre-registered
    ablation, docs/plans/acting_coach_body_state_ablation.md).

    Default OFF preserves the auto-sense status quo (arm A / all current
    behavior). When ``MAXIM_ENABLE_BODY_STATE_PROMPT`` is truthy AND the
    instance has both an embodiment and a memory hub, route the executor's
    Embodiment into ``MemoryHub.embodiment`` so memory_agent's
    ``format_body_state_for_prompt`` populates ``StructuredContext.
    body_state``. This closes the silent gap the 2026-07-14 deep-dive found
    (hub.embodiment was never wired at any production call site) — but only
    behind the experiment flag until the ablation earns a default.
    """
    if instance.embodiment is None or instance.memory_hub is None:
        return
    from maxim.integration.memory_hub import body_state_prompt_enabled

    if body_state_prompt_enabled():
        instance.memory_hub.embodiment = instance.embodiment


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

    # Bio-pipeline (F2: populated by create_full_agent)
    bio_stack: Any | None = None  # BioStack from build_bio_stack
    pain_bus: Any | None = None  # PainBus (shortcut to bio_stack.pain_bus)

    # Execution
    tool_registry: Any | None = None
    executor: Any | None = None

    # Embodiment
    entity: Any | None = None
    embodiment: Any | None = None  # Embodiment wrapper (from build_executor)

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

    def shutdown(self, *, consolidation: Literal["full", "lightweight"] = "full") -> None:
        """Flush memories and clean up resources.

        Args:
            consolidation: Explicit session-end flavor (HANDLE seam, part b —
                mirrors ``end_bio_session``): ``"full"`` (default) runs the
                blocking sleep/replay consolidation via ``on_session_end``;
                ``"lightweight"`` persists state but skips the replay via
                ``on_session_end_lightweight``. The default keeps every
                existing caller byte-identical, and is the count-silent-
                failures choice: wrongly lightweight loses consolidation
                silently, wrongly full is loud-but-harmless slowness.
        """
        if consolidation not in ("full", "lightweight"):
            raise ValueError(f"consolidation must be 'full' or 'lightweight', got {consolidation!r}")
        if self.memory_hub is not None:
            try:
                if consolidation == "lightweight":
                    self.memory_hub.on_session_end_lightweight()
                else:
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

        # Review fix (Arch #2): save cerebellum forward models on shutdown.
        # Without this, learned forward models are lost every session.
        if self.bio_stack is not None:
            try:
                self.bio_stack.save_cerebellum()
            except Exception as e:
                log.warning("Agent %s: cerebellum save failed: %s", self.agent_id, e)


# ---------------------------------------------------------------------------
# Nickname derivation for sim logging
# ---------------------------------------------------------------------------


def _derive_nickname(config: AgentConfig) -> str:
    """Derive a short, thematic display nickname for an agent.

    Priority: entity_spec name > personality first word > agent_id.
    Keeps it <=12 chars for display column alignment.
    """
    # Entity spec: "weapons/rusty_sword" → "Rusty Sword"
    if config.entity_spec:
        # Take the last path component and humanize it
        raw = config.entity_spec.rsplit("/", 1)[-1]
        name = raw.replace("_", " ").replace("-", " ").title()
        return name[:12]

    # Personality: "A grizzled merchant who..." → "Merchant"
    if config.personality:
        # Find the first capitalized or interesting word after common articles
        words = config.personality.split()
        skip = {"a", "an", "the", "is", "was", "are", "very", "quite"}
        for word in words:
            clean = word.strip(".,;:!?\"'")
            if clean.lower() not in skip and len(clean) > 1:
                return clean.capitalize()[:12]

    # NPC agents: "npc_merchant" → "Merchant"
    if config.agent_id.startswith("npc_"):
        return config.agent_id[4:].replace("_", " ").title()[:12]

    # Fallback: use agent_id directly
    return config.agent_id[:12] if config.agent_id else "Agent"


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
        memory_hub = self._create_memory_hub(hippocampus, nac, atl, agent_dir=agent_dir, agent_id=config.agent_id)

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

        # Register a display nickname for sim logging. Prefer entity name
        # (thematic), fall back to personality-derived name, then agent_id.
        nickname = _derive_nickname(config)
        try:
            from maxim.simulation.sim_logger import register_agent_nickname

            register_agent_nickname(config.agent_id, nickname)
        except Exception:
            pass

        log.info(
            "Created agent '%s' (nickname='%s', role=%s, remembers=%s, learns=%s)",
            config.agent_id,
            nickname,
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

    def create_full_agent(
        self,
        config: AgentConfig,
        *,
        tool_registry: Any = None,
        pain_bus: Any = None,
        fear_llm: Any = None,
        auto_load: bool = False,
    ) -> AgentInstance:
        """Create a fully wired agent with bio-stack, executor, and fear gating.

        This is the F2 canonical agent construction entry point.  It
        composes ``create_agent`` (memory subsystems) with
        ``build_bio_stack`` (bio-pipeline) and ``build_executor``
        (tool execution).  The Z1 design decision means the Executor
        is built ONCE per agent and reused across turns.

        Args:
            config: Agent configuration.  ``with_bio_stack``,
                ``with_executor``, and ``with_fear_gate`` control
                which layers are constructed.
            tool_registry: Pre-built ToolRegistry.  Required when
                ``config.with_executor`` is True.  CLI builds this
                via ``build_tool_registry()`` before calling the factory.
            pain_bus: Pre-built PainBus (sim AUT pattern — sandbox needs
                the bus before the rest of the stack).  When provided,
                ``build_bio_stack`` subscribes standard learners to this
                bus instead of constructing a new one.
            fear_llm: LLM router for FearAgent code analysis.  When
                ``None`` (default), FearAgent uses pattern-matching only.
                Pass the agent's LLM router for full LLM-powered safety
                review (sim mode uses this).
            auto_load: Restore persisted state on the ``create_agent``
                SKELETON subsystems only. When ``with_bio_stack=True``
                those skeleton instances are discarded and replaced by
                the bio-stack's — whose restore is governed by
                ``config.load_persisted`` (default True), NOT by this
                flag. Bio-stack persistence is always-on: an agent home
                with prior files is restored unless
                ``load_persisted=False``.

        Returns:
            A fully wired AgentInstance.

        Raises:
            ValueError: ``with_executor`` is True but ``tool_registry``
                was not provided.
        """
        if config.with_executor and tool_registry is None:
            raise ValueError("create_full_agent requires tool_registry when config.with_executor=True")

        # Step 1: Memory subsystems (hippocampus, NAc, ATL, MemoryHub)
        instance = self.create_agent(config, auto_load=auto_load)
        instance.tool_registry = tool_registry or instance.tool_registry

        # Step 2: Bio-stack (PainBus, ReactionBus, cerebellum, etc.)
        # Bio-stack failure propagates — with_bio_stack=True is an intent,
        # not a preference.  Pre-merge review (Exec #3, Arch #3) flagged
        # the silent-degradation risk of catching here.
        if config.with_bio_stack:
            from maxim.runtime.bio_stack import build_bio_stack

            agent_dir = self._resolve_persistence_dir(config)
            bio = build_bio_stack(
                persistence_dir=str(agent_dir),
                pain_bus=pain_bus,
                agent_id=config.agent_id,
                load_persisted=config.load_persisted,
            )
            instance.bio_stack = bio
            instance.pain_bus = bio.pain_bus
            # Upgrade memory subsystems from bio-stack (they have
            # persistence wiring that create_agent's versions may lack).
            # The create_agent versions are immediately discarded — this
            # wastes construction (review #4/#7) but is correct.  A
            # _create_agent_skeleton optimization is deferred.
            #
            # This overwrite is safe ONLY because build_bio_stack itself
            # loads persisted state (hippocampus/NAc/EC/cerebellum) —
            # bio.nac IS the restored instance. Before that landed
            # (nac_cross_session_persistence.md), this line silently
            # discarded create_agent's auto-loaded NAc, which is why a
            # save-only persistence patch TRUNCATED every prior session.
            instance.hippocampus = bio.hippocampus
            instance.nac = bio.nac
            instance.atl = bio.atl
            instance.memory_hub = bio.memory_hub

        # Step 3: Executor (tool execution with ToolPainBridge)
        if config.with_executor and instance.tool_registry is not None:
            if config.with_executor and not config.with_bio_stack:
                log.warning(
                    "Agent %s: executor without bio-stack produces a learning-disabled agent",
                    config.agent_id,
                )
            from maxim.runtime.bootstrap import build_executor

            bio = instance.bio_stack
            # with_pain_bridge controls whether the ToolPainBridge
            # subscribes to the PainBus for out-of-band pain signals.
            # Bridge CONSTRUCTION is always gated on nac (see
            # build_executor docstring): pass nac even when pain_bus
            # is None so the bridge exists for direct attribution
            # (record_tool_complete / record_tool_embodiment_failure).
            _exec_pain_bus = instance.pain_bus if config.with_pain_bridge else None
            executor = build_executor(
                instance.tool_registry,
                pain_bus=_exec_pain_bus,
                nac=instance.nac,
                hippocampus=instance.hippocampus,
                scn=bio.scn if bio is not None else None,
                entity_ref=config.embodiment_ref,
                component_registry=self._component_registry,
                cerebellum=bio.cerebellum if bio is not None else None,
                distributor=bio.distributor if bio is not None else None,
                agent_id=config.agent_id,
            )
            # Review fix (Exec #1): attribute is `embodiment`, not `_embodiment`.
            # The old CLI code had the identical bug — always returned None.
            instance.embodiment = getattr(executor, "embodiment", None)
            instance.executor = executor
            _maybe_wire_body_state(instance)

        # Step 4: Fear gating (wraps executor with FearGatedExecutor)
        if config.with_fear_gate and instance.executor is not None:
            try:
                from maxim.agents.fear_agent import FearAgent
                from maxim.runtime.fear_gate import FearGatedExecutor

                # FearAgent gets the caller's LLM router for code analysis
                # when fear_llm is provided (sim mode).  Otherwise falls
                # back to pattern-matching only (headless, NPC).
                fear_agent = FearAgent(llm=fear_llm)
                instance.executor = FearGatedExecutor(instance.executor, fear_agent)
            except Exception as e:
                log.warning("Agent %s: FearGatedExecutor failed: %s", config.agent_id, e)

        log.info(
            "Created full agent '%s' (bio_stack=%s, executor=%s, fear_gate=%s)",
            config.agent_id,
            instance.bio_stack is not None,
            instance.executor is not None,
            config.with_fear_gate and instance.executor is not None,
        )
        return instance

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
        *,
        agent_dir: Path | None = None,
        agent_id: str,
    ) -> Any:
        """Create a MemoryHub coordinating the agent's memory systems."""
        try:
            from maxim.integration.memory_hub import build_memory_hub
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
            # build_memory_hub calls .connect() internally, so NPC agents
            # now get PlanHistoryBridge + EscalationLearningBridge +
            # FearCircuitBridge.  Pre-Wave-2 these were permanently dead
            # (bare MemoryHub() with no .connect() call).  See
            # memory_hub_unification.md Gap B.
            return build_memory_hub(
                hippocampus=hippocampus,
                scn=scn,
                nac=nac,
                ec=EntorhinalCortex(),
                atl=atl,
                agent_id=agent_id,
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
