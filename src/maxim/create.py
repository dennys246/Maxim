"""Factory namespace for composable Maxim subsystems.

Provides ``maxim.create.*`` functions that instantiate standalone
bio-subsystems, agents, SEM entities, and multi-agent pools.  Each
function is a thin facade over existing internal classes — all heavy
imports are deferred to call time.

Example::

    import maxim

    hippo = maxim.create.hippocampus()
    agent = maxim.create.agent("scout", personality="cautious")
    guard = maxim.create.entity("npcs/guard")
    pool = maxim.create.pool()
"""

from __future__ import annotations

import os
from typing import Any

__all__ = [
    "hippocampus", "nac", "atl", "scn", "angular_gyrus",
    "agent", "pool",
    "entity", "embodiment", "templates",
    "router",
]


# ── Bio-subsystems ─────────────────────────────────────────────────────


def hippocampus(*, persistence_path: str | None = None, **config_kw: Any) -> Any:
    """Create a standalone Hippocampus (episodic memory).

    Args:
        persistence_path: Path for saving/loading memory state.
            If ``None``, memory is ephemeral (not persisted).

    Returns:
        ``Hippocampus`` instance with capture/recall/save/load methods.

    Example::

        hippo = maxim.create.hippocampus(persistence_path="/tmp/memory.json")
        hippo.store_observation("saw a wolf near the cave")
        memories = hippo.recall(query="wolf", limit=3)
        hippo.save()
    """
    from maxim.memory.hippocampus import Hippocampus, HippocampusConfig

    config = HippocampusConfig(persistence_path=persistence_path, **config_kw)
    return Hippocampus(config)


def nac(**config_kw: Any) -> Any:
    """Create a standalone NAc (causal learning engine).

    Returns:
        ``NAc`` instance with observe/predict/save/load methods.

    Example::

        nac = maxim.create.nac()
        nac.observe("action", "ate_mushroom", "effect", "felt_sick", -0.8, 5.0)
        prediction = nac.predict("action", "ate_mushroom")
    """
    from maxim.decisions.nac import NAc, NACConfig

    return NAc(NACConfig(**config_kw))


def atl(*, persistence_path: str | None = None, **config_kw: Any) -> Any:
    """Create a standalone ATL (semantic concept memory).

    Args:
        persistence_path: Path for saving/loading concept state.

    Returns:
        ``ATL`` instance with store/recall/find_or_create methods.

    Example::

        atl = maxim.create.atl(persistence_path="/tmp/concepts.json")
        concept = atl.find_or_create("wolf", category="creature")
    """
    from maxim.memory.atl import ATL, ATLConfig

    config = ATLConfig(persistence_path=persistence_path, **config_kw)
    return ATL(config)


def scn() -> Any:
    """Create a standalone SCN (temporal rhythm indexing).

    Returns:
        ``SCN`` instance with register/query_hour methods.

    Example::

        scn = maxim.create.scn()
        scn.register("mem_001", hour=14, day_of_month=8)
    """
    from maxim.time.scn import SCN

    return SCN()


def angular_gyrus(**config_kw: Any) -> Any:
    """Create a standalone AngularGyrus (algebraic memory + math).

    Returns:
        ``AngularGyrus`` instance with compute/recall methods.
    """
    from maxim.math.angular_gyrus import AngularGyrus, AngularGyrusConfig

    return AngularGyrus(AngularGyrusConfig(**config_kw))


# ── Agents ─────────────────────────────────────────────────────────────


def agent(
    name: str,
    *,
    personality: str | None = None,
    role: str = "npc",
    entity_ref: str | None = None,
    remembers: bool = True,
    learns: bool = True,
    tool_whitelist: set[str] | None = None,
    persistence_dir: str | None = None,
) -> Any:
    """Create a standalone agent with isolated bio-subsystems.

    Each agent gets its own Hippocampus, NAc, ATL, MemoryHub, and
    ToolRegistry.  Does NOT start the agent loop — use ``pool.run_turn()``
    or wire to your own inference.

    Args:
        name: Agent identifier (used for persistence directory naming).
        personality: System prompt overlay injected into LLM context.
        role: ``"pc"``, ``"npc"``, or ``"companion"``.
        entity_ref: SEM component template reference (e.g. ``"npcs/guard"``).
        remembers: Enable episodic memory (Hippocampus).
        learns: Enable causal learning (NAc).
        tool_whitelist: Restrict available tools to this set.
        persistence_dir: Custom persistence directory.

    Returns:
        ``AgentInstance`` with ``.hippocampus``, ``.nac``, ``.atl``,
        ``.tool_registry``, ``.entity``, ``.shutdown()`` methods.

    Example::

        agent = maxim.create.agent("scout", personality="cautious")
        agent.hippocampus.capture(perception="dark cave ahead")
        agent.shutdown()
    """
    from maxim.runtime.agent_factory import AgentConfig, AgentFactory

    factory = AgentFactory()
    config = AgentConfig(
        agent_id=name,
        role=role,
        entity_spec=entity_ref,
        personality=personality,
        remembers=remembers,
        learns=learns,
        tool_whitelist=tool_whitelist,
        persistence_dir=persistence_dir,
    )
    return factory.create_agent(config)


def pool(*, max_workers: int = 4) -> Any:
    """Create a multi-agent pool for concurrent turn execution.

    Args:
        max_workers: Max concurrent agent turns.

    Returns:
        ``AgentPool`` with ``.add()``, ``.run_turn()``, ``.run_round()``,
        ``.export_all_memories()``, ``.shutdown()`` methods.

    Example::

        pool = maxim.create.pool()
        pool.add(maxim.create.agent("guard", personality="stern"))
        pool.add(maxim.create.agent("merchant", personality="cunning"))
        result = pool.run_turn("guard", "A stranger approaches.")
        pool.shutdown()
    """
    from maxim.mesh.bus import LocalMessageBus
    from maxim.runtime.agent_pool import AgentPool

    return AgentPool(bus=LocalMessageBus(), max_workers=max_workers)


# ── SEM Entities ───────────────────────────────────────────────────────


def entity(template_ref: str, *, name: str | None = None, **overrides: Any) -> Any:
    """Instantiate a SEM entity from a component template.

    Args:
        template_ref: Template reference (e.g. ``"npcs/guard"``,
            ``"creatures/wolf"``, ``"bodies/humanoid"``).
        name: Override the entity name.
        **overrides: Additional fields to merge into the template.

    Returns:
        ``Entity`` object with sensors, modulators, children, and
        tree traversal methods.

    Example::

        guard = maxim.create.entity("npcs/guard", name="Captain Aldric")
        guard.sensors["alertness"].value = 0.9
    """
    from maxim.embodiment.component_registry import ComponentRegistry

    registry = ComponentRegistry()
    kw = dict(overrides)
    if name is not None:
        kw["name"] = name
    return registry.instantiate(template_ref, **kw)


def embodiment(root_entity: Any, **config_kw: Any) -> Any:
    """Create an Embodiment runtime from a SEM entity tree.

    Args:
        root_entity: Root ``Entity`` (from ``maxim.create.entity()``
            or manually constructed).

    Returns:
        ``Embodiment`` with ``.read_all()``, ``.evaluate_failures()``,
        ``.body_state_summary()`` methods.

    Example::

        guard = maxim.create.entity("npcs/guard")
        body = maxim.create.embodiment(guard)
        readings = body.read_all()
    """
    from maxim.embodiment.body import Embodiment, EmbodimentConfig

    config = EmbodimentConfig(**config_kw)
    return Embodiment(root_entity, config=config)


def templates() -> dict[str, list[str]]:
    """List available SEM component templates grouped by category.

    Returns:
        Dict mapping category names to lists of template names.
        E.g. ``{"npcs": ["guard", "merchant"], "creatures": ["wolf"]}``.

    Example::

        for category, names in maxim.create.templates().items():
            print(f"{category}: {', '.join(names)}")
    """
    from maxim.embodiment.component_registry import ComponentRegistry

    registry = ComponentRegistry()
    all_refs = registry.list_refs()

    grouped: dict[str, list[str]] = {}
    for ref in all_refs:
        parts = ref.split("/", 1)
        if len(parts) == 2:
            category, name = parts
        else:
            category, name = "other", parts[0]
        grouped.setdefault(category, []).append(name)

    return grouped


# ── LLM Router ─────────────────────────────────────────────────────────


def router(model: str = "mistral-7b") -> Any:
    """Create an LLM router for inference without the full agent loop.

    This is the most direct way to get LLM inference. The router handles
    model selection, JSON repair, token tracking, and backend dispatch.

    Args:
        model: LLM profile name (e.g. ``"mistral-7b"``, ``"claude-sonnet"``).

    Returns:
        ``LLMRouter`` with ``.generate()``, ``.generate_json()`` methods.

    Example::

        llm = maxim.create.router(model="claude-sonnet")
        response = llm.generate("What should I do next?", max_tokens=100)
    """
    # Set env vars in a scoped way — save originals, restore after build
    _saved_enabled = os.environ.get("MAXIM_LLM_ENABLED")
    _saved_profile = os.environ.get("MAXIM_LLM_PROFILE")
    try:
        os.environ["MAXIM_LLM_ENABLED"] = "1"
        os.environ["MAXIM_LLM_PROFILE"] = model
        from maxim.runtime.lane_backends import build_primary_router

        router_instance, _ = build_primary_router()
        return router_instance
    finally:
        # Restore original env state
        if _saved_enabled is None:
            os.environ.pop("MAXIM_LLM_ENABLED", None)
        else:
            os.environ["MAXIM_LLM_ENABLED"] = _saved_enabled
        if _saved_profile is None:
            os.environ.pop("MAXIM_LLM_PROFILE", None)
        else:
            os.environ["MAXIM_LLM_PROFILE"] = _saved_profile
