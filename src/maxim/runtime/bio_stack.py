"""Canonical bio-pipeline construction site (Wave 3, biosystem_unification).

``build_bio_stack`` collapses the ~30-line bio-system construction block
reproduced at every agent entry point into a single function call
returning a frozen ``BioStack`` dataclass.  It composes the four
individual builders shipped in Waves 1 + 2:

    build_reaction_bus  (Wave 1, reactions/bus.py)
    build_pain_bus      (Wave 1, proprioception/pain_bus.py)
    build_memory_hub    (Wave 2, integration/memory_hub.py)
    build_default_network (Wave 2, runtime/bootstrap.py)

Construction order is load-bearing — each builder depends on systems
constructed by prior steps::

    Hippocampus → NAc → SCN → EC → (ATL, AngularGyrus) →
    MemoryHub → PainBus → DefaultNetwork

See also:
    - ``docs/plans/bio_stack_unification.md`` for the plan + audit
    - ``docs/plans/biosystem_unification.md`` for the wave catalog
    - ``docs/architecture/structural_enforcement.md`` for the meta-rule
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from maxim.decisions.nac import NAc
    from maxim.default_network import DefaultNetworkConfig
    from maxim.integration.memory_hub import MemoryHub
    from maxim.math.angular_gyrus import AngularGyrus
    from maxim.memory.atl import ATL
    from maxim.memory.hippocampus import Hippocampus
    from maxim.proprioception.pain_bus import PainBus, PainSignal
    from maxim.reactions.bus import ReactionBus
    from maxim.similarity.ec import EntorhinalCortex
    from maxim.time.scn import SCN

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BioStack:
    """Frozen container for a fully-wired bio-pipeline.

    All fields are populated by ``build_bio_stack``.  Callers access
    individual systems via attribute access (``bio.memory_hub``,
    ``bio.pain_bus``, etc.) and pass them to downstream consumers
    (``build_executor``, ``agent.wire_memory_hub``, etc.).

    Frozen so that bio-system references cannot be accidentally
    swapped after construction.  Internal mutation on the contained
    objects (e.g., ``memory_hub.connect()`` for spatial/salience
    wiring on the Reachy path) is unaffected by the frozen constraint.
    """

    hippocampus: Hippocampus
    nac: NAc
    scn: SCN
    ec: EntorhinalCortex
    atl: ATL | None
    angular_gyrus: AngularGyrus | None
    memory_hub: MemoryHub
    pain_bus: PainBus
    reaction_bus: ReactionBus
    default_network: Any  # DefaultNetwork | None — optional dep


def build_bio_stack(
    *,
    persistence_dir: Path | str | None = None,
    # Pre-built PainBus (sim AUT pattern: bus created early for sandbox,
    # standard learners subscribed here instead of in build_pain_bus).
    pain_bus: "PainBus | None" = None,
    # DefaultNetwork control
    with_default_network: bool = False,
    dn_maxim: object | None = None,
    dn_bus: Any | None = None,
    dn_fear_agent: Any | None = None,
    dn_config: "DefaultNetworkConfig | None" = None,
    dn_frame_size: tuple[int, int] = (640, 480),
    # Extra wiring
    additional_pain_subscribers: tuple[Callable[["PainSignal"], None], ...] = (),
) -> BioStack:
    """Construct the full bio-pipeline as a coherent unit.

    This is the canonical bio-pipeline construction site (Wave 3 of
    biosystem_unification).  Every production agent entry point calls
    this builder instead of hand-rolling ~30 lines of bio-system
    construction.  The builder composes the four individual Wave 1 + 2
    builders in the correct dependency order.

    Args:
        persistence_dir: Root directory for bio-system persistence
            files.  When provided, sub-paths are derived internally
            (``hippocampus.json``, ``atl.json``, ``angular_gyrus.json``).
            When ``None``, bio-systems start in-memory (sim AUT, tests).
        pain_bus: Pre-constructed PainBus.  When provided, the builder
            subscribes the standard learners (hippocampus → memory,
            NAc → causal) to this bus instead of calling
            ``build_pain_bus``.  Use for the sim AUT pattern where the
            sandbox needs a PainBus before the rest of the bio-stack.
        with_default_network: Whether to construct a DefaultNetwork.
            Only the Reachy robot path and sim AUT need one.
        dn_maxim: ``Maxim`` instance for motor control (Reachy only).
            ``None`` = headless/sim.
        dn_bus: ``AgentBus`` for DN message publishing.
        dn_fear_agent: ``FearAgent`` for movement gating within DN.
        dn_config: Pre-built ``DefaultNetworkConfig``.
        dn_frame_size: Video frame dimensions for peripheral calcs.
        additional_pain_subscribers: Extra ``PainSignal`` callbacks
            registered after the standard learners.

    Returns:
        A frozen ``BioStack`` with all systems wired.

    Examples:
        CLI non-sim agent (the most common shape)::

            bio = build_bio_stack(persistence_dir=memory_path)
            agent.wire_memory_hub(bio.memory_hub)
            executor = build_executor(registry, pain_bus=bio.pain_bus, ...)

        Sim AUT with early PainBus for sandbox::

            early_bus = PainBus()  # given to sandbox at sim start
            bio = build_bio_stack(pain_bus=early_bus, with_default_network=True)
            aut_agent.wire_memory_hub(bio.memory_hub)

        Reachy robot with DefaultNetwork::

            bio = build_bio_stack(
                persistence_dir=user_memory(),
                with_default_network=True,
                dn_maxim=self,
                dn_bus=agent_bus,
                dn_fear_agent=fear_agent,
            )
    """
    # -- Step 1: Core bio-systems ------------------------------------------
    from maxim.decisions.nac import NAc
    from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
    from maxim.similarity.ec import EntorhinalCortex
    from maxim.time.scn import SCN

    p = Path(persistence_dir) if persistence_dir is not None else None

    # One-time migration: the CLI historically used "memories.json" for
    # hippocampus persistence. Standardize on "hippocampus.json" (which
    # the Reachy path already used). If the old file exists and the new
    # one doesn't, rename it so existing users don't lose data.
    if p is not None:
        _old_hippo = p / "memories.json"
        _new_hippo = p / "hippocampus.json"
        if _old_hippo.exists() and not _new_hippo.exists():
            try:
                _old_hippo.rename(_new_hippo)
                logger.info("Migrated hippocampus persistence: %s -> %s", _old_hippo, _new_hippo)
            except OSError as e:
                logger.warning("Failed to migrate hippocampus persistence: %s", e)

    hippocampus = Hippocampus(
        config=HippocampusConfig(
            persistence_path=str(p / "hippocampus.json") if p is not None else None,
        )
    )
    nac = NAc()
    scn = SCN()
    ec = EntorhinalCortex()

    # -- Step 2: Optional multi-layer memory (ATL + AngularGyrus) ----------
    atl = None
    angular_gyrus = None
    try:
        from maxim.memory.atl import ATL, ATLConfig

        atl = ATL(
            config=ATLConfig(
                persistence_path=str(p / "atl.json") if p is not None else None,
            )
        )
    except Exception:
        logger.debug("ATL not available")

    try:
        from maxim.math.angular_gyrus import AngularGyrus, AngularGyrusConfig

        angular_gyrus = AngularGyrus(
            config=AngularGyrusConfig(
                persistence_path=str(p / "angular_gyrus.json") if p is not None else None,
            )
        )
    except Exception:
        logger.debug("AngularGyrus not available")

    # -- Step 3: MemoryHub (depends on hippocampus, nac, scn, ec) ----------
    from maxim.integration.memory_hub import build_memory_hub

    memory_hub = build_memory_hub(
        hippocampus=hippocampus,
        scn=scn,
        nac=nac,
        ec=ec,
        atl=atl,
        angular_gyrus=angular_gyrus,
    )

    # -- Step 4: PainBus ---------------------------------------------------
    # Always route through build_pain_bus (the Wave 1 structural door)
    # so that subscriber wiring is centralized. When a pre-built bus
    # is provided (sim AUT pattern: sandbox needs the bus early), pass
    # it via bus= so build_pain_bus subscribes to it instead of
    # constructing a new one. This prevents the N-sites-drift bug class
    # that Wave 1 was designed to eliminate.
    from maxim.proprioception.pain_bus import build_pain_bus

    pain_bus = build_pain_bus(
        hippocampus=hippocampus,
        nac=nac,
        additional_subscribers=additional_pain_subscribers,
        bus=pain_bus,
    )

    # ReactionBus is owned by PainBus — expose it on the stack for
    # callers that need typed Reaction semantics (cerebellum, etc.).
    reaction_bus = pain_bus.reaction_bus

    # -- Step 5: DefaultNetwork (optional) ---------------------------------
    default_network = None
    if with_default_network:
        from maxim.runtime.bootstrap import build_default_network

        default_network = build_default_network(
            nac=nac,
            maxim=dn_maxim,
            bus=dn_bus,
            pain_bus=pain_bus,
            fear_agent=dn_fear_agent,
            config=dn_config,
            frame_size=dn_frame_size,
        )

    return BioStack(
        hippocampus=hippocampus,
        nac=nac,
        scn=scn,
        ec=ec,
        atl=atl,
        angular_gyrus=angular_gyrus,
        memory_hub=memory_hub,
        pain_bus=pain_bus,
        reaction_bus=reaction_bus,
        default_network=default_network,
    )


__all__ = ["BioStack", "build_bio_stack"]
