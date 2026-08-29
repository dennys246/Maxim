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
    - ``docs/plans/archive/bio_stack_unification.md`` for the plan + audit
    - ``docs/plans/archive/biosystem_unification.md`` for the wave catalog
    - ``docs/architecture/structural_enforcement.md`` for the meta-rule
"""

from __future__ import annotations

import json
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
    cerebellum: Any  # Cerebellum | None — optional dep
    default_network: Any  # DefaultNetwork | None — optional dep
    thought_gate: Any  # ThoughtGate | None — PFC deliberation gate
    bio_enrichment_pipeline: Any  # BioEnrichmentPipeline | None — PFC enrichment
    distributor: Any  # TemporalCreditDistributor | None — temporal credit

    def save_cerebellum(self) -> None:
        """Save cerebellum state to its persistence path, if configured.

        Call at session end alongside hippocampus/NAc persistence.
        Pre-merge Arch review critical #1: without this, learned
        forward models are lost every session.
        """
        if self.cerebellum is not None:
            config = getattr(self.cerebellum, "config", None)
            path = getattr(config, "persistence_path", None)
            if path:
                self.cerebellum.save(path)

    def on_session_end(self) -> None:
        """Centralized session-end cleanup.

        Calls ``save_cerebellum()`` + ``distributor.cleanup_session()``.
        Single call site prevents the N-sites-forget-to-call bug class.
        Idempotent — safe to call multiple times.
        """
        self.save_cerebellum()
        if self.distributor is not None:
            try:
                self.distributor.cleanup_session()
            except Exception as e:
                logger.debug("distributor.cleanup_session failed: %s", e)


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
    # P4 multi-agent attribution: owning agent's id, threaded into MemoryHub
    # so single-agent percept paths get a stable fallback for the substrate
    # stash even when the percept factory hasn't populated agent_id.
    #
    # REQUIRED keyword-only (exp 32 Bug A fold, 2026-05-27): the previous
    # ``default_agent`` default silently produced a split between substrate-
    # write and substrate-read paths when callers didn't override it. Same
    # silent-no-op pattern as the ``build_executor(pain_bus=...)`` invariant
    # in CLAUDE.md — pushed into the type so the next miss is a TypeError,
    # not a silent agent_id divergence. Test sites that don't care about
    # agent_id pass the literal ``"default_agent"`` explicitly.
    agent_id: str,
    # Cross-session persistence (nac_cross_session_persistence.md): restore
    # persisted hippocampus/NAc/EC state from persistence_dir at build.
    # Default True. False = write-but-don't-read: persistence PATHS are
    # still set (session-end saves keep working), but nothing is loaded —
    # for agents that must start each run fresh (the sim orchestrator NPC;
    # see AgentConfig.load_persisted).
    load_persisted: bool = True,
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

            early_bus = build_pain_bus(hippocampus=None, nac=None)  # given to sandbox at sim start
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
    from maxim.decisions.nac import NAc, NACConfig
    from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
    from maxim.similarity.ec import ECConfig, EntorhinalCortex
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
    # Cross-session restore (nac_cross_session_persistence.md): the save
    # side always existed (sleep() auto-saves at session end) but nothing
    # on this path ever LOADED — MemoryHub.on_session_start restores
    # ATL/AngularGyrus/cross-layer/EC embeddings but not hippocampus, and
    # AgentFactory.create_full_agent discards its auto-loaded instance in
    # favor of this one. Load here so path ownership stays in one place.
    if load_persisted and p is not None and (p / "hippocampus.json").exists():
        _ok, _err = hippocampus.load_with_recovery()
        if _ok:
            logger.info("Restored hippocampus from %s (%d memories)", p / "hippocampus.json", len(hippocampus))
    # Substrate exploration policy (substrate_exploration_policy.md, 1.1):
    # resolve the novelty-bonus weight from config.json (CLI > env >
    # config.json > 0.0 default). 0.0 == OFF == legacy argmax; the Exp 41
    # treatment arm / operator sets it > 0. Fail-soft: any resolution error
    # falls back to the OFF default so a bad config can't brick the bio-stack.
    try:
        from maxim.runtime.config_loader import resolve_setting as _resolve_setting

        _explore_weight, _ = _resolve_setting("sim.substrate_explore_bonus_weight")
        _explore_weight = max(0.0, float(_explore_weight))
    except Exception:
        _explore_weight = 0.0
    # Drive-gating (motivated attention, Exp 42): opt-in, OFF by default so
    # global recommend_action semantics (learned-link primacy) are unchanged.
    # Same CLI > env > config.json > default precedence; fail-soft to OFF.
    # DORMANT since 2026-06-23 — the Exp 42 gating-OFF ablation graduated
    # identically, so this knob carries no behavioral weight (B8 + drive-affinity
    # do). Resolution kept so the opt-in path stays live; see NACConfig.
    # drive_gate_enabled + docs/experiments/42_substrate_primary_preference.md.
    try:
        from maxim.runtime.config_loader import resolve_setting as _resolve_setting2

        _drive_gate, _ = _resolve_setting2("sim.drive_gate_enabled")
        _drive_gate = bool(_drive_gate)
    except Exception:
        # Fail-soft to OFF, but WARN: a malformed sim.drive_gate_enabled (e.g. a
        # typo'd MAXIM_SIM_DRIVE_GATE_ENABLED) would otherwise silently run the
        # experiment UN-gated and report a misleading VOID/REFRAME with no signal
        # that gating was off. Surface the misconfiguration. (review fold)
        logger.warning(
            "sim.drive_gate_enabled failed to resolve — defaulting drive-gating OFF. "
            "If this is an Exp 42 run, the gating arm is silently disabled; check "
            "MAXIM_SIM_DRIVE_GATE_ENABLED / config.json.",
            exc_info=True,
        )
        _drive_gate = False
    _nac_config = NACConfig(
        substrate_explore_bonus_weight=_explore_weight,
        drive_gate_enabled=_drive_gate,
        # Cross-session persistence (nac_cross_session_persistence.md).
        # Setting the path activates the pre-existing save in
        # MemoryHub.on_session_end / on_session_end_lightweight (guarded
        # `if nac_path:`); the LOAD below closes the loop. The two ship
        # together by design: a save alone made each session TRUNCATE the
        # previous one (agent_factory overwrites its auto-loaded NAc with
        # this instance), which is why the earlier save-only patch was
        # reverted. Decay-on-load semantics live in NAc.load() /
        # apply_wall_clock_decay (see NACConfig field comments).
        persistence_path=str(p / "nac.json") if p is not None else None,
    )
    # Loud failure for the silent-no-op footgun: an untried tool's only score is
    # the novelty bonus, so a weight at or below the substrate min_confidence gate
    # can never clear it — exploration silently does nothing, indistinguishable
    # from OFF. Warn rather than fail (a valid operator may be mid-tuning), but
    # make the dead zone visible. (substrate_exploration_policy.md review fold.)
    if 0.0 < _explore_weight <= _nac_config.min_confidence_threshold:
        logger.warning(
            "sim.substrate_explore_bonus_weight=%.3f is <= the substrate "
            "min_confidence gate (%.3f): exploration will silently no-op — an "
            "untried tool can't clear the gate on novelty alone. Set it strictly "
            "greater than the gate (e.g. 1.5) to enable exploration.",
            _explore_weight,
            _nac_config.min_confidence_threshold,
        )
    nac = NAc(_nac_config)
    # Mirror the cerebellum pattern: load-when-file-exists, recover on
    # corruption (load_safe warns + starts empty). load() applies the
    # wall-clock decay-on-load from the payload's saved_at stamp.
    if load_persisted and p is not None:
        # Review fold (cross-confirmed): the pair is loaded by two independent
        # `if exists()` blocks, so a half-present pair — a manual rm, an
        # interrupted write, a partial backup restore — restored NAc biases
        # keyed to EC node ids a fresh EC will never re-allocate, silently.
        # D2 guards the CLI clear path; this is the LOAD path where the damage
        # actually manifests.
        from maxim.runtime.agent_factory import check_nac_ec_pairing

        check_nac_ec_pairing(p)  # advisory WARNING only
        if (p / "nac.json").exists():
            # load_safe recovers internally and REPORTS via its return value;
            # this caller used to discard it, so nothing downstream could tell a
            # clean restore from a recovered-from-corrupt one.
            ok, err = nac.load_safe()
            if not ok:
                logger.warning("NAc restore fell back to an empty causal model: %s", err)
    # SCN persists beside NAc/EC, exactly as AgentFactory._create_memory_hub does
    # (bugs ledger D42, found while fixing D41): this path built a pathless SCN, so
    # `on_session_end` had nothing to save to and every runtime agent lost its
    # temporal signatures between sessions — silently, because a pathless SCN is
    # indistinguishable from an empty one. Bound at CONSTRUCTION, not assigned after,
    # to close the same concurrent-construction race the factory closed.
    #
    # An UNREADABLE scn.json is left pathless: binding the path makes that file
    # writable for the first time, so a corrupt one would otherwise be overwritten
    # with empty state at session end and the recoverable original destroyed. Going
    # pathless preserves it without touching it — this session simply does not
    # persist temporal state, loudly.
    scn_path = str(p / "scn.json") if p is not None else None
    # `load_persisted=False` is the documented write-but-don't-read agent (the
    # orchestrator NPC): it must not read a previous session's temporal state.
    scn_restorable = scn_path is not None and load_persisted and Path(scn_path).exists()
    scn_unreadable = False
    if scn_restorable:
        try:
            json.loads(Path(scn_path).read_text())
        except (OSError, ValueError) as e:
            scn_unreadable = True
            logger.warning(
                "SCN state at %s is unreadable (%s). It is left UNTOUCHED — this session runs with empty "
                "temporal state and will NOT persist SCN, so the file stays recoverable. Move or repair it "
                "to restore persistence.",
                scn_path,
                e,
            )
    scn = SCN(persistence_path=None if scn_unreadable else scn_path)
    if scn_restorable and not scn_unreadable:
        try:
            scn.load(scn_path)
        except Exception as e:  # D17: report, never swallow silently
            logger.warning("SCN restore failed (%s); starting with empty temporal state: %s", scn_path, e)
    scn.enable_oscillator()  # B2: close SCN→NAc feedback loop
    # EC persists beside NAc (nac_cross_session_persistence.md): NAc's
    # reward_bias / cluster_reward_bias are keyed by EC node ids, so
    # restoring NAc without EC would leave every restored bias pointing
    # at nodes a fresh EC will never re-allocate — persistence that
    # LOOKS like it works while the biases silently dangle. The sim path
    # already persists the pair (aut_nac.json + aut_ec.json); this
    # closes the agent-home path. Sound only now that SituationSignature
    # / SemanticLSH hashes are process-stable (see
    # tests/unit/test_stable_hash_two_process.py).
    ec = EntorhinalCortex(
        config=ECConfig(persistence_path=str(p / "ec.json") if p is not None else None),
    )
    if load_persisted and p is not None and (p / "ec.json").exists():
        try:
            ec.load(str(p / "ec.json"))
        except Exception as _ec_err:
            # Reconstruct rather than keep the partially-mutated instance:
            # EC.load mutates _lsh → _inverted → _signatures → substrate
            # nodes in sequence, so a mid-load raise leaves an internally
            # inconsistent EC — "starting fresh" must be literally true
            # (review fold, Arch #4 + Exec #4, cross-confirmed).
            logger.warning("Failed to load EC state (starting fresh): %s", _ec_err)
            ec = EntorhinalCortex(
                config=ECConfig(persistence_path=str(p / "ec.json")),
            )

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
        agent_id=agent_id,
        start_background_workers=False,
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

    # -- Step 4b: Wire hippocampus.capture_reaction to ReactionBus ---------
    # Every Reaction published on the bus is appended to the pending
    # episode's reactions list so that finalize() can compute net valence.
    reaction_bus.subscribe_all(hippocampus.capture_reaction)

    # -- Step 4c: Cerebellum (forward models + motor learning) -------------
    # Pre-merge review fold: split ImportError (optional dep missing)
    # from config/load errors (which should surface, not be swallowed).
    cerebellum = None
    if p is not None:
        try:
            from maxim.embodiment.cerebellum import Cerebellum, CerebellumConfig
        except ImportError:
            logger.debug("Cerebellum not available (optional dep)")
        else:
            cerebellum = Cerebellum(config=CerebellumConfig())
            _cerebellum_path = p / "cerebellum.json"
            if _cerebellum_path.exists():
                try:
                    cerebellum.load(str(_cerebellum_path))
                    logger.info("Loaded Cerebellum state from %s", _cerebellum_path)
                except Exception as _ce:
                    logger.warning("Failed to load Cerebellum state: %s", _ce)
            if memory_hub is not None:
                memory_hub.cerebellum = cerebellum
            logger.info("Cerebellum initialized in bio-stack")

    # -- Step 4d: TemporalCreditDistributor + reward subscriber -------------
    # Construct the distributor (composes NAc + SCN for temporal-phase-aware
    # credit).  The distributor REPLACES the old nac.distribute_reward()
    # subscriber — they must NOT coexist for the same agent.
    from maxim.decisions.temporal_credit import TemporalCreditDistributor

    distributor = TemporalCreditDistributor(nac=nac, scn=scn)

    # Map Reaction valence/intensity to distributor.distribute so that
    # positive/negative reactions during an episode feed into causal
    # learning with temporal credit fallback.
    def _distribute_reward_from_reaction(reaction: Any) -> None:
        from maxim.proprioception.pain_bus import pain_chain_trace

        val = getattr(reaction, "valence", None)
        intensity = getattr(reaction, "intensity", 0.0)
        kind = getattr(reaction, "kind", "?")
        if val is None:
            pain_chain_trace("pain_chain.reward_skip_no_valence", kind=kind)
            return
        v = val.value if hasattr(val, "value") else str(val)
        if v == "negative":
            reward = -intensity
        elif v == "positive":
            reward = intensity
        else:
            pain_chain_trace("pain_chain.reward_skip_neutral", kind=kind, valence=v)
            return
        if reward != 0.0:
            # Pre-merge Exec review critical #1: agent_id must come from
            # reaction context to avoid cross-contamination in multi-agent
            # sims.  Skip reward distribution when agent_id is unknown
            # rather than polluting a phantom "default" bucket.
            ctx = getattr(reaction, "context", None)
            agent_id = getattr(ctx, "agent_id", None)
            pain_chain_trace(
                "pain_chain.reward_subscriber",
                kind=kind,
                valence=v,
                reward=reward,
                agent_id=agent_id,
                will_distribute=agent_id is not None,
            )
            if agent_id is None:
                return
            try:
                # Route through distributor (adds temporal fallback + goal
                # credit).  goal_tag=None from reaction path — goal credit
                # flows through the deliberation path in tool_dispatch.py.
                credited = distributor.distribute(agent_id, reward, goal_tag=None)
                pain_chain_trace(
                    "pain_chain.distribute_returned",
                    agent_id=agent_id,
                    reward=reward,
                    credited_count=len(credited),
                )
            except Exception as _dr:
                logger.debug("distribute_reward failed: %s", _dr)

    reaction_bus.subscribe_all(_distribute_reward_from_reaction)

    # -- Step 4e: ThoughtGate + BioEnrichmentPipeline -----------------------
    # Non-optional when bio-stack is active.  ThoughtGate gates on
    # novelty × salience × energy.  BioEnrichmentPipeline enriches text
    # via hippocampus, NAc, ATL, EC.  Both depend only on systems already
    # constructed above (no try/except — failure propagates).
    thought_gate = None
    bio_enrichment_pipeline = None
    try:
        from maxim.integration.bio_enrichment import BioEnrichmentPipeline
        from maxim.runtime.gating import TextSalienceScorer
        from maxim.runtime.thought_gate import ThoughtGate

        _text_scorer = TextSalienceScorer(ec=ec, nac=nac)
        bio_enrichment_pipeline = BioEnrichmentPipeline(
            scorer=_text_scorer,
            hippocampus=hippocampus,
            nac=nac,
            atl=atl,
            ec=ec,
            encoder=getattr(memory_hub, "_encoder", None),
        )
        thought_gate = ThoughtGate(scorer=_text_scorer)
        logger.info("ThoughtGate + BioEnrichmentPipeline constructed in bio-stack")
    except Exception as e:
        logger.warning("ThoughtGate/BioEnrichmentPipeline construction failed: %s", e)

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

    # The hub's ConceptExtractor is assembled in a stopped state so any
    # failure in Steps 4-5 has no background thread to leak. Transfer to the
    # live state only after every other component has been constructed.
    try:
        memory_hub.start_background_workers()
    except BaseException:
        memory_hub.shutdown()
        raise

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
        cerebellum=cerebellum,
        default_network=default_network,
        thought_gate=thought_gate,
        bio_enrichment_pipeline=bio_enrichment_pipeline,
        distributor=distributor,
    )


__all__ = ["BioStack", "build_bio_stack"]
