"""The agent loop's perception sections, §1.1-§1.16 (1.3.2 decomposition, slice 5).

``perceive`` is what one pass of ``agent_loop.run_agentic_loop`` does between ``state.update(observation)``
and §1.2 bio-enrichment, in the inline order: §1.1 ``imagine`` (novel entities from the percept text),
§1.15 ``auto_sense`` (passive exteroception + interoception) and §1.16 ``orient_to_audio`` (the
exteroceptive sound direction, the thalamic relay's first consumer). Each body is the inline section moved
verbatim. Before it, ``take_observation`` does §1's observe-and-merge (``sim.next_observation`` then
``state.update``) and returns ``observation`` to the call site: it is read by five later sections, and
``next_observation`` is what surfaces ``sim.current_percept``, which §1.16's gate reads.

The loop unpacks the returned ``PerceptionOutcome`` into the locals it always had, so its consumers are
unchanged: ``imagination_results`` -> §1.2's ``EnrichmentContext.resolved_entities``; ``auto_sense_text``
-> §6's ``context.auto_sense_context``; ``audio_escalate`` -> §6's B1 minimal context and
``has_meaningful_input``. §1.16 also writes ``state.data["_last_audio_orient_az"]`` (its change gate, read
only by itself on a later pass) and world-sets the body's ``azimuth`` sensor; it commands no motion.

**Arguments.** Explicit keywords, never the loop's ``LoopRun`` (rule (d) of the roadmap's import-direction
paragraph): exactly the names each section reads, each the loop's own local passed as-is and named as the
body has always named it, so the bodies are byte for byte the inline block's. That is why
``orient_to_audio`` takes ``_auto_sense_text`` with its underscore: §1.16 appends its line to §1.15's text.
The two initialisations the inline block made between §1.1 and §1.15 open the function that owns each
(``_auto_sense_text`` in ``auto_sense``, ``_audio_escalate_this_tick`` in ``orient_to_audio``). The sections
LOG on the ``maxim.runtime.agent_loop`` logger (``logger`` below is that same object), so their records keep
the agent loop's logger name.

**Known divergence, moved unchanged (#1202).** The percept-text extraction is written three times (§1.1
here, §1.15 here, §1.2 inline in the loop) and they disagree for an attribute-style observation: §1.15 reads
only ``transcript``, so an observation carrying ``cli_input`` and no ``transcript`` feeds imagination and
enrichment but is not auto-sensed. Unifying them changes LLM-primary prompt content; it is #1202's own PR.

**Patch seam.** ``agent_loop`` binds ``perceive`` by name at import (``from maxim.runtime.loop_perception import
perceive``), so a test that replaces the whole step patches ``agent_loop.perceive``; ``perceive`` reads
``imagine``, ``auto_sense`` and ``orient_to_audio`` from this module at call time, so a test that replaces one
section patches it here (``loop_perception.imagine``, ...). ``loop_perception.perceive`` itself is not a seam.

Characterization: ``tests/unit/test_loop_perception_characterization.py`` (written before the move, through
the public ``run_agentic_loop``, and kept green by it).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, NamedTuple

from maxim.utils.logging import log_swallowed_exception

# The SAME logger object as ``agent_loop.logger`` (logging returns one logger per name), so records keep
# the ``maxim.runtime.agent_loop`` name.
logger = logging.getLogger("maxim.runtime.agent_loop")


@dataclass(frozen=True)
class PerceptionOutcome:
    """What one pass's perception (§1.1-§1.16) hands the rest of the pass.

    Runtime-ephemeral: built and consumed inside one pass of ``run_agentic_loop``, never persisted and never
    sent anywhere, so the CC3 forward-compat rule for frozen dataclasses does not apply.
    """

    imagination_results: list  # §1.1 -> §1.2's EnrichmentContext.resolved_entities
    auto_sense_text: str  # §1.15, plus §1.16's line -> §6's context.auto_sense_context
    audio_escalate: bool  # §1.16 (B1) -> §6's minimal context and has_meaningful_input


def take_observation(*, sim: Any, ctrl: Any) -> Any:
    """§1: this tick's observation, merged into ``state.data``.

    A user input in it is read once, by ``take_user_input`` at the top of §1.2, which publishes what a person said to
    memory and clears the state's copy (#1414: consumed once; it used to wait for §3's ``propose_intent``, which the
    loop skips while an LLM answer is pending). The environment, the Default Network and the state are the loop's
    own, read off ``ctrl`` (the setup built it from them, and the loop never rebinds them).
    """
    observation = sim.next_observation(ctrl.environment, ctrl.default_network)
    ctrl.state.update(observation)
    return observation


def perceive(
    *,
    step_num: int,
    observation: Any,
    state: Any,
    sim: Any,
    executor: Any,
    aut_mode: str,
    imagination_trigger: Any | None,
) -> PerceptionOutcome:
    """§1.1 -> §1.15 -> §1.16 of one pass of ``run_agentic_loop``, in the inline order."""
    _imagination_results = imagine(
        step_num=step_num, observation=observation, state=state, imagination_trigger=imagination_trigger
    )
    _auto_sense_text = auto_sense(step_num=step_num, observation=observation, executor=executor)
    _auto_sense_text, _audio_escalate_this_tick = orient_to_audio(
        step_num=step_num,
        sim=sim,
        aut_mode=aut_mode,
        executor=executor,
        state=state,
        _auto_sense_text=_auto_sense_text,
    )
    return PerceptionOutcome(
        imagination_results=_imagination_results,
        auto_sense_text=_auto_sense_text,
        audio_escalate=_audio_escalate_this_tick,
    )


class UserInput(NamedTuple):
    """One tick's input, read once (``take_user_input``): the fields the loop reads, and ``addressed``, the words
    that ADDRESS the agent ("" when none), which are memorized and always enriched."""

    transcript: Any
    hard_override: Any
    cli_input: Any
    is_agentic_voice_input: bool
    addressed: str


# Where the loop puts a reply it makes itself (an auto-approved confirmation, a sim-resolved one, a timeout retry):
# processed like typed input, never memorized as something a person said (owner decision 2026-10-25).
LOOP_REPLY_SLOT = "pending_loop_reply"


def take_user_input(observation: Any, state: Any, agent: Any) -> UserInput:
    """The tick's input, from every channel, read ONCE at the top of the pass (§1.2), and the words that address the
    agent published to memory as they arrive (Session R, owner decisions 2026-10-24/25).

    The input comes from the observation, else the loop's own reply (``LOOP_REPLY_SLOT``, which keeps a person's
    pending input for the next tick), else the interactive CLI / robot / API mailbox ``pending_cli_input``, else the
    voice thread's ``pending_voice_input`` (only forwarded once the wake word opened an agentic session); each slot
    read is popped, so an input is processed once. ``addressed`` is what a person said: the pending CLI and voice
    channels always, the observation's ``cli_input`` when its source addresses the agent (``_addresses``), never the
    loop's own reply or a synthetic ``[ACTION_FOLLOWUP``. It is handed to ``agent.publish_arrival`` (memory hears
    every utterance whatever the loop does next; §3 alone used to publish it, which in Talk never ran) and the
    observation's ``cli_input`` copy in ``state.data`` is cleared, so §3 never publishes it again (#1414).
    """
    transcript = hard_override = cli_input = None
    if hasattr(observation, "get"):
        transcript = observation.get("transcript") or observation.get("raw_transcript_text")
        hard_override = observation.get("hard_override")
        cli_input = observation.get("cli_input")
    elif hasattr(observation, "transcript"):
        transcript = getattr(observation, "transcript", None)
        hard_override = getattr(observation, "hard_override", None)
        cli_input = getattr(observation, "cli_input", None)
    addressed = str(cli_input) if cli_input and _addresses(observation) else ""  # the words the loop processes

    loop_reply = state.data.pop("pending_loop_reply", None)  # LOOP_REPLY_SLOT
    if not cli_input and loop_reply:
        cli_input = loop_reply  # the loop answering itself; a person's pending input waits for the next tick
    else:
        pending_cli = state.data.pop("pending_cli_input", None)  # always read once, discarded behind an observation
        if not cli_input and pending_cli:
            cli_input = addressed = pending_cli

    voice_input = state.data.pop("pending_voice_input", None)
    state.data.pop("pending_voice_transcript", None)
    is_agentic_voice_input = False
    if not cli_input and voice_input:  # CLI takes priority
        cli_input = addressed = voice_input
        is_agentic_voice_input = True  # came from an agentic voice session (wake word already said)
        logger.info("Using voice transcript as user input: %s", voice_input[:50])

    text = str(addressed).strip() if addressed else ""
    if text.startswith("[ACTION_FOLLOWUP"):
        text = ""
    if text:
        if str(state.data.get("cli_input") or "").strip() == text:
            state.data.pop("cli_input", None)  # the runtime owns the state: §3 must not publish it again
        # ``maxim.api`` runs duck-typed embedder agents too: one without ``publish_arrival`` (``agents/base.py``'s
        # explicit default does nothing) has no memory pipeline to tell. A capability probe, not a swallow.
        publish = getattr(agent, "publish_arrival", None)
        if callable(publish):
            publish(state, text, source="voice" if is_agentic_voice_input else "cli")
    return UserInput(transcript, hard_override, cli_input or None, is_agentic_voice_input, text)


def percept_text(observation: Any) -> str:
    """The text a tick's observation carries: a transcript first, then the raw transcript, then CLI input; "" when none.
    The one reading (§1.1 imagination and §1.2 enrichment each kept a copy)."""
    if hasattr(observation, "get"):
        return str(
            observation.get("transcript")
            or observation.get("raw_transcript_text")
            or observation.get("cli_input")
            or ""
        )
    if hasattr(observation, "transcript"):
        return str(getattr(observation, "transcript", "") or getattr(observation, "cli_input", "") or "")
    return ""


# The percept sources that ADDRESS the agent: typed or spoken words (a sim's narrator and scenario text arrive as
# "cli", which is right there: the narrator plays the user). An observation without a source is the interactive
# CLI / API / robot channel, which also addresses it. World events (minecraft, proprioception, sound) do not.
ADDRESSED_SOURCES: frozenset[str] = frozenset({"cli", "voice", "speech", "transcript"})


def _addresses(observation: Any) -> bool:
    """Whether the observation's SOURCE addresses the agent (``ADDRESSED_SOURCES``, or no source at all): read from
    the source, never from which field carries the text (Session R, owner decision D5 2026-10-24). Being addressed
    is salient (innate prior): what a person says is always enriched for recall; deliberation keeps its gate."""
    source = observation.get("source") if hasattr(observation, "get") else getattr(observation, "source", None)
    return source is None or source in ADDRESSED_SOURCES


def imagine(*, step_num: int, observation: Any, state: Any, imagination_trigger: Any | None) -> list:
    """§1.1 IMAGINATION: extract novel entities from the percept text. Returns the trigger's results
    (``ImaginationResult``s, ``[]`` when it is absent, finds no text or raises)."""
    # Post-state.update hook: scan percept text for novel entity
    # phrases, check ComponentIndex for existing matches, and if truly
    # novel, dispatch to ImaginationDesigner for real-time SEM entity
    # generation. Gates: DN arousal + energy budget (checked inside trigger).
    _imagination_results: list = []  # ImaginationResult list for enrichment context
    if imagination_trigger is not None:
        try:
            text = percept_text(observation)
            if text:
                scene_id = state.data.get("current_scene_id") if hasattr(state, "data") else None
                scene_ctx = state.data.get("scene_context") if hasattr(state, "data") else None
                _imagination_results = imagination_trigger.process_percept(
                    text, scene_context=scene_ctx, scene_id=scene_id
                )
            else:
                try:
                    from maxim.simulation.sim_logger import sim_log

                    _obs_keys = list(observation.keys()) if hasattr(observation, "keys") else type(observation).__name__
                    sim_log(
                        "SEM_TRACE",
                        f"Imagination skipped: no percept_text (obs keys: {_obs_keys})",
                        _force_debug=True,
                    )
                except Exception:
                    log_swallowed_exception()
        except Exception as e:
            log_swallowed_exception(e, operation="imagination_trigger", context={"step": step_num})
    return _imagination_results


def _checked_call(executor: Any, tool: Any) -> Any:
    """Run an auto-fire tool directly only when the executor's gates would dispatch it (V6, #1315 item 3:
    ``Executor.check``, the dispatch seam without dispatch); otherwise its typed refusal, which is not shown."""
    refusal = executor.check(tool)
    return refusal if refusal is not None else tool.execute()


def auto_sense(*, step_num: int, observation: Any, executor: Any) -> str:
    """§1.15 AUTO-SENSE: passive perception (exteroception + interoception). Returns the auto-sense text
    (``""`` when there is no new percept or nothing to report)."""
    _auto_sense_text = ""  # populated by section 1.15, set on context at submission

    # On each new percept (not empty ticks), auto-run sense_presence
    # (what's around me?) and sense on self-entity (how do I feel?).
    # Results are injected into StructuredContext so the LLM sees them
    # alongside the narrative — the agent passively perceives its
    # surroundings and body state without choosing to call tools.
    # Check if there's a new percept this tick (reuse observation parsing)
    _has_new_percept = False
    if hasattr(observation, "get"):
        _has_new_percept = bool(
            observation.get("transcript") or observation.get("raw_transcript_text") or observation.get("cli_input")
        )
    elif hasattr(observation, "transcript"):
        _has_new_percept = bool(getattr(observation, "transcript", ""))
    if _has_new_percept and executor is not None:
        try:
            _tool_reg = getattr(executor, "_registry", None) or getattr(executor, "registry", None)
            if _tool_reg is not None:
                _auto_sense_parts = []

                # Exteroception: dispatch every auto_fire tool with no
                # arguments. Pre-W1 this hardcoded ``sense_presence``;
                # the declarative ``auto_fire=True`` metadata on
                # ``SensePresenceTool`` (and any future auto-discovery
                # tool) drives this loop now. The
                # ``get_auto_fire_tools()`` helper preserves the
                # bypass invariant: results are injected into the
                # next prompt as passive perception, never logged to
                # ``actions.jsonl``. See
                # [docs/plans/deferred/sense_tool_registry.md] § "Phase 2".
                _presence_tool = None  # canonical sense_presence instance, for entity_map handoff
                _auto_fire_tools = []
                try:
                    _auto_fire_tools = _tool_reg.get_auto_fire_tools()
                except AttributeError:
                    # Older registries without the helper — fall back
                    # to the legacy by-name lookup so out-of-tree
                    # ToolRegistry subclasses keep working.
                    try:
                        _auto_fire_tools = [_tool_reg.get("sense_presence")]
                    except KeyError:
                        _auto_fire_tools = []
                for _af_tool in _auto_fire_tools:
                    if _af_tool is None:
                        continue
                    try:
                        # V6: a direct call outside dispatch runs only what dispatch would (#1315 item 3)
                        _af_result = _checked_call(executor, _af_tool)
                        if _af_result.success and _af_result.output:
                            _auto_sense_parts.append(str(_af_result.output))
                    except Exception as _exc:
                        log_swallowed_exception(
                            _exc,
                            operation=f"auto_fire:{_af_tool.name}",
                            context={"step": step_num},
                        )
                    # Capture an entity-map source so the
                    # interoception block below can reuse the same
                    # entity tree the auto-fire scan saw. Pre-fold,
                    # this keyed on the literal name "sense_presence"
                    # — re-introducing the implicit-by-name coupling
                    # Phase 2 set out to retire. Capturing by
                    # attribute presence (any auto-fire tool that
                    # exposes ``_entity_map``) keeps the loop name-
                    # agnostic: a future auto-discovery tool that
                    # carries an entity map participates without
                    # needing a hardcoded branch here.
                    if _presence_tool is None and hasattr(_af_tool, "_entity_map"):
                        _presence_tool = _af_tool

                # Interoception: sense self-entity (health, stamina,
                # hunger). This is NOT auto_fire today — it's a
                # special dispatch path that needs to be called
                # once per self-entity with the entity name. The
                # underlying ``sense`` tool stays LLM-callable
                # (``kind="core-universal"``, ``auto_fire=False``)
                # so the agent can also invoke it explicitly.
                # Deferring the metadata-fication of this loop to
                # 1.1+ (multi-arg auto_fire is out of MVP scope).
                _sense = None
                try:
                    _sense = _tool_reg.get("sense")
                except KeyError:
                    pass
                if _sense is not None and _presence_tool is not None and executor.check(_sense) is not None:
                    logger.debug("auto-sense interoception skipped: the executor's gates refuse %r", _sense.name)
                elif _sense is not None and _presence_tool is not None:
                    try:
                        _emap = getattr(_presence_tool, "_entity_map", None)
                        if _emap is not None:
                            _self_ents = _emap.list_self_entities()
                            for _se in _self_ents:
                                _sense_result = _sense.execute(entity_name=_se.name)
                                if _sense_result.success and _sense_result.output:
                                    _auto_sense_parts.append(f"Body state ({_se.name}): {_sense_result.output}")
                    except Exception as _exc:
                        log_swallowed_exception(
                            _exc,
                            operation="auto_sense_self",
                            context={"step": step_num},
                        )
                elif _sense is not None and _presence_tool is None and _auto_fire_tools:
                    # Observable signal that interoception was
                    # skipped despite ``sense`` being registered.
                    # Avoids the silent-no-op that the architecture
                    # review flagged if a future auto-fire tool
                    # roster doesn't expose an entity_map.
                    logger.debug(
                        "auto-sense interoception skipped: no auto-fire tool exposes _entity_map "
                        "(sense tool present but cannot be dispatched per-entity)",
                    )

                if _auto_sense_parts:
                    _auto_sense_text = "\n".join(_auto_sense_parts)

                    try:
                        from maxim.simulation.sim_logger import sim_log

                        _n_entities = _auto_sense_text.count("[SCENE]") + _auto_sense_text.count("[YOU]")
                        sim_log("PERCEPTION", f"auto-sense: {_n_entities} entities, body state updated")
                    except Exception:
                        log_swallowed_exception()
        except Exception as _ase:
            log_swallowed_exception(_ase, operation="auto_sense", context={"step": step_num})
    return _auto_sense_text


def orient_to_audio(
    *, step_num: int, sim: Any, aut_mode: str, executor: Any, state: Any, _auto_sense_text: str
) -> tuple[str, bool]:
    """§1.16 AUDIO ORIENTATION: exteroceptive sound direction (thalamic relay). Returns the auto-sense text
    with this pass's orientation line appended (if any), and whether the sound escalates (B1)."""
    _audio_escalate_this_tick = False  # §1.16: a salient audio percept forces a submission (B1)

    # First consumer of the modality-preserving side-channel
    # (``sim.current_percept``): when this tick's percept is an audio/DoA
    # percept, fold a passive azimuth observation into the auto-sense
    # channel. A SALIENT audio percept ESCALATES to a submission — the
    # thalamic gate: a sub-threshold sound is perceived-but-ignored, an
    # above-threshold one reaches the LLM. Escalation sets
    # ``_audio_escalate_this_tick`` so the has_meaningful_input gate in the
    # loop's §6 (agent_loop.py) does NOT discard the audio-only tick (B1 fix — without this the line
    # was folded, logged, and thrown away before the model ever saw it).
    # GATE (re-gated in Stage 3 of live_audio_orient_wiring.md): the real
    # condition was always "a modality-preserving percept is present this
    # tick" — the old ``sim.is_sim_mode`` check was its proxy, and kept the
    # live path dark. Both adapters now carry ``current_percept``
    # (SimulationAdapter from its percept_source; NullSimulationAdapter
    # from a producer's ``carry_percept`` — the Stage-2 DoA feed), so the
    # gate reads the side-channel directly. Production ticks with no
    # carried percept cost one property read (None → skip, N1 preserved).
    # substrate-primary stays excluded: the drive/EC path reads the sensor
    # directly and §1.16 would double-write (S1).
    if getattr(sim, "current_percept", None) is not None and aut_mode != "substrate-primary":
        try:
            from maxim.embodiment.audio_localization import (
                audio_attention_profile,
                format_audio_orientation,
                is_audio_escalation,
                is_orienting_reflex,
                reflex_oriented_azimuth,
                resolve_orienting_profile,
                should_emit_orientation,
                world_set_azimuth,
            )

            _ap = getattr(sim, "current_percept", None)
            _az = None
            if _ap is not None:
                _ameta = getattr(_ap, "metadata", None) or {}
                _az = _ameta.get("azimuth")
            if _az is not None:
                _sal = getattr(_ap, "salience", 0.0)
                _nov = getattr(_ap, "novelty", 0.0)
                _emb = getattr(executor, "embodiment", None)
                # Per-entity reactivity + orient limits (data-driven; default
                # profile when the body declares no `orienting:` config).
                _oprofile = resolve_orienting_profile(_emb)
                # World-set the body's azimuth sensor on ANY audio percept
                # (before the tier gate) so `listen` can read the current
                # sound direction — the agent can attend even to a
                # sub-threshold sound it chose to notice. Capability-gated +
                # fail-soft: bodies without an `azimuth` sensor are
                # unaffected. Sim mirror of live DoA → azimuth (Track 2 L2).
                # Skipped when a LIVE measurement stream owns the sensor
                # (#508 review fold): on live, the DoA feed already wrote a
                # fresher value than this percept echo, and the anonymous
                # write would be refused by world_set_axis's ownership
                # guard anyway — skipping here keeps routine live audio
                # from opening every session with the refusal WARNING.
                if _emb is not None and "azimuth" not in (getattr(_emb, "live_world_set_sensors", None) or ()):
                    world_set_azimuth(_emb, _az)

                _trace = audio_attention_profile(_sal, _nov)
                # Reflex tier is SIM-ONLY (pre-merge review fold): its
                # world_set models a turn the body then "has made" — on
                # the live path no motor was dispatched, so the modeled
                # oriented azimuth would be a fabricated measurement (the
                # head-frame lesson's failure class). Live reflex-speed
                # orienting is Stage 5's DN behavior, with real motion.
                _reflex = sim.is_sim_mode and _emb is not None and is_orienting_reflex(_sal, _nov, _oprofile)
                _escalates = is_audio_escalation(_sal, _oprofile)

                if _reflex:
                    # REFLEX tier: loud AND sudden → AUTOMATIC orient toward
                    # the sound (superior-colliculus startle), bypassing LLM
                    # deliberation. Model the turn by moving the azimuth
                    # toward center, clamped to the body's physical reach
                    # (max_orient_azimuth). The agent becomes aware AFTER — a
                    # delivered post-reflex notice.
                    _oriented = reflex_oriented_azimuth(_az, _oprofile)
                    world_set_azimuth(_emb, _oriented)
                    state.data["_last_audio_orient_az"] = _oriented
                    _reflex_line = (
                        f"A loud, sudden sound made you orient toward it (it was at azimuth {float(_az):+.2f})."
                    )
                    _auto_sense_text = f"{_auto_sense_text}\n{_reflex_line}" if _auto_sense_text else _reflex_line
                    _audio_escalate_this_tick = True
                    _trace["reflex"] = True
                    _trace["escalated"] = True
                    try:
                        from maxim.simulation.sim_logger import sim_log

                        sim_log(
                            "REACTION",
                            f"orienting reflex: turned toward a loud, sudden sound "
                            f"(was {float(_az):+.2f}, now {_oriented:+.2f})",
                            data=_trace,
                        )
                    except Exception:
                        log_swallowed_exception()
                elif should_emit_orientation(state.data.get("_last_audio_orient_az"), _az):
                    # DELIBERATIVE tier: the agent CHOOSES to attend. Change-
                    # gate skips an unchanged direction (prompt noise — the
                    # first live run re-announced the same direction ~every 2s).
                    _audio_line = format_audio_orientation(_ap)
                    if _audio_line:
                        _auto_sense_text = f"{_auto_sense_text}\n{_audio_line}" if _auto_sense_text else _audio_line
                        # Advance the change-gate on DELIVERY — the line
                        # was folded into auto_sense, so an unchanged
                        # direction must not re-announce next tick. Store
                        # the CLAMPED value (N2) so an out-of-range
                        # reading can't spoof the delta gate. (Pre-fold
                        # this advance was nested under _escalates, so
                        # sub-threshold percepts — the DEFAULT 0.5/0.3
                        # weights, i.e. every live DoA percept — re-folded
                        # the identical direction every fresh reading:
                        # exactly the prompt noise this gate exists to
                        # prevent.)
                        state.data["_last_audio_orient_az"] = max(-1.0, min(1.0, float(_az)))
                        if _escalates:
                            _audio_escalate_this_tick = True
                        _trace["reflex"] = False
                        _trace["escalated"] = _escalates
                        try:
                            from maxim.simulation.sim_logger import sim_log

                            sim_log("PERCEPTION", f"audio-orient: {_audio_line}", data=_trace)
                        except Exception:
                            log_swallowed_exception()
        except Exception as _aoe:
            log_swallowed_exception(_aoe, operation="audio_orientation", context={"step": step_num})
    return _auto_sense_text, _audio_escalate_this_tick
