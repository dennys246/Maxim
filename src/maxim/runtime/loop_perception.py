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
from maxim.utils.input_source import InputSource

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
    own_capture_id: str = ""  # the record ``addressed`` became at arrival, "" if none: recall excludes it (#1573)
    source: InputSource = InputSource.UNKNOWN  # who it came from: the approver when it answers a prompt (#1185)
    answers: str = ""  # the prompt the loop's own reply answers (LOOP_REPLY_FOR_SLOT), "" for any other input (#1185)


# Where the loop puts a reply it makes itself (an auto-approved confirmation, a sim-resolved one, a timeout retry):
# processed like typed input, never memorized as something a person said (owner decision 2026-10-25). Its writer
# also sets ``LOOP_REPLY_SOURCE_SLOT`` (an ``InputSource`` value; ``loop`` when absent): a sim policy's answer is the
# sim's, an auto-yes is the loop's own (#1185).
LOOP_REPLY_SLOT = "pending_loop_reply"
LOOP_REPLY_SOURCE_SLOT = "pending_loop_reply_source"
# The prompt the loop's reply answers ("confirmation" / "timeout_retry"), set with it by every writer and taken with it,
# so the reply is routed to its own prompt whoever made it, the sim's policy or the loop (#1185 delta 7).
LOOP_REPLY_FOR_SLOT = "pending_loop_reply_for"
# Where a tick's other inputs wait, oldest first, behind the one it takes (#1185): the loop's own reply answers a
# pending gating prompt first, and an observation, a typed line and a voice turn that arrive together are taken
# one per tick. Each entry keeps its words, ``source``, ``addressed`` and ``own_capture_id``: it was memorized on
# arrival, so it is processed later with its recall cue, never memorized again. (Read by its literal key: the
# mode-read lint, rule A4, refuses a non-constant key on ``state.data``.)
DEFERRED_INPUT_SLOT = "pending_deferred_input"


def is_tool_followup(text: Any) -> bool:
    """A tool's result fed back to the model (``agent_loop._format_followup``), never a person's words or an answer
    to a prompt: the one test for an ``[ACTION_FOLLOWUP`` line (#1185)."""
    return str(text or "").startswith("[ACTION_FOLLOWUP")


def take_user_input(observation: Any, state: Any, agent: Any, *, sim_mode: bool) -> UserInput:
    """The tick's input, from every channel, read ONCE at the top of the pass (§1.2), and the words that address the
    agent published to memory as they arrive (Session R, owner decisions 2026-10-24/25).

    The channels: the observation's ``cli_input`` (a sim's narrator when ``sim_mode``, else a person), the
    interactive CLI / robot / API mailbox ``pending_cli_input`` (a person; read once, and discarded behind an
    observation as before), and the voice thread's ``pending_voice_input`` (only forwarded once the wake word opened
    an agentic session). Each slot read is popped. ``addressed`` is what a person said: the pending CLI and voice
    channels always, the observation's ``cli_input`` when its source addresses the agent (``_addresses``), never the
    loop's own reply or a tool's ``[ACTION_FOLLOWUP`` result. Every arrival is handed to ``agent.publish_arrival`` as
    it arrives (memory hears every utterance whatever the loop does next; §3 alone used to publish it, which in Talk
    never ran) and the observation's ``cli_input`` copy in ``state.data`` is cleared, so §3 never publishes it again
    (#1414).

    One input is taken per tick (#1185): the loop's own reply first (``LOOP_REPLY_SLOT``, the answer to a pending
    prompt; a person's mailbox input then waits in its slot), else the oldest waiting entry in
    ``DEFERRED_INPUT_SLOT``, then this tick's arrivals in channel order. Every arrival not taken waits there (FIFO),
    a voice turn included, so a steady stream cannot overtake older words. One input is still dropped, unchanged:
    a person's ``pending_cli_input`` that arrives in the same tick as an observation's ``cli_input`` is discarded.
    """
    transcript, hard_override, obs_cli = _observation_fields(observation)
    loop_reply = state.data.pop("pending_loop_reply", None)  # LOOP_REPLY_SLOT
    loop_source = state.data.pop("pending_loop_reply_source", None) or InputSource.LOOP.value  # LOOP_REPLY_SOURCE_SLOT
    loop_answers = str(state.data.pop("pending_loop_reply_for", None) or "")  # LOOP_REPLY_FOR_SLOT

    arrivals: list[dict[str, Any]] = []
    if obs_cli:
        obs_source = (
            InputSource.SELF if is_tool_followup(obs_cli) else (InputSource.SIM if sim_mode else InputSource.OPERATOR)
        )
        arrivals.append(_entry(obs_cli, obs_source, str(obs_cli) if _addresses(observation) else ""))
    if not loop_reply:  # behind the loop's reply, a person's mailbox input waits in its own slot for the next tick
        pending_cli = state.data.pop("pending_cli_input", None)  # always read once, discarded behind an observation
        if pending_cli and not obs_cli:
            arrivals.append(_entry(pending_cli, InputSource.OPERATOR, pending_cli))
    voice_input = state.data.pop("pending_voice_input", None)
    state.data.pop("pending_voice_transcript", None)
    if voice_input:  # never dropped behind another input (#1185): it waits its turn
        arrivals.append(_entry(voice_input, InputSource.OPERATOR, voice_input, voice=True))
    for entry in arrivals:
        _publish_arrival(entry, state, agent)

    waiting = list(state.data.pop("pending_deferred_input", None) or []) + arrivals  # DEFERRED_INPUT_SLOT, oldest first
    if loop_reply:
        taken: dict[str, Any] | None = _entry(loop_reply, InputSource(loop_source), "")
    else:
        taken = waiting.pop(0) if waiting else None
    if waiting:
        state.data["pending_deferred_input"] = waiting  # DEFERRED_INPUT_SLOT
    if taken is None:
        return UserInput(transcript, hard_override, None, False, "", "", InputSource.UNKNOWN)
    if taken["voice"]:
        logger.info("Using voice transcript as user input: %s", str(taken["text"])[:50])
    return UserInput(
        transcript,
        hard_override,
        taken["text"],
        taken["voice"],
        taken["addressed"],
        taken["own_capture_id"],
        InputSource(taken["source"]),
        loop_answers if loop_reply else "",
    )


def _observation_fields(observation: Any) -> tuple[Any, Any, Any]:
    """The observation's transcript, hard override and ``cli_input`` (a dict or a percept-like object)."""
    if hasattr(observation, "get"):
        transcript = observation.get("transcript") or observation.get("raw_transcript_text")
        return transcript, observation.get("hard_override"), observation.get("cli_input")
    if hasattr(observation, "transcript"):
        return (
            getattr(observation, "transcript", None),
            getattr(observation, "hard_override", None),
            getattr(observation, "cli_input", None),
        )
    return None, None, None


def _entry(text: Any, source: InputSource, addressed: Any, *, voice: bool = False) -> dict[str, Any]:
    """One input as it is taken or waits: its words, source, the words that address the agent and the record they
    became (``own_capture_id``, set on publish). Plain data, so the waiting slot is JSON."""
    return {"text": text, "source": source.value, "addressed": addressed or "", "own_capture_id": "", "voice": voice}


def _publish_arrival(entry: dict[str, Any], state: Any, agent: Any) -> None:
    """Memorize an arrival's addressing words once, as it arrives (Session R), and keep the record it became."""
    text = str(entry["addressed"]).strip() if entry["addressed"] else ""
    if is_tool_followup(text):
        text = ""
    entry["addressed"] = text
    if not text:
        return
    if str(state.data.get("cli_input") or "").strip() == text:
        state.data.pop("cli_input", None)  # the runtime owns the state: §3 must not publish it again
    # ``maxim.api`` runs duck-typed embedder agents too: one without ``publish_arrival`` (``agents/base.py``'s
    # explicit default does nothing) has no memory pipeline to tell. A capability probe, not a swallow.
    publish = getattr(agent, "publish_arrival", None)
    if callable(publish):
        published = publish(state, text, source="voice" if entry["voice"] else "cli")
        entry["own_capture_id"] = published if isinstance(published, str) else ""


def context_addresses_the_llm(context: Any, state: Any) -> bool:
    """Whether the tick's context gives the LLM something to answer, beyond new CLI input and a follow-up (moved out
    of ``_run_agentic_loop_body``'s §6 meaningful-input checks, rule T for #1185).

    - Check for speech with maxim keyword: any ``detected_speech`` naming "maxim".
    - Check for maxim keyword in current percept (``has_maxim_keyword``).
    - Check for hard override commands (``hard_override``).
    - A change the person asked for at a confirmation or plan prompt (``state.data["pending_modification"]``,
      #1185): it was consumed as the answer, not sent as a turn, so it must trigger the submit that reads it.
    """
    for speech in context.detected_speech or ():
        if speech and "maxim" in speech.lower():
            return True
    percept = context.current_percept
    if percept and (percept.has_maxim_keyword or percept.hard_override):
        return True
    return bool(state.data.get("pending_modification"))


def percept_enrichment_context(state: Any, imagination_results: Any, user_input: UserInput) -> Any:
    """§1.2's ``EnrichmentContext``: the goal, the entity refs ImaginationTrigger resolved this tick, the turn's words
    and the record they became."""
    from maxim.integration.bio_enrichment import EnrichmentContext

    return EnrichmentContext(
        active_goal=state.data.get("active_goal") if hasattr(state, "data") else None,
        resolved_entities=tuple(r.ref for r in imagination_results if r.ref) if imagination_results else (),
        user_input=user_input.addressed,
        own_capture_id=user_input.own_capture_id,
    )


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
