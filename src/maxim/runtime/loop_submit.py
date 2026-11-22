"""The agent loop's §6 submit, first half: the tick's context and whether it is worth a submission (1.3.6 loop
phase 2, slice 7a).

``prepare_submission`` is what one pass of ``agent_loop.run_agentic_loop`` does inside §6's submit ``try`` before the
request assembly, moved verbatim from the loop's body in three pieces so each function stays at 200 lines or fewer
(owner decision 2026-11-21):

- ``build_tick_context``: ``memory.build_context()``, or a minimal context for a pending follow-up or an escalating
  audio-only pass; the percept's bio-enrichment and auto-sense text; Wire-A's cluster-bias annotations and their sources
  and W1's grayscale-tool annotations (one kill switch, ``MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION``).
- ``gate_submission``: is anything worth a submission? A fresh CLI percept or an escalating audio pass; then, given a
  context, ``_scan_context_inputs``: a new CLI input (voice needs a wake word, Selfy's commands are skipped), spoken
  "maxim", a keyword or hard-override percept, a pending action follow-up (folded into the inputs and cleared, with the
  inputs held for it under planning liveness). Nothing meaningful, or sleep without a wake keyword, drops the context;
  the new input is marked processed.

A raise anywhere here still reaches the loop's ``except`` around the whole submit, as before. By rule (a),
``_take_deferred_inputs`` and ``_followup_synthetic_input`` (called only from this code) moved here too.

**What it returns.** ``SubmitGate``: the five values the request assembly reads (the context or None, the new CLI input,
the follow-up's original query, sleep, the inputs held for the follow-up). Nothing here is carried to a later pass: the
gate's loop variable ``cli_input`` used to rebind the body's local, but the body reassigns that every pass before reading
it.

**Arguments.** Explicit keywords, never the loop's ``LoopRun`` (rule (d)), each the loop's own local named as the body
names it (``_loop_nac``, ``_audio_escalate_this_tick``, ...), so the moved text is byte for byte the inline text,
re-indented. The code LOGS on the ``maxim.runtime.agent_loop`` logger (``logger`` below is that same object).

Characterization: ``tests/unit/test_loop_submit_gate_characterization.py`` (written before the move, through the public
``run_agentic_loop``, and kept green by it); the audio and auto-sense parts are also pinned by
``tests/unit/test_loop_perception_characterization.py``.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from maxim.runtime.loop_controller import is_unread_input
from maxim.runtime.loop_state import operational_mode

# The SAME logger object as ``agent_loop.logger`` (logging returns one logger per name), so records keep
# the ``maxim.runtime.agent_loop`` name.
logger = logging.getLogger("maxim.runtime.agent_loop")


@dataclass(frozen=True)
class SubmitGate:
    """What §6's first half hands the request assembly (module docstring).

    Runtime-ephemeral: built and consumed inside one pass of ``run_agentic_loop``, never persisted and never sent
    anywhere, so the CC3 forward-compat rule for frozen dataclasses does not apply.
    """

    context: Any  # the StructuredContext to submit, or None (nothing worth a submission)
    new_cli_input: str | None
    followup_original_query: str
    is_sleeping: bool
    deferred_inputs: list[str]


@dataclass(frozen=True)
class _ContextScan:
    """``_scan_context_inputs``' results, the four gate locals it may rebind. Runtime-ephemeral, as ``SubmitGate``."""

    new_cli_input: str | None
    has_meaningful_input: bool
    followup_original_query: str
    deferred_inputs: list[str]


def _take_deferred_inputs(context: Any, processed_cli_inputs: Any, liveness_on: bool) -> list[str]:
    """The inputs held back while a planning job was in flight, taken alongside the follow-up that is now being
    submitted (#1048): without this the follow-up branch would drop them, or they would preempt the follow-up.
    Marks them processed. Empty for a loop without planning liveness."""
    if not liveness_on or context is None:
        return []
    taken = [c for c in (context.cli_inputs or []) if is_unread_input(c, processed_cli_inputs)]
    for c in taken:
        processed_cli_inputs.append(c)
    if taken:
        logger.debug("Deferred inputs ride the follow-up: %s", [c[:60] for c in taken])
    return taken


def _followup_synthetic_input(followup: Any) -> str:
    """The follow-up input the LLM receives, with the tool result FRAMED as data (#823).

    The single consumer every ``ActionFollowup`` producer converges on (the loop's result paths,
    the batched-exploration path, and ``LoopController``'s human-confirmed path), so no producer
    can bypass the frame. ``batched_exploration`` may contain fetched pages, so it is framed as
    external, and so is a tool that declares ``untrusted_output`` (recalled or stored text). The query
    is quoted so it cannot shift the parser's ``']: `` split.
    """
    from maxim.modes.definitions import NETWORK_TOOLS
    from maxim.utils.content_safety import frame_tool_output

    tool = followup.tool
    external = tool in NETWORK_TOOLS or tool == "batched_exploration" or followup.untrusted_output
    result = frame_tool_output(tool, followup.result or "", external=external)
    query = str(followup.original_query or "").replace("'", "\u2019")
    return f"[ACTION_FOLLOWUP type={followup.followup_type} tool={tool} mode={followup.mode} query='{query}']: {result}"


def prepare_submission(
    *,
    memory: Any,
    ctrl: Any,
    executor: Any,
    state: Any,
    sim: Any,
    cli_input: Any,
    processed_cli_inputs: Any,
    _audio_escalate_this_tick: bool,
    _percept_enrichment_text: str,
    _auto_sense_text: str,
    _loop_nac: Any,
    _loop_agent_id: str,
    _planning_liveness_on: bool,
) -> SubmitGate:
    """§6, before the request assembly: the tick's context, then the gate (module docstring)."""
    context = build_tick_context(
        memory=memory,
        ctrl=ctrl,
        executor=executor,
        state=state,
        _audio_escalate_this_tick=_audio_escalate_this_tick,
        _percept_enrichment_text=_percept_enrichment_text,
        _auto_sense_text=_auto_sense_text,
        _loop_nac=_loop_nac,
        _loop_agent_id=_loop_agent_id,
    )
    return gate_submission(
        context=context,
        cli_input=cli_input,
        state=state,
        ctrl=ctrl,
        sim=sim,
        processed_cli_inputs=processed_cli_inputs,
        _audio_escalate_this_tick=_audio_escalate_this_tick,
        _planning_liveness_on=_planning_liveness_on,
    )


def build_tick_context(
    *,
    memory: Any,
    ctrl: Any,
    executor: Any,
    state: Any,
    _audio_escalate_this_tick: bool,
    _percept_enrichment_text: str,
    _auto_sense_text: str,
    _loop_nac: Any,
    _loop_agent_id: str,
) -> Any:
    """The tick's context with its enrichment and annotations, or None (module docstring)."""
    context = None
    if hasattr(memory, "build_context"):
        context = memory.build_context()

    # CRITICAL: If there's a pending action followup but no context,
    # create a minimal context to ensure the followup gets processed.
    # This fixes the bug where search results were returned but not
    # processed because memory.build_context() returned None.
    if ctrl.pending_action_followup and context is None:
        from maxim.agents.bus import StructuredContext

        context = StructuredContext(
            timestamp=time.time(),
            mode=operational_mode(executor, state),
            autonomy_level=state.data.get("autonomy_level", "supervised"),
            internet_access=state.data.get("internet_access", False),
        )
        logger.info("Created minimal context for pending action followup")

    # §1.16 B1: a salient audio percept escalated this tick but
    # carries no text, so memory may have built no context — mint
    # a minimal one to carry the auto_sense_context (the azimuth
    # line) into the submission.
    if _audio_escalate_this_tick and context is None:
        from maxim.agents.bus import StructuredContext

        context = StructuredContext(
            timestamp=time.time(),
            mode=operational_mode(executor, state),
            autonomy_level=state.data.get("autonomy_level", "supervised"),
            internet_access=state.data.get("internet_access", False),
        )
        logger.debug("Created minimal context for audio-orient escalation")

    # Inject bio-enrichment context if percept passed novelty gate
    if context is not None and _percept_enrichment_text:
        context.bio_enrichment_context = _percept_enrichment_text

    # Auto-sense context (section 1.15) — passive perception
    if context is not None and _auto_sense_text:
        context.auto_sense_context = _auto_sense_text

    # Wire-A (release_0_9_1.md Stage 2): cluster-bias annotation.
    # Surface NAc agent-wide tool-bias to the LLM proposer so it
    # can read substrate-acquired signal across percept regimes
    # the substrate didn't directly drill (Roy-2c finding).
    #
    # The env var MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION=1 disables
    # the read for the Roy-3 ablation; absence → on by default.
    # Truthy-value parsing is shared with the conftest scrub +
    # test suite via ``annotation_disabled_via_env`` so a future
    # divergence here trips the test layer.
    #
    # ValueError narrowing per pre-merge review (architecture
    # lens): a misconfigured per-agent stash (empty agent_id)
    # MUST surface loudly via the WARNING log so the Roy-3
    # measurement arm doesn't silently degrade to
    # annotation-off behavior. Other exceptions propagate —
    # the producer is on the LLM-submission hot path; a real
    # bug here is correctness-critical (Roy-3 evidence
    # integrity), not a recoverable nuisance.
    #
    # NOTE: the deliberation cycle re-submits the same context
    # object without re-running this hook, so biases captured
    # here are the snapshot at first submission. Bias change
    # within a 1-tick deliberation cycle is bounded by
    # ``reward_bias_alpha`` per call (~0.15); the stale-by-one-
    # tick read is acceptable, and avoiding per-cycle re-reads
    # keeps NAc lock contention bounded.
    from maxim.prompts.cluster_bias_annotation import annotation_disabled_via_env

    if (
        context is not None
        and _loop_nac is not None
        and not annotation_disabled_via_env(os.environ.get("MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION"))
    ):
        try:
            context.cluster_bias_annotations = _loop_nac.get_agent_tool_biases(
                agent_id=_loop_agent_id,
                top_n=5,
            )
            # S1 credit-source provenance: same agent-wide
            # aggregation, keyed by RAW tool signature so the
            # composer can join before the prefix strip.
            # Pre-S1 persisted state has no sources — the
            # empty dict renders the pre-S1 format.
            context.cluster_bias_sources = _loop_nac.get_cluster_reward_sources(
                agent_id=_loop_agent_id,
            )
        except ValueError as e:
            logger.warning(
                "Wire-A annotation skipped due to invalid agent_id (%s); Roy-3 measurement integrity may be affected.",
                e,
            )
            context.cluster_bias_annotations = None
            context.cluster_bias_sources = None

    # W1 sense_tool_registry MVP (grayscale visibility).
    # Surfaces SEM-derived tools the substrate has a
    # non-zero reward bias for but that are NOT in the
    # active scene roster — the Roy-3a-retry gap. The
    # full filter (strip ``tool:`` prefix, exclude
    # active, require SEM kind, cap at top-N) lives in
    # ``prompts/grayscale_tools_annotation.py`` so it
    # can be unit-tested against realistic NAc-shaped
    # input — the pre-merge bio-fidelity review caught
    # a silent prefix mismatch the in-line producer
    # would have hidden in production.
    #
    # Reuses ``cluster_bias_annotations`` populated by
    # the Wire-A block above so the producer makes ONE
    # NAc call per submission, not two.
    #
    # Shares Wire-A's env-var kill switch
    # (MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION=1). The two
    # sections are different surfaces of the same
    # substrate signal; disabling one without the other
    # leaks the signal under a different header and
    # pollutes Roy ablation evidence.
    if (
        context is not None
        and _loop_nac is not None
        and executor is not None
        and not annotation_disabled_via_env(os.environ.get("MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION"))
    ):
        try:
            _gs_registry = getattr(executor, "_registry", None) or getattr(executor, "registry", None)
            # Reuse Wire-A's biases when present; the
            # env-var gate above means the only way
            # ``cluster_bias_annotations`` is None here
            # is the Wire-A ValueError branch (invalid
            # agent_id), in which case grayscale would
            # hit the same error — skip rather than
            # double-log.
            _gs_biases = context.cluster_bias_annotations
            if _gs_biases:
                from maxim.prompts.grayscale_tools_annotation import (
                    build_grayscale_annotations,
                )

                _gs_annotations = build_grayscale_annotations(
                    _gs_biases,
                    _gs_registry,
                    top_n=5,
                )
                if _gs_annotations:
                    context.grayscale_tool_annotations = _gs_annotations
        except (AttributeError, ValueError) as e:
            # Narrowed per architecture-lens review:
            # AttributeError catches broken executor /
            # registry shape (out-of-tree subclass
            # without the new helpers); ValueError
            # mirrors Wire-A's discipline. Other
            # exceptions propagate — the producer is on
            # the LLM-submission hot path and a real
            # bug here is correctness-critical, not a
            # recoverable nuisance.
            logger.warning("Grayscale tools annotation skipped: %s", e)
            context.grayscale_tool_annotations = None
    return context


def gate_submission(
    *,
    context: Any,
    cli_input: Any,
    state: Any,
    ctrl: Any,
    sim: Any,
    processed_cli_inputs: Any,
    _audio_escalate_this_tick: bool,
    _planning_liveness_on: bool,
) -> SubmitGate:
    """Whether the tick is worth a submission, and what the request assembly reads (module docstring)."""
    # Check if there's something meaningful to react to
    # Only submit to LLM if we have:
    # 1. New CLI input with "maxim" keyword - not already processed
    # 2. New speech detected (transcription with "maxim" keyword)
    # 3. Direct address via maxim keyword in recent percepts
    # 4. Hard override commands
    # 5. Exploration mode with high novelty detection (periodic checks)
    #
    # SLEEP STATE: Skip LLM processing unless wake keyword detected
    is_sleeping = state.data.get("processing_state", "awake") == "sleep"
    has_meaningful_input = False
    new_cli_input = None
    _deferred_inputs: list[str] = []

    # If a fresh percept arrived THIS iteration (cli_input from
    # observation), it must be processed even if the same text was
    # seen before.  The context.cli_inputs dedup below catches
    # stale entries from the MemoryAgent deque, but a fresh percept
    # is a new turn — e.g., the orchestrator sending the same probe
    # twice is intentional and each must reach the LLM.
    if cli_input and not is_sleeping:
        new_cli_input = str(cli_input)
        has_meaningful_input = True
    # §1.16 B1: a salient audio percept escalated this tick — it
    # must reach the LLM even with no text input, or the folded
    # azimuth line is discarded at the `context = None` gate below.
    if _audio_escalate_this_tick and not is_sleeping:
        has_meaningful_input = True
    # Track original query from followups for conversation history
    # This ensures followup responses are saved with the original user question
    followup_original_query = ""
    if context:
        _scan = _scan_context_inputs(
            context=context,
            state=state,
            ctrl=ctrl,
            processed_cli_inputs=processed_cli_inputs,
            _planning_liveness_on=_planning_liveness_on,
            new_cli_input=new_cli_input,
            has_meaningful_input=has_meaningful_input,
            followup_original_query=followup_original_query,
            _deferred_inputs=_deferred_inputs,
        )
        new_cli_input, has_meaningful_input = _scan.new_cli_input, _scan.has_meaningful_input
        followup_original_query, _deferred_inputs = _scan.followup_original_query, _scan.deferred_inputs

    # Skip LLM if nothing to react to
    if not has_meaningful_input:
        if sim.is_sim_mode and context and context.cli_inputs:
            _unprocessed = [c for c in context.cli_inputs if c not in processed_cli_inputs]
            if _unprocessed:
                sim.log(
                    "PIPELINE",
                    f"LLM skip: has_meaningful=False but {len(_unprocessed)} unprocessed cli_inputs (all already processed)",
                )
        context = None  # Skip this submission

    # Sleep state: Skip LLM unless wake keyword detected
    # This allows background processing while monitoring for activation
    if is_sleeping and context:
        from maxim.modes.definitions import is_wake_keyword

        has_wake_keyword = False
        if context.cli_inputs:
            for cli in context.cli_inputs:
                if cli and is_wake_keyword(cli):
                    has_wake_keyword = True
                    break
        if not has_wake_keyword and context.detected_speech:
            for speech in context.detected_speech:
                if speech and is_wake_keyword(speech):
                    has_wake_keyword = True
                    break
        if not has_wake_keyword:
            logger.debug("Sleep state: skipping LLM (no wake keyword)")
            context = None

    # Mark new CLI input as processed to prevent duplicate submissions
    # deque(maxlen=20) auto-evicts oldest on overflow — no manual check needed
    if new_cli_input:
        processed_cli_inputs.append(new_cli_input)
    return SubmitGate(
        context=context,
        new_cli_input=new_cli_input,
        followup_original_query=followup_original_query,
        is_sleeping=is_sleeping,
        deferred_inputs=_deferred_inputs,
    )


def _scan_context_inputs(
    *,
    context: Any,
    state: Any,
    ctrl: Any,
    processed_cli_inputs: Any,
    _planning_liveness_on: bool,
    new_cli_input: str | None,
    has_meaningful_input: bool,
    followup_original_query: str,
    _deferred_inputs: list[str],
) -> _ContextScan:
    """The gate's ``if context:`` block (module docstring): it reads the context's inputs, speech, percept and the
    pending follow-up, and may rebind the four gate locals it takes and returns."""
    # Known commands that should NOT be sent to LLM
    # These are handled by Selfy's phrase response system
    SKIP_LLM_COMMANDS = frozenset(
        {
            # System commands
            "maxim shutdown",
            "shutdown maxim",
            "maxim stop",
            "stop maxim",
            "maxim pause",
            "pause maxim",
            "maxim resume",
            "resume maxim",
            # Sleep/wake (processing state)
            "maxim sleep",
            "sleep maxim",
            "maxim nap",
            "maxim rest",
            "maxim wake",
            "wake maxim",
            "maxim wake up",
            "wake up maxim",
            # Strategy switching
            "maxim observe",
            "observe maxim",
            "maxim watch",
            "maxim explore",
            "explore maxim",
            "maxim research",
            "research maxim",
            "maxim assist",
            "maxim help",
            "maxim reflect",
            "maxim reflection",
            "maxim learn",
            "maxim train",
            # Mode switching
            "maxim passive",
            "maxim active",
            "maxim singularity",
        }
    )

    # Check for NEW CLI input (not already processed)
    # CLI input always goes to LLM (maxim prefix presumed)
    # Voice input still requires "maxim" wake word
    input_source = state.data.get("pending_user_input_source", "CLI")
    if context.cli_inputs:
        for cli_input in context.cli_inputs:
            if cli_input and cli_input not in processed_cli_inputs:
                cli_lower = cli_input.lower().strip()
                # CLI source: always process (prefix presumed)
                # Voice source: require "maxim" or "reachy" wake word
                should_process = (input_source == "CLI") or ("maxim" in cli_lower) or ("reachy" in cli_lower)
                if should_process:
                    # Skip LLM for known commands (handled by Selfy)
                    if cli_lower in SKIP_LLM_COMMANDS:
                        logger.info("Skipping LLM for command: %s", cli_input)
                        processed_cli_inputs.append(cli_input)
                        continue
                    new_cli_input = cli_input
                    has_meaningful_input = True
                    break
                else:
                    # Mark voice inputs without wake word as processed
                    processed_cli_inputs.append(cli_input)

    # Check for speech with maxim keyword
    if context.detected_speech:
        for speech in context.detected_speech:
            if speech and "maxim" in speech.lower():
                has_meaningful_input = True
                break

    # Check for maxim keyword in current percept
    if context.current_percept and context.current_percept.has_maxim_keyword:
        has_meaningful_input = True

    # Check for hard override commands
    if context.current_percept and context.current_percept.hard_override:
        has_meaningful_input = True

    # Check for pending action followup (tools that need LLM processing)
    if ctrl.pending_action_followup:
        has_meaningful_input = True
        # Inject the action result into CLI inputs so LLM can process
        followup_query = ctrl.pending_action_followup.original_query
        followup_result = ctrl.pending_action_followup.result or ""
        followup_tool = ctrl.pending_action_followup.tool
        followup_type = ctrl.pending_action_followup.followup_type
        followup_mode = ctrl.pending_action_followup.mode  # noqa: F841 -- moved verbatim; read by nothing (the loop body's locals() hid it from ruff)

        # Preserve original query for conversation history tracking
        # This ensures followup responses are saved with the original user question
        followup_original_query = followup_query

        # Create a synthetic input with followup metadata
        # Format: [ACTION_FOLLOWUP type=X tool=Y query='Z']: result
        synthetic_input = _followup_synthetic_input(ctrl.pending_action_followup)
        if context.cli_inputs:
            context.cli_inputs.append(synthetic_input)
        else:
            context.cli_inputs = [synthetic_input]
        _deferred_inputs = _take_deferred_inputs(context, processed_cli_inputs, _planning_liveness_on)
        # A fold submits the follow-up as itself (the trim keeps it); its original query stays
        # the request's triggering input. The held inputs are already marked processed.
        new_cli_input = None if _deferred_inputs else new_cli_input
        logger.info(
            "Injected action followup into context: type=%s, tool=%s, result_len=%d",
            followup_type,
            followup_tool,
            len(followup_result),
        )
        # Clear the followup after processing
        ctrl.pending_action_followup = None
    return _ContextScan(
        new_cli_input=new_cli_input,
        has_meaningful_input=has_meaningful_input,
        followup_original_query=followup_original_query,
        deferred_inputs=_deferred_inputs,
    )
