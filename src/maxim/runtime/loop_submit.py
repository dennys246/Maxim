"""The agent loop's §6 submit: the tick's context, whether it is worth a submission, the request, and the PFC
deliberation after it (1.3.6 loop phase 2, slices 7a and 7b).

``submit_tick`` is §6's whole submit ``try`` as one pass of ``agent_loop.run_agentic_loop`` runs it, moved verbatim
from the loop's body in pieces so each function stays at 200 lines or fewer (owner decision 2026-11-21). The body
calls it once per due submit and keeps the guard in front of it (substrate-primary, a worker, no pending proposal,
nothing held and the run not stopped, the submit interval) and the tool snapshot ``_all_tools``.

First half (slice 7a), ``prepare_submission``:

- ``build_tick_context``: ``memory.build_context()``, or a minimal context for a pending follow-up or an escalating
  audio-only pass; the percept's bio-enrichment and auto-sense text; Wire-A's cluster-bias annotations and their sources
  and W1's grayscale-tool annotations (one kill switch, ``MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION``).
- ``gate_submission``: is anything worth a submission? A fresh CLI percept or an escalating audio pass; then, given a
  context, ``_scan_context_inputs``: a new CLI input (voice needs a wake word, Selfy's commands are skipped), spoken
  "maxim", a keyword or hard-override percept, a pending action follow-up (folded into the inputs and cleared, with the
  inputs held for it under planning liveness). Nothing meaningful, or sleep without a wake keyword, drops the context;
  the new input is marked processed.

Second half (slice 7b), given a context:

- ``_tick_mode_info``: the operational mode's ``ModeInfo`` (unknown = passive, #829) with the response-token override.
- In ``submit_tick``: the internet status, the new-inputs filter, the surfaced list, the context pool and conversation
  text, the pending plan modification, and the speculative prefetch (consumed here).
- ``_surface_tools``: the roster shown to the model: the mode's tools, the wired body's SEM tools, Wire 3's filter (and
  its ``WIRE_3_FILTER`` sim event), then the permits/forbid filter ``_advertisable_tools`` (#1412).
- ``_describe_surfaced_tools``: the descriptions, Wire 3's felt phrase on each degraded tool, Wire 1's variance.
- ``_submit_request``: the protocol context, the ``inference_start`` event, the shared kwargs, ``submit_tick_context``
  (with the run's inbound comms, #1481), the submit stamp, and the submit's logs.
- ``_deliberate``: after a submit whose section-1.2 PFC gate passed, cycles 2+ through the body's
  ``_run_deliberation_cycles``, or a cycle-1 ready answer used directly.

A raise anywhere in the ``try`` is logged (WARNING plus a DEBUG traceback) and the pass goes on, as inline. By rule (a)
the helpers called only from this code moved here too: ``_take_deferred_inputs``, ``_followup_synthetic_input``,
``_triggering_input``, ``_describe_tools_for_prompt`` with ``_thaw``, and ``_WIRE3_PHRASE_RE``. One helper moved here
that the body still calls: ``_advertisable_tools`` (§4's loop-cap message), which the body reaches through the module
reference ``agent_loop._lsub``, so there is ONE binding for a patch to hit (no other src module imports it).

**Why ``_run_deliberation_cycles`` stays in ``agent_loop``.** It is pinned over 200 lines and listed in the 1.4.0
release gate's ``debt_snapshot`` (``scripts/release_gates.json``); moved verbatim, it would need an
``accepted_new_debt`` record, which is the owner's call (a later slice that also splits it to 200 lines or fewer leaves
no baseline entry and needs none). Rule (c) forbids importing ``agent_loop`` here, so the body passes it in as the
keyword ``_run_deliberation_cycles``, with ``_wait_for_proposal`` (which stays beside it: cycles 2+ inside it and cycle
1 in ``_deliberate`` read the same one function) and its ``LoopRun`` novelty check as ``_is_novel_thought``, under the
names the moved text already calls.

**What it returns.** ``SubmitCarry``: the two body locals §6 rebinds that a LATER pass reads, ``last_surfaced_tools``
(§2's ``record_surfaced_but_unused``) and ``pending_prefetch`` (None once consumed; the body reassigns it only on a pass
with new input). They are returned even after a raise, carrying whatever the ``try`` had rebound by then, exactly as the
inline locals did. That both live as body locals apart from ``ctrl``'s copies is pinned as-is (#1744).
``prepare_submission``'s ``SubmitGate`` carries the five values the request assembly reads (the context or None, the
new CLI input, the follow-up's original query, sleep, the inputs held for the follow-up).

**Arguments.** Explicit keywords, never the loop's ``LoopRun`` (rule (d)), each the loop's own local named as the body
names it (``_loop_nac``, ``_audio_escalate_this_tick``, ...), so the moved text is byte for byte the inline text,
re-indented. One exception: cycle 1's salience was read as ``locals().get("_salience_c1")`` (set only when section
1.2's cycle-1 enrichment ran, so stale or absent otherwise); the body still reads it that way and passes the value as
``_salience_c1``. The code LOGS on the ``maxim.runtime.agent_loop`` logger (``logger`` below is that same object).

Characterization: ``tests/unit/test_loop_submit_gate_characterization.py`` (7a) and
``tests/unit/test_loop_submit_request_characterization.py`` (7b), each written before its move, through the public
``run_agentic_loop``, and kept green by it; the audio and auto-sense parts are also pinned by
``tests/unit/test_loop_perception_characterization.py``.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from maxim.agents.bus import StreamEvent
from maxim.runtime.loop_controller import is_unread_input, take_replan_inputs
from maxim.runtime.loop_external import planning_submit_shared_kwargs, submit_tick_context
from maxim.runtime.loop_planning import annotate_tool_variance
from maxim.runtime.loop_state import operational_mode
from maxim.utils.logging import log_swallowed_exception
from maxim.utils.structured_logging import log_agentic

if TYPE_CHECKING:
    from maxim.agents.llm_worker import ModeInfo

# The SAME logger object as ``agent_loop.logger`` (logging returns one logger per name), so records keep
# the ``maxim.runtime.agent_loop`` name.
logger = logging.getLogger("maxim.runtime.agent_loop")

# Wire 3 (release_0_9_1.md Stage 1): regex matching the felt-sensation
# annotations Embodiment.integrity_to_felt_phrase produces. The §6 Wire 3
# hook (``_describe_surfaced_tools``) uses this to strip a stale annotation before re-applying the
# current-tick one — guards against multi-tick integrity drift accumulating
# multiple suffixes (e.g., integrity 0.55 → ``(feels strained)``, then 0.40
# → ``(feels weakened, prone to failing)``; without the strip, both would
# coexist in the description). The phrases are pinned in tests so a future
# additional band must update both this regex AND
# ``Embodiment.integrity_to_felt_phrase`` together.
_WIRE3_PHRASE_RE = re.compile(r" \((?:feels strained|feels weakened, prone to failing)\)$")


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


@dataclass(frozen=True)
class SubmitCarry:
    """What one §6 submit hands back to the loop's body: the two body locals §6 rebinds that a LATER pass reads
    (module docstring, "What it returns"). Runtime-ephemeral, as ``SubmitGate``."""

    last_surfaced_tools: list[str]  # §2's ``record_surfaced_but_unused`` reads it on the next pass (#1744)
    pending_prefetch: Any  # None once a submit consumed it; the body reassigns it only on a pass with new input


def submit_tick(
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
    _all_tools: set[str],
    _max_response_tokens_override: int | None,
    autonomy_controller: Any,
    _tool_overlay: Any,
    context_pool: Any,
    prefetcher: Any,
    llm_worker: Any,
    agent_states: Any,
    recent_outcomes: Any,
    use_tool_prompting: Any,
    protocol_registry: Any,
    on_event: Any,
    bio_enrichment_pipeline: Any,
    inbound_inbox: Any,
    now: float,
    _pfc_gate_passed: bool,
    stop_event: Any,
    _max_cycles_override: Any,
    percept_source: Any,
    _wms: Any,
    thought_gate: Any,
    step_num: Any,
    _is_novel_thought: Callable[[str], bool],
    _salience_c1: float | None,
    _run_deliberation_cycles: Callable[..., Any],
    _wait_for_proposal: Callable[..., Any],
    last_surfaced_tools: list[str],
    pending_prefetch: Any,
) -> SubmitCarry:
    """§6, the whole submit (module docstring): the gate, then the request and the deliberation; a raise anywhere
    is logged and the pass goes on, as inline."""
    try:
        _submit = prepare_submission(  # §6's first half (slice 7a)
            memory=memory,
            ctrl=ctrl,
            executor=executor,
            state=state,
            sim=sim,
            cli_input=cli_input,
            processed_cli_inputs=processed_cli_inputs,
            _audio_escalate_this_tick=_audio_escalate_this_tick,
            _percept_enrichment_text=_percept_enrichment_text,
            _auto_sense_text=_auto_sense_text,
            _loop_nac=_loop_nac,
            _loop_agent_id=_loop_agent_id,
            _planning_liveness_on=_planning_liveness_on,
        )
        context, new_cli_input, is_sleeping = _submit.context, _submit.new_cli_input, _submit.is_sleeping
        followup_original_query, _deferred_inputs = _submit.followup_original_query, _submit.deferred_inputs

        if context:
            mode_info = _tick_mode_info(
                executor=executor,
                state=state,
                _all_tools=_all_tools,
                _max_response_tokens_override=_max_response_tokens_override,
            )

            # Get internet access status
            internet_access = state.data.get("internet_access", False)
            internet_policy_summary = state.data.get("internet_policy_summary", "")

            # Filter CLI inputs to only include new ones (not already processed)
            # This prevents the LLM from seeing the same input multiple times
            if context.cli_inputs:
                new_inputs = [
                    inp for inp in context.cli_inputs if inp not in processed_cli_inputs or inp == new_cli_input
                ]
                # Only keep the most recent new input
                if new_cli_input:
                    context.cli_inputs = [new_cli_input]
                else:
                    context.cli_inputs = new_inputs[-1:] if new_inputs else []

            available_tools, _wire3_embodiment, _wire3_degraded = _surface_tools(
                mode_info=mode_info,
                _all_tools=_all_tools,
                executor=executor,
                autonomy_controller=autonomy_controller,
            )
            last_surfaced_tools = list(available_tools)

            tool_descriptions = _describe_surfaced_tools(
                available_tools=available_tools,
                executor=executor,
                _tool_overlay=_tool_overlay,
                _wire3_embodiment=_wire3_embodiment,
                _wire3_degraded=_wire3_degraded,
                _loop_nac=_loop_nac,
                _loop_agent_id=_loop_agent_id,
            )

            # Get context pool text
            context_pool_text = context_pool.get_context_text(max_tokens=mode_info.context_window_tokens // 2)

            # Get conversation history for context
            conversation_history_text = context_pool.get_conversation_text(max_turns=5)

            # Check for pending modification request
            pending_modification = state.data.pop("pending_modification", None)

            # Format prefetch context if available
            prefetch_context_text = ""
            skip_exploration = False
            if pending_prefetch is not None:
                prefetch_context_text = prefetcher.format_prefetch_context(pending_prefetch)
                skip_exploration = pending_prefetch.skip_exploration
                # Clear after use
                pending_prefetch = None

            submitted, _shared_submit = _submit_request(
                llm_worker=llm_worker,
                ctrl=ctrl,
                sim=sim,
                memory=memory,
                executor=executor,
                context=context,
                mode_info=mode_info,
                internet_access=internet_access,
                internet_policy_summary=internet_policy_summary,
                available_tools=available_tools,
                tool_descriptions=tool_descriptions,
                context_pool_text=context_pool_text,
                conversation_history_text=conversation_history_text,
                agent_states=agent_states,
                recent_outcomes=recent_outcomes,
                use_tool_prompting=use_tool_prompting,
                is_sleeping=is_sleeping,
                protocol_registry=protocol_registry,
                on_event=on_event,
                autonomy_controller=autonomy_controller,
                new_cli_input=new_cli_input,
                followup_original_query=followup_original_query,
                pending_modification=pending_modification,
                prefetch_context_text=prefetch_context_text,
                skip_exploration=skip_exploration,
                bio_enrichment_pipeline=bio_enrichment_pipeline,
                processed_cli_inputs=processed_cli_inputs,
                _deferred_inputs=_deferred_inputs,
                _planning_liveness_on=_planning_liveness_on,
                inbound_inbox=inbound_inbox,
                now=now,
            )

            _deliberate(
                submitted=submitted,
                _pfc_gate_passed=_pfc_gate_passed,
                bio_enrichment_pipeline=bio_enrichment_pipeline,
                llm_worker=llm_worker,
                stop_event=stop_event,
                ctrl=ctrl,
                _max_cycles_override=_max_cycles_override,
                percept_source=percept_source,
                _shared_submit=_shared_submit,
                autonomy_controller=autonomy_controller,
                inbound_inbox=inbound_inbox,
                state=state,
                _wms=_wms,
                context=context,
                thought_gate=thought_gate,
                step_num=step_num,
                sim=sim,
                _is_novel_thought=_is_novel_thought,
                _salience_c1=_salience_c1,
                _run_deliberation_cycles=_run_deliberation_cycles,
                _wait_for_proposal=_wait_for_proposal,
            )

    except Exception as e:
        import traceback

        logger.warning(f"Failed to submit context to LLM: {type(e).__name__}: {e}")
        logger.debug(f"Full traceback:\n{traceback.format_exc()}")
    return SubmitCarry(last_surfaced_tools=last_surfaced_tools, pending_prefetch=pending_prefetch)


def _tick_mode_info(
    *,
    executor: Any,
    state: Any,
    _all_tools: set[str],
    _max_response_tokens_override: int | None,
) -> ModeInfo:
    """The pass's ``ModeInfo``: the operational mode's definition, its tools, the response-token override."""
    from maxim.agents.llm_worker import ModeInfo
    from maxim.modes.definitions import get_mode

    # Get mode info -- the operator's grant wins, as at dispatch (#829; one reader, #963)
    mode_name = operational_mode(executor, state)

    # Get mode definition for tool access (unknown = passive, as dispatch enforces, #829)
    mode_def = get_mode(mode_name) or get_mode("passive")
    available_tools_for_mode = set()
    if mode_def and _all_tools:
        available_tools_for_mode = mode_def.get_available_tools(_all_tools)

    # Both the budgeter's response reserve and the request's max_tokens
    # read this one field; unset = ModeInfo's own default.
    _mode_info_overrides: dict[str, Any] = (
        {"max_response_tokens": _max_response_tokens_override} if _max_response_tokens_override is not None else {}
    )
    mode_info = ModeInfo(
        name=mode_name,
        goal=mode_def.goal if mode_def else "Respond to user requests",
        context_prompt=mode_def.context_prompt if mode_def else "",
        allowed_tools=available_tools_for_mode,
        forbidden_tools=mode_def.forbidden_tools if mode_def else set(),
        can_access_filesystem=mode_def.can_access_filesystem if mode_def else True,
        can_access_network=mode_def.can_access_network if mode_def else True,
        uses_tool_relevance_filter=(mode_def.uses_tool_relevance_filter if mode_def else False),
        **_mode_info_overrides,
    )
    return mode_info


def _surface_tools(
    *,
    mode_info: Any,
    _all_tools: set[str],
    executor: Any,
    autonomy_controller: Any,
) -> tuple[list[str], Any, dict[str, float]]:
    """The roster the model is shown: the mode's tools, the wired body's SEM tools, Wire 3's filter, the permits
    filter. Returns it with Wire 3's embodiment and degraded integrities, which the descriptions annotate."""
    # Get available tools for this mode (a set; the filters below rebind it to a list)
    available_tools: set[str] | list[str]
    available_tools = mode_info.get_available_tools(_all_tools)

    # Body ownership is not a mode privilege
    # (sem_motor_binding.md Phase 1): SEM affordance
    # tools generated from a wired body join the
    # described tool list past the mode filter —
    # otherwise a live mode's non-empty allowed_tools
    # hides reachy_mini_turn_* and they surface only
    # as bare tokens via the relevance filter (the
    # 'move' bare-token lesson).
    if executor is not None and getattr(executor, "embodiment", None) is not None:
        try:
            from maxim.embodiment.tool_bridge import always_active_sem_tools

            _exec_registry = executor.registry
            for _sem_tool in always_active_sem_tools(_exec_registry):
                if _exec_registry.is_tool_active(_sem_tool.name):
                    available_tools.add(_sem_tool.name)
        except Exception:
            # Silent loss here re-creates the bare-token
            # bug this union exists to fix — log it.
            logger.debug("SEM prompt union failed", exc_info=True)

    # Wire 3 (release_0_9_1.md Stage 1): filter tools
    # routed through critically-damaged components and
    # collect degraded-affordance annotations. Pulls
    # from Embodiment.get_disabled_affordances() /
    # .get_degraded_affordances() which read each
    # modulator's compute_integrity(). Default
    # thresholds: integrity < 0.3 → disabled
    # (filtered out); 0.3 <= integrity < 0.6 →
    # annotated with a felt-sensation phrase
    # ("feels strained" / "feels weakened, prone to
    # failing") so the LLM proposer reads the
    # affordance's cost in proprioceptive voice
    # rather than as a system advisor (bio-fidelity
    # fold). Fail-open at the narrowed exception
    # surface (no embodiment, missing methods, broken
    # modulator shape) — the filter is a no-op but
    # the WARNING surfaces operator-visible signal.
    #
    # NOTE on `last_surfaced_tools`: post-filter is
    # intentional. The learned tool-relevance index
    # (§2, ``loop_intake.take_llm_proposal``, on the
    # next pass) calls `record_surfaced_but_unused`
    # — disabled tools weren't surfaced, so they
    # don't decay. Future: if a disabled tool
    # recovers, the relevance index resumes decay
    # the next tick the tool surfaces again.
    _wire3_embodiment = getattr(executor, "embodiment", None)
    _wire3_degraded: dict[str, float] = {}
    _wire3_disabled: set[str] = set()
    if _wire3_embodiment is not None:
        try:
            _wire3_disabled = _wire3_embodiment.get_disabled_affordances()
            if _wire3_disabled:
                available_tools = [t for t in available_tools if t not in _wire3_disabled]
            _wire3_degraded = _wire3_embodiment.get_degraded_affordances()
        except (AttributeError, TypeError) as e:
            # Narrowed from broad Exception per
            # arch-lens A4: the inner body.py guard
            # already swallows compute_integrity
            # raises with a WARNING. Outer surface
            # failures here are method-shape
            # mismatches (non-Embodiment object
            # plugged into executor.embodiment) —
            # WARN so operator review catches it.
            logger.warning(
                "Wire 3: get_disabled/degraded_affordances shape mismatch — filter no-op: %s",
                e,
            )
            _wire3_degraded = {}
            _wire3_disabled = set()
        # Emit Roy-3 disambiguator (bio-fidelity B2):
        # without this, "Wire 3 hid the tool" and
        # "substrate learned avoidance" are
        # indistinguishable post-hoc. The event
        # lists which affordances were filtered /
        # annotated each LLM submission so Roy-3
        # can quantify Wire 3's effect surface.
        if _wire3_disabled or _wire3_degraded:
            try:
                from maxim.simulation import sim_logger as _sl_w3

                _w3_tick = int(time.time() - _sl_w3._sim_start) if _sl_w3._sim_start > 0.0 else 0
                _sl_w3.sim_log(
                    "WIRE_3_FILTER",
                    f"wire_3: disabled={len(_wire3_disabled)} degraded={len(_wire3_degraded)}",
                    {
                        "tick": _w3_tick,
                        "disabled_tools": sorted(_wire3_disabled),
                        # Pass integrity floats only here — the LLM
                        # sees the felt phrases above.
                        "degraded_integrities": {
                            name: round(integrity, 4) for name, integrity in _wire3_degraded.items()
                        },
                    },
                )
            except ImportError:
                # Non-sim runtime — observability
                # is optional, never load-bearing.
                pass
    available_tools = _advertisable_tools(available_tools, executor, autonomy_controller)
    return available_tools, _wire3_embodiment, _wire3_degraded


def _describe_surfaced_tools(
    *,
    available_tools: list[str],
    executor: Any,
    _tool_overlay: Any,
    _wire3_embodiment: Any,
    _wire3_degraded: dict[str, float],
    _loop_nac: Any,
    _loop_agent_id: str,
) -> dict[str, Any]:
    """The prompt's tool descriptions: described, Wire 3's felt phrases on degraded tools, Wire 1's variance."""
    tool_descriptions = _describe_tools_for_prompt(available_tools, executor, overlay=_tool_overlay)

    # Wire 3: annotate degraded tools' descriptions in
    # place with a felt-sensation phrase (bio-fidelity
    # fold). The annotation lives at the end of the
    # description string so the LLM reads the body's
    # state in proprioceptive voice without losing
    # the tool's normal capability blurb. Uses the
    # per-tool entry's structure (dict for dynamic
    # tools, TOOL_DESCRIPTIONS dict for builtin);
    # skips any tool whose description shape we don't
    # recognise, fail-open.
    #
    # Idempotency under integrity drift (arch A1):
    # if integrity ticks 0.5 → 0.4 → 0.5 across a
    # session, the felt phrase changes per band.
    # The regex strip removes any existing
    # ``(feels …)`` Wire 3 annotation before
    # appending the current one so phrases don't
    # accumulate. Healthy / disabled affordances
    # never enter this loop, so the only way to
    # have an annotation present is via a prior
    # Wire 3 pass.
    if _wire3_degraded and _wire3_embodiment is not None:  # only filled when embodied
        for name, integrity in _wire3_degraded.items():
            entry = tool_descriptions.get(name)
            if not isinstance(entry, dict):
                continue
            base_desc = entry.get("description", "")
            if not isinstance(base_desc, str):
                continue
            phrase = _wire3_embodiment.integrity_to_felt_phrase(integrity)
            if not phrase:
                continue
            annotation = f" ({phrase})"
            # Strip any prior felt annotation pinned
            # by Embodiment.integrity_to_felt_phrase
            # — the two bands give two distinct
            # suffixes which could otherwise stack.
            stripped = _WIRE3_PHRASE_RE.sub("", base_desc)
            # Copy-on-write — TOOL_DESCRIPTIONS is
            # a shared module-level dict; mutating
            # it would poison the description for
            # future calls (and other agents).
            tool_descriptions[name] = {**entry, "description": stripped + annotation}

    annotate_tool_variance(  # Wire 1, moved verbatim to loop_planning (#1481 Rule T, A's grant)
        tool_descriptions=tool_descriptions,
        available_tools=available_tools,
        _loop_nac=_loop_nac,
        _loop_agent_id=_loop_agent_id,
        logger=logger,
    )
    return tool_descriptions


def _submit_request(
    *,
    llm_worker: Any,
    ctrl: Any,
    sim: Any,
    memory: Any,
    executor: Any,
    context: Any,
    mode_info: Any,
    internet_access: Any,
    internet_policy_summary: Any,
    available_tools: list[str],
    tool_descriptions: dict[str, Any],
    context_pool_text: str,
    conversation_history_text: str,
    agent_states: Any,
    recent_outcomes: Any,
    use_tool_prompting: Any,
    is_sleeping: bool,
    protocol_registry: Any,
    on_event: Any,
    autonomy_controller: Any,
    new_cli_input: str | None,
    followup_original_query: str,
    pending_modification: Any,
    prefetch_context_text: str,
    skip_exploration: bool,
    bio_enrichment_pipeline: Any,
    processed_cli_inputs: Any,
    _deferred_inputs: list[str],
    _planning_liveness_on: bool,
    inbound_inbox: Any,
    now: float,
) -> tuple[bool, dict[str, Any]]:
    """The request: protocol context, the inference-start event, the shared kwargs, the submit and its logs."""
    # Get protocol context (re-injected fresh each submission)
    _protocol_context = ""
    if protocol_registry is not None:
        _protocol_context = protocol_registry.get_context_for_llm()

    if callable(on_event):
        try:
            on_event(StreamEvent("inference_start", {}))
        except Exception as e:
            log_swallowed_exception(e, operation="on_event:inference_start")

    # EXPERIMENTAL — hallucination-hint feature.
    # MAXIM_TOOL_FAILURE_HINTS=1 enables (default off).
    # Disable for grounded-language acquisition Phase 0/1.
    # See docs/plans/grounded_language_acquisition.md.
    _failed_tools: list[str] = []
    if os.environ.get("MAXIM_TOOL_FAILURE_HINTS", "0") != "0":
        # __getattr__ on wrappers (InstrumentedExecutor,
        # PainInterceptorExecutor, FearGatedExecutor) walks
        # down to the base Executor where the list lives.
        _failed_tools = list(getattr(executor, "_tools_hallucinated", []))

    _shared_submit = planning_submit_shared_kwargs(
        mode_info=mode_info,
        internet_access=internet_access,
        internet_policy_summary=internet_policy_summary,
        available_tools=available_tools,
        tool_descriptions=tool_descriptions,
        context_pool_text=context_pool_text,
        agent_states=agent_states,
        recent_outcomes=recent_outcomes,
        use_tool_prompting=use_tool_prompting and bool(available_tools),
        conversation_history_text=conversation_history_text,
        is_sleeping=is_sleeping,
        protocol_context=_protocol_context,
    )
    submitted = submit_tick_context(
        llm_worker,
        context=context,
        shared=_shared_submit,
        autonomy_level=autonomy_controller.current_level,
        _failed_tools=_failed_tools,
        triggering_input=_triggering_input(new_cli_input, followup_original_query, pending_modification),
        pending_modification=pending_modification,
        prefetch_context_text=prefetch_context_text,
        skip_exploration=skip_exploration,
        deliberation_available=bio_enrichment_pipeline is not None,
        processed_cli_inputs=processed_cli_inputs,
        _deferred_inputs=_deferred_inputs or take_replan_inputs(ctrl, memory, _planning_liveness_on),
        inbound_feed=inbound_inbox,
    )
    ctrl.last_llm_submit_time = now
    # Log submission for both user input and followups
    if submitted:
        # Check if this is a followup submission
        is_followup = context.cli_inputs and any(inp.startswith("[ACTION_FOLLOWUP") for inp in context.cli_inputs)
        if is_followup:
            logger.info("Submitted followup to LLM for processing")
        elif new_cli_input:
            logger.info(
                "Submitted to LLM: %s",
                new_cli_input[:50] if len(new_cli_input) > 50 else new_cli_input,
            )
        log_agentic(
            "agent_loop",
            "llm_submit",
            {
                "input": new_cli_input[:50] if new_cli_input else "followup" if is_followup else None,
                "mode": mode_info.name,
                "autonomy": autonomy_controller.current_level.value,
                "tools_available": len(available_tools),
            },
        )
        sim.log("EXEC", f"LLM submit: {new_cli_input if new_cli_input else 'followup'}")
    return submitted, _shared_submit


def _deliberate(
    *,
    submitted: bool,
    _pfc_gate_passed: bool,
    bio_enrichment_pipeline: Any,
    llm_worker: Any,
    stop_event: Any,
    ctrl: Any,
    _max_cycles_override: Any,
    percept_source: Any,
    _shared_submit: dict[str, Any],
    autonomy_controller: Any,
    inbound_inbox: Any,
    state: Any,
    _wms: Any,
    context: Any,
    thought_gate: Any,
    step_num: Any,
    sim: Any,
    _is_novel_thought: Callable[[str], bool],
    _salience_c1: float | None,
    _run_deliberation_cycles: Callable[..., Any],
    _wait_for_proposal: Callable[..., Any],
) -> None:
    """PFC multi-cycle deliberation after a submit (section 1.2's gate passed)."""
    # ── PFC multi-cycle deliberation ──────────────────
    # When the gate passed in section 1.2 and the first
    # proposal says ready_to_act=False, run cycles 2+
    # right here where all submission params are in scope.
    if submitted and _pfc_gate_passed and bio_enrichment_pipeline is not None:
        _first = _wait_for_proposal(llm_worker, stop_event, ctrl=ctrl)
        if _first is not None:
            _max_cyc = (
                _max_cycles_override if _max_cycles_override is not None else (3 if percept_source is not None else 2)
            )
            if not _first.ready_to_act:
                # Build a submit closure that captures all params
                _submit_kwargs = dict(
                    **_shared_submit,
                    autonomy_level=autonomy_controller.current_level,
                    triggering_input="",
                    pending_modification=None,
                    prefetch_context="",
                    skip_exploration=False,
                    deliberation_available=True,
                    deferred_inputs=[],
                    inbound_feed=inbound_inbox.continuation() if inbound_inbox is not None else None,
                )

                def _submit_fn(_ctx: Any, _kw: dict = _submit_kwargs) -> bool:
                    return llm_worker.submit_context(context=_ctx, **_kw)

                _active_goal = state.data.get("active_goal") if hasattr(state, "data") else None
                _delib = _run_deliberation_cycles(
                    first_proposal=_first,
                    bio_enrichment=bio_enrichment_pipeline,
                    working_memory=_wms,
                    context=context,
                    submit_fn=_submit_fn,
                    llm_worker=llm_worker,
                    stop_event=stop_event,
                    thought_gate=thought_gate,
                    active_goal=_active_goal,
                    step_num=step_num,
                    max_cycles=_max_cyc,
                    ctrl=ctrl,
                )
                if _delib is not None:
                    ctrl.pending_proposal = _delib
                    sim.log(
                        "DELIBERATION",
                        f"multi-cycle deliberation yielded proposal: "
                        f"tool={_delib.action.get('tool_name') if isinstance(_delib.action, dict) else None}",
                    )
                else:
                    sim.log("DELIBERATION", "multi-cycle deliberation returned None (IDLE)")
            else:
                # ready_to_act == True on cycle 1 — use directly
                ctrl.pending_proposal = _first
                from maxim.simulation.sim_logger import (
                    sim_contemplation,
                    sim_deliberation_end,
                    sim_deliberation_update,
                )

                # Update thinking panel with the AUT's actual reasoning
                # Only if the thought is novel enough vs recent thoughts
                _first_reasoning = _first.reasoning or ""
                if _first_reasoning and _is_novel_thought(_first_reasoning):
                    sim_deliberation_update(
                        _first_reasoning,
                        cycle=1,
                        max_cycles=_max_cyc,
                        salience=_salience_c1,
                    )
                # NOTE: score=0.0 is a placeholder here —
                # the gate decision is not in this scope
                # (different function). Left explicit
                # rather than silently fabricated; wire a
                # real value if this log is ever relied on.
                sim_contemplation(gate_passed=True, refined=False, score=0.0)
                sim_deliberation_end(cycle=1, max_cycles=_max_cyc, summary="Ready to act (cycle 1)")
                if thought_gate is not None:
                    thought_gate.reset_refractory(step_num)


def _triggering_input(new_cli_input: Any, followup_original_query: str, pending_modification: Any) -> str:
    """The input a §6 submit answers: the new input, else the follow-up's original query, else a pending plan
    modification's text (moved out of ``_run_agentic_loop_body``, rule T for #1451)."""
    modification = pending_modification.get("user_modification", "") if pending_modification else ""
    return new_cli_input or followup_original_query or modification


def _thaw(value: Any) -> Any:
    """A read-only mapping (and any nested in it) as plain ``dict``s (#1260's overlay)."""
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    return value


def _advertisable_tools(tools: Any, executor: Any | None, autonomy_controller: Any | None) -> list[str]:
    """The tools the model may be shown: only what will be let through.

    A hard allow/deny list (``tools.allow``/``tools.deny``) used to gate at dispatch only, so the model kept
    seeing, and choosing, tools that could only ever answer with a denial (bugs ledger D82): ``executor.permits``.
    An autonomy hard forbid is the same class (#1412, Console Talk offered ``bash`` and then refused it):
    ``AutonomyController.forbids_tool``, the static ``SafetyConstraints.forbidden_tools``, never the moment's pause.
    """
    return [
        t
        for t in tools
        if (executor is None or executor.permits(t))
        and (autonomy_controller is None or not autonomy_controller.forbids_tool(t))
    ]


def _describe_tools_for_prompt(
    available_tools: Any,
    executor: Any,
    *,
    overlay: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Full tool info for the prompt (description, params, example).

    Builtin tools come from ``TOOL_DESCRIPTIONS`` with the agent's own entries (``overlay``, the run's
    ``LoopRun.tool_overlay``: the narrator's, from its registry, #1260) layered over them; dynamic tools (skills,
    protocols, SEM affordances) are described from the ``Tool`` instance
    via ``Tool.to_json_schema()`` so tools authored in either input_schema
    format render correctly. Pre-CC9 this iterated ``input_schema.items()``
    directly and silently produced empty params for JSONSchema-authored
    tools (the ``@maxim.tool`` decorator path) via the swallow-everything
    except below. Extracted from ``run_agentic_loop`` 2026-09-04 (P21).
    """
    # The overlay is read-only (``NARRATOR_TOOL_DESCRIPTIONS``): thawed into fresh dicts, which every downstream
    # reader expects (the prompt builder renders only ``dict`` entries; Wire 1/3 copy-on-write over them).
    from maxim.modes.definitions import TOOL_DESCRIPTIONS

    described = {**TOOL_DESCRIPTIONS, **{name: _thaw(entry) for name, entry in overlay.items()}}
    tool_descriptions: dict[str, Any] = {}
    for name in available_tools:
        if name in described:
            tool_descriptions[name] = described[name]
            continue
        try:
            tool = executor.registry.get(name)
            json_schema = (
                tool.to_json_schema() if hasattr(tool, "to_json_schema") else {"properties": {}, "required": []}
            )
            properties = json_schema.get("properties", {}) or {}
            required = set(json_schema.get("required", []) or [])
            params: dict[str, str] = {}
            for param_name, prop in properties.items():
                prop_type = prop.get("type", "string") if isinstance(prop, dict) else "string"
                if param_name in required:
                    params[param_name] = str(prop_type)
                else:
                    default = prop.get("default") if isinstance(prop, dict) else None
                    params[param_name] = f"({prop_type}, default={default!r})"
            tool_descriptions[name] = {
                "description": tool.description,
                "params": params,
                "example": None,
                "followup_type": None,
            }
        except (KeyError, Exception):
            pass
    return tool_descriptions
