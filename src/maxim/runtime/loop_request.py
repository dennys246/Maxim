"""The agent loop's §6 submit, whole: the request assembly and the PFC deliberation after 7a's gate (1.3.6 loop
phase 2, slice 7b).

``submit_tick`` is §6's whole submit ``try`` as one pass of ``agent_loop.run_agentic_loop`` runs it, moved verbatim
from the loop's body in pieces so each function stays at 200 lines or fewer (owner decision 2026-11-21). The body
calls it once per due submit and keeps the guard in front of it (substrate-primary, a worker, no pending proposal,
nothing held and the run not stopped, the submit interval) and the tool snapshot ``_all_tools``. Its first statement
is slice 7a's gate, ``loop_submit.prepare_submission`` (the tick's context and whether anything is worth a
submission); this module holds what follows it, given a context (owner decision 2026-11-22: its own module, not
``loop_submit``'s second half):

- ``_tick_mode_info``: the operational mode's ``ModeInfo`` (unknown = passive, #829) with the response-token override.
- In ``submit_tick``: the internet status, the new-inputs filter, the surfaced list, the context pool and conversation
  text, the pending plan modification, and the speculative prefetch (consumed here).
- ``_surface_tools``: the roster shown to the model: the mode's tools, the wired body's SEM tools, Wire 3's filter (and
  its ``WIRE_3_FILTER`` sim event), then the permits/forbid filter ``loop_state._advertisable_tools`` (#1412).
- ``_describe_surfaced_tools``: the descriptions, Wire 3's felt phrase on each degraded tool, Wire 1's variance.
- ``_submit_request``: the protocol context, the ``inference_start`` event, the shared kwargs, ``submit_tick_context``
  (with the run's inbound comms, #1481), the submit stamp, and the submit's logs.
- ``_deliberate``: after a submit whose section-1.2 PFC gate passed, cycles 2+ through the body's
  ``_run_deliberation_cycles``, or a cycle-1 ready answer used directly; either answer goes through §2's one intake,
  ``loop_intake.accept_llm_proposal`` (#1743: it used to be installed on ``ctrl`` past §2's checks, so an action-less
  one blocked every later submit, and the cycles' action-less exits returned None, skipping the intake), with a
  declined turn dropped, not retried (owner decision 2026-11-24). A wait that brings nothing back closes the
  submit's ``inference_start`` with ``inference_end`` (``has_proposal`` False).
- ``log_submit_blocked``: the sim trace of a submit a pending proposal blocks, from ahead of the body's guard.

A raise anywhere in the ``try`` is logged (WARNING plus a DEBUG traceback) and the pass goes on, as inline. By rule (a)
the helpers called only from this code moved here too: ``_triggering``, ``_describe_tools_for_prompt`` with
``_thaw``, and ``_WIRE3_PHRASE_RE``. ``_advertisable_tools``, which §4's loop-cap message also calls, moved to
``loop_state`` (the home of helpers the loop's sections share, beside ``operational_mode``); both callers reach it
through the module reference, so there is ONE binding for a patch to hit (owner decision 2026-11-22).

**Why ``_run_deliberation_cycles`` stays in ``agent_loop``.** It is pinned over 200 lines and listed in the 1.4.0
release gate's ``debt_snapshot`` (``scripts/release_gates.json``); moved verbatim, it would need an
``accepted_new_debt`` record, which is the owner's call (a later slice that also splits it to 200 lines or fewer leaves
no baseline entry and needs none; owner decision 2026-11-22: later). Rule (c) forbids importing ``agent_loop`` here, so
the body passes it in as the keyword ``_run_deliberation_cycles``, with ``_wait_for_proposal`` (which stays beside it:
cycles 2+ inside it and cycle 1 in ``_deliberate`` read the same one function) and its ``LoopRun`` novelty check as
``_is_novel_thought``, under the names the moved text already calls.

**What it returns.** ``SubmitCarry``: the two body locals §6 rebinds that a LATER pass reads, ``last_surfaced_tools``
(§2's ``record_surfaced_but_unused``) and ``pending_prefetch`` (None once consumed; the body reassigns it only on a pass
with new input), and since #1743 the stall counter ``consecutive_llm_fallbacks`` after the deliberation's intake.
They are returned even after a raise, carrying whatever the ``try`` had rebound by then, exactly as the inline locals
did: for the counter that is its value before the submit, which on a raise after the intake ran misnumbers only the
next fallback's log line (``_LLM_STALL_THRESHOLD`` is 1). The intake's ``exhausted`` is not carried: a spent budget
latches ``ctrl.planning_exhausted``, which ``loop_gates.pre_tick_gate`` reads on the next pass, so it aborts even
after a raise and on an idle pass. That both live as body locals apart from ``ctrl``'s copies is pinned as-is (#1744).

**Arguments.** Explicit keywords, never the loop's ``LoopRun`` (rule (d)), each the loop's own local named as the body
names it (``_loop_nac``, ``_tool_overlay``, ...), so the moved text is byte for byte the inline text, re-indented. One
exception: cycle 1's salience was read as ``locals().get("_salience_c1")`` (set only when section 1.2's cycle-1
enrichment ran, so stale or absent otherwise); the body still reads it that way and passes the value as
``_salience_c1``. The code LOGS on the ``maxim.runtime.agent_loop`` logger (``logger`` below is that same object).

Characterization: ``tests/unit/test_loop_submit_request_characterization.py`` (written before the move, through the
public ``run_agentic_loop``, and kept green by it).
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
from maxim.runtime import loop_state as _lstate
from maxim.runtime.loop_controller import take_replan_inputs
from maxim.runtime.loop_external import planning_submit_shared_kwargs, submit_tick_context
from maxim.runtime.loop_intake import IntakeOutcome, accept_llm_proposal
from maxim.runtime.loop_perception import is_tool_followup
from maxim.runtime.loop_planning import annotate_tool_variance
from maxim.runtime.loop_state import operational_mode
from maxim.runtime.loop_submit import prepare_submission
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
class SubmitCarry:
    """What one §6 submit hands back to the loop's body: the three body locals §6 rebinds that a LATER pass reads
    (module docstring, "What it returns"). Runtime-ephemeral, as ``SubmitGate``."""

    last_surfaced_tools: list[str]  # §2's ``record_surfaced_but_unused`` reads it on the next pass (#1744)
    pending_prefetch: Any  # None once a submit consumed it; the body reassigns it only on a pass with new input
    consecutive_llm_fallbacks: int  # #1743: the stall counter after the deliberation's intake (module docstring)


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
    action_sink: Any,
    _consecutive_llm_fallbacks: int,
) -> SubmitCarry:
    """§6, the whole submit (module docstring): the gate, then the request and the deliberation; a raise anywhere
    is logged and the pass goes on, as inline."""
    _intake = IntakeOutcome(exhausted=False, consecutive_llm_fallbacks=_consecutive_llm_fallbacks)
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

            _keep_newest_cli_input(context, processed_cli_inputs, new_cli_input)

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
                state=state,
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

            _intake = _deliberate(
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
                executor=executor,
                action_sink=action_sink,
                on_event=on_event,
                last_surfaced_tools=last_surfaced_tools,
                pending_next_actions=ctrl.pending_next_actions,  # the body's ``pending_next_actions`` is this list
                _planning_liveness_on=_planning_liveness_on,
                _consecutive_llm_fallbacks=_consecutive_llm_fallbacks,
            )

    except Exception as e:
        import traceback

        logger.warning(f"Failed to submit context to LLM: {type(e).__name__}: {e}")
        logger.debug(f"Full traceback:\n{traceback.format_exc()}")
    return SubmitCarry(
        last_surfaced_tools=last_surfaced_tools,
        pending_prefetch=pending_prefetch,
        consecutive_llm_fallbacks=_intake.consecutive_llm_fallbacks,
    )


def log_submit_blocked(*, llm_worker: Any, ctrl: Any, cli_input: Any, sim: Any) -> None:
    """§6's diagnostic, ahead of its guard: trace why a submit is skipped (moved verbatim from the loop's body)."""
    # Diagnostic: trace why LLM submission is skipped
    if llm_worker and ctrl.pending_proposal is not None and cli_input and sim.is_sim_mode:
        _pp_tool = (
            ctrl.pending_proposal.action.get("tool_name", "?")
            if isinstance(ctrl.pending_proposal.action, dict)
            else "?"
        )
        sim.log("PIPELINE", f"LLM gate BLOCKED: ctrl.pending_proposal={_pp_tool}, new cli_input={str(cli_input)[:40]}")


def _keep_newest_cli_input(context: Any, processed_cli_inputs: Any, new_cli_input: str | None) -> None:
    """Narrow the context's CLI inputs to the newest unprocessed one (moved verbatim from ``submit_tick``, #1743)."""
    # Filter CLI inputs to only include new ones (not already processed)
    # This prevents the LLM from seeing the same input multiple times
    if context.cli_inputs:
        new_inputs = [inp for inp in context.cli_inputs if inp not in processed_cli_inputs or inp == new_cli_input]
        # Only keep the most recent new input
        if new_cli_input:
            context.cli_inputs = [new_cli_input]
        else:
            context.cli_inputs = new_inputs[-1:] if new_inputs else []


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
    mode_info: ModeInfo,
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
    available_tools = _lstate._advertisable_tools(available_tools, executor, autonomy_controller)
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
    state: Any,
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
        **_triggering(new_cli_input, followup_original_query, pending_modification, state),
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
        is_followup = context.cli_inputs and any(is_tool_followup(inp) for inp in context.cli_inputs)
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
    executor: Any,
    action_sink: Any,
    on_event: Any,
    last_surfaced_tools: list[str],
    pending_next_actions: list[dict[str, Any]],
    _planning_liveness_on: bool,
    _consecutive_llm_fallbacks: int,
) -> IntakeOutcome:
    """PFC multi-cycle deliberation after a submit (section 1.2's gate passed). The answer it consumes goes through
    ``loop_intake.accept_llm_proposal``, §2's one intake (#1743), given the arguments after ``_wait_for_proposal`` as
    §2 passes them; returns that intake's outcome (an unchanged one when it took no answer)."""

    def _accept(proposal: Any) -> IntakeOutcome:
        return accept_llm_proposal(
            proposal,
            llm_worker=llm_worker,
            ctrl=ctrl,
            sim=sim,
            state=state,
            executor=executor,
            action_sink=action_sink,
            on_event=on_event,
            last_surfaced_tools=last_surfaced_tools,
            pending_next_actions=pending_next_actions,
            _planning_liveness_on=_planning_liveness_on,
            _consecutive_llm_fallbacks=_consecutive_llm_fallbacks,
            declined_is_failure=False,  # a declined turn is not retried (owner decision 2026-11-24)
            ends_turn=True,  # this pass's submit, answered synchronously: the bridge's turn
        )

    _outcome = IntakeOutcome(exhausted=False, consecutive_llm_fallbacks=_consecutive_llm_fallbacks)
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
                _outcome = _accept(_delib)  # the answer it settled on (#1743: never None); the intake decides
                sim.log(
                    "DELIBERATION",
                    f"multi-cycle deliberation yielded proposal: "
                    f"tool={_delib.action.get('tool_name') if isinstance(_delib.action, dict) else None}"
                    if ctrl.pending_proposal is _delib
                    else "multi-cycle deliberation declined: the intake took no action",
                )
            else:
                # ready_to_act == True on cycle 1 — use directly
                _outcome = _accept(_first)
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
                _taken = ctrl.pending_proposal is _first  # the intake's verdict: installed, or declined/dropped
                sim_deliberation_end(
                    cycle=1, max_cycles=_max_cyc, summary="Ready to act (cycle 1)" if _taken else "Declined (cycle 1)"
                )
                if thought_gate is not None:
                    thought_gate.reset_refractory(step_num)
        elif callable(on_event):  # nothing came back (a timeout or a stop): close the submit's inference_start
            try:
                on_event(StreamEvent("inference_end", {"has_proposal": False}))
            except Exception:
                log_swallowed_exception(site="on_event:inference_end")
    return _outcome


def _triggering(new_cli_input: Any, followup_original_query: str, pending_modification: Any, state: Any) -> dict:
    """The input a §6 submit answers (the new input, else the follow-up's original query, else a pending plan
    modification's text; moved out of ``_run_agentic_loop_body``, rule T for #1451), and where it came from: the source
    ``store_user_input`` recorded for that same text (``pending_user_input_source``: "CLI" or "voice"), "" when the
    text is not the stored input (#1519: the prompt labels a voice request as heard on the room mic)."""
    modification = pending_modification.get("user_modification", "") if pending_modification else ""
    text = new_cli_input or followup_original_query or modification
    data = state.data
    stored = new_cli_input and str(data.get("pending_user_input", "")).strip() == str(new_cli_input).strip()
    source = str(data.get("pending_user_input_source", "")) if stored else ""
    return {"triggering_input": text, "triggering_input_source": source}


def _thaw(value: Any) -> Any:
    """A read-only mapping (and any nested in it) as plain ``dict``s (#1260's overlay)."""
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    return value


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
