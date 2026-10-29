"""The agent loop's PLANNING approved path, section 5 (1.3.2 decomposition, slice 4; #1085 PR-a).

At ``AutonomyLevel.PLANNING`` the loop queues every proposal in ``AutonomyController.proposal_queue`` (section 4).
``drain_approved`` executes the entries an approver marked approved. Today that approver is an EMBEDDER calling
``proposal_queue.approve``; this module opens no other approval route (the approval surface is #1185).

Owner decisions (2026-10-08, and 2026-10-09 after the review):

- **One dispatch.** Each approved action runs through the run's bound ``tool_dispatch.execute_and_learn`` with
  ``human_involved=True``, the function the autonomous and the SUPERVISED-confirmed paths call (#1133), so an
  approved action learns exactly what it would have learned autonomously. The credit and the capture are keyed to
  the situation the action was PROPOSED in (``Proposal.source``, #1083). The function never writes the controller:
  it returns the follow-up and the caller assigns it.
- **Approval lifts only the PLANNING level.** ``AutonomyController.approved_action_blocker`` is checked before each
  execution: the critical safety constraints and the pause (``approval_blocker``, the non-level half of
  ``can_execute_action``) and the supervision policy's hard denials (``SupervisionPolicy.hard_deny``: forbidden
  prefixes, categories and tools; owner decision S7). A blocked entry is refused (``blocked``) and not executed.
  The policy's "needs approval" checks (the ``allowed_tools`` list, confidence, ``requires_confirmation``, the
  sandbox/CWD rules) are what the approval satisfies (owner decision 2026-10-09). **The pause check is narrow:**
  while the controller is paused the loop's gate idles and this drain does not run at all, so the check catches
  only a pause landing after this tick's gate (later in the tick, or during the drain). Approved entries do not
  expire, and a later ``resume()`` would run them; that is #1185's.
- **One entry at a time.** ``ProposalQueue.pop_approved`` takes the oldest approved entry, which then executes
  before the next is taken. If one step raises (any ``BaseException``), every entry still approved is refused
  (``drain_aborted``) and the original exception propagates, as it does on the main path.
- **The DRAIN's refusals book no NAc.** ``blocked`` and ``drain_aborted`` are machine refusals, booked through
  ``LoopRun.book_machine_refusal``: the outcome window and the LLM's carryover, no NAc, no situation, no goal
  credit (``no_action`` is reported, not booked). (Elsewhere a refusal still books NAc: §4's hard rejection and
  ``handle_confirmation``'s "no", sim AUTO_REJECT included; #1185.)

A refusal is ordered: the entry is rejected (``status``/``rejected_reason``), then the typed ``plan_refused`` event
and the audit entry, then the booking (#1185 S5), so a raise in the report or the booking cannot leave the entry
approved; such a raise aborts the drain like any other.

**Arguments.** Explicit keywords, never the loop's ``LoopRun`` (rule (d) of the roadmap's import-direction
paragraph). Logs on the ``maxim.runtime.agent_loop`` logger. Characterization:
``tests/unit/test_approved_path_characterization.py`` (written before the move); gates:
``tests/unit/test_approved_path_learns_1085.py``.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from maxim.runtime.loop_state import operational_mode
from maxim.utils.logging import log_swallowed_exception
from maxim.utils.structured_logging import log_agentic

if TYPE_CHECKING:
    from maxim.agents.autonomy import AutonomyController, Proposal
    from maxim.runtime.loop_types import ActionFollowup
    from maxim.runtime.tool_dispatch import ExecutionOutcome

# The SAME logger object as ``agent_loop.logger``, so records keep the ``maxim.runtime.agent_loop`` name.
logger = logging.getLogger("maxim.runtime.agent_loop")

# Wire 1 (release_0_9_1.md Stage 4): regex matching the experience-voice
# annotations the variance-annotation hook produces. Same shape rationale as
# the Wire 3 regex (``agent_loop._WIRE3_PHRASE_RE``) — strip stale annotation before re-applying the
# current-tick one so the description doesn't accumulate suffixes across
# observations.
#
# **Register choice (bio-fidelity fold from pre-merge review):** Wire 3's
# phrases ("feels strained" / "feels weakened, prone to failing") carry a
# SOMATIC voice — they describe proprioceptive body-state. Wire 1's signal
# is METACOGNITIVE — variance over outcome reliability is "what experience
# has taught me about this action", not "what my body senses right now".
# Reusing the "feels X" stem for both collapsed two distinct bio-system
# registers (somatic vs experience-acquired) into one indistinguishable
# surface, so the LLM could not separate "I will fail because the body
# is broken" from "I will fail because the outcome is stochastic". The
# experience-voice phrasing "(unpredictable from prior experience)" /
# "(reliable from prior experience)" aligns with Wire-A's
# "[... from prior experience]" prompt-section register (the only other
# substrate surface that exposes learned variability to the LLM), keeping
# the experience-acquired signals coherent across wires while Wire 3 owns
# the somatic surface alone.
#
# Two phrases match the two bands (high variance / reliable); the middle
# band emits no annotation. Wire 1 annotation appears AFTER any Wire 3
# annotation in the description, so the regex is anchored at end-of-
# string and Wire 3's strip runs first (the two suffixes can co-occur on
# one tool — physical-damage + outcome-variance signals are orthogonal).
_WIRE1_PHRASE_RE = re.compile(r" \((?:unpredictable|reliable) from prior experience\)$")

# Wire 1 thresholds. Bernoulli variance on binary {0, 1} reward maxes at 0.25
# (p = 0.5). The bands are pre-registered in release_0_9_1.md Stage 4:
#   variance >= 0.15 → "(unpredictable from prior experience)"  — ~30/70 or worse
#   variance <  0.05 → "(reliable from prior experience)"       — ~94/6 or better
#   otherwise        → no annotation (middle band)
# Min observation count guards against single-sample noise — Welford variance
# stabilises around n ~= 5 for binary signals.
_WIRE1_HIGH_VARIANCE_THRESHOLD = 0.15
_WIRE1_LOW_VARIANCE_THRESHOLD = 0.05
_WIRE1_MIN_OBSERVATIONS = 5

# Wire 1 ablation gate. Reuses Wire-A's canonical
# ``annotation_disabled_via_env`` parser so the truthy-value set is a single
# source of truth across 0.9.1's two annotation gates. Default OFF
# (annotation ON in 0.9.1 by design). Roy-3 may set this for variance-
# annotation-off arms; the conftest scrub clears it between tests.
_WIRE1_DISABLE_ENV = "MAXIM_DISABLE_VARIANCE_ANNOTATION"
_WIRE1_HIGH_PHRASE = "unpredictable from prior experience"
_WIRE1_LOW_PHRASE = "reliable from prior experience"

# Typed refusal reasons (the ``plan_refused`` event's ``reason`` and the entry's ``rejected_reason``).
BLOCKED = "blocked"
DRAIN_ABORTED = "drain_aborted"
NO_ACTION = "no_action"


def drain_approved(
    *,
    autonomy_controller: AutonomyController,
    execute_and_learn: Callable[..., ExecutionOutcome],
    book_machine_refusal: Callable[..., None],
    executor: Any,
    observation: Any,
    state: Any,
    sim: Any,
) -> ActionFollowup | None:
    """Execute every approved PLANNING proposal, oldest first; return the last follow-up any of them queued.

    ``execute_and_learn`` / ``book_machine_refusal`` are the run's bound callables (``LoopRun``); ``executor`` is the
    run's, whose operational mode a refusal's audit entry records (#963); ``observation`` is this tick's, the
    observation the capture stores (the answer-time observation, as on the confirmed path).

    Exception-safe: the WHOLE per-entry step (the blocker, the execution, that entry's own refusal) runs inside
    one ``try``. On a raise (any ``BaseException``, ``KeyboardInterrupt`` included), every entry still approved
    is first popped and rejected (``drain_aborted``) with no I/O, then each is reported and booked under its own
    guard, and the ORIGINAL exception is re-raised: an ``Exception`` raised in that sweep is logged, never re-raised
    in its place. A second interrupt (a ``BaseException``) raised during the sweep does escape, but by then every
    entry is already rejected, because the reject pass runs before any I/O. So no approved entry survives an
    aborted drain to run later.
    """
    queue = autonomy_controller.proposal_queue
    refuser = _Refuser(
        autonomy_controller=autonomy_controller,
        book_machine_refusal=book_machine_refusal,
        executor=executor,
        state=state,
        sim=sim,
    )
    followup: ActionFollowup | None = None
    while (proposal := queue.pop_approved()) is not None:
        try:
            outcome = _drain_one(
                proposal,
                autonomy_controller=autonomy_controller,
                execute_and_learn=execute_and_learn,
                observation=observation,
                refuser=refuser,
            )
        # BaseException: a KeyboardInterrupt must not leave entries approved either. Re-raised, never swallowed.
        except BaseException as exc:
            _abort_drain(queue, exc, refuser=refuser)
            raise
        if outcome is not None and outcome.followup is not None:
            followup = outcome.followup
    return followup


def _drain_one(
    proposal: Proposal,
    *,
    autonomy_controller: AutonomyController,
    execute_and_learn: Callable[..., ExecutionOutcome],
    observation: Any,
    refuser: _Refuser,
) -> ExecutionOutcome | None:
    """One drained entry: refuse it (typed) or execute it. ``None`` when it was refused."""
    action = proposal.action
    if not action:
        refuser.refuse(proposal, reason=NO_ACTION, detail="the approved proposal carries no action")
        return None
    blocker = autonomy_controller.approved_action_blocker(action)
    if blocker is not None:
        refuser.refuse(proposal, reason=BLOCKED, detail=blocker)
        return None
    return execute_and_learn(
        action=action,
        confidence=proposal.confidence,
        # The situation it was PROPOSED in (#1083). A Proposal built outside the loop has no source: it is then
        # its own record (reasoning, citations) and credits no situation.
        proposal=proposal.source if proposal.source is not None else proposal,
        observation=observation,
        human_involved=True,
    )


def _abort_drain(queue: Any, exc: BaseException, *, refuser: _Refuser) -> None:
    """Refuse every entry still approved after ``exc``: reject them all first (no I/O), then report and book each
    under its own guard, so a failing report cannot leave an entry approved or replace ``exc``."""
    rest: list[Proposal] = []
    while (entry := queue.pop_approved()) is not None:
        _reject(entry, DRAIN_ABORTED)
        rest.append(entry)
    detail = f"an earlier approved action raised {type(exc).__name__}: {exc}"[:300]
    for entry in rest:
        try:
            refuser.report_and_book(entry, reason=DRAIN_ABORTED, detail=detail)
        except Exception:
            log_swallowed_exception(site="loop_planning.py:_abort_drain:drain_aborted")


def _reject(proposal: Proposal, reason: str) -> None:
    proposal.status = "rejected"
    proposal.rejected_reason = reason


@dataclasses.dataclass(frozen=True)
class _Refuser:
    """The run's handles a refusal needs. Runtime-ephemeral (one drain), so CC3 does not apply."""

    autonomy_controller: AutonomyController
    book_machine_refusal: Callable[..., None]
    executor: Any
    state: Any
    sim: Any

    def refuse(self, proposal: Proposal, *, reason: str, detail: str) -> None:
        """Refuse one drained entry, typed: reject it, then report and book it (#1185 S5's order)."""
        _reject(proposal, reason)
        self.report_and_book(proposal, reason=reason, detail=detail)

    def report_and_book(self, proposal: Proposal, *, reason: str, detail: str) -> None:
        """The ``plan_refused`` event, the log and sim lines, the audit entry, then the booking for the LLM (no NAc;
        an entry with no action is not booked)."""
        tool_name = str((proposal.action or {}).get("tool_name") or "unknown")
        log_agentic(
            "agent_loop",
            "plan_refused",
            {"reason": reason, "detail": detail, "tool": tool_name, "proposal_id": proposal.id},
            level="WARNING",
        )
        logger.warning("Approved proposal %s refused (%s): %s -- %s", proposal.id, reason, tool_name, detail)
        self.sim.log("PIPELINE", f"Approved proposal refused ({reason}): {tool_name} -- {detail}")
        self.autonomy_controller.log_action(
            action_type="rejected",
            action=proposal.action,
            reasoning=f"Refused ({reason}): {detail}",
            mode=operational_mode(self.executor, self.state),
            confidence=proposal.confidence,
            human_involved=False,
        )
        if proposal.action:
            self.book_machine_refusal(
                tool_name=tool_name,
                error=f"Refused ({reason}): {detail}",
                reasoning=proposal.reasoning or "",
            )


def annotate_tool_variance(
    *,
    tool_descriptions: Any,
    available_tools: Any,
    _loop_nac: Any,
    _loop_agent_id: Any,
    logger: logging.Logger,
) -> None:
    """Wire 1 (release_0_9_1.md Stage 4): annotate the planning submit's tool descriptions with outcome-variance
    experience phrasing, in place. Moved verbatim from ``agent_loop._run_agentic_loop_body`` (#1481 Rule T; Session
    A's grant 2026-10-28): its free reads are the parameters, ``tool_descriptions`` is only mutated in place, and the
    loop's own ``logger`` is passed in, so the telemetry's logger name does not move."""
    # Wire 1 (release_0_9_1.md Stage 4): annotate
    # tools with outcome-variance experience phrasing.
    # Runs AFTER Wire 3 so an integrity-degraded tool
    # that is ALSO outcome-variable gets both
    # annotations (orthogonal signals: somatic body
    # damage vs. experience-acquired unpredictability).
    # The hybrid bio-system + LLM caveat is documented
    # in docs/plans/deferred/bio_emergent_persona_foundations.md
    # § Wire 1 — variance reaches the LLM through
    # description text, not a pre-filter ranker.
    # A cleaner post-1.0 design would pre-rank tools
    # before the LLM sees them.
    #
    # Idempotency: _WIRE1_PHRASE_RE strips the prior
    # annotation before the current one is appended
    # so the description does not accumulate suffixes
    # across observations. Wire 3 (somatic
    # "feels strained" / "feels weakened, prone to
    # failing") and Wire 1 (experience-voice
    # "unpredictable from prior experience" /
    # "reliable from prior experience") use distinct
    # phrases so the two regexes do not conflict.
    _wire1_annotated_high: list[str] = []
    _wire1_annotated_low: list[str] = []
    _wire1_middle_band: list[tuple[str, float]] = []
    _wire1_felt_phrases: dict[str, str] = {}
    # Gate on bio-system + tool list FIRST so cold-start
    # agents skip the env-var read entirely (one fewer
    # os.environ lookup per submission). The ablation
    # parser is Wire-A's canonical helper — single
    # source of truth across 0.9.1's two gates.
    _risk_profile: dict[str, float] = {}
    if _loop_nac is not None and available_tools:
        from maxim.prompts.cluster_bias_annotation import (
            annotation_disabled_via_env,
        )

        _wire1_disabled_via_env = annotation_disabled_via_env(os.environ.get(_WIRE1_DISABLE_ENV))
        if not _wire1_disabled_via_env:
            try:
                _risk_profile = _loop_nac.get_action_risk_profile(
                    agent_id=_loop_agent_id,
                    min_observations=_WIRE1_MIN_OBSERVATIONS,
                )
            except (ValueError, AttributeError) as e:
                # ValueError: empty agent_id (defensive —
                # _loop_agent_id is non-empty by construction
                # at line ~1025, but a future refactor could
                # introduce a regression here). AttributeError:
                # _loop_nac is something other than an NAc
                # (test stubs / forward-compat). Either case:
                # WARN so operators notice + no-op the wire.
                logger.warning("Wire 1: risk profile fetch failed — annotation no-op: %s", e)
                _risk_profile = {}
        # _risk_profile keys are "tool:<name>" (or
        # "tool:use:<action>" for the generic use
        # tool); available_tools entries are bare
        # names. Strip the "tool:" prefix to match.
        for event_sig, variance in _risk_profile.items():
            if event_sig.startswith("tool:use:"):
                # The compound generic-use signature
                # doesn't map back to a bare tool
                # name in available_tools — the
                # available_tools entry is "use".
                # Skip; Wire 1 doesn't annotate at
                # the action-arg level (no separate
                # description per use:<action>).
                continue
            tool_name = event_sig[len("tool:") :] if event_sig.startswith("tool:") else event_sig
            if tool_name not in available_tools:
                continue
            if variance >= _WIRE1_HIGH_VARIANCE_THRESHOLD:
                phrase = _WIRE1_HIGH_PHRASE
                _wire1_annotated_high.append(tool_name)
                _wire1_felt_phrases[tool_name] = phrase
            elif variance < _WIRE1_LOW_VARIANCE_THRESHOLD:
                phrase = _WIRE1_LOW_PHRASE
                _wire1_annotated_low.append(tool_name)
                _wire1_felt_phrases[tool_name] = phrase
            else:
                # Middle band: no annotation. Strip
                # any prior annotation so an LLM
                # cannot read stale signal after the
                # tool's variance has decayed back
                # to the neutral band.
                _wire1_middle_band.append((tool_name, variance))
                entry = tool_descriptions.get(tool_name)
                if isinstance(entry, dict):
                    base_desc = entry.get("description", "")
                    if isinstance(base_desc, str) and _WIRE1_PHRASE_RE.search(base_desc):
                        tool_descriptions[tool_name] = {
                            **entry,
                            "description": _WIRE1_PHRASE_RE.sub("", base_desc),
                        }
                continue
            entry = tool_descriptions.get(tool_name)
            if not isinstance(entry, dict):
                continue
            base_desc = entry.get("description", "")
            if not isinstance(base_desc, str):
                continue
            stripped = _WIRE1_PHRASE_RE.sub("", base_desc)
            tool_descriptions[tool_name] = {
                **entry,
                "description": f"{stripped} ({phrase})",
            }
        # Emit Roy-3 disambiguator (bio-fidelity
        # measurability): without this event,
        # "Wire 1 annotated the tool" and "LLM
        # ignored the annotation" are
        # indistinguishable post-hoc. Mirrors the
        # Wire 3 WIRE_3_FILTER shape so Roy-3 can
        # count annotation effects uniformly.
        # Payload carries:
        #   - agent_id          (multi-agent attribution)
        #   - high_variance_tools, reliable_tools
        #   - felt_phrases      (exact strings the LLM saw)
        #   - annotated_variances (numeric float per annotated tool)
        #   - middle_band       (tool, variance) tuples for
        #                       counterfactual Roy-3 analysis
        if _wire1_annotated_high or _wire1_annotated_low or _wire1_middle_band:
            try:
                from maxim.simulation import sim_logger as _sl_w1

                _w1_tick = int(time.time() - _sl_w1._sim_start) if _sl_w1._sim_start > 0.0 else 0
                _sl_w1.sim_log(
                    "WIRE_1_ANNOTATION",
                    f"wire_1: high_variance={len(_wire1_annotated_high)} "
                    f"reliable={len(_wire1_annotated_low)} "
                    f"middle={len(_wire1_middle_band)}",
                    {
                        "tick": _w1_tick,
                        "agent_id": _loop_agent_id,
                        "high_variance_tools": sorted(_wire1_annotated_high),
                        "reliable_tools": sorted(_wire1_annotated_low),
                        # felt_phrases is the LLM-visible
                        # text — Roy-3 reads this to
                        # decide what the prompt actually
                        # contained (vs reconstructing
                        # from thresholds + variances).
                        "felt_phrases": dict(_wire1_felt_phrases),
                        # Numeric variance for each
                        # annotated tool — LLM does NOT
                        # see these floats, only the
                        # felt phrase above.
                        "annotated_variances": {
                            (event_sig[len("tool:") :] if event_sig.startswith("tool:") else event_sig): round(
                                variance, 4
                            )
                            for event_sig, variance in _risk_profile.items()
                            if not event_sig.startswith("tool:use:")
                            and (event_sig[len("tool:") :] if event_sig.startswith("tool:") else event_sig)
                            in (_wire1_annotated_high + _wire1_annotated_low)
                        },
                        # Middle-band tools: present
                        # in the profile but not
                        # annotated. The counterfactual
                        # for Roy-3 ablation analysis
                        # ("substrate produced variance
                        # in this band but no
                        # annotation reached the LLM").
                        "middle_band_variances": {name: round(var, 4) for name, var in _wire1_middle_band},
                    },
                )
            except ImportError:
                pass
