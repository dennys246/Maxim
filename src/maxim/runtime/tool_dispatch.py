"""Tool dispatch utilities — outcome recording, parallel execution, agent name.

Extracted from agent_loop.py for single-responsibility decomposition.
The functions here handle tool outcome recording (including NAc causal
learning and energy tracking), parallel action execution, and safe
agent name extraction.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
import time
from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Any

from maxim.decisions.causal_link import Valence as _V
from maxim.runtime.bio_integration import capture_loop_action, record_plan_outcome as _record_plan_outcome
from maxim.runtime.loop_state import operational_mode
from maxim.runtime.loop_types import ActionFollowup
from maxim.tools.base import GATE_REFUSAL_KINDS, ToolErrorKind
from maxim.utils.logging import log_swallowed_exception
from maxim.utils.structured_logging import log_agentic

logger = logging.getLogger(__name__)
# ``execute_and_learn`` logs as the agent loop always has (the ``loop_setup`` precedent): the SAME logger
# object as ``agent_loop.logger``, so its records keep the ``maxim.runtime.agent_loop`` name.
_loop_logger = logging.getLogger("maxim.runtime.agent_loop")

# Outcome-signature token per learning tier. A NEUTRAL outcome must not
# share causal-link identity with a success or a failure: link ids hash
# ``outcome_signature``, so reusing "success" for an ineffective action
# would merge it into the successful link and re-book the very positive
# this fix removes.
_OUTCOME_TOKEN: dict[_V, str] = {
    _V.POSITIVE: "success",
    _V.NEUTRAL: "ineffective",
    _V.NEGATIVE: "failure",
}


# Registry string -> tier. The side_effects channel is JSON-shaped (it
# crosses no wire today, but the registry is a third-party contract), so the
# key carries the enum's VALUE rather than the enum.
_VALENCE_BY_NAME: dict[str, _V] = {
    "positive": _V.POSITIVE,
    "neutral": _V.NEUTRAL,
    "negative": _V.NEGATIVE,
}


@dataclasses.dataclass(frozen=True)
class LearningSideEffects:
    """The learning-relevant signals a tool reports about its own outcome.

    Read from ``ToolOutput.side_effects`` — the typed channel the bio
    pipeline branches on. Consumers read ``side_effects`` and never
    ``metadata``, so a signal filed under ``metadata`` is structurally
    invisible to learning no matter how carefully the tool measured it.
    The append-only key registry lives in ``docs/user/tool_side_effects.md``.

    Homed here rather than in ``agent_loop`` so all three dispatch paths —
    the serial loop, ``execute_parallel_actions`` below, and
    ``runtime/executor.py`` — read the registry through ONE parser. An
    earlier revision put it in ``agent_loop``, which imports FROM this
    module, so the two other consumers had to hand-roll their own read.

    Runtime-ephemeral: constructed and consumed within a single dispatch,
    never persisted and never crossing a wire, so CC3 forward-compat is
    out of scope.
    """

    embodiment_failed: bool = False
    drive_potential_diff: float | None = None
    drive_credit_withheld: bool = False
    drive_relief_channel: str | None = None
    outcome_valence: _V | None = None


def read_learning_side_effects(result: Any) -> LearningSideEffects:
    """Extract the learning tier + credit routing from a tool result.

    * ``embodiment_failures`` — an action that mechanically succeeded but
      HARMED the body (a ``self_effect`` breached a sensor's comfort band)
      is a NEGATIVE learning outcome, so ``record_outcome`` does not book a
      spurious positive that masks the aversion
      (substrate_primary_cradle_readiness.md B5).
    * ``drive_potential_diff`` — motor credit (GAP 1): the drive relief this
      action produced, if it touched a drive sensor. ``record_outcome``
      prefers its SIGN as the cluster reward over the ±1 tool-success.
    * ``drive_credit_withheld`` — sem_motor_binding.md Phase 1:
      drive-touched-but-unmeasured. Suppresses the flat +1 floor for THIS
      action WITHOUT asserting harm.
    * ``drive_relief_channel`` — Phase 2: measured exteroceptive relief
      routes to the direction-bearing cluster instead of interoception.
    * ``outcome_valence`` — D53: the tool's OWN report of what it achieved,
      as distinct from mechanical success. A motion that could not be
      verified, or that moved nothing, is ``"neutral"``; a CONFIRMED
      shortfall is ``"negative"`` — which is a real negative outcome but
      NOT harm, so it must not be laundered through ``embodiment_failures``.

    A tool that reports nothing yields the all-default value, which is the
    historical behaviour: mechanical success means POSITIVE. An
    unrecognised string is ignored rather than guessed at.
    """
    side = getattr(result, "side_effects", None)
    # The registry types this channel ``dict[str, Any] | None``. Anything
    # else is not a valid payload, and reading keys off it would invent
    # signals out of whatever the object returns — so refuse rather than
    # guess. (Found by the pre-merge round: a Mock result made every key
    # read truthy and booked a spurious NEGATIVE.)
    if not isinstance(side, dict) or not side:
        return LearningSideEffects()
    raw = side.get("outcome_valence")
    return LearningSideEffects(
        embodiment_failed=bool(side.get("embodiment_failures")),
        drive_potential_diff=side.get("drive_potential_diff"),
        drive_credit_withheld=bool(side.get("drive_credit_withheld")),
        drive_relief_channel=side.get("drive_relief_channel"),
        outcome_valence=_VALENCE_BY_NAME.get(raw) if isinstance(raw, str) else None,
    )


def _operant_only_credit_enabled() -> bool:
    """True when ``MAXIM_OPERANT_ONLY_CREDIT`` is set (cradle_mother experiment).

    In this mode a learner's action value comes SOLELY from a caregiver's
    contingent operant reward (``NAc.credit_operant_reward``): the substrate
    remembers each action but does NOT book the uniform tool-success cluster
    reward for a driveless action. Probe 3 (``scripts/orient_substrate/
    3_operant_feed_probe.py``) proved the floor otherwise saturates the cluster
    cap and drowns the operant signal. Experiment/harness toggle (env, not
    config) — read per call so tests can flip it; the hot-path cost is one
    ``os.environ.get``. Autouse scrub: tests/conftest.py."""
    from maxim.prompts.cluster_bias_annotation import annotation_disabled_via_env

    return annotation_disabled_via_env(os.environ.get("MAXIM_OPERANT_ONLY_CREDIT"))


def safe_agent_name(agent: Any) -> str:
    """Extract a filesystem-safe agent name from an agent object."""
    raw = None
    try:
        raw = getattr(agent, "state_name", None) or getattr(agent, "agent_name", None) or getattr(agent, "name", None)
    except (AttributeError, TypeError) as e:
        log_swallowed_exception(e, operation="get_agent_name")
        raw = None
    if not raw:
        raw = type(agent).__name__
    name = str(raw).strip() or "agent"
    name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)
    return name.strip("._-") or "agent"


def build_tool_signature(tool_name: str, tool_params: dict[str, Any] | None = None) -> str:
    """Build a compound NAc event signature for a tool call.

    For generic action tools like ``use``, includes the ``action``
    parameter so NAc distinguishes ``use:dodge`` from ``use:open``.
    For all other tools, returns ``tool:<name>``.

    This is the single source of truth for tool→NAc event signature
    format.  All code that records or queries tool signatures MUST
    use this function.
    """
    if tool_params and tool_name == "use":
        action = tool_params.get("action", "")
        if action:
            return f"tool:use:{action}"
    return f"tool:{tool_name}"


def _observe_tool_result(
    nac: Any,
    sig: str,
    learn_valence: _V,
    valence: _V,
    outcome_summary: str,
    elapsed_s: float,
    ctx: dict[str, Any],
    bridge_booked: str | None,
) -> Any:
    """Dispatch's own ``tool_result`` link for one outcome, or ``None`` when it is skipped (#1200 Stage 2, TF7).

    Skipped only for a failure the tool-pain bridge already booked under this EXACT signature (the executor's
    ``ToolOutput.nac_failure_booked`` stamp), so the failure is booked once; every other sink in
    ``record_outcome`` (goal credit, cluster reward, the operant action, energy, the plan bridge) is unchanged.
    A compound (``tool:use:<action>``) or aliased signature, ``None`` (no bridge, a human driving, a speculative
    or never-run call) or a non-failure books as before.
    """
    if bridge_booked is not None and bridge_booked == sig and learn_valence is _V.NEGATIVE:
        return None  # the bridge's booking traced itself (NAc's sim_nac_learn); nothing to trace here (#1221 3)
    link = nac.observe(
        event_type="tool",
        event_signature=sig,
        outcome_type="tool_result",
        outcome_signature=f"{_OUTCOME_TOKEN[learn_valence]}:{outcome_summary}",
        outcome_valence=valence,
        delta_seconds=elapsed_s,
        context=ctx,
    )
    from maxim.simulation.sim_logger import sim_nac  # contained: never raises (#863)

    sim_nac(sig, valence.value, getattr(link, "last_rpe", 0.0) or 0.0, getattr(link, "confidence", 0.5))
    return link


def gate_refusal_of(result: Any) -> ToolErrorKind | None:
    """*result*'s gate refusal kind (``tools/base.py::GATE_REFUSAL_KINDS``), or None: the tool ran, or a tool's own
    refusal (``PERMISSION_DENIED`` keeps booking its NAc negative). What ``record_outcome(refused=)`` is passed."""
    kind = getattr(result, "error_kind", None)
    return kind if isinstance(kind, ToolErrorKind) and kind in GATE_REFUSAL_KINDS else None


def record_outcome(
    *,
    agent_id: str,
    tool_name: str,
    success: bool,
    result_summary: str | None,
    error: str | None,
    reasoning: str,
    recent_outcomes: list[dict[str, Any]],
    max_recent: int,
    llm_worker: Any | None,
    context_pool: Any,
    nac: Any | None = None,
    elapsed_s: float = 0.0,
    active_goal: str | None = None,
    tool_params: dict[str, Any] | None = None,
    cluster_id: str | None = None,
    clusters: dict[str, str] | None = None,
    embodiment_failed: bool = False,
    drive_potential_diff: float | None = None,
    drive_relief_only: bool = False,
    drive_credit_withheld: bool = False,
    drive_relief_channel: str | None = None,
    outcome_valence: "_V | None" = None,
    bridge_booked: str | None,
    refused: ToolErrorKind | None,
) -> None:
    """Record a tool outcome to all sinks including NAc causal learning.

    ``bridge_booked``: the signature the tool-pain bridge booked THIS failure under (``ToolOutput.nac_failure_booked``,
    #1200 Stage 2); see ``_observe_tool_result``. REQUIRED (TF8): forgetting it is a ``TypeError``, not a double count.

    ``refused``: the gate kind when a dispatch gate refused the action (``gate_refusal_of``; #1315 item 3, §1A.2),
    else None. REQUIRED, like ``bridge_booked``. A gate refusal is booked as ``book_machine_refusal`` books one (the
    #1185 precedent): the outcome window, the carryover and the context pool, never NAc, goal credit or a cluster.

    Appends to recent_outcomes, records reasoning carryover on llm_worker,
    adds to context_pool, and (if NAc is wired) records a causal observation
    so NAc learns tool → outcome patterns.

    ``agent_id`` is required (keyword-only) and must be a non-empty
    string so multi-agent paths attribute learning to the right
    agent. Forgetting it is a TypeError, and an empty string is a
    ValueError — pre-merge architecture review caught the empty-
    string bypass as the same band-aid pattern P4 was supposed to
    eliminate. This mirrors ``build_executor(pain_bus=...)`` and
    ``build_pain_bus(hippocampus=..., nac=...)`` — pushing silent-
    no-op invariants into the type. The ``agent_id`` is included in
    the NAc context dict so links can be filtered per-agent at query
    time.

    ``clusters`` is the extero/intero-seam per-modality active-cluster set
    (``LLMProposal.clusters``, ``{modality: cluster_id}``); ``cluster_id``
    is the legacy interoception alias, folded in when the set has no
    interoception entry. Credit is ROUTED by the reward's source:

    * drive-relief (``drive_potential_diff``) and generic tool-success →
      the **interoception** cluster ONLY — never an exteroceptive cluster
      (the write-side complement of de-dilution; probe 3 showed the uniform
      tool-success floor drowns any direction signal it leaks onto).
    * operant/direction (``set_pending_operant_action`` →
      ``credit_operant_reward``) → the **exteroceptive** cluster (audio
      when present) — a caregiver's contingent reward is conditioned on
      WHERE the stimulus is, so the pending action is keyed on the
      direction-bearing cluster.

    Malformed clusters (empty tag/id) raise ``ValueError`` here, OUTSIDE
    the fail-soft NAc block — a degenerate key must be loud, not a
    silently-swallowed no-op.
    """
    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError(
            f"agent_id must be a non-empty string, got {agent_id!r}. "
            "Tool outcome recording is per-agent — empty / missing values "
            "would silently merge attribution across agents."
        )
    nac = _learning_nac(nac, refused)
    ts = time.time()
    recent_outcomes.append(
        {
            "tool": tool_name,
            "success": success,
            "result": result_summary,
            "error": error,
            "timestamp": ts,
        }
    )
    if len(recent_outcomes) > max_recent:
        recent_outcomes.pop(0)

    if llm_worker is not None:
        llm_worker.record_outcome(
            tool_name=tool_name,
            reasoning=reasoning,
            success=success,
            result_summary=(result_summary or "")[:200],
        )

    context_pool.add_outcome(
        tool_name=tool_name,
        success=success,
        result_summary=result_summary,
        error=error,
    )

    # For NAc / cluster / goal LEARNING, an action that harmed the body is a
    # NEGATIVE outcome even if it mechanically "succeeded" — the harm rides in
    # ``ToolOutput.side_effects["embodiment_failures"]`` (e.g. the deceptive
    # hearth's warm_self raises arms.thermal past its comfort band). Without
    # this, a harmful-but-mechanically-successful affordance books a POSITIVE
    # causal link that competes with the ToolPainBridge's direct NEGATIVE
    # attribution and prevents the substrate from learning to avoid it
    # (substrate_primary_cradle_readiness.md B5). The bridge still owns the
    # primary negative attribution; recommend_action's get_negative_outcomes
    # takes the MAX over negative links, so the two paths don't compound
    # harmfully — and flipping the valence here also closes the gap when no
    # ToolPainBridge is wired. The LLM-facing sinks above keep mechanical
    # ``success`` (the result_summary carries the failure detail).
    #
    # **This is a THREE-tier outcome, not a boolean** (D53, 2026-08-31).
    # ``Valence`` is a live ternary and ``causal_link.py::_VALENCE_TO_REWARD``
    # maps it canonically — POSITIVE 1.0 / NEGATIVE 0.0 / NEUTRAL 0.5, the
    # neutral value being the Rescorla-Wagner prior midpoint consumed by
    # ``CausalLink.update_prediction_rw`` and by Welford online variance.
    # An action that RAN but accomplished nothing — a motion clamped at a
    # joint limit, a turn that could not be verified to have reached its
    # target — is exactly "expected outcome, no strong signal": it should
    # move the predictor toward the prior rather than assert either success
    # or harm. Until this was fixed, ``record_outcome`` collapsed the
    # ternary into a boolean in three places, so a refused motion booked a
    # full POSITIVE causal link plus +1.0 goal and cluster credit.
    #
    # Why NEUTRAL and not "route it through embodiment_failures": a clamp is
    # a REFUSAL, not harm. Booking NEGATIVE for a clamped turn that
    # nonetheless centred the sound would invert the bug rather than fix it.
    # The tier keys on the OUTCOME (did anything change) — never on
    # clamp-occurrence.
    #
    # The load-bearing consequence is on the causal surface:
    # ``get_positive_outcomes`` / ``get_negative_outcomes`` both filter on
    # exact valence, so a NEUTRAL link falls into NEITHER and contributes
    # zero to ``recommend_action``'s causal component. Previously every tool
    # that had ever mechanically succeeded carried a flat causal term into
    # action selection — a link's confidence, 0.50 on creation and 0.64+
    # once re-observed, so above the 0.3 ``min_confidence`` gate from the
    # very first observation — and no credit-withholding flag could
    # suppress it, because they all gate the CLUSTER term only.
    # Harm and mechanical failure DOMINATE a tool's self-report: a tool
    # cannot talk its way out of having broken the body. Otherwise the
    # tool's own report wins, because it is the only party that measured
    # what happened; absent a report, mechanical success means POSITIVE
    # (the historical contract every existing tool relies on).
    if embodiment_failed or not success:
        learn_valence = _V.NEGATIVE
    elif outcome_valence is not None:
        learn_valence = outcome_valence
    else:
        learn_valence = _V.POSITIVE
    learn_success = learn_valence is _V.POSITIVE

    # Validate + fold the legacy scalar AFTER the always-on sinks (a
    # malformed set must not lose the outcome record) but BEFORE the
    # fail-soft NAc block, so a degenerate key raises loudly instead of
    # vanishing into logger.debug (pre-merge review: Executor lens flagged
    # the raise-before-sinks ordering; Architecture lens confirmed the
    # loud-guard placement).
    from maxim.decisions.nac import INTEROCEPTION_MODALITY, fold_legacy_cluster_id

    active_clusters = fold_legacy_cluster_id(clusters, cluster_id)
    intero_cluster = active_clusters.get(INTEROCEPTION_MODALITY)
    # Operant credit target: the direction-bearing exteroceptive cluster.
    # Prefer AUDIO_TAG (the shipped exteroceptive channel), else the first
    # non-interoception entry (deterministic: sorted by tag — which cluster
    # the caregiver's reward conditions on under MULTIPLE extero channels is
    # a deferred binding/attention question, see the seam plan), else fall
    # back to interoception (single-cluster bodies — pre-seam behavior; the
    # fallback ALSO captures a transient extero-encode failure upstream,
    # which the encode loop surfaces with its own WARNING).
    from maxim.embodiment.sensory_streams import AUDIO_TAG

    _extero_tags = sorted(t for t in active_clusters if t != INTEROCEPTION_MODALITY)
    operant_cluster = active_clusters.get(AUDIO_TAG) or (
        active_clusters[_extero_tags[0]] if _extero_tags else intero_cluster
    )

    # NAc causal learning: record tool → outcome so predictions improve
    if nac is not None:
        try:
            outcome_summary = (result_summary or error or "")[:50]
            valence = learn_valence
            sig = build_tool_signature(tool_name, tool_params)
            # Tag every NAc observation with agent_id so cross-agent
            # attribution gaps surface as filterable context, not
            # silently merged links.
            ctx: dict[str, Any] = {"agent_id": agent_id}
            if reasoning:
                ctx["goal"] = reasoning[:100]
            _observe_tool_result(nac, sig, learn_valence, valence, outcome_summary, elapsed_s, ctx, bridge_booked)
            # Goal-level credit: if deliberation was active under a goal,
            # credit/penalize that goal so ThoughtGate learns whether
            # deliberation under this goal type produces good outcomes.
            # A NEUTRAL outcome books NOTHING here rather than 0.0. The two
            # are behaviourally identical — ``credit_goal`` accumulates
            # ``current + alpha * reward``, so 0.0 is an exact no-op — but
            # skipping avoids materialising a phantom 0.0 entry for a goal
            # that has no evidence either way.
            if active_goal is not None and learn_valence is not _V.NEUTRAL:
                reward = 1.0 if learn_success else -1.0
                nac.credit_goal(active_goal, reward)

            # G4 closure: cluster-keyed reward bias for substrate-primary
            # action selection. When the proposer captured an EC
            # interoception cluster id at proposal time (only fires from
            # propose_via_substrate today; LLM-primary proposals leave
            # ``cluster_id`` as None), credit the ``(agent, cluster, tool)``
            # triple. ``update_cluster_reward`` is a no-op when cluster_id is
            # None/empty, so this is safe to call unconditionally. See:
            # docs/plans/grounded_language_acquisition.md § Phase 0 G4.
            #
            # Reward magnitude (orient credit path): prefer the drive-comfort
            # ``drive_potential_diff`` from the affordance's side_effects when
            # present — that is the STATE-CONDITIONED signal (turn TOWARD the
            # sound moved azimuth toward center -> positive; away -> negative;
            # warm/feed moved cold/hunger toward comfort -> positive), which the
            # tool-EXECUTION-success signal cannot express (both turns / all warms
            # "succeed"). **Take its SIGN, not its magnitude**: the value is graded
            # progress toward comfort, but a small magnitude (e.g. one warm step
            # ~0.15-0.3, or an azimuth step 0.09) would lose the argmax to the flat
            # +1 non-drive actions get — the #405 Exp-42 floor. Signing to ±1 puts
            # drive-relief actions on the same scale as tool-success while keeping
            # the direction. Exactly-0 net progress books NOTHING (D53): the
            # signing argument requires zero not be rounded UP, and the measured
            # path in tool_bridge already treats measured-zero that way. The
            # producer (tool_bridge) sets it to None when the action touched no
            # drive sensor OR caused COLLATERAL harm (a failure on a sensor its
            # progress didn't account for), so harm-dominates lives there and we
            # fall back to ±1 (=-1 under embodiment_failed). See tool_side_effects.md.
            if active_clusters:
                # Operant-only mode (cradle_mother): when the learner's action
                # value must come SOLELY from a caregiver's contingent reward
                # (mother feeds the infant *because* it oriented), the intrinsic
                # tool-success floor is poison — probe 3's ``tool_floor`` arm
                # showed the uniform +1 saturates both directions to the cluster
                # cap and drowns the operant signal (all arms → chance). So in
                # this mode we (a) remember the action for the mother's later
                # ``credit_operant_reward``, and (b) book a cluster reward ONLY
                # when a REAL drive signal is present — never the tool-success
                # fallback. A driveless turn accrues no cluster bias; the mother
                # is the sole teacher.
                operant_only = _operant_only_credit_enabled()
                if operant_only and operant_cluster:
                    # Remember this action so the mother's later
                    # ``credit_operant_reward`` can reinforce it. Keyed on the
                    # DIRECTION-BEARING cluster (audio when present): the
                    # caregiver's contingency is "you turned toward me", so the
                    # credited (cluster, tool) pair must condition on where the
                    # stimulus was, not on the interoceptive state (seam
                    # routing: operant/direction → exteroceptive cluster).
                    try:
                        nac.set_pending_operant_action(
                            agent_id=agent_id, cluster_id=operant_cluster, tool_signature=sig
                        )
                    except Exception:
                        logger.debug("set_pending_operant_action raised", exc_info=True)
                # abs(...) > epsilon, NOT `!= 0.0`: drive_comfort_progress is a
                # difference of floats, so a genuine zero-progress move (e.g. a
                # mirror move across a nonzero set_point) can leave a ~1e-17
                # residue that exact-equality would mis-credit as ±1. The
                # exactly-0 boundary is load-bearing, so guard it with an
                # epsilon rather than float identity — the residue must land
                # in the NEUTRAL branch below, not be signed into ±1.
                # Credit target: interoception by default. MEASURED
                # exteroceptive relief (Phase 2, sem_motor_binding.md —
                # producer marks drive_relief_channel="exteroceptive")
                # routes to the direction-bearing operant cluster instead:
                # it is source-attributable (conditioned on where the sound
                # was, exactly like the caregiver's operant credit — the
                # probe-3 carve-out), and it is the surface the trained
                # orient policy keys on, so live experience COMPOUNDS the
                # imported biases instead of coexisting beside them. The
                # tool-success floor NEVER routes extero (probe-3 rule).
                credit_cluster = intero_cluster
                # S1 credit provenance: the branch below ALREADY distinguishes
                # why this reward exists — record it so the prompt annotation
                # can say "relieved cold" rather than a bare band label
                # (annotation_context_and_provenance.md, pilot finding F3).
                credit_source: str | None = None
                if drive_potential_diff is not None and abs(drive_potential_diff) > 1e-9:
                    cluster_reward: float | None = 1.0 if drive_potential_diff > 0.0 else -1.0
                    credit_source = "drive_relief"
                    if drive_relief_channel == "exteroceptive":
                        credit_cluster = operant_cluster
                        credit_source = "orient_relief"

                elif drive_potential_diff is not None and learn_valence is not _V.NEGATIVE:
                    # MEASURED exactly-zero net progress, on an action that
                    # did not otherwise fail. (The NEGATIVE guard matters: a
                    # tool that FAILED still books -1 even when its drive
                    # measured nothing — the drive said nothing, but the tool
                    # itself failed, and that is a real negative outcome.)
                    # The drive spec ran
                    # and reported that nothing changed — a turn into the
                    # azimuth wall, a warm on an already-warm body. That is
                    # information, and it is precisely neutral.
                    #
                    # This branch used to fall through to the tool-success
                    # floor below and book +1. Two things in-tree already
                    # contradicted that: ``tool_bridge``'s own MEASURED path
                    # sets ``drive_credit_withheld`` for this same event
                    # ("an honest 'no change'"), and
                    # ``cradle_mother::reactive_mother_tick`` — which an
                    # EARNED result (Exp 52's satiated control arm) depends
                    # on — mints nothing when ``abs(relief) <= 1e-9``. The
                    # modeled path was the odd one out.
                    #
                    # NOTE the epsilon guard lives in the branch ABOVE: a
                    # float difference across a nonzero set_point can leave a
                    # ~1e-17 residue, which lands here rather than being
                    # mis-credited as ±1.
                    cluster_reward = None
                elif operant_only or drive_relief_only or drive_credit_withheld:
                    # drive_credit_withheld (sem_motor_binding.md Phase 1):
                    # a motor-bound LIVE affordance touched a drive sensor a
                    # measurement stream owns — modeled credit is filtered
                    # and measured credit hasn't shipped (Phase 2). The
                    # flat +1 floor here would mint direction-blind cluster
                    # credit for real turns in a silent room (the probe-3
                    # floor-drowning failure, one cluster over).
                    # NO tool-success floor. Two callers need this:
                    # - operant_only (cradle_mother): the mother is the sole teacher.
                    # - drive_relief_only (llm-primary / imagination, Phase 1 of
                    #   substrate_learns_from_experience.md): the LLM issues a BROAD
                    #   always-succeed action stream (say/sense/examine), so the
                    #   uniform +1 floor would flood the interoception cluster with
                    #   "this tool ran" and drown the real drive-relief differential
                    #   (the credit_on_progress hazard, amplified). The substrate
                    #   learns from the body's real drive signal ONLY, never from
                    #   tool execution. A driveless action accrues no cluster bias.
                    cluster_reward = None
                elif learn_valence is _V.NEUTRAL:
                    # The tool ran and accomplished nothing attributable
                    # (a clamped motion, an unverifiable one). No floor:
                    # asserting +1 for an action the tool itself reports
                    # did not happen is the D53 defect. Booking 0.0 would
                    # be an exact no-op anyway
                    # (``current + alpha * 0.0``), so skipping is
                    # equivalent for the bias and additionally avoids
                    # promoting the triple's credit-source to "mixed".
                    cluster_reward = None
                else:
                    cluster_reward = 1.0 if learn_success else -1.0
                    credit_source = "tool_success"
                # Seam routing: drive-relief AND generic tool-success write the
                # INTEROCEPTION cluster only — never an exteroceptive cluster.
                # Direction-bearing clusters (audio) are credited exclusively
                # by source-attributable signals (the caregiver's
                # credit_operant_reward via the pending action above); letting
                # the uniform tool-success floor leak onto them would re-drown
                # the direction signal on the write side (probe 3).
                if cluster_reward is not None and not credit_cluster:
                    # Chosen slot empty (a designed extero-only body, or
                    # the interoception encode failed this tick) — the
                    # reward is dropped by design, but not silently. Note
                    # extero routing rarely lands here: operant_cluster
                    # falls back to intero when only AUDIO_TAG is missing.
                    logger.debug(
                        "cluster reward %+.1f for %s dropped: no credit cluster in %r",
                        cluster_reward,
                        sig,
                        sorted(active_clusters),
                    )
                if cluster_reward is not None and credit_cluster:
                    try:
                        nac.update_cluster_reward(
                            agent_id=agent_id,
                            cluster_id=credit_cluster,
                            tool_signature=sig,
                            reward=cluster_reward,
                            source=credit_source,
                        )
                    except Exception:
                        # Mirrors the surrounding error policy — cluster
                        # learning is best-effort; an exception here must
                        # not crash the agent loop.
                        logger.debug("update_cluster_reward raised", exc_info=True)

        except Exception as e:
            logger.warning("NAc reward signal failed for tool %s: %s", tool_name, e)

    # Energy → NAc: learn which tools are expensive (metabolic budget)
    if nac is not None and elapsed_s > 0:
        _observe_energy_cost(nac, tool_name, elapsed_s, agent_id)


def _learning_nac(nac: Any | None, refused: ToolErrorKind | None) -> Any | None:
    """The NAc ``record_outcome`` may book to: None for a gate refusal (#1315 item 3), which no person made and which
    teaches nothing about the action; a kind that is not a gate kind is a caller's mistake, refused loudly."""
    if refused is None:
        return nac
    if refused not in GATE_REFUSAL_KINDS:
        raise ValueError(f"record_outcome(refused=) takes a gate refusal kind or None, got {refused!r}")
    return None


def _observe_energy_cost(nac: Any, tool_name: str, elapsed_s: float, agent_id: str) -> None:
    """Book a tool's metabolic cost to NAc (moved out of ``record_outcome``, #1315 item 3)."""
    try:
        # Expensive actions (>2s) get NEGATIVE energy valence; cheap ones NEUTRAL
        energy_valence = _V.NEGATIVE if elapsed_s > 2.0 else _V.NEUTRAL
        nac.observe(
            event_type="energy",
            event_signature=f"cost:{tool_name}",
            outcome_type="energy_cost",
            outcome_signature=f"elapsed:{elapsed_s:.1f}s",
            outcome_valence=energy_valence,
            delta_seconds=elapsed_s,
            context={"agent_id": agent_id, "tool": tool_name},
        )
    except Exception:
        log_swallowed_exception()


def _parallel_entry(tool_name: str, parallel_action: dict[str, Any], result: Any) -> dict[str, Any]:
    """One executed parallel action's result entry, carrying its learning fields (moved out of
    ``execute_parallel_actions``, #1315 item 3)."""
    success = getattr(result, "success", True)
    output = getattr(result, "output", None)
    error = getattr(result, "error", None)
    # D53 review fold: this path discarded side_effects entirely, so
    # the learning tier could not reach record_outcome below and a
    # clamped motion dispatched in a parallel batch still booked
    # POSITIVE. The pre-existing NOTE about drive_credit_withheld
    # being "covered today because llm-primary sets
    # drive_relief_only" does NOT extend to the tier: learn_valence
    # is computed above and independently of the cluster block that
    # drive_relief_only gates.
    _side = read_learning_side_effects(result)

    return {
        "tool": tool_name,
        "params": parallel_action.get("params", {}),
        "success": success,
        "result": str(output)[:2000] if output else None,
        "error": error,
        "_embodiment_failed": _side.embodiment_failed,
        "_outcome_valence": _side.outcome_valence,
        "_drive_potential_diff": _side.drive_potential_diff,
        "_drive_credit_withheld": _side.drive_credit_withheld,
        "_drive_relief_channel": _side.drive_relief_channel,
        "_failure_booked": getattr(result, "nac_failure_booked", None),
        "_refused": gate_refusal_of(result),  # executed branch only (#1315 item 3)
    }


#: ``operational_mode`` for the parallel batch, which is handed no loop state: the executor's grant or mode source
#: (the loop always sets one), else ``DEFAULT_RUN_MODE``, exactly as ``operational_mode`` documents for no state.
_NO_LOOP_STATE = SimpleNamespace(data={})


def _audit_parallel_action(
    autonomy_controller: Any, executor: Any, action: dict[str, Any], result: Any, reasoning: str, confidence: float
) -> None:
    """Write the executed parallel action's autonomy audit entry (#1315 item 3, §1A.2): the batch wrote none, so
    the audit trail missed a model that kept naming a refused tool in a batch. (The ``max_consecutive_failures``
    violation it feeds is severity ``error``, which no approval blocker acts on today: #1549.)"""
    success = getattr(result, "success", True)
    autonomy_controller.log_action(
        action_type="executed",
        action=action,
        reasoning=reasoning,
        mode=operational_mode(executor, _NO_LOOP_STATE),
        confidence=confidence,
        outcome="success" if success else "failure",
        error=getattr(result, "error", None),
    )


def execute_parallel_actions(
    *,
    agent_id: str,
    actions: list[dict[str, Any]],
    executor: Any,
    autonomy_controller: Any,
    confidence: float,
    reasoning: str,
    recent_outcomes: list[dict[str, Any]],
    max_recent: int,
    llm_worker: Any | None,
    context_pool: Any,
    nac: Any | None = None,
    active_goal: str | None = None,
    cluster_id: str | None = None,
    clusters: dict[str, str] | None = None,
    drive_relief_only: bool = False,
    proposal: Any = None,
) -> tuple[list[dict[str, Any]], str]:
    """Execute a batch of parallel actions with autonomy gating.

    ``proposal`` (#1481): the proposal whose batch this is. Once the batch has returned, the inbound messages its prompt
    showed whole are acknowledged (``_ack_inbound``), once per batch; a batch that raised acknowledges nothing, and the
    default None acknowledges nothing, so every other caller is unchanged.

    Returns a tuple of (parallel_results, combined_results_text).
    Each result dict has keys: tool, success, result, error, params.

    ``agent_id`` is required (keyword-only) — every per-action
    ``record_outcome`` below tags NAc with this id so multi-agent
    attribution stays per-agent.

    ``active_goal`` is forwarded to per-action ``record_outcome``
    so ThoughtGate goal-credit applies inside the parallel batch.
    Pre-fix the parameter was missing from this signature even though
    the agent-loop call site already passed ``active_goal=`` — any
    parallel-actions batch would have raised TypeError.

    ``cluster_id`` / ``clusters`` (extero/intero seam) are likewise
    forwarded to per-action ``record_outcome`` so the batch's credit
    routing matches single-action dispatch. The seam's pre-merge
    architecture review caught the SECOND recurrence of the
    missing-parameter bug class on this exact signature (``clusters=``
    passed by the agent-loop call site before this parameter existed);
    ``tests/unit/test_modality_seam.py::TestParallelDispatchSignatureContract``
    pins that every kwarg the agent-loop batch site passes is accepted
    here, so a third recurrence fails in CI, not at runtime.
    """
    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError(f"agent_id must be a non-empty string, got {agent_id!r}.")
    parallel_results: list[dict[str, Any]] = []
    all_succeeded = True

    logger.info("Executing %d parallel actions for batched exploration", len(actions))
    log_agentic(
        "agent_loop",
        "parallel_batch_start",
        {"count": len(actions), "tools": [a.get("tool_name") for a in actions]},
    )

    for idx, parallel_action in enumerate(actions):
        tool_name = parallel_action.get("tool_name", "unknown")
        # One result per action, carrying the stamp whichever side of the append a raise lands (#1221 2b).
        recorded = len(parallel_results)
        result: Any = None
        try:
            # Check autonomy for each action
            can_exec, reason = autonomy_controller.can_execute_action(parallel_action, confidence=confidence)
            if not can_exec:
                logger.warning("Parallel action %s rejected: %s", tool_name, reason)
                parallel_results.append(
                    {
                        "tool": tool_name,
                        "success": False,
                        "error": f"Rejected: {reason}",
                        "result": None,
                    }
                )
                continue

            # Execute the action
            result = executor.execute(parallel_action)
            success = getattr(result, "success", True)
            parallel_results.append(_parallel_entry(tool_name, parallel_action, result))

            if not success:
                all_succeeded = False
            _audit_parallel_action(autonomy_controller, executor, parallel_action, result, reasoning, confidence)

            log_agentic(
                "agent_loop",
                "parallel_action_complete",
                {"tool": tool_name, "index": idx, "success": success},
            )

        except Exception as e:
            logger.error("Parallel action %s failed: %s", tool_name, e)
            if len(parallel_results) > recorded:
                continue  # its outcome is recorded, carrying the failure stamp when the bridge booked one
            parallel_results.append(
                {
                    "tool": tool_name,
                    "success": False,
                    "error": str(e),
                    "result": None,
                    "_failure_booked": getattr(result, "nac_failure_booked", None),
                }
            )
            all_succeeded = False

    # Record individual outcomes so LLM has structured history
    for pr in parallel_results:
        record_outcome(
            agent_id=agent_id,
            tool_name=pr["tool"],
            success=pr["success"],
            result_summary=pr.get("result"),
            error=pr.get("error"),
            reasoning=reasoning,
            recent_outcomes=recent_outcomes,
            max_recent=max_recent,
            llm_worker=llm_worker,
            context_pool=context_pool,
            nac=nac,
            active_goal=active_goal,
            tool_params=pr.get("params"),
            embodiment_failed=bool(pr.get("_embodiment_failed")),
            outcome_valence=pr.get("_outcome_valence"),
            # #1133 (D6): the three drive fields the single-action path always read, through the same
            # parser; a batched action now carries its drive relief and its withheld credit too.
            drive_potential_diff=pr.get("_drive_potential_diff"),
            drive_credit_withheld=bool(pr.get("_drive_credit_withheld")),
            drive_relief_channel=pr.get("_drive_relief_channel"),
            # #1200 Stage 2: the signature the bridge booked a failure under (rejected / raised entries: None).
            bridge_booked=pr.get("_failure_booked"),
            # #1315 item 3: a gate refusal books no NAc. The autonomy rejection above stays a booked negative
            # (pre-existing, inconsistent with #1185; #1550).
            refused=pr.get("_refused"),
            cluster_id=cluster_id,
            clusters=clusters,
            # Phase 1 guardrail must reach the BATCH path too: without this, an
            # llm-primary parallel action stream (populated clusters, no
            # drive_potential_diff) would fall to the tool-success floor and flood
            # the interoception cluster — the exact flooding the guard prevents on
            # the single-action path (two-lens review, both lenses CONFIRMED).
            drive_relief_only=drive_relief_only,
        )

    log_agentic(
        "agent_loop",
        "parallel_batch_complete",
        {"count": len(parallel_results), "all_succeeded": all_succeeded},
    )

    # Build combined result text for LLM context
    combined_parts = ["=== BATCHED EXPLORATION RESULTS ==="]
    for pr in parallel_results:
        tool = pr["tool"]
        if pr["success"]:
            result_text = pr["result"] or "[no output]"
            combined_parts.append(f"\n[{tool}] SUCCESS:\n{result_text}")
        else:
            combined_parts.append(f"\n[{tool}] FAILED: {pr.get('error', 'unknown error')}")
    combined_parts.append("\n=== END BATCHED RESULTS ===")
    combined_results = "\n".join(combined_parts)

    # The learning-tier fields are internal to the credit loop above; the
    # returned dicts have a documented shape (tool, success, result, error,
    # params) that callers and the LLM-facing history rely on.
    for pr in parallel_results:
        for key in [k for k in pr if k.startswith("_")]:
            del pr[key]

    _ack_inbound(proposal)
    return parallel_results, combined_results


def _reset_deliberation(executor: Any) -> None:
    """Reset ThinkTool deliberation state when a non-think action fires (L2).

    Single call site for both dispatch paths — prevents drift.
    """
    try:
        _registry = getattr(executor, "registry", None)
        if _registry is not None:
            _think_tool = _registry.get("think")
            if hasattr(_think_tool, "reset_deliberation"):
                _think_tool.reset_deliberation()
    except (KeyError, Exception):
        pass  # think tool not registered or not a ThinkTool


def _followup_result_text(tool_name: str, output: Any, result: Any, limit: int) -> str | None:
    """The RAW text a tool result contributes (the follow-up AND ``result_summary``).

    Deliberately unframed: ``result_summary`` feeds ``record_outcome``, whose NAc outcome signature
    is its first 50 characters -- a frame header there would collapse every outcome of a tool into
    one causal link (#823 review). Framing happens in ``_followup_synthetic_input``.
    """
    if output is not None:
        if tool_name == "internet_search" and isinstance(output, list):
            parts = []
            for i, item in enumerate(output[:10], 1):  # Limit to 10 results
                if isinstance(item, dict):
                    parts.append(
                        f"[{i}] {item.get('title', '')}\n    URL: {item.get('url', '')}\n    {item.get('snippet', '')}"
                    )
            text = "\n\n".join(parts)[:limit]
        else:
            text = str(output)[:limit]
        # For empty results, include metadata message if available
        if not output and hasattr(result, "metadata"):
            msg = result.metadata.get("message", "")
            if msg:
                text = f"[No results: {msg}]"
        return text
    # When output is None but the tool returned an error, include the error text so follow-up
    # re-thinks can see WHY it failed.
    error_msg = getattr(result, "error", None) if result else None
    if error_msg:
        return f"[ERROR: {str(error_msg)[:limit]}]"
    return None


def book_refusal(
    *,
    rec_outcome: Callable[..., Any],
    agent_id: str,
    recent_outcomes: list[dict[str, Any]],
    max_recent: int,
    llm_worker: Any,
    context_pool: Any,
    nac: Any,
    state: Any,
    source: Any,
    tool_name: str,
    error: str,
    reasoning: str,
) -> None:
    """Book an action a person REFUSED (a confirmation or a plan answered "no"), as the loop books a hard
    rejection (#1133, D4).

    Through the run's recorder (``drive_relief_only`` bound) under the hub's ``agent_id`` (the controller's own
    ``agent_name`` drifted from it), credited to the situation the action was PROPOSED in (``source``, the
    ``LLMProposal``: its ``cluster_id`` / ``clusters``). The per-run half is bound once by
    ``loop_setup.build_loop_run`` as ``LoopRun.book_refusal``; callers pass ``source``, ``tool_name``,
    ``error`` and ``reasoning``. Replaces the retired ``LoopController.record_outcome``.
    """
    rec_outcome(
        agent_id=agent_id,
        tool_name=tool_name,
        success=False,
        result_summary=None,
        error=error,
        reasoning=reasoning,
        recent_outcomes=recent_outcomes,
        max_recent=max_recent,
        llm_worker=llm_worker,
        context_pool=context_pool,
        nac=nac,
        active_goal=state.data.get("active_goal") if hasattr(state, "data") else None,
        cluster_id=getattr(source, "cluster_id", None),
        clusters=getattr(source, "clusters", None),
        bridge_booked=None,  # refused: the tool never ran (#1200 Stage 2)
        refused=None,  # a person's "no" still books NAc (#1185); only a gate's refusal does not (#1315 item 3)
    )


def book_machine_refusal(
    *,
    rec_outcome: Callable[..., Any],
    agent_id: str,
    recent_outcomes: list[dict[str, Any]],
    max_recent: int,
    llm_worker: Any,
    context_pool: Any,
    tool_name: str,
    error: str,
    reasoning: str,
) -> None:
    """Book an approved action the MACHINE refused (blocked at drain time, or a drain aborted by a raise; #1085).

    Only a human "no" books NAc (``book_refusal``; owner re-decision 2026-10-08, #1185): a refusal that no person
    made teaches nothing about the action or its situation, so it is booked to the outcome window, the LLM's
    reasoning carryover and the context pool only. There is no ``nac``, no situation and no goal to pass, by
    signature: the recorder gets ``nac=None`` (no causal observation, no cluster reward), no cluster and
    ``active_goal=None`` (no goal credit). Scope: the PLANNING drain's refusals (``loop_planning``); §4's hard
    rejection and a confirmation "no" still book NAc (#1185). Bound once per run by ``loop_setup.build_loop_run``
    as ``LoopRun.book_machine_refusal``.
    """
    rec_outcome(
        agent_id=agent_id,
        tool_name=tool_name,
        success=False,
        result_summary=None,
        error=error,
        reasoning=reasoning,
        recent_outcomes=recent_outcomes,
        max_recent=max_recent,
        llm_worker=llm_worker,
        context_pool=context_pool,
        nac=None,
        active_goal=None,
        cluster_id=None,
        clusters=None,
        bridge_booked=None,  # refused: the tool never ran (#1200 Stage 2)
        refused=None,  # no gate refused it: nac=None above already books no NAc
    )


#: How many rejection follow-ups one user turn gets before the turn ends with the refusal (#1413; owner decision
#: 2026-10-22): the model may re-plan twice; a third refusal in the same turn ends it.
MAX_REJECTION_FOLLOWUPS = 2


def book_hard_rejection(
    *,
    ctrl: Any,
    autonomy_controller: Any,
    rec_outcome: Callable[..., Any],
    agent_id: str,
    nac: Any,
    aut_mode: str,
    proposal: Any,
    action: dict[str, Any],
    reason: str,
    confidence: float,
    active_goal: Any,
) -> ActionFollowup | None:
    """§4's hard rejection: book it, then feed it back to the model or end the turn (#1413; moved from
    ``run_agentic_loop``).

    The audit entry and the outcome booking are the inline branch moved verbatim (a hard rejection still books NAc,
    #1185). Before #1413 that was all it did: no follow-up was scheduled, so the model got no next call, and in
    Console Talk the turn idled to its 180 s timeout with "(no reply — the turn timed out before finishing)".

    Now it returns a ``refused`` follow-up, so the model is told, outside the tool-output frame, that the tool was
    refused and did not run, and re-plans with an allowed tool or replies. At most :data:`MAX_REJECTION_FOLLOWUPS`
    CONSECUTIVE ones: the count starts again with each fresh user input (``LoopController.note_user_input``,
    called where the loop settles a tick's input, whatever its channel) and
    whenever the previous outcome was not a refusal (an action ran in between), so a long autonomous run with no
    user turns is not capped once for its whole life. Beyond that it returns ``None`` and records a BLOCKED action
    through the run's executor (``InstrumentedExecutor.record_block``: ``block_reason`` the refusal, and the
    agent/session telemetry), which settles a Talk bridge at once and names the tool in the chat
    (``console.handle._describe_silent_turn``); without an action sink there is no record.
    The blocked record is written only at the cap, never alongside a follow-up: the Talk bridge ends a turn 2 s
    after its last action, so a record before the re-plan's reply would end the turn first. Every loop with a model
    to re-plan (owner decision 2026-10-22); ``aut_mode="substrate-primary"`` has none (its LLM submission is off,
    so a follow-up would never be consumed and would keep the idle gate awake), and keeps the booking only, as
    before, with no record that would shift that arm's action counts. The caller assigns the follow-up.
    """
    autonomy_controller.log_action(
        action_type="rejected",
        action=action,
        reasoning=f"Rejected: {reason}",
        mode=operational_mode(ctrl.executor, ctrl.state),  # the one reader (#963), on the run's own handles
        confidence=confidence,
    )
    # A refusal chain is broken by anything else: the previous outcome, before this one is booked.
    previous = ctrl.recent_outcomes[-1] if ctrl.recent_outcomes else None
    if not str((previous or {}).get("error") or "").startswith("Rejected by autonomy"):
        ctrl.rejection_followups = 0
    # Record rejection so LLM knows not to re-propose
    rejection_msg = f"Rejected by autonomy: {reason}"
    tool_name = action.get("tool_name", "unknown")
    rec_outcome(
        agent_id=agent_id,
        tool_name=tool_name,
        success=False,
        result_summary=None,
        error=rejection_msg,
        reasoning=proposal.reasoning or "",
        recent_outcomes=ctrl.recent_outcomes,
        max_recent=ctrl.max_recent_outcomes,
        llm_worker=ctrl.llm_worker,
        context_pool=ctrl.context_pool,
        nac=nac,
        active_goal=active_goal,
        cluster_id=proposal.cluster_id,
        clusters=proposal.clusters,
        bridge_booked=None,  # the tool never ran: nothing was bridge-booked
        refused=None,  # the autonomy level, not a dispatch gate: a hard rejection still books NAc (#1185)
    )
    logger.info("Hard rejection recorded for LLM: %s", rejection_msg)

    if aut_mode == "substrate-primary":
        return None
    if ctrl.rejection_followups < MAX_REJECTION_FOLLOWUPS:
        ctrl.rejection_followups += 1
        return ActionFollowup(
            tool=tool_name,
            result=rejection_msg,
            original_query=str(getattr(proposal, "triggering_input", "") or ctrl.rejection_turn or ""),
            followup_type="refused",
            mode=operational_mode(ctrl.executor, ctrl.state),
            timestamp=time.time(),
        )
    record_block = getattr(ctrl.executor, "record_block", None)  # the run's InstrumentedExecutor, given a sink
    if callable(record_block):
        record_block(tool_name, rejection_msg, dict(action.get("params") or {}))
    logger.info("Rejection cap reached (%d) for this turn: ending it on %s", MAX_REJECTION_FOLLOWUPS, tool_name)
    return None


@dataclasses.dataclass(frozen=True)
class ExecutionOutcome:
    """What one ``execute_and_learn`` call did, for its caller to display or route.

    ``raised`` is True when the dispatch raised (the ``except`` branch booked the failure); ``error`` is
    then the exception's text, otherwise the tool's own ``error``. ``followup`` is the follow-up LLM
    cycle the tool asks for, or ``None``: the function never writes the controller, so the caller
    assigns it.

    Runtime-ephemeral: built and consumed within one loop tick, never persisted and never crossing a
    wire, so CC3 forward-compat is out of scope.
    """

    success: bool
    raised: bool
    result: Any
    output: Any
    error: str | None
    result_str: str | None
    followup: ActionFollowup | None


def _tool_followup(
    executor: Any, tool_name: str, result_str: str | None, proposal: Any, followup_type: str, mode: str
) -> ActionFollowup:
    """The follow-up a completed tool queues, stamped with whether the tool it RAN declares untrusted output
    (``Tool.untrusted_output``, Session R): resolved through the executor's one alias resolver, so a ``reflect``
    that ran as ``think`` carries ``think``'s declaration."""
    resolve = getattr(executor, "resolve_tool_name", None)
    registry = getattr(executor, "registry", None)
    ran = resolve(tool_name) if callable(resolve) else tool_name
    tool = registry.find(ran) if registry is not None and hasattr(registry, "find") else None
    return ActionFollowup(
        tool=tool_name,
        result=result_str,
        original_query=getattr(proposal, "triggering_input", ""),
        followup_type=followup_type,
        mode=mode,
        timestamp=time.time(),
        untrusted_output=bool(getattr(tool, "untrusted_output", False)),
    )


def _ack_inbound(proposal: Any) -> None:
    """Acknowledge the inbound comms a dispatched turn's prompt showed whole (#1481). Only the success branch of
    ``LLMWorker._process_request`` sets these fields, so a fallback (and a None proposal) acknowledges nothing; ``ack``
    is idempotent, so a turn that reaches both dispatch paths cannot double-count."""
    feed = getattr(proposal, "inbound_feed", None)
    if feed is not None:
        feed.ack(list(getattr(proposal, "inbound_rendered_ids", ())))


def _record_turn_after_effects(*, action: dict[str, Any], success: bool, proposal: Any, context_pool: Any) -> str:
    """After-effects of a turn whose action was dispatched and RETURNED (the normal path of ``execute_and_learn``; a
    dispatch that raised never reaches here). The seed of #1150's after-effects step (Session A's grant). Returns the
    action's tool name, which the caller reads on."""
    # The inbound comms this turn's prompt showed whole are delivered: the turn was acted on (#1481), even when the
    # tool then failed (``success`` False): the model saw them and acted.
    _ack_inbound(proposal)

    # Track conversation history for response/speak actions
    tool_name = action.get("tool_name", "")
    if tool_name in ("respond", "speak") and success:
        raw_params = action.get("params")
        params = raw_params if isinstance(raw_params, dict) else {}
        response_message = params.get("message") or params.get("text", "")
        triggering_input = getattr(proposal, "triggering_input", "")
        if response_message and triggering_input:
            context_pool.add_conversation_turn(
                user_input=triggering_input,
                assistant_response=response_message,
                tool_used=tool_name,
            )
    return tool_name


def book_agent_fallback_failure(
    e: Exception,
    *,
    rec_outcome: Callable[..., Any],
    agent_id: str,
    action: dict[str, Any],
    result: Any,
    proposal: Any,
    recent_outcomes: list[dict[str, Any]],
    max_recent: int,
    llm_worker: Any,
    context_pool: Any,
    nac: Any,
    state: Any,
) -> None:
    """Book an agent-fallback action whose dispatch raised (moved out of ``run_agentic_loop``'s body, #1315 item 3).

    ``refused`` is ``gate_refusal_of(result)``: None when ``executor.execute`` itself raised (``result`` is still
    None, so the failure books its NAc negative as before), a gate kind only if the gate refused and a later step
    raised. The fallback path's own dispatch is tracked by #1147."""
    log_agentic(
        "agent_loop",
        "error",
        {"context": "agent_fallback_action", "error": str(e)},
        level="ERROR",
    )
    _loop_logger.debug(f"Agent fallback action failed: {e}")

    # Track exception in recent_outcomes for LLM learning
    rec_outcome(
        agent_id=agent_id,
        tool_name=action["tool_name"],
        success=False,
        result_summary=None,
        error=str(e),
        reasoning=getattr(proposal, "reasoning", "") if proposal else "",
        recent_outcomes=recent_outcomes,
        max_recent=max_recent,
        llm_worker=llm_worker,
        context_pool=context_pool,
        nac=nac,
        active_goal=state.data.get("active_goal") if hasattr(state, "data") else None,
        cluster_id=getattr(proposal, "cluster_id", None),
        clusters=getattr(proposal, "clusters", None),
        bridge_booked=getattr(result, "nac_failure_booked", None),
        refused=gate_refusal_of(result),
    )


def _stash_timeout_retry(action: dict[str, Any], sim: Any, state: Any) -> None:
    """Store a succeeded timeout-retry prompt's state for the user's response (moved verbatim out of
    ``execute_and_learn``, #1315 item 3)."""
    # Plan 3.5 R2: fall back to the current agent-level LLM
    # timeout default if the action didn't include _timeout_s.
    # Was hardcoded 60.0 pre-plan (mesh-era value).
    from maxim.agents.llm_worker import DEFAULT_LLM_CALL_TIMEOUT_S

    timeout_s = action.get("_timeout_s", DEFAULT_LLM_CALL_TIMEOUT_S)
    # In sim mode, auto-resolve instead of blocking
    sim_timeout_response = sim.resolve_timeout_retry(timeout_s)
    if sim_timeout_response is not None:
        sim.log("PIPELINE", f"Auto-resolved timeout retry: {sim_timeout_response}")
        state.data["pending_timeout_retry"] = {
            "original_request": action.get("_original_request"),
            "timeout_s": timeout_s,
        }
        state.data["pending_loop_reply"] = sim_timeout_response
    else:
        state.data["pending_timeout_retry"] = {
            "original_request": action.get("_original_request"),
            "timeout_s": timeout_s,
        }


def execute_and_learn(
    *,
    agent: Any,
    agent_name: str,
    agent_id: str,
    executor: Any,
    sim: Any,
    state: Any,
    environment: Any,
    memory: Any,
    hippocampus: Any,
    memory_hub: Any,
    result_cache: Any,
    autonomy_controller: Any,
    rec_outcome: Callable[..., Any],
    recent_outcomes: list[dict[str, Any]],
    max_recent: int,
    llm_worker: Any,
    context_pool: Any,
    nac: Any,
    run_id: str,
    overlay: Mapping[str, Mapping[str, Any]],
    action: dict[str, Any],
    confidence: float,
    proposal: Any,
    observation: Any,
    human_involved: bool,
) -> ExecutionOutcome:
    """Execute one approved action and book everything the agent learns from it (#1133; #1085's core).

    THE dispatch seam, so what an action teaches does not depend on how it was approved. Called today by
    the autonomous path (``run_agentic_loop`` §4), the SUPERVISED-confirmed path
    (``LoopController.handle_confirmation``) and the PLANNING-approved path (``loop_planning.drain_approved``,
    #1085). Tracked exceptions that still dispatch on their own: the parallel batch (``execute_parallel_actions``,
    which reads the same side-effect fields) and the agent-fallback path (#1147). Moved verbatim from §4
    (the autonomous path), in order: the pre-execution snapshot, ``executor.execute``, the tool's learning
    side effects, the ``write_file`` overwrite retry, the timeout-retry prompt, the write cache invalidation,
    the ``tool_called``/``tool_result`` events, the autonomy audit entry, the deliberation reset, the outcome
    credit (``rec_outcome``), the plan outcome, the follow-up, the conversation turn, ``environment.step``,
    ``memory.store_raw``, the Hippocampus capture and ``mark_failure``; an exception books a failed outcome.

    Every parameter is keyword-only with no default, so a caller that forgets one gets a ``TypeError``.
    The first block is per run and is bound ONCE by ``loop_setup.build_loop_run`` as
    ``LoopRun.execute_and_learn`` (a ``functools.partial``): ``rec_outcome`` is the run's recorder (with
    ``drive_relief_only`` bound), ``agent_id`` the hub's, ``memory_hub`` ``None`` when the hub's session
    did not start (no plan outcome then), ``overlay`` the run's own tool descriptions the follow-up type reads
    (``LoopRun.tool_overlay``, #1260). The rest is per action: ``proposal`` is the ``LLMProposal`` the
    action came from (or a sourceless ``autonomy.Proposal``: no situation), carrying the reasoning, citations,
    triggering input and the situation (``clusters``/``cluster_id``/``cluster_margins``) credit and capture key to.
    ``human_involved``: the action was confirmed or approved, by a person OR a policy (the non-interactive
    SUPERVISED auto-yes, sim AUTO_APPROVE); it marks the audit entry and turns the ``write_file`` overwrite
    retry OFF (#1085: what was approved runs exactly). On those paths the two times differ (owner decision D2):
    credit and capture key to PROPOSAL time (``PendingConfirmation.source``; the queued ``Proposal.source``),
    while ``observation`` is the ANSWER-time one the capture stores. Logs on ``maxim.runtime.agent_loop``
    (the ``loop_setup`` precedent). Known defects carried by the move, to be fixed HERE: #1145 (a step after
    the credit raising books a second, negative outcome), #1146 (the overwrite retry's side effects).
    """
    result: Any = None
    success = False
    output: Any = None
    result_str: str | None = None
    queued_followup: ActionFollowup | None = None
    raised: str | None = None
    # Dormant since 2026-10-07 (#1179, runtime/preemption.py): never true, wire_preemption has no caller
    if hasattr(agent, "_execution_tracker") and agent._execution_tracker:
        goal_desc = proposal.reasoning or ""
        robot_handle = getattr(agent, "goal", None)
        robot_handle = getattr(robot_handle, "robot", None) if robot_handle else None
        agent._execution_tracker.capture_before(
            goal_description=goal_desc[:200],
            tool_name=action.get("tool_name", ""),
            tool_params=action.get("params", {}),
            robot=robot_handle,
        )

    # Execute the action
    try:
        exec_start = time.time()
        _loop_logger.info("Starting tool execution: %s", action.get("tool_name"))
        if sim.is_sim_mode:
            sim.log(
                "EXEC",
                f"Executing: {action.get('tool_name')} "
                f"by {agent_name} params={list((action.get('params') or {}).keys())}",
            )
        result = executor.execute(action)
        exec_elapsed = time.time() - exec_start
        success = getattr(result, "success", True)
        _learning_side = read_learning_side_effects(result)
        _embodiment_failed = _learning_side.embodiment_failed
        _drive_potential_diff = _learning_side.drive_potential_diff
        _drive_credit_withheld = _learning_side.drive_credit_withheld
        _drive_relief_channel = _learning_side.drive_relief_channel
        _reported_valence = _learning_side.outcome_valence
        _loop_logger.info(
            "Tool execution completed in %.2fs: %s, success=%s",
            exec_elapsed,
            action.get("tool_name"),
            success,
        )
        if sim.is_sim_mode:
            sim.log(
                "EXEC",
                f"Completed: {action.get('tool_name')} success={success} elapsed={exec_elapsed:.2f}s",
            )

        # Auto-recover: file exists → retry with overwrite (not if a person or a policy approved it: #1085)
        if (
            not success
            and not human_involved
            and action.get("tool_name") == "write_file"
            and "already exists" in str(getattr(result, "error", "")).lower()
        ):
            raw_params = action.get("params")
            safe_params = raw_params if isinstance(raw_params, dict) else {}
            _loop_logger.info(
                "Auto-recovery: retrying write_file with overwrite=True for %s",
                safe_params.get("path", "?"),
            )
            retry_action = dict(action)
            retry_params = dict(safe_params)
            retry_params["overwrite"] = True
            retry_action["params"] = retry_params
            result = executor.execute(retry_action)
            success = getattr(result, "success", True)
            if success:
                _loop_logger.info("Auto-recovery succeeded for write_file")
            else:
                _loop_logger.warning(
                    "Auto-recovery failed for write_file: %s",
                    getattr(result, "error", "unknown"),
                )

        # If this was a timeout retry prompt, store state for user response
        if action.get("_timeout_retry") and success:
            _stash_timeout_retry(action, sim, state)

        # Invalidate cache for write operations to ensure fresh reads
        tool_name = action.get("tool_name", "")
        if tool_name == "write_file" and success:
            written_path = action.get("params", {}).get("path")
            if written_path:
                invalidated = result_cache.invalidate(path=written_path)
                if invalidated > 0:
                    _loop_logger.debug("Invalidated %d cache entries for: %s", invalidated, written_path)

        # Log tool execution
        log_agentic(
            "agent_loop",
            "tool_called",
            {
                "tool": action.get("tool_name"),
                "success": success,
                "source": "llm_worker",
            },
        )

        # Log tool result details
        output = getattr(result, "output", None)
        if output:
            log_agentic(
                "agent_loop",
                "tool_result",
                {
                    "tool": action.get("tool_name"),
                    "output": output if isinstance(output, dict) else str(output)[:100],
                },
            )

        # Log to autonomy controller
        autonomy_controller.log_action(
            action_type="executed",
            action=action,
            reasoning=proposal.reasoning,
            mode=operational_mode(executor, state),
            confidence=confidence,
            citations=proposal.citations,
            outcome="success" if success else "failure",
            human_involved=human_involved,
            error=getattr(result, "error", None),
        )

        # Track outcome for context pool and learning
        # Get followup type to determine result storage and follow-up behavior
        tool_name = action.get("tool_name", "")
        current_mode = operational_mode(executor, state)

        # L2: Reset deliberation state when a non-think action fires.
        if tool_name != "think":
            _reset_deliberation(executor)

        from maxim.modes.definitions import get_tool_followup_type

        followup_type = get_tool_followup_type(tool_name, current_mode, overlay=overlay)

        # Store more result for tools that need processing (up to 3000 chars)
        needs_processing = followup_type in ("process", "respond", "engage")
        result_limit = 3000 if needs_processing else 100
        result_str = _followup_result_text(tool_name, output, result, result_limit)

        rec_outcome(
            agent_id=agent_id,
            tool_name=tool_name or "unknown",
            success=success,
            result_summary=result_str,
            error=getattr(result, "error", None),
            reasoning=getattr(proposal, "reasoning", "") if proposal else "",
            recent_outcomes=recent_outcomes,
            max_recent=max_recent,
            llm_worker=llm_worker,
            context_pool=context_pool,
            nac=nac,
            active_goal=state.data.get("active_goal") if hasattr(state, "data") else None,
            tool_params=action.get("params"),
            cluster_id=getattr(proposal, "cluster_id", None),
            clusters=getattr(proposal, "clusters", None),
            bridge_booked=getattr(result, "nac_failure_booked", None),
            refused=gate_refusal_of(result),
            embodiment_failed=_embodiment_failed,
            drive_potential_diff=_drive_potential_diff,
            drive_credit_withheld=_drive_credit_withheld,
            drive_relief_channel=_drive_relief_channel,
            outcome_valence=_reported_valence,
        )

        # Record plan outcome in MemoryHub for learning. A plan that
        # led to bodily harm is a NEGATIVE plan outcome even if the
        # tool mechanically succeeded (B5) — otherwise the plan path
        # books a positive CausalLink that competes with the tool's
        # learned aversion (the PlanHistoryBridge records under the
        # same tool event signature).
        if memory_hub is not None and gate_refusal_of(result) is None:  # a gate refusal books no plan outcome
            _record_plan_outcome(
                memory_hub=memory_hub,
                goal=proposal.reasoning or "",
                tool_name=tool_name,
                success=success and not _embodiment_failed,
            )

        # If this tool has a followup_type, trigger a follow-up LLM cycle.
        # The followup_type determines how the LLM should handle the results:
        #   "process" - LLM processes results for next action (coding agent)
        #   "respond" - LLM synthesizes results into user response
        #   "engage"  - LLM responds AND offers proactive follow-ups
        # "process" followups fire even on failure so the LLM can
        # learn from the error and retry with a different tool
        # (e.g. sim orchestrator's catch-all 'respond' rejects →
        # LLM should immediately re-think, not stall for 60s).
        # Note: Use 'is not None' to handle empty lists [] which are falsy but still valid output
        if followup_type and ((success and output is not None) or followup_type == "process"):
            queued_followup = _tool_followup(executor, tool_name, result_str, proposal, followup_type, current_mode)
            _loop_logger.info("Tool %s completed with followup_type=%s, queuing follow-up", tool_name, followup_type)

        tool_name = _record_turn_after_effects(
            action=action, success=success, proposal=proposal, context_pool=context_pool
        )

        # Process result
        try:
            followup = environment.step(result)
            if followup:
                state.update(followup)
        except Exception as e:
            log_swallowed_exception(e, operation="environment.step_followup")

        # Store in memory
        try:
            memory.store_raw(
                content={
                    "action": action,
                    "reasoning": proposal.reasoning,
                    "result": getattr(result, "output", None),
                    "success": getattr(result, "success", True),
                },
                metadata={"type": "action_execution"},
            )
        except Exception as e:
            log_swallowed_exception(e, operation="memory.store_raw")

        capture_loop_action(
            hippocampus,
            executor,
            observation,
            state,
            {"goal": proposal.reasoning, "source": "llm_worker"},
            action,
            confidence,
            result,
            run_id,
            agent_id,
            proposal=proposal,
        )

        # Handle failure
        if success is False:
            log_agentic(
                "agent_loop",
                "goal_failed",
                {
                    "tool": action.get("tool_name"),
                    "error": getattr(result, "error", None),
                },
                level="WARNING",
            )
            try:
                state.mark_failure(getattr(result, "error", None))
            except Exception as e:
                log_swallowed_exception(e, operation="state.mark_failure")

    except Exception as e:
        raised = str(e)
        _loop_logger.error(f"Action execution failed: {e}")
        autonomy_controller.log_action(
            action_type="executed",
            action=action,
            reasoning=proposal.reasoning,
            mode=operational_mode(executor, state),
            confidence=confidence,
            outcome="error",
            human_involved=human_involved,
            error=str(e),
        )

        # Track exception in recent_outcomes for LLM learning
        rec_outcome(
            agent_id=agent_id,
            tool_name=action.get("tool_name", "unknown"),
            success=False,
            result_summary=None,
            error=str(e),
            reasoning=getattr(proposal, "reasoning", "") if proposal else "",
            recent_outcomes=recent_outcomes,
            max_recent=max_recent,
            llm_worker=llm_worker,
            context_pool=context_pool,
            nac=nac,
            active_goal=state.data.get("active_goal") if hasattr(state, "data") else None,
            cluster_id=getattr(proposal, "cluster_id", None),
            clusters=getattr(proposal, "clusters", None),
            bridge_booked=getattr(result, "nac_failure_booked", None),
            refused=gate_refusal_of(result),
        )

        # Mark failure in state
        try:
            state.mark_failure(str(e))
        except Exception as mf_err:
            log_swallowed_exception(mf_err, operation="state.mark_failure_exc")
    return ExecutionOutcome(
        success=bool(success) and raised is None,
        raised=raised is not None,
        result=result,
        output=output,
        error=raised if raised is not None else getattr(result, "error", None),
        result_str=result_str,
        followup=queued_followup,
    )
