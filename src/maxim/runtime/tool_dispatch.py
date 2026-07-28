"""Tool dispatch utilities — outcome recording, parallel execution, agent name.

Extracted from agent_loop.py for single-responsibility decomposition.
The functions here handle tool outcome recording (including NAc causal
learning and energy tracking), parallel action execution, and safe
agent name extraction.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

from maxim.utils.logging import log_swallowed_exception
from maxim.utils.structured_logging import log_agentic

logger = logging.getLogger(__name__)


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
) -> None:
    """Record a tool outcome to all sinks including NAc causal learning.

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
    learn_success = success and not embodiment_failed

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
            from maxim.decisions.causal_link import Valence

            outcome_summary = (result_summary or error or "")[:50]
            valence = Valence.POSITIVE if learn_success else Valence.NEGATIVE
            sig = build_tool_signature(tool_name, tool_params)
            # Tag every NAc observation with agent_id so cross-agent
            # attribution gaps surface as filterable context, not
            # silently merged links.
            ctx: dict[str, Any] = {"agent_id": agent_id}
            if reasoning:
                ctx["goal"] = reasoning[:100]
            link = nac.observe(
                event_type="tool",
                event_signature=sig,
                outcome_type="tool_result",
                outcome_signature=f"{'success' if learn_success else 'failure'}:{outcome_summary}",
                outcome_valence=valence,
                delta_seconds=elapsed_s,
                context=ctx,
            )
            # Goal-level credit: if deliberation was active under a goal,
            # credit/penalize that goal so ThoughtGate learns whether
            # deliberation under this goal type produces good outcomes.
            if active_goal is not None:
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
            # the direction. Exactly-0 net progress -> tool-success fallback. The
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
                # exactly-0 -> tool-success boundary is load-bearing, so guard it
                # with an epsilon rather than float identity.
                if drive_potential_diff is not None and abs(drive_potential_diff) > 1e-9:
                    cluster_reward: float | None = 1.0 if drive_potential_diff > 0.0 else -1.0
                elif operant_only or drive_relief_only:
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
                else:
                    cluster_reward = 1.0 if learn_success else -1.0
                # Seam routing: drive-relief AND generic tool-success write the
                # INTEROCEPTION cluster only — never an exteroceptive cluster.
                # Direction-bearing clusters (audio) are credited exclusively
                # by source-attributable signals (the caregiver's
                # credit_operant_reward via the pending action above); letting
                # the uniform tool-success floor leak onto them would re-drown
                # the direction signal on the write side (probe 3).
                if cluster_reward is not None and not intero_cluster:
                    # Cluster context exists (extero) but no interoception
                    # slot: either a designed extero-only body (test-pinned)
                    # or the interoception encode failed this tick — the
                    # reward is dropped by design, but not silently.
                    logger.debug(
                        "cluster reward %+.1f for %s dropped: no interoception cluster in %r",
                        cluster_reward,
                        sig,
                        sorted(active_clusters),
                    )
                if cluster_reward is not None and intero_cluster:
                    try:
                        nac.update_cluster_reward(
                            agent_id=agent_id,
                            cluster_id=intero_cluster,
                            tool_signature=sig,
                            reward=cluster_reward,
                        )
                    except Exception:
                        # Mirrors the surrounding error policy — cluster
                        # learning is best-effort; an exception here must
                        # not crash the agent loop.
                        logger.debug("update_cluster_reward raised", exc_info=True)

            # Sim trace
            try:
                from maxim.simulation.sim_logger import sim_nac

                sim_nac(
                    f"tool:{tool_name}",
                    valence.value,
                    getattr(link, "last_rpe", 0.0) or 0.0,
                    getattr(link, "confidence", 0.5),
                )
            except Exception:
                pass  # sim trace is best-effort
        except Exception as e:
            logger.warning("NAc reward signal failed for tool %s: %s", tool_name, e)

    # Energy → NAc: learn which tools are expensive (metabolic budget)
    if nac is not None and elapsed_s > 0:
        try:
            from maxim.decisions.causal_link import Valence as _V

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
            pass


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
) -> tuple[list[dict[str, Any]], str]:
    """Execute a batch of parallel actions with autonomy gating.

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
            output = getattr(result, "output", None)
            error = getattr(result, "error", None)

            parallel_results.append(
                {
                    "tool": tool_name,
                    "params": parallel_action.get("params", {}),
                    "success": success,
                    "result": str(output)[:2000] if output else None,
                    "error": error,
                }
            )

            if not success:
                all_succeeded = False

            log_agentic(
                "agent_loop",
                "parallel_action_complete",
                {"tool": tool_name, "index": idx, "success": success},
            )

        except Exception as e:
            logger.error("Parallel action %s failed: %s", tool_name, e)
            parallel_results.append(
                {
                    "tool": tool_name,
                    "success": False,
                    "error": str(e),
                    "result": None,
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

    return parallel_results, combined_results
