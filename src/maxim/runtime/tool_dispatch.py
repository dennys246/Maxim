"""Tool dispatch utilities — outcome recording, parallel execution, agent name.

Extracted from agent_loop.py for single-responsibility decomposition.
The functions here handle tool outcome recording (including NAc causal
learning and energy tracking), parallel action execution, and safe
agent name extraction.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from maxim.utils.logging import log_swallowed_exception
from maxim.utils.structured_logging import log_agentic

logger = logging.getLogger(__name__)


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

    # NAc causal learning: record tool → outcome so predictions improve
    if nac is not None:
        try:
            from maxim.decisions.causal_link import Valence

            outcome_summary = (result_summary or error or "")[:50]
            valence = Valence.POSITIVE if success else Valence.NEGATIVE
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
                outcome_signature=f"{'success' if success else 'failure'}:{outcome_summary}",
                outcome_valence=valence,
                delta_seconds=elapsed_s,
                context=ctx,
            )
            # Goal-level credit: if deliberation was active under a goal,
            # credit/penalize that goal so ThoughtGate learns whether
            # deliberation under this goal type produces good outcomes.
            if active_goal is not None:
                reward = 1.0 if success else -1.0
                nac.credit_goal(active_goal, reward)

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
