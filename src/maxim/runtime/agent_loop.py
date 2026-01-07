from __future__ import annotations

import os
import re
import time
import itertools
import json
from typing import Any

from maxim.evaluation.base import Evaluator
from maxim.runtime.state import RuntimeState
from maxim.utils.logging import warn


def _safe_agent_name(agent: Any) -> str:
    raw = None
    try:
        raw = getattr(agent, "agent_name", None) or getattr(agent, "name", None)
    except Exception:
        raw = None
    if not raw:
        raw = type(agent).__name__
    name = str(raw).strip() or "agent"
    name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)
    return name.strip("._-") or "agent"


def _persist_state_json(state: Any, path: str, *, meta: dict[str, Any]) -> None:
    try:
        if hasattr(state, "save_json") and callable(getattr(state, "save_json")):
            try:
                state.save_json(path, meta=meta)
            except TypeError:
                state.save_json(path)
            return
        if hasattr(state, "snapshot") and callable(getattr(state, "snapshot")):
            snap = state.snapshot()
        else:
            snap = {"state": repr(state)}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fp:
            json.dump({"saved_at": time.time(), **meta, **snap}, fp, indent=2, default=str)
        os.replace(tmp, path)
    except Exception as e:
        warn("Failed to persist runtime state: %s", e)


def run_agent_loop(
    agent: Any,
    environment: Any,
    state: Any,
    memory: Any,
    decision_engine: Any,
    executor: Any,
    *,
    evaluators: list[Evaluator] | None = None,
    max_steps: int = 100,
    run_id: str | None = None,
    stop_event: Any | None = None,
    on_step: Any | None = None,
    break_on_no_intent: bool = True,
    idle_sleep_s: float = 0.25,
) -> None:
    """
    Canonical agentic loop:
    observe → agent proposes intent → planner proposes plans → policy constrains → decision engine selects action → executor runs tool
    """
    if evaluators is None:
        evaluators = []

    if not run_id:
        run_id = time.strftime("%Y-%m-%d_%H%M%S")
    agent_name = _safe_agent_name(agent)
    state_path = os.path.join("data", "agents", agent_name, "runtime", f"state_{run_id}.json")
    _persist_state_json(state, state_path, meta={"run_id": run_id, "agent_name": agent_name})

    max_steps_i = int(max_steps or 0)
    step_iter = itertools.count() if max_steps_i <= 0 else range(max_steps_i)
    for _ in step_iter:
        try:
            if stop_event is not None and hasattr(stop_event, "is_set") and stop_event.is_set():
                break
        except Exception:
            pass

        observation = environment.observe()
        state.update(observation)

        intent = None
        if hasattr(agent, "propose_intent"):
            intent = agent.propose_intent(state, memory)
        elif hasattr(agent, "decide"):
            # Legacy fallback: treat `decide()` as a goal provider.
            out = agent.decide(state, memory)
            if isinstance(out, dict):
                intent = out
            elif isinstance(out, str) and out:
                intent = {"goal": out, "confidence": 1.0}

        if not isinstance(intent, dict) or not intent:
            if break_on_no_intent:
                break
            try:
                time.sleep(float(idle_sleep_s))
            except Exception:
                pass
            continue

        goal = intent.get("goal") or intent.get("intent")
        if goal is None:
            if break_on_no_intent:
                break
            try:
                time.sleep(float(idle_sleep_s))
            except Exception:
                pass
            continue

        decision = decision_engine.decide(goal, state, memory)
        if not isinstance(decision, dict) or not decision.get("action"):
            if break_on_no_intent:
                break
            try:
                time.sleep(float(idle_sleep_s))
            except Exception:
                pass
            continue

        action = decision["action"]
        if not isinstance(action, dict):
            warn("Invalid action selected: %r", action)
            if break_on_no_intent:
                break
            try:
                time.sleep(float(idle_sleep_s))
            except Exception:
                pass
            continue

        ctx = {
            "intent": intent,
            "plan": decision.get("plan"),
            "registered_tools": getattr(getattr(executor, "registry", None), "list", lambda: [])(),
        }
        result = executor.execute(action)
        ctx["tool_result"] = result
        eval_results = []
        for evaluator in evaluators:
            try:
                eval_results.append(evaluator.evaluate(ctx))
            except Exception:
                continue
        try:
            if callable(on_step):
                on_step(
                    {
                        "intent": intent,
                        "goal": goal,
                        "decision": decision,
                        "action": action,
                        "tool_result": result,
                        "evaluations": eval_results,
                        "state": state,
                        "memory": memory,
                    }
                )
        except Exception:
            pass

        try:
            followup = environment.step(result)
            if followup:
                state.update(followup)
        except Exception:
            pass

        try:
            memory.store_raw(
                content={
                    "state": state.snapshot(),
                    "intent": intent,
                    "decision": decision,
                    "tool_result": result,
                    "evaluations": eval_results,
                },
                metadata={"type": "episode"},
            )
        except Exception as e:
            warn("Memory store failed: %s", e)

        if getattr(result, "success", True) is False:
            try:
                state.mark_failure(getattr(result, "error", None))
            except Exception:
                pass

        try:
            state.steps_taken += 1
        except Exception:
            pass
        _persist_state_json(state, state_path, meta={"run_id": run_id, "agent_name": agent_name})

        if state.is_done():
            break

    _persist_state_json(state, state_path, meta={"run_id": run_id, "agent_name": agent_name})
