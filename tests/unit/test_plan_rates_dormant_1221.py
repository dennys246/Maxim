"""#1221 item 1: the planner's tool-success-rate chain is Dormant, and it no longer swallows a call into nothing.

``plan_manager`` called ``svc.statistician.get_tool_success_rate(...)``; ``StatisticianAgent`` never had that
method, so every call raised ``AttributeError`` into an ``except Exception: pass`` and ``PhaseContext.
tool_success_rates`` was always empty. Nothing renders that field, and ``PlanHistoryBridge.get_tool_success_rate``
(which counted links, not observations) has no caller. Owner decision 2026-10-12: mark the chain Dormant and stop
the swallow, rather than wire a new planning signal or delete a persisted field.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _calls_get_tool_success_rate(rel: str) -> list[int]:
    tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
    return [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get_tool_success_rate"
    ]


@pytest.mark.xfail(strict=True, reason="#1221: plan_manager calls a get_tool_success_rate the Statistician lacks")
def test_the_planner_calls_no_success_rate_method_that_does_not_exist() -> None:
    from maxim.agents.statistician_agent import StatisticianAgent

    assert not hasattr(StatisticianAgent, "get_tool_success_rate")
    assert _calls_get_tool_success_rate("src/maxim/planning/plan_manager.py") == []


def test_the_dormant_rate_method_has_no_caller() -> None:
    """Guard: PlanHistoryBridge.get_tool_success_rate stays callerless while Dormant (its link counting is wrong;
    reviving it means fixing that first)."""
    callers = []
    for path in sorted((ROOT / "src" / "maxim").rglob("*.py")):
        rel = str(path.relative_to(ROOT))
        if rel != "src/maxim/bridges/planning_bridge.py" and _calls_get_tool_success_rate(rel):
            callers.append(rel)
    assert callers == [] or callers == ["src/maxim/planning/plan_manager.py"]  # the latter until #1221's fix
