"""Expectation validation for scenario testing.

Checks post-run state against scenario-defined expectations:
- action_blocked: FearAgent blocked a tool call
- action_taken: A specific tool was called with matching output
- memory_formed: Hippocampus contains a memory with given content
- pipeline_continued: Pipeline didn't halt after a specific percept
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from maxim.memory.hippocampus import Hippocampus
    from maxim.simulation.scenario_source import Expectation
    from maxim.simulation.sinks import ActionRecord, RecordingSink

logger = logging.getLogger(__name__)


@dataclass
class ExpectationResult:
    """Result of checking a single expectation."""

    expectation: Expectation
    passed: bool
    detail: str = ""


@dataclass
class ScenarioResult:
    """Complete result of running a scenario."""

    scenario_name: str
    passed: bool
    results: list[ExpectationResult] = field(default_factory=list)
    actions: list[ActionRecord] = field(default_factory=list)
    exit_reason: str = "completed"
    duration: float = 0.0

    @property
    def expectations_met(self) -> list[str]:
        return [r.expectation.description or r.expectation.type for r in self.results if r.passed]

    @property
    def expectations_failed(self) -> list[str]:
        return [
            f"{r.expectation.description or r.expectation.type}: {r.detail}"
            for r in self.results
            if not r.passed
        ]


def validate_expectations(
    expectations: list[Expectation],
    sink: RecordingSink,
    hippocampus: Hippocampus | None = None,
    emitted_tags: set[str] | None = None,
) -> list[ExpectationResult]:
    """Validate all expectations against the recorded actions and memory state."""
    results = []
    for exp in expectations:
        if exp.type == "action_blocked":
            results.append(_check_action_blocked(exp, sink))
        elif exp.type == "action_taken":
            results.append(_check_action_taken(exp, sink))
        elif exp.type == "memory_formed":
            results.append(_check_memory_formed(exp, hippocampus))
        elif exp.type == "pipeline_continued":
            results.append(_check_pipeline_continued(exp, sink, emitted_tags))
        else:
            results.append(
                ExpectationResult(
                    expectation=exp, passed=False, detail=f"Unknown expectation type: {exp.type}"
                )
            )
    return results


def _check_action_blocked(exp: Expectation, sink: RecordingSink) -> ExpectationResult:
    """Check that FearAgent blocked an action matching the criteria."""
    blocked = sink.find_blocked(
        tool_pattern=exp.tool_pattern,
        reason_contains=exp.reason_contains,
    )
    if blocked:
        return ExpectationResult(
            expectation=exp,
            passed=True,
            detail=f"Found {len(blocked)} blocked action(s)",
        )
    return ExpectationResult(
        expectation=exp,
        passed=False,
        detail=f"No blocked actions found matching tool_pattern={exp.tool_pattern!r}, "
        f"reason_contains={exp.reason_contains!r}",
    )


def _check_action_taken(exp: Expectation, sink: RecordingSink) -> ExpectationResult:
    """Check that a specific tool was called with matching output."""
    actions = sink.find_actions(tool=exp.tool, output_matches=exp.output_matches)
    if actions:
        return ExpectationResult(
            expectation=exp,
            passed=True,
            detail=f"Found {len(actions)} matching action(s)",
        )
    return ExpectationResult(
        expectation=exp,
        passed=False,
        detail=f"No actions found matching tool={exp.tool!r}, output_matches={exp.output_matches!r}",
    )


def _check_memory_formed(
    exp: Expectation, hippocampus: Hippocampus | None
) -> ExpectationResult:
    """Check that a memory was formed with given content."""
    if hippocampus is None:
        return ExpectationResult(
            expectation=exp, passed=False, detail="No hippocampus available"
        )

    query = exp.memory_contains or ""
    matches = hippocampus.search_by_content(query)

    if matches:
        return ExpectationResult(
            expectation=exp,
            passed=True,
            detail=f"Found {len(matches)} memory/memories containing {query!r}",
        )
    return ExpectationResult(
        expectation=exp,
        passed=False,
        detail=f"No memories found containing {query!r}",
    )


def _check_pipeline_continued(
    exp: Expectation,
    sink: RecordingSink,
    emitted_tags: set[str] | None,
) -> ExpectationResult:
    """Check that the pipeline continued processing after a tagged percept."""
    tag = exp.after_tag
    if not tag:
        return ExpectationResult(
            expectation=exp, passed=False, detail="No after_tag specified"
        )

    if emitted_tags and tag not in emitted_tags:
        return ExpectationResult(
            expectation=exp,
            passed=False,
            detail=f"Tag {tag!r} was never emitted",
        )

    # If there are any actions after the tagged percept was emitted,
    # the pipeline continued
    if sink.actions:
        return ExpectationResult(
            expectation=exp,
            passed=True,
            detail=f"Pipeline produced {len(sink.actions)} action(s) after tag {tag!r}",
        )

    return ExpectationResult(
        expectation=exp,
        passed=False,
        detail=f"No actions recorded after tag {tag!r} — pipeline may have halted",
    )
