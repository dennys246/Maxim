"""Expectation validation for scenario testing.

Checks post-run state against scenario-defined expectations:

Behavioral:
- action_blocked: FearAgent blocked a tool call
- action_taken: A specific tool was called with matching output
- memory_formed: Hippocampus contains a memory with given content
- pipeline_continued: Pipeline didn't halt after a specific percept

Metric:
- action_count_range: Total actions within [min, max]
- tool_success_rate: Tool's success rate >= threshold
- response_latency_ms: Inter-action latency p50/p95 within caps

Bio-system (Phase 0c):
- memory_count_range: Episodic memory count within [min, max]
- concept_formed: ATL formed a concept matching name
- graph_density_above: Hippocampal associative graph density >= threshold
- causal_link_formed: NAc formed a causal link matching event pattern
- prediction_valence: NAc predicts specific valence for a tool
- hallucination_rate_below: Tool hallucination rate < threshold
- tool_used: A specific tool was called at least once
- pain_signal_count: Pain signals fired >= min
"""

from __future__ import annotations

import logging
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
        return [f"{r.expectation.description or r.expectation.type}: {r.detail}" for r in self.results if not r.passed]


def validate_expectations(
    expectations: list[Expectation],
    sink: RecordingSink,
    hippocampus: Hippocampus | None = None,
    emitted_tags: set[str] | None = None,
    *,
    subsystem_snapshot: dict[str, Any] | None = None,
    tool_stats: dict[str, Any] | None = None,
) -> list[ExpectationResult]:
    """Validate all expectations against recorded actions, memory, and bio-system state.

    Args:
        expectations: List of expectations to check.
        sink: RecordingSink with action history.
        hippocampus: Hippocampus instance for memory checks.
        emitted_tags: Scenario tags that were emitted.
        subsystem_snapshot: From AUTIntrospector.benchmark_snapshot() — provides
            hippocampus_stats, nac_stats, pain_stats, concepts, etc.
        tool_stats: From Executor.tool_usage_stats() — provides hallucination_rate, etc.
    """
    snapshot = subsystem_snapshot or {}
    tstats = tool_stats or {}

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
        elif exp.type == "action_count_range":
            results.append(_check_action_count_range(exp, sink))
        elif exp.type == "tool_success_rate":
            results.append(_check_tool_success_rate(exp, sink))
        elif exp.type == "response_latency_ms":
            results.append(_check_response_latency_ms(exp, sink))
        # Bio-system expectation types (Phase 0c)
        elif exp.type == "memory_count_range":
            results.append(_check_memory_count_range(exp, snapshot))
        elif exp.type == "concept_formed":
            results.append(_check_concept_formed(exp, snapshot))
        elif exp.type == "graph_density_above":
            results.append(_check_graph_density(exp, snapshot))
        elif exp.type == "causal_link_formed":
            results.append(_check_causal_link_formed(exp, snapshot))
        elif exp.type == "prediction_valence":
            results.append(_check_prediction_valence(exp, snapshot))
        elif exp.type == "hallucination_rate_below":
            results.append(_check_hallucination_rate(exp, tstats))
        elif exp.type == "tool_used":
            results.append(_check_tool_used(exp, sink))
        elif exp.type == "pain_signal_count":
            results.append(_check_pain_signal_count(exp, snapshot))
        else:
            results.append(
                ExpectationResult(expectation=exp, passed=False, detail=f"Unknown expectation type: {exp.type}")
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


def _check_memory_formed(exp: Expectation, hippocampus: Hippocampus | None) -> ExpectationResult:
    """Check that a memory was formed with given content."""
    if hippocampus is None:
        return ExpectationResult(expectation=exp, passed=False, detail="No hippocampus available")

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
        return ExpectationResult(expectation=exp, passed=False, detail="No after_tag specified")

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


# ── Metric expectation types (Phase: realtime refinement) ────────────────


def _check_action_count_range(exp: Expectation, sink: RecordingSink) -> ExpectationResult:
    """Check that total action count is within [min, max] range."""
    count = len(sink.actions)
    min_count = getattr(exp, "min_count", None) or exp.params.get("min", 0)
    max_count = getattr(exp, "max_count", None) or exp.params.get("max", float("inf"))
    passed = int(min_count) <= count <= int(max_count)
    return ExpectationResult(
        expectation=exp,
        passed=passed,
        detail=f"Action count {count} {'within' if passed else 'outside'} range [{min_count}, {max_count}]",
    )


def _check_tool_success_rate(exp: Expectation, sink: RecordingSink) -> ExpectationResult:
    """Check that a tool's success rate meets a minimum threshold."""
    tool = exp.tool or exp.params.get("tool", "")
    min_rate = float(exp.params.get("min_rate", 0.0))

    matching = [a for a in sink.actions if a.tool_name == tool]
    if not matching:
        return ExpectationResult(
            expectation=exp,
            passed=False,
            detail=f"Tool {tool!r} was never called",
        )

    successes = sum(1 for a in matching if a.result_success)
    rate = successes / len(matching)
    passed = rate >= min_rate
    return ExpectationResult(
        expectation=exp,
        passed=passed,
        detail=f"Tool {tool!r} success rate {rate:.0%} ({successes}/{len(matching)}) "
        f"{'meets' if passed else 'below'} threshold {min_rate:.0%}",
    )


def _check_response_latency_ms(exp: Expectation, sink: RecordingSink) -> ExpectationResult:
    """Check inter-action latency percentiles against thresholds.

    Computes the gap between consecutive action timestamps (in milliseconds)
    and compares p50 / p95 against the configured caps. This is a proxy for
    per-action response latency — it's the actual wall-clock gap between the
    pipeline producing one tool call and the next.

    Params:
        p50_max_ms: optional float — p50 must be <= this
        p95_max_ms: optional float — p95 must be <= this
        min_samples: optional int (default 2) — skip check if fewer actions
    """
    actions = sink.actions
    min_samples = int(exp.params.get("min_samples", 2))
    if len(actions) < min_samples:
        return ExpectationResult(
            expectation=exp,
            passed=False,
            detail=f"Only {len(actions)} action(s) recorded; need >= {min_samples} for latency",
        )

    # Inter-action gaps in ms
    gaps_ms = [(actions[i].timestamp - actions[i - 1].timestamp) * 1000.0 for i in range(1, len(actions))]
    gaps_ms.sort()

    def _percentile(p: float) -> float:
        if not gaps_ms:
            return 0.0
        idx = min(len(gaps_ms) - 1, int(round(p * (len(gaps_ms) - 1))))
        return gaps_ms[idx]

    p50 = _percentile(0.50)
    p95 = _percentile(0.95)

    p50_cap = exp.params.get("p50_max_ms")
    p95_cap = exp.params.get("p95_max_ms")

    failures = []
    if p50_cap is not None and p50 > float(p50_cap):
        failures.append(f"p50={p50:.0f}ms > {float(p50_cap):.0f}ms")
    if p95_cap is not None and p95 > float(p95_cap):
        failures.append(f"p95={p95:.0f}ms > {float(p95_cap):.0f}ms")

    if p50_cap is None and p95_cap is None:
        return ExpectationResult(
            expectation=exp,
            passed=False,
            detail="No thresholds set (provide p50_max_ms and/or p95_max_ms)",
        )

    passed = not failures
    if passed:
        detail = f"Latency OK (p50={p50:.0f}ms, p95={p95:.0f}ms, n={len(gaps_ms)})"
    else:
        detail = "; ".join(failures) + f" (n={len(gaps_ms)})"

    return ExpectationResult(expectation=exp, passed=passed, detail=detail)


# ── Bio-system expectation types (Phase 0c) ───────────────────────────────


def _check_memory_count_range(exp: Expectation, snapshot: dict[str, Any]) -> ExpectationResult:
    """Check that episodic memory count is within [min, max]."""
    hippo = snapshot.get("hippocampus_stats", {})
    count = hippo.get("total_memories", 0)
    if not count and "system_stats" in snapshot:
        count = snapshot["system_stats"].get("hippocampus_memories", 0)
    min_count = int(exp.params.get("min", 0))
    max_count = int(exp.params.get("max", 10000))
    passed = min_count <= count <= max_count
    return ExpectationResult(
        expectation=exp,
        passed=passed,
        detail=f"Memory count {count} {'within' if passed else 'outside'} [{min_count}, {max_count}]",
    )


def _check_concept_formed(exp: Expectation, snapshot: dict[str, Any]) -> ExpectationResult:
    """Check that ATL formed a concept matching the given name."""
    concept_name = exp.params.get("concept_name", "")
    concepts = snapshot.get("concepts", {})
    if not concepts.get("available"):
        return ExpectationResult(expectation=exp, passed=False, detail="ATL not available")

    for c in concepts.get("concepts", []):
        if concept_name.lower() in c.get("name", "").lower():
            return ExpectationResult(
                expectation=exp,
                passed=True,
                detail=f"Found concept matching '{concept_name}': {c.get('name')} (confidence={c.get('confidence', 0):.2f})",
            )
    return ExpectationResult(
        expectation=exp,
        passed=False,
        detail=f"No ATL concept matching '{concept_name}' found",
    )


def _check_graph_density(exp: Expectation, snapshot: dict[str, Any]) -> ExpectationResult:
    """Check that hippocampal associative graph density is above threshold."""
    hippo = snapshot.get("hippocampus_stats", {})
    nodes = hippo.get("graph_nodes", 0)
    edges = hippo.get("graph_edges", 0)
    density = edges / max(nodes, 1)
    min_density = float(exp.params.get("min_density", 0.0))
    passed = density >= min_density
    return ExpectationResult(
        expectation=exp,
        passed=passed,
        detail=f"Graph density {density:.2f} ({edges} edges / {nodes} nodes) "
        f"{'meets' if passed else 'below'} threshold {min_density:.2f}",
    )


def _check_causal_link_formed(exp: Expectation, snapshot: dict[str, Any]) -> ExpectationResult:
    """Check that NAc formed a causal link matching the event pattern."""
    event_contains = exp.params.get("event_contains", "")
    causal = snapshot.get("causal_links", {})
    if not causal.get("available"):
        return ExpectationResult(expectation=exp, passed=False, detail="NAc not available")

    for link in causal.get("links", []):
        if event_contains.lower() in link.get("event", "").lower():
            return ExpectationResult(
                expectation=exp,
                passed=True,
                detail=f"Found causal link matching '{event_contains}': "
                f"{link.get('event')} → {link.get('outcome')} "
                f"(confidence={link.get('confidence', 0):.2f}, observations={link.get('observations', 0)})",
            )
    return ExpectationResult(
        expectation=exp,
        passed=False,
        detail=f"No causal link with event matching '{event_contains}'",
    )


def _check_prediction_valence(exp: Expectation, snapshot: dict[str, Any]) -> ExpectationResult:
    """Check that NAc predicts a specific valence for a tool/event."""
    # This needs introspector.predict_outcome() which isn't in the snapshot.
    # For now, check causal links for valence.
    tool = exp.params.get("tool", exp.tool or "")
    expected = exp.params.get("expected_valence", "negative")
    causal = snapshot.get("causal_links", {})
    if not causal.get("available"):
        return ExpectationResult(expectation=exp, passed=False, detail="NAc not available")

    for link in causal.get("links", []):
        if tool.lower() in link.get("event", "").lower():
            actual = link.get("valence", "").lower()
            if expected.lower() in actual:
                return ExpectationResult(
                    expectation=exp,
                    passed=True,
                    detail=f"NAc predicts '{actual}' valence for '{tool}' (expected: '{expected}')",
                )
            return ExpectationResult(
                expectation=exp,
                passed=False,
                detail=f"NAc predicts '{actual}' valence for '{tool}' (expected: '{expected}')",
            )
    return ExpectationResult(
        expectation=exp,
        passed=False,
        detail=f"No causal link found for tool '{tool}'",
    )


def _check_hallucination_rate(exp: Expectation, tool_stats: dict[str, Any]) -> ExpectationResult:
    """Check that tool hallucination rate is below threshold."""
    max_rate = float(exp.params.get("max_rate", 0.3))
    rate = tool_stats.get("hallucination_rate", 0.0)
    total = tool_stats.get("total_attempts", 0)
    if total == 0:
        return ExpectationResult(expectation=exp, passed=True, detail="No tool calls made (rate=0)")
    passed = rate <= max_rate
    return ExpectationResult(
        expectation=exp,
        passed=passed,
        detail=f"Hallucination rate {rate:.0%} ({tool_stats.get('total_hallucinated', 0)}/{total}) "
        f"{'meets' if passed else 'exceeds'} threshold {max_rate:.0%}",
    )


def _check_tool_used(exp: Expectation, sink: "RecordingSink") -> ExpectationResult:
    """Check that a specific tool was called at least once."""
    tool = exp.tool or exp.params.get("tool", "")
    matching = [a for a in sink.actions if a.tool_name == tool]
    if matching:
        return ExpectationResult(
            expectation=exp,
            passed=True,
            detail=f"Tool '{tool}' used {len(matching)} time(s)",
        )
    return ExpectationResult(
        expectation=exp,
        passed=False,
        detail=f"Tool '{tool}' was never called",
    )


def _check_pain_signal_count(exp: Expectation, snapshot: dict[str, Any]) -> ExpectationResult:
    """Check that pain signals fired at least min times."""
    min_count = int(exp.params.get("min", 1))
    pain = snapshot.get("pain_stats", {})
    if not pain.get("available"):
        return ExpectationResult(expectation=exp, passed=False, detail="PainDetector not available")

    count = pain.get("total_pain_signals", 0)
    passed = count >= min_count
    return ExpectationResult(
        expectation=exp,
        passed=passed,
        detail=f"Pain signals: {count} {'meets' if passed else 'below'} minimum {min_count}",
    )
