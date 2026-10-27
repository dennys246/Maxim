"""ScenarioSource — YAML-driven percept sequences for testing.

Reads a scenario file and emits Percepts on schedule, supporting
both wall-clock (relative) and loop-iteration (step_based) timing.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from maxim.agents.bus import Percept
from maxim.utils.scenario_inputs import read_recorded

logger = logging.getLogger(__name__)


@dataclass
class Expectation:
    """A post-run assertion defined in the scenario YAML."""

    type: str  # action_blocked, action_taken, memory_formed, pipeline_continued
    description: str = ""
    # action_blocked fields
    tool_pattern: str | None = None
    reason_contains: str | None = None
    # action_taken fields
    tool: str | None = None
    output_matches: str | None = None
    # memory_formed fields
    memory_contains: str | None = None
    min_tier: str | None = None
    # pipeline_continued fields
    after_tag: str | None = None
    # action_taken: ``tool`` names a per-embodiment entity tool (``sense_<x>`` / ``<x>_<y>``), not a registered one
    entity_tool: bool = False
    # metric expectation fields (realtime refinement)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioDefinition:
    """Parsed scenario from YAML.

    Core fields (name, percepts, expectations) are required for all scenarios.
    Optional sections add capability for specific runners:
    - tags/metadata: discoverability for benchmark filtering
    - benchmark: tier/weight/metrics config for benchmark runner
    - suite: child scenario list for benchmark suites
    - config: execution defaults (mode, timeout, sandbox)
    """

    name: str
    description: str = ""
    timing: str = "step_based"  # "relative" or "step_based"
    percepts: list[dict[str, Any]] = field(default_factory=list)
    expectations: list[Expectation] = field(default_factory=list)
    # Discoverability metadata (optional)
    tags: list[str] = field(default_factory=list)
    difficulty: str = ""  # easy | medium | hard
    estimated_duration_s: float = 0.0
    subsystems_tested: list[str] = field(default_factory=list)
    tools_tested: list[str] = field(default_factory=list)
    # Optional runner-specific sections (attached by loader)
    benchmark: dict[str, Any] | None = None
    suite: dict[str, Any] | None = None
    config: dict[str, Any] | None = None


#: The ONE table of scoring-threshold operators (#1395): ``load_scenario`` refuses any other key, and
#: ``BenchmarkRunner._check_thresholds`` applies exactly these, so a key cannot be accepted without being checked.
#: Each maps to "the measured value HOLDS the threshold".
THRESHOLD_OPERATORS: dict[str, Any] = {
    "pass": lambda value, threshold: value >= threshold,
    "pass_above": lambda value, threshold: value >= threshold,
    "pass_below": lambda value, threshold: value <= threshold,
}
_ENTITY_TOOL = re.compile(r"^(sense_[a-z0-9_]+|[a-z0-9]+_[a-z0-9_]+)$")


def expectation_tool_problem(tool: str | None, *, entity_tool: bool = False) -> str | None:
    """Why an expectation's ``tool`` cannot match, or ``None``. ``action_taken`` / ``tool_success_rate`` match the
    REGISTERED tool name exactly, so a class name such as ``RespondTool`` could never pass (#1394). A per-embodiment
    entity tool is allowed only with the explicit ``entity_tool: true`` (owner decision 2026-10-26), shaped
    ``sense_<x>`` or ``<x>_<y>`` and not a registered name. Shared by ``load_scenario`` (refuses) and the scenario
    generator (drops the expectation)."""
    if not tool:
        return None
    from maxim.modes.definitions import REGISTERED_TOOL_NAMES

    if entity_tool:
        if tool in REGISTERED_TOOL_NAMES:
            return f"{tool!r} is a registered tool, not an entity tool (drop entity_tool: true)"
        if not _ENTITY_TOOL.match(tool):
            return f"entity_tool {tool!r} is not shaped sense_<x> or <x>_<y>"
        return None
    if tool not in REGISTERED_TOOL_NAMES:
        return (
            f"expectation tool {tool!r} is not a registered tool name (use the name the tool registers under, e.g. "
            "'respond' for RespondTool; mark a per-embodiment entity tool with entity_tool: true)"
        )
    return None


def _check_expectation_tool(path: Path, exp: Expectation) -> None:
    problem = expectation_tool_problem(exp.tool, entity_tool=exp.entity_tool)
    if problem:
        raise ValueError(f"Scenario {path}: {problem}")


def _check_scoring(path: Path, section: str, body: dict[str, Any]) -> None:
    """Each scoring threshold uses only the operators ``BenchmarkRunner._check_thresholds`` knows, and at least one:
    an unknown key (``pas:``) or an empty threshold used to count as applied while checking nothing (#1395)."""
    for metric, threshold in (body.get("scoring") or {}).items():
        if not isinstance(threshold, dict) or not threshold:
            raise ValueError(
                f"Scenario {path}: {section}.scoring.{metric} has no threshold ({sorted(THRESHOLD_OPERATORS)})"
            )
        unknown = sorted(set(threshold) - set(THRESHOLD_OPERATORS))
        if unknown:
            raise ValueError(
                f"Scenario {path}: {section}.scoring.{metric} has unknown key(s) {unknown}; "
                f"known: {sorted(THRESHOLD_OPERATORS)}"
            )


def load_scenario(path: Path) -> ScenarioDefinition:
    """Load and parse a scenario YAML file."""
    try:
        raw = yaml.safe_load(read_recorded(path))
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse scenario YAML {path}: {e}") from e

    if raw is None:
        raw = {}

    expectations = []
    for exp_raw in raw.get("expectations", []):
        exp_type = exp_raw.get("type")
        if not exp_type:
            logger.warning("Expectation missing 'type' field, skipping: %s", exp_raw)
            continue
        # Metric expectations (action_count_range, tool_success_rate,
        # response_latency_ms) carry their thresholds in a `params` dict
        # so the Expectation schema stays stable as new metrics are added.
        # For YAML ergonomics, we also auto-absorb any unknown top-level
        # keys into params.
        known_keys = {
            "type",
            "description",
            "tool_pattern",
            "reason_contains",
            "tool",
            "output_matches",
            "memory_contains",
            "min_tier",
            "after_tag",
            "entity_tool",
            "params",
        }
        extra_params = {k: v for k, v in exp_raw.items() if k not in known_keys}
        params = {**(exp_raw.get("params") or {}), **extra_params}

        expectations.append(
            Expectation(
                type=exp_type,
                description=exp_raw.get("description", ""),
                tool_pattern=exp_raw.get("tool_pattern"),
                reason_contains=exp_raw.get("reason_contains"),
                tool=exp_raw.get("tool"),
                output_matches=exp_raw.get("output_matches"),
                memory_contains=exp_raw.get("memory_contains"),
                min_tier=exp_raw.get("min_tier"),
                after_tag=exp_raw.get("after_tag"),
                entity_tool=bool(exp_raw.get("entity_tool", False)),
                params=params,
            )
        )
        _check_expectation_tool(path, expectations[-1])

    for section in ("benchmark", "suite"):
        _check_scoring(path, section, ((raw.get(section) or {}) if isinstance(raw.get(section), dict) else {}))

    percepts = raw.get("percepts", [])
    if not percepts:
        logger.warning("Scenario %s has no percepts — will complete immediately", path.stem)

    return ScenarioDefinition(
        name=raw.get("name", path.stem),
        description=raw.get("description", ""),
        timing=raw.get("timing", "step_based"),
        percepts=percepts,
        expectations=expectations,
        # Discoverability metadata
        tags=raw.get("tags", []),
        difficulty=raw.get("difficulty", ""),
        estimated_duration_s=float(raw.get("estimated_duration_s", 0)),
        subsystems_tested=raw.get("subsystems_tested", []),
        tools_tested=raw.get("tools_tested", []),
        # Optional runner-specific sections
        benchmark=raw.get("benchmark"),
        suite=raw.get("suite"),
        config=raw.get("config"),
    )


def _percept_from_dict(raw: dict[str, Any], base_time: float) -> Percept:
    """Convert a scenario percept dict to a Percept object."""
    return Percept(
        timestamp=base_time + raw.get("at", 0.0),
        source=raw.get("source", "cli"),
        detections=raw.get("detections", []),
        transcript_chunk=raw.get("transcript_chunk"),
        cli_input=raw.get("cli_input"),
        salience=raw.get("salience", 0.0),
        novelty=raw.get("novelty", 0.0),
        has_voice_command=raw.get("has_voice_command", False),
        has_maxim_keyword=raw.get("has_maxim_keyword", False),
        hard_override=raw.get("hard_override"),
        content=raw.get("content"),
        metadata=raw.get("metadata"),
    )


class ScenarioSource:
    """Emits Percepts from a YAML scenario file.

    Supports two timing modes:
    - step_based: 'at' values are loop iteration counts (deterministic)
    - relative: 'at' values are wall-clock seconds from scenario start

    Example:
        source = ScenarioSource(Path("scenarios/malware_with_pain.yaml"))
        while not source.is_exhausted():
            percept = source.next_percept()
            if percept:
                bus.publish(percept)
            source.advance_step()  # Call once per loop iteration
    """

    def __init__(self, path: Path) -> None:
        self._definition = load_scenario(path)
        self._start_time = time.time()
        self._current_step: int = 0
        self._next_index: int = 0
        self._tag_times: dict[str, float] = {}

        # Sort percepts by 'at' value
        self._definition.percepts.sort(key=lambda p: p.get("at", 0.0))

        logger.info(
            "ScenarioSource loaded: %s (%d percepts, timing=%s)",
            self._definition.name,
            len(self._definition.percepts),
            self._definition.timing,
        )

    @property
    def name(self) -> str:
        return f"scenario:{self._definition.name}"

    @property
    def capabilities(self) -> set[str]:
        """Derive capabilities from the percept sources in the scenario."""
        return {p.get("source", "cli") for p in self._definition.percepts}

    @property
    def definition(self) -> ScenarioDefinition:
        return self._definition

    @property
    def expectations(self) -> list[Expectation]:
        return self._definition.expectations

    def next_percept(self) -> Percept | None:
        """Return the next due Percept, or None if nothing is due yet."""
        if self._next_index >= len(self._definition.percepts):
            return None

        raw = self._definition.percepts[self._next_index]
        at_value = raw.get("at", 0.0)

        if self._definition.timing == "step_based":
            # Step-based: 'at' is loop iteration count
            if self._current_step < int(at_value):
                return None
        else:
            # Relative: 'at' is seconds from start
            elapsed = time.time() - self._start_time
            if elapsed < at_value:
                return None

        # Emit this percept
        percept = _percept_from_dict(raw, self._start_time)
        self._next_index += 1

        # Track emitted tags for pipeline_continued checks
        tag = (raw.get("metadata") or {}).get("scenario_tag")
        if tag:
            self._tag_times.setdefault(tag, time.time())  # first delivery: actions after it continued the pipeline

        logger.debug(
            "ScenarioSource emitting percept %d/%d (source=%s, at=%s)",
            self._next_index,
            len(self._definition.percepts),
            raw.get("source", "cli"),
            at_value,
        )

        # Simulation verbosity logging
        from maxim.simulation.sim_logger import sim_percept

        summary = raw.get("cli_input") or raw.get("transcript_chunk") or raw.get("content") or "signal"
        if len(str(summary)) > 80:
            summary = str(summary)[:77] + "..."
        sim_percept(
            source=raw.get("source", "cli"),
            summary=str(summary),
            step=self._current_step,
            salience=raw.get("salience", 0),
        )

        return percept

    def advance_step(self) -> None:
        """Advance the step counter (call once per loop iteration)."""
        self._current_step += 1

    def is_exhausted(self) -> bool:
        """True when all percepts have been emitted."""
        return self._next_index >= len(self._definition.percepts)

    @property
    def tag_times(self) -> dict[str, float]:
        """When each tag first reached the agent (``time.time()``, the clock ``ActionRecord.timestamp`` uses)."""
        return dict(self._tag_times)

    @property
    def emitted_tags(self) -> set[str]:
        """Tags from emitted percepts (for pipeline_continued validation)."""
        return set(self._tag_times)
