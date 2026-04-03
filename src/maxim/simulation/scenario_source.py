"""ScenarioSource — YAML-driven percept sequences for testing.

Reads a scenario file and emits Percepts on schedule, supporting
both wall-clock (relative) and loop-iteration (step_based) timing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from maxim.agents.bus import Percept

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


@dataclass
class ScenarioDefinition:
    """Parsed scenario from YAML."""

    name: str
    description: str = ""
    timing: str = "step_based"  # "relative" or "step_based"
    percepts: list[dict[str, Any]] = field(default_factory=list)
    expectations: list[Expectation] = field(default_factory=list)


def load_scenario(path: Path) -> ScenarioDefinition:
    """Load and parse a scenario YAML file."""
    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
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
            )
        )

    percepts = raw.get("percepts", [])
    if not percepts:
        logger.warning("Scenario %s has no percepts — will complete immediately", path.stem)

    return ScenarioDefinition(
        name=raw.get("name", path.stem),
        description=raw.get("description", ""),
        timing=raw.get("timing", "step_based"),
        percepts=percepts,
        expectations=expectations,
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
        self._emitted_tags: set[str] = set()

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
            self._emitted_tags.add(tag)

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
    def emitted_tags(self) -> set[str]:
        """Tags from emitted percepts (for pipeline_continued validation)."""
        return set(self._emitted_tags)
