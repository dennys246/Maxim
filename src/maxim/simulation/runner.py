"""ScenarioRunner — orchestrates scenario execution and validation.

Loads a scenario YAML, creates a ScenarioSource, runs the agent loop
with an InstrumentedExecutor and RecordingSink, then validates
expectations against the recorded actions and memory state.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from maxim.simulation.scenario_source import ScenarioSource
from maxim.simulation.sinks import RecordingSink
from maxim.simulation.validation import ScenarioResult, validate_expectations

if TYPE_CHECKING:
    from maxim.memory.hippocampus import Hippocampus

logger = logging.getLogger(__name__)


class ScenarioRunner:
    """Run a scenario YAML and validate expectations.

    Example:
        runner = ScenarioRunner(Path("scenarios/malware_with_pain.yaml"))
        result = runner.run(hippocampus=hippo)
        assert result.passed, result.expectations_failed
    """

    def __init__(self, scenario_path: Path) -> None:
        self._path = scenario_path
        self._source: ScenarioSource | None = None
        self._sink: RecordingSink | None = None

    def run(
        self,
        hippocampus: Hippocampus | None = None,
        max_steps: int = 100,
        pain_bus: Any | None = None,
    ) -> ScenarioResult:
        """Execute scenario and return validated results.

        This is a standalone runner that processes percepts sequentially
        without the full agent loop. For full integration testing,
        use run_agentic_loop() with percept_source parameter.

        Args:
            hippocampus: Optional hippocampus for memory_formed checks.
            max_steps: Maximum loop iterations (safety bound).
            pain_bus: Optional PainBus for routing pain percepts.
        """
        start = time.time()
        self._source = ScenarioSource(self._path)
        self._sink = RecordingSink()

        # Process percepts step by step
        for step in range(max_steps):
            if self._source.is_exhausted():
                break

            percept = self._source.next_percept()
            if percept is not None:
                # Route pain percepts through PainBus if available
                if pain_bus is not None:
                    from maxim.proprioception.pain_bus import route_pain_percept

                    route_pain_percept(percept, pain_bus)

                logger.debug(
                    "Step %d: percept source=%s, cli_input=%s",
                    step,
                    percept.source,
                    percept.cli_input,
                )

            self._source.advance_step()

        duration = time.time() - start
        exit_reason = "exhausted" if self._source.is_exhausted() else "max_steps"

        # Validate expectations
        results = validate_expectations(
            expectations=self._source.expectations,
            sink=self._sink,
            hippocampus=hippocampus,
            emitted_tags=self._source.emitted_tags,
        )

        passed = all(r.passed for r in results)

        return ScenarioResult(
            scenario_name=self._source.definition.name,
            passed=passed,
            results=results,
            actions=self._sink.actions,
            exit_reason=exit_reason,
            duration=duration,
        )

    @property
    def source(self) -> ScenarioSource | None:
        return self._source

    @property
    def sink(self) -> RecordingSink | None:
        return self._sink


def run_scenario(
    path: Path,
    hippocampus: Hippocampus | None = None,
    pain_bus: Any | None = None,
    max_steps: int = 100,
) -> ScenarioResult:
    """Convenience function to run a scenario and return results."""
    runner = ScenarioRunner(path)
    return runner.run(hippocampus=hippocampus, pain_bus=pain_bus, max_steps=max_steps)
