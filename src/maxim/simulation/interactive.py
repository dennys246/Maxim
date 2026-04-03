"""Interactive simulation REPL — describe scenarios in natural language, run them live.

Boots the full agentic pipeline once, then loops: user types a description,
the LLM generates a YAML scenario, and it runs through the already-warm
pipeline. No cold start penalty after first load.

Usage:
    maxim --sim interactive --language-model mistral-7b
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def run_interactive_sim(
    agentic_agent: Any,
    env: Any,
    state: Any,
    memory: Any,
    decision_engine: Any,
    executor: Any,
    *,
    autonomy_controller: Any = None,
    llm_worker: Any = None,
    hippocampus: Any = None,
    memory_hub: Any = None,
    evaluators: list | None = None,
    pain_bus: Any = None,
    llm_profile: str | None = None,
    sim_workspace: Path | None = None,
) -> None:
    """Run the interactive simulation REPL.

    Args:
        All args match run_agentic_loop() — the pipeline is pre-bootstrapped.
        llm_profile: Which LLM profile to use for scenario generation.
        sim_workspace: Directory for sim logs and temp files.
    """
    from maxim.simulation.scenario_source import ScenarioSource
    from maxim.simulation.sim_logger import (
        enable_sim_logging,
        disable_sim_logging,
        sim_log,
        sim_result,
    )
    from maxim.simulation.simulation_generator import generate_scenario
    from maxim.simulation.sinks import RecordingSink
    from maxim.simulation.validation import validate_expectations
    from maxim.runtime.agent_loop import run_agentic_loop

    if sim_workspace is None:
        sim_workspace = Path("data/sim_sandbox")
    sim_workspace.mkdir(parents=True, exist_ok=True)

    print("\n  Interactive Simulation Mode")
    print("  Type a scenario description to generate and run it.")
    print("  Commands: 'quit' to exit, 'last' to rerun last scenario\n")

    last_scenario_path: Path | None = None
    run_count = 0

    while True:
        try:
            description = input("  sim> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting interactive simulation.")
            break

        if not description:
            continue
        if description.lower() in ("quit", "exit", "q"):
            print("  Exiting interactive simulation.")
            break

        # Handle 'last' command
        if description.lower() == "last":
            if last_scenario_path is None or not last_scenario_path.exists():
                print("  No previous scenario to rerun.")
                continue
            scenario_path = last_scenario_path
            print(f"  Rerunning: {scenario_path.name}")
        else:
            # Generate scenario from description
            print("  Generating scenario...")
            try:
                scenario_path = sim_workspace / f"interactive_{run_count:03d}.yaml"
                generate_scenario(
                    description,
                    output_path=scenario_path,
                    llm_profile=llm_profile,
                )
                last_scenario_path = scenario_path
            except Exception as e:
                print(f"  Failed to generate scenario: {e}")
                continue

        # Set up for this run
        run_count += 1
        log_path = str(sim_workspace / f"sim_log_interactive_{run_count:03d}.jsonl")
        enable_sim_logging(log_path=log_path)

        # Create a temp sandbox CWD for this run
        run_tmpdir = Path(tempfile.mkdtemp(
            prefix=f"sim_run_{run_count:03d}_",
            dir=str(sim_workspace),
        ))
        original_cwd = os.getcwd()
        os.chdir(str(run_tmpdir))

        try:
            source = ScenarioSource(scenario_path)
            sink = RecordingSink()

            print(f"  Running: {source.definition.name} "
                  f"({len(source.definition.percepts)} percepts, "
                  f"{len(source.expectations)} expectations)")

            # Run through the already-warm pipeline
            import threading
            stop = threading.Event()

            run_agentic_loop(
                agentic_agent,
                env,
                state,
                memory,
                decision_engine,
                executor,
                autonomy_controller=autonomy_controller,
                llm_worker=llm_worker,
                hippocampus=hippocampus,
                memory_hub=memory_hub,
                evaluators=evaluators or [],
                max_steps=200,
                run_id=f"sim_interactive_{run_count:03d}",
                stop_event=stop,
                percept_source=source,
                action_sink=sink,
                pain_bus=pain_bus,
            )

            # Validate expectations
            results = validate_expectations(
                expectations=source.expectations,
                sink=sink,
                hippocampus=hippocampus,
                emitted_tags=source.emitted_tags,
            )

            passed = all(r.passed for r in results)
            met = sum(1 for r in results if r.passed)
            failed = sum(1 for r in results if not r.passed)

            # Report
            print()
            sim_result(source.definition.name, passed, met, failed)
            for r in results:
                status = "PASS" if r.passed else "FAIL"
                desc = r.expectation.description or r.expectation.type
                print(f"    [{status}] {desc}")
                if not r.passed and r.detail:
                    print(f"           {r.detail}")

            print(f"\n  Actions: {len(sink.actions)}")
            for a in sink.actions[:10]:
                tag = "[BLOCKED]" if a.blocked else ("[OK]" if a.result_success else "[FAIL]")
                print(f"    {tag} {a.tool_name}")
            if len(sink.actions) > 10:
                print(f"    ... and {len(sink.actions) - 10} more")

        except Exception as e:
            print(f"  Scenario execution failed: {e}")
            logger.exception("Interactive sim run failed")
        finally:
            saved_log = disable_sim_logging()
            if saved_log:
                print(f"  Log: {saved_log}")

            # Cleanup sandbox
            os.chdir(original_cwd)
            try:
                import shutil
                shutil.rmtree(str(run_tmpdir))
            except Exception:
                pass

        print()  # Blank line before next prompt
