"""Interactive simulation REPL — conversational mode with LLM-generated continuations.

Boots the full agentic pipeline once, then loops: user types input,
the LLM generator produces contextually-aware follow-up percepts
based on the full conversation history, appends them to the growing
YAML transcript, and runs them through the pipeline.

Each follow-up sees:
- The full prior YAML scenario (what percepts were sent)
- Maxim's responses (what actions it took)
- The user's new input

This produces coherent multi-turn simulations where percepts build
on each other naturally.

Usage:
    maxim --sim interactive --language-model mistral-7b
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# System prompt for generating continuation percepts with prior context
CONTINUATION_PROMPT = """\
You are continuing an interactive simulation for a robotics testing framework.

You will receive:
1. The PRIOR SCENARIO so far (percepts already sent)
2. MAXIM'S RESPONSES (actions the robot took)
3. A NEW USER INPUT describing what happens next

Generate 1-3 NEW percepts that continue the scenario based on the user's input.
These percepts should be contextually aware of what happened before.

PERCEPT TYPES:
- "cli": User text input. Use cli_input field.
- "transcript": User speech. Use transcript_chunk field.
- "vision": Robot sees something. Use detections field.
- "proprioception": Body signal. Set content="pain_signal" with metadata.
- "comms": External message. Use content field.

Return ONLY valid JSON with this format:
{
  "percepts": [
    {
      "at": 0,
      "source": "cli",
      "cli_input": "the user's follow-up",
      "salience": 0.8,
      "novelty": 0.5,
      "metadata": {"scenario_tag": "followup_1"}
    }
  ]
}

RULES:
- Generate 1-3 percepts (keep it focused)
- The FIRST percept MUST be "cli" or "transcript" with text input
  so the LLM agent has something to respond to
- Use step_based timing starting at 0
- Set salience/novelty appropriately for the context
- Include scenario_tags in metadata
- Pain percepts MUST have source="proprioception", content="pain_signal"
- Return ONLY JSON, no prose"""


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
    """Run the interactive simulation REPL with LLM-generated continuations."""
    from maxim.simulation.scenario_source import ScenarioSource, ScenarioDefinition
    from maxim.simulation.sim_logger import (
        enable_sim_logging,
        disable_sim_logging,
        sim_log,
    )
    from maxim.simulation.simulation_generator import generate_scenario, warm_generator, _extract_json
    from maxim.simulation.sinks import RecordingSink
    from maxim.runtime.agent_loop import run_agentic_loop

    if sim_workspace is None:
        sim_workspace = Path("data/sim_sandbox")
    sim_workspace.mkdir(parents=True, exist_ok=True)

    # Pre-load the generator and continuation LLMs so first input is fast
    print("  Pre-loading simulation generator...")
    warm_generator(llm_profile)
    print("  Generator ready.")

    print("\n  Interactive Simulation Mode")
    print("  Type a scenario description or follow-up input.")
    print("  Each input generates contextually-aware percepts.")
    print("  Commands:")
    print("    /new          Start new conversation")
    print("    /save [path]  Save full transcript as YAML")
    print("    /status       Show conversation stats")
    print("    quit          Exit\n")

    conversation_count = 0

    while True:
        conversation_count += 1
        transcript_path = (sim_workspace / f"conversation_{conversation_count:03d}.yaml").resolve()
        log_path = str(sim_workspace / f"sim_log_conv_{conversation_count:03d}.jsonl")

        enable_sim_logging(log_path=log_path)
        sim_log("PIPELINE", f"Conversation {conversation_count} started")

        # Conversation state
        all_percepts: list[dict[str, Any]] = []
        all_actions: list[str] = []  # Summary of Maxim's responses
        total_sink = RecordingSink()
        turn = 0
        restart = False

        # First input — generate initial scenario
        try:
            description = input("  sim> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting.")
            break

        if not description or description.lower() in ("quit", "exit", "q"):
            break

        # Generate initial scenario
        sim_log("PIPELINE", f"Generating initial scenario from: {description[:60]}")
        print("  Generating scenario...")
        try:
            yaml_str = generate_scenario(description, output_path=transcript_path, llm_profile=llm_profile)
            # Show the generated percepts
            try:
                generated = yaml.safe_load(yaml_str) or {}
                for p in generated.get("percepts", []):
                    src = p.get("source", "?")
                    text = p.get("cli_input") or p.get("transcript_chunk") or p.get("content") or "signal"
                    print(f"    [{src}] {str(text)[:70]}")
                exps = generated.get("expectations", [])
                if exps:
                    print(f"    ({len(exps)} expectation(s))")
            except Exception:
                pass
        except Exception as e:
            print(f"  Failed to generate scenario: {e}")
            disable_sim_logging()
            continue

        # Run initial scenario
        print("  Running through agent pipeline...")
        sim_log("PIPELINE", "Running initial scenario turn")
        _run_scenario_turn(
            transcript_path, agentic_agent, env, state, memory,
            decision_engine, executor,
            autonomy_controller=autonomy_controller,
            llm_worker=llm_worker,
            hippocampus=hippocampus,
            memory_hub=memory_hub,
            evaluators=evaluators,
            pain_bus=pain_bus,
            sim_workspace=sim_workspace,
            sink=total_sink,
            turn=turn,
        )

        # Show what Maxim did
        sim_log("PIPELINE", f"Initial turn complete ({len(total_sink.actions)} actions)")
        _show_turn_results(total_sink, from_index=0)
        print("\n  Simulated, what happens next?")

        # Record what happened
        try:
            with open(transcript_path) as f:
                initial = yaml.safe_load(f) or {}
            all_percepts.extend(initial.get("percepts", []))
        except Exception:
            pass
        all_actions.extend(_summarize_actions(total_sink))
        turn += 1

        # Follow-up loop
        while True:
            try:
                user_input = input("  sim> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Exiting.")
                _save_full_transcript(transcript_path, all_percepts, description)
                disable_sim_logging()
                return

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                break

            if user_input.lower() == "/new":
                restart = True
                break

            if user_input.lower().startswith("/save"):
                parts = user_input.split(maxsplit=1)
                save_path = Path(parts[1]).resolve() if len(parts) > 1 else transcript_path
                _save_full_transcript(save_path, all_percepts, description)
                print(f"  Saved: {save_path}")
                continue

            if user_input.lower() == "/status":
                print(f"  Conversation: {conversation_count}, Turn: {turn}")
                print(f"  Percepts sent: {len(all_percepts)}")
                print(f"  Actions recorded: {len(total_sink.actions)}")
                if hippocampus:
                    print(f"  Hippocampus memories: {len(hippocampus)}")
                continue

            # Generate continuation percepts with context
            sim_log("PIPELINE", f"Generating continuation from: {user_input[:60]}")
            print("  Generating continuation...")
            new_percepts = _generate_continuation(
                user_input, all_percepts, all_actions, llm_profile
            )

            if not new_percepts:
                # Fallback: just inject the raw text as a CLI percept
                print("    (Generator returned no percepts, using raw input)")
                new_percepts = [{
                    "at": 0,
                    "source": "cli",
                    "cli_input": user_input,
                    "salience": 0.7,
                    "novelty": 0.5,
                    "metadata": {"scenario_tag": f"followup_{turn}"},
                }]

            # Show what was generated
            for p in new_percepts:
                src = p.get("source", "?")
                text = p.get("cli_input") or p.get("transcript_chunk") or p.get("content") or "signal"
                print(f"    [{src}] {str(text)[:70]}")

            # Append to conversation history
            all_percepts.extend(new_percepts)

            # Write as a temporary scenario file for this turn
            turn_path = (sim_workspace / f"conv_{conversation_count:03d}_turn_{turn:03d}.yaml").resolve()
            turn_scenario = {
                "name": f"turn_{turn}",
                "description": user_input,
                "timing": "step_based",
                "percepts": new_percepts,
                "expectations": [],
            }
            turn_path.write_text(yaml.dump(turn_scenario, default_flow_style=False, sort_keys=False))

            # Run this turn
            _run_scenario_turn(
                turn_path, agentic_agent, env, state, memory,
                decision_engine, executor,
                autonomy_controller=autonomy_controller,
                llm_worker=llm_worker,
                hippocampus=hippocampus,
                memory_hub=memory_hub,
                evaluators=evaluators,
                pain_bus=pain_bus,
                sim_workspace=sim_workspace,
                sink=total_sink,
                turn=turn,
            )

            # Show what Maxim did this turn
            _show_turn_results(total_sink, from_index=len(all_actions))
            all_actions.extend(_summarize_actions(total_sink, from_index=len(all_actions)))
            turn += 1
            print("\n  Simulated, what happens next?")

            # Clean up turn file
            try:
                turn_path.unlink()
            except Exception:
                pass

        # End of conversation — run consolidation now
        if memory_hub is not None:
            try:
                print("  Consolidating memories...")
                session_stats = memory_hub.on_session_end()
                sim_log("PIPELINE", f"Session ended: {session_stats}")
            except Exception as e:
                logger.debug("Failed to end MemoryHub session: %s", e)

        _save_full_transcript(transcript_path, all_percepts, description)
        _print_summary(total_sink)
        print(f"  Transcript: {transcript_path}")

        saved_log = disable_sim_logging()
        if saved_log:
            print(f"  Log: {saved_log}")

        if not restart:
            break

        print()  # Blank line before next conversation


def _run_scenario_turn(
    scenario_path: Path,
    agent: Any, env: Any, state: Any, memory: Any,
    decision_engine: Any, executor: Any,
    *,
    sink: Any,
    turn: int,
    **kwargs: Any,
) -> None:
    """Run a single turn's scenario through the agent loop."""
    from maxim.simulation.scenario_source import ScenarioSource
    from maxim.runtime.agent_loop import run_agentic_loop

    # Create sandbox for this turn
    sim_workspace = kwargs.pop("sim_workspace", Path("data/sim_sandbox"))
    run_tmpdir = Path(tempfile.mkdtemp(
        prefix=f"sim_turn_{turn:03d}_",
        dir=str(sim_workspace),
    ))
    original_cwd = os.getcwd()
    os.chdir(str(run_tmpdir))

    try:
        source = ScenarioSource(scenario_path)
        stop = threading.Event()

        run_agentic_loop(
            agent, env, state, memory, decision_engine, executor,
            max_steps=0,  # Unlimited — stopped by grace period or stop_event
            run_id=f"sim_turn_{turn:03d}",
            stop_event=stop,
            percept_source=source,
            action_sink=sink,
            **kwargs,
        )
    except Exception as e:
        logger.exception("Turn %d failed: %s", turn, e)
        print(f"  Turn failed: {e}")
    finally:
        os.chdir(original_cwd)
        try:
            import shutil
            shutil.rmtree(str(run_tmpdir))
        except Exception:
            pass


# Cached generator agent — avoids reloading the model every continuation
_continuation_agent = None
_continuation_profile = None


def _generate_continuation(
    user_input: str,
    prior_percepts: list[dict],
    prior_actions: list[str],
    llm_profile: str | None,
) -> list[dict] | None:
    """Use the LLM to generate contextual follow-up percepts."""
    global _continuation_agent, _continuation_profile
    from maxim.agents.llm_agent import LLMAgent
    from maxim.simulation.simulation_generator import _extract_json, _clean_percepts

    # Build context for the generator
    context = "PRIOR SCENARIO PERCEPTS:\n"
    for p in prior_percepts[-10:]:  # Last 10 percepts for context
        src = p.get("source", "?")
        text = p.get("cli_input") or p.get("transcript_chunk") or p.get("content") or ""
        context += f"  [{src}] {text}\n"

    context += "\nMAXIM'S RESPONSES:\n"
    for a in prior_actions[-5:]:  # Last 5 actions for context
        context += f"  {a}\n"

    context += f"\nNEW USER INPUT:\n  {user_input}\n"
    context += "\nGenerate follow-up percepts for this continuation."

    # Reuse cached agent (same profile) to avoid reloading the model
    if _continuation_agent is None or _continuation_profile != llm_profile:
        _continuation_agent = LLMAgent(
            profile=llm_profile,
            system_prompt=CONTINUATION_PROMPT,
            temperature=0.2,
            max_tokens=1024,
        )
        _continuation_profile = llm_profile

    raw = _continuation_agent.generate(context, max_tokens=1024)
    result = _extract_json(raw)

    if result and "percepts" in result:
        return _clean_percepts(result["percepts"])

    return None


def _summarize_actions(sink: Any, from_index: int = 0) -> list[str]:
    """Summarize recent actions as strings for context."""
    summaries = []
    for a in sink.actions[from_index:]:
        tag = "[BLOCKED]" if a.blocked else ("[OK]" if a.result_success else "[FAIL]")
        name = a.tool_name
        preview = ""
        if a.result_output and isinstance(a.result_output, dict):
            msg = a.result_output.get("message", "")
            if msg:
                preview = f": {str(msg)[:80]}"
        summaries.append(f"{tag} {name}{preview}")
    return summaries


def _save_full_transcript(path: Path, percepts: list[dict], description: str) -> None:
    """Save the full conversation as a replayable YAML scenario."""
    scenario = {
        "name": "interactive_conversation",
        "description": description,
        "timing": "step_based",
        "percepts": percepts,
        "expectations": [],
    }
    # Re-number steps sequentially
    for i, p in enumerate(scenario["percepts"]):
        p["at"] = i * 2

    try:
        path.write_text(yaml.dump(scenario, default_flow_style=False, sort_keys=False))
    except Exception:
        pass


def _show_turn_results(sink: Any, from_index: int = 0) -> None:
    """Show what Maxim did during this turn."""
    new_actions = sink.actions[from_index:]
    if not new_actions:
        print("  (No actions this turn — LLM may still be processing)")
        return
    for a in new_actions:
        tag = "[BLOCKED]" if a.blocked else ("[OK]" if a.result_success else "[FAIL]")
        name = a.tool_name
        preview = ""
        if a.result_output and isinstance(a.result_output, dict):
            msg = a.result_output.get("message", "")
            if msg:
                preview = f": {str(msg)[:70]}"
        elif a.block_reason:
            preview = f": {str(a.block_reason)[:70]}"
        print(f"  {tag} {name}{preview}")


def _print_summary(sink: Any) -> None:
    """Print conversation summary."""
    print(f"\n  Actions: {len(sink.actions)}")
    for a in sink.actions[:15]:
        tag = "[BLOCKED]" if a.blocked else ("[OK]" if a.result_success else "[FAIL]")
        name = a.tool_name
        preview = ""
        if a.result_output and isinstance(a.result_output, dict):
            msg = a.result_output.get("message", "")
            if msg:
                preview = f": {str(msg)[:60]}"
        print(f"    {tag} {name}{preview}")
    if len(sink.actions) > 15:
        print(f"    ... and {len(sink.actions) - 15} more")
