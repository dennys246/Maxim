"""Research Orchestrator — coordinates Researcher, Writer, and Reviewer agents.

Sequences the three agents through the research protocol:
1. Researcher runs experiments via simulation tools
2. Writer produces a structured paper from experiment data
3. Reviewer validates claims and requests revisions (max 3 rounds)

CLI: maxim --sim research --goal "..." [--campaign <yaml>] [--aut-model <model>]
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from maxim.mesh.bus import LocalMessageBus

logger = logging.getLogger(__name__)

MAX_REVISION_ROUNDS = 3


@dataclass
class ResearchResult:
    """Result from a completed research protocol run."""

    goal: str
    experiments_count: int = 0
    paper_path: str = ""
    review_verdict: str = "not_reviewed"
    review_confidence: float = 0.0
    revision_rounds: int = 0
    duration_s: float = 0.0
    cost_usd: float = 0.0
    aut_model: str = ""
    orchestrator_model: str = ""
    session_id: str = ""


def start_research_mode(
    goal: str,
    campaign: str | None = None,
    language_model: str | None = None,
    aut_model: str | None = None,
    debug: bool = False,
    sandbox_backend: str = "auto",
    max_turns: int = 50,
) -> ResearchResult:
    """Boot research mode: Researcher → Writer → Reviewer → revision loop.

    This is the main entry point called from cli.py when --sim research is used.

    Args:
        goal: The research question (e.g., "hippocampal recall under interference")
        campaign: Path to campaign YAML(s) for the researcher to run
        language_model: LLM profile for orchestrator/research agents
        aut_model: Separate LLM for the agent-under-test (dual-LLM mode)
        debug: Enable verbose tracing
        sandbox_backend: Sandbox type for sim runs (auto/tmpdir/docker)
        max_turns: Max simulation turns per experiment

    Returns:
        ResearchResult with protocol summary
    """
    from maxim.simulation.orchestrator import start_simulation_mode
    from maxim.simulation.research_tools import ExperimentLog
    from maxim.simulation.research_agents import WriterAgent, ReviewerAgent

    start_time = time.time()
    session_id = time.strftime("%Y%m%d_%H%M%S")
    session_dir = Path("data") / "sim_reports" / f"research_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    # ── Setup ────────────────────────────────────────────────────────────
    bus = LocalMessageBus()
    experiment_log = ExperimentLog(session_dir=session_dir, agent_nickname="researcher")

    # Build the LLM router for Writer + Reviewer (research agents)
    research_llm = None
    try:
        from maxim.models.language.router import LLMRouter, load_llm_config
        from maxim.runtime.lane_backends import build_primary_router

        research_llm, _ = build_primary_router(logger=logger)
        if research_llm is None:
            config = load_llm_config()
            if config.enabled:
                research_llm = LLMRouter(config)
        if research_llm is not None:
            research_llm.warmup()
            research_llm.wait_ready(timeout=120.0)
    except Exception as e:
        logger.warning("Research LLM init failed: %s — Writer/Reviewer will use heuristics", e)

    # ── Phase 1: Researcher ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  RESEARCH PROTOCOL")
    print(f"  Goal: {goal}")
    if campaign:
        print(f"  Campaign: {campaign}")
    if aut_model:
        print(f"  AUT model: {aut_model}")
    print(f"{'=' * 60}\n")

    print("  Phase 1: Running experiments (Researcher agent)...")

    # Load campaign YAML and extract percept texts for the researcher
    campaign_turns: list[dict] = []
    if campaign:
        try:
            from maxim.simulation.scenario_source import load_scenario
            from pathlib import Path as _Path

            scenario = load_scenario(_Path(campaign))
            for p in scenario.percepts:
                text = p.get("cli_input", "")
                if text:
                    campaign_turns.append(
                        {
                            "text": text.strip(),
                            "phase": p.get("metadata", {}).get("phase", ""),
                            "role": p.get("metadata", {}).get("experiment_role", ""),
                            "tag": p.get("metadata", {}).get("scenario_tag", ""),
                        }
                    )
            print(f"  Loaded {len(campaign_turns)} campaign turns from {campaign}")
        except Exception as e:
            logger.warning("Failed to load campaign YAML: %s", e)

    # Build the researcher goal with concrete campaign steps
    researcher_goal = goal
    if campaign_turns:
        turn_lines = []
        for i, turn in enumerate(campaign_turns, 1):
            phase_label = f" [{turn['phase']}]" if turn["phase"] else ""
            # Show full text (up to 500 chars) so the LLM has the complete narrative
            turn_lines.append(f"--- TURN {i}{phase_label} ---\n{turn['text'][:500]}")
        turns_block = "\n\n".join(turn_lines)
        researcher_goal = (
            f"{goal}\n\n"
            f"CAMPAIGN PROTOCOL\n"
            f"=================\n"
            f"You MUST deliver the following narrative turns to the AUT, in order.\n"
            f'For each turn, call: send_message(text="<the narrative text below>")\n\n'
            f"IMPORTANT RULES:\n"
            f"- Send the narrative text VERBATIM — do NOT paraphrase, summarize, or adapt it.\n"
            f"- Do NOT invent your own probes or adversarial variations.\n"
            f"- Do NOT skip turns or reorder them.\n"
            f"- Wait for the AUT response after each turn before sending the next.\n\n"
            f"{turns_block}\n\n"
            f"--- END OF CAMPAIGN TURNS ---\n\n"
            f"After ALL {len(campaign_turns)} turns are sent and responses collected:\n"
            f"1. Use inspect_aut(query='memory_recall', params={{'keyword': 'Verath'}}) to check memory survival\n"
            f"2. Use inspect_aut(query='system_stats') for graph topology\n"
            f"3. Record results with record_experiment\n"
            f"4. Call finish_simulation with your findings"
        )
    elif campaign:
        researcher_goal = f"{goal}\n\nCampaign file: {campaign} (failed to load — run manually with send_message)"

    # Run the researcher via the existing simulation orchestrator
    sim_result = start_simulation_mode(
        goal=researcher_goal,
        persona="researcher",
        max_turns=max_turns,
        debug=debug,
        sandbox_backend=sandbox_backend,
        aut_model=aut_model,
        no_sim_env=True,  # Research mode doesn't need pain-triggering files
    )

    # The experiment log was populated by the researcher's record_experiment calls
    # during the sim run. Reload it from disk (the sim writes to sim_tmpdir,
    # and the report copies it to the session dir).
    # Check multiple locations for the experiments file
    exp_count = len(experiment_log)
    if exp_count == 0:
        # Try loading from the sim report directory
        sim_reports = Path("data") / "sim_reports"
        if sim_reports.exists():
            for d in sorted(sim_reports.iterdir(), reverse=True):
                exp_file = d / "experiments.jsonl"
                if exp_file.exists() and d.name != f"research_{session_id}":
                    experiment_log = ExperimentLog(session_dir=d, agent_nickname="researcher")
                    exp_count = len(experiment_log)
                    if exp_count > 0:
                        logger.info("Loaded %d experiments from %s", exp_count, d)
                        break

    print(f"  Researcher completed: {exp_count} experiments recorded")
    print(f"  Sim result: {sim_result.turns} turns, {sim_result.total_actions} actions")

    # ── Phase 2: Writer ──────────────────────────────────────────────────
    print("\n  Phase 2: Writing paper (Writer agent)...")

    writer = WriterAgent(
        llm=research_llm,
        experiment_log=experiment_log,
        bus=bus,
        session_dir=session_dir,
    )
    draft = writer.run(goal)
    sections_written = len(draft.sections)
    print(f"  Writer completed: {sections_written} sections written")

    # ── Phase 3: Reviewer ────────────────────────────────────────────────
    print("\n  Phase 3: Peer review (Reviewer agent)...")

    reviewer = ReviewerAgent(
        llm=research_llm,
        experiment_log=experiment_log,
        bus=bus,
        session_dir=session_dir,
    )

    review = reviewer.review(draft, goal)
    revision_round = 0

    # ── Revision loop ────────────────────────────────────────────────────
    while review.verdict == "revise" and revision_round < MAX_REVISION_ROUNDS:
        revision_round += 1
        print(f"\n  Revision round {revision_round}/{MAX_REVISION_ROUNDS}...")
        print(f"    Issues: {len(review.issues)}, Requests: {len(review.revision_requests)}")

        # Writer revises based on feedback
        draft = writer.revise(review)
        print(f"    Writer revised {len(review.section_feedback)} sections")

        # Reviewer re-evaluates
        review = reviewer.review(draft, goal)
        print(f"    Reviewer verdict: {review.verdict} (confidence: {review.confidence:.2f})")

    # ── Final report ─────────────────────────────────────────────────────
    duration = time.time() - start_time
    cost = 0.0
    if research_llm is not None:
        cost = getattr(research_llm, "session_cost", 0.0) or 0.0

    result = ResearchResult(
        goal=goal,
        experiments_count=exp_count,
        paper_path=str(draft.output_path) if draft.output_path else "",
        review_verdict=review.verdict,
        review_confidence=review.confidence,
        revision_rounds=revision_round,
        duration_s=duration,
        cost_usd=cost,
        aut_model=aut_model or "",
        orchestrator_model=language_model or "",
        session_id=session_id,
    )

    # Save protocol result
    result_path = session_dir / "research_result.json"
    result_path.write_text(
        json.dumps(
            {
                "goal": result.goal,
                "experiments_count": result.experiments_count,
                "paper_path": result.paper_path,
                "review_verdict": result.review_verdict,
                "review_confidence": result.review_confidence,
                "revision_rounds": result.revision_rounds,
                "duration_s": round(result.duration_s, 1),
                "cost_usd": round(result.cost_usd, 4),
                "aut_model": result.aut_model,
                "orchestrator_model": result.orchestrator_model,
                "session_id": result.session_id,
                "bus_messages": bus.message_count,
                "review_issues": review.issues,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Save bus message history
    bus_path = session_dir / "bus_history.json"
    bus_history = bus.get_history()
    bus_path.write_text(
        json.dumps(
            [m.to_dict() for m in bus_history],
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print("  RESEARCH PROTOCOL COMPLETE")
    print(f"  Goal: {goal}")
    print(f"  Experiments: {exp_count}")
    print(f"  Paper: {draft.output_path}")
    print(f"  Verdict: {review.verdict} (confidence: {review.confidence:.2f})")
    print(f"  Revisions: {revision_round}")
    print(f"  Duration: {duration:.1f}s")
    if cost > 0:
        print(f"  Cost: ${cost:.4f}")
    print(f"  Session: {session_dir}")
    print(f"{'=' * 60}\n")

    return result
