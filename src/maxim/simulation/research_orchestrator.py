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
    finish_reason: str = ""


def _persist_research_result(
    result: ResearchResult,
    session_dir: Path,
    bus: LocalMessageBus,
    *,
    review_issues: list[str] | None = None,
) -> None:
    """Persist the protocol result and bus history for every terminal path."""
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
                "finish_reason": result.finish_reason,
                "bus_messages": bus.message_count,
                "review_issues": review_issues or [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    bus_path = session_dir / "bus_history.json"
    bus_history = bus.get_history()
    bus_path.write_text(
        json.dumps(
            [message.to_dict() for message in bus_history],
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


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

    from maxim.simulation.sim_logger import display_status, display_summary

    # ── Phase 1: Researcher ──────────────────────────────────────────────
    # separator removed — display tier handles formatting
    display_status("RESEARCH PROTOCOL")
    display_status(f"Goal: {goal}")
    if campaign:
        display_status(f"Campaign: {campaign}")
    if aut_model:
        display_status(f"AUT model: {aut_model}")
    # separator removed

    display_status("Phase 1: Running experiments...")

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
                            "salience": p.get("salience", 0.8),
                            "novelty": p.get("novelty", 0.7),
                        }
                    )
            display_status(f"Loaded {len(campaign_turns)} campaign turns from {campaign}")
        except Exception as e:
            logger.warning("Failed to load campaign YAML: %s", e)

    # Build the researcher goal — for non-campaign runs only.
    # Campaign runs bypass the orchestrator LLM entirely.
    researcher_goal = goal
    if not campaign_turns and campaign:
        researcher_goal = f"{goal}\n\nCampaign file: {campaign} (failed to load — run manually with send_message)"

    # Run the simulation. Campaign turns are injected directly through
    # the bridge, and post-campaign analysis runs programmatically.
    # The orchestrator LLM is only used for non-campaign simulations.
    # Pass our experiment_log so the orchestrator's research tools write to
    # the same log the Writer/Reviewer will read from (D-0a fix).
    sim_result = start_simulation_mode(
        goal=researcher_goal,
        mode="research",
        max_turns=max_turns,
        debug=debug,
        sandbox_backend=sandbox_backend,
        aut_model=aut_model,
        no_sim_env=True,
        pre_campaign_turns=campaign_turns if campaign_turns else None,
        experiment_log=experiment_log,
    )

    from maxim.simulation.sim_types import is_simulation_run_failure

    sim_finish_reason = str(sim_result.finish_reason or "")
    if is_simulation_run_failure(sim_finish_reason):
        duration = time.time() - start_time
        result = ResearchResult(
            goal=goal,
            review_verdict="aborted",
            duration_s=duration,
            aut_model=aut_model or "",
            orchestrator_model=language_model or "",
            session_id=session_id,
            finish_reason=sim_finish_reason,
        )
        _persist_research_result(
            result,
            session_dir,
            bus,
            review_issues=[f"Underlying simulation was unusable: {sim_finish_reason}"],
        )
        display_summary(
            [
                "RESEARCH PROTOCOL ABORTED",
                f"Goal: {goal}",
                f"Simulation finish: {sim_finish_reason}",
                "Writer/reviewer skipped: underlying experiment is not valid evidence",
                f"Session: {session_dir}",
            ]
        )
        return result

    # Record analysis as an experiment — works for both campaign and non-campaign runs (D-0b fix).
    if sim_result.campaign_analysis:
        analysis = sim_result.campaign_analysis
        stats_data = analysis.get("system_stats", {})
        turns_data = analysis.get("turns", [])

        # Extract memory recall data dynamically — check memory_recall dict
        # for any seed keyword results (D-0d fix: no hardcoded keys).
        memory_recall = analysis.get("memory_recall", {})
        recall_summary_parts = []
        any_recall_found = False
        for keyword, recall_data in memory_recall.items():
            found = bool(recall_data) and "error" not in recall_data
            count = recall_data.get("count", 0) if isinstance(recall_data, dict) else 0
            recall_summary_parts.append(f"{keyword}: {'found' if found else 'not found'} ({count} hit(s))")
            if found:
                any_recall_found = True

        hippo_stats = stats_data.get("hippocampus", {}) if isinstance(stats_data, dict) else {}
        mem_count = hippo_stats.get("total_memories", 0)
        graph_edges = hippo_stats.get("graph_edges", 0)

        recall_summary = "; ".join(recall_summary_parts) if recall_summary_parts else "no recall queries"
        n_turns = len(turns_data)

        experiment_log.record(
            hypothesis=f"Simulation goal: {goal}",
            method=f"{'Direct bridge injection of ' + str(n_turns) + ' campaign turns' if n_turns else 'Free-form simulation with orchestrator LLM'}",
            result=f"Memory recall: {recall_summary}. {mem_count} total memories, {graph_edges} graph edges.",
            conclusion=f"AUT formed {mem_count} episodic memories with {graph_edges} associative edges. Recall: {recall_summary}.",
            tags=["auto_recorded", "campaign" if n_turns else "freeform"],
            metrics={
                "any_recall_found": 1.0 if any_recall_found else 0.0,
                "total_memories": mem_count,
                "graph_edges": graph_edges,
                "campaign_turns": n_turns,
                "duration_s": sim_result.duration_s,
            },
        )

    exp_count = len(experiment_log)
    display_status(f"Researcher completed: {exp_count} experiments recorded")
    display_status(f"Sim result: {sim_result.turns} turns, {sim_result.total_actions} actions")

    # ── Phase 2: Writer ──────────────────────────────────────────────────
    display_status("Phase 2: Writing paper...")

    writer = WriterAgent(
        llm=research_llm,
        experiment_log=experiment_log,
        bus=bus,
        session_dir=session_dir,
    )
    draft = writer.run(goal)
    sections_written = len(draft.sections)
    display_status(f"Writer completed: {sections_written} sections written")

    # ── Phase 3: Reviewer ────────────────────────────────────────────────
    display_status("Phase 3: Peer review...")

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
        display_status(f"Revision round {revision_round}/{MAX_REVISION_ROUNDS}...")
        display_status(f"Issues: {len(review.issues)}, Requests: {len(review.revision_requests)}")

        # Writer revises based on feedback
        draft = writer.revise(review)
        display_status(f"Writer revised {len(review.section_feedback)} sections")

        # Reviewer re-evaluates
        review = reviewer.review(draft, goal)
        display_status(f"Reviewer verdict: {review.verdict} (confidence: {review.confidence:.2f})")

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
        finish_reason=sim_finish_reason,
    )

    _persist_research_result(result, session_dir, bus, review_issues=review.issues)

    # Print summary
    summary_lines = [
        "RESEARCH PROTOCOL COMPLETE",
        f"Goal: {goal}",
        f"Experiments: {exp_count}",
        f"Paper: {draft.output_path}",
        f"Verdict: {review.verdict} (confidence: {review.confidence:.2f})",
        f"Revisions: {revision_round}",
        f"Duration: {duration:.1f}s",
    ]
    if cost > 0:
        summary_lines.append(f"Cost: ${cost:.4f}")
    summary_lines.append(f"Session: {session_dir}")
    display_summary(summary_lines)

    return result
