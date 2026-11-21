"""Standalone experiment runner — run a campaign and get structured results.

Wraps ``start_simulation_mode()`` with campaign-specific defaults and
returns an ``ExperimentResult`` containing the simulation result plus
``Observer`` analysis.  Designed for scripting experiments without
bootstrapping the full interactive CLI.

Example::

    from maxim.simulation.experiment import run_campaign

    result = run_campaign(
        campaign="scenarios/experiments/hippocampal_recall_short.yaml",
        aut_model="mistral-7b",
        seed_keywords=["Verath"],
    )
    print(f"Memory survived: {result.memory_survived('Verath')}")
    print(f"Summary: {result.summary}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Structured result from a campaign experiment run."""

    campaign_path: str
    aut_model: str
    sim_turns: int
    sim_duration_s: float
    finish_reason: str

    # Observer analysis (from full_analysis)
    analysis: dict[str, Any] = field(default_factory=dict)
    summary: str = ""

    # The delivery record of each campaign turn, as campaign_runner.run_precampaign_turns writes it:
    # {"turn": <1-based>, "phase", "actions", "response", "timed_out", ...} or {"turn", "phase", "error"}
    campaign_turns: list[dict[str, Any]] = field(default_factory=list)

    # What the run DID, carried from the SimulationResult so a scenario's expectations are judged against it
    # (#1310: the benchmark validated against an empty sink and no hippocampus, so no model could pass).
    actions: list[dict[str, Any]] = field(default_factory=list)  # serialized ActionRecords
    tool_stats: dict[str, Any] = field(default_factory=dict)  # Executor.tool_usage_stats()
    subsystem_snapshot: dict[str, Any] = field(default_factory=dict)  # Observer.benchmark_snapshot()
    hippocampus: Any = None  # the AUT's hippocampus, for memory expectations
    # When each campaign turn's scenario_tag was delivered without error (its ``sent_at``); None when the run reported
    # no delivery. The one record of delivery: ``delivered_tags`` below is derived from it, so the two cannot drift.
    delivered_tag_times: dict[str, float] | None = None

    @property
    def delivered_tags(self) -> set[str] | None:
        """The scenario tags delivered without error (the keys of ``delivered_tag_times``)."""
        return set(self.delivered_tag_times) if self.delivered_tag_times is not None else None

    def memory_survived(self, keyword: str) -> bool:
        """Check if a specific memory survived the campaign."""
        recall = self.analysis.get("memory_recall", {}).get(keyword, {})
        return recall.get("count", 0) > 0

    def recall_count(self, keyword: str) -> int:
        """Get the number of recall hits for a keyword."""
        recall = self.analysis.get("memory_recall", {}).get(keyword, {})
        return recall.get("count", 0)

    @property
    def hippocampus_memories(self) -> int:
        stats = self.analysis.get("system_stats", {})
        return stats.get("hippocampus_memories", 0)

    @property
    def causal_links(self) -> int:
        stats = self.analysis.get("system_stats", {})
        return stats.get("nac_causal_links", 0)


def _undeliverable(p: Any) -> str | None:
    """Why percept ``p`` cannot be one scripted turn, or ``None`` when it can: the ONE test both modes of
    :func:`load_campaign_turns` apply (#1621). A turn is a non-blank ``cli_input`` string (``take_user_input`` reads
    a blank line as unaddressed, so it would publish nothing, and ``run_precampaign_turns`` measures ``len(text)``)."""
    if not isinstance(p, dict):
        return "is not a mapping"
    text = p.get("cli_input", "")
    if not (isinstance(text, str) and text.strip()):
        return f"has no cli_input line (cli_input {text!r} is not a non-blank line)"
    return None


def load_campaign_turns(campaign_path: str, *, strict: bool = False) -> list[dict[str, Any]]:
    """Load campaign YAML and extract turns for direct injection.

    Returns dicts matching the format ``start_simulation_mode()``
    expects for ``pre_campaign_turns``: ``text``, ``phase``,
    ``salience``, ``novelty``, plus ``metadata`` for experiment use.

    This is the ONE loader of scripted turns (#1621): ``run_campaign``, ``maxim.imagine(scenario=)``, research mode's
    campaign and the CLI's ``--sim-script`` all read their turns here, so no caller reads its own key. The file is
    parsed by ``scenario_source.load_scenario`` (one parser, two views: its expectation and scoring checks run here
    too), so a file that is not YAML, whose top level is not a mapping, or whose ``percepts`` is not a list raises
    ``ValueError`` in either mode; an empty file has no percepts.

    A percept is a turn when :func:`_undeliverable` passes it (a non-blank ``cli_input`` string). By default any
    other percept is skipped (``run_campaign``'s behaviour) and one WARNING names every skipped percept.
    ``strict=True`` (``--sim-script``, ``imagine``, research mode, where every percept must be one turn) raises
    ``ValueError`` instead, for an empty ``percepts`` list, a percept that is not a turn, a percept whose
    ``metadata`` is not a mapping, or one carrying a percept-level ``phase`` (the old ``imagine`` format; the phase
    is read from ``metadata.phase`` only, so it would silently be ``""``). The turn dicts are the same either way. A
    caller that cannot run on zero turns refuses an empty result itself (``run_campaign``).
    """
    path = Path(campaign_path)
    if not path.exists():
        raise FileNotFoundError(f"Campaign file not found: {campaign_path}")

    from maxim.simulation.scenario_source import load_scenario

    percepts = load_scenario(path).percepts
    if not isinstance(percepts, list):
        raise ValueError(f"{campaign_path}: 'percepts' is not a list (got {type(percepts).__name__})")
    if strict and not percepts:
        raise ValueError(f"{campaign_path}: no non-empty 'percepts' list")
    turns = []
    skipped: list[str] = []
    for i, p in enumerate(percepts):
        problem = _undeliverable(p)
        if strict and problem is not None:
            raise ValueError(f"{campaign_path}: percept {i} {problem}")
        if strict and not isinstance(p.get("metadata", {}), dict):
            raise ValueError(f"{campaign_path}: percept {i}'s metadata is not a mapping")
        if strict and "phase" in p:
            raise ValueError(f"{campaign_path}: percept {i} carries a percept-level 'phase'; put it in metadata.phase")
        meta = (p.get("metadata") or {}) if problem is None else {}
        if not isinstance(meta, dict):
            problem = "has a metadata that is not a mapping"
        if problem is not None:
            skipped.append(f"{i} ({problem})")
            continue
        if "phase" in p:
            logger.warning(
                "%s: percept %d's percept-level 'phase' is ignored (read from metadata.phase)", campaign_path, i
            )
        text = p["cli_input"]
        turns.append(
            {
                "at": p.get("at", i),
                "text": text,
                "cli_input": text,  # backward compat for ExperimentResult consumers
                "phase": meta.get("phase", ""),
                "salience": p.get("salience", 0.8),
                "novelty": p.get("novelty", 0.7),
                "metadata": meta,
            }
        )
    if skipped:
        logger.warning("%s: not delivered as turns: percept %s", campaign_path, "; ".join(skipped))
    return turns


def delivered_scenario_tag_times(
    turns: list[dict[str, Any]], campaign_analysis: dict[str, Any] | None
) -> dict[str, float] | None:
    """The ``sent_at`` of every campaign turn delivered without error, keyed by its ``scenario_tag`` (the first
    delivery when a tag repeats), so ``pipeline_continued`` can require an action AFTER the tag (#1391). ``None`` when
    the run reported no delivery at all; a delivered turn without ``sent_at`` (an older record) is left out, so its
    tag's order stays unknown and fails closed."""
    records = (campaign_analysis or {}).get("turns")
    if not isinstance(records, list):
        return None
    times: dict[str, float] = {}
    for record in records:
        if not isinstance(record, dict) or "error" in record:
            continue
        index, sent_at = record.get("turn"), record.get("sent_at")
        if isinstance(index, int) and 1 <= index <= len(turns) and isinstance(sent_at, (int, float)):
            tag = (turns[index - 1].get("metadata") or {}).get("scenario_tag")
            if tag:
                times.setdefault(str(tag), float(sent_at))
    return times


def run_campaign(
    campaign: str,
    aut_model: str = "mistral-7b",
    seed_keywords: list[str] | None = None,
    goal: str = "run campaign experiment",
    mode: str = "generative",
    max_turns: int = 50,
    response_timeout: float = 60.0,
    debug: bool = False,
) -> ExperimentResult:
    """Run a campaign and return structured results with introspection.

    This is the recommended entry point for scripted experiments. It:
    1. Loads campaign YAML and extracts turns
    2. Runs ``start_simulation_mode()`` with direct injection
    3. Performs ``Observer.full_analysis()`` on the result
    4. Returns an ``ExperimentResult`` with all data

    Args:
        campaign: Path to campaign YAML file.
        aut_model: Model profile for the AUT (e.g., "mistral-7b").
        seed_keywords: Keywords to search for in post-campaign analysis.
        goal: Simulation goal string.
        mode: Orchestrator flow-shape label (default "generative").
        max_turns: Maximum simulation turns.
        response_timeout: Timeout per turn in seconds.
        debug: Enable verbose tracing.

    Returns:
        ExperimentResult with simulation data + introspection analysis.
    """
    from maxim.simulation.orchestrator import start_simulation_mode

    # Load campaign turns for direct injection
    turns = load_campaign_turns(campaign)
    if not turns:
        raise ValueError(f"No turns found in campaign: {campaign}")

    logger.info(
        "Running experiment: %s (%d turns, model=%s)",
        campaign,
        len(turns),
        aut_model,
    )

    # Run the simulation with campaign turns injected directly
    sim_result = start_simulation_mode(
        goal=goal,
        mode=mode,
        max_turns=max_turns,
        response_timeout=response_timeout,
        debug=debug,
        aut_model=aut_model,
        pre_campaign_turns=turns,
    )

    # Build the experiment result
    result = ExperimentResult(
        campaign_path=campaign,
        aut_model=aut_model,
        sim_turns=sim_result.turns,
        sim_duration_s=sim_result.duration_s,
        finish_reason=sim_result.finish_reason,
        campaign_turns=list((sim_result.campaign_analysis or {}).get("turns", [])),
        actions=list(sim_result.actions),
        tool_stats=dict(sim_result.tool_stats),
        subsystem_snapshot=dict(sim_result.subsystem_snapshot),
        delivered_tag_times=delivered_scenario_tag_times(turns, sim_result.campaign_analysis),
    )

    # Try to get introspection data from the sim result
    # The orchestrator stores aut_introspector as an attribute if available
    introspector = getattr(sim_result, "introspector", None)
    if introspector is not None:
        result.hippocampus = introspector.hippocampus
        try:
            result.analysis = introspector.full_analysis(seed_keywords=seed_keywords)
            result.summary = introspector.summarize(result.analysis)
        except Exception as e:
            logger.warning("Post-experiment introspection failed: %s", e)
            result.summary = f"Introspection failed: {e}"

    return result
