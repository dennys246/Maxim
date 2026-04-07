"""Standalone experiment runner — run a campaign and get structured results.

Wraps ``start_simulation_mode()`` with campaign-specific defaults and
returns an ``ExperimentResult`` containing the simulation result plus
``AUTIntrospector`` analysis.  Designed for scripting experiments without
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

    # AUTIntrospector analysis (from full_analysis)
    analysis: dict[str, Any] = field(default_factory=dict)
    summary: str = ""

    # Raw campaign turn results (if available)
    campaign_turns: list[dict[str, Any]] = field(default_factory=list)

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


def load_campaign_turns(campaign_path: str) -> list[dict[str, Any]]:
    """Load campaign YAML and extract turns for direct injection."""
    import yaml

    path = Path(campaign_path)
    if not path.exists():
        raise FileNotFoundError(f"Campaign file not found: {campaign_path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    percepts = data.get("percepts", [])
    return [
        {
            "at": p.get("at", i),
            "cli_input": p.get("cli_input", ""),
            "metadata": p.get("metadata", {}),
        }
        for i, p in enumerate(percepts)
        if p.get("cli_input")
    ]


def run_campaign(
    campaign: str,
    aut_model: str = "mistral-7b",
    seed_keywords: list[str] | None = None,
    goal: str = "run campaign experiment",
    persona: str = "campaign",
    max_turns: int = 50,
    response_timeout: float = 60.0,
    debug: bool = False,
) -> ExperimentResult:
    """Run a campaign and return structured results with introspection.

    This is the recommended entry point for scripted experiments. It:
    1. Loads campaign YAML and extracts turns
    2. Runs ``start_simulation_mode()`` with direct injection
    3. Performs ``AUTIntrospector.full_analysis()`` on the result
    4. Returns an ``ExperimentResult`` with all data

    Args:
        campaign: Path to campaign YAML file.
        aut_model: Model profile for the AUT (e.g., "mistral-7b").
        seed_keywords: Keywords to search for in post-campaign analysis.
        goal: Simulation goal string.
        persona: Orchestrator persona (default "campaign" for hands-off).
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
        persona=persona,
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
    )

    # Try to get introspection data from the sim result
    # The orchestrator stores aut_introspector as an attribute if available
    introspector = getattr(sim_result, "introspector", None)
    if introspector is not None:
        try:
            result.analysis = introspector.full_analysis(seed_keywords=seed_keywords)
            result.summary = introspector.summarize(result.analysis)
        except Exception as e:
            logger.warning("Post-experiment introspection failed: %s", e)
            result.summary = f"Introspection failed: {e}"

    return result
