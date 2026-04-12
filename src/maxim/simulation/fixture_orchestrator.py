"""FixtureDrivenOrchestrator — run YAML fixture scenarios without a narrator LLM.

S1 of simulator_upgrades_plan. Reads a YAML fixture (same schema as
ScenarioSource), drives percepts through a SimulationBridge, and collects
substrate-relevant bio-system state snapshots at end-of-run. No narrator,
no live LLM required for the orchestrator itself — the AUT's LLM calls
are handled by whatever backend is wired (including MockLLMBackend from S2).

Usage:
    result = FixtureDrivenOrchestrator(fixture_path).run(bridge, aut_state)
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from maxim.simulation.scenario_source import load_scenario

logger = logging.getLogger(__name__)


@dataclass
class FixtureResult:
    """Result of a fixture-driven run — structured for substrate analysis."""

    fixture_name: str = ""
    fixture_path: str = ""
    turns_delivered: int = 0
    duration_s: float = 0.0
    finish_reason: str = "complete"

    # Per-turn records: what was sent and what the AUT did
    turn_records: list[dict[str, Any]] = field(default_factory=list)

    # Substrate state snapshots (collected at end-of-run)
    substrate_metrics: dict[str, Any] = field(default_factory=dict)

    # Expectation results (from scenario YAML)
    expectation_results: list[dict[str, Any]] = field(default_factory=list)
    expectations_passed: int = 0
    expectations_total: int = 0


class FixtureDrivenOrchestrator:
    """Drive a YAML fixture through the agent loop without a narrator LLM.

    The orchestrator:
    1. Loads a scenario YAML via ScenarioSource (existing schema)
    2. For each percept, injects it through the bridge and waits for AUT response
    3. Collects turn-by-turn action records
    4. At end-of-run, snapshots bio-system state into substrate_metrics
    5. Checks scenario expectations against observed behavior

    The bridge's send_and_wait handles all timing/settling.
    """

    def __init__(
        self,
        fixture_path: Path,
        *,
        settle_s: float = 2.0,
        turn_timeout: float = 30.0,
    ) -> None:
        self._fixture_path = Path(fixture_path)
        self._definition = load_scenario(self._fixture_path)
        self._settle_s = settle_s
        self._turn_timeout = turn_timeout

        logger.info(
            "FixtureDrivenOrchestrator: loaded %s (%d percepts, %d expectations)",
            self._definition.name,
            len(self._definition.percepts),
            len(self._definition.expectations),
        )

    @property
    def name(self) -> str:
        return self._definition.name

    def run(
        self,
        bridge: Any,
        *,
        hippocampus: Any | None = None,
        nac: Any | None = None,
        memory_hub: Any | None = None,
        pain_bus: Any | None = None,
        percept_trace_buffer: Any | None = None,
    ) -> FixtureResult:
        """Drive the fixture through the bridge and collect results.

        Args:
            bridge: SimulationBridge connected to the AUT
            hippocampus: AUT's Hippocampus (for state snapshot)
            nac: AUT's NAc (for state snapshot)
            memory_hub: AUT's MemoryHub (for ATL access)
            pain_bus: AUT's PainBus/ReactionBus (for reaction history)
            percept_trace_buffer: AUT's PerceptTraceBuffer (for trace snapshot)
        """
        start = time.time()
        turn_records: list[dict[str, Any]] = []

        # Drive each percept through the bridge
        for i, percept_dict in enumerate(self._definition.percepts):
            text = (
                percept_dict.get("cli_input")
                or percept_dict.get("content")
                or percept_dict.get("transcript_chunk")
                or ""
            )
            source = percept_dict.get("source", "cli")
            salience = percept_dict.get("salience", 0.8)
            novelty = percept_dict.get("novelty", 0.7)

            if source == "cli" and text:
                # Text percepts go through send_and_wait for action collection
                result = bridge.send_and_wait(
                    text,
                    salience=salience,
                    novelty=novelty,
                    timeout=self._turn_timeout,
                    settle_s=self._settle_s,
                )
                turn_records.append(
                    {
                        "index": i,
                        "source": source,
                        "text": text[:500],
                        "salience": salience,
                        "novelty": novelty,
                        "response": result.get("response"),
                        "actions": [a.tool_name for a in result.get("actions", [])],
                        "blocked": [a.tool_name for a in result.get("blocked", [])],
                        "timed_out": result.get("timed_out", False),
                        "duration_ms": result.get("duration_ms", 0),
                    }
                )
            elif source.startswith("sensor:") or source == "proprioception":
                # Sensor/pain percepts go through inject_pain or inject_sensor
                if source == "proprioception" or "pain" in source:
                    intensity = percept_dict.get("salience", 0.5)
                    pain_type = (percept_dict.get("metadata") or {}).get("pain_type", "fixture_signal")
                    bridge.inject_pain(
                        pain_type=pain_type,
                        intensity=intensity,
                    )
                    turn_records.append(
                        {
                            "index": i,
                            "source": source,
                            "pain_type": pain_type,
                            "intensity": intensity,
                        }
                    )
                else:
                    # Generic sensor — inject as CLI with metadata for now
                    # (inject_vision stub is S1 scope, deferred to S1+)
                    if text:
                        result = bridge.send_and_wait(
                            text,
                            salience=salience,
                            novelty=novelty,
                            timeout=self._turn_timeout,
                            settle_s=self._settle_s,
                        )
                        turn_records.append(
                            {
                                "index": i,
                                "source": source,
                                "text": text[:500],
                                "response": result.get("response"),
                                "actions": [a.tool_name for a in result.get("actions", [])],
                                "timed_out": result.get("timed_out", False),
                                "duration_ms": result.get("duration_ms", 0),
                            }
                        )
            else:
                # Narrative or unknown source — inject as text if content exists
                if text:
                    result = bridge.send_and_wait(
                        text,
                        salience=salience,
                        novelty=novelty,
                        timeout=self._turn_timeout,
                        settle_s=self._settle_s,
                    )
                    turn_records.append(
                        {
                            "index": i,
                            "source": source,
                            "text": text[:500],
                            "response": result.get("response"),
                            "actions": [a.tool_name for a in result.get("actions", [])],
                            "timed_out": result.get("timed_out", False),
                            "duration_ms": result.get("duration_ms", 0),
                        }
                    )

        duration_s = time.time() - start

        # Signal bridge that fixture is done
        bridge.finish()

        # Collect substrate state snapshots
        substrate_metrics = self._collect_substrate_state(
            hippocampus=hippocampus,
            nac=nac,
            memory_hub=memory_hub,
            pain_bus=pain_bus,
            percept_trace_buffer=percept_trace_buffer,
        )

        # Check expectations
        expectation_results = self._check_expectations(bridge, turn_records)
        passed = sum(1 for e in expectation_results if e.get("pass"))

        result = FixtureResult(
            fixture_name=self._definition.name,
            fixture_path=str(self._fixture_path),
            turns_delivered=len(turn_records),
            duration_s=round(duration_s, 2),
            turn_records=turn_records,
            substrate_metrics=substrate_metrics,
            expectation_results=expectation_results,
            expectations_passed=passed,
            expectations_total=len(expectation_results),
        )

        logger.info(
            "Fixture %s complete: %d turns in %.1fs, expectations %d/%d",
            self._definition.name,
            result.turns_delivered,
            duration_s,
            passed,
            len(expectation_results),
        )

        return result

    def _collect_substrate_state(
        self,
        *,
        hippocampus: Any | None,
        nac: Any | None,
        memory_hub: Any | None,
        pain_bus: Any | None,
        percept_trace_buffer: Any | None,
    ) -> dict[str, Any]:
        """Snapshot bio-system state for substrate analysis."""
        metrics: dict[str, Any] = {}

        # Hippocampus episodes
        if hippocampus is not None:
            try:
                metrics["hippocampus"] = {
                    "episode_count": len(hippocampus),
                }
            except Exception as e:
                logger.debug("hippocampus snapshot failed: %s", e)

        # NAc causal links
        if nac is not None:
            try:
                total_links = sum(len(v) for v in nac._links.values())
                top_links = []
                for sig_links in nac._links.values():
                    for link in sig_links:
                        top_links.append(
                            {
                                "event": getattr(link, "event_signature", ""),
                                "outcome": getattr(link, "outcome_signature", ""),
                                "confidence": round(getattr(link, "confidence", 0), 4),
                                "observations": getattr(link, "observation_count", 0),
                            }
                        )
                top_links.sort(key=lambda x: x["confidence"], reverse=True)
                metrics["nac"] = {
                    "total_links": total_links,
                    "top_links": top_links[:10],
                }
            except Exception as e:
                logger.debug("nac snapshot failed: %s", e)

        # ATL semantic nodes (via MemoryHub)
        if memory_hub is not None:
            try:
                atl = getattr(memory_hub, "_atl", None) or getattr(memory_hub, "atl", None)
                if atl is not None:
                    node_count = len(atl) if hasattr(atl, "__len__") else 0
                    atl_metrics: dict[str, Any] = {"node_count": node_count}
                    # P1: substrate modality breakdown
                    if hasattr(atl, "_modality_index"):
                        atl_metrics["substrate_text_nodes"] = len(atl._modality_index.get("text", set()))
                        atl_metrics["substrate_vision_nodes"] = len(atl._modality_index.get("vision", set()))
                    metrics["atl"] = atl_metrics
            except Exception as e:
                logger.debug("atl snapshot failed: %s", e)

        # EC substrate nodes (P1)
        ec = getattr(memory_hub, "ec", None)
        if ec is not None and hasattr(ec, "substrate_node_count"):
            try:
                metrics["ec_substrate"] = {
                    "node_count": ec.substrate_node_count,
                }
            except Exception as e:
                logger.debug("ec substrate snapshot failed: %s", e)

        # PerceptTraceBuffer
        if percept_trace_buffer is not None:
            try:
                snapshot = percept_trace_buffer.snapshot()
                metrics["percept_trace"] = {
                    "active_entries": len(snapshot),
                    "current_tick": percept_trace_buffer.current_tick,
                    "entries": [
                        {
                            "agent_id": e.agent_id,
                            "percept_id": e.percept_id,
                            "tick": e.tick,
                            "activation": round(e.activation_strength, 4),
                        }
                        for e in snapshot[:20]  # Cap at 20 for report size
                    ],
                }
            except Exception as e:
                logger.debug("percept_trace snapshot failed: %s", e)

        # ReactionBus history (via PainBus wrapper or direct)
        if pain_bus is not None:
            try:
                reaction_bus = getattr(pain_bus, "reaction_bus", pain_bus)
                if hasattr(reaction_bus, "history"):
                    history = reaction_bus.history
                    metrics["reactions"] = {
                        "total_reactions": len(history),
                        "by_kind": {},
                    }
                    for r in history:
                        kind = getattr(r, "kind", "unknown")
                        metrics["reactions"]["by_kind"][kind] = metrics["reactions"]["by_kind"].get(kind, 0) + 1
            except Exception as e:
                logger.debug("reaction_bus snapshot failed: %s", e)

        return metrics

    def _check_expectations(
        self,
        bridge: Any,
        turn_records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Check scenario expectations against observed behavior."""
        import re

        results: list[dict[str, Any]] = []
        all_actions = bridge.get_all_actions()

        for exp in self._definition.expectations:
            check: dict[str, Any] = {
                "type": exp.type,
                "description": exp.description,
                "pass": False,
            }

            if exp.type == "action_blocked":
                pattern = exp.tool_pattern or ""
                blocked = [a for a in all_actions if a.blocked]
                if pattern:
                    matched = [a for a in blocked if re.search(pattern, a.tool_name, re.IGNORECASE)]
                    check["pass"] = len(matched) > 0
                    check["matched"] = len(matched)
                else:
                    check["pass"] = len(blocked) > 0
                    check["matched"] = len(blocked)

            elif exp.type == "action_taken":
                tool = exp.tool or ""
                taken = [a for a in all_actions if not a.blocked and a.tool_name == tool]
                check["pass"] = len(taken) > 0
                check["matched"] = len(taken)

            elif exp.type == "memory_formed":
                # Check turn records for any memory-related actions
                memory_actions = [
                    a for a in all_actions if "memory" in a.tool_name.lower() or "recall" in a.tool_name.lower()
                ]
                check["pass"] = len(memory_actions) > 0
                check["matched"] = len(memory_actions)

            elif exp.type == "pipeline_continued":
                # Check that the agent continued processing after a tagged percept
                tag = exp.after_tag
                if tag:
                    # Find the turn with this tag, check actions exist after it
                    tagged_idx = None
                    for rec in turn_records:
                        if tag in str(rec.get("text", "")):
                            tagged_idx = rec["index"]
                            break
                    if tagged_idx is not None:
                        later = [r for r in turn_records if r["index"] > tagged_idx and r.get("actions")]
                        check["pass"] = len(later) > 0
                    else:
                        check["detail"] = f"tag '{tag}' not found in turns"
                else:
                    check["pass"] = True  # No tag = vacuously true

            else:
                check["detail"] = f"unknown expectation type: {exp.type}"

            results.append(check)

        return results

    def to_report_dict(self, result: FixtureResult) -> dict[str, Any]:
        """Convert FixtureResult to a dict suitable for SimulationReport.substrate_metrics."""
        return asdict(result)
