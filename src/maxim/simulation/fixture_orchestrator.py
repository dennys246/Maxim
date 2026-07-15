"""FixtureDrivenOrchestrator — run YAML fixture scenarios without a narrator LLM.

S1 of simulator_upgrades_plan. Reads a YAML fixture (same schema as
ScenarioSource), drives percepts through a SimulationBridge, and collects
substrate-relevant bio-system state snapshots at end-of-run. No narrator,
no live LLM required for the orchestrator itself — the AUT's LLM calls
are handled by whatever backend is wired (including MockLLMBackend from S2).

**W2 Bug B fix (Fix B, 2026-05-27, [docs/plans/deferred/imagination_substrate_signals.md]):**
``run()`` accepts an optional substrate-aware scene-load pre-trigger that
calls ``Narrator.generate_scene_manifest(llm_router, goal, nac_top_biases=...)``
and routes the result through ``imagination_trigger.process_manifest(...)``.
This mirrors the AUT orchestrator path at ``orchestrator.py::start_simulation_mode``
where W2 originally landed — exp 32 surfaced that W2's hookup site was
structurally bypassed by fixture-driven test arms (Roy's ``roy_1_holdout``
runs through this orchestrator and never reached W2's generative-narrator
manifest call). Fix B extends the substrate→scene pipeline to the fixture
path so the LLM can act on Wire-A's strongly-rewarding annotations when the
named tool isn't otherwise in scene.

**Open Question #5 (self-reinforcing preference loops) is INTENTIONALLY
deferred** in Fix B, matching the W2 MVP precedent: Wire-A's tau-300
cluster-bias decay is the natural inhibitor; an empirical-grounding gate
(``≥N% of past sessions`` constraint) becomes a follow-up plan if the next
Roy iteration shows pathological reinforcement. See W2's plan-doc Open
Question 5 for the deeper rationale.

Usage:
    # Basic — no substrate-aware pre-trigger (S1 contract preserved):
    result = FixtureDrivenOrchestrator(fixture_path).run(bridge, aut_state)

    # With Fix B substrate-aware pre-trigger:
    result = FixtureDrivenOrchestrator(fixture_path).run(
        bridge,
        nac=aut_nac,
        memory_hub=aut_memory_hub,
        imagination_trigger=aut_imagination_trigger,
        llm_router=llm_router,
        goal=goal,
    )
"""

from __future__ import annotations

import logging
import os
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

    # Fix B observability: entities materialized by the substrate-aware
    # scene-load pre-trigger (see ``_substrate_pretrigger``). Empty tuple
    # when the pre-trigger didn't fire OR materialized zero entities.
    # Surfaces to Roy analyzers as ``substrate_metrics["pretrigger_entities"]``
    # via ``_collect_substrate_state``, plus the per-arm session_dir's
    # report JSON so cross-arm scene-divergence (Roy methodology concern
    # raised by pre-merge architecture-lens BLOCK 3) is post-hoc auditable.
    pretrigger_entities: tuple[str, ...] = ()


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
        # Fix B (W2 to fixture scene-load) — optional substrate-aware
        # pre-trigger. ALL FOUR (nac, imagination_trigger, llm_router, goal)
        # must be non-None for the pre-trigger to fire. See class docstring
        # + docs/plans/deferred/imagination_substrate_signals.md.
        imagination_trigger: Any | None = None,
        llm_router: Any | None = None,
        goal: str | None = None,
    ) -> FixtureResult:
        """Drive the fixture through the bridge and collect results.

        Args:
            bridge: SimulationBridge connected to the AUT
            hippocampus: AUT's Hippocampus (for state snapshot)
            nac: AUT's NAc (for state snapshot AND substrate-aware
                pre-trigger source — see imagination_trigger arg below)
            memory_hub: AUT's MemoryHub (for ATL access + canonical
                agent_id resolution for the substrate-aware pre-trigger)
            pain_bus: AUT's PainBus/ReactionBus (for reaction history)
            percept_trace_buffer: AUT's PerceptTraceBuffer (for trace snapshot)
            imagination_trigger: Fix B — AUT's ``ImaginationTrigger``.
                When non-None alongside ``llm_router`` and ``goal``,
                the orchestrator calls ``generate_scene_manifest`` with
                ``NAc.get_agent_tool_biases(agent_id=memory_hub.agent_id)``
                and routes the result through
                ``imagination_trigger.process_manifest(...)`` BEFORE the
                percept loop runs. This materializes substrate-favored
                entities into the fixture's scene so SEM-derived tools
                like ``sense_food_source`` become invokable when Wire-A's
                annotation names them as strongly rewarding.
            llm_router: Fix B — LLMRouter for the manifest LLM call.
                Required (alongside ``imagination_trigger`` and ``goal``)
                for the pre-trigger to fire.
            goal: Fix B — goal string passed to ``generate_scene_manifest``.
                In Roy's per-arm framing this is templated as e.g.
                ``"roy:roy-3a:arm_a"``; the orchestrator augments it with
                the fixture's name/description so the manifest LLM has
                useful scene context.

        Gating semantics:
            - All FOUR Fix B parameters (``nac``, ``imagination_trigger``,
              ``llm_router``, ``goal``) MUST be non-None for the pre-
              trigger to fire. Otherwise the orchestrator runs the
              original S1 contract (no pre-trigger) unchanged. Pre-merge
              review caught an earlier draft that promised "three" while
              the code required four — see ``_substrate_pretrigger``'s
              docstring for the rationale (without ``nac`` there is no
              substrate signal; the gate would silently no-op while
              burning a manifest LLM call).
            - ``MAXIM_DISABLE_IMAGINATION_SUBSTRATE_SIGNAL=1`` disables
              the pre-trigger for Roy ablation runs (shared with W2's
              kill switch; same truthy parser).
            - Empty biases short-circuit cleanly — no manifest LLM
              call when the substrate has nothing to surface.
        """
        start = time.time()
        turn_records: list[dict[str, Any]] = []

        # ── Fix B: substrate-aware scene-load pre-trigger ────────────────
        # Mirrors orchestrator.py's W2 hookup at the generative-narrator
        # path. Closes the gap exp 32 surfaced: fixture-driven test arms
        # bypass W2 because they never call generate_scene_manifest.
        # Returns the materialized entity names so they land on
        # ``FixtureResult.pretrigger_entities`` for post-hoc Roy cross-arm
        # scene-divergence auditing (pre-merge review BLOCK 3).
        pretrigger_entities = self._substrate_pretrigger(
            nac=nac,
            memory_hub=memory_hub,
            imagination_trigger=imagination_trigger,
            llm_router=llm_router,
            goal=goal,
        )

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
            pretrigger_entities=pretrigger_entities,
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

    def _substrate_pretrigger(
        self,
        *,
        nac: Any | None,
        memory_hub: Any | None,
        imagination_trigger: Any | None,
        llm_router: Any | None,
        goal: str | None,
    ) -> tuple[str, ...]:
        """Fix B — substrate-aware scene-load pre-trigger for fixture path.

        Mirrors W2's hookup at ``orchestrator.py::start_simulation_mode``
        line ~1467. **Fires when all FOUR parameters (nac, imagination_trigger,
        llm_router, goal) are non-None** — the executor-lens pre-merge review
        caught that an earlier docstring said "three". The four-param shape is
        load-bearing: without ``nac`` there is no substrate signal to surface,
        and the gate would silently no-op while burning a manifest LLM call.

        Materializes substrate-favored entities into the fixture's scene by:

        1. Resolving the agent_id from ``memory_hub.agent_id`` (canonical
           AUT identifier post-Fix-A; falls back to ``"sim_aut"`` if the
           hub is missing the field — should not happen in practice).
        2. Calling ``NAc.get_agent_tool_biases(agent_id, top_n=5)`` to
           collect substrate-acquired tool preferences.
        3. Composing a fixture-aware goal that augments the caller's
           goal with the scenario name + description, so the manifest
           LLM has useful scene context (Roy's per-arm goal templates
           like ``"roy:roy-3a:arm_a"`` are otherwise uninformative).
        4. Calling ``generate_scene_manifest(llm_router, goal, nac_top_biases)``
           — same shared substrate-voice rendering as W2's generative
           hookup.
        5. Routing the manifest text through
           ``imagination_trigger.process_manifest(scene_id="fixture_pretrigger")``
           to materialize entities into scene.

        Observability (pre-merge review BLOCK 2 fold): every gate emits a
        matching ``sim_log("SEM_TRACE", ...)`` event so Roy's JSONL post-hoc
        analyzers can distinguish "Fix B fired and materialized N entities"
        from "Fix B skipped on empty biases" from "Fix B disabled via kill
        switch" from "Fix B never reached the gate." Mirrors W2's emission
        pattern at orchestrator.py:1482-1487.

        Returns:
            Tuple of materialized entity names (or empty tuple when the
            pre-trigger short-circuited at any gate). The caller stashes
            this on ``FixtureResult.pretrigger_entities`` so cross-arm
            scene-divergence is post-hoc auditable (pre-merge review
            architecture-lens BLOCK 3 fold).

        Fail-soft: any exception in the pre-trigger path is logged at
        WARNING but does NOT abort the run. The S1 contract (drive
        percepts → snapshot state) remains intact regardless of pre-
        trigger success.
        """
        # Lazy-imported once per call: sim_log + the env-var helper. Module
        # cache makes subsequent imports dict-lookup-cost (~µs); mirrors W2.
        from maxim.simulation.sim_logger import sim_log

        if imagination_trigger is None or llm_router is None or goal is None or nac is None:
            return ()

        try:
            from maxim.prompts.cluster_bias_annotation import annotation_disabled_via_env

            # Shared kill switch with W2's generative hookup. Roy ablation
            # arms can disable both surfaces of the substrate signal in one
            # env-var flip per CLAUDE.md "opt-in env vars need autouse
            # scrubs" + conftest's existing scrub for this var.
            if annotation_disabled_via_env(os.environ.get("MAXIM_DISABLE_IMAGINATION_SUBSTRATE_SIGNAL")):
                sim_log(
                    "SEM_TRACE", "Fix B fixture pre-trigger: skipped via MAXIM_DISABLE_IMAGINATION_SUBSTRATE_SIGNAL"
                )
                return ()

            # Resolve canonical agent_id. Post-Fix-A (PR #290), memory_hub.agent_id
            # is always set from AgentConfig.agent_id via build_bio_stack's
            # required keyword-only contract — so the ``or "sim_aut"`` fallback
            # should never fire in production. Kept defensive for raw-orchestrator
            # callers (tests, future adapters) that omit memory_hub.
            agent_id = (getattr(memory_hub, "agent_id", None) if memory_hub is not None else None) or "sim_aut"

            try:
                nac_top_biases = nac.get_agent_tool_biases(agent_id=agent_id, top_n=5)
            except ValueError as e:
                # Post-Fix-A this branch should not fire (agent_id is required +
                # non-empty in AgentConfig). Logged loudly so a future regression
                # surfaces in Roy's measurement arm rather than silently degrading.
                logger.warning(
                    "Fix B substrate-aware fixture pre-trigger skipped due to invalid agent_id (%s)",
                    e,
                )
                sim_log("SEM_TRACE", f"Fix B fixture pre-trigger: skipped (invalid agent_id: {e})")
                return ()

            if not nac_top_biases:
                # Empty substrate — pre-trigger is a no-op. No need to
                # burn an LLM call on an empty manifest.
                sim_log(
                    "SEM_TRACE",
                    f"Fix B fixture pre-trigger: skipped (NAc has no biases for agent_id={agent_id})",
                )
                return ()

            from maxim.simulation.narrator import generate_scene_manifest

            # Fixture-aware goal: caller's goal (e.g., Roy's arm template)
            # alone is uninformative; augment with the fixture's name + any
            # description from the YAML so the manifest LLM has scene context.
            fixture_name = self._definition.name
            fixture_description = getattr(self._definition, "description", "") or ""
            scene_goal = goal
            if fixture_name:
                scene_goal = f"{goal} (fixture: {fixture_name})"
                if fixture_description:
                    scene_goal = f"{scene_goal} — {fixture_description}"

            sim_log(
                "SEM_TRACE",
                f"Fix B fixture pre-trigger: generating manifest (agent_id={agent_id}, biases={len(nac_top_biases)})",
            )
            manifest_text = generate_scene_manifest(
                llm_router,
                scene_goal,
                nac_top_biases=nac_top_biases,
            )

            if not manifest_text:
                sim_log("SEM_TRACE", "Fix B fixture pre-trigger: manifest empty (goal too vague or LLM failed)")
                return ()

            results = imagination_trigger.process_manifest(
                manifest_text,
                scene_id="fixture_pretrigger",
            )
            entity_names = tuple(str(r) for r in (results or ()))
            sim_log(
                "SEM_TRACE",
                f"Fix B fixture pre-trigger: {len(entity_names)} entities resolved {list(entity_names)}",
            )
            return entity_names
        except Exception as e:
            # Fail-soft: pre-trigger failure must NOT abort the fixture run.
            # The S1 contract (drive percepts → snapshot state) is preserved.
            logger.warning("Fix B fixture pre-trigger failed: %s", e, exc_info=True)
            try:
                sim_log("SEM_TRACE", f"Fix B fixture pre-trigger: failed ({type(e).__name__}: {e})")
            except Exception:
                # Don't let the trace emission itself break fail-soft.
                pass
            return ()

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
