"""Campaign runner — drives generative, DM, and pre-campaign turn delivery.

Extracted from orchestrator.py for single-responsibility decomposition.
Each function runs one campaign mode through the simulation bridge and
returns structured results. The orchestrator calls exactly one of these
based on flags, then sets stop_event.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def run_generative_campaign(
    *,
    goal: str,
    bridge: Any,
    llm_router: Any,
    arc_yaml: str | None,
    max_turns: int,
    tool_registry: Any,
    session_dir_base: str,
) -> Any | None:
    """Run a generative campaign — narrator drives a multi-turn story.

    Returns the GenerativeCampaignResult, or None on failure.
    """
    from maxim.simulation.sim_logger import display_status, display_summary

    display_status(f"Generative Campaign: {goal}")
    display_status(f"Max turns: {max_turns}")
    if arc_yaml:
        display_status(f"Arc: {arc_yaml}")
    time.sleep(0.5)  # Let AUT start up

    try:
        from maxim.simulation.generative_runner import (
            run_generative_campaign as _run,
        )

        result = _run(
            goal=goal,
            bridge=bridge,
            llm=llm_router,
            arc_yaml=arc_yaml,
            max_turns=max_turns,
            tool_registry=tool_registry,
            session_dir=session_dir_base,
        )
        display_summary([f"Generative campaign complete: {result.turns_completed} turns"])
        return result
    except Exception as e:
        logger.warning("Generative campaign failed: %s", e)
        display_summary([f"Generative campaign error: {e}"])
        return None


def run_dm_campaign(
    *,
    dm_campaign: Any,
    bridge: Any,
    llm_router: Any,
    aut_registry: Any,
    aut_executor: Any,
    aut_hippocampus: Any | None,
    aut_nac: Any | None,
    aut_memory_hub: Any | None,
    aut_pain_bus: Any | None,
) -> dict[str, Any]:
    """Run a DM campaign — DM runtime drives encounters.

    Returns the DM rollup dict (campaign results + bio-system expectations).
    """
    from maxim.simulation.sim_logger import display_status, display_summary

    display_status(f"DM Campaign: {dm_campaign.name}")
    display_status(f"Goal: {dm_campaign.goal}")
    display_status(f"Encounters: {len(dm_campaign.encounters)}")
    display_status(f"Seed: {dm_campaign.seed}")
    time.sleep(1.0)  # Let AUT start up

    dm_rollup: dict[str, Any] = {}
    try:
        from maxim.simulation.dm_runtime import DMRuntime
        from maxim.simulation.tools_dm import ChooseTool

        # Register ChooseTool for the AUT
        dm_choose_tool = ChooseTool()
        aut_registry.register(dm_choose_tool)

        dm = DMRuntime(
            campaign=dm_campaign,
            bridge=bridge,
            llm_router=llm_router,
            choose_tool=dm_choose_tool,
            executor=aut_executor,
        )

        # ── Instantiate SEM entities from campaign specs ─────────────
        entity_registry: dict[str, Any] = {}
        try:
            from maxim.embodiment.spec import _parse_entity

            # Player character
            if dm_campaign.pc_spec and isinstance(dm_campaign.pc_spec, dict):
                pc_spec = dm_campaign.pc_spec
                if "name" in pc_spec or "entity_type" in pc_spec:
                    entity_registry["pc"] = _parse_entity(pc_spec)

            # NPCs
            for npc_name, npc_spec in (dm_campaign.npc_specs or {}).items():
                if isinstance(npc_spec, dict) and ("name" in npc_spec or "entity_type" in npc_spec):
                    entity_registry[npc_name] = _parse_entity(npc_spec)

            # World objects
            for obj_name, obj_spec in (dm_campaign.object_specs or {}).items():
                if isinstance(obj_spec, dict) and ("name" in obj_spec or "entity_type" in obj_spec):
                    entity_registry[obj_name] = _parse_entity(obj_spec)

            if entity_registry:
                dm.init_entities(
                    entity_registry=entity_registry,
                    tool_registry=aut_registry,
                )
                logger.info(
                    "DM: Instantiated %d SEM entities: %s",
                    len(entity_registry),
                    list(entity_registry.keys()),
                )
        except Exception as e:
            logger.warning("DM: Entity instantiation failed (campaign will run without SEM entities): %s", e)

        try:
            dm_state = dm.run()
        except KeyboardInterrupt:
            logger.info("DM Campaign interrupted by user")
            display_summary(["DM Campaign interrupted (Ctrl+C)"])
            dm_state = dm._state  # Partial state
            dm_state.finish_reason = "cancel"
        dm_rollup = dm.get_rollup()

        display_summary(
            [
                f"DM Campaign complete: {dm_state.turn_count} turns, "
                f"{len(dm_state.choices_made)} choices, "
                f"{len(dm_state.dice_rolls)} dice rolls",
                f"Finish: {dm_state.finish_reason}",
            ]
        )

        # Check bio-system expectations
        exp_results = dm.check_expectations(
            hippocampus=aut_hippocampus,
            nac=aut_nac,
            scn=getattr(aut_memory_hub, "scn", None) if aut_memory_hub else None,
            pain_bus=aut_pain_bus,
        )
        dm_rollup["bio_systems"] = exp_results
        if exp_results.get("checks"):
            passed = sum(1 for c in exp_results["checks"].values() if c.get("pass"))
            total = len(exp_results["checks"])
            status = "ALL PASS" if exp_results["all_pass"] else f"{passed}/{total} passed"
            display_summary([f"Bio-system expectations: {status}"])
    except Exception as e:
        logger.error("DM Campaign failed: %s", e)
        display_summary([f"DM Campaign error: {e}"])
        dm_rollup = {"error": str(e)}

    bridge.finish()
    return dm_rollup


def run_precampaign_turns(
    *,
    turns: list[dict[str, Any]],
    bridge: Any,
    introspector: Any | None,
) -> dict[str, Any]:
    """Deliver pre-parsed campaign turns directly through the bridge.

    Returns a campaign_analysis dict containing turn results and
    post-campaign analysis.
    """
    from maxim.simulation.sim_logger import display_status, display_summary, display_response

    display_status(f"Delivering {len(turns)} campaign turns directly to AUT...")
    time.sleep(1.0)  # Let AUT start up

    campaign_results: list[dict[str, Any]] = []
    for i, turn in enumerate(turns, 1):
        text = turn.get("text", "")
        phase = turn.get("phase", "")
        sal = turn.get("salience", 0.8)
        nov = turn.get("novelty", 0.7)
        phase_label = f" [{phase}]" if phase else ""
        display_status(f"Turn {i}/{len(turns)}{phase_label}: sending ({len(text)} chars)...")
        try:
            result = bridge.send_and_wait(text, salience=sal, novelty=nov)
            actions = [a.tool_name for a in result.get("actions", [])]
            blocked = len(result.get("blocked", []))
            response = result.get("response", "")
            resp_preview = (
                (response[:80] + "...") if response and len(response) > 80 else (response or "(no verbal response)")
            )
            display_status(
                f"AUT: {len(actions)} action(s) {actions}, {blocked} blocked, {result.get('duration_ms', 0):.0f}ms"
            )
            display_response(resp_preview)
            campaign_results.append(
                {
                    "turn": i,
                    "phase": phase,
                    "text_len": len(text),
                    "actions": actions,
                    "blocked": blocked,
                    "response": response,
                    "timed_out": result.get("timed_out", False),
                    "duration_ms": result.get("duration_ms", 0),
                }
            )
        except Exception as e:
            logger.warning("Campaign turn %d failed: %s", i, e)
            campaign_results.append({"turn": i, "phase": phase, "error": str(e)})
    display_summary([f"Campaign delivery complete: {len(campaign_results)} turns delivered"])

    # Post-campaign analysis
    display_status("Running post-campaign analysis...")
    campaign_analysis: dict[str, Any] = {"turns": campaign_results}
    if introspector is not None:
        try:
            analysis = introspector.full_analysis(seed_keywords=[])
            campaign_analysis.update(analysis)

            stats = analysis.get("system_stats", {})
            mem_count = stats.get("hippocampus_memories", "?")
            links = stats.get("nac_causal_links", "?")
            display_status(f"System stats: {mem_count} memories, {links} causal links")
            display_status(f"Summary: {introspector.summarize(analysis)}")
        except Exception as e:
            logger.warning("Post-campaign analysis failed: %s", e)
            display_status(f"(analysis failed: {e})")

    # Save analysis
    try:
        analysis_path = Path("data") / "sim_reports" / f"campaign_analysis_{time.strftime('%Y%m%d_%H%M%S')}.json"
        analysis_path.parent.mkdir(parents=True, exist_ok=True)
        from maxim.utils.atomic_io import atomic_write_json

        atomic_write_json(str(analysis_path), campaign_analysis)
        display_status(f"Analysis saved: {analysis_path}")
    except Exception:
        pass

    display_status("Post-campaign analysis complete.")
    return campaign_analysis
