"""Party DM Runtime — multi-agent campaign execution.

Extends DMRuntime with NPC agents that have real memory and learning.
Each NPC is a full AgentInstance (Hippocampus, NAc, personality) created
via AgentFactory.  During each encounter:

1. DM delivers scene narrative to ALL agents (PC + NPCs)
2. NPC agents react first (generate dialogue, update internal state)
3. PC agent observes NPC reactions + scene, makes choice
4. DM resolves choice, applies effects
5. All agents witness the outcome (feeds into their hippocampus)

This means NPCs *remember* prior encounters, *learn* causal patterns
from the PC's behavior, and *adapt* their dialogue accordingly.

Example::

    party = PartyDMRuntime(campaign, bridge, agent_pool, agent_factory)
    state = party.run()  # Blocks until campaign ends
    memories = party.get_agent_memories()  # Per-NPC memory export
"""

from __future__ import annotations

import logging
import time
from typing import Any

from maxim.simulation.dm_runtime import CampaignState, DMRuntime
from maxim.simulation.dm_schema import CampaignDef, EncounterDef

log = logging.getLogger(__name__)


class PartyDMRuntime(DMRuntime):
    """DM runtime with NPC agents that have real memory + learning.

    Extends the base DMRuntime encounter loop with multi-agent turns:
    NPC agents receive the same scene narrative as the PC, generate
    responses, and have their reactions folded into the stimulus
    the PC sees.

    Parameters
    ----------
    campaign : CampaignDef
        Parsed campaign definition with ``party_mode: true``.
    bridge : SimulationBridge
        Bridge for sending percepts to the PC agent.
    agent_pool : AgentPool
        Pool managing all NPC agent instances.
    agent_factory : AgentFactory
        Factory for creating NPC agents from campaign specs.
    llm_router : Any
        LLM router for choice classification fallback.
    choice_resolution : str
        How conflicting party choices resolve. Default ``"pc_decides"``.
    """

    def __init__(
        self,
        campaign: CampaignDef,
        bridge: Any,
        agent_pool: Any,
        agent_factory: Any,
        llm_router: Any = None,
        rng: Any = None,
        choose_tool: Any = None,
        executor: Any = None,
        choice_resolution: str = "pc_decides",
        component_registry: Any = None,
    ) -> None:
        super().__init__(
            campaign=campaign,
            bridge=bridge,
            llm_router=llm_router,
            rng=rng,
            choose_tool=choose_tool,
            executor=executor,
        )
        self._pool = agent_pool
        self._factory = agent_factory
        self._choice_resolution = choice_resolution
        self._component_registry = component_registry
        self._npc_responses: dict[str, str] = {}  # Last NPC responses per encounter

        # Share the LLM router with the agent pool so NPCs can generate
        # real dialogue instead of placeholder responses.
        if llm_router is not None and hasattr(agent_pool, "_llm_router"):
            agent_pool._llm_router = llm_router

        # Set up NPC agents from campaign specs
        self._setup_npc_agents()

    def _setup_npc_agents(self) -> None:
        """Create NPC agents from campaign npc_specs via AgentFactory.

        Each NPC gets:
        - Entity from ComponentRegistry (if spec has 'ref')
        - Personality from metadata.persona_prompt
        - Full bio-stack (hippocampus + NAc)
        """
        for npc_name, npc_spec in self._campaign.npc_specs.items():
            # Skip if already in pool (e.g., from a resumed campaign)
            if self._pool.has(f"npc_{npc_name}") if hasattr(self._pool, 'has') else f"npc_{npc_name}" in self._pool.agent_ids:
                continue

            persona = ""
            if isinstance(npc_spec, dict):
                persona = npc_spec.get("metadata", {}).get("persona_prompt", "")

            # Check for entity ref in spec
            entity_ref = None
            if isinstance(npc_spec, dict):
                entity_ref = npc_spec.get("ref") or npc_spec.get("entity_ref")

            # Determine model tier from spec
            model_tier = "small"
            if isinstance(npc_spec, dict):
                model_tier = npc_spec.get("model_tier", "small")

            # Determine if NPC should remember/learn
            remembers = True
            learns = True
            if isinstance(npc_spec, dict):
                remembers = npc_spec.get("remembers", True)
                learns = npc_spec.get("learns", True)

            try:
                instance = self._factory.create_npc_agent(
                    npc_name=npc_name,
                    entity_ref=entity_ref,
                    personality=persona,
                    model_profile=model_tier,
                    remembers=remembers,
                    learns=learns,
                )
                self._pool.add(instance)
                log.info("Party DM: created NPC agent '%s' (entity=%s, persona=%s...)",
                         npc_name, entity_ref or "none", persona[:40])
            except Exception as e:
                log.warning("Party DM: failed to create NPC agent '%s': %s", npc_name, e)

    def run(self) -> CampaignState:
        """Run the campaign with multi-agent turns.

        Overrides DMRuntime.run() to add NPC agent turns between
        scene delivery and PC choice resolution.
        """
        log.info("Party DM: Starting campaign '%s' (party_mode, %d NPCs)",
                 self._campaign.name, self._pool.size)

        while not self._state.finished:
            enc_name = self._state.current_encounter
            encounter = self._campaign.encounters.get(enc_name)
            if encounter is None:
                self._state.finished = True
                self._state.finish_reason = f"unknown encounter: {enc_name}"
                log.error("Party DM: Unknown encounter '%s'", enc_name)
                break

            self._state.encounters_visited.append(enc_name)
            self._state.turn_count += 1
            log.info("Party DM: Encounter '%s' (turn %d)", enc_name, self._state.turn_count)

            # Set up ChooseTool for this encounter's choices
            self._setup_encounter_choices(encounter)

            # Set up scene (register/deregister entity tools)
            if self._scene:
                self._scene.enter_encounter(encounter)

            # --- PHASE 1: Compose base scene narrative ---
            base_stimulus = self._compose_stimulus(encounter)

            # --- PHASE 2: NPC agents react to the scene ---
            npc_reactions = self._run_npc_turns(encounter, base_stimulus)

            # --- PHASE 3: Deliver enriched stimulus to PC ---
            enriched_stimulus = self._enrich_with_npc_reactions(
                base_stimulus, npc_reactions, encounter
            )
            response = self._deliver_and_wait(enriched_stimulus)

            # --- PHASE 4: Resolve choice + effects ---
            if encounter.choices:
                choice = self._get_choice(response, encounter)
                log.info("Party DM: PC chose '%s'", choice)

                self._state.choices_made.append({
                    "encounter": enc_name,
                    "choice": choice,
                    "turn": self._state.turn_count,
                    "timestamp": time.time(),
                    "npc_reactions": {k: v[:100] for k, v in npc_reactions.items()},
                })

                # Apply on_choice effects
                effects = encounter.on_choice.get(choice, {})
                if effects.get("flags"):
                    for flag in effects["flags"]:
                        self._state.flags.add(flag.lower())

                # Resolve dice
                dice_spec = encounter.dice.get(choice)
                if dice_spec:
                    self._resolve_dice(choice, dice_spec, encounter)

                # --- PHASE 5: All agents witness the outcome ---
                outcome_text = f"The player chose to {choice}."
                self._broadcast_outcome(encounter, choice, outcome_text)

                # Follow branch
                target = encounter.branches.get(choice, "__END__")
                if target == "__END__":
                    self._state.finished = True
                    self._state.finish_reason = f"campaign_end:{enc_name}:{choice}"
                else:
                    self._state.current_encounter = target.lower()
            else:
                # No choices — auto-advance
                enc_order = self._campaign.encounter_order
                try:
                    idx = enc_order.index(enc_name)
                    if idx + 1 < len(enc_order):
                        self._state.current_encounter = enc_order[idx + 1]
                    else:
                        self._state.finished = True
                        self._state.finish_reason = "all_encounters_complete"
                except ValueError:
                    self._state.finished = True
                    self._state.finish_reason = f"encounter_not_in_order:{enc_name}"

        log.info("Party DM: Campaign '%s' complete — %d turns, %d NPCs",
                 self._campaign.name, self._state.turn_count, self._pool.size)
        return self._state

    def _run_npc_turns(
        self,
        encounter: EncounterDef,
        stimulus: str,
    ) -> dict[str, str]:
        """Run turns for active NPC agents in this encounter.

        Each NPC receives the scene narrative and produces a response.
        Responses are collected and will be folded into the PC's stimulus.

        Returns dict of npc_name → response text.
        """
        active_npc_ids = [
            f"npc_{name}" for name in encounter.active_npcs
            if f"npc_{name}" in self._pool.agent_ids
        ]

        if not active_npc_ids:
            return {}

        # Build per-NPC percepts (same scene, different framing)
        percepts = {}
        for npc_id in active_npc_ids:
            npc_name = npc_id.removeprefix("npc_")
            npc_spec = self._campaign.npc_specs.get(npc_name, {})
            persona = ""
            if isinstance(npc_spec, dict):
                persona = npc_spec.get("metadata", {}).get("persona_prompt", "")

            # Frame the scene for this NPC
            npc_percept = f"[Scene: {stimulus}]\n"
            if persona:
                npc_percept += f"[Your personality: {persona}]\n"
            npc_percept += "How do you react to this scene? Respond in character."
            percepts[npc_id] = npc_percept

        # Run all NPC turns (concurrent)
        results = self._pool.run_round(percepts, concurrent=True)

        # Collect responses
        reactions: dict[str, str] = {}
        for npc_id, result in results.items():
            npc_name = npc_id.removeprefix("npc_")
            if result.error:
                log.warning("Party DM: NPC '%s' failed to respond: %s", npc_name, result.error)
            elif result.response:
                reactions[npc_name] = result.response
                log.debug("Party DM: NPC '%s' reacted: %s...", npc_name, result.response[:60])

        self._npc_responses = reactions
        return reactions

    def _enrich_with_npc_reactions(
        self,
        base_stimulus: str,
        npc_reactions: dict[str, str],
        encounter: EncounterDef,
    ) -> str:
        """Fold NPC reactions into the stimulus the PC sees.

        The PC sees: base scene + NPC dialogue/reactions + choice prompt.
        """
        if not npc_reactions:
            return base_stimulus

        # Insert NPC reactions before the choice prompt
        parts = [base_stimulus.rsplit("\n\nWhat do you do?", 1)[0]]

        for npc_name, reaction in npc_reactions.items():
            npc_spec = self._campaign.npc_specs.get(npc_name, {})
            display_name = npc_name
            if isinstance(npc_spec, dict):
                display_name = npc_spec.get("metadata", {}).get("role", npc_name)
            parts.append(f'\n{npc_name} ({display_name}): "{reaction}"')

        # Re-add choice prompt if present
        if encounter.choices:
            choices_str = ", ".join(encounter.choices)
            parts.append(f"\n\nWhat do you do? Options: {choices_str}")

        return "\n".join(parts)

    def _broadcast_outcome(
        self,
        encounter: EncounterDef,
        choice: str,
        outcome_text: str,
    ) -> None:
        """Broadcast the encounter outcome to all NPC agents.

        This feeds into their hippocampus — they *witness* what happened
        and can recall it in future encounters.
        """
        active_npc_ids = [
            f"npc_{name}" for name in encounter.active_npcs
            if f"npc_{name}" in self._pool.agent_ids
        ]

        if not active_npc_ids:
            return

        percepts = {npc_id: outcome_text for npc_id in active_npc_ids}
        self._pool.run_round(percepts, concurrent=False)  # Sequential for outcome recording

    def get_agent_memories(self) -> dict[str, dict[str, Any]]:
        """Export all NPC agent memories for post-campaign analysis."""
        return self._pool.export_all_memories()

    def get_rollup(self) -> dict[str, Any]:
        """Extended rollup including per-NPC memory state."""
        rollup = super().get_rollup()
        rollup["party_mode"] = True
        rollup["npc_agent_count"] = self._pool.size
        rollup["npc_memories"] = self.get_agent_memories()
        return rollup

    def shutdown(self) -> None:
        """Shut down NPC agent pool and flush memories."""
        self._pool.shutdown()
