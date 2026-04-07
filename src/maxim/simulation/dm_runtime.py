"""DM Runtime — campaign state machine, choice classification, encounter resolution.

Drives a D&D-style campaign through the generative campaign infrastructure.
The DM runtime composes stimuli from encounter definitions + NPC state,
delivers them via the simulation bridge, classifies AUT responses to
determine choices, and advances the campaign state.

Example:
    dm = DMRuntime(campaign, bridge, embodiment, rng)
    dm.run()  # Blocks until campaign ends or __END__ reached
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

from maxim.simulation.dm_schema import CampaignDef, EncounterDef, roll_dice

log = logging.getLogger(__name__)


@dataclass
class CampaignState:
    """Mutable state tracking campaign progress."""

    current_encounter: str = ""
    current_act: int = 0
    choices_made: list[dict[str, Any]] = field(default_factory=list)
    flags: set[str] = field(default_factory=set)
    dice_rolls: list[dict[str, Any]] = field(default_factory=list)
    encounters_visited: list[str] = field(default_factory=list)
    finished: bool = False
    finish_reason: str = ""
    turn_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_encounter": self.current_encounter,
            "current_act": self.current_act,
            "choices_made": self.choices_made,
            "flags": sorted(self.flags),
            "dice_rolls": self.dice_rolls,
            "encounters_visited": self.encounters_visited,
            "finished": self.finished,
            "finish_reason": self.finish_reason,
            "turn_count": self.turn_count,
        }


class DMRuntime:
    """Campaign state machine driving encounters through the sim bridge.

    Parameters
    ----------
    campaign : CampaignDef
        Parsed campaign definition.
    bridge : SimulationBridge
        Bridge for sending percepts to the AUT and waiting for responses.
    llm_router : Any
        LLM router for choice classification fallback.
    rng : random.Random | None
        Seeded RNG for dice rolls. Created from campaign.seed if None.
    """

    def __init__(
        self,
        campaign: CampaignDef,
        bridge: Any,
        llm_router: Any = None,
        rng: random.Random | None = None,
    ) -> None:
        self._campaign = campaign
        self._bridge = bridge
        self._llm = llm_router
        self._rng = rng or random.Random(campaign.seed)
        self._state = CampaignState(
            current_encounter=campaign.first_encounter,
        )

    @property
    def state(self) -> CampaignState:
        return self._state

    @property
    def campaign(self) -> CampaignDef:
        return self._campaign

    def run(self) -> CampaignState:
        """Run the campaign to completion. Blocks until __END__ or error."""
        log.info("DM: Starting campaign '%s' (seed=%d)", self._campaign.name, self._campaign.seed)

        while not self._state.finished:
            enc_name = self._state.current_encounter
            encounter = self._campaign.encounters.get(enc_name)
            if encounter is None:
                self._state.finished = True
                self._state.finish_reason = f"unknown encounter: {enc_name}"
                log.error("DM: Unknown encounter '%s'", enc_name)
                break

            self._state.encounters_visited.append(enc_name)
            self._state.turn_count += 1
            log.info("DM: Encounter '%s' (turn %d)", enc_name, self._state.turn_count)

            # Compose and deliver stimulus
            stimulus = self._compose_stimulus(encounter)
            response = self._deliver_and_wait(stimulus)

            # Classify choice
            if encounter.choices:
                choice = self._classify_choice(response, encounter)
                log.info("DM: AUT chose '%s'", choice)

                # Record choice
                self._state.choices_made.append(
                    {
                        "encounter": enc_name,
                        "choice": choice,
                        "turn": self._state.turn_count,
                        "timestamp": time.time(),
                    }
                )

                # Apply on_choice effects
                effects = encounter.on_choice.get(choice, {})
                if effects.get("flags"):
                    for flag in effects["flags"]:
                        self._state.flags.add(flag.lower())

                # Resolve dice if required for this choice
                dice_spec = encounter.dice.get(choice)
                if dice_spec:
                    self._resolve_dice(choice, dice_spec, encounter)

                # Follow branch
                target = encounter.branches.get(choice, "__END__")
                if target == "__END__":
                    self._state.finished = True
                    self._state.finish_reason = f"campaign_end:{enc_name}:{choice}"
                    log.info("DM: Campaign ended at '%s' via choice '%s'", enc_name, choice)
                else:
                    self._state.current_encounter = target.lower()
            else:
                # No choices — auto-advance to next encounter in act order
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

        log.info(
            "DM: Campaign '%s' complete — %d turns, %d choices, %d dice rolls",
            self._campaign.name,
            self._state.turn_count,
            len(self._state.choices_made),
            len(self._state.dice_rolls),
        )
        return self._state

    def _compose_stimulus(self, encounter: EncounterDef) -> str:
        """Build the narrative text for an encounter.

        Combines scene text with NPC dialogue hints based on current flags.
        """
        parts = [encounter.scene]

        # Add NPC dialogue based on current flags
        for npc_name in encounter.active_npcs:
            npc_spec = self._campaign.npc_specs.get(npc_name, {})
            hints = encounter.dialogue_hints
            _persona = npc_spec.get("metadata", {}).get("persona_prompt", "")  # Used by NarrativeModulator in Slice 2

            # Find the best matching dialogue hint (flag-based)
            hint_text = hints.get("default", "")
            for flag in self._state.flags:
                if flag in hints:
                    hint_text = hints[flag]
                    break

            if hint_text:
                npc_display = npc_spec.get("metadata", {}).get("role", npc_name)
                parts.append(f'\n{npc_name} ({npc_display}): "{hint_text}"')

        # Add choice prompt if applicable
        if encounter.choices:
            choices_str = ", ".join(encounter.choices)
            parts.append(f"\n\nWhat do you do? Options: {choices_str}")

        return "\n".join(parts)

    def _deliver_and_wait(self, stimulus: str) -> dict[str, Any]:
        """Send stimulus to AUT via bridge and wait for response."""
        try:
            result = self._bridge.send_and_wait(stimulus, salience=0.8, novelty=0.7)
            return result if isinstance(result, dict) else {"raw": str(result)}
        except Exception as e:
            log.warning("DM: Bridge delivery failed: %s", e)
            return {"error": str(e)}

    def _classify_choice(self, response: dict[str, Any], encounter: EncounterDef) -> str:
        """Map AUT response to one of the encounter's declared choices.

        Uses LLM one-shot classification. Falls back to first choice on failure.
        """
        if not encounter.choices:
            return ""

        # Extract response text from the bridge result
        response_text = ""
        if isinstance(response, dict):
            response_text = response.get("raw", response.get("message", str(response)))
        if not response_text:
            response_text = str(response)

        # Try keyword matching first (fast, no LLM call)
        response_lower = response_text.lower()
        for choice in encounter.choices:
            if choice.lower() in response_lower:
                return choice

        # LLM fallback — one-shot classification
        if self._llm is not None:
            try:
                choices_str = ", ".join(encounter.choices)
                prompt = (
                    f"The player was given these choices: {choices_str}\n"
                    f'The player responded: "{response_text[:300]}"\n'
                    f"Which choice did the player pick? Reply with ONLY the choice name, nothing else."
                )
                llm_result = self._llm.generate(prompt, max_tokens=20)
                if llm_result:
                    result_lower = llm_result.strip().lower()
                    for choice in encounter.choices:
                        if choice.lower() in result_lower:
                            return choice
            except Exception as e:
                log.debug("DM: LLM classification failed: %s", e)

        # Fallback — first choice
        log.warning("DM: Could not classify response, defaulting to '%s'", encounter.choices[0])
        return encounter.choices[0]

    def _resolve_dice(
        self,
        choice: str,
        dice_spec: dict[str, Any],
        encounter: EncounterDef,
    ) -> None:
        """Resolve a dice check for a choice."""
        roll_spec = dice_spec.get("roll", "1d20")
        dc = dice_spec.get("dc", 10)
        success_flag = dice_spec.get("success_flag", "")

        result = roll_dice(roll_spec, self._rng)
        success = result >= dc

        self._state.dice_rolls.append(
            {
                "encounter": encounter.name,
                "choice": choice,
                "roll_spec": roll_spec,
                "result": result,
                "dc": dc,
                "success": success,
                "timestamp": time.time(),
            }
        )

        if success and success_flag:
            self._state.flags.add(success_flag.lower())

        log.info(
            "DM: Dice %s = %d vs DC %d → %s%s",
            roll_spec,
            result,
            dc,
            "SUCCESS" if success else "FAILURE",
            f" (flag: {success_flag})" if success and success_flag else "",
        )

        # Deliver dice result as narrative
        outcome_text = f"[Dice roll: {roll_spec} = {result} vs DC {dc} → {'SUCCESS' if success else 'FAILURE'}]"
        try:
            self._bridge.send_and_wait(outcome_text, salience=0.6, novelty=0.3)
        except Exception:
            pass

    def get_rollup(self) -> dict[str, Any]:
        """Generate campaign rollup for report.json."""
        return {
            "campaign": {
                "name": self._campaign.name,
                "goal": self._campaign.goal,
                "seed": self._campaign.seed,
                "encounters_completed": len(self._state.encounters_visited),
                "encounters_visited": self._state.encounters_visited,
                "choices_made": self._state.choices_made,
                "dice_rolls": self._state.dice_rolls,
                "flags": sorted(self._state.flags),
                "finish_reason": self._state.finish_reason,
                "turn_count": self._state.turn_count,
            },
        }
