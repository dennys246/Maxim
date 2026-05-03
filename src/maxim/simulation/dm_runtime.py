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

import ast
import logging
import operator
import random
import time
from dataclasses import dataclass, field
from typing import Any

from maxim.simulation.dm_schema import CampaignDef, EncounterDef, roll_dice

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safe arithmetic expression evaluator (replaces eval() for cascade exprs)
# ---------------------------------------------------------------------------

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval_expr(expr: str, variables: dict[str, float]) -> float:
    """Evaluate a simple arithmetic expression safely using AST parsing.

    Supports: +, -, *, /, //, %, **, (), variable names from *variables* dict.
    Rejects: attribute access, function calls, string ops, imports, subscripts.

    Examples:
        _safe_eval_expr("-(roll + damage_bonus)", {"roll": 4, "damage_bonus": 2})  → -6.0
        _safe_eval_expr("hp * 0.5", {"hp": 20})  → 10.0
    """
    tree = ast.parse(expr, mode="eval")

    def _eval_node(node: ast.expr) -> float:
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise ValueError(f"Unknown variable: {node.id!r}")
            return float(variables[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            return float(_SAFE_OPS[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
            return float(_SAFE_OPS[type(node.op)](_eval_node(node.operand)))
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")

    return _eval_node(tree)


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
        choose_tool: Any = None,
        executor: Any = None,
        prompt_handler: Any = None,
    ) -> None:
        self._campaign = campaign
        self._bridge = bridge
        self._llm = llm_router
        self._rng = rng or random.Random(campaign.seed)
        self._choose_tool = choose_tool
        self._executor = executor
        self._prompt_handler = prompt_handler
        self._registered_aliases: list[str] = []
        self._scene: SceneState | None = None
        self._entity_registry: dict[str, Any] = {}
        self._cascade_resolver: CascadeResolver | None = None
        self._state = CampaignState(
            current_encounter=campaign.first_encounter,
        )

    @property
    def state(self) -> CampaignState:
        return self._state

    @property
    def campaign(self) -> CampaignDef:
        return self._campaign

    def init_entities(
        self,
        entity_registry: dict[str, Any],
        tool_registry: Any = None,
        embodiment: Any = None,
        cerebellum: Any = None,
    ) -> None:
        """Initialize entity registry and scene state.

        Call after creating SEM entities from campaign specs. Enables
        scene management (entity enter/leave) and cascade resolution.
        """
        self._entity_registry = entity_registry
        self._cascade_resolver = CascadeResolver(entity_registry)
        self._scene = SceneState(
            entity_registry=entity_registry,
            tool_registry=tool_registry,
            embodiment=embodiment,
            cerebellum=cerebellum,
        )

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

            # Set up ChooseTool for this encounter's choices
            self._setup_encounter_choices(encounter)

            # Set up scene (register/deregister entity tools)
            if self._scene:
                self._scene.enter_encounter(encounter)

            # Compose and deliver stimulus
            stimulus = self._compose_stimulus(encounter)
            _interactive_choice = self._prompt_handler is not None and bool(encounter.choices)

            if _interactive_choice:
                # Interactive mode: display narrative to the human, prompt
                # for their choice, then inform the AUT what happened.
                from maxim.simulation.sim_logger import display_scene

                display_scene(stimulus)
                choice = self._prompt_human_choice(encounter)
                log.info("DM: Player chose '%s'", choice)
                # Tell the AUT the narrative + choice outcome so it can
                # generate flavour text / maintain conversational context.
                self._deliver_and_wait(f"{stimulus}\n\n[Player chose: {choice}]")
            else:
                # Automated mode: AUT decides via ChooseTool or text classification
                response = self._deliver_and_wait(stimulus)

            if encounter.choices:
                if not _interactive_choice:
                    choice = self._get_choice(response, encounter)
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

                # Apply entity swaps (equip/unequip)
                for swap in effects.get("swap_entity", []):
                    self._apply_entity_swap(swap)

                # Evaluate reveal conditions (contextual visibility)
                if self._entity_registry:
                    revealed = evaluate_reveal_conditions(self._entity_registry)
                    for ent_name, item_name in revealed:
                        log.info("DM: Revealed %s.%s", ent_name, item_name)

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

    def _setup_encounter_choices(self, encounter: EncounterDef) -> None:
        """Set up ChooseTool and aliases for the current encounter's choices."""
        # Clean up previous encounter's aliases
        if self._executor and self._registered_aliases:
            self._executor.remove_aliases(self._registered_aliases)
            self._registered_aliases = []

        if not encounter.choices:
            if self._choose_tool:
                self._choose_tool.set_choices([])
            return

        # Update ChooseTool with current choices
        if self._choose_tool:
            self._choose_tool.set_choices(encounter.choices)

        # Register choice names as aliases to 'choose' in the executor
        if self._executor:
            aliases = {c.lower(): "choose" for c in encounter.choices}
            # Also common variants
            for c in encounter.choices:
                aliases[c.lower().replace("_", "")] = "choose"
                aliases[c.lower().replace("_", " ")] = "choose"
            self._executor.register_aliases(aliases)
            self._registered_aliases = list(aliases.keys())

    def _get_choice(self, response: dict[str, Any], encounter: EncounterDef) -> str:
        """Get the AUT's choice, preferring ChooseTool result over text classification."""
        # Check if ChooseTool was used
        if self._choose_tool and self._choose_tool.last_choice:
            choice = self._choose_tool.last_choice
            # Reset for next encounter
            self._choose_tool._last_choice = None
            return choice

        # Fall back to text/LLM classification
        return self._classify_choice(response, encounter)

    def _prompt_human_choice(self, encounter: EncounterDef) -> str:
        """Prompt the human player for a choice via SimPromptHandler.

        If the user types free text that doesn't match any choice, it's
        sent to the AUT as a percept so the agent can respond in character.
        The choice prompt then repeats, letting the user roleplay or ask
        questions before committing to an actual choice.

        Returns the choice string (lowercased to match encounter keys).
        On timeout, defaults to the first choice.
        """
        from maxim.interactive.prompts import PromptRequest, PromptType

        display_choices = [c.replace("_", " ").title() for c in encounter.choices]

        while True:
            request = PromptRequest(
                prompt_type=PromptType.SINGLE_CHOICE,
                question="What do you do?",
                options=tuple(display_choices),
                default=display_choices[0] if display_choices else None,
                timeout_sec=600.0,
            )
            response = self._prompt_handler.prompt(request)

            if response.timed_out:
                log.info("DM: Player timed out — defaulting to '%s'", encounter.choices[0])
                return encounter.choices[0]

            value = response.value.strip()
            if not value:
                continue

            value_lower = value.lower().replace(" ", "_")

            # Try exact match against encounter choices
            for c in encounter.choices:
                if c.lower() == value_lower:
                    return c
            # Fuzzy: check if the display label matches
            for i, dc in enumerate(display_choices):
                if dc.lower() == value.lower():
                    return encounter.choices[i]

            # Not a recognized choice — treat as free-text roleplay.
            # Send to the AUT as a percept so the agent can respond,
            # then re-prompt for the actual choice.
            log.info("DM: Free-text input from player: %s", value[:80])
            try:
                from maxim.simulation.sim_logger import display_scene

                display_scene(f"  You: {value}")
            except Exception:
                pass

            try:
                self._bridge.send_and_wait(
                    f"[Player says: {value}]",
                    salience=0.8,
                    novelty=0.7,
                )
            except Exception:
                pass

    def _compose_stimulus(self, encounter: EncounterDef) -> str:
        """Build the narrative text for an encounter.

        Combines scene text with NPC dialogue hints based on current flags.
        When entities are instantiated (via init_entities), includes live
        sensor state so the agent perceives the actual game state — not
        just the static scene text.
        """
        parts = [encounter.scene]

        # Add live entity sensor state if SceneState is active
        if self._scene and self._entity_registry:
            entity_state = self._format_entity_state(encounter)
            if entity_state:
                parts.append(f"\n[Game State]\n{entity_state}")

        # Add NPC dialogue based on current flags
        for npc_name in encounter.active_npcs:
            npc_spec = self._campaign.npc_specs.get(npc_name, {})
            hints = encounter.dialogue_hints
            metadata = npc_spec.get("metadata", {})
            persona = metadata.get("persona_prompt", "")

            # Find the best matching dialogue hint (flag-based)
            hint_text = hints.get("default", "")
            for flag in self._state.flags:
                if flag in hints:
                    hint_text = hints[flag]
                    break

            if hint_text:
                npc_display = metadata.get("role", npc_name)
                # Inject persona into the stimulus so distinct NPCs reach the
                # LLM with visibly different framing.
                if persona:
                    parts.append(f"\n[{npc_name} — {persona}]")
                parts.append(f'\n{npc_name} ({npc_display}): "{hint_text}"')

        # Add choice prompt if applicable
        if encounter.choices:
            choices_str = ", ".join(encounter.choices)
            parts.append(f"\n\nWhat do you do? Options: {choices_str}")

        return "\n".join(parts)

    def _format_entity_state(self, encounter: EncounterDef) -> str:
        """Format live entity sensor values for the stimulus.

        Reads scalar sensors from active NPCs and world objects,
        presenting them as observable game state the agent can perceive.
        Only includes visible sensors (not hidden ones).
        """
        lines: list[str] = []

        for npc_name in encounter.active_npcs:
            entity = self._entity_registry.get(npc_name)
            if entity is None:
                continue
            sensor_parts = []
            for sensor_name, sensor in getattr(entity, "sensors", {}).items():
                # Skip hidden sensors
                vis = getattr(sensor, "_visibility", None)
                if vis == "hidden":
                    continue
                try:
                    reading = sensor.read()
                    val = reading.value if hasattr(reading, "value") else reading
                    if isinstance(val, (int, float)):
                        sensor_parts.append(
                            f"{sensor_name}={val:.1f}" if isinstance(val, float) else f"{sensor_name}={val}"
                        )
                except Exception:
                    pass
            if sensor_parts:
                lines.append(f"  {npc_name}: {', '.join(sensor_parts)}")

        for obj_name in encounter.world_objects:
            entity = self._entity_registry.get(obj_name)
            if entity is None:
                continue
            sensor_parts = []
            for sensor_name, sensor in getattr(entity, "sensors", {}).items():
                vis = getattr(sensor, "_visibility", None)
                if vis == "hidden":
                    continue
                try:
                    reading = sensor.read()
                    val = reading.value if hasattr(reading, "value") else reading
                    if isinstance(val, (int, float)):
                        sensor_parts.append(
                            f"{sensor_name}={val:.1f}" if isinstance(val, float) else f"{sensor_name}={val}"
                        )
                except Exception:
                    pass
            if sensor_parts:
                lines.append(f"  {obj_name}: {', '.join(sensor_parts)}")

        return "\n".join(lines)

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

        # Extract response text and action names from the bridge result
        response_text = ""
        action_names: list[str] = []
        if isinstance(response, dict):
            response_text = response.get("response", response.get("raw", response.get("message", "")))
            # Also collect tool names the AUT called — these are implicit choices
            for action in response.get("actions", []):
                if hasattr(action, "tool_name"):
                    action_names.append(action.tool_name)
                elif isinstance(action, dict):
                    action_names.append(action.get("tool_name", ""))
                elif isinstance(action, str):
                    action_names.append(action)
        if not response_text:
            response_text = str(response)

        # Combine all text for matching: response + tool names + tool reasoning
        search_text = f"{response_text} {' '.join(action_names)}".lower()

        # Try keyword matching first (fast, no LLM call)
        for choice in encounter.choices:
            if choice.lower().replace("_", " ") in search_text or choice.lower() in search_text:
                return choice
            # Also check if an AUT tool name matches a choice (e.g., "accept_job" tool)
            for action_name in action_names:
                if choice.lower() in action_name.lower() or action_name.lower() in choice.lower():
                    return choice

        # LLM fallback — one-shot classification via generate_json
        if self._llm is not None:
            try:
                choices_str = ", ".join(encounter.choices)
                prompt = (
                    f"The player was given these choices: {choices_str}\n"
                    f'The player responded: "{response_text[:300]}"\n'
                    f'Which choice did the player pick? Reply with JSON: {{"choice": "<choice_name>"}}'
                )
                llm_result = self._llm.generate_json(prompt, max_tokens=30)
                if isinstance(llm_result, dict):
                    picked = llm_result.get("choice", "").lower()
                    for choice in encounter.choices:
                        if choice.lower() == picked or choice.lower() in picked:
                            return choice
                elif isinstance(llm_result, str):
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

        # In interactive mode, show dice results to the human player.
        if self._prompt_handler is not None:
            try:
                from maxim.simulation.sim_logger import display_scene

                display_scene(outcome_text)
            except Exception:
                pass

        try:
            self._bridge.send_and_wait(outcome_text, salience=0.6, novelty=0.3)
        except Exception:
            pass

    def _apply_entity_swap(self, swap: dict[str, Any]) -> None:
        """Swap an entity in the registry — detach old, instantiate + attach new.

        Swap spec format (in campaign YAML ``on_choice``)::

            swap_entity:
              - old: cybernetic_arm        # entity name to remove
                new_ref: bodies/megarm_v3  # component registry ref to instantiate
                new_name: megarm_v3        # name for the new entity instance
                parent: runner             # (optional) parent entity to attach to

        Handles tool re-registration if SceneState is active.
        """
        old_name = swap.get("old", "")
        new_ref = swap.get("new_ref", "")
        new_name = swap.get("new_name", old_name)
        parent_name = swap.get("parent", "")

        if not new_ref:
            log.warning("DM: swap_entity missing 'new_ref'")
            return

        # Detach + deregister old entity
        old_entity = self._entity_registry.pop(old_name, None)
        if old_entity is not None:
            if hasattr(old_entity, "detach"):
                old_entity.detach()
            if self._scene:
                self._scene._deregister_entity_tools(old_name)
            log.info("DM: Detached entity '%s'", old_name)

        # Instantiate new entity from component registry
        try:
            from maxim.embodiment.component_registry import ComponentRegistry

            registry = ComponentRegistry()
            new_entity = registry.instantiate(new_ref, name=new_name)
        except Exception as e:
            log.error("DM: Failed to instantiate '%s': %s", new_ref, e)
            return

        # Reparent if requested
        if parent_name:
            parent = self._entity_registry.get(parent_name)
            if parent is not None and hasattr(new_entity, "reparent"):
                new_entity.reparent(parent)

        # Register in entity registry + scene
        self._entity_registry[new_name] = new_entity
        if self._scene:
            self._scene._entity_registry[new_name] = new_entity
            self._scene._register_entity_tools(new_name)
        if self._cascade_resolver:
            self._cascade_resolver._entities[new_name] = new_entity

        log.info(
            "DM: Swapped '%s' → '%s' (ref=%s)%s",
            old_name,
            new_name,
            new_ref,
            f" parent={parent_name}" if parent_name else "",
        )

    def check_expectations(
        self,
        hippocampus: Any = None,
        nac: Any = None,
        scn: Any = None,
        pain_bus: Any = None,
    ) -> dict[str, Any]:
        """Check bio-system expectations from campaign YAML.

        Returns a dict with per-system pass/fail results.
        """
        expectations = self._campaign.expectations
        if not expectations:
            return {"all_pass": True, "checks": {}}

        checks: dict[str, dict[str, Any]] = {}

        # Hippocampus checks
        if "hippocampus" in expectations and hippocampus is not None:
            exp = expectations["hippocampus"]
            mem_count = len(hippocampus)
            min_captures = exp.get("min_episodic_captures", 0)
            checks["hippocampus_captures"] = {
                "expected": min_captures,
                "actual": mem_count,
                "pass": mem_count >= min_captures,
            }

        # NAc checks
        if "nac" in expectations and nac is not None:
            exp = expectations["nac"]
            nac_stats = nac.stats()
            total_obs = nac_stats.get("total_observations", 0)
            min_obs = exp.get("min_observations", 0)
            checks["nac_observations"] = {
                "expected": min_obs,
                "actual": total_obs,
                "pass": total_obs >= min_obs,
            }
            if "prediction_confidence_above" in exp:
                threshold = exp["prediction_confidence_above"]
                # Check if any link has confidence above threshold
                max_conf = 0.0
                for links in nac._links.values():
                    for link in links:
                        max_conf = max(max_conf, link.confidence)
                checks["nac_confidence"] = {
                    "expected": threshold,
                    "actual": max_conf,
                    "pass": max_conf >= threshold,
                }

        # SCN checks
        if "scn" in expectations and scn is not None:
            exp = expectations["scn"]
            scn_stats = scn.stats()
            bins_used = scn_stats.get("circadian_bins_used", 0)
            min_bins = exp.get("temporal_bins_used", 0)
            checks["scn_bins"] = {
                "expected": min_bins,
                "actual": bins_used,
                "pass": bins_used >= min_bins,
            }

        # Pain checks
        if "pain" in expectations and pain_bus is not None:
            exp = expectations["pain"]
            pain_stats = pain_bus.get_stats()
            total_pain = pain_stats.get("total_published", 0)
            min_signals = exp.get("min_signals", 0)
            checks["pain_signals"] = {
                "expected": min_signals,
                "actual": total_pain,
                "pass": total_pain >= min_signals,
            }

        all_pass = all(c.get("pass", True) for c in checks.values())

        result = {"all_pass": all_pass, "checks": checks}

        # Log results
        for name, check in checks.items():
            icon = "✓" if check["pass"] else "✗"
            log.info(
                "DM Expectation %s %s: expected=%s actual=%s",
                icon,
                name,
                check["expected"],
                check["actual"],
            )

        return result

    def get_rollup(self) -> dict[str, Any]:
        """Generate campaign rollup for report.json."""
        # Include entity snapshots if scene state exists
        entity_snapshots = {}
        if self._scene:
            entity_snapshots = self._scene.snapshot()

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
                "entity_snapshots": entity_snapshots,
            },
        }


# ---------------------------------------------------------------------------
# SceneState — tracks active entities per encounter
# ---------------------------------------------------------------------------


class SceneState:
    """Tracks which entities are active in the current encounter.

    Manages dynamic tool registration/deregistration as NPCs and
    objects enter and leave scenes.
    """

    def __init__(
        self,
        entity_registry: dict[str, Any],
        tool_registry: Any = None,
        embodiment: Any = None,
        cerebellum: Any = None,
    ) -> None:
        self._entity_registry = entity_registry  # name → Entity
        self._tool_registry = tool_registry
        self._embodiment = embodiment
        self._cerebellum = cerebellum
        self._active_npcs: list[str] = []
        self._active_objects: list[str] = []
        self._registered_tools: dict[str, list[str]] = {}  # entity_name → [tool_names]

    def enter_encounter(self, encounter: EncounterDef) -> None:
        """Update scene when entering a new encounter.

        Registers tools for newly-active entities.
        Deregisters tools for entities no longer in scene.
        """
        new_npcs = [n for n in encounter.active_npcs if n in self._entity_registry]
        new_objects = [o for o in encounter.world_objects if o in self._entity_registry]

        # Deregister tools for departing entities
        departing = set(self._active_npcs + self._active_objects) - set(new_npcs + new_objects)
        for name in departing:
            self._deregister_entity_tools(name)

        # Register tools for arriving entities
        arriving = set(new_npcs + new_objects) - set(self._active_npcs + self._active_objects)
        for name in arriving:
            self._register_entity_tools(name)

        self._active_npcs = new_npcs
        self._active_objects = new_objects

        log.debug(
            "Scene: %d NPCs (%s), %d objects (%s)",
            len(new_npcs),
            ", ".join(new_npcs),
            len(new_objects),
            ", ".join(new_objects),
        )

    def _register_entity_tools(self, name: str) -> None:
        """Register tools for an entity entering the scene."""
        if self._tool_registry is None:
            return
        entity = self._entity_registry.get(name)
        if entity is None:
            return
        try:
            from maxim.embodiment.tool_bridge import generate_tools_for_entity

            tools = generate_tools_for_entity(
                entity,
                self._tool_registry,
                embodiment=self._embodiment,
                cerebellum=self._cerebellum,
            )
            self._registered_tools[name] = [t.name for t in tools]
            log.info("Scene: registered %d tools for %s", len(tools), name)
        except Exception as e:
            log.debug("Scene: failed to register tools for %s: %s", name, e)

    def _deregister_entity_tools(self, name: str) -> None:
        """Deregister tools for an entity leaving the scene."""
        if self._tool_registry is None:
            return
        tool_names = self._registered_tools.pop(name, [])
        for tname in tool_names:
            try:
                self._tool_registry.deregister(tname)
            except Exception:
                pass
        if tool_names:
            log.info("Scene: deregistered %d tools for %s", len(tool_names), name)

    def get_active_entity(self, name: str) -> Any:
        """Get an active entity by name."""
        if name in self._active_npcs or name in self._active_objects:
            return self._entity_registry.get(name)
        return None

    def format_scene_for_prompt(self) -> str:
        """Format active scene entities for the LLM prompt."""
        lines = ["=== Scene ==="]

        if self._active_npcs:
            lines.append("Active NPCs:")
            for name in self._active_npcs:
                entity = self._entity_registry.get(name)
                if entity is None:
                    continue
                # Show visible sensors only
                sensors = []
                for sname, sensor in entity.sensors.items():
                    if entity.get_visibility(sname) == "visible":
                        try:
                            val = sensor.read().value
                            if isinstance(val, (int, float)):
                                sensors.append(f"{sname}={val:.1f}" if isinstance(val, float) else f"{sname}={val}")
                        except Exception:
                            pass
                role = entity.metadata.get("role", entity.entity_type)
                sensor_str = f": {', '.join(sensors)}" if sensors else ""
                lines.append(f"  - {name} ({role}){sensor_str}")

        if self._active_objects:
            lines.append("Objects:")
            for name in self._active_objects:
                entity = self._entity_registry.get(name)
                if entity is None:
                    continue
                sensors = []
                for sname, sensor in entity.sensors.items():
                    if entity.get_visibility(sname) == "visible":
                        try:
                            val = sensor.read().value
                            if isinstance(val, (int, float)):
                                sensors.append(f"{sname}={val:.1f}" if isinstance(val, float) else f"{sname}={val}")
                        except Exception:
                            pass
                sensor_str = f": {', '.join(sensors)}" if sensors else ""
                lines.append(f"  - {name} ({entity.entity_type}){sensor_str}")

        return "\n".join(lines) if len(lines) > 1 else ""

    def snapshot(self) -> dict[str, Any]:
        """Snapshot all entity sensor states for the rollup report."""
        result = {}
        for name, entity in self._entity_registry.items():
            sensors = {}
            for sname, sensor in entity.sensors.items():
                try:
                    val = sensor.read().value
                    if isinstance(val, (int, float)):
                        sensors[sname] = val
                except Exception:
                    pass
            if sensors:
                result[name] = sensors
        return result


# ---------------------------------------------------------------------------
# Cascade Resolver
# ---------------------------------------------------------------------------


class CascadeResolver:
    """Resolves cascade specs by reading/writing entity sensors.

    Role references (self, wielder, target) are resolved against a
    context dict mapping role names to Entity objects.
    """

    def __init__(self, entity_registry: dict[str, Any]) -> None:
        self._entities = entity_registry

    def resolve(
        self,
        cascade: Any,
        roles: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a cascade: reads → writes → side_effects.

        Args:
            cascade: CascadeSpec with reads, writes, side_effects.
            roles: Maps role names to Entity objects
                   (e.g., {"self": sword, "wielder": derek, "target": guard}).

        Returns:
            Dict of read values keyed by role name.
        """

        read_values: dict[str, float] = {}

        # Phase 1: Reads (gather values)
        for ref in cascade.reads:
            val = self._read_sensor(ref.ref, roles)
            if val is not None:
                role_name = ref.role or ref.ref.split(".")[-1]
                read_values[role_name] = val

        # Phase 2: Writes (apply changes)
        for ref in cascade.writes:
            self._write_sensor(ref, roles, read_values)

        # Phase 3: Side effects (same as writes but semantically separate)
        for ref in cascade.side_effects:
            self._write_sensor(ref, roles, read_values)

        return read_values

    def _resolve_entity_sensor(
        self,
        ref_path: str,
        roles: dict[str, Any],
    ) -> tuple[Any, str] | None:
        """Resolve a ref path like 'wielder.strength.modifier' to (entity, sensor_name).

        Returns (entity, sensor_name) or None if not found.
        """
        parts = ref_path.split(".")
        if len(parts) < 2:
            return None

        # First part is the role or entity name
        role_or_name = parts[0]
        entity = roles.get(role_or_name) or self._entities.get(role_or_name)
        if entity is None:
            return None

        # Navigate through children for intermediate parts
        for part in parts[1:-1]:
            child = entity.find(part) if hasattr(entity, "find") else None
            if child is None:
                return None
            entity = child

        sensor_name = parts[-1]
        return (entity, sensor_name)

    def _read_sensor(self, ref_path: str, roles: dict[str, Any]) -> float | None:
        """Read a sensor value from a ref path."""
        result = self._resolve_entity_sensor(ref_path, roles)
        if result is None:
            return None
        entity, sensor_name = result
        sensor = entity.sensors.get(sensor_name)
        if sensor is None:
            # Try vital_metrics fallback
            return entity.vital_metrics.get(sensor_name)
        try:
            val = sensor.read().value
            return float(val) if isinstance(val, (int, float)) else None
        except Exception:
            return None

    def _write_sensor(
        self,
        ref: Any,
        roles: dict[str, Any],
        read_values: dict[str, float],
    ) -> None:
        """Write a sensor value from a cascade ref."""
        result = self._resolve_entity_sensor(ref.ref, roles)
        if result is None:
            if not ref.optional:
                log.debug("Cascade: ref '%s' not found", ref.ref)
            return
        entity, sensor_name = result

        # Compute new value
        if ref.value is not None:
            # Absolute set
            new_val = ref.value
        elif ref.delta is not None:
            # Additive delta
            current = entity.vital_metrics.get(sensor_name, 0.0)
            new_val = current + ref.delta
        elif ref.expr:
            # Safe arithmetic expression evaluation — no eval().
            # Supports: +, -, *, /, (), variable substitution from read_values.
            # Rejects: attribute access, function calls, string ops, imports.
            try:
                new_val = _safe_eval_expr(ref.expr, read_values)
            except (ValueError, TypeError, ZeroDivisionError) as e:
                log.debug("Cascade: expr '%s' failed: %s", ref.expr, e)
                return
        else:
            return

        # Apply to vital_metrics (sensors read from here via SpecSensor)
        entity.vital_metrics[sensor_name] = float(new_val)
        log.debug("Cascade: %s.%s = %.2f", entity.name, sensor_name, new_val)


# ---------------------------------------------------------------------------
# Visibility Evaluator
# ---------------------------------------------------------------------------


def evaluate_reveal_conditions(
    entity_registry: dict[str, Any],
    roles: dict[str, Any] | None = None,
) -> list[tuple[str, str]]:
    """Check all contextual visibility conditions and reveal items that pass.

    Returns list of (entity_name, revealed_item_name) tuples.
    """
    from maxim.simulation.dm_schema import RevealCondition

    revealed: list[tuple[str, str]] = []
    roles = roles or {}

    for name, entity in entity_registry.items():
        visibility = entity.metadata.get("visibility", {})
        reveal_when = entity.metadata.get("reveal_when", {})

        for item_name, condition_data in reveal_when.items():
            # Skip if already visible
            if visibility.get(item_name) == "visible":
                continue

            try:
                condition = RevealCondition.from_dict(condition_data) if isinstance(condition_data, dict) else None
                if condition is None:
                    continue

                # Resolve the ref to get the sensor value
                ref_parts = condition.ref.split(".")
                role_or_name = ref_parts[0]
                target_entity = roles.get(role_or_name) or entity_registry.get(role_or_name)
                if role_or_name == "self":
                    target_entity = entity

                if target_entity is None:
                    continue

                # Navigate to sensor
                for part in ref_parts[1:-1]:
                    child = target_entity.find(part) if hasattr(target_entity, "find") else None
                    if child is None:
                        break
                    target_entity = child
                else:
                    sensor_name = ref_parts[-1]
                    val = target_entity.vital_metrics.get(sensor_name)
                    if val is None:
                        sensor = target_entity.sensors.get(sensor_name)
                        if sensor:
                            try:
                                val = float(sensor.read().value)
                            except Exception:
                                pass

                    if val is not None and condition.evaluate(val):
                        entity.reveal(item_name)
                        revealed.append((name, item_name))
                        log.info("Visibility: revealed %s.%s", name, item_name)
            except Exception as e:
                log.debug("Visibility eval failed for %s.%s: %s", name, item_name, e)

    return revealed
