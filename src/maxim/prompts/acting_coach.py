"""Acting Coach — meta-prompt scaffolding for affordance exploration.

The Acting Coach is a "default policy" that bio-systems continuously
modulate, not override.  It produces a prompt section that encourages
the agent to explore its physical capabilities (SEM affordances) while
annotating exploration directives with learned experience from:

1. **NAc valence** — causal links inject learned caution or preference.
2. **Pain anticipation** — body_state carries anticipated pain context.
3. **Cerebellum predictions** — motor programs predict affordance outcomes.

Composition order: base coach directive → NAc caution annotations →
pain anticipation context → cerebellum predictions.  Each layer ADDS
information; none removes the base directive.  The agent always *can*
explore — the bio-systems inform *how cautiously*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActingCoachConfig:
    """Configuration for the Acting Coach prompt section.

    Attributes:
        role_values: What the character cares about (e.g., survival, curiosity, duty).
        speech_register: How the character speaks (formal/casual/archaic/terse).
        failure_mode: Stress response style (defensive/aggressive/creative/withdrawal).
        continuity_contract: What the character remembers between turns.
        embodiment_guidance: Whether to include affordance exploration directives.
            Automatically enabled when entity tools are available.
        body_state_layers: Whether Layers 2+4 (pain anticipation + drive
            modulation) may render. Both layers also require a non-empty
            ``body_state`` — which no production path supplied before the
            Exp 44 wiring (see docs/plans/acting_coach_body_state_ablation.md).
            This flag exists for the arm-B ablation (body_state present,
            coaching off); do NOT overload ``embodiment_guidance`` for that —
            it gates Layers 1-4 plus the exploration directives together.
        exploration_intensity: How aggressively to encourage exploration (0.0-1.0).
            Higher values produce stronger "try everything" directives.
            Lower values produce more cautious, observation-focused guidance.
    """

    role_values: tuple[str, ...] = ("curiosity", "survival")
    speech_register: str = "casual"
    failure_mode: str = "creative"
    continuity_contract: str = "You remember your past actions, their outcomes, and the lessons you drew from them."
    embodiment_guidance: bool = True
    exploration_intensity: float = 0.7
    body_state_layers: bool = True


def acting_coach_config_from_env() -> ActingCoachConfig:
    """Build the production ActingCoachConfig, honoring ablation env vars.

    ``MAXIM_DISABLE_COACH_BODY_LAYERS`` (parsed by the canonical
    ``cluster_bias_annotation.annotation_disabled_via_env`` — 1/true/t/yes/
    y/on, case-insensitive, whitespace-tolerant) turns Layers 2+4 off for
    the Exp 44 arm-B ablation (docs/plans/acting_coach_body_state_ablation.md).
    Producer sites (cli.py, simulation/orchestrator.py) call this instead of
    ``ActingCoachConfig()`` so the env read lives in ONE place;
    ``compose_acting_coach_section`` itself stays env-free (unit tests pass
    explicit configs — Wire-A producer-site pattern).
    """
    import os

    from maxim.prompts.cluster_bias_annotation import annotation_disabled_via_env

    disabled = annotation_disabled_via_env(os.environ.get("MAXIM_DISABLE_COACH_BODY_LAYERS"))
    return ActingCoachConfig(body_state_layers=not disabled)


def compose_acting_coach_section(
    config: ActingCoachConfig,
    *,
    available_tools: set[str] | None = None,
    causal_context: list[dict[str, Any]] | None = None,
    body_state: str = "",
    motor_programs: list[dict[str, Any]] | None = None,
) -> str:
    """Compose the Acting Coach prompt section.

    Args:
        config: Acting coach configuration.
        available_tools: Currently available tool names. When entity-specific
            tools are present, embodiment guidance activates.
        causal_context: NAc causal predictions from StructuredContext.
            Each entry: {"event": str, "outcome": str, "valence": str,
            "confidence": float, "context_match": float}.
        body_state: Body state interoception string (carries pain anticipation).
        motor_programs: Cerebellum motor programs from StructuredContext.
            Each entry: {"name": str, "goal": str, "confidence": float,
            "success_rate": float, "steps": list[str], "risks": list[str]}.

    Returns:
        Formatted prompt section string. Empty string if nothing to add.
    """
    lines: list[str] = ["=== Acting Coach ==="]

    # --- Base directive: role values + speech register ---
    values_text = ", ".join(config.role_values)
    lines.append(f"Your core values: {values_text}.")
    lines.append(f"Speech register: {config.speech_register}.")
    lines.append(f"Under stress, you tend toward {config.failure_mode} responses.")
    lines.append(config.continuity_contract)
    lines.append("")

    # --- Embodiment guidance (when entity tools are present) ---
    has_entity_tools = _has_entity_tools(available_tools)
    if config.embodiment_guidance and has_entity_tools:
        lines.append("=== Physical Exploration ===")
        has_discovery = available_tools is not None and "sense_tools" in available_tools
        if config.exploration_intensity >= 0.5:
            if has_discovery:
                lines.append(
                    "You have a physical form. Your most relevant capabilities "
                    "are already available as tools. To find more — describe "
                    "what you want to do to sense_tools. Don't ask for "
                    "permission — act. Vary your approach: try different force "
                    "levels, angles, or techniques. Sense your body to observe "
                    "the effects of each action."
                )
            else:
                lines.append(
                    "You have physical capabilities. Explore them. Try different "
                    "parameters. Observe the sensor changes after each action. "
                    "Don't ask for permission — act. Vary your approach: try "
                    "different force levels, angles, or techniques. Each action "
                    "teaches you something about this body and this world."
                )
        else:
            if has_discovery:
                lines.append(
                    "You have a physical form. Your most relevant capabilities "
                    "are available as tools. To find more, describe what you "
                    "want to do to sense_tools. Sense your body before and "
                    "after each action. Start with cautious parameters."
                )
            else:
                lines.append(
                    "You have physical capabilities. Use them thoughtfully. "
                    "Observe the sensor readings before and after each action. "
                    "Start with cautious parameters and increase gradually."
                )
        lines.append("")

        # --- Layer 1: NAc valence modulation ---
        nac_annotations = _compose_nac_annotations(causal_context, available_tools)
        if nac_annotations:
            lines.append("=== Learned Experience (from your past actions) ===")
            lines.extend(nac_annotations)
            lines.append("")

        # --- Layer 2: Pain anticipation (from body_state) ---
        if config.body_state_layers:
            pain_note = _compose_pain_anticipation(body_state)
            if pain_note:
                lines.append(pain_note)
                lines.append("")

        # --- Layer 3: Cerebellum forward model predictions ---
        cerebellum_notes = _compose_cerebellum_predictions(motor_programs)
        if cerebellum_notes:
            lines.append("=== Predicted Outcomes (from forward models) ===")
            lines.extend(cerebellum_notes)
            lines.append("")

        # --- Layer 4: Drive modulation (interoceptive needs) ---
        if config.body_state_layers:
            drive_note = _compose_drive_modulation(body_state)
            if drive_note:
                lines.append(drive_note)
                lines.append("")

    return "\n".join(lines)


def _has_entity_tools(available_tools: set[str] | None) -> bool:
    """Check if any available tools look like SEM entity tools.

    Detection: ``sense_tools`` present (SEM discovery mode, S2) OR
    entity tools matching the naming patterns ``{entity}_{affordance}``
    and ``read_{entity}_{sensor}`` from ``embodiment/tool_bridge.py``.

    Since ``acting_coach`` is only set when ``entity_ref`` is provided,
    false positives here just produce slightly broader guidance (not harmful).
    """
    if not available_tools:
        return False
    # S2: sense_tools presence is the primary indicator
    if "sense_tools" in available_tools:
        return True
    _BUILTIN_TOOLS = {
        "say",
        "think",
        "memory_recall",
        "internet_search",
        "examine",
        "respond",
        "choose",
        "request_interaction",
        "set_scene",
        "sleep",
        "observe_self",
        "web_search",
        "bash",
        "read_file",
        "write_file",
        "glob",
        "navigate",
        "plan",
        "sense",
        "sense_tools",
    }
    for tool in available_tools:
        if tool not in _BUILTIN_TOOLS and (tool.startswith("read_") or "_" in tool):
            return True
    return False


def _compose_nac_annotations(
    causal_context: list[dict[str, Any]] | None,
    available_tools: set[str] | None,
) -> list[str]:
    """Annotate available affordance tools with NAc learned valence.

    Filters causal_context to entries whose 'event' matches an available
    tool, then composes caution (negative valence) or preference
    (positive valence) annotations.
    """
    if not causal_context or not available_tools:
        return []

    annotations: list[str] = []
    tool_set = {t.lower() for t in available_tools}

    for pred in causal_context:
        event = pred.get("event", "")
        if event.lower() not in tool_set:
            continue
        valence = pred.get("valence", "neutral")
        confidence = pred.get("confidence", 0.0)
        outcome = pred.get("outcome", "unknown outcome")

        if confidence < 0.3:
            continue

        if valence == "negative":
            conf_word = "high" if confidence >= 0.7 else "moderate"
            annotations.append(
                f'- {event}: has caused problems before — "{outcome}" '
                f"({conf_word} confidence). Try lower intensity or "
                f"test on weaker targets first."
            )
        elif valence == "positive":
            conf_word = "reliably" if confidence >= 0.7 else "sometimes"
            annotations.append(f'- {event}: {conf_word} succeeded — "{outcome}". Consider trying this approach first.')

    return annotations


def _compose_pain_anticipation(body_state: str) -> str:
    """Extract pain anticipation signals from body_state.

    The body_state string carries anticipated pain context when the
    perceived_pain:anticipated system fires (hippocampal pattern-match
    on current context against past pain episodes).  We look for
    'anticipated' or 'anxiety' keywords as indicators.
    """
    if not body_state:
        return ""
    # body_state is a narrative string built by body.py.  When
    # anticipatory pain is active, it contains phrases like
    # "anticipated pain" or "anxiety".
    lower = body_state.lower()
    if "anticipated" in lower or "anxiety" in lower:
        return (
            "Caution — your body is signaling anticipated discomfort "
            "based on similar past situations. Proceed, but stay aware "
            "of sensor changes."
        )
    return ""


def _compose_drive_modulation(body_state: str) -> str:
    """Report which specific drive signals need attention — per sensor, without
    prescribing a modality-specific action.

    ``format_body_state_for_prompt`` renders one line per sensor carrying a
    ``(DRIVE: <descriptor>)`` annotation, e.g.
    ``- body.temperature: 0.9C (DRIVE: outside comfort band, discomfort 0.30)``
    or ``- body.hunger: 0.8 (DRIVE: deprived, intensity 0.30)``. This function
    names the specific signal(s) that need attention and leaves the *action* to
    the LLM, which sees the full Body State + its affordance tools and can map
    each signal to the right response.

    Data-driven by construction — the guidance reflects whatever sensors/drives
    the body actually declares. It deliberately does NOT hardcode a per-drive
    branch: the pre-fix code emitted thermal "seek shelter/warmth" text for ANY
    homeostatic breach, which is a category error for exteroceptive drives (a
    sound-bearing "centeredness" drive would tell the robot to "seek warmth"
    when a sound was off to the side). Adding a second ``if drive == "..."``
    branch would just move the smell; naming the sensor and staying
    action-neutral removes it. See review SF-4 / embodiment_runtime_wiring.md
    Layer 3b.
    """
    if not body_state:
        return ""

    _MARKER = "(DRIVE:"
    discomfort: list[str] = []  # homeostatic breach — a signal outside its comfortable range
    unmet: list[str] = []  # entropic deprivation — an unmet need
    building: list[str] = []  # a need that is rising but not yet urgent

    for line in body_state.splitlines():
        idx = line.find(_MARKER)
        if idx == -1:
            continue
        descriptor = line[idx + len(_MARKER) :]
        end = descriptor.find(")")
        if end != -1:
            descriptor = descriptor[:end]
        descriptor = descriptor.strip().lower()

        # Sensor label: the "- {entity}.{sensor}:" prefix, when present.
        sensor = ""
        head = line[:idx].strip().removeprefix("- ")
        if ":" in head:
            sensor = head.split(":", 1)[0].strip()

        if "outside comfort band" in descriptor:
            discomfort.append(sensor)
        elif "deprived" in descriptor:
            unmet.append(sensor)
        elif descriptor == "rising":
            building.append(sensor)

    def _named(labels: list[str], fallback: str) -> str:
        named = ", ".join(s for s in labels if s)
        return named or fallback

    drive_notes: list[str] = []
    if discomfort:
        drive_notes.append(
            f"Outside its comfortable range: {_named(discomfort, 'a bodily signal')}. "
            "Act to bring it back — the right action depends on which signal it is "
            "(see the Body State above)."
        )
    if unmet:
        drive_notes.append(f"Unmet need: {_named(unmet, 'a bodily need')}. Prioritize addressing it.")
    if building:
        drive_notes.append(
            f"Building, not yet urgent: {_named(building, 'a need')}. Keep it in mind as you plan your next actions."
        )

    if not drive_notes:
        return ""
    return "=== Body Needs ===\n" + "\n".join(drive_notes)


def _compose_cerebellum_predictions(
    motor_programs: list[dict[str, Any]] | None,
) -> list[str]:
    """Annotate affordances with cerebellum forward model predictions.

    Motor programs carry predicted outcomes and confidence levels.
    Negative predictions (low success_rate, high risks) get caution
    annotations; confident predictions get outcome forecasts.
    """
    if not motor_programs:
        return []

    notes: list[str] = []
    for prog in motor_programs[:5]:
        name = prog.get("name", "?")
        confidence = prog.get("confidence", 0.0)
        success_rate = prog.get("success_rate", 0.0)
        risks = prog.get("risks", [])

        if confidence < 0.3:
            continue

        if success_rate < 0.4 and risks:
            risk_text = ", ".join(risks[:2])
            notes.append(f"- {name}: predicted low success ({success_rate:.0%}). Risks: {risk_text}.")
        elif success_rate >= 0.7:
            goal = prog.get("goal", "")
            notes.append(f"- {name}: predicted success ({success_rate:.0%})" + (f" — {goal}" if goal else "") + ".")

    return notes
