"""Motor-only AUT prompt — Phase 0 of grounded_language_acquisition.md.

Substrate-primary mode skips the LLM, but downstream tooling (logging,
display, debug, future hybrid modes) still benefits from a structured
percept rendering. This module composes that rendering with NO English
narration and NO linguistic affordance descriptions — only:

  * drive states (numeric)
  * sensor readings (numeric)
  * available motor primitives (tool names only — no descriptions)

Pre-linguistic by design. The Acting Coach (``prompts/acting_coach``)
adds linguistic role-framing, exploration hortatives, and bio-modulation
narratives — all banned here. If a future caller imports both, that's
a wiring bug; substrate-primary mode chooses one or the other.

Read alongside:
- docs/plans/grounded_language_acquisition.md Phase 0
- src/maxim/runtime/agent_loop.py::propose_via_substrate
- src/maxim/prompts/acting_coach.py (the layer this replaces)
"""

from __future__ import annotations

from typing import Any


def compose_motor_only_percept(
    *,
    drives: dict[str, float] | None = None,
    sensors: dict[str, float] | None = None,
    available_tools: list[str] | tuple[str, ...] | set[str] | None = None,
) -> str:
    """Render substrate state as a motor-only percept block.

    The output is a fixed-shape, deterministic, English-free dump:

        DRIVES
        hunger 0.823
        thirst 0.211
        SENSORS
        head.thermal 0.500
        arms.pressure 0.000
        TOOLS
        infant_humanoid_pick_up
        arms_use
        head_look

    Empty sections render their header only ("DRIVES\\n" with no rows).
    The motor-only AUT does NOT receive descriptions, examples, role
    framing, or any narrator-supplied prose. Numeric scales are 6-char
    fixed (3-decimal float) so token counts stay constant across ticks.

    Args:
        drives: ``{drive_name: value in [0, 1]}`` — typically extracted
            via ``runtime.agent_loop._read_drive_states``.
        sensors: Additional sensor readings beyond the drive-mapped set.
            Same numeric contract as drives.
        available_tools: Tool names. Order is preserved if a list/tuple
            is passed; sets are sorted for deterministic output.

    Returns:
        Formatted percept string. Empty when all three inputs are empty.
    """
    sections: list[str] = []

    drives = drives or {}
    sensors = sensors or {}

    if not drives and not sensors and not available_tools:
        return ""

    sections.append("DRIVES")
    for name in sorted(drives):
        sections.append(f"{name} {drives[name]:.3f}")

    sections.append("SENSORS")
    for name in sorted(sensors):
        sections.append(f"{name} {sensors[name]:.3f}")

    sections.append("TOOLS")
    if available_tools:
        if isinstance(available_tools, (list, tuple)):
            tool_iter: list[str] = list(available_tools)
        else:
            tool_iter = sorted(available_tools)
        sections.extend(tool_iter)

    return "\n".join(sections)


def compose_motor_only_aut_prompt(
    *,
    drives: dict[str, float] | None = None,
    sensors: dict[str, float] | None = None,
    available_tools: list[str] | tuple[str, ...] | set[str] | None = None,
    last_action: dict[str, Any] | None = None,
    last_outcome: str | None = None,
) -> str:
    """Compose the full motor-only AUT prompt.

    Distinct from ``compose_motor_only_percept`` — this is the full
    structured "what the AUT sees this tick" view, including the most
    recent action + outcome (also in motor-only form: tool name +
    success/failure flag, no English).

    Args:
        drives: ``{drive_name: value}``.
        sensors: ``{sensor_name: value}``.
        available_tools: Iterable of tool names.
        last_action: ``{"tool_name": str, "params": dict}`` from the
            most recent ``executor.execute`` call. None on the first
            tick or when the substrate produced no proposal.
        last_outcome: ``"success"`` / ``"failure"`` — collapsed to
            two values to keep the prompt linguistically sterile.

    Returns:
        Formatted prompt block. Empty string when nothing to render.
    """
    blocks: list[str] = []

    percept = compose_motor_only_percept(drives=drives, sensors=sensors, available_tools=available_tools)
    if percept:
        blocks.append(percept)

    if last_action is not None:
        tool = last_action.get("tool_name", "")
        outcome = last_outcome or ""
        if tool or outcome:
            blocks.append("LAST_ACTION")
            if tool:
                blocks.append(tool)
            if outcome:
                blocks.append(outcome)

    return "\n".join(blocks)
