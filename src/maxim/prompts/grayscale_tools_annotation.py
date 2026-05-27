"""Grayscale-tools section composer (W1 sense_tool_registry MVP).

Surfaces SEM-derived tools the substrate has a non-zero reward bias for
but that are NOT in the active scene roster. The motivating bug (Roy-3a-
retry NULL outcome): Wire-A annotated ``sense_food_source [strongly
rewarding]`` for the substrate, but the test fixture had no food entity,
so the tool was silently absent from the LLM-visible tools list. The
substrate had decided "this tool matters" but the LLM had no way to act
on the signal because it couldn't see the tool exists elsewhere.

This module renders the candidate set as a separate prompt section the
LLM reads alongside ``=== Available Tools ===``. The annotation reads
``[not in current location]`` so the LLM understands the tool is
*known but presently uninvokable* — invoking it would fail because the
target entity is absent. The expected reasoning move is "reach for an
equivalent active tool" via ``sense_tools`` or by choosing a different
active affordance.

Design choices:

- Rendered as an adjacent IMPORTANT section (not folded into the
  Available Tools block) so the active-tools rendering pipeline stays
  unchanged: the Wire 1 / Wire 3 description annotations, the mode-
  filtered tool-relevance branch, and the failed-tools hint all keep
  working without modification.
- The bias magnitude is folded into one of two bands — "strongly
  rewarding from prior experience" (bias >= +0.2), "strongly aversive
  from prior experience" (bias <= -0.2), or no band suffix when within
  the neutral band. Mirrors the Wire-A composer's band-based phrasing
  so the LLM reads grayscale tools in the same voice as substrate
  associations.
- The descriptions are passed in pre-resolved by the producer (it knows
  the registered tool instance) — this composer does no registry
  lookups.

See [docs/plans/sense_tool_registry.md] § "Phase 3" and the load-bearing
invariant "SEM-derived tools must stay scene-scoped at execution time"
(grayscale is about visibility, not invokability).
"""

from __future__ import annotations

# Bands are intentionally a touch wider than Wire-A's so the grayscale
# block surfaces only tools the substrate has a clearly load-bearing
# signal for. Neutral bands (|bias| < 0.05) are pre-filtered by the
# producer's NAc.get_agent_tool_biases call (top-N by |bias|), so the
# composer only sees the strong tail.
STRONG_BAND_THRESHOLD = 0.05  # below |this|, render with no experience suffix


def _bias_phrase(bias: float) -> str:
    """Render the magnitude+sign of a bias as a short experience phrase."""
    if bias >= STRONG_BAND_THRESHOLD:
        return "rewarding from prior experience"
    if bias <= -STRONG_BAND_THRESHOLD:
        return "aversive from prior experience"
    return ""


def compose_grayscale_tools_section(
    annotations: list[tuple[str, float, str]] | None,
) -> str:
    """Render (tool_name, bias, description) tuples as a prompt section.

    Returns an empty string when ``annotations`` is None or empty so the
    prompt builder skips the section entirely — no clutter for cold-
    start agents or scenes where the substrate's biased tools are all
    already active.

    Example output:

        === Tools known but not in current location ===
          sense_food_source — Scan for food sources [not in current location, rewarding from prior experience]
          forest_gather     — Gather forage in the forest [not in current location]

    The annotation has two parts in brackets:
    - ``not in current location`` (always present) — declares the tool
      is knowable but presently uninvokable.
    - ``<experience phrase>`` (optional) — only when the substrate bias
      magnitude exceeds the strong-band threshold. Matches Wire-A's
      "from prior experience" voice.
    """
    if not annotations:
        return ""

    lines: list[str] = ["=== Tools known but not in current location ==="]
    # Column-align tool names for readability — same convention as the
    # Wire-A renderer. ``default=0`` defends against an empty list
    # surviving the upstream guards (impossible today but harmless).
    max_name_len = max((len(name) for name, _bias, _desc in annotations), default=0)
    for name, bias, description in annotations:
        # Trim descriptions to keep the section budget bounded — full
        # descriptions render in the active tools section when the tool
        # transitions in; the grayscale view is a discoverability hint,
        # not a tool-call reference.
        trimmed_desc = (description or "").strip()
        if len(trimmed_desc) > 80:
            trimmed_desc = trimmed_desc[:77] + "..."

        phrase = _bias_phrase(bias)
        bracket = "not in current location"
        if phrase:
            bracket = f"{bracket}, {phrase}"

        name_padded = name.ljust(max_name_len)
        if trimmed_desc:
            lines.append(f"  {name_padded} — {trimmed_desc} [{bracket}]")
        else:
            lines.append(f"  {name_padded} [{bracket}]")
    return "\n".join(lines)


__all__ = [
    "STRONG_BAND_THRESHOLD",
    "compose_grayscale_tools_section",
]
