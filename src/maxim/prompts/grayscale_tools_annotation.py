"""Grayscale-tools section composer + producer (W1 sense_tool_registry MVP).

Surfaces SEM-derived tools the substrate has a non-zero reward bias for
but that are NOT in the active scene roster. The motivating bug (Roy-3a-
retry NULL outcome): Wire-A annotated ``sense_food_source`` as strongly
rewarding for the substrate, but the test fixture had no food entity, so
the tool was silently absent from the LLM-visible tools list. The
substrate had decided "this tool matters" but the LLM had no way to act
on the signal because it couldn't see the tool exists elsewhere.

This module exposes two surfaces:

- ``build_grayscale_annotations(biases, registry)`` — producer-side
  filter that takes NAc-shaped ``(tool:<name>, bias)`` tuples (the
  output format of ``NAc.get_agent_tool_biases``) and a ToolRegistry,
  and returns ``(tool_name, bias, description)`` tuples ready for the
  composer. Strips the canonical ``tool:`` prefix, filters out tools
  that are core / active / non-SEM, and caps the result at ``top_n``.
- ``compose_grayscale_tools_section(annotations)`` — pure renderer on
  the tuple list. The LLM-facing block reads ``[not in current
  location, <intensity> from prior experience]`` so it understands the
  tool is *known but presently uninvokable* — invoking it would fail
  because the target entity is absent.

Substrate-voice consistency: the intensity phrase ("strongly rewarding",
"mildly aversive", etc.) is produced by Wire-A's ``bias_to_band`` so the
two prompt surfaces (Wire-A's ``=== Substrate associations ===`` block
and grayscale's ``=== Tools known but not in current location ===``
block) tell the LLM a coherent story about the same NAc bias number. A
post-merge review of an earlier W1 draft caught this: rendering the same
tool with a different intensity descriptor in adjacent prompt sections
confused the substrate-voice the agent reads.

Env-var gate: grayscale shares Wire-A's
``MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION`` kill switch. The two sections
are different surfaces of the same substrate signal; disabling one
without the other would leak the signal under a different header and
pollute Roy iteration ablation evidence. The producer's caller is
responsible for honoring the env var (consumer-side gating would not
prevent the producer hot path running).

See [docs/plans/sense_tool_registry.md] § "Phase 3" and the load-bearing
invariant "SEM-derived tools must stay scene-scoped at execution time"
(grayscale is about visibility, not invokability).
"""

from __future__ import annotations

from typing import Any, Protocol

from maxim.prompts.cluster_bias_annotation import bias_to_band, strip_tool_prefix


class _RegistryLike(Protocol):
    """Minimal subset of ToolRegistry the producer reads.

    Defined as a Protocol so the producer can be unit-tested with a
    lightweight stub without importing the full ToolRegistry.
    """

    def list(self) -> list[str]: ...
    def get(self, name: str) -> Any: ...
    def get_tools_by_kind(self, kind: str) -> list[Any]: ...


def build_grayscale_annotations(
    biases: list[tuple[str, float]] | None,
    registry: _RegistryLike | None,
    *,
    top_n: int = 5,
) -> list[tuple[str, float, str]]:
    """Filter NAc biases to the grayscale candidate set.

    Applies four filters in order:

    1. **Prefix strip** — ``NAc.get_agent_tool_biases`` returns
       ``tool:<name>`` (or ``tool:use:<action>``) per the canonical
       ``build_tool_signature`` format. Bare tool names live in the
       registry, so the prefix must be stripped BEFORE set lookups —
       this is the bug a pre-merge review caught (the producer's
       compare keys silently mismatched, leaving the grayscale list
       empty in production).
    2. **Active-list exclusion** — tools already in the active roster
       render in the normal tools block; the grayscale view is for
       inactive-but-knowable tools only.
    3. **SEM-kind requirement** — only ``sem-modulator-derived`` tools
       grayscale. Core tools are always active; non-SEM scene tools
       are out of scope for the MVP.
    4. **top_n cap** — bounds the prompt-section budget. Matches
       Wire-A's ``top_n=5`` default so the two sections render at
       similar weight.

    Returns an empty list when ``biases`` is falsy, when ``registry``
    is None, or when no candidate survives the filter (cold-start
    agents, scenes where every biased tool is already active).
    """
    if not biases or registry is None:
        return []

    try:
        active = set(registry.list())
        sem_names = {t.name for t in registry.get_tools_by_kind("sem-modulator-derived")}
    except AttributeError:
        # Registry doesn't expose the metadata-aware helpers — out-of-
        # tree subclass or partial-build situation. Skip silently;
        # grayscale is purely additive, never load-bearing for
        # correctness.
        return []

    annotations: list[tuple[str, float, str]] = []
    for raw_signature, bias in biases:
        tool_name = strip_tool_prefix(raw_signature)
        if tool_name in active:
            continue
        if tool_name not in sem_names:
            continue
        try:
            description = getattr(registry.get(tool_name), "description", "") or ""
        except KeyError:
            description = ""
        annotations.append((tool_name, bias, description))
        if len(annotations) >= top_n:
            break
    return annotations


def compose_grayscale_tools_section(
    annotations: list[tuple[str, float, str]] | None,
) -> str:
    """Render ``(tool_name, bias, description)`` tuples as a prompt section.

    Returns an empty string when ``annotations`` is None or empty so the
    prompt builder skips the section entirely — no clutter for cold-
    start agents or scenes where the substrate's biased tools are all
    already active.

    Example output:

        === Tools known but not in current location ===
          sense_food_source — Scan for food sources [not in current location, strongly rewarding from prior experience]
          forest_gather     — Gather forage in the forest [not in current location, neutral / mixed]

    The annotation has two parts in brackets:
    - ``not in current location`` (always present) — declares the tool
      is knowable but presently uninvokable.
    - ``<intensity band>`` — Wire-A's ``bias_to_band`` mapping. "neutral
      / mixed" renders without the "from prior experience" suffix
      (matches Wire-A's renderer); the non-neutral bands carry the
      suffix because they encode decision-relevant signal.
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

        band = bias_to_band(bias)
        if band == "neutral / mixed":
            bracket = f"not in current location, {band}"
        else:
            bracket = f"not in current location, {band} from prior experience"

        name_padded = name.ljust(max_name_len)
        if trimmed_desc:
            lines.append(f"  {name_padded} — {trimmed_desc} [{bracket}]")
        else:
            lines.append(f"  {name_padded} [{bracket}]")
    return "\n".join(lines)


__all__ = [
    "build_grayscale_annotations",
    "compose_grayscale_tools_section",
]
