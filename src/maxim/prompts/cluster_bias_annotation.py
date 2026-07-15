"""Cluster bias annotation — Wire-A prompt section (0.9.1 Stage 2).

Renders ``NAc.get_agent_tool_biases`` output as a structured prompt
block so the LLM proposer can read substrate-acquired tool-level
reward signal across percept regimes the substrate didn't directly
drill. The interim signal-surfacing fix for the Roy-2c finding (the
cluster wire has the right tool keys but wrong cluster keys; LLM
proposer never sees the cluster-keyed bias map otherwise).

**Why agent-wide aggregation, not active-cluster restriction:**
Roy-2c proved priming-acquired EC clusters and engineered test-fixture
EC clusters are structurally disjoint under the LinguisticEncoder gap.
Restricting Wire-A's rendering to "clusters that match the current
percept's active set" reproduces exactly the bug this wire exists to
fix — the active-cluster intersection with priming clusters is empty
in the failure mode. The plan calls this out explicitly in
[release_0_9_1.md § Stage 2](docs/plans/archive/release_0_9_1.md).

**Why static signal-surfacing, not dynamic edge formation:** Wire-A
renders associations the substrate already encoded via tool-name keys
that survive the encoder gap. The *structural* fix for cross-modal
edge formation is [cross_modal_substrate_binding.md](
docs/plans/archive/cross_modal_substrate_binding.md) (cancelled by Roy-4) or
[jepa_cross_modal_alignment.md](docs/plans/deferred/jepa_cross_modal_alignment.md)
(1.2+ research direction). Wire-A coexists with whichever structural
fix eventually ships; the llm-primary proposer doesn't consume
bound-edge neighborhoods anyway, so Wire-A remains the llm-primary
read surface even after the structural fix lands.

**Why hand-coded bias bands are bio-defensible:** the 5 thresholds
(0.5 / 0.1 / -0.1 / -0.5) are *display-layer translation* for the LLM
consumer, not substrate encoding. Specifically:

(a) The bias **values** are 100% substrate-earned via
``NAc.update_cluster_reward``'s ``reward_bias_alpha`` accumulation
over Roy-priming reward signals. No human picks them.

(b) The band **labels** map those numeric values to phrases the LLM
proposer can read. The substrate didn't earn the word "strongly
rewarding" — humans picked it as the most evocative gloss for
``bias >= 0.5``.

(c) No band-derived signal flows back into encoders, EC, NAc state, or
any other substrate write path. The bands are pure I/O.

Per [feedback_interim_contamination.md](
.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_interim_contamination.md),
the canonical contamination pattern is "hand-curated decisions
UPSTREAM of substrate encoding." Wire-A's bands are downstream of
encoding — the substrate produces the bias, the bands narrate it.
This is the same shape as ``decision_engine`` rendering numeric
confidences as text for log lines: the rendering layer adds nothing
substrate didn't already say.
"""

from __future__ import annotations

# Bias-band thresholds — pre-registered in release_0_9_1.md Stage 2.
# Tests pin these exact boundaries; do not relax without a new plan.
_BAND_STRONG_REWARDING = 0.5
_BAND_MILD_REWARDING = 0.1
_BAND_NEUTRAL_LOWER = -0.1
_BAND_MILD_AVERSIVE = -0.5

# Truthy values that disable Wire-A's annotation at the agent-loop
# producer site. Shared with the conftest scrub and the test suite
# so a future env-var divergence here would be caught by tests AND
# match the producer's actual parsing. Matches the spelling pattern
# the rest of 0.9.1's opt-in env-var gates use (e.g.,
# similarity/ec.py::_ec_trace_enabled inverts this with a falsy-set
# for opt-in semantics).
TRUTHY_DISABLE_VALUES: frozenset[str] = frozenset({"1", "true", "t", "yes", "y", "on"})


def annotation_disabled_via_env(raw: str | None) -> bool:
    """Return True when the env-var value matches a disable signal.

    Centralized parser so the producer (``agent_loop.py``), the
    conftest scrub, and the test suite all consult one truthy-set
    definition. Case-insensitive; whitespace-tolerant. ``None`` /
    empty / unrecognized values return False (annotation stays ON,
    matching the 0.9.1 default).
    """
    if not raw:
        return False
    return raw.strip().lower() in TRUTHY_DISABLE_VALUES


def bias_to_band(bias: float) -> str:
    """Map a numeric reward bias to a human-readable band label.

    Five bands per the plan:
    - ``bias >= 0.5``         → ``"strongly rewarding"``
    - ``0.1 <= bias < 0.5``   → ``"mildly rewarding"``
    - ``-0.1 <= bias < 0.1``  → ``"neutral / mixed"``
    - ``-0.5 < bias < -0.1``  → ``"mildly aversive"``
    - ``bias <= -0.5``        → ``"strongly aversive"``

    Boundary semantics are inclusive on the rewarding side (>=) and
    exclusive on the negative band's lower edge (so -0.5 EXACTLY
    decodes to ``"strongly aversive"``, not ``"mildly aversive"``).
    """
    if bias >= _BAND_STRONG_REWARDING:
        return "strongly rewarding"
    if bias >= _BAND_MILD_REWARDING:
        return "mildly rewarding"
    if bias > _BAND_NEUTRAL_LOWER:
        # -0.1 < bias < 0.1
        return "neutral / mixed"
    if bias > _BAND_MILD_AVERSIVE:
        # -0.5 < bias <= -0.1
        return "mildly aversive"
    return "strongly aversive"


def _strip_tool_prefix(tool_signature: str) -> str:
    """Strip the ``tool:`` prefix produced by ``build_tool_signature``.

    The signature format is ``tool:<name>`` or ``tool:use:<action>``
    (the generic ``use`` tool keys per-action). The annotation block
    renders bare tool names so the LLM reads the surface form it
    sees in its tool description list.
    """
    if tool_signature.startswith("tool:"):
        return tool_signature[len("tool:") :]
    return tool_signature


def compose_cluster_bias_annotation_section(
    biases: list[tuple[str, float]] | None,
) -> str:
    """Render top-N (tool, bias) pairs as a prompt section.

    Returns an empty string when ``biases`` is None, empty, or contains
    only neutral entries — the prompt builder skips empty sections so
    the AUT prompt stays uncluttered when the substrate has no
    learned signal to share.

    Layout matches the plan's example output:

        === Substrate associations from prior experience ===
          sense_food_source       [strongly rewarding from prior experience]
          infant_humanoid_pick_up [neutral / mixed]

    The "from prior experience" suffix is appended to the rewarding /
    aversive bands but not the neutral band, since neutral signals are
    informational rather than directive. This is a deliberate
    rendering choice for the LLM's reading: the rewarding/aversive
    bands carry decision-relevant content; neutral entries surface
    "the substrate has seen this tool but doesn't lean either way."
    """
    if not biases:
        return ""

    # Filter out neutral entries IF they would be the only entries —
    # an all-neutral block adds tokens without signal. Keep neutral
    # entries when they appear alongside non-neutral ones, since the
    # ordering itself is informative ("these tools have signal; these
    # don't").
    has_non_neutral = any(bias_to_band(b) != "neutral / mixed" for _, b in biases)
    if not has_non_neutral:
        return ""

    lines: list[str] = ["=== Substrate associations from prior experience ==="]
    # Compute the max tool-name width for column-aligned annotations.
    # ``default=0`` defends against a future predicate filtering rendered
    # to empty after the upstream non-empty guards passed — current code
    # paths can't reach this, but max([]) raises ValueError and that's
    # a silent-noise failure under a future caller addition.
    rendered = [(_strip_tool_prefix(tool), bias) for tool, bias in biases]
    max_name_len = max((len(name) for name, _ in rendered), default=0)
    for name, bias in rendered:
        band = bias_to_band(bias)
        if band == "neutral / mixed":
            annotation = f"[{band}]"
        else:
            annotation = f"[{band} from prior experience]"
        # Two-space minimum gap; pad name to max width.
        lines.append(f"  {name.ljust(max_name_len)}  {annotation}")
    return "\n".join(lines)


__all__ = [
    "TRUTHY_DISABLE_VALUES",
    "annotation_disabled_via_env",
    "bias_to_band",
    "compose_cluster_bias_annotation_section",
    "strip_tool_prefix",
]


# Public alias for the prefix-stripper. W1's grayscale producer needs the
# same prefix discipline, so the helper is exported rather than copied.
# Keeping the underscored name for backward compatibility with any
# in-module callers.
strip_tool_prefix = _strip_tool_prefix
