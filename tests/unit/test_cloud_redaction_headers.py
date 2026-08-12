"""Cloud redaction ↔ PromptBuilder header pins (2026-08-12 privacy audit).

The audit found ``_SECTION_RULES`` had silently drifted from the live
prompt format: the ``memory_contents`` rule matched the legacy
``RELEVANT MEMORIES:`` marker while PromptBuilder emits
``=== Relevant Memories ===``, and the ``workspace_manifest`` rule
matched a header no producer emits — so hippocampus recalls and
workspace file listings shipped to cloud verbatim under every policy
including ``strict``, with zero test coverage. These tests close both
directions of that drift:

1. Structural: the exact header literals the rules depend on must
   exist in the PromptBuilder source, so a header rename fails here
   and forces a rule update.
2. Behavioral: a prompt carrying the live sensitive sections must come
   back redacted under strict/standard and intact under relaxed.
"""

from __future__ import annotations

from pathlib import Path

import maxim.agents.prompt_builder as prompt_builder_mod
from maxim.utils.cloud_redaction import _SECTION_RULES, CloudRedactionFilter


def _rules_match(header: str) -> bool:
    return any(pattern.match(header.strip()) for pattern, _, _, _ in _SECTION_RULES)


def _make_filter(policy: str) -> CloudRedactionFilter:
    return CloudRedactionFilter.from_config(global_cfg={"policy": policy})


# Live sensitive headers, with representative parameterized instances.
_LIVE_SENSITIVE_HEADERS = [
    "=== Relevant Memories ===",
    "=== Conversation History ===",
    "=== Recent Action Outcomes ===",
    "=== Recent Speech ===",
    "=== Context ===",
    "=== PROJECT DIRECTORY (12 entries, CWD: Maxim) ===",
    "=== EXISTING WORKSPACE (1 file) ===",
    "=== EXISTING WORKSPACE (7 files) ===",
    "=== Your inner deliberation (private — not speech) ===",
    "=== Your prior reasoning ===",
]

# Source literals the rules are pinned against. If PromptBuilder renames
# one of these, this test fails — update BOTH the producer and the
# matching rule in cloud_redaction._SECTION_RULES in the same commit.
_PINNED_SOURCE_LITERALS = [
    '"=== Relevant Memories ==="',
    '"=== Conversation History ===\\n"',
    '"=== Recent Action Outcomes ==="',
    '"=== Recent Speech ==="',
    '"=== Context ===\\n"',
    'f"=== PROJECT DIRECTORY ({n_entries} entries, CWD: {cwd_name}) ==="',
    "f\"=== EXISTING WORKSPACE ({n_files} file{'s' if n_files != 1 else ''}) ===\"",
    '"=== Your inner deliberation (private — not speech) ==="',
    '"=== Your prior reasoning ==="',
]


class TestSectionRulesPinnedToPromptBuilder:
    def test_every_live_sensitive_header_matches_a_rule(self) -> None:
        for header in _LIVE_SENSITIVE_HEADERS:
            assert _rules_match(header), f"no _SECTION_RULES pattern matches live header {header!r}"

    def test_pinned_literals_exist_in_prompt_builder_source(self) -> None:
        source = Path(prompt_builder_mod.__file__).read_text()
        for literal in _PINNED_SOURCE_LITERALS:
            assert literal in source, (
                f"pinned header literal {literal} no longer in prompt_builder.py — "
                f"update _SECTION_RULES in cloud_redaction.py and this pin together"
            )


class TestRedactionBehaviorOnLiveFormat:
    """The audit's empirical repro, kept as a regression guard."""

    _USER_PROMPT = "\n".join(
        [
            "=== Relevant Memories ===",
            "- [episodic, salience=0.91] user said their password hint is SENTINEL_MEMORY",
            "",
            "=== EXISTING WORKSPACE (2 files) ===",
            "notes_on_denny.txt",
            "SENTINEL_FILENAME.txt",
            "=== Your inner deliberation (private — not speech) ===",
            "SENTINEL_DELIBERATION plan to persuade the user",
            "=== Instructions ===",
            "Do the thing.",
        ]
    )

    def test_strict_redacts_memories_workspace_deliberation(self) -> None:
        result = _make_filter("strict").redact("system prompt", self._USER_PROMPT)
        assert "SENTINEL_MEMORY" not in result.user
        assert "SENTINEL_FILENAME" not in result.user
        assert "SENTINEL_DELIBERATION" not in result.user
        assert "[REDACTED: memory_contents]" in result.user
        assert "[REDACTED: workspace_manifest]" in result.user
        assert "[REDACTED: inner_deliberation]" in result.user
        # Non-sensitive sections survive.
        assert "Do the thing." in result.user

    def test_standard_matches_strict_but_reports_its_own_name(self) -> None:
        result = _make_filter("standard").redact("system prompt", self._USER_PROMPT)
        assert "SENTINEL_MEMORY" not in result.user
        assert result.policy == "standard"

    def test_relaxed_passes_private_sections_through(self) -> None:
        result = _make_filter("relaxed").redact("system prompt", self._USER_PROMPT)
        assert "SENTINEL_MEMORY" in result.user
        assert "SENTINEL_DELIBERATION" in result.user
