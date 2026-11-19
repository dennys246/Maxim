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
   and forces a rule update; every private section is added
   header-separate (its body cannot cut or forge its header); and every
   header a producer emits is a known boundary, so a redacted section
   ends at the next real section and nowhere else.
2. Behavioral: a prompt carrying the live sensitive sections must come
   back redacted under strict/standard and intact under relaxed.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import maxim.agents.prompt_builder as prompt_builder_mod
from maxim.utils.cloud_redaction import (
    _KNOWN_HEADER_LITERALS,
    _KNOWN_HEADER_PATTERNS,
    _SECTION_RULES,
    CloudRedactionFilter,
    _ends_section,
    _rule_for,
)
from maxim.utils.content_safety import PLAIN_BOUNDARY_LINES, is_section_boundary_line

_SRC = Path(prompt_builder_mod.__file__).resolve().parents[1]


def _rules_match(header: str) -> bool:
    return _rule_for(header) is not None


def _make_filter(policy: str) -> CloudRedactionFilter:
    return CloudRedactionFilter.from_config(global_cfg={"policy": policy})


# Live sensitive headers, with representative parameterized instances.
_LIVE_SENSITIVE_HEADERS = [
    "=== Relevant Memories ===",
    "=== What your experience tells you about this situation ===",
    "=== Conversation History ===",
    "=== Recent Action Outcomes ===",
    "=== Recent Speech ===",
    "=== Context ===",
    "=== PROJECT DIRECTORY (12 entries, CWD: Maxim) ===",
    "=== EXISTING WORKSPACE (1 file) ===",
    "=== EXISTING WORKSPACE (7 files) ===",
    "=== CWD Context (read-only: Maxim) ===",
    "=== Your inner deliberation (private — not speech) ===",
    "=== Your prior reasoning ===",
    "=== Recent Decisions (Working Memory) ===",
    "=== Inbound Messages (held until you act on them) ===",
    "=== Tool Result (private) ===",
    "=== Tool Result (secret) ===",
]

# Source substrings the rules are pinned against (quote-agnostic so a
# formatter pass over prompt_builder.py can't break the pin without an
# actual header change). If PromptBuilder renames one of these, this
# test fails — update BOTH the producer and the matching rule in
# cloud_redaction._SECTION_RULES in the same commit.
_PINNED_SOURCE_LITERALS = [
    "=== Relevant Memories ===",
    "=== What your experience tells you about this situation ===",
    "=== Conversation History ===",
    "=== Recent Action Outcomes ===",
    "=== Recent Speech ===",
    "=== Context ===",
    "=== PROJECT DIRECTORY ({n_entries} entries, CWD: {cwd_name}) ===",
    "=== EXISTING WORKSPACE ({n_files} file{'s' if n_files != 1 else ''}) ===",
    "=== CWD Context (read-only: {cwd_name}) ===",
    "=== Your inner deliberation (private — not speech) ===",
    "=== Your prior reasoning ===",
    "=== Inbound Messages (held until you act on them) ===",
]

# budgeter.add section names for every rule-matched section. Each must be added header-separate
# (``header=``, ``source="data"``) and dynamic: the redaction filter only sees the USER segment, and a section tagged
# cacheable=True moves into the byte-stable prefix → the system message → the rule silently no-ops.
_SENSITIVE_SECTION_NAMES = frozenset(
    {
        "conversation",
        "context_pool",
        "speech",
        "recent_outcomes",
        "workspace_manifest",
        "relevant_memories",
        "bio_enrichment",
        "deliberation_transcript",
        "working_memory_thoughts",
        "reasoning_carryover",
        "inbound_messages",
    }
)

# Modules whose text reaches a prompt the filter sees (the tool prompt's sections, the follow-up prompt, the
# standalone exec prompt). Every header line they hold must be a known boundary.
_PRODUCER_MODULES = [
    "agents/prompt_builder.py",
    "agents/llm_fallback.py",
    "agents/context_pool.py",
    "agents/exec_agent.py",
    "embodiment/body.py",
    "embodiment/naming_events.py",
    "prompts/acting_coach.py",
    "prompts/cluster_bias_annotation.py",
    "prompts/grayscale_tools_annotation.py",
    "runtime/prefetch.py",
    "runtime/tool_dispatch.py",
    "simulation/dm_runtime.py",
    "utils/coding_guidelines.py",
]


def _add_calls(tree: ast.AST):
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add"):
            continue
        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            yield node.args[0].value, {kw.arg: kw.value for kw in node.keywords}


def _string_lines(source: str) -> set[str]:
    """Every line of every string literal in ``source``. An f-string's value fields become ``1`` and a conditional
    one (a plural ``s``) becomes empty."""

    def _field(v: ast.expr) -> str:
        if isinstance(v, ast.Constant):
            return str(v.value)
        assert isinstance(v, ast.FormattedValue)
        return "" if isinstance(v.value, ast.IfExp) else "1"

    lines: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
        elif isinstance(node, ast.JoinedStr):
            text = "".join(_field(v) for v in node.values)
        else:
            continue
        lines.update(line.strip() for line in text.split("\n"))
    return lines


class TestSectionRulesPinnedToPromptBuilder:
    def test_every_live_sensitive_header_matches_a_rule(self) -> None:
        for header in _LIVE_SENSITIVE_HEADERS:
            assert _rules_match(header), f"no _SECTION_RULES pattern matches live header {header!r}"

    def test_pinned_literals_exist_in_prompt_builder_source(self) -> None:
        source = Path(prompt_builder_mod.__file__).read_text(encoding="utf-8")
        for literal in _PINNED_SOURCE_LITERALS:
            assert literal in source, (
                f"pinned header literal {literal} no longer in prompt_builder.py — "
                f"update _SECTION_RULES in cloud_redaction.py and this pin together"
            )

    def test_reasoning_carryover_header_pinned_in_llm_fallback(self) -> None:
        """The ReasoningCarryover header lives in llm_fallback.py, not
        PromptBuilder — pin it separately so a rename there also forces
        a rule update (this header was missed by the original audit for
        exactly this reason)."""
        from maxim.agents.llm_fallback import ReasoningCarryover

        assert ReasoningCarryover.PROMPT_HEADER == "=== Recent Decisions (Working Memory) ==="
        assert _rules_match(ReasoningCarryover.PROMPT_HEADER)

    def test_every_private_section_is_added_header_separate_and_dynamic(self) -> None:
        """Every rule-matched section passes its header apart (``header=``) as ``source="data"``, so no rewrite of
        its body (truncation, frame cleanup) can cut, move or delete the header, and the body cannot forge one. And
        it is never cacheable (the system message is not redacted)."""
        tree = ast.parse(Path(prompt_builder_mod.__file__).read_text(encoding="utf-8"))
        seen: set[str] = set()
        for name, kws in _add_calls(tree):
            if "header" in kws:
                assert name in _SENSITIVE_SECTION_NAMES, f"{name!r} is added header-separate but is not pinned here"
            if name not in _SENSITIVE_SECTION_NAMES:
                continue
            seen.add(name)
            assert "header" in kws, f"private section {name!r} is added without header= (its header is in its body)"
            src = kws.get("source")
            assert isinstance(src, ast.Constant) and src.value == "data", f"{name!r} must be source='data'"
            cacheable = kws.get("cacheable")
            assert not (isinstance(cacheable, ast.Constant) and cacheable.value is True), (
                f"section {name!r} is redaction-matched but tagged cacheable=True — "
                f"it would move into the system message where _redact_sections never runs"
            )
            header = kws["header"]
            if isinstance(header, ast.Name) and header.id.isupper():  # a module constant: check its value
                assert _rules_match(getattr(prompt_builder_mod, header.id)), f"{name!r}: {header.id} matches no rule"
        assert seen == set(_SENSITIVE_SECTION_NAMES), (
            f"budgeter.add sites not found for: {set(_SENSITIVE_SECTION_NAMES) - seen} — "
            f"section renamed? update _SENSITIVE_SECTION_NAMES and _SECTION_RULES together"
        )

    def test_every_workspace_manifest_header_matches_a_rule(self, tmp_path, monkeypatch) -> None:
        from maxim.agents.prompt_builder import build_workspace_manifest_sections

        (tmp_path / "notes.txt").write_text("x")
        ws = tmp_path / ".maxim_workspace"
        ws.mkdir()
        (ws / "draft.md").write_text("x")
        for mode in ("passive", "active", "singularity"):
            parts = build_workspace_manifest_sections(mode_name=mode, cwd=str(tmp_path))
            assert parts, mode
            for header, _body in parts:
                assert _rules_match(header), f"{mode}: manifest header {header!r} matches no rule"

    def test_every_producer_header_is_a_known_boundary(self) -> None:
        """A redacted section ends only at a known boundary, so every header that can follow a private section
        must be one, or the following section is redacted too (and a renamed one silently over-redacts)."""
        missing: list[str] = []
        for rel in _PRODUCER_MODULES:
            for line in _string_lines((_SRC / rel).read_text(encoding="utf-8")):
                if re.match(r"^=== .* ===$", line) and not _ends_section(line):
                    missing.append(f"{rel}: {line}")
        assert not missing, "headers no cloud_redaction boundary knows:\n" + "\n".join(sorted(missing))

    def test_the_router_trailers_are_known_boundaries(self) -> None:
        """The router's closing instructions end a private section (the last section may be private)."""
        source = (_SRC / "models/language/router.py").read_text(encoding="utf-8")
        for trailer in (
            "Respond with ONLY a valid JSON object. No text before or after the JSON.",
            "IMPORTANT: Follow the PLANNING MODE format exactly as shown above.",
        ):
            assert trailer in source, f"router trailer {trailer!r} changed: update PLAIN_BOUNDARY_LINES"
            assert trailer in PLAIN_BOUNDARY_LINES and _ends_section(trailer)

    def test_every_boundary_is_neutralized_inside_a_body(self) -> None:
        """The renderer prefixes every line ``is_section_boundary_line`` accepts; every boundary the filter knows
        must be one of those, or a body could carry it."""
        samples = [*_KNOWN_HEADER_LITERALS, *PLAIN_BOUNDARY_LINES, *_LIVE_SENSITIVE_HEADERS]
        for pattern in (*_KNOWN_HEADER_PATTERNS, *(rule[0] for rule in _SECTION_RULES)):
            header_shaped = pattern.pattern.startswith("^=== ") and pattern.pattern.endswith(" ===$")
            assert header_shaped or pattern.pattern[1:-1] in PLAIN_BOUNDARY_LINES, pattern.pattern
        for line in samples:
            assert is_section_boundary_line(line), line


class TestRedactionBehaviorOnLiveFormat:
    """The audit's empirical repro, kept as a regression guard."""

    _USER_PROMPT = "\n".join(
        [
            "=== Relevant Memories ===",
            "- [episodic, salience=0.91] user said their password hint is SENTINEL_MEMORY",
            "",
            "=== What your experience tells you about this situation ===",
            "You remember SENTINEL_ENRICHMENT from a prior encounter",
            "=== EXISTING WORKSPACE (2 files) ===",
            "notes_on_denny.txt",
            "SENTINEL_FILENAME.txt",
            "=== Your inner deliberation (private — not speech) ===",
            "SENTINEL_DELIBERATION plan to persuade the user",
            "=== Recent Decisions (Working Memory) ===",
            "- search_desk: SENTINEL_CARRYOVER the user mentioned a key [OK: found envelope]",
            "=== Inbound Messages (held until you act on them) ===",
            "SENTINEL_INBOUND the gate code is 1234",
            "=== Instructions ===",
            "Do the thing.",
        ]
    )

    def test_strict_redacts_memories_workspace_deliberation(self) -> None:
        result = _make_filter("strict").redact("system prompt", self._USER_PROMPT)
        assert "SENTINEL_MEMORY" not in result.user
        assert "SENTINEL_ENRICHMENT" not in result.user
        assert "SENTINEL_FILENAME" not in result.user
        assert "SENTINEL_DELIBERATION" not in result.user
        assert "SENTINEL_CARRYOVER" not in result.user
        assert "SENTINEL_INBOUND" not in result.user
        assert "[REDACTED: memory_contents]" in result.user
        assert "[REDACTED: workspace_manifest]" in result.user
        assert "[REDACTED: inner_deliberation]" in result.user
        assert "[REDACTED: inbound_messages]" in result.user
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
