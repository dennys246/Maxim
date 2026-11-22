"""Cloud redaction filter for LLM prompts."""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Any

from maxim.utils.content_safety import (
    PLAIN_BOUNDARY_LINES,
    frame_spans,
    is_blank_line,
    is_section_boundary_line,
    split_lines,
)


class DataSensitivity(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    PRIVATE = "private"
    SECRET = "secret"


_PUBLIC, _INTERNAL, _PRIVATE, _SECRET = (
    DataSensitivity.PUBLIC,
    DataSensitivity.INTERNAL,
    DataSensitivity.PRIVATE,
    DataSensitivity.SECRET,
)

# What each tool's result is, for the follow-up prompt (``tool_result_header``). Every registered tool
# (modes/definitions.py::REGISTERED_TOOL_NAMES) has an entry, pinned by tests/unit/test_cloud_redaction_headers.py, so
# a new tool is classified on purpose; a name not here (a tool generated per entity at run time) is private.
# PUBLIC: content from the open web. INTERNAL: the agent's own state and simulated/embodied actions. PRIVATE: files,
# memories, perception of people and sound, messages, the agent's own reasoning. SECRET: command execution.
TOOL_SENSITIVITY: dict[str, DataSensitivity] = {
    # the open web
    "internet_search": _PUBLIC,
    "web_search": _PUBLIC,
    "http_fetch": _PUBLIC,
    # files, code and the workspace
    "read_file": _PRIVATE,
    "write_file": _PRIVATE,
    "edit_file": _PRIVATE,
    "list_directory": _PRIVATE,
    "glob": _PRIVATE,
    "search_code": _PRIVATE,
    "git_diff": _PRIVATE,
    "git_commit": _PRIVATE,
    "read_data_file": _PRIVATE,
    "read_sandbox_file": _PRIVATE,
    "write_sandbox_file": _PRIVATE,
    "list_sandbox": _PRIVATE,
    "create_sandbox_script": _PRIVATE,
    "read_other_output": _PRIVATE,
    "list_other_outputs": _PRIVATE,
    "write_shared_output": _PRIVATE,
    "request_directory_change": _PRIVATE,
    "batched_exploration": _PRIVATE,
    # command execution
    "bash": _SECRET,
    "execute_file": _SECRET,
    "execute_sandbox_script": _SECRET,
    "run_tests": _SECRET,
    # messages and people
    "send_message": _PRIVATE,
    "call_user": _PRIVATE,
    "sense_presence": _PRIVATE,
    "focus_on_sound": _PRIVATE,
    # memories and the agent's own reasoning
    "memory_recall": _PRIVATE,
    "similarity_search": _PRIVATE,
    "concept_query": _PRIVATE,
    "causal_links": _PRIVATE,
    "temporal_patterns": _PRIVATE,
    "think": _PRIVATE,
    # the agent's own state, controls and replies
    "respond": _INTERNAL,
    "maxim_command": _INTERNAL,
    "mode_switch": _INTERNAL,
    "autonomy_level": _INTERNAL,
    "display_mode": _INTERNAL,
    "internet_access_toggle": _INTERNAL,
    "energy_status": _INTERNAL,
    "system_stats": _INTERNAL,
    "explain": _INTERNAL,
    "math": _INTERNAL,
    "sleep": _INTERNAL,
    "pain_history": _INTERNAL,
    "sense_tools": _INTERNAL,
    "focus_interests": _INTERNAL,
    "novelty_track": _INTERNAL,
    "predict_outcome": _INTERNAL,
    # embodied and simulated actions and worlds
    "move": _INTERNAL,
    "sense": _INTERNAL,
    "say": _INTERNAL,
    "speak": _INTERNAL,
    "track_target": _INTERNAL,
    "examine": _INTERNAL,
    "choose": _INTERNAL,
    "actor": _INTERNAL,
    "request_interaction": _INTERNAL,
    "set_entity_sensor": _INTERNAL,
    "inject_pain": _INTERNAL,
    "damage_component": _INTERNAL,
    "inspect_aut": _INTERNAL,
    "observe_actions": _INTERNAL,
    "design_entity": _INTERNAL,
    "browse_components": _INTERNAL,
    "browse_encounters": _INTERNAL,
    "set_scene": _INTERNAL,
    "scene_summary": _INTERNAL,
    "generate_scenario": _INTERNAL,
    "emit_campaign": _INTERNAL,
    "extend_simulation": _INTERNAL,
    "finish_simulation": _INTERNAL,
    "spawn_sub_simulation": _INTERNAL,
    "check_completion": _INTERNAL,
    "query_experiments": _INTERNAL,
    "record_experiment": _INTERNAL,
    "analyze_results": _INTERNAL,
}

MAXIM_COMMAND_SENSITIVITY: dict[str, DataSensitivity] = {
    "request_shutdown": DataSensitivity.SECRET,
    "request_sleep": DataSensitivity.PRIVATE,
    "request_observe": DataSensitivity.INTERNAL,
    "goto_pose": DataSensitivity.PRIVATE,
    "move": DataSensitivity.SECRET,
    "look_at_image": DataSensitivity.PRIVATE,
}


def tool_result_header(tool_name: str) -> str:
    """The header a follow-up prompt puts above a tool's result. It names the tool's sensitivity
    (``TOOL_SENSITIVITY``), so the filter redacts the result by the class its tool belongs to. A tool missing from
    the table (or "" for an unparseable follow-up) is private: the filter fails closed."""
    sensitivity = TOOL_SENSITIVITY.get(tool_name, DataSensitivity.PRIVATE)
    return f"=== Tool Result ({sensitivity.value}) ==="


@dataclass(slots=True)
class RedactionPolicy:
    name: str
    allow_internal: bool = True
    allow_private: bool = False
    allow_secret: bool = False


@dataclass(slots=True)
class RedactionResult:
    system: str
    user: str
    categories_sent: list[str]
    categories_redacted: list[str]
    policy: str


# What the LLM router actually sent in each successful cloud call made under ``record_dispatches`` (the caller's
# context, carried into the router's worker thread by ``contextvars.copy_context``). A caller that must know whether
# a section reached the model (the inbound acknowledgement) reads it instead of the prompt it built.
_DISPATCHES: ContextVar[list[RedactionResult] | None] = ContextVar("maxim_redaction_dispatches", default=None)


@contextmanager
def record_dispatches() -> Iterator[list[RedactionResult]]:
    """Collect the redaction result of every successful cloud call made in this context (a local call adds none)."""
    sink: list[RedactionResult] = []
    token = _DISPATCHES.set(sink)
    try:
        yield sink
    finally:
        _DISPATCHES.reset(token)


def note_dispatch(result: RedactionResult | None) -> None:
    """Record what a successful call sent (the router calls it; ``None`` is a call that was not redacted)."""
    sink = _DISPATCHES.get()
    if sink is not None and result is not None:
        sink.append(result)


# Section rules are pinned against the headers PromptBuilder ACTUALLY
# emits (tests/unit/test_cloud_redaction_headers.py) — the 2026-08-12
# privacy audit found the previous rule set had silently drifted from
# the live prompt format: the ``memory_contents`` and
# ``workspace_manifest`` patterns matched no producer, so hippocampus
# recalls and workspace file listings shipped to cloud verbatim under
# every policy including ``strict``. When PromptBuilder renames or adds
# a sensitive section header, the structural test fails and forces a
# rule update here — pattern-based redaction with no header pin is a
# silent no-op waiting to happen.
#
# A redacted ``=== X ===`` section runs from its header to the first blank line (sections are joined by one) or the
# next KNOWN boundary (``_ends_section``: a rule header, a header in ``_KNOWN_HEADER_LITERALS`` /
# ``_KNOWN_HEADER_PATTERNS``, or a plain boundary line such as the router's closing instruction), whichever comes
# first, or to the end of the text. An unknown ``=== X ===`` line never ends it: that errs toward over-redaction.
# Every producer of these sections renders them by identity (``PromptBudgeter.add(header=...)`` or
# ``content_safety.render_headed_section``), so a body holds neither a blank line nor a boundary line; the renderer
# and this filter share ``content_safety.is_blank_line`` and ``is_section_boundary_line``.
_SECTION_RULES: list[tuple[re.Pattern, str, DataSensitivity]] = [
    (re.compile(r"^=== Conversation History ===$"), "conversation_history", DataSensitivity.PRIVATE),
    (re.compile(r"^=== Recent Action Outcomes ===$"), "tool_outputs", DataSensitivity.PRIVATE),
    (re.compile(r"^=== Recent Speech ===$"), "speech_transcripts", DataSensitivity.PRIVATE),
    # workspace_manifest: legacy static header + the live parameterized
    # headers from build_workspace_manifest_sections (the CWD listing is
    # the same file-listing class as the workspace one).
    (re.compile(r"^=== Workspace Manifest ===$"), "workspace_manifest", DataSensitivity.PRIVATE),
    (
        re.compile(r"^=== PROJECT DIRECTORY \(\d+ entries, CWD: .*\) ===$"),
        "workspace_manifest",
        DataSensitivity.PRIVATE,
    ),
    (re.compile(r"^=== EXISTING WORKSPACE \(\d+ files?\) ===$"), "workspace_manifest", DataSensitivity.PRIVATE),
    (re.compile(r"^=== CWD Context \(read-only: .*\) ===$"), "workspace_manifest", DataSensitivity.PRIVATE),
    (re.compile(r"^=== Context ===$"), "context_pool", DataSensitivity.PRIVATE),
    # memory_contents: live PromptBuilder header, the bio-enrichment
    # section (whose EnrichmentResult.memories lines are the same
    # hippocampus recall content — half-redacting memories while their
    # sibling section ships them was the review round's cross-confirmed
    # gap), + legacy exec_agent prompt marker (still emitted by the
    # standalone-agent path).
    (re.compile(r"^=== Relevant Memories ===$"), "memory_contents", DataSensitivity.PRIVATE),
    (
        re.compile(r"^=== What your experience tells you about this situation ===$"),
        "memory_contents",
        DataSensitivity.PRIVATE,
    ),
    # The standalone exec prompt's markers (agents/exec_agent.py::_build_llm_context, an f-string that does not
    # neutralize what it interpolates). A blank line does not end them; a known boundary or the end of the text does.
    (re.compile(r"^RELEVANT MEMORIES:$"), "memory_contents", DataSensitivity.PRIVATE),
    (re.compile(r"^RECENT CLI INPUTS:$"), "cli_inputs", DataSensitivity.PRIVATE),
    # inner_deliberation: the header itself classifies the section as
    # private, and the content is the LLM's own verbatim reasoning —
    # the exact leak class the hivemind bundle scrub closes on the
    # substrate side.
    (
        re.compile(r"^=== Your inner deliberation \(private — not speech\) ===$"),
        "inner_deliberation",
        DataSensitivity.PRIVATE,
    ),
    (re.compile(r"^=== Your prior reasoning ===$"), "inner_deliberation", DataSensitivity.PRIVATE),
    # ReasoningCarryover (llm_fallback.py::ReasoningCarryover.PROMPT_HEADER —
    # NOT a PromptBuilder header): each line is `- {tool}: {reasoning[:80]}
    # [OK|FAIL: {result_summary[:100]}]` — the LLM's own reasoning plus
    # raw tool output, i.e. inner_deliberation + tool_outputs content
    # under a header neither rule matched (found post-review while
    # enumerating what strict still ships).
    (
        re.compile(r"^=== Recent Decisions \(Working Memory\) ===$"),
        "inner_deliberation",
        DataSensitivity.PRIVATE,
    ),
    # Other people's messages held for the agent (#1481): SMS, voice, the command API.
    (
        re.compile(r"^=== Inbound Messages \(held until you act on them\) ===$"),
        "inbound_messages",
        DataSensitivity.PRIVATE,
    ),
    # A follow-up prompt's tool result (tool_result_header), redacted by its tool's sensitivity. The public header is
    # a rule too, so it is a known boundary; _should_redact keeps it.
    *((re.compile(rf"^=== Tool Result \({level.value}\) ===$"), "tool_outputs", level) for level in DataSensitivity),
]

# Every other header a producer emits into a prompt that reaches the filter, pinned against the producers' source by
# tests/unit/test_cloud_redaction_headers.py. Each one ends a redacted section; a parameterized one is a pattern.
_KNOWN_HEADER_LITERALS: frozenset[str] = frozenset(
    f"=== {name} ==="
    for name in (
        "Acting Coach",
        "Active Concepts",
        "Active Protocols",
        "Additional Tools",
        "Agent States",
        "Available Motor Programs",
        "Available Tools",
        "BASH SCRIPTING BEST PRACTICES",
        "BATCHED EXPLORATION RESULTS",
        "BATCHED TOOL CALLS",
        "Body Needs",
        "Body State (pain-relevant)",
        "Body State",
        "Body Tool Discipline",
        "Body Utterances",
        "Budget Context (USD)",
        "CORRECT FORMAT",
        "CORRECT FORMAT EXAMPLE 1",
        "CORRECT FORMAT EXAMPLE 2",
        "CURRENT DATE/TIME (IMPORTANT)",
        "Causal Predictions (learned from experience)",
        "Correction",
        "Current Observation",
        "DOCKER BEST PRACTICES",
        "END BATCHED RESULTS",
        "FILE WORKSPACE",
        "FILE WORKSPACE REMINDER",
        "File Operation Rules",
        "GLOB PATTERN GUIDE",
        "HTML/CSS BEST PRACTICES",
        "Historical Context",
        "Instructions",
        "JAVASCRIPT BEST PRACTICES",
        "JSON BEST PRACTICES",
        "Learned Associations (from experience)",
        "Learned Experience (from your past actions)",
        "MARKDOWN & RESEARCH DOCUMENT GUIDELINES",
        "New Instructions",
        "OPERATIONAL STATE",
        "PYTHON BEST PRACTICES",
        "Physical Exploration",
        "Planning Rule",
        "Predicted Outcomes (from forward models)",
        "RESEARCH DOCUMENT STRUCTURE",
        "Recent Observations",
        "SQL BEST PRACTICES",
        "Scene",
        "Semantic Knowledge",
        "Substrate associations from prior experience",
        "Tool Parameters",
        "Tool Selection",
        "Tools You've Hallucinated",
        "Tools known but not in current location",
        "User Request",
        "WRONG (DO NOT DO THIS)",
        "What you perceive right now",
        "YAML BEST PRACTICES",
        "Your Tools",
    )
)
_KNOWN_HEADER_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"^=== Entity: .* ===$"),
    re.compile(r"^=== DISCOVERY PLAN \(topics: .*\) ===$"),
    re.compile(r"^=== FILE DISCOVERY \(topic: .*\) ===$"),
    re.compile(r"^=== Statistical Patterns \(\d+ active\) ===$"),
)

# A frame's closer (``<</TOOL_OUTPUT id=…>>``, ``utils/content_safety.py::frame_tool_output``) is not a path: redacting it
# left every framed body unclosed (#1519). Frame bodies have their own marker lookalikes defanged, so only a real
# closer reads this way.
_PATH_PATTERN = re.compile(r"(?:(?:[A-Za-z]:\\)|/(?!TOOL_OUTPUT id=[0-9a-f]+>>))[^ \n\t\r]+")
_SECRET_PATTERN = re.compile(r"(?i)(api[_-]?key|secret|token|bearer)\s*[:=]\s*[^ \n\t\r]+")


def _rule_for(line: str) -> tuple[str, DataSensitivity] | None:
    """The (category, sensitivity) of the private section ``line`` opens, or None."""
    stripped = line.strip()
    for pattern, category, sensitivity in _SECTION_RULES:
        if pattern.match(stripped):
            return category, sensitivity
    return None


def _opening_rule(line: str, start: int, spans: list[tuple[int, int]]) -> tuple[str, DataSensitivity, str] | None:
    """The (category, sensitivity, header) of the private section the ``\n``-line ``line`` (at offset ``start`` of
    its text) opens: the line itself, or, failing closed (#1519), a piece of it that another reader shows as a line
    of its own (``content_safety.split_lines``: a U+2028 or other break the renderer did not normalize). A real
    private header is always a line of its own by construction (``check_section_header``, sections joined by
    ``\n``), so a piece opens only on header-shaped DATA, and the cost is over-redaction. A line or piece wholly
    inside a complete frame (``_in_frame``) is framed body and opens nothing; the rest of a frame closer's line is
    outside its frame and is read. Only an opening is read by pieces; a section ENDS on ``\n`` lines only
    (``_ends_section``), so a piece can start a redaction but never cut one short."""
    pieces = [(start, line)]
    offset = start
    for chunk in split_lines(line, keepends=True):
        pieces.append((offset, split_lines(chunk)[0]))  # the piece, its break left off
        offset += len(chunk)
    for piece_start, piece in pieces:
        if _in_frame(spans, piece_start, piece_start + len(piece)):
            continue
        rule = _rule_for(piece)
        if rule is not None:
            return rule[0], rule[1], piece
    return None


def _section_openings(text: str) -> tuple[list[str], list[tuple[str, DataSensitivity, str] | None], list[bool]]:
    """``text``'s ``\n``-lines, the private section each opens (``_opening_rule``, or None), and whether each lies
    wholly inside a complete frame (``_in_frame``): the one reading the redactor, the prefix gate
    (``has_private_section``) and the sent-categories report all use (#1519)."""
    lines = text.split("\n")
    spans, _, _ = frame_spans(text)
    openings: list[tuple[str, DataSensitivity, str] | None] = []
    inside: list[bool] = []
    offset = 0
    for line in lines:
        openings.append(_opening_rule(line, offset, spans))
        inside.append(_in_frame(spans, offset, offset + len(line)))
        offset += len(line) + 1
    return lines, openings, inside


def has_private_section(text: str) -> bool:
    """Whether ``text`` holds a line that opens a private section (any rule, whatever the policy): exactly where the
    redactor would open one (``_section_openings``), so the router's prefix gate is as fail-closed as the redactor."""
    return any(rule is not None for rule in _section_openings(text)[1])


def _ends_at_blank(header_line: str) -> bool:
    """Whether the section ``header_line`` opens ends at its first blank line: every ``=== X ===`` section is
    rendered by identity, the exec prompt's plain markers are not (their text may hold blank lines)."""
    return header_line.strip() not in PLAIN_BOUNDARY_LINES


def _ends_section(line: str) -> bool:
    """Whether ``line`` is a known section boundary. Only a line ``is_section_boundary_line`` accepts can be one, so
    a neutralized body line never is."""
    if not is_section_boundary_line(line):
        return False
    stripped = line.strip()
    return (
        stripped in _KNOWN_HEADER_LITERALS
        or stripped in PLAIN_BOUNDARY_LINES
        or any(p.match(stripped) for p in _KNOWN_HEADER_PATTERNS)
        or _rule_for(stripped) is not None
    )


def _in_frame(spans: list[tuple[int, int]], start: int, end: int) -> bool:
    """Whether the text at ``[start, end)`` lies wholly inside a COMPLETE untrusted-content frame (#1519), by the one
    pairing rule (``content_safety.frame_spans``), past its opener. Such text is framed external data, so it is body
    whatever it says: a framed blank line never ends a PRIVATE section early, and a framed private-shaped header is
    not itself redacted. Defense in depth: ``frame_tool_output`` already quotes every body line that reads as a
    header or a boundary, so a real frame's body holds neither. Fail-closed: an opener whose closer never comes (a
    cut), one repeated before its closer, and a pair whose span crosses a section header (``content_safety.
    is_section_header``, which every header in ``_SECTION_RULES`` satisfies: an opener forged in unframed text with
    a live id) give no span, and text only partly inside a frame (the rest of a closer's line) is outside."""
    return any(span_start < start and end <= span_end for span_start, span_end in spans)


def _section_end(lines: list[str], i: int, *, at_blank: bool, inside: list[bool]) -> int:
    """The index past a redacted section's body starting at ``i``: the first known boundary (``_ends_section``)
    or, for a section that ends at one (``_ends_at_blank``), blank line, that is not inside a complete frame
    (``_in_frame``, #1519); the end of the text otherwise. A framed line never ends a section early."""
    while i < len(lines):
        if not inside[i] and (_ends_section(lines[i]) or (at_blank and is_blank_line(lines[i]))):
            break
        i += 1
    return i


class CloudRedactionFilter:
    """Redacts sensitive sections before cloud dispatch."""

    def __init__(self, policy: RedactionPolicy) -> None:
        self._policy = policy

    @classmethod
    def from_config(
        cls,
        *,
        provider_cfg: dict[str, Any] | None = None,
        global_cfg: dict[str, Any] | None = None,
    ) -> "CloudRedactionFilter":
        provider_cfg = provider_cfg or {}
        global_cfg = global_cfg or {}
        policy_name = provider_cfg.get("redaction_policy") or global_cfg.get("policy") or "strict"
        policy = cls._policy_from_name(str(policy_name))
        return cls(policy)

    @staticmethod
    def _policy_from_name(name: str) -> RedactionPolicy:
        """Map a policy name to its behavior.

        ``standard`` currently has the same flags as ``strict`` (no
        distinct middle behavior is implemented yet) but keeps its own
        name so the cloud audit log reports the policy the operator
        actually configured instead of silently claiming ``strict``.
        Unknown names fall back to ``strict`` — over-redaction is the
        safe direction.
        """
        normalized = str(name or "").strip().lower()
        if normalized == "relaxed":
            return RedactionPolicy(name="relaxed", allow_internal=True, allow_private=True, allow_secret=False)
        if normalized == "standard":
            return RedactionPolicy(name="standard", allow_internal=True, allow_private=False, allow_secret=False)
        return RedactionPolicy(name="strict", allow_internal=True, allow_private=False, allow_secret=False)

    def redact(self, system: str, user: str) -> RedactionResult:
        categories_redacted: set[str] = set()
        categories_sent: set[str] = set()

        redacted_system, sys_redacted = self._redact_text(system)
        if sys_redacted:
            categories_redacted.update(sys_redacted)
        redacted_user, user_redacted = self._redact_sections(user)
        redacted_user, text_redacted = self._redact_text(redacted_user)
        categories_redacted.update(user_redacted)
        categories_redacted.update(text_redacted)

        # Everything not explicitly redacted is assumed sent
        categories_sent = self._infer_sent_categories(user, categories_redacted)

        return RedactionResult(
            system=redacted_system,
            user=redacted_user,
            categories_sent=sorted(categories_sent),
            categories_redacted=sorted(categories_redacted),
            policy=self._policy.name,
        )

    def _should_redact(self, sensitivity: DataSensitivity) -> bool:
        if sensitivity == DataSensitivity.PUBLIC:
            return False
        if sensitivity == DataSensitivity.INTERNAL:
            return not self._policy.allow_internal
        if sensitivity == DataSensitivity.PRIVATE:
            return not self._policy.allow_private
        if sensitivity == DataSensitivity.SECRET:
            return not self._policy.allow_secret
        return True

    def _redact_sections(self, text: str) -> tuple[str, set[str]]:
        """Replace each redacted section, from its header up to its end (``_SECTION_RULES``), with one marker.

        A section ENDS on ``\\n`` lines only: the renderer turns every other line break in a private body into
        ``\\n``, and a line that still holds one is a single line here too, never a boundary. A section OPENS on any
        reader's piece of a line (``_section_openings``), so a break can start a redaction but never cut one short.
        """
        lines, openings, inside = _section_openings(text)  # framed text is body, never a header (#1519)
        redacted_categories: set[str] = set()
        out: list[str] = []
        i = 0
        while i < len(lines):
            matched = openings[i]
            if matched is not None and self._should_redact(matched[1]):
                category = matched[0]
                redacted_categories.add(category)
                out.append(f"[REDACTED: {category}]")
                at_blank = _ends_at_blank(matched[2])
                i = _section_end(lines, i + 1, at_blank=at_blank, inside=inside)
                continue
            out.append(lines[i])
            i += 1
        return "\n".join(out), redacted_categories

    def _redact_text(self, text: str) -> tuple[str, set[str]]:
        redacted: set[str] = set()
        new_text = text
        if _SECRET_PATTERN.search(new_text):
            new_text = _SECRET_PATTERN.sub("[REDACTED_SECRET]", new_text)
            redacted.add("credentials")
        if _PATH_PATTERN.search(new_text):
            new_text = _PATH_PATTERN.sub("[REDACTED_PATH]", new_text)
            redacted.add("file_paths")
        return new_text, redacted

    def _infer_sent_categories(self, original: str, redacted: set[str]) -> set[str]:
        sent: set[str] = set()
        for matched in _section_openings(original)[1]:  # the openings the redactor reads (#1519)
            if matched is not None and matched[0] not in redacted:
                sent.add(matched[0])
        # Always include core context as sent
        sent.add("system_context")
        return sent


__all__ = [
    "CloudRedactionFilter",
    "RedactionPolicy",
    "RedactionResult",
    "DataSensitivity",
    "TOOL_SENSITIVITY",
    "MAXIM_COMMAND_SENSITIVITY",
    "has_private_section",
    "note_dispatch",
    "record_dispatches",
    "tool_result_header",
]
