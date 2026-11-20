"""Prompt assembly and budget management for LLM requests.

Self-contained priority-based section assembler. Zero coupling to
LLMWorker state — only needs a token counter.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Callable, Literal

logger = logging.getLogger(__name__)

# Upper bound on the fraction of the prompt budget the byte-stable cacheable
# prefix may consume in build_segmented(). Keeps a guaranteed floor for the
# dynamic bucket's MANDATORY sections (the user request) even when the stable
# set is unusually large. Fixed (not per-turn) so it never perturbs the prefix.
_STABLE_BUDGET_FRACTION = 0.7


# Who wrote a section's text (#1042). "builder": the prompt builder's own guidance, which names tools, so it passes
# through the budgeter's ``builder_gate`` (an agent is told only about tools it has). "data": what the agent reads
# (a tool result, a memory, an input, its own roster), which is never rewritten. Required on every ``add``: a new
# section cannot skip the decision.
SectionSource = Literal["builder", "data"]


class _Unset(Enum):
    """``PromptBudgeter.add``'s ``header`` when the caller did not pass one."""

    UNSET = "unset"


_UNSET = _Unset.UNSET


class SectionPriority(IntEnum):
    """Priority tiers for prompt sections. Lower = higher priority."""

    MANDATORY = 0  # Never dropped (instructions, user request)
    CRITICAL = 1  # Dropped only under extreme pressure (identity, tools, planning)
    IMPORTANT = 2  # Truncatable first, then dropped (conversation, context pool)
    NICE_TO_HAVE = 3  # Dropped first (foundational, mode context, speech)


@dataclass
class PromptSection:
    """A single section of the assembled prompt."""

    name: str
    content: str
    priority: SectionPriority
    token_count: int
    insertion_order: int
    truncatable: bool = False
    min_tokens: int = 0
    truncate_fn: Callable[[str, int], str] | None = None
    # When True, this section is part of the byte-stable cacheable prefix
    # (prompt_caching_for_cloud_backends.md). Stable sections are assembled
    # independently of dynamic content so the prefix is byte-identical across
    # turns within a session/phase. Default False (dynamic).
    cacheable: bool = False
    # When True (requires ``cacheable``), the section changes far more often
    # than the rest of the stable prefix — the scene roster's tool manifest
    # changes at every encounter/act boundary, while guidance and the
    # foundational text change at most with mode/autonomy. ``build_segmented``
    # emits every phase-scoped section AFTER every session-stable one, so a
    # roster change invalidates only the tail of the cached prefix (P21).
    phase_scoped: bool = False
    # A section rendered by identity (``PromptBudgeter.add(header=...)``): the header line is kept apart from
    # ``content`` (the body), so truncation and frame cleanup act on the body only and the header is always emitted.
    header: str | None = None

    @property
    def rendered(self) -> str:
        """The section as it appears in the prompt: the header line, if any, then the body."""
        return self.content if self.header is None else f"{self.header}\n{self.content}"


class PromptBudgeter:
    """Assembles prompt sections within a token budget.

    Processes sections by priority tier (MANDATORY first, NICE_TO_HAVE last).
    Truncatable sections are shortened before being dropped entirely.
    Final output preserves the original insertion order of included sections.
    """

    def __init__(
        self,
        total_budget: int,
        response_reserve: int,
        token_counter: Any,
        template_overhead: int = 100,
        *,
        builder_gate: Callable[[str], str] | None,
    ) -> None:
        """``builder_gate`` (required, #1042) rewrites every ``source="builder"`` section: the prompt builder passes
        its roster gate. ``None`` must be said explicitly (a budgeter with no roster to gate on), so a new production
        budgeter cannot skip the gate by omission."""
        self._builder_gate = builder_gate
        self._total_budget = total_budget
        self._response_reserve = response_reserve
        self._template_overhead = template_overhead
        self._counter = token_counter
        self._sections: list[PromptSection] = []
        self._insertion_idx = 0

    @property
    def prompt_budget(self) -> int:
        """Tokens available for the actual prompt content."""
        return max(0, self._total_budget - self._response_reserve - self._template_overhead)

    def add(
        self,
        name: str,
        content: str,
        priority: SectionPriority,
        truncatable: bool = False,
        min_tokens: int = 0,
        truncate_fn: Callable[[str, int], str] | None = None,
        cacheable: bool = False,
        phase_scoped: bool = False,
        *,
        source: SectionSource,
        frame_wording: str | None = None,
        header: str | None | Literal[_Unset.UNSET] = _UNSET,
    ) -> None:
        """Add a section to the budget. Empty content is silently ignored.

        ``source`` (required, #1042) says who wrote the text: ``"builder"`` sections pass through the
        ``builder_gate``; ``"data"`` sections are kept as written.

        ``phase_scoped=True`` is only meaningful on a cacheable section (it
        places the section at the END of the stable prefix); asking for it
        on a dynamic section is a caller bug, raised loudly rather than
        silently ignored.

        ``frame_wording`` names what this section's frames hold (#1481); None is the tool-output default.

        ``header`` is required with ``source="data"`` (``header=None`` says the section has none), so a new data
        section cannot skip the decision. A ``=== X ===`` line renders the section by identity: ``content`` is the
        body, and every rewrite (frame cleanup, truncation) acts on the body, which is neutralized
        (``content_safety.neutralize_section_body``) so it holds neither a section boundary nor a blank line. The
        header line is therefore always the section's first line. Every private section (the ones
        ``utils/cloud_redaction.py::_SECTION_RULES`` redact) is added this way. Every data section also loses any
        ``\x1e`` (the prompt segment delimiter, ``prompt_builder.PROMPT_SEGMENT_DELIMITER``) to a line break.
        """
        if phase_scoped and not cacheable:
            raise ValueError(f"prompt section {name!r}: phase_scoped=True requires cacheable=True")
        if source not in ("builder", "data"):
            raise ValueError(f"prompt section {name!r}: source must be 'builder' or 'data', not {source!r}")
        if header is _UNSET:
            if source == "data":
                raise TypeError(f"prompt section {name!r}: a source='data' section passes header= (None for none)")
            header = None
        if header is not None:
            from maxim.utils.content_safety import check_section_header, neutralize_section_body

            if source != "data":
                raise ValueError(f"prompt section {name!r}: a header-separate section is source='data'")
            check_section_header(header)
        if source == "data" and content:
            content = content.replace("\x1e", "\n")  # never the segment delimiter, cacheable or not
        if source == "builder" and self._builder_gate is not None and content:
            content = self._builder_gate(content)
        if content and "TOOL_OUTPUT" in content:
            # Recalled and tool text arrives framed, sometimes cut on the way (Session R, rule 2): a section never
            # carries a partial frame, and one holding a frame states the frame rule first, budgeted with it.
            from maxim.utils.content_safety import framed_section

            content = (
                framed_section(content) if frame_wording is None else framed_section(content, wording=frame_wording)
            )
        if not content or not content.strip():
            return
        if header is not None:
            content = neutralize_section_body(content)
        token_count = self._counter.count_tokens(content if header is None else f"{header}\n{content}")
        self._sections.append(
            PromptSection(
                name=name,
                content=content,
                priority=priority,
                token_count=token_count,
                insertion_order=self._insertion_idx,
                truncatable=truncatable,
                min_tokens=min_tokens,
                truncate_fn=truncate_fn,
                cacheable=cacheable,
                phase_scoped=phase_scoped,
                header=header,
            )
        )
        self._insertion_idx += 1

    def _fit(self, sections: list[PromptSection], budget: int) -> tuple[list[PromptSection], list[str], int]:
        """Select/truncate ``sections`` to fit within ``budget``.

        Priority-tier order (MANDATORY first); truncatable sections shorten
        before being dropped. Returns (included, dropped_names, tokens_used).
        Pure with respect to ``self`` (only reads the token counter).
        """
        included: list[PromptSection] = []
        dropped: list[str] = []
        used = 0

        by_tier: dict[SectionPriority, list[PromptSection]] = {}
        for s in sections:
            by_tier.setdefault(s.priority, []).append(s)

        for tier in sorted(by_tier.keys()):
            for section in by_tier[tier]:
                remaining = budget - used
                if section.token_count <= remaining:
                    included.append(section)
                    used += section.token_count
                elif (
                    section.truncatable
                    and section.truncate_fn is not None
                    and remaining >= section.min_tokens
                    and remaining > 0
                ):
                    truncated = self._truncated(section, remaining)
                    if truncated.token_count <= remaining and truncated.content.strip():
                        included.append(truncated)
                        used += truncated.token_count
                        logger.debug(
                            "Truncated section '%s': %d→%d tokens",
                            section.name,
                            section.token_count,
                            truncated.token_count,
                        )
                    else:
                        dropped.append(section.name)
                else:
                    dropped.append(section.name)

        return included, dropped, used

    def _truncated(self, section: PromptSection, remaining: int) -> PromptSection:
        """``section`` with its body cut to fit ``remaining`` tokens, header included. A truncator may cut a frame,
        front or back (Session R, rule 2); a header-separate body stays neutralized whatever the truncator returns,
        and its header is never offered to the truncator, so it cannot be cut."""
        assert section.truncate_fn is not None
        header_tokens = 0 if section.header is None else self._counter.count_tokens(section.header + "\n")
        body = section.truncate_fn(section.content, max(0, remaining - header_tokens))
        if "TOOL_OUTPUT" in body:
            from maxim.utils.content_safety import drop_unbalanced_frames

            body = drop_unbalanced_frames(body)[0]
        if section.header is not None:
            from maxim.utils.content_safety import neutralize_section_body

            body = neutralize_section_body(body)
        cut = dataclasses.replace(section, content=body)
        return dataclasses.replace(cut, token_count=self._counter.count_tokens(cut.rendered))

    @staticmethod
    def _log_included(included: list[PromptSection]) -> None:
        """DEBUG observability: one event enumerating the sections that made
        it into the prompt, with per-section char sizes.

        Exists for the Exp 44 mechanism checks
        (docs/plans/deferred/acting_coach_body_state_ablation.md): with MAXIM_LOG_FILE
        set (root logger at DEBUG), the JSONL carries one ``prompt_sections``
        line per prompt build, so arm verification (body_state / acting_coach
        present, sizes changing across turns, no unintended section diffs
        across arms) needs no prompt-text dump. Name:size pairs only — no
        content — so the event is safe at DEBUG in every mode.
        """
        logger.debug(
            "prompt_sections %s",
            ",".join(
                f"{s.name}:{len(s.rendered)}"
                for s in sorted(included, key=lambda x: (not x.cacheable, x.phase_scoped, x.insertion_order))
            ),
        )

    @staticmethod
    def _emit(included: list[PromptSection]) -> str:
        """Join included sections in original insertion order."""
        ordered = sorted(included, key=lambda s: s.insertion_order)
        return "\n\n".join(s.rendered for s in ordered)

    @staticmethod
    def _emit_stable(included: list[PromptSection]) -> str:
        """Join the stable segment: session-stable sections first (insertion
        order), then phase-scoped ones (insertion order).

        A prefix cache is invalidated from the first differing byte onward,
        so the content that changes most often must sit LAST. Insertion
        order alone put the scene-roster tool manifest (changes at every
        encounter, and whenever a Wire 1/Wire 3 phrase is spliced into a
        description) ahead of ~2.5k tokens of guidance that changes far less
        often (identity still carries the processing state and autonomy
        level), which made every encounter boundary a full-prefix miss
        (Phase 0 of the sandbox plan measured 559 of 3,450 cacheable tokens
        surviving). Ordering does not make the tail cacheable — it keeps the
        head cached when the tail changes.
        """
        ordered = sorted(included, key=lambda s: (s.phase_scoped, s.insertion_order))
        return "\n\n".join(s.rendered for s in ordered)

    def build(self) -> tuple[str, list[str]]:
        """Assemble the prompt within budget.

        Returns:
            (prompt_text, list of dropped section names)
        """
        budget = self.prompt_budget
        included, dropped, used = self._fit(self._sections, budget)
        if dropped:
            logger.info("Prompt budget %d/%d — dropped sections: %s", used, budget, ", ".join(dropped))
        self._log_included(included)
        return self._emit(included), dropped

    def build_segmented(self) -> tuple[str, str, list[str]]:
        """Assemble a byte-stable cacheable prefix + a dynamic remainder.

        Splits sections by the ``cacheable`` flag and budgets the stable
        sections FIRST, with a budget that does not depend on per-turn dynamic
        content. This guarantees the stable text is byte-identical across turns
        within a session/phase — the prerequisite for prompt caching
        (docs/plans/archive/prompt_caching_for_cloud_backends.md).

        The stable bucket is capped at ``_STABLE_BUDGET_FRACTION`` of the prompt
        budget so a pathologically large stable set (e.g. a huge singularity
        tool manifest) can never starve the dynamic MANDATORY sections (the
        user request). The cap is a fixed fraction — independent of per-turn
        state — so it does not itself perturb the stable text.

        Within the stable text, session-stable sections come first and
        ``phase_scoped`` ones last (see ``_emit_stable``), so a phase
        boundary invalidates only the tail of the cached prefix.

        Returns:
            (stable_text, dynamic_text, dropped_section_names)
        """
        budget = self.prompt_budget
        stable_sections = [s for s in self._sections if s.cacheable]
        dynamic_sections = [s for s in self._sections if not s.cacheable]

        stable_cap = int(budget * _STABLE_BUDGET_FRACTION)
        stable_included, stable_dropped, stable_used = self._fit(stable_sections, stable_cap)

        dynamic_budget = max(0, budget - stable_used)
        dyn_included, dyn_dropped, dyn_used = self._fit(dynamic_sections, dynamic_budget)

        dropped = stable_dropped + dyn_dropped
        if dropped:
            logger.info(
                "Prompt budget segmented %d(stable)+%d(dynamic)/%d — dropped: %s",
                stable_used,
                dyn_used,
                budget,
                ", ".join(dropped),
            )
        self._log_included(stable_included + dyn_included)
        return self._emit_stable(stable_included), self._emit(dyn_included), dropped


# ── Truncation Helpers ──────────────────────────────────────────────────────


def _compact_conversation(content: str, max_turns: int) -> str:
    """Pre-pass: hard-cap conversation history to last N turns, always keeping the first turn."""
    lines = content.split("\n")
    turn_starts = []
    for i, line in enumerate(lines):
        if line.startswith("User:") or line.startswith("Maxim:"):
            turn_starts.append(i)

    if len(turn_starts) <= max_turns:
        return content

    # Always keep first turn + last (max_turns - 1) turns
    keep_starts = [turn_starts[0]]  # First turn always pinned
    remaining = max_turns - 1
    if remaining > 0:
        keep_starts.extend(turn_starts[-remaining:])

    # Build compacted content
    keep_set = set(keep_starts)
    result_lines = []
    current_kept = False
    for i, line in enumerate(lines):
        if i in keep_set:
            current_kept = True
        elif line.startswith("User:") or line.startswith("Maxim:"):
            current_kept = i in keep_set

        if current_kept:
            result_lines.append(line)

    return "\n".join(result_lines)


def _truncate_conversation(content: str, max_tokens: int, counter: Any) -> str:
    """Drop oldest User+Maxim turn pairs from the front."""
    lines = content.split("\n")
    # Find turn boundaries (lines starting with "User:" or "Maxim:")
    turn_starts: list[int] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("User:") or stripped.startswith("Maxim:"):
            turn_starts.append(i)

    if len(turn_starts) < 2:
        return content

    # Try removing from the front, one turn boundary at a time
    for cut_idx in range(1, len(turn_starts)):
        candidate = "\n".join(lines[turn_starts[cut_idx] :])
        if counter.count_tokens(candidate) <= max_tokens:
            return candidate
    # Last resort: return just the last turn
    return "\n".join(lines[turn_starts[-1] :])


def _truncate_context_pool(content: str, max_tokens: int, counter: Any) -> str:
    """Drop oldest lines from the front of the context pool."""
    lines = content.split("\n")
    for start in range(len(lines)):  # the body only: its header is the budgeter's, never offered here
        candidate = "\n".join(lines[start:])
        if counter.count_tokens(candidate) <= max_tokens:
            return candidate
    return lines[-1] if lines else ""


def _truncate_tool_guidance(content: str, max_tokens: int, counter: Any) -> str:
    """Remove Example lines, then indented detail lines."""
    lines = content.split("\n")
    # First pass: remove lines containing "Example" or "e.g."
    filtered = [line for line in lines if "Example" not in line and "e.g." not in line]
    candidate = "\n".join(filtered)
    if counter.count_tokens(candidate) <= max_tokens:
        return candidate
    # Second pass: remove indented detail lines (4+ spaces or tab)
    filtered = [line for line in filtered if not line.startswith("    ") and not line.startswith("\t")]
    return "\n".join(filtered)


def _truncate_reasoning_carryover(content: str, max_tokens: int, counter: Any) -> str:
    """Drop oldest entries from the reasoning carryover."""
    lines = content.split("\n")
    # Keep header line, truncate entry lines from front
    header_lines: list[str] = []
    entry_lines: list[str] = []
    for line in lines:
        if line.startswith("- "):
            entry_lines.append(line)
        else:
            header_lines.append(line)

    header = "\n".join(header_lines)
    lead = header + "\n" if header_lines else ""  # a header-separate body has no header line to keep
    for start in range(1, len(entry_lines)):
        candidate = lead + "\n".join(entry_lines[start:])
        if counter.count_tokens(candidate) <= max_tokens:
            return candidate
    return header


def _truncate_manifest(content: str, max_tokens: int, counter: Any) -> str:
    """Truncate a manifest section by removing file entries from the end."""
    lines = content.split("\n")
    # Separate header/rule lines from indented file entries
    header_lines: list[str] = []
    entry_lines: list[str] = []
    for line in lines:
        if (
            line.startswith("  ")
            and not line.startswith("  ...")
            and not line.startswith("  1.")
            and not line.startswith("  2.")
            and not line.startswith("  3.")
        ):
            entry_lines.append(line)
        else:
            header_lines.append(line)

    # Remove entries from the end until it fits
    while entry_lines and counter.count_tokens("\n".join(header_lines + entry_lines)) > max_tokens:
        entry_lines.pop()

    if entry_lines:
        # Reconstruct: headers before entries, then remaining headers after
        return "\n".join(header_lines[:2] + entry_lines + header_lines[2:])
    return "\n".join(header_lines)
