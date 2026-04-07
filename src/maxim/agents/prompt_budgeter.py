"""Prompt assembly and budget management for LLM requests.

Self-contained priority-based section assembler. Zero coupling to
LLMWorker state — only needs a token counter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable

logger = logging.getLogger(__name__)


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
    ) -> None:
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
    ) -> None:
        """Add a section to the budget. Empty content is silently ignored."""
        if not content or not content.strip():
            return
        token_count = self._counter.count_tokens(content)
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
            )
        )
        self._insertion_idx += 1

    def build(self) -> tuple[str, list[str]]:
        """Assemble the prompt within budget.

        Returns:
            (prompt_text, list of dropped section names)
        """
        budget = self.prompt_budget
        included: list[PromptSection] = []
        dropped: list[str] = []
        used = 0

        # Process by priority tier
        by_tier: dict[SectionPriority, list[PromptSection]] = {}
        for s in self._sections:
            by_tier.setdefault(s.priority, []).append(s)

        for tier in sorted(by_tier.keys()):
            for section in by_tier[tier]:
                remaining = budget - used
                if section.token_count <= remaining:
                    # Fits entirely
                    included.append(section)
                    used += section.token_count
                elif (
                    section.truncatable
                    and section.truncate_fn is not None
                    and remaining >= section.min_tokens
                    and remaining > 0
                ):
                    # Truncate to fit
                    truncated = section.truncate_fn(section.content, remaining)
                    new_count = self._counter.count_tokens(truncated)
                    if new_count <= remaining and truncated.strip():
                        included.append(
                            PromptSection(
                                name=section.name,
                                content=truncated,
                                priority=section.priority,
                                token_count=new_count,
                                insertion_order=section.insertion_order,
                                truncatable=section.truncatable,
                                min_tokens=section.min_tokens,
                                truncate_fn=section.truncate_fn,
                            )
                        )
                        used += new_count
                        logger.debug(
                            "Truncated section '%s': %d→%d tokens", section.name, section.token_count, new_count
                        )
                    else:
                        dropped.append(section.name)
                else:
                    dropped.append(section.name)

        if dropped:
            logger.info("Prompt budget %d/%d — dropped sections: %s", used, budget, ", ".join(dropped))

        # Re-sort by original insertion order to preserve prompt flow
        included.sort(key=lambda s: s.insertion_order)
        prompt_text = "\n\n".join(s.content for s in included)
        return prompt_text, dropped


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
    for start in range(1, len(lines)):
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
    for start in range(1, len(entry_lines)):
        candidate = header + "\n" + "\n".join(entry_lines[start:])
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
