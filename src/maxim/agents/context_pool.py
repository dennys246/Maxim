"""Context pool for managing growing LLM context with summarization.

The context pool accumulates observations, agent states, and outcomes,
automatically summarizing when the pool exceeds a configurable size.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from maxim.agents.bus import Percept

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ContextPoolConfig:
    """Configuration for the context pool."""

    # Maximum tokens before triggering summarization
    max_tokens: int = 2000

    # How many tokens the summary should target
    summary_target_tokens: int = 500

    # Maximum number of entries before forced summarization
    max_entries: int = 50

    # How many recent entries to always keep unsummarized
    keep_recent: int = 5

    # Whether to include agent states in context
    include_agent_states: bool = True

    # Whether to include tool outcomes in context
    include_outcomes: bool = True

    # Whether to include abstraction stream
    include_abstractions: bool = True

    # Persistence path (None = no persistence)
    persistence_path: str | None = None

    @classmethod
    def from_env(cls) -> ContextPoolConfig:
        """Load configuration from environment variables."""
        return cls(
            max_tokens=int(os.getenv("MAXIM_CONTEXT_POOL_MAX_TOKENS", "2000")),
            summary_target_tokens=int(os.getenv("MAXIM_CONTEXT_POOL_SUMMARY_TOKENS", "500")),
            max_entries=int(os.getenv("MAXIM_CONTEXT_POOL_MAX_ENTRIES", "50")),
            keep_recent=int(os.getenv("MAXIM_CONTEXT_POOL_KEEP_RECENT", "5")),
            include_agent_states=os.getenv("MAXIM_CONTEXT_POOL_AGENT_STATES", "true").lower() == "true",
            include_outcomes=os.getenv("MAXIM_CONTEXT_POOL_OUTCOMES", "true").lower() == "true",
            include_abstractions=os.getenv("MAXIM_CONTEXT_POOL_ABSTRACTIONS", "true").lower() == "true",
            persistence_path=os.getenv("MAXIM_CONTEXT_POOL_PATH"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry Types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ContextEntry:
    """A single entry in the context pool."""

    timestamp: float
    entry_type: str  # "percept", "outcome", "agent_state", "abstraction", "summary"
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    # Estimated token count (rough: ~4 chars per token)
    @property
    def estimated_tokens(self) -> int:
        return max(1, len(self.content) // 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "type": self.entry_type,
            "content": self.content,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextEntry:
        return cls(
            timestamp=float(data.get("timestamp", 0)),
            entry_type=str(data.get("type", "unknown")),
            content=str(data.get("content", "")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class AgentStateEntry:
    """Snapshot of an agent's state."""

    agent_name: str
    state: str
    goal: str | None
    last_action: str | None
    success: bool | None
    timestamp: float = field(default_factory=time.time)

    def to_context_entry(self) -> ContextEntry:
        parts = [f"Agent '{self.agent_name}' in state '{self.state}'"]
        if self.goal:
            parts.append(f"Goal: {self.goal}")
        if self.last_action:
            outcome = "succeeded" if self.success else "failed" if self.success is False else "pending"
            parts.append(f"Last action: {self.last_action} ({outcome})")
        return ContextEntry(
            timestamp=self.timestamp,
            entry_type="agent_state",
            content=" | ".join(parts),
            metadata={
                "agent": self.agent_name,
                "state": self.state,
                "success": self.success,
            },
        )


@dataclass
class AbstractionEntry:
    """High-level abstraction or insight."""

    category: str  # "pattern", "insight", "summary", "novelty"
    content: str
    confidence: float = 1.0
    source: str = "system"
    timestamp: float = field(default_factory=time.time)

    def to_context_entry(self) -> ContextEntry:
        return ContextEntry(
            timestamp=self.timestamp,
            entry_type="abstraction",
            content=f"[{self.category.upper()}] {self.content}",
            metadata={
                "category": self.category,
                "confidence": self.confidence,
                "source": self.source,
            },
        )


@dataclass
class ConversationTurn:
    """A single turn in the conversation history."""

    user_input: str
    assistant_response: str
    timestamp: float = field(default_factory=time.time)
    tool_used: str | None = None  # e.g., "respond", "speak"

    def to_dict(self) -> dict[str, Any]:
        return {
            "user": self.user_input,
            "assistant": self.assistant_response,
            "timestamp": self.timestamp,
            "tool": self.tool_used,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationTurn":
        return cls(
            user_input=str(data.get("user", "")),
            assistant_response=str(data.get("assistant", "")),
            timestamp=float(data.get("timestamp", time.time())),
            tool_used=data.get("tool"),
        )

    def to_context_entry(self) -> ContextEntry:
        return ContextEntry(
            timestamp=self.timestamp,
            entry_type="conversation",
            content=f'User: "{self.user_input[:100]}" → Maxim: "{self.assistant_response[:100]}"',
            metadata={
                "user": self.user_input,
                "assistant": self.assistant_response,
                "tool": self.tool_used,
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# Context Pool
# ─────────────────────────────────────────────────────────────────────────────


class ContextPool:
    """
    Growing context pool with automatic summarization.

    The pool accumulates:
    - Percept summaries (what Maxim observed)
    - Agent states (what agents are doing)
    - Tool outcomes (what happened)
    - Abstractions (high-level insights)

    When the pool exceeds max_tokens or max_entries, older entries
    are summarized into a compact summary, keeping recent entries intact.
    """

    def __init__(
        self,
        config: ContextPoolConfig | None = None,
        summarizer: Callable[[str], str] | None = None,
    ):
        self.config = config or ContextPoolConfig()
        self._summarizer = summarizer
        self._entries: list[ContextEntry] = []
        self._summaries: list[str] = []  # Accumulated summaries
        self._conversation_history: list[ConversationTurn] = []  # Conversation turns
        self._lock = threading.RLock()  # RLock for reentrant safety
        self._total_observations = 0

        # Performance: running token total (O(1) instead of O(n) per add)
        self._total_tokens = 0

        # Performance: cached context text (invalidated on modification)
        self._cached_context_text: str | None = None

        # Load from persistence if configured
        if self.config.persistence_path:
            self._load()

    def set_summarizer(self, summarizer: Callable[[str], str]) -> None:
        """Set the LLM summarizer function."""
        self._summarizer = summarizer

    # ─────────────────────────────────────────────────────────────────────────
    # Adding Entries
    # ─────────────────────────────────────────────────────────────────────────

    def add_percept(self, percept: Percept) -> None:
        """Add a percept observation to the pool."""
        # Convert percept to compact string representation
        parts = []

        if percept.transcript_chunk:
            parts.append(f'Heard: "{percept.transcript_chunk[:100]}"')

        if percept.detections:
            objects = [d.get("label", "object") for d in percept.detections[:5]]
            parts.append(f"Saw: {', '.join(objects)}")

        if percept.cli_input:
            parts.append(f'User input: "{percept.cli_input[:100]}"')

        if percept.has_maxim_keyword:
            parts.append("(addressed directly)")

        if percept.novelty > 0.5:
            parts.append(f"[novelty: {percept.novelty:.1f}]")

        if not parts:
            return  # Nothing notable to record

        content = " | ".join(parts)
        entry = ContextEntry(
            timestamp=percept.timestamp,
            entry_type="percept",
            content=content,
            metadata={
                "source": percept.source,
                "salience": percept.salience,
                "novelty": percept.novelty,
            },
        )

        self._add_entry(entry)
        self._total_observations += 1

    def add_outcome(
        self,
        tool_name: str,
        success: bool,
        result_summary: str | None = None,
        error: str | None = None,
    ) -> None:
        """Add a tool execution outcome."""
        if not self.config.include_outcomes:
            return

        status = "succeeded" if success else "failed"
        parts = [f"Tool '{tool_name}' {status}"]

        if result_summary:
            parts.append(f": {result_summary[:100]}")
        elif error:
            parts.append(f": {error[:100]}")

        entry = ContextEntry(
            timestamp=time.time(),
            entry_type="outcome",
            content="".join(parts),
            metadata={
                "tool": tool_name,
                "success": success,
            },
        )
        self._add_entry(entry)

    def add_agent_state(self, state: AgentStateEntry) -> None:
        """Add an agent state snapshot."""
        if not self.config.include_agent_states:
            return
        self._add_entry(state.to_context_entry())

    def add_abstraction(self, abstraction: AbstractionEntry) -> None:
        """Add a high-level abstraction."""
        if not self.config.include_abstractions:
            return
        self._add_entry(abstraction.to_context_entry())

    def add_raw(self, entry_type: str, content: str, **metadata: Any) -> None:
        """Add a raw entry."""
        entry = ContextEntry(
            timestamp=time.time(),
            entry_type=entry_type,
            content=content,
            metadata=metadata,
        )
        self._add_entry(entry)

    def add_conversation_turn(
        self,
        user_input: str,
        assistant_response: str,
        tool_used: str | None = None,
    ) -> None:
        """Add a conversation turn (user input + assistant response pair)."""
        turn = ConversationTurn(
            user_input=user_input,
            assistant_response=assistant_response,
            tool_used=tool_used,
        )
        with self._lock:
            self._conversation_history.append(turn)
            # Keep only recent conversation history (last 20 turns)
            if len(self._conversation_history) > 20:
                self._conversation_history = self._conversation_history[-20:]

        # Also add as context entry for the general pool
        self._add_entry(turn.to_context_entry())

    def get_conversation_history(self, max_turns: int = 10) -> list[dict[str, Any]]:
        """Get recent conversation history as list of dicts."""
        with self._lock:
            turns = self._conversation_history[-max_turns:]
            return [turn.to_dict() for turn in turns]

    def get_conversation_text(self, max_turns: int = 10) -> str:
        """Get conversation history formatted as text for prompts."""
        with self._lock:
            turns = self._conversation_history[-max_turns:]
            if not turns:
                return ""

            lines = []
            for turn in turns:
                lines.append(f"User: {turn.user_input}")
                lines.append(f"Maxim: {turn.assistant_response}")
            return "\n".join(lines)

    def _add_entry(self, entry: ContextEntry) -> None:
        """Add entry and trigger summarization if needed."""
        with self._lock:
            self._entries.append(entry)

            # O(1) token update instead of O(n) sum
            self._total_tokens += entry.estimated_tokens

            # Invalidate cached context text
            self._cached_context_text = None

            # Check if summarization is needed (now O(1))
            if self._total_tokens > self.config.max_tokens or len(self._entries) > self.config.max_entries:
                self._summarize_old_entries()

    # ─────────────────────────────────────────────────────────────────────────
    # Summarization
    # ─────────────────────────────────────────────────────────────────────────

    def _summarize_old_entries(self) -> None:
        """Summarize older entries to reduce context size."""
        if len(self._entries) <= self.config.keep_recent:
            return

        # Split into old (to summarize) and recent (to keep)
        old_entries = self._entries[: -self.config.keep_recent]
        recent_entries = self._entries[-self.config.keep_recent :]

        # Build text to summarize
        text_to_summarize = self._entries_to_text(old_entries)

        # Get summary from LLM or use fallback
        if self._summarizer:
            try:
                summary = self._summarizer(text_to_summarize)
            except Exception as e:
                logger.warning(f"Summarization failed: {e}")
                summary = self._fallback_summarize(old_entries)
        else:
            summary = self._fallback_summarize(old_entries)

        # Store summary and replace entries
        self._summaries.append(summary)

        # Create summary entry
        summary_entry = ContextEntry(
            timestamp=time.time(),
            entry_type="summary",
            content=f"[Previous context summary] {summary}",
            metadata={"entries_summarized": len(old_entries)},
        )

        self._entries = [summary_entry] + recent_entries

        # Recalculate token total after major modification
        self._total_tokens = sum(e.estimated_tokens for e in self._entries)

        # Invalidate cached context text
        self._cached_context_text = None

        logger.debug(f"Summarized {len(old_entries)} entries into {summary_entry.estimated_tokens} tokens")

    def _fallback_summarize(self, entries: list[ContextEntry]) -> str:
        """Create a simple summary without LLM."""
        # Group by type
        by_type: dict[str, list[str]] = {}
        for entry in entries:
            by_type.setdefault(entry.entry_type, []).append(entry.content)

        parts = []
        for entry_type, contents in by_type.items():
            if len(contents) <= 2:
                parts.append(f"{entry_type}: {'; '.join(contents)}")
            else:
                parts.append(f'{entry_type}: {len(contents)} events including "{contents[-1][:50]}..."')

        return " | ".join(parts)

    def _entries_to_text(self, entries: list[ContextEntry]) -> str:
        """Convert entries to plain text for summarization."""
        lines = []
        for entry in entries:
            timestamp_str = time.strftime("%H:%M:%S", time.localtime(entry.timestamp))
            lines.append(f"[{timestamp_str}] {entry.content}")
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # Context Retrieval
    # ─────────────────────────────────────────────────────────────────────────

    def get_context_text(self, max_tokens: int | None = None) -> str:
        """Get the full context as text for LLM prompts (cached when no truncation)."""
        with self._lock:
            # Fast path: return cached text if available and no truncation needed
            if max_tokens is None and self._cached_context_text is not None:
                return self._cached_context_text

            # Build the text
            parts = []
            if self._summaries:
                parts.append("=== Historical Context ===")
                parts.extend(self._summaries[-3:])  # Keep last 3 summaries

            # Add current entries
            if self._entries:
                parts.append("=== Recent Observations ===")
                for entry in self._entries:
                    timestamp_str = time.strftime("%H:%M:%S", time.localtime(entry.timestamp))
                    parts.append(f"[{timestamp_str}] {entry.content}")

            text = "\n".join(parts)

            # Cache the untruncated version
            if max_tokens is None:
                self._cached_context_text = text
            else:
                # Truncate if needed
                max_chars = max_tokens * 4
                if len(text) > max_chars:
                    text = text[-max_chars:]

            return text

    def get_recent_entries(self, count: int = 10) -> list[ContextEntry]:
        """Get the most recent entries."""
        with self._lock:
            return self._entries[-count:]

    def get_entries_by_type(self, entry_type: str) -> list[ContextEntry]:
        """Get all entries of a specific type."""
        with self._lock:
            return [e for e in self._entries if e.entry_type == entry_type]

    @property
    def stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                "entries": len(self._entries),
                "summaries": len(self._summaries),
                "conversation_turns": len(self._conversation_history),
                "estimated_tokens": self._total_tokens,  # Use cached total (O(1))
                "total_observations": self._total_observations,
                "types": list(set(e.entry_type for e in self._entries)),
            }

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load from persistence file."""
        if not self.config.persistence_path:
            return

        try:
            if os.path.exists(self.config.persistence_path):
                with open(self.config.persistence_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                self._entries = [ContextEntry.from_dict(e) for e in data.get("entries", [])]
                self._summaries = data.get("summaries", [])
                self._conversation_history = [
                    ConversationTurn.from_dict(t) for t in data.get("conversation_history", [])
                ]
                self._total_observations = data.get("total_observations", 0)

                # Initialize token count from loaded entries
                self._total_tokens = sum(e.estimated_tokens for e in self._entries)

                logger.info(
                    f"Loaded context pool: {len(self._entries)} entries, "
                    f"{len(self._summaries)} summaries, {len(self._conversation_history)} conversation turns"
                )
        except Exception as e:
            logger.warning(f"Failed to load context pool: {e}")

    def save(self) -> None:
        """Save to persistence file."""
        if not self.config.persistence_path:
            return

        try:
            os.makedirs(os.path.dirname(self.config.persistence_path) or ".", exist_ok=True)

            with self._lock:
                data = {
                    "entries": [e.to_dict() for e in self._entries],
                    "summaries": self._summaries,
                    "conversation_history": [t.to_dict() for t in self._conversation_history],
                    "total_observations": self._total_observations,
                    "saved_at": time.time(),
                }

            temp_path = f"{self.config.persistence_path}.tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(temp_path, self.config.persistence_path)

        except Exception as e:
            logger.warning(f"Failed to save context pool: {e}")

    def clear(self) -> None:
        """Clear all entries, summaries, and conversation history."""
        with self._lock:
            self._entries.clear()
            self._summaries.clear()
            self._conversation_history.clear()
            self._total_observations = 0


# ─────────────────────────────────────────────────────────────────────────────
# Summarization Prompt Builder
# ─────────────────────────────────────────────────────────────────────────────


def build_summarization_prompt(context_text: str, target_tokens: int = 500) -> str:
    """Build a prompt for LLM summarization of context."""
    return f"""Summarize the following observations into a compact summary.
Keep the most important information: what was observed, what happened, and key outcomes.
Target length: approximately {target_tokens} tokens.

Context to summarize:
{context_text}

Summary:"""
