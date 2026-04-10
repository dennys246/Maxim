# tools/learned_index.py
"""Keyword-weighted hashtable for tool relevance scoring.

Auto-extracts keywords from tool metadata at registration time.
Learns from tool execution outcomes to refine weights and discover
new keyword associations via Rescorla-Wagner inspired updates.

Thread-safe: all mutations to _index and _tool_keywords are
protected by a lock.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolKeywordEntry:
    """A keyword associated with a tool, with learned weight."""

    word: str
    weight: float = 0.5  # 0.0 = never relevant, 1.0 = always relevant
    source: str = "auto"  # "auto" (extracted), "manual" (declared), or "learned" (discovered)
    observations: int = 0  # Times this word co-occurred with tool execution


STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "and",
        "but",
        "or",
        "nor",
        "not",
        "no",
        "so",
        "if",
        "than",
        "that",
        "this",
        "it",
        "its",
        "use",
        "used",
        "using",
    }
)

_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    """Tokenize text for index lookup — lowercase, split, filter stopwords."""
    return {tok for tok in _SPLIT_RE.split(text.lower()) if len(tok) > 2 and tok not in STOPWORDS}


class LearnedToolIndex:
    """Keyword-weighted hashtable for tool relevance scoring.

    Auto-extracts keywords from tool metadata at registration time.
    Learns from tool execution outcomes to refine weights and discover
    new keyword associations.

    Usage::

        index = LearnedToolIndex()
        for tool in registry.all_tools():
            index.register_tool(tool)
        index.load(str(user_memory() / "tool_index.json"))  # from maxim.utils.paths

        # At prompt build time:
        relevant, background = index.get_relevant_tools("pick up the red cup")

        # After tool execution:
        index.record_outcome("pick up the red cup", "grab", success=True)
    """

    LEARNING_RATE = 0.1
    DECAY_RATE = 0.2
    RELEVANCE_THRESHOLD = 0.3
    # Below this registry size, the filter is a token-savings illusion: prompt
    # overhead from the section split costs more than the trimmed descriptions
    # would. At or below SMALL_REGISTRY_THRESHOLD, get_relevant_tools returns
    # every tool as "relevant" (background empty), giving the model the full
    # manifest in one section. Tunable; current registries land around 27-50
    # tools depending on enabled subsystems.
    SMALL_REGISTRY_THRESHOLD = 60
    # Floor for the relevant set on cold-start scoring above the small-registry
    # threshold. Was 3 (severe — produced 3 dict-order tools and caused
    # systematic tool hallucination); raised to a value that gives the model
    # enough surface area to find a real match for any reasonable query.
    MIN_TOOLS = 15
    NEW_KEYWORD_INITIAL_WEIGHT = 0.2
    NEW_KEYWORD_MAX_PER_OUTCOME = 2

    def __init__(self) -> None:
        self._tool_keywords: dict[str, dict[str, ToolKeywordEntry]] = {}
        self._index: dict[str, list[tuple[str, ToolKeywordEntry]]] = {}
        self._lock = threading.Lock()

    # ── Registration ──────────────────────────────────────────

    def register_tool(self, tool: Any) -> None:
        """Auto-extract keywords from tool metadata and register.

        Safe to call multiple times — skips if already registered
        (preserves learned weights).
        """
        name = tool.name
        with self._lock:
            if name in self._tool_keywords:
                return
            keywords = self._extract_keywords(tool)
            self._tool_keywords[name] = {}
            for word in keywords:
                entry = ToolKeywordEntry(word=word, weight=0.5, source="auto")
                self._tool_keywords[name][word] = entry
                self._index.setdefault(word, []).append((name, entry))

    def register_manual_keywords(self, tool_name: str, keywords: set[str]) -> None:
        """Add manually declared keywords (higher initial weight)."""
        with self._lock:
            for word in keywords:
                word = word.lower()
                if tool_name in self._tool_keywords and word in self._tool_keywords[tool_name]:
                    continue
                entry = ToolKeywordEntry(word=word, weight=0.7, source="manual")
                self._tool_keywords.setdefault(tool_name, {})[word] = entry
                self._index.setdefault(word, []).append((tool_name, entry))

    # ── Scoring ───────────────────────────────────────────────

    def score_tools(self, goal_text: str) -> dict[str, float]:
        """Score all tools against a goal string.

        Returns ``{tool_name: normalized_relevance_score}`` sorted descending.
        Scores use average weight of matched keywords with a match-count boost,
        preventing verbose tool descriptions from dominating.
        """
        tokens = _tokenize(goal_text)
        tool_matches: dict[str, list[float]] = {}
        with self._lock:
            for token in tokens:
                for tool_name, entry in self._index.get(token, []):
                    tool_matches.setdefault(tool_name, []).append(entry.weight)

        scores: dict[str, float] = {}
        for tool_name, weights in tool_matches.items():
            avg = sum(weights) / len(weights)
            boost = 1.0 + 0.2 * min(len(weights) - 1, 3)
            scores[tool_name] = avg * boost
        return dict(sorted(scores.items(), key=lambda x: -x[1]))

    def get_relevant_tools(self, goal_text: str) -> tuple[list[str], list[str]]:
        """Partition tools into relevant (full schema) and background (name only).

        For small registries (<= SMALL_REGISTRY_THRESHOLD) returns every tool
        as relevant — the filter overhead exceeds its savings at that scale,
        and trimming risks hiding the right tool from a cold-signal query.

        For larger registries: returns the highest-scoring matches (above
        RELEVANCE_THRESHOLD) plus a MIN_TOOLS floor. The floor padding draws
        from the highest-scoring sub-threshold tools first, falling back to
        registration order only when scoring produced no signal at all.
        """
        with self._lock:
            all_tools = list(self._tool_keywords.keys())

        if len(all_tools) <= self.SMALL_REGISTRY_THRESHOLD:
            return list(all_tools), []

        scores = self.score_tools(goal_text)
        sorted_tools = sorted(scores.items(), key=lambda x: -x[1])

        relevant: list[str] = []
        for name, score in sorted_tools:
            if score >= self.RELEVANCE_THRESHOLD:
                relevant.append(name)
            else:
                break

        # Floor: ensure the relevant set has at least MIN_TOOLS so the model
        # always sees enough surface area. Prefer the next-highest-scoring
        # tools (still informative ranking) over arbitrary dict-order picks.
        if len(relevant) < self.MIN_TOOLS:
            relevant_set = set(relevant)
            for name, _score in sorted_tools:
                if name in relevant_set:
                    continue
                relevant.append(name)
                relevant_set.add(name)
                if len(relevant) >= self.MIN_TOOLS:
                    break

        # If scoring produced no candidates at all (cold registry, no learned
        # signal yet, no keyword matches), fall back to registration order
        # padding rather than returning fewer than MIN_TOOLS.
        if len(relevant) < self.MIN_TOOLS:
            relevant_set = set(relevant)
            for name in all_tools:
                if name in relevant_set:
                    continue
                relevant.append(name)
                relevant_set.add(name)
                if len(relevant) >= self.MIN_TOOLS:
                    break

        relevant_set = set(relevant)
        background = [n for n in all_tools if n not in relevant_set]
        return relevant, background

    # ── Learning ──────────────────────────────────────────────

    def record_outcome(
        self,
        goal_text: str,
        tool_name: str,
        success: bool,
    ) -> None:
        """Update keyword weights based on tool execution outcome.

        On success: strengthen existing keywords AND discover new ones
        from goal tokens not yet in the tool's keyword set.

        On failure: count the observation but DON'T weaken the weight.
        Tool failure means the execution failed, not that the keyword
        association is wrong.
        """
        tokens = _tokenize(goal_text)
        with self._lock:
            tool_kw = self._tool_keywords.get(tool_name)
            if not tool_kw:
                return

            matched = tokens & set(tool_kw.keys())
            for word in matched:
                entry = tool_kw[word]
                entry.observations += 1
                if success:
                    entry.weight += self.LEARNING_RATE * (1.0 - entry.weight)
                # Failure: observation counted, no weight change
                entry.weight = max(0.01, min(1.0, entry.weight))

            # Discover new keywords from successful executions
            if success:
                new_tokens = tokens - set(tool_kw.keys())
                created = 0
                for word in new_tokens:
                    if created >= self.NEW_KEYWORD_MAX_PER_OUTCOME:
                        break
                    if len(word) <= 2 or word in STOPWORDS:
                        continue
                    entry = ToolKeywordEntry(
                        word=word,
                        weight=self.NEW_KEYWORD_INITIAL_WEIGHT,
                        source="learned",
                        observations=1,
                    )
                    tool_kw[word] = entry
                    self._index.setdefault(word, []).append((tool_name, entry))
                    created += 1

    def record_surfaced_but_unused(
        self,
        goal_text: str,
        surfaced_tools: list[str],
        used_tool: str,
    ) -> None:
        """Decay keywords for tools surfaced in the prompt but not chosen.

        This is the primary negative signal: the LLM saw the tool schema
        and decided not to use it.
        """
        tokens = _tokenize(goal_text)
        with self._lock:
            for tool_name in surfaced_tools:
                if tool_name == used_tool:
                    continue
                tool_kw = self._tool_keywords.get(tool_name)
                if not tool_kw:
                    continue
                matched = tokens & set(tool_kw.keys())
                for word in matched:
                    entry = tool_kw[word]
                    entry.weight -= self.LEARNING_RATE * self.DECAY_RATE * entry.weight
                    entry.weight = max(0.01, min(1.0, entry.weight))

    # ── Keyword extraction ────────────────────────────────────

    def _extract_keywords(self, tool: Any) -> set[str]:
        """Auto-extract keywords from tool name, description, and params."""
        parts: list[str] = []
        if hasattr(tool, "name") and tool.name:
            parts.append(tool.name)
            # Split compound names: "edit_file" → "edit", "file"
            parts.extend(tool.name.replace("_", " ").replace("-", " ").split())
        if hasattr(tool, "description") and tool.description:
            parts.append(tool.description)
        if hasattr(tool, "input_schema") and isinstance(tool.input_schema, dict):
            parts.extend(tool.input_schema.keys())
        # Include MANUAL_KEYWORDS if declared on the tool class
        manual = getattr(tool, "MANUAL_KEYWORDS", None)
        if isinstance(manual, (set, frozenset, list, tuple)):
            parts.extend(manual)
        return _tokenize(" ".join(parts))

    # ── Persistence ───────────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist learned weights to JSON (atomic write)."""
        with self._lock:
            data = {
                tool_name: {
                    word: {
                        "weight": round(entry.weight, 4),
                        "source": entry.source,
                        "observations": entry.observations,
                    }
                    for word, entry in keywords.items()
                }
                for tool_name, keywords in self._tool_keywords.items()
            }
        try:
            from maxim.utils.atomic_io import atomic_write_json

            atomic_write_json(path, data)
        except Exception as e:
            logger.debug("Failed to save tool index: %s", e)

    def load(self, path: str) -> None:
        """Load learned weights from JSON.

        Only updates weights for tools that are currently registered.
        Restores learned keywords from previous sessions.
        """
        try:
            with open(path) as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return

        with self._lock:
            for tool_name, keywords in data.items():
                if tool_name not in self._tool_keywords:
                    continue
                for word, kw_data in keywords.items():
                    if word in self._tool_keywords[tool_name]:
                        entry = self._tool_keywords[tool_name][word]
                        entry.weight = kw_data.get("weight", entry.weight)
                        entry.observations = kw_data.get("observations", 0)
                        entry.source = kw_data.get("source", entry.source)
                    elif kw_data.get("source") == "learned":
                        entry = ToolKeywordEntry(
                            word=word,
                            weight=kw_data.get("weight", self.NEW_KEYWORD_INITIAL_WEIGHT),
                            source="learned",
                            observations=kw_data.get("observations", 0),
                        )
                        self._tool_keywords[tool_name][word] = entry
                        self._index.setdefault(word, []).append((tool_name, entry))

    # ── Stats ─────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return index statistics."""
        with self._lock:
            total_keywords = sum(len(kw) for kw in self._tool_keywords.values())
            learned = sum(1 for kw in self._tool_keywords.values() for e in kw.values() if e.source == "learned")
            return {
                "tools_registered": len(self._tool_keywords),
                "total_keywords": total_keywords,
                "learned_keywords": learned,
                "index_entries": len(self._index),
            }


__all__ = ["LearnedToolIndex", "ToolKeywordEntry"]
