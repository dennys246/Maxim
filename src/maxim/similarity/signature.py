"""Situation signature for multi-modal similarity.

Captures multiple dimensions of a situation for efficient matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from maxim.utils.seeding import stable_hash_32


@dataclass(frozen=True, slots=True)
class SituationSignature:
    """Multi-dimensional fingerprint of a situation.

    Captures multiple aspects of a situation in a hashable form
    that enables efficient similarity computation.

    Dimensions:
    - semantic_hash: LSH hash of text embedding (goal, reasoning)
    - structural_hash: Hash of action type, tool, outcome type
    - temporal_hash: SCN bin signature (hour, day, week, month)
    - context_hash: Hash of key context features (mode, objects, etc.)

    KNOWN LIMITATION (Phase 2):
    The semantic_hash dimension requires Phase 4's semantic embedding model.
    Until Phase 4 is implemented, all memories have identical semantic hashes,
    meaning similarity queries rely only on structural, temporal, and context.

    Attributes:
        semantic_hash: LSH buckets from text embedding
        structural_hash: Hash of action/tool/outcome
        temporal_hash: (hour_bin, day_bin, week_bin, month_bin)
        context_hash: Hash of context features
        tool_name: Tool used (for index)
        outcome_type: Outcome type (for index)
        mode: Active mode (for index)
        goal_keywords: Keywords from goal (for similarity)
    """

    semantic_hash: tuple[int, ...]
    structural_hash: int
    temporal_hash: tuple[int, int, int, int]
    context_hash: int

    # Optional: raw feature values for weighted similarity
    tool_name: str = ""
    outcome_type: str = ""
    mode: str = ""
    goal_keywords: tuple[str, ...] = ()

    # Phase 2 warning tracking
    _semantic_fallback_warned: bool = False

    @classmethod
    def from_memory(
        cls,
        memory: Any,
        semantic_hasher: Any = None,
    ) -> SituationSignature:
        """Create signature from an EpisodicMemory.

        Args:
            memory: The memory to create a signature for.
            semantic_hasher: Optional semantic LSH. If None, semantic dimension
                           is set to null hash and similarity relies on other dimensions.

        Returns:
            SituationSignature for the memory.
        """
        import logging

        logger = logging.getLogger(__name__)

        # Extract text for semantic hashing
        text_parts = []
        if hasattr(memory, "decision"):
            if hasattr(memory.decision, "intent"):
                intent = memory.decision.intent
                if isinstance(intent, dict):
                    text_parts.append(str(intent.get("goal", "")))
                else:
                    text_parts.append(str(intent))
            if hasattr(memory.decision, "reasoning"):
                text_parts.append(str(memory.decision.reasoning or ""))
        semantic_text = " ".join(text_parts)

        # Compute semantic hash via LSH
        if semantic_hasher and semantic_text:
            semantic_hash = semantic_hasher.hash(semantic_text)
        else:
            # Phase 2 fallback: null semantic hash
            semantic_hash = (0,) * 8

            if not cls._semantic_fallback_warned:
                cls._semantic_fallback_warned = True
                logger.info(
                    "EC: semantic_hasher not provided, using null semantic hash. "
                    "Semantic similarity disabled until Phase 4."
                )

        # Compute structural hash
        tool_name = ""
        outcome_type = ""
        if hasattr(memory, "action"):
            tool_name = getattr(memory.action, "tool_name", "") or ""
        if hasattr(memory, "outcome"):
            outcome = memory.outcome
            if hasattr(outcome, "success"):
                outcome_type = "success" if outcome.success else "failure"

        # stable_hash_32, NOT builtin hash(): these ints are persisted via
        # EC.save()/load() and compared for exact equality in
        # _structural_match/_context_match — a randomized hash can never
        # match across a process boundary (--resume-sim, cross-session).
        structural_str = f"{tool_name}:{outcome_type}"
        structural_hash = stable_hash_32(structural_str)

        # Compute temporal hash from timestamp
        temporal_hash = (0, 0, 0, 0)
        if hasattr(memory, "timestamp"):
            from datetime import datetime

            ts = memory.timestamp
            dt = datetime.fromtimestamp(ts)
            hour_bin = dt.hour
            day_bin = dt.weekday()
            week_bin = (dt.day - 1) // 7
            month_bin = dt.month - 1
            temporal_hash = (hour_bin, day_bin, week_bin, month_bin)

        # Compute context hash
        mode = ""
        context_features = []
        if hasattr(memory, "context"):
            ctx = memory.context
            mode = getattr(ctx, "active_mode", "") or ""
        if hasattr(memory, "perception"):
            perc = memory.perception
            if hasattr(perc, "detected_objects"):
                context_features.extend(perc.detected_objects or [])
            if hasattr(perc, "detected_people"):
                context_features.extend(perc.detected_people or [])

        context_str = f"{mode}:{':'.join(sorted(context_features))}"
        context_hash = stable_hash_32(context_str)

        # Extract goal keywords
        goal_keywords = tuple(word.lower() for word in semantic_text.split()[:10] if len(word) > 3)

        return cls(
            semantic_hash=semantic_hash,
            structural_hash=structural_hash,
            temporal_hash=temporal_hash,
            context_hash=context_hash,
            tool_name=tool_name,
            outcome_type=outcome_type,
            mode=mode,
            goal_keywords=goal_keywords,
        )

    def _hamming_distance(self, other: SituationSignature) -> int:
        """Compute Hamming distance between semantic hashes."""
        if len(self.semantic_hash) != len(other.semantic_hash):
            return len(self.semantic_hash)  # Max distance
        return sum(a != b for a, b in zip(self.semantic_hash, other.semantic_hash))

    def _structural_match(self, other: SituationSignature) -> bool:
        """Check if structural hashes match exactly."""
        return self.structural_hash == other.structural_hash

    def temporal_distance(self, other: SituationSignature) -> float:
        """Compute temporal distance (0.0 = same time, 1.0 = max different)."""
        # Weighted by time scale importance
        weights = (0.4, 0.3, 0.2, 0.1)  # hour > day > week > month
        max_diffs = (24, 7, 4, 12)

        total_dist = 0.0
        for (a, b), max_diff, weight in zip(zip(self.temporal_hash, other.temporal_hash), max_diffs, weights):
            # Circular distance for cyclic time
            diff = min(abs(a - b), max_diff - abs(a - b))
            total_dist += weight * (diff / max_diff)

        return total_dist

    def _context_match(self, other: SituationSignature) -> bool:
        """Check if context hashes match exactly."""
        return self.context_hash == other.context_hash

    def keyword_overlap(self, other: SituationSignature) -> float:
        """Compute Jaccard similarity of goal keywords."""
        if not self.goal_keywords or not other.goal_keywords:
            return 0.0
        intersection = set(self.goal_keywords) & set(other.goal_keywords)
        union = set(self.goal_keywords) | set(other.goal_keywords)
        return len(intersection) / len(union) if union else 0.0

    def similarity(self, other: SituationSignature) -> float:
        """Compute weighted composite similarity (0.0-1.0)."""
        # Semantic similarity (from LSH hamming)
        if self.semantic_hash and other.semantic_hash:
            hamming = self._hamming_distance(other)
            semantic_sim = 1.0 - (hamming / max(1, len(self.semantic_hash)))
        else:
            semantic_sim = 0.5  # Neutral when no semantic hash

        # Structural similarity (exact match bonus)
        structural_sim = 1.0 if self._structural_match(other) else 0.0

        # Temporal similarity (inverse distance)
        temporal_dist = self.temporal_distance(other)
        temporal_sim = 1.0 - temporal_dist

        # Context similarity
        context_sim = 1.0 if self._context_match(other) else 0.0

        # Keyword overlap
        keyword_sim = self.keyword_overlap(other)

        # Weighted combination
        weights = {
            "semantic": 0.35,
            "structural": 0.25,
            "temporal": 0.15,
            "context": 0.15,
            "keyword": 0.10,
        }

        return (
            weights["semantic"] * semantic_sim
            + weights["structural"] * structural_sim
            + weights["temporal"] * temporal_sim
            + weights["context"] * context_sim
            + weights["keyword"] * keyword_sim
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "semantic_hash": list(self.semantic_hash),
            "structural_hash": self.structural_hash,
            "temporal_hash": list(self.temporal_hash),
            "context_hash": self.context_hash,
            "tool_name": self.tool_name,
            "outcome_type": self.outcome_type,
            "mode": self.mode,
            "goal_keywords": list(self.goal_keywords),
        }

    @classmethod
    def from_dict(cls, data: dict) -> SituationSignature:
        """Deserialize from dictionary."""
        return cls(
            semantic_hash=tuple(data.get("semantic_hash", [0] * 8)),
            structural_hash=data.get("structural_hash", 0),
            temporal_hash=tuple(data.get("temporal_hash", [0, 0, 0, 0])),
            context_hash=data.get("context_hash", 0),
            tool_name=data.get("tool_name", ""),
            outcome_type=data.get("outcome_type", ""),
            mode=data.get("mode", ""),
            goal_keywords=tuple(data.get("goal_keywords", [])),
        )


__all__ = ["SituationSignature"]
