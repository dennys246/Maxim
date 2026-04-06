"""UMR — Uniform Memory Reference for cross-agent addressing.

Format: {nickname}.{region}.{id}

Examples:
    researcher.hippo.exp_001   — experiment 001 in researcher's hippocampus
    writer.paper.methods       — methods section of writer's paper
    reviewer.review.round_1    — reviewer's first-round review
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UMR:
    """Parsed uniform memory reference."""

    nickname: str  # Agent nickname ("researcher")
    region: str  # Memory region or namespace ("hippo", "paper", "review")
    ref_id: str  # Specific item ID ("exp_001", "methods")

    def __str__(self) -> str:
        return f"{self.nickname}.{self.region}.{self.ref_id}"

    @classmethod
    def build(cls, nickname: str, region: str, ref_id: str) -> UMR:
        return cls(nickname=nickname, region=region, ref_id=ref_id)


def parse_umr(s: str) -> UMR:
    """Parse a UMR string into its components.

    Raises ValueError if the string doesn't have exactly 3 dot-separated parts.
    """
    parts = s.split(".", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid UMR '{s}': expected nickname.region.id (3 dot-separated parts)")
    return UMR(nickname=parts[0], region=parts[1], ref_id=parts[2])
