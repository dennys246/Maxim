"""Memory record types for Angular Gyrus mathematical knowledge.

Defines MathMemory and CompressedMathMemory — the data structures
stored in the Angular Gyrus MemoryLayer.  Every record carries a dual
verbal/code representation, bridging natural language and computation.

MathMemory extends the MemoryRecord ABC (memory/types.py), inheriting
common tracking fields (id, timestamp, created_at, accessed_at,
access_count, long_term, consolidated_at, touch()).

CompressedMathMemory extends CompressedRecord, inheriting the above
plus edge_count.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from maxim.math.types import MathCategory
from maxim.memory.types import CompressedRecord, MemoryRecord


@dataclass
class MathMemory(MemoryRecord):
    """A mathematical memory — fact, formula, method, or learned pattern.

    Stored in the Angular Gyrus MemoryLayer.  Dual representation:
    every record has both a verbal (natural language) and code
    (executable) form, reflecting the angular gyrus's role as a
    language-mediated math region.

    Inherits from MemoryRecord: id, timestamp, created_at, accessed_at,
    access_count, long_term, consolidated_at, touch().
    """

    # --- Math-specific fields ---
    name: str = ""  # "linear_regression", "pi", "area_circle"
    category: MathCategory = MathCategory.FACT
    domain: str = ""  # "statistics", "arithmetic", "geometry"

    # Dual representation (angular gyrus is language-mediated)
    verbal: str = ""  # "The mean of a dataset is the sum divided by count"
    code: str = ""  # "sum(data) / len(data)"

    # Mathematical properties
    inputs: list[str] = field(default_factory=list)  # ["list[float]"]
    outputs: list[str] = field(default_factory=list)  # ["float"]

    # Provenance & confidence
    source: str = ""  # "built_in", "learned", "derived"
    confidence: float = 1.0
    observation_count: int = 0  # Times used/verified

    def keywords(self) -> set[str]:
        """Extract keywords from math memory fields."""
        kws: set[str] = set()
        if self.name:
            kws.add(self.name.lower())
        if self.domain:
            kws.add(self.domain.lower())
        return kws

    def to_context_dict(self) -> dict[str, Any]:
        """Format math memory for LLM context."""
        return {
            "type": "math",
            "name": self.name,
            "domain": self.domain,
            "verbal": self.verbal,
            "confidence": self.confidence,
            "category": self.category.name,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON persistence."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "access_count": self.access_count,
            "long_term": self.long_term,
            "consolidated_at": self.consolidated_at,
            "name": self.name,
            "category": self.category.name,
            "domain": self.domain,
            "verbal": self.verbal,
            "code": self.code,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "source": self.source,
            "confidence": self.confidence,
            "observation_count": self.observation_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MathMemory:
        """Deserialize from JSON."""
        category_str = data.get("category", "FACT")
        try:
            category = MathCategory[category_str]
        except KeyError:
            category = MathCategory.FACT

        return cls(
            id=data["id"],
            timestamp=data["timestamp"],
            created_at=data.get("created_at", data["timestamp"]),
            accessed_at=data.get("accessed_at", data["timestamp"]),
            access_count=data.get("access_count", 1),
            long_term=data.get("long_term", False),
            consolidated_at=data.get("consolidated_at"),
            name=data.get("name", ""),
            category=category,
            domain=data.get("domain", ""),
            verbal=data.get("verbal", ""),
            code=data.get("code", ""),
            inputs=data.get("inputs", []),
            outputs=data.get("outputs", []),
            source=data.get("source", ""),
            confidence=data.get("confidence", 1.0),
            observation_count=data.get("observation_count", 0),
        )


@dataclass
class CompressedMathMemory(CompressedRecord):
    """Lightweight compressed form of a math memory (~100 bytes).

    Keeps essential fields for pattern matching and recall scoring.
    Drops: verbal, code, inputs, outputs.

    Inherits from CompressedRecord: id, timestamp, created_at, accessed_at,
    access_count, long_term, consolidated_at, edge_count, touch().
    """

    # Compressed math fields
    name: str = ""
    category: MathCategory = MathCategory.FACT
    domain: str = ""
    confidence: float = 1.0

    def keywords(self) -> set[str]:
        """Extract keywords from compressed math memory."""
        kws: set[str] = set()
        if self.name:
            kws.add(self.name.lower())
        if self.domain:
            kws.add(self.domain.lower())
        return kws

    def to_context_dict(self) -> dict[str, Any]:
        """Format compressed math memory for LLM context."""
        return {
            "type": "compressed_math",
            "name": self.name,
            "domain": self.domain,
            "confidence": self.confidence,
            "category": self.category.name,
        }

    @classmethod
    def from_math_record(cls, record: MathMemory, edge_count: int = 0) -> CompressedMathMemory:
        """Compress a full MathMemory to lightweight form."""
        return cls(
            id=record.id,
            timestamp=record.timestamp,
            created_at=record.created_at,
            accessed_at=record.accessed_at,
            access_count=record.access_count,
            long_term=record.long_term,
            consolidated_at=record.consolidated_at,
            edge_count=edge_count,
            name=record.name,
            category=record.category,
            domain=record.domain,
            confidence=record.confidence,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON persistence."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "access_count": self.access_count,
            "long_term": self.long_term,
            "consolidated_at": self.consolidated_at,
            "edge_count": self.edge_count,
            "name": self.name,
            "category": self.category.name,
            "domain": self.domain,
            "confidence": self.confidence,
            "_compressed": True,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompressedMathMemory:
        """Deserialize from JSON."""
        category_str = data.get("category", "FACT")
        try:
            category = MathCategory[category_str]
        except KeyError:
            category = MathCategory.FACT

        return cls(
            id=data["id"],
            timestamp=data.get("timestamp", data.get("created_at", 0.0)),
            created_at=data.get("created_at", 0.0),
            accessed_at=data.get("accessed_at", 0.0),
            access_count=data.get("access_count", 1),
            long_term=data.get("long_term", False),
            consolidated_at=data.get("consolidated_at"),
            edge_count=data.get("edge_count", 0),
            name=data.get("name", ""),
            category=category,
            domain=data.get("domain", ""),
            confidence=data.get("confidence", 1.0),
        )


__all__ = [
    "MathMemory",
    "CompressedMathMemory",
]
