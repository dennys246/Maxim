"""Numerical Workspace — session-scoped working memory for agents.

A shared scratchpad where agents store and retrieve numerical values
across loop iterations.  The IPS auto-categorizes each stored value
by magnitude.  Computation history is logged for learning and replay.

Saved/loaded as part of the mandatory consolidation cycle in
MemoryHub.on_session_end() / on_session_start().
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass
from typing import Any

from maxim.math.ips import IPS
from maxim.math.types import (
    AnalysisResult,
    ApproximateResult,
    ExactResult,
    MagnitudeCategory,
    NumericalValue,
)


@dataclass
class WorkspaceConfig:
    """Configuration for the NumericalWorkspace."""

    max_registers: int = 100
    history_limit: int = 500
    auto_categorize: bool = True


class NumericalWorkspace:
    """Working memory for numerical values — shared across agents.

    Values persist across agent loop iterations within a session.
    """

    def __init__(self, ips: IPS, config: WorkspaceConfig | None = None) -> None:
        self._ips = ips
        self.config = config or WorkspaceConfig()

        self._registers: dict[str, NumericalValue] = {}
        self._history: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=self.config.history_limit
        )

    def store(self, name: str, value: float | list[float], source: str = "") -> None:
        """Store a named value. IPS auto-categorizes magnitude."""
        if len(self._registers) >= self.config.max_registers and name not in self._registers:
            # Evict oldest by timestamp
            oldest = min(self._registers.values(), key=lambda v: v.timestamp)
            del self._registers[oldest.name]

        if self.config.auto_categorize:
            if isinstance(value, list):
                # Categorize the mean for lists
                avg = sum(value) / len(value) if value else 0.0
                magnitude = self._ips.categorize(avg)
            else:
                magnitude = self._ips.categorize(value)
        else:
            magnitude = MagnitudeCategory.MODERATE

        self._registers[name] = NumericalValue(
            name=name,
            value=value,
            magnitude=magnitude,
            source=source,
            timestamp=time.time(),
        )

    def recall(self, name: str) -> NumericalValue | None:
        """Retrieve a named value."""
        return self._registers.get(name)

    def recall_all(self) -> dict[str, NumericalValue]:
        """Get all current workspace values."""
        return dict(self._registers)

    def clear(self, name: str | None = None) -> None:
        """Clear one or all registers."""
        if name is not None:
            self._registers.pop(name, None)
        else:
            self._registers.clear()

    def record_computation(
        self,
        operation: str,
        inputs: dict[str, Any],
        result: ExactResult | ApproximateResult | AnalysisResult,
    ) -> None:
        """Log a computation to history for learning and replay."""
        entry = {
            "timestamp": time.time(),
            "operation": operation,
            "inputs": inputs,
            "result_type": type(result).__name__,
        }

        if isinstance(result, ExactResult):
            entry["value"] = result.value
            entry["verbal"] = result.verbal
        elif isinstance(result, ApproximateResult):
            entry["value"] = result.value
            entry["confidence"] = result.confidence
            entry["magnitude"] = result.magnitude.name
        elif isinstance(result, AnalysisResult):
            entry["method"] = result.method
            entry["parameters"] = result.parameters
            entry["verbal"] = result.verbal
            entry["confidence"] = result.confidence

        self._history.append(entry)

    def get_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Recent computation history (newest first)."""
        items = list(self._history)
        items.reverse()
        return items[:limit]

    def summarize(self) -> str:
        """Natural language summary of workspace state for LLM context.

        Example: 'Workspace: speed=2.3 (MODERATE), count=1547 (LARGE)'
        """
        if not self._registers:
            return "Workspace: empty"

        parts: list[str] = []
        for nv in sorted(self._registers.values(), key=lambda v: v.timestamp, reverse=True):
            if isinstance(nv.value, list):
                val_str = f"[{len(nv.value)} items]"
            else:
                val_str = f"{nv.value:.4g}"
            parts.append(f"{nv.name}={val_str} ({nv.magnitude.name})")

        return f"Workspace: {', '.join(parts)}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize workspace state for persistence."""
        return {
            "registers": {
                name: {
                    "name": nv.name,
                    "value": nv.value,
                    "magnitude": nv.magnitude.name,
                    "source": nv.source,
                    "timestamp": nv.timestamp,
                }
                for name, nv in self._registers.items()
            },
            "history": list(self._history),
        }

    def load_from_dict(self, data: dict[str, Any]) -> None:
        """Restore workspace state from persisted data."""
        self._registers.clear()
        self._history.clear()

        for _name, rd in data.get("registers", {}).items():
            try:
                magnitude = MagnitudeCategory[rd.get("magnitude", "MODERATE")]
            except KeyError:
                magnitude = MagnitudeCategory.MODERATE

            self._registers[rd["name"]] = NumericalValue(
                name=rd["name"],
                value=rd["value"],
                magnitude=magnitude,
                source=rd.get("source", ""),
                timestamp=rd.get("timestamp", 0.0),
            )

        for entry in data.get("history", []):
            self._history.append(entry)
