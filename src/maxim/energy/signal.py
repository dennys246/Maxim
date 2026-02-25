"""Energy signal types - Represents energy expenditure events.

Energy signals are the fundamental unit of energy tracking, analogous to
pain signals in the pain detection system. Each signal represents a
discrete energy expenditure with its source, magnitude, and context.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EnergyType(Enum):
    """Categories of energy expenditure.

    Different types allow for domain-specific tracking and analysis,
    while maintaining a unified energy accounting system.
    """

    # Compute-related energy
    LLM_TOKENS = "llm_tokens"           # Token-based energy (input + output)
    LLM_LATENCY = "llm_latency"         # Time waiting for LLM response
    LLM_COST = "llm_cost"               # USD-normalized cost signal
    COMPUTE_TIME = "compute_time"       # General CPU/GPU compute time

    # Movement-related energy
    MOTOR_COMMAND = "motor_command"     # Energy to execute movement
    MOTOR_CURRENT = "motor_current"     # Actual motor current draw (if available)

    # Sensor-related energy
    VISION_INFERENCE = "vision_inference"   # Vision model inference
    AUDIO_PROCESSING = "audio_processing"   # Audio transcription/TTS

    # Meta energy
    ATTENTION = "attention"             # Cognitive attention/focus cost
    MEMORY_ACCESS = "memory_access"     # Memory retrieval cost


@dataclass
class EnergySignal:
    """A recorded energy expenditure event.

    Attributes:
        energy_type: Category of energy expenditure.
        amount: Energy amount in normalized units (0-100 scale typical).
        timestamp: When the energy was expended.
        source: Identifier for what produced this signal (e.g., "llm_worker").
        context: Additional metadata about the expenditure.
    """

    energy_type: EnergyType
    amount: float                       # Normalized energy units
    timestamp: float = field(default_factory=time.time)
    source: str = ""                    # e.g., "llm_worker", "motor_cortex"
    duration_ms: float = 0.0           # How long the activity took
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "energy_type": self.energy_type.value,
            "amount": round(self.amount, 4),
            "timestamp": self.timestamp,
            "source": self.source,
            "duration_ms": round(self.duration_ms, 2),
            "context": self.context,
        }

    @property
    def is_significant(self) -> bool:
        """Check if this is a significant energy expenditure."""
        # Thresholds for different energy types
        thresholds = {
            EnergyType.LLM_TOKENS: 1.0,
            EnergyType.LLM_LATENCY: 0.5,
            EnergyType.LLM_COST: 1.0,
            EnergyType.MOTOR_COMMAND: 0.5,
            EnergyType.VISION_INFERENCE: 0.3,
        }
        threshold = thresholds.get(self.energy_type, 0.1)
        return self.amount >= threshold


@dataclass
class EnergyBudget:
    """Current energy budget state for a domain.

    Tracks available energy and spending rate to enable
    energy-aware decision making.
    """

    domain: EnergyType
    total_capacity: float = 100.0       # Maximum energy available
    current_level: float = 100.0        # Current available energy
    recharge_rate: float = 1.0          # Energy recovered per second
    last_update: float = field(default_factory=time.time)

    def spend(self, amount: float) -> bool:
        """Attempt to spend energy.

        Args:
            amount: Energy to spend.

        Returns:
            True if energy was available, False if insufficient.
        """
        self._recharge()
        if amount > self.current_level:
            return False
        self.current_level -= amount
        return True

    def _recharge(self) -> None:
        """Apply passive energy regeneration."""
        now = time.time()
        elapsed = now - self.last_update
        self.current_level = min(
            self.total_capacity,
            self.current_level + elapsed * self.recharge_rate
        )
        self.last_update = now

    @property
    def percentage(self) -> float:
        """Current energy as percentage of capacity."""
        self._recharge()
        return (self.current_level / self.total_capacity) * 100

    @property
    def is_low(self) -> bool:
        """Check if energy is below 25%."""
        return self.percentage < 25.0

    @property
    def is_critical(self) -> bool:
        """Check if energy is below 10%."""
        return self.percentage < 10.0


__all__ = [
    "EnergyType",
    "EnergySignal",
    "EnergyBudget",
]
