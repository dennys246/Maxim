"""LLM energy tracker - Tracks token usage and inference costs.

Monitors LLM API calls and estimates energy based on:
- Token counts (input and output)
- Model type (larger models = more energy)
- Latency (time waiting for response)

This enables:
- Tracking "cognitive energy" expenditure
- Learning which prompts are expensive
- Making energy-aware model routing decisions
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from maxim.energy.signal import EnergySignal, EnergyType
from maxim.energy.tracker import EnergyConfig, EnergyTracker

logger = logging.getLogger(__name__)


@dataclass
class LLMEnergyConfig(EnergyConfig):
    """Configuration for LLM energy tracking.

    Attributes:
        input_token_cost: Energy per input token.
        output_token_cost: Energy per output token (typically higher).
        latency_cost_per_second: Energy cost for waiting.
        model_multipliers: Energy multipliers by model name.
    """

    # Token costs (relative energy units)
    input_token_cost: float = 0.001      # 1000 input tokens = 1 energy
    output_token_cost: float = 0.003     # 333 output tokens = 1 energy (generation is expensive)

    # Latency cost (represents opportunity cost of waiting)
    latency_cost_per_second: float = 0.1  # 10 seconds = 1 energy

    # Model-specific multipliers (larger = more energy)
    model_multipliers: dict[str, float] = field(default_factory=lambda: {
        # Claude models
        "claude-3-haiku": 0.5,
        "claude-3-sonnet": 1.0,
        "claude-3-opus": 2.0,
        "claude-opus-4-5": 2.5,
        "claude-sonnet-4-5": 1.2,
        "claude-haiku-4-5": 0.6,
        # OpenAI models (if used)
        "gpt-4o": 1.5,
        "gpt-4o-mini": 0.4,
        "gpt-4-turbo": 1.8,
        # Local models (if used)
        "local": 0.2,            # Local inference is cheap
        "ollama": 0.3,
    })

    # Default multiplier for unknown models
    default_multiplier: float = 1.0


class LLMEnergyTracker(EnergyTracker):
    """Tracks energy expenditure from LLM API calls.

    Records token usage and latency for each LLM call, converts
    to normalized energy units, and provides statistics.

    Example:
        tracker = LLMEnergyTracker()

        # Record an LLM call
        signal = tracker.record(
            input_tokens=500,
            output_tokens=150,
            model="claude-3-haiku",
            latency_ms=1200,
            context={"prompt_type": "planning"},
        )

        # Get stats
        stats = tracker.get_window_stats()
        print(f"Tokens this minute: {stats['total_tokens']}")
    """

    name = "llm"
    energy_types = {EnergyType.LLM_TOKENS, EnergyType.LLM_LATENCY}

    def __init__(self, config: LLMEnergyConfig | None = None) -> None:
        """Initialize the LLM energy tracker.

        Args:
            config: LLM-specific configuration. Uses defaults if None.
        """
        super().__init__(config or LLMEnergyConfig())
        self._llm_config = config or LLMEnergyConfig()

        # LLM-specific accumulators
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_latency_ms = 0.0
        self._call_count = 0

    def record(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str = "",
        latency_ms: float = 0.0,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> EnergySignal:
        """Record an LLM API call.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            model: Model name/ID for multiplier lookup.
            latency_ms: Response latency in milliseconds.
            context: Additional context (prompt type, agent, etc.).

        Returns:
            EnergySignal representing this call's energy cost.
        """
        # Calculate token energy
        multiplier = self._get_model_multiplier(model)

        token_energy = (
            input_tokens * self._llm_config.input_token_cost +
            output_tokens * self._llm_config.output_token_cost
        ) * multiplier

        # Calculate latency energy
        latency_energy = (latency_ms / 1000.0) * self._llm_config.latency_cost_per_second

        # Total energy
        total_energy = token_energy + latency_energy

        # Build context
        signal_context = context or {}
        signal_context.update({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "model": model,
            "multiplier": multiplier,
            "token_energy": round(token_energy, 4),
            "latency_energy": round(latency_energy, 4),
        })

        signal = EnergySignal(
            energy_type=EnergyType.LLM_TOKENS,
            amount=total_energy,
            timestamp=time.time(),
            source=self.name,
            duration_ms=latency_ms,
            context=signal_context,
        )

        # Update LLM-specific accumulators
        with self._lock:
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            self._total_latency_ms += latency_ms
            self._call_count += 1

        self._record_signal(signal)
        return signal

    def _get_model_multiplier(self, model: str) -> float:
        """Get energy multiplier for a model.

        Args:
            model: Model name/ID.

        Returns:
            Multiplier (1.0 = baseline).
        """
        if not model:
            return self._llm_config.default_multiplier

        # Try exact match first
        model_lower = model.lower()
        if model_lower in self._llm_config.model_multipliers:
            return self._llm_config.model_multipliers[model_lower]

        # Try prefix matching
        for key, mult in self._llm_config.model_multipliers.items():
            if key in model_lower or model_lower in key:
                return mult

        return self._llm_config.default_multiplier

    def get_llm_stats(self, window_seconds: float | None = None) -> dict[str, Any]:
        """Get LLM-specific statistics.

        Args:
            window_seconds: Window for rate calculations.

        Returns:
            Dict with token counts, rates, and costs.
        """
        base_stats = self.get_window_stats(window_seconds)

        with self._lock:
            return {
                **base_stats,
                # Totals
                "total_input_tokens": self._total_input_tokens,
                "total_output_tokens": self._total_output_tokens,
                "total_tokens": self._total_input_tokens + self._total_output_tokens,
                "total_latency_ms": round(self._total_latency_ms, 2),
                "call_count": self._call_count,
                # Averages
                "avg_tokens_per_call": round(
                    (self._total_input_tokens + self._total_output_tokens) /
                    max(self._call_count, 1),
                    1
                ),
                "avg_latency_ms": round(
                    self._total_latency_ms / max(self._call_count, 1),
                    2
                ),
            }

    def get_token_budget_status(
        self,
        budget_tokens: int = 100000,
    ) -> dict[str, Any]:
        """Check token usage against a budget.

        Args:
            budget_tokens: Token budget to check against.

        Returns:
            Dict with usage, remaining, and percentage.
        """
        with self._lock:
            used = self._total_input_tokens + self._total_output_tokens

        return {
            "budget": budget_tokens,
            "used": used,
            "remaining": budget_tokens - used,
            "percentage": round((used / budget_tokens) * 100, 1),
            "is_over_budget": used > budget_tokens,
        }

    def clear(self) -> None:
        """Clear history and reset LLM-specific counters."""
        super().clear()
        with self._lock:
            self._total_input_tokens = 0
            self._total_output_tokens = 0
            self._total_latency_ms = 0.0
            self._call_count = 0


__all__ = [
    "LLMEnergyTracker",
    "LLMEnergyConfig",
]
