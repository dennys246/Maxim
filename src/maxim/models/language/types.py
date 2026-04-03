"""LLM type definitions — pure data types with no business logic.

Extracted from router.py for modularity. These types are used by
LLMRouter, backends (openai, anthropic), and external callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class LLMResponse:
    """Structured response from any LLM backend."""

    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    latency_ms: float = 0.0
    provider: str = ""
    stop_reason: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    cached_input_tokens: int = 0
    uncached_input_tokens: int = 0


@dataclass(slots=True)
class RoutingPolicy:
    """Governs routing and budget enforcement for providers."""

    provider_priority: list[str] = field(default_factory=list)
    fallback_on_rate_limit: bool = True
    fallback_on_timeout: bool = True
    fallback_on_budget_exceeded: str = "local"
    require_cloud_opt_in: bool = True
    context_window_routing: bool = True
    max_cost_per_request: float = 0.50
    max_cost_per_hour: float = 1.00
    max_cost_per_day: float = 10.00
    max_cost_per_month: float = 100.00
    max_session_cost: float = 5.00  # Hard ceiling — rejects ALL requests when hit
    cost_warning_threshold: float = 0.80
    cost_critical_threshold: float = 0.95


@dataclass
class ProviderState:
    """Tracks provider health and backoff state."""

    backoff_until: float = 0.0
    consecutive_errors: int = 0
    last_error: str = ""
    last_success: float = 0.0
