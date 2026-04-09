"""Cloud dispatch constants — pricing, model downgrades, JSON prompts.

Extracted from router.py for single-responsibility decomposition.
These are module-level constants and static config loaders used by LLMRouter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from maxim.models.language.cost_tracker import CostTrackerConfig, ModelPricing
from maxim.models.language.types import RoutingPolicy


# ─── Default pricing table ──────────────────────────────────────────────
DEFAULT_PRICING: dict[str, ModelPricing] = {
    "claude-sonnet-4-5-20250514": ModelPricing(3.00, 15.00, 0.30),
    "claude-haiku-4-5-20251001": ModelPricing(0.80, 4.00, 0.08),
    "claude-opus-4-5-20250514": ModelPricing(15.00, 75.00, 1.50),
    "gpt-4o": ModelPricing(2.50, 10.00, 1.25),
    "gpt-4o-mini": ModelPricing(0.15, 0.60, 0.075),
    "local": ModelPricing(0.0, 0.0, 0.0),
}

# ─── Model downgrade map for budget tiers ────────────────────────────────
MODEL_DOWNGRADE_MAP: dict[str, str] = {
    "claude-opus-4-5-20250514": "claude-sonnet-4-5-20250514",
    "claude-sonnet-4-5-20250514": "claude-haiku-4-5-20251001",
    "gpt-4o": "gpt-4o-mini",
    "gpt-4-turbo": "gpt-4o-mini",
}


# ─── Centralized JSON instruction prompts ────────────────────────────────
# Used across all LLM→JSON paths for consistency.  The quote-escaping
# guidance is critical for small/medium models (7B–14B) that embed
# narrative dialogue with unescaped double quotes.

JSON_RULES = (
    "CRITICAL JSON RULES:\n"
    "- Output ONLY valid JSON. No text before or after the JSON object.\n"
    '- Escape ALL double quotes inside string values with backslash: \\"  \n'
    '- Example: {"message": "She said \\"hello\\" to me"}\n'
    "- Never use unescaped double quotes inside a JSON string value.\n"
    "- If including dialogue or quotes, always backslash-escape them."
)

SYSTEM_TOOL_RESPONSE = (
    "You are Maxim, an intelligent robot assistant. "
    "You MUST respond with valid JSON only. No explanations outside JSON. "
    "Select the most appropriate tool based on the context and user request. "
    "If unsure, use 'respond' to communicate with the user.\n\n" + JSON_RULES
)

SYSTEM_JSON_ONLY = (
    "You are a JSON-only response system. "
    "Output ONLY valid JSON. No explanations, no code, no markdown. "
    "If the user prompt contains partial JSON, complete it. "
    "Never explain how to create JSON - just output the JSON directly.\n\n" + JSON_RULES
)

SYSTEM_ROUTE = (
    "You are Maxim, a local robot assistant. "
    "Return ONLY a single JSON object (no prose) describing the next action.\n\n" + JSON_RULES
)


# ─── Static config loaders (called from LLMRouter.__init__) ──────────────


def normalize_providers(cfg: Any) -> dict[str, dict[str, Any]]:
    """Normalize the providers dict from LLMConfig."""
    providers = cfg.providers if isinstance(cfg.providers, dict) else {}
    if providers:
        return {str(k): v for k, v in providers.items() if isinstance(k, str) and isinstance(v, dict)}
    return {
        "local": {
            "type": cfg.backend or "llama_cpp",
            "model": cfg.model,
            "model_base": cfg.model_base,
            "model_path": cfg.model_path,
            "n_ctx": cfg.n_ctx,
        }
    }


def load_routing_policy(raw: dict[str, Any]) -> RoutingPolicy:
    """Parse a routing policy from raw config dict."""
    data = raw if isinstance(raw, dict) else {}
    return RoutingPolicy(
        provider_priority=list(data.get("provider_priority", []))
        if isinstance(data.get("provider_priority"), list)
        else [],
        fallback_on_rate_limit=bool(data.get("fallback_on_rate_limit", True)),
        fallback_on_timeout=bool(data.get("fallback_on_timeout", True)),
        fallback_on_budget_exceeded=str(data.get("fallback_on_budget_exceeded", "local") or "local"),
        require_cloud_opt_in=bool(data.get("require_cloud_opt_in", True)),
        context_window_routing=bool(data.get("context_window_routing", True)),
        max_cost_per_request=float(data.get("max_cost_per_request", 0.50) or 0.0),
        max_cost_per_hour=float(data.get("max_cost_per_hour", 1.00) or 0.0),
        max_cost_per_day=float(data.get("max_cost_per_day", 10.00) or 0.0),
        max_cost_per_month=float(data.get("max_cost_per_month", 100.00) or 0.0),
        max_session_cost=float(data.get("max_session_cost", 5.00) or 0.0),
        cost_warning_threshold=float(data.get("cost_warning_threshold", 0.80) or 0.0),
        cost_critical_threshold=float(data.get("cost_critical_threshold", 0.95) or 0.0),
    )


def load_pricing_table(cfg: Any) -> dict[str, ModelPricing]:
    """Build the pricing table from defaults + config overrides."""
    pricing: dict[str, ModelPricing] = dict(DEFAULT_PRICING)
    raw = cfg.pricing if isinstance(cfg.pricing, dict) else {}
    for model, entry in raw.items():
        if not isinstance(model, str) or not isinstance(entry, dict):
            continue
        try:
            pricing[model] = ModelPricing(
                input_price=float(entry.get("input_price", entry.get("input", 0.0)) or 0.0),
                output_price=float(entry.get("output_price", entry.get("output", 0.0)) or 0.0),
                cached_input_price=float(entry.get("cached_input_price", entry.get("cached_input", 0.0)) or 0.0),
            )
        except Exception:
            continue
    return pricing


def load_cost_config(cfg: Any) -> CostTrackerConfig:
    """Build cost tracker config from LLMConfig."""
    raw = cfg.routing if isinstance(cfg.routing, dict) else {}
    return CostTrackerConfig(
        state_path=str(raw.get("cost_state_path", str(Path.home() / ".maxim" / "util" / "cost_state.json"))),
        persistence_interval_s=float(raw.get("cost_persistence_interval_s", 10.0) or 10.0),
        persistence_interval_n=int(raw.get("cost_persistence_interval_n", 5) or 5),
        reserved_budget_ratio=float(raw.get("reserved_budget_ratio", 0.2) or 0.2),
        min_spend_samples=int(raw.get("min_spend_samples", 5) or 5),
    )
