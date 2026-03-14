"""Foundational context loading and configuration for LLM worker.

Module-level configuration and caching functions with no class dependency.
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)


_COST_BRIDGE_DEFAULTS: dict[str, float] = {
    "cost_energy_scale": 100.0,  # $1.00 -> 100 energy units
}


def _load_cost_bridge_config(path: str = "data/util/energy.json") -> dict[str, float]:
    cfg = dict(_COST_BRIDGE_DEFAULTS)
    if not path or not os.path.exists(path):
        return cfg
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return cfg
    if not isinstance(raw, dict):
        return cfg
    bridge = raw.get("cost_bridge")
    if not isinstance(bridge, dict):
        return cfg
    try:
        cfg["cost_energy_scale"] = float(bridge.get("cost_energy_scale", cfg["cost_energy_scale"]))
    except Exception:
        pass
    return cfg


_CLOUD_PROVIDER_TYPES = {
    "anthropic",
    "claude",
    "openai",
    "openai_compatible",
    "openai_compat",
}


def _is_cloud_provider_type(provider_type: str) -> bool:
    return str(provider_type or "").strip().lower().replace("-", "_") in _CLOUD_PROVIDER_TYPES


# ─────────────────────────────────────────────────────────────────────────────
# Foundational Documents (Constitution & Agent Rules)
# ─────────────────────────────────────────────────────────────────────────────

_foundational_context_cache: str | None = None


def _load_foundational_context() -> str:
    """Load CONSTITUTION.md and AGENTS.md as foundational LLM context.

    Caches the result to avoid repeated file I/O.
    Returns a condensed version suitable for prompts.
    """
    global _foundational_context_cache
    if _foundational_context_cache is not None:
        return _foundational_context_cache

    from pathlib import Path

    parts: list[str] = []

    # Find repo root (look for CONSTITUTION.md or AGENTS.md)
    current = Path(__file__).resolve()
    repo_root = None
    for parent in current.parents:
        if (parent / "CONSTITUTION.md").exists() or (parent / "AGENTS.md").exists():
            repo_root = parent
            break

    if repo_root is None:
        logger.debug("Could not find repo root for foundational documents")
        _foundational_context_cache = ""
        return ""

    # Load CONSTITUTION.md - extract key principles (condensed for prompt)
    constitution_path = repo_root / "CONSTITUTION.md"
    if constitution_path.exists():
        try:
            # Extract key constitutional principles (condensed for prompt efficiency)
            parts.append("=== CORE PRINCIPLES (from Constitution) ===")
            parts.append("Priority Order: 1) Physical Safety 2) Ethics 3) Guidelines 4) Helpfulness")
            parts.append("")
            parts.append("Hard Constraints (NEVER violate):")
            parts.append("- Never move toward a person who said 'stop' or shows distress")
            parts.append("- Never continue movement after unexpected collision")
            parts.append("- Never attempt to prevent being powered off")
            parts.append("- Never fabricate information or claim false certainty")
            parts.append("")
            parts.append("Core Values: Honesty, transparency, respect for persons, avoiding harm")
            parts.append("When uncertain: Ask rather than assume. Halt rather than proceed blindly.")
            logger.debug("Loaded constitution principles")
        except Exception as e:
            logger.warning("Failed to load CONSTITUTION.md: %s", e)

    # Load AGENTS.md - extract agent behavior rules
    agents_path = repo_root / "AGENTS.md"
    if agents_path.exists():
        try:
            # Extract key agent behavior rules (condensed for prompt efficiency)
            parts.append("")
            parts.append("=== AGENT BEHAVIOR RULES (from AGENTS.md) ===")
            parts.append("Agents THINK but do not ACT directly.")
            parts.append("")
            parts.append("You MAY: Read state, query memory, propose intents, evaluate outcomes")
            parts.append("You MAY NOT: Execute tools directly, mutate state, control execution loops")
            parts.append("")
            parts.append("Output: Structured intent (JSON), never imperative commands")
            parts.append("Coordination: Through state and decision engine, not direct agent calls")
            logger.debug("Loaded agent rules")
        except Exception as e:
            logger.warning("Failed to load AGENTS.md: %s", e)

    _foundational_context_cache = "\n".join(parts) if parts else ""
    return _foundational_context_cache
