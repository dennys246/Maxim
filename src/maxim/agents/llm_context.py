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


def _load_cost_bridge_config(path: str = "") -> dict[str, float]:
    cfg = dict(_COST_BRIDGE_DEFAULTS)
    if not path:
        from maxim.utils.paths import resolve_user_state

        path = str(resolve_user_state("util/energy.json"))
    if not os.path.exists(path):
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
    """The foundational preamble: the Constitution's core principles plus the agent behaviour rules.

    Gated on the constitution SHIPPED IN THE PACKAGE (``maxim/_data/CONSTITUTION.md``, D32). It used to
    walk up from this file looking for a repo-root ``CONSTITUTION.md`` / ``AGENTS.md`` -- present in a
    checkout, absent in every installed wheel -- so every pip user's agent ran with an EMPTY preamble.
    The repo-root file stays the source; ``tests/unit/test_foundational_preamble.py`` fails when the
    packaged copy drifts from it. Cached: the text is session-stable (it is a cacheable prompt section).
    """
    global _foundational_context_cache
    if _foundational_context_cache is not None:
        return _foundational_context_cache

    from importlib import resources

    if not resources.files("maxim").joinpath("_data", "CONSTITUTION.md").is_file():
        logger.warning(
            "maxim/_data/CONSTITUTION.md is missing from the install: the agent has no foundational preamble"
        )
        _foundational_context_cache = ""
        return ""

    parts = [
        "=== CORE PRINCIPLES (from Constitution) ===",
        "Priority Order: 1) Physical Safety 2) Ethics 3) Guidelines 4) Helpfulness",
        "",
        "Hard Constraints (NEVER violate):",
        "- Never move toward a person who said 'stop' or shows distress",
        "- Never continue movement after unexpected collision",
        "- Never attempt to prevent being powered off",
        "- Never fabricate information or claim false certainty",
        "",
        "Core Values: Honesty, transparency, respect for persons, avoiding harm",
        "When uncertain: Ask rather than assume. Halt rather than proceed blindly.",
        "",
        "=== AGENT BEHAVIOR RULES ===",
        "Agents THINK but do not ACT directly.",
        "",
        "You MAY: Read state, query memory, propose intents, evaluate outcomes",
        "You MAY NOT: Execute tools directly, mutate state, control execution loops",
        "",
        "Output: Structured intent (JSON), never imperative commands",
        "Coordination: Through state and decision engine, not direct agent calls",
    ]
    _foundational_context_cache = "\n".join(parts)
    return _foundational_context_cache
