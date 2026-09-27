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


_PREAMBLE_START = "<!-- runtime-preamble:start -->"
_PREAMBLE_END = "<!-- runtime-preamble:end -->"


def extract_runtime_preamble(constitution: str) -> str:
    """The Constitution's "Runtime Preamble" block: the text between its markers, fence stripped."""
    _head, start, tail = constitution.partition(_PREAMBLE_START)
    block, end, _ = tail.partition(_PREAMBLE_END)
    if not start or not end:
        # Both markers or nothing: a lost END would otherwise inject the rest of the document.
        return ""
    lines = [line for line in block.strip().splitlines() if not line.startswith("```")]
    return "\n".join(lines).strip()


def _load_foundational_context() -> str:
    """The foundational preamble, read VERBATIM from the Constitution (D32, owner decision 2026-09-27).

    It was a hard-coded paraphrase gated on a repo-root CONSTITUTION.md existing: the document and the
    prompt could drift silently, and every pip install (no repo root) ran with an EMPTY preamble. The
    text now lives in CONSTITUTION.md's marked "Runtime Preamble" section, read from the copy shipped
    as package data (``maxim/_data/CONSTITUTION.md``; ``tests/unit/test_foundational_preamble.py`` fails
    when it drifts from the repo root). Cached: the text is session-stable (a cacheable prompt section).
    """
    global _foundational_context_cache
    if _foundational_context_cache is not None:
        return _foundational_context_cache

    from importlib import resources

    packaged = resources.files("maxim").joinpath("_data").joinpath("CONSTITUTION.md")
    text = extract_runtime_preamble(packaged.read_text(encoding="utf-8")) if packaged.is_file() else ""
    if not text:
        logger.warning(
            "No Runtime Preamble in maxim/_data/CONSTITUTION.md (missing file or markers): the agent has "
            "no foundational preamble"
        )
    _foundational_context_cache = text
    return _foundational_context_cache
