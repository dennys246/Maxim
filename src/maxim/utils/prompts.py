"""Prompt loader utility for external prompt management.

Loads prompts from data/prompts/ folder, enabling:
- Separation of prompts from code
- Easy prompt iteration without code changes
- Version control of prompt changes
- Hot-reloading capability (optional)
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default prompts directory relative to project root
_PROMPTS_DIR: Path | None = None


class ResponseFormat(Enum):
    """Response formatting styles for different modes."""

    MINIMAL = "minimal"  # Single phrase/sentence (observe, sleep)
    BRIEF = "brief"  # 2-3 sentences max (exploration)
    CONVERSATIONAL = "conversational"  # Normal dialogue (live, active-assistance)
    DETAILED = "detailed"  # Thorough explanations (reflection)
    ACADEMIC = "academic"  # Research paper style with citations (research)


def _find_prompts_dir() -> Path:
    """Find the prompts directory.

    Search order:
    1. Bundled package data (src/maxim/_data/prompts/)
    2. Repo checkout (walk up from this file to find data/prompts/)
    3. CWD/data/prompts/ (legacy fallback)
    """
    global _PROMPTS_DIR

    if _PROMPTS_DIR is not None:
        return _PROMPTS_DIR

    # 1. Bundled package data (works after pip install)
    try:
        from maxim.utils.paths import bundled_data

        bundled = bundled_data() / "prompts"
        if bundled.is_dir():
            _PROMPTS_DIR = bundled
            return _PROMPTS_DIR
    except Exception:
        pass

    # 2. Repo checkout (walk up from this file → data/prompts/)
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "data" / "prompts"
        if candidate.is_dir():
            _PROMPTS_DIR = candidate
            return _PROMPTS_DIR

    # 3. CWD fallback (legacy)
    cwd_candidate = Path.cwd() / "data" / "prompts"
    if cwd_candidate.is_dir():
        _PROMPTS_DIR = cwd_candidate
        return _PROMPTS_DIR

    # Fall back to user data home — never create directories in CWD
    try:
        from maxim.utils.paths import data_home

        fallback = data_home() / "prompts"
        fallback.mkdir(parents=True, exist_ok=True)
        _PROMPTS_DIR = fallback
        return _PROMPTS_DIR
    except Exception:
        # Last resort: return CWD candidate without creating it
        _PROMPTS_DIR = cwd_candidate
        return _PROMPTS_DIR


def set_prompts_dir(path: Path | str) -> None:
    """Override the prompts directory location."""
    global _PROMPTS_DIR
    _PROMPTS_DIR = Path(path)


def get_prompts_dir() -> Path:
    """Get the current prompts directory."""
    return _find_prompts_dir()


def clear_cache() -> None:
    """Clear the prompt cache (for hot-reloading)."""
    load_prompt.cache_clear()
    load_json_prompt.cache_clear()


@lru_cache(maxsize=64)
def load_prompt(category: str, name: str, default: str = "") -> str:
    """Load a prompt template from file.

    Args:
        category: Subdirectory (e.g., "modes", "strategies/reactive")
        name: Prompt name without extension
        default: Default value if file not found

    Returns:
        Prompt text or default if not found
    """
    prompts_dir = _find_prompts_dir()
    path = prompts_dir / category / f"{name}.txt"

    if path.exists():
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception as e:
            logger.warning(f"Failed to load prompt {path}: {e}")
            return default

    # Try .md extension as fallback
    md_path = path.with_suffix(".md")
    if md_path.exists():
        try:
            return md_path.read_text(encoding="utf-8").strip()
        except Exception as e:
            logger.warning(f"Failed to load prompt {md_path}: {e}")
            return default

    logger.debug(f"Prompt not found: {path}")
    return default


@lru_cache(maxsize=32)
def load_json_prompt(category: str, name: str) -> dict[str, Any]:
    """Load a JSON prompt configuration.

    Args:
        category: Subdirectory (e.g., "fallbacks")
        name: Prompt name without extension

    Returns:
        Parsed JSON dict or empty dict if not found
    """
    prompts_dir = _find_prompts_dir()
    path = prompts_dir / category / f"{name}.json"

    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Failed to load JSON prompt {path}: {e}")
            return {}

    logger.debug(f"JSON prompt not found: {path}")
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────


def get_mode_prompt(mode_name: str) -> str:
    """Load a mode's context prompt."""
    # Normalize name (active-assistance -> active_assistance)
    normalized = mode_name.lower().replace("-", "_")
    return load_prompt("modes", normalized)


def get_strategy_prompt(strategy_name: str, category: str = "") -> str:
    """Load a strategy's approach prompt.

    Args:
        strategy_name: Strategy name
        category: Optional subcategory (reactive, proactive, learning, exploration)
    """
    normalized = strategy_name.lower().replace("-", "_")

    # Try with category first
    if category:
        prompt = load_prompt(f"strategies/{category}", normalized)
        if prompt:
            return prompt

    # Try each category
    for cat in ["reactive", "proactive", "learning", "exploration"]:
        prompt = load_prompt(f"strategies/{cat}", normalized)
        if prompt:
            return prompt

    # Fallback to root strategies folder
    return load_prompt("strategies", normalized)


def get_format_prompt(format_name: str | ResponseFormat) -> str:
    """Load a response format template."""
    if isinstance(format_name, ResponseFormat):
        format_name = format_name.value
    return load_prompt("formats", format_name.lower())


def get_system_prompt(name: str) -> str:
    """Load a system prompt (identity, safety, etc.)."""
    return load_prompt("system", name)


def get_agent_prompt(agent_name: str) -> str:
    """Load an agent's system prompt."""
    return load_prompt("agents", agent_name)


def get_fallback_responses() -> dict[str, Any]:
    """Load fallback response templates."""
    return load_json_prompt("fallbacks", "responses")


# ─────────────────────────────────────────────────────────────────────────────
# Response Configuration
# ─────────────────────────────────────────────────────────────────────────────


def get_mode_response_config(mode_name: str) -> dict[str, Any]:
    """Get response configuration for a mode.

    Returns dict with:
        - max_response_tokens: int
        - context_window_tokens: int
        - response_format: str
        - format_prompt: str (loaded from formats/)
    """
    config = load_json_prompt("modes", f"{mode_name}_config")

    if not config:
        # Return defaults based on mode
        defaults = {
            "observe": {
                "max_response_tokens": 128,
                "context_window_tokens": 512,
                "response_format": "minimal",
            },
            "sleep": {
                "max_response_tokens": 64,
                "context_window_tokens": 256,
                "response_format": "minimal",
            },
            "exploration": {
                "max_response_tokens": 256,
                "context_window_tokens": 1024,
                "response_format": "brief",
            },
            "live": {
                "max_response_tokens": 512,
                "context_window_tokens": 2048,
                "response_format": "conversational",
            },
            "active-assistance": {
                "max_response_tokens": 768,
                "context_window_tokens": 2048,
                "response_format": "detailed",
            },
            "reflection": {
                "max_response_tokens": 1024,
                "context_window_tokens": 3072,
                "response_format": "detailed",
            },
            "train": {
                "max_response_tokens": 512,
                "context_window_tokens": 2048,
                "response_format": "conversational",
            },
            "research": {
                "max_response_tokens": 2048,
                "context_window_tokens": 4096,
                "response_format": "academic",
            },
        }
        normalized = mode_name.lower().replace("_", "-")
        config = defaults.get(
            normalized,
            {
                "max_response_tokens": 512,
                "context_window_tokens": 2048,
                "response_format": "conversational",
            },
        )

    # Load the format prompt
    format_name = config.get("response_format", "conversational")
    config["format_prompt"] = get_format_prompt(format_name)

    return config


def build_mode_system_prompt(mode_name: str, include_format: bool = True) -> str:
    """Build a complete system prompt for a mode.

    Combines:
    - Base system identity
    - Mode-specific context prompt
    - Response format instructions (optional)
    """
    parts = []

    # Base identity
    identity = get_system_prompt("identity")
    if identity:
        parts.append(identity)

    # Mode context
    mode_prompt = get_mode_prompt(mode_name)
    if mode_prompt:
        parts.append(mode_prompt)

    # Response format
    if include_format:
        config = get_mode_response_config(mode_name)
        format_prompt = config.get("format_prompt", "")
        if format_prompt:
            parts.append(format_prompt)

    return "\n\n".join(parts)
