"""Response configuration loading for key and phrase mappings.

Loads and parses JSON configuration files for keyboard shortcuts and
voice command phrase responses.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Pattern

logger = logging.getLogger(__name__)


@dataclass
class KeyResponse:
    """Configuration for a keyboard shortcut response."""

    call: str
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    pause_training: bool = False


@dataclass
class PhraseResponse:
    """Configuration for a voice phrase response."""

    call: str
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    pause_training: bool = False
    wake_word: bool = False
    requires_agentic: bool = True
    cooldown_s: float = 2.0
    pattern: Pattern[str] | None = None
    normalized: str = ""


def _get_config_candidates(
    env_var: str,
    filename: str,
) -> list[str]:
    """Get list of candidate paths to search for a config file.

    Args:
        env_var: Environment variable name that may override path.
        filename: Base filename (e.g., "key_responses.json").

    Returns:
        List of paths to check, in priority order.
    """
    candidates: list[str] = []

    # Environment variable override
    env_path = str(os.getenv(env_var, "")).strip()
    if env_path:
        candidates.append(env_path)

    # Current working directory
    candidates.append(os.path.join(os.getcwd(), "data", "util", filename))
    candidates.append(os.path.join(os.getcwd(), filename))

    # Repository root (relative to this file)
    try:
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        candidates.append(os.path.join(repo_root, "data", "util", filename))
        candidates.append(os.path.join(repo_root, filename))
    except Exception:
        pass

    return candidates


def _compile_phrase_pattern(phrase: str) -> Pattern[str] | None:
    """Compile a phrase into a regex pattern for matching.

    Args:
        phrase: The phrase to compile.

    Returns:
        Compiled regex pattern, or None if invalid.
    """
    raw = str(phrase or "").strip()
    if not raw:
        return None

    escaped = re.escape(raw)
    pattern = escaped

    try:
        # Add word boundaries if phrase starts/ends with word characters
        if re.match(r"^\w", raw) and re.search(r"\w$", raw):
            pattern = rf"\b{escaped}\b"
        return re.compile(pattern, flags=re.IGNORECASE)
    except Exception:
        return None


def normalize_trigger_text(text: str) -> str:
    """Normalize text for phrase matching.

    Converts to lowercase, removes punctuation, collapses whitespace.

    Args:
        text: Raw text to normalize.

    Returns:
        Normalized text.
    """
    raw = str(text or "").strip().lower()
    if not raw:
        return ""

    # Remove non-word characters except spaces
    cleaned = re.sub(r"[^\w\s]", " ", raw, flags=re.UNICODE)
    return " ".join(cleaned.split())


def normalize_transcript_text(text: str) -> str:
    """Normalize transcript text with alias expansion.

    Like normalize_trigger_text, but also:
    - Removes standalone 's' tokens (common speech artifact)
    - Expands known aliases (e.g., "maximum" -> "maxim")

    Args:
        text: Raw transcript text.

    Returns:
        Normalized text with aliases expanded.
    """
    normalized = normalize_trigger_text(text)
    if not normalized:
        return ""

    raw_tokens = normalized.split()

    # Remove standalone 's' (common whisper artifact)
    tokens = [t for t in raw_tokens if t and t != "s"]

    # Known aliases
    aliases = {
        "maximum": "maxim",
        "maximums": "maxim",
        "maxims": "maxim",
    }

    changed = tokens != raw_tokens
    for idx, token in enumerate(tokens):
        replacement = aliases.get(token)
        if replacement:
            tokens[idx] = replacement
            changed = True

    return " ".join(tokens) if changed else normalized


def load_key_responses(
    log: logging.Logger | None = None,
) -> dict[str, dict[str, Any]]:
    """Load keyboard shortcut responses from JSON configuration.

    Searches for configuration file in standard locations, falling back
    to built-in defaults if not found.

    Args:
        log: Logger for warnings.

    Returns:
        Dict mapping key characters to response specifications.
    """
    log = log or logger

    # Default responses
    default: dict[str, dict[str, Any]] = {
        "c": {"call": "center_vision", "pause_training": True},
        "u": {"call": "mark_trainable_moment"},
        **{str(i): {"call": "label_outcome", "args": [i]} for i in range(10)},
    }

    candidates = _get_config_candidates("MAXIM_KEY_RESPONSES", "key_responses.json")

    raw = None
    for path in candidates:
        if path and os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as fp:
                    raw = json.load(fp)
                break
            except Exception as e:
                log.warning("Failed to load key responses from '%s': %s", path, e)
                return default

    if not isinstance(raw, dict):
        return default

    parsed: dict[str, dict[str, Any]] = {}
    for key, spec in raw.items():
        if not isinstance(key, str) or not key:
            continue

        if isinstance(spec, str):
            parsed[key] = {"call": spec}
            continue

        if not isinstance(spec, dict):
            continue

        call = spec.get("call") or spec.get("method")
        if not isinstance(call, str) or not call:
            continue

        parsed[key] = {
            "call": call,
            "args": spec.get("args") if isinstance(spec.get("args"), list) else [],
            "kwargs": spec.get("kwargs") if isinstance(spec.get("kwargs"), dict) else {},
            "pause_training": bool(spec.get("pause_training", False)),
        }

    merged = dict(default)
    merged.update(parsed)
    return merged


def load_phrase_responses(
    log: logging.Logger | None = None,
) -> dict[str, dict[str, Any]]:
    """Load voice phrase responses from JSON configuration.

    Searches for configuration file in standard locations, falling back
    to built-in defaults if not found. Compiles regex patterns for
    efficient phrase matching.

    Args:
        log: Logger for warnings.

    Returns:
        Dict mapping phrases to response specifications.
    """
    log = log or logger

    # Default responses
    default: dict[str, dict[str, Any]] = {
        # Shutdown commands
        "maxim shutdown": {
            "call": "request_shutdown",
            "requires_agentic": False,
            "cooldown_s": 2.0,
        },
        "shutdown maxim": {
            "call": "request_shutdown",
            "requires_agentic": False,
            "cooldown_s": 2.0,
        },
        # Sleep processing state commands
        "maxim sleep": {
            "call": "request_sleep",
            "requires_agentic": False,
            "cooldown_s": 2.0,
        },
        "sleep maxim": {
            "call": "request_sleep",
            "requires_agentic": False,
            "cooldown_s": 2.0,
        },
        "maxim nap": {
            "call": "request_sleep",
            "requires_agentic": False,
            "cooldown_s": 2.0,
        },
        "maxim rest": {
            "call": "request_sleep",
            "requires_agentic": False,
            "cooldown_s": 2.0,
        },
        # Wake from sleep
        "maxim wake up": {
            "call": "request_wake",
            "requires_agentic": False,
            "cooldown_s": 2.0,
        },
        "wake up maxim": {
            "call": "request_wake",
            "requires_agentic": False,
            "cooldown_s": 2.0,
        },
        "maxim wake": {
            "call": "request_wake",
            "requires_agentic": False,
            "cooldown_s": 2.0,
        },
        "wake maxim": {
            "call": "request_wake",
            "requires_agentic": False,
            "cooldown_s": 2.0,
        },
        # Strategy commands
        "maxim observe": {
            "call": "request_strategy_observe",
            "requires_agentic": False,
            "cooldown_s": 2.0,
        },
        "observe maxim": {
            "call": "request_strategy_observe",
            "requires_agentic": False,
            "cooldown_s": 2.0,
        },
        "maxim watch": {
            "call": "request_strategy_observe",
            "requires_agentic": False,
            "cooldown_s": 2.0,
        },
        "maxim explore": {
            "call": "request_strategy_explore",
            "requires_agentic": False,
            "cooldown_s": 2.0,
        },
        "explore maxim": {
            "call": "request_strategy_explore",
            "requires_agentic": False,
            "cooldown_s": 2.0,
        },
        "maxim research": {
            "call": "request_strategy_research",
            "requires_agentic": False,
            "cooldown_s": 2.0,
        },
        "research maxim": {
            "call": "request_strategy_research",
            "requires_agentic": False,
            "cooldown_s": 2.0,
        },
        "maxim assist": {
            "call": "request_strategy_assist",
            "requires_agentic": False,
            "cooldown_s": 2.0,
        },
        "maxim help": {
            "call": "request_strategy_assist",
            "requires_agentic": False,
            "cooldown_s": 2.0,
        },
        "maxim reflect": {
            "call": "request_strategy_reflect",
            "requires_agentic": False,
            "cooldown_s": 2.0,
        },
        "maxim reflection": {
            "call": "request_strategy_reflect",
            "requires_agentic": False,
            "cooldown_s": 2.0,
        },
        "maxim learn": {
            "call": "request_strategy_learn",
            "requires_agentic": False,
            "cooldown_s": 2.0,
        },
        "maxim train": {
            "call": "request_strategy_learn",
            "requires_agentic": False,
            "cooldown_s": 2.0,
        },
        # Mode commands
        "maxim passive": {
            "call": "request_mode_passive",
            "requires_agentic": False,
            "cooldown_s": 2.0,
        },
        "maxim active": {
            "call": "request_mode_active",
            "requires_agentic": False,
            "cooldown_s": 2.0,
        },
        "maxim singularity": {
            "call": "request_mode_singularity",
            "requires_agentic": False,
            "cooldown_s": 2.0,
        },
        # Wake words
        "maxim": {"call": "wake_up_agentic", "wake_word": True, "cooldown_s": 2.0},
        "reachy": {"call": "wake_up_agentic", "wake_word": True, "cooldown_s": 2.0},
        # Other commands
        "center": {
            "call": "center_vision",
            "pause_training": True,
            "requires_agentic": True,
            "cooldown_s": 2.0,
        },
    }

    candidates = _get_config_candidates(
        "MAXIM_PHRASE_RESPONSES", "phrase_responses.json"
    )

    raw = None
    for path in candidates:
        if path and os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as fp:
                    raw = json.load(fp)
                break
            except Exception as e:
                log.warning("Failed to load phrase responses from '%s': %s", path, e)
                return _finalize_phrase_responses(default)

    if not isinstance(raw, dict):
        return _finalize_phrase_responses(default)

    parsed: dict[str, dict[str, Any]] = {}
    for phrase, spec in raw.items():
        if not isinstance(phrase, str) or not phrase.strip():
            continue
        phrase = phrase.strip()

        if isinstance(spec, str):
            spec = {"call": spec}
        if not isinstance(spec, dict):
            continue

        call = spec.get("call") or spec.get("method")
        if not isinstance(call, str) or not call:
            continue

        wake_word = bool(spec.get("wake_word", False))
        requires_agentic = bool(spec.get("requires_agentic", not wake_word))

        cooldown_s = spec.get("cooldown_s")
        try:
            cooldown_s = float(cooldown_s) if cooldown_s is not None else 2.0
        except Exception:
            cooldown_s = 2.0
        if float(cooldown_s) <= 0:
            cooldown_s = 2.0

        parsed[phrase] = {
            "call": call,
            "args": spec.get("args") if isinstance(spec.get("args"), list) else [],
            "kwargs": spec.get("kwargs") if isinstance(spec.get("kwargs"), dict) else {},
            "pause_training": bool(spec.get("pause_training", False)),
            "wake_word": wake_word,
            "requires_agentic": requires_agentic,
            "cooldown_s": float(cooldown_s),
        }

    merged = dict(default)
    merged.update(parsed)
    return _finalize_phrase_responses(merged)


def _finalize_phrase_responses(
    responses: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Add compiled patterns and normalized text to phrase responses.

    Args:
        responses: Raw phrase responses dict.

    Returns:
        Responses with _pattern and _normalized fields added.
    """
    for phrase, spec in responses.items():
        if isinstance(spec, dict):
            if "_pattern" not in spec:
                spec["_pattern"] = _compile_phrase_pattern(phrase)
            if "_normalized" not in spec:
                spec["_normalized"] = normalize_trigger_text(phrase)
    return responses


__all__ = [
    "KeyResponse",
    "PhraseResponse",
    "load_key_responses",
    "load_phrase_responses",
    "normalize_trigger_text",
    "normalize_transcript_text",
]
