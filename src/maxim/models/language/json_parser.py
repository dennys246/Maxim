"""Robust JSON extraction from LLM text output.

LLMs — especially small/medium models (7B–14B) — produce JSON with common
defects: unescaped quotes inside string values, literal newlines, trailing
commas, truncated output, ChatML token leakage, and preamble commentary.

This module provides a multi-stage repair pipeline:

  Stage 0  Strip ChatML tokens, commentary, code fences → find first `{`
  Stage 1  Direct ``json.loads()`` (fast path for well-formed output)
  Stage 2  Sanitize control characters (\\n, \\t inside strings)
  Stage 3  ``json_repair`` library (handles unescaped quotes, trailing
           commas, missing brackets — covers ~95% of LLM defects)
  Stage 4  Structural repair (add missing braces/brackets for truncated output)

All JSON parsing of LLM output should go through ``_extract_json_object()``.
"""

from __future__ import annotations

import json
from typing import Any

from maxim.utils.logging import info, warn


# ─── Stage 2: control character sanitization ─────────────────────────────


def _sanitize_json_string(text: str) -> str:
    """Escape literal control characters inside JSON string values.

    LLMs sometimes output literal newlines/tabs inside JSON strings,
    which is invalid JSON.  This walks the text tracking string
    boundaries and replaces control chars with their escape sequences.
    """
    result: list[str] = []
    in_string = False
    escape_next = False

    for char in text:
        if escape_next:
            result.append(char)
            escape_next = False
            continue

        if char == "\\":
            result.append(char)
            escape_next = True
            continue

        if char == '"':
            in_string = not in_string
            result.append(char)
            continue

        if in_string:
            if char == "\n":
                result.append("\\n")
            elif char == "\r":
                result.append("\\r")
            elif char == "\t":
                result.append("\\t")
            elif ord(char) < 32:
                result.append(f"\\u{ord(char):04x}")
            else:
                result.append(char)
        else:
            result.append(char)

    return "".join(result)


# ─── Stage 3: json_repair library ────────────────────────────────────────


def _repair_with_library(text: str) -> str | None:
    """Use the ``json_repair`` library for robust repair.

    Handles unescaped quotes, trailing commas, missing brackets,
    single-quoted strings, and many other LLM-specific JSON defects.
    Returns the repaired string, or None if the library isn't available.
    """
    try:
        from json_repair import repair_json

        return repair_json(text, return_objects=False)  # type: ignore[return-value]
    except ImportError:
        return None
    except Exception:
        return None


# ─── Helpers ─────────────────────────────────────────────────────────────


def _find_first_json_object(text: str) -> str | None:
    """Find the first complete JSON object in text by matching braces.

    Handles cases where the LLM outputs extra content after the first
    valid JSON object.
    """
    if not text or not text.startswith("{"):
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[: i + 1]

    # No matching brace found — return the whole thing for repair
    return text


def _try_structural_repair(text: str) -> str:
    """Repair truncated JSON by closing unclosed strings/brackets/braces."""
    # Single-pass counting
    open_braces = close_braces = open_brackets = close_brackets = 0
    quote_count = 0
    prev_char = ""
    for char in text:
        if char == "{":
            open_braces += 1
        elif char == "}":
            close_braces += 1
        elif char == "[":
            open_brackets += 1
        elif char == "]":
            close_brackets += 1
        elif char == '"' and prev_char != "\\":
            quote_count += 1
        prev_char = char

    missing_braces = open_braces - close_braces
    missing_brackets = max(0, open_brackets - close_brackets)

    if missing_braces <= 0 and missing_brackets <= 0 and quote_count % 2 == 0:
        return text  # Nothing to repair

    repaired = text.rstrip().rstrip(",")

    # Close unclosed strings
    if quote_count % 2 == 1:
        repaired += '"'

    # Add closing brackets/braces
    repaired += "]" * missing_brackets + "}" * missing_braces
    info("JSON structural repair: added %d braces, %d brackets", missing_braces, missing_brackets)
    return repaired


# ─── Pre-processing ──────────────────────────────────────────────────────

# ChatML tokens that LLMs sometimes leak into output
_CHATML_TOKENS = ("<|im_end|>", "<|im_start|>", "<|assistant|>", "<|user|>", "<|system|>")

# Commentary preambles — LLM is explaining instead of outputting JSON
_COMMENTARY_PREFIXES = (
    "the provided",
    "the answer",
    "to answer",
    "i will",
    "i would",
    "here is",
    "here's",
    "let me",
    "this is",
    "the question",
    "the response",
    "as requested",
    "the json",
    "valid json",
    "the format",
    "to create",
    "we can",
    "you can",
)


def _preprocess(raw: str) -> str | None:
    """Strip non-JSON wrapping from LLM output.  Returns None if the
    text is pure commentary with no JSON object."""

    # Strip ChatML tokens
    for token in _CHATML_TOKENS:
        if token in raw:
            raw = raw.split(token)[0].strip()

    # Detect commentary
    raw_lower = raw.lower()
    for pattern in _COMMENTARY_PREFIXES:
        if raw_lower.startswith(pattern):
            return None

    # Strip code fences
    raw = raw.replace("```json", "```").replace("```JSON", "```")
    if "```" in raw:
        parts = raw.split("```")
        if len(parts) >= 3:
            raw = parts[1].strip()
        else:
            raw = raw.replace("```", "").strip()

    # Find first opening brace
    start = raw.find("{")
    if start < 0:
        return None

    return raw[start:]


# ─── Main entry point ────────────────────────────────────────────────────


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract a JSON dict from LLM text output.

    This is the SINGLE entry point for all LLM→JSON parsing in the
    codebase.  It implements a multi-stage repair pipeline, trying
    progressively more aggressive fixes until one succeeds or all fail.

    Returns the parsed dict, or None if extraction fails.
    """
    raw = str(text or "").strip()
    if not raw:
        return None

    # ── Stage 0: Pre-process (strip tokens, commentary, fences) ──────
    cleaned = _preprocess(raw)
    if cleaned is None:
        warn("JSON extraction failed: no JSON object found in LLM output")
        return None

    # Find the first complete JSON object by brace-matching
    json_candidate = _find_first_json_object(cleaned) or cleaned

    # ── Stage 1: Direct parse (fast path) ────────────────────────────
    try:
        obj = json.loads(json_candidate)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    # ── Stage 2: Sanitize control characters ─────────────────────────
    try:
        sanitized = _sanitize_json_string(json_candidate)
        obj = json.loads(sanitized)
        if isinstance(obj, dict):
            info("JSON parse succeeded after control-char sanitization")
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    # ── Stage 3: json_repair library (handles unescaped quotes, ──────
    #    trailing commas, single-quoted strings, and many more)
    repaired_str = _repair_with_library(json_candidate)
    if repaired_str is not None:
        try:
            obj = json.loads(repaired_str)
            if isinstance(obj, dict):
                info("JSON parse succeeded via json_repair library")
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    # ── Stage 4: Structural repair (truncated output) ────────────────
    structurally_repaired = _try_structural_repair(json_candidate)
    if structurally_repaired != json_candidate:
        # Try direct parse of structurally repaired text
        try:
            obj = json.loads(structurally_repaired)
            if isinstance(obj, dict):
                info("JSON parse succeeded after structural repair")
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

        # Try json_repair on structurally repaired text
        repaired_str = _repair_with_library(structurally_repaired)
        if repaired_str is not None:
            try:
                obj = json.loads(repaired_str)
                if isinstance(obj, dict):
                    info("JSON parse succeeded via json_repair after structural repair")
                    return obj
            except (json.JSONDecodeError, ValueError):
                pass

    # ── All stages failed ────────────────────────────────────────────
    warn(
        "JSON parse failed (all stages): len=%d | last_50: %s",
        len(json_candidate),
        json_candidate[-50:] if len(json_candidate) > 50 else json_candidate,
    )
    return None
