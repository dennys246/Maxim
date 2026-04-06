from __future__ import annotations

import json
from typing import Any

from maxim.utils.logging import info, warn


def _sanitize_json_string(text: str) -> str:
    """Escape control characters inside JSON strings.

    LLMs sometimes output literal newlines/tabs inside JSON string values,
    which is invalid JSON. This function escapes them properly.
    """
    result = []
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
            # Escape control characters inside strings
            if char == "\n":
                result.append("\\n")
            elif char == "\r":
                result.append("\\r")
            elif char == "\t":
                result.append("\\t")
            elif ord(char) < 32:
                # Other control characters - escape as unicode
                result.append(f"\\u{ord(char):04x}")
            else:
                result.append(char)
        else:
            result.append(char)

    return "".join(result)


def _repair_unescaped_quotes(text: str) -> str:
    """Attempt to escape unescaped double quotes inside JSON string values.

    When an LLM embeds narrative text containing dialogue (e.g., She said
    "hello") into a JSON string, the inner quotes break parsing.  This
    function heuristically detects quotes that appear inside a string value
    (not at a structural boundary) and escapes them.

    Strategy: walk the text tracking JSON structural context.  A quote that
    appears inside a string value and is followed by content that doesn't
    look like a JSON key or structural token is treated as a literal that
    needs escaping.
    """
    import re

    # Quick check: if it parses, no repair needed
    try:
        json.loads(text)
        return text
    except Exception:
        pass

    # Find all string value positions: after `"key":` patterns, the next
    # quote opens a string value.  We re-escape any unescaped quotes
    # inside that value by looking for the *correct* closing quote
    # (one followed by , or } or ] or whitespace then one of those).
    # This is a heuristic — it won't handle all edge cases but covers the
    # common case of narrative dialogue embedded in tool params.
    structural_close = re.compile(r'"\s*[,}\]]')

    result = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\":
            # Escaped char — pass through both characters
            result.append(text[i : i + 2])
            i += 2
            continue
        if ch == '"':
            # Opening quote of a string — find the structural close
            result.append('"')
            i += 1
            # Scan for the closing quote (one followed by structural char)
            while i < n:
                ic = text[i]
                if ic == "\\":
                    result.append(text[i : i + 2])
                    i += 2
                    continue
                if ic == '"':
                    # Is this the structural close?
                    rest = text[i:]
                    if structural_close.match(rest):
                        result.append('"')
                        i += 1
                        break
                    # Not structural — escape it
                    result.append('\\"')
                    i += 1
                    continue
                result.append(ic)
                i += 1
            continue
        result.append(ch)
        i += 1

    return "".join(result)


def _find_first_json_object(text: str) -> str | None:
    """Find the first complete JSON object in text by matching braces.

    This handles cases where the LLM outputs multiple JSON objects or
    extra content after the first valid JSON.
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
                # Found the matching closing brace
                return text[: i + 1]

    # No matching brace found - return the whole thing for repair attempt
    return text


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None

    # Strip ChatML tokens that LLMs sometimes include in output
    chatml_tokens = ["<|im_end|>", "<|im_start|>", "<|assistant|>", "<|user|>", "<|system|>"]
    for token in chatml_tokens:
        if token in raw:
            # Take only content before the first ChatML token
            raw = raw.split(token)[0].strip()

    # Detect commentary patterns - LLM is explaining instead of outputting JSON
    commentary_patterns = [
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
    ]
    raw_lower = raw.lower()
    for pattern in commentary_patterns:
        if raw_lower.startswith(pattern):
            # LLM is commenting, not outputting JSON
            return None

    # Extract FIRST code block only (not all of them)
    raw = raw.replace("```json", "```").replace("```JSON", "```")
    if "```" in raw:
        parts = raw.split("```")
        if len(parts) >= 3:
            # Take only the FIRST code block content
            raw = parts[1].strip()
        else:
            raw = raw.replace("```", "").strip()

    start = raw.find("{")
    if start < 0:
        warn("JSON extraction failed: no opening brace found")
        return None

    # Find the MATCHING closing brace for the first opening brace
    # This handles cases where there's extra content after the first JSON object
    json_candidate = _find_first_json_object(raw[start:])

    if json_candidate is None:
        # No opening brace or completely malformed
        warn("JSON extraction failed: couldn't find JSON object")
        return None

    # Check if we got a complete JSON (ends with })
    if not json_candidate.rstrip().endswith("}"):
        # Truncated - try to repair by adding missing braces
        # Single-pass counting (6x faster than 6 separate .count() calls)
        open_braces = close_braces = open_brackets = close_brackets = 0
        quote_count = 0
        prev_char = ""
        for char in json_candidate:
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

        # Strip trailing incomplete content
        json_candidate = json_candidate.rstrip().rstrip(",")

        # Close unclosed strings
        if quote_count % 2 == 1:
            json_candidate += '"'

        # Add closing brackets/braces
        json_candidate += "]" * missing_brackets + "}" * missing_braces
        info("JSON repair: added %d braces, %d brackets", missing_braces, missing_brackets)

    try:
        obj = json.loads(json_candidate)
    except json.JSONDecodeError as e:
        # Try sanitizing (control chars, unescaped quotes) on any parse failure
        try:
            sanitized = _sanitize_json_string(json_candidate)
            obj = json.loads(sanitized)
            info("JSON parse succeeded after sanitizing")
        except Exception:
            # Last resort: try repairing unescaped quotes inside string values
            try:
                repaired = _repair_unescaped_quotes(json_candidate)
                obj = json.loads(repaired)
                info("JSON parse succeeded after quote repair")
            except Exception:
                warn(
                    "JSON parse failed: %s | len=%d | last_50: %s",
                    str(e)[:80],
                    len(json_candidate),
                    json_candidate[-50:] if len(json_candidate) > 50 else json_candidate,
                )
                return None
    except Exception as e:
        warn(
            "JSON parse failed: %s | len=%d | last_50: %s",
            str(e)[:80],
            len(json_candidate),
            json_candidate[-50:] if len(json_candidate) > 50 else json_candidate,
        )
        return None

    return obj if isinstance(obj, dict) else None
