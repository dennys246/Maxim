"""File reference detection from user input.

Detects file names, path patterns, and user intent (create vs modify)
from natural-language input. Used by the speculative pre-fetcher to
decide which files to discover and read ahead of the LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Patterns to detect file references in user input (pre-compiled for performance)
FILE_PATTERNS = [
    # Explicit file extensions
    re.compile(r"(\b\w+\.py\b)", re.IGNORECASE),  # Python files
    re.compile(r"(\b\w+\.js\b)", re.IGNORECASE),  # JavaScript
    re.compile(r"(\b\w+\.ts\b)", re.IGNORECASE),  # TypeScript
    re.compile(r"(\b\w+\.json\b)", re.IGNORECASE),  # JSON
    re.compile(r"(\b\w+\.yaml\b)", re.IGNORECASE),  # YAML
    re.compile(r"(\b\w+\.yml\b)", re.IGNORECASE),  # YAML alt
    re.compile(r"(\b\w+\.md\b)", re.IGNORECASE),  # Markdown
    re.compile(r"(\b\w+\.txt\b)", re.IGNORECASE),  # Text
    re.compile(r"(\b\w+\.sh\b)", re.IGNORECASE),  # Shell scripts
    re.compile(r"(\b\w+\.css\b)", re.IGNORECASE),  # CSS
    re.compile(r"(\b\w+\.html\b)", re.IGNORECASE),  # HTML
    # Path patterns
    re.compile(r"(src/[\w/]+\.?\w*)", re.IGNORECASE),  # src/ paths
    re.compile(r"(lib/[\w/]+\.?\w*)", re.IGNORECASE),  # lib/ paths
    re.compile(r"(tests?/[\w/]+\.?\w*)", re.IGNORECASE),  # test(s)/ paths
    re.compile(r"(config/[\w/]+\.?\w*)", re.IGNORECASE),  # config/ paths
]

# Keywords indicating file modification intent
MODIFY_KEYWORDS = frozenset({
    "update", "modify", "change", "edit", "fix", "add to", "append",
    "remove from", "delete from", "refactor", "improve", "enhance",
})

# Keywords indicating new file creation
CREATE_KEYWORDS = frozenset({
    "create", "new", "make", "write", "generate", "build",
})


@dataclass
class FileReference:
    """A detected file reference from user input."""

    pattern: str  # The matched pattern (e.g., "hello.py")
    is_path: bool  # Whether it looks like a full path
    intent: str  # "modify", "create", or "unknown"
    confidence: float  # How confident we are in this detection


def detect_file_references(text: str) -> list[FileReference]:
    """Detect file references in user input.

    Returns list of FileReference objects sorted by confidence.
    """
    refs: list[FileReference] = []
    text_lower = text.lower()

    # Detect modification vs creation intent
    has_modify_intent = any(kw in text_lower for kw in MODIFY_KEYWORDS)
    has_create_intent = any(kw in text_lower for kw in CREATE_KEYWORDS)

    if has_create_intent and not has_modify_intent:
        intent = "create"
    elif has_modify_intent:
        intent = "modify"
    else:
        intent = "unknown"

    seen = set()
    for compiled_pattern in FILE_PATTERNS:
        for match in compiled_pattern.finditer(text):  # Use pre-compiled pattern
            file_ref = match.group(1)
            if file_ref.lower() in seen:
                continue
            seen.add(file_ref.lower())

            is_path = "/" in file_ref
            confidence = 0.9 if is_path else 0.7

            refs.append(FileReference(
                pattern=file_ref,
                is_path=is_path,
                intent=intent,
                confidence=confidence,
            ))

    # Sort by confidence descending
    refs.sort(key=lambda r: -r.confidence)
    return refs


def detect_file_intent(text: str) -> str:
    """Detect whether user wants to create or modify a file.

    Returns: "create", "modify", or "unknown"
    """
    text_lower = text.lower()

    has_modify = any(kw in text_lower for kw in MODIFY_KEYWORDS)
    has_create = any(kw in text_lower for kw in CREATE_KEYWORDS)

    if has_create and not has_modify:
        return "create"
    elif has_modify:
        return "modify"
    return "unknown"
