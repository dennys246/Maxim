from __future__ import annotations


# ─────────────────────────────────────────────────────────────────────────────
# PLANNING MODE APPROVAL DETECTION
# ─────────────────────────────────────────────────────────────────────────────
# Keywords for approval detection (case-insensitive)
_APPROVAL_YES = frozenset(
    {
        "yes",
        "y",
        "yeah",
        "yep",
        "yup",
        "sure",
        "ok",
        "okay",
        "approve",
        "approved",
        "go",
        "go ahead",
        "do it",
        "proceed",
        "execute",
        "run",
        "confirm",
        "confirmed",
        "accept",
        "accepted",
        "sounds good",
        "looks good",
        "that works",
        "perfect",
        "great",
        "good",
        "fine",
        "correct",
        "right",
    }
)

_APPROVAL_NO = frozenset(
    {
        "no",
        "n",
        "nope",
        "nah",
        "stop",
        "cancel",
        "abort",
        "reject",
        "rejected",
        "deny",
        "denied",
        "don't",
        "dont",
        "do not",
        "never",
        "negative",
        "wrong",
        "incorrect",
        "bad",
        "not that",
    }
)


def detect_approval_intent(text: str) -> tuple[str, str | None]:
    """
    Detect user intent from text: approval, rejection, or modification.

    Returns:
        Tuple of (intent, modification_text):
        - ("approve", None) - user approved the plan
        - ("reject", None) - user rejected the plan
        - ("modify", "new instructions") - user wants to modify the plan
        - ("unknown", None) - could not determine intent
    """
    if not text:
        return ("unknown", None)

    text_lower = text.lower().strip()
    text_words = set(text_lower.split())

    # Check for exact match or word-level match for approval
    if text_lower in _APPROVAL_YES or text_words & _APPROVAL_YES:
        # But make sure it's not a modification (has other content)
        # Short responses like "yes" are approval, but "yes but change X" is modify
        if len(text_lower) < 20 or text_lower in _APPROVAL_YES:
            return ("approve", None)

    # Check for rejection
    if text_lower in _APPROVAL_NO or text_words & _APPROVAL_NO:
        if len(text_lower) < 20 or text_lower in _APPROVAL_NO:
            return ("reject", None)

    # Check for modification indicators
    modify_indicators = [
        "but",
        "instead",
        "change",
        "modify",
        "update",
        "different",
        "actually",
        "rather",
        "how about",
        "what if",
        "can you",
        "could you",
        "would you",
        "please",
        "also",
        "add",
        "remove",
    ]
    for indicator in modify_indicators:
        if indicator in text_lower:
            return ("modify", text)

    # If text is short and starts with approval/rejection word
    first_word = text_words.pop() if text_words else ""
    if first_word in _APPROVAL_YES:
        return ("approve", None)
    if first_word in _APPROVAL_NO:
        return ("reject", None)

    # Default: treat longer unknown text as modification request
    if len(text_lower) > 10:
        return ("modify", text)

    return ("unknown", None)
