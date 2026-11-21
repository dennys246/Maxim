from __future__ import annotations

import re


# ─────────────────────────────────────────────────────────────────────────────
# PLANNING MODE APPROVAL DETECTION
# ─────────────────────────────────────────────────────────────────────────────
# #1185: this parses a HUMAN's answer to a plan-approval prompt, so it is a safety gate and fails closed.
# Approval needs the WHOLE reply (lowercased, punctuation stripped, whitespace collapsed) to be one of
# ``_APPROVE_PHRASES``; an approval word inside a longer reply never approves. A rejection or negation word
# anywhere means the reply is never approval: a short one without an approval word rejects, one that also
# carries an approval word is contradictory and comes out "unknown" (the caller re-asks). The old parser
# tested a word-set intersection for YES words first, so "stop, don't run it" (the token "run"), "not ok" and
# "no good" approved, and it read ``set.pop()``, so mixed replies flipped with PYTHONHASHSEED.
_APPROVE_PHRASES = frozenset(
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
        "lgtm",
        "go ahead",
        "do it",
        "proceed",
        "confirm",
        "confirmed",
        "accept",
        "accepted",
        "sounds good",
        "looks good",
        "yes please",
    }
)
_REJECT_PHRASES = frozenset(
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
        "do not",
        "never",
        "not now",
        "not yet",
    }
)
# Single words that carry approval anywhere in a reply: used only to detect a CONTRADICTION with a negation.
# They never approve on their own; only a whole ``_APPROVE_PHRASES`` match does. Verbs naming the action
# ("run", "execute") are not assent, so "don't run it" is a rejection, not a contradiction.
_APPROVAL_WORDS = frozenset(
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
        "lgtm",
        "go",
        "proceed",
        "confirm",
        "confirmed",
        "accept",
        "accepted",
        "good",
        "fine",
        "great",
        "perfect",
        "correct",
        "right",
    }
)
# Words that negate or refuse anywhere in a reply (any "...n't" contraction counts too, see ``_is_negation``).
_NEGATION_WORDS = frozenset(
    {
        "no",
        "n",
        "nope",
        "nah",
        "not",
        "stop",
        "cancel",
        "abort",
        "reject",
        "rejected",
        "deny",
        "denied",
        "dont",
        "never",
        "wait",
        "hold",
        "wrong",
        "incorrect",
        "bad",
    }
)
# A reply this short with a negation and no approval word is a rejection; a longer one is a modification
# request (the plan is dropped and the LLM sees the text), which also never executes the plan.
_SHORT_REJECT_MAX_WORDS = 4

_MODIFY_INDICATORS = (
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
)


def _normalize(text: str) -> list[str]:
    """Lowercase words with surrounding punctuation stripped; apostrophes kept inside a word ("don't")."""
    text = text.lower().replace("\u2019", "'").replace("\u2018", "'")
    words = []
    for raw in re.split(r"[^\w']+", text):
        word = raw.strip("'")
        if word:
            words.append(word)
    return words


def _is_negation(word: str) -> bool:
    return word in _NEGATION_WORDS or word.endswith("n't")


def detect_approval_intent(text: str) -> tuple[str, str | None]:
    """
    Detect a human's intent from their answer to a plan-approval prompt. Fails closed (#1185): ambiguous input
    is never "approve".

    Returns:
        Tuple of (intent, modification_text):
        - ("approve", None) - the whole reply is an approval phrase
        - ("reject", None) - the whole reply is a rejection phrase, or a short negated reply with no approval word
        - ("modify", text) - a longer reply asking for changes (the plan is dropped, never executed)
        - ("unknown", None) - could not determine intent, or the reply contradicts itself; the caller re-asks
    """
    words = _normalize(text or "")
    if not words:
        return ("unknown", None)
    phrase = " ".join(words)
    if phrase in _APPROVE_PHRASES:
        return ("approve", None)
    if phrase in _REJECT_PHRASES:
        return ("reject", None)

    has_negation = any(_is_negation(w) for w in words)
    has_approval_word = any(w in _APPROVAL_WORDS for w in words)
    if has_negation and has_approval_word:
        return ("unknown", None)  # contradictory ("ok but don't run it", "not ok", "yes no"): re-ask
    if has_negation:
        if len(words) <= _SHORT_REJECT_MAX_WORDS:
            return ("reject", None)
        return ("modify", text)

    padded = f" {phrase} "
    if any(f" {indicator} " in padded for indicator in _MODIFY_INDICATORS):
        return ("modify", text)
    if has_approval_word:
        return ("unknown", None)  # an approval word in a reply that is not an approval phrase: re-ask
    if len(phrase) > 10:
        return ("modify", text)
    return ("unknown", None)
