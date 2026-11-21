from __future__ import annotations

import re


# ─────────────────────────────────────────────────────────────────────────────
# HUMAN YES/NO ANSWERS (plan approval, action confirmation, timeout retry)
# ─────────────────────────────────────────────────────────────────────────────
# #1185: these parse a HUMAN's answer to a prompt that gates an action, so they are a safety gate and fail
# closed. One vocabulary for every prompt (owner decision 2026-11-21, the issue's narrow sets): a reply is
# "yes" only when the WHOLE reply, lowercased with surrounding sentence punctuation and whitespace removed, is
# one of ``_YES``, and "no" only when it is one of ``_NO``. Everything else that carries an approval word, a
# negation, a question mark or a non-ASCII symbol (an emoji such as a thumbs-down) is "unknown" and the caller
# asks again: a refusal books NAc, so only a plain human "no" may produce one.
#
# The old parser tested a word-set intersection for YES words before NO words, so "stop, don't run it" (the
# token "run"), "not ok" and "no good" approved; "No." was unknown; and a mixed reply read ``set.pop()``, so its
# answer flipped with PYTHONHASHSEED.
_YES = frozenset({"yes", "y", "approve"})
_NO = frozenset({"no", "n", "reject", "cancel", "stop"})

# Characters stripped from both ends of a reply before the whole-reply match: sentence punctuation, quotes and
# whitespace. Anything else left over ("yes :(", "no 👎") keeps the reply from matching.
_EDGE_CHARS = " \t\r\n.,!;'\""

# Words that carry assent anywhere in a reply. They never approve on their own (only a whole ``_YES`` reply
# does); a reply containing one that is not a plain yes/no is ambiguous and re-asks rather than becoming a
# modification that drops the plan. Verbs naming the action ("run", "execute") are not assent.
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
        "agree",
        "agreed",
        "good",
        "fine",
        "great",
        "perfect",
        "correct",
        "right",
    }
)
# Words that negate or refuse anywhere in a reply (any "...n't" contraction counts too, see ``_is_negation``).
# A reply containing one that is not a plain "no" re-asks: "no problem", "why not", "don't stop" and "wait"
# must not book a refusal.
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
    "also",
    "add",
    "remove",
)


def _fold_quotes(text: str) -> str:
    return text.lower().replace("\u2019", "'").replace("\u2018", "'")


def classify_yes_no(text: str | None) -> str | None:
    """The one yes/no reading of a human's answer to a gating prompt: ``"yes"``, ``"no"`` or ``None``.

    ``"yes"`` only when the whole reply (lowercased, sentence punctuation, quotes and whitespace stripped from
    its ends, inner whitespace collapsed) is one of {yes, y, approve}; ``"no"`` only for {no, n, reject,
    cancel, stop}. Anything else, a question ("yes?") included, is ``None``.
    """
    if not text or "?" in text:
        return None
    reply = " ".join(_fold_quotes(text).strip(_EDGE_CHARS).split())
    if reply in _YES:
        return "yes"
    if reply in _NO:
        return "no"
    return None


def _words(text: str) -> list[str]:
    return [w.strip("'") for w in re.findall(r"[a-z0-9']+", _fold_quotes(text)) if w.strip("'")]


def _is_negation(word: str) -> bool:
    return word in _NEGATION_WORDS or word.endswith("n't")


def _has_symbol(text: str) -> bool:
    """A non-ASCII symbol (an emoji, a dingbat): it may carry a polarity this parser cannot read."""
    return any(not ch.isascii() and not ch.isalnum() and not ch.isspace() and ch not in "\u2018\u2019" for ch in text)


def detect_approval_intent(text: str | None) -> tuple[str, str | None]:
    """
    Detect a human's intent from their answer to a plan-approval prompt. Fails closed (#1185): only a plain
    yes approves, only a plain no rejects, and anything ambiguous re-asks.

    Returns:
        Tuple of (intent, modification_text):
        - ("approve", None) - ``classify_yes_no`` read "yes"
        - ("reject", None) - ``classify_yes_no`` read "no"
        - ("modify", text) - free text with no approval word, no negation, no question and no symbol
        - ("unknown", None) - anything else; the caller asks again
    """
    answer = classify_yes_no(text)
    if answer == "yes":
        return ("approve", None)
    if answer == "no":
        return ("reject", None)
    if not text or not text.strip() or "?" in text or _has_symbol(text):
        return ("unknown", None)
    words = _words(text)
    if any(w in _APPROVAL_WORDS or _is_negation(w) for w in words):
        return ("unknown", None)
    lowered = f" {' '.join(words)} "
    if len(text.strip()) > 10 or any(f" {indicator} " in lowered for indicator in _MODIFY_INDICATORS):
        return ("modify", text)
    return ("unknown", None)
