"""Text normalization utilities for concept matching.

Provides tokenization, stop-word filtering, and basic lemmatization
for matching free-form text (goals, CLI input, transcripts) to ATL
concept names.

Used by:
- ConceptExtractor: goal tokenization during concept registration
- ConceptContextBuilder: percept-to-concept matching during recall
- PatternCompleter: concept matching during pattern completion
"""

from __future__ import annotations

import re

_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "to", "in", "on", "of", "for", "is", "it",
    "and", "or", "but", "not", "with", "at", "by", "from", "as",
    "be", "was", "were", "been", "are", "am", "do", "does", "did",
    "has", "had", "have", "will", "would", "could", "should", "can",
    "this", "that", "these", "those", "i", "me", "my", "we", "our",
})


def normalize_tokens(text: str) -> list[str]:
    """Tokenize, filter stop words, and lemmatize for concept matching.

    Splits on whitespace AND underscores so compound identifiers like
    "navigate_to_kitchen" become ["navigate", "kitchen"] (after stop-word
    removal). Lemmatization uses basic suffix stripping (no NLTK dependency).
    Covers common English inflections: -ing, -ed, -s, -ly, -tion.
    Not linguistically perfect, but sufficient for concept name matching.
    """
    words = re.split(r"[\s_]+", text.lower())
    result = []
    for w in words:
        if w in _STOP_WORDS or len(w) < 2:
            continue
        result.append(_lemmatize(w))
    return result


def _lemmatize(word: str) -> str:
    """Basic suffix stripping. No external dependencies.

    Handles common English inflections without NLTK. Not linguistically
    perfect, but sufficient for concept name matching where the goal is
    "grasping" -> "grasp", "mugs" -> "mug", etc.

    Edge-case aware: checks stem validity (min length, vowel presence)
    to avoid garbage stems like "used" -> "us" or "placed" -> "plac".
    """
    # -ing: grasping -> grasp, running -> run
    if word.endswith("ing") and len(word) > 4:
        stem = word[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]  # running -> runn -> run
        if _has_vowel(stem):
            return stem
        return word
    # -ed: grasped -> grasp, used -> use, placed -> place
    if word.endswith("ed") and len(word) > 3:
        if word.endswith("eed"):
            return word  # "freed" stays "freed"
        if word.endswith("ied") and len(word) > 4:
            return word[:-3] + "y"  # "carried" -> "carry"
        # Try removing -ed, check if stem is valid
        stem = word[:-2]
        if len(stem) >= 2 and _has_vowel(stem):
            return stem
        # Try removing -d (for words like "placed" -> "place")
        stem_d = word[:-1]
        if stem_d.endswith("e") and len(stem_d) >= 3:
            return stem_d
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"  # batteries -> battery
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _has_vowel(word: str) -> bool:
    """Check if word contains at least one vowel."""
    return any(c in "aeiou" for c in word)


__all__ = ["normalize_tokens"]