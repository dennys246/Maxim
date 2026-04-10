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

_STOP_WORDS: frozenset[str] = frozenset(
    {
        # Articles and core determiners
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        # Prepositions
        "to",
        "in",
        "on",
        "of",
        "for",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "onto",
        "over",
        "under",
        "after",
        "before",
        "during",
        "through",
        "about",
        "against",
        # Conjunctions
        "and",
        "or",
        "but",
        "not",
        "so",
        "if",
        "then",
        "than",
        "because",
        "while",
        "although",
        "though",
        "unless",
        # Auxiliary verbs and copulas
        "is",
        "it",
        "its",
        "be",
        "was",
        "were",
        "been",
        "being",
        "are",
        "am",
        "do",
        "does",
        "did",
        "doing",
        "has",
        "had",
        "have",
        "having",
        "will",
        "would",
        "could",
        "should",
        "shall",
        "can",
        "may",
        "might",
        "must",
        # Pronouns
        "i",
        "me",
        "my",
        "mine",
        "myself",
        "we",
        "us",
        "our",
        "ours",
        "you",
        "your",
        "yours",
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
        "they",
        "them",
        "their",
        "theirs",
        # Common filler and hedge words that are rarely meaningful concepts
        "best",
        "good",
        "bad",
        "now",
        "here",
        "there",
        "when",
        "where",
        "what",
        "who",
        "why",
        "how",
        "some",
        "any",
        "all",
        "no",
        "yes",
        "also",
        "only",
        "just",
        "very",
        "too",
        "one",
        "two",
        "such",
        "still",
        "even",
        "much",
        "many",
        "more",
        "most",
        "less",
        "few",
        "both",
        "other",
        "another",
    }
)

# Leading/trailing punctuation that sneaks in from free-form text
# (e.g. "plan," "action." "it'"). Strip from both ends after lowercasing.
_PUNCT_STRIP = ".,;:!?\"'`()[]{}<>—–-"


def normalize_tokens(text: str) -> list[str]:
    """Tokenize, filter stop words, and lemmatize for concept matching.

    Splits on whitespace, underscores, and apostrophes so compound
    identifiers like "navigate_to_kitchen" become ["navigate", "kitchen"]
    and contractions like "it's" become ["it", "s"] (both filtered by the
    stop-word / length-2 rules). Also strips leading/trailing punctuation
    so "plan," and "action." normalize to "plan" and "action" before the
    stop-word filter runs. Lemmatization uses basic suffix stripping (no
    NLTK dependency) and is conservative about stem length to avoid
    garbage stems like "tak" from "taking".
    """
    words = re.split(r"[\s_'\u2019]+", text.lower())
    result: list[str] = []
    for raw in words:
        w = raw.strip(_PUNCT_STRIP)
        if not w or w in _STOP_WORDS or len(w) < 3:
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
    # -ing: grasping -> grasp, running -> run.
    # Split the two cases so we can require a longer stem for non-doubled
    # strips (which are prone to garbage stems like "taking" -> "tak")
    # while still allowing legitimate short stems from doubled-consonant
    # strips (running -> runn -> run is correct at 3 chars).
    if word.endswith("ing") and len(word) > 4:
        stem = word[:-3]
        doubled = len(stem) >= 2 and stem[-1] == stem[-2]
        if doubled:
            stem = stem[:-1]  # running -> runn -> run
            if len(stem) >= 3 and _has_vowel(stem):
                return stem
        else:
            # Non-doubled: require len >= 4 so "taking" stays "taking"
            # rather than degenerating into "tak". A real stemmer would
            # know "taking" -> "take", but we prefer the original word
            # intact over producing non-words.
            if len(stem) >= 4 and _has_vowel(stem):
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
