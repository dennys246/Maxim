"""Text normalization utilities for concept matching.

Provides tokenization, stop-word filtering, and basic lemmatization
for matching free-form text (goals, CLI input, transcripts) to ATL
concept names.

Used by:
- ConceptExtractor: goal tokenization during concept registration
- ConceptContextBuilder: percept-to-concept matching during recall
- PatternCompleter: concept matching during pattern completion
- since Session R (#1064/#1447, owner decision D3 2026-10-24): every memory store's text keying, on the write and
  read side alike -- the Hippocampus ranker and cue, the NAc keyword scan, the ATL token index, the enrichment
  query, the association index and the records' ``keywords()``. One standard, so a word the user said is keyed the
  same way wherever it is memorized and wherever it is recalled.

``normalize_tokens`` is FROZEN: ATL concept names are persisted from its output, so its known quirks ("used" ->
"us", "status" -> "statu", tokens under 3 characters dropped, so "Al" and "TV" are lost) are not fixed in place;
a change here orphans stored concepts and needs its own migration.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

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

    Edge-case aware about stem length and vowels, but it does NOT reach the base form of a silent-e verb: "placed"
    -> "plac", "moved" -> "mov", "used" -> "us", while "place" and "move" stay whole; a doubled consonant stays
    doubled before "-ed" ("stopped" -> "stopp"). Frozen with ``normalize_tokens`` (ATL names are persisted from
    it); the content keys converge these forms on top of it, in ``settle_key`` (#1620).
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


# Record fields that describe a capture rather than hold what was said or seen: never content (Session R, N6).
# ``bio_response`` is a past ThinkTool render of recalled memories; tokenizing it would re-surface every utterance it
# once recalled (an echo).
_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "source",
        "modality",
        "has_maxim",
        "detections_count",
        "_skill_name",
        "bio_response",
        "salience",
        "novelty",
        "spatial_source",
        "entity_source",
        "perceived_intensity",
    }
)


def text_values(value: Any, *, depth: int = 3) -> Iterator[str]:
    """The readable text in a stored value: strings, and the strings inside dicts and lists to ``depth``, with
    metadata keys skipped and every framed region removed (Session R).

    A frame in a record is always DERIVED text, a past render of something recalled (stored user input is never
    framed), so its body is not this record's content. A partial frame left by a cut is dropped the same way.
    """
    if isinstance(value, str):
        if "TOOL_OUTPUT" in value:
            from maxim.utils.content_safety import drop_unbalanced_frames, outside_tool_output

            value = drop_unbalanced_frames(outside_tool_output(value))[0]
        if value:
            yield value
    elif depth > 0 and isinstance(value, dict):
        for key, inner in value.items():
            if key not in _METADATA_KEYS:
                yield from text_values(inner, depth=depth - 1)
    elif depth > 0 and isinstance(value, (list, tuple)):
        for inner in value:
            yield from text_values(inner, depth=depth - 1)


def _kept(token: str) -> bool:
    """A convergence step may produce ``token`` only when the tokenizer keeps it and does not lemmatize it to a stop
    word ("whos" from "whose" would settle to "who"); ``settle_key`` then normalizes it to its fixed point, so "clos"
    from "close" settles to "clo", as "closed" and "closing" do."""
    again = normalize_tokens(token)
    return bool(again) and again[0] not in _STOP_WORDS


# The longest token the keys converge: an English word's inflections are what converge, and a longer token (a run in
# a tool's output) is no word; each step re-normalizes the token, so converging an n-character run costs O(n^2).
_CONVERGE_MAX_LEN = 40
# The longest token settled to its fixed point. A longer one is no word either (a run in a tool's output, "ed" * 16000):
# it settles as it did before #1620, in at most four passes, because the frozen ``_lemmatize`` strips one suffix per
# pass and a fixed point over n characters costs O(n^2). Idempotency is promised for a token up to this length.
_SETTLE_MAX_LEN = 64


def _converge(token: str) -> str:
    """One pass of the content keys' convergence over a normalized word (#1620): the regular inflections of one word
    reach one key, which the frozen ``_lemmatize`` does not do ("move", "moved", "moving", "moves" keyed as "move",
    "mov", "moving", "move"). Three steps, each taken only when the tokenizer keeps its result (``_kept``):

    - a three-letter stem the frozen ``-ing`` branch refused is cut: "moving" -> "mov", "asking" -> "ask";
    - a doubled final consonant other than "s" is single: "stopp" (from "stopped") -> "stop", "pull" -> "pul" (as
      "pulling" keys); an "-ss" word keeps it, since "bras" would settle to "bra";
    - a final "e" after a consonant or "u" is dropped: "move" -> "mov", "place" -> "plac", "issue" -> "issu", and
      "close" -> "clos", which ``settle_key`` settles to "clo" with "closed" and "closing".

    The keys are not words ("mov", "plac") and are never shown; they are what a write and a read both reach. The cost
    is a merge of words that differ only by a final "e" ("plan" and "plane", "made" and "mad", "care" and "car",
    "hope" and "hop", "done" and the "don" of "don't"): a recall that shares a key it should not, never one that
    misses a key it should share. Only a lowercase ASCII word of at most ``_CONVERGE_MAX_LEN`` letters converges:
    "1000" stays apart from "100", a hyphenated or marked-up token ("fast-suite", "sweep**") keeps its frozen key,
    and a long run of one letter is not re-normalized letter by letter. Not converged, because the frozen tokenizer
    drops or keeps them before this runs: "used" ("us", under 3 characters) against "use"/"using", "agreed"/"freed"
    (its "-eed" rule) against "agree"/"free", "accessing"/"passing" ("acc", "pas") against "access"/"pass", and
    irregular forms ("hid", "hidden", "slept", "hung"). Persisted ATL concept names (frozen ``normalize_tokens`` output such as "plac") reach
    the same key as the words they came from."""
    if len(token) > _CONVERGE_MAX_LEN or not (token.isascii() and token.isalpha()):
        return token
    if token.endswith("ing") and len(token) > 4:
        stem = token[:-3]
        if len(stem) >= 3 and _has_vowel(stem) and _kept(stem):
            token = stem
    if len(token) >= 4 and token[-1] == token[-2] and token[-1] not in "aeiouys" and _kept(token[:-1]):
        token = token[:-1]
    if len(token) >= 4 and token[-1] == "e" and token[-2] not in "aeioy" and _kept(token[:-1]):
        token = token[:-1]
    return token


def settle_key(token: str) -> str:
    """One WORD as a content key: normalized until it stops changing, then converged (``_converge``, #1620), the two
    repeated until neither changes it; a key the tokenizer would drop (under 3 characters, "used" -> "us") keeps its
    last form. Every change shortens the word, so the loops end at a fixed point, never at a cap (a capped loop left
    "hmmmmmmmmmmmm" keyed as "hmmmm", which settled again to "hmm"), except for a token longer than
    ``_SETTLE_MAX_LEN``, which is no word: it settles in at most four passes, as before #1620, and is not converged.
    Idempotent for a token up to that length: a keyed word keys to itself, so a store that is handed keys (NAc's
    keyword scan) settles them rather than re-tokenizing them."""
    if len(token) > _SETTLE_MAX_LEN:
        for _ in range(4):
            again = normalize_tokens(token)
            if not again or again == [token]:
                return token
            token = again[0]
        return token
    while True:
        while True:
            again = normalize_tokens(token)
            if not again or again == [token]:
                break
            token = again[0]
        converged = _converge(token)
        if converged == token:
            return token
        token = converged


def key_tokens(text: str) -> list[str]:
    """The content KEYS of ``text``: ``normalize_tokens``, each token settled to a fixed point, so keying is
    idempotent (design review S1/SF2: "settings" -> "setting" -> "set" when a keyed word was keyed again, which lost
    NAc and ATL matches). Every store keys text through this, on the write and the read side; ``normalize_tokens``
    itself stays frozen (ATL names are persisted from it)."""
    return [settle_key(t) for t in normalize_tokens(text)]


def content_tokens(*values: Any) -> set[str]:
    """``key_tokens`` over the readable text of ``values``: the one content key every store matches on."""
    tokens: set[str] = set()
    for value in values:
        for text in text_values(value):
            tokens.update(key_tokens(text))
    return tokens


# How much of what an episode heard is kept and rendered (#1444, #1137): ONE bound, used by enrichment's render and by
# compression's ``CompressedMemory.heard``, so a compressed record renders as its episode did. A reviewed constant: it
# holds a stated fact or a question; three recalled episodes stay well under the enrichment section's budget.
RECALLED_WORDS_MAX_CHARS = 240


def bounded_words(text: str, limit: int = RECALLED_WORDS_MAX_CHARS) -> str:
    """``text`` NFKC-normalized and whitespace-collapsed, THEN cut to ``limit`` (normalization first, so it cannot
    grow past the bound). A cut ends in "...", NFKC's own fixed point (a "…" expanded to "..." on the next pass and
    pushed a re-bound over the limit), so bounding is idempotent and a compressed record renders as its episode did."""
    import unicodedata

    text = " ".join(unicodedata.normalize("NFKC", text or "").split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def same_words(text: str) -> str:
    """``text`` as the same words compare: NFKC, case-folded, whitespace collapsed. Not the content keys, which are
    lossy ("My name is Al" and "My name?" key alike; "yes" keys to nothing)."""
    import unicodedata

    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


__all__ = [
    "normalize_tokens",
    "key_tokens",
    "settle_key",
    "text_values",
    "content_tokens",
    "same_words",
    "bounded_words",
    "RECALLED_WORDS_MAX_CHARS",
]
