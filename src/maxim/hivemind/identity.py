"""Identity-bearing concept detection (Hivemind shareability, PR C).

v1_refinement.md §B5 PR C. Simplified port from the archived Mother
plan (``docs/plans/archive/mother_maxim_plan.md`` "Identity-bearing
concept detection" section). Detects whether a textual label refers
to a specific named entity (a person, a place, a unique role-with-name)
versus a generic shareable concept.

Used by the bundle composer (PR D) to quarantine identity-bearing
patterns from substrate bundles — those stay local to the contributor's
Maxim. False positives are conservative (good: pattern stays local
unnecessarily). False negatives leak (bad: identity-bearing pattern
shipped out). The threshold defaults err on the side of false positives.

The Mother plan's full identity map walked the ATL concept graph and
SEM entity registry to enumerate the user's entire entity space. That
approach is ~200 LOC + requires the full bio-stack at scrub time. This
port is the substrate-only heuristic: scan a label string for
identity-bearing surface features, no graph traversal required. Pairs
with PR D's per-domain filter (the reserved ``_identity`` domain marker
quarantines EC nodes; this module quarantines NAc links by event
signature).

A future 1.1+ revision can layer ATL-graph traversal on top via an
optional ``identity_map: set[str]`` parameter; today the heuristic
runs without bio-system context.

Known limitations (pre-merge review fold, IMPORTANT findings)
-------------------------------------------------------------

1. **Game-substrate over-flagging.** At ``threshold=1`` on a D&D /
   fantasy-substrate Oasis, every sentence-cased monster token
   (``"Dragon roared"``, ``"Goblin fled"``) trips the proper-noun
   signal even though these are generic creature templates the bundle
   composer probably wants to share. PR D's bundle composer should
   consider ``threshold=2`` (or higher) for the bundle-composition
   default in game-substrate domains; ``threshold=1`` is the right
   default for general-purpose Oases. NOT addressed in 1.0 by an
   internal allow-list — a curated "this is generic" lexicon would
   need a community-curated set, deferred to 1.1+.

2. **English-skew.** The default keyword set is English-only
   (honorifics, ownership markers). Non-English contributors should
   pass ``identity_keywords=`` overrides until a per-language keyword
   registry ships in 1.1+. The heuristic's proper-noun signal still
   works on non-English text whose script uses casing (Latin, Cyrillic,
   Greek); abugidas + abjads (Arabic, Hebrew, Devanagari) lack the
   casing signal entirely.

3. **Bio-system context unused.** Unlike the Mother plan's full ATL/SEM
   walk, this port has no signal that ``"Aldric"`` is a known NPC vs a
   real-world person. The reserved ``identity_map: set[str]`` parameter
   slot is documented for 1.1+ extension.
"""

from __future__ import annotations

# Honorifics + titles that signal "this token is a personal name marker."
# Conservative: small, English-skewed list. Operators / community
# Oases can extend via the ``identity_keywords`` parameter.
_DEFAULT_IDENTITY_KEYWORDS: frozenset[str] = frozenset(
    {
        # Honorifics
        "mr",
        "mrs",
        "ms",
        "miss",
        "dr",
        "prof",
        "professor",
        "sir",
        "lady",
        "lord",
        "captain",
        "sergeant",
        "officer",
        # Pronoun-like / ownership markers
        "my",
        "myself",
        "user",
        "owner",
        # Pattern markers that often introduce a name
        "named",
        "called",
        "aka",
    }
)

# Common English stop-words to exclude from the proper-noun count.
# Capitalized stop-words ("The", "A", "And") would otherwise inflate
# the count for a sentence-cased event signature.
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "is",
        "in",
        "on",
        "at",
        "to",
        "of",
        "for",
        "by",
        "from",
        "with",
        "as",
        "this",
        "that",
        "it",
        "if",
        "do",
        "did",
        "be",
        "was",
        "were",
        "had",
        "has",
        "have",
    }
)

# Reserved marker used by PR D to scope EC nodes the user wants quarantined.
# A node with ``domain=IDENTITY_DOMAIN_MARKER`` is excluded from every
# substrate bundle. Underscore-prefixed by convention so it sits in the
# reserved namespace (cf. ``_consensus`` in ``merge.py``).
IDENTITY_DOMAIN_MARKER: str = "_identity"


def is_identity_bearing(
    text: str,
    *,
    threshold: int = 1,
    identity_keywords: frozenset[str] | None = None,
) -> bool:
    """Detect whether ``text`` looks identity-bearing.

    Heuristic:

    1. Tokenize on whitespace.
    2. For each token, strip surrounding punctuation. Count two signals:

       - **Proper-noun-shape**: capitalized initial letter, length > 2,
         lowercase form NOT in the stop-word list.
       - **Identity-keyword**: lowercase form in ``identity_keywords``
         (or :data:`_DEFAULT_IDENTITY_KEYWORDS` if not supplied).

    3. Return ``True`` if either signal count meets ``threshold``.

    Parameters
    ----------
    text
        The label to evaluate. NAc event_signatures, tool descriptions,
        ATL concept names, or free-form percept text are typical inputs.
        Empty / whitespace returns ``False``.
    threshold
        Minimum count for either signal to flag identity-bearing.
        Default ``1`` — a single proper noun OR a single identity
        keyword is enough. Raise to ``2`` for a stricter filter.
    identity_keywords
        Override the default keyword set (e.g., to add domain-specific
        terms or community-curated lexicons). Passing the empty set
        disables keyword matching, leaving only the proper-noun signal.

    Examples
    --------
    >>> is_identity_bearing("opened door")
    False
    >>> is_identity_bearing("met Dave at the market")
    True
    >>> is_identity_bearing("greeted my friend")
    True
    >>> is_identity_bearing("attacked the dragon")
    False
    """
    if not text or not text.strip():
        return False
    keywords = identity_keywords if identity_keywords is not None else _DEFAULT_IDENTITY_KEYWORDS

    proper_noun_count = 0
    keyword_count = 0
    for token in text.split():
        # Strip surrounding punctuation so e.g. ``"Dave,"`` and ``"Dave"``
        # both register as proper-noun-shaped.
        clean = token.strip(".,!?;:'\"()[]{}")
        if not clean:
            continue
        lower = clean.lower()
        if lower in keywords:
            keyword_count += 1
            if keyword_count >= threshold:
                return True
            continue
        # Fold (pre-merge review IMPORTANT): drop the ``len > 2`` guard
        # that pre-fold excluded valid two-letter proper nouns ("Li",
        # "Wu", "Yi", "JK"). Per the conservative-tilt contract, short
        # names should trip the heuristic — the guard contradicted it.
        # Single-character tokens are still skipped (no real two-letter-
        # or-shorter false positives beyond stop-words, which are
        # already excluded).
        if len(clean) > 1 and clean[0].isupper() and lower not in _STOP_WORDS:
            proper_noun_count += 1
            if proper_noun_count >= threshold:
                return True
    return False


def filter_identity_bearing_links(
    links_dict: dict[str, list[dict]],
    *,
    threshold: int = 1,
    identity_keywords: frozenset[str] | None = None,
) -> dict[str, list[dict]]:
    """Return a copy of ``links_dict`` with identity-bearing event_sigs dropped.

    Pure function: ``links_dict`` is not mutated. Lists are shallow-copied;
    link dicts are NOT deep-copied (the caller is the bundle composer
    which immediately serializes to JSON, so aliasing is safe). The
    bundle composer (PR D) calls this on ``nac_dump["links"]`` after
    merging but before bundle composition.

    Parameters mirror :func:`is_identity_bearing`.
    """
    out: dict[str, list[dict]] = {}
    for evt_sig, links in links_dict.items():
        if is_identity_bearing(evt_sig, threshold=threshold, identity_keywords=identity_keywords):
            continue
        out[evt_sig] = list(links)
    return out


def is_identity_domain(domain: str | None) -> bool:
    """Return ``True`` if ``domain`` is the reserved identity marker.

    Used by PR D's EC bundle composition: nodes with
    ``domain == IDENTITY_DOMAIN_MARKER`` are excluded from substrate
    bundles regardless of any other domain filter. Tolerates ``None``
    (most nodes are undomained — return ``False``).
    """
    return domain == IDENTITY_DOMAIN_MARKER


__all__ = [
    "IDENTITY_DOMAIN_MARKER",
    "filter_identity_bearing_links",
    "is_identity_bearing",
    "is_identity_domain",
]
