"""Identity-bearing concept detection tests (Hivemind shareability, PR C).

v1_refinement.md §B5 PR C. Validates the heuristic that detects whether
a text label refers to a specific named entity. Conservative tilt:
false positives (legitimate patterns staying local) are preferred over
false negatives (identity-bearing patterns leaking out).
"""

from __future__ import annotations

from maxim.hivemind.identity import (
    IDENTITY_DOMAIN_MARKER,
    filter_identity_bearing_links,
    is_identity_bearing,
    is_identity_domain,
)


# ─────────────────────────────────────────────────────────────────────────
# Empty / trivial inputs
# ─────────────────────────────────────────────────────────────────────────


def test_empty_string_is_not_identity_bearing() -> None:
    """The empty string is the trivial no-op input."""
    assert not is_identity_bearing("")


def test_whitespace_only_is_not_identity_bearing() -> None:
    """A whitespace string has no tokens — not identity-bearing."""
    assert not is_identity_bearing("   \t\n  ")


# ─────────────────────────────────────────────────────────────────────────
# Generic shareable concepts — should NOT trip the detector
# ─────────────────────────────────────────────────────────────────────────


def test_lowercase_action_phrase_not_identity_bearing() -> None:
    """A lowercase event-sig-like phrase passes through."""
    assert not is_identity_bearing("opened door")


def test_attacked_dragon_not_identity_bearing() -> None:
    """A common-noun phrase even with action verb passes through."""
    assert not is_identity_bearing("attacked the dragon")


def test_sentence_case_with_only_stop_word_capitalization_not_identity() -> None:
    """A sentence-cased "The door creaked open" is not identity-bearing —
    "The" is a stop-word, so the only capitalized initial isn't counted.
    """
    assert not is_identity_bearing("The door creaked open")


def test_tool_signature_not_identity_bearing() -> None:
    """A tool-style event signature like ``tool:open_door`` is not
    identity-bearing — no capitalized tokens, no identity keywords.
    """
    assert not is_identity_bearing("tool:open_door:success")


# ─────────────────────────────────────────────────────────────────────────
# Identity-bearing labels — SHOULD trip the detector
# ─────────────────────────────────────────────────────────────────────────


def test_proper_noun_name_is_identity_bearing() -> None:
    """A simple "met Dave" pattern is identity-bearing — Dave is a
    proper-noun-shaped token.
    """
    assert is_identity_bearing("met Dave at the market")


def test_my_friend_is_identity_bearing() -> None:
    """The ``my`` ownership marker flags the phrase as identity-bearing
    (the listener can't disambiguate without context).
    """
    assert is_identity_bearing("greeted my friend")


def test_dr_smith_is_identity_bearing() -> None:
    """A title (``dr``) + a proper noun both trip the detector."""
    assert is_identity_bearing("consulted Dr Smith")


def test_named_X_is_identity_bearing() -> None:
    """The ``named`` keyword flags a phrase as identity-bearing."""
    assert is_identity_bearing("the NPC named Aldric")


def test_user_keyword_is_identity_bearing() -> None:
    """The reserved ``user`` keyword (own identity in conversation) flags."""
    assert is_identity_bearing("the user requested help")


def test_short_two_letter_proper_noun_is_identity_bearing() -> None:
    """Fold regression guard (pre-merge review IMPORTANT): short proper
    nouns like ``Li``, ``Wu``, ``Yi``, ``JK`` are valid identity markers
    in non-Western names and initials. Pre-fold the ``len > 2`` guard
    silently let them through despite the conservative-tilt contract.
    """
    assert is_identity_bearing("met Li at the market")
    assert is_identity_bearing("greeted Wu and his daughter")
    # The two-character lowercase form is NOT identity-bearing — capitalization
    # is the actual signal.
    assert not is_identity_bearing("met li at the market")


# ─────────────────────────────────────────────────────────────────────────
# Threshold parameter
# ─────────────────────────────────────────────────────────────────────────


def test_higher_threshold_relaxes_detection() -> None:
    """At threshold=2, a single proper noun no longer trips."""
    assert is_identity_bearing("met Dave", threshold=1)
    assert not is_identity_bearing("met Dave", threshold=2)


def test_higher_threshold_still_trips_on_multiple_signals() -> None:
    """Two proper nouns at threshold=2 still trips."""
    assert is_identity_bearing("Dave and Sarah went to Portland", threshold=2)


# ─────────────────────────────────────────────────────────────────────────
# identity_keywords parameter
# ─────────────────────────────────────────────────────────────────────────


def test_custom_keywords_extend_detection() -> None:
    """Operators can extend the keyword set with domain-specific terms."""
    custom = frozenset({"avatar", "handle"})
    assert is_identity_bearing("the avatar greeted me", identity_keywords=custom)
    # Without the custom set, lowercase "avatar" alone is not identity-bearing.
    assert not is_identity_bearing("the avatar greeted me", identity_keywords=frozenset())


def test_empty_keyword_set_disables_keyword_signal() -> None:
    """Passing an empty frozenset disables keyword matching — only proper
    nouns trip detection.
    """
    # "my friend" uses the keyword signal in the default set.
    assert not is_identity_bearing("my friend visited", identity_keywords=frozenset())
    # The proper-noun signal still works.
    assert is_identity_bearing("Dave visited", identity_keywords=frozenset())


# ─────────────────────────────────────────────────────────────────────────
# Punctuation handling
# ─────────────────────────────────────────────────────────────────────────


def test_punctuation_attached_to_proper_noun_strips() -> None:
    """``Dave,`` strips to ``Dave`` and trips detection."""
    assert is_identity_bearing("met Dave, the merchant")


def test_punctuation_only_token_is_skipped() -> None:
    """A pure-punctuation token doesn't break iteration."""
    assert not is_identity_bearing("opened , the door")


# ─────────────────────────────────────────────────────────────────────────
# filter_identity_bearing_links
# ─────────────────────────────────────────────────────────────────────────


def test_filter_drops_identity_bearing_event_sig() -> None:
    """Event sigs that look identity-bearing are dropped from the output."""
    links = {
        "tool:open_door:success": [{"id": "link-1"}],
        "met Dave at the market": [{"id": "link-2"}],
        "attacked the dragon": [{"id": "link-3"}],
    }
    filtered = filter_identity_bearing_links(links)
    assert set(filtered.keys()) == {"tool:open_door:success", "attacked the dragon"}


def test_filter_is_pure() -> None:
    """``filter_identity_bearing_links`` does not mutate its input."""
    original = {
        "tool:open_door:success": [{"id": "link-1"}],
        "met Dave at the market": [{"id": "link-2"}],
    }
    original_copy = {k: list(v) for k, v in original.items()}
    filter_identity_bearing_links(original)
    assert original == original_copy


def test_filter_preserves_list_objects_shallow() -> None:
    """The filtered output is a shallow copy — list objects are NEW but the
    contained link dicts are the same references (caller serializes to JSON
    next, so aliasing is safe).
    """
    inner_link = {"id": "link-1"}
    links = {"tool:open_door:success": [inner_link]}
    filtered = filter_identity_bearing_links(links)
    # The list is a copy.
    assert filtered["tool:open_door:success"] is not links["tool:open_door:success"]
    # The link dict is the same reference.
    assert filtered["tool:open_door:success"][0] is inner_link


def test_filter_empty_dict_returns_empty() -> None:
    """No links → empty output, no error."""
    assert filter_identity_bearing_links({}) == {}


# ─────────────────────────────────────────────────────────────────────────
# is_identity_domain
# ─────────────────────────────────────────────────────────────────────────


def test_identity_domain_marker_is_underscore_prefixed() -> None:
    """The marker is reserved in the ``_*`` namespace by convention,
    matching ``_consensus`` in ``merge.py``.
    """
    assert IDENTITY_DOMAIN_MARKER.startswith("_")


def test_is_identity_domain_true_for_marker() -> None:
    assert is_identity_domain(IDENTITY_DOMAIN_MARKER)


def test_is_identity_domain_false_for_none() -> None:
    """An undomained node is not identity-marked."""
    assert not is_identity_domain(None)


def test_is_identity_domain_false_for_other_domains() -> None:
    assert not is_identity_domain("combat")
    assert not is_identity_domain("cooking")
    assert not is_identity_domain("")
