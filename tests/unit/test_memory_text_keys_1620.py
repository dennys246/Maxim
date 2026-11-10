"""#1620: the content keys converge the regular inflections of one word (memory/text.py::settle_key).

The frozen ``_lemmatize`` strips "-ed"/"-ing" without reaching a base form ("moved" -> "mov" while "move" stays
whole), so a memory "the guard moved the key" shared no key with a cue "where did the guard move it?". The keys now
converge on top of the frozen tokenizer; ``normalize_tokens`` itself is unchanged (ATL names are persisted from it).
"""

from __future__ import annotations

import time

import pytest

from maxim.memory import text as T

FORMS = [
    ("move moved moving moves", "mov"),
    ("locate located locating locates", "locat"),
    ("place placed placing places", "plac"),
    ("hide hiding hides", "hid"),
    ("guard guarded guarding guards", "guard"),
    ("stop stopped stopping stops", "stop"),
    ("store stored storing stores", "stor"),
    ("take taking takes", "tak"),
    ("ask asked asking asks", "ask"),
    ("issue issued issuing issues", "issu"),
    ("pull pulled pulling pulls", "pul"),
    ("drip dripped dripping drips", "drip"),
    ("close closed closing closes", "clo"),
    ("house houses housed housing", "hou"),
    ("plan planned planning", "plan"),
]

# Not converged, stated: the frozen tokenizer decides these before the keys converge ("used" -> "us" is under 3
# characters, the "-eed" rule keeps "agreed", "accessing" -> "acces" settles to "acce"), or they are irregular. Pinned,
# so a change to any of them is seen. An "-ss" word keeps its double "s": undoubling it chains into the "-s" rule
# ("brass" -> "bras" -> "bra").
GAPS = {
    "use": ["use"],
    "used": ["us"],
    "using": ["using"],
    "agree": ["agree"],
    "agreed": ["agreed"],
    "access": ["access"],
    "accessing": ["acc"],
    "pass": ["pass"],
    "passing": ["pas"],
    "brass": ["brass"],
    "1000": ["1000"],
    "100": ["100"],
    "****": ["****"],
    "hid": ["hid"],
}


@pytest.mark.parametrize(("forms", "key"), FORMS)
def test_the_regular_inflections_of_a_word_reach_one_key(forms: str, key: str) -> None:
    assert {w: T.key_tokens(w) for w in forms.split()} == {w: [key] for w in forms.split()}


def test_the_stated_gaps_stay_as_stated() -> None:
    assert {w: T.key_tokens(w) for w in GAPS} == GAPS


def test_the_stated_cost_words_differing_by_a_final_e_share_a_key() -> None:
    for a, b in (("plan", "plane"), ("made", "mad"), ("care", "car"), ("hope", "hop")):
        assert T.key_tokens(a) == T.key_tokens(b), (a, b)


def test_a_long_run_is_keyed_without_converging_it() -> None:
    """A token past ``_CONVERGE_MAX_LEN`` is no word: it keeps its frozen key, and keying it stays linear (each
    convergence step re-normalizes the token, so a 30,000-letter run took 5.8 s)."""
    run = "z" * 30_000
    started = time.perf_counter()
    assert T.key_tokens(run) == T.normalize_tokens(run) == [run]
    assert time.perf_counter() - started < 1.0
    for chain in ("ed" * 16_000, "ing" * 10_000, "stopp" + "ed" * 8_000):  # one frozen suffix strip per pass
        started = time.perf_counter()
        T.key_tokens(chain)
        assert time.perf_counter() - started < 1.0, chain[:12]


def test_normalize_tokens_is_unchanged() -> None:
    """The frozen tokenizer keeps its quirks: ATL concept names are persisted from its output."""
    assert T.normalize_tokens("moved placed moving located stopped") == ["mov", "plac", "moving", "locat", "stopp"]


def test_a_persisted_quirky_concept_name_reaches_its_words_key() -> None:
    """An ATL name persisted from the frozen tokenizer ("plac", from "placed") keys as the words it came from, so the
    name_token index still matches them (atl.py keys ``name_token`` through ``key_tokens``)."""
    for name, word in (("plac", "placing"), ("mov", "move"), ("locat", "location is located"), ("stopp", "stop")):
        assert T.key_tokens(name)[0] in T.key_tokens(word), (name, word)


IDEMPOTENT = [
    "move",
    "moved",
    "moving",
    "stopped",
    "pull",
    "issue",
    "houses",
    "used",
    "freeing",
    "accessing",
    "cape",
    "settings",
    "closing",
    "whose",
    "hmmmmmmmmmmmm",
    "shhhhhhhhhhhhh",
    "abcdeffffffffffffffffff",
    "m" + "o" * 37 + "ving",
    "p" * 39 + "ed",
    "x" * 60 + "ing",
]


@pytest.mark.parametrize("word", IDEMPOTENT)
def test_settle_key_is_idempotent(word: str) -> None:
    """Every step shortens the word, so settling ends at a fixed point, never at a loop cap (a capped loop keyed
    "hmmmmmmmmmmmm" as "hmmmm", which settled again to "hmm")."""
    for key in T.key_tokens(word):
        assert T.settle_key(key) == key


def test_a_key_is_never_one_the_tokenizer_rewrites() -> None:
    """A key is the tokenizer's fixed point, never a stop word: "close" settles through "clos" to "clo" (as "closed"
    does), "butt" is not "but", and "whose" is not "who"."""
    assert T.key_tokens("butt whose") == ["butt", "whose"]
    for word in ("move", "store", "pull", "moving", "stopped", "issue", "close", "house", "hmmmmmmmmmmmm"):
        (key,) = T.key_tokens(word)
        assert T.normalize_tokens(key) == [key] and key not in T._STOP_WORDS


def test_a_base_form_cue_reaches_an_inflected_memory() -> None:
    """The ranker (``hippocampus_retrieval._rank_by_relevance``, the cue path) gives a base-form cue a hit on a memory
    that holds the inflected form, and the other way round."""
    from maxim.memory.hippocampus_retrieval import _rank_by_relevance
    from maxim.memory.types import Context, EpisodicMemory, Perception

    now = time.time()
    moved = EpisodicMemory(
        id="moved", timestamp=now - 10, perception=Perception(observations={"text": "the warden moved the brass ladle"})
    )
    other = EpisodicMemory(
        id="other", timestamp=now, perception=Perception(observations={"text": "rain falls on the roof"})
    )
    placed = EpisodicMemory(id="placed", timestamp=now - 20, context=Context(active_goal="hide the placed lantern"))
    assert [m.id for m in _rank_by_relevance([moved, other, placed], "where did the warden move it", 1)] == ["moved"]
    assert [m.id for m in _rank_by_relevance([moved, other, placed], "placing a lantern", 1)] == ["placed"]
    assert [m.id for m in _rank_by_relevance([moved, other, placed], "hiding", 1)] == ["placed"]
