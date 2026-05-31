"""Provenance + substrate domain fields on CausalLink (Hivemind shareability, PR A).

v1_refinement.md §B5 (Hivemind shareability infrastructure) adds two
shareability fields to ``CausalLink``:

- ``source``: opaque contributor ID. ``"local"`` for links learned on
  this Maxim; some other ID (e.g. ``"oasis-abc123"``, ``"consensus"``)
  for links merged in from a substrate bundle. Enables trust management
  and the raw-vs-primed experimental distinction in
  ``maxim_hivemind.md``.
- ``domain``: optional substrate-domain tag (``"combat"``, ``"cooking"``,
  ``"medical"``, ...). ``None`` for undomained / generic links. Lets
  Oases subscribe to specific domains.

These tests pin:

1. Default values on a fresh instance (``"local"`` and ``None``).
2. Round-trip through ``to_dict`` / ``from_dict`` preserving custom
   values.
3. Backward-compatible default substitution when the dict lacks the
   fields (existing on-disk dumps load cleanly).
"""

from __future__ import annotations

from maxim.decisions.causal_link import CausalLink, TemporalDelta, Valence


def _make_link(
    *,
    source: str = "local",
    domain: str | None = None,
    contributors: tuple[str, ...] = (),
) -> CausalLink:
    return CausalLink(
        id="link-1",
        event_type="tool",
        event_signature="tool:open_door",
        event_context={"location": "hallway"},
        outcome_type="tool_result",
        outcome_signature="tool_result:open_door:success",
        outcome_valence=Valence.POSITIVE,
        temporal_delta=TemporalDelta(),
        source=source,
        domain=domain,
        contributors=contributors,
    )


# ─────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────


def test_defaults_source_local_domain_none() -> None:
    """A fresh CausalLink defaults to ``source="local"`` and ``domain=None``.

    These defaults preserve the "no Hivemind" semantics: every link a
    Maxim learns on its own is locally-sourced and undomained until a
    bundle import or domain-aware learning path sets otherwise.
    """
    link = _make_link()
    assert link.source == "local"
    assert link.domain is None


# ─────────────────────────────────────────────────────────────────────────
# Custom values round-trip
# ─────────────────────────────────────────────────────────────────────────


def test_to_dict_includes_source_and_domain() -> None:
    """``to_dict`` carries both fields so persistence preserves them."""
    link = _make_link(source="oasis-abc123", domain="combat")
    data = link.to_dict()
    assert data["source"] == "oasis-abc123"
    assert data["domain"] == "combat"


def test_round_trip_preserves_source_and_domain() -> None:
    """``to_dict`` → ``from_dict`` preserves custom values exactly."""
    original = _make_link(source="consensus", domain="medical")
    restored = CausalLink.from_dict(original.to_dict())
    assert restored.source == "consensus"
    assert restored.domain == "medical"


def test_round_trip_preserves_none_domain() -> None:
    """``None`` domain round-trips as ``None`` (not coerced to empty string)."""
    original = _make_link(source="oasis-1", domain=None)
    restored = CausalLink.from_dict(original.to_dict())
    assert restored.domain is None


# ─────────────────────────────────────────────────────────────────────────
# Backward compatibility: pre-B5 dumps lack the new fields
# ─────────────────────────────────────────────────────────────────────────


def test_from_dict_missing_source_defaults_to_local() -> None:
    """Pre-B5 dumps that lack ``source`` load as ``source="local"``.

    Every persisted ``aut_nac.json`` written before the Hivemind
    shareability infrastructure landed lacks the ``source`` key. The
    loader MUST tolerate the absence and substitute ``"local"`` so
    existing user sessions continue to load unchanged.
    """
    original = _make_link(source="oasis-x", domain="combat")
    data = original.to_dict()
    del data["source"]
    restored = CausalLink.from_dict(data)
    assert restored.source == "local"
    # domain is unaffected by removing source
    assert restored.domain == "combat"


def test_from_dict_missing_domain_defaults_to_none() -> None:
    """Pre-B5 dumps that lack ``domain`` load as ``domain=None``."""
    original = _make_link(source="oasis-x", domain="combat")
    data = original.to_dict()
    del data["domain"]
    restored = CausalLink.from_dict(data)
    assert restored.domain is None
    # source is unaffected by removing domain
    assert restored.source == "oasis-x"


def test_from_dict_missing_both_defaults_to_local_none() -> None:
    """Both fields absent → both defaults applied."""
    original = _make_link()
    data = original.to_dict()
    del data["source"]
    del data["domain"]
    restored = CausalLink.from_dict(data)
    assert restored.source == "local"
    assert restored.domain is None


# ─────────────────────────────────────────────────────────────────────────
# contributors (fan-in audit trail for PR B's nac.merge())
# ─────────────────────────────────────────────────────────────────────────


def test_contributors_defaults_to_empty_tuple() -> None:
    """A solo-source link has an empty ``contributors`` tuple.

    Single-contributor links are the common case (every locally-learned
    link and every directly-imported single-source link). Empty tuple
    is the "no fan-in" sentinel.
    """
    link = _make_link()
    assert link.contributors == ()


def test_contributors_round_trip_through_json() -> None:
    """Fan-in contributors round-trip as JSON list ↔ Python tuple.

    PR B's ``nac.merge()`` will populate ``contributors = ("A", "B")``
    when aggregating two contributors. The dataclass field stays
    immutable (tuple), the JSON representation is a list, and
    ``from_dict`` restores the tuple.
    """
    original = _make_link(
        source="consensus",
        contributors=("oasis-1", "oasis-2", "oasis-3"),
    )
    data = original.to_dict()
    assert data["contributors"] == ["oasis-1", "oasis-2", "oasis-3"]
    restored = CausalLink.from_dict(data)
    assert restored.contributors == ("oasis-1", "oasis-2", "oasis-3")
    assert isinstance(restored.contributors, tuple)


def test_from_dict_missing_contributors_defaults_to_empty_tuple() -> None:
    """Pre-B5 dumps that lack ``contributors`` load as ``()``.

    Same backward-compat rule as ``source`` / ``domain``: a pre-Hivemind
    ``aut_nac.json`` lacks the field; the loader fills the default so
    existing user sessions continue loading.
    """
    original = _make_link(source="oasis-x")
    data = original.to_dict()
    del data["contributors"]
    restored = CausalLink.from_dict(data)
    assert restored.contributors == ()
