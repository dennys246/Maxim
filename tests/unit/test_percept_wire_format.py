"""Pin the Percept wire-format vs. session-format split (C10 prep).

The wire-dict ships peer → leader through the (to-be-implemented in 1.1)
perception transport. The session-dict persists to the leader's session log.
The two formats serve different audiences:

- Session-dict: minimal disk footprint for replay; includes substrate
  references the leader already computed (``substrate_node_id``).
- Wire-dict: full raw-observation fidelity so the receiving leader can
  run substrate encoding from scratch; substrate fields and
  leader-computed derived signals are intentionally absent.

These tests pin the contract so a silent shape drift post-1.0 (which would
be a session-format-drift class of bug) gets caught here.
"""

from __future__ import annotations

from maxim.agents.bus import Percept
from maxim.agents.percept_context import PerceptContext


def _make_representative_percept() -> Percept:
    """Build a Percept with every wire-relevant field populated.

    Substrate fields (``embedding``, ``substrate_node_id``) are populated
    here so the "wire-dict NEVER carries substrate" assertion exercises
    the actual omission, not a falsy-default coincidence.
    """
    ctx = PerceptContext(
        channel="mesh",
        sender="remote:reachy-001",
        agent_id="leader-agent",
    )
    return Percept(
        timestamp=1234567890.5,
        source="vision",
        detections=[{"label": "cup", "confidence": 0.91}],
        transcript_chunk="hello world",
        transcript_chunk_index=42,
        file_changed="/tmp/foo.txt",
        cli_input="/help",
        salience=0.7,
        novelty=0.4,
        has_voice_command=True,
        has_maxim_keyword=False,
        hard_override=None,
        explore_command={"verb": "look", "target": "cup"},
        content="user said hello",
        metadata={"trace_id": "abc"},
        raw_transcript_text="hello world, full transcript here",
        maxim_runtime={"loop_tick": 17},
        context=ctx,
        modality="vision",
        embedding=[0.1, 0.2, 0.3, 0.4],
        substrate_node_id="ec-node-77",
    )


# Whitelist — these are the fields a network-backed PerceptSource MAY ship.
# Adding a field to this list = a deliberate decision to tunnel it; removing
# is a wire-format change that needs the _format_version bump path.
EXPECTED_WIRE_FIELDS: frozenset[str] = frozenset(
    {
        "_format_version",  # CC1 contract — wire payload versioned at root
        "timestamp",
        "source",
        "detections",
        "transcript_chunk",
        "transcript_chunk_index",
        "file_changed",
        "cli_input",
        "has_voice_command",
        "has_maxim_keyword",
        "hard_override",
        "explore_command",
        "content",
        "metadata",
        "raw_transcript_text",
        "sensory",
        "context",
        "modality",
    }
)


def test_wire_dict_field_set_matches_whitelist():
    """The wire-dict field set is an explicit whitelist (not a blacklist).

    A blacklist would silently include any new Percept field by default;
    a whitelist forces the wire-format question to be answered each time
    a Percept field is added.
    """
    p = _make_representative_percept()
    actual = set(p.to_wire_dict().keys())
    assert actual == EXPECTED_WIRE_FIELDS, (
        f"Wire-dict field set drifted from contract.\n"
        f"  Unexpected: {sorted(actual - EXPECTED_WIRE_FIELDS)}\n"
        f"  Missing:    {sorted(EXPECTED_WIRE_FIELDS - actual)}\n"
        "If this drift is intentional, update EXPECTED_WIRE_FIELDS and bump "
        "the _format_version on the wire-dict path."
    )


def test_wire_dict_excludes_substrate_fields():
    """Embedding and substrate_node_id are NEVER on the wire.

    The leader owns substrate (EC, ATL, LinguisticEncoder); the peer
    ships raw observations. Bio-fidelity rule: peer is "a sensor," not
    "a partial cognition." Populating substrate fields on the source
    Percept must not leak them onto the wire.
    """
    p = _make_representative_percept()
    assert p.embedding is not None  # populated on the source Percept
    assert p.substrate_node_id is not None  # populated on the source Percept

    wire = p.to_wire_dict()
    assert "embedding" not in wire
    assert "substrate_node_id" not in wire


def test_wire_dict_excludes_leader_computed_signals():
    """Salience, novelty, and maxim_runtime stay off the wire.

    Salience/novelty are leader-derived (comparison against substrate
    state); maxim_runtime is leader-internal. A peer cannot compute
    these without the leader's substrate.
    """
    p = _make_representative_percept()
    wire = p.to_wire_dict()
    assert "salience" not in wire
    assert "novelty" not in wire
    assert "maxim_runtime" not in wire


def test_wire_dict_carries_transcript_chunk_index_session_dict_omits_it():
    """An explicit divergence between the two formats.

    The session-dict ``to_dict`` deliberately omits
    ``transcript_chunk_index`` (it's a peer-side sequencing artifact
    not needed for replay); the wire-dict includes it so the receiving
    leader can detect peer-side sequence gaps. Pinning this divergence
    catches the failure mode where a future edit accidentally collapses
    the two paths into one.
    """
    p = _make_representative_percept()
    assert "transcript_chunk_index" in p.to_wire_dict()
    assert "transcript_chunk_index" not in p.to_dict()


def test_wire_dict_round_trips_without_loss():
    """to_wire_dict → from_wire_dict preserves the wire-shipped fields."""
    p = _make_representative_percept()
    wire = p.to_wire_dict()
    restored = Percept.from_wire_dict(wire)

    assert restored.timestamp == p.timestamp
    assert restored.source == p.source
    assert restored.detections == p.detections
    assert restored.transcript_chunk == p.transcript_chunk
    assert restored.transcript_chunk_index == p.transcript_chunk_index
    assert restored.file_changed == p.file_changed
    assert restored.cli_input == p.cli_input
    assert restored.has_voice_command == p.has_voice_command
    assert restored.has_maxim_keyword == p.has_maxim_keyword
    assert restored.hard_override == p.hard_override
    assert restored.explore_command == p.explore_command
    assert restored.content == p.content
    assert restored.metadata == p.metadata
    assert restored.raw_transcript_text == p.raw_transcript_text
    assert restored.modality == p.modality

    # PerceptContext rehydrates as a typed instance, not a bare dict.
    assert isinstance(restored.context, PerceptContext)
    assert restored.context.channel == p.context.channel
    assert restored.context.sender == p.context.sender
    assert restored.context.agent_id == p.context.agent_id


def test_wire_dict_round_trip_leaves_substrate_fields_unset_on_receive():
    """A leader reconstructing a peer-shipped Percept must NOT inherit
    a substrate reference from the wire (none was shipped).

    The receiving side will populate ``embedding`` and ``substrate_node_id``
    by running substrate encoding on the rehydrated Percept; until then
    they must be None — anything else would be a leak.
    """
    p = _make_representative_percept()
    wire = p.to_wire_dict()
    restored = Percept.from_wire_dict(wire)

    assert restored.embedding is None
    assert restored.substrate_node_id is None


def test_wire_dict_handles_minimal_percept():
    """Round-trip a Percept with only required fields populated.

    A peer that ships only a timestamp + source (e.g., a heartbeat
    percept) must not crash the wire-format path.
    """
    p = Percept(timestamp=100.0, source="idle")
    wire = p.to_wire_dict()
    restored = Percept.from_wire_dict(wire)
    assert restored.timestamp == 100.0
    assert restored.source == "idle"
    assert restored.detections == []
    assert restored.context is None


def test_wire_dict_handles_none_context():
    """A Percept with no PerceptContext serializes context as None."""
    p = Percept(timestamp=1.0, source="vision", context=None)
    wire = p.to_wire_dict()
    assert wire["context"] is None
    restored = Percept.from_wire_dict(wire)
    assert restored.context is None


def test_wire_dict_and_session_dict_field_sets_diverge():
    """The wire-dict and session-dict are NOT the same shape.

    Defensive guard against a future copy-paste edit that accidentally
    collapses the two paths into one. The substantive distinctions
    (wire excludes substrate; session excludes peer-side raw observations)
    rest on the field sets actually differing.
    """
    p = _make_representative_percept()
    wire_keys = set(p.to_wire_dict().keys())
    session_keys = set(p.to_dict().keys())
    assert wire_keys != session_keys, (
        "Wire-dict and session-dict have the same field set; the two contracts have "
        "collapsed. Either path's invariants (no substrate on wire / no peer-raw on "
        "session) are no longer enforced by shape divergence."
    )


def test_wire_dict_stamps_format_version():
    """The wire-dict carries _format_version at root (CC1 contract).

    Matches the broader 'every persisted/wire-shipped JSON payload
    carries _format_version' invariant. The Hivemind bundle does the
    same for its manifest; this is the perception-transport equivalent.
    """
    from maxim.utils.format_version import FORMAT_VERSION

    p = _make_representative_percept()
    wire = p.to_wire_dict()
    assert wire["_format_version"] == FORMAT_VERSION


def test_wire_dict_from_wire_tolerates_missing_format_version():
    """Legacy/test producers that omit _format_version still deserialize.

    Mirrors :func:`check_format_version`'s "absent field → 0.x + warn"
    behavior. The peer-side ecosystem may include older builds; the
    receiver must not crash on them.
    """
    p = _make_representative_percept()
    wire = p.to_wire_dict()
    wire.pop("_format_version")
    # Should reconstruct cleanly with a single warning (suppressed
    # after first occurrence per the check_format_version contract).
    restored = Percept.from_wire_dict(wire)
    assert restored.timestamp == p.timestamp
    assert restored.source == p.source


def test_wire_dict_from_wire_strips_format_version_from_kwargs():
    """``_format_version`` must not leak into the Percept constructor.

    ``Percept`` has no ``_format_version`` field; if the loader passes
    it through the ``**kwargs`` filter, the dataclass raises TypeError.
    The valid_fields filter handles this incidentally today, but pin
    it explicitly so a future loader refactor (e.g. switching to
    ``**data`` directly) doesn't silently regress.
    """
    p = _make_representative_percept()
    wire = p.to_wire_dict()
    assert "_format_version" in wire
    # No raise → the filter or explicit pop is doing its job.
    restored = Percept.from_wire_dict(wire)
    assert not hasattr(restored, "_format_version")


def test_from_wire_dict_rejects_injected_leader_only_fields():
    """Inbound allowlist (2026-08-12 audit): a buggy or malicious peer
    shipping wire-EXCLUDED fields must not get them rehydrated — they
    are leader-derived values (substrate refs, salience) the leader
    computes itself. Outbound exclusion alone is only half the
    invariant."""
    p = _make_representative_percept()
    wire = p.to_wire_dict()
    wire["embedding"] = [0.1, 0.2, 0.3]
    wire["substrate_node_id"] = "ec-node-77"
    wire["salience"] = 0.99
    wire["novelty"] = 0.88
    wire["maxim_runtime"] = {"internal": True}

    restored = Percept.from_wire_dict(wire)

    assert restored.embedding is None
    assert restored.substrate_node_id is None
    assert restored.salience == 0.0
    assert restored.novelty == 0.0
    assert restored.maxim_runtime is None


def test_wire_fields_constant_matches_to_wire_dict_output():
    """_PERCEPT_WIRE_FIELDS is the single source of truth for both wire
    directions — pin it against what to_wire_dict actually emits so the
    two cannot drift apart."""
    from maxim.agents.bus import _PERCEPT_WIRE_FIELDS

    p = _make_representative_percept()
    wire = p.to_wire_dict()
    emitted = set(wire.keys()) - {"_format_version", "context"}
    assert emitted == set(_PERCEPT_WIRE_FIELDS)
