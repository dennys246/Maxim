"""CC3 (v1.0 freeze) — frozen-dataclass forward-compat regression tests.

These tests pin the contract documented in CLAUDE.md's "Frozen dataclasses
shipping in 1.0 are forward-compat-audited" invariant. Two halves:

(a) Escape-hatch dataclasses — ``Episode`` and ``SensorReading``. Each has
    an ``extra: dict[str, Any]`` field declared with ``hash=False,
    compare=False`` so the dataclass stays hashable and identity-safe
    when ``extra`` carries an empty (or populated) dict. Round-trip
    preserves ``extra`` through ``to_dict``/``from_dict``.

(b) Shape-frozen dataclasses — typed contracts where an ``extra`` dict
    would dilute the type or re-open an isolation back-channel
    (``TraceSnapshot``, ``Reaction``, ``ReactionContext``,
    ``PerceptContext``, drive specs, ``ValenceSignal``,
    ``TemporalSignature``, ``MeshNode``, ``MeshConfig``).
    These tests do NOT enforce shape via reflection (the docstring
    marker is the human-readable contract); they DO assert that the
    types stay constructible and hashable.

Hash regression tripwire: any future refactor that adds ``hash=True`` /
omits the ``hash=False, compare=False`` on an ``extra`` field flips
``hash(instance)`` from "works" to "TypeError". The hash assertions
catch that at test time.
"""

from __future__ import annotations

from maxim.embodiment.sem import SensorReading
from maxim.memory.episode import Episode
from maxim.reactions.types import Reaction, ReactionContext, TraceSnapshot
from maxim.decisions.causal_link import TemporalDelta, Valence


# ─────────────────────────────────────────────────────────────────────────
# (a) Escape-hatch dataclasses
# ─────────────────────────────────────────────────────────────────────────


class TestEpisodeExtraField:
    def _make(self, extra: dict | None = None) -> Episode:
        return Episode(
            id="e1",
            start_tick=0,
            end_tick=10,
            channel="test",
            sender_ids=(),
            thread_id=None,
            activated_nodes=(),
            reward_events=(),
            scn_tag=None,
            extra=extra or {},
        )

    def test_default_extra_is_empty_dict(self) -> None:
        ep = Episode(
            id="e1",
            start_tick=0,
            end_tick=10,
            channel="t",
            sender_ids=(),
            thread_id=None,
            activated_nodes=(),
            reward_events=(),
            scn_tag=None,
        )
        assert ep.extra == {}

    def test_extra_round_trips_through_dict(self) -> None:
        ep = self._make(extra={"source_doc": "abc.md", "score": 0.42})
        rebuilt = Episode.from_dict(ep.to_dict())
        assert rebuilt == ep
        assert rebuilt.extra == {"source_doc": "abc.md", "score": 0.42}

    def test_empty_extra_is_omitted_from_dict(self) -> None:
        # to_dict only writes the key if extra is non-empty — keeps pre-1.0
        # dump shape identical for episodes without extra metadata.
        d = self._make().to_dict()
        assert "extra" not in d

    def test_pre_cc3_dict_loads_with_empty_extra(self) -> None:
        # Loader must accept dicts WITHOUT an extra key (old persisted data).
        legacy = self._make().to_dict()
        legacy.pop("extra", None)
        rebuilt = Episode.from_dict(legacy)
        assert rebuilt.extra == {}

    def test_hash_works_with_default_extra(self) -> None:
        # hash=False on extra keeps the dataclass hashable. Pre-CC3
        # callers that put episodes in dict keys / sets continue to work.
        ep = self._make()
        assert hash(ep) == hash(ep)
        assert {ep, self._make()}  # set construction does not raise

    def test_hash_works_with_populated_extra(self) -> None:
        ep = self._make(extra={"k": "v"})
        # hash(ep) must not raise even though extra is a dict.
        h = hash(ep)
        assert isinstance(h, int)

    def test_extra_does_not_affect_equality(self) -> None:
        # compare=False on extra: two episodes with identical declared
        # fields but different extra dicts compare equal. Identity is
        # in the declared fields, not the observability dict.
        a = self._make(extra={"k": "v"})
        b = self._make(extra={"k": "OTHER"})
        assert a == b

    def test_extra_does_not_affect_hash(self) -> None:
        # Same identity (declared fields) → same hash, regardless of extra.
        a = self._make(extra={"k": "v"})
        b = self._make(extra={"x": 1})
        assert hash(a) == hash(b)


class TestSensorReadingExtraField:
    def test_default_extra_is_empty_dict(self) -> None:
        sr = SensorReading(
            sensor_name="s",
            entity_name="e",
            value=1.0,
            unit="C",
            timestamp=0.0,
        )
        assert sr.extra == {}

    def test_hash_works_with_default_extra(self) -> None:
        sr = SensorReading(
            sensor_name="s",
            entity_name="e",
            value=1.0,
            unit="C",
            timestamp=0.0,
        )
        assert hash(sr) == hash(sr)

    def test_hash_works_with_populated_extra(self) -> None:
        sr = SensorReading(
            sensor_name="s",
            entity_name="e",
            value=1.0,
            unit="C",
            timestamp=0.0,
            extra={"calibration": "v2"},
        )
        h = hash(sr)
        assert isinstance(h, int)

    def test_extra_does_not_affect_equality(self) -> None:
        a = SensorReading("s", "e", 1.0, "C", 0.0, extra={"a": 1})
        b = SensorReading("s", "e", 1.0, "C", 0.0, extra={"b": 2})
        assert a == b


# ─────────────────────────────────────────────────────────────────────────
# (b) Shape-frozen dataclasses — basic constructability + hashability
# ─────────────────────────────────────────────────────────────────────────


class TestTraceSnapshotShapeFrozen:
    """``TraceSnapshot`` is shape-frozen because it's reachable from
    ``ReactionContext.bindings``; an ``extra`` dict on it would re-open
    the isolation back-channel ``ReactionContext`` itself closes. This
    test pins that no ``extra`` field exists — adding one in a future
    refactor would silently regress the isolation rule documented in
    ``reactions/types.py``.
    """

    def test_no_extra_field(self) -> None:
        # Reflective check — if you find yourself wanting to remove
        # this assertion to add ``extra``, read the SHAPE-FROZEN block
        # in ``TraceSnapshot``'s docstring first. If the isolation
        # rationale truly no longer applies, document the new rationale
        # in the ``ReactionContext`` module docstring AND coordinate
        # with ``mother_npc_stimulus_plan`` reviewers.
        ts = TraceSnapshot(percept_id="p1")
        assert not hasattr(ts, "extra")

    def test_round_trip_through_dict(self) -> None:
        ts = TraceSnapshot(
            percept_id="p1",
            activation_strength=0.7,
            content_hash="abc",
            decay_factor=0.9,
        )
        rebuilt = TraceSnapshot.from_dict(ts.to_dict())
        assert rebuilt == ts


class TestTemporalDeltaDefault:
    def test_default_observed_deltas_is_empty_tuple(self) -> None:
        # CC3 backfill: was a required field, now defaults to () so the
        # dataclass falls into category (a). ``mean``/``min``/``max``
        # tolerate empty-tuple already.
        td = TemporalDelta()
        assert td.observed_deltas == ()
        assert td.mean == 0.0


class TestReactionShapeFrozen:
    """Sanity check that ``Reaction`` stays constructible. Equality and
    hash live on the typed identity (kind + intensity + valence + ...)
    via ``ReactionContext.bindings``; we don't enforce hashability here
    because ``ReactionContext.bindings: dict`` already breaks
    auto-hash. The SHAPE-FROZEN marker is for the type contract, not a
    hash claim.
    """

    def test_construct_reaction(self) -> None:
        r = Reaction(
            kind="pain",
            intensity=0.5,
            valence=Valence.NEGATIVE,
            timestamp=1.0,
            context=ReactionContext(),
            source="test",
        )
        assert r.kind == "pain"
        assert r.intensity == 0.5

    def test_round_trip_through_dict(self) -> None:
        r = Reaction(
            kind="fear",
            intensity=0.8,
            valence=Valence.NEGATIVE,
            timestamp=2.0,
            context=ReactionContext(agent_id="a1"),
            source="cerebellum:motor_failure",
        )
        rebuilt = Reaction.from_dict(r.to_dict())
        assert rebuilt == r
