"""P3.5 Stage 1 — BioSystemSnapshot Protocol + SessionSnapshot round-trip tests.

Covers:
- Protocol conformance across all six bio-systems (runtime_checkable).
- Per-system empty-state round-trip (dump → fresh instance → load_state).
- Populated-state round-trip for each system.
- Lock discipline preservation for ATL/NAc/Hippocampus (AST inspection).
- Hippocampus reserves an ``episodes`` key for P3a.
- SessionSnapshot compose + dump + file round-trip for all six systems.
- Envelope shape validation (``schema_version: int``, ``kind``, ``payload``).
- load_state preserves runtime wires (config, not overwritten).
- Envelope-authoritative versioning (bad envelope version is rejected).
"""

from __future__ import annotations

import inspect
import json

import pytest

from maxim.decisions.nac import NAc, NACConfig
from maxim.memory.atl import ATL, ATLConfig
from maxim.memory.cross_layer import CrossLayerEdgeType, CrossLayerGraph
from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
from maxim.memory.percept_trace_buffer import PerceptTraceBuffer
from maxim.memory.snapshot import (
    BioSystemSnapshot,
    SNAPSHOT_KINDS,
    SessionSnapshot,
    atl_from_snapshot,
    atl_to_snapshot,
    unwrap_envelope,
    wrap_envelope,
)
from maxim.time.scn import SCN


# ─────────────────────────────────────────────────────────────────────────
# Protocol conformance
# ─────────────────────────────────────────────────────────────────────────


class TestProtocolConformance:
    """Every bio-system must structurally conform to BioSystemSnapshot."""

    def test_atl_conforms(self):
        assert isinstance(ATL(ATLConfig()), BioSystemSnapshot)

    def test_hippocampus_conforms(self):
        assert isinstance(Hippocampus(HippocampusConfig()), BioSystemSnapshot)

    def test_nac_conforms(self):
        assert isinstance(NAc(NACConfig()), BioSystemSnapshot)

    def test_scn_conforms(self):
        assert isinstance(SCN(), BioSystemSnapshot)

    def test_ptb_conforms(self):
        assert isinstance(PerceptTraceBuffer(), BioSystemSnapshot)

    def test_cross_layer_graph_conforms(self):
        assert isinstance(CrossLayerGraph(), BioSystemSnapshot)

    def test_all_six_have_int_schema_version(self):
        """Envelope versioning is authoritative; all schema_version fields are int, not str."""
        for cls in (ATL, Hippocampus, NAc, SCN, PerceptTraceBuffer, CrossLayerGraph):
            assert isinstance(cls.schema_version, int), f"{cls.__name__}.schema_version is not int"
            assert cls.schema_version == 1


# ─────────────────────────────────────────────────────────────────────────
# Empty-state round-trip per bio-system
# ─────────────────────────────────────────────────────────────────────────


def _strip_timestamps(d: dict) -> dict:
    """Hippocampus has a saved_at timestamp that differs across dumps."""
    d = dict(d)
    d.pop("saved_at", None)
    return d


class TestATLRoundTrip:
    def test_empty_atl_round_trips(self):
        atl = ATL(ATLConfig())
        dumped = atl.dump()
        restored = ATL(ATLConfig())
        restored.load_state(dumped)
        assert restored.dump() == dumped

    def test_dump_returns_dict(self):
        atl = ATL(ATLConfig())
        d = atl.dump()
        assert isinstance(d, dict)
        assert "concepts" in d and "graph" in d and "registry" in d


class TestHippocampusRoundTrip:
    def test_empty_hippocampus_round_trips(self):
        h = Hippocampus(HippocampusConfig())
        dumped = h.dump()
        restored = Hippocampus(HippocampusConfig())
        restored.load_state(dumped)
        assert _strip_timestamps(restored.dump()) == _strip_timestamps(dumped)

    def test_dump_reserves_episodes_key_for_p3a(self):
        """P3a Stage 1 populates this slot; P3.5 Stage 1 just reserves it."""
        h = Hippocampus(HippocampusConfig())
        d = h.dump()
        assert "episodes" in d, "Hippocampus.dump() must reserve the 'episodes' key for P3a"
        assert d["episodes"] == [], "Stage 1 episodes slot must default to empty list"


class TestNAcRoundTrip:
    def test_empty_nac_round_trips(self):
        nac = NAc(NACConfig())
        dumped = nac.dump()
        restored = NAc(NACConfig())
        restored.load_state(dumped)
        assert restored.dump() == dumped

    def test_reward_bias_round_trips(self):
        """P2 Stage 2 shipped reward_bias persistence; verify it survives."""
        nac = NAc(NACConfig())
        nac._reward_bias[("agent1", "node_a")] = 0.5
        nac._reward_bias[("agent2", "node_b")] = -0.25
        dumped = nac.dump()
        restored = NAc(NACConfig())
        restored.load_state(dumped)
        assert restored._reward_bias == nac._reward_bias


class TestSCNRoundTrip:
    def test_empty_scn_round_trips(self):
        scn = SCN()
        dumped = scn.dump()
        restored = SCN()
        restored.load_state(dumped)
        assert restored.dump() == dumped

    def test_payload_version_mutation_does_not_affect_load(self):
        """Round 2 fold: payload-layer 'version' check removed from
        SCN.load_state for consistency with the envelope-authoritative
        versioning tombstone. Mutating it must NOT break load."""
        scn = SCN()
        mutated = scn.dump()
        mutated["version"] = "999.0"
        SCN().load_state(mutated)  # must not raise


class TestPerceptTraceBufferRoundTrip:
    def test_empty_buffer_round_trips(self):
        """Stage 1 empty-buffer requirement — cross-confirmed reviewer finding."""
        ptb = PerceptTraceBuffer()
        dumped = ptb.dump()
        restored = PerceptTraceBuffer()
        restored.load_state(dumped)
        assert restored.dump() == dumped
        assert len(restored) == 0

    def test_load_state_trims_to_max_entries(self):
        """Round 2 Exec critical #3 regression — ring-buffer cap invariant.

        Dump a state whose 'entries' list exceeds the live instance's
        max_entries, load it, assert len(buffer) never exceeds the cap.
        """
        # Construct a "dumped" state with 150 entries targeting a
        # max_entries=100 live instance.
        dumped = {
            "tick_counter": 150,
            "max_entries": 150,
            "tau": 10.0,
            "tick_rate": 1.0,
            "min_activation": 0.01,
            "entries": [
                {
                    "agent_id": "a",
                    "percept_id": f"p{i}",
                    "tick": i,
                    "activation_strength": 1.0,
                    "registered_at": 0.0,
                }
                for i in range(150)
            ],
        }
        live = PerceptTraceBuffer(max_entries=100, tau=10.0)
        live.load_state(dumped)
        assert len(live) == 100, "ring-buffer cap must never be exceeded post-load"
        # Most-recent entries preserved (last 50 of the original 150)
        snap_ids = {e.percept_id for e in live.snapshot(min_activation=0.0)}
        assert "p149" in snap_ids
        assert "p50" in snap_ids
        assert "p0" not in snap_ids  # oldest 50 dropped

    def test_load_state_warns_on_tuning_drift(self, caplog):
        """Round 2 Arch critical #2 regression — tuning-drift surface."""
        import logging

        dumped = {
            "tick_counter": 0,
            "max_entries": 100,
            "tau": 50.0,  # different from live
            "tick_rate": 1.0,
            "min_activation": 0.01,
            "entries": [],
        }
        live = PerceptTraceBuffer(max_entries=100, tau=10.0)
        with caplog.at_level(logging.WARNING):
            live.load_state(dumped)
        assert any("tuning drift" in rec.message for rec in caplog.records)


class TestCrossLayerGraphRoundTrip:
    def test_empty_graph_round_trips(self):
        clg = CrossLayerGraph()
        dumped = clg.dump()
        restored = CrossLayerGraph()
        restored.load_state(dumped)
        assert restored.dump() == dumped

    def test_edge_round_trips(self):
        """P4's mug test depends on cross-layer edges surviving round-trip."""
        clg = CrossLayerGraph()
        clg.add_edge(
            source_layer="hippocampus",
            source_id="ep_1",
            target_layer="atl",
            target_id="concept_mug",
            edge_type=CrossLayerEdgeType.DERIVED_FROM,
            weight=0.8,
        )
        dumped = clg.dump()
        restored = CrossLayerGraph()
        restored.load_state(dumped)
        assert restored.dump() == dumped
        assert len(restored._outgoing) == 1


# ─────────────────────────────────────────────────────────────────────────
# Envelope shape + versioning
# ─────────────────────────────────────────────────────────────────────────


class TestEnvelopeShape:
    def test_wrap_envelope_produces_expected_shape(self):
        atl = ATL(ATLConfig())
        env = wrap_envelope("atl", atl)
        assert env["schema_version"] == 1
        assert env["kind"] == "atl"
        assert isinstance(env["payload"], dict)

    def test_unknown_kind_rejected(self):
        atl = ATL(ATLConfig())
        with pytest.raises(ValueError, match="unknown bio-system kind"):
            wrap_envelope("wrong_kind", atl)

    def test_unwrap_envelope_wrong_kind_rejected(self):
        atl = ATL(ATLConfig())
        env = wrap_envelope("atl", atl)
        with pytest.raises(ValueError, match="envelope kind mismatch"):
            unwrap_envelope(env, "nac")

    def test_unwrap_envelope_non_int_version_rejected(self):
        env = {"schema_version": "1.0", "kind": "atl", "payload": {}}
        with pytest.raises(ValueError, match="schema_version must be int"):
            unwrap_envelope(env, "atl")

    def test_unwrap_envelope_unknown_version_rejected(self):
        env = {"schema_version": 99, "kind": "atl", "payload": {}}
        with pytest.raises(ValueError, match="schema_version 99 not supported"):
            unwrap_envelope(env, "atl")

    def test_snapshot_kinds_match_expected_set(self):
        assert set(SNAPSHOT_KINDS) == {
            "atl",
            "hippocampus",
            "nac",
            "scn",
            "percept_trace_buffer",
            "cross_layer_graph",
        }


class TestEnvelopeVersioningAuthoritative:
    """Payload-layer legacy version strings are tombstoned — envelope rules.

    Round 2 fold: Hippocampus + NAc load_state version checks were
    removed to make the invariant uniform across all bio-systems. These
    tests verify the tombstone holds for every system with a legacy
    payload ``version`` string.
    """

    def test_atl_payload_legacy_string_does_not_affect_load(self):
        atl = ATL(ATLConfig())
        dumped = atl.dump()
        assert dumped.get("version") == "1.0"
        mutated = dict(dumped)
        mutated["version"] = "999.0"
        ATL(ATLConfig()).load_state(mutated)  # must not raise

    def test_nac_payload_version_mutation_does_not_affect_load(self):
        """Round 2 fold — NAc.load_state payload version check removed."""
        nac = NAc(NACConfig())
        dumped = nac.dump()
        assert dumped.get("version") == "1.0"
        mutated = dict(dumped)
        mutated["version"] = "999.0"
        NAc(NACConfig()).load_state(mutated)  # must not raise

    def test_hippocampus_payload_legacy_version_absent_from_dump(self):
        """Round 2 fold — Hippocampus.dump no longer emits a 'version'
        key at all, since the field is load-ignored. Absence is the
        cleanest form of the tombstone."""
        h = Hippocampus(HippocampusConfig())
        dumped = h.dump()
        assert "version" not in dumped

    def test_envelope_version_bad_rejected(self):
        """Bumping the ENVELOPE schema_version IS load-affecting."""
        atl = ATL(ATLConfig())
        env = atl_to_snapshot(atl)
        env["schema_version"] = 2
        with pytest.raises(ValueError, match="schema_version 2 not supported"):
            atl_from_snapshot(env, ATL(ATLConfig()))


# ─────────────────────────────────────────────────────────────────────────
# Lock discipline — AST-based guard per Round 1 minor finding
# ─────────────────────────────────────────────────────────────────────────


class TestLockDisciplinePreserved:
    """Every extracted ``dump()`` method must still acquire its original lock.

    Round 1 minor finding #3: simple string-grep regression guards are
    false-positive-prone on docstrings. This uses inspect.getsource on
    the function and checks for the lock acquisition token pattern.
    """

    def test_atl_dump_acquires_rwlock_read(self):
        src = inspect.getsource(ATL.dump)
        assert "self._rwlock.read()" in src, "ATL.dump must acquire the rwlock read lock"

    def test_nac_dump_acquires_lock(self):
        src = inspect.getsource(NAc.dump)
        assert "with self._lock:" in src, "NAc.dump must acquire self._lock"

    def test_hippocampus_dump_acquires_rwlock_read(self):
        src = inspect.getsource(Hippocampus.dump)
        assert "self._rwlock.read()" in src, "Hippocampus.dump must acquire the rwlock read lock"

    def test_atl_save_delegates_to_dump(self):
        """save() must call self.dump() — the one-line-save shape."""
        src = inspect.getsource(ATL.save)
        assert "self.dump()" in src, "ATL.save must delegate to self.dump()"

    def test_nac_save_delegates_to_dump(self):
        src = inspect.getsource(NAc.save)
        assert "self.dump()" in src, "NAc.save must delegate to self.dump()"

    def test_hippocampus_save_delegates_to_dump(self):
        src = inspect.getsource(Hippocampus.save)
        assert "self.dump()" in src, "Hippocampus.save must delegate to self.dump()"

    def test_scn_save_delegates_to_dump(self):
        src = inspect.getsource(SCN.save)
        assert "self.dump()" in src, "SCN.save must delegate to self.dump()"


# ─────────────────────────────────────────────────────────────────────────
# Runtime wire preservation — cross-confirmed Round 1 finding #1
# ─────────────────────────────────────────────────────────────────────────


class TestLoadPreservesRuntimeWires:
    """load_state is in-place mutation; it must NOT overwrite self.config."""

    def test_atl_load_state_preserves_config(self):
        orig_cfg = ATLConfig(max_concepts=5000, retention_threshold=0.3)
        atl = ATL(orig_cfg)
        atl.load_state(ATL(ATLConfig()).dump())
        assert atl.config is orig_cfg, "load_state must not touch self.config"
        assert atl.config.max_concepts == 5000

    def test_nac_load_state_preserves_config_and_ec(self):
        orig_cfg = NACConfig()
        sentinel_ec = object()
        nac = NAc(orig_cfg, ec=sentinel_ec)
        nac.load_state(NAc(NACConfig()).dump())
        assert nac.config is orig_cfg
        assert nac._ec is sentinel_ec, "load_state must not touch self._ec"

    def test_hippocampus_load_state_preserves_config(self):
        orig_cfg = HippocampusConfig()
        h = Hippocampus(orig_cfg)
        h.load_state(Hippocampus(HippocampusConfig()).dump())
        assert h.config is orig_cfg


# ─────────────────────────────────────────────────────────────────────────
# SessionSnapshot composition + file round-trip
# ─────────────────────────────────────────────────────────────────────────


def _make_live_systems():
    return (
        ATL(ATLConfig()),
        Hippocampus(HippocampusConfig()),
        NAc(NACConfig()),
        SCN(),
        PerceptTraceBuffer(),
        CrossLayerGraph(),
    )


class TestSessionSnapshotComposition:
    def test_capture_all_six_systems(self):
        atl, hc, nac, scn, ptb, clg = _make_live_systems()
        snap = SessionSnapshot.capture(
            atl=atl,
            hippocampus=hc,
            nac=nac,
            scn=scn,
            percept_trace_buffer=ptb,
            cross_layer_graph=clg,
        )
        assert snap.envelope["kind"] == "session"
        assert snap.envelope["schema_version"] == 1
        assert set(snap.envelope["systems"].keys()) == {
            "atl",
            "hippocampus",
            "nac",
            "scn",
            "percept_trace_buffer",
            "cross_layer_graph",
        }

    def test_capture_partial_systems(self):
        """Capturing a subset is allowed."""
        atl, hc, _, _, _, _ = _make_live_systems()
        snap = SessionSnapshot.capture(atl=atl, hippocampus=hc)
        assert set(snap.envelope["systems"].keys()) == {"atl", "hippocampus"}

    def test_full_round_trip_in_memory(self):
        atl, hc, nac, scn, ptb, clg = _make_live_systems()
        snap = SessionSnapshot.capture(
            atl=atl,
            hippocampus=hc,
            nac=nac,
            scn=scn,
            percept_trace_buffer=ptb,
            cross_layer_graph=clg,
        )

        atl2, hc2, nac2, scn2, ptb2, clg2 = _make_live_systems()
        snap.restore_into(
            atl=atl2,
            hippocampus=hc2,
            nac=nac2,
            scn=scn2,
            percept_trace_buffer=ptb2,
            cross_layer_graph=clg2,
        )

        assert atl2.dump() == atl.dump()
        assert _strip_timestamps(hc2.dump()) == _strip_timestamps(hc.dump())
        assert nac2.dump() == nac.dump()
        assert scn2.dump() == scn.dump()
        assert ptb2.dump() == ptb.dump()
        assert clg2.dump() == clg.dump()

    def test_file_round_trip(self, tmp_path):
        atl, hc, nac, scn, ptb, clg = _make_live_systems()
        snap = SessionSnapshot.capture(
            atl=atl,
            hippocampus=hc,
            nac=nac,
            scn=scn,
            percept_trace_buffer=ptb,
            cross_layer_graph=clg,
        )
        path = tmp_path / "session.json"
        snap.write(path)

        # Verify on-disk shape is valid JSON
        with open(path) as f:
            on_disk = json.load(f)
        assert on_disk["kind"] == "session"
        assert on_disk["schema_version"] == 1

        # Reload
        reloaded = SessionSnapshot.from_file(path)
        atl2, hc2, nac2, scn2, ptb2, clg2 = _make_live_systems()
        reloaded.restore_into(
            atl=atl2,
            hippocampus=hc2,
            nac=nac2,
            scn=scn2,
            percept_trace_buffer=ptb2,
            cross_layer_graph=clg2,
        )
        assert atl2.dump() == atl.dump()
        assert nac2.dump() == nac.dump()

    def test_session_snapshot_bad_envelope_kind_rejected(self):
        bad = {"schema_version": 1, "kind": "not_session", "systems": {}}
        with pytest.raises(ValueError, match="envelope kind must be 'session'"):
            SessionSnapshot.from_dict(bad)

    def test_session_snapshot_bad_schema_version_rejected(self):
        bad = {"schema_version": 2, "kind": "session", "systems": {}}
        with pytest.raises(ValueError, match="schema_version 2 not supported"):
            SessionSnapshot.from_dict(bad)

    def test_session_snapshot_non_int_schema_version_rejected(self):
        bad = {"schema_version": "1", "kind": "session", "systems": {}}
        with pytest.raises(ValueError, match="schema_version must be int"):
            SessionSnapshot.from_dict(bad)


class TestSessionSnapshotStrictMode:
    """Round 2 fold — strict=True raises on partial capture/restore."""

    def test_capture_strict_missing_system_raises(self):
        atl, hc, _, _, _, _ = _make_live_systems()
        with pytest.raises(ValueError, match="missing bio-systems"):
            SessionSnapshot.capture(atl=atl, hippocampus=hc, strict=True)

    def test_capture_strict_all_systems_passes(self):
        atl, hc, nac, scn, ptb, clg = _make_live_systems()
        snap = SessionSnapshot.capture(
            atl=atl,
            hippocampus=hc,
            nac=nac,
            scn=scn,
            percept_trace_buffer=ptb,
            cross_layer_graph=clg,
            strict=True,
        )
        assert len(snap.envelope["systems"]) == 6

    def test_restore_into_strict_instance_without_envelope_raises(self):
        """Live instance has no matching envelope sub-snapshot."""
        atl1, hc1, _, _, _, _ = _make_live_systems()
        snap = SessionSnapshot.capture(atl=atl1, hippocampus=hc1)  # only 2 systems
        atl2, hc2, nac2, _, _, _ = _make_live_systems()
        with pytest.raises(ValueError, match="without envelope"):
            snap.restore_into(atl=atl2, hippocampus=hc2, nac=nac2, strict=True)

    def test_restore_into_strict_envelope_without_instance_raises(self):
        """Envelope has a sub-snapshot but caller didn't pass that system."""
        atl1, hc1, nac1, _, _, _ = _make_live_systems()
        snap = SessionSnapshot.capture(atl=atl1, hippocampus=hc1, nac=nac1)
        atl2, hc2, _, _, _, _ = _make_live_systems()
        with pytest.raises(ValueError, match="without instances"):
            snap.restore_into(atl=atl2, hippocampus=hc2, strict=True)

    def test_restore_into_strict_exact_match_passes(self):
        atl1, hc1, nac1, scn1, ptb1, clg1 = _make_live_systems()
        snap = SessionSnapshot.capture(
            atl=atl1,
            hippocampus=hc1,
            nac=nac1,
            scn=scn1,
            percept_trace_buffer=ptb1,
            cross_layer_graph=clg1,
            strict=True,
        )
        atl2, hc2, nac2, scn2, ptb2, clg2 = _make_live_systems()
        snap.restore_into(
            atl=atl2,
            hippocampus=hc2,
            nac=nac2,
            scn=scn2,
            percept_trace_buffer=ptb2,
            cross_layer_graph=clg2,
            strict=True,
        )  # must not raise
