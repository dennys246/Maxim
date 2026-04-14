"""BioSystemSnapshot Protocol + SessionSnapshot composition (P3.5 Stage 1).

This module provides a unified serialization-shape contract for all bio-
systems that need to round-trip state across process boundaries (P3a
persistence round-trip, P4 cross-modal mug test subprocess, P5 stress
persistence, future sim checkpoint/restore).

Protocol vs storage-target protocols
====================================

This Protocol (``BioSystemSnapshot``) describes the **serialization
shape** a live bio-system produces for an in-memory snapshot. It is
orthogonal to ``maxim.memory.store.EpisodicStore`` / ``CausalStore`` /
``SemanticStore``, which describe **storage backends** (where persistence
writes — filesystem, database, ...). The two compose: a Mother Maxim
deployment that wants SessionSnapshot state to land in Postgres wraps
the bio-system in a DB-backed storage target AND implements this
Protocol for the snapshot shape. Neither replaces the other.

Envelope-authoritative versioning
=================================

Every sub-snapshot is wrapped in an envelope:

    {"schema_version": 1, "kind": "atl", "payload": <bio-system dict>}

The ``schema_version`` at the ENVELOPE layer is the ONLY authoritative
version. Payload-layer legacy version strings (ATL's ``"1.0"``,
Hippocampus's ``"3.0"``, NAc's ``"1.0"``, SCN's ``"3.0"``,
CrossLayerGraph's ``"1.0"``) are **TOMBSTONED** — no new migrations bump
them. All future forward-migration logic lives at the envelope layer
via ``SessionSnapshot.migrate`` (Stage 2+). Bumping both layers in the
same change would create an ambiguous contract that Stage 2 migration
tooling cannot reason about.

Load semantics: in-place mutation
=================================

``BioSystemSnapshot.load_state(state)`` is **instance-level and mutating**.
It does NOT construct a fresh instance. This matches the existing
``bio_system.save/load(path)`` contract and preserves runtime wires
(ATL config + semantics callbacks, NAc.ec, Hippocampus scn reference,
SCN persistence_path, CrossLayerGraph._layers). A classmethod factory
shape was rejected during P3.5 plan review because it cannot accept the
required init params on ATL/NAc/Hippocampus and cannot re-establish
runtime-only wires.

Method naming
=============

The Protocol uses ``load_state`` (not ``load``) to avoid colliding with
the pre-existing ``load(path: str | None)`` method that every bio-system
already exports for filesystem I/O. Renaming those path-based methods
across 37+ call sites is out of scope for Stage 1. Both methods coexist:
``save(path)`` → ``atomic_write_json(path, self.dump())`` and
``load(path)`` → ``self.load_state(json.load(open(path)))``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from maxim.utils.atomic_io import atomic_write_json

if TYPE_CHECKING:
    from maxim.decisions.nac import NAc
    from maxim.memory.atl import ATL
    from maxim.memory.cross_layer import CrossLayerGraph
    from maxim.memory.hippocampus import Hippocampus
    from maxim.memory.percept_trace_buffer import PerceptTraceBuffer
    from maxim.time.scn import SCN


# ─────────────────────────────────────────────────────────────────────────
# Protocol
# ─────────────────────────────────────────────────────────────────────────


@runtime_checkable
class BioSystemSnapshot(Protocol):
    """Protocol for bio-systems that can dump/load their state as a dict.

    Six bio-systems conform in P3.5 Stage 1:

    - ``maxim.memory.atl.ATL``
    - ``maxim.memory.hippocampus.Hippocampus`` (via ``PersistenceMixin``)
    - ``maxim.decisions.nac.NAc``
    - ``maxim.time.scn.SCN``
    - ``maxim.memory.percept_trace_buffer.PerceptTraceBuffer``
    - ``maxim.memory.cross_layer.CrossLayerGraph``

    Locking caveat (Round 2 Arch important #5): ``dump()`` on most
    bio-systems acquires the bio-system's internal read lock (ATL
    rwlock, NAc mutex, Hippocampus rwlock, PerceptTraceBuffer mutex)
    to produce a point-in-time consistent dict. Callers that want to
    snapshot a running system from a hot-path thread should spawn the
    ``dump()`` call on a worker thread rather than blocking the hot
    path on lock contention. ``load_state()`` similarly acquires write
    locks where applicable; callers must ensure no concurrent writers
    are active against the target bio-system before calling.
    """

    schema_version: int

    def dump(self) -> dict[str, Any]: ...

    def load_state(self, state: dict[str, Any]) -> None: ...


# ─────────────────────────────────────────────────────────────────────────
# Envelope helpers
# ─────────────────────────────────────────────────────────────────────────


SNAPSHOT_KINDS: tuple[str, ...] = (
    "atl",
    "hippocampus",
    "nac",
    "scn",
    "percept_trace_buffer",
    "cross_layer_graph",
)


def wrap_envelope(kind: str, system: BioSystemSnapshot) -> dict[str, Any]:
    """Wrap a bio-system's dump in the P3.5 envelope shape.

    Shape: ``{"schema_version": 1, "kind": kind, "payload": system.dump()}``

    The envelope version is always ``1`` in Stage 1. Stage 2+ migration
    may bump this.
    """
    if kind not in SNAPSHOT_KINDS:
        raise ValueError(f"unknown bio-system kind: {kind!r} (expected one of {SNAPSHOT_KINDS})")
    return {
        "schema_version": system.schema_version,
        "kind": kind,
        "payload": system.dump(),
    }


def unwrap_envelope(envelope: dict[str, Any], expected_kind: str) -> dict[str, Any]:
    """Verify envelope shape and return the payload dict.

    Raises ``ValueError`` if ``kind`` or ``schema_version`` are wrong.
    Stage 2+ migration tooling will hook here to upgrade payloads before
    returning.
    """
    kind = envelope.get("kind")
    if kind != expected_kind:
        raise ValueError(f"envelope kind mismatch: expected {expected_kind!r}, got {kind!r}")

    version = envelope.get("schema_version")
    if not isinstance(version, int):
        raise ValueError(f"envelope schema_version must be int, got {type(version).__name__}: {version!r}")
    if version != 1:
        raise ValueError(f"envelope schema_version {version} not supported in Stage 1 (migration lands in Stage 2)")

    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise ValueError(f"envelope payload must be dict, got {type(payload).__name__}")
    return payload


# ─────────────────────────────────────────────────────────────────────────
# Per-bio-system adapter functions (symmetric dump/load pairs)
#
# These are thin wrappers that the SessionSnapshot orchestrator uses to
# drive each bio-system through the envelope. They also serve as a stable
# seam for Stage 2 migration functions to hook into.
# ─────────────────────────────────────────────────────────────────────────


def atl_to_snapshot(atl: ATL) -> dict[str, Any]:
    return wrap_envelope("atl", atl)


def atl_from_snapshot(envelope: dict[str, Any], into: ATL) -> None:
    into.load_state(unwrap_envelope(envelope, "atl"))


def hippocampus_to_snapshot(hippocampus: Hippocampus) -> dict[str, Any]:
    return wrap_envelope("hippocampus", hippocampus)


def hippocampus_from_snapshot(envelope: dict[str, Any], into: Hippocampus) -> None:
    into.load_state(unwrap_envelope(envelope, "hippocampus"))


def nac_to_snapshot(nac: NAc) -> dict[str, Any]:
    return wrap_envelope("nac", nac)


def nac_from_snapshot(envelope: dict[str, Any], into: NAc) -> None:
    into.load_state(unwrap_envelope(envelope, "nac"))


def scn_to_snapshot(scn: SCN) -> dict[str, Any]:
    return wrap_envelope("scn", scn)


def scn_from_snapshot(envelope: dict[str, Any], into: SCN) -> None:
    into.load_state(unwrap_envelope(envelope, "scn"))


def ptb_to_snapshot(ptb: PerceptTraceBuffer) -> dict[str, Any]:
    return wrap_envelope("percept_trace_buffer", ptb)


def ptb_from_snapshot(envelope: dict[str, Any], into: PerceptTraceBuffer) -> None:
    into.load_state(unwrap_envelope(envelope, "percept_trace_buffer"))


def cross_layer_graph_to_snapshot(clg: CrossLayerGraph) -> dict[str, Any]:
    return wrap_envelope("cross_layer_graph", clg)


def cross_layer_graph_from_snapshot(envelope: dict[str, Any], into: CrossLayerGraph) -> None:
    into.load_state(unwrap_envelope(envelope, "cross_layer_graph"))


# ─────────────────────────────────────────────────────────────────────────
# SessionSnapshot
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class SessionSnapshot:
    """Composed snapshot across all six P3.5 bio-systems.

    Constructed either from a live set of bio-system instances via
    ``SessionSnapshot.capture(...)`` or from a JSON file via
    ``SessionSnapshot.from_file(path)``. Loaded back into a fresh set of
    wired instances via ``restore_into(...)``.

    Stage 1 ships full round-trip for all six systems. Stage 2 adds
    migration tooling + subprocess round-trip harness.
    """

    envelope: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def capture(
        cls,
        *,
        atl: ATL | None = None,
        hippocampus: Hippocampus | None = None,
        nac: NAc | None = None,
        scn: SCN | None = None,
        percept_trace_buffer: PerceptTraceBuffer | None = None,
        cross_layer_graph: CrossLayerGraph | None = None,
        strict: bool = False,
    ) -> SessionSnapshot:
        """Build a SessionSnapshot from live bio-system instances.

        All bio-system arguments are keyword-only and optional. A
        system that is not provided is omitted from the envelope's
        ``systems`` dict.

        When ``strict=False`` (default), any subset of systems can be
        provided and the snapshot captures only those. This supports
        debug / single-system checkpoint workflows.

        When ``strict=True``, ALL six bio-systems must be provided;
        any missing system raises ``ValueError``. Use this for P4
        mug-test + P5 stress harness call sites that MUST capture the
        full session — Round 2 Exec + Arch cross-confirmed finding.
        """
        if strict:
            missing = [
                name
                for name, obj in (
                    ("atl", atl),
                    ("hippocampus", hippocampus),
                    ("nac", nac),
                    ("scn", scn),
                    ("percept_trace_buffer", percept_trace_buffer),
                    ("cross_layer_graph", cross_layer_graph),
                )
                if obj is None
            ]
            if missing:
                raise ValueError(f"SessionSnapshot.capture(strict=True) missing bio-systems: {missing}")

        systems: dict[str, dict[str, Any]] = {}
        if atl is not None:
            systems["atl"] = atl_to_snapshot(atl)
        if hippocampus is not None:
            systems["hippocampus"] = hippocampus_to_snapshot(hippocampus)
        if nac is not None:
            systems["nac"] = nac_to_snapshot(nac)
        if scn is not None:
            systems["scn"] = scn_to_snapshot(scn)
        if percept_trace_buffer is not None:
            systems["percept_trace_buffer"] = ptb_to_snapshot(percept_trace_buffer)
        if cross_layer_graph is not None:
            systems["cross_layer_graph"] = cross_layer_graph_to_snapshot(cross_layer_graph)

        return cls(
            envelope={
                "schema_version": 1,
                "kind": "session",
                "systems": systems,
            }
        )

    def restore_into(
        self,
        *,
        atl: ATL | None = None,
        hippocampus: Hippocampus | None = None,
        nac: NAc | None = None,
        scn: SCN | None = None,
        percept_trace_buffer: PerceptTraceBuffer | None = None,
        cross_layer_graph: CrossLayerGraph | None = None,
        strict: bool = False,
    ) -> None:
        """Mutate the provided live bio-systems in place from this snapshot.

        Each instance MUST be already constructed + wired; this method
        only replaces state, not identity.

        When ``strict=False`` (default), missing sub-snapshot for a
        provided instance is silently skipped, and missing instance
        for a present sub-snapshot is also silently skipped. This
        matches the Stage 1 debug-friendly contract.

        When ``strict=True``, both mismatches raise ``ValueError``:
        every provided instance must have a matching sub-snapshot, and
        every sub-snapshot in the envelope must have a matching
        instance. Use this for P4 / P5 harness call sites.
        """
        self._validate_envelope()
        systems = self.envelope.get("systems", {})

        if strict:
            provided: dict[str, Any] = {
                "atl": atl,
                "hippocampus": hippocampus,
                "nac": nac,
                "scn": scn,
                "percept_trace_buffer": percept_trace_buffer,
                "cross_layer_graph": cross_layer_graph,
            }
            provided_keys = {name for name, obj in provided.items() if obj is not None}
            envelope_keys = set(systems.keys())
            missing_snapshots = provided_keys - envelope_keys
            missing_instances = envelope_keys - provided_keys
            if missing_snapshots or missing_instances:
                raise ValueError(
                    "SessionSnapshot.restore_into(strict=True) mismatch: "
                    f"instances without envelope={sorted(missing_snapshots)}, "
                    f"envelope without instances={sorted(missing_instances)}"
                )

        if atl is not None and "atl" in systems:
            atl_from_snapshot(systems["atl"], atl)
        if hippocampus is not None and "hippocampus" in systems:
            hippocampus_from_snapshot(systems["hippocampus"], hippocampus)
        if nac is not None and "nac" in systems:
            nac_from_snapshot(systems["nac"], nac)
        if scn is not None and "scn" in systems:
            scn_from_snapshot(systems["scn"], scn)
        if percept_trace_buffer is not None and "percept_trace_buffer" in systems:
            ptb_from_snapshot(systems["percept_trace_buffer"], percept_trace_buffer)
        if cross_layer_graph is not None and "cross_layer_graph" in systems:
            cross_layer_graph_from_snapshot(systems["cross_layer_graph"], cross_layer_graph)

    def dump(self) -> dict[str, Any]:
        """Return the envelope dict for external serialization."""
        return self.envelope

    def write(self, path: str | Path) -> None:
        """Persist this snapshot to disk atomically."""
        atomic_write_json(str(path), self.envelope)

    @classmethod
    def from_dict(cls, envelope: dict[str, Any]) -> SessionSnapshot:
        snapshot = cls(envelope=envelope)
        snapshot._validate_envelope()
        return snapshot

    @classmethod
    def from_file(cls, path: str | Path) -> SessionSnapshot:
        with open(path, encoding="utf-8") as f:
            envelope = json.load(f)
        return cls.from_dict(envelope)

    def _validate_envelope(self) -> None:
        if not isinstance(self.envelope, dict):
            raise ValueError(f"envelope must be dict, got {type(self.envelope).__name__}")
        if self.envelope.get("kind") != "session":
            raise ValueError(f"envelope kind must be 'session', got {self.envelope.get('kind')!r}")
        version = self.envelope.get("schema_version")
        if not isinstance(version, int):
            raise ValueError(f"envelope schema_version must be int, got {type(version).__name__}: {version!r}")
        if version != 1:
            raise ValueError(
                f"SessionSnapshot schema_version {version} not supported in Stage 1 (migration lands in Stage 2)"
            )
        systems = self.envelope.get("systems")
        if not isinstance(systems, dict):
            raise ValueError(f"envelope systems must be dict, got {type(systems).__name__}")


__all__ = [
    "BioSystemSnapshot",
    "SNAPSHOT_KINDS",
    "SessionSnapshot",
    "atl_from_snapshot",
    "atl_to_snapshot",
    "cross_layer_graph_from_snapshot",
    "cross_layer_graph_to_snapshot",
    "hippocampus_from_snapshot",
    "hippocampus_to_snapshot",
    "nac_from_snapshot",
    "nac_to_snapshot",
    "ptb_from_snapshot",
    "ptb_to_snapshot",
    "scn_from_snapshot",
    "scn_to_snapshot",
    "unwrap_envelope",
    "wrap_envelope",
]
