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

import contextlib
import copy
import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from maxim.utils.atomic_io import atomic_write_json

logger = logging.getLogger(__name__)

# Stage 2: latest envelope schema version. Bump in lockstep with a registered
# migration in ``_SESSION_MIGRATIONS`` below. ``unwrap_envelope`` /
# ``SessionSnapshot._validate_envelope`` accept any version ≤ this and run the
# migration chain to upgrade in place.
LATEST_SESSION_SCHEMA_VERSION: int = 1
LATEST_SUBSYSTEM_SCHEMA_VERSION: int = 1

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

    Raises ``ValueError`` if ``kind`` is wrong. Older sub-system envelopes
    are upgraded to ``LATEST_SUBSYSTEM_SCHEMA_VERSION`` via the
    ``_SUBSYSTEM_MIGRATIONS`` chain (Stage 2+).
    """
    kind = envelope.get("kind")
    if kind != expected_kind:
        raise ValueError(f"envelope kind mismatch: expected {expected_kind!r}, got {kind!r}")

    upgraded = migrate_subsystem_envelope(envelope, target_version=LATEST_SUBSYSTEM_SCHEMA_VERSION)

    payload = upgraded.get("payload")
    if not isinstance(payload, dict):
        raise ValueError(f"envelope payload must be dict, got {type(payload).__name__}")
    return payload


# ─────────────────────────────────────────────────────────────────────────
# Migration registries (Stage 2)
#
# A migration is a pure function that takes an envelope at version N and
# returns the equivalent envelope at version N+1. ``_run_migration_chain``
# walks the registry one step at a time. Unknown source versions raise
# ``ValueError`` — the registry is the single source of truth for what
# the loader can read.
# ─────────────────────────────────────────────────────────────────────────


MigrationFn = Callable[[dict[str, Any]], dict[str, Any]]

# Session-level migrations. Keyed by the FROM version; each value
# upgrades to FROM + 1. Empty in Stage 2 because LATEST_SESSION_SCHEMA_VERSION
# is 1 and there is no v0-in-the-wild — but a synthetic legacy fixture is
# registered below for the migration test harness.
_SESSION_MIGRATIONS: dict[int, MigrationFn] = {}

# Sub-system envelope migrations. Same shape as session-level.
_SUBSYSTEM_MIGRATIONS: dict[int, MigrationFn] = {}


def _run_migration_chain(
    envelope: dict[str, Any],
    *,
    target_version: int,
    registry: dict[int, MigrationFn],
    label: str,
) -> dict[str, Any]:
    """Walk ``registry`` one step per call until envelope reaches target_version.

    Raises ``ValueError`` on missing migration or unreachable target.

    **Aliasing contract (Stage 2 review I1+M5 fold).** The caller's
    envelope dict is NEVER mutated and the return value is NEVER the
    same dict object as the input — including nested values. The chain
    deep-copies at the entry so callers that mutate the result post-call
    cannot corrupt their input envelope or any of its nested payloads.
    Migration is a process-boundary operation, not a hot path, so
    ``copy.deepcopy`` cost is acceptable.
    """
    if not isinstance(envelope, dict):
        raise ValueError(f"{label} migration input must be dict, got {type(envelope).__name__}")
    version = envelope.get("schema_version")
    if not isinstance(version, int):
        raise ValueError(f"{label} envelope schema_version must be int, got {type(version).__name__}: {version!r}")
    if version > target_version:
        raise ValueError(
            f"{label} envelope schema_version {version} is newer than this build's "
            f"latest ({target_version}) — refusing to downgrade"
        )

    # Deep-copy at the entry so the no-op path also returns a fully
    # caller-independent dict (Stage 2 review I1+M5 fold).
    current = copy.deepcopy(envelope)
    while True:
        cur_version = current["schema_version"]
        if cur_version == target_version:
            return current
        migration = registry.get(cur_version)
        if migration is None:
            raise ValueError(
                f"{label} envelope schema_version {cur_version} has no migration to {cur_version + 1} "
                f"(registry keys: {sorted(registry.keys())})"
            )
        upgraded = migration(current)
        if not isinstance(upgraded, dict):
            raise ValueError(
                f"{label} migration {cur_version}→{cur_version + 1} returned non-dict: {type(upgraded).__name__}"
            )
        next_version = upgraded.get("schema_version")
        if not isinstance(next_version, int):
            raise ValueError(
                f"{label} migration {cur_version}→{cur_version + 1} produced "
                f"non-int schema_version: {type(next_version).__name__}: {next_version!r}"
            )
        if next_version != cur_version + 1:
            raise ValueError(
                f"{label} migration {cur_version}→{cur_version + 1} produced schema_version {next_version!r}"
            )
        current = upgraded


def migrate_session_envelope(envelope: dict[str, Any], *, target_version: int | None = None) -> dict[str, Any]:
    """Upgrade a session-level envelope through the migration chain.

    Pure forward migration. Not in-place — returns a new dict at each step.
    """
    if target_version is None:
        target_version = LATEST_SESSION_SCHEMA_VERSION
    return _run_migration_chain(
        envelope,
        target_version=target_version,
        registry=_SESSION_MIGRATIONS,
        label="session",
    )


def migrate_subsystem_envelope(envelope: dict[str, Any], *, target_version: int | None = None) -> dict[str, Any]:
    """Upgrade a sub-system envelope through the migration chain."""
    if target_version is None:
        target_version = LATEST_SUBSYSTEM_SCHEMA_VERSION
    return _run_migration_chain(
        envelope,
        target_version=target_version,
        registry=_SUBSYSTEM_MIGRATIONS,
        label="subsystem",
    )


def register_session_migration(from_version: int) -> Callable[[MigrationFn], MigrationFn]:
    """Decorator: register a session-envelope migration ``from_version → from_version+1``."""

    def _decorator(fn: MigrationFn) -> MigrationFn:
        if from_version in _SESSION_MIGRATIONS:
            raise ValueError(f"session migration from version {from_version} already registered")
        _SESSION_MIGRATIONS[from_version] = fn
        return fn

    return _decorator


def register_subsystem_migration(from_version: int) -> Callable[[MigrationFn], MigrationFn]:
    """Decorator: register a sub-system envelope migration ``from_version → from_version+1``.

    **Lockstep contract (Stage 2 review I3 fold).** The sub-system
    registry is **kind-agnostic** — a single integer keys the entire
    family of sub-system envelopes. This means when one sub-system
    bumps its envelope shape (e.g., NAc payload restructure warranting
    v2), `LATEST_SUBSYSTEM_SCHEMA_VERSION` bumps to 2 and EVERY
    sub-system's adapter functions must accept v2 envelopes (typically
    via a no-op pass-through migration for the systems that didn't
    actually change). The migration function itself can branch on
    ``envelope["kind"]`` if it needs per-kind logic. This avoids the
    `(kind, version)` tuple registry that would otherwise be needed,
    at the cost of forcing all six sub-systems to bump in lockstep.
    Mother Maxim's per-deployment overrides are deferred to post-1.0.
    """

    def _decorator(fn: MigrationFn) -> MigrationFn:
        if from_version in _SUBSYSTEM_MIGRATIONS:
            raise ValueError(f"subsystem migration from version {from_version} already registered")
        _SUBSYSTEM_MIGRATIONS[from_version] = fn
        return fn

    return _decorator


@contextlib.contextmanager
def isolated_migrations() -> Iterator[None]:
    """Context manager: snapshot + clear + restore both migration registries.

    Stage 2 review I2 fold — moves the test-isolation pattern out of
    private-global mutation in test fixtures and into a stable public
    API. Tests that need to register synthetic migrations should wrap
    their setup in this context so:

    1. They start from a CLEAN registry (no risk of duplicate-source
       collision against a real production migration registered at
       module import time — this is the M1 hardening).
    2. The pre-test registry is restored on exit even if the test
       raises mid-way.

    Usage::

        with isolated_migrations():
            @register_session_migration(0)
            def _v0_to_v1(env): ...
            assert migrate_session_envelope(legacy)["schema_version"] == 1

    A future refactor that puts the registry inside a class only
    needs to update this one function; tests stay stable.
    """
    saved_session = dict(_SESSION_MIGRATIONS)
    saved_subsystem = dict(_SUBSYSTEM_MIGRATIONS)
    _SESSION_MIGRATIONS.clear()
    _SUBSYSTEM_MIGRATIONS.clear()
    try:
        yield
    finally:
        _SESSION_MIGRATIONS.clear()
        _SESSION_MIGRATIONS.update(saved_session)
        _SUBSYSTEM_MIGRATIONS.clear()
        _SUBSYSTEM_MIGRATIONS.update(saved_subsystem)


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

        from maxim.utils.format_version import FORMAT_VERSION

        return cls(
            envelope={
                "_format_version": FORMAT_VERSION,
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

        **Atomicity (Stage 2 review C1 fold).** ``restore_into`` is
        all-or-nothing across the six sub-systems. Before mutating any
        target, every target's CURRENT state is captured via its own
        ``dump()`` into a rollback table. If any sub-system's
        ``load_state`` raises (malformed payload, internal invariant
        violation, migration error), every already-mutated target is
        restored from its rollback dump before the exception propagates.
        Required because P4 mug-test and P5 stress persistence both
        consume this in hot loops where a torn restore would brick a
        multi-agent session with no indication of which systems landed.

        Rollback failure is logged via ``logger.exception`` and the
        original exception is re-raised. Two failures in a single
        restore is a "your bio-system is broken" condition and is
        loud-by-design — no swallowing.

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

        provided: dict[str, Any] = {
            "atl": atl,
            "hippocampus": hippocampus,
            "nac": nac,
            "scn": scn,
            "percept_trace_buffer": percept_trace_buffer,
            "cross_layer_graph": cross_layer_graph,
        }

        if strict:
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

        # Build the apply list — only kinds that have BOTH an instance
        # and an envelope entry. Order matches SNAPSHOT_KINDS for
        # determinism so review tests can pin "ATL applies before NAc."
        adapter_table: dict[str, tuple[Any, Any]] = {
            "atl": (atl, atl_from_snapshot),
            "hippocampus": (hippocampus, hippocampus_from_snapshot),
            "nac": (nac, nac_from_snapshot),
            "scn": (scn, scn_from_snapshot),
            "percept_trace_buffer": (percept_trace_buffer, ptb_from_snapshot),
            "cross_layer_graph": (cross_layer_graph, cross_layer_graph_from_snapshot),
        }
        apply_list: list[tuple[str, Any, Any, dict[str, Any]]] = []
        for kind in SNAPSHOT_KINDS:
            instance, adapter = adapter_table[kind]
            if instance is None or kind not in systems:
                continue
            apply_list.append((kind, instance, adapter, systems[kind]))

        # Phase 1: snapshot every target's CURRENT state for rollback.
        # Done up-front so a dump() failure on a wedged instance
        # surfaces BEFORE any state is mutated.
        rollback: list[tuple[str, Any, dict[str, Any]]] = []
        for kind, instance, _adapter, _envelope in apply_list:
            try:
                rollback.append((kind, instance, instance.dump()))
            except Exception as e:
                raise RuntimeError(
                    f"SessionSnapshot.restore_into: failed to capture rollback state for {kind!r} "
                    f"before any mutation; aborting cleanly. Underlying error: {e}"
                ) from e

        # Phase 2: apply each adapter. On any failure, walk the
        # already-mutated prefix in reverse and restore from rollback.
        applied: list[tuple[str, Any]] = []  # (kind, instance) pairs that were mutated
        try:
            for kind, instance, adapter, env in apply_list:
                adapter(env, instance)
                applied.append((kind, instance))
        except Exception as primary_exc:
            # Rollback in reverse-applied order. For every
            # already-mutated target, look up its pre-mutation dump and
            # call load_state to restore.
            rollback_index = {kind: state for kind, _inst, state in rollback}
            rollback_failures: list[str] = []
            for kind, instance in reversed(applied):
                try:
                    instance.load_state(rollback_index[kind])
                except Exception as rollback_exc:
                    rollback_failures.append(f"{kind}: {rollback_exc}")
            if rollback_failures:
                logger.exception(
                    "SessionSnapshot.restore_into: rollback FAILED for %s after primary failure; "
                    "bio-systems are in a torn state. Primary error follows.",
                    rollback_failures,
                )
            raise primary_exc

    def dump(self) -> dict[str, Any]:
        """Return a deep copy of the envelope dict for external serialization.

        Stage 2 review M5 fold — returning ``self.envelope`` by reference
        let callers mutate snapshot state by accident. ``deepcopy`` is
        cheap relative to the size of bio-system payloads.
        """
        return copy.deepcopy(self.envelope)

    def write(self, path: str | Path) -> None:
        """Persist this snapshot to disk atomically."""
        atomic_write_json(str(path), self.envelope)

    @classmethod
    def from_dict(cls, envelope: dict[str, Any]) -> SessionSnapshot:
        # Stage 2: upgrade older session envelopes through the migration
        # chain before validating. ``migrate_session_envelope`` is a no-op
        # when the input is already at LATEST_SESSION_SCHEMA_VERSION.
        if not isinstance(envelope, dict):
            raise ValueError(f"envelope must be dict, got {type(envelope).__name__}")
        if envelope.get("kind") != "session":
            raise ValueError(f"envelope kind must be 'session', got {envelope.get('kind')!r}")
        upgraded = migrate_session_envelope(envelope)
        snapshot = cls(envelope=upgraded)
        snapshot._validate_envelope()
        return snapshot

    @classmethod
    def from_file(cls, path: str | Path) -> SessionSnapshot:
        with open(path, encoding="utf-8") as f:
            envelope = json.load(f)
        from maxim.utils.format_version import check_format_version

        check_format_version(envelope, "session_snapshot", log=logger)
        return cls.from_dict(envelope)

    def _validate_envelope(self) -> None:
        if not isinstance(self.envelope, dict):
            raise ValueError(f"envelope must be dict, got {type(self.envelope).__name__}")
        if self.envelope.get("kind") != "session":
            raise ValueError(f"envelope kind must be 'session', got {self.envelope.get('kind')!r}")
        version = self.envelope.get("schema_version")
        if not isinstance(version, int):
            raise ValueError(f"envelope schema_version must be int, got {type(version).__name__}: {version!r}")
        if version != LATEST_SESSION_SCHEMA_VERSION:
            raise ValueError(
                f"SessionSnapshot schema_version {version} != latest "
                f"{LATEST_SESSION_SCHEMA_VERSION} — call SessionSnapshot.from_dict to auto-migrate"
            )
        systems = self.envelope.get("systems")
        if not isinstance(systems, dict):
            raise ValueError(f"envelope systems must be dict, got {type(systems).__name__}")


__all__ = [
    "LATEST_SESSION_SCHEMA_VERSION",
    "LATEST_SUBSYSTEM_SCHEMA_VERSION",
    "BioSystemSnapshot",
    "SNAPSHOT_KINDS",
    "SessionSnapshot",
    "atl_from_snapshot",
    "atl_to_snapshot",
    "cross_layer_graph_from_snapshot",
    "cross_layer_graph_to_snapshot",
    "hippocampus_from_snapshot",
    "hippocampus_to_snapshot",
    "isolated_migrations",
    "migrate_session_envelope",
    "migrate_subsystem_envelope",
    "nac_from_snapshot",
    "nac_to_snapshot",
    "ptb_from_snapshot",
    "ptb_to_snapshot",
    "register_session_migration",
    "register_subsystem_migration",
    "scn_from_snapshot",
    "scn_to_snapshot",
    "unwrap_envelope",
    "wrap_envelope",
]
