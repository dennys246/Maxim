"""Runtime drain state for mesh nodes.

Plan 4 Stage C2. This module is the **mutable state layer** for mesh
drain, deliberately separated from the declarative ``mesh.yml``
(topology) surface. The split matches the Kubernetes spec-vs-status
model:

- ``mesh.yml`` is read-only from every Maxim code path — operators
  edit it by hand and restart the daemon to reload topology.
- ``~/.maxim/util/drained_nodes.{role}.txt`` is the mutable drain set,
  written by ``maxim peer --node X drain`` / ``resume`` and read by
  ``maxim peer list-nodes`` / ``maxim doctor check_mesh_nodes``.

This separation is a **load-bearing invariant** (see the CLAUDE.md
lesson "mesh.yml is declarative; ~/.maxim/util/ is mutable state"):
C3's admin API writes to this layer, never to ``mesh.yml``. Keeping
the two layers strictly disjoint means there's no reconciliation
contract to document, no merge-conflict story for a git-tracked
config file, and no way for a non-human writer to corrupt
operator-authored topology.

The four issues pre-merge review CC2 raised against an earlier
drain-state design are each addressed explicitly here:

1. **Role detection timing.** ``MAXIM_ROLE`` is read at drain-state
   path resolution time. The caller (``peer/cli.py``) MUST invoke
   :func:`maxim.runtime.role.detect_and_apply_role` at the top of
   ``run_peer_connect_subcommand`` before any drain/resume verb
   reaches this module. Regression test in
   ``test_drain_state.py::TestRoleIsolation``.

2. **Read/write race.** Every mutation acquires a ``filelock.FileLock``
   on a sibling ``drained_nodes.{role}.txt.lock`` file. Cross-platform
   via the ``filelock`` library (POSIX + Windows).

3. **Orphan validation.** Drain operations MUST pass the current
   mesh's node set so this module can reject unknown names at drain
   time (exit 2 with a known-node list). Read operations (``list``)
   return orphan names alongside the valid drain set so callers can
   surface a warning without hard-failing — operators mid-edit
   shouldn't lose their work.

4. **Permission preservation.** Drain state writes use
   ``atomic_write_text(preserve_mode=True)`` so pre-existing mode
   bits survive round-trips. Drain state itself isn't secret, but the
   preserved-mode flag is infrastructure shared with any C3 file
   that might contain credentials.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from filelock import FileLock

from maxim.utils.atomic_io import atomic_write_text
from maxim.utils.paths import resolve_user_state


class DrainError(ValueError):
    """Raised when a drain operation cannot be performed.

    Carries a short diagnostic and (optionally) a list of known node
    names so the CLI can render an operator-readable error with a
    "known: [...]" hint.
    """

    def __init__(self, message: str, *, known_nodes: list[str] | None = None) -> None:
        self.known_nodes = known_nodes or []
        super().__init__(message)


@dataclass(frozen=True)
class DrainReadResult:
    """Return value for :func:`read_drained_nodes`.

    ``drained`` is the set of names currently persisted as drained.
    ``orphans`` is the subset of ``drained`` that does NOT appear in
    the supplied mesh node set. Orphans are reported, not filtered
    out — callers decide whether to warn, fail, or clean up.
    """

    drained: frozenset[str]
    orphans: frozenset[str]

    @property
    def active(self) -> frozenset[str]:
        """Drain entries that match a real mesh node."""
        return self.drained - self.orphans


# ─── path helpers ────────────────────────────────────────────────────────


def _role() -> str:
    """Return the current role for drain state scoping.

    Reads ``MAXIM_ROLE`` (set by ``runtime/role.py::detect_and_apply_role``
    per Plan 2 R2a). Defaults to ``leader`` only if the env var is
    genuinely absent — callers that hit this default on a peer machine
    have a bug: they skipped the role-detection call the module
    docstring requires.

    Unexpected values (empty string, mixed case, whitespace) are
    lowercased and stripped. Anything that doesn't match
    ``{leader, peer, solo}`` falls back to ``leader`` with no warning
    because this module is imported during every ``maxim peer ...``
    invocation and a noisy warning would flood the happy path. The
    regression test locks the behavior.
    """
    raw = os.environ.get("MAXIM_ROLE", "").strip().lower()
    if raw in ("leader", "peer", "solo"):
        return raw
    return "leader"


def drain_state_path() -> Path:
    """Return the role-scoped path for the drain state file."""
    return resolve_user_state(f"util/drained_nodes.{_role()}.txt")


def _lock_path(state_path: Path) -> Path:
    return state_path.with_suffix(state_path.suffix + ".lock")


def _lock(state_path: Path) -> FileLock:
    """Build a ``FileLock`` for the given drain state path.

    The lock file is a sibling with a ``.lock`` suffix. ``FileLock``
    handles both POSIX (``fcntl.flock``) and Windows (``msvcrt.locking``)
    natively — no platform carve-out needed. Timeout is 10s, which
    is generous for a file-bound lock but prevents indefinite hangs
    if another process crashed mid-write without releasing.
    """
    state_path.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(_lock_path(state_path)), timeout=10)


# ─── read path ───────────────────────────────────────────────────────────


def _load_names(state_path: Path) -> set[str]:
    """Parse the drain state file into a set of names.

    Empty/missing file → empty set. Blank lines and comment lines
    (``#``-prefixed) are ignored.
    """
    if not state_path.is_file():
        return set()
    try:
        content = state_path.read_text()
    except OSError:
        return set()
    return {line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith("#")}


def read_drained_nodes(known_node_names: set[str] | None = None) -> DrainReadResult:
    """Return the current drain state, partitioned into active + orphans.

    Parameters
    ----------
    known_node_names
        If provided, any drained entry not in this set is reported as
        an orphan (still in ``drained``, also in ``orphans``). If
        omitted, ``orphans`` is always empty — callers that can't
        supply the mesh node set accept the "no orphan detection"
        trade-off.
    """
    state_path = drain_state_path()
    with _lock(state_path):
        drained = frozenset(_load_names(state_path))
    if known_node_names is None:
        return DrainReadResult(drained=drained, orphans=frozenset())
    orphans = drained - known_node_names
    return DrainReadResult(drained=drained, orphans=orphans)


# ─── write path ──────────────────────────────────────────────────────────


def _serialize(names: set[str]) -> str:
    """Render a drain set as sorted newline-separated text with a header."""
    body = "\n".join(sorted(names))
    header = (
        "# Maxim drain state (Plan 4 C2). Role-scoped, one node name per line.\n"
        "# Edit via `maxim peer --node <name> drain|resume` — direct edits work\n"
        "# but the CLI surface is safer and will validate against mesh.yml.\n"
    )
    return header + body + ("\n" if body else "")


def _write(state_path: Path, names: set[str]) -> None:
    """Atomic write with preserve_mode so pre-existing file mode bits survive.

    Drain state itself isn't secret, but the ``preserve_mode=True``
    invocation exercises the shared utility pattern C3 will use for
    credential-bearing files. Locking this in at C2 catches regressions
    in the shared utility that would otherwise only surface in C3.
    """
    atomic_write_text(
        str(state_path),
        _serialize(names),
        preserve_mode=True,
    )


def drain_node(name: str, known_node_names: set[str]) -> frozenset[str]:
    """Add ``name`` to the drain set under lock. Returns the new drain set.

    Parameters
    ----------
    name
        The node to drain. Must appear in ``known_node_names`` or
        :class:`DrainError` is raised with the known list attached.
    known_node_names
        The current mesh's node names (from
        ``MeshConfig.nodes``). Used for orphan validation.

    Raises
    ------
    DrainError
        If ``name`` is not in ``known_node_names``. Idempotent for
        already-drained names — those return the current set without
        error.
    """
    if name not in known_node_names:
        raise DrainError(
            f"unknown node {name!r}",
            known_nodes=sorted(known_node_names),
        )

    state_path = drain_state_path()
    with _lock(state_path):
        current = _load_names(state_path)
        current.add(name)
        _write(state_path, current)
        return frozenset(current)


def resume_node(name: str, known_node_names: set[str]) -> frozenset[str]:
    """Remove ``name`` from the drain set under lock. Returns the new set.

    Idempotent for names that aren't currently drained — returns the
    current set without error. Unknown names (not in
    ``known_node_names``) ARE still rejected — we want the operator
    to know they typed the wrong name rather than silently succeeding
    against nothing.
    """
    if name not in known_node_names:
        raise DrainError(
            f"unknown node {name!r}",
            known_nodes=sorted(known_node_names),
        )

    state_path = drain_state_path()
    with _lock(state_path):
        current = _load_names(state_path)
        current.discard(name)
        _write(state_path, current)
        return frozenset(current)


__all__ = [
    "DrainError",
    "DrainReadResult",
    "drain_node",
    "drain_state_path",
    "read_drained_nodes",
    "resume_node",
]
