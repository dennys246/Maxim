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

from filelock import FileLock, Timeout as _FileLockTimeout

from maxim.utils.atomic_io import atomic_write_text
from maxim.utils.paths import resolve_user_state


class DrainError(ValueError):
    """Raised when a drain operation cannot be performed.

    Carries a short diagnostic and (optionally) a list of known node
    names so the CLI can render an operator-readable error with a
    "known: [...]" hint.

    **A4 note (C2 pre-merge review):** ``known_nodes`` is pre-materialized
    structured data, not a rendered string. Consumers that don't
    need the known-node hint (e.g. a future C3 admin API JSON
    response that renders the error differently) can ignore it —
    the lookup is already done at the call site so there's no
    incremental cost to carrying it. Do NOT pre-render the
    ``"known: [...]"`` suffix into ``str(exc)`` itself; keep the
    structured field separate so every caller can render it in its
    own shape.
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
    per Plan 2 R2a). The three valid values are ``leader``, ``peer``,
    ``solo``; whitespace + case are normalized.

    **A1 guard (C2 pre-merge review):** unexpected non-empty values
    raise :class:`DrainError` rather than silently falling back to
    ``leader``. The C2 pre-design review flagged silent fallback as
    a band-aid per CLAUDE.md's no-band-aid rule: after
    ``detect_and_apply_role`` runs there IS no other happy path, so
    a mismatch means either the operator mis-set the env var or a
    caller skipped the required detect-and-apply call. Both should
    be loud. The genuine ``""`` case (env var absent) still falls
    back to ``leader`` to handle tests and standalone scripts.
    """
    raw = os.environ.get("MAXIM_ROLE", "").strip().lower()
    if raw in ("leader", "peer", "solo"):
        return raw
    if raw == "":
        # Env var genuinely absent — tests and standalone scripts
        # that skip detect_and_apply_role get a sensible default.
        return "leader"
    # Non-empty, non-matching value is a caller bug: either
    # detect_and_apply_role wasn't called, or the operator set
    # MAXIM_ROLE to something invalid.
    raise DrainError(
        f"unrecognized MAXIM_ROLE={raw!r} (expected one of: leader, peer, solo). "
        "Caller must invoke maxim.runtime.role.detect_and_apply_role first, "
        "or the operator needs to unset MAXIM_ROLE."
    )


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

    **E2 fold (C2 pre-merge review):** inline ``#`` comments are also
    stripped. The module docstring invites hand-editing, and the
    original single-character startswith check treated
    ``mac-studio  # needs rebuild`` as a literal node name that
    silently became an orphan. Now we split on ``#``, take the first
    field, and strip. Lines that are empty after stripping the
    comment are skipped (``  # just a comment`` is a comment line,
    not an empty node name).
    """
    if not state_path.is_file():
        return set()
    try:
        content = state_path.read_text()
    except OSError:
        return set()
    names: set[str] = set()
    for raw in content.splitlines():
        # Strip inline comment (everything after the first `#`)
        # before whitespace-trimming so `foo  # bar` becomes `foo`.
        stripped = raw.split("#", 1)[0].strip()
        if stripped:
            names.add(stripped)
    return names


def read_drained_nodes(known_node_names: set[str] | None = None) -> DrainReadResult:
    """Return the current drain state, partitioned into active + orphans.

    Parameters
    ----------
    known_node_names
        If provided, any drained entry not in this set is reported as
        an orphan (still in ``drained``, also in ``orphans``). If
        omitted (``None``) OR empty set, ``orphans`` is always empty
        — callers that can't supply a meaningful mesh node set opt
        out of orphan detection. Passing an empty set used to report
        every entry as an orphan, which was never useful (an empty
        mesh has nothing to validate against).

    Raises
    ------
    DrainError
        If the filelock timeout elapses (E3 fold). The original raw
        ``filelock.Timeout`` surfaced to operators as an untyped
        traceback.
    """
    state_path = drain_state_path()
    try:
        with _lock(state_path):
            drained = frozenset(_load_names(state_path))
    except _FileLockTimeout as e:
        raise DrainError(
            f"drain state locked — another maxim process holds "
            f"{_lock_path(state_path)} for more than 10s. Check with "
            f"`lsof {_lock_path(state_path)}` and retry."
        ) from e
    if not known_node_names:
        # None or empty set both opt out of orphan detection.
        # Empty-set used to report every entry as an orphan which
        # is never the right behavior (E8 fold).
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
    """Atomic write for drain state.

    **A3 fold (C2 pre-merge review):** previously used
    ``preserve_mode=True`` "to exercise the shared utility pattern C3
    will use for credential-bearing files." That was a synthetic
    reason — drain state is operator-visible topology, not a secret.
    C3 credential files use :func:`atomic_write_secret` directly
    (which the same review added alongside this fold). Keeping this
    as plain ``atomic_write_text`` means the "this file contains
    secrets" signal only fires when it actually does.
    """
    atomic_write_text(str(state_path), _serialize(names))


def drain_node(
    name: str,
    known_node_names: set[str],
    *,
    self_name: str | None = None,
    force_self: bool = False,
) -> frozenset[str]:
    """Add ``name`` to the drain set under lock. Returns the new drain set.

    Parameters
    ----------
    name
        The node to drain. Must appear in ``known_node_names`` or
        :class:`DrainError` is raised with the known list attached.
    known_node_names
        The current mesh's node names (from
        ``MeshConfig.nodes``). Used for orphan validation.
    self_name
        The ``mesh.yml::self`` value, if known. When ``name ==
        self_name`` and ``force_self`` is False, raises
        :class:`DrainError` — draining yourself strands in-flight
        requests and is almost always a mistake. **A2 fold
        (C2 pre-merge review):** this guard used to live in the CLI
        layer only, which meant any future caller (C3 admin API,
        test fixtures, recovery scripts) that invoked ``drain_node``
        directly bypassed the safety check. Enforcing it at the
        state layer means every writer gets the guarantee.
    force_self
        Override the self-drain guard. The ``--force-self`` flag
        on ``maxim peer --node <self> drain`` sets this explicitly.

    Raises
    ------
    DrainError
        If ``name`` is not in ``known_node_names``, or if
        ``name == self_name`` without ``force_self``, or if the
        ``filelock.FileLock`` timeout elapses (E3 fold).
    """
    if name not in known_node_names:
        raise DrainError(
            f"unknown node {name!r}",
            known_nodes=sorted(known_node_names),
        )
    if self_name is not None and name == self_name and not force_self:
        raise DrainError(
            f"refusing to drain self ({name!r}) — this strands in-flight "
            "requests and is almost always a mistake. "
            "Pass force_self=True (CLI: --force-self) to override.",
            known_nodes=sorted(known_node_names),
        )

    state_path = drain_state_path()
    try:
        with _lock(state_path):
            current = _load_names(state_path)
            current.add(name)
            _write(state_path, current)
            return frozenset(current)
    except _FileLockTimeout as e:
        raise DrainError(
            f"drain state locked — another maxim process holds "
            f"{_lock_path(state_path)} for more than 10s. Check with "
            f"`lsof {_lock_path(state_path)}` and retry."
        ) from e


def resume_node(name: str, known_node_names: set[str]) -> frozenset[str]:
    """Remove ``name`` from the drain set under lock. Returns the new set.

    Idempotent for names that aren't currently drained — returns the
    current set without error. Unknown names (not in
    ``known_node_names``) ARE still rejected — we want the operator
    to know they typed the wrong name rather than silently succeeding
    against nothing.

    Raises :class:`DrainError` on unknown node or lock timeout (E3 fold).
    """
    if name not in known_node_names:
        raise DrainError(
            f"unknown node {name!r}",
            known_nodes=sorted(known_node_names),
        )

    state_path = drain_state_path()
    try:
        with _lock(state_path):
            current = _load_names(state_path)
            current.discard(name)
            _write(state_path, current)
            return frozenset(current)
    except _FileLockTimeout as e:
        raise DrainError(
            f"drain state locked — another maxim process holds "
            f"{_lock_path(state_path)} for more than 10s. Check with "
            f"`lsof {_lock_path(state_path)}` and retry."
        ) from e


def clear_drain_for_removed_node(name: str) -> bool:
    """Unconditionally remove ``name`` from the drain set under lock.

    Plan 4 C3.2. Used by :func:`maxim.peer.mesh_setup.run_remove_node`
    when an operator removes a node that's currently drained — the
    drain entry would otherwise become an orphan that surfaces as a
    warning in ``list-drained`` and ``maxim doctor``.

    Unlike :func:`resume_node` this does NOT validate ``name``
    against the mesh node set, because the caller is in the middle
    of removing the node and the validity check would race with its
    own mutation. The caller is responsible for ensuring this is
    only invoked as part of a ``remove-node`` flow, never as a
    user-facing verb.

    Returns
    -------
    bool
        True if the entry was removed, False if it wasn't drained
        in the first place. Caller uses this to render a visible
        message ("also cleared from drain state") only when
        something actually changed.

    Raises
    ------
    DrainError
        If the filelock timeout elapses.
    """
    state_path = drain_state_path()
    try:
        with _lock(state_path):
            current = _load_names(state_path)
            if name not in current:
                return False
            current.discard(name)
            _write(state_path, current)
            return True
    except _FileLockTimeout as e:
        raise DrainError(
            f"drain state locked — another maxim process holds "
            f"{_lock_path(state_path)} for more than 10s. Check with "
            f"`lsof {_lock_path(state_path)}` and retry."
        ) from e


__all__ = [
    "DrainError",
    "DrainReadResult",
    "clear_drain_for_removed_node",
    "drain_node",
    "drain_state_path",
    "read_drained_nodes",
    "resume_node",
]
