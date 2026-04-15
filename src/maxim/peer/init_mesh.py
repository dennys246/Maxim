"""``maxim peer init-mesh`` — synthesize ``mesh.yml`` from ``peer.yml``.

Plan 4 Stage C3.1 (deferred from C2 per the C2 pre-design review).
The original C2 plan included ``init-mesh`` to unblock drain/resume
on ``peer.yml``-only installs, but the C2 scope was pivoted to
"runtime drain state layer" and ``init-mesh`` was deferred so that
C2 could ship the drain mechanism cleanly first.

This is the convenience verb that fills the gap: an operator who
ran ``maxim peer connect`` (which writes ``peer.yml``) but never
created a ``mesh.yml`` by hand will hit "Cannot drain: drain
requires mesh.yml" when they try ``maxim peer --node leader drain``.
``init-mesh`` synthesizes a one-node ``mesh.yml`` from the existing
``peer.yml`` so drain/resume works immediately.

**peer.yml is left in place by design.** ``runtime/role.py`` reads
``peer.yml`` existence as part of the role detection decision order
(per the Plan 2 R2a CLAUDE.md lesson). Deleting or moving
``peer.yml`` post-init-mesh would break role detection silently on
the next ``maxim`` invocation. The two files coexist: ``peer.yml``
is the role-detection signal + simple-single-leader config;
``mesh.yml`` is the multi-node topology surface that drain/resume
+ ``list-nodes`` consume.

Decision tree (locked from the C2 pre-design review E7 finding —
the original spec for the failed Option A1 ``migrate-config`` verb,
which we now reuse here):

============ ============ ======== =================================================== ====
``peer.yml`` ``mesh.yml`` ``--force`` Action                                              Exit
============ ============ ======== =================================================== ====
absent       absent       —        "nothing to convert"                                   1
absent       present      —        "mesh.yml already exists, nothing to do"               0
present      absent       —        synthesize mesh.yml from peer.yml                      0
present      present      no       refuse with --force hint                               2
present      present      yes      back up mesh.yml → mesh.yml.bak, then synthesize       0
============ ============ ======== =================================================== ====

The backup-on-force step is load-bearing: ``--force`` is the
explicit "I know what I'm doing" override, but it MUST NOT silently
destroy the operator's existing ``mesh.yml``. The ``.bak`` file is
written via ``shutil.copy2`` (preserves mtime + mode) before the
new ``mesh.yml`` is written. If the operator wants to keep multiple
backups, that's their job — ``init-mesh`` always overwrites
``mesh.yml.bak``.
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Sequence

from maxim.peer.config import peer_config_path, read_peer_config
from maxim.peer.mesh_config import (
    MeshConfig,
    mesh_config_path,
    synthesize_from_peer_config,
    write_mesh_config,
)


_USAGE = """\
Usage: maxim peer init-mesh [--force]

Synthesize ~/.config/maxim/mesh.yml from the existing peer.yml so
that `maxim peer --node X drain|resume` and `list-nodes` work on a
peer.yml-only install.

Options:
  --force    Overwrite an existing mesh.yml (backed up to
             mesh.yml.bak first). Refuses without this flag.
             Also refuses if mesh.yml.bak already exists — delete
             or rename the existing backup first to avoid losing
             your original on a double `--force`.

peer.yml is NOT modified — it stays in place because runtime role
detection reads its existence as part of the leader/peer decision.
"""


def run_init_mesh(argv: Sequence[str]) -> int:
    """Entry point for ``maxim peer init-mesh``.

    Returns the CLI exit code per the decision tree in the module
    docstring. Flags supported: ``--force``, ``-h`` / ``--help``.

    **E1 fold (C3.1 pre-merge review):** the unknown-option filter
    excludes ``-h``/``--help`` so that ``init-mesh --bogus --help``
    surfaces the bogus error instead of silently swallowing it via
    the help short-circuit. Help still wins over ``--force`` alone
    (operator can ask for help mid-typing) but doesn't mask other
    unknown flags.
    """
    known_flags = ("--force", "-h", "--help")
    unknown = [a for a in argv if a not in known_flags]
    if unknown:
        print(f"Unknown option(s): {' '.join(unknown)}", file=sys.stderr)
        print(_USAGE, file=sys.stderr)
        return 2
    if any(a in ("-h", "--help") for a in argv):
        print(_USAGE)
        return 0
    force = "--force" in argv

    peer_path = peer_config_path()
    mesh_path = mesh_config_path()
    peer_present = peer_path.is_file()
    mesh_present = mesh_path.is_file()

    # Decision tree row 1: nothing to convert
    if not peer_present and not mesh_present:
        print("✗ Nothing to convert: neither peer.yml nor mesh.yml exists.", file=sys.stderr)
        print(f"  → Run `maxim peer connect <url>` first to create {peer_path}.", file=sys.stderr)
        return 1

    # Decision tree row 2: peer.yml absent but mesh.yml present —
    # operator already has a mesh, no-op success.
    #
    # A5 fold (C3.1 pre-merge review): surface the unusual state
    # rather than silent-success. peer.yml absence isn't a hard
    # failure — runtime/role.py falls through to mesh.yml — but the
    # operator who invoked init-mesh expecting "something to happen"
    # should know why nothing did.
    if not peer_present and mesh_present:
        print(f"ℹ {mesh_path} already exists and there is no peer.yml to convert.")
        print("  Nothing to do.")
        print()
        print("  Note: peer.yml is absent. runtime/role.py will fall through to")
        print("  mesh.yml directly for role detection, which is fine but unusual.")
        print("  If you want a peer.yml as well, run `maxim peer connect <url>`.")
        return 0

    # peer.yml is present from here. Read it before doing anything
    # destructive so a malformed peer.yml fails before we touch
    # mesh.yml.
    peer = read_peer_config()
    if peer is None:
        print(
            f"✗ {peer_path} exists but could not be parsed.",
            file=sys.stderr,
        )
        print("  → Check the file for syntax errors, or recreate via `maxim peer connect`.", file=sys.stderr)
        return 1

    # Decision tree row 4: mesh.yml already exists, no --force
    if mesh_present and not force:
        print(f"✗ {mesh_path} already exists.", file=sys.stderr)
        print("  → If you want to overwrite, re-run with --force:", file=sys.stderr)
        print("      maxim peer init-mesh --force", file=sys.stderr)
        print("    The existing mesh.yml will be backed up to mesh.yml.bak.", file=sys.stderr)
        print("    (--force refuses if mesh.yml.bak already exists — delete it first", file=sys.stderr)
        print("    or rename it to keep multiple history slots.)", file=sys.stderr)
        return 2

    # Synthesize the mesh in memory.
    mesh = synthesize_from_peer_config()
    if mesh is None:
        # Should be unreachable — we just confirmed peer is not None
        # — but a concurrent peer.yml deletion between the
        # read_peer_config and synthesize_from_peer_config calls
        # could trigger this.
        print(
            f"✗ Could not synthesize mesh from {peer_path}.",
            file=sys.stderr,
        )
        return 1

    # Decision tree row 5: mesh.yml exists + --force → backup first.
    #
    # A2 fold (C3.1 pre-merge review): refuse if mesh.yml.bak already
    # exists. Without this guard, double `--force` silently destroys
    # the operator's original mesh.yml: the second --force backs up
    # the first --force's synthesized output, original is gone. The
    # refuse-if-exists rule matches the explicit "I-know-what-I'm-
    # doing" ethos of --force itself; operator must consciously
    # delete or rename the existing .bak to proceed.
    #
    # A6 fold: backup-via-shutil.copy2 is non-atomic, but new write
    # via atomic_write_secret IS atomic. The asymmetry is the SAFEST
    # failure mode: a crash between the backup and the new write
    # leaves a partial .bak (or nothing) AND the original mesh.yml
    # intact. Operator loses nothing. Do NOT "fix" this by making
    # the backup atomic too — that would change the failure mode
    # to "operator might lose the original under concurrent crash".
    backup_path = None
    if mesh_present:
        backup_path = mesh_path.with_suffix(mesh_path.suffix + ".bak")
        if backup_path.is_file():
            print(
                f"✗ {backup_path} already exists.",
                file=sys.stderr,
            )
            print("  → Refusing to overwrite an existing backup. Either:", file=sys.stderr)
            print(f"      rm {backup_path}", file=sys.stderr)
            print(f"      mv {backup_path} {backup_path}.old", file=sys.stderr)
            print("    then re-run `maxim peer init-mesh --force`.", file=sys.stderr)
            return 2
        try:
            shutil.copy2(str(mesh_path), str(backup_path))
        except OSError as e:
            print(
                f"✗ Failed to back up {mesh_path} → {backup_path}: {e}",
                file=sys.stderr,
            )
            print("  Refusing to proceed without a backup.", file=sys.stderr)
            return 1

    # Write the new mesh.yml. write_mesh_config routes through
    # atomic_write_secret because mesh.yml::cluster_key is a secret
    # per the C2 invariant; the function also chmods to 0o600 on
    # first write.
    try:
        written = write_mesh_config(mesh)
    except OSError as e:
        print(f"✗ Failed to write {mesh_path}: {e}", file=sys.stderr)
        if backup_path:
            print(f"  → Your original mesh.yml is safe at {backup_path}", file=sys.stderr)
        return 1

    # Success rendering — show the operator what landed.
    print(f"✓ Synthesized {written} from {peer_path}")
    _print_mesh_summary(mesh)
    if backup_path:
        print(f"  → Existing mesh.yml backed up to {backup_path}")
    print()
    print("You can now run:")
    print(f"  maxim peer --node {mesh.self_name} drain --force-self")
    print("  maxim peer list-drained")
    print("  maxim peer list-nodes")
    return 0


def _print_mesh_summary(mesh: MeshConfig) -> None:
    """Render a 3-line summary of what init-mesh just wrote."""
    node_count = len(mesh.nodes)
    node_word = "node" if node_count == 1 else "nodes"
    first = mesh.nodes[0]
    print(f"  → {node_count} {node_word} ({first.role}, {first.url})")
    print("  → cluster_key copied from peer.yml::api_key")
    print(f"  → self set to {mesh.self_name!r}")
    print("  → peer.yml left in place (still used for role detection)")


__all__ = ["run_init_mesh"]
