"""Operator-explicit one-shot setup verbs that mutate ``mesh.yml``.

Plan 4 Stage C3.1 (``init-mesh``) + C3.2 (``add-node`` / ``remove-node``).

Renamed from ``init_mesh.py`` in C3.2 to reflect the broader
"setup verb" category as the file grew from one verb to three. All
three verbs share the same shape:

1. Read the existing ``mesh.yml`` (or synthesize from ``peer.yml`` for
   ``init-mesh``)
2. Mutate the in-memory :class:`MeshConfig`
3. Write back via :func:`maxim.peer.mesh_config.write_mesh_config`

This is the **single sanctioned class of writers** of ``mesh.yml`` per
the C2 invariant ("``mesh.yml`` is declarative; runtime code paths
are read-only"). The CI grep allow-list in
``.github/workflows/test.yml`` enforces that only this module + the
test file may call ``write_mesh_config``. Future C3 features that
need to mutate ``mesh.yml`` (e.g. cluster key rotation) should be
added here OR moved to ``~/.maxim/util/`` instead — the choice is
documented in
``.claude/projects/.../memory/feedback_strict_grep_caller_allowlist.md``.

------------------------------------------------------------------------
``maxim peer init-mesh [--force]`` (C3.1)
------------------------------------------------------------------------

Synthesizes ``mesh.yml`` from ``peer.yml`` so an operator who ran
``maxim peer connect`` (which writes ``peer.yml``) but never hand-
created a ``mesh.yml`` can immediately use drain/resume + list-nodes.

**peer.yml is left in place by design.** ``runtime/role.py`` reads
``peer.yml`` existence as part of the role detection decision order
(per the Plan 2 R2a CLAUDE.md lesson). Deleting or moving
``peer.yml`` post-init-mesh would break role detection silently. The
two files coexist: ``peer.yml`` is the role-detection signal +
simple-single-leader config; ``mesh.yml`` is the multi-node topology
surface that drain/resume + ``list-nodes`` consume.

Decision tree (locked from C2 pre-design review E7):

============ ============ ======== =================================================== ====
``peer.yml`` ``mesh.yml`` ``--force`` Action                                              Exit
============ ============ ======== =================================================== ====
absent       absent       —        "nothing to convert"                                   1
absent       present      —        already-converted no-op (info note about role detect)  0
present      absent       —        synthesize mesh.yml from peer.yml                      0
present      present      no       refuse with --force hint                               2
present      present      yes, .bak absent  back up + synthesize                          0
present      present      yes, .bak present refuse + cleanup hint (A2 fold)               2
============ ============ ======== =================================================== ====

------------------------------------------------------------------------
``maxim peer add-node <name> --url <url> [--role peer|leader] [--force]`` (C3.2)
------------------------------------------------------------------------

Append a new node to ``mesh.yml::nodes``. Default role is ``peer``.
Refuses if ``<name>`` already exists unless ``--force`` (which
replaces the existing entry, not duplicates).

URL validation is **syntax-only** — matches C1's "parse-time
validation is offline; DNS deferred to probe time" rule. The
operator can run ``maxim peer --node <name> status`` immediately
after to verify reachability.

Refuses if ``mesh.yml`` doesn't exist yet (operator must run
``init-mesh`` first or hand-create one).

------------------------------------------------------------------------
``maxim peer remove-node <name>`` (C3.2)
------------------------------------------------------------------------

Remove a node from ``mesh.yml::nodes``. Side effect: clears any
drain state for ``<name>`` (with a visible warning) so removing a
drained node doesn't leave an orphan in the drain state file.

Refuses if ``<name>`` is ``mesh.yml::self`` — you can't delete the
node identity the running daemon is bound to. The error message
documents the workaround (edit ``mesh.yml`` by hand to change
``self``, restart, then ``remove-node`` the old self).

Refuses if ``mesh.yml`` doesn't exist yet.

------------------------------------------------------------------------
Common contracts
------------------------------------------------------------------------

- All three verbs use :func:`write_mesh_config` (the single
  sanctioned writer); the CI grep allow-list enforces this at
  PR-review time.
- All three verbs validate field contents at construction time via
  :class:`MeshConfig.__post_init__` (yaml-safe + non-empty nodes +
  ``self_name`` matches a node, the last enforced via the C3.2 A1
  fold).
- All three verbs call :func:`runtime.role.detect_and_apply_role`
  via the dispatcher in :func:`maxim.peer.cli.run_peer_connect_subcommand`
  — they can rely on ``MAXIM_ROLE`` being set when they run.
- Backup-on-write is non-atomic by design (the safest failure mode
  is partial backup + intact original, NOT atomic backup +
  half-written original). Don't "fix" this.
- **No filelock around mesh.yml read→mutate→write** (E5/A3 fold,
  C3.2 pre-merge review). Two parallel ``add-node`` / ``remove-node``
  calls would race the same way drain state would have without
  ``filelock.FileLock`` — last writer wins, silently dropping the
  other's mutation. This is a *conscious trade-off* for C3.2: setup
  verbs are operator-explicit one-shots, expected to be
  serial-by-construction (one operator at a time). The C2 invariant
  ("mesh.yml is declarative; only operator-explicit setup verbs
  may write") makes this acceptable. **If a future C3.3+ feature
  wires automatic writes to mesh.yml** — which the C2 invariant
  explicitly forbids — the lock gap becomes a correctness bug. The
  right C3.3+ resolution is "don't write to mesh.yml from automatic
  paths" (write to ``~/.maxim/util/`` instead), NOT "add a filelock
  to setup verbs." If the rule changes, both this docstring AND the
  CLAUDE.md C2 invariant lesson must change in the same commit.
"""

from __future__ import annotations

import logging
import shutil
import sys
from collections.abc import Sequence

from maxim.peer.config import peer_config_path, read_peer_config
from maxim.peer.drain_state import DrainError, _clear_drain_unconditional
from maxim.peer.mesh_config import (
    MeshConfig,
    MeshConfigError,
    MeshNode,
    NodeRole,
    mesh_config_path,
    read_mesh_config,
    synthesize_from_peer_config,
    write_mesh_config,
)

logger = logging.getLogger(__name__)


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


# ─── Plan 4 C3.2: add-node / remove-node ────────────────────────────────


_ADD_NODE_USAGE = """\
Usage: maxim peer add-node <name> --url <url> [--role peer|leader] [--force]

Append a new node to ~/.config/maxim/mesh.yml::nodes.

Required:
  <name>          Unique node name (must not contain whitespace+# or
                  newlines — see mesh_config._validate_yaml_safe)
  --url <url>     Node URL (e.g. https://peer.example.com/v1).
                  Validation is SYNTAX-ONLY at add time; reachability
                  is checked on the next `list-nodes` or doctor probe.
                  Matches C1's "DNS deferred to probe time" rule.

Options:
  --role <role>   peer (default) or leader.
  --force         Replace an existing node with the same name. Without
                  --force, add-node refuses if <name> already exists
                  and points the operator at remove-node + re-add.
                  On a brand-new name (nothing to replace), --force is
                  silently a no-op — the verb is idempotent for new
                  nodes the same way `mkdir -p` is for new directories.

Refuses if mesh.yml does not exist yet — run `maxim peer init-mesh`
first or hand-create one.
"""


_REMOVE_NODE_USAGE = """\
Usage: maxim peer remove-node <name>

Remove a node from ~/.config/maxim/mesh.yml::nodes.

Side effect: clears any drain state for <name> with a visible
"also cleared from drain state" message. Otherwise the orphan would
surface as a warning in `list-drained` and `maxim doctor`.

Refuses if:
  - mesh.yml does not exist (nothing to remove from)
  - <name> is not in mesh.yml::nodes (operator typo guard)
  - <name> is mesh.yml::self (you can't delete the running daemon's
    own identity — restart with a different self first)

Refuses if removing <name> would leave mesh.yml with zero nodes —
the parser requires at least one node, so the resulting file would
fail to parse on the next read.
"""


# E1 fold (C3.2 pre-merge review): replace the "__help__" magic
# string with a sentinel object so the help-vs-error branch in
# run_add_node can't accidentally swap order under future edits.
# Identity-comparison via `is _HELP_REQUESTED` is unambiguous and
# the static type checker can flag misuse.
_HELP_REQUESTED = object()


def _parse_add_node_argv(argv: Sequence[str]) -> tuple[int, object | None, dict]:
    """Parse argv for ``add-node``. Returns (exit_code, signal, opts).

    Three return shapes:

    - ``(2, "error string", {})`` — bad input, caller prints error + usage
    - ``(0, _HELP_REQUESTED, {})`` — operator passed -h/--help
    - ``(0, None, opts)`` — happy path, opts has name/url/role/force

    Split out so the parsing logic is testable in isolation and
    extending the option set in C3.3 (e.g. ``--public-key`` for
    cluster key rotation) is a single function edit.
    """
    args = list(argv)
    if any(a in ("-h", "--help") for a in args):
        return 0, _HELP_REQUESTED, {}
    if not args:
        return 2, "Missing required <name> argument", {}

    name = args.pop(0)
    if name.startswith("-"):
        return 2, f"Expected node name, got option {name!r}", {}

    url: str | None = None
    role: NodeRole = "peer"
    force = False

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--url":
            if i + 1 >= len(args):
                return 2, "--url requires a value", {}
            url = args[i + 1]
            i += 2
        elif a == "--role":
            if i + 1 >= len(args):
                return 2, "--role requires a value", {}
            role_val = args[i + 1]
            if role_val not in ("peer", "leader"):
                return 2, f"--role must be 'peer' or 'leader', got {role_val!r}", {}
            role = role_val  # type: ignore[assignment]
            i += 2
        elif a == "--force":
            force = True
            i += 1
        else:
            return 2, f"Unknown option: {a}", {}

    if url is None:
        return 2, "--url is required", {}

    return 0, None, {"name": name, "url": url, "role": role, "force": force}


def run_add_node(argv: Sequence[str]) -> int:
    """Entry point for ``maxim peer add-node``. Plan 4 C3.2.

    Decision tree:

    - ``mesh.yml`` absent → exit 1 (run init-mesh first)
    - ``mesh.yml`` malformed → exit 2 (caller fixes the schema error)
    - ``<name>`` already exists + no ``--force`` → exit 2 (refuse, hint)
    - ``<name>`` already exists + ``--force`` → replace in place
    - new name → append
    - URL/name/role contains yaml-unsafe characters → exit 2 (raised
      by ``MeshNode.__post_init__``)

    Validation is syntax-only — reachability is the next
    ``list-nodes`` / doctor probe's job, matching C1's "DNS deferred
    to probe time" invariant.
    """
    rc, signal, opts = _parse_add_node_argv(argv)
    if signal is _HELP_REQUESTED:
        print(_ADD_NODE_USAGE)
        return 0
    if rc != 0:
        # signal here is the error message string (not the help sentinel)
        print(signal, file=sys.stderr)
        print(_ADD_NODE_USAGE, file=sys.stderr)
        return rc

    name: str = opts["name"]
    url: str = opts["url"]
    role: NodeRole = opts["role"]
    force: bool = opts["force"]

    mesh_path = mesh_config_path()
    if not mesh_path.is_file():
        print(f"✗ {mesh_path} does not exist.", file=sys.stderr)
        print("  → Run `maxim peer init-mesh` first to create it from peer.yml,", file=sys.stderr)
        print("    or create ~/.config/maxim/mesh.yml by hand.", file=sys.stderr)
        return 1

    try:
        mesh = read_mesh_config(mesh_path)
    except MeshConfigError as e:
        print(f"✗ {e}", file=sys.stderr)
        print("  → Fix the schema error in mesh.yml first.", file=sys.stderr)
        return 2
    if mesh is None:
        # Race: file disappeared between is_file() and read.
        print(f"✗ {mesh_path} disappeared between check and read.", file=sys.stderr)
        return 1

    existing = mesh.get_node(name)
    if existing is not None and not force:
        print(f"✗ Node {name!r} already exists in mesh.yml.", file=sys.stderr)
        print(f"    {existing.role}: {existing.url}", file=sys.stderr)
        print("  → To replace, re-run with --force:", file=sys.stderr)
        print(f"      maxim peer add-node {name} --url {url} --role {role} --force", file=sys.stderr)
        print("  → Or remove the existing entry first:", file=sys.stderr)
        print(f"      maxim peer remove-node {name}", file=sys.stderr)
        return 2

    # Build the new node. MeshNode.__post_init__ validates the
    # yaml-safe characters; ValueError propagates to the operator.
    try:
        new_node = MeshNode(name=name, url=url, role=role)
    except ValueError as e:
        print(f"✗ Invalid node fields: {e}", file=sys.stderr)
        return 2

    if existing is not None:
        # Force-replace: keep node order stable so an operator's
        # `list-nodes` output doesn't shuffle on a force-add.
        new_nodes = tuple(new_node if n.name == name else n for n in mesh.nodes)
    else:
        # Append in operator-typed order — this also matches
        # `list-nodes` / `list-drained` rendering.
        new_nodes = mesh.nodes + (new_node,)

    try:
        # CC13 (closed 2026-08-13): the reserved mesh-auth fields are
        # forwarded so activating them cannot be silently reset by
        # add-node. NOTE what still gates activation: parse_mesh_config
        # never populates these and to_yaml never emits them — extending
        # the FROZEN dialect is the activation work's own architectural
        # decision (TOML/PyYAML escape hatches per the parser invariant).
        new_mesh = MeshConfig(
            cluster_key=mesh.cluster_key,
            self_name=mesh.self_name,
            nodes=new_nodes,
            protocol_version=mesh.protocol_version,
            cluster_keys=mesh.cluster_keys,
            cluster_trust_anchors=mesh.cluster_trust_anchors,
            cluster_auth_mode=mesh.cluster_auth_mode,
        )
    except ValueError as e:
        print(f"✗ Resulting mesh.yml would be invalid: {e}", file=sys.stderr)
        return 2

    try:
        write_mesh_config(new_mesh, mesh_path)
    except OSError as e:
        print(f"✗ Failed to write {mesh_path}: {e}", file=sys.stderr)
        return 1

    if existing is not None:
        print(f"✓ Replaced node {name!r} in {mesh_path}")
        print(f"  → was: {existing.role}, {existing.url}")
        print(f"  → now: {new_node.role}, {new_node.url}")
    else:
        print(f"✓ Added node {name!r} to {mesh_path}")
        print(f"  → {new_node.role}, {new_node.url}")
    print()
    print("You can verify with:")
    print(f"  maxim peer --node {name} status")
    print("  maxim peer list-nodes")
    return 0


def run_remove_node(argv: Sequence[str]) -> int:
    """Entry point for ``maxim peer remove-node``. Plan 4 C3.2.

    Decision tree:

    - ``-h``/``--help`` → exit 0 (usage)
    - missing ``<name>`` → exit 2
    - ``mesh.yml`` absent → exit 1 (nothing to remove from)
    - ``<name>`` not in mesh.yml::nodes → exit 2 (operator typo guard)
    - ``<name> == mesh.yml::self`` → exit 2 with workaround hint
      (you can't delete the running daemon's identity)
    - removing ``<name>`` would leave 0 nodes → exit 2 (parser
      requires at least one)
    - happy path → drop ``<name>``, write, clear drain state, print
      summary

    The drain state cleanup is automatic + visible (option (c) from
    the proposal): operator gets one verb, sees what happened, no
    orphans left behind.
    """
    args = list(argv)
    if any(a in ("-h", "--help") for a in args):
        print(_REMOVE_NODE_USAGE)
        return 0
    if not args:
        print("Missing required <name> argument", file=sys.stderr)
        print(_REMOVE_NODE_USAGE, file=sys.stderr)
        return 2
    name = args[0]
    if name.startswith("-"):
        print(f"Expected node name, got option {name!r}", file=sys.stderr)
        print(_REMOVE_NODE_USAGE, file=sys.stderr)
        return 2
    if len(args) > 1:
        print(f"Unexpected extra arguments: {' '.join(args[1:])}", file=sys.stderr)
        print(_REMOVE_NODE_USAGE, file=sys.stderr)
        return 2

    mesh_path = mesh_config_path()
    if not mesh_path.is_file():
        print(f"✗ {mesh_path} does not exist.", file=sys.stderr)
        print("  → Nothing to remove. Run `maxim peer init-mesh` to create one.", file=sys.stderr)
        return 1

    try:
        mesh = read_mesh_config(mesh_path)
    except MeshConfigError as e:
        print(f"✗ {e}", file=sys.stderr)
        print("  → Fix the schema error in mesh.yml first.", file=sys.stderr)
        return 2
    if mesh is None:
        print(f"✗ {mesh_path} disappeared between check and read.", file=sys.stderr)
        return 1

    existing = mesh.get_node(name)
    if existing is None:
        known = ", ".join(n.name for n in mesh.nodes)
        print(f"✗ Unknown node: {name!r} (known: {known})", file=sys.stderr)
        return 2

    if name == mesh.self_name:
        print(
            f"✗ Refusing to remove self ({name!r}) — you can't delete the running",
            file=sys.stderr,
        )
        print(
            "  daemon's own identity from mesh.yml::nodes (the parser requires",
            file=sys.stderr,
        )
        print("  self to match a node).", file=sys.stderr)
        print("  → To change identity:", file=sys.stderr)
        print("      1. Edit mesh.yml::self to a different existing node by hand", file=sys.stderr)
        print("      2. Restart the daemon under the new self", file=sys.stderr)
        print(f"      3. Re-run `maxim peer remove-node {name}` under the new identity", file=sys.stderr)
        return 2

    # E4/A5 fold (C3.2 pre-merge review, cross-confirmed): this branch
    # is structurally unreachable through C3.2 verbs — the self-refusal
    # above guarantees `name != self_name`, and a 1-node mesh always
    # has that node == self (parser constraint that self_name must
    # match a node, now also enforced at MeshConfig.__post_init__ via
    # the A1 fold). So a 1-node mesh hits the self-refusal first, and
    # multi-node meshes never trigger this branch.
    #
    # Kept as defensive depth for two reasons:
    # 1. If a future C3.3+ verb (rename-node, set-self) loosens the
    #    self-refusal, this branch becomes the new error surface and
    #    must produce an actionable message — not a raw ValueError
    #    from MeshConfig.__post_init__.
    # 2. The C2 invariant says invariants are enforced at the state
    #    layer, not the CLI layer. This is the state-layer enforcement.
    #
    # No regression test exercises this branch directly because there's
    # no operator-reachable path to it. test_single_node_mesh_hits_self_guard_first
    # documents the unreachability explicitly.
    if len(mesh.nodes) <= 1:  # pragma: no cover - defensive only
        print(
            f"✗ Refusing to remove {name!r}: mesh would be empty.",
            file=sys.stderr,
        )
        print(
            "  → Add another node first (`maxim peer add-node ...`), or remove",
            file=sys.stderr,
        )
        print("    mesh.yml entirely if you no longer need a multi-node setup.", file=sys.stderr)
        return 2

    new_nodes = tuple(n for n in mesh.nodes if n.name != name)
    try:
        # CC13 (closed 2026-08-13): reserved mesh-auth fields forwarded —
        # see the add-node note above for what still gates activation.
        new_mesh = MeshConfig(
            cluster_key=mesh.cluster_key,
            self_name=mesh.self_name,
            nodes=new_nodes,
            protocol_version=mesh.protocol_version,
            cluster_keys=mesh.cluster_keys,
            cluster_trust_anchors=mesh.cluster_trust_anchors,
            cluster_auth_mode=mesh.cluster_auth_mode,
        )
    except ValueError as e:
        # Should be unreachable — we just validated len > 1 — but
        # MeshConfig.__post_init__ might add new validation later.
        print(f"✗ Resulting mesh.yml would be invalid: {e}", file=sys.stderr)
        return 2

    try:
        write_mesh_config(new_mesh, mesh_path)
    except OSError as e:
        print(f"✗ Failed to write {mesh_path}: {e}", file=sys.stderr)
        return 1

    # Clear drain state for the removed node. This is operator-visible
    # so they know what happened — option (c) from the C3.2 proposal.
    # Imports are at module top per E6 fold; the underscore-prefixed
    # function name signals this is the private "no validation" helper
    # used only by remove-node, not the public resume_node API.
    drain_cleared = False
    try:
        drain_cleared = _clear_drain_unconditional(name)
    except DrainError as e:
        # Drain state file is locked — write succeeded, but the
        # cleanup didn't. Surface the failure but don't roll back the
        # mesh.yml write (it's already committed and a partial state
        # is recoverable: operator can manually `resume` the orphan
        # later).
        #
        # E3 fold (C3.2 pre-merge review): emit a structured WARN log
        # alongside the stderr print so script-driven callers get a
        # machine-readable signal that an orphan was left. Matches
        # Plan 4's observability discipline (provider_silenced,
        # role_divergence, etc.).
        logger.warning(
            "mesh_setup_drain_cleanup_failed: removed node %r from mesh.yml "
            "but drain state cleanup failed (%s) — orphan entry may remain",
            name,
            e,
            extra={
                "event": "mesh_setup_drain_cleanup_failed",
                "data": {"node_name": name, "error": str(e)},
            },
        )
        print(f"⚠ Removed {name!r} from mesh.yml, but drain state cleanup failed: {e}", file=sys.stderr)
        print(
            f"  → The drain state file may now have an orphan entry for {name!r}.",
            file=sys.stderr,
        )
        print("  → Run `maxim peer list-drained` to confirm and clean up manually.", file=sys.stderr)
        return 0  # mesh.yml write succeeded — partial success, exit 0

    print(f"✓ Removed node {name!r} from {mesh_path}")
    print(f"  → was: {existing.role}, {existing.url}")
    if drain_cleared:
        print(f"  → also cleared {name!r} from drain state")
    print()
    print("You can verify with:")
    print("  maxim peer list-nodes")
    return 0


__all__ = ["run_add_node", "run_init_mesh", "run_remove_node"]
