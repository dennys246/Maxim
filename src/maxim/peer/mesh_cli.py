"""CLI verbs for ``maxim peer list-nodes`` and ``maxim peer --node <name> ...``.

Plan 4 Stage C1 + C2 + C3.3 + C3.5 + C3.6. Extends ``maxim peer``
with mesh topology verbs backed by :mod:`maxim.peer.mesh_config`.
Node probes route through
:meth:`maxim.models.language.maxim_peer_backend._MaximPeerBackend.for_url`
+ ``health_check`` — the canonical probe entry point (Plan 3 R2.6).

Verbs (C1 + C2 + C3.3 + C3.5 + C3.6):

- ``maxim peer list-nodes [--json]`` — table/JSON of nodes + live status
  with drained nodes shown inline
- ``maxim peer list-drained`` — dump the current drain set (C2)
- ``maxim peer --node <name> status`` / ``health`` — per-node probe
- ``maxim peer --node <name> drain`` — add to drain state (C2)
- ``maxim peer --node <name> resume`` — clear from drain state (C2)
- ``maxim peer --node <name> install <extras>`` — mesh-aware install
  with drain → install → resume composition (C3.3)
- ``maxim peer --node <name> update`` — mesh-aware update with
  drain → update → resume composition (C3.5)
- ``maxim peer --node <name> restart`` — mesh-aware restart with
  drain → restart → resume composition (C3.5)
- ``maxim peer --node <name> llm <model>`` — mesh-aware LLM swap with
  drain → swap → resume composition (C3.6)

Drain state lives in ``~/.maxim/util/drained_nodes.{role}.txt`` via
:mod:`maxim.peer.drain_state`. The split between ``mesh.yml``
(declarative topology) and the runtime drain state file (mutable
state) is a load-bearing invariant — see the CLAUDE.md lesson
"mesh.yml is declarative; ~/.maxim/util/ is mutable state."

Sibling module :mod:`maxim.peer.mesh_setup` houses the operator-
invoked one-shot setup verbs that mutate ``mesh.yml`` itself
(``init-mesh``, ``add-node``, ``remove-node``); this module
(``mesh_cli``) handles the read-only-from-mesh.yml verbs
(``list-nodes``, ``--node <status|health|...>``).

Deferred to remaining Plan 4 work: admin API endpoints, per-agent
rate limiting, request-trace ring buffer, cluster key rotation.
See :doc:`/docs/plans/reactive_peer_mesh_roadmap` for the full arc.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from maxim.peer.drain_state import (
    DrainError,
    drain_node as _drain_node,
    drain_node_if_absent as _drain_node_if_absent,
    read_drained_nodes,
    resume_node as _resume_node,
)
from maxim.peer.admin_core import (
    llm_swap_on_target,
    restart_on_target,
    update_on_target,
)
from maxim.peer.install_core import (
    KNOWN_EXTRAS,
    _looks_like_url,
    classify_install_tokens,
    install_on_target,
)
from maxim.peer.mesh_config import (
    MeshConfig,
    MeshConfigError,
    MeshNode,
    read_or_synthesize_mesh_config,
)
from maxim.peer.probe_classify import classify_probe_outcome


# Exit code 3 — install succeeded but the post-install auto-resume
# failed. Distinguishable from exit code 1 (install itself failed)
# so operators tailing exit codes can tell "node is upgraded but
# stuck in drain" apart from "node failed to upgrade and is stuck
# in drain." Fold I4 from the C3.3 pre-merge review.
_EXIT_RESUME_FAILED_POST_OP = 3


MESH_USAGE_HEAD = """\
Usage: maxim peer list-nodes [--json]
       maxim peer list-drained
       maxim peer --node <name> <verb>

Verbs: status, health, drain, resume, install, update, restart, llm

Mesh topology verbs. Requires mesh.yml (or falls back to peer.yml as
a synthesized one-node mesh). Node probes go through
_MaximPeerBackend.health_check(). Drain state lives at
~/.maxim/util/drained_nodes.{role}.txt. The install/update/restart/llm
verbs compose drain → op → resume around shared cores in install_core
and admin_core.
"""


@dataclass
class _NodeProbeReport:
    node: MeshNode
    status: str  # ok|fail|warn|info
    message: str
    fix: str | None = None
    latency_ms: float | None = None
    drained: bool = False  # Plan 4 C2: drained nodes get status=info, no probe


# ─── public entry points ────────────────────────────────────────────────


def run_list_nodes(argv: Sequence[str]) -> int:
    """``maxim peer list-nodes [--json]`` — table of nodes + live status.

    Plan 4 C2: drained nodes (from the runtime drain state file) are
    shown as ``info`` / "drained (not probed)" without making a
    network call. Orphan drain entries — names in the drain state
    file that no longer match any node in mesh.yml, e.g. after an
    operator edited mesh.yml to remove a node — surface as a warning
    footer, not a hard fail.
    """
    as_json = "--json" in argv
    mesh, err = _load_mesh_or_report_error()
    if mesh is None:
        return err

    known_names = {n.name for n in mesh.nodes}
    drain_result = read_drained_nodes(known_names)

    reports: list[_NodeProbeReport] = []
    for node in mesh.nodes:
        if node.name in drain_result.active:
            reports.append(_drained_report(node))
        else:
            reports.append(_probe_node(node, mesh.cluster_key))

    if as_json:
        _print_json(mesh, reports, orphans=sorted(drain_result.orphans))
    else:
        _print_table(mesh, reports, orphans=sorted(drain_result.orphans))
    # Non-zero exit if any node is failing (warn is tolerated, matching
    # `maxim doctor`'s exit-code contract). Orphan drain entries are a
    # warning, not a failure — operator may be mid-edit.
    return 1 if any(r.status == "fail" for r in reports) else 0


def run_list_drained(argv: Sequence[str]) -> int:
    """``maxim peer list-drained`` — print the current drain set.

    Plan 4 C2. Operator-friendly dump of the role-scoped drain state
    file. Reports active drains (node name matches mesh.yml) and
    orphans (drain entry with no matching node — operator edited
    mesh.yml after draining) separately so the operator can fix
    orphans with ``resume``.
    """
    del argv  # no options yet
    mesh, err = _load_mesh_or_report_error()
    if mesh is None:
        return err
    known_names = {n.name for n in mesh.nodes}
    drain_result = read_drained_nodes(known_names)

    if not drain_result.drained:
        print("No nodes drained.")
        return 0

    print(f"Drained nodes ({len(drain_result.drained)}):")
    for name in sorted(drain_result.active):
        print(f"  ⊝ {name}")
    if drain_result.orphans:
        print()
        print(f"⚠ Orphan drain entries ({len(drain_result.orphans)}):")
        for name in sorted(drain_result.orphans):
            print(f"  ⊝ {name}  (no such node in mesh.yml)")
        print("  → Run `maxim peer --node <name> resume` to clean up,")
        print("    or re-add the node to mesh.yml.")
    return 0


def run_node_subcommand(argv: Sequence[str]) -> int:
    """``maxim peer --node <name> <verb>`` dispatcher.

    ``argv`` is everything after ``peer`` (i.e., starts with
    ``["--node", "<name>", "<verb>"]``).

    Plan 4 C1: ``status`` / ``health``. Plan 4 C2: ``drain`` /
    ``resume``. Plan 4 C3.3: ``install``. Plan 4 C3.5: ``update`` /
    ``restart``. Plan 4 C3.6: ``llm``.

    Drain + resume reject unknown nodes with exit 2 and list the
    known names. Drain is idempotent for already-drained nodes (exit
    0, informational message); resume is idempotent for not-drained
    nodes.

    ``--force-self`` flag after the drain verb is required to drain
    the node matching ``mesh.yml::self``: draining yourself strands
    in-flight requests and is almost always a mistake. The override
    exists because there are legitimate cases (e.g., graceful
    shutdown scripts) but it must be explicit.

    Install takes positional tokens after the verb:
    ``maxim peer --node <name> install <extras_or_packages>``.
    Composes drain → install → resume around the shared
    :func:`maxim.peer.cli._install_on_target` core. Refuses self,
    refuses a positional URL (use ``maxim peer install`` for that),
    and leaves the node drained on install failure with a loud
    message pointing at the manual resume command.
    """
    valid_verbs = "status|health|drain|resume|install|update|restart|llm"
    if not argv or argv[0] != "--node":
        print(f"Usage: maxim peer --node <name> <{valid_verbs}>", file=sys.stderr)
        return 2
    if len(argv) < 2:
        print(
            f"Missing node name: maxim peer --node <name> <{valid_verbs}>",
            file=sys.stderr,
        )
        return 2
    if len(argv) < 3:
        print(
            f"Missing verb: maxim peer --node {argv[1]} <{valid_verbs}>",
            file=sys.stderr,
        )
        return 2
    name = argv[1]
    verb = argv[2]
    verb_args = list(argv[3:])

    mesh, err = _load_mesh_or_report_error()
    if mesh is None:
        return err
    node = mesh.get_node(name)
    if node is None:
        known = ", ".join(n.name for n in mesh.nodes)
        print(f"Unknown node: {name!r} (known: {known})", file=sys.stderr)
        return 2

    if verb in ("status", "health"):
        # Plan 4 C2: drained nodes short-circuit to an info report.
        drained = read_drained_nodes({n.name for n in mesh.nodes}).active
        if node.name in drained:
            report = _drained_report(node)
        else:
            report = _probe_node(node, mesh.cluster_key)
        _print_single_node(report)
        return 0 if report.status in ("ok", "info") else 1
    if verb == "drain":
        return _run_drain(mesh, node, force_self="--force-self" in set(verb_args))
    if verb == "resume":
        return _run_resume(mesh, node)
    if verb == "install":
        return _run_node_install(mesh, node, verb_args)
    if verb == "update":
        return _run_node_update(mesh, node, verb_args)
    if verb == "restart":
        return _run_node_restart(mesh, node, verb_args)
    if verb == "llm":
        return _run_node_llm(mesh, node, verb_args)
    print(
        f"Unknown --node verb: {verb!r} (expected {valid_verbs})",
        file=sys.stderr,
    )
    return 2


def _run_drain(mesh: MeshConfig, node: MeshNode, *, force_self: bool) -> int:
    """Execute ``drain`` for one node. Returns the CLI exit code.

    Self-drain requires ``--force-self`` because it strands in-flight
    requests and is almost always a mistake. The override flag is
    explicit so scripts that genuinely need it (graceful shutdowns)
    have to opt in.

    **A2 fold (C2 pre-merge review):** the self-drain guard now also
    lives in :func:`drain_state.drain_node`. The CLI still parses
    ``--force-self`` and still catches the DrainError to render the
    operator-readable hint, but if any future caller (C3 admin API,
    test fixture, recovery script) invokes ``drain_node`` directly,
    the state layer enforces the same guarantee.
    """
    known_names = {n.name for n in mesh.nodes}
    already = node.name in read_drained_nodes(known_names).active
    try:
        new_set = _drain_node(
            node.name,
            known_names,
            self_name=mesh.self_name,
            force_self=force_self,
        )
    except DrainError as e:
        # The state layer raises on (a) unknown node (unreachable
        # here because mesh.get_node validated already, except for
        # concurrent mesh.yml edits), (b) self-drain without
        # force_self, and (c) filelock timeout. Render a friendly
        # CLI message for the self case; the other cases fall
        # through to the generic format.
        if node.name == mesh.self_name and not force_self:
            print(
                f"✗ Refusing to drain self ({node.name!r}) — this strands in-flight\n"
                f"  requests and is almost always a mistake.\n"
                f"  → If you're sure, re-run with --force-self:\n"
                f"      maxim peer --node {node.name} drain --force-self",
                file=sys.stderr,
            )
        else:
            print(f"✗ {e}", file=sys.stderr)
        return 2
    if already:
        print(f"ℹ {node.name!r} already drained. Drain set: {sorted(new_set)}")
    else:
        print(f"✓ Drained {node.name!r}. Drain set: {sorted(new_set)}")
    return 0


def _run_resume(mesh: MeshConfig, node: MeshNode) -> int:
    """Execute ``resume`` for one node. Returns the CLI exit code."""
    known_names = {n.name for n in mesh.nodes}
    was_drained = node.name in read_drained_nodes(known_names).active
    try:
        new_set = _resume_node(node.name, known_names)
    except DrainError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 2
    if not was_drained:
        print(f"ℹ {node.name!r} was not drained. Drain set: {sorted(new_set) or '[]'}")
    else:
        print(f"✓ Resumed {node.name!r}. Drain set: {sorted(new_set) or '[]'}")
    return 0


def _redact_url_for_error(url: str) -> str:
    """Redact a URL down to ``<scheme>://...`` for operator error messages.

    Fold M3: ``f"positional URL {arg!r}"`` in error messages leaks
    the operator's typed URL, which could contain secrets in the
    path (e.g. ``https://host/sk-abc``). Redacting to scheme-only
    matches how ``peer/cli.py::truncate_key`` handles similar cases.
    Keep the scheme so the operator can confirm the parser saw a
    URL, not some other syntax.
    """
    if "://" not in url:
        return "<url>"
    scheme = url.split("://", 1)[0]
    return f"{scheme}://..."


def _parse_node_install_tokens(verb_args: list[str]) -> list[str]:
    """Extract raw install tokens from the ``--node <name> install`` argv tail.

    Mirrors :func:`maxim.peer.cli._cmd_install`'s argv parser for
    behavioral consistency across entry points:

    - Tokens starting with ``--`` are ignored (future flags).
    - Tokens that **look like URLs** (contain ``"://"`` per
      :func:`install_core._looks_like_url`) are **rejected** — unlike
      the positional-URL path through ``maxim peer install``, this
      verb resolves the target URL from ``mesh.yml::nodes`` by name.
      A positional URL here would silently override the mesh lookup,
      which is exactly the kind of dual-source-of-truth footgun the
      Plan 4 C2 invariant warns against. The URL detection uses
      ``"://"`` rather than ``startswith("http")`` so real PyPI
      packages like ``httpx``, ``http-client``, ``httpie`` are
      accepted as legitimate install tokens (Fold I3).
    - Everything else is comma-split into raw install tokens, which
      the caller passes to :func:`classify_install_tokens` for the
      ``KNOWN_EXTRAS`` classification.

    The ``ValueError`` message redacts the URL down to scheme-only
    to avoid leaking secrets that operators typed into argv (Fold M3).

    Returns the flat list of raw tokens (no classification yet, no
    whitespace stripping — the classifier handles both).
    """
    raw_tokens: list[str] = []
    for arg in verb_args:
        if _looks_like_url(arg):
            raise ValueError(
                f"positional URL {_redact_url_for_error(arg)} is not allowed "
                f"for the mesh-aware install verb — use "
                f"`maxim peer install <extras> <url>` for the no-mesh.yml "
                f"positional-URL form instead."
            )
        if arg.startswith("--"):
            continue  # future flags
        raw_tokens.extend(arg.split(","))
    return raw_tokens


def _run_node_install(
    mesh: MeshConfig,
    node: MeshNode,
    verb_args: list[str],
) -> int:
    """Execute ``--node <name> install <tokens>``.

    Plan 4 C3.3. Mesh-aware install with drain → install → resume
    composition. Resolves the target URL + cluster key from
    ``mesh.yml::nodes`` by name (unlike the positional-URL
    ``maxim peer install`` verb, which uses ``peer.yml``), drains the
    node via :func:`drain_state.drain_node_if_absent` (atomic
    drain-if-absent — closes the CC2 TOCTOU finding from the C3.3
    pre-merge review), delegates the HTTP body to
    :func:`install_core.install_on_target`, then resumes the node on
    success **only if this function was the one that drained it**.

    **Failure mode semantics (explicit design choice):** if drain
    succeeds but the install itself fails, the node is **left
    drained** with a loud "still drained → run resume" message.
    Auto-resume-on-failure would mask install failures behind a clean
    exit. This matches the C2 "fail loud, don't auto-recover from
    operator-explicit verbs" pattern.

    **Was-drained sticky semantics:** if the node was already drained
    when this verb started (operator drained it earlier for a
    different reason), we skip BOTH the drain step AND the
    auto-resume. The operator's prior drain intent is sticky — we
    don't clobber it on our way out. The atomic check inside
    :func:`drain_node_if_absent` closes the TOCTOU window that
    existed in the pre-fold C3.3 code (``read_drained_nodes`` then
    ``drain_node`` outside a shared lock).

    **Stacked-failure safety:** the install call is wrapped in
    ``try/except BaseException`` so ANY exception
    (``KeyboardInterrupt``, a programming bug inside
    :func:`install_on_target`, etc.) prints the "still drained" hint
    to stderr before re-raising. Without this, an abrupt interruption
    mid-install would leave the node drained with zero operator-
    facing clue (Fold I6).

    **Self-guard:** refuses if ``node.name == mesh.self_name`` because
    remote-installing on yourself is a round-trip through your own
    admin endpoint when ``pip install pymaxim[<extras>]`` would do
    the same thing locally. Matches the ``remove-node`` self-guard
    pattern from C3.2. The state-layer ``drain_node_if_absent`` ALSO
    enforces the self-drain guard, so this step 1 check is defense-
    in-depth. We keep both because the CLI-layer message is
    operator-friendly (points at local pip) while the state-layer
    message is generic — matches C2 A2 fold reasoning.

    Returns the CLI exit code:

    - ``0`` — install succeeded (and resume succeeded or was skipped)
    - ``1`` — install failed (node left drained with loud hint)
    - ``2`` — refused pre-install (self, unknown, bad tokens, drain fail)
    - ``3`` — install succeeded but post-install auto-resume failed
      (Fold I4 — distinguishable from ``1`` so operators can tell
      "upgraded but stuck in drain" from "failed and stuck in drain")
    """
    # 1. Self-guard — match C3.2 remove-node pattern. See docstring
    # note on defense-in-depth with the state-layer self-drain guard.
    if node.name == mesh.self_name:
        print(
            f"✗ Refusing to install on self ({node.name!r}) — remote-installing\n"
            f"  on yourself is a round-trip through your own admin endpoint.\n"
            f"  → Use `pip install pymaxim[<extras>]` directly on this machine.",
            file=sys.stderr,
        )
        return 2

    # 2. Parse install tokens — rejects positional URL with a hint to
    # use the no-mesh.yml fallback verb.
    try:
        raw_tokens = _parse_node_install_tokens(verb_args)
    except ValueError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 2

    # 3. Empty-tokens guard — matches _cmd_install's usage shape.
    if not raw_tokens:
        print(
            f"Usage: maxim peer --node {node.name} install <extras_or_packages>",
            file=sys.stderr,
        )
        print("  extras: " + ", ".join(sorted(KNOWN_EXTRAS)), file=sys.stderr)
        print(
            f"  Example: maxim peer --node {node.name} install semantic",
            file=sys.stderr,
        )
        return 2

    # 4. Classify into extras + packages. The classifier drops
    # whitespace-only tokens; if every token was whitespace, surface
    # the same usage error rather than POSTing an empty body.
    extras, packages = classify_install_tokens(raw_tokens)
    if not extras and not packages:
        print(
            "✗ No valid install tokens after stripping whitespace.",
            file=sys.stderr,
        )
        return 2

    # 5. Atomic drain-if-absent. Returns (drain_set, we_added_it).
    # If ``we_added_it`` is False, the node was already drained at
    # the moment we held the lock — sticky operator intent, skip
    # auto-resume. If True, we added the entry and are responsible
    # for resuming on success. The TOCTOU window between
    # ``read_drained_nodes`` and ``drain_node`` that existed in the
    # pre-fold code is closed by collapsing both operations into a
    # single filelock critical section inside the state layer
    # (Fold CC2).
    known_names = {n.name for n in mesh.nodes}
    try:
        _, drained_here = _drain_node_if_absent(
            node.name,
            known_names,
            self_name=mesh.self_name,
            force_self=False,
        )
    except DrainError as e:
        print(f"✗ Failed to drain {node.name!r}: {e}", file=sys.stderr)
        return 2

    if drained_here:
        print(f"✓ Drained {node.name!r} for install.")
    else:
        print(f"ℹ {node.name!r} already drained by operator — skipping drain/resume bookkeeping.")

    # 6. Delegate to the shared install core, wrapped so any
    # unexpected exception (KeyboardInterrupt, programming bug not
    # caught by install_on_target's own try/except) still prints the
    # "still drained" hint before re-raising (Fold I6). Install
    # failure (install_rc != 0) is NOT raised — it's returned as an
    # exit code and handled by the success/fail branches below.
    try:
        install_rc = install_on_target(
            node.url,
            mesh.cluster_key,
            extras,
            packages,
        )
    except BaseException:
        # BaseException covers KeyboardInterrupt + SystemExit, not
        # just Exception subclasses. Then re-raise so the caller
        # sees the original failure mode.
        if drained_here:
            print(
                f"\n⚠ Install interrupted. Node {node.name!r} is STILL DRAINED.\n"
                f"  → After handling the interruption, run:\n"
                f"      maxim peer --node {node.name} resume",
                file=sys.stderr,
            )
        raise

    # 7. Success path — resume only if we drained it ourselves. If
    # the resume itself fails (filelock timeout or similar), warn the
    # operator and return exit code 3 so operators tailing return
    # codes can distinguish "install succeeded but node is stuck in
    # drain" from "install failed" (Fold I4).
    if install_rc == 0:
        if drained_here:
            try:
                _resume_node(node.name, known_names)
                print(f"✓ Resumed {node.name!r} after install.")
            except DrainError as e:
                print(
                    f"⚠ Install succeeded but resume failed: {e}\n  → Run: maxim peer --node {node.name} resume",
                    file=sys.stderr,
                )
                return _EXIT_RESUME_FAILED_POST_OP
        return 0

    # 8. Failure path — leave drained with loud message. The
    # operator is responsible for debugging and running resume after
    # they've confirmed the node is healthy. This is the "fail loud,
    # don't auto-recover from operator-explicit verbs" pattern from
    # C2. The banner is the LAST thing on stderr — the core's own
    # error output comes first, then this banner, so the operator
    # sees the actionable recovery hint at the bottom of their
    # terminal.
    if drained_here:
        print(
            f"\n✗ Install failed. Node {node.name!r} is STILL DRAINED.\n"
            f"  → Debug the failure above, then run:\n"
            f"      maxim peer --node {node.name} resume",
            file=sys.stderr,
        )
    return install_rc


def _run_node_update(
    mesh: MeshConfig,
    node: MeshNode,
    verb_args: list[str],
) -> int:
    """Execute ``--node <name> update [--dry-run] [--force] [--branch <b>]``.

    Plan 4 C3.5. Mesh-aware update with drain → update → resume
    composition. Same pattern as :func:`_run_node_install`.

    ``--dry-run`` skips drain/resume (preview only, no mutation).
    Self-guard refuses updating yourself (use ``maxim peer update``
    directly on the leader).

    Returns the CLI exit code:

    - ``0`` — update succeeded (and resume succeeded or was skipped)
    - ``1`` — update failed (node left drained with loud hint)
    - ``2`` — refused pre-update (self, unknown, drain fail)
    - ``3`` — update succeeded but post-update auto-resume failed
    """
    # Self-guard FIRST — before parsing flags. Even dry-run previews
    # against self are a round-trip through your own admin endpoint.
    # Review fold: both lenses cross-confirmed that the original
    # ordering (dry-run before self-guard) silently bypassed the guard.
    if node.name == mesh.self_name:
        print(
            f"✗ Refusing to update self ({node.name!r}) — remote-updating\n"
            f"  yourself is a round-trip through your own admin endpoint.\n"
            f"  → Use `maxim peer update` directly on this machine.",
            file=sys.stderr,
        )
        return 2

    # Parse verb args (after self-guard so flag parsing doesn't mask
    # the self-check).
    branch = "main"
    dry_run = False
    force = False
    mode = "auto"
    version: str | None = None
    i = 0
    while i < len(verb_args):
        a = verb_args[i]
        if a == "--branch" and i + 1 < len(verb_args):
            branch = verb_args[i + 1]
            i += 2
            continue
        if a == "--dev":
            mode = "dev"
            if i + 1 < len(verb_args) and not verb_args[i + 1].startswith("-"):
                i += 1
                branch = verb_args[i]
        elif a == "--version" and i + 1 < len(verb_args):
            version = verb_args[i + 1]
            i += 2
            continue
        elif a in ("--dry-run", "--preview"):
            dry_run = True
        elif a in ("--force", "-f"):
            force = True
        i += 1

    if version and mode == "dev":
        print("--version and --dev are mutually exclusive.", file=sys.stderr)
        return 2

    # Dry-run is read-only — no drain needed.
    if dry_run:
        return update_on_target(
            node.url,
            mesh.cluster_key,
            branch=branch,
            dry_run=True,
            force=False,
            mode=mode,
            version=version,
        )

    # Atomic drain-if-absent.
    known_names = {n.name for n in mesh.nodes}
    try:
        _, drained_here = _drain_node_if_absent(
            node.name,
            known_names,
            self_name=mesh.self_name,
            force_self=False,
        )
    except DrainError as e:
        print(f"✗ Failed to drain {node.name!r}: {e}", file=sys.stderr)
        return 2

    if drained_here:
        print(f"✓ Drained {node.name!r} for update.")
    else:
        print(f"ℹ {node.name!r} already drained by operator — skipping drain/resume bookkeeping.")

    # Delegate to the shared core.
    try:
        op_rc = update_on_target(
            node.url,
            mesh.cluster_key,
            branch=branch,
            dry_run=False,
            force=force,
            mode=mode,
            version=version,
        )
    except BaseException:
        if drained_here:
            print(
                f"\n⚠ Update interrupted. Node {node.name!r} is STILL DRAINED.\n"
                f"  → After handling the interruption, run:\n"
                f"      maxim peer --node {node.name} resume",
                file=sys.stderr,
            )
        raise

    # Resume on success.
    if op_rc == 0:
        if drained_here:
            try:
                _resume_node(node.name, known_names)
                print(f"✓ Resumed {node.name!r} after update.")
            except DrainError as e:
                print(
                    f"⚠ Update succeeded but resume failed: {e}\n  → Run: maxim peer --node {node.name} resume",
                    file=sys.stderr,
                )
                return _EXIT_RESUME_FAILED_POST_OP
        return 0

    # Failure path.
    if drained_here:
        print(
            f"\n✗ Update failed. Node {node.name!r} is STILL DRAINED.\n"
            f"  → Debug the failure above, then run:\n"
            f"      maxim peer --node {node.name} resume",
            file=sys.stderr,
        )
    return op_rc


def _run_node_restart(
    mesh: MeshConfig,
    node: MeshNode,
    verb_args: list[str],
) -> int:
    """Execute ``--node <name> restart``.

    Plan 4 C3.5. Mesh-aware restart with drain → restart → resume
    composition. Same pattern as :func:`_run_node_install`.

    Self-guard refuses restarting yourself (use ``maxim peer restart``
    directly on the leader).

    Returns the CLI exit code:

    - ``0`` — restart succeeded (and resume succeeded or was skipped)
    - ``1`` — restart failed (node left drained with loud hint)
    - ``2`` — refused pre-restart (self, unknown, drain fail)
    - ``3`` — restart succeeded but post-restart auto-resume failed
    """
    del verb_args  # no options yet

    # Self-guard.
    if node.name == mesh.self_name:
        print(
            f"✗ Refusing to restart self ({node.name!r}) — remote-restarting\n"
            f"  yourself is a round-trip through your own admin endpoint.\n"
            f"  → Use `maxim peer restart` directly on this machine.",
            file=sys.stderr,
        )
        return 2

    # Atomic drain-if-absent.
    known_names = {n.name for n in mesh.nodes}
    try:
        _, drained_here = _drain_node_if_absent(
            node.name,
            known_names,
            self_name=mesh.self_name,
            force_self=False,
        )
    except DrainError as e:
        print(f"✗ Failed to drain {node.name!r}: {e}", file=sys.stderr)
        return 2

    if drained_here:
        print(f"✓ Drained {node.name!r} for restart.")
    else:
        print(f"ℹ {node.name!r} already drained by operator — skipping drain/resume bookkeeping.")

    # Delegate to the shared core.
    try:
        op_rc = restart_on_target(node.url, mesh.cluster_key)
    except BaseException:
        if drained_here:
            print(
                f"\n⚠ Restart interrupted. Node {node.name!r} is STILL DRAINED.\n"
                f"  → After handling the interruption, run:\n"
                f"      maxim peer --node {node.name} resume",
                file=sys.stderr,
            )
        raise

    # Resume on success.
    if op_rc == 0:
        if drained_here:
            try:
                _resume_node(node.name, known_names)
                print(f"✓ Resumed {node.name!r} after restart.")
            except DrainError as e:
                print(
                    f"⚠ Restart succeeded but resume failed: {e}\n  → Run: maxim peer --node {node.name} resume",
                    file=sys.stderr,
                )
                return _EXIT_RESUME_FAILED_POST_OP
        return 0

    # Failure path.
    if drained_here:
        print(
            f"\n✗ Restart failed. Node {node.name!r} is STILL DRAINED.\n"
            f"  → Debug the failure above, then run:\n"
            f"      maxim peer --node {node.name} resume",
            file=sys.stderr,
        )
    return op_rc


def _run_node_llm(
    mesh: MeshConfig,
    node: MeshNode,
    verb_args: list[str],
) -> int:
    """Execute ``--node <name> llm <model>``.

    Plan 4 C3.6. Mesh-aware LLM swap with drain → swap → resume
    composition. Same pattern as :func:`_run_node_install`.

    This is the key enabler for C5 capacity-aware routing — per-node
    model assignment means the router can know which node runs which
    model.

    Self-guard refuses swapping yourself (use ``maxim peer llm``
    directly on the leader).

    Returns the CLI exit code:

    - ``0`` — swap succeeded (and resume succeeded or was skipped)
    - ``1`` — swap failed (node left drained with loud hint)
    - ``2`` — refused pre-swap (self, unknown, no model, drain fail)
    - ``3`` — swap succeeded but post-swap auto-resume failed
    """
    # Parse model from verb_args.
    model: str | None = None
    for arg in verb_args:
        if arg.startswith("--"):
            continue
        model = arg
        break

    if not model:
        print(
            f"Usage: maxim peer --node {node.name} llm <model>",
            file=sys.stderr,
        )
        print("  Example: maxim peer --node leader-desk llm qwen2.5-14b", file=sys.stderr)
        return 2

    # Self-guard.
    if node.name == mesh.self_name:
        print(
            f"✗ Refusing to swap LLM on self ({node.name!r}) — remote-swapping\n"
            f"  yourself is a round-trip through your own admin endpoint.\n"
            f"  → Use `maxim peer llm {model}` directly on this machine.",
            file=sys.stderr,
        )
        return 2

    # Atomic drain-if-absent.
    known_names = {n.name for n in mesh.nodes}
    try:
        _, drained_here = _drain_node_if_absent(
            node.name,
            known_names,
            self_name=mesh.self_name,
            force_self=False,
        )
    except DrainError as e:
        print(f"✗ Failed to drain {node.name!r}: {e}", file=sys.stderr)
        return 2

    if drained_here:
        print(f"✓ Drained {node.name!r} for LLM swap.")
    else:
        print(f"ℹ {node.name!r} already drained by operator — skipping drain/resume bookkeeping.")

    # Delegate to the shared core.
    try:
        op_rc = llm_swap_on_target(node.url, mesh.cluster_key, model)
    except BaseException:
        if drained_here:
            print(
                f"\n⚠ LLM swap interrupted. Node {node.name!r} is STILL DRAINED.\n"
                f"  → After handling the interruption, run:\n"
                f"      maxim peer --node {node.name} resume",
                file=sys.stderr,
            )
        raise

    # Resume on success.
    if op_rc == 0:
        if drained_here:
            try:
                _resume_node(node.name, known_names)
                print(f"✓ Resumed {node.name!r} after LLM swap.")
            except DrainError as e:
                print(
                    f"⚠ LLM swap succeeded but resume failed: {e}\n  → Run: maxim peer --node {node.name} resume",
                    file=sys.stderr,
                )
                return _EXIT_RESUME_FAILED_POST_OP
        return 0

    # Failure path.
    if drained_here:
        print(
            f"\n✗ LLM swap failed. Node {node.name!r} is STILL DRAINED.\n"
            f"  → Debug the failure above, then run:\n"
            f"      maxim peer --node {node.name} resume",
            file=sys.stderr,
        )
    return op_rc


# ─── internals ──────────────────────────────────────────────────────────


def _load_mesh_or_report_error() -> tuple[MeshConfig | None, int]:
    """Load mesh.yml (or synthesize from peer.yml), printing operator-
    readable errors to stderr on failure. Returns (mesh, exit_code).
    """
    try:
        mesh = read_or_synthesize_mesh_config()
    except MeshConfigError as e:
        print(f"✗ {e}", file=sys.stderr)
        print("  → Fix the schema error above, or run `maxim doctor`.", file=sys.stderr)
        return None, 2
    if mesh is None:
        print("✗ No mesh.yml or peer.yml configured on this machine.", file=sys.stderr)
        print("  → Run `maxim peer connect <url>` to set up a single leader, or", file=sys.stderr)
        print("    create ~/.config/maxim/mesh.yml with a nodes: list.", file=sys.stderr)
        return None, 1
    return mesh, 0


def _drained_report(node: MeshNode) -> _NodeProbeReport:
    """Build a ``_NodeProbeReport`` for a drained node without making
    a network call. Plan 4 C2: drain is operator-intentional, not a
    failure signal.
    """
    return _NodeProbeReport(
        node=node,
        status="info",
        message=f"drained (not probed) — {node.url}",
        fix=None,
        latency_ms=None,
        drained=True,
    )


def _probe_node(node: MeshNode, cluster_key: str) -> _NodeProbeReport:
    """Probe one node via ``_MaximPeerBackend.for_url(...).health_check()``.

    Lazy backend import avoids pulling the backend stack when the verb
    doesn't need it. ``ImportError`` is caught defensively in case the
    `llm-server` extra isn't installed; any *other* exception from
    ``health_check`` is a bug — ``_MaximPeerBackend.health_check`` is
    contractually required to return a ``ProbeResult`` rather than
    raise (Plan 3 R2.6), so catching ``Exception`` broadly would hide
    real regressions as silent warnings.
    """
    try:
        from maxim.models.language.maxim_peer_backend import _MaximPeerBackend
    except ImportError as e:
        return _NodeProbeReport(
            node=node,
            status="warn",
            message=f"peer backend import failed: {e}",
            fix="Install the llm-server extra: pip install -e '.[llm-server]'",
        )

    backend = _MaximPeerBackend.for_url(node.url, api_key=cluster_key)
    result = backend.health_check(enable_stage2=True)

    outcome = getattr(result, "outcome", "other")
    detail = getattr(result, "detail", "")
    latency = getattr(result, "latency_ms", None)
    classification = classify_probe_outcome(outcome, detail, latency, node.url)
    return _NodeProbeReport(
        node=node,
        status=classification.status,
        message=classification.message,
        fix=classification.fix,
        latency_ms=latency,
    )


# ─── output rendering ───────────────────────────────────────────────────


_STATUS_SYMBOLS = {
    "ok": "✓",
    "fail": "✗",
    "warn": "⚠",
    "info": "ℹ",  # Plan 4 C2: drained nodes render with status=info
}


def _print_table(
    mesh: MeshConfig,
    reports: list[_NodeProbeReport],
    *,
    orphans: list[str] | None = None,
) -> None:
    """Human-readable table output.

    Plan 4 C2: drained nodes render with the ``⊝`` symbol and skip
    the network-probe line. Orphan drain entries (drain state names
    that no longer match any mesh.yml node) render as a trailing
    warning block — not a hard fail, operator may be mid-edit.
    """
    self_name = mesh.self_name
    drained_count = sum(1 for r in reports if r.drained)
    print()
    header = f"━━━ Mesh: {len(reports)} node(s), self={self_name}"
    if drained_count:
        header += f", {drained_count} drained"
    print(header + " ━━━")
    name_w = max((len(r.node.name) for r in reports), default=4)
    role_w = max((len(r.node.role) for r in reports), default=6)
    for r in reports:
        # Drained nodes get a dedicated symbol to distinguish from healthy
        # info results (future-proofing if other info cases appear).
        symbol = "⊝" if r.drained else _STATUS_SYMBOLS.get(r.status, "?")
        marker = " (self)" if r.node.name == self_name else ""
        print(f"  {symbol} {r.node.name.ljust(name_w)}  {r.node.role.ljust(role_w)}  {r.node.url}{marker}")
        print(f"      → {r.message}")
        if r.fix:
            for line in r.fix.splitlines():
                print(f"        {line}")
    if orphans:
        print()
        print(f"  ⚠ Orphan drain entries ({len(orphans)}):")
        for name in orphans:
            print(f"    {name}  (no such node in mesh.yml)")
        print("    → Run `maxim peer --node <name> resume` to clean up.")
    print()


def _print_single_node(report: _NodeProbeReport) -> None:
    symbol = "⊝" if report.drained else _STATUS_SYMBOLS.get(report.status, "?")
    print(f"{symbol} {report.node.name} ({report.node.role}) {report.node.url}")
    print(f"  → {report.message}")
    if report.fix:
        for line in report.fix.splitlines():
            print(f"    {line}")


def _print_json(
    mesh: MeshConfig,
    reports: list[_NodeProbeReport],
    *,
    orphans: list[str] | None = None,
) -> None:
    """Machine-readable output. Reuses the shape of
    ``maxim doctor --json`` so operator tooling can parse both with
    the same schema.

    Plan 4 C2 adds ``drained`` boolean per node and a top-level
    ``orphans`` array for drain state entries that don't match any
    mesh.yml node.
    """
    output = {
        "self": mesh.self_name,
        "protocol_version": mesh.protocol_version,
        "nodes": [
            {
                "name": r.node.name,
                "url": r.node.url,
                "role": r.node.role,
                "status": r.status,
                "message": r.message,
                "fix": r.fix,
                "latency_ms": r.latency_ms,
                "drained": r.drained,
            }
            for r in reports
        ],
        "orphans": orphans or [],
        "worst_status": _worst_status(reports),
    }
    print(json.dumps(output, indent=2))


def _worst_status(reports: list[_NodeProbeReport]) -> str:
    worst = "ok"
    for r in reports:
        if r.status == "fail":
            return "fail"
        if r.status == "warn":
            worst = "warn"
    return worst


__all__ = [
    "run_list_drained",
    "run_list_nodes",
    "run_node_subcommand",
]
