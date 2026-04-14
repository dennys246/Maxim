"""CLI verbs for ``maxim peer list-nodes`` and ``maxim peer --node <name> ...``.

Plan 4 Stage C1 + C2. Extends ``maxim peer`` with mesh topology verbs
backed by :mod:`maxim.peer.mesh_config`. Node probes route through
:meth:`maxim.models.language.maxim_peer_backend._MaximPeerBackend.for_url`
+ ``health_check`` — the canonical probe entry point (Plan 3 R2.6).

Verbs (C1 + C2):

- ``maxim peer list-nodes [--json]`` — table/JSON of nodes + live status
  with drained nodes shown inline
- ``maxim peer list-drained`` — dump the current drain set (C2)
- ``maxim peer --node <name> status`` / ``health`` — per-node probe
- ``maxim peer --node <name> drain`` — add to drain state (C2)
- ``maxim peer --node <name> resume`` — clear from drain state (C2)

Drain state lives in ``~/.maxim/util/drained_nodes.{role}.txt`` via
:mod:`maxim.peer.drain_state`. The split between ``mesh.yml``
(declarative topology) and the runtime drain state file (mutable
state) is a load-bearing invariant — see the CLAUDE.md lesson
"mesh.yml is declarative; ~/.maxim/util/ is mutable state."

Deferred to C3: ``--node install`` + VRAM precheck, ``--node refresh``,
``add-node``, ``remove-node``, ``init-mesh`` (peer.yml → mesh.yml
converter), admin API endpoints, per-agent rate limiting, request-trace
ring buffer, cluster key rotation.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from maxim.peer.drain_state import (
    DrainError,
    drain_node as _drain_node,
    read_drained_nodes,
    resume_node as _resume_node,
)
from maxim.peer.mesh_config import (
    MeshConfig,
    MeshConfigError,
    MeshNode,
    read_or_synthesize_mesh_config,
)
from maxim.peer.probe_classify import classify_probe_outcome


MESH_USAGE_HEAD = """\
Usage: maxim peer list-nodes [--json]
       maxim peer list-drained
       maxim peer --node <name> <status|health|drain|resume>

Mesh topology verbs. Requires mesh.yml (or falls back to peer.yml as
a synthesized one-node mesh). Node probes go through
_MaximPeerBackend.health_check(). Drain state lives at
~/.maxim/util/drained_nodes.{role}.txt.
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

    Plan 4 C2 verbs: ``drain`` / ``resume`` (in addition to C1's
    ``status`` / ``health``). Drain rejects unknown nodes with exit 2
    and lists the known names; resume has the same behavior. Drain is
    idempotent for already-drained nodes (exit 0, informational
    message); resume is idempotent for not-drained nodes.

    ``--force-self`` flag after the verb is required to drain the
    node matching ``mesh.yml::self``: draining yourself strands
    in-flight requests and is almost always a mistake. The override
    exists because there are legitimate cases (e.g., graceful
    shutdown scripts) but it must be explicit.
    """
    valid_verbs = "status|health|drain|resume"
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
    verb_flags = set(argv[3:])

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
        return _run_drain(mesh, node, force_self="--force-self" in verb_flags)
    if verb == "resume":
        return _run_resume(mesh, node)
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
    """
    if node.name == mesh.self_name and not force_self:
        print(
            f"✗ Refusing to drain self ({node.name!r}) — this strands in-flight\n"
            f"  requests and is almost always a mistake.\n"
            f"  → If you're sure, re-run with --force-self:\n"
            f"      maxim peer --node {node.name} drain --force-self",
            file=sys.stderr,
        )
        return 2
    known_names = {n.name for n in mesh.nodes}
    already = node.name in read_drained_nodes(known_names).active
    try:
        new_set = _drain_node(node.name, known_names)
    except DrainError as e:
        # Should be unreachable — we validated via mesh.get_node above —
        # but a concurrent mesh.yml edit between our get_node and the
        # drain_state call could trigger it. Report clearly.
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
