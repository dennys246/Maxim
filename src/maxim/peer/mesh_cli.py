"""CLI verbs for ``maxim peer list-nodes`` and ``maxim peer --node <name> ...``.

Plan 4 Stage C1. Extends ``maxim peer`` with **read-only** topology
verbs backed by :mod:`maxim.peer.mesh_config`. Node probes route
through :meth:`maxim.models.language.maxim_peer_backend._MaximPeerBackend.for_url`
+ ``health_check`` — the canonical probe entry point (Plan 3 R2.6).

Verbs shipped in C1:

- ``maxim peer list-nodes [--json]`` — table or JSON of nodes + live status
- ``maxim peer --node <name> status`` — per-node probe
- ``maxim peer --node <name> health`` — alias for status

Deferred to C2 (intentional, pre-merge review flagged the original
drain design as under-specified): ``--node drain`` / ``--node resume``,
``mesh.yml::drain`` schema field, runtime drain state persistence,
``--node install`` + VRAM precheck, ``--node refresh``, ``add-node``,
``remove-node``, admin API endpoints, per-agent rate limiting,
request-trace ring buffer, cluster key rotation.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from maxim.peer.mesh_config import (
    MeshConfig,
    MeshConfigError,
    MeshNode,
    read_or_synthesize_mesh_config,
)
from maxim.peer.probe_classify import classify_probe_outcome


MESH_USAGE_HEAD = """\
Usage: maxim peer list-nodes [--json]
       maxim peer --node <name> <status|health>

Read-only mesh topology verbs. Requires mesh.yml (or falls back to
peer.yml as a synthesized one-node mesh). Node probes go through
_MaximPeerBackend.health_check().
"""


@dataclass
class _NodeProbeReport:
    node: MeshNode
    status: str  # ok|fail|warn
    message: str
    fix: str | None = None
    latency_ms: float | None = None


# ─── public entry points ────────────────────────────────────────────────


def run_list_nodes(argv: Sequence[str]) -> int:
    """``maxim peer list-nodes [--json]`` — table of nodes + live status."""
    as_json = "--json" in argv
    mesh, err = _load_mesh_or_report_error()
    if mesh is None:
        return err
    reports = [_probe_node(n, mesh.cluster_key) for n in mesh.nodes]
    if as_json:
        _print_json(mesh, reports)
    else:
        _print_table(mesh, reports)
    # Non-zero exit if any node is failing (warn is tolerated, matching
    # `maxim doctor`'s exit-code contract).
    return 1 if any(r.status == "fail" for r in reports) else 0


def run_node_subcommand(argv: Sequence[str]) -> int:
    """``maxim peer --node <name> <verb>`` dispatcher.

    ``argv`` is everything after ``peer`` (i.e., starts with
    ``["--node", "<name>", "<verb>"]``).
    """
    if not argv or argv[0] != "--node":
        print("Usage: maxim peer --node <name> <status|health>", file=sys.stderr)
        return 2
    if len(argv) < 2:
        print("Missing node name: maxim peer --node <name> <status|health>", file=sys.stderr)
        return 2
    if len(argv) < 3:
        print(f"Missing verb: maxim peer --node {argv[1]} <status|health>", file=sys.stderr)
        return 2
    name = argv[1]
    verb = argv[2]

    mesh, err = _load_mesh_or_report_error()
    if mesh is None:
        return err
    node = mesh.get_node(name)
    if node is None:
        known = ", ".join(n.name for n in mesh.nodes)
        print(f"Unknown node: {name!r} (known: {known})", file=sys.stderr)
        return 2

    if verb in ("status", "health"):
        report = _probe_node(node, mesh.cluster_key)
        _print_single_node(report)
        return 0 if report.status == "ok" else 1
    print(f"Unknown --node verb: {verb!r} (expected status|health)", file=sys.stderr)
    return 2


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
}


def _print_table(mesh: MeshConfig, reports: list[_NodeProbeReport]) -> None:
    """Human-readable table output."""
    self_name = mesh.self_name
    print()
    print(f"━━━ Mesh: {len(reports)} node(s), self={self_name} ━━━")
    name_w = max((len(r.node.name) for r in reports), default=4)
    role_w = max((len(r.node.role) for r in reports), default=6)
    for r in reports:
        symbol = _STATUS_SYMBOLS.get(r.status, "?")
        marker = " (self)" if r.node.name == self_name else ""
        print(f"  {symbol} {r.node.name.ljust(name_w)}  {r.node.role.ljust(role_w)}  {r.node.url}{marker}")
        print(f"      → {r.message}")
        if r.fix:
            for line in r.fix.splitlines():
                print(f"        {line}")
    print()


def _print_single_node(report: _NodeProbeReport) -> None:
    symbol = _STATUS_SYMBOLS.get(report.status, "?")
    print(f"{symbol} {report.node.name} ({report.node.role}) {report.node.url}")
    print(f"  → {report.message}")
    if report.fix:
        for line in report.fix.splitlines():
            print(f"    {line}")


def _print_json(mesh: MeshConfig, reports: list[_NodeProbeReport]) -> None:
    """Machine-readable output. Reuses the shape of
    ``maxim doctor --json`` so operator tooling can parse both with
    the same schema.
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
            }
            for r in reports
        ],
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
    "run_list_nodes",
    "run_node_subcommand",
]
