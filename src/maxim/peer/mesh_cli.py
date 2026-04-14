"""CLI verbs for ``maxim peer list-nodes`` and ``maxim peer --node <name> ...``.

Plan 4 Stage C1. Extends ``maxim peer`` with read-only topology verbs
backed by :mod:`maxim.peer.mesh_config`. Node probes route through
:meth:`maxim.models.language.maxim_peer_backend._MaximPeerBackend.for_url`
+ ``health_check`` — the canonical probe entry point (Plan 3 R2.6).

Verbs shipped in C1:

- ``maxim peer list-nodes [--json]`` — table or JSON of nodes + live status
- ``maxim peer --node <name> status`` — per-node probe
- ``maxim peer --node <name> health`` — alias for status
- ``maxim peer --node <name> drain`` — persist drain state
- ``maxim peer --node <name> resume`` — clear drain state

Deferred to C2/C3: ``--node <name> install``, ``--node <name> refresh``,
``add-node``, ``remove-node``, admin API endpoints, per-agent rate
limiting, request-trace ring buffer, cluster key rotation.
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
    drain_node,
    read_drained_nodes,
    read_or_synthesize_mesh_config,
    resume_node,
)


MESH_USAGE_HEAD = """\
Usage: maxim peer list-nodes [--json]
       maxim peer --node <name> <status|health|drain|resume>

Read-only mesh topology verbs. Requires mesh.yml (or falls back to
peer.yml as a synthesized one-node mesh). Node probes go through
_MaximPeerBackend.health_check().
"""


@dataclass
class _NodeProbeReport:
    node: MeshNode
    status: str  # ok|fail|warn|info|drained
    message: str
    fix: str | None = None
    latency_ms: float | None = None
    drained: bool = False


# ─── public entry points ────────────────────────────────────────────────


def run_list_nodes(argv: Sequence[str]) -> int:
    """``maxim peer list-nodes [--json]`` — table of nodes + live status."""
    as_json = "--json" in argv
    mesh, err = _load_mesh_or_report_error()
    if mesh is None:
        return err
    drained = read_drained_nodes()
    reports = [_probe_node(n, mesh.cluster_key, drained) for n in mesh.nodes]
    if as_json:
        _print_json(mesh, reports)
    else:
        _print_table(mesh, reports)
    # Non-zero exit if any node is failing (warn is tolerated).
    return 1 if any(r.status == "fail" for r in reports) else 0


def run_node_subcommand(argv: Sequence[str]) -> int:
    """``maxim peer --node <name> <verb>`` dispatcher.

    ``argv`` is everything after ``peer`` (i.e., starts with
    ``["--node", "<name>", "<verb>"]``).
    """
    if len(argv) < 3 or argv[0] != "--node":
        print("Usage: maxim peer --node <name> <status|health|drain|resume>", file=sys.stderr)
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
        drained = read_drained_nodes()
        report = _probe_node(node, mesh.cluster_key, drained)
        _print_single_node(report)
        return 0 if report.status in ("ok", "info", "drained") else 1
    if verb == "drain":
        new = drain_node(name)
        print(f"✓ Drained {name}. Drain set: {sorted(new) or '[]'}")
        return 0
    if verb == "resume":
        new = resume_node(name)
        print(f"✓ Resumed {name}. Drain set: {sorted(new) or '[]'}")
        return 0
    print(f"Unknown --node verb: {verb!r} (expected status|health|drain|resume)", file=sys.stderr)
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


def _probe_node(node: MeshNode, cluster_key: str, drained: set[str]) -> _NodeProbeReport:
    """Probe one node via ``_MaximPeerBackend.for_url(...).health_check()``.

    Drained nodes are reported without probing — drain is intentional,
    not a failure signal.
    """
    if node.name in drained:
        return _NodeProbeReport(
            node=node,
            status="drained",
            message="drained (operator-intentional)",
            drained=True,
        )
    # Lazy import: avoids pulling the backend stack for non-probe verbs
    # like `drain` / `resume`.
    try:
        from maxim.models.language.maxim_peer_backend import _MaximPeerBackend
    except Exception as e:
        return _NodeProbeReport(
            node=node,
            status="warn",
            message=f"peer backend import failed: {e}",
            fix="Install the llm-server extra: pip install -e '.[llm-server]'",
        )

    try:
        backend = _MaximPeerBackend.for_url(node.url, api_key=cluster_key)
        result = backend.health_check(enable_stage2=True)
    except Exception as e:
        return _NodeProbeReport(
            node=node,
            status="warn",
            message=f"probe raised: {type(e).__name__}: {e}",
        )

    return _classify_probe_result(node, result)


def _classify_probe_result(node: MeshNode, result) -> _NodeProbeReport:  # type: ignore[no-untyped-def]
    """Map ``ProbeResult.outcome`` to an operator-readable report.

    Specific-before-general ordering matches the Plan 2 R2c contract for
    typed exception handling — ``auth_rejected`` and ``inference_broken``
    are classified first before falling to the generic ``down`` bucket.
    """
    outcome = getattr(result, "outcome", "other")
    detail = getattr(result, "detail", "")
    latency = getattr(result, "latency_ms", None)

    if outcome == "ok":
        return _NodeProbeReport(
            node=node,
            status="ok",
            message=f"reachable ({detail})",
            latency_ms=latency,
        )
    if outcome == "auth_rejected":
        return _NodeProbeReport(
            node=node,
            status="fail",
            message=f"auth rejected: {detail}",
            fix=(
                "Cluster key mismatch. Rotate on the leader and redistribute:\n"
                "  On leader: maxim tunnel key rotate && maxim tunnel key export\n"
                "  Then update mesh.yml::cluster_key on this node."
            ),
            latency_ms=latency,
        )
    if outcome == "inference_broken":
        return _NodeProbeReport(
            node=node,
            status="fail",
            message=f"chat endpoint broken: {detail}",
            fix=(
                "Leader is reachable but chat/completions is failing. Check:\n"
                "  1. A model is loaded: maxim peer llm --status\n"
                "  2. Leader logs: maxim peer logs\n"
                "  3. VRAM pressure: maxim doctor --as leader"
            ),
            latency_ms=latency,
        )
    if outcome in ("timeout", "connection_refused", "dns_fail", "tls_error", "http_5xx", "other"):
        return _NodeProbeReport(
            node=node,
            status="warn",
            message=f"{outcome.replace('_', ' ')}: {detail}",
            fix=f"Is {node.url} reachable from here? Try: curl -v {node.url}/models",
            latency_ms=latency,
        )
    return _NodeProbeReport(
        node=node,
        status="warn",
        message=f"unknown outcome {outcome!r}: {detail}",
        latency_ms=latency,
    )


# ─── output rendering ───────────────────────────────────────────────────


_STATUS_SYMBOLS = {
    "ok": "✓",
    "fail": "✗",
    "warn": "⚠",
    "info": "ℹ",
    "drained": "⊝",
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
        latency = f" [{r.latency_ms:.0f}ms]" if r.latency_ms is not None else ""
        print(f"  {symbol} {r.node.name.ljust(name_w)}  {r.node.role.ljust(role_w)}  {r.node.url}{marker}")
        print(f"      → {r.message}{latency}")
        if r.fix:
            for line in r.fix.splitlines():
                print(f"        {line}")
    print()


def _print_single_node(report: _NodeProbeReport) -> None:
    symbol = _STATUS_SYMBOLS.get(report.status, "?")
    latency = f" [{report.latency_ms:.0f}ms]" if report.latency_ms is not None else ""
    print(f"{symbol} {report.node.name} ({report.node.role}) {report.node.url}")
    print(f"  → {report.message}{latency}")
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
                "drained": r.drained,
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
