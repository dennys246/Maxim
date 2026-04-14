"""Mesh config: multi-node cluster topology + drain state.

Plan 4 Stage C1. Sibling of ``peer.yml`` (leader URL + key for a single
peer-leader pair) — ``mesh.yml`` describes a named set of nodes that
talk to each other through a shared cluster key.

Stored at the same platform-specific location as ``peer.yml``:

  POSIX:   ``~/.config/maxim/mesh.yml``
  Windows: ``%APPDATA%\\maxim\\mesh.yml``

Schema::

    cluster_key: sk-...
    self: leader-desk            # must match one entry in nodes:
    protocol_version: 1
    nodes:
      - name: leader-desk
        url: http://192.168.1.10:8099/v1
        role: leader
      - name: mac-studio
        url: https://mac.example.com/v1
        role: peer
    drain:                       # optional; names of nodes to skip
      - mac-studio

**Fallback path.** When ``mesh.yml`` is absent,
:func:`read_or_synthesize_mesh_config` loads the legacy ``peer.yml``
and synthesizes a one-node mesh (the leader only). Existing users see
zero behavior change: the new verbs just show one node.

**Parse-time validation is syntax-only.** URL reachability (DNS, SSRF)
is deferred to probe time via
:meth:`maxim.models.language.maxim_peer_backend._MaximPeerBackend.health_check`.
This keeps parse offline-safe for tests and startup.

**Drain state is persisted separately** at
``~/.maxim/util/drained_nodes.{role}.txt`` (one node name per line).
Role comes from ``MAXIM_ROLE`` env var (Plan 2 R2a) so leader drain
state does not leak to peer drain state on the same machine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from maxim.peer.config import peer_config_path, read_peer_config
from maxim.utils.atomic_io import atomic_write_text
from maxim.utils.paths import resolve_user_state

NodeRole = Literal["leader", "peer"]


class MeshConfigError(ValueError):
    """Raised on malformed ``mesh.yml``. Carries a line number when available."""

    def __init__(self, message: str, *, line: int | None = None) -> None:
        self.line = line
        if line is not None:
            super().__init__(f"mesh.yml line {line}: {message}")
        else:
            super().__init__(f"mesh.yml: {message}")


@dataclass(frozen=True)
class MeshNode:
    name: str
    url: str
    role: NodeRole


@dataclass(frozen=True)
class MeshConfig:
    cluster_key: str
    self_name: str
    nodes: tuple[MeshNode, ...]
    protocol_version: int = 1
    drain: tuple[str, ...] = field(default_factory=tuple)

    def get_node(self, name: str) -> MeshNode | None:
        for n in self.nodes:
            if n.name == name:
                return n
        return None

    def self_node(self) -> MeshNode | None:
        return self.get_node(self.self_name)


def mesh_config_path() -> Path:
    """Return the platform-appropriate mesh config path (next to peer.yml)."""
    return peer_config_path().parent / "mesh.yml"


def _validate_url_syntax(url: str, *, line: int) -> None:
    """Syntax-only URL validation. DNS/SSRF is deferred to probe time."""
    try:
        parsed = urlparse(url)
    except Exception as e:
        raise MeshConfigError(f"malformed url {url!r}: {e}", line=line) from e
    if parsed.scheme not in ("http", "https"):
        raise MeshConfigError(
            f"url {url!r} must use http:// or https:// (got {parsed.scheme!r})",
            line=line,
        )
    if not parsed.hostname:
        raise MeshConfigError(f"url {url!r} has no hostname", line=line)


def parse_mesh_config(content: str) -> MeshConfig:
    """Parse mesh.yml content without PyYAML (matches peer.yml style).

    Supports the minimal YAML dialect used by ``peer.yml``: top-level
    ``key: value`` lines plus a ``nodes:`` list where each entry is a
    ``- name: foo`` block followed by indented ``key: value`` lines.
    Good enough for a 2-5 node mesh; if operators want nested anchors
    they can file an issue and we bring in PyYAML.

    Raises :class:`MeshConfigError` with a line number on any problem.
    """
    cluster_key: str | None = None
    self_name: str | None = None
    protocol_version = 1
    nodes: list[MeshNode] = []
    drain: list[str] = []

    lines = content.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        line_no = i + 1
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        # Top-level scalars
        if raw[:1] not in (" ", "\t") and ":" in stripped and not stripped.startswith("-"):
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip()
            if key == "cluster_key":
                cluster_key = value
                i += 1
                continue
            if key == "self":
                self_name = value
                i += 1
                continue
            if key == "protocol_version":
                try:
                    protocol_version = int(value)
                except ValueError as e:
                    raise MeshConfigError(
                        f"protocol_version must be an integer, got {value!r}",
                        line=line_no,
                    ) from e
                i += 1
                continue
            if key == "nodes":
                i, nodes = _parse_nodes_block(lines, i + 1)
                continue
            if key == "drain":
                i, drain = _parse_drain_block(lines, i + 1)
                continue
            raise MeshConfigError(f"unknown top-level key {key!r}", line=line_no)

        raise MeshConfigError(f"unexpected line: {stripped!r}", line=line_no)

    if not cluster_key:
        raise MeshConfigError("missing required field 'cluster_key'")
    if not self_name:
        raise MeshConfigError("missing required field 'self' (must name one of the nodes)")
    if not nodes:
        raise MeshConfigError("missing required field 'nodes' (must list at least one node)")

    node_names = {n.name for n in nodes}
    if self_name not in node_names:
        raise MeshConfigError(f"'self: {self_name}' does not match any entry in nodes: (known: {sorted(node_names)})")

    if protocol_version != 1:
        raise MeshConfigError(f"unsupported protocol_version {protocol_version} (this build speaks 1)")

    return MeshConfig(
        cluster_key=cluster_key,
        self_name=self_name,
        protocol_version=protocol_version,
        nodes=tuple(nodes),
        drain=tuple(drain),
    )


def _parse_nodes_block(lines: list[str], start: int) -> tuple[int, list[MeshNode]]:
    """Parse a ``nodes:`` list block. Returns (next_index, nodes)."""
    nodes: list[MeshNode] = []
    current: dict[str, str] = {}
    current_line: int | None = None
    i = start
    while i < len(lines):
        raw = lines[i]
        line_no = i + 1
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        # End of indented block: a non-indented line closes the list.
        if raw[:1] not in (" ", "\t"):
            break
        if stripped.startswith("- "):
            if current:
                nodes.append(_finalize_node(current, line=current_line or line_no))
                current = {}
            current_line = line_no
            body = stripped[2:].strip()
            if body:
                key, _, value = body.partition(":")
                current[key.strip()] = value.strip()
        else:
            key, _, value = stripped.partition(":")
            current[key.strip()] = value.strip()
        i += 1
    if current:
        nodes.append(_finalize_node(current, line=current_line or start + 1))
    return i, nodes


def _parse_drain_block(lines: list[str], start: int) -> tuple[int, list[str]]:
    """Parse a ``drain:`` list block. Returns (next_index, names)."""
    names: list[str] = []
    i = start
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        if raw[:1] not in (" ", "\t"):
            break
        if stripped.startswith("- "):
            names.append(stripped[2:].strip())
        i += 1
    return i, names


def _finalize_node(raw: dict[str, str], *, line: int) -> MeshNode:
    name = raw.get("name", "").strip()
    url = raw.get("url", "").strip()
    role = raw.get("role", "").strip()
    if not name:
        raise MeshConfigError("node entry missing 'name'", line=line)
    if not url:
        raise MeshConfigError(f"node {name!r} missing 'url'", line=line)
    if role not in ("leader", "peer"):
        raise MeshConfigError(
            f"node {name!r} has invalid role {role!r} (expected 'leader' or 'peer')",
            line=line,
        )
    _validate_url_syntax(url, line=line)
    return MeshNode(name=name, url=url, role=role)  # type: ignore[arg-type]


def read_mesh_config(path: Path | None = None) -> MeshConfig | None:
    """Load the mesh config from disk, or None if missing.

    Raises :class:`MeshConfigError` on malformed content. Callers that
    want a "fall back to peer.yml if absent" path should use
    :func:`read_or_synthesize_mesh_config` instead.
    """
    p = path or mesh_config_path()
    if not p.is_file():
        return None
    try:
        content = p.read_text()
    except OSError:
        return None
    return parse_mesh_config(content)


def synthesize_from_peer_config() -> MeshConfig | None:
    """Build a one-node mesh from the legacy ``peer.yml`` (leader only).

    Used by :func:`read_or_synthesize_mesh_config` when ``mesh.yml``
    doesn't exist. Returns None if ``peer.yml`` is also absent.

    The synthesized mesh has:

    - ``self_name = "leader"`` (single-node topology; this peer sees
      just the leader)
    - A single ``MeshNode(name="leader", url=peer.url, role="leader")``
    - ``cluster_key`` set from ``peer.api_key``
    - Empty drain list
    """
    peer = read_peer_config()
    if peer is None:
        return None
    node = MeshNode(name="leader", url=peer.url, role="leader")
    return MeshConfig(
        cluster_key=peer.api_key,
        self_name="leader",
        nodes=(node,),
        protocol_version=1,
        drain=(),
    )


def read_or_synthesize_mesh_config() -> MeshConfig | None:
    """Return a :class:`MeshConfig` from ``mesh.yml`` OR a synthesized
    one from ``peer.yml``. Returns None if neither source exists.

    This is the entry point callers should use when they want to
    operate on "whatever the user has configured" without caring which
    of the two files it came from.
    """
    mesh = read_mesh_config()
    if mesh is not None:
        return mesh
    return synthesize_from_peer_config()


# ─── drain state persistence ─────────────────────────────────────────────


def _drain_state_path() -> Path:
    """Role-scoped drain state file. Role from ``MAXIM_ROLE`` (Plan 2 R2a)."""
    role = os.environ.get("MAXIM_ROLE", "leader")
    return resolve_user_state(f"util/drained_nodes.{role}.txt")


def read_drained_nodes() -> set[str]:
    """Return the set of node names currently drained (role-scoped)."""
    path = _drain_state_path()
    if not path.is_file():
        return set()
    try:
        content = path.read_text()
    except OSError:
        return set()
    return {line.strip() for line in content.splitlines() if line.strip()}


def write_drained_nodes(names: set[str]) -> Path:
    """Persist the drain set atomically. Returns the path written."""
    path = _drain_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(sorted(names))
    if content:
        content += "\n"
    atomic_write_text(str(path), content)
    return path


def drain_node(name: str) -> set[str]:
    """Add ``name`` to the drain set. Returns the new set."""
    current = read_drained_nodes()
    current.add(name)
    write_drained_nodes(current)
    return current


def resume_node(name: str) -> set[str]:
    """Remove ``name`` from the drain set. Returns the new set."""
    current = read_drained_nodes()
    current.discard(name)
    write_drained_nodes(current)
    return current


__all__ = [
    "MeshConfig",
    "MeshConfigError",
    "MeshNode",
    "NodeRole",
    "drain_node",
    "mesh_config_path",
    "parse_mesh_config",
    "read_drained_nodes",
    "read_mesh_config",
    "read_or_synthesize_mesh_config",
    "resume_node",
    "synthesize_from_peer_config",
    "write_drained_nodes",
]
