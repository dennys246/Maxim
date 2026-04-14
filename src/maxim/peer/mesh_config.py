"""Mesh config: multi-node cluster topology for ``maxim peer list-nodes``.

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

**Fallback path.** When ``mesh.yml`` is absent,
:func:`read_or_synthesize_mesh_config` loads the legacy ``peer.yml``
and synthesizes a one-node mesh (the leader only). Existing users see
zero behavior change: the new verbs just show one node.

**Parse-time validation is syntax-only.** URL reachability (DNS, SSRF)
is deferred to probe time via
:meth:`maxim.models.language.maxim_peer_backend._MaximPeerBackend.health_check`.
This keeps parse offline-safe for tests and startup.

**Parser dialect.** Deliberately trivial: flat ``key: value`` top-level
scalars plus a single nested ``nodes:`` list of ``- name: foo`` blocks
with indented ``key: value`` continuation lines. Tabs and inline
comments are rejected loudly — if you need anchors, quoted strings,
tab indentation, or embedded comments, edit ``mesh.yml`` via a
generator, do NOT bolt those features onto this parser. The
architectural escape hatches are PyYAML as an optional extra or
switching the config format to TOML (stdlib ``tomllib``); both are
open questions for C2.

**Drain state is NOT in this module.** Plan 4 C1 review flagged the
original two-layer design (``mesh.yml::drain`` + runtime state file)
as under-specified: no reconciliation contract, no role-detection
timing story, read/write race, no orphan validation. Drain + resume
verbs defer to Plan 4 C2 with a proper design pass. The ``drain:``
field is intentionally absent from the schema for C1.

**Probe classification is NOT in this module.** The shared helper
that maps ``ProbeResult.outcome`` to an operator-readable
``ProbeClassification`` lives in :mod:`maxim.peer.probe_classify`
so that callers can use it without pulling the parser layer. Round 2
review (A4R2) flagged the previous in-module placement as coupling
concerns that should stay separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from maxim.peer.config import peer_config_path, read_peer_config

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


# ─── parser ─────────────────────────────────────────────────────────────


def _strip_inline_comment(value: str) -> str:
    """Remove trailing ``# ...`` from a value.

    Round 2 review E1: the naive ``value.find("#")`` silently truncates
    legitimate ``#`` characters in values — a ``cluster_key: sk-abc#literal``
    would become ``sk-abc`` with no error, surfacing later as
    ``auth_rejected`` with no visible root cause. Require whitespace
    before the ``#`` (the common ``foo  # comment`` pattern) so bare
    ``#`` characters inside a value are preserved.
    """
    stripped = value.strip()
    # Find ``#`` only when preceded by whitespace (or at start-of-string
    # — empty ``#`` comments are caller-rejected elsewhere). Scan so a
    # ``#`` preceded by a non-space character is left in place.
    for i in range(1, len(stripped)):
        if stripped[i] == "#" and stripped[i - 1].isspace():
            return stripped[:i].rstrip()
    return stripped


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

    **Loud errors:** rejects tabs, bare ``-`` entries, duplicate node
    names, unknown top-level keys, and inline comments inside values
    (comments only allowed on their own lines).

    Raises :class:`MeshConfigError` with a line number on any problem.
    """
    cluster_key: str | None = None
    self_name: str | None = None
    protocol_version = 1
    nodes: list[MeshNode] = []

    lines = content.splitlines()
    # Reject tab indentation outright — mixing tabs and spaces is the
    # top source of silent YAML mis-parse bugs.
    for idx, raw in enumerate(lines, start=1):
        if "\t" in raw:
            raise MeshConfigError(
                "tab characters are not allowed; indent with spaces only",
                line=idx,
            )

    i = 0
    while i < len(lines):
        raw = lines[i]
        line_no = i + 1
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        # Top-level scalars: no leading whitespace, has a ``:``, not a list item.
        if not raw.startswith(" ") and ":" in stripped and not stripped.startswith("-"):
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = _strip_inline_comment(value)
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
            raise MeshConfigError(f"unknown top-level key {key!r}", line=line_no)

        raise MeshConfigError(f"unexpected line: {stripped!r}", line=line_no)

    if not cluster_key:
        raise MeshConfigError("missing required field 'cluster_key'")
    if not self_name:
        raise MeshConfigError("missing required field 'self' (must name one of the nodes)")
    if not nodes:
        raise MeshConfigError("missing required field 'nodes' (must list at least one node)")

    node_names = [n.name for n in nodes]
    seen: set[str] = set()
    for name in node_names:
        if name in seen:
            raise MeshConfigError(f"duplicate node name {name!r} in nodes:")
        seen.add(name)

    if self_name not in seen:
        raise MeshConfigError(f"'self: {self_name}' does not match any entry in nodes: (known: {sorted(seen)})")

    if protocol_version != 1:
        raise MeshConfigError(f"unsupported protocol_version {protocol_version} (this build speaks 1)")

    return MeshConfig(
        cluster_key=cluster_key,
        self_name=self_name,
        protocol_version=protocol_version,
        nodes=tuple(nodes),
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
        if not raw.startswith(" "):
            break
        # Bare dash ``- `` is always an error — silent entry corruption
        # bit us during pre-merge review.
        if stripped == "-":
            raise MeshConfigError("empty list entry '-'; each node needs 'name: <val>'", line=line_no)
        if stripped.startswith("- "):
            if current:
                nodes.append(_finalize_node(current, line=current_line or line_no))
                current = {}
            current_line = line_no
            body = stripped[2:].strip()
            if body:
                key, _, value = body.partition(":")
                current[key.strip()] = _strip_inline_comment(value)
        else:
            key, _, value = stripped.partition(":")
            current[key.strip()] = _strip_inline_comment(value)
        i += 1
    if current:
        nodes.append(_finalize_node(current, line=current_line or start + 1))
    return i, nodes


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


# ─── disk I/O ───────────────────────────────────────────────────────────


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
    )


def read_or_synthesize_mesh_config() -> MeshConfig | None:
    """Return a :class:`MeshConfig` from ``mesh.yml`` OR a synthesized
    one from ``peer.yml``. Returns None if neither source exists.
    """
    mesh = read_mesh_config()
    if mesh is not None:
        return mesh
    return synthesize_from_peer_config()


__all__ = [
    "MeshConfig",
    "MeshConfigError",
    "MeshNode",
    "NodeRole",
    "mesh_config_path",
    "parse_mesh_config",
    "read_mesh_config",
    "read_or_synthesize_mesh_config",
    "synthesize_from_peer_config",
]
