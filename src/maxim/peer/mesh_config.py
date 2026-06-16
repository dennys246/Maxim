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

import logging
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from maxim.peer.config import peer_config_path, read_peer_config
from maxim.utils.atomic_io import atomic_write_secret

logger = logging.getLogger(__name__)

NodeRole = Literal["leader", "peer"]


class MeshConfigError(ValueError):
    """Raised on malformed ``mesh.yml``. Carries a line number when available."""

    def __init__(self, message: str, *, line: int | None = None) -> None:
        self.line = line
        if line is not None:
            super().__init__(f"mesh.yml line {line}: {message}")
        else:
            super().__init__(f"mesh.yml: {message}")


def _validate_yaml_safe(field_name: str, value: str) -> None:
    """E3 fold (C3.1 pre-merge review): reject characters that would
    break round-trip through ``parse_mesh_config``.

    The FROZEN parser dialect:
    - Splits on newlines (``\\n``/``\\r``) — embedded newlines would
      produce extra parse lines, silently corrupting the file
    - Strips inline ``#`` comments preceded by whitespace — values
      like ``sk abc # comment-looking`` round-trip lossy because
      ``# comment-looking`` is dropped
    - Doesn't support quoted strings — leading/trailing whitespace is
      ambiguous

    The writer must reject these at construction time so a future
    code path (e.g. C3 cluster-key rotation that constructs a
    MeshConfig with a fresh secret) can't silently corrupt mesh.yml.
    """
    if "\n" in value or "\r" in value:
        raise ValueError(
            f"{field_name}={value!r} contains newline characters; "
            "the FROZEN parser dialect treats newlines as line breaks "
            "so the value would round-trip lossily."
        )
    # Match `<whitespace>#` — the inline-comment trigger in the parser
    for i in range(1, len(value)):
        if value[i] == "#" and value[i - 1].isspace():
            raise ValueError(
                f"{field_name}={value!r} contains whitespace followed by '#'; "
                "the FROZEN parser strips this as an inline comment so the "
                "value would round-trip lossily. Use '#' without preceding "
                "whitespace if the literal '#' is required (e.g. "
                "'sk-cluster#tag' parses correctly)."
            )


@dataclass(frozen=True)
class MeshNode:
    """One node entry in ``mesh.yml``.

    SHAPE-FROZEN at 1.0 (CC3). The hand-rolled ``mesh.yml`` parser
    dialect is FROZEN per CLAUDE.md — adding a new ``MeshNode`` field
    requires extending both the parser and ``to_yaml_lines`` (which
    enforces round-trip integrity per the docstring on that method).
    Adding any field post-1.0 is a major-version-bump change for the
    cross-host wire format; the parser-frozen dialect is the gate.
    """

    name: str
    url: str
    role: NodeRole

    def __post_init__(self) -> None:
        # E3 fold: defense in depth — reject parser-incompatible
        # values at construction time. The synthesizer + the
        # init-mesh verb both go through this, so a bad peer.yml
        # surfaces as a ValueError immediately rather than producing
        # a corrupt mesh.yml that the next read fails on.
        _validate_yaml_safe("MeshNode.name", self.name)
        _validate_yaml_safe("MeshNode.url", self.url)
        _validate_yaml_safe("MeshNode.role", self.role)

    def to_yaml_lines(self) -> list[str]:
        """A3 fold (C3.1 pre-merge review): serialize this node as
        the list of YAML lines it contributes to a parent
        :meth:`MeshConfig.to_yaml`.

        Pushing serialization into the owning type means any future
        field added to ``MeshNode`` (e.g. ``public_key`` for C3
        cluster key rotation) MUST extend this method to land in the
        round-trip — there's no way for ``MeshConfig.to_yaml`` to
        silently drop the new field. Without this method, the parent
        serializer would have field-by-field reach-in code that
        silently goes stale when the dataclass grows.
        """
        return [
            f"  - name: {self.name}",
            f"    url: {self.url}",
            f"    role: {self.role}",
        ]


@dataclass(frozen=True)
class MeshConfig:
    """Top-level ``mesh.yml`` contents.

    SHAPE-FROZEN at 1.0 (CC3). Same rationale as :class:`MeshNode` —
    parser-frozen YAML dialect on the cross-host wire. ``protocol_version``
    is the deliberate forward-compat knob; bump that integer when the
    wire format changes incompatibly. Adding any other field post-1.0
    requires the parser, ``to_yaml`` writer, and round-trip test all
    to land together — *except* the CC13 reserved-null fields below,
    which are a deliberate carve-out (the slot is named at 1.0; the
    parser + writer + round-trip land together only when 1.1 activates
    a field). See the next paragraph.

    **CC13 auth format-freeze (reserved-null sibling fields).** The three
    ``cluster_*`` fields below are reserved at 1.0 so 1.1+ mesh-auth
    schemes (key rotation, asymmetric trust anchors, mTLS) can land
    WITHOUT changing this frozen dataclass shape — which would itself be
    a breaking change per the CC3 SHAPE-FROZEN contract. They are always
    ``None`` at 1.0: no code path populates them, and the FROZEN parser /
    ``to_yaml`` writer neither read nor emit them yet (a 1.0 ``mesh.yml``
    cannot contain them — the parser rejects unknown top-level keys). When
    1.1 *activates* a field it lands the parser read + writer emit +
    round-trip test together, per the contract above; until then the
    fields exist only on the dataclass so the names are frozen and the
    1.1 parser widening is non-breaking. Declared as ``tuple[str, ...]``
    (not ``list``) to keep the frozen dataclass hashable, matching
    ``nodes``.

    - ``cluster_keys`` — accepted-on-read rotation list; ``cluster_key``
      stays the single active write key when this is populated.
    - ``cluster_trust_anchors`` — trusted public keys for asymmetric
      mesh auth.
    - ``cluster_auth_mode`` — ``"bearer"`` (default / ``None``),
      ``"asymmetric"``, or ``"mtls"``.
    """

    cluster_key: str
    self_name: str
    nodes: tuple[MeshNode, ...]
    protocol_version: int = 1
    # CC13 reserved-null auth-scheme siblings — see class docstring.
    cluster_keys: tuple[str, ...] | None = None
    cluster_trust_anchors: tuple[str, ...] | None = None
    cluster_auth_mode: str | None = None

    def __post_init__(self) -> None:
        # E3 fold: validate yaml-safe characters on the top-level
        # scalars too. The dataclass is frozen so this only runs at
        # construction; subsequent attribute access is read-only.
        _validate_yaml_safe("MeshConfig.cluster_key", self.cluster_key)
        _validate_yaml_safe("MeshConfig.self_name", self.self_name)
        # I9 fold (C3.3 pre-merge review, Blast Radius): empty string
        # cluster_key would send ``Authorization: Bearer `` (trailing
        # space, empty token) to any admin endpoint → cryptic 401
        # from the leader. The parser rejects a missing key at read
        # time but ``_validate_yaml_safe`` only checks forbidden
        # characters, not empty. Hoist the check here so
        # ``synthesize_from_peer_config`` and direct-construction
        # paths (tests, future verbs) can't produce a construct that
        # silently produces a mis-auth request. Same pattern as E7
        # (empty nodes) and C3.2 A1 (self_name in nodes).
        if not self.cluster_key:
            raise ValueError(
                "MeshConfig.cluster_key must be non-empty — an empty "
                "cluster key would produce an empty Bearer token on "
                "every admin request, yielding cryptic 401s."
            )
        # E7 fold (C3.1 pre-merge review): empty nodes tuple round-
        # trips to parser-rejecting output. parse_mesh_config requires
        # at least one node; the writer must not produce something
        # that fails to parse. Reject at construction time.
        if not self.nodes:
            raise ValueError(
                "MeshConfig.nodes must contain at least one MeshNode "
                "(parser requires the same; round-trip would fail otherwise)"
            )
        # A1 fold (C3.2 pre-merge review, cross-confirmed E9 + A1):
        # parse_mesh_config validates `self_name in nodes` but the
        # constructor did not. C3.2's add-node / remove-node verbs
        # construct MeshConfig directly, bypassing the parser-side
        # check. Today it's unreachable (remove-node refuses self,
        # add-node preserves existing self_name), but the gap will
        # bite the first C3.3 verb that touches self_name (rename-node,
        # set-self, cluster-key rotation that changes node identity).
        # Hoist the check now so write-side and read-side parity holds.
        # Same E7 pattern from C3.1 applied to a different invariant.
        node_names = {n.name for n in self.nodes}
        if self.self_name not in node_names:
            raise ValueError(
                f"MeshConfig.self_name={self.self_name!r} does not match any node "
                f"in nodes (known: {sorted(node_names)}). The parser requires "
                f"self to match a node name; constructing a config that violates "
                f"this would produce a mesh.yml that fails to parse on next read."
            )

    def get_node(self, name: str) -> MeshNode | None:
        for n in self.nodes:
            if n.name == name:
                return n
        return None

    def self_node(self) -> MeshNode | None:
        return self.get_node(self.self_name)

    def to_yaml(self) -> str:
        """Serialize to the FROZEN parser dialect.

        Plan 4 C3.1: round-trips with :func:`parse_mesh_config`.
        Mirrors :meth:`maxim.peer.config.PeerConfig.to_yaml` — same
        sister-module pattern, same "no PyYAML dep" rationale, same
        sorted-and-stable field order.

        The output is intentionally minimal: no comments, no inline
        annotations, no fancy formatting. Operators who hand-edit
        ``mesh.yml`` and want comments should know that the
        ``init-mesh`` verb (and any future C3 verb that rewrites the
        file) will strip them — comments survive only as long as no
        CLI verb touches the file. The FROZEN parser invariant
        (no quoted strings, no anchors, no tab indent) constrains
        the writer to a similarly minimal dialect.

        Field order: ``cluster_key`` → ``self`` → ``protocol_version``
        → ``nodes:`` (list, in declaration order). The order matters
        because a future operator-readable diff between two
        ``mesh.yml`` files should be stable across rewrites.

        **A3 fold (C3.1 pre-merge review):** node serialization
        delegates to :meth:`MeshNode.to_yaml_lines` so that any
        future ``MeshNode`` field addition forces the writer to
        include it (rather than this method silently reaching into
        only the fields it knows about).
        """
        lines = [
            f"cluster_key: {self.cluster_key}",
            f"self: {self.self_name}",
            f"protocol_version: {self.protocol_version}",
            "nodes:",
        ]
        for node in self.nodes:
            lines.extend(node.to_yaml_lines())
        return "\n".join(lines) + "\n"


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


def write_mesh_config(cfg: MeshConfig, path: Path | None = None) -> Path:
    """Persist ``cfg`` to disk via the FROZEN parser dialect.

    Plan 4 C3.1. Mirrors :func:`maxim.peer.config.write_peer_config`
    in shape, but routes the write through
    :func:`maxim.utils.atomic_io.atomic_write_secret` because
    ``mesh.yml::cluster_key`` is a secret per the C2 invariant
    ("``mesh.yml`` is declarative; credential-bearing files use
    ``atomic_write_secret``").

    On first write (file did not exist), ``atomic_write_secret`` has
    no pre-existing mode to preserve, so this function explicitly
    chmods to ``0o600`` afterwards. On rewrites, the mode bits the
    operator set are preserved and the explicit chmod is a no-op.
    POSIX-only chmod; Windows skips the permission tighten the same
    way ``write_peer_config`` does.

    **Concurrent reader safety (blast A3 fold):** ``atomic_write_secret``
    uses ``os.replace`` under the hood, which is an atomic rename on
    POSIX — concurrent readers (``maxim peer list-nodes`` in another
    shell, or doctor running concurrently) will observe either the
    old file or the new file, never a half-written file. No reader-
    side lock is needed.

    **Sanctioned caller (architecture A1 fold from C3.1 + C3.2 reframe):**
    the only sanctioned callers of this function live in
    :mod:`maxim.peer.mesh_setup` — the operator-invoked one-shot
    setup verbs (``init-mesh``, ``add-node``, ``remove-node`` as of
    Plan 4 C3.2). The C2 invariant ("mesh.yml is declarative;
    runtime code paths are read-only") permits operator-explicit
    edits but forbids automatic runtime / admin-API writes to
    ``mesh.yml``. A CI grep enforces the allow-list in
    ``.github/workflows/test.yml``: any new caller of
    ``write_mesh_config`` outside ``mesh_setup.py`` will fail CI
    with a hint pointing at this docstring. If you have a legitimate
    new caller (e.g. a future operator-invoked verb that grows
    ``mesh_setup.py``), update both the allow-list AND the C2
    CLAUDE.md invariant lesson in the same commit so the rule
    remains coherent. **First ask whether the new write belongs in
    `~/.maxim/util/` instead** — automatic mutations always do.

    **E4 fold (C3.1 pre-merge review):** chmod failure on first
    write is no longer silently swallowed — it surfaces via
    ``logger.warning`` so the failure shows up in ``MAXIM_LOG_FILE``
    and ``maxim doctor``. We still don't fail-hard because some
    filesystems (WSL1, certain network mounts) don't support chmod
    and the alternative is refusing to ship the config at all.
    """
    p = path or mesh_config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    is_new = not p.is_file()
    atomic_write_secret(str(p), cfg.to_yaml())
    if is_new and platform.system() != "Windows":
        try:
            os.chmod(p, 0o600)
        except OSError as e:
            # E4 fold: surface this via the structured logger so it
            # shows up in MAXIM_LOG_FILE / doctor. Brand-new file
            # would otherwise land at umask (typically 0o644) with
            # the cluster_key world-readable.
            logger.warning(
                "write_mesh_config: chmod 0o600 on first write failed for %s: %s. "
                "Cluster key may be readable to other users on this system. "
                "Run `chmod 0600 %s` manually if this is a concern.",
                p,
                e,
                p,
            )
    return p


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
    "write_mesh_config",
]
