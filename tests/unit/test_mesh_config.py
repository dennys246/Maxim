"""Tests for maxim.peer.mesh_config (Plan 4 Stage C1)."""

from __future__ import annotations

import sys

import pytest

from maxim.peer.mesh_config import (
    MeshConfig,
    MeshConfigError,
    MeshNode,
    parse_mesh_config,
    read_mesh_config,
    read_or_synthesize_mesh_config,
    synthesize_from_peer_config,
    write_mesh_config,
)
from maxim.peer.probe_classify import ProbeClassification, classify_probe_outcome


VALID_YAML = """\
cluster_key: sk-cluster-abc
self: leader-desk
protocol_version: 1
nodes:
  - name: leader-desk
    url: http://192.168.1.10:8099/v1
    role: leader
  - name: mac-studio
    url: https://mac.example.com/v1
    role: peer
"""


class TestParseMeshConfig:
    def test_round_trip_happy_path(self) -> None:
        cfg = parse_mesh_config(VALID_YAML)
        assert cfg.cluster_key == "sk-cluster-abc"
        assert cfg.self_name == "leader-desk"
        assert cfg.protocol_version == 1
        assert len(cfg.nodes) == 2
        assert cfg.nodes[0] == MeshNode(
            name="leader-desk",
            url="http://192.168.1.10:8099/v1",
            role="leader",
        )
        assert cfg.nodes[1].name == "mac-studio"
        assert cfg.nodes[1].role == "peer"

    def test_self_must_match_a_node(self) -> None:
        yaml = VALID_YAML.replace("self: leader-desk", "self: ghost")
        with pytest.raises(MeshConfigError, match="'self: ghost' does not match"):
            parse_mesh_config(yaml)

    def test_missing_self_rejected(self) -> None:
        yaml = "\n".join(line for line in VALID_YAML.splitlines() if not line.startswith("self:"))
        with pytest.raises(MeshConfigError, match="missing required field 'self'"):
            parse_mesh_config(yaml)

    def test_missing_cluster_key_rejected(self) -> None:
        yaml = VALID_YAML.replace("cluster_key: sk-cluster-abc\n", "")
        with pytest.raises(MeshConfigError, match="missing required field 'cluster_key'"):
            parse_mesh_config(yaml)

    def test_missing_nodes_rejected(self) -> None:
        yaml = "cluster_key: sk-abc\nself: leader-desk\n"
        with pytest.raises(MeshConfigError, match="missing required field 'nodes'"):
            parse_mesh_config(yaml)

    def test_unknown_role_rejected_with_line_number(self) -> None:
        yaml = VALID_YAML.replace("role: peer", "role: overlord")
        with pytest.raises(MeshConfigError, match=r"mesh\.yml line \d+:.*invalid role 'overlord'") as exc:
            parse_mesh_config(yaml)
        assert exc.value.line is not None

    def test_malformed_url_scheme_rejected_with_line_number(self) -> None:
        yaml = VALID_YAML.replace("http://192.168.1.10:8099/v1", "ftp://bad/v1")
        with pytest.raises(MeshConfigError, match="must use http:// or https://") as exc:
            parse_mesh_config(yaml)
        assert exc.value.line is not None

    def test_url_without_hostname_rejected(self) -> None:
        yaml = VALID_YAML.replace("http://192.168.1.10:8099/v1", "http:///v1")
        with pytest.raises(MeshConfigError, match="has no hostname"):
            parse_mesh_config(yaml)

    def test_unknown_top_level_key_rejected(self) -> None:
        yaml = VALID_YAML + "clutser_key: typo\n"  # intentional typo
        with pytest.raises(MeshConfigError, match="unknown top-level key 'clutser_key'"):
            parse_mesh_config(yaml)

    def test_unsupported_protocol_version_rejected(self) -> None:
        yaml = VALID_YAML.replace("protocol_version: 1", "protocol_version: 99")
        with pytest.raises(MeshConfigError, match="unsupported protocol_version 99"):
            parse_mesh_config(yaml)

    def test_non_integer_protocol_version_rejected(self) -> None:
        yaml = VALID_YAML.replace("protocol_version: 1", "protocol_version: latest")
        with pytest.raises(MeshConfigError, match="protocol_version must be an integer"):
            parse_mesh_config(yaml)

    def test_comments_and_blank_lines_tolerated(self) -> None:
        yaml = "# header\n\n" + VALID_YAML + "\n# trailing comment\n"
        cfg = parse_mesh_config(yaml)
        assert len(cfg.nodes) == 2

    def test_comment_between_nodes_entries(self) -> None:
        """Pre-merge review F18: comments inside nodes: block."""
        yaml = """\
cluster_key: sk-x
self: a
nodes:
  - name: a
    url: http://a/v1
    role: leader
  # interstitial comment
  - name: b
    url: http://b/v1
    role: peer
"""
        cfg = parse_mesh_config(yaml)
        assert len(cfg.nodes) == 2
        assert cfg.nodes[0].name == "a"
        assert cfg.nodes[1].name == "b"

    def test_drain_field_is_rejected_as_unknown_key(self) -> None:
        """C1 intentionally does not support ``drain:``. Pre-merge review
        flagged the two-layer design (config + runtime state) as
        under-specified; drain/resume defer to C2 with a proper design pass.
        """
        yaml = VALID_YAML + "drain:\n  - mac-studio\n"
        with pytest.raises(MeshConfigError, match="unknown top-level key 'drain'"):
            parse_mesh_config(yaml)


class TestParserHardening:
    """Parser hardening added in response to pre-merge review findings."""

    def test_tab_indentation_rejected(self) -> None:
        """F1: tab-indented input silently mis-parses."""
        yaml = "cluster_key: sk-x\nself: a\nnodes:\n\t- name: a\n\t  url: http://a/v1\n\t  role: leader\n"
        with pytest.raises(MeshConfigError, match="tab characters are not allowed"):
            parse_mesh_config(yaml)

    def test_inline_comment_on_value_stripped(self) -> None:
        """F2: ``name: foo  # note`` must not make a node named ``foo  # note``."""
        yaml = """\
cluster_key: sk-x
self: leader-desk  # primary node
nodes:
  - name: leader-desk  # not part of the name
    url: http://a/v1  # LAN
    role: leader
"""
        cfg = parse_mesh_config(yaml)
        assert cfg.self_name == "leader-desk"
        assert cfg.nodes[0].name == "leader-desk"
        assert cfg.nodes[0].url == "http://a/v1"

    def test_bare_dash_entry_rejected(self) -> None:
        """F3: dangling ``- `` silently corrupts the next node."""
        yaml = """\
cluster_key: sk-x
self: a
nodes:
  -
  - name: a
    url: http://a/v1
    role: leader
"""
        with pytest.raises(MeshConfigError, match="empty list entry"):
            parse_mesh_config(yaml)

    def test_duplicate_node_name_rejected(self) -> None:
        """F5: duplicate node names must be loud."""
        yaml = """\
cluster_key: sk-x
self: a
nodes:
  - name: a
    url: http://a/v1
    role: leader
  - name: a
    url: http://b/v1
    role: peer
"""
        with pytest.raises(MeshConfigError, match="duplicate node name 'a'"):
            parse_mesh_config(yaml)

    def test_cluster_key_with_hash_is_preserved(self) -> None:
        """Round 2 E1: ``cluster_key: sk-abc#literal`` must NOT silently
        truncate to ``sk-abc``. The naive comment stripper would have
        produced ``auth_rejected`` at probe time with no visible root
        cause — a silent-corruption bug.
        """
        yaml = """\
cluster_key: sk-cluster#notes-are-part-of-key
self: a
nodes:
  - name: a
    url: http://a/v1
    role: leader
"""
        cfg = parse_mesh_config(yaml)
        assert cfg.cluster_key == "sk-cluster#notes-are-part-of-key"

    def test_url_with_hash_preserved_if_no_whitespace(self) -> None:
        """Same E1 concern for URLs with fragments — though URLs typically
        don't have fragments in the OpenAI-compatible base path, the
        parser must not corrupt them silently.
        """
        yaml = """\
cluster_key: sk-x
self: a
nodes:
  - name: a
    url: http://a/v1#fragment
    role: leader
"""
        cfg = parse_mesh_config(yaml)
        assert cfg.nodes[0].url == "http://a/v1#fragment"

    def test_inline_comment_still_stripped_when_preceded_by_space(self) -> None:
        """E1 regression guard: the space-before-# rule must still strip
        legitimate trailing comments.
        """
        yaml = """\
cluster_key: sk-abc  # this is a trailing comment
self: a
nodes:
  - name: a
    url: http://a/v1  # the leader
    role: leader
"""
        cfg = parse_mesh_config(yaml)
        assert cfg.cluster_key == "sk-abc"
        assert cfg.nodes[0].url == "http://a/v1"


class TestReadMeshConfig:
    def test_file_absent_returns_none(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        assert read_mesh_config() is None

    def test_reads_file_on_disk(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        mesh_path = tmp_path / "maxim" / "mesh.yml"
        mesh_path.parent.mkdir(parents=True)
        mesh_path.write_text(VALID_YAML)
        cfg = read_mesh_config()
        assert cfg is not None
        assert cfg.self_name == "leader-desk"

    def test_malformed_file_raises(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        mesh_path = tmp_path / "maxim" / "mesh.yml"
        mesh_path.parent.mkdir(parents=True)
        mesh_path.write_text(
            "cluster_key: sk-x\nself: nope\nnodes:\n  - name: a\n    url: http://a/v1\n    role: leader\n"
        )
        with pytest.raises(MeshConfigError, match="does not match"):
            read_mesh_config()


class TestSynthesizeFromPeerConfig:
    def test_returns_none_when_no_peer_config(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        assert synthesize_from_peer_config() is None

    def test_builds_one_node_mesh_from_peer_yml(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        peer_path = tmp_path / "maxim" / "peer.yml"
        peer_path.parent.mkdir(parents=True)
        peer_path.write_text("url: https://leader.example.com/v1\napi_key: sk-peer-abc\n")
        cfg = synthesize_from_peer_config()
        assert cfg is not None
        assert cfg.cluster_key == "sk-peer-abc"
        assert cfg.self_name == "leader"
        assert len(cfg.nodes) == 1
        assert cfg.nodes[0].url == "https://leader.example.com/v1"
        assert cfg.nodes[0].role == "leader"

    def test_read_or_synthesize_prefers_mesh_yml(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        (tmp_path / "maxim").mkdir(parents=True)
        (tmp_path / "maxim" / "peer.yml").write_text("url: https://old.example.com/v1\napi_key: sk-old\n")
        (tmp_path / "maxim" / "mesh.yml").write_text(VALID_YAML)
        cfg = read_or_synthesize_mesh_config()
        assert cfg is not None
        assert cfg.cluster_key == "sk-cluster-abc"  # from mesh.yml, not peer.yml

    def test_read_or_synthesize_falls_back_when_no_mesh(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        (tmp_path / "maxim").mkdir(parents=True)
        (tmp_path / "maxim" / "peer.yml").write_text("url: https://leader.example.com/v1\napi_key: sk-fallback\n")
        cfg = read_or_synthesize_mesh_config()
        assert cfg is not None
        assert cfg.cluster_key == "sk-fallback"


class TestMeshConfigToYaml:
    """Plan 4 C3.1: round-trip serializer for the FROZEN dialect.

    The serializer must produce output that ``parse_mesh_config``
    can read back into an equal :class:`MeshConfig`. This is the
    load-bearing invariant for ``init-mesh`` (which writes via
    ``write_mesh_config`` → ``to_yaml``).
    """

    def test_round_trip_one_node(self):
        cfg = MeshConfig(
            cluster_key="sk-test",
            self_name="leader",
            nodes=(MeshNode(name="leader", url="https://x.example/v1", role="leader"),),
            protocol_version=1,
        )
        parsed = parse_mesh_config(cfg.to_yaml())
        assert parsed == cfg

    def test_round_trip_multi_node(self):
        cfg = MeshConfig(
            cluster_key="sk-cluster-abc",
            self_name="leader-desk",
            nodes=(
                MeshNode(name="leader-desk", url="http://192.168.1.10:8099/v1", role="leader"),
                MeshNode(name="mac-studio", url="https://mac.example.com/v1", role="peer"),
                MeshNode(name="tablet", url="http://10.0.0.5/v1", role="peer"),
            ),
            protocol_version=1,
        )
        parsed = parse_mesh_config(cfg.to_yaml())
        assert parsed == cfg
        # Node order MUST be preserved across round-trip
        assert [n.name for n in parsed.nodes] == [n.name for n in cfg.nodes]

    def test_serialized_format_is_stable(self):
        """Two serializations of the same config produce byte-equal
        output. Stable diffs across rewrites are operator-friendly."""
        cfg = MeshConfig(
            cluster_key="sk-test",
            self_name="leader",
            nodes=(MeshNode(name="leader", url="http://a/v1", role="leader"),),
        )
        assert cfg.to_yaml() == cfg.to_yaml()

    def test_serialized_format_starts_with_cluster_key(self):
        """Field order is documented as cluster_key → self → protocol_version → nodes."""
        cfg = MeshConfig(
            cluster_key="sk-test",
            self_name="a",
            nodes=(MeshNode(name="a", url="http://a/v1", role="leader"),),
        )
        lines = cfg.to_yaml().splitlines()
        assert lines[0] == "cluster_key: sk-test"
        assert lines[1] == "self: a"
        assert lines[2] == "protocol_version: 1"
        assert lines[3] == "nodes:"

    def test_no_pyyaml_dep_in_output(self):
        """The dialect is the FROZEN parser dialect — no quoted
        strings, no anchors. The C1 invariant survives the writer."""
        cfg = MeshConfig(
            cluster_key="sk",
            self_name="a",
            nodes=(MeshNode(name="a", url="http://a/v1", role="leader"),),
        )
        yaml = cfg.to_yaml()
        assert '"' not in yaml
        assert "'" not in yaml
        assert "&" not in yaml  # no YAML anchors
        assert "*" not in yaml  # no YAML aliases

    def test_round_trip_preserves_protocol_version_default(self):
        cfg = MeshConfig(
            cluster_key="sk",
            self_name="a",
            nodes=(MeshNode(name="a", url="http://a/v1", role="leader"),),
            # protocol_version uses dataclass default = 1
        )
        parsed = parse_mesh_config(cfg.to_yaml())
        assert parsed.protocol_version == 1


class TestClusterAuthFormatFreeze:
    """CC13 auth format-freeze: the three reserved-null ``cluster_*`` sibling
    fields exist on the dataclass but are NOT serialized or parsed at 1.0.
    These tests pin the freeze so 1.1's activation of any field is a
    deliberate, reviewed change (parser + writer + round-trip land together).
    """

    def _one_node_cfg(self, **overrides):
        base = dict(
            cluster_key="sk-test",
            self_name="leader",
            nodes=(MeshNode(name="leader", url="https://x.example/v1", role="leader"),),
        )
        base.update(overrides)
        return MeshConfig(**base)

    def test_reserved_fields_default_to_none(self):
        cfg = self._one_node_cfg()
        assert cfg.cluster_keys is None
        assert cfg.cluster_trust_anchors is None
        assert cfg.cluster_auth_mode is None

    def test_default_config_round_trips_with_reserved_fields_none(self):
        """A config with the reserved fields at their None default round-trips
        cleanly — the reserved slots do not perturb the wire format."""
        cfg = self._one_node_cfg()
        parsed = parse_mesh_config(cfg.to_yaml())
        assert parsed == cfg
        assert parsed.cluster_keys is None
        assert parsed.cluster_trust_anchors is None
        assert parsed.cluster_auth_mode is None

    def test_to_yaml_does_not_emit_reserved_fields_at_1_0(self):
        """The writer is byte-stable: even when the reserved fields are
        populated (which no 1.0 code path does), ``to_yaml`` does not emit
        them — activation is 1.1 work that lands the writer change with it."""
        cfg = self._one_node_cfg(
            cluster_keys=("sk-old", "sk-new"),
            cluster_trust_anchors=("ed25519:AAAA",),
            cluster_auth_mode="asymmetric",
        )
        yaml = cfg.to_yaml()
        assert "cluster_keys" not in yaml
        assert "cluster_trust_anchors" not in yaml
        assert "cluster_auth_mode" not in yaml

    @pytest.mark.parametrize(
        "reserved_block",
        [
            "cluster_keys:\n  - sk-old\n  - sk-new\n",
            "cluster_trust_anchors:\n  - ed25519:AAAA\n",
            "cluster_auth_mode: asymmetric\n",
        ],
    )
    def test_parser_rejects_reserved_keys_at_1_0(self, reserved_block):
        """The FROZEN parser still rejects the reserved keys as unknown
        top-level keys at 1.0. 1.1 teaching the parser to read them is a
        non-breaking widening — but it must be a deliberate change, not a
        silent acceptance. Mirrors ``test_drain_field_is_rejected_as_unknown_key``.
        """
        yaml = VALID_YAML + reserved_block
        with pytest.raises(MeshConfigError, match="unknown top-level key"):
            parse_mesh_config(yaml)

    def test_reserved_fields_are_hashable(self):
        """The reserved fields use ``tuple`` (not ``list``) so the frozen
        dataclass stays hashable per the CC3 forward-compat audit."""
        cfg = self._one_node_cfg(cluster_keys=("sk-a", "sk-b"))
        # Must not raise — list fields would make this a TypeError.
        assert hash(cfg) == hash(cfg)


class TestWriteMeshConfig:
    """Disk-I/O wrapper around to_yaml(). Must use atomic_write_secret
    because mesh.yml::cluster_key is a secret per the C2 invariant.
    """

    def test_write_then_read_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        cfg = MeshConfig(
            cluster_key="sk-test",
            self_name="leader",
            nodes=(MeshNode(name="leader", url="https://x/v1", role="leader"),),
        )
        path = write_mesh_config(cfg)
        assert path.is_file()
        loaded = read_mesh_config(path)
        assert loaded == cfg

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only chmod")
    def test_first_write_chmods_to_0600(self, tmp_path, monkeypatch):
        import os

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        cfg = MeshConfig(
            cluster_key="sk-secret",
            self_name="leader",
            nodes=(MeshNode(name="leader", url="https://x/v1", role="leader"),),
        )
        path = write_mesh_config(cfg)
        mode = os.stat(path).st_mode & 0o777
        assert mode == 0o600

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only chmod")
    def test_rewrite_preserves_existing_mode(self, tmp_path, monkeypatch):
        """Operator who chmods their mesh.yml to a custom mode (say
        0o640 for shared-group access) shouldn't see it widened on
        the next write."""
        import os

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        cfg = MeshConfig(
            cluster_key="sk",
            self_name="leader",
            nodes=(MeshNode(name="leader", url="https://x/v1", role="leader"),),
        )
        path = write_mesh_config(cfg)
        os.chmod(path, 0o640)

        cfg2 = MeshConfig(
            cluster_key="sk-rotated",
            self_name="leader",
            nodes=(MeshNode(name="leader", url="https://x/v1", role="leader"),),
        )
        write_mesh_config(cfg2)
        mode = os.stat(path).st_mode & 0o777
        assert mode == 0o640


class TestYamlSafeValidation:
    """E3 fold (C3.1 pre-merge review): MeshNode and MeshConfig
    reject parser-incompatible characters at construction time so a
    future code path can't silently corrupt mesh.yml via round-trip.
    """

    def test_node_name_with_newline_rejected(self):
        with pytest.raises(ValueError, match="newline"):
            MeshNode(name="leader\nrogue", url="http://a/v1", role="leader")

    def test_node_url_with_carriage_return_rejected(self):
        with pytest.raises(ValueError, match="newline"):
            MeshNode(name="a", url="http://a/v1\r\n", role="leader")

    def test_node_url_with_inline_comment_rejected(self):
        """`http://a/v1 # comment` would round-trip to `http://a/v1`
        because the parser strips inline comments. Lossy."""
        with pytest.raises(ValueError, match="inline comment"):
            MeshNode(name="a", url="http://a/v1 # comment", role="leader")

    def test_cluster_key_with_newline_rejected(self):
        with pytest.raises(ValueError, match="newline"):
            MeshConfig(
                cluster_key="sk\nrogue",
                self_name="a",
                nodes=(MeshNode(name="a", url="http://a/v1", role="leader"),),
            )

    def test_cluster_key_with_inline_comment_rejected(self):
        """`sk-cluster # not-secret` would round-trip lossy."""
        with pytest.raises(ValueError, match="inline comment"):
            MeshConfig(
                cluster_key="sk-cluster # not-secret",
                self_name="a",
                nodes=(MeshNode(name="a", url="http://a/v1", role="leader"),),
            )

    def test_cluster_key_with_hash_no_whitespace_allowed(self):
        """C1 E1 fold: the parser preserves `#` when there's no
        preceding whitespace (e.g., `sk-abc#literal`). The writer
        validation must mirror this — bare `#` is fine, only
        whitespace-then-`#` is rejected."""
        cfg = MeshConfig(
            cluster_key="sk-cluster#tag",
            self_name="a",
            nodes=(MeshNode(name="a", url="http://a/v1", role="leader"),),
        )
        # Round trip through to_yaml + parse — must preserve the literal
        round_tripped = parse_mesh_config(cfg.to_yaml())
        assert round_tripped.cluster_key == "sk-cluster#tag"

    def test_self_name_with_newline_rejected(self):
        with pytest.raises(ValueError, match="newline"):
            MeshConfig(
                cluster_key="sk",
                self_name="a\nb",
                nodes=(MeshNode(name="a", url="http://a/v1", role="leader"),),
            )


class TestEmptyNodesRejected:
    """E7 fold (C3.1 pre-merge review): empty nodes tuple round-trips
    to parser-rejecting output. Reject at construction time so the
    invariant holds throughout the dataclass lifecycle.
    """

    def test_empty_nodes_tuple_rejected(self):
        with pytest.raises(ValueError, match="at least one MeshNode"):
            MeshConfig(cluster_key="sk", self_name="a", nodes=())

    def test_one_node_minimum_accepted(self):
        # Sanity check: the minimum valid mesh has exactly one node
        cfg = MeshConfig(
            cluster_key="sk",
            self_name="a",
            nodes=(MeshNode(name="a", url="http://a/v1", role="leader"),),
        )
        assert len(cfg.nodes) == 1


class TestSelfNameMustMatchNode:
    """A1 fold (C3.2 pre-merge review, cross-confirmed E9 + A1):
    parse_mesh_config validates self_name in nodes but the
    constructor previously did not. Hoist the check so write-side
    and read-side parity holds. Same E7 pattern from C3.1 applied
    to a different invariant.
    """

    def test_self_name_not_in_nodes_rejected(self):
        with pytest.raises(ValueError, match="does not match any node"):
            MeshConfig(
                cluster_key="sk",
                self_name="ghost",
                nodes=(MeshNode(name="real", url="http://r/v1", role="leader"),),
            )

    def test_self_name_matches_one_node_accepted(self):
        cfg = MeshConfig(
            cluster_key="sk",
            self_name="real",
            nodes=(MeshNode(name="real", url="http://r/v1", role="leader"),),
        )
        assert cfg.self_name == "real"

    def test_self_name_matches_in_multi_node_mesh(self):
        cfg = MeshConfig(
            cluster_key="sk",
            self_name="leader-desk",
            nodes=(
                MeshNode(name="leader-desk", url="http://a/v1", role="leader"),
                MeshNode(name="mac-studio", url="http://b/v1", role="peer"),
            ),
        )
        assert cfg.self_name == "leader-desk"

    def test_error_lists_known_node_names(self):
        with pytest.raises(ValueError, match="known: \\['mac-studio', 'tablet'\\]"):
            MeshConfig(
                cluster_key="sk",
                self_name="ghost",
                nodes=(
                    MeshNode(name="mac-studio", url="http://a/v1", role="leader"),
                    MeshNode(name="tablet", url="http://b/v1", role="peer"),
                ),
            )


class TestMeshNodeToYamlLines:
    """A3 fold (C3.1 pre-merge review): node serialization is owned
    by MeshNode so future field additions force the writer to include
    them. This pre-empts the C3 break when MeshNode gains a
    `public_key` field for cluster key rotation.
    """

    def test_node_to_yaml_lines_shape(self):
        node = MeshNode(name="leader", url="https://x/v1", role="leader")
        lines = node.to_yaml_lines()
        assert lines == [
            "  - name: leader",
            "    url: https://x/v1",
            "    role: leader",
        ]

    def test_mesh_to_yaml_uses_node_to_yaml_lines(self):
        """Regression guard: if a future refactor field-by-field
        reach-in code returns to MeshConfig.to_yaml, this test should
        catch it. We assert that the output for a 2-node mesh has
        exactly 6 node-related lines (2 nodes × 3 lines each)."""
        cfg = MeshConfig(
            cluster_key="sk",
            self_name="a",
            nodes=(
                MeshNode(name="a", url="http://a/v1", role="leader"),
                MeshNode(name="b", url="http://b/v1", role="peer"),
            ),
        )
        lines = cfg.to_yaml().splitlines()
        # 4 top-level lines (cluster_key, self, protocol_version, nodes:)
        # + 6 node lines (2 nodes × 3 lines) = 10 total
        assert len(lines) == 10


class TestWritePeerConfigUsesAtomicWriteSecret:
    """A4 fold (C3.1 pre-merge review): write_peer_config now routes
    through atomic_write_secret like write_mesh_config. Validates the
    C2 invariant in production code at both credential-bearing
    writers, eliminating the latent inconsistency between them.
    """

    def test_write_then_read_round_trip(self, tmp_path, monkeypatch):
        from maxim.peer.config import PeerConfig, read_peer_config, write_peer_config

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        cfg = PeerConfig(url="https://x/v1", api_key="sk-test")
        path = write_peer_config(cfg)
        assert path.is_file()
        loaded = read_peer_config()
        assert loaded == cfg

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only chmod")
    def test_first_write_is_0600(self, tmp_path, monkeypatch):
        import os

        from maxim.peer.config import PeerConfig, write_peer_config

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        cfg = PeerConfig(url="https://x/v1", api_key="sk-secret")
        path = write_peer_config(cfg)
        mode = os.stat(path).st_mode & 0o777
        assert mode == 0o600

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only chmod")
    def test_rewrite_preserves_existing_mode(self, tmp_path, monkeypatch):
        """A4 fold validates that the C2 preserve_mode behavior holds
        for peer.yml as well: an operator who chmods peer.yml to a
        custom mode (e.g., 0o640 for shared-group) shouldn't see it
        widened on the next write."""
        import os

        from maxim.peer.config import PeerConfig, write_peer_config

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        cfg = PeerConfig(url="https://x/v1", api_key="sk-1")
        path = write_peer_config(cfg)
        os.chmod(path, 0o640)

        cfg2 = PeerConfig(url="https://x/v1", api_key="sk-2")
        write_peer_config(cfg2)
        mode = os.stat(path).st_mode & 0o777
        assert mode == 0o640


class TestWriteMeshConfigChmodWarning:
    """E4 fold (C3.1 pre-merge review): chmod failure on first write
    is no longer silently swallowed — it surfaces via logger.warning
    so the failure shows up in MAXIM_LOG_FILE / doctor logs.
    """

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX chmod path only")
    def test_chmod_failure_logged(self, tmp_path, monkeypatch, caplog):
        import logging
        import os

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        cfg = MeshConfig(
            cluster_key="sk-secret",
            self_name="a",
            nodes=(MeshNode(name="a", url="http://a/v1", role="leader"),),
        )

        # Monkey-patch os.chmod to fail
        real_chmod = os.chmod

        def _failing_chmod(path, mode):
            if str(path).endswith("mesh.yml"):
                raise OSError("simulated chmod failure")
            real_chmod(path, mode)

        monkeypatch.setattr(os, "chmod", _failing_chmod)

        with caplog.at_level(logging.WARNING, logger="maxim.peer.mesh_config"):
            write_mesh_config(cfg)

        # File still got written (atomic_write_secret succeeded)
        assert (tmp_path / "maxim" / "mesh.yml").is_file()
        # Warning surfaced
        assert any("chmod 0o600 on first write failed" in record.message for record in caplog.records)


class TestClassifyProbeOutcome:
    """Shared classifier — single source of truth for probe outcome mapping.

    Specific-before-general ordering is load-bearing (Plan 2 R2c).
    """

    def test_returns_frozen_dataclass(self) -> None:
        """Round 2 A3R2: structured result, not a tuple — fields stay stable
        across C2/C3 additions.
        """
        result = classify_probe_outcome("ok", "HTTP 200", 5.0, "http://a/v1")
        assert isinstance(result, ProbeClassification)
        with pytest.raises(Exception):
            result.status = "fail"  # type: ignore[misc]

    def test_ok_returns_reachable_with_latency(self) -> None:
        r = classify_probe_outcome("ok", "HTTP 200", 5.0, "http://a/v1")
        assert r.status == "ok"
        assert "reachable" in r.message
        assert "5ms" in r.message
        assert r.fix is None

    def test_ok_detail_flows_through_message(self) -> None:
        """Round 2 A1R2: callers pass richer detail through, no post-hoc
        override. Doctor passes ``f"{role}, {url}"`` as detail; mesh_cli
        passes the raw probe detail.
        """
        r = classify_probe_outcome("ok", "leader, http://192.168.1.10/v1", 42.0, "http://192.168.1.10/v1")
        assert r.status == "ok"
        assert "leader, http://192.168.1.10/v1" in r.message
        assert "42ms" in r.message

    def test_auth_rejected_is_fail_with_key_rotate_hint(self) -> None:
        r = classify_probe_outcome("auth_rejected", "HTTP 401", 12.0, "http://a/v1")
        assert r.status == "fail"
        assert "auth rejected" in r.message
        assert r.fix is not None
        assert "tunnel key rotate" in r.fix

    def test_inference_broken_is_fail_with_chat_hint(self) -> None:
        r = classify_probe_outcome("inference_broken", "stage2: HTTP 500", 800.0, "http://a/v1")
        assert r.status == "fail"
        assert "chat endpoint broken" in r.message
        assert r.fix is not None
        assert "maxim peer llm --status" in r.fix

    def test_network_outcomes_warn(self) -> None:
        for outcome in ("timeout", "connection_refused", "dns_fail", "tls_error", "http_5xx", "other"):
            r = classify_probe_outcome(outcome, "detail", None, "http://a/v1")
            assert r.status == "warn", f"{outcome} should be warn"
            assert "curl -v" in (r.fix or "")

    def test_unknown_outcome_falls_through_to_warn(self) -> None:
        r = classify_probe_outcome("plasma_storm", "?", None, "http://a/v1")
        assert r.status == "warn"
        assert "unknown outcome" in r.message

    def test_no_latency_omits_bracket(self) -> None:
        r = classify_probe_outcome("ok", "HTTP 200", None, "http://a/v1")
        assert "ms]" not in r.message
