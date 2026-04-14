"""Tests for maxim.peer.mesh_config (Plan 4 Stage C1)."""

from __future__ import annotations

import pytest

from maxim.peer.mesh_config import (
    MeshConfigError,
    MeshNode,
    parse_mesh_config,
    read_mesh_config,
    read_or_synthesize_mesh_config,
    synthesize_from_peer_config,
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
