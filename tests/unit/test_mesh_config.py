"""Tests for maxim.peer.mesh_config (Plan 4 Stage C1)."""

from __future__ import annotations

import pytest

from maxim.peer.mesh_config import (
    MeshConfigError,
    MeshNode,
    drain_node,
    parse_mesh_config,
    read_drained_nodes,
    read_mesh_config,
    read_or_synthesize_mesh_config,
    resume_node,
    synthesize_from_peer_config,
    write_drained_nodes,
)


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
drain:
  - mac-studio
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
        assert cfg.drain == ("mac-studio",)

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
        with pytest.raises(MeshConfigError, match=r"mesh\.yml line \d+:.*invalid role 'overlord'"):
            parse_mesh_config(yaml)

    def test_malformed_url_scheme_rejected(self) -> None:
        yaml = VALID_YAML.replace("http://192.168.1.10:8099/v1", "ftp://bad/v1")
        with pytest.raises(MeshConfigError, match="must use http:// or https://"):
            parse_mesh_config(yaml)

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

    def test_optional_drain_absent(self) -> None:
        yaml = "\n".join(
            line for line in VALID_YAML.splitlines() if not (line.startswith("drain") or line.strip() == "- mac-studio")
        )
        cfg = parse_mesh_config(yaml)
        assert cfg.drain == ()


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


class TestDrainState:
    def test_round_trip(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path))
        monkeypatch.setenv("MAXIM_ROLE", "leader")
        from maxim.utils import paths

        paths._reset_caches()

        write_drained_nodes({"mac-studio", "tablet"})
        assert read_drained_nodes() == {"mac-studio", "tablet"}

    def test_drain_and_resume(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path))
        monkeypatch.setenv("MAXIM_ROLE", "leader")
        from maxim.utils import paths

        paths._reset_caches()

        drain_node("mac-studio")
        assert read_drained_nodes() == {"mac-studio"}
        drain_node("tablet")
        assert read_drained_nodes() == {"mac-studio", "tablet"}
        resume_node("mac-studio")
        assert read_drained_nodes() == {"tablet"}
        resume_node("tablet")
        assert read_drained_nodes() == set()

    def test_role_isolation(self, tmp_path, monkeypatch) -> None:
        """Leader and peer drain state files must not leak into each other."""
        monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path))
        from maxim.utils import paths

        paths._reset_caches()

        monkeypatch.setenv("MAXIM_ROLE", "leader")
        write_drained_nodes({"mac-studio"})

        monkeypatch.setenv("MAXIM_ROLE", "peer")
        assert read_drained_nodes() == set()
        write_drained_nodes({"tablet"})

        monkeypatch.setenv("MAXIM_ROLE", "leader")
        assert read_drained_nodes() == {"mac-studio"}

    def test_empty_file_handled(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path))
        monkeypatch.setenv("MAXIM_ROLE", "leader")
        from maxim.utils import paths

        paths._reset_caches()

        write_drained_nodes(set())
        assert read_drained_nodes() == set()
