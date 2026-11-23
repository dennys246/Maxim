"""Line 6 item 7: ``maxim peer connect`` says where every lane goes, and a remote lane is never silently dropped.

- **The no-op.** ``apply_lane_env_overrides`` patched only the tiers this host's hardware detected, and only large
  had a placeholder (``build_primary_router``). So ``lanes.medium.remote_url`` on a GPU box (detected: large and
  small) was ignored, with nothing logged. A remote URL now creates any of the three tiers.
- **The connect.** Owner decision 2026-11-23: the default stays large-only (the leader serves one generation at a
  time), ``--all-lanes`` points all three at the leader, and connect prints each lane's destination. A large-only
  connect clears medium and small left by an earlier ``--all-lanes``, and ``forget`` clears every tier.
"""

from __future__ import annotations

import pytest

from maxim.peer.cli import run_peer_connect_subcommand
from maxim.runtime.config_loader import load_config, reset_config_cache
from maxim.runtime.lane_models import REMOTE_CAPABLE_TIERS, apply_lane_env_overrides
from maxim.runtime.worker_pool import LaneConfig

LEADER = "http://100.64.0.7:8099/v1"


def _gpu_box() -> dict[str, LaneConfig]:
    """What detect_tiers gives a CUDA box: large and small, no medium."""
    return {
        "large": LaneConfig(name="large", max_workers=1, model_profile="qwen2.5-14b-instruct", requires_gpu=True),
        "small": LaneConfig(name="small", max_workers=2, model_profile="smollm-1.7b-instruct", device="cpu"),
    }


@pytest.fixture
def no_lane_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for tier in REMOTE_CAPABLE_TIERS:
        for suffix in ("REMOTE_URL", "REMOTE_MODEL", "REMOTE_API_KEY", "TIMEOUT_S"):
            monkeypatch.delenv(f"MAXIM_LANE_{tier.upper()}_{suffix}", raising=False)


def test_a_remote_url_on_an_undetected_tier_creates_that_lane(monkeypatch, no_lane_env) -> None:
    monkeypatch.setenv("MAXIM_LANE_MEDIUM_REMOTE_URL", LEADER)
    out = apply_lane_env_overrides(_gpu_box())
    assert out["medium"].remote_url == LEADER
    assert out["medium"].requires_gpu is False  # the leader's hardware serves it
    assert out["large"].remote_url is None  # detected tiers are untouched without their own URL


def test_large_still_gets_its_lane_on_a_box_with_no_large_tier(monkeypatch, no_lane_env) -> None:
    """The removed build_primary_router placeholder's case: a Mac or CPU box with no large tier."""
    monkeypatch.setenv("MAXIM_LANE_LARGE_REMOTE_URL", LEADER)
    cpu_box = {"small": _gpu_box()["small"]}
    out = apply_lane_env_overrides(cpu_box)
    assert out["large"].remote_url == LEADER
    assert out["large"].max_workers == 1


def test_an_undetected_tier_without_a_url_is_not_invented(no_lane_env) -> None:
    out = apply_lane_env_overrides(_gpu_box())
    assert set(out) == {"large", "small"}


@pytest.fixture
def peer_home(tmp_path, monkeypatch: pytest.MonkeyPatch, no_lane_env):
    """config.json and peer.yml in a tmp dir; hardware detection pinned to a CUDA box with nothing downloaded."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    import maxim.peer.config as peer_config

    monkeypatch.setattr(peer_config, "peer_config_path", lambda: tmp_path / "maxim" / "peer.yml")
    monkeypatch.setattr("maxim.runtime.lane_models.detect_tiers", lambda *a, **k: _gpu_box())
    reset_config_cache()
    yield tmp_path
    reset_config_cache()


def _connect(*extra: str) -> int:
    return run_peer_connect_subcommand(["connect", LEADER, "--key", "k" * 32, "--skip-test", *extra])


def _remote_urls() -> dict[str, str | None]:
    reset_config_cache()
    lanes = load_config().lanes
    return {tier: getattr(lanes, tier).remote_url for tier in REMOTE_CAPABLE_TIERS}


def test_connect_points_only_large_at_the_leader_and_says_where_the_rest_go(peer_home, capsys) -> None:
    assert _connect() == 0
    assert _remote_urls() == {"large": LEADER, "medium": None, "small": None}
    out = capsys.readouterr().out
    assert f"large   → leader ({LEADER})" in out
    assert "small   → local, smollm-1.7b-instruct on cpu" in out
    assert "medium  → none on this machine" in out
    assert "--all-lanes" in out


def test_connect_all_lanes_points_every_lane_at_the_leader(peer_home, capsys) -> None:
    assert _connect("--all-lanes") == 0
    assert _remote_urls() == dict.fromkeys(REMOTE_CAPABLE_TIERS, LEADER)
    out = capsys.readouterr().out
    for tier in REMOTE_CAPABLE_TIERS:
        assert f"{tier:<7} → leader ({LEADER})" in out


def test_a_later_large_only_connect_clears_the_other_lanes(peer_home) -> None:
    assert _connect("--all-lanes") == 0
    assert _connect() == 0
    assert _remote_urls() == {"large": LEADER, "medium": None, "small": None}


def test_forget_clears_every_lane(peer_home, capsys) -> None:
    assert _connect("--all-lanes") == 0
    assert run_peer_connect_subcommand(["forget"]) == 0
    assert _remote_urls() == dict.fromkeys(REMOTE_CAPABLE_TIERS)
    out = capsys.readouterr().out
    for tier in REMOTE_CAPABLE_TIERS:
        assert f"Cleared lanes.{tier}.*" in out


def test_a_failed_hardware_detection_still_saves_and_says_so(peer_home, monkeypatch, capsys) -> None:
    def _boom(*_a, **_k):
        raise RuntimeError("no torch")

    monkeypatch.setattr("maxim.runtime.lane_models.detect_tiers", _boom)
    assert _connect() == 0
    assert _remote_urls()["large"] == LEADER
    assert "maxim doctor --as peer" in capsys.readouterr().out


def test_cloud_is_accepted_and_marks_the_url_public_facing(peer_home) -> None:
    """``--cloud`` was read from argv but rejected by the parser as an unknown option (#1780)."""
    from maxim.peer.config import read_peer_config

    assert _connect("--cloud") == 0
    assert read_peer_config().is_cloud is True
