"""#1780 (1.3.5 line 6 item 7): ``maxim peer connect`` says what the leader serves, and accepts ``--cloud``.

Owner decision 2026-11-23, after the review round: medium and small lanes serve no model call in production (only
``manager.get_backend("large")`` is built, in ``lane_backends.build_primary_router``), so connect does not grow an
``--all-lanes`` flag. It says the large lane, the one model calls go through, now points at the leader. ``--cloud`` was read from argv but
rejected by the parser as an unknown option.
"""

from __future__ import annotations

import pytest

from maxim.peer.cli import run_peer_connect_subcommand
from maxim.runtime.config_loader import reset_config_cache

LEADER = "http://100.64.0.7:8099/v1"


@pytest.fixture
def peer_home(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """config.json and peer.yml in a tmp dir."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    import maxim.peer.config as peer_config

    monkeypatch.setattr(peer_config, "peer_config_path", lambda: tmp_path / "maxim" / "peer.yml")
    reset_config_cache()
    yield tmp_path
    reset_config_cache()


def _connect(*extra: str) -> int:
    return run_peer_connect_subcommand(["connect", LEADER, "--key", "k" * 32, "--skip-test", *extra])


def test_connect_says_the_large_lane_now_points_at_the_leader(peer_home, capsys) -> None:
    assert _connect() == 0
    out = capsys.readouterr().out
    assert "large lane, which now points at this" in out
    assert "medium and small lanes serve no model calls" in out
    assert "unreachable when maxim starts, the large lane runs locally instead" in out


def test_the_usage_text_lists_cloud(capsys) -> None:
    assert run_peer_connect_subcommand(["connect", "--help"]) == 0
    assert "--cloud          Mark the URL public-facing" in capsys.readouterr().out


def test_cloud_is_accepted_and_marks_the_url_public_facing(peer_home, capsys) -> None:
    from maxim.peer.config import read_peer_config

    assert _connect("--cloud") == 0
    assert read_peer_config().is_cloud is True
    out = capsys.readouterr().out
    assert "cloud-provider routing is unchanged" in out
    assert "auto-enabled" not in out  # the old banner claimed a cloud gate the flag never sets
