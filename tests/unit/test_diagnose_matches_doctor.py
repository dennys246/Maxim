"""`maxim.diagnose()` and `maxim doctor --json` run ONE probe set (roadmap 1.3.1).

The CLI exports MAXIM_ROLE at startup (runtime.role.detect_and_apply_role) before any subcommand, and
the doctor read only that env var -- so in a Python session (nothing exported) `diagnose()` fell back
to "auto", skipped the remote-leader probe, and reported all-passed while the CLI exited 1 on that
probe. The doctor now resolves the role through the CLI's own resolver."""

from __future__ import annotations

import json

import pytest


@pytest.fixture
def peer_machine(tmp_path, monkeypatch):
    """A machine configured as a peer by its peer.yml -- nothing exported in the environment."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("MAXIM_ROLE", raising=False)
    monkeypatch.delenv("MAXIM_LANE_LARGE_REMOTE_URL", raising=False)
    from maxim.peer.config import PeerConfig, write_peer_config

    write_peer_config(PeerConfig(url="http://leader.test:8100/v1", api_key="sk-test-key"))
    from maxim.runtime.role import detect_role

    assert detect_role()[0] == "peer", "fixture: the peer.yml must make this machine a peer"


def _cli_json(monkeypatch, capsys) -> tuple[dict, int]:
    """`maxim doctor --json` as the CLI runs it: the role exported first, as cli.py::main does."""
    from maxim.doctor.cli import run_doctor_subcommand
    from maxim.runtime.role import detect_role

    monkeypatch.setenv("MAXIM_ROLE", detect_role()[0])
    capsys.readouterr()
    code = run_doctor_subcommand(["--json"])
    return json.loads(capsys.readouterr().out), code


def test_both_entry_points_run_the_same_checks(peer_machine, monkeypatch, capsys) -> None:
    import maxim

    report = maxim.diagnose()  # a Python session: nothing exported yet
    api_names = sorted(check.name for check in report.all_checks)
    cli, code = _cli_json(monkeypatch, capsys)
    cli_names = sorted(result["name"] for section in cli["sections"] for result in section["checks"])

    assert "Remote leader probe" in api_names, "diagnose() skipped the peer probe the CLI runs"
    assert api_names == cli_names
    # ...and they agree on the verdict (the probe cannot reach leader.test: the suite has no network)
    assert report.all_passed is (code == 0)
