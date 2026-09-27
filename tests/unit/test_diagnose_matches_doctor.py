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
    api_checks = sorted((check.name, check.status) for check in report.all_checks)
    cli, code = _cli_json(monkeypatch, capsys)
    cli_checks = sorted((r["name"], r["status"]) for section in cli["sections"] for r in section["checks"])

    assert "Remote leader probe" in {name for name, _ in api_checks}, "diagnose() skipped the peer probe"
    assert api_checks == cli_checks  # same checks AND the same status for each
    # ...and they agree on the verdict (the probe cannot reach leader.test: the suite has no network)
    assert report.all_passed is (code == 0)


def test_a_configured_key_is_never_sent_to_a_url_the_caller_named(peer_machine, monkeypatch) -> None:
    """diagnose(peer=<any url>) filled in this machine's leader key and sent it to that host."""
    from maxim.doctor import checks

    seen = []
    monkeypatch.setattr(
        checks,
        "check_peer_auth",
        lambda url, key: seen.append((url, key)) or checks.CheckResult(name="auth", status="info", message="captured"),
    )
    import maxim

    maxim.diagnose(peer="https://attacker.example/v1")
    maxim.diagnose(peer="http://leader.test:8100/v1/")  # the configured leader (trailing slash)
    assert seen == [("https://attacker.example/v1", None), ("http://leader.test:8100/v1/", "sk-test-key")]


def test_the_configured_key_follows_its_leader_however_the_url_is_spelled(peer_machine) -> None:
    """Scheme/host case and a trailing /v1 name the same leader; another port or host does not."""
    from maxim.doctor.checks import _configured_peer_key_for

    for same in ("http://leader.test:8100/v1", "HTTP://Leader.Test:8100/v1/", "http://leader.test:8100"):
        assert _configured_peer_key_for(same) == "sk-test-key", same
    for other in ("http://leader.test:8101/v1", "http://leader.test.evil:8100/v1", "https://leader.test:8100/v1"):
        assert _configured_peer_key_for(other) is None, other


def test_the_role_row_reports_where_the_cli_resolved_the_role(monkeypatch) -> None:
    """The CLI exports MAXIM_ROLE at startup; the doctor's role row must still name the real source,
    not read its own export back as an explicit env setting."""
    monkeypatch.delenv("MAXIM_ROLE", raising=False)
    from maxim.doctor.checks import check_resolved_config
    from maxim.runtime.role import apply_role

    apply_role("leader", "default")  # what detect_and_apply_role does on an unconfigured machine
    row = next(r for r in check_resolved_config() if r.name == "role")
    assert row.message == "leader  [source=default]" and row.status == "info"


def test_an_exported_role_that_overrides_config_json_warns(tmp_path, monkeypatch) -> None:
    """Every env-shadowed row warns with a fix; the role row's special case must not lose that."""
    from maxim.doctor.checks import check_resolved_config
    from maxim.runtime.role import detect_and_apply_role

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(
        "maxim.doctor.checks._read_config_for_doctor", lambda cfg, field: "solo" if field == "role" else None
    )
    monkeypatch.setenv("MAXIM_ROLE", "peer")
    detect_and_apply_role(["doctor"])
    row = next(r for r in check_resolved_config() if r.name == "role")
    assert row.status == "warn" and "shadows config.json=solo" in row.message and row.fix


def test_the_role_row_is_the_same_from_both_entry_points_on_an_unconfigured_machine(tmp_path, monkeypatch) -> None:
    """The case resolved_role exists for: a default leader. The CLI exports it before the doctor runs."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("MAXIM_ROLE", raising=False)
    monkeypatch.delenv("MAXIM_LANE_LARGE_REMOTE_URL", raising=False)
    from maxim.doctor.checks import check_resolved_config
    from maxim.runtime.role import detect_and_apply_role, detect_role

    if detect_role() != ("leader", "default"):
        pytest.skip("this machine resolves a configured role even with HOME/XDG isolated")

    def _row():
        r = next(r for r in check_resolved_config() if r.name == "role")
        return r.status, r.message

    api_row = _row()  # a Python session: nothing applied
    detect_and_apply_role(["doctor", "--json"])  # the CLI's startup
    assert _row() == api_row == ("info", "leader  [source=default]")


def test_the_env_configured_key_follows_its_own_url(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))  # no peer.yml
    monkeypatch.setenv("MAXIM_LANE_LARGE_REMOTE_URL", "https://Leader.example.com/v1")
    monkeypatch.setenv("MAXIM_LANE_LARGE_REMOTE_API_KEY", "sk-env")
    from maxim.doctor.checks import _configured_peer_key_for

    assert _configured_peer_key_for("https://leader.example.com") == "sk-env"
    assert _configured_peer_key_for("https://other.example.com/v1") is None
