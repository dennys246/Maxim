"""#1806 PR 0: one Reachy daemon probe, ``hardware/reachy/probe.py::probe_daemon`` + ``refusal``.

Every network step is patched at the probe's own seams (``resolve_ipv4``'s ``socket.gethostbyname``,
``tcp_connect_error``, ``read_daemon_status``'s ``fetch_url``) so the probe's composition really runs. No socket
leaves the process.
"""

from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

import pytest

from maxim.hardware.reachy import probe as reachy_probe
from maxim.utils import http as maxim_http

RUNNING = {
    "version": "1.8.3",
    "state": "running",
    "media_released": False,
    "hardware_id": "0123456789abcdef",
    "wlan_ip": "10.6.0.63",
    "simulation_enabled": False,
    "mockup_sim_enabled": False,
    "robot_name": "reachy_mini",
}


def _resp(body):
    r = MagicMock()
    r.json.return_value = body
    return r


def _probe(target="10.6.0.63", *, status=RUNNING, sdk=(1, 8), tcp=None, fetch=None, resolve=None):
    fetch_mock = MagicMock(return_value=_resp(status)) if fetch is None else fetch
    with (
        patch.object(reachy_probe, "installed_sdk_version", return_value=sdk),
        patch("socket.gethostbyname", **(resolve or {"return_value": "10.6.0.63"})) as lookup,
        patch.object(reachy_probe, "tcp_connect_error", return_value=tcp) as dial,
        patch("maxim.utils.http.fetch_url", fetch_mock),
    ):
        outcome = reachy_probe.probe_daemon(target, deadline_s=4.0)
    return outcome, lookup, dial, fetch_mock


# ── the stages ─────────────────────────────────────────────────────────────────────────────────────────────────


def test_a_running_matching_daemon_carries_every_status_field_and_refuses_nothing():
    outcome, lookup, dial, fetch = _probe()
    assert outcome.failure is None and outcome.address == "10.6.0.63" and outcome.era == "match"
    assert (outcome.version, outcome.state, outcome.media_released) == ("1.8.3", "running", False)
    assert (outcome.hardware_id, outcome.wlan_ip, outcome.robot_name) == (
        "0123456789abcdef",
        "10.6.0.63",
        "reachy_mini",
    )
    assert outcome.simulation_enabled is False and outcome.mockup_sim_enabled is False
    assert reachy_probe.refusal(outcome, None) is None
    lookup.assert_not_called()  # an IPv4 literal is never looked up
    assert dial.call_args.args[:2] == ("10.6.0.63", 8000)


def test_the_status_read_dials_the_resolved_literal_directly_with_a_deadline():
    outcome, lookup, dial, fetch = _probe("reachy-mini.local", resolve={"return_value": "192.168.1.40"})
    lookup.assert_called_once_with("reachy-mini.local")
    assert dial.call_args.args[0] == "192.168.1.40"
    assert fetch.call_args.args[0] == "http://192.168.1.40:8000/api/daemon/status"
    kw = fetch.call_args.kwargs
    assert kw["address_policy"] == "private" and kw["proxy"] == "never" and 0 < kw["deadline_s"] <= 4.0
    assert outcome.target == "reachy-mini.local" and outcome.address == "192.168.1.40"


def test_an_unresolvable_name_stops_before_any_dial():
    outcome, _, dial, fetch = _probe("nope.local", resolve={"side_effect": socket.gaierror("nodename")})
    assert outcome.failure == "resolve" and outcome.address is None
    dial.assert_not_called()
    fetch.assert_not_called()
    assert reachy_probe.refusal(outcome, None).startswith("unreachable")


def test_a_closed_port_stops_before_the_status_read():
    outcome, _, _, fetch = _probe(tcp="connection refused")
    assert outcome.failure == "tcp" and outcome.error == "connection refused" and outcome.reachable is False
    fetch.assert_not_called()
    assert "10.6.0.63:8000" in reachy_probe.refusal(outcome, None)


@pytest.mark.parametrize(
    "fetch",
    [
        MagicMock(side_effect=maxim_http.HTTPConnectionError("refused")),
        MagicMock(return_value=_resp(["not", "an", "object"])),
        MagicMock(return_value=MagicMock(json=MagicMock(side_effect=ValueError("not json")))),
    ],
)
def test_an_unreadable_status_is_reachable_but_refused(fetch):
    outcome, _, _, _ = _probe(fetch=fetch)
    assert outcome.failure == "status" and outcome.reachable is True and outcome.status_read is False
    assert outcome.media_released is None
    assert "status unreadable" in reachy_probe.refusal(outcome, None)


# ── era / minor / state ────────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("sdk", "version", "era"),
    [
        ((1, 2), "1.8.3", "era_mismatch"),
        ((1, 8), "1.2.6", "era_mismatch"),
        ((1, 2), "1.2.6", "era_mismatch"),  # a zenoh-era SDK is never a match
        ((1, 8), "1.9.0", "minor_drift"),
        ((1, 8), "1.8.4", "match"),
        (None, "1.8.3", "sdk_missing"),
        ((1, 8), "garbage", "unknown"),
    ],
)
def test_era_verdicts_and_every_non_match_refuses(sdk, version, era):
    outcome, *_ = _probe(status={**RUNNING, "version": version}, sdk=sdk)
    assert outcome.era == era
    assert (reachy_probe.refusal(outcome, None) is None) == (era == "match")


@pytest.mark.parametrize("state", ["starting", "error", None])
def test_a_daemon_not_running_refuses(state):
    status = {**RUNNING, "state": state}
    outcome, *_ = _probe(status=status)
    assert "not 'running'" in reachy_probe.refusal(outcome, None)


# ── untrusted status text ──────────────────────────────────────────────────────────────────────────────────────


def test_status_fields_are_type_checked_and_bounded():
    hostile = {
        **RUNNING,
        "robot_name": "\x1b]0;pwned\x07" + "r" * 500,
        "hardware_id": {"nested": "object"},
        "media_released": "false",  # a string is not the daemon's bool
        "simulation_enabled": 1,
    }
    outcome, *_ = _probe(status=hostile)
    assert outcome.robot_name is not None and len(outcome.robot_name) <= 128
    assert all(ch.isprintable() for ch in outcome.robot_name) and "\x1b" not in outcome.robot_name
    assert outcome.hardware_id is None
    assert outcome.media_released is None and outcome.simulation_enabled is None


def test_extra_cannot_shadow_a_declared_field():
    with pytest.raises(ValueError, match="collide"):
        reachy_probe.ProbeOutcome(extra={"hardware_id": "x"})
    assert reachy_probe.ProbeOutcome(extra={"later": 1}).extra == {"later": 1}


def test_a_meaningless_deadline_is_refused():
    with pytest.raises(ValueError):
        reachy_probe.probe_daemon("10.6.0.63", deadline_s=0)


# ── the consumers route through it ─────────────────────────────────────────────────────────────────────────────


def test_controller_connect_takes_the_1704_pre_connect_state_from_the_probe():
    """The controller's #1704 ownership read is the probe's ``media_released``, not a second status read."""
    import sys

    from maxim.hardware.reachy.controller import ReachyMiniController

    mini = MagicMock()
    mini.media.camera = None
    mini.media.audio = None
    mini.media_released = True
    mod = MagicMock()
    mod.ReachyMini.return_value = mini
    c = ReachyMiniController(host="10.42.0.1", connection_mode="network", media_backend="no_media")
    with (
        patch.dict(sys.modules, {"reachy_mini": mod}),
        patch.object(ReachyMiniController, "_probe_state_shape", lambda self: None),
        patch.object(reachy_probe, "installed_sdk_version", return_value=(1, 8)),
        patch.object(reachy_probe, "tcp_connect_error", return_value=None),
        patch.object(reachy_probe, "read_daemon_status", return_value={**RUNNING, "media_released": False}) as read,
    ):
        assert c.connect(timeout=2.0) is True
    read.assert_called_once()
    assert read.call_args.args[:2] == ("10.42.0.1", 8000)
    assert c._media_released_by_connect is True
    assert mod.ReachyMini.call_args.kwargs["automatic_body_yaw"] is False


def test_controller_connect_survives_an_unreadable_status():
    import sys

    from maxim.hardware.reachy.controller import ReachyMiniController

    mod = MagicMock()
    c = ReachyMiniController(host="10.42.0.1", connection_mode="network")
    with (
        patch.dict(sys.modules, {"reachy_mini": mod}),
        patch.object(ReachyMiniController, "_probe_state_shape", lambda self: None),
        patch.object(reachy_probe, "installed_sdk_version", return_value=(1, 8)),
        patch.object(reachy_probe, "tcp_connect_error", return_value=None),
        patch.object(reachy_probe, "read_daemon_status", side_effect=reachy_probe.DaemonStatusUnreadable("x")),
    ):
        assert c.connect(timeout=2.0) is True  # a failed status read stays non-fatal at connect
    mod.ReachyMini.assert_called_once()


def test_maxim_diagnostics_reports_the_probe_outcome(monkeypatch, capsys):
    from maxim.utils import reachy_diagnostics as diag

    monkeypatch.setenv("MAXIM_REACHY_HOST", "10.6.0.63")
    monkeypatch.setattr(diag, "test_ping", lambda host: (True, "ok"))
    monkeypatch.setattr(diag, "test_port", lambda host, port: (True, "open"))
    monkeypatch.setattr("sys.argv", ["maxim-diagnostics"])
    monkeypatch.setattr("maxim.runtime.config_loader.engage_offline_if_configured", lambda: None)
    probed: list[str] = []

    def fake_probe(target, **kw):
        probed.append(target)
        return reachy_probe.ProbeOutcome(target=target, address=target, failure="tcp", error="connection refused")

    monkeypatch.setattr(reachy_probe, "probe_daemon", fake_probe)
    monkeypatch.setattr(maxim_http, "fetch_url", MagicMock(return_value=MagicMock(status=200)))
    assert diag.main() == 1
    out = capsys.readouterr().out
    assert probed == ["10.6.0.63"]
    assert "connection refused" in out


def test_maxim_diagnostics_warns_when_a_connect_check_would_refuse(monkeypatch, capsys):
    from maxim.utils import reachy_diagnostics as diag

    monkeypatch.setattr(diag, "test_ping", lambda host: (True, "ok"))
    monkeypatch.setattr(diag, "test_port", lambda host, port: (True, "open"))
    monkeypatch.setattr(
        reachy_probe,
        "probe_daemon",
        lambda target, **kw: reachy_probe.ProbeOutcome(
            target=target, address=target, version="1.9.0", state="running", sdk_version=(1, 8), era="minor_drift"
        ),
    )
    monkeypatch.setattr(maxim_http, "fetch_url", MagicMock(return_value=MagicMock(status=200)))
    assert diag.diagnose_reachy("10.6.0.63") == 0
    out = capsys.readouterr().out
    assert "version=1.9.0 state=running" in out and "minor-version mismatch" in out
