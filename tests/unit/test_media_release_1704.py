"""#1704: a ``no_media`` connect must not leave the daemon's media released after disconnect.

SDK 1.8.3's ``ReachyMini(media_backend="no_media")`` calls ``release_media()`` when the daemon owns the
camera and microphone, and the release is daemon-wide. The SDK re-acquires only in ``ReachyMini.__exit__``;
``ReachyMiniController.disconnect()`` dropped the handle, so the daemon stayed released (``media_released:
true``, WebRTC signalling on :8443 refused) and every later camera client failed to connect (2026-11-19 robot
run 1 → run 2, ``[Errno 61] Connection refused``).

The controller re-acquires on disconnect only what ITS connect released: the daemon reported
``media_released: false`` before the connect and the SDK handle reports ``media_released`` after it. A daemon
already released by someone else stays released, and an unknown pre-connect state is never guessed.
"""

from __future__ import annotations

import logging
import sys
from unittest.mock import MagicMock, patch


from maxim.hardware.reachy import probe as reachy_probe
from maxim.hardware.reachy.controller import ReachyMiniController


def _status_or_unreadable(status: dict | None):
    """The probe's status-read seam (#1806): the daemon's JSON object, or an unreadable status for ``None``."""

    def read(*_a, **_k):
        if status is None:
            raise reachy_probe.DaemonStatusUnreadable("unit test: no status")
        return dict(status)

    return read


def _sdk_mini(*, released_by_connect: bool, acquire_works: bool = True) -> MagicMock:
    """A ReachyMini handle with the SDK's ``media_released`` property and ``acquire_media()``."""
    mini = MagicMock()
    mini.media.camera = None
    mini.media.audio = None
    mini.media_released = released_by_connect

    def acquire_media() -> None:
        if acquire_works:
            mini.media_released = False

    mini.acquire_media.side_effect = acquire_media
    return mini


def _connect(
    monkeypatch, mini: MagicMock, status: dict | None, *, c: ReachyMiniController | None = None, expect: bool = True
) -> ReachyMiniController:
    monkeypatch.setattr(reachy_probe, "read_daemon_status", _status_or_unreadable(status))
    monkeypatch.setattr(reachy_probe, "installed_sdk_version", lambda: (1, 8))
    # connect's one REST readback (GET /api/state/full) is not what these tests are about: no network.
    monkeypatch.setattr(ReachyMiniController, "_probe_state_shape", lambda self: None)
    mod = MagicMock()
    mod.ReachyMini.return_value = mini
    if c is None:
        c = ReachyMiniController(host="10.42.0.1", connection_mode="network", media_backend="no_media")
    with (
        patch.dict(sys.modules, {"reachy_mini": mod}),
        patch.object(reachy_probe, "tcp_connect_error", return_value=None),
    ):
        assert c.connect(timeout=2.0) is expect
    return c


def test_a_no_media_connect_that_released_media_reacquires_it_on_disconnect(monkeypatch):
    mini = _sdk_mini(released_by_connect=True)
    c = _connect(monkeypatch, mini, {"state": "running", "media_released": False})
    c.disconnect()
    assert mini.acquire_media.call_count == 1
    assert mini.media_released is False


def test_a_connect_that_released_nothing_never_acquires(monkeypatch):
    mini = _sdk_mini(released_by_connect=False)
    c = _connect(monkeypatch, mini, {"state": "running", "media_released": False})
    c.disconnect()
    assert mini.acquire_media.call_count == 0


def test_media_released_before_the_connect_stays_released(monkeypatch):
    """Someone else released the daemon's media (direct camera access): not this controller's to undo."""
    mini = _sdk_mini(released_by_connect=True)
    c = _connect(monkeypatch, mini, {"state": "running", "media_released": True})
    c.disconnect()
    assert mini.acquire_media.call_count == 0


def test_an_unknown_pre_connect_state_is_never_guessed_and_is_reported(monkeypatch, caplog):
    mini = _sdk_mini(released_by_connect=True)
    with caplog.at_level(logging.WARNING, logger="maxim.hardware.reachy.controller"):
        c = _connect(monkeypatch, mini, None)
        c.disconnect()
    assert mini.acquire_media.call_count == 0
    assert any("media" in r.getMessage() and "acquire" in r.getMessage() for r in caplog.records)


def test_a_failed_reacquire_is_reported_and_the_disconnect_still_completes(monkeypatch, caplog):
    mini = _sdk_mini(released_by_connect=True, acquire_works=False)
    c = _connect(monkeypatch, mini, {"state": "running", "media_released": False})
    with caplog.at_level(logging.WARNING, logger="maxim.hardware.reachy.controller"):
        c.disconnect()
    assert mini.acquire_media.call_count == 1
    assert c._mini is None
    assert any("/api/media/acquire" in r.getMessage() for r in caplog.records)


# ── review-round folds (Executor, Architecture, Wire integrity, 2026-11-19) ───────────────────────────────


def test_the_reacquire_runs_before_the_ssh_tunnel_stops(monkeypatch):
    """Under tunnel=True the SDK client's host is the tunnel's 127.0.0.1 end: acquire first, then the tunnel."""
    order: list[str] = []
    mini = _sdk_mini(released_by_connect=True)
    mini.acquire_media.side_effect = lambda: (order.append("acquire"), setattr(mini, "media_released", False))
    c = _connect(monkeypatch, mini, {"state": "running", "media_released": False})
    monkeypatch.setattr(c, "_stop_ssh_tunnel", lambda: order.append("tunnel_stop"))
    c.disconnect()
    assert order == ["acquire", "tunnel_stop"]


def test_an_acquire_that_raises_is_reported_and_ownership_carries_over(monkeypatch, caplog):
    """A dead link (the soft-reconnect path) can raise inside acquire_media after its POST: report, keep owning."""
    mini = _sdk_mini(released_by_connect=True)
    mini.acquire_media.side_effect = TimeoutError("get_status timed out")
    c = _connect(monkeypatch, mini, {"state": "running", "media_released": False})
    with caplog.at_level(logging.WARNING, logger="maxim.hardware.reachy.controller"):
        c.disconnect()
    assert c._mini is None
    assert c._media_released_by_connect is True
    messages = [r.getMessage() for r in caplog.records]
    assert any("acquire_media() raised" in m for m in messages)
    assert any("could not be confirmed" in m for m in messages)


def test_a_failed_reacquire_is_retried_by_the_next_disconnect_after_a_reconnect(monkeypatch):
    """acquire fails on a dead link → reconnect (the daemon still reports released) → the final disconnect acquires."""
    first = _sdk_mini(released_by_connect=True, acquire_works=False)
    c = _connect(monkeypatch, first, {"state": "running", "media_released": False})
    c.disconnect()
    assert first.acquire_media.call_count == 1
    second = _sdk_mini(released_by_connect=True)
    _connect(monkeypatch, second, {"state": "running", "media_released": True}, c=c)
    c.disconnect()
    assert second.acquire_media.call_count == 1
    assert second.media_released is False
    assert c._media_released_by_connect is False


def test_a_connect_that_fails_after_the_release_undoes_it(monkeypatch):
    mini = _sdk_mini(released_by_connect=True)
    monkeypatch.setattr(
        ReachyMiniController, "_assert_auto_yaw_off", lambda self: (_ for _ in ()).throw(RuntimeError("ws dropped"))
    )
    _connect(monkeypatch, mini, {"state": "running", "media_released": False}, expect=False)
    assert mini.acquire_media.call_count == 1
    assert mini.media_released is False


def test_the_real_sdk_acquire_does_not_release_again_under_no_media():
    """Bound to the installed SDK (skips without it): acquire_media() rebuilds the media manager through
    _configure_mediamanager("no_media") while _media_released is still True, so it must not release again.
    The fix relies on that; an SDK bump that changes it fails here."""
    import types

    import pytest

    sdk = pytest.importorskip("reachy_mini.reachy_mini")
    mini = sdk.ReachyMini.__new__(sdk.ReachyMini)
    mini.client = MagicMock()
    mini.client.acquire_media.return_value = True
    mini.client.get_status.return_value = types.SimpleNamespace(
        no_media=False, camera_specs_name="", version="1.8.3", media_released=True, wlan_ip=None
    )
    mini._media_released = True
    mini._media_backend = "no_media"
    mini._log_level = "WARNING"
    mini.connection_mode = "network"
    mini._daemon_http_url = "http://127.0.0.1:8000"
    mini.logger = logging.getLogger("reachy_mini_test")
    mini.media_manager = MagicMock()
    mini._warn_if_daemon_version_mismatch = lambda status: None
    mini.acquire_media()
    assert mini.client.acquire_media.call_count == 1
    assert mini.client.release_media.call_count == 0
    assert mini.media_released is False


# ── #1712: the controller closes the SDK client; the edges the #1704 rounds left ──────────────────────────


def test_disconnect_closes_the_sdk_client_after_the_reacquire_and_before_the_tunnel(monkeypatch):
    """The controller is the handle's only long-lived holder (Selfy.mini reads through it): it closes the client
    the SDK's __exit__ would, instead of leaving the WebSocket thread to __del__."""
    order: list[str] = []
    mini = _sdk_mini(released_by_connect=True)
    mini.acquire_media.side_effect = lambda: (order.append("acquire"), setattr(mini, "media_released", False))
    mini.client.disconnect.side_effect = lambda: order.append("client_disconnect")
    c = _connect(monkeypatch, mini, {"state": "running", "media_released": False})
    monkeypatch.setattr(c, "_stop_ssh_tunnel", lambda: order.append("tunnel_stop"))
    c.disconnect()
    assert order == ["acquire", "client_disconnect", "tunnel_stop"]
    assert c._mini is None
    assert mini.media_manager.close.call_count == 0  # the media lifecycle owner closes it (#1435), not the controller


def test_a_raising_stream_stop_still_drops_the_handle_and_closes_the_client(monkeypatch):
    mini = _sdk_mini(released_by_connect=False)
    c = _connect(monkeypatch, mini, {"state": "running", "media_released": False})
    c._video_stream = MagicMock()
    c._video_stream.stop.side_effect = RuntimeError("stream wedged")
    c.disconnect()
    assert c._mini is None
    assert c._video_stream is None
    assert mini.client.disconnect.call_count == 1


def test_a_client_disconnect_that_raises_is_reported_and_the_disconnect_completes(monkeypatch, caplog):
    mini = _sdk_mini(released_by_connect=False)
    mini.client.disconnect.side_effect = OSError("socket already closed")
    c = _connect(monkeypatch, mini, {"state": "running", "media_released": False})
    with caplog.at_level(logging.WARNING, logger="maxim.hardware.reachy.controller"):
        c.disconnect()
    assert c._mini is None
    assert any("client's disconnect() raised" in r.getMessage() for r in caplog.records)


def test_a_failed_connect_drops_its_handle_so_the_next_disconnect_does_not_retry_on_it(monkeypatch, caplog):
    """An unconfirmed re-acquire on the failed connect's handle is not retried on that dead handle; the
    ownership still carries to the next connect (the #1704 rule), and the WARNING names the failed connect."""
    mini = _sdk_mini(released_by_connect=True, acquire_works=False)
    monkeypatch.setattr(
        ReachyMiniController, "_assert_auto_yaw_off", lambda self: (_ for _ in ()).throw(RuntimeError("ws dropped"))
    )
    with caplog.at_level(logging.WARNING, logger="maxim.hardware.reachy.controller"):
        c = _connect(monkeypatch, mini, {"state": "running", "media_released": False}, expect=False)
    assert c._mini is None
    assert mini.client.disconnect.call_count == 1
    assert c._media_released_by_connect is True
    assert any("during a failed connect" in r.getMessage() for r in caplog.records)
    c.disconnect()
    assert mini.acquire_media.call_count == 1


def test_a_camera_connect_onto_released_media_names_the_fix(monkeypatch, caplog):
    """A killed no_media client leaves the daemon released, and the SDK never re-acquires for a camera backend:
    the connect says so and names POST /api/media/acquire, instead of a bare Errno 61 later."""
    mini = _sdk_mini(released_by_connect=False)
    c = ReachyMiniController(host="10.42.0.1", connection_mode="network", media_backend="default")
    with caplog.at_level(logging.WARNING, logger="maxim.hardware.reachy.controller"):
        _connect(monkeypatch, mini, {"state": "running", "media_released": True}, c=c)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("media_released: true" in m and "10.42.0.1:8000/api/media/acquire" in m for m in warnings)


def test_a_no_media_connect_onto_released_media_is_not_warned(monkeypatch, caplog):
    mini = _sdk_mini(released_by_connect=True)
    with caplog.at_level(logging.WARNING, logger="maxim.hardware.reachy.controller"):
        _connect(monkeypatch, mini, {"state": "running", "media_released": True})
    assert not any("before this connect (media_backend" in r.getMessage() for r in caplog.records)


# ── #1712 part 2: SDK-only scripts undo their own no_media release ───────────────────────────────────────

_SCRIPTS = __import__("pathlib").Path(__file__).resolve().parents[2] / "scripts"


def _load(monkeypatch, name: str, rel: str):
    import importlib.util

    monkeypatch.syspath_prepend(str(_SCRIPTS / "orient_backbone"))
    monkeypatch.syspath_prepend(str(_SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / rel)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_release_mini_runs_the_sdks_own_exit_and_tolerates_none_and_a_dead_link(monkeypatch, capsys):
    smoke = _load(monkeypatch, "live_1_smoke", "orient_backbone/live_1_smoke.py")
    mini = MagicMock()
    smoke.release_mini(mini)
    mini.__exit__.assert_called_once_with(None, None, None)
    smoke.release_mini(None)
    dead = MagicMock()
    dead.__exit__.side_effect = TimeoutError("no route")
    smoke.release_mini(dead)
    assert "POST /api/media/acquire" in capsys.readouterr().out


def test_live_rig_releases_at_exit_on_close_and_before_a_reconnect(monkeypatch):
    common = _load(monkeypatch, "live_common", "orient_backbone/live_common.py")
    first, second = MagicMock(), MagicMock()
    registered: list = []
    monkeypatch.setattr(common.atexit, "register", lambda fn, *a: registered.append(fn))
    monkeypatch.setattr(common, "connect_and_wake", lambda host: (first, MagicMock()))
    monkeypatch.setattr(common, "make_rest_reader", lambda host: MagicMock())
    rig = common.LiveRig("10.42.0.1")
    assert registered == [rig.close]
    sdk = MagicMock()
    sdk.ReachyMini.return_value = second
    monkeypatch.setitem(sys.modules, "reachy_mini", sdk)
    monkeypatch.setattr(common.time, "sleep", lambda _s: None)
    rig._reconnect()
    first.__exit__.assert_called_once_with(None, None, None)  # the dead session's release is undone first
    assert rig.mini is second
    rig.close()
    rig.close()  # idempotent: the atexit call after an explicit close does nothing more
    second.__exit__.assert_called_once_with(None, None, None)
    assert rig.mini is None


def test_connect_and_wake_undoes_the_release_when_waking_fails(monkeypatch):
    common = _load(monkeypatch, "live_common", "orient_backbone/live_common.py")
    mini = MagicMock()
    mini.wake_up.side_effect = ConnectionError("ws dropped")
    sdk = MagicMock()
    sdk.ReachyMini.return_value = mini
    monkeypatch.setitem(sys.modules, "reachy_mini", sdk)
    monkeypatch.setitem(sys.modules, "reachy_mini.utils", MagicMock())
    monkeypatch.setattr(common.time, "sleep", lambda _s: None)
    import pytest

    with pytest.raises(ConnectionError):
        common.connect_and_wake("10.42.0.1")
    mini.__exit__.assert_called_once_with(None, None, None)


def test_signals_that_skip_atexit_are_turned_into_a_normal_exit(monkeypatch):
    """SIGTERM/SIGHUP's default handlers end the process without atexit (kill, timeout, a closed tmux pane)."""
    import signal

    smoke = _load(monkeypatch, "live_1_smoke", "orient_backbone/live_1_smoke.py")
    installed: dict = {}
    foreign = object()
    current = {signal.SIGTERM: signal.SIG_DFL, signal.SIGHUP: foreign}
    monkeypatch.setattr(smoke.signal, "getsignal", lambda sig: current[sig])
    monkeypatch.setattr(smoke.signal, "signal", lambda sig, fn: installed.__setitem__(sig, fn))
    smoke.exit_on_hangup_and_term()
    assert list(installed) == [signal.SIGTERM]  # a handler someone else installed is left alone
    import pytest

    with pytest.raises(SystemExit) as exited:
        installed[signal.SIGTERM](signal.SIGTERM, None)
    assert exited.value.code == 128 + signal.SIGTERM


def test_release_mini_reports_a_reacquire_the_daemon_refused(monkeypatch, capsys):
    """acquire_media() signals a refused or unreachable POST only by leaving the handle released (no raise)."""
    smoke = _load(monkeypatch, "live_1_smoke", "orient_backbone/live_1_smoke.py")
    mini = MagicMock()
    mini.media_released = True  # __exit__ ran, but the daemon never re-acquired
    smoke.release_mini(mini)
    assert "POST /api/media/acquire" in capsys.readouterr().out
    ok = MagicMock()
    ok.media_released = False
    smoke.release_mini(ok)
    assert "media_released" not in capsys.readouterr().out


def test_release_mini_still_closes_the_client_when_exit_raises_midway(monkeypatch):
    """__exit__ can raise in its status read after the POST, before its own client.disconnect()."""
    smoke = _load(monkeypatch, "live_1_smoke", "orient_backbone/live_1_smoke.py")
    mini = MagicMock()
    mini.__exit__.side_effect = TimeoutError("get_status timed out")
    smoke.release_mini(mini)
    assert mini.client.disconnect.call_count == 1


def test_live_1_smoke_registers_the_release_right_after_its_no_media_connect():
    """The lint cannot guard this file (release_mini's own __exit__ satisfies it), so pin the registration:
    in main, the signal hook precedes the connect and atexit.register(release_mini, mini) follows it."""
    import ast

    tree = ast.parse((_SCRIPTS / "orient_backbone" / "live_1_smoke.py").read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)]
    connect = min(n.lineno for n in calls if getattr(n.func, "id", None) == "ReachyMini")
    registered = [
        n.lineno
        for n in calls
        if getattr(n.func, "attr", None) == "register"
        and getattr(n.func.value, "id", None) == "atexit"
        and [getattr(a, "id", None) for a in n.args] == ["release_mini", "mini"]
    ]
    hooked = [n.lineno for n in calls if getattr(n.func, "id", None) == "exit_on_hangup_and_term"]
    assert len(registered) == 1 and registered[0] > connect
    assert len(hooked) == 1 and hooked[0] < connect


def test_a_move_that_times_out_during_the_disconnect_leaves_no_latch_for_the_next_connection(monkeypatch):
    """disconnect resets the settle latch, then re-acquires media (an HTTP POST, seconds on a bad link) before it
    drops the handle. A chunk that times out in that window still holds its handle and latches; the reset after
    the swap clears it, so the reconnected controller's first move is not refused (delta round 2, #1712)."""
    mini = _sdk_mini(released_by_connect=True)
    c = _connect(monkeypatch, mini, {"state": "running", "media_released": False})

    def acquire_while_a_chunk_times_out() -> None:
        c._unsettled = ("chunk", None, None)  # what _run_move's TimeoutError branch writes, mid-disconnect
        mini.media_released = False

    mini.acquire_media.side_effect = acquire_while_a_chunk_times_out
    c.disconnect()
    assert mini.acquire_media.call_count == 1
    assert c._unsettled is None
