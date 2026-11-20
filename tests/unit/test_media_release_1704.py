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


from maxim.hardware.reachy.controller import ReachyMiniController


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
    monkeypatch.setattr(ReachyMiniController, "_daemon_status", staticmethod(lambda *a, **k: status))
    monkeypatch.setattr(ReachyMiniController, "_installed_sdk_version", staticmethod(lambda: (1, 8)))
    # connect's one REST readback (GET /api/state/full) is not what these tests are about: no network.
    monkeypatch.setattr(ReachyMiniController, "_probe_state_shape", lambda self: None)
    mod = MagicMock()
    mod.ReachyMini.return_value = mini
    if c is None:
        c = ReachyMiniController(host="10.42.0.1", connection_mode="network", media_backend="no_media")
    with (
        patch.dict(sys.modules, {"reachy_mini": mod}),
        patch.object(ReachyMiniController, "_port_open", return_value=True),
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
