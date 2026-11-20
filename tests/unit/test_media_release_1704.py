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

import pytest

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


def _connect(monkeypatch, mini: MagicMock, status: dict | None) -> ReachyMiniController:
    monkeypatch.setattr(ReachyMiniController, "_daemon_status", staticmethod(lambda *a, **k: status))
    monkeypatch.setattr(ReachyMiniController, "_installed_sdk_version", staticmethod(lambda: (1, 8)))
    mod = MagicMock()
    mod.ReachyMini.return_value = mini
    c = ReachyMiniController(host="10.42.0.1", connection_mode="network", media_backend="no_media")
    with (
        patch.dict(sys.modules, {"reachy_mini": mod}),
        patch.object(ReachyMiniController, "_port_open", return_value=True),
    ):
        assert c.connect(timeout=2.0) is True
    return c


@pytest.mark.xfail(strict=True, reason="#1704 red gate: disconnect() drops the handle without re-acquiring")
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


@pytest.mark.xfail(strict=True, reason="#1704 red gate: an unknown pre-connect media state is not reported")
def test_an_unknown_pre_connect_state_is_never_guessed_and_is_reported(monkeypatch, caplog):
    mini = _sdk_mini(released_by_connect=True)
    with caplog.at_level(logging.WARNING, logger="maxim.hardware.reachy.controller"):
        c = _connect(monkeypatch, mini, None)
        c.disconnect()
    assert mini.acquire_media.call_count == 0
    assert any("media" in r.getMessage() and "acquire" in r.getMessage() for r in caplog.records)


@pytest.mark.xfail(strict=True, reason="#1704 red gate: a failed re-acquire is not reported")
def test_a_failed_reacquire_is_reported_and_the_disconnect_still_completes(monkeypatch, caplog):
    mini = _sdk_mini(released_by_connect=True, acquire_works=False)
    c = _connect(monkeypatch, mini, {"state": "running", "media_released": False})
    with caplog.at_level(logging.WARNING, logger="maxim.hardware.reachy.controller"):
        c.disconnect()
    assert mini.acquire_media.call_count == 1
    assert c._mini is None
    assert any("/api/media/acquire" in r.getMessage() for r in caplog.records)
