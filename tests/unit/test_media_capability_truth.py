"""Capability truth for media flags (2026-08-01 live-smoke fold).

Under ``media_backend: no_media`` the Reachy SDK's media manager exists
with ``camera=None`` / ``audio=None`` — but the runtime hardcoded
``has_vision=True`` / ``has_audio=True`` for any connected robot, so the
capture loops polled absent devices at loop rate and the SDK logged
"Camera/Audio system is not initialized." dozens of times per second.

Pins:

- ``derive_media_capabilities`` answers from the ACTUAL media state and
  only downgrades on positive evidence (mocks / un-introspectable robots
  keep True);
- ``look()`` / ``listen()`` return None without touching the SDK when the
  capability is absent.
"""

from __future__ import annotations

import logging

from maxim.runtime.capabilities import derive_media_capabilities


class _Media:
    def __init__(self, camera=None, audio=None):
        self.camera = camera
        self.audio = audio


class _Mini:
    def __init__(self, media):
        self.media = media


class _Robot:
    def __init__(self, mini):
        self.mini = mini


class TestDeriveMediaCapabilities:
    def test_no_media_backend_downgrades_both(self):
        robot = _Robot(_Mini(_Media(camera=None, audio=None)))
        assert derive_media_capabilities(robot) == (False, False)

    def test_live_devices_stay_true(self):
        robot = _Robot(_Mini(_Media(camera=object(), audio=object())))
        assert derive_media_capabilities(robot) == (True, True)

    def test_camera_only(self):
        robot = _Robot(_Mini(_Media(camera=object(), audio=None)))
        assert derive_media_capabilities(robot) == (True, False)

    def test_unintrospectable_robot_keeps_permissive_true(self):
        # SimulatedController mocks / third-party controllers without a
        # .mini.media chain must see NO behavior change.
        class _NoMini:
            pass

        assert derive_media_capabilities(_NoMini()) == (True, True)
        assert derive_media_capabilities(_Robot(mini=None)) == (True, True)

    def test_media_without_device_attrs_keeps_true(self):
        class _OpaqueMedia:
            pass

        assert derive_media_capabilities(_Robot(_Mini(_OpaqueMedia()))) == (True, True)


class _ExplodingMedia:
    """Any SDK touch fails the test — the gate must fire first."""

    def __getattr__(self, name):
        raise AssertionError(f"SDK media touched ({name}) despite absent capability")


class _Caps:
    def __init__(self, has_vision, has_audio):
        self.has_vision = has_vision
        self.has_audio = has_audio


def _stub_media_loop(has_vision, has_audio):
    from maxim.embodied_runtime.media_loop import MediaLoopMixin

    class _Stub(MediaLoopMixin):
        def __init__(self):
            self.mini = _Mini(_ExplodingMedia())
            self._capabilities = _Caps(has_vision, has_audio)
            self.log = logging.getLogger("test-media-cap")

    return _Stub()


class TestLookListenCapabilityGates:
    def test_look_returns_none_without_touching_sdk(self):
        stub = _stub_media_loop(has_vision=False, has_audio=True)
        assert stub.look(show=False) is None

    def test_listen_returns_none_without_touching_sdk(self):
        stub = _stub_media_loop(has_vision=True, has_audio=False)
        assert stub.listen() is None
