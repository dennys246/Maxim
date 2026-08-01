"""robots.yaml drives the LIVE robot connection (2026-07-31 fix).

Pre-fix, the selfy connect path built its controller config inline and
silently ignored ``~/.maxim/robots.yaml`` — while the connect-failure
message told the operator to "set host: <ip> in robots.yaml". Pins:

- ``resolve_connection_config``: exact-id match, unambiguous-primary
  fallback, never-guess-among-unmarked-robots, declared-beats-defaults;
- ``RobotRegistry.connect_robot`` filters the free-form config dict to
  the controller constructor's accepted kwargs, so wiring-layer keys
  (``body``, ``audio_localization``) no longer TypeError the
  construction into a bogus "Failed to connect".
"""

from __future__ import annotations

from maxim.hardware.config import RobotConfig, RobotsConfig, resolve_connection_config
from maxim.hardware.controller import RobotController
from maxim.hardware.registry import RobotRegistry


def _cfg(robots: dict[str, dict]) -> RobotsConfig:
    return RobotsConfig(robots=[RobotConfig.from_dict(rid, data) for rid, data in robots.items()])


_DEFAULTS = {"robot_name": "reachy_mini", "media_backend": "default"}


class TestResolveConnectionConfig:
    def test_declared_host_reaches_the_merged_config(self):
        cfg = _cfg(
            {
                "reachy_mini": {
                    "type": "reachy_mini",
                    "primary": True,
                    "config": {"host": "10.6.0.63", "connection_mode": "network", "body": "bodies/reachy_mini"},
                }
            }
        )
        merged = resolve_connection_config(cfg, "reachy_mini", defaults=_DEFAULTS)
        assert merged["host"] == "10.6.0.63"
        assert merged["connection_mode"] == "network"
        assert merged["robot_name"] == "reachy_mini"  # default preserved

    def test_primary_fallback_when_id_differs(self):
        # The runtime name ("reachy_mini") routinely differs from the yaml
        # key ("primary") — same rule as _resolve_body_wiring.
        cfg = _cfg({"my_desk_robot": {"type": "reachy_mini", "primary": True, "config": {"host": "10.6.0.63"}}})
        merged = resolve_connection_config(cfg, "reachy_mini", defaults=_DEFAULTS)
        assert merged["host"] == "10.6.0.63"

    def test_single_robot_fallback_without_primary_flag(self):
        cfg = _cfg({"only_one": {"type": "reachy_mini", "config": {"host": "10.6.0.63"}}})
        merged = resolve_connection_config(cfg, "something_else", defaults=_DEFAULTS)
        assert merged["host"] == "10.6.0.63"

    def test_ambiguous_multi_robot_never_guesses(self):
        cfg = _cfg(
            {
                "robot_a": {"type": "reachy_mini", "config": {"host": "10.0.0.1"}},
                "robot_b": {"type": "reachy_mini", "config": {"host": "10.0.0.2"}},
            }
        )
        merged = resolve_connection_config(cfg, "reachy_mini", defaults=_DEFAULTS)
        assert "host" not in merged  # defaults only — a foreign robot's host must not leak in

    def test_declared_keys_win_over_defaults(self):
        cfg = _cfg({"r": {"type": "reachy_mini", "primary": True, "config": {"media_backend": "gstreamer"}}})
        merged = resolve_connection_config(cfg, "r", defaults=_DEFAULTS)
        assert merged["media_backend"] == "gstreamer"

    def test_empty_config_returns_defaults(self):
        merged = resolve_connection_config(RobotsConfig(), "reachy_mini", defaults=_DEFAULTS)
        assert merged == _DEFAULTS


class _RecordingController(RobotController):
    """Minimal controller recording its constructor kwargs."""

    def __init__(self, robot_id: str = "rec", *, robot_name: str = "x", host: str | None = None) -> None:
        super().__init__(robot_id)
        self.received_host = host
        self.received_robot_name = robot_name

    @property
    def robot_type(self):
        return "recording"

    def connect(self, timeout=30.0):
        from maxim.hardware.capabilities import RobotConnectionState

        self._update_state(connection_state=RobotConnectionState.CONNECTED)
        return True

    def disconnect(self):
        pass

    def goto_target(self, target):
        return True

    def look_at_pixel(self, target):
        return True

    def get_current_pose(self):
        return {}

    def wake_up(self):
        return True

    def goto_sleep(self):
        return True

    def start_recording(self):
        return True

    def stop_recording(self):
        return True

    def get_video_stream(self):
        return None

    def get_audio_stream(self):
        return None


class TestConnectRobotConfigFilter:
    def test_wiring_layer_keys_do_not_break_construction(self):
        """The full robots.yaml config dict (incl. body/audio_localization)
        must connect cleanly — pre-fix the blind splat raised TypeError and
        surfaced as a bogus 'Failed to connect'."""
        registry = RobotRegistry()
        registry.register_controller_type("recording", _RecordingController)
        controller = registry.connect_robot(
            robot_id="r1",
            robot_type="recording",
            config={
                "robot_name": "reachy_mini",
                "host": "10.6.0.63",
                "body": "bodies/reachy_mini",  # wiring-layer key
                "audio_localization": False,  # wiring-layer key
            },
        )
        assert controller is not None
        assert controller.received_host == "10.6.0.63"
        assert controller.received_robot_name == "reachy_mini"

    def test_var_kwargs_controller_receives_everything(self):
        received = {}

        class _KwController(_RecordingController):
            def __init__(self, robot_id: str = "kw", **kwargs) -> None:
                RobotController.__init__(self, robot_id)
                received.update(kwargs)

        registry = RobotRegistry()
        registry.register_controller_type("kw", _KwController)
        assert registry.connect_robot(robot_id="r2", robot_type="kw", config={"anything": 1}) is not None
        assert received == {"anything": 1}
