"""world_set_axis ownership guard (reflex canonicalization, structural half).

roadmap_1_1_to_1_3.md §"Reflex canonicalization": when a LIVE measurement
stream owns a sensor (``Embodiment.live_world_set_sensors``), an anonymous
``world_set_axis`` write must be REFUSED — a modeled value would fabricate a
reading the next real measurement reverts (the phantom-credit / ``head=None``
failure class). The check lives inside ``world_set_axis`` itself, with an
explicit ``owner=`` opt-out for the legitimate live writers (DoA feed, motor
readback), so forgetting the guard is a refusal rather than a silent
fabrication.
"""

from __future__ import annotations

import pytest

from maxim.embodiment import audio_localization
from maxim.embodiment.audio_localization import world_set_axis, world_set_azimuth


class _Root:
    def __init__(self, sensors: dict, vital: dict) -> None:
        self.sensors = sensors
        self.vital_metrics = vital


class _Emb:
    def __init__(self, root, live_owned: set[str] | None = None) -> None:
        self.root = root
        self.live_world_set_sensors = live_owned if live_owned is not None else set()


@pytest.fixture(autouse=True)
def _reset_refusal_warn_dedup():
    audio_localization._OWNERSHIP_REFUSAL_WARNED.clear()
    yield
    audio_localization._OWNERSHIP_REFUSAL_WARNED.clear()


def _emb(live_owned: set[str] | None = None) -> _Emb:
    return _Emb(_Root(sensors={"azimuth": object()}, vital={"azimuth": 0.0}), live_owned)


def test_anonymous_write_to_live_owned_sensor_is_refused():
    emb = _emb(live_owned={"azimuth"})
    assert world_set_azimuth(emb, -0.6) is False
    assert emb.root.vital_metrics["azimuth"] == 0.0


def test_declared_owner_writes_live_owned_sensor():
    emb = _emb(live_owned={"azimuth"})
    assert world_set_azimuth(emb, -0.6, owner="doa_feed") is True
    assert emb.root.vital_metrics["azimuth"] == -0.6


def test_unowned_sensor_keeps_legacy_anonymous_write():
    """Sim / scripted-substrate bodies (empty live set) are unaffected."""
    emb = _emb(live_owned=set())
    assert world_set_azimuth(emb, 0.4) is True
    assert emb.root.vital_metrics["azimuth"] == 0.4


def test_body_without_live_set_attribute_keeps_legacy_write():
    class _Bare:
        def __init__(self, root) -> None:
            self.root = root

    emb = _Bare(_Root(sensors={"azimuth": object()}, vital={"azimuth": 0.0}))
    assert world_set_azimuth(emb, 0.4) is True


def test_refusal_warns_once_per_sensor(caplog):
    emb = _emb(live_owned={"azimuth"})
    with caplog.at_level("WARNING", logger="maxim.embodiment.audio_localization"):
        world_set_azimuth(emb, -0.6)
        world_set_azimuth(emb, 0.3)
    warnings = [r for r in caplog.records if "refused anonymous write" in r.getMessage()]
    assert len(warnings) == 1


def test_owner_gate_applies_to_generic_axis():
    root = _Root(sensors={"head_yaw": object()}, vital={"head_yaw": 0.0})
    emb = _Emb(root, live_owned={"head_yaw"})
    assert world_set_axis(emb, "head_yaw", 10.0) is False
    assert world_set_axis(emb, "head_yaw", 10.0, owner="motor_readback") is True
    assert root.vital_metrics["head_yaw"] == 10.0
