"""Issue #783 — `WaterTrial.check_fingerprint` must see the live encoding equation.

The frozen-apparatus guard for Exp 60/61/R3/62 read `SensorEncoderConfig().pattern_threshold` (a
fresh default, not the encoder the trial books through), omitted `gain_exponent`/`gain_modalities`
(the world channel's encoding equation), and checked 3 of the body's 17 declared world ranges —
excluding `light_level` and `time_of_day`, which carry most of the gain mass. Each live mutation
below PASSED the old guard; each must now refuse. The identity arm is the a-priori anchor: the
offline agent IS the shipped body at shipped defaults, so it must pass with zero drift.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import survival_world.water_trial as WT  # noqa: E402
from survival_world.exp60_run import FROZEN as FROZEN60  # noqa: E402
from survival_world.scripted_water import ScriptedWaterBridge  # noqa: E402


@pytest.fixture
def trial(tmp_path: Path):
    from maxim.simulation.minecraft_harness import build_minecraft_aut
    from survival_world.common import make_fresh_encoder

    srv = ScriptedWaterBridge(shore={"x": 10.0, "y": 64.0, "z": 10.0}, submerged={"x": 10.0, "y": 60.0, "z": 20.0})
    aut = build_minecraft_aut(
        agent_id="enc_fp", bridge_port=srv.port, persistence_dir=str(tmp_path), entity_ref="bodies/minecraft_player"
    )
    return WT.WaterTrial(
        aut=aut,
        rcon=None,
        username="maxim",
        geom={"shore": [10.0, 64.0, 10.0], "submerged": [10.0, 60.0, 20.0]},
        frozen=FROZEN60,
        probe_cap_s=3.0,
        train_cap_s=8.0,
        persistence_dir=tmp_path,
        agent_id="enc_fp",
        encoder=make_fresh_encoder(aut),
    )


def _refusal(trial) -> str:
    with pytest.raises(WT.Refusal) as exc:
        trial.check_fingerprint(FROZEN60["usable_oxygen_max"])
    return str(exc.value)


# ── the identity anchor ──


def test_shipped_body_at_defaults_passes_and_records_its_identity(trial) -> None:
    live = trial.check_fingerprint(FROZEN60["usable_oxygen_max"])
    assert live["encoding"] == WT.APPARATUS_ENCODING
    assert len(live["encoding"]["world_ranges"]) == 17


def test_frozen_encoding_agrees_with_exp60s_three_ranges() -> None:
    """The new constant must not contradict the record it widens."""
    for k, (lo, hi) in FROZEN60["fingerprint"]["sensor_ranges"].items():
        assert WT.APPARATUS_ENCODING["world_ranges"][k] == {"lo": lo, "hi": hi}
    assert (
        WT.APPARATUS_ENCODING["encoder_config"]["pattern_threshold"]
        == FROZEN60["fingerprint"]["encoder_pattern_threshold"]
    )


# ── live mutations: each passed the pre-#783 guard ──


def test_harness_encoder_gain_exponent_refuses(trial) -> None:
    trial.encoder.config.gain_exponent = 2.0
    assert "encoder_config.gain_exponent" in _refusal(trial)


def test_harness_encoder_gain_modalities_refuses(trial) -> None:
    trial.encoder.config.gain_modalities = frozenset({"world", "audio"})
    assert "encoder_config.gain_modalities" in _refusal(trial)


def test_harness_encoder_pattern_threshold_is_read_live(trial) -> None:
    """The old read was a fresh default, blind to the encoder actually in use."""
    trial.encoder.config.pattern_threshold = 0.9
    assert "encoder_pattern_threshold" in _refusal(trial)


def test_loop_encoder_default_change_refuses(trial, monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe loop default-constructs its encoder; a changed class default re-keys it."""
    import maxim.similarity.encoder as enc

    @dataclasses.dataclass
    class Shifted(enc.SensorEncoderConfig):
        gain_exponent: float = 2.0

    monkeypatch.setattr(enc, "SensorEncoderConfig", Shifted)
    assert "loop.encoder_config.gain_exponent" in _refusal(trial)


@pytest.mark.parametrize("sensor", ["light_level", "time_of_day"])
def test_rerange_of_a_high_mass_sensor_refuses(trial, monkeypatch: pytest.MonkeyPatch, sensor: str) -> None:
    """Re-ranging light_level to [-15, 15] moves shore-vs-submerged 0.7874 → 0.5832 (#783)."""
    import maxim.runtime.agent_loop as al

    real = al._read_world_ranges

    def reranged(executor):
        r = dict(real(executor))
        lo, hi = r[sensor]
        r[sensor] = (-hi, hi)
        return r

    monkeypatch.setattr(al, "_read_world_ranges", reranged)
    assert f"world_ranges.{sensor}" in _refusal(trial)


# ── pure ──


def test_identity_enumerates_every_config_field() -> None:
    """A field added to the config later must join the identity, and so drift against a frozen one."""
    from maxim.similarity.encoder import SensorEncoderConfig

    @dataclasses.dataclass
    class Grown(SensorEncoderConfig):
        new_knob: float = 1.0

    ident = WT.encoding_identity(Grown(), {})
    assert "new_knob" in ident["encoder_config"]
    assert WT.encoding_drift(ident, {**WT.APPARATUS_ENCODING, "world_ranges": {}}) == ["encoder_config.new_knob"]


def test_reversed_range_and_roster_change_are_drift() -> None:
    frozen = WT.APPARATUS_ENCODING
    ranges = {k: (v["lo"], v["hi"]) for k, v in frozen["world_ranges"].items()}
    cfg_live = dict(frozen["encoder_config"])
    base = {"encoder_config": cfg_live, "world_ranges": {k: {"lo": a, "hi": b} for k, (a, b) in ranges.items()}}
    assert WT.encoding_drift(base, frozen) == []
    flipped = dict(base["world_ranges"], oxygen={"lo": 40.0, "hi": 0.0})
    assert WT.encoding_drift({**base, "world_ranges": flipped}, frozen) == ["world_ranges.oxygen"]
    dropped = {k: v for k, v in base["world_ranges"].items() if k != "time_of_day"}
    assert WT.encoding_drift({**base, "world_ranges": dropped}, frozen) == ["world_ranges.time_of_day"]
