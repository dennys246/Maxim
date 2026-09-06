"""Guards for 1.1.4 PR 2 — the `modality:` sensor declaration and the world channel.

Pins: the schema passthrough; declaration-driven channel membership (world
purely declared, audio = legacy tuple ∪ declared); byte-identical behavior
for bodies that declare nothing (every pre-PR-2 body); the value/range
lockstep for declared sensors under both place-code arms; the modulator
sub-sensor `drive: null` crash fix; and `recommend_action` accepting a
three-modality cluster set with the world cluster's bias in the sum.
"""

from __future__ import annotations

import pytest

from maxim.embodiment.sensory_streams import AUDIO_TAG, WORLD_TAG
from maxim.embodiment.spec import _parse_entity
from maxim.runtime.agent_loop import (
    _SUBSTRATE_CHANNELS,
    _read_exteroceptive_ranges,
    _read_exteroceptive_states,
    _read_world_ranges,
    _read_world_states,
)


class _FakeExecutor:
    def __init__(self, root):
        class _E:
            pass

        self.embodiment = _E()
        self.embodiment.root = root


def _player_body():
    return _parse_entity(
        {
            "name": "player",
            "sensors": {
                "health": {"range": [0, 20], "initial": 20, "modality": "world"},
                "light_level": {"range": [0, 15], "initial": 7, "modality": "world"},
                "hunger": {
                    "range": [0, 1],
                    "initial": 0.2,
                    "drive": {"drift_mode": "entropic", "drift_direction": "up", "deprivation_threshold": 0.7},
                },
                "wind_direction": {"range": [-1, 1], "initial": 0.4, "modality": "audio"},
            },
        }
    )


class TestModalityDeclaration:
    def test_schema_carries_the_declared_modality(self):
        body = _player_body()
        assert body.sensors["health"].reading_schema["modality"] == "world"
        assert "modality" not in body.sensors["hunger"].reading_schema

    def test_world_channel_reads_only_declared_world_sensors(self):
        ex = _FakeExecutor(_player_body())
        vals = _read_world_states(ex)
        assert vals == {"health": 20.0, "light_level": 7.0}
        assert _read_world_ranges(ex) == {"health": (0.0, 20.0), "light_level": (0.0, 15.0)}

    def test_declared_audio_sensor_joins_the_audio_channel_raw(self):
        ex = _FakeExecutor(_player_body())
        vals = _read_exteroceptive_states(ex)
        assert vals == {"wind_direction": 0.4}
        assert _read_exteroceptive_ranges(ex) == {"wind_direction": (-1.0, 1.0)}

    def test_declared_audio_lockstep_survives_place_code(self, monkeypatch):
        """Declared audio sensors join RAW on BOTH walks when the legacy tuple
        is place-coded — a value without its range silently re-folds (P1)."""
        monkeypatch.setenv("MAXIM_PLACE_CODE_EXTEROCEPTION", "1")
        body = _parse_entity(
            {
                "name": "b",
                "sensors": {
                    "azimuth": {"range": [-1, 1], "initial": 0.5},
                    "wind_direction": {"range": [-1, 1], "initial": 0.4, "modality": "audio"},
                },
            }
        )
        ex = _FakeExecutor(body)
        vals = _read_exteroceptive_states(ex)
        ranges = _read_exteroceptive_ranges(ex)
        assert "wind_direction" in vals and "wind_direction" in ranges
        assert any(k.startswith("azdir_azimuth_") for k in vals)
        # lockstep: every valued sensor has a range under place code
        assert set(vals) <= set(ranges)

    def test_undeclared_bodies_are_byte_identical(self):
        """Every bundled body declares nothing → world channel empty, audio
        channel exactly the legacy tuple behavior (no declared merge)."""
        from maxim.embodiment.component_registry import ComponentRegistry

        registry = ComponentRegistry()
        checked = 0
        for ref in registry.list_refs("bodies"):
            if ref in (
                "bodies/minecraft_player",
                "bodies/minecraft_bench",
                "bodies/minecraft_bench_satiated",
            ):
                # The legitimate world-channel feeders, each with its case
                # argued here per this gate's own rule:
                # - minecraft_player: THE declaring body (1.1.4 PR 3);
                #   behavior pinned by test_minecraft_seam.py.
                # - minecraft_bench (+ its satiated twin, which inherits
                #   the world sensors via `extends` — the registry resolves
                #   the merge, so both appear here): the Exp 56 frozen
                #   benchmark apparatus
                #   (exp56_four_arm_sharing_preregistration.md §Apparatus)
                #   — a six-sensor world channel inside L11's per-channel
                #   budget, pinned by test_exp56_harness.py::TestBenchBody.
                # Any OTHER body joining this list must argue its case in
                # review.
                continue
            spec = registry.get(ref)
            body = _parse_entity(dict(spec.get("entity", spec)))
            ex = _FakeExecutor(body)
            assert _read_world_states(ex) == {}, f"{ref} unexpectedly feeds the world channel"
            from maxim.runtime.agent_loop import _read_declared_modality_states

            assert _read_declared_modality_states(ex, AUDIO_TAG) == {}, (
                f"{ref} unexpectedly declares audio sensors — audio channel content would change"
            )
            checked += 1
        assert checked >= 10, "bundled-body sweep degenerated"

    def test_channel_registry_has_three_channels_world_last(self):
        assert [ch.tag for ch in _SUBSTRATE_CHANNELS] == ["interoception", AUDIO_TAG, WORLD_TAG]


class TestModalityValidation:
    """An unknown or misplaced `modality:` must fail LOUDLY at parse time —
    a silent no-op here is a sensor belonging to no channel, indistinguishable
    from working (both review lenses, PR 2 round)."""

    def _body(self, sensor_spec):
        return {"name": "b", "sensors": {"s": sensor_spec}}

    def test_typo_raises(self):
        with pytest.raises(ValueError, match="unknown sensor modality"):
            _parse_entity(self._body({"range": [0, 1], "modality": "wolrd"}))

    def test_sensorymodality_vocabulary_is_not_declarable(self):
        with pytest.raises(ValueError, match="unknown sensor modality"):
            _parse_entity(self._body({"range": [0, 1], "modality": "sound"}))

    def test_interoception_is_rejected_with_the_drive_hint(self):
        with pytest.raises(ValueError, match="drive"):
            _parse_entity(self._body({"range": [0, 1], "modality": "interoception"}))

    def test_modality_on_a_modulator_sub_sensor_is_rejected(self):
        with pytest.raises(ValueError, match="entity-level"):
            _parse_entity(
                {
                    "name": "b",
                    "modulators": {
                        "arms": {"sensors": {"t": {"range": [0, 1], "modality": "world"}}},
                    },
                }
            )

    def test_modality_null_means_undeclared(self):
        body = _parse_entity(self._body({"range": [0, 1], "modality": None}))
        assert "modality" not in body.sensors["s"].reading_schema


class TestChildEntityDeclaredAudio:
    def test_child_sensor_sharing_a_tuple_name_still_joins_when_root_lacks_it(self):
        """Dedupe is by actual legacy emission, not tuple name: a CHILD
        entity's `azimuth` declaring audio joins the channel when the root
        has no azimuth sensor (executor-lens review, PR 2 round)."""
        body = _parse_entity(
            {
                "name": "root",
                "sensors": {},
                "children": [
                    {
                        "name": "ear",
                        "sensors": {"azimuth": {"range": [-1, 1], "initial": 0.3, "modality": "audio"}},
                    }
                ],
            }
        )
        ex = _FakeExecutor(body)
        assert _read_exteroceptive_states(ex) == {"azimuth": 0.3}
        assert _read_exteroceptive_ranges(ex) == {"azimuth": (-1.0, 1.0)}


class TestModulatorSubSensorDriveNull:
    def test_drive_null_on_a_modulator_sub_sensor_parses(self):
        """The `extends`-child drive-removal idiom, sub-sensor edition —
        crashed with `'NoneType' object has no attribute 'get'` before the
        PR 2 fix (`\"drive\" in ms_spec` reached `_parse_drive_spec(None)`)."""
        body = _parse_entity(
            {
                "name": "b",
                "modulators": {
                    "arms": {
                        "sensors": {
                            "thermal": {"range": [0, 1], "initial": 0.5, "drive": None},
                        },
                    },
                },
            }
        )
        assert "arms.thermal" not in body.drive_specs


class TestThreeChannelSelection:
    def test_recommend_action_sums_the_world_cluster_bias(self):
        """A three-modality cluster set flows through `recommend_action` and
        the world cluster's learned bias reaches the score sum."""
        from maxim.decisions.nac import NAc
        from maxim.runtime.tool_dispatch import build_tool_signature

        nac = NAc()
        agent = "a"
        sig_x = build_tool_signature("tool_x", None)
        for _ in range(12):
            nac.update_cluster_reward(
                agent_id=agent,
                cluster_id="world-cluster-1",
                tool_signature=sig_x,
                reward=1.0,
                source="operant",
            )
        with_world = nac.recommend_action(
            agent_id=agent,
            available_tools=["tool_x", "tool_y"],
            current_clusters={"interoception": "ic-1", WORLD_TAG: "world-cluster-1"},
            min_confidence=0.0,
        )
        assert with_world is not None and with_world["tool_name"] == "tool_x", (
            "world cluster bias did not reach the recommend_action score sum"
        )
        without_world = nac.recommend_action(
            agent_id=agent,
            available_tools=["tool_x", "tool_y"],
            current_clusters={"interoception": "ic-1"},
            min_confidence=0.0,
        )
        # without the world cluster there is no learned signal at all — either
        # nothing is recommended or tool_x is not preferred BY the bias
        if without_world is not None and without_world["tool_name"] == "tool_x":
            assert with_world.get("confidence", 1.0) >= without_world.get("confidence", 0.0)
