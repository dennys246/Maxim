"""#1125: drive-value RECORD readers must find a modulator drive where it lives.

The real ``bodies/infant_humanoid`` declares ``arms.thermal``, ``arms.pressure`` and
``head.thermal`` as ROOT ``drive_specs``, but their values live on the modulators
(``root.modulators["arms"].vital_metrics["thermal"]``). Two readers look a drive up as
``root.vital_metrics[name]`` and so never find them:

- ``Executor._drive_pressure_snapshot``, the memory-strength encoding record (Phase 2b-ii):
  an infant whose arm is burning records no arm pressure at all;
- ``Embodiment.body_state_summary``, the Body State text the LLM reads behind
  ``MAXIM_ENABLE_BODY_STATE_PROMPT`` (default off): the burning arm is not in it.

Scope (owner decision 2026-10-08): RECORDS only. The credit readers in ``tool_bridge``
(``pre_values`` / ``_drive_potential_diff`` / ``_drive_progress_by_drive``) are equally blind,
but resolving them inverts Exp 42's harm credit at arm saturation; that is #1161.

Every test runs the REAL body. The expected numbers are written out by hand from the infant's
YAML, not computed through the code under test.
"""

from __future__ import annotations

from typing import Any

import pytest

from maxim.embodiment.body import Embodiment
from maxim.embodiment.component_registry import ComponentRegistry
from maxim.runtime.executor import Executor
from maxim.tools.base import Tool, ToolOutput
from maxim.tools.registry import ToolRegistry

_QUALIFIED = ("arms.pressure", "arms.thermal", "head.thermal")
_ROOT = ("azimuth", "core_temperature", "hunger", "thirst")


class _Noop(Tool):
    name = "noop"
    description = "does nothing to the body"
    input_schema: dict[str, Any] = {}

    def execute(self, **kwargs: Any) -> ToolOutput:
        return ToolOutput(success=True)


def _infant() -> tuple[Any, Embodiment]:
    """The real infant, every drive off its comfort so each one has something to report."""
    root = ComponentRegistry().instantiate("bodies/infant_humanoid")
    emb = Embodiment(root=root)
    root.vital_metrics.update(hunger=0.75, thirst=0.4, core_temperature=-0.4, azimuth=0.3)
    root.modulators["arms"].vital_metrics.update(thermal=0.8, pressure=0.9)
    root.modulators["head"].vital_metrics["thermal"] = -0.7
    return root, emb


def _pressure_record(emb: Embodiment) -> dict[str, float]:
    """What the production entry stamps: ``Executor.execute`` -> ``drive_pressure_before``."""
    registry = ToolRegistry()
    registry.register(_Noop())
    executor = Executor(tool_registry=registry)
    executor.embodiment = emb
    result = executor.execute({"tool_name": "noop", "params": {}})
    assert result.success
    return dict(result.drive_pressure_before or ())


def _summary_sensors(emb: Embodiment) -> dict[str, dict[str, Any]]:
    (body,) = [e for e in emb.body_state_summary() if e["entity"] == "infant_humanoid"]
    return body["sensors"]


# ── red gates: the record readers are blind to modulator drives ─────────────────────────────────


@pytest.mark.xfail(
    strict=True, reason="#1125: the pressure record reads root.vital_metrics, so modulator drives are absent"
)
def test_the_pressure_record_carries_every_modulator_drive():
    _, emb = _infant()
    record = _pressure_record(emb)
    # Homeostatic pressure = (|v - set_point| - band) / (span - band), from the YAML:
    # arms.thermal 0.8, band 0.5, span 1 -> 0.3/0.5; arms.pressure 0.9, band 0.6 -> 0.3/0.4;
    # head.thermal -0.7, band 0.5 -> 0.2/0.5.
    assert {name: record.get(name) for name in _QUALIFIED} == {
        "arms.thermal": pytest.approx(0.6),
        "arms.pressure": pytest.approx(0.75),
        "head.thermal": pytest.approx(0.4),
    }


@pytest.mark.xfail(
    strict=True, reason="#1125: the pressure record reads root.vital_metrics, so modulator drives are absent"
)
def test_the_pressure_record_follows_the_arm_as_it_cools():
    root, emb = _infant()
    hot = _pressure_record(emb)["arms.thermal"]
    root.modulators["arms"].vital_metrics["thermal"] = 0.3  # back inside the comfort band
    assert hot > 0.0
    assert _pressure_record(emb)["arms.thermal"] == 0.0


@pytest.mark.xfail(strict=True, reason="#1125: body_state_summary skips every dotted drive name")
def test_the_body_state_summary_shows_every_modulator_drive():
    _, emb = _infant()
    sensors = _summary_sensors(emb)
    assert {name: sensors.get(name) for name in _QUALIFIED} == {
        "arms.thermal": {"value": 0.8, "unit": "celsius_norm", "drive": "outside comfort band, discomfort 0.12"},
        "arms.pressure": {"value": 0.9, "unit": "ratio", "drive": "outside comfort band, discomfort 0.09"},
        "head.thermal": {"value": -0.7, "unit": "celsius_norm", "drive": "outside comfort band, discomfort 0.08"},
    }


@pytest.mark.xfail(strict=True, reason="#1125: body_state_summary skips every dotted drive name")
def test_the_burning_arm_reaches_the_prompt_text_and_the_coach():
    """Behind ``MAXIM_ENABLE_BODY_STATE_PROMPT`` the LLM reads this text, and Acting Coach Layer 4
    names the drives that need attention from it."""
    from maxim.prompts.acting_coach import _compose_drive_modulation

    _, emb = _infant()
    text = emb.format_body_state_for_prompt()
    assert "- infant_humanoid.arms.thermal: 0.8celsius_norm (DRIVE: outside comfort band, discomfort 0.12)" in text
    assert "infant_humanoid.arms.thermal" in _compose_drive_modulation(text)


# ── green: root drives read exactly as before ────────────────────────────────────────────────────


def test_root_drive_pressures_are_unchanged():
    """Pinned before and after the fix (values captured at origin/main f1c833ab)."""
    _, emb = _infant()
    record = _pressure_record(emb)
    assert {name: record.get(name) for name in _ROOT} == {
        "azimuth": pytest.approx(0.2222222222222222),
        "core_temperature": pytest.approx(0.2),
        "hunger": 1.0,
        "thirst": pytest.approx(0.3333333333333335),
    }


def test_root_drive_summary_entries_are_unchanged():
    """Pinned before and after the fix (captured at origin/main f1c833ab)."""
    _, emb = _infant()
    sensors = _summary_sensors(emb)
    assert {name: sensors[name] for name in (*_ROOT, "stamina", "visibility", "carrying_weight")} == {
        "azimuth": {"value": 0.3, "unit": "normalized", "drive": "outside comfort band, discomfort 0.06"},
        "core_temperature": {"value": -0.4, "unit": "celsius_norm", "drive": "outside comfort band, discomfort 0.23"},
        "hunger": {"value": 0.75, "unit": "ratio", "drive": "deprived, intensity 0.30"},
        "thirst": {"value": 0.4, "unit": "ratio", "drive": "rising"},
        "stamina": {"value": 1.0, "unit": "ratio"},
        "visibility": {"value": 0.8, "unit": "ratio"},
        "carrying_weight": {"value": 0.0, "unit": "ratio"},
    }


def test_the_body_state_prompt_stays_off_by_default(monkeypatch):
    """The summary reaches an LLM only behind the flag; the default production wiring is untouched."""
    from types import SimpleNamespace

    from maxim.runtime.agent_factory import _maybe_wire_body_state

    monkeypatch.delenv("MAXIM_ENABLE_BODY_STATE_PROMPT", raising=False)
    _, emb = _infant()
    instance = SimpleNamespace(embodiment=emb, memory_hub=SimpleNamespace(embodiment=None))
    _maybe_wire_body_state(instance)
    assert instance.memory_hub.embodiment is None


# ── D2: the selection readers and both records agree with the one resolver, on the real body ────


@pytest.mark.xfail(strict=True, reason="#1125: both records skip modulator drives, so they disagree with the resolver")
def test_selection_and_record_readers_agree_with_the_resolver():
    """``_read_drive_states`` and ``substrate_telemetry`` resolve qualified drives with their own walk
    (left as they are, owner decision D2); the pressure record and the Body State read through the
    embodiment's one resolution rule (``sem._resolve_sensor_slot`` / ``_read_sensor_value``). A
    future divergence between any of them fails here.

    OUTSIDE this set, on purpose:
    - ``tool_bridge``'s credit reads (``pre_values`` / ``_drive_potential_diff`` /
      ``_drive_progress_by_drive``) and ``cradle_mother``: knowingly blind to modulator drives until #1161;
    - ``Embodiment.evaluate_failures``: reads root keys FIRST, so a dotted root orphan shadows the
      modulator's value (the #874 shadow, guarded by ``test_set_entity_sensor_value_874``);
    - ``Embodiment.tick_vital_drift``: a writer with its own qualified walk;
    - ``naming_events.collect_sensor_values``: a flat walk over every sub-sensor, not only drives.
    """
    from maxim.embodiment.sem import drive_pressure
    from maxim.embodiment.tool_bridge import _resolve_sensor_slot
    from maxim.runtime.substrate_proposal import _read_drive_ranges, _read_drive_states
    from maxim.simulation.substrate_telemetry import _drive_snapshot

    root, emb = _infant()
    executor = Executor(tool_registry=ToolRegistry())
    executor.embodiment = emb
    drives = sorted(root.drive_specs)
    assert drives == sorted((*_QUALIFIED, *_ROOT))  # the infant's full roster, so nothing is skipped

    resolved: dict[str, float] = {}
    for name in drives:
        slot = _resolve_sensor_slot(root, name)
        assert slot is not None, name
        metrics, key, _, _ = slot
        resolved[name] = float(metrics[key])
    assert len(set(resolved.values())) == len(drives)  # distinct values, so a crossed wire cannot agree

    states = _read_drive_states(executor)
    telemetry = _drive_snapshot(executor)["drives"]
    summary = _summary_sensors(emb)
    ranges = _read_drive_ranges(executor)
    record = _pressure_record(emb)
    for name in drives:
        assert states[name] == resolved[name], name
        assert telemetry[name] == resolved[name], name
        assert summary[name]["value"] == resolved[name], name
        expected_pressure = drive_pressure(root.drive_specs[name], resolved[name], *ranges[name])
        assert record[name] == pytest.approx(expected_pressure), name
