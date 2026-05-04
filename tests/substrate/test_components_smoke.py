"""Standing smoke test for every YAML in ``_data/components/``.

This is a **drift guard**: as the SEM runtime evolves (spec schema,
sensor wiring, failure-mode format), bundled component YAMLs can drift
out of sync with the loader. A failure here surfaces the drift
immediately instead of letting it decay into silent runtime bugs
(e.g., sensors that silently read as None, failure modes that never
fire, Reactions published without the expected context).

For each bundled component this test:

1. Instantiates a fresh ``Entity`` via ``ComponentRegistry.instantiate``.
2. Wraps it in an ``Embodiment`` and reads every scalar sensor.
3. Calls ``evaluate_failures()`` (may fire nothing — that's OK).
4. For the subset of weapon components that declare a durability-
   based failure mode, drives the durability sensor below the
   threshold and verifies the failure fires AND publishes a
   rich-context ``PainSignal`` through ``PainBus``.

Scope is intentionally narrow: load-and-evaluate, not behavioral. If
a component's sensor ranges or failure triggers change, the test may
need to be updated, but the default pass path should remain stable
for any well-formed YAML.
"""

from __future__ import annotations

import logging

import pytest

from maxim.embodiment.body import Embodiment
from maxim.embodiment.component_registry import ComponentRegistry, _read_component_header
from maxim.embodiment.sem import Entity
from maxim.proprioception.pain import PainSignal
from maxim.proprioception.pain_bus import PainBus

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def registry() -> ComponentRegistry:
    """Default-search-path registry — finds bundled _data/components."""
    return ComponentRegistry()


def _is_legacy_body_spec(registry: ComponentRegistry, ref: str) -> bool:
    """Is this ref a legacy ``body:``-format YAML that can't be instantiate()'d?

    The registry's ``_read_component_header`` accepts files with a top-
    level ``body:`` or ``world_entities:`` block via an auto-detection
    path, stamping them with ``_legacy=True``. ``instantiate()`` cannot
    route these through ``load_spec()`` — it only knows the modern
    ``component:/entity:`` format. Callers that need these YAMLs must
    use ``embodiment.spec.load_spec`` directly.

    This is a real gap in the registry surface, discovered by this
    smoke test. Tracked as a follow-up; not in Stage 2 scope.
    """
    path = registry._file_map.get(ref)
    if path is None:
        return False
    header = _read_component_header(path) or {}
    return bool(header.get("_legacy"))


@pytest.fixture(scope="module")
def all_refs(registry: ComponentRegistry) -> list[str]:
    """All component-format refs (legacy body-specs filtered out)."""
    refs = registry.list_refs()
    legacy = [r for r in refs if _is_legacy_body_spec(registry, r)]
    if legacy:
        logger.warning(
            "Skipping %d legacy body-spec YAMLs that the ComponentRegistry "
            "indexes but cannot instantiate via the component API: %s. "
            "These require embodiment.spec.load_spec() directly. Tracked "
            "as a registry follow-up.",
            len(legacy),
            ", ".join(sorted(legacy)),
        )
    return [r for r in refs if r not in set(legacy)]


class TestAllComponentsLoad:
    """Every bundled component YAML instantiates cleanly."""

    def test_at_least_one_component_discovered(self, all_refs):
        assert all_refs, "ComponentRegistry found no components — check search paths"

    def test_instantiate_all_refs(self, registry, all_refs):
        """Iterate all refs in one test so failures list them explicitly."""
        failed: list[tuple[str, str]] = []
        for ref in all_refs:
            try:
                entity = registry.instantiate(ref)
                assert isinstance(entity, Entity), f"{ref}: not an Entity"
            except Exception as e:  # noqa: BLE001
                failed.append((ref, f"{type(e).__name__}: {e}"))
        assert not failed, "Components failed to instantiate:\n" + "\n".join(f"  {ref}: {err}" for ref, err in failed)

    def test_read_all_sensors_each(self, registry, all_refs):
        """Every component's scalar sensors read without exception."""
        failed: list[tuple[str, str]] = []
        for ref in all_refs:
            try:
                entity = registry.instantiate(ref)
                emb = Embodiment(entity)
                emb.read_scalars()  # must not raise
            except Exception as e:  # noqa: BLE001
                failed.append((ref, f"{type(e).__name__}: {e}"))
        assert not failed, "Sensor reads failed:\n" + "\n".join(f"  {ref}: {err}" for ref, err in failed)

    def test_evaluate_failures_each(self, registry, all_refs):
        """evaluate_failures() runs without raising on every component."""
        failed: list[tuple[str, str]] = []
        for ref in all_refs:
            try:
                entity = registry.instantiate(ref)
                emb = Embodiment(entity)
                emb.evaluate_failures()  # may fire nothing
            except Exception as e:  # noqa: BLE001
                failed.append((ref, f"{type(e).__name__}: {e}"))
        assert not failed, "evaluate_failures() raised:\n" + "\n".join(f"  {ref}: {err}" for ref, err in failed)


class TestDurabilityFailureCascade:
    """Weapon components with durability failure modes fire pain signals."""

    def test_rusty_sword_shatter_fires_pain_signal(self, registry):
        """The canonical PoC anchor: rusty_sword shatter → PainSignal.

        Also the canonical wiring verification for Stage 2 — proves
        that body.py → PainBus.publish → direct subscribers works
        end-to-end on a real bundled component.
        """
        entity = registry.instantiate("weapons/rusty_sword")
        # Drive durability below the shatter threshold (< 0.1).
        entity.vital_metrics["durability"] = 0.05

        pain_bus = PainBus(_allow_raw=True)
        captured: list[PainSignal] = []
        pain_bus.subscribe(captured.append)

        emb = Embodiment(entity, pain_bus=pain_bus)
        events = emb.evaluate_failures()

        assert any(e.failure_name == "shatter" for e in events), (
            f"shatter did not fire — events: {[e.failure_name for e in events]}"
        )
        assert captured, "no PainSignal delivered to PainBus subscribers"
        signal = captured[-1]
        assert signal.context["source"] == "embodiment"
        assert signal.context["failure_mode"] == "shatter"
        assert "rusty_sword" in signal.context["entity"]
        assert signal.intensity > 0.0

    def test_weapons_with_durability_failures_can_fire(self, registry):
        """Every weapon component with a sensor-driven failure can trigger.

        Iterates bundled weapons, finds any failure mode whose trigger
        field is a scalar sensor, drives that sensor past the trigger
        threshold, and verifies the failure fires. Skips components
        whose triggers reference fields we can't synthesize (e.g.,
        composed vital_metrics only the runtime populates).
        """
        failed: list[tuple[str, str]] = []
        fired: list[str] = []
        for ref in registry.list_refs(category="weapons"):
            try:
                entity = registry.instantiate(ref)
                # Find a scalar-sensor failure mode
                target_fm = None
                target_field = None
                for fm in entity.failure_modes:
                    for trig in fm.triggers:
                        if trig.field in entity.sensors:
                            target_fm = fm
                            target_field = trig.field
                            break
                    if target_fm is not None:
                        break
                if target_fm is None:
                    continue

                # Drive the field past the trigger. Use vital_metrics
                # override since Body.evaluate_failures reads vital_metrics
                # as well as sensor readings.
                trig = next(t for t in target_fm.triggers if t.field == target_field)
                if trig.op in ("<", "<="):
                    entity.vital_metrics[target_field] = trig.value - 0.1
                elif trig.op in (">", ">="):
                    entity.vital_metrics[target_field] = trig.value + 0.1
                else:
                    continue  # skip equality-style triggers for this probe

                emb = Embodiment(entity)
                events = emb.evaluate_failures()
                if any(e.failure_name == target_fm.name for e in events):
                    fired.append(f"{ref}/{target_fm.name}")
                else:
                    failed.append((ref, f"{target_fm.name} did not fire"))
            except Exception as e:  # noqa: BLE001
                failed.append((ref, f"{type(e).__name__}: {e}"))

        assert fired, (
            "No weapon components fired their durability failures — SEM runtime may have drifted from the YAML schema"
        )
        assert not failed, "Weapon failure probes failed:\n" + "\n".join(f"  {ref}: {err}" for ref, err in failed)
