"""Minecraft world backend — SEM affordances become real game actions (1.1.4 PR 3).

The world-seam counterpart of ``hardware/reachy/motor_backend.py``, copying
its contract exactly (`docs/plans/world_seam_1_1_4.md` §PR 3; the plan's
"copy the Reachy pattern" instruction):

* ``attach_backends``-shaped factory (:func:`minecraft_modulator_factory`)
  threaded through ``build_executor(modulator_factory=)`` — never attached
  post-hoc (push-silent-no-ops-into-types: a forgotten attach is a virtual
  world that LIES success).
* **Honesty contract:** an affordance succeeds only when the bridge confirms
  it (``action_result.ok``); a timeout or refused action is a REAL failure
  (``ModulatorResult(success=False)``) — unknown is not success, and an
  unconfirmable effect must not book one.
* **World-owned sensors:** the backend declares :attr:`world_owned_sensors`
  (unioned into ``Embodiment.live_world_set_sensors`` by ``build_executor``),
  and writes the game's MEASURED post-action state back through the
  canonical ``world_set_axis(owner="minecraft_bridge")`` — the modeled
  ``self_effect`` on those keys is filtered, so declarative deltas serve
  stub/sim runs and the game's truth is the single writer on live.
* Construction is I/O-free (the Exp 54 offline-factory discipline): talk to
  the bridge in ``execute()``/``sync_world_sensors()``, never in
  ``__init__``.

Continuous sensor flow: :meth:`sync_world_sensors` pulls the client's latest
snapshot into the entity's declared world sensors — call it per tick (the
PR 4 harness's loop hook) and after every action (done internally). Only
sensors the body DECLARES are written; unknown snapshot keys are ignored
(the body YAML is the contract for what this agent can sense).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_OWNER = "minecraft_bridge"


class MinecraftWorldBackend:
    """Modulator backend mapping SEM affordances to bridge actions."""

    def __init__(
        self,
        *,
        client: Any,
        entity: Any = None,
        modulator_name: str,
        entity_name: str,
        world_sensors: tuple[str, ...] = (),
    ) -> None:
        self._client = client
        self._entity = entity
        self._modulator_name = modulator_name
        self._entity_name = entity_name
        self._embodiment: Any = None  # bound by build_executor post-construction
        self.world_owned_sensors: tuple[str, ...] = tuple(world_sensors)

    def bind_embodiment(self, embodiment: Any) -> None:
        """Called by ``build_executor`` after the Embodiment wrapper exists."""
        self._embodiment = embodiment

    # ── sensors ──────────────────────────────────────────────────────────

    def sync_world_sensors(self) -> int:
        """Write the client's latest snapshot into the entity's declared
        world sensors via ``world_set_axis``. Returns how many were written.
        Fail-soft per sensor; an unbound embodiment writes nothing."""
        if self._embodiment is None or self._client is None:
            return 0
        state = {}
        try:
            state = self._client.latest_state() or {}
        except Exception:
            logger.debug("minecraft backend: latest_state read failed", exc_info=True)
            return 0
        from maxim.embodiment.audio_localization import world_set_axis

        written = 0
        for name in self.world_owned_sensors:
            if name not in state:
                continue
            try:
                if world_set_axis(self._embodiment, name, float(state[name]), owner=_OWNER):
                    written += 1
            except Exception:
                logger.debug("minecraft backend: world-set of %r failed", name, exc_info=True)
        return written

    # ── actions ──────────────────────────────────────────────────────────

    def execute(self, affordance: str, params: dict[str, Any]) -> Any:
        from maxim.embodiment.sem import ModulatorResult

        def _result(**kw: Any) -> Any:
            return ModulatorResult(
                modulator_name=self._modulator_name,
                entity_name=self._entity_name,
                affordance=affordance,
                params=params,
                **kw,
            )

        client = self._client
        if client is None:
            return _result(success=False, error="Minecraft bridge client not connected")
        try:
            result = client.call_action(affordance, params)
        except Exception as exc:
            return _result(success=False, error=f"bridge action failed: {exc}")
        ok = bool(result.get("ok"))
        # Post-action truth: the action_result carries a fresh snapshot; the
        # client has already absorbed it, so one sync writes the measured
        # world into the body regardless of ok/fail (a failed mine attempt
        # still cost time; the world may have moved).
        self.sync_world_sensors()
        if not ok:
            return _result(success=False, error=str(result.get("detail") or "action refused by the game"))
        return _result(success=True)


def minecraft_modulator_factory(client: Any, *, world_sensors: tuple[str, ...]):
    """``attach_backends``-shaped factory for ``build_executor(modulator_factory=)``.

    Attaches a :class:`MinecraftWorldBackend` to every spec-declared
    modulator of the player entity. ``world_sensors`` names the body sensors
    the bridge's snapshots own — pass the body's declared ``modality: world``
    sensor names (the harness derives them from the parsed body so YAML and
    backend cannot drift).
    """

    def factory(entity: Any, mod_name: str, _spec_modulator: Any) -> MinecraftWorldBackend:
        return MinecraftWorldBackend(
            client=client,
            entity=entity,
            modulator_name=mod_name,
            entity_name=getattr(entity, "name", "?"),
            world_sensors=world_sensors,
        )

    return factory


def declared_world_sensor_names(entity: Any) -> tuple[str, ...]:
    """The entity tree's ``modality: world`` sensor names — the derivation
    the harness passes to :func:`minecraft_modulator_factory` so the YAML
    stays the single source of truth for what the bridge may write."""
    names: list[str] = []
    walk = getattr(entity, "walk", None)
    for ent in walk() if callable(walk) else (entity,):
        for name, sensor in (getattr(ent, "sensors", {}) or {}).items():
            schema = getattr(sensor, "reading_schema", {}) or {}
            if schema.get("modality") == "world":
                names.append(name)
    return tuple(names)
