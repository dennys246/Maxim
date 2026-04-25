"""Embodiment resolution — LOD (Level of Detail) for SEM entities.

Controls how much sensor detail the agent sees based on the
``--deep-embodiment`` flag:

- **Level 2 (default):** Each modulator exposes ONE ``integrity`` value
  (aggregated from sub-sensors).  Damage targets the component as a
  whole.  Suitable for 7-14B models.

- **Level 3 (``--deep-embodiment``):** Sub-sensors individually exposed.
  Damage types route through ``damage_affinities``.  Suitable for 30B+
  models that can reason about 30+ sensors simultaneously.

The entity spec is always parsed at full fidelity (level 3).  This
module controls the *runtime view* — which sensors the agent sees in
tool descriptions, prompt context, and sensor readings.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from maxim.embodiment.sem import Entity

log = logging.getLogger(__name__)

# Cache the resolved depth to avoid repeated env lookups
_resolved_depth: int | None = None


def get_embodiment_depth(args: Any = None) -> int:
    """Resolve the embodiment depth level (2 or 3).

    Checks (in order):
    1. ``args.deep_embodiment`` (CLI flag)
    2. ``MAXIM_DEEP_EMBODIMENT`` env var
    3. Default: 2
    """
    global _resolved_depth
    if _resolved_depth is not None:
        return _resolved_depth

    depth = 2

    if args is not None and getattr(args, "deep_embodiment", False):
        depth = 3
    elif os.environ.get("MAXIM_DEEP_EMBODIMENT", "").strip() in ("1", "true", "yes"):
        depth = 3

    _resolved_depth = depth
    if depth == 3:
        log.info("Deep embodiment enabled (level 3): sub-sensors + damage affinities active")
    return depth


def reset_depth() -> None:
    """Reset cached depth (for testing)."""
    global _resolved_depth
    _resolved_depth = None


def get_visible_sensors(entity: Entity, depth: int = 2) -> dict[str, float]:
    """Return the sensor readings visible to the agent at the given depth.

    Level 2: entity-level sensors + per-modulator ``integrity`` (single value).
    Level 3: entity-level sensors + all modulator sub-sensors.
    """
    readings: dict[str, float] = {}

    # Entity-level sensors (always visible)
    for sname, sval in entity.vital_metrics.items():
        # Skip computed modulator integrity keys at level 2
        # (they're exposed as named integrity values below)
        if "." in sname and depth < 3:
            continue
        readings[sname] = sval

    # Per-modulator sensors
    for mod_name, mod in entity.modulators.items():
        if not hasattr(mod, "vital_metrics") or not mod.vital_metrics:
            continue

        if depth >= 3:
            # Level 3: expose all sub-sensors
            for ms_name, ms_val in mod.vital_metrics.items():
                readings[f"{mod_name}.{ms_name}"] = ms_val
        else:
            # Level 2: expose only aggregated integrity
            if hasattr(mod, "compute_integrity"):
                readings[f"{mod_name}.integrity"] = mod.compute_integrity()

    return readings


def format_sensor_summary(entity: Entity, depth: int = 2) -> str:
    """Format a human-readable sensor summary for prompt injection.

    Level 2: "wing: 85% | torso: 95% | health: 90%"
    Level 3: "wing.membrane: 80% | wing.bone: 90% | wing.joint: 85% | ..."
    """
    sensors = get_visible_sensors(entity, depth)
    if not sensors:
        return ""

    parts = []
    for name, val in sorted(sensors.items()):
        parts.append(f"{name}: {val:.0%}")
    return " | ".join(parts)
