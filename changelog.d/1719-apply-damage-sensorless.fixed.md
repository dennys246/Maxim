- **Damage aimed at a body part with no sub-sensors is refused, not dropped** (#1719):
  `SpecModulator.apply_damage` used to return `1.0` on such a part and apply nothing, with no log or error, so a
  host calling it directly on a flat body saw the damage vanish. It now raises `maxim.exceptions.DamageRefused`
  (a `MaximError` and a `ValueError`, carrying `.modulator`). `damage_component` is unchanged: it already refused
  such parts (#873). Hosts that call `apply_damage` directly should give the part sub-sensors, or catch
  `DamageRefused`. `compute_integrity()` still reads `1.0` for capability-only parts.
