- **`SpecModulator.apply_damage` raises instead of returning on a part with no sub-sensors** (#1719): a direct
  caller that read the old `1.0` now gets `maxim.exceptions.DamageRefused` (also a `ValueError`; `.modulator`
  names the part). To migrate, check `bool(part.vital_metrics)` before calling, or catch `DamageRefused` and
  report it; do not turn it back into a default integrity.
