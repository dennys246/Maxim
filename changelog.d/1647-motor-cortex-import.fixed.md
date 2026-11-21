- **`maxim.models.movement.motor_cortex` refuses typed without keras** (#1647): importing it without the `training`
  extra now raises `OptionalDependencyError` (`import_name="keras"`, `extra="training"`) with the
  `pymaxim[training]` hint, instead of a plain `ImportError`. It is still an `ImportError` subclass, so existing
  handlers catch it. The fresh-import sweep now declares each module that belongs to an extra (the two console
  modules and `motor_cortex`): such a module must import when its extra is installed, and when the extra is absent
  it may fail only with that extra's own refusal. This was one of the two causes of the nightly slow lane's red runs
  since 2026-11-01.
