- **`maxim.models.movement.motor_cortex` imports without keras** (#1647): `LayerScale`, `MotorCortex` and the
  `motor_cortex` alias are built on first access, and a missing keras surfaces there as the typed
  `OptionalDependencyError` with the `pymaxim[training]` hint, instead of a plain `ImportError` from importing the
  module. The TensorFlow GPU env defaults are now set when the model is built, not on import. This was one of the two
  causes of the nightly slow lane's red runs since 2026-11-01.
