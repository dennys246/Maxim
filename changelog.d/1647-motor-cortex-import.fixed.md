- **`maxim.models.movement.motor_cortex` refuses typed without keras** (#1647): importing it without the `training`
  extra now raises `OptionalDependencyError` (`import_name="keras"`, `extra="training"`) with the
  `pymaxim[training]` hint, instead of a plain `ImportError`. It is still an `ImportError` subclass, so existing
  handlers catch it. The fresh-import sweep now declares each module that belongs to an extra (the two console
  modules and `motor_cortex`): such a module must import when its extra is installed, and when the extra is absent
  it may fail only with that extra's own refusal. This was one of the two causes of the nightly slow lane's red runs
  since 2026-11-01.
- **The `console` extra declares `pydantic>=2.7,<3.0`** (#1647): `console/schemas.py` imports pydantic directly and
  uses its v2 API, but the extra received it only through fastapi, whose own range still allows pydantic 1. A test
  now checks that every third-party name an extra-owned module imports at module level is a direct requirement of
  its extra.
