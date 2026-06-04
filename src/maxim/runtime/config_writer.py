"""Canonical writer for ``~/.config/maxim/config.json`` (C2 IM2 fold).

This module is the ONLY sanctioned writer surface for ``config.json``.
CI grep allow-lists this file + its test file as the only callers of
``atomic_write_json`` against the config-json path, mirroring the
``mesh_setup.py`` discipline that ``write_mesh_config`` is the only
sanctioned writer for ``mesh.yml``.

Concurrency safety (I-5 fold from the pre-implementation two-lens
review):

- ``filelock.FileLock`` acquired BEFORE any disk read
- Re-read inside the lock (no caching across the lock boundary)
- Atomic write via :func:`maxim.utils.atomic_io.atomic_write_json`
- Lock released after the write completes

The lock-acquire-after-read pattern would let a stale in-memory
dataclass from process A clobber process B's just-written field when
two ``maxim config set`` invocations race from two tmux panes. The
regression-guard pattern is the same as
``tests/integration/test_drain_state_concurrent.py``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from maxim.exceptions import ConfigurationError
from maxim.runtime.config_loader import (
    CONFIG_FORMAT_VERSION,
    LaneTierConfig,
    MaximConfig,
    config_path,
    load_config,
)
from maxim.utils.atomic_io import atomic_write_json
from maxim.utils.format_version import with_format_version

logger = logging.getLogger(__name__)


def _lock_path_for(target: Path) -> str:
    """Return the FileLock path co-located with the config file."""
    return str(target) + ".lock"


def _serialize_for_json(config: MaximConfig) -> dict[str, Any]:
    """Convert :class:`MaximConfig` to the JSON-serializable shape.

    ``dataclasses.asdict`` produces almost what we want, but
    :class:`LaneTierConfig`'s ``extra`` dict must remain a top-level
    sibling of the declared fields when round-tripped (otherwise a
    future-grown field would be lost). We inline ``extra`` here.

    **Post-implementation Architecture #4 fold:** the collision check
    moved into :class:`LaneTierConfig.__post_init__`. A malformed
    LaneTierConfig now cannot reach this writer at all — the
    constructor rejects it with ``ConfigurationError``. The defensive
    assertion below guards against the impossible case where the
    invariant is bypassed (e.g., via a hypothetical caller that
    constructs the dataclass without ``__post_init__`` running — not
    possible with frozen dataclasses, but cheap to document).
    """
    payload = asdict(config)
    # Inline lanes.<tier>.extra back into the tier dict
    lanes = payload.get("lanes", {})
    for tier_name in ("large", "medium", "small"):
        tier = lanes.get(tier_name, {})
        extras = tier.pop("extra", {}) or {}
        for k, v in extras.items():
            assert k not in tier, (
                f"config_writer: LaneTierConfig.__post_init__ should have "
                f"caught the collision on lanes.{tier_name}.extra[{k!r}] "
                f"at construction. Reached the writer — invariant bypassed."
            )
            tier[k] = v
    return payload


def write_config(
    config: MaximConfig,
    path: Path | None = None,
) -> Path:
    """Atomically persist a :class:`MaximConfig` to ``config.json``.

    Holds ``filelock.FileLock`` for the duration of the write. The
    caller is responsible for having computed the *full* config they
    want persisted — use :func:`mutate_config` for the safe RMW path
    that re-reads under the lock.

    Returns the path written.
    """
    target = path if path is not None else config_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = _serialize_for_json(config)
    # Post-implementation Architecture #1 fold: route _format_version
    # stamping through the canonical with_format_version helper. The
    # pre-fold direct assignment silently bypassed CC1's fail-loud-
    # on-stale-conflict semantics (raises ValueError when payload
    # already carries a mismatched _format_version).
    payload = with_format_version(payload, CONFIG_FORMAT_VERSION)

    try:
        from filelock import FileLock
    except ImportError as e:
        raise ConfigurationError(
            "config_writer: filelock package is required for safe "
            "concurrent writes. Install via `pip install filelock`."
        ) from e

    lock = FileLock(_lock_path_for(target))
    with lock:
        atomic_write_json(str(target), payload)

    logger.info("config_writer: wrote %s", target)
    return target


def mutate_config(
    mutator: Callable[[MaximConfig], MaximConfig],
    path: Path | None = None,
) -> tuple[MaximConfig, Path]:
    """Safely apply a mutation to ``config.json`` under the file lock.

    The mutator receives the freshly-read :class:`MaximConfig` (read
    INSIDE the lock — no in-memory cache across the lock boundary) and
    must return the new config to persist. This is the canonical
    read-modify-write pattern for the ``maxim config set`` verb and
    any future caller that needs to apply a delta atomically.

    Returns ``(new_config, path_written)``.

    Per I-5 fold from the pre-implementation review: lock-acquire
    happens BEFORE the read so a concurrent writer in another process
    can't slip a write between our read and our write.
    """
    target = path if path is not None else config_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        from filelock import FileLock
    except ImportError as e:
        raise ConfigurationError(
            "config_writer: filelock package is required for safe "
            "concurrent writes. Install via `pip install filelock`."
        ) from e

    lock = FileLock(_lock_path_for(target))
    with lock:
        current = load_config(target)
        new = mutator(current)
        payload = _serialize_for_json(new)
        payload["_format_version"] = CONFIG_FORMAT_VERSION
        atomic_write_json(str(target), payload)

    logger.info("config_writer: mutated %s", target)
    return new, target


def set_field(
    field_path: str,
    value: Any,
    path: Path | None = None,
) -> tuple[MaximConfig, Path]:
    """Set a single field by dot-path and persist.

    Coerces string values to the schema-correct type per the same
    dispatch the env-var path uses (`_coerce_for_field`). Validates
    the resulting dataclass shape.

    Raises :class:`ConfigurationError` on unknown field path, type
    mismatch, range violation, enum miss, or invalid api_key_ref.
    """
    from maxim.runtime.config_loader import _coerce_for_field, _FIELD_TO_ENV

    if field_path not in _FIELD_TO_ENV:
        raise ConfigurationError(
            f"config_writer: unknown field path {field_path!r}. Valid paths: {sorted(_FIELD_TO_ENV.keys())}"
        )

    # Coerce string-input values via the same dispatch the env-var
    # path uses, so CLI input ("4") gets converted to int 4. Non-string
    # callers (Python API) can pass typed values directly; we skip
    # coercion for those.
    if isinstance(value, str):
        coerced = _coerce_for_field(value, field_path)
    else:
        coerced = value

    def mutator(current: MaximConfig) -> MaximConfig:
        return _apply_field_to_config(current, field_path, coerced)

    return mutate_config(mutator, path=path)


def _apply_field_to_config(
    config: MaximConfig,
    field_path: str,
    value: Any,
) -> MaximConfig:
    """Return a new :class:`MaximConfig` with ``field_path`` set to ``value``.

    Walks the dot path via dataclasses.replace, rebuilding the
    intermediate frozen dataclasses as needed.
    """
    from dataclasses import replace

    parts = field_path.split(".")
    if len(parts) == 1:
        return replace(config, **{parts[0]: value})

    # Walk down to the leaf parent
    section_name = parts[0]
    section = getattr(config, section_name)
    new_section = _apply_field_to_section(section, parts[1:], value, section_name)
    return replace(config, **{section_name: new_section})


def _apply_field_to_section(
    section: Any,
    remaining_parts: list[str],
    value: Any,
    section_path: str,
) -> Any:
    """Recursively apply a field assignment to a nested section."""
    from dataclasses import replace

    if len(remaining_parts) == 1:
        # Special-case LaneTierConfig: a string passed for
        # remote_api_key_ref must pass the load-time validation that
        # rejects inline strings (cross-confirmed I-3/IM3 fold).
        if (
            isinstance(section, LaneTierConfig)
            and remaining_parts[0] == "remote_api_key_ref"
            and isinstance(value, str)
        ):
            from maxim.runtime.config_loader import _validate_api_key_ref

            value = _validate_api_key_ref(value, f"{section_path}.remote_api_key_ref")
        return replace(section, **{remaining_parts[0]: value})

    next_section_name = remaining_parts[0]
    next_section = getattr(section, next_section_name)
    new_next = _apply_field_to_section(
        next_section,
        remaining_parts[1:],
        value,
        f"{section_path}.{next_section_name}",
    )
    return replace(section, **{next_section_name: new_next})


__all__ = [
    "mutate_config",
    "set_field",
    "write_config",
]
