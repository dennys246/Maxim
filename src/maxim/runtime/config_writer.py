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
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from maxim.exceptions import ConfigurationError
from maxim.runtime.config_loader import (
    CONFIG_FORMAT_VERSION,
    LaneTierConfig,
    LaneTierPlacement,
    MaximConfig,
    config_path,
    load_config,
)
from maxim.utils.atomic_io import atomic_write_json, atomic_write_secret
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
        # lane_capability_placement_split.md 3b: inline each placement entry's
        # ``extra`` the same way, so a forward-grown placement field survives a
        # write/read round-trip instead of double-nesting under an "extra" key.
        for entry in tier.get("placement", []) or []:
            if not isinstance(entry, dict):
                continue
            entry_extras = entry.pop("extra", {}) or {}
            for k, v in entry_extras.items():
                assert k not in entry, (
                    f"config_writer: LaneTierPlacement.__post_init__ should have "
                    f"caught the collision on lanes.{tier_name}.placement extra[{k!r}]."
                )
                entry[k] = v
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

    # Long-lived processes (maxim serve) re-read config after setup writes —
    # without this the get_config singleton serves stale config to the next
    # in-process lane build (post-merge review Exec B2).
    from maxim.runtime.config_loader import invalidate_config_cache

    invalidate_config_cache()
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

    # Same invalidation as write_config — mutate_config writes directly and
    # does NOT route through write_config (post-merge review Exec B2).
    from maxim.runtime.config_loader import invalidate_config_cache

    invalidate_config_cache()
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


def apply_mesh_setup(
    leader_url: str,
    api_key: str,
    *,
    remote_model: str | None = None,
    path: Path | None = None,
) -> tuple[Path, Path]:
    """Console SETUP seam helper — connect this box to a leader as a peer.

    Writes a resolvable large-tier PEER placement (``role=peer`` +
    ``lanes.large.{remote_url, remote_api_key_ref[, remote_model]}``) so
    ``resolve_setting`` / ``derive_placement`` read it back as a remote large
    lane. The API key lands as a **ref**: it is written to a mode-0600 file via
    :func:`atomic_write_secret` and only that file PATH is stored in config —
    never the inline key (which ``_validate_api_key_ref`` rejects at load time).

    This is a thin convenience over the sanctioned single-writer path: the config
    fields are applied atomically through :func:`mutate_config` (one RMW under the
    file lock). The app must NOT hand-assemble the nested lane dict or know the
    ``remote_api_key_ref`` rules — that is what this helper is for.

    Args:
        leader_url: the leader's inference URL (e.g. ``https://maxim.example.com``).
        api_key: the raw key to store as a ref (written to a 0600 file, not config).
        remote_model: optional served-model name to request from the leader.
        path: config.json path override (tests). Secrets land next to it.

    Returns ``(secret_path, config_path_written)``.
    """
    if not leader_url:
        raise ConfigurationError("apply_mesh_setup: leader_url is required")
    if not api_key:
        raise ConfigurationError("apply_mesh_setup: api_key is required")

    target = path if path is not None else config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    secret_path = _write_secret_ref(target.parent / "mesh_leader_api_key", api_key)

    def mutator(current: MaximConfig) -> MaximConfig:
        new = _apply_field_to_config(current, "role", "peer")
        new = _apply_field_to_config(new, "lanes.large.remote_url", leader_url)
        new = _apply_field_to_config(new, "lanes.large.remote_api_key_ref", str(secret_path))
        if remote_model:
            new = _apply_field_to_config(new, "lanes.large.remote_model", remote_model)
        return new

    _, written = mutate_config(mutator, path=path)
    return secret_path, written


def _write_secret_ref(secret_path: Path, key: str) -> Path:
    """Write ``key`` to ``secret_path`` as a mode-0600 file and return the path.

    ``atomic_write_secret`` PRESERVES an existing mode but does not set one on a
    fresh file (it inherits umask → 0644). A key file must be 0600, so tighten
    umask around the create AND chmod explicitly — the belt-and-suspenders the
    peer.yml→config migration uses (config_unification C4). The stored ref is
    always the file PATH; the inline key never touches config.json.
    """
    _prev_umask = os.umask(0o077)
    try:
        atomic_write_secret(str(secret_path), key)
        os.chmod(str(secret_path), 0o600)
    finally:
        os.umask(_prev_umask)
    return secret_path


def apply_cloud_setup(
    provider: str,
    profile: str,
    api_key: str,
    *,
    monthly_budget_usd: float | None = None,
    path: Path | None = None,
) -> tuple[Path, Path]:
    """Console SETUP seam helper — configure a cloud provider as the large lane.

    Writes a resolvable large-tier **CLOUD placement** (``lanes.large.placement``
    = one ``cloud`` entry carrying ``model=<profile>`` + ``api_key_ref``) plus
    ``cloud.enabled=true``, a non-zero ``cloud.max_lanes`` (so the cloud gate
    admits the lane), and the session budget. The key lands as a **ref** (0600
    file), never inline — the placement's ``api_key_ref`` is resolved into the
    provider env var at lane-build time (the two-site injection fix in
    ``lane_backends`` makes a cloud-profile placement actually reach the backend).

    Mirrors :func:`apply_mesh_setup`: one atomic RMW through :func:`mutate_config`,
    the app never hand-assembles the placement dict or knows the ref rules.

    Args:
        provider: provider label (e.g. ``anthropic``) — used only to name the
            secret file so multiple providers don't collide.
        profile: the cloud model profile to run (e.g. ``claude-sonnet``).
        api_key: the raw provider key to store as a ref.
        monthly_budget_usd: optional cap → ``cloud.session_budget_usd``.
        path: config.json path override (tests). Secrets land next to it.

    Returns ``(secret_path, config_path_written)``.
    """
    from dataclasses import replace

    if not provider:
        raise ConfigurationError("apply_cloud_setup: provider is required")
    if not profile:
        raise ConfigurationError("apply_cloud_setup: profile is required")
    if not api_key:
        raise ConfigurationError("apply_cloud_setup: api_key is required")

    target = path if path is not None else config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    # Sanitize the provider label for the filename (never trust it as a path).
    safe_provider = "".join(c for c in provider if c.isalnum() or c in ("-", "_")) or "cloud"
    secret_path = _write_secret_ref(target.parent / f"cloud_{safe_provider}_api_key", api_key)

    def mutator(current: MaximConfig) -> MaximConfig:
        placement = LaneTierPlacement(origin="cloud", model=profile, api_key_ref=str(secret_path))
        large = replace(current.lanes.large, placement=(placement,))
        lanes = replace(current.lanes, large=large)
        cloud_kwargs: dict[str, Any] = {"enabled": True, "max_lanes": max(1, current.cloud.max_lanes)}
        if monthly_budget_usd is not None:
            cloud_kwargs["session_budget_usd"] = float(monthly_budget_usd)
        cloud = replace(current.cloud, **cloud_kwargs)
        return replace(current, lanes=lanes, cloud=cloud)

    _, written = mutate_config(mutator, path=path)
    return secret_path, written


def placement_resolvable(tier: str = "large") -> tuple[bool, str, str]:
    """Is ``tier``'s LLM placement resolvable right now? The SETUP read-side.

    The counterpart to :func:`apply_mesh_setup` / :func:`apply_cloud_setup`:
    those WRITE a placement, this asks whether one is in place — so a caller
    (the Reachy bootstrap, the console's setup wizard, a first-run check) can
    branch on "is this box configured to think yet?" **without knowing config
    vocabulary**. Before this helper, callers had to reach for
    ``resolve_setting('lanes.large.remote_url')`` and friends and re-implement
    the local/mesh/cloud precedence by hand — leaking exactly the schema
    knowledge the SETUP seam exists to hide.

    Resolution mirrors the runtime's own order: an explicit ``placement`` wins;
    otherwise the legacy ``remote_url`` (mesh) / cloud-profile fields are
    classified the way ``derive_placement`` does; otherwise a local profile.

    Returns ``(resolvable, kind, detail)`` where ``kind`` is one of
    ``"mesh"`` / ``"cloud"`` / ``"local"`` / ``"none"`` and ``detail`` is a
    human-readable, UI-safe summary (never contains a key — only refs/URLs).
    Never raises: an unreadable config answers ``(False, "none", <why>)``.
    """
    try:
        cfg = load_config()
    except Exception as e:  # malformed config.json — answer, don't explode
        return False, "none", f"config could not be loaded: {type(e).__name__}: {e}"

    lane = getattr(cfg.lanes, tier, None)
    if lane is None:
        return False, "none", f"unknown tier {tier!r} (expected large/medium/small)"

    # 1. Explicit placement wins (the authoritative carrier).
    placement = getattr(lane, "placement", ()) or ()
    if placement:
        primary = placement[0]
        origin = str(getattr(primary, "origin", "") or "")
        model = getattr(primary, "model", None)
        url = getattr(primary, "remote_url", None) or getattr(primary, "url", None)
        if origin == "peer":
            return (bool(url), "mesh", f"peer placement → {url}" if url else "peer placement missing a url")
        if origin == "cloud":
            ok = bool(model or url)
            return ok, "cloud", f"cloud placement → {model or url}" if ok else "cloud placement missing model/url"
        if origin == "local":
            return (bool(model), "local", f"local placement → {model}" if model else "local placement missing a model")
        return False, "none", f"placement has an unrecognized origin {origin!r}"

    # 2. Legacy fields, classified the way derive_placement does.
    if getattr(lane, "remote_url", None):
        return True, "mesh", f"remote lane → {lane.remote_url}"
    if getattr(cfg.cloud, "enabled", False) and getattr(cfg.llm, "profile", None):
        return True, "cloud", f"cloud enabled with profile {cfg.llm.profile}"
    if getattr(cfg.llm, "profile", None):
        return True, "local", f"local profile {cfg.llm.profile}"
    return False, "none", "no placement, remote_url, or llm.profile configured"


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
