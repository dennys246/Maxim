"""Canonical writer for ``~/.config/maxim/config.json`` (C2 IM2 fold).

This module is the ONLY sanctioned writer surface for ``config.json``.
CI grep allow-lists this file + its test file as the only callers of
``atomic_write_json`` against the config-json path, mirroring the
``mesh_setup.py`` discipline that ``write_mesh_config`` is the only
sanctioned writer for ``mesh.yml``.

The IM2 caller gate in ``.github/workflows/test.yml`` enumerates the
sanctioned callers of ``write_config`` / ``mutate_config`` /
``set_field``: the operator-explicit config verbs (``config_cli``,
``peer/cli``) plus, test-side, each suite that exercises a real
``config set`` → ``load_config`` round trip. Test files are listed
individually rather than by a ``tests/`` wildcard so a new test-side
writer stays visible in review; a section whose round trip is only
simulated by hand-writing JSON would not catch the failure these
tests exist for (a value the writer emits and the loader drops).

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
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from maxim.exceptions import ConfigurationError
from maxim.runtime.config_loader import (
    CONFIG_FORMAT_VERSION,
    _check_format_version,
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
    # A config parsed from an OLDER (or same-version) file is this build's schema, so it is written at
    # this build's version: passing the old version through made the canonical stamp refuse once
    # CONFIG_FORMAT_VERSION moved (#856). A config parsed from a NEWER file lost the keys this build
    # tolerated; writing it would drop them silently and stamp the file down (#974), so that stays a
    # loud refusal.
    loaded_version = payload.pop("_format_version", None)
    if isinstance(loaded_version, str) and loaded_version:
        if _check_format_version({"_format_version": loaded_version})[1]:
            raise ConfigurationError(
                f"config_writer: this config was loaded from a newer config.json (_format_version "
                f"{loaded_version!r}; this build writes {CONFIG_FORMAT_VERSION}). Writing it would drop the "
                "settings this build does not know. Upgrade Maxim, or edit the file by hand (#974)."
            )
    # Post-implementation Architecture #1 fold: route _format_version
    # stamping through the canonical with_format_version helper.
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
        # A NEWER file: this build would drop the settings it does not know and stamp the file down
        # (#974). Refuse, and name the explicit way out.
        _refuse_if_newer(_read_raw_config(target))
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


# ─────────────────────────────────────────────────────────────────────────────
# A newer config.json after a downgrade (#974): refuse, then an explicit transition
# ─────────────────────────────────────────────────────────────────────────────

PRESERVED_FILENAME = "config.preserved.json"
_PRESERVED_FORMAT_VERSION = "1.0"
_PRESERVED_MAX_BYTES = 1_000_000  # the sidecar is read before anything is shown; bound it

# The settings whose restore cannot widen what an agent may do, where Maxim reads from or writes to,
# what it contacts, or what it may discard: tuning knobs only. EVERYTHING else -- every other field, any
# field a later build adds -- is security-relevant and flagged in the restore diff (fail safe, #974
# review). Deliberately OFF the list: `llm.backend` (pytorch is a different model-loading and code path),
# `memory.strategy` (changes what is forgotten), `data.budget_gb` (None lifts the disk cap).
# `llm.max_response_tokens` / `llm.deliberation_max_cycles` stay ON it: they scale per-turn cloud spend,
# but `cloud.session_budget_usd`, which bounds that spend, is flagged.
_NON_SECURITY_FIELDS = frozenset(
    {
        "llm.n_ctx",
        "llm.max_response_tokens",
        "llm.deliberation_max_cycles",
        "lanes.large.timeout_s",
        "lanes.medium.timeout_s",
        "lanes.small.timeout_s",
        "auto_spawn.timeout_s",
        "memory.k",
        "memory.retro_cutoff_us",
        "memory.retro_tau_us",
        "memory.s_base",
        "sim.aut_turn_timeout_s",
        "sim.drive_gate_enabled",
        "sim.substrate_explore_bonus_weight",
    }
)


def safe_display(text: object, limit: int = 120) -> str:
    """``text`` made safe to print: control characters shown escaped (a crafted key or value cannot
    move the cursor, clear a line or add one) and the result truncated. For every name or value the
    #974 flow prints, in messages and logs alike."""
    raw = text if isinstance(text, str) else json.dumps(text, ensure_ascii=True, default=str)
    escaped = "".join(ch if ch.isprintable() else ch.encode("unicode_escape").decode("ascii") for ch in raw)
    return escaped if len(escaped) <= limit else escaped[: limit - 3] + "..."


def is_security_relevant(field_path: str) -> bool:
    """Whether restoring ``field_path`` needs the operator's particular attention: everything except the
    explicit tuning knobs in ``_NON_SECURITY_FIELDS`` (fail safe: a field added later is flagged)."""
    return field_path not in _NON_SECURITY_FIELDS


def preserved_path(target: Path | None = None) -> Path:
    """The sidecar beside ``config.json`` that holds settings set aside by ``downgrade_config``."""
    return (target if target is not None else config_path()).with_name(PRESERVED_FILENAME)


def _read_raw_config(target: Path) -> dict[str, Any] | None:
    """The config file as raw JSON (None when absent or empty); the loader's own read would drop what
    this build does not know, which is exactly what #974 must see."""
    if not target.is_file():
        return None
    text = target.read_text(encoding="utf-8").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"config.json: invalid JSON at {target}: {e.msg}") from e
    return data if isinstance(data, dict) else None


def _is_newer(raw: dict[str, Any] | None) -> bool:
    return raw is not None and bool(_check_format_version(raw)[1])


def _refuse_if_newer(raw: dict[str, Any] | None) -> None:
    if not _is_newer(raw):
        return
    assert raw is not None
    from maxim.runtime.config_loader import unknown_config_entries

    names = sorted(unknown_config_entries(raw))
    shown = ", ".join(safe_display(n, 60) for n in names[:8]) + (" ..." if len(names) > 8 else "")
    raise ConfigurationError(
        f"config.json was written by a newer Maxim (_format_version {safe_display(raw.get('_format_version'), 20)}; "
        f"this build writes {CONFIG_FORMAT_VERSION}). Writing it would drop {len(names)} setting(s) this build does "
        f"not know ({shown}). Run `maxim config downgrade` to keep the settings this build knows and set the rest "
        "aside in config.preserved.json, or upgrade Maxim (#974)."
    )


def _read_preserved(sidecar: Path) -> dict[str, Any]:
    """The sidecar's content, size-capped (one bounded read) and shape-checked; an empty record when
    there is none."""
    empty: dict[str, Any] = {"_format_version": _PRESERVED_FORMAT_VERSION, "entries": {}}
    if not sidecar.exists():
        return empty
    if not sidecar.is_file():
        raise ConfigurationError(f"{sidecar.name} is not a regular file; move it aside")
    with open(sidecar, "rb") as fh:
        blob = fh.read(_PRESERVED_MAX_BYTES + 1)
    if len(blob) > _PRESERVED_MAX_BYTES:
        raise ConfigurationError(
            f"{sidecar.name} is too large (over {_PRESERVED_MAX_BYTES} bytes); refusing to read it"
        )
    try:
        data = json.loads(blob.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ConfigurationError(f"{sidecar.name}: not valid JSON ({type(e).__name__})") from e
    from maxim.utils.format_version import check_format_version

    if not isinstance(data, dict) or not isinstance(data.get("entries", {}), dict):
        raise ConfigurationError(f"{sidecar.name}: not a preserved-settings record")
    check_format_version(data, "config_preserved", log=logger)
    entries = {}
    for name, entry in data.get("entries", {}).items():
        if not isinstance(entry, dict) or "value" not in entry:
            raise ConfigurationError(f"{sidecar.name}: entry {safe_display(name, 60)} is malformed")
        entries[name] = entry
    return {**data, "entries": entries}


def _write_preserved(sidecar: Path, record: dict[str, Any]) -> None:
    from maxim.utils.atomic_io import atomic_write_text

    record = {**record, "_format_version": _PRESERVED_FORMAT_VERSION}
    # 0600 from creation: API-key REFERENCES and paths can be among the preserved settings.
    atomic_write_text(str(sidecar), json.dumps(record, indent=2, sort_keys=True), initial_mode=0o600)


@dataclass(frozen=True)
class DowngradeResult:
    """What ``downgrade_config`` did. Runtime-only (never persisted), so outside CC3."""

    config: Path
    sidecar: Path
    preserved: tuple[str, ...]


def downgrade_config(path: Path | None = None) -> DowngradeResult:
    """Rewrite a NEWER config.json for this build, setting aside what this build does not know (#974).

    Operator-explicit (``maxim config downgrade``). The settings this build knows are kept and written
    at this build's version; the others move to ``config.preserved.json`` (0600), which nothing reads
    for configuration. The sidecar is written BEFORE the config, so a failure never loses a setting.
    Refuses a file that is not newer: there is nothing to set aside.
    """
    import time

    from filelock import FileLock

    from maxim.runtime.config_loader import unknown_config_entries

    target = path if path is not None else config_path()
    sidecar = preserved_path(target)
    with FileLock(_lock_path_for(target)):
        raw = _read_raw_config(target)
        if not _is_newer(raw):  # a future MAJOR raises inside: it cannot be downgraded, only upgraded to
            raise ConfigurationError(
                f"config.json is not from a newer Maxim (it matches this build's {CONFIG_FORMAT_VERSION} or is "
                "older): nothing to set aside"
            )
        assert raw is not None
        entries = unknown_config_entries(raw)
        record = _read_preserved(sidecar)
        source = str(raw.get("_format_version"))
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        # Per-entry provenance, shown at restore: an entry from an older downgrade keeps ITS version and
        # time, so a stale value is visible as stale; a key preserved again takes the newer value.
        record["entries"] = {
            **record["entries"],
            **{p: {"value": v, "source_format_version": source, "preserved_at": stamp} for p, v in entries.items()},
        }
        _write_preserved(sidecar, record)

        kept = load_config(target)  # the newer file parses with its unknown keys tolerated and dropped
        payload = with_format_version(
            {k: v for k, v in _serialize_for_json(kept).items() if k != "_format_version"}, CONFIG_FORMAT_VERSION
        )
        atomic_write_json(str(target), payload)

    from maxim.runtime.config_loader import invalidate_config_cache

    invalidate_config_cache()
    logger.warning(
        "config_writer: downgraded %s; set aside %d setting(s) in %s: %s",
        target,
        len(entries),
        sidecar,
        ", ".join(safe_display(n, 60) for n in sorted(entries)),
    )
    return DowngradeResult(config=target, sidecar=sidecar, preserved=tuple(sorted(entries)))


@dataclass(frozen=True)
class RestoreRow:
    """One preserved setting as the operator is shown it before a restore. Runtime-only, outside CC3."""

    path: str
    current: Any
    preserved: Any
    security_relevant: bool
    source_format_version: str
    preserved_at: str
    superseded: int = 0  # older set-aside values of this same field that this newer one replaces


def _preserved_sort_key(entry: dict[str, Any]) -> str:
    """Oldest-first ordering by the entry's ``preserved_at``; anything that is not an ISO-8601 UTC
    timestamp (missing, null, a planted string) sorts OLDEST, so it can never win a collision."""
    import re

    stamp = entry.get("preserved_at")
    return stamp if isinstance(stamp, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", stamp) else ""


def _without_fields(entry_path: str, value: Any, fields_to_drop: set[str]) -> Any:
    """``value`` (the value preserved at ``entry_path``) with the given dotted fields removed, and any
    section left empty by that removed too."""
    pruned = json.loads(json.dumps(value))
    prefix = entry_path + "."
    for field_path in fields_to_drop:
        if not field_path.startswith(prefix) or not isinstance(pruned, dict):
            continue
        parts = field_path[len(prefix) :].split(".")
        trail = [pruned]
        for part in parts[:-1]:
            nxt = trail[-1].get(part)
            if not isinstance(nxt, dict):
                break
            trail.append(nxt)
        else:
            trail[-1].pop(parts[-1], None)
            for depth in range(len(trail) - 1, 0, -1):  # drop sections the removal emptied
                if not trail[depth]:
                    trail[depth - 1].pop(parts[depth - 1], None)
    return pruned


def _set_path(data: dict[str, Any], field_path: str, value: Any) -> None:
    node = data
    parts = field_path.split(".")
    for part in parts[:-1]:
        child = node.get(part)
        node[part] = child = dict(child) if isinstance(child, dict) else {}
        node = child
    node[parts[-1]] = value


def _get_path(data: dict[str, Any], field_path: str) -> Any:
    node: Any = data
    for part in field_path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def _flatten_to_fields(path: str, value: Any, leaves: frozenset[str], out: dict[str, Any]) -> None:
    """Split a preserved value into one entry per schema field: a whole section (``tools``,
    ``console``, ``lanes.large``) is never restored wholesale, so every field is merged and SHOWN on its
    own (#974 review: a wholesale section silently reset the fields it lacked). A key the schema does not
    know stays under its own path. Paths are dot-joined, so a key containing a dot is ambiguous with a
    nested one; either way the result is validated and flagged like any other field."""
    prefix = path + "."
    if isinstance(value, dict) and path not in leaves and any(leaf.startswith(prefix) for leaf in leaves):
        for key, sub in value.items():
            _flatten_to_fields(prefix + str(key), sub, leaves, out)
    else:
        out[path] = value


def restore_preserved(path: Path | None = None, *, confirm: Callable[[list[RestoreRow]], bool]) -> list[str]:
    """Bring back settings ``downgrade_config`` set aside, once THIS build knows them (#974).

    Preserved values are split into one entry per schema field and merged field by field, never a
    whole section at a time. Only fields the current schema knows are candidates; the merged config is
    re-validated by the same strict parser as any config.json (a bad value refuses the whole restore).
    ``confirm`` is shown every candidate: current value, preserved value, whether it is security-relevant
    (fail safe), and the version and time it was set aside. Nothing is written unless it returns True.
    Fields this build still does not know stay in the sidecar. Returns the paths restored.

    This is an API, like editing config.json directly; the CLI is what asks a person (it needs an
    interactive terminal and has no flag to skip the question).
    """
    from filelock import FileLock

    from maxim.runtime.config_loader import _parse_config_dict, config_field_paths

    target = path if path is not None else config_path()
    sidecar = preserved_path(target)
    # A lane tier's `extra` is stored INLINE in the tier on disk, so it is not a restorable leaf; a
    # preserved unknown lane field stays set aside (#974 review).
    leaves = frozenset(p for p in config_field_paths() if not (p.startswith("lanes.") and p.endswith(".extra")))
    with FileLock(_lock_path_for(target)):
        record = _read_preserved(sidecar)
        fields: dict[str, dict[str, Any]] = {}
        superseded: dict[str, int] = {}
        entry_fields: dict[str, set[str]] = {}  # each ORIGINAL sidecar entry -> the fields it holds
        # Oldest first, so when two set-aside values land on one field (a section from one downgrade, a
        # field from a later one) the NEWEST wins, and the row says how many it replaced (#974 review).
        for entry_path, entry in sorted(record["entries"].items(), key=lambda kv: _preserved_sort_key(kv[1])):
            flat: dict[str, Any] = {}
            _flatten_to_fields(entry_path, entry["value"], leaves, flat)
            entry_fields[entry_path] = set(flat)
            for field_path, value in flat.items():
                if field_path in fields:
                    superseded[field_path] = superseded.get(field_path, 0) + 1
                fields[field_path] = {**entry, "value": value}
        raw = _read_raw_config(target) or {}
        _refuse_if_newer(raw)
        # What is IN EFFECT now (defaults included), not just what the file spells out: an absent
        # `console.sandbox` is `false`, and the diff must say so.
        # JSON-normalized, so a tuple-typed field compares equal to the list the sidecar holds.
        in_effect = json.loads(
            json.dumps(
                _serialize_for_json(_parse_config_dict({**raw, "_format_version": CONFIG_FORMAT_VERSION})), default=str
            )
        )
        candidates = {p: e for p, e in fields.items() if p in leaves}
        restorable = {p: e for p, e in candidates.items() if _get_path(in_effect, p) != e["value"]}
        # Already in effect: nothing to ask about. They are dropped from the sidecar only as part of a
        # CONFIRMED restore; with nothing to restore, the sidecar is left exactly as it is (the operator
        # never saw its contents, so nothing in it may be discarded).
        unchanged = set(candidates) - set(restorable)
        if not restorable:
            return []
        merged = json.loads(json.dumps(raw))  # a deep copy; the current file is never mutated
        for field_path, entry in restorable.items():
            _set_path(merged, field_path, entry["value"])
        merged["_format_version"] = CONFIG_FORMAT_VERSION
        parsed = _parse_config_dict(merged)  # the same validation as any config.json; raises on a bad value

        rows = [
            RestoreRow(
                path=p,
                current=_get_path(in_effect, p),
                preserved=e["value"],
                security_relevant=is_security_relevant(p),
                source_format_version=str(e.get("source_format_version", "?")),
                preserved_at=str(e.get("preserved_at", "?")),
                superseded=superseded.get(p, 0),
            )
            for p, e in sorted(restorable.items())
        ]
        if not confirm(rows):
            return []
        payload = with_format_version(
            {k: v for k, v in _serialize_for_json(parsed).items() if k != "_format_version"}, CONFIG_FORMAT_VERSION
        )
        atomic_write_json(str(target), payload)
        # An original entry leaves the sidecar only when EVERY field it holds was restored or is already
        # in effect; otherwise it stays in its original form, so no value is discarded unseen (an older
        # value of a field this build does not know included). Fields it holds that are now in effect
        # are simply not asked about next time.
        settled = set(restorable) | unchanged
        remaining = {}
        for key, entry in record["entries"].items():
            held = entry_fields.get(key, set())
            if held <= settled:
                continue
            # Kept for its still-unknown keys, with every SETTLED field pruned out of it: a value already
            # restored (or superseded by one that was, and counted in that row) must never be offered
            # again over a later operator choice (#974 review).
            remaining[key] = {**entry, "value": _without_fields(key, entry["value"], held & settled)}
        if remaining:
            _write_preserved(sidecar, {**record, "entries": remaining})
        else:
            sidecar.unlink()

    from maxim.runtime.config_loader import invalidate_config_cache

    invalidate_config_cache()
    logger.warning(
        "config_writer: restored %d preserved setting(s) into %s: %s",
        len(rows),
        target,
        ", ".join(safe_display(r.path, 60) for r in rows),
    )
    return [r.path for r in rows]


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

    Since the 2026-09-04 console-auth review fold, ``atomic_write_secret``
    itself guarantees 0600 from fd creation on fresh files (``initial_mode``) —
    the umask-tighten + chmod here predate that fix and stay as harmless
    belt-and-suspenders (the peer.yml→config migration's pattern,
    config_unification C4); do NOT copy this shape into new callers, the
    writer alone now suffices. The stored ref is always the file PATH; the
    inline key never touches config.json.
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
    if tier not in ("large", "medium", "small"):
        return False, "none", f"unknown tier {tier!r} (expected large/medium/small)"
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

    # 2. Legacy fields — resolved through `resolve_setting` so the ENV layer of
    # the canonical CLI > env > config.json > default chain counts. Reading
    # config.json alone would answer "none" on a box configured via
    # MAXIM_LANE_LARGE_REMOTE_URL / MAXIM_LLM_PROFILE (the peer-migration and
    # every documented env setup), pushing a working operator back through the
    # setup wizard (review finding).
    def _resolved(field_path: str, fallback: Any = None) -> Any:
        try:
            from maxim.runtime.config_loader import resolve_setting

            result = resolve_setting(field_path)
            return result[0] if isinstance(result, tuple) else result
        except Exception:
            return fallback

    remote_url = _resolved(f"lanes.{tier}.remote_url", getattr(lane, "remote_url", None))
    if remote_url:
        return True, "mesh", f"remote lane → {remote_url}"
    profile = _resolved("llm.profile", getattr(cfg.llm, "profile", None))
    cloud_enabled = _resolved("cloud.enabled", getattr(cfg.cloud, "enabled", False))
    if cloud_enabled and profile:
        return True, "cloud", f"cloud enabled with profile {profile}"
    if profile:
        return True, "local", f"local profile {profile}"
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
