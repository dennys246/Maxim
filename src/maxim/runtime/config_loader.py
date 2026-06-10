"""Instance-level operator config loader (C1 of config_unification.md).

Reads ``~/.config/maxim/config.json`` and exposes a typed
:class:`MaximConfig` dataclass plus the :func:`resolve_setting`
precedence chain that absorbs ~22 daily-use ``MAXIM_*`` env vars onto
a single source of truth.

Path convention mirrors :func:`maxim.peer.config.peer_config_path` —
XDG on POSIX, ``%APPDATA%\\maxim\\`` on Windows. Same directory as
``peer.yml`` / ``mesh.yml`` / ``profiles.yml`` per CLAUDE.md's
"declarative config layer" invariant.

Precedence (CLI > env > config > default) is exposed via
:func:`resolve_setting`. The function logs at INFO on convergence
(both env and config set to the same value — operator's two-sources-
of-truth confusion is the bug class to surface) AND at WARNING on
divergence (env shadows config with a different value).

Schema versioning at 1.0 (CC1): every ``config.json`` written by
Maxim carries ``"_format_version": "1.0"`` at root. The file is
CLI-canonical (the ``maxim config`` verbs are the operator's primary
path; hand-edit is the escape hatch), so CC1 applies. This is the
intentional divergence from ``profiles.yml`` (hand-edit-canonical, no
``_format_version``) discussed in config_unification.md's
"Schema versioning" subsection.

Pre-implementation two-lens review fold (2026-06-01) anchors:

- C-1: ``_env_is_set`` rejects empty-string env vars as unset
- C-2: ``_coerce_*`` uses the pinned truthy/falsy/range table
- CR3: ``resolve_setting`` logs convergence AND divergence
- I-3 / IM3: ``_validate_api_key_ref`` rejects inline strings at load time
- I-4: ``_warned_envs`` module set fires the deprecation INFO once per startup
- I-6 / IM4: ``_parse_config_dict`` ties unknown-key handling to ``_format_version``
- IM1: every section dataclass declares its CC3 path in its docstring
- N1: ``_format_version`` is the first field of :class:`MaximConfig`
"""

from __future__ import annotations

import json
import logging
import os
import platform
import threading
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

from maxim.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Schema constants (FROZEN at 1.0)
# ─────────────────────────────────────────────────────────────────────────────

CONFIG_FORMAT_VERSION: str = "1.0"
"""Inaugural format version. Bumped per CC1 on shape changes:
minor (1.1+) for additive non-breaking fields, major (2.0) for
required-field breaking changes (loader migration required)."""

_VALID_ROLES: frozenset[str] = frozenset({"leader", "peer", "solo"})
_VALID_BACKENDS: frozenset[str] = frozenset({"llama_cpp", "pytorch"})
_VALID_REDACTION_POLICIES: frozenset[str] = frozenset({"standard", "relaxed", "strict"})
_VALID_TIER_NAMES: frozenset[str] = frozenset({"large", "medium", "small"})

# C-2 fold: canonical truthy/falsy sets. Match the existing
# MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION parser + the leader-UX PR's
# profile-loader. Empty string is rejected by `_env_is_set` before
# coercion runs.
_TRUTHY: frozenset[str] = frozenset({"1", "true", "yes", "on"})
_FALSY: frozenset[str] = frozenset({"0", "false", "no", "off"})


# ─────────────────────────────────────────────────────────────────────────────
# Section dataclasses (CC3 path declarations per IM1 fold)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LLMConfigSection:
    """LLM core settings.

    SHAPE-FROZEN at 1.0 (CC3) — path (b) per config_unification.md IM1
    fold. ``backend`` is a frozen enum surface; an ``extra:`` dict
    would dilute typed validation. Adding optional fields with defaults
    is non-breaking; adding required fields is a 2.0 break.
    """

    enabled: bool = True
    profile: str | None = None
    n_ctx: int = 8192
    backend: Literal["llama_cpp", "pytorch"] = "llama_cpp"
    auto_download: bool = False


_LANE_TIER_DECLARED_FIELDS: frozenset[str] = frozenset(
    {"remote_url", "remote_model", "remote_api_key_ref", "timeout_s", "placement"}
)

_VALID_PLACEMENT_ORIGINS: frozenset[str] = frozenset({"local", "cloud", "peer"})

_LANE_TIER_PLACEMENT_DECLARED_FIELDS: frozenset[str] = frozenset({"origin", "model", "url", "api_key_ref", "timeout_s"})


@dataclass(frozen=True)
class LaneTierPlacement:
    """One declarative provider candidate for a lane tier (config.json).

    The declarative twin of the runtime ``worker_pool.ProviderPlacement``: it
    carries an **unresolved** ``api_key_ref`` (file path or ``keyring:`` URI)
    that is resolved into ``ProviderPlacement.api_key`` when the lane is built —
    deliberately split so "ref" never names a resolved value (mirrors
    ``LaneTierConfig.remote_api_key_ref`` → ``LaneConfig.remote_api_key``).

    ESCAPE-HATCH at 1.0 (CC3) — path (a). ``extra`` gathers forward-growth
    metadata; ``__post_init__`` rejects keys that collide with declared fields.
    ``origin`` is one of ``"local"`` / ``"cloud"`` / ``"peer"`` — validated at
    parse time (``_parse_lane_tier``) with JSON context. Field-coherence
    (PEER needs url, LOCAL needs model, …) is validated at the producer boundary
    via ``worker_pool.validate_placement_coherence`` when the placement is
    resolved into a ``ProviderPlacement``.
    """

    origin: str
    model: str | None = None
    url: str | None = None
    api_key_ref: str | None = None
    timeout_s: float | None = None
    extra: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def __post_init__(self) -> None:
        collisions = _LANE_TIER_PLACEMENT_DECLARED_FIELDS & set(self.extra.keys())
        if collisions:
            raise ConfigurationError(
                f"LaneTierPlacement: extra dict contains key(s) that collide "
                f"with declared fields: {sorted(collisions)}. extra is for "
                f"forward-growth additive metadata only."
            )


@dataclass(frozen=True)
class LaneTierConfig:
    """Per-tier remote routing.

    ESCAPE-HATCH at 1.0 (CC3) — path (a) per config_unification.md IM1
    fold. The only section type where (a) is genuinely right:
    ``remote_url`` / ``remote_model`` / ``remote_api_key_ref`` are
    likely to grow forward (``remote_health_path``,
    ``remote_timeout_s``, ``remote_routing_weight``, ...).

    ``extra`` values MUST be JSON-serializable (str/int/float/bool/None
    plus nested list/dict only) per CC3 — ndarray/datetime raise at
    persist time. Producers prefer declared fields; ``extra`` is for
    genuinely additive metadata only. The ``hash=False, compare=False``
    spec is load-bearing per CC3 escape-hatch rules.

    **Post-implementation Architecture #4 fold:** the escape-hatch
    ``extra`` dict must NOT contain keys that collide with declared
    fields (``remote_url``, ``remote_model``, ``remote_api_key_ref``).
    Without this guard, the C2 writer's ``_serialize_for_json`` step
    would silently shadow declared fields on round-trip. Push the
    invariant into the type via ``__post_init__`` so a malformed
    LaneTierConfig is rejected at construction time, not at write
    time. Mirrors CLAUDE.md's "push silent-no-op invariants into
    types, not helpers" discipline.
    """

    remote_url: str | None = None
    remote_model: str | None = None
    remote_api_key_ref: str | None = None
    # llm_timeout_scalability.md Stage 2: per-tier read timeout for the
    # inference HTTP call. ``None`` → backend default
    # (``_MaximPeerBackend`` 300s, ``_OpenAIBackend`` 60s). Set explicitly
    # for large self-hosted lanes running 30B+ models where defaults
    # under-provision the wait window (qwen2.5-32b on Mac Mini can hit
    # 150s+ TTFT on growing-context prompts). Resolved through the
    # standard CLI > env > config.json > default precedence chain via
    # ``resolve_setting('lanes.<tier>.timeout_s', ...)``.
    timeout_s: float | None = None
    # lane_capability_placement_split.md Phase 3b: explicit ordered placement
    # for this tier (declarative twin of LaneConfig.placement). ``()`` (the
    # default) means "derive from the remote_* fields above" — existing configs
    # behave identically. When non-empty, it is the operator's declarative
    # placement; CLI flags (--llm/--cloud-lane) still edit it per the
    # CLI > config precedence chain. Resolved into worker_pool.ProviderPlacement
    # (api_key_ref → api_key) at lane-build time.
    placement: tuple[LaneTierPlacement, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def __post_init__(self) -> None:
        collisions = _LANE_TIER_DECLARED_FIELDS & set(self.extra.keys())
        if collisions:
            raise ConfigurationError(
                f"LaneTierConfig: extra dict contains key(s) that collide "
                f"with declared fields: {sorted(collisions)}. extra is for "
                f"forward-growth additive metadata only — declared fields "
                f"go in their own slots."
            )
        if self.timeout_s is not None:
            if not isinstance(self.timeout_s, (int, float)):
                raise ConfigurationError(
                    f"LaneTierConfig.timeout_s must be a number, got "
                    f"{type(self.timeout_s).__name__}: {self.timeout_s!r}"
                )
            if self.timeout_s <= 0:
                raise ConfigurationError(
                    f"LaneTierConfig.timeout_s must be positive, got {self.timeout_s}. Use None for backend default."
                )


@dataclass(frozen=True)
class LanesConfigSection:
    """Per-tier remote routing config keyed by frozen tier-name set.

    SHAPE-FROZEN at 1.0 (CC3) — path (b) per config_unification.md IM1
    fold. Keyed by the lane-tier-names invariant (``large``, ``medium``,
    ``small``). Adding a new tier post-1.0 is a major-version bump.
    """

    large: LaneTierConfig = field(default_factory=LaneTierConfig)
    medium: LaneTierConfig = field(default_factory=LaneTierConfig)
    small: LaneTierConfig = field(default_factory=LaneTierConfig)


@dataclass(frozen=True)
class CloudConfigSection:
    """Cloud-LLM fallback config.

    SHAPE-FROZEN at 1.0 (CC3) — path (b). ``redaction_policy`` is a
    frozen enum; cloud-LLM contract is tightly coupled to the
    redaction layer.
    """

    enabled: bool = False
    max_lanes: int = 0
    fallback_model: str | None = None
    session_budget_usd: float = 5.0
    redaction_policy: Literal["standard", "relaxed", "strict"] = "standard"


@dataclass(frozen=True)
class ProxyConfigSection:
    """Leader-proxy admission control.

    SHAPE-FROZEN at 1.0 (CC3) — path (b).
    """

    max_concurrent: int = 4
    rate_limit_rpm: int = 0


@dataclass(frozen=True)
class AutoSpawnConfigSection:
    """Auto-spawn behavior for llama-cpp-server and cloudflared.

    SHAPE-FROZEN at 1.0 (CC3) — path (b).
    """

    llm_server: bool = True
    tunnel: bool = True
    port: int = 8100
    timeout_s: int = 120


@dataclass(frozen=True)
class DataConfigSection:
    """Data paths + disk budget.

    SHAPE-FROZEN at 1.0 (CC3) — path (b).
    """

    home: str | None = None
    budget_gb: float | None = None


@dataclass(frozen=True)
class MaximConfig:
    """Top-level Maxim instance config.

    SHAPE-FROZEN at 1.0 (CC3) — path (b) per config_unification.md IM1
    fold. Top-level shape is the schema contract; section additions
    go inside section types, not at root. Adding optional fields with
    defaults is non-breaking; adding required fields requires
    ``_format_version 2.0`` + a migration step in the same commit.

    Field order: ``_format_version`` is declared first per the
    underscore-sort-first convention (N1 fold from the review round —
    matches ``maxim.utils.format_version`` precedent so the field
    appears before payload keys in ``json.dumps(..., sort_keys=True)``
    output).
    """

    _format_version: str = CONFIG_FORMAT_VERSION
    role: Literal["leader", "peer", "solo"] | None = None
    llm: LLMConfigSection = field(default_factory=LLMConfigSection)
    lanes: LanesConfigSection = field(default_factory=LanesConfigSection)
    cloud: CloudConfigSection = field(default_factory=CloudConfigSection)
    proxy: ProxyConfigSection = field(default_factory=ProxyConfigSection)
    auto_spawn: AutoSpawnConfigSection = field(default_factory=AutoSpawnConfigSection)
    data: DataConfigSection = field(default_factory=DataConfigSection)


# ─────────────────────────────────────────────────────────────────────────────
# Absorbed env-var registry + field-path mapping
# ─────────────────────────────────────────────────────────────────────────────


_FIELD_TO_ENV: dict[str, str] = {
    "role": "MAXIM_ROLE",
    "llm.enabled": "MAXIM_LLM_ENABLED",
    "llm.profile": "MAXIM_LLM_PROFILE",
    "llm.n_ctx": "MAXIM_LLM_N_CTX",
    "llm.backend": "MAXIM_LLM_BACKEND",
    "llm.auto_download": "MAXIM_AUTO_DOWNLOAD_MODELS",
    "lanes.large.remote_url": "MAXIM_LANE_LARGE_REMOTE_URL",
    "lanes.large.remote_model": "MAXIM_LANE_LARGE_REMOTE_MODEL",
    "lanes.large.remote_api_key_ref": "MAXIM_LANE_LARGE_REMOTE_API_KEY",
    "lanes.large.timeout_s": "MAXIM_LANE_LARGE_TIMEOUT_S",
    "lanes.medium.remote_url": "MAXIM_LANE_MEDIUM_REMOTE_URL",
    "lanes.medium.remote_model": "MAXIM_LANE_MEDIUM_REMOTE_MODEL",
    "lanes.medium.remote_api_key_ref": "MAXIM_LANE_MEDIUM_REMOTE_API_KEY",
    "lanes.medium.timeout_s": "MAXIM_LANE_MEDIUM_TIMEOUT_S",
    "lanes.small.remote_url": "MAXIM_LANE_SMALL_REMOTE_URL",
    "lanes.small.remote_model": "MAXIM_LANE_SMALL_REMOTE_MODEL",
    "lanes.small.remote_api_key_ref": "MAXIM_LANE_SMALL_REMOTE_API_KEY",
    "lanes.small.timeout_s": "MAXIM_LANE_SMALL_TIMEOUT_S",
    "cloud.enabled": "MAXIM_LLM_CLOUD_ENABLED",
    "cloud.max_lanes": "MAXIM_MAX_CLOUD_LANES",
    "cloud.fallback_model": "MAXIM_CLOUD_FALLBACK_MODEL",
    "cloud.session_budget_usd": "MAXIM_CLOUD_SESSION_BUDGET",
    "cloud.redaction_policy": "MAXIM_LLM_REDACTION_POLICY",
    "proxy.max_concurrent": "MAXIM_PROXY_MAX_CONCURRENT",
    "proxy.rate_limit_rpm": "MAXIM_PROXY_RATE_LIMIT_RPM",
    "auto_spawn.llm_server": "MAXIM_AUTO_SPAWN_LLM_SERVER",
    "auto_spawn.tunnel": "MAXIM_AUTO_SPAWN_TUNNEL",
    "auto_spawn.port": "MAXIM_AUTO_SPAWN_PORT",
    "auto_spawn.timeout_s": "MAXIM_AUTO_SPAWN_TIMEOUT_S",
    "data.home": "MAXIM_DATA_HOME",
    "data.budget_gb": "MAXIM_DATA_BUDGET_GB",
}


_ABSORBED_ENV_VARS: frozenset[str] = frozenset(_FIELD_TO_ENV.values())
"""Module-level frozenset of every env var the schema absorbs. Used by
the once-per-startup deprecation INFO log (I-4 fold) — when a
``resolve_setting`` call hits an env-var present in this set, the
loader logs an INFO suggesting migration to ``config.json`` exactly
once across the process lifetime."""


_warned_envs: set[str] = set()
"""Module-level set of env-var names we have already emitted the
deprecation INFO for in this process. Cleared by
:func:`_reset_warned_envs` (test-only helper). Production code must
NOT clear this — the "fire once per startup" semantics are load-bearing
to keep the log from spamming per-lane-resolution dispatch."""


# ─────────────────────────────────────────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────────────────────────────────────────


def config_path() -> Path:
    """Return the platform-appropriate ``config.json`` path.

    Mirrors :func:`maxim.peer.config.peer_config_path` so all four
    declarative config files (``peer.yml``, ``mesh.yml``,
    ``profiles.yml``, ``config.json``) live in the same directory.
    """
    if platform.system() == "Windows":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
        return base / "maxim" / "config.json"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "maxim" / "config.json"


# ─────────────────────────────────────────────────────────────────────────────
# Env-var presence test (C-1 fold)
# ─────────────────────────────────────────────────────────────────────────────


def _env_is_set(name: str) -> bool:
    """Return True iff env var is set to a non-empty, non-whitespace value.

    Fold C-1: POSIX shells can ``export FOO=`` (empty string) and
    the result is still "present" to :func:`os.environ.get`. The
    2026-06-01 Mac Mini trigger was exactly a leaked-then-emptied env
    var silently shadowing ``config.json``. Empty-after-strip is
    treated as unset.
    """
    raw = os.environ.get(name)
    return raw is not None and raw.strip() != ""


# ─────────────────────────────────────────────────────────────────────────────
# Coercion table (C-2 fold)
# ─────────────────────────────────────────────────────────────────────────────


def _coerce_bool(raw: str, field_path: str) -> bool:
    """Parse a string env value into bool per the C-2 coercion table.

    Truthy: ``{"1", "true", "yes", "on"}`` (case-insensitive, post-strip).
    Falsy: ``{"0", "false", "no", "off"}``. Anything else raises
    :class:`ConfigurationError`. Empty string is rejected upstream by
    :func:`_env_is_set`.
    """
    normalized = raw.strip().lower()
    if normalized in _TRUTHY:
        return True
    if normalized in _FALSY:
        return False
    raise ConfigurationError(
        f"config: {field_path}: expected bool-like value (one of {sorted(_TRUTHY | _FALSY)}), got {raw!r}"
    )


def _coerce_int(raw: str, field_path: str, *, min_val: int | None = None, max_val: int | None = None) -> int:
    """Parse a string env value into int + range-validate per C-2."""
    try:
        value = int(raw.strip())
    except ValueError as e:
        raise ConfigurationError(f"config: {field_path}: expected int, got {raw!r}") from e
    if min_val is not None and value < min_val:
        raise ConfigurationError(f"config: {field_path}: value {value} below minimum {min_val}")
    if max_val is not None and value > max_val:
        raise ConfigurationError(f"config: {field_path}: value {value} above maximum {max_val}")
    return value


def _coerce_float(raw: str, field_path: str, *, min_val: float | None = None) -> float:
    """Parse a string env value into float + lower-bound check per C-2."""
    try:
        value = float(raw.strip())
    except ValueError as e:
        raise ConfigurationError(f"config: {field_path}: expected float, got {raw!r}") from e
    if min_val is not None and value < min_val:
        raise ConfigurationError(f"config: {field_path}: value {value} below minimum {min_val}")
    return value


def _coerce_enum(raw: str, field_path: str, valid: frozenset[str]) -> str:
    """Match a string env value against an enum set (case-insensitive, post-strip)."""
    normalized = raw.strip().lower()
    if normalized not in valid:
        raise ConfigurationError(f"config: {field_path}: expected one of {sorted(valid)}, got {raw!r}")
    return normalized


def _coerce_role(raw: str, field_path: str) -> str:
    """Coerce a role value. Auto-converts the legacy ``client`` name to ``peer``.

    The legacy term comes from ``runtime/leader_mode.py::RoleName`` which
    used ``client`` for what ``runtime/role.py`` calls ``peer``. The
    C3 unification (see config_unification.md) collapses both onto the
    ``role.py`` vocabulary; this coercion runs at load time to keep
    existing operator setups working.
    """
    normalized = raw.strip().lower()
    if normalized == "client":
        logger.warning(
            "config: %s='client' is the legacy leader_mode.py term; "
            "coerced to 'peer'. Update your config / env var to 'peer'.",
            field_path,
        )
        return "peer"
    if normalized in _VALID_ROLES:
        return normalized
    raise ConfigurationError(f"config: {field_path}: expected one of {sorted(_VALID_ROLES)}, got {raw!r}")


def _validate_api_key_ref(value: str, field_path: str) -> str:
    """Validate an api_key_ref value at load time per cross-confirmed I-3 + IM3 fold.

    Two modes ONLY: file path (starts with ``/`` or ``~``) or keyring
    URI (matches ``keyring:<service>:<account>``). Any other string is
    rejected with a fix-hint — the original draft's "inline plain
    string with deprecation WARNING" mode was identified by both
    review lenses as a footgun that lets ``maxim config set ... sk-abc``
    cheerfully write mode-0644 plaintext keys to disk.

    Empty string and ``None`` are both treated as unset (no ref).
    """
    if not value:
        return value
    if value.startswith("/") or value.startswith("~"):
        return value
    if value.startswith("keyring:"):
        if value.count(":") < 2:
            raise ConfigurationError(
                f"config: {field_path}: keyring URI must be 'keyring:<service>:<account>', got {value!r}"
            )
        return value
    raise ConfigurationError(
        f"config: {field_path}: must be a file path (starting with '/' or '~') "
        f"or a keyring URI ('keyring:<service>:<account>'). "
        f"Inline plaintext API keys are NOT supported — got {value!r}. "
        f"Fix: write the key to a mode-0600 file (e.g., ~/.config/maxim/api_key) "
        f"and reference it by path."
    )


def _coerce_for_field(raw: str, field_path: str) -> Any:
    """Dispatch coercion based on field path. Returns the typed value.

    Note: ``remote_api_key_ref`` env values are NOT validated against
    the path/keyring schema here. The cross-confirmed I-3/IM3 fold
    (inline-string rejected) applies to ``config.json`` writes and
    parse-time validation only — the legacy ``MAXIM_LANE_<TIER>_
    REMOTE_API_KEY`` env var legitimately holds an inline key value
    (that's its pre-C4 semantics). :func:`resolve_api_key_ref` handles
    all three forms (path, keyring URI, inline) for downstream
    consumers.
    """
    # Booleans
    if field_path in {
        "llm.enabled",
        "llm.auto_download",
        "cloud.enabled",
        "auto_spawn.llm_server",
        "auto_spawn.tunnel",
    }:
        return _coerce_bool(raw, field_path)
    # Integer fields with range constraints
    if field_path == "llm.n_ctx":
        return _coerce_int(raw, field_path, min_val=256)
    if field_path in {"cloud.max_lanes", "proxy.max_concurrent", "proxy.rate_limit_rpm"}:
        return _coerce_int(raw, field_path, min_val=0)
    if field_path == "auto_spawn.port":
        return _coerce_int(raw, field_path, min_val=1, max_val=65535)
    if field_path == "auto_spawn.timeout_s":
        return _coerce_int(raw, field_path, min_val=1)
    # Float fields
    if field_path == "cloud.session_budget_usd":
        return _coerce_float(raw, field_path, min_val=0.0)
    if field_path == "data.budget_gb":
        return _coerce_float(raw, field_path, min_val=0.0)
    if field_path.startswith("lanes.") and field_path.endswith(".timeout_s"):
        # llm_timeout_scalability.md Stage 2: strictly positive (matches
        # LaneTierConfig.__post_init__ validation). Zero or negative
        # would mean "give up immediately," which is never useful and
        # would brick the lane.
        value = _coerce_float(raw, field_path, min_val=0.0)
        if value <= 0.0:
            raise ConfigurationError(f"config: {field_path}: must be positive, got {value}")
        return value
    # Enum fields
    if field_path == "role":
        return _coerce_role(raw, field_path)
    if field_path == "llm.backend":
        return _coerce_enum(raw, field_path, _VALID_BACKENDS)
    if field_path == "cloud.redaction_policy":
        return _coerce_enum(raw, field_path, _VALID_REDACTION_POLICIES)
    # API key refs from env: pass through as-is (legacy semantics —
    # MAXIM_LANE_*_REMOTE_API_KEY holds the inline key directly).
    # config.json validation happens in _parse_lane_tier; CLI set_field
    # validation happens in _apply_field_to_section.
    if field_path.endswith(".remote_api_key_ref"):
        return raw.strip()
    # Plain strings (no transformation beyond strip for tidy values)
    return raw.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass walking helpers
# ─────────────────────────────────────────────────────────────────────────────


def _walk_dot_path(obj: Any, dot_path: str) -> Any:
    """Walk a dataclass via dot-separated field names. Returns the leaf value."""
    parts = dot_path.split(".")
    for part in parts:
        obj = getattr(obj, part)
    return obj


def _read_from_config(config: MaximConfig | None, field_path: str) -> Any | None:
    """Read a field value from the config dataclass.

    Returns ``None`` when the field would shadow a default — i.e., the
    config either is ``None`` or the value equals the default for that
    field. This lets the precedence chain distinguish "operator set
    this in config.json" from "this is the schema's default."
    """
    if config is None:
        return None
    try:
        actual = _walk_dot_path(config, field_path)
    except AttributeError:
        return None
    default = _walk_dot_path(MaximConfig(), field_path)
    if actual == default:
        return None
    return actual


def _builtin_default(field_path: str) -> Any:
    """Return the schema default for a field via dot path."""
    return _walk_dot_path(MaximConfig(), field_path)


# ─────────────────────────────────────────────────────────────────────────────
# Precedence chain (CR3 + C-1 + I-4 folds)
# ─────────────────────────────────────────────────────────────────────────────


def resolve_setting(
    field_path: str,
    cli_value: Any | None = None,
    *,
    config: MaximConfig | None = None,
) -> tuple[Any, str]:
    """Resolve a config field via the four-layer precedence chain.

    Returns ``(effective_value, source)`` where source is one of
    ``"cli" | "env" | "config" | "default"``.

    Precedence: CLI > env > config.json > builtin default.

    Logging:

    - **Convergence** (env set AND config sets same value): INFO log
      noting that ``config.json`` is ignored until env var is unset
      (CR3 fold).
    - **Divergence** (env set AND config sets different value): WARNING
      log naming both values (CR3 fold).
    - **Deprecation** (env set for an absorbed var): INFO log
      suggesting migration, fired exactly once per env var per startup
      via the module-level :data:`_warned_envs` set (I-4 fold).
    """
    if field_path not in _FIELD_TO_ENV:
        raise ConfigurationError(
            f"config: unknown field path {field_path!r}. Valid paths: {sorted(_FIELD_TO_ENV.keys())}"
        )

    if cli_value is not None:
        return cli_value, "cli"

    # 2026-06-04 fold: when callers don't pass an explicit ``config``,
    # auto-load via the cached :func:`get_config`. Pre-fold, ``_read_from_config(None, ...)``
    # silently returned ``None``, which made every ``resolve_setting(field_path)``
    # call (no kwarg) read as "config layer is empty" and fall through to the
    # schema default — even when ``config.json`` had a real operator value.
    # Symptom: ``maxim doctor`` showed contradictory rows like "llm.n_ctx:
    # using default (8192)" alongside "llm.n_ctx: 13312 [source=config.json]"
    # because some doctor helpers passed ``config=cfg`` and others didn't.
    # Now both paths converge on the same cached config.
    if config is None:
        try:
            config = get_config()
        except Exception:
            # If load_config raised (parse error, permission denied), preserve
            # the legacy fall-through-to-default behaviour rather than
            # propagating the error to every caller. The dedicated
            # ``check_resolved_config`` path surfaces config.json load failures
            # via its own ``fail`` row.
            config = None
    env_name = _FIELD_TO_ENV[field_path]
    config_value = _read_from_config(config, field_path)

    if _env_is_set(env_name):
        env_raw = os.environ[env_name]
        env_value = _coerce_for_field(env_raw, field_path)

        # CR3 fold: log on every layer interaction, not just mismatch.
        if config_value is not None:
            if config_value == env_value:
                logger.info(
                    "config: %s has source=env AND config.json sets the same "
                    "value (%r). config.json is ignored until env var is unset.",
                    field_path,
                    env_value,
                )
            else:
                logger.warning(
                    "config override: %s=%r (env) shadows %r (config.json)",
                    field_path,
                    env_value,
                    config_value,
                )

        # I-4 fold: once-per-startup deprecation INFO.
        if env_name in _ABSORBED_ENV_VARS and env_name not in _warned_envs:
            _warned_envs.add(env_name)
            logger.info(
                "config: %s is set. Consider migrating to config.json via "
                "`maxim config set %s <value>` (env vars deprecated in 1.1).",
                env_name,
                field_path,
            )

        return env_value, "env"

    if config_value is not None:
        return config_value, "config"

    return _builtin_default(field_path), "default"


def _reset_warned_envs() -> None:
    """Test-only helper: clear the deprecation-INFO dedupe set.

    Production code must NOT call this — the "fire once per startup"
    semantics are load-bearing to prevent per-request log spam in
    lane-resolution paths.
    """
    _warned_envs.clear()


def resolve_api_key_ref(ref: str | None) -> str | None:
    """Resolve a ``lanes.<tier>.remote_api_key_ref`` value to the actual
    key string.

    Two resolution modes per the cross-confirmed I-3 + IM3 fold (inline
    plaintext rejected at load time):

    - **File path** (starts with ``/`` or ``~``): read the file as text.
      Returns the stripped contents. ``None`` if the file is missing or
      unreadable. ``maxim doctor`` reports the missing/wrong-mode case
      eagerly.
    - **Keyring URI** (``keyring:<service>:<account>``): resolves via
      :mod:`keyring`. ``None`` if the keyring package is not installed
      or the entry is absent — does not raise (per I-1 fold —
      ``maxim doctor`` / ``maxim config get`` must remain runnable when
      keyring is missing).

    Returns ``None`` when ``ref`` is ``None`` / empty / unresolvable.
    Callers that need a loud failure on missing key should branch on
    the ``None`` return and raise their own typed exception (e.g.,
    :class:`maxim.models.language.types.BackendAuthFailed` at lane
    backend construction time).
    """
    if not ref:
        return None
    if ref.startswith("/") or ref.startswith("~"):
        try:
            expanded = Path(ref).expanduser()
            if not expanded.is_file():
                return None
            return expanded.read_text(encoding="utf-8").strip() or None
        except OSError as e:
            # Log the failure category, not the path — paths can encode
            # operator filesystem layout that's not for the log file.
            # The value the function received is a REFERENCE (file path),
            # not the secret itself; the cross-confirmed I-3/IM3 fold
            # forbids inline secrets in this position. CodeQL's data-
            # flow analysis flags this because the surrounding function
            # name contains "api_key", but `ref` is metadata, not key.
            logger.debug(
                "resolve_api_key_ref: failed to read file-mode reference: %s",
                type(e).__name__,
            )
            return None
    if ref.startswith("keyring:"):
        try:
            import keyring as _keyring  # type: ignore[import-untyped]
        except ImportError:
            logger.warning(
                "config: lanes.*.remote_api_key_ref uses keyring but the "
                "keyring package is not installed (run `pip install keyring`)."
            )
            return None
        parts = ref.split(":", 2)
        if len(parts) < 3:
            return None
        _, service, account = parts
        try:
            return _keyring.get_password(service, account) or None
        except Exception as e:
            # Log the failure category, not the URI — service+account
            # can be operator metadata. `ref` is a REFERENCE, not the
            # key itself.
            logger.warning("config: keyring lookup failed: %s", type(e).__name__)
            return None
    # Inline key (legacy env-var semantics): when the value isn't a
    # file path or keyring URI, treat it as an already-resolved key.
    # config.json values are validated at load time to reject this
    # shape per the cross-confirmed I-3/IM3 fold; this branch only
    # fires for values that came from env. Returning the value as-is
    # preserves the pre-C4 MAXIM_LANE_<TIER>_REMOTE_API_KEY semantics
    # where the env var directly holds the inline key.
    return ref.strip() or None


# ─────────────────────────────────────────────────────────────────────────────
# Schema validation + JSON parsing (I-6 + IM4 folds tied to _format_version)
# ─────────────────────────────────────────────────────────────────────────────


def _check_format_version(data: dict[str, Any]) -> tuple[str, bool]:
    """Read ``_format_version`` from the parsed config and classify it.

    Returns ``(version_string, is_future_minor)`` where
    ``is_future_minor`` is True when the file declares a newer minor
    version than the loader knows but the major version matches —
    triggering the I-6/IM4 forward-compat tolerance for unknown keys.

    Raises :class:`ConfigurationError` on a future major (the breaking
    case per CC1).

    **Post-implementation Architecture #1 fold:** the version-string
    extraction routes through :func:`maxim.utils.format_version.check_format_version`,
    the canonical helper per CC1. This wraps it with the
    major.minor classification needed for the I-6/IM4 unknown-key
    forward-compat rule. The pre-fold code rolled its own version
    extraction, silently bypassing CC1's fail-loud-on-stale-conflict
    semantics — exactly the "two parallel implementations of a
    freeze-critical contract" pattern CLAUDE.md targets.
    """
    from maxim.utils.format_version import (
        FORMAT_VERSION as _FV_DEFAULT,
        LEGACY_VERSION,
        check_format_version,
    )

    # Type-check the raw field BEFORE routing through the canonical
    # helper. A non-string value (e.g., ``_format_version: 1.0`` as a
    # number) is most likely an operator typo and should fail loudly —
    # the canonical helper at maxim.utils.format_version treats non-
    # string values as "missing" (returns LEGACY_VERSION). For
    # config.json specifically we want stricter behavior because the
    # file is CLI-canonical and the CLI never writes a non-string
    # version.
    raw_field = data.get("_format_version")
    if raw_field is not None and (not isinstance(raw_field, str) or not raw_field):
        raise ConfigurationError(f"config.json: _format_version must be a non-empty string, got {raw_field!r}")

    # check_format_version returns a string version or LEGACY_VERSION
    # when the field is absent. We treat absent as "same as loader"
    # for our minor-version classification (the C1 contract says
    # empty/missing config returns defaults — same as a 1.0 file with
    # no payload).
    raw = check_format_version(data, "config_json", log=logger)
    if raw == LEGACY_VERSION:
        # Pre-1.0 file or absent field. Treat as same-major.
        raw = _FV_DEFAULT

    # Parse major.minor; tolerate trailing patch/identifiers via splitting.
    try:
        major_str, minor_str, *_ = raw.split(".") + ["0"]
        major = int(major_str)
        minor = int(minor_str)
    except (ValueError, IndexError) as e:
        raise ConfigurationError(
            f"config.json: _format_version {raw!r} is not a valid major.minor version string"
        ) from e

    known_major, known_minor = (int(p) for p in CONFIG_FORMAT_VERSION.split(".")[:2])

    if major > known_major:
        raise ConfigurationError(
            f"config.json: _format_version {raw!r} is a future major version "
            f"(loader knows {CONFIG_FORMAT_VERSION}). Major version bumps are "
            f"breaking per CC1 — upgrade Maxim or downgrade the config file."
        )
    if major < known_major:
        # Older major — for 1.x there is no 0.x case to handle; the
        # file simply predates the loader's known version. The format-
        # version check_format_version helper in utils/ would warn
        # once; here we accept and treat unknown keys strictly because
        # the file is implicitly older not newer.
        return raw, False

    is_future_minor = minor > known_minor
    return raw, is_future_minor


def _parse_lane_tier(
    section: Any,
    path: str,
    *,
    tolerate_unknown: bool,
) -> LaneTierConfig:
    """Parse a single ``lanes.<tier>`` section into :class:`LaneTierConfig`.

    The escape-hatch ``extra`` dict gathers any unknown keys; per CC3
    path (a) the values must be JSON-serializable but the dict has no
    typed schema. ``tolerate_unknown`` is unused here (the whole point
    of path (a) is that unknown keys go to ``extra``); it's accepted
    for signature symmetry with :func:`_parse_typed_section`.
    """
    del tolerate_unknown  # always tolerant for escape-hatch sections
    if section is None:
        return LaneTierConfig()
    if not isinstance(section, dict):
        raise ConfigurationError(f"config.json: {path} must be a mapping, got {type(section).__name__}")

    declared = {"remote_url", "remote_model", "remote_api_key_ref", "timeout_s", "placement"}
    extra: dict[str, Any] = {k: v for k, v in section.items() if k not in declared}

    remote_url = section.get("remote_url")
    remote_model = section.get("remote_model")
    remote_api_key_ref = section.get("remote_api_key_ref")
    timeout_s = section.get("timeout_s")

    for fname, fval in (
        ("remote_url", remote_url),
        ("remote_model", remote_model),
        ("remote_api_key_ref", remote_api_key_ref),
    ):
        if fval is not None and not isinstance(fval, str):
            raise ConfigurationError(f"config.json: {path}.{fname} must be a string or null, got {type(fval).__name__}")

    if timeout_s is not None:
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
            raise ConfigurationError(
                f"config.json: {path}.timeout_s must be a number or null, got {type(timeout_s).__name__}"
            )
        # LaneTierConfig.__post_init__ rejects ≤ 0; surface here at parse
        # time with a clearer JSON-context message before construction.
        if timeout_s <= 0:
            raise ConfigurationError(
                f"config.json: {path}.timeout_s must be positive, got {timeout_s}. "
                f"Use null (or omit) for the backend default."
            )
        timeout_s = float(timeout_s)

    if isinstance(remote_api_key_ref, str):
        remote_api_key_ref = _validate_api_key_ref(remote_api_key_ref, f"{path}.remote_api_key_ref")

    placement = _parse_lane_tier_placement(section.get("placement"), f"{path}.placement")

    return LaneTierConfig(
        remote_url=remote_url,
        remote_model=remote_model,
        remote_api_key_ref=remote_api_key_ref,
        timeout_s=timeout_s,
        placement=placement,
        extra=extra,
    )


def _parse_lane_tier_placement(raw: Any, path: str) -> tuple[LaneTierPlacement, ...]:
    """Parse a ``lanes.<tier>.placement`` array into ``LaneTierPlacement``s.

    ``None`` / absent → ``()`` (derive from the legacy remote_* fields). Each
    entry is a mapping with a required ``origin`` (``local``/``cloud``/``peer``)
    and optional ``model`` / ``url`` / ``api_key_ref`` / ``timeout_s``. Origin
    value + types + api_key_ref are validated here with JSON context; field
    coherence is validated at the producer boundary when the entry is resolved
    into a runtime ``ProviderPlacement``.
    """
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ConfigurationError(f"config.json: {path} must be a list, got {type(raw).__name__}")

    entries: list[LaneTierPlacement] = []
    for i, item in enumerate(raw):
        ipath = f"{path}[{i}]"
        if not isinstance(item, dict):
            raise ConfigurationError(f"config.json: {ipath} must be a mapping, got {type(item).__name__}")

        origin = item.get("origin")
        if not isinstance(origin, str) or origin not in _VALID_PLACEMENT_ORIGINS:
            raise ConfigurationError(
                f"config.json: {ipath}.origin must be one of {sorted(_VALID_PLACEMENT_ORIGINS)}, got {origin!r}"
            )

        for fname in ("model", "url", "api_key_ref"):
            fval = item.get(fname)
            if fval is not None and not isinstance(fval, str):
                raise ConfigurationError(
                    f"config.json: {ipath}.{fname} must be a string or null, got {type(fval).__name__}"
                )

        timeout_s = item.get("timeout_s")
        if timeout_s is not None:
            if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
                raise ConfigurationError(
                    f"config.json: {ipath}.timeout_s must be a number or null, got {type(timeout_s).__name__}"
                )
            if timeout_s <= 0:
                raise ConfigurationError(f"config.json: {ipath}.timeout_s must be positive, got {timeout_s}.")
            timeout_s = float(timeout_s)

        api_key_ref = item.get("api_key_ref")
        if isinstance(api_key_ref, str):
            api_key_ref = _validate_api_key_ref(api_key_ref, f"{ipath}.api_key_ref")

        declared = _LANE_TIER_PLACEMENT_DECLARED_FIELDS
        entry_extra = {k: v for k, v in item.items() if k not in declared}
        entries.append(
            LaneTierPlacement(
                origin=origin,
                model=item.get("model"),
                url=item.get("url"),
                api_key_ref=api_key_ref,
                timeout_s=timeout_s,
                extra=entry_extra,
            )
        )
    return tuple(entries)


def _parse_typed_section(
    section: Any,
    section_path: str,
    section_cls: type,
    *,
    tolerate_unknown: bool,
) -> Any:
    """Generic shape-frozen section parser.

    Tied to ``_format_version`` per I-6 + IM4 fold:
    ``tolerate_unknown=True`` (future-minor case) logs WARNING and
    drops unknown keys; ``tolerate_unknown=False`` (same-version case)
    raises :class:`ConfigurationError` to surface typos.
    """
    if section is None:
        return section_cls()
    if not isinstance(section, dict):
        raise ConfigurationError(f"config.json: {section_path} must be a mapping, got {type(section).__name__}")

    known_fields = {f.name for f in fields(section_cls)}
    unknown = set(section.keys()) - known_fields
    if unknown:
        if tolerate_unknown:
            logger.warning(
                "config.json: %s has unknown key(s) %s — tolerated because "
                "_format_version is a future minor. Loader will ignore them.",
                section_path,
                sorted(unknown),
            )
        else:
            raise ConfigurationError(
                f"config.json: {section_path} has unknown key(s) "
                f"{sorted(unknown)}; valid keys are {sorted(known_fields)}"
            )

    kwargs: dict[str, Any] = {}
    for fld in fields(section_cls):
        if fld.name not in section:
            continue
        raw = section[fld.name]
        path = f"{section_path}.{fld.name}"
        kwargs[fld.name] = _coerce_json_field(raw, path, fld.type)

    return section_cls(**kwargs)


def _coerce_json_field(raw: Any, field_path: str, expected_type: Any) -> Any:
    """Validate and coerce a JSON-parsed value against the dataclass field type.

    This is the JSON path — env-var coercion is in
    :func:`_coerce_for_field`. JSON already gives us typed values
    (bool, int, float, str, None), so the work here is mostly range +
    enum + api_key_ref validation.
    """
    # Treat the type as a string annotation; ``get_type_hints`` would
    # work but introduces ForwardRef churn at module import time.
    type_str = str(expected_type)

    if raw is None:
        if "| None" in type_str or "Optional" in type_str:
            return None
        raise ConfigurationError(f"config.json: {field_path}: null not allowed")

    # bool — note: ``int`` matches ``"int"`` substring, so check bool
    # FIRST and use a word-boundary-ish match. ``"bool"`` only ever
    # appears in bool / bool | None annotations.
    if expected_type is bool or "bool" in type_str:
        if not isinstance(raw, bool):
            raise ConfigurationError(f"config.json: {field_path}: expected bool, got {type(raw).__name__}")
        return raw

    # float — check BEFORE int because ``"int"`` is a substring of
    # ``"int | None"`` but NOT of ``"float | None"``. Also Python's
    # ``isinstance(x, float)`` rejects ints, so use the explicit
    # int-or-float test below.
    if expected_type is float or "float" in type_str:
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            raise ConfigurationError(f"config.json: {field_path}: expected float, got {type(raw).__name__}")
        return _range_check_float(float(raw), field_path)

    # int with range
    if expected_type is int or "int" in type_str:
        if not isinstance(raw, int) or isinstance(raw, bool):
            raise ConfigurationError(f"config.json: {field_path}: expected int, got {type(raw).__name__}")
        return _range_check_int(raw, field_path)

    # str + str | None + Literal[...] (Literal annotations are all
    # string-valued enums in this schema)
    if expected_type is str or "str" in type_str or type_str.startswith("Literal["):
        if not isinstance(raw, str):
            raise ConfigurationError(f"config.json: {field_path}: expected string, got {type(raw).__name__}")
        # Enum subsets
        if field_path == "role":
            return _coerce_role(raw, field_path)
        if field_path == "llm.backend":
            return _coerce_enum(raw, field_path, _VALID_BACKENDS)
        if field_path == "cloud.redaction_policy":
            return _coerce_enum(raw, field_path, _VALID_REDACTION_POLICIES)
        return raw

    raise ConfigurationError(
        f"config.json: {field_path}: unhandled type {expected_type!r} (loader bug — please report)"
    )


def _range_check_int(value: int, field_path: str) -> int:
    """Apply per-field range constraints to a JSON-parsed int."""
    if field_path == "llm.n_ctx" and value < 256:
        raise ConfigurationError(f"config.json: {field_path}={value} below minimum 256")
    if (
        field_path
        in {
            "cloud.max_lanes",
            "proxy.max_concurrent",
            "proxy.rate_limit_rpm",
        }
        and value < 0
    ):
        raise ConfigurationError(f"config.json: {field_path}={value} below minimum 0")
    if field_path == "auto_spawn.port" and not (1 <= value <= 65535):
        raise ConfigurationError(f"config.json: {field_path}={value} must be in 1..65535")
    if field_path == "auto_spawn.timeout_s" and value < 1:
        raise ConfigurationError(f"config.json: {field_path}={value} below minimum 1")
    return value


def _range_check_float(value: float, field_path: str) -> float:
    """Apply per-field range constraints to a JSON-parsed float."""
    if field_path in {"cloud.session_budget_usd", "data.budget_gb"} and value < 0:
        raise ConfigurationError(f"config.json: {field_path}={value} below minimum 0.0")
    return value


def _parse_lanes_section(section: Any, tolerate_unknown: bool) -> LanesConfigSection:
    """Parse the ``lanes`` section.

    Tier names are FROZEN per the existing lane-tier-names invariant
    (``large``, ``medium``, ``small``). Unknown tier names always
    raise :class:`ConfigurationError`, regardless of
    ``_format_version`` — even a future-minor writer cannot introduce
    a new tier without a major bump (per the LanesConfigSection CC3
    docstring).
    """
    del tolerate_unknown  # tier names are stricter than the version gate
    if section is None:
        return LanesConfigSection()
    if not isinstance(section, dict):
        raise ConfigurationError(f"config.json: lanes must be a mapping, got {type(section).__name__}")

    unknown_tiers = set(section.keys()) - _VALID_TIER_NAMES
    if unknown_tiers:
        raise ConfigurationError(
            f"config.json: lanes has unknown tier name(s) {sorted(unknown_tiers)}; "
            f"valid tier names are {sorted(_VALID_TIER_NAMES)} (FROZEN per the "
            f"lane-tier-names invariant). Adding a new tier requires a major "
            f"_format_version bump."
        )

    return LanesConfigSection(
        large=_parse_lane_tier(section.get("large"), "lanes.large", tolerate_unknown=False),
        medium=_parse_lane_tier(section.get("medium"), "lanes.medium", tolerate_unknown=False),
        small=_parse_lane_tier(section.get("small"), "lanes.small", tolerate_unknown=False),
    )


def _parse_config_dict(data: dict[str, Any]) -> MaximConfig:
    """Validate and parse a JSON-decoded dict into :class:`MaximConfig`.

    Tied to ``_format_version`` per I-6 + IM4 fold — the same-version
    case rejects unknown keys at every level; the future-minor case
    tolerates them. Future major raises :class:`ConfigurationError`.
    """
    if not isinstance(data, dict):
        raise ConfigurationError(f"config.json: root must be a JSON object, got {type(data).__name__}")

    version, is_future_minor = _check_format_version(data)

    known_top_level = {f.name for f in fields(MaximConfig)}
    unknown_top = set(data.keys()) - known_top_level
    if unknown_top:
        if is_future_minor:
            logger.warning(
                "config.json: unknown top-level key(s) %s tolerated because "
                "_format_version=%r is a future minor (loader knows %s).",
                sorted(unknown_top),
                version,
                CONFIG_FORMAT_VERSION,
            )
        else:
            raise ConfigurationError(
                f"config.json: unknown top-level key(s) {sorted(unknown_top)}; valid keys are {sorted(known_top_level)}"
            )

    # Parse role at top level
    role_raw = data.get("role")
    if role_raw is None:
        role: str | None = None
    elif isinstance(role_raw, str):
        role = _coerce_role(role_raw, "role")
    else:
        raise ConfigurationError(f"config.json: role must be a string or null, got {type(role_raw).__name__}")

    llm = _parse_typed_section(data.get("llm"), "llm", LLMConfigSection, tolerate_unknown=is_future_minor)
    lanes = _parse_lanes_section(data.get("lanes"), tolerate_unknown=is_future_minor)
    cloud = _parse_typed_section(data.get("cloud"), "cloud", CloudConfigSection, tolerate_unknown=is_future_minor)
    proxy = _parse_typed_section(data.get("proxy"), "proxy", ProxyConfigSection, tolerate_unknown=is_future_minor)
    auto_spawn = _parse_typed_section(
        data.get("auto_spawn"),
        "auto_spawn",
        AutoSpawnConfigSection,
        tolerate_unknown=is_future_minor,
    )
    data_section = _parse_typed_section(data.get("data"), "data", DataConfigSection, tolerate_unknown=is_future_minor)

    return MaximConfig(
        _format_version=version,
        role=role,  # type: ignore[arg-type]
        llm=llm,
        lanes=lanes,
        cloud=cloud,
        proxy=proxy,
        auto_spawn=auto_spawn,
        data=data_section,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public loader entry point + lazy singleton + IM5 auto-migration shim
# ─────────────────────────────────────────────────────────────────────────────


_loaded_config: MaximConfig | None = None
_loaded_config_lock = threading.Lock()
_loaded_config_path: Path | None = None

# IM5 fold: track whether the peer.yml → config.json migration has been
# attempted in this process. Idempotent — once attempted, subsequent
# load_config() calls skip the check. Tests reset via reset_config_cache().
_migration_attempted: bool = False


def _maybe_migrate_from_peer_yml(target: Path) -> None:
    """IM5 fold auto-migration: write a minimal ``config.json`` from
    ``peer.yml`` on first startup when conditions match.

    Triggers iff:
    - ``config.json`` does NOT exist at ``target``
    - ``peer.yml`` exists at the canonical location
    - cloudflared config does NOT exist (preserves the legitimate
      leader-with-stale-peer.yml case per the C3 seven-rank order;
      a leader machine should never have its role auto-flipped to peer)

    Writes ``config.json`` with ``role=peer`` and
    ``lanes.large.{remote_url, remote_api_key_ref, remote_model}``
    populated from peer.yml fields. peer.yml is left in place — never
    deleted by the shim. Subsequent startups skip the migration
    because ``config.json`` now exists.

    **Post-implementation Executor I2 fold:** the
    ``_migration_attempted`` flag is set AFTER the work completes
    (either successfully OR after a definitive "not applicable" check
    like config.json present, peer.yml absent, cloudflared present).
    The pre-fold ordering set the flag BEFORE the work, so a
    transient OSError on first write (permission denied, disk full,
    etc.) would permanently skip migration for the rest of the
    process — even after the operator fixed permissions and re-ran.
    Now retry storms are bounded by the "not applicable" check (which
    happens before any write attempt) and recoverable write failures
    leave the flag unset so the next load_config call can try again.

    Failures (write permission, etc.) degrade gracefully: log WARNING
    and continue. The caller's load_config will still produce a valid
    defaults-only MaximConfig.
    """
    global _migration_attempted
    if _migration_attempted:
        return

    if target.is_file():
        # Not applicable — config.json already exists. Set the flag so
        # we don't repeatedly stat the file on every load_config call.
        _migration_attempted = True
        return

    # Read peer.yml; bail if absent
    try:
        from maxim.peer.config import read_peer_config
    except Exception:
        _migration_attempted = True  # not applicable
        return
    try:
        peer = read_peer_config()
    except Exception as e:
        logger.debug("config: peer.yml read failed during migration check: %s", e)
        _migration_attempted = True  # not applicable
        return
    if peer is None:
        _migration_attempted = True  # not applicable
        return

    # Cloudflared check (preserves leader case)
    try:
        from maxim.runtime.role import _cloudflared_config_exists

        if _cloudflared_config_exists() is not None:
            logger.debug(
                "config: peer.yml present but cloudflared config also present; "
                "skipping auto-migration (preserves leader case per IM5 fold)."
            )
            _migration_attempted = True  # definitive — leader case
            return
    except Exception:
        pass

    # Build the minimal MaximConfig and write
    try:
        # Local import to avoid the writer module pulling load_config back
        # into a cycle at module-init time.
        from maxim.runtime.config_writer import write_config
    except Exception as e:
        logger.warning("config: auto-migration import failed (%s); skipping", e)
        # Do NOT set the flag — import errors might be transient (e.g.,
        # a partial install). Next load_config retries.
        return

    migrated = MaximConfig(
        role="peer",
        lanes=LanesConfigSection(
            large=LaneTierConfig(
                remote_url=peer.url,
                remote_model=peer.model,
                # peer.yml stores the API key inline (legacy mode-0600
                # file). The migration writes a file-path reference to
                # the canonical api_key file so the config.json mode
                # stays 0644. Operators who hand-edited their peer.yml
                # to a different path can override post-migration via
                # `maxim config set lanes.large.remote_api_key_ref`.
                remote_api_key_ref=_peer_api_key_ref_for_migration(peer.api_key),
            ),
        ),
    )

    try:
        written = write_config(migrated, path=target)
    except Exception as e:
        logger.warning(
            "config: auto-migration write failed (%s); peer.yml stays as the "
            "active routing source for now (compat read). Next load_config "
            "will retry the migration.",
            e,
        )
        # Per Executor I2 fold: leave flag unset so transient write
        # failures (permission denied, disk full) can recover on a
        # subsequent retry.
        return

    _migration_attempted = True  # success
    logger.info(
        "config: auto-migrated peer.yml → %s (peer.yml preserved for 1.x compat, retired in 2.0)",
        written,
    )


def _peer_api_key_ref_for_migration(api_key: str) -> str | None:
    """Write the peer.yml inline API key to ``~/.config/maxim/api_key``
    mode-0600 and return a file-path reference.

    If the canonical api_key file already exists with a DIFFERENT key,
    leave it alone and return None — the operator has a key file path
    already and we should not silently clobber it. A subsequent
    ``maxim config set lanes.large.remote_api_key_ref`` can wire the
    explicit reference.
    """
    if not api_key:
        return None
    try:
        api_key_path = config_path().parent / "api_key"
        if api_key_path.exists():
            try:
                existing = api_key_path.read_text(encoding="utf-8").strip()
            except OSError:
                existing = ""
            if existing and existing != api_key.strip():
                # Existing different key — don't clobber. Operator
                # follow-up needed.
                logger.warning(
                    "config: peer.yml api_key differs from existing %s; "
                    "leaving the file alone. Run `maxim config set "
                    "lanes.large.remote_api_key_ref <path>` to wire a "
                    "reference manually.",
                    api_key_path,
                )
                return None
            # Same content already — just reference it
            return str(api_key_path)
        from maxim.utils.atomic_io import atomic_write_secret

        api_key_path.parent.mkdir(parents=True, exist_ok=True)
        # Post-implementation Executor C3 fold (umask race): on a
        # brand-new file ``atomic_write_secret`` inherits the
        # caller's umask, which is typically 0o022 — exposing the
        # API key as world-readable during the brief window between
        # the write and the subsequent chmod. Tighten the umask to
        # 0o077 around the write so the file is created at 0o600
        # from the start. Restore the original umask afterward.
        original_umask = os.umask(0o077)
        try:
            atomic_write_secret(str(api_key_path), api_key.strip() + "\n")
        finally:
            os.umask(original_umask)
        # Defensive chmod for the resave case (an existing file at a
        # wider mode wouldn't be re-tightened by the umask change).
        try:
            api_key_path.chmod(0o600)
        except OSError:
            pass
        return str(api_key_path)
    except Exception as e:
        logger.warning(
            "config: failed to write api_key file during migration (%s); "
            "skipping remote_api_key_ref (operator can wire it later)",
            e,
        )
        return None


def load_config(path: Path | None = None) -> MaximConfig:
    """Read ``config.json`` from disk, validate, return :class:`MaximConfig`.

    Returns a defaults-only :class:`MaximConfig` when the file is
    missing or empty. Raises :class:`ConfigurationError` on JSON parse
    errors, schema violations, or future-major versions.

    Triggers the IM5 fold auto-migration shim (peer.yml → config.json)
    on first call when conditions match — see
    :func:`_maybe_migrate_from_peer_yml`.

    Does NOT cache. For the lazily-cached production singleton use
    :func:`get_config`.
    """
    effective_path = path if path is not None else config_path()

    # IM5 fold auto-migration runs BEFORE the read so first-startup
    # peer.yml-only setups get their data written to config.json and
    # subsequently picked up by the rest of this function. Idempotent —
    # the shim sets _migration_attempted and short-circuits on
    # subsequent calls.
    _maybe_migrate_from_peer_yml(effective_path)

    if not effective_path.is_file():
        return MaximConfig()

    try:
        raw = effective_path.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigurationError(f"config.json: failed to read {effective_path}: {e}") from e

    stripped = raw.strip()
    if not stripped:
        return MaximConfig()

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as e:
        raise ConfigurationError(
            f"config.json: invalid JSON at {effective_path} (line {e.lineno}, col {e.colno}): {e.msg}"
        ) from e

    return _parse_config_dict(data)


def get_config(path: Path | None = None) -> MaximConfig:
    """Return the process-wide lazy-cached :class:`MaximConfig`.

    The first call loads from ``path`` (or :func:`config_path` default)
    and caches; subsequent calls return the cached instance unless
    ``path`` differs from the cached one (then re-loads + re-caches
    under that new path).

    For test isolation use :func:`reset_config_cache` from the
    ``_isolate_config_json_env`` fixture in ``tests/conftest.py``.
    """
    global _loaded_config, _loaded_config_path
    with _loaded_config_lock:
        target = path if path is not None else config_path()
        if _loaded_config is None or _loaded_config_path != target:
            _loaded_config = load_config(target)
            _loaded_config_path = target
        return _loaded_config


def reset_config_cache() -> None:
    """Test-only helper: clear the lazy-loaded config singleton + the
    IM5 fold auto-migration idempotency flag.

    Pairs with :func:`_reset_warned_envs` for test isolation. Production
    code must not call this — the singleton is process-wide by design.
    """
    global _loaded_config, _loaded_config_path, _migration_attempted
    with _loaded_config_lock:
        _loaded_config = None
        _loaded_config_path = None
        _migration_attempted = False


__all__ = [
    "CONFIG_FORMAT_VERSION",
    "AutoSpawnConfigSection",
    "CloudConfigSection",
    "DataConfigSection",
    "LaneTierConfig",
    "LanesConfigSection",
    "LLMConfigSection",
    "MaximConfig",
    "ProxyConfigSection",
    "config_path",
    "get_config",
    "load_config",
    "reset_config_cache",
    "resolve_api_key_ref",
    "resolve_setting",
]
