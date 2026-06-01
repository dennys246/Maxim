"""User-defined Maxim profile loader (L2 of leader_ux_profile_management.md).

Reads ``~/.config/maxim/profiles.yml`` at import time and merges its
``profiles:`` entries into ``_BUILTIN_PROFILES``, ``_PROFILE_ALIASES``,
and ``LLM_MODELS`` so that any GGUF on HuggingFace (or any local path)
becomes a first-class Maxim profile.

Path convention mirrors ``peer/config.py::peer_config_path`` — XDG on
POSIX, ``%APPDATA%\\maxim\\`` on Windows. Same dir as ``peer.yml`` and
``mesh.yml`` per CLAUDE.md's "declarative config layer" invariant.

Schema (FROZEN at 1.0 — see leader_ux_profile_management.md):

    profiles:
      my-qwen-32b-q5:
        backend: llama_cpp                 # llama_cpp | pytorch
        prompt_style: chatml               # chatml | mistral_instruct |
                                           # llama3_instruct | llama2_chat |
                                           # phi3 | phi | gemma
        # ONE OF download: OR local_path: REQUIRED
        download:
          hf_repo: bartowski/Qwen2.5-32B-Instruct-GGUF
          hf_file: Qwen2.5-32B-Instruct-Q5_K_M.gguf
        # local_path: ~/models/custom.gguf

        # OPTIONAL
        n_ctx: 32768                       # default 8192
        stop: ["<|im_end|>"]               # default per prompt_style
        aliases: [qwen32q5]                # default []
        arch:                              # default None (coarse VRAM est.)
          n_layers: 64
          n_kv_heads: 8
          head_dim: 128
          kv_type_bytes: 2
          weights_gb: 23.0

Precedence on name collision: user profile WINS over built-in, with a
WARNING logged at load time. This matches the codebase's avoid-silent-
no-op discipline (the operator opted in by writing the file; silently
falling back to the built-in would mask their intent).

Loading is intentionally synchronous + import-time. Live reload is out
of scope at 1.0 (mirrors peer.yml / mesh.yml).

**Schema versioning at 1.0 (CC3-analog):** the YAML schema is the
shape-frozen analog of CC3 option (b) — there is NO ``extra:`` escape
hatch on user profiles by design (the typed validation enum surface
would be diluted, and unknown keys are how typos like ``prompt_styel``
silently land). The file does not carry ``_format_version`` at 1.0
because it is primarily operator-authored (per CC1: that field marks
files Maxim writes). When ``maxim model add`` rewrites the file, the
schema version stays implicit ``1``. Post-1.0 evolution:

- Adding an OPTIONAL field with a default is non-breaking.
- Adding a REQUIRED field is breaking and requires a top-level
  ``version: 2`` key + a migration path through the loader.

The validation pipeline is intentionally atomic: every profile entry
is staged first, then applied in one pass. A single bad profile blocks
every user profile from merging so operators can't end up in a
"half-loaded, half-not" state.
"""

from __future__ import annotations

import logging
import os
import platform
from pathlib import Path
from typing import Any

import yaml

from maxim.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Schema constants (FROZEN at 1.0)
# ─────────────────────────────────────────────────────────────────────────────

_VALID_BACKENDS: frozenset[str] = frozenset({"llama_cpp", "pytorch"})

# Drawn from the existing prompt_style values used by built-in profiles.
# A new value must be wired into the chat-template dispatch layer
# (currently in models/language/prompts.py) before being added here.
_VALID_PROMPT_STYLES: frozenset[str] = frozenset(
    {
        "chatml",
        "mistral_instruct",
        "llama2_chat",
        "llama3_instruct",
        "phi",
        "phi3",
        "gemma",
    }
)

_DEFAULT_N_CTX: int = 8192

# Default stop tokens by prompt_style — mirrors what the built-in
# profiles declare. Used only when the user omits `stop:`.
_DEFAULT_STOPS: dict[str, list[str]] = {
    "chatml": ["<|im_end|>", "<|endoftext|>"],
    "mistral_instruct": ["</s>"],
    "llama2_chat": ["</s>"],
    "llama3_instruct": ["<|eot_id|>"],
    "phi": ["<|endoftext|>"],
    "phi3": ["<|end|>", "<|endoftext|>"],
    "gemma": ["<end_of_turn>"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────────────────────────────────────────


def profiles_config_path() -> Path:
    """Return the platform-appropriate ``profiles.yml`` path.

    Mirrors :func:`maxim.peer.config.peer_config_path` so all three
    declarative config files (peer.yml, mesh.yml, profiles.yml) live in
    the same directory.
    """
    if platform.system() == "Windows":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
        return base / "maxim" / "profiles.yml"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "maxim" / "profiles.yml"


# ─────────────────────────────────────────────────────────────────────────────
# Schema validation
# ─────────────────────────────────────────────────────────────────────────────


def _validate_arch(arch: Any, profile_name: str) -> dict[str, Any]:
    """Validate an optional ``arch:`` block. Returns the dict as-is on
    success; raises :class:`ConfigurationError` on shape problems."""
    if arch is None:
        return {}
    if not isinstance(arch, dict):
        raise ConfigurationError(
            f"profile {profile_name!r}: 'arch' must be a mapping with "
            f"n_layers/n_kv_heads/head_dim/kv_type_bytes/weights_gb."
        )
    required = ("n_layers", "n_kv_heads", "head_dim", "kv_type_bytes", "weights_gb")
    missing = [f for f in required if f not in arch]
    if missing:
        raise ConfigurationError(
            f"profile {profile_name!r}: 'arch' is missing required fields: "
            f"{missing}. See tests/unit/test_profile_metadata.py for the "
            f"field documentation. Omit the entire 'arch:' block to skip "
            f"VRAM estimation entirely."
        )
    return dict(arch)


def _validate_and_normalize(profile_name: str, raw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one user profile entry and return ``(profile_dict, download_dict)``.

    ``profile_dict`` is suitable for insertion into ``_BUILTIN_PROFILES``.
    ``download_dict`` is suitable for insertion into ``LLM_MODELS`` when
    ``download:`` is specified — ``None`` (empty dict) when ``local_path:``
    is used instead.

    Raises :class:`ConfigurationError` with a clear message for every
    schema violation. The error message names the offending profile and
    the specific field so the operator can fix it in one pass.
    """
    if not isinstance(raw, dict):
        raise ConfigurationError(
            f"profile {profile_name!r}: each entry under 'profiles:' must be a mapping. Got {type(raw).__name__}."
        )

    backend = raw.get("backend")
    if not backend or backend not in _VALID_BACKENDS:
        raise ConfigurationError(
            f"profile {profile_name!r}: 'backend' is required and must be "
            f"one of {sorted(_VALID_BACKENDS)}. Got {backend!r}."
        )

    prompt_style = raw.get("prompt_style")
    if not prompt_style or prompt_style not in _VALID_PROMPT_STYLES:
        raise ConfigurationError(
            f"profile {profile_name!r}: 'prompt_style' is required and must "
            f"be one of {sorted(_VALID_PROMPT_STYLES)}. Got {prompt_style!r}."
        )

    download = raw.get("download")
    local_path = raw.get("local_path")
    if download is not None and local_path is not None:
        raise ConfigurationError(
            f"profile {profile_name!r}: specify exactly one of 'download:' or 'local_path:', not both."
        )
    if download is None and local_path is None:
        raise ConfigurationError(
            f"profile {profile_name!r}: must specify either 'download:' "
            f"(with hf_repo + hf_file) or 'local_path:' (a path to a "
            f"local GGUF file)."
        )

    n_ctx = raw.get("n_ctx", _DEFAULT_N_CTX)
    # isinstance(True, int) is True in Python — guard against the bool
    # special case so `n_ctx: true` doesn't silently land as n_ctx=1.
    if isinstance(n_ctx, bool) or not isinstance(n_ctx, int) or n_ctx <= 0:
        raise ConfigurationError(f"profile {profile_name!r}: 'n_ctx' must be a positive integer; got {n_ctx!r}.")

    stop = raw.get("stop")
    if stop is None:
        stop = list(_DEFAULT_STOPS.get(prompt_style, []))
    elif not isinstance(stop, list) or not all(isinstance(s, str) for s in stop):
        raise ConfigurationError(f"profile {profile_name!r}: 'stop' must be a list of strings.")

    aliases = raw.get("aliases", [])
    if not isinstance(aliases, list) or not all(isinstance(a, str) for a in aliases):
        raise ConfigurationError(f"profile {profile_name!r}: 'aliases' must be a list of strings.")

    arch = _validate_arch(raw.get("arch"), profile_name)

    # Build the normalized profile dict matching _BUILTIN_PROFILES shape.
    # `model` auto-derives from profile_name; `model_base` auto-derives
    # from hf_repo basename when download is used, or from profile_name
    # otherwise. Both are internal — not user-facing fields.
    profile_dict: dict[str, Any] = {
        "backend": backend,
        "model": profile_name,
        "prompt_style": prompt_style,
        "stop": stop,
        "n_ctx": n_ctx,
    }
    if arch:
        profile_dict["arch"] = arch

    download_dict: dict[str, Any] = {}
    if download is not None:
        if not isinstance(download, dict):
            raise ConfigurationError(f"profile {profile_name!r}: 'download' must be a mapping with hf_repo + hf_file.")
        hf_repo = download.get("hf_repo")
        hf_file = download.get("hf_file")
        if not hf_repo or not isinstance(hf_repo, str):
            raise ConfigurationError(
                f"profile {profile_name!r}: 'download.hf_repo' is required "
                f"and must be a string (e.g., 'bartowski/Some-Model-GGUF')."
            )
        if not hf_file or not isinstance(hf_file, str):
            raise ConfigurationError(
                f"profile {profile_name!r}: 'download.hf_file' is required and must be a string (the GGUF filename)."
            )
        # model_base is the filename stem (with .gguf + quant suffix
        # stripped). This is load-bearing: build_model_path constructs
        # the on-disk lookup as `{model_base}.{quant}.gguf` with a
        # case-insensitive fallback to `{model_base}-{quant.lower()}.gguf`.
        # Deriving from the repo basename instead (the original L2
        # behavior) made the lookup target diverge from the downloader's
        # actual filename and silently broke every user-added profile
        # at first `--llm` use. The pre-merge architecture review caught
        # this; see leader_ux_profile_management.md fold round.
        profile_dict["model_base"] = _derive_model_base_from_filename(hf_file)
        download_dict = {
            "description": f"User-defined profile {profile_name!r} (from {hf_repo})",
            "size_gb": None,
            "expected_bytes": None,
            "quantization": _infer_quantization_from_filename(hf_file),
            "url": f"https://huggingface.co/{hf_repo}/resolve/main/{hf_file}",
            "filename": hf_file,
        }
    else:
        # local_path: must be a string. Expansion happens at use time.
        if not isinstance(local_path, str):
            raise ConfigurationError(f"profile {profile_name!r}: 'local_path' must be a string.")
        # Auto-derive model_base from the profile name for local-path
        # profiles. The download path is bypassed entirely.
        expanded_local_path = str(Path(local_path).expanduser())
        profile_dict["model_base"] = profile_name
        # model_path is the explicit on-disk path consulted by
        # load_llm_config (config.py) before falling back to
        # build_model_path. Without this, local_path entries silently
        # routed through build_model_path's `{model_base}.{quant}.gguf`
        # construction — which never resolves to the user's actual file.
        # Pre-merge executor review caught this; see fold notes.
        profile_dict["model_path"] = expanded_local_path
        download_dict = {
            "description": f"User-defined profile {profile_name!r} (local file)",
            "size_gb": None,
            "expected_bytes": None,
            "quantization": None,
            "url": None,  # No download — local file
            "filename": None,
            "local_path": expanded_local_path,
        }

    return profile_dict, download_dict


_QUANT_SUFFIXES: tuple[str, ...] = (
    "Q2_K",
    "Q3_K_S",
    "Q3_K_M",
    "Q3_K_L",
    "Q4_0",
    "Q4_K_S",
    "Q4_K_M",
    "Q5_0",
    "Q5_K_S",
    "Q5_K_M",
    "Q6_K",
    "Q8_0",
    "F16",
    "F32",
)


def _infer_quantization_from_filename(filename: str) -> str | None:
    """Best-effort substring match for the quant suffix. Returns None
    if no recognized suffix is found — the download path tolerates that."""
    upper = filename.upper()
    for suffix in _QUANT_SUFFIXES:
        if suffix in upper:
            return suffix
    return None


def _derive_model_base_from_filename(filename: str) -> str:
    """Derive the ``model_base`` value used by ``build_model_path`` to
    locate the downloaded file on disk.

    The on-disk lookup in ``build_model_path`` constructs the canonical
    filename as ``{model_base}.{quant}.gguf`` with a case-insensitive
    fallback to ``{model_base}-{quant.lower()}.gguf``. To make either
    pattern match the GGUF that the downloader writes, ``model_base``
    must equal the filename with the ``.gguf`` extension and quant
    suffix stripped.

    Example::

        "Qwen2.5-32B-Instruct-Q5_K_M.gguf"  →  "Qwen2.5-32B-Instruct"
        "model-q4_k_m.gguf"                  →  "model"

    If no recognized quant suffix is present, returns the filename with
    only the ``.gguf`` extension stripped. The case-insensitive fallback
    in ``build_model_path`` still accommodates non-canonical suffix
    casing as long as the stem matches.

    This fixes the bug where the original loader set ``model_base`` from
    the HF repo basename (e.g. ``"Qwen2.5-32B-Instruct-GGUF"``), which
    ``build_model_path`` could never resolve back to the downloaded
    ``Qwen2.5-32B-Instruct-Q5_K_M.gguf`` file on disk.
    """
    stem = filename
    # Strip .gguf (case-insensitive)
    lower = stem.lower()
    if lower.endswith(".gguf"):
        stem = stem[: -len(".gguf")]
    # Strip recognized quant suffix (case-insensitive, with separator
    # being '.' or '-' or none).
    stem_upper = stem.upper()
    for suffix in _QUANT_SUFFIXES:
        for sep in (".", "-", ""):
            tail = f"{sep}{suffix}"
            if stem_upper.endswith(tail):
                return stem[: -len(tail)]
    return stem


# ─────────────────────────────────────────────────────────────────────────────
# Top-level load + merge
# ─────────────────────────────────────────────────────────────────────────────


def load_user_profiles(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Parse the user profiles file and return a mapping of
    ``profile_name -> raw entry dict``.

    Returns an empty dict when:
      - the file does not exist,
      - the file is empty / null,
      - the file's ``profiles:`` key is missing or null.

    Raises :class:`ConfigurationError` with file-location context when
    the YAML itself is unparseable. Per-profile schema validation
    happens later in :func:`apply_user_profiles`.
    """
    target = path if path is not None else profiles_config_path()
    if not target.is_file():
        return {}
    try:
        content = target.read_text()
    except OSError as exc:
        raise ConfigurationError(f"could not read profiles file at {target}: {exc}") from exc
    if not content.strip():
        return {}
    try:
        # yaml.safe_load is the only sanctioned entry — yaml.load is
        # unsafe (allows arbitrary Python object construction). The CI
        # grep in .github/workflows/test.yml enforces this.
        data = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"profiles file at {target} is not valid YAML: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigurationError(
            f"profiles file at {target}: top-level must be a mapping with a 'profiles:' key. Got {type(data).__name__}."
        )
    profiles = data.get("profiles")
    if profiles is None:
        return {}
    if not isinstance(profiles, dict):
        raise ConfigurationError(
            f"profiles file at {target}: 'profiles:' must be a mapping "
            f"of profile_name -> profile_dict. Got {type(profiles).__name__}."
        )
    return profiles


def apply_user_profiles(
    builtin_profiles: dict[str, dict[str, Any]],
    profile_aliases: dict[str, str],
    llm_models: dict[str, dict[str, Any]],
    *,
    path: Path | None = None,
) -> None:
    """Load + validate + merge user profiles into the three core dicts.

    User profiles win on slug / alias collision with built-ins; each
    collision is logged at WARNING level so the operator sees it.

    All three dicts are mutated in place. The merge is atomic in spirit
    (validation runs first across the whole file; on any failure no
    dict is touched) — implemented by collecting validated entries
    into a staging dict and applying them at the end.
    """
    raw_entries = load_user_profiles(path=path)
    if not raw_entries:
        return

    # Validate everything first so a partial application can't happen.
    staged_profiles: dict[str, dict[str, Any]] = {}
    staged_downloads: dict[str, dict[str, Any]] = {}
    staged_aliases: dict[str, str] = {}
    aliases_seen: dict[str, str] = {}  # alias -> profile_name (for intra-user collision detection)

    for profile_name, raw in raw_entries.items():
        if not isinstance(profile_name, str) or not profile_name.strip():
            raise ConfigurationError(
                f"profile names under 'profiles:' must be non-empty strings; got {profile_name!r}."
            )
        if "/" in profile_name or ":" in profile_name:
            raise ConfigurationError(
                f"profile name {profile_name!r} must not contain '/' or ':' "
                f"(those characters break HF cache pathnames)."
            )
        profile_dict, download_dict = _validate_and_normalize(profile_name, raw)
        staged_profiles[profile_name] = profile_dict
        staged_downloads[profile_name] = download_dict

        # Intra-user alias collision: two user profiles claiming the
        # same alias is a hard error per the plan doc — operator must
        # resolve before we can pick one.
        for alias in raw.get("aliases", []) or []:
            normalized = str(alias).strip().lower().replace("_", "-").replace(" ", "")
            if not normalized:
                continue
            if normalized in aliases_seen and aliases_seen[normalized] != profile_name:
                raise ConfigurationError(
                    f"alias {normalized!r} is claimed by both "
                    f"{aliases_seen[normalized]!r} and {profile_name!r}. "
                    f"Each alias must map to exactly one user profile."
                )
            aliases_seen[normalized] = profile_name
            staged_aliases[normalized] = profile_name

    # Apply: user wins, log collisions.
    for profile_name, profile_dict in staged_profiles.items():
        if profile_name in builtin_profiles:
            logger.warning(
                "User profile %r overrides bundled profile (loaded from %s).",
                profile_name,
                path or profiles_config_path(),
            )
        builtin_profiles[profile_name] = profile_dict

    for profile_name, download_dict in staged_downloads.items():
        llm_models[profile_name] = download_dict

    for alias, profile_name in staged_aliases.items():
        if alias in profile_aliases and profile_aliases[alias] != profile_name:
            logger.warning(
                "User alias %r (-> %r) overrides bundled alias (-> %r).",
                alias,
                profile_name,
                profile_aliases[alias],
            )
        profile_aliases[alias] = profile_name
