"""Maxim API key generation and storage for securing peer access.

When a leader exposes its spawned llama-cpp-server via a Cloudflare tunnel,
anyone who knows the hostname could hit the endpoint. This module generates
and persists a Bearer token that the server validates via its --api_key flag.

Layout:
- Key file:  ~/.config/maxim/api_key  (POSIX; Windows uses %APPDATA%\\maxim\\api_key)
- Format:    single-line URL-safe random string, 43 chars (256-bit entropy)
- Perms:     0600 on POSIX; Windows relies on user-profile ACLs
"""

from __future__ import annotations

import logging
import os
import platform
import secrets
from pathlib import Path


KEY_BYTES = 32  # 256-bit random key → 43 char base64url-ish string

# The module holds NAMED keys under the one maxim config dir. "api_key" is the
# mesh key (peer → leader inference auth; every parameter default preserves its
# original single-key behaviour byte-identically). "console_token" is the
# console credential (maxim serve bearer auth) — a SEPARATE file on purpose:
# one credential must not grant both inference and console admin
# (docs/plans/console_tunnel_hardening.md, decision A1). Its mxc_ prefix makes
# a leaked token self-identifying and secret-scanner-teachable, GitHub-style.
MESH_KEY_NAME = "api_key"
CONSOLE_KEY_NAME = "console_token"
CONSOLE_TOKEN_PREFIX = "mxc_"


def key_file_path(name: str = MESH_KEY_NAME) -> Path:
    """Return the platform-appropriate path for the named key file."""
    if platform.system() == "Windows":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
        return base / "maxim" / name
    # POSIX (Linux, macOS, WSL)
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "maxim" / name


def generate_key(prefix: str = "") -> str:
    """Return a fresh 256-bit URL-safe random key string, optionally prefixed."""
    return prefix + secrets.token_urlsafe(KEY_BYTES)


def key_exists(name: str = MESH_KEY_NAME) -> bool:
    return key_file_path(name).is_file()


def read_key(name: str = MESH_KEY_NAME) -> str | None:
    """Return the stored key, or None if not set."""
    path = key_file_path(name)
    if not path.is_file():
        return None
    try:
        content = path.read_text().strip()
        return content or None
    except OSError:
        return None


def write_key(key: str, name: str = MESH_KEY_NAME) -> Path:
    """Persist a key to disk with restrictive permissions on POSIX.

    Creates parent directories if missing. Overwrites any existing key.
    Returns the written path. The mesh key keeps this original writer
    (migrating it is out of PR-2 scope); NEW named keys go through
    ``_write_key_secret`` below, which rides the canonical
    ``atomic_write_secret``.
    """
    path = key_file_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(key + "\n")
    # 0600 on POSIX — owner read/write only
    if platform.system() != "Windows":
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    return path


def _write_key_secret(key: str, name: str) -> Path:
    """Persist a named key via the canonical credential writer (0600 + atomic).

    ``atomic_write_secret`` PRESERVES a pre-existing mode but cannot invent one
    for a brand-new file (its docstring assigns that to the caller), so the
    0600 chmod after the write is load-bearing for first writes; rewrites then
    preserve it.
    """
    from maxim.utils.atomic_io import atomic_write_secret

    path = key_file_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_secret(str(path), key + "\n")
    if platform.system() != "Windows":
        try:
            os.chmod(path, 0o600)
        except OSError:
            logging.getLogger(__name__).warning("could not chmod 0600 on %s", path, exc_info=True)
    return path


def ensure_key() -> str:
    """Return the existing mesh key if present, else generate, save, and return a new one."""
    existing = read_key()
    if existing:
        return existing
    key = generate_key()
    write_key(key)
    return key


def rotate_key() -> str:
    """Generate a new mesh key, overwrite the file, return the new value."""
    key = generate_key()
    write_key(key)
    return key


# ─── console token (bearer credential for `maxim serve`) ───────────────────


def read_console_token() -> str | None:
    """Return the stored console token, or None if not set."""
    return read_key(CONSOLE_KEY_NAME)


def ensure_console_token() -> str:
    """Return the existing console token, else generate (mxc_-prefixed), save, return."""
    existing = read_console_token()
    if existing:
        return existing
    token = generate_key(prefix=CONSOLE_TOKEN_PREFIX)
    _write_key_secret(token, CONSOLE_KEY_NAME)
    return token


def rotate_console_token() -> str:
    """Generate a new console token, overwrite the file, return the new value.

    Every device that stored the old token is logged out on its next request —
    rotation is the console's revocation story (hardening plan, decision A5).
    """
    token = generate_key(prefix=CONSOLE_TOKEN_PREFIX)
    _write_key_secret(token, CONSOLE_KEY_NAME)
    return token


def truncate_for_display(key: str, *, keep: int = 6) -> str:
    """Return 'abcdef…XYZ123' for safer logging (first + last `keep` chars)."""
    if len(key) <= keep * 2 + 1:
        return key
    return f"{key[:keep]}…{key[-keep:]}"


# ─── shell snippet rendering ───────────────────────────────────────────────

ENV_VAR = "MAXIM_LANE_LARGE_REMOTE_API_KEY"


def render_snippets(key: str) -> dict[str, str]:
    """Return a dict of {shell_name: copy-paste snippet} for exporting the key.

    Shells covered:
    - bash_zsh: Linux, macOS, WSL — most common default
    - fish:     fish shell (POSIX)
    - powershell: Windows + PowerShell Core (cross-platform)
    - cmd:      Windows command prompt
    """
    return {
        "bash_zsh": _bash_snippet(key),
        "fish": _fish_snippet(key),
        "powershell": _powershell_snippet(key),
        "cmd": _cmd_snippet(key),
    }


def _bash_snippet(key: str) -> str:
    return (
        f"# bash / zsh — current session only:\n"
        f'export {ENV_VAR}="{key}"\n'
        f"\n"
        f"# Recommended: persist via peer config (no shell edits needed):\n"
        f"# maxim peer connect <leader-url> --key {key[:8]}...\n"
    )


def _fish_snippet(key: str) -> str:
    return f'# fish — universal (persists across sessions automatically):\nset -Ux {ENV_VAR} "{key}"\n'


def _powershell_snippet(key: str) -> str:
    return (
        f"# PowerShell (Windows, Linux, macOS) — current session:\n"
        f'$env:{ENV_VAR} = "{key}"\n'
        f"\n"
        f"# Persist in your profile (run once):\n"
        f'Add-Content $PROFILE "`n$env:{ENV_VAR} = `"{key}`""\n'
    )


def _cmd_snippet(key: str) -> str:
    return (
        f":: Windows cmd — persists across sessions (machine-wide for your user):\n"
        f'setx {ENV_VAR} "{key}"\n'
        f"\n"
        f":: Current session only:\n"
        f"set {ENV_VAR}={key}\n"
    )


def format_all_snippets(key: str) -> str:
    """Return a single human-readable block with all shell variants."""
    snippets = render_snippets(key)
    out = []
    labels = {
        "bash_zsh": "━━━ bash / zsh (Linux, macOS, WSL) ━━━",
        "fish": "━━━ fish ━━━",
        "powershell": "━━━ PowerShell (Windows, cross-platform) ━━━",
        "cmd": "━━━ Windows cmd ━━━",
    }
    for shell, snippet in snippets.items():
        out.append(labels.get(shell, shell))
        out.append(snippet)
    return "\n".join(out)


__all__ = [
    "ENV_VAR",
    "KEY_BYTES",
    "key_file_path",
    "generate_key",
    "key_exists",
    "read_key",
    "write_key",
    "ensure_key",
    "rotate_key",
    "truncate_for_display",
    "render_snippets",
    "format_all_snippets",
]
