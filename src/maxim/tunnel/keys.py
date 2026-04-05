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

import os
import platform
import secrets
from pathlib import Path


KEY_BYTES = 32  # 256-bit random key → 43 char base64url-ish string


def key_file_path() -> Path:
    """Return the platform-appropriate key file path."""
    if platform.system() == "Windows":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
        return base / "maxim" / "api_key"
    # POSIX (Linux, macOS, WSL)
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "maxim" / "api_key"


def generate_key() -> str:
    """Return a fresh 256-bit URL-safe random key string."""
    return secrets.token_urlsafe(KEY_BYTES)


def key_exists() -> bool:
    return key_file_path().is_file()


def read_key() -> str | None:
    """Return the stored key, or None if not set."""
    path = key_file_path()
    if not path.is_file():
        return None
    try:
        content = path.read_text().strip()
        return content or None
    except OSError:
        return None


def write_key(key: str) -> Path:
    """Persist a key to disk with restrictive permissions on POSIX.

    Creates parent directories if missing. Overwrites any existing key.
    Returns the written path.
    """
    path = key_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(key + "\n")
    # 0600 on POSIX — owner read/write only
    if platform.system() != "Windows":
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    return path


def ensure_key() -> str:
    """Return the existing key if present, else generate, save, and return a new one."""
    existing = read_key()
    if existing:
        return existing
    key = generate_key()
    write_key(key)
    return key


def rotate_key() -> str:
    """Generate a new key, overwrite the file, return the new value."""
    key = generate_key()
    write_key(key)
    return key


def truncate_for_display(key: str, *, keep: int = 6) -> str:
    """Return 'abcdef…XYZ123' for safer logging (first + last `keep` chars)."""
    if len(key) <= keep * 2 + 1:
        return key
    return f"{key[:keep]}…{key[-keep:]}"


# ─── shell snippet rendering ───────────────────────────────────────────────

ENV_VAR = "MAXIM_LANE_INFER_REMOTE_API_KEY"


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
        f'# bash / zsh — current session:\n'
        f'export {ENV_VAR}="{key}"\n'
        f'\n'
        f'# Persist in your shell rc file (pick one):\n'
        f"echo 'export {ENV_VAR}=\"{key}\"' >> ~/.bashrc\n"
        f"echo 'export {ENV_VAR}=\"{key}\"' >> ~/.zshrc\n"
    )


def _fish_snippet(key: str) -> str:
    return (
        f'# fish — universal (persists across sessions automatically):\n'
        f'set -Ux {ENV_VAR} "{key}"\n'
    )


def _powershell_snippet(key: str) -> str:
    return (
        f'# PowerShell (Windows, Linux, macOS) — current session:\n'
        f'$env:{ENV_VAR} = "{key}"\n'
        f'\n'
        f'# Persist in your profile (run once):\n'
        f'Add-Content $PROFILE "`n$env:{ENV_VAR} = `"{key}`""\n'
    )


def _cmd_snippet(key: str) -> str:
    return (
        f':: Windows cmd — persists across sessions (machine-wide for your user):\n'
        f'setx {ENV_VAR} "{key}"\n'
        f'\n'
        f":: Current session only:\n"
        f'set {ENV_VAR}={key}\n'
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
