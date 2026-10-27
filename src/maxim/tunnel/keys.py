"""Maxim API key generation and storage for securing peer access.

A leader serves inference through its LeaderProxy (:8099), reachable on the
LAN and through a Cloudflare tunnel. This module generates and persists the
Bearer token the proxy requires; the leader's loopback llama-cpp-server gets
the same key through its environment (``API_KEY``), never argv.

Layout:
- Key file:  ~/.config/maxim/api_key  (POSIX; Windows uses %APPDATA%\\maxim\\api_key)
- A non-leader's spawned llama-cpp-server is keyed too, with its own per-host
  ``llm_server_key`` beside it (#1314; see ``ensure_local_server_key``).
- Format:    single-line URL-safe random string, 43 chars (256-bit entropy)
- Perms:     0600 on POSIX; Windows relies on user-profile ACLs
"""

from __future__ import annotations

import contextlib
import logging
import os
import platform
import secrets
import stat
from collections.abc import Callable, Iterator
from pathlib import Path

from maxim.exceptions import KeyFileUnusable, LeaderRefusal


KEY_BYTES = 32  # 256-bit random key → 43 char base64url-ish string

# The module holds NAMED keys under the one maxim config dir. "api_key" is the
# mesh key (peer → leader inference auth; every parameter default preserves its
# original single-key behaviour byte-identically). "console_token" is the
# console credential (maxim serve bearer auth) — a SEPARATE file on purpose:
# one credential must not grant both inference and console admin
# (docs/plans/deferred/console_tunnel_hardening.md, decision A1). Its mxc_ prefix makes
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
    """Whether anything is at the key path. Only a missing file is False: a path that cannot be stat'd, or is not a
    regular file, is True, so a caller reports it instead of treating the key as absent (#1311). Never raises; never
    reads through ``Path.is_file``, which raises on Python <=3.13 and swallows on 3.14 (#1312)."""
    try:
        os.stat(key_file_path(name))
    except FileNotFoundError:
        return False
    except (OSError, ValueError):
        return True
    return True


def read_key(name: str = MESH_KEY_NAME) -> str | None:
    """Return the stored key, or None if it is missing or unusable in any way (never raises). A leading BOM is
    stripped, as the leader's strict reader does, so a peer never sends a different bearer than the leader accepts."""
    try:
        raw, _mode = _read_key_file(key_file_path(name))
    except LeaderKeyError:
        return None
    return raw.lstrip("\ufeff").strip() or None


def key_file_state(name: str = MESH_KEY_NAME) -> tuple[str, str | None]:
    """``("missing", None)``, ``("unusable", reason)`` or ``("ok", None)`` for the named key file (never raises).

    For callers that must tell an absent key (generate one; "no key set") from one that is there but cannot be used
    (refuse; report the reason): an unusable key must not be advised away as a mismatch (#1311), and an ensure never
    overwrites one (#1402). An empty file reports ``unusable`` here, though an ensure regenerates it.
    """
    try:
        raw, _mode = _read_key_file(key_file_path(name))
    except LeaderKeyMissing:
        return "missing", None
    except LeaderKeyError as e:
        return "unusable", e.reason
    if not raw.lstrip("\ufeff").strip():
        return "unusable", _EMPTY_KEY_REASON
    return "ok", None


_EMPTY_KEY_REASON = "key file is empty"


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
    """Persist a named key via the canonical credential writer.

    ``atomic_write_secret`` owns the whole mode story since the 2026-09-04
    review fold: 0600 from fd creation on first write (the secret never sits
    umask-wide, .tmp included), exact-mode preservation on rewrites.
    """
    from maxim.utils.atomic_io import atomic_write_secret

    path = key_file_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_secret(str(path), key + "\n")
    return path


def ensure_key() -> str:
    """Return the mesh key, generating one only when the file is missing or empty (#1402).

    A file that exists but cannot be used raises :class:`~maxim.exceptions.KeyFileUnusable`, never an overwrite.
    """
    return _ensure_named_key(MESH_KEY_NAME, lambda key: write_key(key))


def rotate_key() -> str:
    """Generate a new mesh key, overwrite the file, return the new value."""
    key = generate_key()
    with _key_creation_lock(key_file_path()):
        write_key(key)
    return key


def _key_unusable_fix(path: Path, name: str) -> str:
    """How to repair an unusable key file. Never "rotate": a rotation keeps the unusable mode (#1402 design pass)."""
    holders = (
        "every peer then needs `maxim peer key set <key>`"
        if name == MESH_KEY_NAME
        else "every signed-in device must pair or sign in again"
    )
    repair = (
        f"Fix the file ({path})"
        if platform.system() == "Windows"
        else f"Fix the file (`chmod 600 {path}` if it is unreadable; move it aside if it is a directory or not a key)"
    )
    return f"{repair}, or remove it so the next start generates a new one ({holders})."


def _existing_key(name: str, path: Path) -> str | None:
    """The key in the named file; None when it is missing or empty (both generate); KeyFileUnusable otherwise."""
    try:
        raw, _mode = _read_key_file(path)
    except LeaderKeyMissing:
        return None
    except LeaderKeyError as e:
        # Plain KeyFileUnusable: a console token is not a leader key, and this is not a LeaderRefusal.
        raise KeyFileUnusable(path=path, reason=e.reason, fix_hint=_key_unusable_fix(path, name)) from e
    key = raw.lstrip("\ufeff").strip()
    if not key:
        logging.getLogger(__name__).info("key file is empty; generating a new key: %s", path)
        return None
    return key


@contextlib.contextmanager
def _key_creation_lock(path: Path, refusal: Callable[[Path, str], KeyFileUnusable] | None = None) -> Iterator[None]:
    """The per-file lock every creator and rotator in this module holds, so an ensure never writes over a key another
    process has just created. ``peer key set``, the peer connect writer and the config-loader migration write the
    mesh key without it (atomically, so never half a file); routing them through it is #1402's follow-up."""
    from maxim.utils.process_lock import LockContended, file_lock

    try:
        with file_lock(path.with_name(path.name + ".lock"), timeout_s=_LOCAL_KEY_LOCK_TIMEOUT_S):
            yield
    except LockContended as e:
        if refusal is not None:
            raise refusal(path, "another process holds the key-creation lock") from e
        raise KeyFileUnusable(
            path=path,
            reason="another process holds the key-creation lock",
            fix_hint="Another maxim process is creating or rotating this key; retry once it finishes.",
        ) from e


def _ensure_named_key(
    name: str,
    write: Callable[[str], object],
    *,
    prefix: str = "",
    read: Callable[[Path], str | None] | None = None,
    refusal: Callable[[Path, str], KeyFileUnusable] | None = None,
) -> str:
    """The named key, created when (and only when) its file is missing or empty: the ONE ensure for key files.

    ``read`` returns the key, None for a missing or empty file, or raises the key's typed refusal for any other
    unusable file (default: :func:`_existing_key`); ``refusal`` builds that refusal for a creation failure (default: a
    plain :class:`~maxim.exceptions.KeyFileUnusable`). Read first without a lock (a valid key in a read-only directory
    still works). Creation happens under the key's lock with a re-read inside it (another process may have just
    written it), an empty file is unlinked first so the new file is created fresh rather than written over the old
    one's mode, and the value returned is the one read back from disk, so concurrent first starts agree (#1314, #1402).
    """
    path = key_file_path(name)
    read = read or (lambda p: _existing_key(name, p))
    refusal = refusal or (lambda p, reason: KeyFileUnusable(path=p, reason=reason, fix_hint=_key_unusable_fix(p, name)))
    key = read(path)
    if key is not None:
        return key
    try:
        with _key_creation_lock(path, refusal):
            if read(path) is None:
                path.unlink(missing_ok=True)
                write(generate_key(prefix=prefix))
    except OSError as e:
        raise refusal(path, f"cannot create key file ({type(e).__name__})") from e
    key = read(path)
    if key is None:
        raise refusal(path, "key file is missing or empty after it was written")
    return key


# ─── leader key (fail-closed read for the leader role) ─────────────────────

# A generated key is 43 chars; a shorter one is a typo or a placeholder.
MIN_LEADER_KEY_CHARS = 16

_LEADER_KEY_FIX = "Run `maxim tunnel key rotate` on the leader, then `maxim peer key set <key>` on each peer."


class LeaderKeyError(KeyFileUnusable, LeaderRefusal):
    """The leader role has no usable mesh key, so the leader refuses to start.

    A leader serves inference and admin endpoints on every interface; without a
    key it would serve them to anyone who can reach it. ``read_key`` returns
    None for every failure (its non-leader callers want that), so the leader
    path reads through :func:`require_leader_key` instead, and every way the
    key can be unusable is a distinct ``reason`` naming the resolved ``path``.
    """

    def __init__(self, *, path: Path, reason: str, fix_hint: str = _LEADER_KEY_FIX) -> None:
        super().__init__(path=path, reason=reason, fix_hint=fix_hint, message=f"leader key unusable ({reason}): {path}")


class LeaderKeyMissing(LeaderKeyError):
    """No key file at the path: the one :class:`LeaderKeyError` a caller may treat as "no key" (``oasis serve`` then
    needs ``--insecure``). Only ``FileNotFoundError`` is missing; a path that cannot be stat'd or opened, a parent that
    is not a directory, or something other than a regular file is an unusable key, never an absent one (#1311)."""

    def __init__(self, *, path: Path) -> None:
        super().__init__(path=path, reason="no key file")


# A generated key is 43 characters; a key file larger than this is not a key.
_MAX_KEY_FILE_BYTES = 64 * 1024


def _read_key_file(path: Path) -> tuple[str, int]:
    """The text and permission bits of the regular file at ``path``, read through ONE file descriptor.

    Opened, fstat'd and read through the same descriptor, so the file cannot be swapped (for a FIFO, which would hang
    startup, or for a world-readable file after the mode check) between the check and the read. Raises
    :class:`LeaderKeyMissing` or :class:`LeaderKeyError`, never a raw ``OSError`` (#1311, #1312).
    """
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(path, flags)
    except FileNotFoundError as e:
        raise LeaderKeyMissing(path=path) from e
    except (OSError, ValueError) as e:
        # Which layer refused: a parent that cannot be searched means the path cannot even be stat'd.
        try:
            os.stat(path)
        except (OSError, ValueError):
            raise LeaderKeyError(path=path, reason=f"cannot stat key file ({type(e).__name__})") from e
        raise LeaderKeyError(path=path, reason=f"cannot read key file ({type(e).__name__})") from e
    try:
        try:
            st = os.fstat(fd)
        except OSError as e:
            raise LeaderKeyError(path=path, reason=f"cannot stat key file ({type(e).__name__})") from e
        if not stat.S_ISREG(st.st_mode):
            raise LeaderKeyError(path=path, reason="key path is not a regular file")
        data = b""
        try:
            while len(data) <= _MAX_KEY_FILE_BYTES:  # a network filesystem may return a short read
                chunk = os.read(fd, _MAX_KEY_FILE_BYTES + 1 - len(data))
                if not chunk:
                    break
                data += chunk
        except OSError as e:
            raise LeaderKeyError(path=path, reason=f"cannot read key file ({type(e).__name__})") from e
        if len(data) > _MAX_KEY_FILE_BYTES:
            raise LeaderKeyError(path=path, reason="key file is too large to be a key")
        try:
            return data.decode("utf-8"), stat.S_IMODE(st.st_mode)
        except UnicodeDecodeError as e:
            raise LeaderKeyError(path=path, reason=f"cannot read key file ({type(e).__name__})") from e
    finally:
        os.close(fd)


def require_leader_key() -> str:
    """Return the mesh key for a leader, or raise :class:`LeaderKeyError`.

    Refuses a missing, unreadable, empty, short, non-ASCII or whitespace-bearing
    key, and on POSIX a key file group- or world-readable (the host may have
    other users). The path is resolved from ``XDG_CONFIG_HOME``/``HOME`` like
    every other key read, so a service started under a different home fails
    here, loudly, instead of serving without auth.
    """
    path = key_file_path(MESH_KEY_NAME)
    raw, mode = _read_key_file(path)  # LeaderKeyMissing for no file; LeaderKeyError for any unusable one
    return _validated_key(path, raw, mode, LeaderKeyError, chmod_fix=f"Run `chmod 600 {path}` on the leader.")


def _validated_key(path: Path, raw: str, mode: int, error: type[LeaderKeyError], *, chmod_fix: str) -> str:
    """The key in ``raw`` (a key file read with :func:`_read_key_file`), or ``error`` naming why it is unusable: on
    POSIX a file group- or world-readable, then an empty, non-ASCII, whitespace-bearing or short key."""
    if platform.system() != "Windows" and mode & 0o077:
        raise error(path=path, reason=f"key file mode is {mode & 0o777:o}, readable by other users", fix_hint=chmod_fix)
    key = raw.lstrip("\ufeff").strip()
    if not key:
        raise error(path=path, reason="key file is empty")
    if not key.isascii() or any(c.isspace() for c in key):
        raise error(path=path, reason="key contains non-ASCII or whitespace characters")
    if len(key) < MIN_LEADER_KEY_CHARS:
        raise error(path=path, reason=f"key is shorter than {MIN_LEADER_KEY_CHARS} characters")
    return key


# ─── the local server key (a non-leader's spawned llama-cpp-server, #1314) ────────────────────────────

LOCAL_SERVER_KEY_NAME = "llm_server_key"
# How long a first start waits for another process creating the key (two sims, a console and a sim).
_LOCAL_KEY_LOCK_TIMEOUT_S = 30.0


def _local_key_fix(path: Path) -> str:
    if platform.system() == "Windows":
        return (
            f"Fix the file ({path}), or remove it so the next start generates a new one; if %APPDATA%\\maxim "
            "cannot be written, fix its permissions."
        )
    return (
        f"Fix the file (`chmod 600 {path}`), or remove it so the next start generates a new one; if the config "
        "directory cannot be written, set XDG_CONFIG_HOME to a writable directory."
    )


class LocalServerKeyError(LeaderKeyError):
    """A non-leader's spawned llama-cpp-server has no usable key, so the server is not started.

    Every spawned server requires a key (#1314): a solo's binds loopback, but without one any local process or user
    on a shared host could use it, and a stale tunnel ingress pointing at its port would expose it. A file that
    exists but is unusable, or a key that cannot be created, refuses here, naming the path; it is never overwritten
    and the server never starts unkeyed.
    """

    def __init__(self, *, path: Path, reason: str, fix_hint: str | None = None) -> None:
        super().__init__(path=path, reason=reason, fix_hint=fix_hint or _local_key_fix(path))
        self.message = f"local LLM server key unusable ({reason}): {path}"
        self.args = (self.message,)


def _read_local_server_key(path: Path) -> str | None:
    """The local server key at ``path``; None when the file is missing or empty; :class:`LocalServerKeyError` when
    unusable."""
    try:
        raw, mode = _read_key_file(path)
    except LeaderKeyMissing:
        return None
    except LeaderKeyError as e:
        raise LocalServerKeyError(path=path, reason=e.reason) from e
    if not raw.lstrip("\ufeff").strip():
        return None  # empty = missing, as for every key file (owner decision 2026-10-25, #1402)
    return _validated_key(path, raw, mode, LocalServerKeyError, chmod_fix=_local_key_fix(path))


def read_local_server_key() -> str | None:
    """For READERS (doctor, harnesses): the local server key, None if none exists yet. Never creates it: a key
    created here would match no running server."""
    return _read_local_server_key(key_file_path(LOCAL_SERVER_KEY_NAME))


def ensure_local_server_key() -> str:
    """For the SPAWN side only: the local server key, created when (and only when) the file is missing or empty.

    Through :func:`_ensure_named_key` (the lock, the re-read inside it, the value read back from disk). A file that
    exists but is unusable, or a key that cannot be created, is a :class:`LocalServerKeyError` naming the path; it is
    never overwritten (#1402).
    """
    return _ensure_named_key(
        LOCAL_SERVER_KEY_NAME,
        lambda key: _write_key_secret(key, LOCAL_SERVER_KEY_NAME),
        read=_read_local_server_key,
        refusal=lambda path, reason: LocalServerKeyError(path=path, reason=reason),
    )


def candidate_server_keys() -> dict[str, str]:
    """Every key a local llama-cpp-server on this host may hold, by name (``"mesh"``, ``"local"``), for READERS that
    must not guess which role spawned it (doctor, harnesses). Read-only; an absent or unusable file is left out."""
    found: dict[str, str] = {}
    try:
        found["mesh"] = require_leader_key()
    except LeaderKeyError:
        pass
    try:
        local = read_local_server_key()
    except LocalServerKeyError:
        local = None
    if local is not None:
        found["local"] = local
    return found


# ─── console token (bearer credential for `maxim serve`) ───────────────────


def read_console_token() -> str | None:
    """Return the stored console token, or None if not set.

    This sits in the per-request auth path (rotation-bites-immediately), so
    an EXISTING-but-unreadable file — which 401s every device while looking
    like a plain missing token — warns loudly instead of failing silently
    (review fold: the mystery-401 debugging session).
    """
    token = read_key(CONSOLE_KEY_NAME)
    if token is None and key_exists(CONSOLE_KEY_NAME):
        logging.getLogger(__name__).warning(
            "console token file exists but is empty or unreadable — every request will 401: %s",
            key_file_path(CONSOLE_KEY_NAME),
        )
    return token


def ensure_console_token() -> str:
    """Return the console token, generating one (mxc_-prefixed) only when the file is missing or empty (#1402).

    A file that exists but cannot be used raises :class:`~maxim.exceptions.KeyFileUnusable`, never an overwrite:
    that would log out every device with no message naming why.
    """
    return _ensure_named_key(
        CONSOLE_KEY_NAME, lambda token: _write_key_secret(token, CONSOLE_KEY_NAME), prefix=CONSOLE_TOKEN_PREFIX
    )


def rotate_console_token() -> str:
    """Generate a new console token, overwrite the file, return the new value.

    Every device that stored the old token is logged out on its next request —
    rotation is the console's revocation story (hardening plan, decision A5).
    """
    token = generate_key(prefix=CONSOLE_TOKEN_PREFIX)
    with _key_creation_lock(key_file_path(CONSOLE_KEY_NAME)):
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
