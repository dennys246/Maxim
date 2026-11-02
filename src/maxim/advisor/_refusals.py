"""The checks ``maxim.advisor.open()`` runs before it builds anything (#1315 item 1, §4.2 "Refusals at open()").

Each refusal is an :class:`~maxim.exceptions.AdvisorCompositionError`. Nothing here writes, spawns or connects.
:func:`code_loading_vars_set` is the ONE place the observer iterates the environment: the refusal is by prefix
(``DYLD_*``, ``GGML_*``, ``LLAMA_*``), which cannot be checked without listing the names, and the env read-set gate
allows exactly this function's iteration and no other (``test_advisor_gate.py::ENV_WILDCARD_ALLOW``).
"""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass

from maxim.exceptions import AdvisorCompositionError

# Acting at exec, they only mark a hostile environment; acting at dlopen (libllama/libggml load at open()), or read
# by llama-cpp-python to pick its shared library, they load code (§4.2).
CODE_LOADING_EXACT = frozenset(
    {"LD_PRELOAD", "LD_AUDIT", "DYLD_INSERT_LIBRARIES", "LD_LIBRARY_PATH", "LLAMA_CPP_LIB_PATH", "LLAVA_CPP_LIB"}
)
# Refused by prefix: dyld applies DYLD_LIBRARY_PATH by leaf name even to an absolute CDLL path, GGML_BACKEND_PATH
# loads code, and llama.cpp's C side reads LLAMA_* and GGML_* where Python cannot see it.
CODE_LOADING_PREFIXES = ("DYLD_", "GGML_", "LLAMA_")


def code_loading_vars_set() -> list[str]:
    """The pinned code-loading variables present in the environment, sorted. The observer's only env iteration."""
    return sorted(k for k in os.environ if k in CODE_LOADING_EXACT or k.startswith(CODE_LOADING_PREFIXES))


def refuse_code_loading_env() -> None:
    names = code_loading_vars_set()
    if names:
        raise AdvisorCompositionError(f"open() refuses a code-loading environment: {', '.join(names)} set")


def check_model(model: str | os.PathLike[str]) -> str:
    """``model`` as a path string, refused unless it names an existing regular file (never a profile or a URL)."""
    path = os.fspath(model)
    try:
        st = os.stat(path)
    except (OSError, ValueError) as e:
        raise AdvisorCompositionError(f"the model is not an existing file: {path!r}") from e
    if not stat.S_ISREG(st.st_mode):
        raise AdvisorCompositionError(f"the model is not a regular file: {path!r}")
    return path


_HASH_CHUNK = 1 << 20


def _identity(st: os.stat_result) -> tuple[int, int, int, int]:
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def pin_model(path: str) -> tuple[tuple[int, int, int, int], str]:
    """Open ``path`` once, refuse it unless it is a regular file, and hash it from that fd. Returns the file's
    identity (device, inode, size, mtime) and its sha256, so :func:`check_model_unchanged` can refuse a file
    swapped before the load (review S1/S2/A3). A read failure is a refusal, never a raw ``OSError``."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    except OSError as e:
        raise AdvisorCompositionError(f"the model could not be opened: {path!r} ({e.strerror})") from e
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise AdvisorCompositionError(f"the model is not a regular file: {path!r}")
        digest = hashlib.sha256()
        offset = 0
        while chunk := os.pread(fd, _HASH_CHUNK, offset):
            digest.update(chunk)
            offset += len(chunk)
    except OSError as e:
        raise AdvisorCompositionError(f"the model could not be read: {path!r} ({e.strerror})") from e
    finally:
        os.close(fd)
    return _identity(st), digest.hexdigest()


def check_model_unchanged(path: str, identity: tuple[int, int, int, int]) -> None:
    """Refuse when ``path`` no longer names the file :func:`pin_model` hashed: the hash would name a model that did
    not run. llama.cpp loads by path, so a swap and swap-back inside the load is not caught (the stated residual)."""
    try:
        now = _identity(os.stat(path))
    except OSError as e:
        raise AdvisorCompositionError(f"the model vanished during open(): {path!r} ({e.strerror})") from e
    if now != identity:
        raise AdvisorCompositionError(f"the model changed during open(): {path!r} is not the file that was hashed")


@dataclass(frozen=True)
class CheckedDirs:
    """The home and report directories, held open; every check ran on these fds. In-process only (CC3 n/a)."""

    home_fd: int
    report_fd: int

    def close(self) -> None:
        for fd in (self.home_fd, self.report_fd):
            os.close(fd)


def _open_dir(path: str | os.PathLike[str], what: str) -> int:
    try:
        return os.open(os.fspath(path), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as e:  # missing, not a directory, or a symlink at the leaf (ELOOP/ENOTDIR)
        raise AdvisorCompositionError(f"{what} is not an existing, non-symlink directory: {path!r} ({e})") from e


def _ancestry(fd: int) -> list[tuple[int, int]]:
    """``(st_dev, st_ino)`` of ``fd``'s directory and each ancestor up to the root, walked through the fd by
    ``..`` (physical: a symlinked parent cannot hide one). ``os.stat`` with ``dir_fd`` opens nothing."""
    chain: list[tuple[int, int]] = []
    st = os.fstat(fd)
    up = "."
    while True:
        ident = (st.st_dev, st.st_ino)
        if chain and chain[-1] == ident:  # ".." of the root is the root
            return chain
        chain.append(ident)
        up = ".." if up == "." else up + "/.."
        st = os.stat(up, dir_fd=fd)


def open_checked_dirs(home_dir: str | os.PathLike[str], report_dir: str | os.PathLike[str]) -> CheckedDirs:
    """Open both directories ``O_DIRECTORY|O_NOFOLLOW`` and check them on the fds: each writable, neither the
    other, neither inside the other (so no ``realpath`` window: the fds are what every later write goes through)."""
    home_fd = _open_dir(home_dir, "home_dir")
    try:
        report_fd = _open_dir(report_dir, "report_dir")
    except BaseException:
        os.close(home_fd)
        raise
    dirs = CheckedDirs(home_fd=home_fd, report_fd=report_fd)
    try:
        for fd, what in ((home_fd, "home_dir"), (report_fd, "report_dir")):
            if not os.access(".", os.W_OK | os.X_OK, dir_fd=fd):
                raise AdvisorCompositionError(f"{what} is not writable: {home_dir if fd == home_fd else report_dir!r}")
        home_chain, report_chain = _ancestry(home_fd), _ancestry(report_fd)
        if home_chain[0] in report_chain or report_chain[0] in home_chain:
            raise AdvisorCompositionError(
                f"home_dir and report_dir overlap (one is, or is inside, the other): {home_dir!r}, {report_dir!r}"
            )
    except BaseException:
        dirs.close()
        raise
    return dirs
