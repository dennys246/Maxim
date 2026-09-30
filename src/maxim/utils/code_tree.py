"""Which code a checkout holds, decided by what is on disk (M1, #998).

Two questions every experiment record answers about the code that produced it:

- **Is the tree clean?** (:func:`tree_difference` / :func:`tree_dirty`) -- every code path is on disk exactly
  as HEAD has it: the same git blob, computed from the bytes on disk with no filters; the same executable
  bit; the same symlink target; nothing extra and nothing missing.
- **Which code, exactly?** (:func:`code_tree_sha256`) -- a digest of every code path's on-disk content, so
  a dirty run still names the code it ran.

Both ask git only for the SET of paths (:func:`code_paths`) and read the content themselves. That is the
point: ``git status`` and ``git diff`` can be made to report a dirty tree as clean by config and index
state (a per-user exclude file, ``assume-unchanged``, ``core.fileMode=false``, replace refs, a
``GIT_DIR`` leaked from a parent process). Every git call here drops those inputs (:func:`git_out`), and
anything that cannot be established reads as dirty / ``"unknown"``: an unestablishable tree is what the
gate exists to stop.

The boundary it keeps: a path the repo's own **committed** ``.gitignore`` files exclude is not code, and
those rules are compared with the code (the root ``.gitignore`` is always in the path set).

**Stdlib only, by contract.** ``scripts/_provenance.py`` loads this file by path from its own tree and must
never import ``maxim`` (the installed package may be a different checkout), so nothing here may import
``maxim`` or a third-party package. Regression guard: ``tests/unit/test_clean_flag_998.py``.
"""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from pathlib import Path

# The code scope, as ``scripts/_provenance.py::DIRTY_SCOPE``; the root ``.gitignore`` is always added.
SCOPE = ("src", "scripts")

# Inherited git variables that would point a call at another tree, rewrite HEAD's objects, or change
# how a pathspec matches. Dropped from every call.
_INHERITED = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_COMMON_DIR",
    "GIT_REPLACE_REF_BASE",
    "GIT_LITERAL_PATHSPECS",
    "GIT_GLOB_PATHSPECS",
    "GIT_NOGLOB_PATHSPECS",
    "GIT_ICASE_PATHSPECS",
)

_MODE_KIND = {b"100644": b"file", b"100755": b"executable", b"120000": b"symlink"}


def git_out(repo: Path | str, *args: str, stdin: bytes | None = None, ok: tuple[int, ...] = (0,)) -> bytes | None:
    """``git`` output for ``repo``, or ``None`` on failure. Git reads no user or system config and no
    inherited ``GIT_CONFIG_*`` variables; inherited location, replace and pathspec variables are dropped;
    replace refs are not followed; the repo's own config cannot add an excludes file, match paths
    case-insensitively or decompose unicode names. So neither the environment nor per-user config can
    change what git reports. (Not reading user, system or ``GIT_CONFIG_*`` config is defense in depth: no
    setting found that moves these listings is left unpinned by a ``-c`` above, and git reads
    ``core.worktree`` only from the repo's own config, which :func:`_top_level_problem` catches.)"""
    env = {k: v for k, v in os.environ.items() if k not in _INHERITED and not k.startswith("GIT_CONFIG")}
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env["GIT_NO_REPLACE_OBJECTS"] = "1"
    try:
        r = subprocess.run(
            [
                "git",
                *("-c", "core.precomposeunicode=true", "-c", "core.ignorecase=false"),
                *("-c", f"core.excludesFile={os.devnull}"),
                *args,
            ],
            cwd=Path(repo),
            input=stdin,
            capture_output=True,
            timeout=60,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout if r.returncode in ok else None


def head_hash(repo: Path | str, length: int | None = None) -> str:
    """HEAD's commit id (abbreviated to ``length``, or git's default abbreviation when ``None``), read
    through :func:`git_out`, so it names the same tree the clean flag judged. ``"unknown"`` on failure."""
    short = "--short" if length is None else f"--short={length}"
    out = git_out(repo, "rev-parse", short, "HEAD")
    return out.decode().strip() if out else "unknown"


def head_commit(repo: Path | str) -> str:
    """HEAD's FULL commit id, read through :func:`git_out`. Every provenance stamp uses this one length, so a
    harness row and the sim reports it spawned compare byte for byte (M1b, #1003). ``"unknown"`` on failure."""
    out = git_out(repo, "rev-parse", "HEAD")
    return out.decode().strip() if out else "unknown"


def code_paths(repo: Path | str, scope: tuple[str, ...] = SCOPE) -> set[bytes] | None:
    """Every path whose content is the code under ``scope``: the root ``.gitignore``; every tracked path;
    every untracked path the repo's ``.gitignore`` files do not exclude; every path in HEAD (so a deletion
    shows); and every untracked ``.gitignore`` -- listed with no excludes, so one that ignores itself and
    the code beside it still counts -- unless its directory is already excluded by a ``.gitignore``
    committed in HEAD
    (a ``.pytest_cache/``, a venv, a ``node_modules`` package ship their own, and nothing there is code).
    ``None`` when git fails."""
    repo = Path(repo)
    if _top_level_problem(repo) is not None:
        return None
    listed: set[bytes] = set()
    head_paths = git_out(repo, "ls-tree", "-r", "-z", "--name-only", "HEAD", "--", ".gitignore", *scope)
    worktree = git_out(
        repo, "ls-files", "-z", "--cached", "--others", "--exclude-per-directory=.gitignore", "--", ".gitignore", *scope
    )
    untracked_rules = git_out(
        repo, "ls-files", "-z", "--others", "--", ".gitignore", *(f":(glob){d}/**/.gitignore" for d in scope)
    )
    if head_paths is None or worktree is None or untracked_rules is None:
        return None
    in_head = {p for p in head_paths.split(b"\0") if p}
    listed.update(in_head)
    listed.update(p for p in worktree.split(b"\0") if p)
    candidates = sorted(p for p in untracked_rules.split(b"\0") if p)
    listed.update(p for p in candidates if b"/" not in p)  # the root .gitignore always counts
    candidates = [p for p in candidates if b"/" in p]
    if not candidates:
        return listed
    # HEAD is the authority on which rules are tracked (their content is compared to HEAD too), so no index
    # state -- ``git rm --cached``, a staged rule -- can change which directories they excuse.
    tracked_rules = {p for p in in_head if p == b".gitignore" or p.endswith(b"/.gitignore")}
    # The directory itself, WITHOUT a trailing slash: with one, check-ignore judges an empty-named child
    # (so the candidate's own rules, or a ``dir/*`` pattern that a ``!dir/*.py`` re-includes from, answer).
    dirs = b"".join(os.path.dirname(p) + b"\0" for p in candidates)
    verdicts = git_out(repo, "check-ignore", "-v", "-z", "-n", "--no-index", "--stdin", stdin=dirs, ok=(0, 1))
    if verdicts is None:
        return None
    fields = verdicts.split(b"\0")
    excluded = set()
    for source, _line, pattern, path in zip(fields[0::4], fields[1::4], fields[2::4], fields[3::4]):
        if source in tracked_rules and pattern and not pattern.startswith(b"!"):
            excluded.add(path)
    listed.update(p for p in candidates if os.path.dirname(p) not in excluded)
    return listed


def _top_level_problem(repo: Path) -> str | None:
    """Why ``repo`` cannot be judged, or ``None`` when it is the top of its own work tree."""
    top = git_out(repo, "rev-parse", "--show-toplevel")
    if top is None:
        return "git cannot read it as a repository (not a repo, git failed, or refused, e.g. dubious ownership)"
    if os.path.realpath(os.fsdecode(top.strip())) != os.path.realpath(repo):
        return f"not the top of its git work tree (git reports {os.fsdecode(top.strip())})"
    return None


def on_disk(repo: Path | str, rel: bytes, blob_hash: str | None = None) -> tuple[bytes, bytes] | None:
    """What ``rel`` is on disk, as ``(kind, content)``: ``deleted``; ``symlink`` with its target; ``file`` or
    ``executable`` (git's rule: the owner exec bit) with the sha256 of its bytes. With ``blob_hash`` (a git
    object format), a symlink's or file's content is instead its git blob id, as hex. ``None`` when it
    cannot be read as one of those, or resolves outside the repo (reached through a symlinked directory)."""
    repo = Path(repo)
    root = os.path.realpath(repo)
    path = repo / os.fsdecode(rel)
    if os.path.commonpath([root, os.path.realpath(path.parent)]) != root:
        return None
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return b"deleted", b""
    except OSError:
        return None
    if stat.S_ISLNK(st.st_mode):
        target = os.fsencode(os.readlink(path))
        if blob_hash is None:
            return b"symlink", target
        return b"symlink", hashlib.new(blob_hash, b"blob %d\0" % len(target) + target).hexdigest().encode()
    if not stat.S_ISREG(st.st_mode):
        return None
    content = hashlib.new(blob_hash or "sha256")
    if blob_hash is not None:
        content.update(b"blob %d\0" % st.st_size)
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                content.update(block)
    except OSError:
        return None
    kind = b"executable" if st.st_mode & stat.S_IXUSR else b"file"
    return kind, (content.digest() if blob_hash is None else content.hexdigest().encode())


def code_tree_sha256(repo: Path | str, scope: tuple[str, ...] = SCOPE) -> str:
    """sha256 naming the exact code under ``scope``: every path in :func:`code_paths`, in path order, each
    framed and length-prefixed with what it is on disk (:func:`on_disk`). No git config or index state can
    hide a change. Index state can move the digest, but only toward "different": a path staged and not
    in HEAD is listed even when absent from disk. When :func:`tree_difference` finds nothing, the listed
    set is exactly HEAD's, so a clean tree's digest is fixed by HEAD alone. ``"unknown"`` when git fails or
    a path cannot be read as a file, a symlink or absent (an untracked nested repo is a directory)."""
    listed = code_paths(repo, scope)
    if listed is None:
        return "unknown"

    def framed(field: bytes) -> bytes:
        return len(field).to_bytes(8, "big") + field

    digest = hashlib.sha256()
    for rel in sorted(listed):
        seen = on_disk(repo, rel)
        if seen is None:
            return "unknown"
        digest.update(framed(rel) + framed(seen[0]) + framed(seen[1]))
    return digest.hexdigest()


def tree_difference(repo: Path | str, scope: tuple[str, ...] = SCOPE) -> str | None:
    """``None`` when every path in :func:`code_paths` is on disk exactly as HEAD has it; otherwise the first
    difference found, as ``"<path>: <why>"``, so a refusal can name it. Unknown is a difference: a git
    failure, an unknown object format, a path git cannot map (a submodule), an unreadable or odd path.
    It judges the code on disk: a staged edit reverted on disk, ``git rm --cached`` or an index flag leaves a
    tree that matches HEAD clean. Index state can only make a tree read dirty, never clean: a path staged
    but not in HEAD is a difference even when absent from disk. Line-ending conversion (autocrlf) is a
    difference, because the bytes on disk are not HEAD's."""
    problem = _top_level_problem(Path(repo))
    if problem is not None:
        return f"({repo}): {problem}"
    listed = code_paths(repo, scope)
    if listed is None:
        return "(the code paths): git could not list them"
    fmt_out = git_out(repo, "rev-parse", "--show-object-format")
    fmt = fmt_out.decode().strip() if fmt_out else ""
    if fmt not in ("sha1", "sha256"):
        return f"(the repository): unknown object format {fmt!r}"
    tree = git_out(repo, "ls-tree", "-r", "-z", "HEAD", "--", ".gitignore", *scope)
    if tree is None:
        return "(HEAD): git could not read it"
    head: dict[bytes, tuple[bytes, bytes]] = {}
    for entry in (e for e in tree.split(b"\0") if e):
        meta, _, rel = entry.partition(b"\t")
        mode, _kind, oid = meta.split(b" ")
        head[rel] = (mode, oid)
    for rel in sorted(listed):
        name = os.fsdecode(rel)
        if rel not in head:
            return f"{name}: not in HEAD"
        mode, oid = head[rel]
        if mode not in _MODE_KIND:
            return f"{name}: HEAD mode {mode.decode()} (a submodule or unknown entry)"
        seen = on_disk(repo, rel, blob_hash=fmt)
        if seen is None:
            return f"{name}: cannot be read as a file or symlink inside the repo"
        if seen[0] == b"deleted":
            return f"{name}: deleted"
        if seen[0] != _MODE_KIND[mode]:
            return f"{name}: is {seen[0].decode()}, HEAD has {_MODE_KIND[mode].decode()}"
        if seen[1] != oid:
            return f"{name}: content differs from HEAD"
    return None


def tree_dirty(repo: Path | str, scope: tuple[str, ...] = SCOPE) -> bool:
    """True unless the code on disk is exactly HEAD's (see :func:`tree_difference`)."""
    return tree_difference(repo, scope) is not None
