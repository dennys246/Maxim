"""Shared provenance guard for experiment harnesses that spawn `maxim` sub-sims.

WHY THIS EXISTS (2026-07-28, Exp 42b)
-------------------------------------
A 40-sub-sim behavioural re-validation was invalidated because the sub-sims
imported a DIFFERENT checkout than the one under test. Three things conspired,
and each of them is silent on its own:

1. `maxim` is a console script; it resolves `import maxim` purely through
   `sys.path`. A venv can carry stale editable `.pth` files pointing at other
   checkouts (or deleted worktrees) from an old `pip install -e`.
2. `PYTHONPATH=src` beats those `.pth` entries — but it is RELATIVE, so it
   silently resolves to nothing unless the launch cwd is the repo root.
3. The shell that launched the run used
   `source .venv/bin/activate && export PYTHONPATH=src`; the `source` failed,
   `&&` short-circuited, and the export never ran.

Nothing errored. Every sub-sim "succeeded". And the `git_hash` recorded in the
run records came from the *harness file's* directory, so the JSONL looked
authoritative while describing code that was never executed. The mistake only
surfaced days later via an unrelated missing-artifact symptom — by which point
the run could not be proven either way.

THE RULE
--------
`git_hash` answers "where does the harness live?". That is NOT the question.
The question is "which code did the sub-sims execute?" — and a harness that
cannot answer it produces results that mean nothing, whether or not they
happen to be correct.

So: every harness that spawns `maxim` MUST call :func:`assert_repo_interpreter`
before its first sub-sim, and SHOULD record :func:`executed_code_provenance`
into each run record so the artifact is self-auditing forever after.

THE SECOND DOOR (2026-08-26, Exp 53/53b — roadmap 1.1.x item 16.7)
------------------------------------------------------------------
The rule above was scoped to harnesses that SPAWN `maxim`. The in-process
family (`scripts/orient_*/`, which imports `maxim` and drives the robot
directly) inherited the vocabulary but not the enforcement: it *stamped*
``working_tree_dirty_src_scripts: true`` into every Exp 53/53b start record
and kept going. Stamping is detection; refusing is enforcement. So:

* :func:`preflight_gated_record` — a harness about to write a GATED record
  (anything under ``docs/experiments/data/``) from a dirty ``src``/``scripts``
  tree gets :class:`DirtyTreeError` (harness policy: exit 3) unless the
  operator passed ``--allow-dirty``; the returned dict then carries
  ``allow_dirty: True`` and the harness stamps it into EVERY record so the
  write-up cannot silently omit it.
* :func:`in_process_code_provenance` — the in-process counterpart of
  :func:`executed_code_provenance`: the caller hands over ``maxim.__file__``
  (this module still imports nothing from `maxim`) and gets the executed
  tree's hash + dirty flag, refusing when the imported package is not this
  repo's ``src``.

Full history: docs/lessons/experiment-prereg-precedes-data.md.

This module is deliberately stdlib-only and does NOT import `maxim` — it is
imported by path from the harness's own directory, so it is guaranteed to come
from the same tree as the harness that calls it.
"""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import sys
from pathlib import Path

__all__ = [
    "GATED_DATA_DIR",
    "DirtyTreeError",
    "ProvenanceError",
    "assert_repo_interpreter",
    "executed_code_provenance",
    "in_process_code_provenance",
    "is_gated_path",
    "preflight_gated_record",
    "preflight_gated_record_or_exit",
    "evidence_out_path",
    "evidence_out_paths",
    "evidence_out_paths_or_exit",
    "EVIDENCE_DIR",
    "resolved_maxim_file",
    "working_tree_dirty",
]

# Anything written here is a GATED record: it backs a ledger row, a result
# doc, or a release gate. The refuse path below applies to this tree only.
GATED_DATA_DIR = Path("docs/experiments/data")

# The paths whose dirtiness makes a run's code-under-test unestablishable.
DIRTY_SCOPE = ("src", "scripts")


class ProvenanceError(RuntimeError):
    """The interpreter would import a `maxim` outside the harness's repo."""


class DirtyTreeError(ProvenanceError):
    """A gated record was about to be written from a dirty src/scripts tree.

    Harness policy is exit 3 (the same code :func:`assert_repo_interpreter`
    callers use), unless the operator passed ``--allow-dirty`` — in which case
    the record itself must say so (``allow_dirty: true``).
    """


def _shebang_interpreter(binary: str) -> str:
    """The interpreter the console script itself runs under.

    Probing with ``sys.executable`` would test the HARNESS's interpreter, which
    need not be the one `maxim` uses — that gap is exactly where a mismatch hides.
    """
    try:
        first = Path(binary).read_text().splitlines()[0]
    except UnicodeDecodeError:
        # Not a text script — the caller passed a raw interpreter (e.g. a
        # harness that spawns `[sys.executable, "-m", "maxim"]`). The binary
        # IS the interpreter; probing through sys.executable here would
        # re-open the harness-vs-subsim gap this module exists to close.
        return binary
    except (OSError, IndexError):
        return sys.executable
    return first.lstrip("#!").strip() or sys.executable


def resolved_maxim_file(binary: str, *, timeout: float = 60.0) -> str | None:
    """Return `maxim.__file__` as the sub-sims would resolve it, or None."""
    interp = _shebang_interpreter(binary)
    try:
        probe = subprocess.run(
            [interp, "-c", "import maxim,sys; sys.stdout.write(maxim.__file__)"],
            env=os.environ.copy(),  # same env the sub-sims inherit
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return None
    out = probe.stdout.strip()
    return out if probe.returncode == 0 and out else None


def _git_hash(cwd: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def assert_repo_interpreter(repo_root: Path | str, binary: str, *, exempt: bool = False) -> str | None:
    """Raise :class:`ProvenanceError` unless `maxim` resolves inside ``repo_root``.

    ``exempt`` is for mock/dry runs that never spawn a sub-sim. Returns the
    resolved ``maxim.__file__`` on success (None when exempt).
    """
    if exempt:
        return None
    root = Path(repo_root).resolve()
    src = (root / "src").resolve()
    resolved = resolved_maxim_file(binary)
    if resolved is None:
        raise ProvenanceError(
            f"cannot import `maxim` with the interpreter behind {binary}.\n"
            f"  Activate the right venv, then: export PYTHONPATH={src}"
        )
    imported_root = Path(resolved).resolve().parent.parent
    if imported_root == src:
        return resolved
    raise ProvenanceError(
        "the `maxim` package the sub-sims would import is NOT this repo's src.\n"
        f"  harness repo src : {src}\n"
        f"  imported maxim   : {imported_root}\n"
        "  → the run would measure the WRONG CODE and its results would be meaningless.\n"
        f"  Fix: export PYTHONPATH={src}\n"
        "       (ABSOLUTE — a relative 'src' silently resolves to nothing off the repo root;\n"
        "        and put it on its own line, never `source ... && export ...`, because a\n"
        "        failing `source` short-circuits the export without erroring.)\n"
        "  Also check your venv's site-packages for stale `__editable__*.pth` files left by\n"
        "  an old `pip install -e` from another/deleted checkout. Best cure: give each\n"
        "  worktree its own venv + editable install so PYTHONPATH is never load-bearing."
    )


def working_tree_dirty(repo_root: Path | str, scope: tuple[str, ...] = DIRTY_SCOPE) -> bool:
    """True when ``git status`` reports any change (incl. untracked) under ``scope``.

    A git failure counts as DIRTY: an unestablishable tree is the exact thing
    the refuse path exists to stop, so unknown must not read as clean.
    """
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain", "--", *scope],
            cwd=Path(repo_root),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return True
    if r.returncode != 0:
        return True
    return bool(r.stdout.strip())


def code_tree_sha256(repo_root: Path | str, scope: tuple[str, ...] = DIRTY_SCOPE) -> str:
    """sha256 naming the exact code under ``scope`` (M1): every tracked path, every untracked path the repo's
    own ``.gitignore`` files do not exclude, and every path in HEAD, in path order, each framed with its
    content -- a regular file's bytes (and whether it is executable), a symlink's target string, or
    deleted. It hashes what is ON DISK and asks git only for the path set, so no git config or index state
    (an external diff, textconv, a per-user exclude file, assume-unchanged, ``core.fileMode``, autocrlf)
    can move it or hide a change; inherited git location and pathspec variables (``GIT_DIR``,
    ``GIT_WORK_TREE``, ``GIT_LITERAL_PATHSPECS``, ...) are dropped so a parent process cannot point it
    elsewhere or blind a listing. The boundary it keeps: a path the repo's ``.gitignore`` files exclude is
    assumed not to be code, and every rule that decides it (the root ``.gitignore``, any ``.gitignore`` in
    scope, tracked or not) is hashed with the code. ``"unknown"`` when git fails, a path is neither a
    file, a symlink nor absent (an untracked nested repo is a directory), or a path resolves outside the
    repo. Its twin in ``src/maxim/simulation/report.py::_code_tree_sha256`` must stay identical (pinned by AST)."""
    repo = Path(repo_root)
    inherited = (
        *("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_COMMON_DIR"),
        *("GIT_LITERAL_PATHSPECS", "GIT_GLOB_PATHSPECS", "GIT_NOGLOB_PATHSPECS", "GIT_ICASE_PATHSPECS"),
    )
    env = {k: v for k, v in os.environ.items() if k not in inherited}
    env["GIT_OPTIONAL_LOCKS"] = "0"
    listed: set[bytes] = set()
    for args in (
        # The root .gitignore is hashed with the code: it decides which paths count as code.
        ["ls-files", "-z", "--cached", "--others", "--exclude-per-directory=.gitignore", "--", ".gitignore", *scope],
        ["ls-tree", "-r", "-z", "--name-only", "HEAD", "--", ".gitignore", *scope],
        # An untracked .gitignore is listed with no excludes, so one that ignores itself (and the code
        # beside it) still moves the digest.
        ["ls-files", "-z", "--others", "--", ".gitignore", *(f":(glob){d}/**/.gitignore" for d in scope)],
    ):
        try:
            r = subprocess.run(
                ["git", "-c", "core.precomposeunicode=true", *args], cwd=repo, capture_output=True, timeout=60, env=env
            )
        except (OSError, subprocess.TimeoutExpired):
            return "unknown"
        if r.returncode != 0:
            return "unknown"
        listed.update(p for p in r.stdout.split(b"\0") if p)

    def framed(field: bytes) -> bytes:
        return len(field).to_bytes(8, "big") + field

    root = os.path.realpath(repo)
    digest = hashlib.sha256()
    for rel in sorted(listed):
        path = repo / os.fsdecode(rel)
        if os.path.commonpath([root, os.path.realpath(path.parent)]) != root:
            return "unknown"  # reached through a symlinked directory: not this repo's code
        try:
            st = os.lstat(path)
        except FileNotFoundError:
            digest.update(framed(rel) + framed(b"deleted"))
            continue
        except OSError:
            return "unknown"
        if stat.S_ISLNK(st.st_mode):
            digest.update(framed(rel) + framed(b"symlink") + framed(os.fsencode(os.readlink(path))))
        elif stat.S_ISREG(st.st_mode):
            content = hashlib.sha256()
            try:
                with open(path, "rb") as f:
                    for block in iter(lambda: f.read(1 << 20), b""):
                        content.update(block)
            except OSError:
                return "unknown"
            kind = b"executable" if st.st_mode & 0o111 else b"file"
            digest.update(framed(rel) + framed(kind) + framed(content.digest()))
        else:
            return "unknown"
    return digest.hexdigest()


def is_gated_path(repo_root: Path | str, out_path: Path | str | None) -> bool:
    """True when ``out_path`` resolves inside ``<repo_root>/docs/experiments/data/``."""
    if out_path is None:
        return False
    gated_root = (Path(repo_root) / GATED_DATA_DIR).resolve()
    try:
        return Path(out_path).resolve().is_relative_to(gated_root)
    except (OSError, ValueError):
        return False


def preflight_gated_record(
    repo_root: Path | str, out_path: Path | str | None, *, allow_dirty: bool = False
) -> dict[str, bool]:
    """Refuse to write a gated record from a dirty tree unless ``allow_dirty``.

    Returns ``{"gated", "working_tree_dirty_src_scripts", "allow_dirty"}`` so
    the caller can stamp the outcome into the record. ``allow_dirty`` in the
    result is True only when it was needed AND granted (gated + dirty +
    ``--allow-dirty``): a clean tree needs no allowance and must not claim one.
    Raises :class:`DirtyTreeError` (harness policy: exit 3) when the write is
    gated, the tree is dirty, and no allowance was given. Non-gated writes
    (``/tmp`` logs, dry runs elsewhere) are never refused — the flag is still
    reported so the record can carry it.
    """
    root = Path(repo_root).resolve()
    gated = is_gated_path(root, out_path)
    dirty = working_tree_dirty(root)
    if gated and dirty and not allow_dirty:
        raise DirtyTreeError(
            f"refusing to write a GATED record ({Path(out_path).resolve().relative_to(root)}) "
            f"from a DIRTY tree: `git status --porcelain -- {' '.join(DIRTY_SCOPE)}` is not empty in {root}.\n"
            "  A result whose code-under-test cannot be established is not a validation "
            "(Exp 42b corollary; Exp 53/53b release-day incident).\n"
            "  Fix: commit (and merge) the harness/src changes first, then re-run from the clean tree —\n"
            "       or pass --allow-dirty, which stamps `allow_dirty: true` into every record so the\n"
            "       write-up cannot omit it (docs/lessons/experiment-prereg-precedes-data.md)."
        )
    return {
        "gated": gated,
        "working_tree_dirty_src_scripts": dirty,
        "allow_dirty": bool(gated and dirty and allow_dirty),
    }


def preflight_gated_record_or_exit(
    repo_root: Path | str, out_path: Path | str | None, *, allow_dirty: bool = False
) -> dict[str, bool]:
    """:func:`preflight_gated_record` with the harness policy applied: print + exit 3."""
    try:
        return preflight_gated_record(repo_root, out_path, allow_dirty=allow_dirty)
    except DirtyTreeError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(3) from exc


def in_process_code_provenance(
    repo_root: Path | str,
    maxim_file: str | None,
    *,
    out_path: Path | str | None = None,
    allow_dirty: bool = False,
) -> dict[str, object]:
    """Provenance for a harness that IMPORTS `maxim` in-process (no sub-sim).

    ``maxim_file`` is the caller's ``maxim.__file__`` — this module stays
    maxim-free. Raises :class:`ProvenanceError` when that package is not this
    repo's ``src`` (the run would measure the wrong code), and delegates the
    gated-write refusal to :func:`preflight_gated_record` when ``out_path`` is
    given. The returned dict is the ``provenance`` block harnesses stamp into
    their start record; ``allow_dirty`` is present only when it was granted.
    """
    root = Path(repo_root).resolve()
    src = (root / "src").resolve()
    executed = Path(maxim_file or "").resolve() if maxim_file else None
    if executed is None or not executed.is_relative_to(src):
        raise ProvenanceError(
            f"the imported `maxim` is {executed}, not this repo's src ({src}).\n"
            "  The run would measure the WRONG CODE — fix PYTHONPATH (absolute, its own line) and re-run."
        )
    gate = preflight_gated_record(root, out_path, allow_dirty=allow_dirty)
    prov: dict[str, object] = {
        "executed_maxim_file": str(executed),
        "executed_git_hash": _git_hash_short12(root),
        "working_tree_dirty_src_scripts": gate["working_tree_dirty_src_scripts"],
        "code_tree_sha256": code_tree_sha256(root),
        "python": sys.executable,
        "pythonpath": os.environ.get("PYTHONPATH", ""),
    }
    if gate["allow_dirty"]:
        prov["allow_dirty"] = True
    return prov


def _git_hash_short12(cwd: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return r.stdout.strip() or "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def executed_code_provenance(
    repo_root: Path | str,
    binary: str,
    *,
    out_path: Path | str | None = None,
    allow_dirty: bool = False,
) -> dict[str, object]:
    """Provenance describing the CODE THAT RAN, for embedding in run records.

    ``harness_git_hash`` is where the harness file lives; ``executed_git_hash``
    is the tree the sub-sims actually import. When they disagree, the run is
    suspect — record both so the artifact can be audited long after the shell
    history is gone. Since 2026-08-29 the block also carries
    ``working_tree_dirty_src_scripts`` (both harness families stamp it), and
    when ``out_path`` is given the gated-record refusal applies exactly as for
    the in-process family (:func:`preflight_gated_record`; ``allow_dirty`` is
    stamped only when it was needed and granted).
    """
    root = Path(repo_root).resolve()
    gate = preflight_gated_record(root, out_path, allow_dirty=allow_dirty)
    resolved = resolved_maxim_file(binary)
    executed_root = Path(resolved).resolve().parent.parent.parent if resolved else None
    prov: dict[str, object] = {
        "harness_repo": str(root),
        "harness_git_hash": _git_hash(root),
        "executed_maxim_file": resolved or "unresolved",
        "executed_git_hash": _git_hash(executed_root) if executed_root else "unknown",
        "working_tree_dirty_src_scripts": gate["working_tree_dirty_src_scripts"],
        # The digest describes the tree the dirty flag (and any allowance) was judged on, and only when that
        # is the tree the sub-sims import: an allowance for one tree must never bind to another's code.
        "code_tree_sha256": code_tree_sha256(root) if executed_root == root else "unknown",
        "pythonpath": os.environ.get("PYTHONPATH", ""),
    }
    if gate["allow_dirty"]:
        prov["allow_dirty"] = True
    return prov


# ── D27: committed-evidence writes are opt-in ────────────────────────────────
# The module's THIRD door (after spawn-provenance and in-process gated
# records): harnesses that UPDATE committed evidence under docs/experiments/
# route their output paths through evidence_out_paths(_or_exit).

EVIDENCE_DIR = Path("docs/experiments")


def evidence_out_paths(
    repo_root: Path | str,
    committed_paths: "list[Path | str]",
    *,
    write_experiment_results: bool,
    allow_dirty: bool = False,
) -> "list[Path]":
    """D27: a harness updates COMMITTED evidence only with the explicit opt-in.

    Any path resolving inside ``<repo>/docs/experiments/`` (the S4 results
    JSONs and their committed ``.md`` reports alike) is GOVERNED:

    * without ``--write-experiment-results`` every governed path is REDIRECTED
      into one fresh temp directory (names preserved) and both locations are
      printed — an ordinary or degraded run can never replace real evidence as
      a side effect (the D25 failure class, scripts surface);
    * with the flag, the write additionally refuses a dirty ``src/``+
      ``scripts/`` tree (:class:`DirtyTreeError`, harness policy exit 3)
      unless ``allow_dirty`` — replacing evidence is a deliberate, reviewable
      act performed from established code, mirroring
      ``tests/substrate/conftest.py::publish_sweep_results``.

    Paths outside ``docs/experiments/`` pass through untouched. All governed
    paths share one temp dir so paired artifacts (json + md) stay together.
    """
    import tempfile

    root = Path(repo_root).resolve()
    governed_root = (root / EVIDENCE_DIR).resolve()
    resolved = [Path(p).resolve() for p in committed_paths]
    governed = [p for p in resolved if p.is_relative_to(governed_root)]
    if not governed:
        return resolved
    if not write_experiment_results:
        tmp = Path(tempfile.mkdtemp(prefix="maxim-evidence-"))
        out: "list[Path]" = []
        for p in resolved:
            if p in governed:
                redirected = tmp / p.name
                print(
                    f"[evidence] NOT updating committed record {p.relative_to(root)} "
                    f"(no --write-experiment-results); writing {redirected}"
                )
                out.append(redirected)
            else:
                out.append(p)
        return out
    if working_tree_dirty(root) and not allow_dirty:
        raise DirtyTreeError(
            "refusing to OVERWRITE committed evidence "
            f"({', '.join(str(p.relative_to(root)) for p in governed)}) from a DIRTY tree "
            f"(`git status --porcelain -- {' '.join(DIRTY_SCOPE)}` is not empty in {root}).\n"
            "  A degraded or in-progress run must not replace real evidence (D25/D27).\n"
            "  Fix: commit the harness/src changes and re-run from the clean tree, or pass --allow-dirty."
        )
    return resolved


def evidence_out_path(
    repo_root: Path | str,
    committed_path: "Path | str",
    *,
    write_experiment_results: bool,
    allow_dirty: bool = False,
) -> Path:
    """Single-path convenience over :func:`evidence_out_paths`."""
    return evidence_out_paths(
        repo_root,
        [committed_path],
        write_experiment_results=write_experiment_results,
        allow_dirty=allow_dirty,
    )[0]


def evidence_out_paths_or_exit(
    repo_root: Path | str,
    committed_paths: "list[Path | str]",
    *,
    write_experiment_results: bool,
    allow_dirty: bool = False,
) -> "list[Path]":
    """:func:`evidence_out_paths` with the harness exit-3 policy applied.

    Mirrors :func:`preflight_gated_record_or_exit`: a dirty-tree refusal
    prints ``[FAIL]`` and exits 3 instead of raising a traceback — the
    documented contract for every gated/evidence refusal in this repo.
    """
    try:
        return evidence_out_paths(
            repo_root,
            committed_paths,
            write_experiment_results=write_experiment_results,
            allow_dirty=allow_dirty,
        )
    except DirtyTreeError as exc:
        print(f"[FAIL] evidence-write preflight: {exc}", file=sys.stderr)
        raise SystemExit(3) from exc
