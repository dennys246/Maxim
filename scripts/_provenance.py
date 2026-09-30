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
from the same tree as the harness that calls it. The clean flag and the code
digest are ``src/maxim/utils/code_tree.py``'s (stdlib-only by contract), loaded
the same way: by file path from THIS tree, never through ``sys.path`` (#998).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

__all__ = [
    "GATED_DATA_DIR",
    "HARNESS_RUN_ID_ENV",
    "DirtyTreeError",
    "OwnReportError",
    "SimRunFailed",
    "failed_row",
    "stamp_harness_row",
    "stamp_verdict",
    "is_failed_row",
    "spawn_evidence",
    "depends_on",
    "find_own_report",
    "harness_run_id",
    "list_sessions",
    "sim_evidence",
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
    "working_tree_difference",
    "working_tree_dirty",
]


def _load_code_tree():
    """``src/maxim/utils/code_tree.py`` from THIS tree, by path: the harness's code judged by its own tree's
    rules, whatever ``maxim`` the interpreter would import."""
    path = Path(__file__).resolve().parents[1] / "src" / "maxim" / "utils" / "code_tree.py"
    spec = importlib.util.spec_from_file_location("_maxim_code_tree", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_code_tree = _load_code_tree()

# Anything written here is a GATED record: it backs a ledger row, a result
# doc, or a release gate. The refuse path below applies to this tree only.
GATED_DATA_DIR = Path("docs/experiments/data")

# The paths whose dirtiness makes a run's code-under-test unestablishable (plus the root .gitignore).
DIRTY_SCOPE = _code_tree.SCOPE


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
    # The FULL commit id, as the sim report stamps it, so a harness row and its sims compare byte for byte
    # (M1b, #1003; before, harness stamps were git's default abbreviation and sims 12 characters).
    return _code_tree.head_commit(cwd)


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
    """True unless the code on disk under ``scope`` is exactly HEAD's (#998; ``utils/code_tree.py``). Decided
    by content, not ``git status``, which per-user config and index state could make report a dirty tree
    as clean. Unknown counts as DIRTY: an unestablishable tree is what the refuse path exists to stop."""
    return _code_tree.tree_dirty(repo_root, scope)


def working_tree_difference(repo_root: Path | str, scope: tuple[str, ...] = DIRTY_SCOPE) -> str | None:
    """``None`` when clean, else the first difference as ``"<path>: <why>"`` (``utils/code_tree.py``)."""
    return _code_tree.tree_difference(repo_root, scope)


def code_tree_sha256(repo_root: Path | str, scope: tuple[str, ...] = DIRTY_SCOPE) -> str:
    """The M1 code digest (``utils/code_tree.py``): names the exact code on disk, dirty or not."""
    return _code_tree.code_tree_sha256(repo_root, scope)


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
    difference = working_tree_difference(root)
    dirty = difference is not None
    if gated and dirty and not allow_dirty:
        raise DirtyTreeError(
            f"refusing to write a GATED record ({Path(out_path).resolve().relative_to(root)}) "
            f"from a DIRTY tree: the code on disk in {root} is not HEAD's ({difference}).\n"
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
        "executed_git_hash": _git_hash(root),
        **_run_id_stamp(),
        # Stamped HERE, never by the writer (M1b PR 5a): the evidence gate judges an in-process row by its own
        # provenance and a spawning row by the sims it echoes, so a writer must not choose which.
        "harness_family": "in_process",
        "working_tree_dirty_src_scripts": gate["working_tree_dirty_src_scripts"],
        "code_tree_sha256": code_tree_sha256(root),
        "python": sys.executable,
        "pythonpath": os.environ.get("PYTHONPATH", ""),
    }
    if gate["allow_dirty"]:
        prov["allow_dirty"] = True
    return prov


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
        **_run_id_stamp(),
        "harness_family": "spawning",
    }
    if gate["allow_dirty"]:
        prov["allow_dirty"] = True
    return prov


# ── M1b (#1003): the harness run id, and finding the harness's OWN sim report ─────────────────────────
# A harness mints one run id per process and sets it on every sub-sim's environment; the sim stamps it into
# ``report.json`` (``simulation/report.py::HARNESS_RUN_ID_ENV``), and the harness then reads back exactly the
# report its spawn wrote -- not the newest directory -- and echoes that report's evidence into its row: the
# sim reports live under the gitignored ``data/``, so the committed row is the only record a gate can read.
# The run id is a join key, never evidence: a report binds to a harness only when a harness row names it.

HARNESS_RUN_ID_ENV = "MAXIM_HARNESS_RUN_ID"  # == simulation/report.py::HARNESS_RUN_ID_ENV (pinned by a test)
_RUN_ID: dict[str, str] = {}


def harness_run_id() -> str:
    """This harness process's run id: minted on first call, the same on every later one. ALWAYS fresh -- an
    id inherited from the environment (a parent harness, or a stray shell export) is kept as the parent, never
    reused, so a sim can only ever carry the id of the harness that spawned it. Call it at harness start,
    mock runs included, and set ``env[HARNESS_RUN_ID_ENV] = run_id`` on every spawn."""
    if "id" not in _RUN_ID:
        import uuid

        _RUN_ID["parent"] = os.environ.get(HARNESS_RUN_ID_ENV, "").strip()
        _RUN_ID["id"] = uuid.uuid4().hex
    return _RUN_ID["id"]


def _run_id_stamp() -> dict[str, str]:
    run_id = harness_run_id()
    return {"harness_run_id": run_id, **({"parent_harness_run_id": _RUN_ID["parent"]} if _RUN_ID["parent"] else {})}


class SimRunFailed(ProvenanceError):
    """A spawned sim did not produce a usable run. ``sims`` is the evidence of every report involved (the one
    found, or each candidate), so the harness's failed row says what ran; see :func:`failed_row`."""

    def __init__(self, message: str, *, sims: list[dict[str, object]], detail: dict[str, object] | None = None):
        super().__init__(message)
        self.sims = sims
        self.detail = detail or {}


class OwnReportError(SimRunFailed):
    """The harness could not identify the ONE report its spawn wrote (none, or several, carried its run id).
    ``sims`` is the evidence of every new session directory; ``detail`` has the before/after listings."""


def list_sessions(data_home: Path | str) -> set[str]:
    """The session directory names under ``<data_home>/sim_reports/`` (directories only)."""
    reports = Path(data_home) / "sim_reports"
    return {p.name for p in reports.iterdir() if p.is_dir()} if reports.is_dir() else set()


def _read_report(session_dir: Path) -> dict | None:
    try:
        data = json.loads((session_dir / "report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def find_own_report(data_home: Path | str, run_id: str, before: set[str]) -> tuple[Path, dict]:
    """The one report written since ``before`` (a :func:`list_sessions` snapshot taken inside the spawn
    function, after the home is prepared and immediately before the subprocess) whose
    ``provenance.harness_run_id`` is ``run_id``. None, or several, raise :class:`OwnReportError`: a harness
    never guesses which sim it measured."""
    reports = Path(data_home) / "sim_reports"
    after = list_sessions(data_home)
    new = sorted(after - set(before))
    candidates = [(reports / name, _read_report(reports / name)) for name in new]
    mine = [
        (d, r) for d, r in candidates if r is not None and (r.get("provenance") or {}).get("harness_run_id") == run_id
    ]
    if len(mine) != 1:
        raise OwnReportError(
            f"expected exactly one new report carrying harness_run_id={run_id} under {reports}, found {len(mine)} "
            f"(new session dirs: {new})",
            sims=[sim_evidence(d, r) for d, r in candidates],
            detail={"before": sorted(before), "after": sorted(after)},
        )
    return mine[0]


def spawn_evidence(data_home: Path | str, run_id: str, before: set[str], *, returncode: int) -> tuple[Path, dict]:
    """After a spawn has exited: the report it wrote (:func:`find_own_report`), refusing a run whose sim
    exited non-zero (a typed abort or an error, D22): :class:`SimRunFailed` carries the report's evidence, so
    the abort is recorded rather than dropped. Returns ``(session_dir, report)``."""
    try:
        session_dir, report = find_own_report(data_home, run_id, before)
    except OwnReportError as exc:
        if returncode == 0:
            raise
        raise SimRunFailed(
            f"sub-sim exited {returncode} and no report of its own was found: {exc}", sims=exc.sims, detail=exc.detail
        ) from exc
    if returncode != 0:
        raise SimRunFailed(
            f"sub-sim exited {returncode} (finish_reason={report.get('finish_reason')!r})",
            sims=[sim_evidence(session_dir, report)],
        )
    return session_dir, report


def failed_row(exc: BaseException) -> dict[str, object]:
    """The fields a harness writes for a run that failed, instead of dropping it (#1003): ``status:
    "failed"``, the reason, and whatever sim evidence the failure carried. Readers exclude these rows from
    trial counts (a row with no ``status`` is a legacy ok row)."""
    return {
        "record_kind": "harness_row",
        "status": "failed",
        # First line only, capped: a sub-sim's stderr tail belongs in its log, not in a committed record.
        "reason": f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"[:500],
        "sims": list(getattr(exc, "sims", []) or []),
        **({"failure_detail": exc.detail} if getattr(exc, "detail", None) else {}),
    }


def stamp_harness_row(row: dict, *, mock: bool) -> dict:
    """A harness row as the evidence gate reads it (M1b PR 5a), stamped at the one place a writer writes it:
    ``record_kind``, ``status`` (``failed`` when the row carries a ``refusal`` or was already failed, else ``ok``)
    and an explicit ``mock`` (a missing ``mock`` is never read as "not mock"). Returns the same dict. A row already
    carrying some other ``status`` is refused: overwriting it would erase how the run ended."""
    if row.get("status") not in (None, "ok", "failed"):
        raise ValueError(f"harness row status {row['status']!r} is neither 'ok' nor 'failed'")
    row["record_kind"] = "harness_row"
    refused = row.get("refusal") is not None  # the verdicts' own reading: an empty message is still a refusal
    row["status"] = "failed" if refused or row.get("status") == "failed" else "ok"
    row["mock"] = bool(mock)
    return row


def stamp_verdict(
    verdict: dict, *, repo_root: Path | str, kind: str, data: Path | str, data_bytes: bytes, scope: dict
) -> dict:
    """A verdict as the evidence gate reads it (M1b PR 5a): ``record_kind``, its ``kind`` (the gate owns each
    kind's pass values), the rows file it judged as a REPO-RELATIVE path with the sha256 of ``data_bytes`` — the
    bytes the verdict parsed, read ONCE by the caller, so rows appended while it ran cannot slip under the hash — and the scope that selects every row it read: its selectors (``run_ids``,
    ``campaign_id``), or ``{"all_rows": True}`` when it reads the whole file. An empty scope or a ``None`` selector
    is refused, so "every row" is never the reading of a writer that forgot. Returns the same dict."""
    if not scope or any(v is None for v in scope.values()):
        raise ValueError(f"verdict scope {scope!r}: name the selectors, or {{'all_rows': True}} for the whole file")
    root = Path(repo_root).resolve()
    path = Path(data).resolve()
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        rel = str(data)  # outside the repo: the gate refuses it (not a tracked record)
    verdict["record_kind"] = "verdict"
    verdict["kind"] = kind
    verdict["data"] = rel
    verdict["data_sha256"] = hashlib.sha256(data_bytes).hexdigest()
    verdict["scope"] = scope
    return verdict


def is_failed_row(row: object) -> bool:
    """True for a harness row recording a failed run. Legacy rows carry no ``status`` and count as ok."""
    return isinstance(row, dict) and row.get("status") == "failed"


# The report fields a harness row echoes for each sim it names (the gate judges the row alone).
_EVIDENCE_TOP = ("record_kind", "finish_reason", "ts")
_EVIDENCE_PROVENANCE = (
    "harness_run_id",
    "executed_git_hash",
    "end_executed_git_hash",
    "code_tree_sha256",
    "end_code_tree_sha256",
    "working_tree_dirty_src_scripts",
    "code_changed_during_run",
    "configured_n_ctx",
    "resume",
)
_EVIDENCE_ROLE_SUFFIXES = ("_profile", "_router_n_ctx", "_budget_n_ctx")


def sim_evidence(session_dir: Path | str, report: dict | None) -> dict[str, object]:
    """A sim report's evidence, projected for a harness row: its session id, what it is, how it ended, when,
    and the code/model/context it ran under (each ``provenance`` field copied as stamped; a field the report
    lacks is ``None``, which no gate reads as established). ``report`` ``None`` = no readable report."""
    prov = (report or {}).get("provenance") or {}
    out: dict[str, object] = {"session_id": Path(session_dir).name, "report_found": report is not None}
    for key in _EVIDENCE_TOP:
        out[key] = (report or {}).get(key)
    for key in _EVIDENCE_PROVENANCE:
        out[key] = prov.get(key)
    for key in sorted(prov):
        if key.endswith(_EVIDENCE_ROLE_SUFFIXES):
            out[key] = prov[key]
    return out


def depends_on(data_home: Path | str, before: set[str]) -> list[dict[str, object]]:
    """The evidence of every session already in the home when the sim spawned: the state it inherited
    (a copied or transplanted home, or a ``--resume-sim`` target). Empty for a fresh home."""
    reports = Path(data_home) / "sim_reports"
    return [sim_evidence(reports / name, _read_report(reports / name)) for name in sorted(before)]


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
    difference = working_tree_difference(root)
    if difference is not None and not allow_dirty:
        raise DirtyTreeError(
            "refusing to OVERWRITE committed evidence "
            f"({', '.join(str(p.relative_to(root)) for p in governed)}) from a DIRTY tree "
            f"(the code on disk in {root} is not HEAD's: {difference}).\n"
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
