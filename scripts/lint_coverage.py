#!/usr/bin/env python3
"""Coverage as a ratchet (roadmap 1.3.2 item 5, the GATE half).

Runs in the ``unit-tests`` job right after the required fast suite, which now carries
``--cov=src/maxim --cov-branch --cov-report=json:coverage.json`` (one run, no second pass).
It reads that job's ``coverage.json`` and enforces four things.

**Owner decisions (2026-10-04).** Floors are measured in CI's own environment on the gate
PR and pinned rounded DOWN to 0.1 pt (not the Codex card's v1.3.1 number). A floored scope
fails when measured < floor, and also when measured > floor + 1.0 (raise the floor in the
same PR). Coverage runs inside the required fast suite. Diff coverage: at least 80% of the
added or modified executable lines under ``src/maxim``, files on the reviewed exclusion list
left out. The exclusion list is committed and reviewed, every entry with a reason.

**1. Floors** (``scripts/coverage_floors.json``; statement coverage =
``covered_lines / num_statements``, branch numbers printed, never gated). One ``overall``
floor and one per package: the first path component under ``src/maxim``, with top-level
modules grouped as ``maxim/<root>``. Each floor is ``{percent, missing}``: ``missing`` is the
uncovered statement count pinned with it.

- every run: measured more than k statements under ``percent`` fails (``k = max(ceil(statements
  × 0.001), K_MIN)``, ``K_MIN = 5``, i.e. measured < percent − 100·k/statements); measured > percent + ``band`` fails
  ("raise it"). **The k tolerance below a floor** (owner decision 2026-10-04, "tolerance below
  floor"; minimum ``K_MIN = 5`` statements, owner decision 2026-10-05, because the observed CI
  noise is about ±2 statements whatever the package size, so ``K_MIN`` is twice that): CI coverage moves by a few statements between runs with no code change (maxim/memory
  measured 760 then 762 missing, 85.59% under an 85.6 floor), and a floor rounded down to 0.1
  leaves only 0–0.1 pt of random slack. It is bounded and NON-cumulative: it applies only to this
  HEAD comparison against the committed pin, never to the pin rules below, and pins never move
  with it — so any number of PRs together can drop at most k statements under an unchanged
  floor, and a PR that changes a floor is held to the measurement by the CHANGED/NEW rules. No
  HEAD rule compares measured ``missing`` with a pinned ``missing`` (those comparisons are pin
  rules), so none needed the tolerance;
  ``band`` must be 1.0 and ``min_package_statements`` (N) 200; a package with >= N
  statements needs a floor; a ``null`` floor fails closed printing
  ``floor missing: measured X`` (the bootstrap: the file is committed with nulls, CI prints
  the values, they are committed on the same branch); a floor whose package directory is
  absent is an orphan. **Hysteresis:** a floored package keeps its floor while its
  directory exists, even under N.
- pull requests (against the merge-base copy, ``git show``): ``band``/N/format may not
  change; a new or changed floor is verified against the measurement by the rules under
  "Every new or changed floor is verified data" below (lowered only to the measurement, only
  when uncovered statements did not grow; the ``missing`` pin never rises); a floor may be
  removed only when its directory is gone at HEAD. Package-level renames
  (``git diff -M --name-status``) carry a removed floor to its target, which then follows the
  CHANGED rules against it; with no rename detected, every NEW floor in a PR that removes a
  floored package is held to each removed one the same way.

**2. Diff coverage** (pull requests). Changed lines come from
``git -c diff.renames=false diff --no-renames --no-ext-diff --no-textconv -U0 <merge-base>
-- src/maxim`` against the working tree, so MOVED code counts as changed (the owner's
rule: moved code arrives tested). Each changed line that carries code maps to the start
line of its innermost enclosing ``ast.stmt`` (decorator line too), deduplicated (S1); a clause
header that is not a statement (``except ...:``, ``else:``, ``finally:``, ``case ...:``) maps to
the first statement of its block, so widening an ``except`` reads as covered only if the
handler ran; the
statement is executable when coverage lists it executed or missing. A changed file absent
from ``coverage.json`` and not excluded fails (S2). A changed statement coverage reports
in ``excluded_lines`` counts as UNCOVERED, except a body of imports only under a plain
``if TYPE_CHECKING:`` (S3). Below 80% fails; every uncovered changed line is printed, with
per-file diff coverage. A PR with no executable ``src`` change passes.

**3. Configuration is fixed** (S4). The whole ``[tool.coverage]`` table must equal
:data:`CANONICAL_CONFIG`, except ``run.omit``, which must equal the two fixed globs followed
by exactly the exclusion list's paths; the fixed globs must match no ``src/maxim`` file. It
fails on a ``.coveragerc``, a ``[coverage:`` section in ``setup.cfg``/``tox.ini``, a
``--cov-config``/``--no-cov``/``--cov-fail-under``/``--cov-append``/``-p no:cov`` anywhere
pytest or the workflows read options, any ``COVERAGE_*``/``COV_CORE_*`` name in a workflow,
and ``import coverage``/``import pytest_cov``/``Coverage.current`` in ``tests/`` or ``src/``.
``coverage.json`` must exist and carry ``meta.version`` == the pinned coverage and
``meta.branch_coverage`` true, and every file it names must exist (S8); the job deletes any
earlier ``.coverage``/``coverage.json``/``coverage.xml`` before pytest (S9).

**4. Exclusions and the ledger** (``scripts/coverage_exclusions.json``). Each exclusion is
``{path, reason, covered_by, ref}``; ``covered_by`` must be null, because no lane measures
coverage outside this job (a value such as ``"nightly-model-cache"`` can come back once that
lane produces coverage data the lint reads); the
path must be an existing ``src/maxim`` ``.py`` file. On a PR a NEW exclusion needs a ``ref``
(``#N`` or a github.com PR/issue URL) and is printed loudly. ``ledger`` is append-only (the
base list, as parsed records, is an exact prefix) and holds two kinds of allowance per file:

- ``pragma``: the lexical count of matches of EVERY ``exclude_lines`` regex in the file
  (S3) — not only ``pragma: no cover``, since ``if TYPE_CHECKING:`` hides a whole block —
  except two shapes that hide nothing but themselves, so ordinary code (and every module a
  decomposition creates) does not need a ledger entry: an imports-only ``if TYPE_CHECKING:``
  (no ``else``; the same AST check as the diff rule) and a ``raise NotImplementedError`` that
  is its own statement (no compound statement starts on its line);
- ``excluded_statements``: an excluded file's AST statement count (S5).

Every run, a file's count may not exceed its last ledger entry (0 without one). On a PR, a
count that rose against the merge-base (renames mapped through ``git diff -M``) needs a NEW
entry for that file in the same diff; every new entry must equal the measured count; a new
entry above the base count needs a ``ref``. A newly excluded file's base count is 0.

**Every new or changed floor is verified data** (pull requests; base pin ``b``, head pin ``h``,
measurement ``m``; ``k = max(ceil(statements × 0.001), K_MIN)`` statements of run-to-run noise):

- a NEW floor (bootstrap, a new package): ``h.percent >= round_down(100 × (covered − k) /
  statements)`` and ``h.missing <= m.missing``;
- a CHANGED floor: ``h.missing <= min(m.missing, b.missing)`` — the missing pin NEVER rises.
  LOWERING (``h.percent < b.percent``) needs, with no tolerance, ``m.missing <= b.missing``
  (covered code was deleted; uncovered code did not grow) and ``h.percent >= round_down(m)``
  (lowered exactly to the measurement). Raising or keeping ``percent`` keeps the k tolerance;
- a rename carry-over target follows the CHANGED rules against the removed package's floor;
  and a split may not lose the sum: over each group of removed packages that share targets,
  the targets' measured ``missing`` summed may not exceed the removed packages' pinned
  ``missing`` summed (plus the pin of any target that already had a floor at base).

Why this shape: three review rounds each found a walk-down when floors were allowed a
tolerance on the way DOWN (a floors-only lower; a high ``missing`` pin spent by the next PR;
a lower-then-repin pair, 68.4 → 67.9 over 7 PRs). The invariant closes the class: a floor
falls only to the measurement and only when uncovered statements did not grow, and the pin
those PRs compare against can only fall. Deleting covered code still lowers a floor (missing
unchanged, the measurement falls, pin ``round_down(m)``). Cost: a PR that adds uncovered code
and must raise a floor pins the base's lower ``missing``, so a later deletion must also cover
those statements before its floor can drop.

**Bootstrap** (the gate PR, when the base has neither committed file). The floors file is
committed with ``null`` values; the lint fails closed and prints a paste-ready measured floors
file. Take it from TWO CI runs (re-run the job), save each as JSON and pin the per-scope minimum:
``python scripts/lint_coverage.py --merge-floors run1.json run2.json`` (lowest percent, lowest
missing). That minimum passes against either run whenever the runs differ by at most ``k``
covered statements per scope (every floor is NEW on the gate PR); a scope that varies more is flaky and needs a look. Floor diff rules
against base do not apply (every floor is new, so it is checked against the measurement); an
exclusion is grandfathered when its path matched a non-fixed ``omit`` pattern in the base
``pyproject.toml`` (today ``selfy.py``), and any other needs a ``ref``.

**The gate checks its own wiring**, from the parsed YAML of the ``unit-tests`` job in
``.github/workflows/test.yml`` (a comment satisfies nothing): every checkout has
``with.fetch-depth == 0``; a step runs the exact coverage pin; exactly one step measures, with the
cleanup and the exact ``--cov`` arguments; exactly one later step runs exactly
``python scripts/lint_coverage.py`` and carries only ``name`` and ``run`` (no ``if:``, ``env:``,
``shell:``, ``with:``, ``|| true``); the only step allowed between the two is the MemoryHub step,
matched by its run line, so nothing can rewrite ``coverage.json``; no ``continue-on-error``.

**When rules run:** every event runs 1, 3 and the head half of 4. On ``pull_request`` the
diff rules are the gate and any git failure is an error (``_lint_git.must_not_skip``,
exit 2) — the job checks out with ``fetch-depth: 0``. Locally they run when a merge-base
exists (the diff is against the working tree, so untracked files are not seen). A push to main
runs them too, against the last push whose ``unit-tests`` job passed (``_lint_git.push_base``,
#1089), with the same exit-2 rule. Other events run the head rules only. Every run prints overall + per-package statement and branch
coverage, so the numbers live in the CI log.

**Residuals (not mechanized):**

- tests that execute code but assert nothing: coverage without behaviour. The push's
  quality rule (behavioural tests, deletion probes) is enforced by review;
- dilution: 80% of a diff can be met by covering trivial lines while the risky ones stay
  uncovered; the printed uncovered lines are for the reviewer;
- code that only runs in a subprocess or a fork is not measured (pytest-cov 7 does not
  measure subprocesses; tests do not run sims by rule);
- ``scripts/`` is out of scope (the floors are ``src/maxim`` only);
- determined evasion: ``sys.settrace``/``sys.monitoring`` tampering, ``__import__`` of the
  coverage API, or a plugin handle from ``config.pluginmanager`` are not detected; the
  bans catch forgetting, not malice;
- timing-dependent branches can move a scope by a few statements between runs (``k`` absorbs
  it in ``percent``; ``missing`` takes the lower run; see "Bootstrap");
- the pinned ``missing`` goes stale between floor changes: it is verified only when its floor
  is new or changed, so a floor untouched for many PRs carries an old ``missing``, and the
  drop rule then compares against that old number;
- deleting UNCOVERED code lowers measured ``missing`` below the pin, and that headroom can
  then be spent removing tests elsewhere in the scope without the floor rules noticing (the
  percent may still have to hold). It costs deleting real code; review reads it;
- deleting a package and recreating it under a new name later resets it as a NEW floor with
  the k tolerance: bounded at about k statements per cycle, and each cycle costs a real
  deletion plus 80% diff coverage on the recreated code;
- pre-suite tampering: a step BEFORE the measuring suite (or a test-dependency install) can
  plant a ``sitecustomize.py`` / ``.pth`` file or patch the environment so the measurement
  itself lies; the CI check only constrains the steps between the suite and the gate. Review
  reads the job;
- the pragma swap: removing one exclusion match and adding ``# pragma: no cover`` to a large
  function's ``def`` keeps the file's count while hiding far more code. The ledger counts
  matches, not excluded lines; the diff rule still counts the changed ``def`` line as
  uncovered, and review must read it;
- merging floored package X into an existing floored package Y: X's floor carries to Y, so
  the merge passes only when Y's floor is at or above X's or Y's measured ``missing`` is at or
  below X's pinned ``missing``; when it is not, the merge cannot pass as one PR (move X to a
  NEW package name instead, which carries X's floor, then merge it in a later PR). There is
  no exception mechanism for floors by design.

Regression guard: tests/unit/test_lint_coverage.py (fixture git repos + synthetic
``coverage.json`` driving ``main()``; each mechanism deletion-proven) and the "Coverage
ratchet (roadmap 1.3.2 item 5)" step in .github/workflows/test.yml (unit-tests job).

Exits: 0 clean; 1 a rule failed (stderr); 2 the measurement or git could not answer.
Stdlib plus ``pyyaml`` (to read the workflow structurally); does not import ``maxim`` or ``coverage``.
"""

from __future__ import annotations

import ast
import fnmatch
import io
import json
import math
import os
import re
import sys
import time
import tokenize
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import yaml  # noqa: E402  (pyyaml: pinned in the unit-tests job's first install, not the guarded one)
from _lint_git import GitUnavailable, base_ref, changed_files, git, must_not_skip, show  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SCOPE = "src/maxim"
FLOORS_REL = "scripts/coverage_floors.json"
EXCLUSIONS_REL = "scripts/coverage_exclusions.json"
COVERAGE_JSON = "coverage.json"

COVERAGE_VERSION = "7.13.3"  # == the `coverage==` pin in the unit-tests job
BAND = 1.0
TICK = 0.1  # a floor's resolution; k = max(ceil(statements * TICK / 100), K_MIN) statements of run-to-run noise
# The minimum k (owner decision 2026-10-05). CI coverage noise is ABSOLUTE, about ±2 statements whatever a package's
# size (maxim/retrieval, 296 statements: 59 missing in CI run 5 against 61 in earlier runs), so a size-proportional k
# of 1 made small floors unpinnable. K_MIN is twice the observed noise. Fixed here: the floors file cannot carry it
# (its keys are exact), so no PR can change it short of editing the lint, which review reads.
K_MIN = 5
MIN_PACKAGE_STATEMENTS = 200
DIFF_THRESHOLD = 80.0
FLOORS_FORMAT_VERSION = 1
EXCLUSIONS_FORMAT_VERSION = 1
ROOT_PACKAGE = "maxim/<root>"

FIXED_OMIT = ["*/tests/*", "*/__pycache__/*"]
WORKFLOW_REL = ".github/workflows/test.yml"
CI_JOB = "unit-tests"
CI_COV_ARGS = "--cov=src/maxim --cov-branch --cov-report=json:coverage.json --cov-report="
CI_LINT_RUN = "run: python scripts/lint_coverage.py"
CI_CLEAN = "rm -f .coverage coverage.json coverage.xml"
CI_BETWEEN_RUN = "python -m pytest tests/integration/test_memory_hub.py -q"  # the one step allowed in between
CANONICAL_EXCLUDE_LINES = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
]
CANONICAL_CONFIG: dict = {
    "run": {"source": [SCOPE], "omit": None, "branch": True},  # omit: FIXED_OMIT + the exclusion paths
    "report": {"exclude_lines": CANONICAL_EXCLUDE_LINES, "show_missing": True},
}
# Only null: no nightly lane measures coverage today, so a `covered_by` claim would be unmeasured.
# A value (e.g. "nightly-model-cache") comes back when that lane produces coverage data the lint can read.
COVERED_BY = {None}
LEDGER_KINDS = ("pragma", "excluded_statements")

_REF_RE = re.compile(r"#\d+|https://github\.com/[\w.-]+/[\w.-]+/(?:pull|issues)/\d+")
_BANNED_OPTS_RE = re.compile(r"--cov-config\b|--no-cov(?![\w-])|--cov-fail-under\b|--cov-append\b|no:(?:pytest_)?cov\b")
_BANNED_ENV_RE = re.compile(r"\b(?:COVERAGE|COV_CORE)_[A-Z_]+")
_BANNED_IMPORT_RE = re.compile(
    r"^\s*(?:import|from)\s+(?:coverage|pytest_cov)\b|\bCoverage\.current\b|^\s*import\s+[\w.]+\s*,\s*coverage\b",
    re.MULTILINE,
)
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


class MeasurementError(RuntimeError):
    """coverage.json is missing, stale or unreadable — exit 2, never pass."""


class FileFormatError(ValueError):
    """A committed gate file does not parse — reported as a rule failure."""


# ── coverage.json ─────────────────────────────────────────────────────────────


@dataclass
class FileCov:
    executed: set[int]
    missing: set[int]
    excluded: set[int]
    statements: int
    covered: int
    branches: int
    covered_branches: int


@dataclass
class ScopeCov:
    statements: int = 0
    covered: int = 0
    branches: int = 0
    covered_branches: int = 0

    @property
    def missing(self) -> int:
        return self.statements - self.covered

    @property
    def percent(self) -> float:
        return 100.0 * self.covered / self.statements if self.statements else 100.0

    def add(self, f: FileCov) -> None:
        self.statements += f.statements
        self.covered += f.covered
        self.branches += f.branches
        self.covered_branches += f.covered_branches


def load_coverage(root: Path) -> dict[str, FileCov]:
    path = root / COVERAGE_JSON
    if not path.is_file():
        raise MeasurementError(
            f"{COVERAGE_JSON} not found — the fast suite must run with --cov-report=json:{COVERAGE_JSON} first"
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data["meta"]
        raw_files = data["files"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise MeasurementError(f"{COVERAGE_JSON} unreadable: {e}") from e
    if meta.get("version") != COVERAGE_VERSION:
        raise MeasurementError(
            f"{COVERAGE_JSON} was written by coverage {meta.get('version')!r}, the pin is {COVERAGE_VERSION} (stale file?)"
        )
    if meta.get("branch_coverage") is not True:
        raise MeasurementError(f"{COVERAGE_JSON} was not measured with branch coverage (--cov-branch)")
    out: dict[str, FileCov] = {}
    for key, f in raw_files.items():
        p = Path(key)
        rel = p.relative_to(root).as_posix() if p.is_absolute() and p.is_relative_to(root) else p.as_posix()
        if not rel.startswith(SCOPE + "/"):
            raise MeasurementError(f"{COVERAGE_JSON} names a file outside {SCOPE}: {key}")
        if not (root / rel).is_file():
            raise MeasurementError(f"{COVERAGE_JSON} names {rel}, which does not exist (stale file?)")
        try:
            s = f["summary"]
            out[rel] = FileCov(
                executed=set(f["executed_lines"]),
                missing=set(f["missing_lines"]),
                excluded=set(f["excluded_lines"]),
                statements=int(s["num_statements"]),
                covered=int(s["covered_lines"]),
                branches=int(s.get("num_branches", 0)),
                covered_branches=int(s.get("covered_branches", 0)),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise MeasurementError(f"{COVERAGE_JSON}: entry for {key} unreadable: {e}") from e
    if not out:
        raise MeasurementError(f"{COVERAGE_JSON} measured no files")
    return out


def package_of(rel: str) -> str:
    parts = rel[len(SCOPE) + 1 :].split("/")
    return ROOT_PACKAGE if len(parts) == 1 else f"maxim/{parts[0]}"


def package_present(root: Path, pkg: str) -> bool:
    src = root / SCOPE
    if pkg == ROOT_PACKAGE:
        return any(src.glob("*.py"))
    d = src / pkg.split("/", 1)[1]
    return d.is_dir() and any(d.rglob("*.py"))


def measure_scopes(cov: dict[str, FileCov]) -> tuple[ScopeCov, dict[str, ScopeCov]]:
    overall = ScopeCov()
    pkgs: dict[str, ScopeCov] = {}
    for rel, f in cov.items():
        overall.add(f)
        pkgs.setdefault(package_of(rel), ScopeCov()).add(f)
    return overall, pkgs


def round_down(pct: float) -> float:
    return math.floor(pct * 10 + 1e-9) / 10


def missing_slack(m: ScopeCov) -> int:
    """k: run-to-run coverage noise in STATEMENTS — one 0.1-pt tick of the scope rounded up, but never under
    ``K_MIN`` (noise is absolute, about ±2 statements). Used by the below-floor HEAD tolerance and the new/raised
    floor tolerance; bounded per scope and non-cumulative, because pins never move with it."""
    return max(math.ceil(m.statements * TICK / 100), K_MIN)


# ── floors ────────────────────────────────────────────────────────────────────


@dataclass
class Floor:
    percent: float | None
    missing: int | None


@dataclass
class Floors:
    band: float
    min_statements: int
    version: int
    overall: Floor
    packages: dict[str, Floor] = field(default_factory=dict)


def _parse_floor(name: str, v: object) -> Floor:
    if not isinstance(v, dict) or set(v) != {"percent", "missing"}:
        raise FileFormatError(f"floor {name!r} must be exactly {{percent, missing}}: {v!r}")
    pct, miss = v["percent"], v["missing"]
    if pct is not None:
        if not isinstance(pct, (int, float)) or isinstance(pct, bool) or not 0 <= pct <= 100:
            raise FileFormatError(f"floor {name!r}: percent must be a number in [0, 100] or null: {pct!r}")
        if abs(pct * 10 - round(pct * 10)) > 1e-6:
            raise FileFormatError(f"floor {name!r}: percent has more than one decimal: {pct!r}")
    if miss is not None and (not isinstance(miss, int) or isinstance(miss, bool) or miss < 0):
        raise FileFormatError(f"floor {name!r}: missing must be a non-negative int or null: {miss!r}")
    if (pct is None) != (miss is None):
        raise FileFormatError(f"floor {name!r}: percent and missing are pinned together (both or neither null)")
    return Floor(float(pct) if pct is not None else None, miss)


def parse_floors(text: str) -> Floors:
    try:
        d = json.loads(text)
    except json.JSONDecodeError as e:
        raise FileFormatError(f"not JSON: {e}") from e
    keys = {"floors_format_version", "band", "min_package_statements", "overall", "packages"}
    if not isinstance(d, dict) or set(d) != keys:
        raise FileFormatError(f"top-level keys must be exactly {sorted(keys)}")
    if not isinstance(d["packages"], dict):
        raise FileFormatError("packages must be an object")
    return Floors(
        band=d["band"],
        min_statements=d["min_package_statements"],
        version=d["floors_format_version"],
        overall=_parse_floor("overall", d["overall"]),
        packages={k: _parse_floor(k, v) for k, v in d["packages"].items()},
    )


def _fmt_pct(x: float) -> str:
    return f"{x:.2f}%"


def floor_head_rules(root: Path, fl: Floors, overall: ScopeCov, pkgs: dict[str, ScopeCov]) -> list[str]:
    out: list[str] = []
    if fl.version != FLOORS_FORMAT_VERSION:
        out.append(f"{FLOORS_REL}: floors_format_version must be {FLOORS_FORMAT_VERSION}, got {fl.version!r}")
    if fl.band != BAND:
        out.append(f"{FLOORS_REL}: band must be {BAND} (owner decision 2026-10-04), got {fl.band!r}")
    if fl.min_statements != MIN_PACKAGE_STATEMENTS:
        out.append(f"{FLOORS_REL}: min_package_statements must be {MIN_PACKAGE_STATEMENTS}, got {fl.min_statements!r}")

    def check(name: str, f: Floor, m: ScopeCov) -> None:
        if f.percent is None:
            out.append(
                f"floor missing: {name} measured {_fmt_pct(m.percent)} (missing {m.missing}) — pin "
                f'{{"percent": {round_down(m.percent)}, "missing": {m.missing}}}'
            )
        elif m.percent < f.percent - below_tolerance(m) - 1e-9:
            # Owner decision 2026-10-04 ("tolerance below floor"): fail only beyond k statements under the floor.
            out.append(
                f"{name}: {_fmt_pct(m.percent)} is below its floor {f.percent}% by more than the "
                f"{missing_slack(m)}-statement noise tolerance ({_fmt_pct(f.percent - below_tolerance(m))}; "
                f"missing {m.missing})"
            )
        elif m.percent > f.percent + BAND + 1e-9:
            out.append(
                f"{name}: {_fmt_pct(m.percent)} is more than {BAND} pt above its floor {f.percent}% — raise it to "
                f'{{"percent": {round_down(m.percent)}, "missing": {m.missing}}}'
            )

    check("overall", fl.overall, overall)
    for pkg, f in sorted(fl.packages.items()):
        if not package_present(root, pkg):
            out.append(f"{pkg}: floored package has no directory at HEAD (orphan) — remove the entry")
        elif pkg not in pkgs or pkgs[pkg].statements == 0:
            out.append(f"{pkg}: floored package has no measured statements (all excluded?)")
        else:
            check(pkg, f, pkgs[pkg])
    for pkg, m in sorted(pkgs.items()):
        if pkg not in fl.packages and m.statements >= MIN_PACKAGE_STATEMENTS:
            out.append(
                f"{pkg}: {m.statements} statements (>= {MIN_PACKAGE_STATEMENTS}) and no floor — add "
                f'"{pkg}": {{"percent": {round_down(m.percent)}, "missing": {m.missing}}}'
            )
    return out


def package_renames(root: Path, base: str) -> dict[str, set[str]]:
    """{base package: packages its renamed files moved to} from ``git diff -M --name-status``."""
    out: dict[str, set[str]] = {}
    for new, old in changed_files(root, base, SCOPE):
        if old and old != new and old.startswith(SCOPE + "/") and new.startswith(SCOPE + "/"):
            a, b = package_of(old), package_of(new)
            if a != b:
                out.setdefault(a, set()).add(b)
    return out


def floor_diff_rules(
    root: Path,
    base_fl: Floors,
    fl: Floors,
    overall: ScopeCov,
    pkgs: dict[str, ScopeCov],
    renames: dict[str, set[str]],
) -> list[str]:
    out: list[str] = []
    for attr, label in (
        ("band", "band"),
        ("min_statements", "min_package_statements"),
        ("version", "floors_format_version"),
    ):
        if getattr(base_fl, attr) != getattr(fl, attr):
            out.append(f"{label} changed {getattr(base_fl, attr)!r} -> {getattr(fl, attr)!r}: it is fixed")

    def changed_entry(name: str, b: Floor | None, h: Floor, m: ScopeCov | None) -> None:
        """NEW entry (``b`` None): the measurement within k statements, ``missing`` <= measured. CHANGED entry:
        ``missing`` never rises (<= min(measured, base)); LOWERING needs measured missing <= base missing AND
        percent >= round_down(measured), no tolerance; raising (or unchanged percent) keeps the k tolerance."""
        if m is None or h.percent is None or h.missing is None:
            return
        based = b is not None and b.percent is not None and b.missing is not None
        if based and (b.percent, b.missing) == (h.percent, h.missing):
            return
        k = missing_slack(m)
        lowest = round_down(100.0 * (m.covered - k) / m.statements) if m.statements else 0.0
        cap = min(m.missing, b.missing) if based else m.missing
        if h.missing > cap:
            out.append(
                f"{name}: floor pins missing {h.missing} above "
                + (f"min(measured {m.missing}, base {b.missing})" if based else f"the measured {m.missing}")
                + f" — a pinned missing never rises; pin {cap}"
            )
        if based and h.percent < b.percent - 1e-9:
            if m.missing > b.missing:
                out.append(
                    f"{name}: floor lowered {b.percent}% -> {h.percent}% but measured missing {m.missing} > the base's "
                    f"pinned missing {b.missing} — cover {m.missing - b.missing} statement(s) first: a floor drops "
                    "only when covered code is deleted, not when uncovered code is added or tests are removed"
                )
            if h.percent + 1e-9 < round_down(m.percent):
                out.append(
                    f"{name}: floor lowered to {h.percent}%, below the measurement {_fmt_pct(m.percent)} — a lowered "
                    f"floor goes exactly to round_down(measured) = {round_down(m.percent)} (no tolerance)"
                )
        elif h.percent + 1e-9 < lowest:
            out.append(
                f"{name}: {'new' if not based else 'changed'} floor {h.percent}% is below the measured "
                f"{_fmt_pct(m.percent)} less {k} statement(s) of run-to-run noise ({lowest}%) — pin {round_down(m.percent)}"
            )

    changed_entry("overall", base_fl.overall, fl.overall, overall)
    if base_fl.overall.percent is not None and fl.overall.percent is None:
        out.append("overall: floor set back to null")
    removed = []
    for pkg, b in sorted(base_fl.packages.items()):
        h = fl.packages.get(pkg)
        if h is None:
            if b.percent is None:
                continue
            if package_present(root, pkg):
                out.append(
                    f"{pkg}: floor removed while its directory still exists (a floor stays until the package is gone)"
                )
            else:
                removed.append((pkg, b))
            continue
        if b.percent is not None and h.percent is None:
            out.append(f"{pkg}: floor set back to null")
            continue
        changed_entry(pkg, b, h, pkgs.get(pkg))
    new_keys = sorted(k for k in fl.packages if k not in base_fl.packages)
    carried: set[str] = set()
    targets_of: dict[str, list[str]] = {}
    for pkg, b in removed:
        # A carry-over target follows the CHANGED rules against the removed package's floor. With no rename
        # detected, every new floor in this diff is a candidate target (fail closed).
        targets_of[pkg] = sorted(renames.get(pkg, set())) or new_keys
        for t in targets_of[pkg]:
            h = fl.packages.get(t)
            if h is None or h.percent is None:
                out.append(f"{t}: received {pkg}'s renamed files; the floor carries over — add a floor for {t}")
                continue
            carried.add(t)
            changed_entry(f"{t} (carrying {pkg})", b, h, pkgs.get(t))
    # The SUM rule: a split must not lose uncovered statements between its pieces (each piece alone is under the
    # carried pin; together they may not be). Removed packages that share a target are one group; the group's
    # budget is the sum of its removed packages' base `missing` plus the base `missing` of every target that
    # already had a floor at base (a merge into an existing package brings that package's own pin along).
    groups: list[tuple[set[str], set[str]]] = []
    for pkg in targets_of:
        srcs, tgts = {pkg}, set(targets_of[pkg])
        for g in [g for g in groups if g[1] & tgts]:
            groups.remove(g)
            srcs |= g[0]
            tgts |= g[1]
        groups.append((srcs, tgts))
    base_missing = {pkg: b.missing for pkg, b in removed}
    for srcs, tgts in groups:
        if not tgts or any(base_missing[s] is None for s in srcs):
            continue
        budget = sum(base_missing[s] or 0 for s in srcs)
        budget += sum(base_fl.packages[t].missing or 0 for t in tgts if t in base_fl.packages)
        measured = sum(pkgs[t].missing for t in tgts if t in pkgs)
        if measured > budget:
            out.append(
                f"{' + '.join(sorted(tgts))}: received {', '.join(sorted(srcs))}; together they measure missing "
                f"{measured} > the carried pinned missing {budget} — cover {measured - budget} statement(s) "
                "(a split may not lose uncovered statements between its pieces)"
            )
    for k in new_keys:
        if k not in carried:
            changed_entry(k, None, fl.packages[k], pkgs.get(k))
    return out


def below_tolerance(m: ScopeCov) -> float:
    """The HEAD check's allowance under a floor, in percentage points: k statements of the scope (k as in
    ``missing_slack``). Pins never move with it, so it does not accumulate across PRs."""
    return 100.0 * missing_slack(m) / m.statements if m.statements else 0.0


def suggested_floors(fl: Floors, overall: ScopeCov, pkgs: dict[str, ScopeCov]) -> dict:
    """The floors file as measured: every floored package plus every package at or over N."""
    keys = set(fl.packages) | {k for k, m in pkgs.items() if m.statements >= MIN_PACKAGE_STATEMENTS}

    def pin(m: ScopeCov | None) -> dict:
        return {"percent": round_down(m.percent), "missing": m.missing} if m else {"percent": None, "missing": None}

    return {
        "floors_format_version": FLOORS_FORMAT_VERSION,
        "band": BAND,
        "min_package_statements": MIN_PACKAGE_STATEMENTS,
        "overall": pin(overall),
        "packages": {k: pin(pkgs.get(k)) for k in sorted(keys)},
    }


# ── exclusions + ledger ───────────────────────────────────────────────────────


@dataclass
class Exclusions:
    entries: list[dict]
    ledger: list[dict]

    @property
    def paths(self) -> list[str]:
        return [e["path"] for e in self.entries]

    def allowance(self, kind: str, rel: str) -> int:
        n = 0
        for x in self.ledger:
            if x["kind"] == kind and x["file"] == rel:
                n = x["count"]
        return n


def _ref_ok(ref: object) -> bool:
    return isinstance(ref, str) and bool(_REF_RE.fullmatch(ref))


def parse_exclusions(text: str) -> Exclusions:
    try:
        d = json.loads(text)
    except json.JSONDecodeError as e:
        raise FileFormatError(f"not JSON: {e}") from e
    keys = {"exclusions_format_version", "exclusions", "ledger"}
    if not isinstance(d, dict) or set(d) != keys:
        raise FileFormatError(f"top-level keys must be exactly {sorted(keys)}")
    if d["exclusions_format_version"] != EXCLUSIONS_FORMAT_VERSION:
        raise FileFormatError(f"exclusions_format_version must be {EXCLUSIONS_FORMAT_VERSION}")
    if not isinstance(d["exclusions"], list) or not isinstance(d["ledger"], list):
        raise FileFormatError("exclusions and ledger must be lists")
    seen: set[str] = set()
    for e in d["exclusions"]:
        if not isinstance(e, dict) or set(e) != {"path", "reason", "covered_by", "ref"}:
            raise FileFormatError(f"exclusion must be exactly {{path, reason, covered_by, ref}}: {e!r}")
        if not isinstance(e["path"], str) or e["path"] in seen:
            raise FileFormatError(f"exclusion path must be a unique string: {e!r}")
        seen.add(e["path"])
        if not (isinstance(e["reason"], str) and e["reason"].strip()):
            raise FileFormatError(f"exclusion needs a non-empty reason: {e['path']}")
        if e["covered_by"] not in COVERED_BY:
            raise FileFormatError(f"exclusion covered_by must be null (no lane measures it): {e!r}")
        if e["ref"] is not None and not _ref_ok(e["ref"]):
            raise FileFormatError(f"exclusion ref must be '#NNN', a github.com PR/issue URL or null: {e!r}")
    for x in d["ledger"]:
        if not isinstance(x, dict) or set(x) != {"kind", "file", "count", "reason", "ref"}:
            raise FileFormatError(f"ledger entry must be exactly {{kind, file, count, reason, ref}}: {x!r}")
        if x["kind"] not in LEDGER_KINDS or not isinstance(x["file"], str):
            raise FileFormatError(f"ledger entry kind must be one of {LEDGER_KINDS}: {x!r}")
        if not isinstance(x["count"], int) or isinstance(x["count"], bool) or x["count"] < 0:
            raise FileFormatError(f"ledger count must be a non-negative int: {x!r}")
        if not (isinstance(x["reason"], str) and x["reason"].strip()):
            raise FileFormatError(f"ledger entry needs a non-empty reason: {x!r}")
        if x["ref"] is not None and not _ref_ok(x["ref"]):
            raise FileFormatError(f"ledger ref must be '#NNN', a github.com PR/issue URL or null: {x!r}")
    return Exclusions(list(d["exclusions"]), list(d["ledger"]))


_TC_RX, _NIE_RX = CANONICAL_EXCLUDE_LINES[1], CANONICAL_EXCLUDE_LINES[2]


def _exempt_lines(text: str) -> tuple[set[int], set[int]]:
    """(imports-only ``if TYPE_CHECKING:`` lines, lone ``raise NotImplementedError`` lines): matches that
    exclude nothing but themselves and imports. An unparseable file exempts nothing."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set(), set()
    tc: set[int] = set()
    raises: set[int] = set()
    starts: dict[int, list[ast.stmt]] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.stmt):
            starts.setdefault(n.lineno, []).append(n)
        if isinstance(n, ast.If) and _imports_only_tc(n):
            tc.add(n.lineno)
    for line, nodes in starts.items():
        if all(isinstance(n, ast.Raise) for n in nodes):
            raises.add(line)  # no compound statement starts on this line, so only the raise is excluded
    return tc, raises


def exclusion_matches(text: str) -> int:
    """Lexical matches of every ``exclude_lines`` regex (coverage matches them over the whole text), except an
    imports-only ``if TYPE_CHECKING:`` and a ``raise NotImplementedError`` that is its own statement."""
    tc, raises = _exempt_lines(text)
    n = 0
    for rx in CANONICAL_EXCLUDE_LINES:
        for m in re.finditer(rx, text, re.MULTILINE):
            line = text.count("\n", 0, m.start()) + 1
            if (rx == _TC_RX and line in tc) or (rx == _NIE_RX and line in raises):
                continue
            n += 1
    return n


def statement_count(text: str) -> int:
    return sum(isinstance(n, ast.stmt) for n in ast.walk(ast.parse(text)))


def _count(kind: str, text: str) -> int:
    return exclusion_matches(text) if kind == "pragma" else statement_count(text)


def _src_files(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in (root / SCOPE).rglob("*.py"))


def exclusion_head_rules(root: Path, ex: Exclusions) -> list[str]:
    out: list[str] = []
    for p in ex.paths:
        if not (p.startswith(SCOPE + "/") and p.endswith(".py") and (root / p).is_file()):
            out.append(f"exclusion {p}: not an existing {SCOPE} .py file (orphan) — remove the entry")
            continue
        n = statement_count((root / p).read_text(encoding="utf-8"))
        allow = ex.allowance("excluded_statements", p)
        if n > allow:
            out.append(
                f"exclusion {p}: {n} statements exceed its ledger allowance {allow} — append "
                f'{{"kind": "excluded_statements", "file": "{p}", "count": {n}, "reason": ..., "ref": "#N"}}'
            )
    for rel in _src_files(root):
        n = exclusion_matches((root / rel).read_text(encoding="utf-8", errors="replace"))
        allow = ex.allowance("pragma", rel)
        if n > allow:
            out.append(
                f"{rel}: {n} coverage-exclusion matches exceed its ledger allowance {allow} — append "
                f'{{"kind": "pragma", "file": "{rel}", "count": {n}, "reason": ..., "ref": "#N"}}'
            )
    return out


def _omit_extras(pyproject_text: str) -> list[str]:
    try:
        cov = tomllib.loads(pyproject_text).get("tool", {}).get("coverage", {})
    except tomllib.TOMLDecodeError:
        return []
    return [p for p in cov.get("run", {}).get("omit", []) if p not in FIXED_OMIT]


def exclusion_diff_rules(root: Path, base: str, ex: Exclusions, base_ex: Exclusions | None) -> list[str]:
    out: list[str] = []
    if base_ex is None:  # bootstrap: grandfather what the base pyproject already omitted
        pats = _omit_extras(show(root, base, "pyproject.toml"))
        base_excluded = {p for p in ex.paths if any(fnmatch.fnmatch(p, pat) for pat in pats)}
        base_ledger: list[dict] = []
    else:
        base_excluded = set(base_ex.paths)
        base_ledger = base_ex.ledger
        if ex.ledger[: len(base_ledger)] != base_ledger:
            return [
                "ledger is append-only: the merge-base list is not an exact prefix of this one (edited, reordered or removed)"
            ]
    for e in ex.entries:
        if e["path"] not in base_excluded:
            print(f"coverage: NEW EXCLUSION {e['path']} (ref {e['ref']}): {e['reason']}")
            if not _ref_ok(e["ref"]):
                out.append(f"exclusion {e['path']}: a new exclusion needs a ref ('#NNN' or a github.com PR/issue URL)")

    renamed = {new: old for new, old in changed_files(root, base, SCOPE)}

    def base_count(kind: str, rel: str) -> int:
        src = renamed.get(rel, rel)
        if kind == "excluded_statements" and src not in base_excluded and rel not in base_excluded:
            return 0
        text = show(root, base, src)
        return _count(kind, text) if text else 0

    def head_count(kind: str, rel: str) -> int | None:
        p = root / rel
        return _count(kind, p.read_text(encoding="utf-8", errors="replace")) if p.is_file() else None

    new_entries = ex.ledger[len(base_ledger) :]
    for x in new_entries:
        h = head_count(x["kind"], x["file"])
        if h is None or x["count"] != h:
            out.append(
                f"new ledger entry {x['kind']} {x['file']} = {x['count']}: must equal the measured count ({h}) — "
                "no pre-approving a later rise"
            )
            continue
        if x["count"] > base_count(x["kind"], x["file"]) and not _ref_ok(x["ref"]):
            out.append(f"new ledger entry {x['kind']} {x['file']} = {x['count']} raises the count: it needs a ref")
    granted = {(x["kind"], x["file"]) for x in new_entries}
    checks = [("pragma", rel) for rel in sorted(renamed) if (root / rel).is_file()]
    checks += [("excluded_statements", p) for p in ex.paths if (root / p).is_file()]
    for kind, rel in checks:
        h, b = head_count(kind, rel), base_count(kind, rel)
        if h is not None and h > b and (kind, rel) not in granted:
            out.append(
                f"{rel}: {kind} count rose {b} -> {h} on this branch — needs a new ref'd ledger entry in this diff"
            )
    return out


# ── configuration (S4) ────────────────────────────────────────────────────────


def ci_step_rules(root: Path) -> list[str]:
    """The gate's own CI wiring, read from the PARSED workflow (a comment satisfies nothing): a disabled,
    softened or bypassed step would make every other rule here vacuous."""
    p = root / WORKFLOW_REL
    where = f"{WORKFLOW_REL} `{CI_JOB}`"
    try:
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
        steps = doc["jobs"][CI_JOB]["steps"]
        job = doc["jobs"][CI_JOB]
    except (OSError, yaml.YAMLError, KeyError, TypeError) as e:
        return [f"{where}: cannot read the job's steps ({e}) — the coverage gate's wiring cannot be verified"]
    if not isinstance(steps, list) or not all(isinstance(s, dict) for s in steps):
        return [f"{where}: steps must be a list of mappings"]
    out: list[str] = []

    def run(s: dict) -> str:
        r = s.get("run")
        return r if isinstance(r, str) else ""

    def flat(s: dict) -> str:
        return re.sub(r"\s+", " ", run(s).replace("\\\n", " ")).strip()

    checkout = [s for s in steps if str(s.get("uses", "")).startswith("actions/checkout@")]
    if not checkout or any((s.get("with") or {}).get("fetch-depth") != 0 for s in checkout):
        out.append(f"{where}: every checkout step needs `with: fetch-depth: 0` (the diff rules need the merge-base)")
    if not any(f'"coverage=={COVERAGE_VERSION}"' in run(s) for s in steps):
        out.append(f"{where}: missing the exact coverage pin {COVERAGE_VERSION} in a step's run")
    if not isinstance(job, dict):
        return [f"{where}: the job must be a mapping"]
    if "defaults" in doc or "defaults" in job:
        out.append(f"{where}: `defaults` (workflow or job level) is banned — it can rewrite every step's shell")
    if "if" in job:
        out.append(f"{where}: a job-level `if:` is banned on the job that carries the coverage gate")
    if "continue-on-error" in job or any("continue-on-error" in s for s in steps):
        out.append(f"{where}: continue-on-error is banned in the job that carries the coverage gate")
    suite = [i for i, s in enumerate(steps) if "--cov" in run(s)]
    if len(suite) != 1 or CI_COV_ARGS not in flat(steps[suite[0]]) or CI_CLEAN not in flat(steps[suite[0]]):
        out.append(
            f"{where}: needs exactly one measuring step, running `{CI_CLEAN}` and the fast suite with `{CI_COV_ARGS}`"
        )
    gate = [i for i, s in enumerate(steps) if "lint_coverage.py" in run(s)]
    if len(gate) != 1 or run(steps[gate[0]]).strip() != CI_LINT_RUN[5:]:
        out.append(f"{where}: needs exactly one step whose run is `{CI_LINT_RUN[5:]}` (no `|| true`)")
        return out
    g = steps[gate[0]]
    if set(g) != {"name", "run"}:
        out.append(
            f"{where}: the coverage-ratchet step may carry only `name` and `run` (found {sorted(set(g) - {'name', 'run'})})"
        )
    if len(suite) == 1:
        if gate[0] < suite[0]:
            out.append(f"{where}: the coverage-ratchet step must come after the fast suite that measures")
        for s in steps[suite[0] + 1 : gate[0]]:
            if set(s) != {"name", "run"} or run(s).strip() != CI_BETWEEN_RUN:
                out.append(
                    f"{where}: step {s.get('name')!r} sits between the measuring suite and the coverage ratchet — "
                    f"only the MemoryHub step (`{CI_BETWEEN_RUN}`) may, so nothing can rewrite coverage.json"
                )
    return out


def config_rules(root: Path, ex_paths: list[str]) -> list[str]:
    out: list[str] = ci_step_rules(root)
    try:
        pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        return [f"pyproject.toml unreadable: {e}"]
    cov = pyproject.get("tool", {}).get("coverage")
    expected = json.loads(json.dumps(CANONICAL_CONFIG))
    expected["run"]["omit"] = FIXED_OMIT + list(ex_paths)
    if cov != expected:
        out.append(
            "[tool.coverage] differs from the canonical table in scripts/lint_coverage.py "
            f"(omit must be {FIXED_OMIT} + exactly the exclusion list's paths, in order): got {cov!r}"
        )
    for rel in _src_files(root):
        absolute = (root / rel).as_posix()
        for pat in FIXED_OMIT:
            if fnmatch.fnmatch(absolute, pat) or fnmatch.fnmatch(rel, pat):
                out.append(f"{rel}: matched by the fixed omit glob {pat!r} — it would be silently unmeasured")
    if (root / ".coveragerc").exists():
        out.append(".coveragerc present: coverage configuration lives only in pyproject.toml [tool.coverage]")
    for name in ("setup.cfg", "tox.ini"):
        p = root / name
        if p.is_file() and re.search(r"^\s*\[coverage:", p.read_text(errors="replace"), re.MULTILINE):
            out.append(f"{name} has a [coverage:...] section: coverage configuration lives only in pyproject.toml")
    addopts = pyproject.get("tool", {}).get("pytest", {}).get("ini_options", {}).get("addopts", "")
    sources = [("pyproject.toml addopts", addopts if isinstance(addopts, str) else " ".join(addopts))]
    for name in ("pytest.ini", "tox.ini", "setup.cfg", "tests/pytest.ini"):
        if (root / name).is_file():
            sources.append((name, (root / name).read_text(errors="replace")))
    wf = root / ".github" / "workflows"
    workflows = sorted(wf.glob("*.yml")) + sorted(wf.glob("*.yaml")) if wf.is_dir() else []
    for p in workflows:
        text = p.read_text(errors="replace")
        sources.append((p.relative_to(root).as_posix(), text))
        for m in sorted(set(_BANNED_ENV_RE.findall(text))):
            out.append(f"{p.relative_to(root).as_posix()}: names {m} — coverage is configured only in pyproject.toml")
    for name, text in sources:
        for m in sorted(set(_BANNED_OPTS_RE.findall(text))):
            out.append(f"{name}: {m} is banned (it moves or disables the measurement)")
    for top in ("tests", SCOPE):
        for p in sorted((root / top).rglob("*.py")):
            if _BANNED_IMPORT_RE.search(p.read_text(errors="replace")):
                out.append(
                    f"{p.relative_to(root).as_posix()}: imports or drives the coverage API (banned in tests/ and src/)"
                )
    return out


# ── diff coverage (S1-S3, S6) ─────────────────────────────────────────────────


def changed_lines(root: Path, base: str) -> dict[str, set[int]]:
    """{path: added/modified line numbers at the working tree}, rename detection OFF (S6)."""
    text = git(
        root,
        "-c",
        "diff.renames=false",
        "-c",
        "core.quotePath=false",
        "diff",
        "--no-renames",
        "--no-ext-diff",
        "--no-textconv",
        "--no-color",
        "--no-relative",
        "--src-prefix=a/",
        "--dst-prefix=b/",
        "-U0",
        base,
        "--",
        SCOPE,
    )
    out: dict[str, set[int]] = {}
    cur: str | None = None
    for ln in text.splitlines():
        if ln.startswith("+++ "):
            target = ln[4:].split("\t", 1)[0]  # git appends a TAB after a path that contains a space
            cur = target[2:] if target.startswith("b/") else None
            if cur is not None and cur.endswith(".py"):
                out.setdefault(cur, set())
            else:
                cur = None
            continue
        m = _HUNK_RE.match(ln)
        if m and cur is not None:
            start, n = int(m.group(1)), int(m.group(2) if m.group(2) is not None else 1)
            out[cur].update(range(start, start + n))
    return out


def _code_lines(text: str) -> set[int]:
    skip = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT}
    skip |= {tokenize.ENCODING, tokenize.ENDMARKER}
    lines: set[int] = set()
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type not in skip:
            lines.update(range(tok.start[0], tok.end[0] + 1))
    return lines


def _is_tc(node: ast.expr) -> bool:
    return (isinstance(node, ast.Name) and node.id == "TYPE_CHECKING") or (
        isinstance(node, ast.Attribute) and node.attr == "TYPE_CHECKING"
    )


@dataclass
class _Stmt:
    first: int
    end: int
    node: ast.stmt
    parent: ast.AST | None


def _statements(tree: ast.AST) -> list[_Stmt]:
    out: list[_Stmt] = []
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            if isinstance(child, ast.stmt):
                decos = getattr(child, "decorator_list", [])
                first = min([child.lineno] + [d.lineno for d in decos])
                out.append(_Stmt(first, child.end_lineno or child.lineno, child, parent))
    return out


def _clause_headers(stmts: list[_Stmt]) -> list[tuple[int, int, _Stmt]]:
    """(start, end, first statement of the block) for every clause header that is not a statement of its own:
    ``except ...:``, ``else:``, ``finally:`` and ``case ...:``. A changed header maps to its block's first
    statement, not to the enclosing ``try``/``if``/``match`` (which would read as covered when the block is not)."""
    by_node = {id(s.node): s for s in stmts}
    out: list[tuple[int, int, _Stmt]] = []

    def block(start: int, body: list[ast.stmt]) -> None:
        if body and id(body[0]) in by_node:
            first = by_node[id(body[0])]
            if start <= first.first - 1:
                out.append((start, first.first - 1, first))

    for s in stmts:
        n = s.node
        prev_end = n.body[-1].end_lineno if getattr(n, "body", None) else n.lineno
        if isinstance(n, (ast.Try, getattr(ast, "TryStar", ast.Try))):
            for h in n.handlers:
                block(h.lineno, h.body)
                prev_end = h.end_lineno or prev_end
            if n.orelse:
                block(prev_end + 1, n.orelse)
                prev_end = n.orelse[-1].end_lineno or prev_end
            if n.finalbody:
                block(prev_end + 1, n.finalbody)
        elif isinstance(n, (ast.If, ast.For, ast.AsyncFor, ast.While)) and n.orelse:
            block((prev_end or n.lineno) + 1, n.orelse)  # an `elif` is its own If: the range holds only blank lines
        elif isinstance(n, ast.Match):
            for c in n.cases:
                block(c.pattern.lineno, c.body)
    return out


def _imports_only_tc(node: ast.AST | None) -> bool:
    return (
        isinstance(node, ast.If)
        and _is_tc(node.test)
        and not node.orelse
        and all(isinstance(b, (ast.Import, ast.ImportFrom)) for b in node.body)
    )


def _typing_imports_only(st: _Stmt) -> bool:
    """The S3 exception: a plain ``if TYPE_CHECKING:`` (no else) whose body is imports only, or a statement in it."""
    return _imports_only_tc(st.node if isinstance(st.node, ast.If) else st.parent)


def diff_coverage(root: Path, base: str, cov: dict[str, FileCov], excluded: set[str]) -> tuple[list[str], list[str]]:
    """(failures, report lines)."""
    fails: list[str] = []
    report: list[str] = []
    total = covered = 0
    for rel, lines in sorted(changed_lines(root, base).items()):
        path = root / rel
        if not path.is_file() or rel in excluded or not lines:
            continue
        f = cov.get(rel)
        if f is None:
            fails.append(f"{rel}: changed but absent from {COVERAGE_JSON} and not on the exclusion list (S2)")
            continue
        text = path.read_text(encoding="utf-8")
        try:
            stmts = _statements(ast.parse(text))
            headers = _clause_headers(stmts)
            code = _code_lines(text)
        except (SyntaxError, tokenize.TokenError) as e:
            fails.append(f"{rel}: cannot parse ({e})")
            continue
        seen: dict[int, bool] = {}
        for ln in sorted(lines & code):
            enclosing = [s for s in stmts if s.first <= ln <= s.end]
            if not enclosing:
                continue
            st = max(enclosing, key=lambda s: (s.first, -s.end))
            clause = [h for h in headers if h[0] <= ln <= h[1] and h[0] >= st.first]
            if clause:
                st = max(clause, key=lambda h: h[0])[2]
            keys = (st.node.lineno, st.first)
            if st.first in seen:
                continue
            if any(k in f.executed for k in keys):
                seen[st.first] = True
            elif any(k in f.missing for k in keys):
                seen[st.first] = False
            elif any(k in f.excluded for k in keys):
                if not _typing_imports_only(st):
                    seen[st.first] = False  # S3: excluded code in a diff is uncovered code
        if not seen:
            continue
        n, c = len(seen), sum(seen.values())
        total += n
        covered += c
        src = text.splitlines()
        report.append(f"  {rel}: {c}/{n} changed statements covered ({100.0 * c / n:.1f}%)")
        for ln in sorted(k for k, v in seen.items() if not v):
            report.append(f"    UNCOVERED {rel}:{ln}: {src[ln - 1].strip()[:100]}")
    if total:
        pct = 100.0 * covered / total
        report.insert(0, f"coverage: diff coverage {covered}/{total} changed executable statements = {pct:.1f}%")
        if pct + 1e-9 < DIFF_THRESHOLD:
            fails.append(
                f"diff coverage {pct:.1f}% is below {DIFF_THRESHOLD:.0f}% ({covered}/{total}) — the uncovered "
                "changed lines are listed above"
            )
    else:
        report.insert(0, "coverage: diff coverage — no executable src/maxim change")
    return fails, report


# ── main ─────────────────────────────────────────────────────────────────────


def _print_totals(overall: ScopeCov, pkgs: dict[str, ScopeCov], fl: Floors | None) -> None:
    def line(name: str, m: ScopeCov) -> str:
        f = (fl.overall if name == "overall" else fl.packages.get(name)) if fl else None
        pin = "" if f is None else f" floor {f.percent}% (missing {f.missing})"
        br = 100.0 * m.covered_branches / m.branches if m.branches else 100.0
        return (
            f"  {name}: {_fmt_pct(m.percent)} of {m.statements} statements, missing {m.missing}; "
            f"branches {br:.2f}% of {m.branches}{pin}"
        )

    print("coverage: statement coverage (gated) and branch coverage (printed only):")
    print(line("overall", overall))
    for k in sorted(pkgs):
        print(line(k, pkgs[k]))


def _read(root: Path, rel: str, parse):
    try:
        return parse((root / rel).read_text(encoding="utf-8")), None
    except OSError as e:
        return None, f"{rel}: unreadable ({e})"
    except FileFormatError as e:
        return None, f"{rel}: {e}"


def merge_floors(texts: list[str]) -> dict:
    """The per-scope MINIMUM of several measured floors files (the paste-ready JSON a run prints): lowest
    percent AND lowest missing (``missing`` is one-sided, so the lower pin passes against every run).
    A scope in only one file keeps that file's value."""
    docs = [json.loads(t) for t in texts]

    def merge(vals: list[dict]) -> dict:
        vals = [v for v in vals if v and v.get("percent") is not None]
        if not vals:
            return {"percent": None, "missing": None}
        return {"percent": min(v["percent"] for v in vals), "missing": min(v["missing"] for v in vals)}

    keys = sorted({k for d in docs for k in d["packages"]})
    return {
        "floors_format_version": FLOORS_FORMAT_VERSION,
        "band": BAND,
        "min_package_statements": MIN_PACKAGE_STATEMENTS,
        "overall": merge([d["overall"] for d in docs]),
        "packages": {k: merge([d["packages"].get(k) for d in docs]) for k in keys},
    }


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--merge-floors"]:
        if len(argv) < 3:
            print("usage: lint_coverage.py --merge-floors RUN1.json RUN2.json [...]", file=sys.stderr)
            return 2
        print(json.dumps(merge_floors([Path(a).read_text(encoding="utf-8") for a in argv[1:]]), indent=2))
        return 0
    started = time.monotonic()
    root = REPO_ROOT
    event = os.environ.get("GITHUB_EVENT_NAME")
    failures: list[str] = []
    fl, err = _read(root, FLOORS_REL, parse_floors)
    failures += [err] if err else []
    ex, err = _read(root, EXCLUSIONS_REL, parse_exclusions)
    failures += [err] if err else []
    if ex is not None:
        failures += config_rules(root, ex.paths)
        failures += exclusion_head_rules(root, ex)
    try:
        cov = load_coverage(root)
    except MeasurementError as e:
        sys.stdout.flush()  # stdout and stderr interleave in CI logs: keep the printed report readable
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        print(f"coverage: ERROR — failing closed: {e}", file=sys.stderr)
        return 2
    overall, pkgs = measure_scopes(cov)
    _print_totals(overall, pkgs, fl)
    if fl is not None:
        failures += floor_head_rules(root, fl, overall, pkgs)
        if fl.overall.percent is None or any(f.percent is None for f in fl.packages.values()):
            print(f"coverage: measured floors for {FLOORS_REL} (rounded down to 0.1; review before committing):")
            print(json.dumps(suggested_floors(fl, overall, pkgs), indent=2))

    if event in (None, "", "pull_request", "push"):  # push: against the last green push (_lint_git.push_base, #1089)
        try:
            base = base_ref(root)
        except GitUnavailable as e:
            if must_not_skip(str(e)):
                return 2
            print(f"INFO: no merge-base available; diff-scoped rules skipped ({e})")
        else:
            try:
                if fl is not None:
                    base_text = show(root, base, FLOORS_REL)
                    if base_text:
                        try:
                            base_fl = parse_floors(base_text)
                        except FileFormatError as e:
                            failures.append(f"{FLOORS_REL} at the merge-base: {e}")
                        else:
                            failures += floor_diff_rules(root, base_fl, fl, overall, pkgs, package_renames(root, base))
                    else:
                        print(f"coverage: {FLOORS_REL} absent at the merge-base — bootstrap, every floor is new")
                        empty = Floors(fl.band, fl.min_statements, fl.version, Floor(None, None))
                        failures += floor_diff_rules(root, empty, fl, overall, pkgs, {})
                if ex is not None:
                    base_text = show(root, base, EXCLUSIONS_REL)
                    try:
                        base_ex = parse_exclusions(base_text) if base_text else None
                    except FileFormatError as e:
                        failures.append(f"{EXCLUSIONS_REL} at the merge-base: {e}")
                    else:
                        failures += exclusion_diff_rules(root, base, ex, base_ex)
                    dfails, report = diff_coverage(root, base, cov, set(ex.paths))
                    print("\n".join(report))
                    failures += dfails
            except GitUnavailable as e:
                print(f"ERROR: git could not answer during the diff-scoped rules ({e})", file=sys.stderr)
                return 2
    else:
        print(f"coverage: {event} event — head rules only (floors, configuration, ledger)")

    print(f"coverage: {time.monotonic() - started:.1f}s")
    if failures:
        sys.stdout.flush()  # the paste-ready floors JSON (stdout) must not split around this list (stderr)
        print("coverage ratchet FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("coverage ratchet: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
