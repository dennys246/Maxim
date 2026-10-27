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
  rules), so none needed the tolerance (the "raise it to" HINT prints ``min(measured, pinned)``
  so that following it passes the pin rule, #1409; it gates nothing);
  ``band`` must be 1.0 and ``min_package_statements`` (N) 200; a package with >= N
  statements needs a floor; a ``null`` floor fails closed printing
  ``floor missing: measured X`` (the bootstrap: the file is committed with nulls, CI prints
  the values, they are committed on the same branch); a floor whose package directory is
  absent is an orphan. **Hysteresis:** a floored package keeps its floor while its
  directory exists, even under N.
- pull requests (against the merge-base copy, ``git show``): ``band``/N/format may not
  change (format: except v1 -> v2, section 1b); a new or changed floor is verified against the measurement by the rules under
  "Every new or changed floor is verified data" below (lowered only to the measurement, only
  when uncovered statements did not grow; the ``missing`` pin never rises); a floor may be
  removed only when its directory is gone at HEAD. Package-level renames
  (``git diff -M --name-status``) carry a removed floor to its target, which then follows the
  CHANGED rules against it; with no rename detected, every NEW floor in a PR that removes a
  floored package is held to each removed one the same way.

**1b. Per-file floors** (owner decision 2026-10-13; format v2). ``files`` maps a normalized repo-relative POSIX
path (``src/maxim/....py``: no ``./``, ``//``, ``..``, ``\\`` or leading ``/`` — a format error otherwise) to the
same ``{percent, missing}`` floor. File floors are OPT-IN (no size threshold) and are enforced beside the package
floor, independently, every run: a PR may have to edit both, and neither substitutes for the other (a package's
~100-statement band can otherwise pay for a file's deleted tests). The file's measured scope is its own
``coverage.json`` entry, and every rule of section 1 applies unchanged, k and ``K_MIN`` included:

- every run: the k tolerance below, the band above, ``null`` fails closed printing the pin; the key must be in the
  case-exact ``src/maxim`` directory listing (never ``Path.is_file()``, which answers True for a case variant on
  APFS) and must not be or pass through a symlink; a floored file that is excluded (so unmeasured) fails; a key whose
  file is gone is an orphan. Under ``--local`` no floor gates (a note, M60a);
- pull requests: NEW/CHANGED/LOWERING exactly as for packages; set back to ``null`` fails; a file floor may be
  removed only when its file is gone (case-exact listing); a ``-M`` rename carries the floor to its target (CHANGED
  rules); with no rename detected every new file floor is a candidate target, and the sum rule applies. That holds
  for a retirement PR too (N5): a ``retired_files`` entry stops the "gone" failure, but the removed floor is still
  carried to every NEW file floor in the same diff — stricter than the design needs, fail-closed; land new file
  floors in a separate PR from a retirement;
- **SF-1, a floored file does not vanish:** a floored file gone at HEAD with no detected rename FAILS, unless the
  same diff appends a ``retired_files`` entry ``{path, ref, reason}`` for it. ``retired_files`` is append-only (the
  base list is an exact prefix, like the ledger); every entry needs a ``ref`` and a reason; a new entry is printed
  loudly and must name a file that had a floor at the base and is gone at HEAD (file and floor);
- **SF-2, the split rule** (``SPLIT_STATEMENTS`` = S = 25; owner decisions 2026-10-13). TRIGGER: S or more of a
  floored file's BASE statements are EDITED OR DELETED (a modified line counts as removed) — not the net count, so padding the file back with trivial statements
  hides nothing. A base statement is removed when a deleted line of the diff (``git diff -M -U0`` base..HEAD, ``-``
  hunks in base line numbers, against the ``-M`` rename target for a rename) maps onto it by the diff-coverage
  mapping (innermost enclosing ``ast.stmt``; a clause header to its block's first statement), or when its line
  survives but no longer starts a statement of the same type (the code was wrapped in a string literal); distinct
  statements are counted. A gone file (or one whose path is now a symlink, N4) removes all of its base statements.
  SUCCESSORS: every non-excluded ``src/maxim`` .py file that is ADDED in the diff (absent from the base tree — which
  includes the ``-M`` rename target of an UNRELATED file), or EXISTING (present at the base) and grown by S or more
  AST statements against the base. Each successor of S or more statements needs its own file floor in this PR: a
  floor it did not have at the base is NEW (NEW rules), one it had follows the CHANGED rules; a smaller ADDED
  successor only joins the sum. SUM: the source's measured ``missing`` plus every successor's measured ``missing``
  may not exceed the source's base ``missing`` pin plus the base pin of every successor floored at the base, with no
  tolerance (the package sum rule's code). This is STRICT by design: an existing successor with no floor at the base
  brings no pin, so its WHOLE measured ``missing`` — old uncovered code included — is charged to the source's pin.
  The remedy is to move code into NEW modules (or floor the destination in an earlier PR). It closes the shim trick
  (move everything to a new module, leave a re-export shim carrying the floor), the pre-created target (PR1 adds a
  one-line module, PR2 moves the code into it), padding (move code out, add as many trivial statements back) and
  laundering test erosion through a move. Its costs, accepted: an unrelated module added (or an unrelated file grown
  by S) in the same PR as a slice joins the group, so land it separately. With NO successor the sum is skipped
  (owner decision 2026-10-13): the code went nowhere that grew by S, so an in-place rewrite of S or more statements
  is held by the source's own HEAD floor rule (k tolerance) and the CHANGED/LOWERING rules, not by a no-tolerance
  sum that thread noise would false-fail; code spread over several small files is the thin-slicing residual;
- format: ``floors_format_version`` is 2 and HEAD must be 2. The one version change a diff may make is base v1 ->
  HEAD v2, the v1 base read as ``files = {}`` and ``retired_files = []`` (every file floor is then NEW); every other
  change (2 -> 1, 2 -> 3) fails.

**2. Diff coverage** (pull requests). Changed lines come from
``git -c diff.renames=false diff --no-renames --no-ext-diff --no-textconv -U0 <merge-base>
-- src/maxim`` against the working tree, so MOVED code counts as changed (the owner's
rule: moved code arrives tested). Hunk bodies are consumed by their header counts (:func:`parse_unified_diff`, shared
with the split rule; hardened 2026-10-13: an added column-0 ``++x`` line once read as a ``+++`` header and silently
dropped the file's later hunks from this existing rule). Each changed line that carries code maps to the start
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
- a CHANGED floor: ``h.missing <= min(m.missing + K_MIN, b.missing)`` — the missing pin NEVER rises above its base;
  ``K_MIN`` (5 statements, the absolute noise, never the size-scaled k) lets a pin taken from one CI run pass on the
  pin commit's own run (M60a, owner decision 2026-10-27). A file floor and a rename target take the same rule.
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
covered statements per scope (every floor is NEW on the gate PR). **A NEW floor — the bootstrap, a new package, a
new file floor — is pinned from 2 or more CI runs** (commit it ``null``, take the printed pins from two runs, merge);
a CHANGED or LOWERED floor is verified against the one run of its own PR, which is what the rules above print.
``--merge-floors`` merges and prints ONLY the scopes that are ``null`` in (or absent from) the committed floors file
(SF-A4: a spread over k in an unrelated scope must not block a new pin; ``--all-scopes`` merges every scope, the
bootstrap shape), and the output holds only those entries: copy only the new entry into the committed file. The
printed floors block appears in the CI log only while some floor is ``null``; for a second run, re-run the failed
job. The input is that printed block saved as JSON — the uploaded ``coverage.json`` artifact is NOT an accepted
input. It enforces the spread of the runs it is given (SF-3, owner decision 2026-10-13): it exits 1 and prints no
floors when a merged scope is measured in fewer than two of the runs, when the runs' ``missing`` differ by more than
the scope's k, when the runs carry different ``retired_files``, or when two inputs are byte-identical (the same
file given twice). Every entry is validated as a floor (``{percent, missing}``, both or neither null) and a bad one
is a format error (exit 2). The gate cannot see where a pin came from: whether the inputs are two different CI
runs, or runs of this commit, is NOT detected (two copies differing by one byte pass) — review reads the pin's
provenance (mechanization backlog M52). A flaky scope is fixed (the non-deterministic test), never absorbed into a
floor. The printed pins carry no statement count, so k is computed from a LOWER bound on it (``100·missing / (100 −
percent)``, the percent being rounded down): the check can only be stricter than the HEAD tolerance. Duplicate JSON
keys are refused in the floors file, the exclusions file and the ``--merge-floors`` inputs (``coverage.json`` is
read with plain ``json.loads``: the coverage tool writes it). Floor diff rules
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

**"The merge-base" below means the base:** on a pull request the merge-base with ``main``; on a push to main the
last push whose ``unit-tests`` job passed (``_lint_git.push_base``, #1089).

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
- per-file floors, thin slicing (owner decision 2026-10-13, documented, not mechanized): the split rule fires per
  PR, so a floored file's code can leave in steps of S − 1 = 24 removed statements per PR, and code moved within one
  PR into several EXISTING files that each grow by fewer than S statements makes none of them a successor. Each step
  is bounded: the moved code is changed code, so it costs 80% diff coverage, and the destination's package floor
  holds it (its band, ~100 statements of a large package); the source's own floor still re-pins honestly.
  Cumulative-since-pin removal tracking would close it (mechanization backlog M51);
- a no-tolerance lowering can be refused by run-to-run noise: when covered code leaves a floored scope (a file
  under decomposition most of all), ``m.missing`` may measure 1–2 statements over the base pin by noise alone and
  the lowering is refused ("cover N statement(s)"). The remedy is a re-run of the job, or covering those
  statements — never a tolerance;
- a file floor on a small file: k = ``K_MIN`` = 5 statements is loose there (10 pt of a 50-statement file), bounded
  and non-cumulative because pins never move with it;
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
import posixpath
import re
import subprocess
import sys
import time
import tokenize
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import yaml  # noqa: E402  (pyyaml: pinned in the unit-tests job's first install, not the guarded one)
from _lint_git import GitUnavailable, base_ref, changed_files, gh_api, git, must_not_skip, show  # noqa: E402
from _lint_allowance import append_only_problem, ref_ok  # noqa: E402

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
FLOORS_FORMAT_VERSION = 2  # v2 (2026-10-13): per-file floors (`files`) + `retired_files`; a v1 base reads files={}
# The split rule (owner decision 2026-10-13, SF-2): a floored file whose AST statement count falls by at least this
# many against the base hands its code to every file the diff ADDS; a successor this size needs its own file floor.
SPLIT_STATEMENTS = 25
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

_BANNED_OPTS_RE = re.compile(r"--cov-config\b|--no-cov(?![\w-])|--cov-fail-under\b|--cov-append\b|no:(?:pytest_)?cov\b")
_BANNED_ENV_RE = re.compile(r"\b(?:COVERAGE|COV_CORE)_[A-Z_]+")
_BANNED_IMPORT_RE = re.compile(
    r"^\s*(?:import|from)\s+(?:coverage|pytest_cov)\b|\bCoverage\.current\b|^\s*import\s+[\w.]+\s*,\s*coverage\b",
    re.MULTILINE,
)
_HUNK_FULL_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


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


def load_coverage(root: Path, path: Path | None = None) -> dict[str, FileCov]:
    """The measurement at ``root``'s ``coverage.json``, or at ``path`` (``--pins-from-run``: a CI runner's file, whose
    absolute keys are re-rooted at ``src/maxim/``; every file must still exist under ``root``)."""
    foreign = path is not None
    path = root / COVERAGE_JSON if path is None else path
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
        if foreign and f"/{SCOPE}/" in rel:
            rel = rel[rel.index(f"/{SCOPE}/") + 1 :]  # /home/runner/work/Maxim/Maxim/src/maxim/x.py -> src/maxim/x.py
        if not rel.startswith(SCOPE + "/"):
            raise MeasurementError(f"{COVERAGE_JSON} names a file outside {SCOPE}: {key}")
        # A CI run of a PR measures the merge with main, which may hold files this branch lacks; its pins are for
        # that merge, and CI re-measures the merge on the pin commit, so a foreign file need not exist here.
        if not foreign and not (root / rel).is_file():
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
    files: dict[str, Floor] = field(default_factory=dict)  # repo-relative POSIX path -> floor (v2)
    retired: list[dict] = field(default_factory=list)  # `retired_files`: append-only {path, ref, reason} (v2)


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict:
    """``object_pairs_hook`` for the gate files: a duplicate key is an error. ``json.loads`` keeps the LAST value,
    so a reviewer reading the first (high) one would approve a file the gate reads low."""
    out: dict = {}
    for k, v in pairs:
        if k in out:
            raise FileFormatError(f"duplicate JSON key {k!r} (the parser would keep only the last value)")
        out[k] = v
    return out


def _loads(text: str) -> object:
    try:
        return json.loads(text, object_pairs_hook=_strict_pairs)
    except json.JSONDecodeError as e:
        raise FileFormatError(f"not JSON: {e}") from e


def file_key_problem(k: object) -> str | None:
    """A ``files`` / ``retired_files`` key must already be the normalized repo-relative POSIX path of a
    ``src/maxim`` ``.py`` file: one file, one spelling (``./``, ``//``, ``..``, ``\\`` or a leading ``/`` would let
    a second key name the same file)."""
    if not isinstance(k, str):
        return f"file key must be a string: {k!r}"
    if (
        "\\" in k
        or k.startswith("/")
        or ".." in k.split("/")
        or posixpath.normpath(k) != k
        or not k.startswith(SCOPE + "/")
        or not k.endswith(".py")
    ):
        return f"file key {k!r} must be a normalized repo-relative path {SCOPE}/....py (no ./, //, .., \\ or leading /)"
    return None


def _parse_retired(v: object) -> list[dict]:
    if not isinstance(v, list):
        raise FileFormatError("retired_files must be a list")
    seen: set[str] = set()
    for x in v:
        if not isinstance(x, dict) or set(x) != {"path", "ref", "reason"}:
            raise FileFormatError(f"retired_files entry must be exactly {{path, ref, reason}}: {x!r}")
        if problem := file_key_problem(x["path"]):
            raise FileFormatError(f"retired_files: {problem}")
        if x["path"] in seen:
            raise FileFormatError(f"retired_files: {x['path']} retired twice")
        seen.add(x["path"])
        if not ref_ok(x["ref"]):
            raise FileFormatError(f"retired_files entry needs a ref ('#NNN' or a github.com PR/issue URL): {x!r}")
        if not (isinstance(x["reason"], str) and x["reason"].strip()):
            raise FileFormatError(f"retired_files entry needs a non-empty reason: {x!r}")
    return list(v)


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


_V1_KEYS = frozenset({"floors_format_version", "band", "min_package_statements", "overall", "packages"})
_V2_KEYS = _V1_KEYS | {"files", "retired_files"}


def parse_floors(text: str) -> Floors:
    """v2 (``files`` + ``retired_files``) or v1 (no per-file floors: read as ``files = {}``, one way only — HEAD
    must be v2, :func:`floor_head_rules`; the only version change a diff may make is base v1 -> HEAD v2)."""
    d = _loads(text)
    if not isinstance(d, dict):
        raise FileFormatError("the floors file must be a JSON object")
    keys = _V1_KEYS if d.get("floors_format_version") == 1 else _V2_KEYS
    if set(d) != keys:
        raise FileFormatError(f"top-level keys must be exactly {sorted(keys)}")
    if not isinstance(d["packages"], dict):
        raise FileFormatError("packages must be an object")
    files: dict[str, Floor] = {}
    if keys == _V2_KEYS:
        if not isinstance(d["files"], dict):
            raise FileFormatError("files must be an object")
        for k, v in d["files"].items():
            if problem := file_key_problem(k):
                raise FileFormatError(problem)
            files[k] = _parse_floor(k, v)
    return Floors(
        band=d["band"],
        min_statements=d["min_package_statements"],
        version=d["floors_format_version"],
        overall=_parse_floor("overall", d["overall"]),
        packages={k: _parse_floor(k, v) for k, v in d["packages"].items()},
        files=files,
        retired=_parse_retired(d["retired_files"]) if keys == _V2_KEYS else [],
    )


def _fmt_pct(x: float) -> str:
    return f"{x:.2f}%"


def file_scopes(cov: dict[str, FileCov]) -> dict[str, ScopeCov]:
    """Each measured file folded into a :class:`ScopeCov`, keyed by its case-exact ``coverage.json`` path."""
    out: dict[str, ScopeCov] = {}
    for rel, f in cov.items():
        out[rel] = ScopeCov()
        out[rel].add(f)
    return out


def _symlinked(root: Path, rel: str) -> bool:
    """True when ``rel`` or any directory on its way down from the repo root is a symlink."""
    p = root
    for part in rel.split("/"):
        p = p / part
        if p.is_symlink():
            return True
    return False


def file_present(root: Path, rel: str, listing: set[str]) -> bool:
    """Case-exact existence (NIT-1): ``rel`` is in the ``src/maxim`` directory listing (``set(_src_files(root))``)
    and is no symlink. ``Path.is_file()`` would answer True for a case variant on a case-insensitive filesystem."""
    return rel in listing and not _symlinked(root, rel)


def floor_head_rules(
    root: Path,
    fl: Floors,
    overall: ScopeCov,
    pkgs: dict[str, ScopeCov],
    only: set[str] | None = None,
    *,
    files: dict[str, ScopeCov],
) -> list[str]:
    """The HEAD floor rules. ``only`` restricts the per-scope checks to those packages (a file floor gates when its
    package is in it) and skips ``overall``; the file-format checks always run. ``--local`` passes the empty set, so
    only the format checks gate there (M60a); CI never passes it.
    ``files`` is :func:`file_scopes` of the measurement (required: a file floor must never be skipped by omission)."""
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
            # The pin this hint prints must pass floor_diff_rules' "a pinned missing never rises" (#1409): a package
            # that rose past the band while adding statements keeps its current missing pin, never the measured one.
            pin_missing = m.missing if f.missing is None else min(m.missing, f.missing)
            out.append(
                f"{name}: {_fmt_pct(m.percent)} is more than {BAND} pt above its floor {f.percent}% — raise it to "
                f'{{"percent": {round_down(m.percent)}, "missing": {pin_missing}}}'
            )

    if only is None:
        check("overall", fl.overall, overall)
    for pkg, f in sorted(fl.packages.items()):
        if only is not None and pkg not in only:
            continue
        if not package_present(root, pkg):
            out.append(f"{pkg}: floored package has no directory at HEAD (orphan) — remove the entry")
        elif pkg not in pkgs or pkgs[pkg].statements == 0:
            out.append(f"{pkg}: floored package has no measured statements (all excluded?)")
        else:
            check(pkg, f, pkgs[pkg])
    for pkg, m in sorted(pkgs.items()):
        if only is not None and pkg not in only:
            continue
        if pkg not in fl.packages and m.statements >= MIN_PACKAGE_STATEMENTS:
            out.append(
                f"{pkg}: {m.statements} statements (>= {MIN_PACKAGE_STATEMENTS}) and no floor — add "
                f'"{pkg}": {{"percent": {round_down(m.percent)}, "missing": {m.missing}}}'
            )
    # 1b. Per-file floors: the same check, k and K_MIN unchanged, on the file's own measurement.
    listing = set(_src_files(root)) if fl.files else set()
    for rel, f in sorted(fl.files.items()):
        if only is not None and package_of(rel) not in only:
            continue
        if _symlinked(root, rel):
            out.append(f"{rel}: a file floor's key may not be (or pass through) a symlink — key the real file")
        elif rel not in listing:
            out.append(f"{rel}: floored file not found at HEAD (case-exact) — orphan: remove or carry the entry")
        elif rel not in files or files[rel].statements == 0:
            out.append(f"{rel}: floored file has no measured statements (excluded? a floored file must be measured)")
        else:
            check(rel, f, files[rel])
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


Hunk = tuple[int, int, int, int]  # (old_start, old_count, new_start, new_count) of one ``-U0`` hunk


@dataclass
class FileDiff:
    """What the per-file diff rules read from git, gathered once by :func:`file_diff`."""

    renames: dict[str, str]  # base path -> HEAD path (``git diff -M --name-status`` base..HEAD)
    added: set[str]  # ``src/maxim`` .py files in the HEAD listing that the base tree does not have
    base_text: Callable[[str], str]  # a file's content at the base ('' when it did not exist there)
    # base path -> its ``-U0`` hunks (old_start, old_count, new_start, new_count), base..HEAD with ``-M``, so a
    # renamed file's hunks are against its rename target; a file absent here is unchanged
    hunks: dict[str, list[Hunk]] = field(default_factory=dict)
    modified: set[str] = field(default_factory=set)  # changed in place: present at the base and at HEAD, same path


@dataclass
class DiffFile:
    old: str | None  # the base path (``--- a/<path>``); None for an added file
    new: str | None  # the HEAD / working-tree path (``+++ b/<path>``); None for a deleted file
    hunks: list[Hunk] = field(default_factory=list)


def _diff_path(header: str) -> str | None:
    target = header[4:].split("\t", 1)[0]  # git appends a TAB after a path that contains a space
    return target[2:] if target[:2] in ("a/", "b/") else None  # /dev/null: absent on that side


def parse_unified_diff(text: str) -> list[DiffFile]:
    """The ONE parser of ``git diff`` output for this lint (diff coverage and the split rule). Each hunk BODY is
    consumed by the counts in its ``@@ -a,b +c,d @@`` header (a missing count is 1; ``\\ No newline at end of file``
    counts for neither side), and ``diff --git`` / ``---`` / ``+++`` are recognised only in a file's header, between
    ``diff --git`` and its first ``@@``. A deleted source line ``-- 1`` prints as ``--- 1`` and an added ``++x`` as
    ``+++ x``: read as headers they silently dropped every later hunk of the file (the delta review of 2026-10-13;
    the added-line half hardened diff coverage, an existing rule). Output it cannot account for raises
    ``GitUnavailable`` (exit 2), never a partial answer. Lines split on ``\\n`` ONLY, and both callers read git's
    output as bytes (``git(..., raw=True)``): ``splitlines()`` and text-mode newline translation would split one diff
    line at a form feed, a lone CR, ``\\x85`` or U+2028 and turn legal source into a refused diff on every PR."""
    out: list[DiffFile] = []
    cur: DiffFile | None = None
    in_header = False
    # "\n" only: str.splitlines() also splits on \x0c, \x0b, \x1c-\x1e, \x85, U+2028/2029 (legal inside a source
    # line), which would turn one diff line into two and make the count-based hunk read refuse the output
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    i = 0
    while i < len(lines):
        ln = lines[i]
        i += 1
        if ln.startswith("diff --git "):
            cur = DiffFile(None, None)
            out.append(cur)
            in_header = True
        elif cur is not None and in_header and ln.startswith("--- "):
            cur.old = _diff_path(ln)
        elif cur is not None and in_header and ln.startswith("+++ "):
            cur.new = _diff_path(ln)
        elif cur is not None and (m := _HUNK_FULL_RE.match(ln)):
            in_header = False
            h = (int(m[1]), int(m[2] if m[2] is not None else 1), int(m[3]), int(m[4] if m[4] is not None else 1))
            cur.hunks.append(h)
            old_left, new_left = h[1], h[3]
            while old_left or new_left:
                if i >= len(lines):
                    raise GitUnavailable(f"git diff output ends inside a hunk of {cur.new or cur.old}")
                body = lines[i]
                i += 1
                if body.startswith("\\"):
                    continue
                if body[:1] == "-" and old_left:
                    old_left -= 1
                elif body[:1] == "+" and new_left:
                    new_left -= 1
                elif body[:1] == " " and old_left and new_left:
                    old_left, new_left = old_left - 1, new_left - 1
                else:
                    raise GitUnavailable(
                        f"git diff hunk body does not match its header in {cur.new or cur.old}: {body!r}"
                    )
        elif in_header or ln.startswith("\\") or cur is None:
            continue  # header lines (index, mode, similarity, rename, Binary) and a trailing no-newline marker
        else:
            raise GitUnavailable(f"git diff line outside any hunk or header: {ln!r}")
    return out


def _diff_hunks(root: Path, base: str) -> tuple[dict[str, list[Hunk]], set[str]]:
    """({base path: its -U0 hunks}, {paths changed in place}) from ONE ``git diff -M -U0`` base..HEAD — the rename
    pairing of :func:`changed_files` (same command, same detection), so a renamed file diffs against its target."""
    text = git(
        root,
        "-c",
        "core.quotePath=false",
        "diff",
        "-M",
        "--no-ext-diff",
        "--no-textconv",
        "--no-color",
        "--no-relative",
        "--src-prefix=a/",
        "--dst-prefix=b/",
        "-U0",
        base,
        "HEAD",
        "--",
        SCOPE,
        raw=True,  # bytes: no newline translation of a lone CR (parse_unified_diff)
    )
    hunks: dict[str, list[Hunk]] = {}
    modified: set[str] = set()
    for f in parse_unified_diff(text):
        if f.old is None:
            continue
        hunks.setdefault(f.old, []).extend(f.hunks)
        if f.new == f.old:
            modified.add(f.old)
    return hunks, modified


def file_diff(root: Path, base: str) -> FileDiff:
    renames = {old: new for new, old in changed_files(root, base, SCOPE, unquoted=True) if old and old != new}
    at_base = set(git(root, "ls-tree", "-r", "-z", "--name-only", base, SCOPE).split("\0")) - {""}
    added = {rel for rel in _src_files(root) if rel not in at_base}
    hunks, modified = _diff_hunks(root, base)
    return FileDiff(renames, added, lambda rel: show(root, base, rel), hunks, modified)


#: Used when neither the base nor HEAD has a file floor: no file rule consults git then, so no git call is made (and a
#: run without file floors gains no exit-2 path). Every file rule iterates a floors map, so an empty map reads nothing.
NO_FILE_DIFF = FileDiff({}, set(), lambda rel: "")


def removed_statements(base_src: str, hunks: list[Hunk], head_src: str) -> int:
    """How many of the base file's statements a diff EDITS OR DELETES (SF-2's trigger; a modified line counts): the distinct base statements a
    deleted line maps onto (the diff-coverage mapping, :func:`_line_statements`), plus every untouched base statement
    whose first line no longer starts a statement of the same type at HEAD (code wrapped in a string literal: no line
    deleted, the statements gone). Raises ``SyntaxError``/``tokenize.TokenError`` on an unparseable side."""
    deleted: set[int] = set()
    for os_, oc, _, _ in hunks:
        deleted.update(range(os_, os_ + oc))

    def key(n: ast.stmt) -> tuple[int, int]:
        return n.lineno, n.col_offset  # a position, not id(): the mapping and the walk below parse separate trees

    gone = {key(st.node) for _, st in _line_statements(base_src, deleted)}
    head = {(n.lineno, type(n)) for n in ast.walk(ast.parse(head_src)) if isinstance(n, ast.stmt)}
    for n in ast.walk(ast.parse(base_src)):
        if not isinstance(n, ast.stmt) or key(n) in gone:
            continue
        if n.lineno in deleted:  # a deleted line counts only through the mapping above; it has no HEAD position
            continue
        # The unchanged line's HEAD number: shifted by every hunk that ends before it (an insertion, oc = 0, sits
        # AFTER its old_start line).
        shift = sum(nc - oc for os_, oc, _, nc in hunks if n.lineno > (os_ + oc - 1 if oc else os_))
        if (n.lineno + shift, type(n)) not in head:
            gone.add(key(n))
    return len(gone)


def _sum_rule(
    out: list[str],
    edges: dict[str, set[str]],
    budget: Callable[[set[str], set[str]], int | None],
    measured: Callable[[str], int],
    why: str,
) -> None:
    """The SUM rule, shared by package splits and the per-file split rule: sources that share a target are one
    group, and the group's targets' measured ``missing`` summed may not exceed ``budget(sources, targets)`` (no
    tolerance). Each piece alone can sit under its pin while together they lose uncovered statements."""
    groups: list[tuple[set[str], set[str]]] = []
    for src in edges:
        srcs, tgts = {src}, set(edges[src])
        for g in [g for g in groups if g[1] & tgts]:
            groups.remove(g)
            srcs |= g[0]
            tgts |= g[1]
        groups.append((srcs, tgts))
    for srcs, tgts in groups:
        allowed = budget(srcs, tgts) if tgts else None
        if allowed is None:
            continue
        total = sum(measured(t) for t in tgts)
        if total > allowed:
            out.append(
                f"{' + '.join(sorted(tgts))}: received {', '.join(sorted(srcs))}; together they measure missing "
                f"{total} > the carried pinned missing {allowed} — cover {total - allowed} statement(s) ({why})"
            )


def pin_rises(base_fl: Floors, fl: Floors, measured: Callable[[str], ScopeCov | None] = lambda _key: None) -> list[str]:
    """The measurement-free half of ``floor_diff_rules``' "a pinned missing never rises": every floor present at both
    base and head under the same key (overall, a package, a file) whose head ``missing`` exceeds its base ``missing``.
    It compares the two floor files only, so ``--local`` gates it exactly as CI does (#1270: #1268 raised
    maxim/console 173 -> 207 and its local run printed that as a note). Renames and splits stay CI's: their pins carry
    across keys and are verified against measurements. ``measured`` (a key's measurement, when the run has one) only
    sharpens the printed pin to ``min(measured, base)``, the cap ``changed_entry`` enforces; it never changes what
    fails."""
    pairs = [("overall", base_fl.overall, fl.overall)]
    pairs += [(k, base_fl.packages.get(k), h) for k, h in fl.packages.items()]
    pairs += [(k, base_fl.files.get(k), h) for k, h in fl.files.items()]
    out = []
    for name, b, h in pairs:
        if b is None or b.missing is None or h.missing is None or h.missing <= b.missing:
            continue
        m = measured(name)
        cap = b.missing if m is None else min(m.missing, b.missing)
        out.append(
            f"{name}: floor pins missing {h.missing} above the base's pinned {b.missing} — a pinned missing never rises;"
            f" cover {h.missing - b.missing} statement(s) and pin {cap}"
        )
    return out


def floor_diff_rules(
    root: Path,
    base_fl: Floors,
    fl: Floors,
    overall: ScopeCov,
    pkgs: dict[str, ScopeCov],
    renames: dict[str, set[str]],
    *,
    files: dict[str, ScopeCov],
    fd: FileDiff,
    excluded: set[str],
) -> list[str]:
    """The pin rules against the base, for ``overall``, the ``packages`` namespace and the ``files`` namespace
    (the same rules, run twice), plus the file-only rules: ``retired_files`` (SF-1) and the split rule (SF-2).
    ``files``/``fd``/``excluded`` are required keyword arguments: a file rule must never be skipped by omission."""
    out: list[str] = []
    for attr, label in (("band", "band"), ("min_statements", "min_package_statements")):
        if getattr(base_fl, attr) != getattr(fl, attr):
            out.append(f"{label} changed {getattr(base_fl, attr)!r} -> {getattr(fl, attr)!r}: it is fixed")
    # The one version change a diff may make: a v1 base (read as files = {}) becomes v2. One way only.
    if base_fl.version != fl.version and not (base_fl.version == 1 and fl.version == FLOORS_FORMAT_VERSION):
        out.append(f"floors_format_version changed {base_fl.version!r} -> {fl.version!r}: it is fixed (only 1 -> 2)")

    def changed_entry(name: str, b: Floor | None, h: Floor, m: ScopeCov | None) -> None:
        """NEW entry (``b`` None): the measurement within k statements, ``missing`` <= measured. CHANGED entry:
        ``missing`` never rises (<= min(measured + K_MIN, base), M60a); LOWERING needs measured missing <= base missing AND
        percent >= round_down(measured), no tolerance; raising (or unchanged percent) keeps the k tolerance."""
        if m is None or h.percent is None or h.missing is None:
            return
        based = b is not None and b.percent is not None and b.missing is not None
        if based and (b.percent, b.missing) == (h.percent, h.missing):
            return
        k = missing_slack(m)
        lowest = round_down(100.0 * (m.covered - k) / m.statements) if m.statements else 0.0
        # M60a: a CHANGED pin may sit up to K_MIN (the absolute noise, never the size-scaled k) above this run's
        # measured missing, because a pin taken from one CI run is verified on the pin commit's own run (+-2
        # statements). The base side stays exact: a pinned missing never rises. NEW floors stay exact (two runs).
        cap = min(m.missing + K_MIN, b.missing) if based else m.missing
        if h.missing > cap:
            pin = min(m.missing, b.missing) if based else m.missing
            out.append(
                f"{name}: floor pins missing {h.missing} above "
                + (
                    f"min(measured {m.missing} + {K_MIN} of noise, base {b.missing})"
                    if based
                    else f"the measured {m.missing}"
                )
                + f" — a pinned missing never rises; pin {pin}"
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

    def namespace(
        base_map: dict[str, Floor],
        head_map: dict[str, Floor],
        measured: dict[str, ScopeCov],
        present: Callable[[str], bool],
        renamed_to: Callable[[str], set[str]],
        still_there: str,
        vanished: Callable[[str], None],
    ) -> None:
        """One floor namespace (packages or files): removal only when gone, null-back, CHANGED/NEW verification,
        the rename carry and the sum rule. ``vanished`` is called for a removed floor with no detected rename."""
        removed = []
        for key, b in sorted(base_map.items()):
            h = head_map.get(key)
            if h is None:
                if b.percent is None:
                    continue
                if present(key):
                    out.append(f"{key}: floor removed while its {still_there}")
                else:
                    removed.append((key, b))
                continue
            if b.percent is not None and h.percent is None:
                out.append(f"{key}: floor set back to null")
                continue
            changed_entry(key, b, h, measured.get(key))
        new_keys = sorted(k for k in head_map if k not in base_map)
        carried: set[str] = set()
        targets_of: dict[str, list[str]] = {}
        for key, b in removed:
            # A carry-over target follows the CHANGED rules against the removed floor. With no rename detected,
            # every new floor in this diff is a candidate target (fail closed).
            detected = sorted(renamed_to(key))
            if not detected:
                vanished(key)
            targets_of[key] = detected or new_keys
            for t in targets_of[key]:
                h = head_map.get(t)
                if h is None or h.percent is None:
                    out.append(f"{t}: received {key}'s renamed files; the floor carries over — add a floor for {t}")
                    continue
                carried.add(t)
                changed_entry(f"{t} (carrying {key})", b, h, measured.get(t))
        # The group's budget is the sum of its removed floors' base `missing` plus the base `missing` of every target
        # that already had a floor at base (a merge into an existing floored scope brings that scope's own pin).
        base_missing = {key: b.missing for key, b in removed}

        def budget(srcs: set[str], tgts: set[str]) -> int | None:
            if any(base_missing[s] is None for s in srcs):
                return None
            return sum(base_missing[s] or 0 for s in srcs) + sum(
                base_map[t].missing or 0 for t in tgts if t in base_map and t not in srcs
            )

        _sum_rule(
            out,
            {k: set(v) for k, v in targets_of.items()},
            budget,
            lambda t: measured[t].missing if t in measured else 0,
            "a split may not lose uncovered statements between its pieces",
        )
        for k in new_keys:
            if k not in carried:
                changed_entry(k, None, head_map[k], measured.get(k))

    changed_entry("overall", base_fl.overall, fl.overall, overall)
    if base_fl.overall.percent is not None and fl.overall.percent is None:
        out.append("overall: floor set back to null")
    namespace(
        base_fl.packages,
        fl.packages,
        pkgs,
        lambda p: package_present(root, p),
        lambda p: renames.get(p, set()),
        "directory still exists (a floor stays until the package is gone)",
        lambda p: None,  # packages >= N must carry a floor anyway; files are opt-in, hence SF-1 below
    )

    # SF-1: retired_files is append-only; a NEW entry is ref'd (parse), printed, and retires only a floor the base
    # had for a file that is gone at HEAD. It is the ONLY way a floored file may vanish without a detected rename.
    listing = set(_src_files(root))
    retired_now: set[str] = set()
    if problem := append_only_problem(base=base_fl.retired, head=fl.retired, what="retired_files is"):
        out.append(problem)
    else:
        for x in fl.retired[len(base_fl.retired) :]:
            p = x["path"]
            print(f"coverage: RETIRED FILE FLOOR {p} (ref {x['ref']}): {x['reason']}")
            bf = base_fl.files.get(p)
            if bf is None or bf.percent is None:
                out.append(f"retired_files {p}: no file floor at the base to retire")
            elif p in fl.files or file_present(root, p, listing):
                out.append(f"retired_files {p}: the file or its floor is still here at HEAD — retire only a gone file")
            else:
                retired_now.add(p)

    def vanished_file(rel: str) -> None:
        if rel not in retired_now:
            out.append(
                f"{rel}: floored file is gone at HEAD with no detected rename — a file floor does not vanish: keep the "
                "rename detectable (git mv, then edit) so the floor carries, or append a ref'd retired_files entry "
                '{"path": ..., "ref": "#N", "reason": ...}'
            )

    namespace(
        base_fl.files,
        fl.files,
        files,
        lambda r: file_present(root, r, listing),
        lambda r: {fd.renames[r]} if r in fd.renames else set(),
        "file still exists (a file floor stays until the file is gone)",
        vanished_file,
    )
    out += split_rule(root, base_fl, fl, files, fd, excluded, listing)
    return out


def split_rule(
    root: Path,
    base_fl: Floors,
    fl: Floors,
    files: dict[str, ScopeCov],
    fd: FileDiff,
    excluded: set[str],
    listing: set[str],
) -> list[str]:
    """SF-2 (owner decisions 2026-10-13): the floor follows the code. When a diff EDITS OR DELETES ``SPLIT_STATEMENTS`` or more
    of a floored file F's base statements (:func:`removed_statements` against F's ``-M`` rename target when one was
    detected; all of them when F is gone or its path is now a symlink), its successors are every non-excluded
    ``src/maxim`` .py file the diff ADDS and every EXISTING one whose AST statement count grew by ``SPLIT_STATEMENTS``
    or more. Each successor of ``SPLIT_STATEMENTS`` or more statements needs its own file floor (NEW or CHANGED rules,
    by :func:`floor_diff_rules`' namespace pass), and F's measured ``missing`` plus every successor's may not exceed
    F's base ``missing`` pin plus the base pins of successors floored at the base (the shared sum rule, no tolerance;
    an existing successor with no base floor is charged its whole ``missing``). With no successor the sum is skipped."""
    out: list[str] = []
    edges: dict[str, set[str]] = {}
    counts: dict[tuple[str, bool], int | None] = {}

    def count(rel: str, at_base: bool) -> int | None:
        if (rel, at_base) not in counts:
            text = fd.base_text(rel) if at_base else (root / rel).read_text(encoding="utf-8")
            try:
                counts[rel, at_base] = statement_count(text)
            except (SyntaxError, ValueError) as e:
                out.append(f"{rel}: cannot count its statements for the split rule ({e})")
                counts[rel, at_base] = None
        return counts[rel, at_base]

    for f_key, b in sorted(base_fl.files.items()):
        if b.percent is None or b.missing is None:
            continue
        base_src = fd.base_text(f_key)
        if not base_src:
            continue
        here = f_key if f_key in listing else fd.renames.get(f_key)
        here = here if here is not None and file_present(root, here, listing) else None  # N4: a symlink is gone
        if here is None:
            removed = count(f_key, True)
        else:
            try:
                removed = removed_statements(
                    base_src, fd.hunks.get(f_key, []), (root / here).read_text(encoding="utf-8")
                )
            except (SyntaxError, ValueError, tokenize.TokenError) as e:
                out.append(f"{f_key}: cannot map its removed statements for the split rule ({e})")
                continue
        if removed is None or removed < SPLIT_STATEMENTS:
            continue
        successors = []
        for s in sorted((fd.added | fd.modified) - excluded - {f_key} - ({here} if here else set())):
            if s not in listing or not s.endswith(".py"):
                continue
            n = count(s, False)
            if n is None:
                continue
            if s not in fd.added:  # EXISTING: a successor only when it grew by S or more
                before = count(s, True)
                if before is None or n - before < SPLIT_STATEMENTS:
                    continue
            successors.append(s)
            if s not in files and n:
                out.append(f"{s}: successor of {f_key} (split rule) is not measured in {COVERAGE_JSON}")
            if n >= SPLIT_STATEMENTS and s not in fl.files:
                how = "added" if s in fd.added else f"grew by {n - (count(s, True) or 0)} statements"
                out.append(
                    f"{s}: {n} statements ({how}) while this diff removed {removed} of floored {f_key}'s statements "
                    f"— a successor of >= {SPLIT_STATEMENTS} statements needs its own file floor (split rule): add "
                    + (
                        f'"{s}": {{"percent": {round_down(files[s].percent)}, "missing": {files[s].missing}}}'
                        if s in files
                        else f'"{s}": {{"percent": null, "missing": null}}'
                    )
                )
        if successors:  # no successor: the HEAD floor rule (k) and the CHANGED/LOWERING rules hold the source alone
            edges[f_key] = set(successors) | ({here} if here else set())
    _sum_rule(
        out,
        edges,
        lambda srcs, tgts: sum(base_fl.files[s].missing or 0 for s in srcs)
        + sum(base_fl.files[t].missing or 0 for t in tgts if t in base_fl.files and t not in srcs),
        lambda t: files[t].missing if t in files else 0,
        f"split rule: a floored file that lost >= {SPLIT_STATEMENTS} statements keeps its pin across its successors; "
        "an existing successor without a base floor is charged its whole missing — move code into NEW modules",
    )
    return out


def below_tolerance(m: ScopeCov) -> float:
    """The HEAD check's allowance under a floor, in percentage points: k statements of the scope (k as in
    ``missing_slack``). Pins never move with it, so it does not accumulate across PRs."""
    return 100.0 * missing_slack(m) / m.statements if m.statements else 0.0


def suggested_floors(fl: Floors, overall: ScopeCov, pkgs: dict[str, ScopeCov], files: dict[str, ScopeCov]) -> dict:
    """The floors file as measured: every floored package plus every package at or over N, every floored file (file
    floors are opt-in, so no file is added), and ``retired_files`` as committed."""
    keys = set(fl.packages) | {k for k, m in pkgs.items() if m.statements >= MIN_PACKAGE_STATEMENTS}

    def pin(m: ScopeCov | None) -> dict:
        return {"percent": round_down(m.percent), "missing": m.missing} if m else {"percent": None, "missing": None}

    return {
        "floors_format_version": FLOORS_FORMAT_VERSION,
        "band": BAND,
        "min_package_statements": MIN_PACKAGE_STATEMENTS,
        "overall": pin(overall),
        "packages": {k: pin(pkgs.get(k)) for k in sorted(keys)},
        "files": {k: pin(files.get(k)) for k in sorted(fl.files)},
        "retired_files": list(fl.retired),
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


def parse_exclusions(text: str) -> Exclusions:
    d = _loads(text)
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
        if e["ref"] is not None and not ref_ok(e["ref"]):
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
        if x["ref"] is not None and not ref_ok(x["ref"]):
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
        if problem := append_only_problem(base=base_ledger, head=ex.ledger, what="ledger is"):
            return [problem]
    for e in ex.entries:
        if e["path"] not in base_excluded:
            print(f"coverage: NEW EXCLUSION {e['path']} (ref {e['ref']}): {e['reason']}")
            if not ref_ok(e["ref"]):
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
        if x["count"] > base_count(x["kind"], x["file"]) and not ref_ok(x["ref"]):
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
    # Composite actions run inside a workflow's job, so they are read too (#1117 review: the setup steps moved there).
    workflows += sorted((root / ".github" / "actions").glob("**/action.y*ml"))
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
        raw=True,  # bytes: no newline translation of a lone CR (parse_unified_diff)
    )
    out: dict[str, set[int]] = {}
    for f in parse_unified_diff(text):  # count-based: an added ``++x`` line no longer reads as a ``+++`` header
        if f.new is not None and f.new.endswith(".py"):
            lines = out.setdefault(f.new, set())
            for _, _, start, n in f.hunks:
                lines.update(range(start, start + n))
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
    """Reached only on a line coverage.py's ``if TYPE_CHECKING:`` regex excluded; NOT mypy's set (see ``_lint_allowance.py``)."""
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


def _line_statements(text: str, lines: set[int]) -> list[tuple[int, _Stmt]]:
    """Each line of ``lines`` that carries code, in order, with the statement it maps to: its innermost enclosing
    ``ast.stmt`` (decorator lines included), or for a clause header that is not a statement (``except``, ``else``,
    ``finally``, ``case``) its block's first statement. Shared by diff coverage (changed lines at HEAD) and the split
    rule (deleted lines at the base). Raises ``SyntaxError``/``tokenize.TokenError``."""
    stmts = _statements(ast.parse(text))
    headers = _clause_headers(stmts)
    code = _code_lines(text)
    out: list[tuple[int, _Stmt]] = []
    for ln in sorted(lines & code):
        enclosing = [s for s in stmts if s.first <= ln <= s.end]
        if not enclosing:
            continue
        st = max(enclosing, key=lambda s: (s.first, -s.end))
        clause = [h for h in headers if h[0] <= ln <= h[1] and h[0] >= st.first]
        if clause:
            st = max(clause, key=lambda h: h[0])[2]
        out.append((ln, st))
    return out


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
            mapped = _line_statements(text, lines)
        except (SyntaxError, tokenize.TokenError) as e:
            fails.append(f"{rel}: cannot parse ({e})")
            continue
        seen: dict[int, bool] = {}
        for _, st in mapped:
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


def _print_totals(overall: ScopeCov, pkgs: dict[str, ScopeCov], fl: Floors | None, files: dict[str, ScopeCov]) -> None:
    def line(name: str, m: ScopeCov) -> str:
        f = (fl.overall if name == "overall" else fl.packages.get(name) or fl.files.get(name)) if fl else None
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
    for k in sorted(fl.files if fl else {}):  # floored files only (opt-in)
        print(line(k, files[k]) if k in files else f"  {k}: not measured")


def _read(root: Path, rel: str, parse):
    try:
        return parse((root / rel).read_text(encoding="utf-8")), None
    except OSError as e:
        return None, f"{rel}: unreadable ({e})"
    except FileFormatError as e:
        return None, f"{rel}: {e}"


def _statements_at_least(v: dict) -> int:
    """A LOWER bound on a scope's statement count from its printed pin: the true percent lies in [p, p + 0.1), so
    statements = 100·missing / (100 − p_true) >= 100·missing / (100 − p). 0 when the pin carries no information."""
    p, m = v["percent"], v["missing"]
    if not m or p >= 100:
        return 0
    return math.ceil(100.0 * m / (100.0 - p) - 1e-6)


def merge_floors(texts: list[str], committed: Floors | None) -> tuple[dict, list[str]]:
    """(the per-scope MINIMUM of several measured floors files, problems). The input is the paste-ready JSON a run
    prints; the minimum is the lowest percent AND the lowest missing (``missing`` is one-sided, so the lower pin
    passes against every run). Only the NEW scopes are merged (SF-A4): those ``null`` in, or absent from, the
    ``committed`` floors file — every scope when ``committed`` is None (``--all-scopes``) — so a spread in an unrelated
    scope cannot block a new pin, and the result holds only those entries. The merge REFUSES (SF-3, owner decision
    2026-10-13) a merged scope measured in fewer than two of the runs, or whose runs' ``missing`` differ by more than
    its k (a non-deterministic test is fixed before pinning, never absorbed into a floor), and byte-identical inputs.
    It enforces the spread of the runs it is given; it cannot see where they came from. k is computed from a lower
    bound on the scope's statement count (:func:`_statements_at_least`), so the spread check can only be stricter than
    the HEAD tolerance, never looser. Every entry is validated by :func:`_parse_floor` (a ``FileFormatError``)."""
    docs = []
    for i, t in enumerate(texts):
        d = _loads(t)
        if not isinstance(d, dict) or not isinstance(d.get("overall"), dict) or not isinstance(d.get("packages"), dict):
            raise FileFormatError(f"run {i + 1}: a run's floors JSON needs `overall` and `packages`")
        if not isinstance(d.get("files", {}), dict):
            raise FileFormatError(f"run {i + 1}: `files` must be an object")
        _parse_floor("overall", d["overall"])
        for k, v in d["packages"].items():
            _parse_floor(k, v)
        for k, v in d.get("files", {}).items():
            if problem := file_key_problem(k):
                raise FileFormatError(f"run {i + 1}: {problem}")
            _parse_floor(k, v)
        docs.append(d)
    problems: list[str] = []
    for i in range(len(texts)):
        for j in range(i):
            if texts[i] == texts[j]:
                problems.append(f"runs {j + 1} and {i + 1} are byte-identical — the same run given twice is one run")

    def new(name: str, have: dict[str, Floor] | None) -> bool:
        if committed is None:
            return True
        f = committed.overall if name == "overall" else (have or {}).get(name)
        return f is None or f.percent is None

    def merge(name: str, vals: list[dict | None]) -> dict:
        vals = [v for v in vals if v and v.get("percent") is not None]
        if not vals:
            return {"percent": None, "missing": None}
        if len(vals) < 2:
            problems.append(f"{name}: measured in {len(vals)} of {len(docs)} runs — a floor is pinned from 2 or more")
        spread = max(v["missing"] for v in vals) - min(v["missing"] for v in vals)
        k = missing_slack(ScopeCov(statements=max(_statements_at_least(v) for v in vals)))
        if spread > k:
            problems.append(
                f"{name}: the runs' missing differ by {spread} statements, more than its k = {k} — a flaky scope "
                "needs a look (fix the non-deterministic test) before a floor is pinned from it"
            )
        return {"percent": min(v["percent"] for v in vals), "missing": min(v["missing"] for v in vals)}

    retired = [d.get("retired_files", []) for d in docs]
    if any(r != retired[0] for r in retired):
        problems.append("the runs carry different retired_files lists — they were not measured on one floors file")
    pkg_keys = sorted(
        k for k in {k for d in docs for k in d["packages"]} if new(k, committed.packages if committed else None)
    )
    file_keys = sorted(
        k for k in {k for d in docs for k in d.get("files", {})} if new(k, committed.files if committed else None)
    )
    merged: dict = {}
    if committed is None:
        merged.update(
            floors_format_version=FLOORS_FORMAT_VERSION, band=BAND, min_package_statements=MIN_PACKAGE_STATEMENTS
        )
    if new("overall", None):
        merged["overall"] = merge("overall", [d["overall"] for d in docs])
    if committed is None or pkg_keys:
        merged["packages"] = {k: merge(k, [d["packages"].get(k) for d in docs]) for k in pkg_keys}
    if committed is None or file_keys:
        merged["files"] = {k: merge(k, [d.get("files", {}).get(k) for d in docs]) for k in file_keys}
    if committed is None:
        merged["retired_files"] = retired[0]
    elif not merged:
        problems.append(
            f"no scope is null in (or absent from) the committed {FLOORS_REL}: nothing new to pin "
            "(--all-scopes merges every scope)"
        )
    return merged, problems


# ── --pins-from-run: CI's printed pins, applied in one command (M60a) ─────────────────────────────────────────────

RUN_SUITE_STEP = "Run required fast suite"
COVERAGE_ARTIFACT = "coverage-json"


def _download_artifact(run_id: str, name: str, dest: Path) -> Path:
    try:
        r = subprocess.run(
            ["gh", "run", "download", run_id, "-n", name, "-D", str(dest)], capture_output=True, text=True, timeout=300
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GitUnavailable(f"gh run download {run_id} -n {name}: {exc}") from exc
    if r.returncode != 0:
        raise GitUnavailable(f"gh run download {run_id} -n {name}: {r.stderr.strip() or r.returncode}")
    return dest / COVERAGE_JSON


def run_measurement(
    root: Path,
    run_id: str,
    *,
    api: Callable[[str], object] = gh_api,
    download: Callable[[str, str, Path], Path] = _download_artifact,
    tmp: Path,
) -> dict[str, FileCov]:
    """The ``coverage.json`` of CI run ``run_id``, bound to this checkout: the run's head SHA is ``HEAD``, its latest
    attempt's ``unit-tests`` job ran the measuring suite to ``success`` (a ``-x`` failure leaves partial data, and the
    upload runs on ``always()``), and the artifact is that attempt's (``coverage-json-<attempt>``; the unsuffixed
    legacy name only on attempt 1). Raises :class:`MeasurementError` naming the refusal, :class:`GitUnavailable` when
    ``gh`` cannot answer."""
    run = api(f"repos/{{owner}}/{{repo}}/actions/runs/{run_id}")
    if not isinstance(run, dict):
        raise MeasurementError(f"run {run_id}: unexpected API response")
    head = git(root, "rev-parse", "HEAD").strip()
    if run.get("head_sha") != head:
        raise MeasurementError(
            f"run {run_id} measured {str(run.get('head_sha'))[:12]}, not this checkout's HEAD {head[:12]}: "
            "pin from the run of the commit you are on"
        )
    if not str(run.get("path", "")).endswith("workflows/test.yml") or run.get("event") not in ("pull_request", "push"):
        raise MeasurementError(
            f"run {run_id} is {run.get('path')!r} on {run.get('event')!r}: pin from a Tests run of a pull_request or push"
        )
    attempt = run.get("run_attempt")
    if not isinstance(attempt, int) or attempt < 1:
        raise MeasurementError(f"run {run_id}: no run_attempt in the API response")
    jobs = api(f"repos/{{owner}}/{{repo}}/actions/runs/{run_id}/attempts/{attempt}/jobs?per_page=100")
    job = next((j for j in (jobs or {}).get("jobs", []) if isinstance(j, dict) and j.get("name") == "unit-tests"), None)
    if job is None:
        raise MeasurementError(f"run {run_id} attempt {attempt} has no unit-tests job")
    step = next((s for s in job.get("steps") or [] if isinstance(s, dict) and s.get("name") == RUN_SUITE_STEP), None)
    if step is None or step.get("conclusion") != "success":
        found = "absent" if step is None else step.get("conclusion")
        raise MeasurementError(
            f"run {run_id} attempt {attempt}: the step {RUN_SUITE_STEP!r} is {found}, not success: a failed or "
            "skipped suite leaves partial coverage data"
        )
    names = {
        a.get("name")
        for a in (api(f"repos/{{owner}}/{{repo}}/actions/runs/{run_id}/artifacts?per_page=100") or {}).get(
            "artifacts", []
        )
        if isinstance(a, dict) and not a.get("expired")
    }
    wanted = f"{COVERAGE_ARTIFACT}-{attempt}"
    if wanted not in names:
        if attempt == 1 and COVERAGE_ARTIFACT in names:
            wanted = COVERAGE_ARTIFACT  # runs uploaded before the attempt-bound name (M60a)
        else:
            raise MeasurementError(
                f"run {run_id}: no unexpired artifact {wanted!r} (found {sorted(n for n in names if n)})"
            )
    return load_coverage(root, download(run_id, wanted, tmp))


def pins_from_measurement(
    fl: Floors, overall: ScopeCov, pkgs: dict[str, ScopeCov], files: dict[str, ScopeCov]
) -> tuple[dict[str, Floor], list[str]]:
    """The pins the CI hint prints for every floored scope more than ``band`` above its floor (``round_down`` of the
    measurement, ``missing`` capped by the current pin: a pinned missing never rises), and notes for what is left
    alone: a scope below its floor (a fall is fixed by covering code, never by a pin), a null floor (NEW floors come
    from two runs, ``--merge-floors``), an unmeasured scope."""
    pins: dict[str, Floor] = {}
    notes: list[str] = []
    scopes = [("overall", fl.overall, overall)]
    scopes += [(k, f, pkgs.get(k)) for k, f in sorted(fl.packages.items())]
    scopes += [(k, f, files.get(k)) for k, f in sorted(fl.files.items())]
    for name, f, m in scopes:
        if m is None or m.statements == 0:
            notes.append(f"{name}: not measured by the run — left alone")
        elif f.percent is None:
            notes.append(f"{name}: null floor — a NEW floor is pinned from two runs (--merge-floors), left alone")
        elif m.percent > f.percent + BAND + 1e-9:
            pins[name] = Floor(round_down(m.percent), m.missing if f.missing is None else min(m.missing, f.missing))
        elif f.missing is not None and f.missing > m.missing + K_MIN:
            notes.append(
                f"{name}: its pinned missing {f.missing} is more than {K_MIN} above the run's {m.missing}; if this "
                f"branch changed that pin, CI refuses it: pin {min(m.missing, f.missing)}"
            )
        elif m.percent < f.percent - below_tolerance(m) - 1e-9:
            notes.append(
                f"{name}: {_fmt_pct(m.percent)} is below its floor {f.percent}% — cover code; a pin cannot fix a fall"
            )
    return pins, notes


def apply_pins(text: str, pins: dict[str, Floor]) -> str:
    """``coverage_floors.json`` with ``pins`` written in place: key order and every other byte as the file had them
    (it round-trips through ``json.dumps(indent=2)``)."""
    doc = json.loads(text)
    for name, f in pins.items():
        slot = doc if name == "overall" else doc["files" if name.startswith(SCOPE + "/") else "packages"]
        slot[name] = {"percent": f.percent, "missing": f.missing}
    return json.dumps(doc, indent=2) + "\n"


def pins_from_run_main(root: Path, run_id: str) -> int:
    import tempfile

    if not run_id.isdigit():
        print(f"usage: lint_coverage.py --pins-from-run RUN_ID (a number, got {run_id!r})", file=sys.stderr)
        return 2
    try:
        text = (root / FLOORS_REL).read_text(encoding="utf-8")
        fl = parse_floors(text)
        with tempfile.TemporaryDirectory() as td:
            cov = run_measurement(root, run_id, tmp=Path(td))
    except (OSError, FileFormatError, GitUnavailable) as e:
        print(f"coverage: --pins-from-run: {e}", file=sys.stderr)
        return 2
    except MeasurementError as e:
        print(f"coverage: --pins-from-run REFUSED: {e}", file=sys.stderr)
        return 1
    overall, pkgs = measure_scopes(cov)
    pins, notes = pins_from_measurement(fl, overall, pkgs, file_scopes(cov))
    for n in notes:
        print(f"coverage: --pins-from-run: {n}")
    if not pins:
        print(
            f"coverage: --pins-from-run: run {run_id} puts no scope more than {BAND} pt above its floor; nothing to pin"
        )
        return 0
    old = {"overall": fl.overall, **fl.packages, **fl.files}
    for name, f in pins.items():
        print(f"coverage: pin {name}: {old[name].percent}%/{old[name].missing} -> {f.percent}%/{f.missing}")
    (root / FLOORS_REL).write_text(apply_pins(text, pins), encoding="utf-8")  # a tracked file: git is the undo
    print(f"coverage: wrote {len(pins)} pin(s) to {FLOORS_REL} from run {run_id}; commit them, and CI verifies them")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--merge-floors"]:
        every = "--all-scopes" in argv[1:]
        runs = [a for a in argv[1:] if a != "--all-scopes"]
        if len(runs) < 2:
            print("usage: lint_coverage.py --merge-floors [--all-scopes] RUN1.json RUN2.json [...]", file=sys.stderr)
            return 2
        try:
            # Only the scopes null in (or absent from) the committed floors file are merged (SF-A4).
            committed = None if every else parse_floors((REPO_ROOT / FLOORS_REL).read_text(encoding="utf-8"))
            merged, problems = merge_floors([Path(a).read_text(encoding="utf-8") for a in runs], committed)
        except (OSError, FileFormatError) as e:
            print(f"coverage: --merge-floors: {e}", file=sys.stderr)
            return 2
        if problems:
            print("coverage: --merge-floors REFUSED (no floors printed):", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            return 1
        print(json.dumps(merged, indent=2))
        return 0
    if argv[:1] == ["--pins-from-run"]:
        if len(argv) != 2:
            print("usage: lint_coverage.py --pins-from-run RUN_ID", file=sys.stderr)
            return 2
        return pins_from_run_main(REPO_ROOT, argv[1])
    # --local: the pre-push run scripts/suite_at_commit.sh makes on the exact commit to be pushed. Floors are measured
    # in CI's environment (macOS reads 1-2 pt off Linux), so locally NO floor gates, in either direction (M60a, owner
    # decision 2026-10-27; it gated the touched packages' floors before, which gave false reds and false greens): every
    # floor result and a PR's own floor-pin edits are printed as notes -- except a pinned ``missing`` that rose above
    # its base, which reads no measurement and gates as in CI (pin_rises, #1270; a stale origin/main can misplace the
    # base: fetch first). The floors file's format and fixed fields, diff coverage (#1232) and every other rule gate
    # exactly as in CI. Never in CI: ci_step_rules pins the step to the bare command, and this refuses to run under
    # GitHub Actions.
    local = argv[:1] == ["--local"]
    if local and os.environ.get("GITHUB_ACTIONS"):
        print("coverage: --local is the pre-push check and cannot run inside GitHub Actions", file=sys.stderr)
        return 2
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
    files = file_scopes(cov)
    _print_totals(overall, pkgs, fl, files)
    if fl is not None:
        if local:
            gating = floor_head_rules(root, fl, overall, pkgs, only=set(), files=files)  # the format checks only
            for note in floor_head_rules(root, fl, overall, pkgs, files=files):
                if note not in gating:
                    print(f"coverage (local floor result, measured in CI's environment, not gated): {note}")
            failures += gating
        else:
            failures += floor_head_rules(root, fl, overall, pkgs, files=files)
        if fl.overall.percent is None or any(f.percent is None for f in [*fl.packages.values(), *fl.files.values()]):
            print(f"coverage: measured floors for {FLOORS_REL} (rounded down to 0.1; review before committing):")
            print(json.dumps(suggested_floors(fl, overall, pkgs, files), indent=2))

    if event in (None, "", "pull_request", "push"):  # push: against the last green push (_lint_git.push_base, #1089)
        try:
            base = base_ref(root)
        except GitUnavailable as e:
            if must_not_skip(str(e)):
                return 2
            print(f"INFO: no base available; diff-scoped rules skipped ({e})")
        else:
            try:
                if fl is not None:
                    excluded = set(ex.paths) if ex is not None else set()
                    base_text = show(root, base, FLOORS_REL)
                    if base_text:
                        try:
                            base_fl = parse_floors(base_text)
                        except FileFormatError as e:
                            failures.append(f"{FLOORS_REL} at the base: {e}")
                        else:
                            # git is asked only when a file floor exists on either side (NO_FILE_DIFF's note)
                            fd = file_diff(root, base) if base_fl.files or fl.files else NO_FILE_DIFF
                            pin = floor_diff_rules(
                                root,
                                base_fl,
                                fl,
                                overall,
                                pkgs,
                                package_renames(root, base),
                                files=files,
                                fd=fd,
                                excluded=excluded,
                            )
                            if local:  # a PR's floor edits are CI measurements: CI verifies them, not this box,
                                # except the half that reads no measurement: a pinned missing that rose (#1270)
                                scopes = {"overall": overall, **pkgs, **files}
                                rises = pin_rises(base_fl, fl, scopes.get)
                                failures += rises
                                gated = {r.split(":", 1)[0] for r in rises}  # gated above: not also a "not gated" note
                                for note in pin:
                                    if note.split(":", 1)[0] not in gated:
                                        print(f"coverage (local, floor pin rule, verified in CI, not gated): {note}")
                            else:
                                failures += pin
                    else:
                        print(f"coverage: {FLOORS_REL} absent at the base — bootstrap, every floor is new")
                        empty = Floors(fl.band, fl.min_statements, fl.version, Floor(None, None))
                        fd = file_diff(root, base) if fl.files else NO_FILE_DIFF
                        failures += floor_diff_rules(
                            root, empty, fl, overall, pkgs, {}, files=files, fd=fd, excluded=excluded
                        )
                if ex is not None:
                    base_text = show(root, base, EXCLUSIONS_REL)
                    try:
                        base_ex = parse_exclusions(base_text) if base_text else None
                    except FileFormatError as e:
                        failures.append(f"{EXCLUSIONS_REL} at the base: {e}")
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
