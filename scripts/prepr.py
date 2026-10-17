#!/usr/bin/env python3
"""Run the CI lint job's checks locally, read from the workflow itself (backlog M59).

The pre-PR list used to be followed by attention ("run every ``jobs.lint`` step, the inline greps included"), and
#1275 failed CI on the PR-title lint anyway. This script parses ``.github/workflows/test.yml`` AT RUN TIME and runs:

1. every ``run:`` step of ``jobs.lint``, in workflow order, each finishing before the next (ruff check + ruff
   format --check are the first of them, exactly as CI runs them);
2. the unit-tests job's one step that runs ``scripts/run_mypy_public.sh`` (the targeted mypy check, M65).

**prepr is a gate, so it refuses rather than guesses.** A developer reads ``prepr: PASS`` as clearance to push. Any
condition under which prepr cannot reproduce CI's verdict is exit 2 (REFUSED), or an operator-chosen downgrade
(``--allow-*``, ``--no-fetch``, ``--no-pr``, PR text typed by hand) that is printed and turns the outcome into
``prepr: TAINTED-PASS (...)``, exit 3. A plain ``prepr: PASS`` (exit 0) means nothing was downgraded. A non-Linux
platform is printed as a separate ``NOTE:`` line and does not change the exit code: the operator cannot change it
per run, and a gate that never exits 0 on a Mac would be read past.

Inputs and how each could lie (the full note is the M59 approach note; each row is a test in test_prepr.py):

=========================  ====================================================  =====================================
input                      how it could lie                                      defence
=========================  ====================================================  =====================================
origin/main + fetch        stale ref: "contains main" but not today's main       fetch first; failure -> exit 2;
                                                                                 ``--no-fetch`` taints
shallow clone / merge-base empty merge-base: every diff-scoped lint SKIPS         shallow or no merge-base -> exit 2
                           locally (``_lint_git.must_not_skip`` only errors
                           under GITHUB_EVENT_NAME) -> vacuous PASS
HEAD vs CI's merge commit  CI tests ``refs/pull/N/merge``; a branch behind main  ``rev-list HEAD..origin/main`` must be
                           is judged on a tree nobody ran locally                empty (then the merge tree == HEAD's);
                                                                                 ``--allow-behind`` taints
working tree               uncommitted/untracked files: the tree-wide steps see  dirty (untracked included), or an
                           them, base..HEAD lints and CI do not; gitignored      ignored *.py under src/ tests/
                           *.py that tree-globbing lints read                    scripts/ -> exit 2; ``--allow-dirty``
                                                                                 taints
gh PR lookup               a CLOSED/MERGED PR; another branch's PR; a fork's     state must be OPEN (its text is CI's,
                           same-named branch                                     wherever its head is); a cross-repo
                                                                                 PR, a head owner != origin's owner,
                                                                                 or headRefName != this branch ->
                                                                                 exit 2
typed ``--title/--body``   hand-typed text silently replacing the PR's           differs (CRLF/trailing space
                                                                                 normalized) from an OPEN PR -> exit 2;
                                                                                 with no OPEN PR, taints
PR text secrets            CI's secret scan reads the PR title+body only under   the secret-scan step is classified
                           ``pull_request``; the local-mode step does not;       (unclassifiable mention ->
                           a substring test misses ``python -m ...``             PlanError);
                                                                                 ``lint_secrets.scan_range(...,
                                                                                 pr_text=...)`` called directly; a
                                                                                 finding fails, a load/git/config
                                                                                 error refuses
``--no-pr``                skipping the PR-title lint reads as a full PASS       skips only steps reading the PR
                                                                                 title/body; taints
ruff/mypy versions         another version reports other findings; a ``>=``,     ``<tool> --version`` on the step's
                           unpinned or ``-r file`` install leaves CI's version   PATH vs the dropped ``==`` pin ->
                           unknowable                                            exit 2 unless ``--allow-pin-mismatch``
                                                                                 (other pins WARN); no exact pin ->
                                                                                 PlanError
setup-python version       3.11 here vs 3.12 in CI: parse/ast lints differ       major.minor must match -> else exit 2
                                                                                 unless ``--allow-python-mismatch``
job/step keys, ``uses:``   ``if:``/``defaults``/``container``/``strategy``/...   allow-lists; anything else is a
                           change what runs; a ``uses:`` lint action skipped     PlanError (exit 2); only checkout
                                                                                 (fetch-depth 0) and setup-python skip
``env:`` / ``${{ }}``      an expression CI evaluates, passed literally here     three understood in ``env:`` (PR
                                                                                 title, PR body, github.token:
                                                                                 dropped); else, and anywhere in a
                                                                                 ``run:`` body, PlanError
event/runner/operator env  a leaked GITHUB_* flips lints to PR/push mode;        GITHUB_*, RUNNER_*, ACTIONS_*,
                           BASH_ENV is sourced by non-interactive bash; an       MAXIM_*, BASH_ENV, ENV, PYTHONPATH,
                           operator's PYTHONPATH/MYPYPATH/MAXIM_* (CI has none)  MYPYPATH, PYTHONWARNINGS/STARTUP/
                                                                                 HOME/USERBASE, VIRTUAL_ENV removed,
                                                                                 never faked
shell                      no ``shell:`` is ``bash -e {0}`` (NO pipefail);       per-step argv from ``shell:``; only
                           ``shell: bash`` is ``bash --noprofile --norc -eo      bash/absent, else PlanError
                           pipefail {0}``
install lines              ``pip install x && check`` dropped whole; ``cd d &&   plain-form installs dropped; an
                           pip install`` kept (mutates the operator's env)       install with ``& ; | ` $(`` or after
                                                                                 a separator -> PlanError; so is an
                                                                                 editable/VCS/path install (its
                                                                                 ruff/mypy versions are unknowable)
OS (BSD vs GNU, bash 3.2)  macOS grep/sed/find and bash 3.2 vs ubuntu GNU/bash 5 a ``NOTE:`` line on non-Linux
python3 / pyyaml           steps resolve another python3; no yaml -> traceback   interpreter dir first on PATH;
                                                                                 prepr.sh probes ``import yaml``
=========================  ====================================================  =====================================

The payload-reader guard (test_prepr.py) is a tripwire for literal ``["pull_request"]`` and
``.get("pull_request")`` in scripts/: a variable key, ``getattr``, ``pop``, an ``in`` test, or jq in a shell step
would slip past it (a ``${{ github.event.pull_request.* }}`` in a run body is already a PlanError, E1); CI stays
authoritative either way. An ssh host alias (``git@github-work:o/r``) is refused, not resolved. Residuals, stated rather than defended: CI runs the lints in ``pull_request`` mode and prepr in local mode; the one
verdict difference found (the PR-text secret scan) is reproduced above by a direct call. Other operator environment
variables are inherited. mypy's view of
third-party packages is whatever is installed here (``--ignore-missing-imports`` turns a missing one into Any). CI's
network namespace, the fast suite and the coverage ratchet are not reproduced (``scripts/suite_at_commit.sh``).

**The skip rule, and the only one.** A ``uses:`` step (checkout / setup-python only) is skipped. Inside a run script,
a logical line (backslash continuations joined) that is a plain install command (``pip``/``pip3``/``pip3.X``,
``python[3[.X]] -m pip``, ``uv pip``, ``sudo [-H] pip``, options such as ``-q`` before ``install`` allowed) is dropped: local installs are the operator's, and the
dropped ``==`` pins are compared with what is installed. A step left with no command line is skipped as
install-only. Nothing else is filtered: the greps, the ``git fetch`` the prereg lint runs, and every comment line run
as written. ``tests/unit/test_prepr.py`` pins the selection to the workflow with an independent expected-drop list.

**Environment.** Workflow-, job- and step-level literal ``env:`` are honoured. Steps run with the shell GitHub would
use for their ``shell:`` key, from the repo root, with this interpreter's directory first on ``PATH`` (so
``python3`` is the one running prepr). ``PYTHONPATH`` is REMOVED, not set, because CI does not set it (it
``pip install -e``s the checkout): no lint step imports ``maxim``, and mypy resolves the command-line files' own
``src/`` before site-packages, so a worktree's editable-install shadowing does not reach any step. Setting it (or
inheriting the operator's) would make prepr differ from CI.

    scripts/prepr.sh [--title T] [--body B] [--no-pr] [--no-fetch] [--allow-dirty] [--allow-behind]
                     [--allow-pin-mismatch] [--allow-python-mismatch] [--fail-fast] [--verbose] [--list]

exit codes:
  0  prepr: PASS          every step passed, nothing downgraded (a NOTE: platform line may still print)
  1  prepr: FAIL          a step (or the PR-text secret scan) failed
  2  prepr: REFUSED       the plan could not be built, or CI's conditions could not be reproduced
  3  prepr: TAINTED-PASS  every step passed under an operator downgrade: --allow-*, --no-fetch, --no-pr, or PR
                          text typed by hand
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

import yaml

# The docstring's exit-code block, reused as --help's epilog (test_prepr pins that they agree).
EXIT_CODES = (__doc__ or "")[(__doc__ or "").index("exit codes:") :]
REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_REL = ".github/workflows/test.yml"
LINT_JOB = "lint"
MYPY_JOB = "unit-tests"
MYPY_SCRIPT = "scripts/run_mypy_public.sh"

# An install command (the only commands the skip rule drops, and only in their plain form).
_INSTALL_CMD = (
    r"(?:sudo\s+(?:-\S+\s+)*)?"
    r"(?:uv\s+pip|pip3?(?:\.\d+)?|python(?:3(?:\.\d+)?)?\s+-m\s+pip)(?:\s+-\S+)*\s+install\b"
)
INSTALL_LINE = re.compile(r"^\s*" + _INSTALL_CMD)
# An install command after a shell separator, mid-line: never dropped, never run (PlanError).
INSTALL_AFTER_SEPARATOR = re.compile(r"(?:&|\||;|\(|`|\bthen\b|\bdo\b|\belse\b)\s*" + _INSTALL_CMD)
SHELL_METACHARS = ("&", ";", "|", "`", "$(")
# The expressions prepr understands; the value is the local source (None = drop the variable).
EXPRESSIONS: dict[str, str | None] = {
    "${{ github.event.pull_request.title }}": "title",
    "${{ github.event.pull_request.body }}": "body",
    "${{ github.token }}": None,
}
# GitHub's command line per `shell:` value (docs: "Using a specific shell"). Absent is NOT the same as `bash`.
SHELLS: dict[str | None, tuple[str, ...]] = {
    None: ("bash", "-e"),
    "bash": ("bash", "--noprofile", "--norc", "-eo", "pipefail"),
}
# Keys that do not change what a job's steps run or how. Anything else (defaults, container, services, strategy,
# if, ...) is a PlanError.
JOB_KEYS = frozenset({"name", "runs-on", "timeout-minutes", "permissions", "env", "steps", "needs", "concurrency"})
RUN_STEP_KEYS = frozenset({"name", "run", "env", "shell"})
USES_STEP_KEYS = frozenset({"name", "uses", "with"})
# The only `uses:` actions prepr may skip, with the only `with:` keys each may carry.
SKIPPABLE_USES: dict[str, frozenset[str]] = {
    "actions/checkout": frozenset({"fetch-depth"}),
    "actions/setup-python": frozenset({"python-version"}),
}
# Removed from every step's environment, never faked: lints then run in local mode against the merge-base.
SCRUB_PREFIXES = ("GITHUB_", "RUNNER_", "ACTIONS_")
# VIRTUAL_ENV is removed too: steps find python3/ruff/mypy through PATH (this interpreter's directory first), and a
# venv interpreter locates its environment from the pyvenv.cfg beside it, not from VIRTUAL_ENV (test-pinned).
SCRUB_VARS = (
    "BASH_ENV",
    "ENV",
    "PYTHONPATH",
    "MYPYPATH",
    "PYTHONWARNINGS",
    "PYTHONSTARTUP",
    "PYTHONHOME",
    "PYTHONUSERBASE",
    "VIRTUAL_ENV",
    "PYTHONOPTIMIZE",
    "PYTHONINSPECT",
)
SCRUB_MAXIM_PREFIX = "MAXIM_"  # CI sets none; an operator's MAXIM_* toggles must not reach a step
# Roots whose gitignored *.py files CI never has but tree-globbing lints (lint_orphan_modules) would read here.
IGNORED_PY_ROOTS = ("src", "tests", "scripts")
# Pins whose mismatch refuses (they decide what the step reports); other pins only WARN.
STRICT_PINS = ("ruff", "mypy")
PR_SECRET_SCAN = "Secret scan of the PR title + body (lint_secrets.scan_range, as CI's pull_request run)"
PIN = re.compile(r"""["']?([A-Za-z0-9_.\-\[\]]+)==([A-Za-z0-9_.+\-]+)["']?""")


class PlanError(RuntimeError):
    """The workflow holds something prepr cannot reproduce faithfully."""


class Refusal(RuntimeError):
    """The local conditions cannot reproduce CI's verdict (exit 2)."""


@dataclass(frozen=True)
class Step:
    """One workflow step as prepr will run it (runtime-only; never persisted)."""

    job: str
    name: str
    script: str  # the run script with install lines dropped
    env: dict[str, str] = field(default_factory=dict)
    pr_env: dict[str, str] = field(default_factory=dict)  # VAR -> "title" | "body"
    dropped: tuple[str, ...] = ()  # the install commands removed by the skip rule
    shell: tuple[str, ...] = SHELLS[None]
    secret_scan: bool = False  # runs lint_secrets in diff mode: CI also scans the PR text there


@dataclass(frozen=True)
class Plan:
    steps: tuple[Step, ...]
    skipped: tuple[tuple[str, str, str], ...]  # (job, step name, reason)
    python_versions: frozenset[str] = frozenset()
    lint_timeout_minutes: int | None = None

    @property
    def needs(self) -> frozenset[str]:
        """PR fields some step needs; the secret-scan step needs the title (CI scans the PR text there)."""
        scan = {"title"} if any(s.secret_scan for s in self.steps) else set()
        return frozenset({v for s in self.steps for v in s.pr_env.values()} | scan)


def _logical_lines(script: str) -> list[list[str]]:
    """Group physical lines into logical commands (a trailing backslash continues the line)."""
    groups: list[list[str]] = []
    cur: list[str] = []
    for line in script.splitlines():
        cur.append(line)
        if not line.rstrip().endswith("\\"):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def _code_part(line: str) -> str:
    """The line before its trailing shell comment (a ``#`` at the start or after whitespace, outside quotes)."""
    quote = ""
    prev = " "
    i = 0
    while i < len(line):
        c = line[i]
        if quote:
            if c == "\\" and quote == '"':
                i += 2
                prev = "x"
                continue
            if c == quote:
                quote = ""
        elif c == "\\":
            i += 2
            prev = "x"
            continue
        elif c in "'\"":
            quote = c
        elif c == "#" and prev.isspace():
            return line[:i]
        prev = c
        i += 1
    return line


def strip_installs(script: str, where: str = "step") -> tuple[str, tuple[str, ...]]:
    """(script without its plain install commands, the dropped commands). A compound install is a PlanError."""
    kept: list[str] = []
    dropped: list[str] = []
    for group in _logical_lines(script):
        code = _code_part(" ".join(g.rstrip().rstrip("\\").strip() for g in group)).strip()
        if INSTALL_LINE.match(code):
            bad = [m for m in SHELL_METACHARS if m in code]
            if bad:
                raise PlanError(
                    f"{where}: install line {code!r} also carries {bad}: dropping it would drop the rest of the "
                    "command, running it would install into this environment"
                )
            dropped.append(code)
        elif INSTALL_AFTER_SEPARATOR.search(code):
            raise PlanError(f"{where}: an install command inside a compound line {code!r} cannot be dropped or run")
        else:
            kept.extend(group)
    return "\n".join(kept) + ("\n" if kept else ""), tuple(dropped)


def strict_pin_problem(cmd: str) -> str | None:
    """Why a dropped install leaves a STRICT_PINS tool's CI version unknowable (None when it does not).

    The pin check compares ``tool==X`` with the local tool; ``>=``, ``~=``, a spaced ``==``, no pin, or a
    requirements/constraints file would leave CI's version unknown, so prepr refuses rather than compare nothing.
    """
    try:
        tokens = shlex.split(cmd)
    except ValueError as e:
        return f"cannot parse install line {cmd!r} ({e})"
    for tok in tokens:
        if tok in ("-i", "--index-url", "--extra-index-url", "-f", "--find-links") or tok.startswith(
            ("--index-url=", "--extra-index-url=", "--find-links=")
        ):
            return f"install line {cmd!r} uses another package index/source: an exact pin may resolve to other contents"
        if (
            tok in ("-e", "--editable")
            or tok.startswith("--editable=")
            or (tok.startswith("-e") and not tok.startswith("--") and len(tok) > 2)
            or "git+" in tok
            or "://" in tok
            or tok == "."
            or tok.startswith((".[", "./", "../", "/", "~"))
            or tok.endswith((".whl", ".tar.gz", ".zip"))
        ):
            return f"install line {cmd!r} installs from a path/VCS/editable source, whose ruff/mypy pins are unknowable"
        if tok in ("-r", "--requirement", "-c", "--constraint") or tok.startswith(("--requirement=", "--constraint=")):
            return f"install line {cmd!r} reads a file prepr cannot check pins in"
        if tok.startswith(("-r", "-c")) and len(tok) > 2 and not tok.startswith("--"):
            return f"install line {cmd!r} reads a file prepr cannot check pins in"
        m = re.match(r"([A-Za-z0-9_.\-]+)", tok)
        if not m or m.group(1).lower() not in STRICT_PINS:
            continue
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+(?:\[[^\]]*\])?==[A-Za-z0-9_.+\-]+", tok):
            return f"install line {cmd!r} names {m.group(1)} without an exact `==` pin (CI's version is unknowable)"
    return None


SECRET_SCAN_CMD = re.compile(
    r"python(?:3(?:\.\d+)?)?\s+(?:scripts/lint_secrets\.py|-m\s+scripts\.lint_secrets)(?P<args>(?:\s+[\w./=:-]+)*)"
)


# The only lint_secrets arguments prepr accepts (exact spelling: argparse would also accept a prefix). Both are
# whole-tree modes that never read the PR text; `--repo` (hidden) and anything else is a PlanError.
SECRET_SCAN_FLAGS = frozenset({"--all", "--staged"})


def is_secret_scan(script: str, where: str = "step") -> bool:
    """Whether this run script runs ``lint_secrets`` in its diff mode (the mode that, in CI, scans the PR text).

    Every command line mentioning ``scripts/lint_secrets`` (or ``scripts.lint_secrets``) must be ``python[3[.X]] scripts/lint_secrets.py [args]`` or
    ``python[3[.X]] -m scripts.lint_secrets [args]``; any other mention (an echo, ``./scripts/...``, a compound line)
    is a PlanError, because prepr could not tell whether CI would scan the PR text there. ``--all``/``--staged`` are
    whole-tree modes that never read the PR text.
    """
    found = False
    for group in _logical_lines(script):
        code = _code_part(" ".join(g.rstrip().rstrip("\\").strip() for g in group)).strip()
        if not re.search(r"scripts[/.]lint_secrets", code):
            continue
        m = SECRET_SCAN_CMD.fullmatch(code)
        if m is None:
            raise PlanError(f"{where}: cannot classify {code!r} as a secret-scan invocation (or as not one)")
        args = m.group("args").split()
        unknown = [a for a in args if a not in SECRET_SCAN_FLAGS]
        if unknown:
            raise PlanError(
                f"{where}: lint_secrets argument(s) {unknown} are not ones prepr reproduces (exact flags only)"
            )
        if not args:
            found = True
    return found


def has_command(script: str) -> bool:
    return any(ln.strip() and not ln.strip().startswith("#") for ln in script.splitlines())


def _resolve_env(raw: object, where: str) -> tuple[dict[str, str], dict[str, str]]:
    """(literal env, PR-sourced env) from an ``env:`` mapping; an unknown expression is a PlanError."""
    if raw is None:
        return {}, {}
    if not isinstance(raw, dict):
        raise PlanError(f"{where}: env must be a mapping")
    env: dict[str, str] = {}
    pr: dict[str, str] = {}
    for k, v in raw.items():
        text = "" if v is None else str(v)
        if "${{" not in text:
            env[str(k)] = text
            continue
        if text.strip() not in EXPRESSIONS:
            raise PlanError(f"{where}: env {k}={text!r} uses an expression prepr cannot reproduce")
        source = EXPRESSIONS[text.strip()]
        if source is not None:
            pr[str(k)] = source
    return env, pr


def _job(workflow: dict, job: str) -> tuple[dict, list[dict]]:
    try:
        spec = workflow["jobs"][job]
        steps = spec["steps"]
    except (KeyError, TypeError) as e:
        raise PlanError(f"{WORKFLOW_REL}: no job {job!r} with steps ({e})") from e
    if not isinstance(steps, list) or not all(isinstance(s, dict) for s in steps):
        raise PlanError(f"{WORKFLOW_REL}: {job} steps must be a list of mappings")
    extra = set(spec) - JOB_KEYS
    if extra:
        raise PlanError(f"job {job}: key(s) {sorted(extra)} change how its steps run; prepr cannot reproduce them")
    return spec, steps


def _uses_step(job: str, step: dict) -> str | None:
    """Validate a skippable ``uses:`` step; returns its setup-python version, if it is one."""
    uses = str(step.get("uses"))
    action = uses.split("@", 1)[0]
    name = str(step.get("name") or uses)
    if action not in SKIPPABLE_USES:
        raise PlanError(f"{job} step {name!r}: `uses: {uses}` does work prepr cannot reproduce")
    extra = set(step) - USES_STEP_KEYS
    with_ = step.get("with") or {}
    if not isinstance(with_, dict):
        raise PlanError(f"{job} step {name!r}: `with:` must be a mapping")
    extra |= {f"with.{k}" for k in set(with_) - SKIPPABLE_USES[action]}
    if extra:
        raise PlanError(f"{job} step {name!r}: key(s) {sorted(extra)} cannot be reproduced locally")
    if action == "actions/checkout" and str(with_.get("fetch-depth")) != "0":
        raise PlanError(f"{job} step {name!r}: checkout without `fetch-depth: 0` (CI would judge a shallow clone)")
    if action == "actions/setup-python":
        if "python-version" not in with_:
            raise PlanError(f"{job} step {name!r}: setup-python without `python-version`")
        return str(with_["python-version"])
    return None


def _to_step(job: str, step: dict, base_env: dict[str, str], base_pr: dict[str, str]) -> Step | None:
    name = str(step.get("name") or str(step["run"]).splitlines()[0])
    where = f"{job} step {name!r}"
    extra = set(step) - RUN_STEP_KEYS
    if extra:
        raise PlanError(f"{where}: key(s) {sorted(extra)} cannot be reproduced locally")
    shell = step.get("shell")
    if shell not in SHELLS:
        raise PlanError(f"{where}: `shell: {shell}` (only bash, or no shell key, is reproduced)")
    if "${{" in str(step["run"]):
        raise PlanError(f"{where}: a `${{{{ }}}}` expression in the run body is evaluated by CI, not by bash")
    env, pr = _resolve_env(step.get("env"), where)
    script, dropped = strip_installs(str(step["run"]), where)
    for cmd in dropped:
        problem = strict_pin_problem(cmd)
        if problem:
            raise PlanError(f"{where}: {problem}")
    if not has_command(script):
        return None
    return Step(
        job, name, script, {**base_env, **env}, {**base_pr, **pr}, dropped, SHELLS[shell], is_secret_scan(script, where)
    )


def build_plan(workflow: dict) -> Plan:
    """The lint job's run steps, then the unit-tests step that runs the mypy script."""
    if "defaults" in workflow:
        raise PlanError("workflow `defaults` would change every step's shell; prepr cannot reproduce it")
    wf_env, wf_pr = _resolve_env(workflow.get("env"), "workflow")
    steps: list[Step] = []
    skipped: list[tuple[str, str, str]] = []
    pythons: set[str] = set()

    spec, lint_steps = _job(workflow, LINT_JOB)
    job_env, job_pr = _resolve_env(spec.get("env"), f"job {LINT_JOB}")
    for step in lint_steps:
        if "run" not in step:
            if "uses" not in step:
                raise PlanError(f"{LINT_JOB} step {step.get('name')!r}: neither run: nor uses:")
            v = _uses_step(LINT_JOB, step)
            if v is not None:
                pythons.add(v)
            skipped.append((LINT_JOB, str(step.get("name") or step.get("uses")), "no run: (uses/setup)"))
            continue
        s = _to_step(LINT_JOB, step, {**wf_env, **job_env}, {**wf_pr, **job_pr})
        if s is None:
            skipped.append((LINT_JOB, str(step.get("name")), "install-only"))
        else:
            steps.append(s)
    timeout = spec.get("timeout-minutes")

    spec, unit_steps = _job(workflow, MYPY_JOB)
    mypy = [s for s in unit_steps if MYPY_SCRIPT in str(s.get("run", ""))]
    if len(mypy) != 1:
        raise PlanError(f"{MYPY_JOB}: expected exactly one step running {MYPY_SCRIPT}, found {len(mypy)}")
    for step in unit_steps[: unit_steps.index(mypy[0])]:
        if str(step.get("uses", "")).split("@", 1)[0] == "actions/setup-python":
            v = _uses_step(MYPY_JOB, step)
            if v is not None:
                pythons.add(v)
    job_env, job_pr = _resolve_env(spec.get("env"), f"job {MYPY_JOB}")
    s = _to_step(MYPY_JOB, mypy[0], {**wf_env, **job_env}, {**wf_pr, **job_pr})
    if s is None:
        raise PlanError(f"{MYPY_JOB}: the mypy step has no command after the install lines")
    steps.append(s)
    if not pythons:
        raise PlanError("no actions/setup-python step: CI's python version is unknown")
    return Plan(tuple(steps), tuple(skipped), frozenset(pythons), int(timeout) if timeout is not None else None)


def load_plan(root: Path = REPO_ROOT) -> Plan:
    return build_plan(yaml.safe_load((root / WORKFLOW_REL).read_text(encoding="utf-8")) or {})


def tool_version(tool: str, path: str) -> str | None:
    """``<tool> --version`` as the step would resolve it (its PATH); None when not found."""
    exe = shutil.which(tool, path=path)
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=60).stdout.split()
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out[1] if len(out) > 1 else None


def check_pins(
    plan: Plan,
    path: str,
    version: Callable[[str], str] | None = None,
    tool: Callable[[str, str], str | None] | None = None,
) -> tuple[list[str], list[str]]:
    """(errors, warnings) for the dropped ``==`` pins. ruff/mypy mismatches are errors: they decide what is reported."""
    version = version or metadata.version
    tool = tool or tool_version
    errors: list[str] = []
    warns: list[str] = []
    seen: set[str] = set()
    for s in plan.steps:
        for cmd in s.dropped:
            for pkg, want in PIN.findall(cmd):
                base = pkg.split("[", 1)[0].lower()
                if base in seen:
                    continue
                seen.add(base)
                if base in STRICT_PINS:
                    have = tool(base, path)
                    if have != want:
                        errors.append(
                            f"{base} {have or 'NOT FOUND'} on the steps' PATH, CI pins {want}: its findings may differ"
                        )
                    continue
                try:
                    got = version(base)
                except metadata.PackageNotFoundError:
                    warns.append(f"WARN: {base}=={want} is pinned in CI but not installed here")
                    continue
                if got != want:
                    warns.append(f"WARN: {base} {got} installed, CI pins {want}; its findings may differ from CI's")
    return errors, warns


def _norm(text: str) -> str:
    """Typed vs PR text comparison only: CRLF -> LF, trailing whitespace dropped (CI scans the raw text)."""
    return "\n".join(ln.rstrip() for ln in text.replace("\r\n", "\n").split("\n")).rstrip()


# Accepted origin forms, host ANCHORED (``https://evil.com/github.com/o/r`` is not GitHub).
ORIGIN_URL = re.compile(
    r"^(?:https?://(?:[^@/]+@)?github\.com(?::\d+)?/|git@github\.com:|ssh://(?:[^@/]+@)?github\.com(?::\d+)?/)"
    r"(?P<owner>[^/]+)/[^/]+?(?:\.git)?/?$"
)
ORIGIN_FORMS = (
    "https://github.com/OWNER/REPO(.git), git@github.com:OWNER/REPO(.git), ssh://git@github.com[:PORT]/OWNER/REPO"
)


def repo_owner(root: Path = REPO_ROOT) -> str:
    """The GitHub owner of the remote named ``origin``. Refuses (fail-closed) when it cannot be read."""
    url = _git(root, "remote", "get-url", "origin").stdout.strip()
    m = ORIGIN_URL.match(url)
    if m is None:
        raise Refusal(
            f"cannot read the repo owner from origin {url or '(no remote named origin)'!r}; accepted forms: "
            f"{ORIGIN_FORMS}. Fix it with `git remote set-url origin <url>` (or name the GitHub remote origin)"
        )
    return m.group("owner")


def current_branch(root: Path) -> str:
    """The checked-out branch's full name without ``refs/heads/`` ('' when detached). Not ``--abbrev-ref``/``--short``,
    which answer ``heads/x`` when a tag shares the branch's name."""
    ref = _git(root, "symbolic-ref", "-q", "HEAD").stdout.strip()
    return ref.removeprefix("refs/heads/") if ref.startswith("refs/heads/") else ""


def resolve_pr_text(
    title: str | None,
    body: str | None,
    gh: Callable[[], dict | None],
    branch: str,
    owner: Callable[[], str | None],
) -> tuple[str | None, str | None, bool]:
    """(title, body, typed): the text CI will see, and whether it was typed by hand rather than read from the PR.

    An OPEN PR's text is what CI reads, wherever its head is (the text does not depend on the head commit); the PR
    must be THIS branch's (``headRefName`` == ``branch``), else refused. A typed ``--title``/``--body`` that differs
    from an OPEN PR's text is refused, never silently preferred. With no OPEN PR, typed text is used and taints.
    """
    pr = gh()
    open_pr = pr if pr and pr.get("state") == "OPEN" else None
    if open_pr is None:
        typed = title is not None or body is not None
        if title is None:
            return None, body, typed
        return title, body if body is not None else "", True
    if open_pr.get("isCrossRepository") is not False:
        raise Refusal(f"PR #{open_pr.get('number')} comes from a fork (or gh did not say): not this repo's branch")
    login = str((open_pr.get("headRepositoryOwner") or {}).get("login") or "")
    expected = owner()
    if expected is None or login.lower() != expected.lower():
        raise Refusal(
            f"PR #{open_pr.get('number')}'s head is owned by {login or '?'!r}, origin by {expected or '?'!r}: "
            "not this repo's branch"
        )
    if open_pr.get("headRefName") != branch:
        raise Refusal(
            f"gh resolved PR #{open_pr.get('number')} for branch {open_pr.get('headRefName')!r}, "
            f"but HEAD is on {branch or '(detached)'!r}: not this branch's PR"
        )
    pr_title, pr_body = str(open_pr.get("title") or ""), str(open_pr.get("body") or "")
    which = [
        n
        for n, v, w in (("--title", title, pr_title), ("--body", body, pr_body))
        if v is not None and _norm(v) != _norm(w)
    ]
    if which:
        raise Refusal(
            f"{' and '.join(which)} differ from open PR #{open_pr.get('number')}'s text, which CI will read: "
            "drop them, or edit the PR"
        )
    return pr_title, pr_body, False


def scan_pr_text(root: Path, base: str, head: str, text: str) -> int:
    """lint_secrets' own pull_request computation, ``scan_range(..., pr_text=...)``, called directly (no faked event).

    CI's Secret scan step scans the PR title and body only under ``pull_request``; prepr's steps run in local mode,
    so without this a secret in the PR text would pass here and fail there. 0 = clean.
    """
    spec = importlib.util.spec_from_file_location("_prepr_lint_secrets", root / "scripts/lint_secrets.py")
    if spec is None or spec.loader is None:
        raise Refusal("cannot load scripts/lint_secrets.py for the PR-text secret scan")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # any import-time failure: reported as a refusal, never a traceback
        raise Refusal(
            f"cannot load scripts/lint_secrets.py for the PR-text secret scan ({type(e).__name__}: {e})"
        ) from e
    missing = [n for n in ("GitUnavailable", "ConfigError", "scan_range", "_report") if not hasattr(mod, n)]
    if missing:
        raise Refusal(f"scripts/lint_secrets.py lacks {missing}: prepr's PR-text scan no longer matches it")
    try:
        scan = mod.scan_range(root, base, head, pr_text=text)
    except (mod.GitUnavailable, mod.ConfigError) as e:
        raise Refusal(f"the PR-text secret scan cannot run ({type(e).__name__}: {e})") from e
    return int(mod._report([scan], "branch range + PR title/body, as CI's pull_request run"))


def gh_pr_view(root: Path = REPO_ROOT) -> dict | None:
    if shutil.which("gh") is None:
        print("INFO: gh not installed; no PR lookup", file=sys.stderr)
        return None
    try:
        r = subprocess.run(
            ["gh", "pr", "view", "--json", "number,state,headRefName,isCrossRepository,headRepositoryOwner,title,body"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"INFO: gh pr view failed ({e})", file=sys.stderr)
        return None
    if r.returncode != 0:
        print(f"INFO: gh pr view: {r.stderr.strip() or 'no PR for this branch'}", file=sys.stderr)
        return None
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def step_env(step: Step, pr_text: dict[str, str], base: dict[str, str] | None = None) -> dict[str, str]:
    env = {
        k: v
        for k, v in (os.environ if base is None else base).items()
        if not k.startswith((*SCRUB_PREFIXES, SCRUB_MAXIM_PREFIX)) and k not in SCRUB_VARS
    }
    env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}"
    env.update(step.env)
    for var, source in step.pr_env.items():
        env[var] = pr_text[source]
    return env


def run_step(step: Step, env: dict[str, str], log: Path, root: Path, verbose: bool) -> tuple[bool, float]:
    script = log.with_suffix(".sh")
    script.write_text(step.script, encoding="utf-8")
    t0 = time.monotonic()
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            [*step.shell, str(script)],
            cwd=root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            fh.write(line)
            if verbose:
                sys.stdout.write(line)
        rc = proc.wait()
    return rc == 0, time.monotonic() - t0


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)


def git_preflight(root: Path, *, fetch: bool, allow_behind: bool, allow_dirty: bool) -> tuple[str, str, list[str]]:
    """(HEAD, merge-base, taints). Raises :class:`Refusal` when the local tree cannot stand for CI's."""
    taints: list[str] = []
    shallow = _git(root, "rev-parse", "--is-shallow-repository")
    if shallow.returncode != 0 or shallow.stdout.strip() != "false":
        raise Refusal(
            f"shallow or unreadable clone ({shallow.stdout.strip() or shallow.stderr.strip()}): CI has full history"
        )
    if fetch:
        r = _git(root, "fetch", "-q", "origin", "main")
        if r.returncode != 0:
            raise Refusal(f"git fetch origin main failed ({r.stderr.strip()}); fix it, or pass --no-fetch (taints)")
    else:
        taints.append("--no-fetch (origin/main may be stale)")
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    mb = _git(root, "merge-base", "origin/main", "HEAD")
    if mb.returncode != 0 or not mb.stdout.strip() or not head:
        raise Refusal(f"no merge-base with origin/main ({mb.stderr.strip()}): the diff-scoped lints would skip")
    behind = _git(root, "rev-list", "--count", "HEAD..origin/main")
    n_behind = int(behind.stdout.strip()) if behind.returncode == 0 and behind.stdout.strip().isdigit() else -1
    if n_behind != 0:
        msg = f"HEAD is {n_behind if n_behind > 0 else 'an unknown number of'} commit(s) behind origin/main"
        if not allow_behind:
            raise Refusal(f"{msg}: CI tests the merge with main; merge or rebase onto origin/main first")
        taints.append(f"--allow-behind ({msg})")
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    dirty = [ln for ln in status.stdout.splitlines() if ln.strip()]
    if status.returncode != 0 or dirty:
        what = f"{len(dirty)} uncommitted/untracked path(s)" if status.returncode == 0 else "git status failed"
        if not allow_dirty:
            raise Refusal(f"{what}: CI sees only commits; commit (or pass --allow-dirty, which taints)")
        taints.append(f"--allow-dirty ({what})")
    ign = _git(root, "ls-files", "-o", "-i", "--exclude-standard", "--", *IGNORED_PY_ROOTS)
    ignored_py = [ln for ln in ign.stdout.splitlines() if ln.endswith(".py")]
    if ign.returncode != 0 or ignored_py:
        what = (
            f"{len(ignored_py)} gitignored *.py file(s) under {'/, '.join(IGNORED_PY_ROOTS)}/ (CI has none; "
            "lint_orphan_modules globs these roots)"
            if ign.returncode == 0
            else "git ls-files --ignored failed"
        )
        if not allow_dirty:
            raise Refusal(f"{what}: move or delete them (or pass --allow-dirty, which taints)")
        taints.append(f"--allow-dirty ({what})")
    return head, mb.stdout.strip(), taints


def platform_note() -> str | None:
    if sys.platform.startswith("linux"):
        return None
    bash = subprocess.run(["bash", "-c", "echo $BASH_VERSION"], capture_output=True, text=True).stdout.strip()
    return f"platform {sys.platform}, bash {bash or '?'} (CI: ubuntu-latest, GNU tools)"


def main(argv: list[str] | None = None, root: Path = REPO_ROOT) -> int:
    ap = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n", 1)[0],
        epilog=EXIT_CODES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--title", help="the PR title: refused if it differs from this branch's OPEN PR; taints only with no OPEN PR"
    )
    ap.add_argument(
        "--body", help="the PR body: refused if it differs from this branch's OPEN PR; taints only with no OPEN PR"
    )
    ap.add_argument("--no-pr", action="store_true", help="skip only the steps reading the PR title/body (taints)")
    ap.add_argument("--no-fetch", action="store_true", help="do not `git fetch origin main` first (taints)")
    ap.add_argument("--allow-dirty", action="store_true", help="run with uncommitted/untracked files (taints)")
    ap.add_argument("--allow-behind", action="store_true", help="run though HEAD lacks origin/main (taints)")
    ap.add_argument("--allow-pin-mismatch", action="store_true", help="run with other ruff/mypy versions (taints)")
    ap.add_argument("--allow-python-mismatch", action="store_true", help="run under another python (taints)")
    ap.add_argument("--fail-fast", action="store_true", help="stop at the first failing step")
    ap.add_argument("--verbose", "-v", action="store_true", help="stream every step's output")
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    args = ap.parse_args(argv)

    try:
        plan = load_plan(root)
    except (PlanError, OSError, yaml.YAMLError) as e:
        print(f"ERROR: cannot build the plan from {WORKFLOW_REL}: {e}", file=sys.stderr)
        return 2
    if args.list:
        for job, name, why in plan.skipped:
            print(f"SKIP  [{job}] {name} ({why})")
        for s in plan.steps:
            extra = f" (dropped: {'; '.join(s.dropped)})" if s.dropped else ""
            print(f"RUN   [{s.job}] {s.name} [{' '.join(s.shell)}]{extra}")
        return 0

    try:
        if args.no_pr and (args.title is not None or args.body is not None):
            raise Refusal("--no-pr contradicts --title/--body")
        head, mb, taints = git_preflight(
            root, fetch=not args.no_fetch, allow_behind=args.allow_behind, allow_dirty=args.allow_dirty
        )
        here = f"{sys.version_info.major}.{sys.version_info.minor}"
        if plan.python_versions != {here}:
            msg = f"python {here} here, CI sets up {sorted(plan.python_versions)}"
            if not args.allow_python_mismatch:
                raise Refusal(f"{msg} (pass --allow-python-mismatch, which taints)")
            taints.append(f"--allow-python-mismatch ({msg})")
        steps = list(plan.steps)
        no_pr_skipped: list[str] = []
        if args.no_pr:
            no_pr_skipped = [s.name for s in steps if s.pr_env]
            steps = [s for s in steps if not s.pr_env]
            title, body = None, None
            taints.append(f"--no-pr ({len(no_pr_skipped)} step(s) reading the PR text skipped; no PR-text secret scan)")
        else:
            title, body, typed = resolve_pr_text(
                args.title, args.body, lambda: gh_pr_view(root), current_branch(root), lambda: repo_owner(root)
            )
            if typed:
                taints.append("PR text typed by hand (no open PR to read it from)")
            if "title" in plan.needs and title is None:
                users = [s.name for s in steps if "title" in s.pr_env.values()]
                raise Refusal(
                    f"no OPEN PR found for this branch; pass --title '<the PR title>' or --no-pr "
                    f"(needed by: {'; '.join(users)})"
                )
        path = step_env(steps[0], {"title": "", "body": ""})["PATH"] if steps else os.environ.get("PATH", "")
        pin_errors, pin_warns = check_pins(plan, path)
        if pin_errors:
            if not args.allow_pin_mismatch:
                raise Refusal("; ".join(pin_errors) + " (pass --allow-pin-mismatch, which taints)")
            taints.append(f"--allow-pin-mismatch ({'; '.join(pin_errors)})")
        scan_result: tuple[bool, float] | None = None
        if title is not None and any(s.secret_scan for s in steps):
            s0 = time.monotonic()
            scan_ok = scan_pr_text(root, mb, head, f"{title}\n{body or ''}") == 0
            scan_result = (scan_ok, time.monotonic() - s0)
    except Refusal as e:
        print(f"ERROR: prepr refuses: {e}", file=sys.stderr)
        print("prepr: REFUSED (exit 2)")
        return 2
    note = platform_note()

    print(f"prepr: HEAD {head[:12]}, base (merge-base with origin/main) {mb[:12]}")
    if title is not None:
        print(f"prepr: PR title {title!r}")
    for w in pin_warns:
        print(w)
    for t in taints:
        print(f"TAINT: {t}")
    if note:
        print(f"NOTE: {note}")
    for name in no_pr_skipped:
        print(f"SKIP  (no-pr) {name}")
    pr_text = {"title": title or "", "body": body or ""}

    log_dir = Path(tempfile.mkdtemp(prefix="prepr-"))
    failed: list[str] = []
    ran = 0
    lint_secs = 0.0
    t0 = time.monotonic()
    if scan_result is not None:
        print(f"{'PASS' if scan_result[0] else 'FAIL'}  {scan_result[1]:6.1f}s  [prepr] {PR_SECRET_SCAN}", flush=True)
        if not scan_result[0]:
            failed.append(PR_SECRET_SCAN)
    for i, s in enumerate(steps, 1):
        log = log_dir / f"{i:02d}.log"
        ok, secs = run_step(s, step_env(s, pr_text), log, root, args.verbose)
        ran += 1
        lint_secs += secs if s.job == LINT_JOB else 0.0
        print(f"{'PASS' if ok else 'FAIL'}  {secs:6.1f}s  [{s.job}] {s.name}", flush=True)
        if not ok:
            failed.append(s.name)
            if not args.verbose:
                tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
                print("\n".join(f"      | {ln}" for ln in tail))
            if args.fail_fast:
                break
    total = time.monotonic() - t0
    if plan.lint_timeout_minutes is not None and lint_secs > plan.lint_timeout_minutes * 60:
        print(f"WARN: the lint steps took {lint_secs:.0f}s here; CI kills the job at {plan.lint_timeout_minutes} min")
    print(f"prepr: {ran - len(failed)}/{len(steps)} steps passed ({ran} run) in {total:.0f}s; logs in {log_dir}")
    if note:
        print(f"NOTE: {note}")
    suffix = f" (tainted: {'; '.join(taints)})" if taints else ""
    if failed:
        print(f"prepr: FAIL ({len(failed)}): {'; '.join(failed)}{suffix}")
        return 1
    if taints:
        print(f"prepr: TAINTED-PASS{suffix}")
        return 3
    print("prepr: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
