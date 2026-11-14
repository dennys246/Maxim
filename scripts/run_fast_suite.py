#!/usr/bin/env python3
"""Run the fast suite as N deterministic shards in one job, then combine coverage and junit (#1645).

    python scripts/run_fast_suite.py --shards 4 -- <the suite's pytest arguments, verbatim>
    python scripts/run_fast_suite.py --shards 1 [--no-cov] -- <arguments>   # serial: the same order, no partition
    python3 scripts/run_fast_suite.py --check-map                          # the shard map's diff rules (lint job)

Each shard is one ``python -m pytest`` process that COLLECTS THE WHOLE SUITE and keeps its own files
(``tests/_shard_plugin.py``), so every run executes the same tests in the same order in the same process, which is the
property serial coverage is reproducible under. The ratchet's rules are unchanged; this script only has to be honest.
CI's `unit-tests` job runs it with :data:`CI_SHARDS` shards; the step's exact run text is :data:`CI_STEP_RUN`, which
``lint_coverage.ci_step_rules``, ``lint_release_gates`` rule 9 and ``tests/unit/test_ci_workflow_shape.py`` pin.

Fail-closed rules (the #1645 approach note, v2; every one exits non-zero):
- refuses to start if ``coverage.json``, ``fast-suite.xml``, any ``.coverage*`` or any shard output exists (the step's
  clean removes them; deleting them here would be invisible to the lint);
- forwards the pytest arguments VERBATIM and changes only an allowlist: ``--junitxml=fast-suite.xml`` becomes the
  shard's own file, ``--cov-report=json:coverage.json`` becomes ``--cov-report=``, and it adds
  ``-p tests._shard_plugin --shard k/N --shard-record ...`` and ``-p no:cacheprovider``. An argument that would
  conflict (``--shard*``, ``-n``, ``--dist``, ``--cov-append``, ``--cov-config``, another ``--junitxml`` or json
  report) is refused. ``--no-cov`` drops every ``--cov*`` argument (suite_at_commit.sh's quick mode; never CI's);
- each shard's environment names its own coverage data file (``.coverage.shard<kk>``) and drops every other
  coverage.py and pytest-cov environment name (``lint_coverage.config_rules`` scans this file for them);
- any shard's non-zero or negative return code fails the run (exit 5, "no tests", included); a shard stopped by
  ``-x`` is named as such, and a failed run skips the junit merge and the coverage combine;
- reconciliation is by SETS: every shard saw the same ``whole`` item set and the same modules; the kept sets are
  pairwise disjoint and their union is ``whole``; no shard is empty; the merged junit's cases equal that union, with
  no duplicate (classname, name);
- coverage: exactly one data file per shard, all with arc data, combined by name; only the combine writes
  ``coverage.json`` and only the merge writes ``fast-suite.xml``.

``--shards 1`` runs ``python -m pytest`` once with the arguments verbatim (plus ``-p no:cacheprovider``): no plugin, no
partition, the serial order. The rig keeps it until ``RIG_KNOWN_FAILURES`` is re-calibrated under sharding.

The map (``scripts/suite_shards.json``): duplicate keys, an index outside ``0..n-1`` or a different ``n`` fail (at
every shard's start and in ``--check-map``). ``--check-map`` also fails a test module renamed (``git diff -M``
against the base) whose new path is unmapped while its old key exists ("carry the entry"), and prints a key with no
file as a NOTE.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
JUNIT = "fast-suite.xml"
JSON_REPORT = "--cov-report=json:coverage.json"
REFUSED_PREFIXES = ("--shard", "-n", "--numprocesses", "--dist", "--cov-append", "--cov-config")
SHARD_TIMEOUT_S = 50 * 60  # below the job's timeout-minutes (60), so a hung shard names itself first
MAP_REL = "scripts/suite_shards.json"
ENV_DROP = ("COVERAGE_", "COV_CORE_")  # every coverage.py / pytest-cov environment name a shard must not inherit
# pytest's knobs that add options, load plugins or stop plugin autoload (which drops pytest-cov): a shard runs only
# the step's arguments (#1645 review; lint_coverage.config_rules also bans naming them in a workflow).
PYTEST_ENV_DROP = frozenset({"PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_DISABLE_PLUGIN_AUTOLOAD"})

# CI's measuring step, as one text (#1645 switch). Its consumers compare the whitespace-flattened run (flat_run).
CI_SHARDS = 4
CI_CLEAN = (
    "rm -f .coverage .coverage.* coverage.json coverage.xml fast-suite.xml fast-suite.shard*.xml shard*.collect.json"
)
CI_COV_ARGS = "--cov=src/maxim --cov-branch --cov-report=json:coverage.json --cov-report="
CI_PYTEST_ARGS = (
    'tests/ -x -q -m "not slow" --tb=short --ignore=tests/integration/test_memory_hub.py '
    f"--require-extras=console,sign --junitxml={JUNIT} {CI_COV_ARGS}"
)
CI_STEP_RUN = (
    f"{CI_CLEAN}\n"
    f"SUITE='\"$PY\" scripts/run_fast_suite.py --shards {CI_SHARDS} -- {CI_PYTEST_ARGS}'\n"
    'bash scripts/ci_netns.sh "$SUITE"\n'
)


def flat_run(text: str) -> str:
    """A step's run text with line continuations joined and whitespace collapsed (how its pins compare it)."""
    return re.sub(r"\s+", " ", text.replace("\\\n", " ")).strip()


class RunnerError(Exception):
    """A fail-closed refusal: printed, exit 2."""


class MapError(ValueError):
    """The shard map is malformed (the plugin turns it into a pytest usage error)."""


def parse_map(text: str, n: int) -> dict[str, int]:
    """``suite_shards.json``'s ``shards`` for ``n``: duplicate keys, a non-index value or another ``n`` raise."""

    def no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        keys = [k for k, _ in pairs]
        dup = sorted({k for k in keys if keys.count(k) > 1})
        if dup:
            raise MapError(f"{MAP_REL}: duplicate keys {dup}")
        return dict(pairs)

    try:
        data = json.loads(text, object_pairs_hook=no_duplicates)
    except json.JSONDecodeError as e:
        raise MapError(f"{MAP_REL}: not JSON ({e})") from e
    if data.get("_format_version") != "1.0" or not isinstance(data.get("shards"), dict):
        raise MapError(f"{MAP_REL}: needs _format_version 1.0 and a `shards` object")
    if data.get("n") != n:
        raise MapError(f"{MAP_REL}: balanced for n={data.get('n')!r}, this run has N={n}")
    out: dict[str, int] = {}
    for key, value in data["shards"].items():
        if type(value) is not int or not 0 <= value < n:
            raise MapError(f"{MAP_REL}: {key!r} -> {value!r} is not a shard index in 0..{n - 1}")
        out[key] = value
    return out


def without_cov(args: list[str]) -> list[str]:
    """``--no-cov``: the arguments without any ``--cov*`` (the quick local run; it is never the measurement)."""
    return [a for a in args if not a.startswith("--cov")]


def _refuse_conflicts(args: list[str]) -> None:
    junits = [a for a in args if a.startswith("--junitxml")]
    if junits != [f"--junitxml={JUNIT}"]:
        raise RunnerError(f"needs exactly one `--junitxml={JUNIT}` (found {junits})")
    jsons = [a for a in args if a.startswith("--cov-report=json")]
    if jsons and jsons != [JSON_REPORT]:
        raise RunnerError(f"the only json report allowed is `{JSON_REPORT}` (found {jsons})")
    for a in args:
        if a.startswith(REFUSED_PREFIXES):
            raise RunnerError(f"argument {a!r} conflicts with sharding")
    for i, a in enumerate(args):
        if a == "-p" and i + 1 < len(args) and args[i + 1] in ("no:cov", "no:pytest_cov", "tests._shard_plugin"):
            raise RunnerError(f"argument `-p {args[i + 1]}` conflicts with sharding")


def _no_cacheprovider(args: list[str]) -> list[str]:
    has = any(a == "-p" and b == "no:cacheprovider" for a, b in zip(args, args[1:]))
    return [] if has else ["-p", "no:cacheprovider"]


def serial_argv(args: list[str]) -> list[str]:
    """``--shards 1``: the arguments verbatim, plus ``-p no:cacheprovider`` unless they carry it."""
    _refuse_conflicts(args)
    return [*args, *_no_cacheprovider(args)]


def shard_argv(args: list[str], k: int, n: int) -> list[str]:
    """The step's pytest arguments for shard k: verbatim, with only the allowlisted replacements and additions."""
    _refuse_conflicts(args)
    out: list[str] = []
    for a in args:
        if a == f"--junitxml={JUNIT}":
            out.append(f"--junitxml=fast-suite.shard{k:02d}.xml")
        elif a == JSON_REPORT:
            out.append("--cov-report=")
        else:
            out.append(a)
    return [
        *out,
        "-p",
        "tests._shard_plugin",
        f"--shard={k}/{n}",
        f"--shard-record=shard{k:02d}.collect.json",
        *_no_cacheprovider(args),
    ]


def scrubbed_env() -> dict[str, str]:
    """The environment without coverage's names (ENV_DROP) or pytest's selection/plugin knobs (PYTEST_ENV_DROP)."""
    return {key: v for key, v in os.environ.items() if not key.startswith(ENV_DROP) and key not in PYTEST_ENV_DROP}


def shard_env(k: int) -> dict[str, str]:
    env = scrubbed_env()
    env["COVERAGE_FILE"] = f".coverage.shard{k:02d}"  # the one allowlisted name (lint_coverage.config_rules)
    return env


def refuse_leftovers(cwd: Path) -> None:
    left = [p.name for p in cwd.iterdir() if p.name in ("coverage.json", JUNIT) or p.name.startswith(".coverage")]
    left += [p.name for p in cwd.glob("fast-suite.shard*.xml")] + [p.name for p in cwd.glob("shard*.collect.json")]
    if left:
        raise RunnerError(
            f"refusing to start: outputs from an earlier run exist ({sorted(left)}); the step cleans them"
        )


def reconcile(records: list[dict], n: int) -> list[str]:
    """Set reconciliation of the shards' collection records; returns problems (empty = sound)."""
    problems: list[str] = []
    whole = set(records[0]["whole"])
    if len(whole) != len(records[0]["whole"]):
        problems.append("shard 0's collection has duplicate nodeids")
    for r in records:
        if r["n"] != n:
            problems.append(f"shard {r['shard']} ran with n={r['n']}")
        if set(r["whole"]) != whole:
            problems.append(
                f"shard {r['shard']} collected a different item set from shard 0 "
                f"({len(set(r['whole']) ^ whole)} differ)"
            )
        if r["modules"] != records[0]["modules"]:
            problems.append(f"shard {r['shard']} visited a different module set from shard 0")
        if r.get("collect_outcomes") != records[0].get("collect_outcomes"):
            problems.append(f"shard {r['shard']} had different collection-time skips/errors from shard 0")
        if not r["selected"]:
            problems.append(f"shard {r['shard']} kept no tests (an empty shard)")
    seen: set[str] = set()
    for r in records:
        mine = set(r["selected"])
        if mine & seen:
            problems.append(f"shard {r['shard']} overlaps an earlier shard on {len(mine & seen)} items")
        seen |= mine
    if seen != whole:
        problems.append(f"the shards' union differs from the collection ({len(seen ^ whole)} items)")
    return problems


def _case_key(nodeid: str) -> tuple[str, str]:
    from _pytest.junitxml import mangle_test_address

    parts = mangle_test_address(nodeid)
    return ".".join(parts[:-1]), parts[-1]


def merge_junit(cwd: Path, n: int, expected: set[str], collect_outcomes: dict[str, str]) -> list[str]:
    """Writes fast-suite.xml from the shards' files; returns problems.

    Item cases (``expected``) must appear exactly once across the shards. A collection-time case (a collector skipped
    or failed at collection, ``collect_outcomes``) appears in EVERY shard, because each collects the whole suite; it
    must appear exactly once in each, and the merge keeps shard 0's copy only. Anything else fails."""
    problems: list[str] = []
    want = {_case_key(nodeid) for nodeid in expected}
    collect_keys = {_case_key(nodeid) for nodeid in collect_outcomes}
    if want & collect_keys:
        problems.append(f"{len(want & collect_keys)} case(s) are both an item and a collection-time report")
    root = ET.Element("testsuites")
    got: list[tuple[str, str]] = []
    for k in range(n):
        tree = ET.parse(cwd / f"fast-suite.shard{k:02d}.xml")
        shard_collect: list[tuple[str, str]] = []
        for suite in tree.getroot().iter("testsuite"):
            for tc in list(suite.iter("testcase")):
                case = (tc.get("classname", ""), tc.get("name", ""))
                if case in collect_keys:
                    shard_collect.append(case)
                    if k:
                        suite.remove(tc)  # the same collection-time case, already kept from shard 0
                else:
                    got.append(case)
            root.append(suite)
        if sorted(shard_collect) != sorted(collect_keys):
            problems.append(
                f"shard {k}'s junit has collection-time cases {sorted(shard_collect)}, "
                f"expected each of {sorted(collect_keys)} once"
            )
    if len(got) != len(set(got)):
        problems.append(f"the merged junit has {len(got) - len(set(got))} duplicate (classname, name) cases")
    if set(got) != want:
        problems.append(
            f"the merged junit's cases differ from the shards' kept items "
            f"(missing {len(want - set(got))}, extra {len(set(got) - want)})"
        )
    ET.ElementTree(root).write(cwd / JUNIT, encoding="utf-8", xml_declaration=True)
    return problems


def combine_coverage(cwd: Path, n: int, py: str) -> list[str]:
    problems: list[str] = []
    files: list[str] = []
    for k in range(n):
        mine = sorted(p.name for p in cwd.glob(f".coverage.shard{k:02d}*"))
        if len(mine) != 1:
            problems.append(f"shard {k}: expected exactly one coverage data file, found {mine}")
        files += mine
    if problems:
        return problems
    import coverage

    for f in files:
        data = coverage.CoverageData(basename=str(cwd / f))
        data.read()
        if not data.has_arcs():
            problems.append(f"{f}: no arc (branch) data")
    if problems:
        return problems
    for cmd in ([py, "-m", "coverage", "combine", *files], [py, "-m", "coverage", "json", "-q", "-o", "coverage.json"]):
        rc = subprocess.run(cmd, cwd=cwd, env=scrubbed_env()).returncode
        if rc != 0:
            problems.append(f"`{' '.join(cmd)}` exited {rc}")
    return problems


def _launch(cwd: Path, py: str, argvs: list[list[str]]) -> tuple[list[int | None], list[float]]:
    """Starts the shards together; returns each one's return code (None: killed at the timeout) and finish time."""
    n = len(argvs)
    procs, logs, started = [], [], time.monotonic()
    for k in range(n):
        cmd = [py, "-m", "pytest", *argvs[k]]
        print(f"run_fast_suite: shard {k}/{n}: {' '.join(cmd)}", flush=True)
        log = open(cwd / f"shard{k:02d}.log", "wb")  # noqa: SIM115 - closed below
        logs.append(log)
        procs.append(subprocess.Popen(cmd, cwd=cwd, env=shard_env(k), stdout=log, stderr=subprocess.STDOUT))
    rcs: list[int | None] = [None] * n
    times: list[float] = [0.0] * n
    pending = set(range(n))
    while pending:  # poll, so each shard's own finish time is recorded (not the order it was waited on)
        for k in sorted(pending):
            rc = procs[k].poll()
            if rc is not None:
                rcs[k], times[k] = rc, time.monotonic() - started
                pending.discard(k)
        if pending and time.monotonic() - started > SHARD_TIMEOUT_S:
            for k in pending:
                procs[k].kill()
                procs[k].wait()
                times[k] = time.monotonic() - started  # rcs[k] stays None: reported as a timeout
            pending.clear()
        if pending:
            time.sleep(1.0)
    for log in logs:
        log.close()
    order = sorted(range(n), key=lambda k: (rcs[k] == 0, k))  # a failing shard's output first
    for k in order:
        print(f"\n===== shard {k}/{n}: rc={rcs[k]} ({times[k] / 60:.1f} min to finish) =====")
        sys.stdout.write((cwd / f"shard{k:02d}.log").read_text(encoding="utf-8", errors="replace")[-20000:])
    return rcs, times


def rc_problems(rcs: list[int | None], exitfirst: bool) -> list[str]:
    problems: list[str] = []
    for k, rc in enumerate(rcs):
        if rc is None:
            problems.append(f"shard {k} timed out after {SHARD_TIMEOUT_S // 60} min and was killed")
        elif rc == 1 and exitfirst:
            problems.append(f"shard {k} stopped at its first failure (-x)")
        elif rc != 0:
            why = " (pytest's 'no tests collected')" if rc == 5 else " (killed by a signal)" if rc < 0 else ""
            problems.append(f"shard {k} exited {rc}{why}")
    return problems


def results(cwd: Path, n: int, py: str, rcs: list[int | None], args: list[str]) -> list[str]:
    """Every check after the shards ran: return codes, the set reconciliation, the junit merge, the combine."""
    problems = rc_problems(rcs, exitfirst="-x" in args or "--exitfirst" in args)
    records = []
    for k in range(n):
        path = cwd / f"shard{k:02d}.collect.json"
        if path.exists():
            records.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            problems.append(f"shard {k} wrote no collection record")
    if len(records) == n:
        problems += reconcile(records, n)
    if problems:
        return problems + ["the junit merge and the coverage combine did not run (a shard failed or did not reconcile)"]
    if not all((cwd / f"fast-suite.shard{k:02d}.xml").exists() for k in range(n)):
        return ["a shard wrote no junit"]
    expected = set().union(*(set(r["selected"]) for r in records))
    problems += merge_junit(cwd, n, expected, records[0].get("collect_outcomes") or {})
    if any(a.startswith("--cov=") for a in args):
        problems += combine_coverage(cwd, n, py)
    return problems


def _plan(n: int, args: list[str], cwd: Path) -> list[list[str]]:
    """Each process's pytest arguments (one for ``--shards 1``); raises :class:`RunnerError` before anything runs."""
    if n < 1:
        raise RunnerError("--shards must be >= 1")
    if not args or args[0] != "tests/":
        raise RunnerError("the pytest arguments must start with `tests/` (the suite's own selection)")
    refuse_leftovers(cwd)
    if n == 1:
        return [serial_argv(args)]
    return [shard_argv(args, k, n) for k in range(n)]


def run(n: int, args: list[str], cwd: Path, py: str) -> int:
    try:
        argvs = _plan(n, args, cwd)
    except RunnerError as e:
        print(f"run_fast_suite: {e}", file=sys.stderr)
        return 2
    if n == 1:
        cmd = [py, "-m", "pytest", *argvs[0]]
        print(f"run_fast_suite: serial (--shards 1): {' '.join(cmd)}", flush=True)
        return subprocess.run(cmd, cwd=cwd, env=scrubbed_env()).returncode
    rcs, times = _launch(cwd, py, argvs)
    problems = results(cwd, n, py, rcs, args)
    print(f"\nrun_fast_suite: per-shard minutes {[round(t / 60, 1) for t in times]}")
    if problems:
        print("run_fast_suite: FAILED\n  - " + "\n  - ".join(problems), file=sys.stderr)
        return 1
    print(f"run_fast_suite: {n} shards passed and reconcile")
    return 0


def map_problems(root: Path, renames: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
    """(failures, notes) for the shard map at ``root``, given the diff's test-module renames as (new, old)."""
    try:
        mapping = parse_map((root / MAP_REL).read_text(encoding="utf-8"), CI_SHARDS)
    except (OSError, MapError) as e:
        return [str(e)], []
    notes = [
        f"{MAP_REL}: {key!r} names no file (a stale key; drop it at the next rebalance)"
        for key in mapping
        if not (root / key).is_file()
    ]
    failures = [
        f'{old} was renamed to {new}, whose path is unmapped: carry the entry ("{new}": {mapping[old]}) in '
        f"{MAP_REL}, or the module changes shard"
        for new, old in renames
        if old in mapping and new not in mapping
    ]
    return failures, notes


def check_map(root: Path) -> int:
    """``--check-map`` (the lint job): the map parses for CI's N, and no renamed test module drops its entry."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _lint_git import GitUnavailable, base_ref, changed_files, must_not_skip

    try:
        renames = [(new, old) for new, old in changed_files(root, base_ref(root), "tests") if old and old != new]
    except GitUnavailable as e:
        if must_not_skip(f"no base for the shard map's rename rule: {e}"):
            return 2
        print(f"run_fast_suite --check-map: no base ({e}); the rename rule is skipped locally")
        renames = []
    failures, notes = map_problems(root, renames)
    for note in notes:
        print(f"NOTE: {note}")
    for f in failures:
        print(f"run_fast_suite --check-map: {f}", file=sys.stderr)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shards", type=int)
    ap.add_argument("--no-cov", action="store_true", help="drop every --cov* argument (a quick local run)")
    ap.add_argument("--check-map", action="store_true", help="check the shard map's diff rules, run nothing")
    ap.add_argument("pytest_args", nargs=argparse.REMAINDER)
    ns = ap.parse_args(argv)
    if ns.check_map:
        if ns.shards is not None or ns.no_cov or ns.pytest_args:
            ap.error("--check-map takes no other arguments")
        return check_map(REPO)
    if ns.shards is None:
        ap.error("--shards is required")
    args = ns.pytest_args[1:] if ns.pytest_args[:1] == ["--"] else ns.pytest_args
    if ns.no_cov:
        args = without_cov(args)
    return run(ns.shards, args, Path.cwd(), sys.executable)


if __name__ == "__main__":
    sys.exit(main())
