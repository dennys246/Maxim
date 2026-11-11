#!/usr/bin/env python3
"""Run the fast suite as N deterministic shards in one job, then combine coverage and junit (#1645).

    python scripts/run_fast_suite.py --shards 4 -- <the suite's pytest arguments, verbatim>

Each shard is one ``python -m pytest`` process that COLLECTS THE WHOLE SUITE and keeps its own files
(``tests/_shard_plugin.py``), so every run executes the same tests in the same order in the same process, which is the
property serial coverage is reproducible under. The ratchet's rules are unchanged; this script only has to be honest.

Fail-closed rules (the #1645 approach note, v2; every one exits non-zero):
- refuses to start if ``coverage.json``, ``fast-suite.xml`` or any ``.coverage*`` exists (the step's clean removes
  them; deleting them here would be invisible to the lint);
- forwards the pytest arguments VERBATIM and changes only an allowlist: ``--junitxml=fast-suite.xml`` becomes the
  shard's own file, ``--cov-report=json:coverage.json`` becomes ``--cov-report=``, and it adds
  ``-p tests._shard_plugin --shard k/N --shard-record ...`` and ``-p no:cacheprovider``. An argument that would
  conflict (``--shard*``, ``-n``, ``--dist``, ``--cov-append``, ``--cov-config``, another ``--junitxml`` or json
  report) is refused;
- each shard's environment sets ``COVERAGE_FILE=.coverage.shard<kk>`` and drops every other ``COVERAGE_*`` and
  ``COV_CORE_*`` name;
- any shard's non-zero or negative return code fails the run (exit 5, "no tests", included);
- reconciliation is by SETS: every shard saw the same ``whole`` item set and the same modules; the kept sets are
  pairwise disjoint and their union is ``whole``; no shard is empty; the merged junit's cases equal that union, with
  no duplicate (classname, name);
- coverage: exactly one data file per shard, all with arc data, combined by name; only the combine writes
  ``coverage.json`` and only the merge writes ``fast-suite.xml``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
JUNIT = "fast-suite.xml"
JSON_REPORT = "--cov-report=json:coverage.json"
REFUSED_PREFIXES = ("--shard", "-n", "--numprocesses", "--dist", "--cov-append", "--cov-config")
SHARD_TIMEOUT_S = 50 * 60


class RunnerError(Exception):
    """A fail-closed refusal: printed, exit 2."""


def shard_argv(args: list[str], k: int, n: int) -> list[str]:
    """The step's pytest arguments for shard k: verbatim, with only the allowlisted replacements and additions."""
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
        "-p",
        "no:cacheprovider",
    ]


def shard_env(k: int) -> dict[str, str]:
    env = {key: v for key, v in os.environ.items() if not key.startswith(("COVERAGE_", "COV_CORE_"))}
    env["COVERAGE_FILE"] = f".coverage.shard{k:02d}"
    return env


def refuse_leftovers(cwd: Path) -> None:
    left = [p.name for p in cwd.iterdir() if p.name in ("coverage.json", JUNIT) or p.name.startswith(".coverage")]
    left += [p.name for p in cwd.glob("fast-suite.shard*.xml")] + [p.name for p in cwd.glob("shard*.collect.json")]
    if left:
        raise RunnerError(
            f"refusing to start: outputs from an earlier run exist ({sorted(left)}); the step cleans them"
        )


def junit_cases(path: Path) -> list[tuple[str, str]]:
    return [(tc.get("classname", ""), tc.get("name", "")) for tc in ET.parse(path).iter("testcase")]


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
    env = {key: v for key, v in os.environ.items() if not key.startswith(("COVERAGE_", "COV_CORE_"))}
    for cmd in ([py, "-m", "coverage", "combine", *files], [py, "-m", "coverage", "json", "-q", "-o", "coverage.json"]):
        rc = subprocess.run(cmd, cwd=cwd, env=env).returncode
        if rc != 0:
            problems.append(f"`{' '.join(cmd)}` exited {rc}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("pytest_args", nargs=argparse.REMAINDER)
    ns = ap.parse_args(argv)
    args = ns.pytest_args[1:] if ns.pytest_args[:1] == ["--"] else ns.pytest_args
    n, cwd, py = ns.shards, Path.cwd(), sys.executable
    try:
        if n < 1:
            raise RunnerError("--shards must be >= 1")
        if not args or args[0] != "tests/":
            raise RunnerError("the pytest arguments must start with `tests/` (the suite's own selection)")
        refuse_leftovers(cwd)
        argvs = [shard_argv(args, k, n) for k in range(n)]
    except RunnerError as e:
        print(f"run_fast_suite: {e}", file=sys.stderr)
        return 2
    procs, logs, started = [], [], time.monotonic()
    for k in range(n):
        cmd = [py, "-m", "pytest", *argvs[k]]
        print(f"run_fast_suite: shard {k}/{n}: {' '.join(cmd)}", flush=True)
        log = open(cwd / f"shard{k:02d}.log", "wb")  # noqa: SIM115 - closed below
        logs.append(log)
        procs.append(subprocess.Popen(cmd, cwd=cwd, env=shard_env(k), stdout=log, stderr=subprocess.STDOUT))
    rcs: list[int | None] = []
    times: list[float] = []
    for k, p in enumerate(procs):
        try:
            rcs.append(p.wait(timeout=max(1.0, SHARD_TIMEOUT_S - (time.monotonic() - started))))
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
            rcs.append(None)
        times.append(time.monotonic() - started)
        logs[k].close()
    order = sorted(range(n), key=lambda k: (rcs[k] == 0, k))  # a failing shard's output first
    for k in order:
        print(f"\n===== shard {k}/{n}: rc={rcs[k]} ({times[k] / 60:.1f} min to finish) =====")
        sys.stdout.write((cwd / f"shard{k:02d}.log").read_text(encoding="utf-8", errors="replace")[-20000:])
    problems: list[str] = []
    for k, rc in enumerate(rcs):
        if rc is None:
            problems.append(f"shard {k} timed out after {SHARD_TIMEOUT_S // 60} min and was killed")
        elif rc != 0:
            why = " (pytest's 'no tests collected')" if rc == 5 else " (killed by a signal)" if rc < 0 else ""
            problems.append(f"shard {k} exited {rc}{why}")
    records = []
    for k in range(n):
        path = cwd / f"shard{k:02d}.collect.json"
        if path.exists():
            records.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            problems.append(f"shard {k} wrote no collection record")
    if len(records) == n:
        problems += reconcile(records, n)
        expected = set().union(*(set(r["selected"]) for r in records))
        if all((cwd / f"fast-suite.shard{k:02d}.xml").exists() for k in range(n)):
            problems += merge_junit(cwd, n, expected, records[0].get("collect_outcomes") or {})
        else:
            problems.append("a shard wrote no junit")
    if any(a.startswith("--cov=") for a in args):
        problems += combine_coverage(cwd, n, py)
    print(f"\nrun_fast_suite: per-shard minutes {[round(t / 60, 1) for t in times]}")
    if problems:
        print("run_fast_suite: FAILED\n  - " + "\n  - ".join(problems), file=sys.stderr)
        return 1
    print(f"run_fast_suite: {n} shards passed and reconcile")
    return 0


if __name__ == "__main__":
    sys.exit(main())
