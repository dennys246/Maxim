#!/usr/bin/env python
"""The R2 extras matrix's helper: each optional extra is tested against the REAL library at its floor and its latest
(standard R2, #1335; docs/plans/readiness_and_extras_standards.md §"R2's lane", whose design pass this follows).

The ``extras-matrix`` job in ``.github/workflows/test.yml`` runs one cell per ``{extra} × {floor, latest}``. Each cell
installs ONLY ``pymaxim[<extra>]`` plus pytest, and calls this helper:

``scope --extra X``
    On a pull request only (the YAML never asks on a push or a nightly, which run every cell, so a bug here cannot
    empty the release gate): ``run=true`` when the PR's diff touches a path in the extra's table, the shared lane
    paths, or when the diff cannot be read. Written as ``key=value`` lines for ``$GITHUB_OUTPUT``.
``pins --extra X --resolution {floor,latest} -o FILE``
    ``==`` constraints for every requirement of the extra. ``floor``: each requirement's ``>=`` bound (a requirement
    without one is refused, exit 2). The floor is the extra's DIRECT requirements only (``lowest-direct``): their own
    dependencies resolve to what pip picks today, so an old-onnxruntime incompatibility under piper-tts 1.3.0 is out
    of scope. ``latest``: what an unconstrained ``pip install --dry-run --ignore-installed``
    of the extra resolves today. ``[extras]`` are stripped (pip refuses them in a constraints file).
``verify --extra X --pins FILE``
    After the install: every pinned requirement whose marker holds here is installed at exactly its pin
    (``packaging.Version`` equality), and the installed distributions are exactly the metadata closure of
    ``pymaxim[<extra>]`` + ``pytest`` (+ pip/setuptools/wheel): nothing outside the extra can make it pass.
``tests --extra X``
    The test files the cell runs: those that apply the ``extras_<extra>`` marker (``check_lane_roster.marked_files``,
    the same set the roster and its PR-time twin collect). Not ``tests/``: a cell holds only the extra, and the rest
    of the suite does not collect without the other extras; a marked test elsewhere fails the roster check.
``assets --extra X --dest DIR``
    Provision what the extra's tests need, before the network namespace (tts: the pinned voice, sha256-verified).

Exit 0 ok; 1 a check failed; 2 bad input (an unknown extra, a requirement with no floor, an unreadable pip report).
Pinned by tests/unit/test_extras_matrix.py.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Any change here can break every cell, so a PR touching one runs them all.
# The lane itself: what every cell installs, imports or runs (the conftest and the guards and plugins it loads, the
# helper and what it imports, the network namespace).
SHARED_PATHS = (
    "pyproject.toml",
    "tests/conftest.py",
    "tests/network_guard.py",
    "tests/docker_guard.py",
    "tests/script_module_guard.py",
    "tests/multi_agent_fixtures.py",
    ".github/workflows/test.yml",
    "scripts/extras_matrix.py",
    "scripts/_lint_git.py",
    "scripts/check_lane_roster.py",
    "scripts/ci_netns.sh",
)

# extra -> the paths (prefixes) whose change makes a PR run that extra's cells: the extra's code, what its marked tests
# exercise, its consumers, its tests and its roster. A path missing here is caught by the nightly within a day, and by
# the release gate; it is not a hole in the guard, only a later red.
EXTRAS: dict[str, tuple[str, ...]] = {
    "console": (
        "src/maxim/console/",
        "src/maxim/cli.py",  # `maxim serve` dispatches to console.server.run_serve
        "src/maxim/tunnel/keys.py",  # the console token `maxim serve` ensures and the test reads
        "src/maxim/runtime/heartbeat.py",  # `maxim serve` starts the health monitor
        "tests/unit/test_console_ws_real_uvicorn.py",
        ".github/constraints/console.txt",
        "scripts/lane_rosters/extras-console.json",
    ),
    "tts": (
        "src/maxim/models/audio/",
        "src/maxim/models/download.py",
        "src/maxim/console/announcer.py",
        "src/maxim/utils/response_output.py",
        "src/maxim/utils/audio.py",
        "src/maxim/utils/optional_deps.py",  # _piper_installed: the provisioner's piper probe
        "src/maxim/console/server.py",  # _pairing_of, is_ready's consumer
        "src/maxim/embodied_runtime/agentic_runtime.py",  # builds the agentic runtime's TTSEngine
        # ensure_tts_voice's path (the assets step and the provisioner): fetch, data paths, storage preflight, lock
        "src/maxim/utils/http.py",
        "src/maxim/utils/paths.py",
        "src/maxim/utils/storage.py",
        "src/maxim/utils/process_lock.py",
        # cli.py (`maxim --tts`) also builds a TTSEngine, but nearly every PR touches it; the nightly covers it.
        "tests/unit/test_tts_real_piper.py",
        "tests/unit/test_piper_tts_synthesis.py",
        "tests/unit/test_tts_voice_provisioning.py",
        "scripts/lane_rosters/extras-tts.json",
    ),
}

_BASELINE = {"pip", "setuptools", "wheel"}  # the interpreter's own tooling, present before any install


class InputError(Exception):
    """Exit 2: the helper cannot answer (not: the answer is no)."""


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirements(extra: str, pyproject: Path = REPO / "pyproject.toml") -> list[str]:
    if extra not in EXTRAS:
        raise InputError(f"unknown extra {extra!r} (the matrix table knows: {', '.join(sorted(EXTRAS))})")
    deps = tomllib.loads(pyproject.read_text())["project"]["optional-dependencies"]
    if extra not in deps:
        raise InputError(f"pyproject.toml declares no {extra!r} extra")
    return list(deps[extra])


def _constraint(req: str, version: str) -> str:
    from packaging.requirements import Requirement

    r = Requirement(req)
    return f"{r.name}=={version}" + (f"; {r.marker}" if r.marker else "")


def floor_pins(reqs: list[str]) -> list[str]:
    from packaging.requirements import Requirement

    out = []
    for req in reqs:
        lows = [s.version for s in Requirement(req).specifier if s.operator == ">="]
        if len(lows) != 1:
            raise InputError(f"{req!r} declares no single >= floor; R2 tests every extra at its floor")
        out.append(_constraint(req, lows[0]))
    return out


def latest_pins(reqs: list[str], report: dict) -> list[str]:
    """``report`` is pip's ``--report`` JSON for an unconstrained install of the extra."""
    from packaging.requirements import Requirement

    try:
        resolved = {_norm(item["metadata"]["name"]): item["metadata"]["version"] for item in report["install"]}
    except (KeyError, TypeError) as exc:
        raise InputError(f"unreadable pip report: {exc!r}") from exc
    out = []
    for req in reqs:
        r = Requirement(req)
        if r.marker is not None and not r.marker.evaluate():
            continue  # not installed on this runner, so nothing to pin
        if _norm(r.name) not in resolved:
            raise InputError(f"the pip report resolves no {r.name!r}")
        out.append(_constraint(req, resolved[_norm(r.name)]))
    return out


def _resolve_latest_report(extra: str) -> dict:
    cmd = [sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed", "--quiet", "--report", "-"]
    r = subprocess.run([*cmd, f".[{extra}]"], cwd=REPO, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise InputError(f"pip --dry-run failed ({r.returncode}): {r.stderr[-500:]}")
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError as exc:
        raise InputError(f"unreadable pip report: {exc}") from exc


def check_pins(pins: list[str], installed: dict[str, str]) -> list[str]:
    """Problems: a pinned requirement (marker true here) not installed at exactly its pin."""
    from packaging.requirements import Requirement
    from packaging.version import Version

    problems = []
    for line in pins:
        r = Requirement(line)
        if r.marker is not None and not r.marker.evaluate():
            continue
        want = next(iter(r.specifier)).version
        have = installed.get(_norm(r.name))
        if have is None:
            problems.append(f"{r.name}: pinned {want}, not installed")
        elif Version(have) != Version(want):
            problems.append(f"{r.name}: pinned {want}, installed {have}")
    return problems


def closure(roots: list[str], requires_of) -> set[str]:  # noqa: ANN001 (requires_of: name -> list[str] | None)
    """Normalized names reachable from ``roots`` through each distribution's ``Requires-Dist``, extras and markers
    honoured. ``requires_of(name)`` returns ``None`` for a distribution that is not installed."""
    from packaging.requirements import Requirement

    seen: set[tuple[str, frozenset[str]]] = set()
    names: set[str] = set()
    todo = [Requirement(r) for r in roots]
    while todo:
        req = todo.pop()
        key = (_norm(req.name), frozenset(_norm(e) for e in req.extras))
        if key in seen:
            continue
        seen.add(key)
        names.add(key[0])
        for dep in requires_of(req.name) or []:
            d = Requirement(dep)
            if d.marker is None or any(d.marker.evaluate({"extra": e}) for e in (req.extras or {""})):
                todo.append(d)
    return names


def check_only_the_extra(extra: str, installed: set[str], requires_of) -> list[str]:  # noqa: ANN001
    expected = closure([f"pymaxim[{extra}]", "pytest"], requires_of) | _BASELINE
    stray = sorted(installed - expected)
    return [f"installed but outside pymaxim[{extra}] + pytest: {', '.join(stray)}"] if stray else []


def _installed() -> dict[str, str]:
    from importlib.metadata import distributions

    return {_norm(d.metadata["Name"]): d.version for d in distributions()}


def _requires_of(name: str) -> list[str] | None:
    from importlib.metadata import PackageNotFoundError, requires

    try:
        return requires(name) or []
    except PackageNotFoundError:
        return None


def scope(extra: str, changed: list[str] | None) -> tuple[bool, str]:
    """(run, why). ``changed`` is ``None`` when the diff could not be read: then the cell runs."""
    if extra not in EXTRAS:
        raise InputError(f"unknown extra {extra!r}")
    if changed is None:
        return True, "the PR's diff could not be read, so the cell runs"
    hits = sorted(p for p in changed if p in SHARED_PATHS or any(p.startswith(pre) for pre in EXTRAS[extra]))
    if hits:
        return True, f"the diff touches {', '.join(hits[:5])}" + (" ..." if len(hits) > 5 else "")
    return False, f"the diff touches nothing of the {extra} extra's (scripts/extras_matrix.py::EXTRAS)"


def _changed_paths() -> list[str] | None:
    sys.path.insert(0, str(REPO / "scripts"))
    import _lint_git

    try:
        base = _lint_git.base_ref(REPO)
        pairs = _lint_git.changed_files(REPO, base, ".", suffix="")
    except _lint_git.GitUnavailable as exc:
        print(f"scope: {exc}", file=sys.stderr)
        return None
    return sorted({p for pair in pairs for p in pair if p})  # both sides of a rename; "" marks an added file


def marked_test_files(extra: str) -> list[str]:
    if extra not in EXTRAS:
        raise InputError(f"unknown extra {extra!r}")
    sys.path.insert(0, str(REPO / "scripts"))
    import check_lane_roster

    return check_lane_roster.marked_files(f"extras_{extra}")


def assets(extra: str, dest: Path) -> int:
    if extra == "tts":
        from maxim.models import download

        dest.mkdir(parents=True, exist_ok=True)
        ok = download.ensure_tts_voice(download.DEFAULT_TTS, auto=True, models_dir=dest)
        print(f"assets: {download.DEFAULT_TTS} {'verified' if ok else 'NOT provisioned'} in {dest}")
        return 0 if ok else 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("scope", "pins", "verify", "tests", "assets"):
        p = sub.add_parser(name)
        p.add_argument("--extra", required=True)
        if name == "pins":
            p.add_argument("--resolution", choices=("floor", "latest"), required=True)
            p.add_argument("-o", "--output", type=Path, required=True)
        if name == "verify":
            p.add_argument("--pins", type=Path, required=True)
        if name == "assets":
            p.add_argument("--dest", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "scope":
            run, why = scope(args.extra, _changed_paths())
            print(f"run={'true' if run else 'false'}\nwhy={why}")
            return 0
        if args.cmd == "pins":
            reqs = requirements(args.extra)
            pins = (
                floor_pins(reqs)
                if args.resolution == "floor"
                else latest_pins(reqs, _resolve_latest_report(args.extra))
            )
            args.output.write_text("\n".join(pins) + "\n")
            print("\n".join(pins))
            return 0
        if args.cmd == "verify":
            installed = _installed()
            pins = [ln for ln in args.pins.read_text().splitlines() if ln.strip()]
            problems = check_pins(pins, installed) + check_only_the_extra(args.extra, set(installed), _requires_of)
            for line in sorted(f"{n}=={v}" for n, v in installed.items()):
                print(f"installed: {line}")
            for p in problems:
                print(f"FAIL: {p}", file=sys.stderr)
            return 1 if problems else 0
        if args.cmd == "tests":
            files = marked_test_files(args.extra)
            if not files:
                raise InputError(f"no test file applies the extras_{args.extra} marker")
            print(" ".join(files))
            return 0
        return assets(args.extra, args.dest)
    except InputError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
