"""scripts/run_fast_suite.py: CI's fast suite as N deterministic shards (#1645), and every fail-closed rule it carries.

The approach note's deletion probes (docs/plans/ci_sharded_suite_1645_approach.md, "Deletion probes (build)") each
have a test here that drives the REAL runner over a stub ``pytest`` (a package in the run's cwd, which ``python -m
pytest`` imports first). The stub writes what a shard writes (its collection record, its junit, a real coverage data
file made by the coverage CLI) and records the argv and coverage environment it was given, so the S1 test proves each
shard's argv is the pinned step's arguments plus the allowlist and nothing else. Each probe's stub tweak is one
defect the runner must turn red.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts import lint_coverage as LC
from scripts import run_fast_suite as R

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run_fast_suite.py"
# The step's pytest arguments as the GATE owns them (lint_coverage.CI_PYTEST_ARGS), as the shell hands them over: not
# the runner's own constant, so the runner and this test cannot agree with each other by construction (#1645 delta).
CI_ARGS = shlex.split(LC.CI_PYTEST_ARGS)

pytestmark = pytest.mark.timeout(120)

STUB = r"""
import json, os, signal, subprocess, sys
from pathlib import Path

argv = sys.argv[1:]


def opt(prefix):
    return next((a.split("=", 1)[1] for a in argv if a.startswith(prefix + "=")), None)


names = sorted(k for k in os.environ if k.startswith(("COVERAGE", "COV_CORE", "PYTEST_ADDOPTS", "PYTEST_PLUGINS",
                                                       "PYTEST_DISABLE_PLUGIN_AUTOLOAD")))
spec = opt("--shard")
if spec is None:  # --shards 1: the serial run records its argv and exits
    Path("argv-serial.json").write_text(json.dumps({"argv": argv, "env": names}))
    sys.exit(int(os.environ.get("STUB_SERIAL_RC", "0")))
k, n = (int(x) for x in spec.split("/"))
me = json.loads(os.environ.get("STUB", "{}")).get(str(k), {})
Path(f"argv{k}.json").write_text(json.dumps({"argv": argv, "env": names, "file": os.environ.get("COVERAGE_FILE")}))
items = [f"tests/test_s{i}.py::test_it" for i in range(n)]
whole = items + me.get("whole_extra", [])
whole = [me["swap"] if me.get("swap") and x == items[-1] else x for x in whole]
selected = [] if me.get("empty") else [items[k]]
record = {"_format_version": "1.0", "shard": k, "n": n, "whole": whole, "selected": selected,
          "modules": sorted({i.split("::")[0] for i in items}), "module_shards": {}, "collect_outcomes": {}}
Path(opt("--shard-record")).write_text(json.dumps(record))
cases = "" if me.get("drop_case") else f'<testcase classname="tests.test_s{k}" name="test_it"/>' * (2 if me.get("dup_case") else 1)
Path(opt("--junitxml")).write_text(f'<testsuites><testsuite name="pytest">{cases}</testsuite></testsuites>')
if any(a.startswith("--cov=") for a in argv) and not me.get("no_data"):
    branch = ["--branch"] if "--cov-branch" in argv and not me.get("line_only") else []
    subprocess.run([sys.executable, "-m", "coverage", "run", *branch, "t.py"], check=True)
if me.get("signal"):
    os.kill(os.getpid(), signal.SIGKILL)
sys.exit(me.get("rc", 0))
"""


def _env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("COVERAGE_", "COV_CORE_", "PYTEST_"))}
    env.update(extra)
    return env


def _cwd(tmp_path: Path) -> Path:
    (tmp_path / "pytest").mkdir()
    (tmp_path / "pytest" / "__init__.py").write_text("")
    (tmp_path / "pytest" / "__main__.py").write_text(STUB)
    (tmp_path / "t.py").write_text("x = 1\nif x:\n    y = 2\n")
    return tmp_path


def _run(cwd: Path, *argv: str, stub: dict | None = None, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(RUNNER), *argv],
        cwd=cwd,
        env=_env(STUB=json.dumps(stub or {}), **env),
        capture_output=True,
        text=True,
        timeout=100,
    )


def _sharded(cwd: Path, stub: dict | None = None, **env: str) -> subprocess.CompletedProcess:
    return _run(cwd, "--shards", "4", "--", *CI_ARGS, stub=stub, **env)


def _expected_shard_argv(k: int, n: int) -> list[str]:
    swap = {"--junitxml=fast-suite.xml": f"--junitxml=fast-suite.shard{k:02d}.xml", R.JSON_REPORT: "--cov-report="}
    allowlist = ["-p", "tests._shard_plugin", f"--shard={k}/{n}", f"--shard-record=shard{k:02d}.collect.json"]
    return [swap.get(a, a) for a in CI_ARGS] + allowlist + ["-p", "no:cacheprovider"]


# ── S1: each shard runs the pinned step's arguments plus the allowlist, nothing else ─────────────────────────────


def test_each_shard_gets_the_steps_arguments_verbatim_plus_only_the_allowlist(tmp_path):
    """A runner that dropped `--require-extras`, `--cov-branch` or `-m "not slow"`, or added an option, fails here; a
    coverage environment name injected into the job (COVERAGE_RCFILE) never reaches a shard, and neither does a pytest
    knob that would narrow the selection or drop pytest-cov (PYTEST_ADDOPTS, PYTEST_PLUGINS, plugin autoload off)."""
    cwd = _cwd(tmp_path)
    r = _sharded(cwd, COVERAGE_RCFILE=str(cwd / "no-such-rc"), COV_CORE_PROBE="1", PYTEST_ADDOPTS="-k notmemory",
                 PYTEST_PLUGINS="x", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")  # fmt: skip
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr
    for k in range(R.CI_SHARDS):
        seen = json.loads((cwd / f"argv{k}.json").read_text())
        assert seen["argv"] == _expected_shard_argv(k, R.CI_SHARDS)
        assert seen["env"] == ["COVERAGE_FILE"] and seen["file"] == f".coverage.shard{k:02d}"
    assert json.loads((cwd / "coverage.json").read_text())["meta"]["branch_coverage"] is True
    assert (cwd / "fast-suite.xml").read_text().count("<testcase") == R.CI_SHARDS
    assert "4 shards passed and reconcile" in r.stdout and "per-shard minutes" in r.stdout


def test_the_pinned_step_arguments_carry_the_suites_selection_and_measurement():
    """What the stub test compares against: the gate's own arguments (lint_coverage.CI_PYTEST_ARGS), which the runner's
    constant must equal token for token (the gate checks the flattened text; this checks what the shell splits)."""
    assert shlex.split(R.CI_PYTEST_ARGS) == CI_ARGS
    for must in ("tests/", "-x", "not slow", "--ignore=tests/integration/test_memory_hub.py",
                 "--require-extras=console,sign", "--junitxml=fast-suite.xml", "--cov=src/maxim", "--cov-branch",
                 R.JSON_REPORT):  # fmt: skip
        assert must in CI_ARGS, must
    assert f"-- {R.CI_PYTEST_ARGS}'" in R.CI_STEP_RUN and R.CI_STEP_RUN.startswith(R.CI_CLEAN + "\n")


# ── the deletion probes: every one must turn the run red ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "stub,needle",
    [
        ({"2": {"rc": 5}}, "shard 2 exited 5 (pytest's 'no tests collected')"),
        ({"1": {"signal": True}}, "shard 1 exited -9 (killed by a signal)"),
        ({"3": {"rc": 1}}, "shard 3 stopped at its first failure (-x)"),
        ({"0": {"empty": True}}, "shard 0 kept no tests (an empty shard)"),
        # a test module in no shard: every shard collects it, none keeps it
        ({str(k): {"whole_extra": ["tests/test_orphan.py::test_it"]} for k in range(4)}, "the shards' union differs"),
        # count-equal but set-different: shard 2 collected one different item
        ({"2": {"swap": "tests/test_other.py::test_it"}}, "shard 2 collected a different item set from shard 0"),
        ({"1": {"drop_case": True}}, "the merged junit's cases differ from the shards' kept items (missing 1"),
        ({"1": {"dup_case": True}}, "the merged junit has 1 duplicate (classname, name) cases"),
        ({"3": {"no_data": True}}, "shard 3: expected exactly one coverage data file, found []"),
        ({"2": {"line_only": True}}, ".coverage.shard02: no arc (branch) data"),
    ],
    ids=[
        "exit-5",
        "signal",
        "x-stop",
        "empty-shard",
        "file-in-no-shard",
        "set-different",
        "dropped-case",
        "duplicated-case",
        "deleted-data-file",
        "line-only-data",
    ],  # fmt: skip
)
def test_a_defect_in_any_shard_fails_the_run(tmp_path, stub, needle):
    cwd = _cwd(tmp_path)
    r = _sharded(cwd, stub)
    assert r.returncode == 1, r.stdout[-3000:] + r.stderr
    assert needle in r.stderr, r.stderr
    if any(m in needle for m in ("exited", "stopped", "kept no tests", "union", "different item")):
        assert not (cwd / "coverage.json").exists() and not (cwd / "fast-suite.xml").exists()


@pytest.mark.parametrize("leftover", [".coverage.x", ".coverage", "coverage.json", "fast-suite.xml",
                                      "fast-suite.shard00.xml", "shard01.collect.json"])  # fmt: skip
def test_a_leftover_output_refuses_the_start(tmp_path, leftover):
    cwd = _cwd(tmp_path)
    (cwd / leftover).write_text("stale")
    r = _sharded(cwd)
    assert r.returncode == 2 and "refusing to start" in r.stderr and leftover in r.stderr
    assert not list(cwd.glob("argv*.json")), "nothing may run"


@pytest.mark.parametrize(
    "extra,needle",
    [
        (["--cov-append"], "conflicts with sharding"),
        (["-n", "4"], "conflicts with sharding"),
        (["--shard=0/2"], "conflicts with sharding"),
        (["-p", "no:cov"], "conflicts with sharding"),
        (["--junitxml=other.xml"], "needs exactly one `--junitxml=fast-suite.xml`"),
        (["--cov-report=json:other.json"], "the only json report allowed"),
    ],
)
def test_an_argument_the_runner_cannot_forward_is_refused(tmp_path, extra, needle):
    r = _run(_cwd(tmp_path), "--shards", "4", "--", *CI_ARGS, *extra)
    assert r.returncode == 2 and needle in r.stderr, r.stderr


def test_the_arguments_must_start_with_the_suites_selection(tmp_path):
    r = _run(_cwd(tmp_path), "--shards", "4", "--", *CI_ARGS[1:])
    assert r.returncode == 2 and "must start with `tests/`" in r.stderr


# ── --shards 1 and --no-cov (suite_at_commit.sh, the rig) ───────────────────────────────────────────────────────


def test_one_shard_is_the_serial_run_verbatim(tmp_path):
    """No plugin, no partition, no report rewrites: the serial order, plus `-p no:cacheprovider` (never doubled)."""
    cwd = _cwd(tmp_path)
    r = _run(cwd, "--shards", "1", "--", *CI_ARGS, STUB_SERIAL_RC="3", PYTEST_ADDOPTS="-k notmemory")
    assert r.returncode == 3, r.stderr  # pytest's own status, unchanged
    assert json.loads((cwd / "argv-serial.json").read_text())["env"] == []  # no PYTEST_ADDOPTS in the serial run
    assert json.loads((cwd / "argv-serial.json").read_text())["argv"] == [*CI_ARGS, "-p", "no:cacheprovider"]
    (cwd / "argv-serial.json").unlink()
    r = _run(cwd, "--shards", "1", "--", *CI_ARGS, "-p", "no:cacheprovider")
    assert json.loads((cwd / "argv-serial.json").read_text())["argv"] == [*CI_ARGS, "-p", "no:cacheprovider"]


def test_no_cov_drops_every_cov_argument_and_skips_the_combine(tmp_path):
    cwd = _cwd(tmp_path)
    r = _run(cwd, "--shards", "4", "--no-cov", "--", *CI_ARGS)
    assert r.returncode == 0, r.stderr
    argv = json.loads((cwd / "argv0.json").read_text())["argv"]
    assert not [a for a in argv if a.startswith("--cov")] and not (cwd / "coverage.json").exists()
    assert (cwd / "fast-suite.xml").exists()


# ── the shard map ────────────────────────────────────────────────────────────────────────────────────────────────


def _map(shards: str, n: int = 4) -> str:
    return '{"_format_version": "1.0", "n": %d, "shards": {%s}}' % (n, shards)


@pytest.mark.parametrize(
    "text,needle",
    [
        (_map('"tests/a.py": 0, "tests/a.py": 1'), "duplicate keys ['tests/a.py']"),
        (_map('"tests/a.py": 4'), "is not a shard index in 0..3"),
        (_map('"tests/a.py": true'), "is not a shard index"),
        (_map('"tests/a.py": 0', n=3), "balanced for n=3"),
        ('{"n": 4, "shards": {}}', "needs _format_version 1.0"),
    ],
)
def test_a_malformed_map_fails(text, needle):
    with pytest.raises(R.MapError, match=None) as e:
        R.parse_map(text, 4)
    assert needle in str(e.value)


def test_the_committed_map_parses_for_cis_shard_count():
    assert R.parse_map((ROOT / R.MAP_REL).read_text(encoding="utf-8"), R.CI_SHARDS)


def test_the_shard_plugin_stops_on_a_map_the_runner_refuses(tmp_path):
    """The plugin partitions at every shard's start with the runner's own parser: a duplicate key stops the shard."""
    from tests import _shard_plugin as P

    (tmp_path / "scripts").mkdir()
    (tmp_path / R.MAP_REL).write_text(_map('"tests/a.py": 0, "tests/a.py": 1'))
    with pytest.raises(pytest.UsageError, match="duplicate keys"):
        P.load_map(tmp_path, 4)
    (tmp_path / R.MAP_REL).write_text(_map('"tests/a.py": 3'))
    assert P.load_map(tmp_path, 4) == {"tests/a.py": 3}


def _map_root(tmp_path: Path, shards: str, files: list[str]) -> Path:
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / R.MAP_REL).write_text(_map(shards))
    for f in files:
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text("")
    return tmp_path


def test_a_renamed_module_must_carry_its_entry(tmp_path):
    root = _map_root(tmp_path, '"tests/old.py": 2', ["tests/new.py"])
    failures, notes = R.map_problems(root, [("tests/new.py", "tests/old.py")])
    assert failures and "carry the entry" in failures[0] and '"tests/new.py": 2' in failures[0]
    assert notes and "tests/old.py" in notes[0]  # the stale key is a note, never a failure


def test_a_carried_rename_or_an_unmapped_rename_passes(tmp_path):
    root = _map_root(tmp_path, '"tests/old.py": 2, "tests/new.py": 2', ["tests/new.py"])
    assert R.map_problems(root, [("tests/new.py", "tests/old.py")])[0] == []
    root2 = _map_root(tmp_path / "b", '"tests/x.py": 1', ["tests/x.py", "tests/new.py"])
    assert R.map_problems(root2, [("tests/new.py", "tests/unmapped_old.py")]) == ([], [])


def test_check_map_reads_the_diff_and_fails_a_dropped_entry(tmp_path):
    """End to end through git: a rename against main whose new path is unmapped."""
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    root = _map_root(tmp_path, '"tests/test_old.py": 1', [])
    (root / "tests").mkdir()
    (root / "tests" / "test_old.py").write_text("def test_it():\n    assert True\n" * 20)
    (root / "scripts" / "run_fast_suite.py").write_text(RUNNER.read_text(encoding="utf-8"))
    (root / "scripts" / "_lint_git.py").write_text((ROOT / "scripts" / "_lint_git.py").read_text(encoding="utf-8"))
    for cmd in (["init", "-q", "-b", "main"], ["add", "-A"], ["commit", "-qm", "base"], ["checkout", "-qb", "f"],
                ["mv", "tests/test_old.py", "tests/test_new.py"], ["commit", "-qam", "rename"]):  # fmt: skip
        subprocess.run([*git, *cmd], cwd=root, check=True, capture_output=True)
    env = _env()
    env.pop("GITHUB_EVENT_NAME", None)
    r = subprocess.run([sys.executable, str(root / "scripts" / "run_fast_suite.py"), "--check-map"], cwd=root,
                       env=env, capture_output=True, text=True, timeout=60)  # fmt: skip
    assert r.returncode == 1 and "carry the entry" in r.stderr, r.stdout + r.stderr


def test_the_lint_job_runs_the_map_check():
    steps = yaml.safe_load((ROOT / ".github" / "workflows" / "test.yml").read_text())["jobs"]["lint"]["steps"]
    found = [s for s in steps if "run_fast_suite.py" in str(s.get("run", ""))]
    assert len(found) == 1 and set(found[0]) == {"name", "run"}
    assert found[0]["run"].strip() == "python3 scripts/run_fast_suite.py --check-map"


@pytest.mark.parametrize("rel", ["scripts/run_fast_suite.py", "tests/_shard_plugin.py"])
def test_the_runner_and_plugin_catch_no_broad_exception(rel):
    """The approach note puts the runner "in the silent-swallow lint's scope", but that lint scans src/maxim only
    (#1645 review). This is the runner's own check: no bare ``except``, no ``except Exception``/``BaseException``
    (alone or in a tuple), so every unexpected error exits non-zero instead of being turned into a pass."""
    import ast

    broad = {"Exception", "BaseException"}
    tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            types = node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
            names = {t.id for t in types if isinstance(t, ast.Name)} | {
                t.attr for t in types if isinstance(t, ast.Attribute)
            }
            if node.type is None or names & broad:
                offenders.append(node.lineno)
    assert not offenders, f"{rel}: broad exception handlers at lines {offenders}"
