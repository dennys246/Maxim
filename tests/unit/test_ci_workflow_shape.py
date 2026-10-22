"""The CI escape paths stay closed (roadmap 1.3.2 item 7, #940; the 2026-09-27 Codex card's Test/CI findings).

Structural pins on `.github/workflows/test.yml`, read PARSED (a comment satisfies nothing):
- no install step may swallow its own failure (`|| echo`, `|| true`);
- the fast suite runs inside a loopback-only network namespace and arms the positive control there
  (tests/unit/test_network_boundary.py);
- both nightly lanes install and run their tests, offline against a warmed model cache, through ONE setup action
  (.github/actions/model-cache-setup, #1117), and each is held to its exact roster (scripts/check_lane_roster.py);
- the claims lint runs in the lint job (roadmap 1.3.2 item 8);
- the secret scan runs in the lint job and, whole-tree, in a nightly job (#1081);
- the release gate runs in the lint job, and its runtime twin reads the fast suite's JUnit report in the unit-tests
  job (owner decision 2026-10-15; scripts/lint_release_gates.py also checks both from either side);
- the targeted mypy check's file list lives only in scripts/run_mypy_public.sh, which the unit-tests step runs
  (backlog M65).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WF = yaml.safe_load((ROOT / ".github/workflows/test.yml").read_text())
ACTION = yaml.safe_load((ROOT / ".github/actions/model-cache-setup/action.yml").read_text())
LANE_JOBS = ("model-cache-tests", "slow-tests")


def _steps(job: str) -> list[dict]:
    return WF["jobs"][job]["steps"]


def _run(step: dict) -> str:
    return re.sub(r"\s+", " ", str(step.get("run", "")).replace("\\\n", " "))


def _commands(step: dict) -> str:
    """The step's shell without its comment lines (a comment may mention `|| echo` without running it)."""
    lines = [ln for ln in str(step.get("run", "")).splitlines() if not ln.strip().startswith("#")]
    return re.sub(r"\s+", " ", "\n".join(lines).replace("\\\n", " "))


def _all_steps() -> list[tuple[str, dict]]:
    """Every workflow job's steps AND the composite action's (design pass: once the install moved into the action,
    a guard walking only the workflow went blind to it)."""
    out = [(job, step) for job, spec in WF["jobs"].items() for step in spec.get("steps", [])]
    return out + [("model-cache-setup", step) for step in ACTION["runs"]["steps"]]


def test_no_install_step_swallows_its_failure():
    offenders = [
        (where, step.get("name"))
        for where, step in _all_steps()
        if re.search(r"\bpip install\b", _commands(step)) and "||" in _commands(step)
    ]
    assert not offenders, offenders


def test_no_job_installs_the_package_without_its_core_dependencies():
    """2026-10-09: `pip install --no-deps -e .` left httpx/rich/json-repair/filelock to arrive transitively;
    huggingface-hub 2.x moved httpx -> httpx2 and main went red at an unchanged commit (filelock was already 4.x
    against our <4 pin). The package is installed WITH its declared dependencies everywhere."""
    offenders = [
        (where, step.get("name"))
        for where, step in _all_steps()
        if re.search(r"\bpip install\b[^\n]*--no-deps[^\n]*(-e\s+)?\.(\s|$|\[)", _commands(step))
    ]
    assert not offenders, offenders


def test_every_test_run_is_inside_the_loopback_only_network_namespace():
    """The fast suite and both nightly lanes (architecture review: the boundary covered one step). The MemoryHub step
    is the stated exception: the coverage gate pins its exact form (scripts/lint_coverage.py::ci_step_rules)."""
    for job, marker in (
        ("unit-tests", "--cov"),
        ("slow-tests", '-m "slow"'),
        ("model-cache-tests", "requires_model_cache"),
    ):
        runs = [_run(s) for s in _steps(job) if marker in _run(s) and "pytest" in _run(s)]
        assert len(runs) == 1, (job, runs)
        assert "bash scripts/ci_netns.sh" in runs[0] and '"$PY" -m pytest tests/' in runs[0], job


def test_the_netns_script_drops_privilege_and_arms_the_control():
    script = (Path(__file__).resolve().parents[2] / "scripts" / "ci_netns.sh").read_text()
    assert "unshare --net" in script and "ip link set lo up" in script and "setpriv" in script
    assert "MAXIM_EXPECT_NETNS=1" in script and '"PY=$(command -v python)"' in script
    assert '"PATH=$PATH"' in script  # captured outside sudo: secure_path would lose setup-python's interpreter


def test_the_netns_script_carries_the_callers_environment_by_name_and_checks_it_arrived():
    """O20 (2026-10-20): Ubuntu 26's sudo-rs ignores ``sudo -E``, so every job variable (MAXIM_TEST_TTS_VOICE_DIR,
    HF_HOME, MAXIM_RUN_MODEL_TESTS, ...) silently stopped reaching the namespace: the tts cells and both nightlies
    went red, and unit-tests stayed green without them. The environment now crosses sudo by NAME, and the innermost
    shell fails (exit 3) unless every preserved name arrived."""
    script = (ROOT / "scripts" / "ci_netns.sh").read_text()
    code = "\n".join(ln for ln in script.splitlines() if not ln.lstrip().startswith("#"))
    assert "sudo -E" not in code and '--preserve-env="$NETNS_ENV_NAMES"' in code
    assert '"MAXIM_NETNS_ENV_NAMES=$NETNS_ENV_NAMES"' in code  # names only: values never go on sudo's argv
    # The arrival check runs in the innermost shell, after setpriv, and refuses with exit 3.
    after_setpriv = code[code.index("setpriv") :]
    assert "${!n+x}" in after_setpriv and "exit 3" in after_setpriv
    # Secret-shaped names (the runner's tokens, a later HF_TOKEN or *_API_KEY) stay out of the suite.
    assert ".*TOKEN.*|.*_KEY|.*SECRET.*|.*PASSWORD.*" in code
    # Names bash or sudo recreate themselves are not counted (they would make the check pass with nothing carried), and
    # only grep's "no match" is tolerated, so a failing compgen/paste cannot become an empty list.
    assert "PWD|SHLVL|_|OLDPWD|SUDO_.*" in code and "|| [ $? -eq 1 ]" in code and "|| true" not in code


def test_the_shared_setup_action_installs_caches_and_warms():
    """No PR loads the action (both lanes are nightly-only), so its shape is pinned here (design pass): a run step
    without `shell:` is rejected only when a nightly loads it."""
    assert ACTION["runs"]["using"] == "composite"
    steps = ACTION["runs"]["steps"]
    assert all(s.get("shell") == "bash" for s in steps if "run" in s), [s.get("name") for s in steps]
    runs = " ".join(_run(s) for s in steps)
    assert '".[semantic,test,console,sign]"' in runs and "spacy download en_core_web_sm" in runs
    cache = next(s for s in steps if s.get("name") == "Restore model cache")  # the pip cache precedes it (#1318)
    assert str(cache.get("uses", "")).startswith("actions/cache@")
    assert "warm-list.txt" in cache["with"]["key"] and cache["with"]["path"] == "/home/runner/.cache/huggingface"
    warm = next(s for s in steps if s.get("name") == "Warm the cache")
    assert "if" not in warm  # unconditional: a partial or stale cache hit leaves models missing
    assert [s.get("name") for s in steps].index("Derive the required model list") < steps.index(cache)


@pytest.mark.parametrize(
    ("job", "marker", "lane", "xml"),
    [
        ("slow-tests", '-m "slow"', "slow", "slow-results.xml"),
        ("model-cache-tests", '-m "requires_model_cache"', "model-cache", "model-cache-results.xml"),
    ],
)
def test_each_lane_uses_the_shared_setup_runs_offline_and_checks_its_roster(job, marker, lane, xml):
    steps = _steps(job)
    assert sum(s.get("uses") == "./.github/actions/model-cache-setup" for s in steps) == 1, job
    assert not any(str(s.get("uses", "")).startswith("actions/cache@") for s in steps), "inlined cache: use the action"
    setup = next(i for i, s in enumerate(steps) if s.get("uses") == "./.github/actions/model-cache-setup")
    run = next(i for i, s in enumerate(steps) if marker in _run(s))
    assert setup < run
    env = steps[run]["env"]
    assert env["MAXIM_RUN_MODEL_TESTS"] == "1" and env["HF_HUB_OFFLINE"] == "1" and env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["HF_HOME"] == "/home/runner/.cache/huggingface"
    assert "--require-extras=console,sign" in _run(steps[run]) and f"--junitxml={xml}" in _run(steps[run])
    check = next(s for s in steps if "check_lane_roster.py" in _run(s))
    # its own step, always(): inside the run step a pytest failure would skip it and hide "not its roster"
    assert check is not steps[run] and check.get("if") == "always()"
    assert _run(check).strip() == f"python3 scripts/check_lane_roster.py --lane {lane} {xml}"


def test_the_claims_lint_runs_in_the_lint_job():
    assert any(_run(s).strip() == "python3 scripts/lint_claims_sync.py" for s in _steps("lint"))


def test_tests_run_on_prs_to_any_base():
    """D63: a stacked PR (base = a sibling branch) gets Tests at open, not only after retarget."""
    on = WF.get("on") or WF.get(True)  # PyYAML reads the bare key `on` as True
    assert "branches" not in (on["pull_request"] or {})


def test_the_settings_drift_check_runs_nightly_from_the_protected_environment():
    job = WF["jobs"]["repo-settings"]
    assert "(nightly)" in job["name"] and job["environment"] == "settings-check"
    step = next(s for s in job["steps"] if "check_repo_settings.py" in _run(s))
    assert step["env"]["GH_TOKEN"] == "${{ secrets.SETTINGS_READ_TOKEN }}" and "--static" not in _run(step)
    assert any(_run(s).strip() == "python3 scripts/check_repo_settings.py --static" for s in _steps("lint"))


def test_the_secret_scan_runs_on_every_pr_and_on_the_whole_tree_nightly():
    """#1081: the diff-mode step in the required lint job, and an explicit `--all` run in a (nightly) job the release
    gate reads (on a schedule the diff range is empty by construction, so the lint-job step alone would be vacuous)."""
    assert any(_commands(s).strip() == "python3 scripts/lint_secrets.py" for s in _steps("lint"))
    job = WF["jobs"]["secret-scan"]
    assert "(nightly)" in job["name"] and "schedule" in job["if"]
    assert any(_run(s).strip() == "python3 scripts/lint_secrets.py --all" for s in job["steps"])


def test_the_release_gate_runs_in_the_lint_job_and_its_twin_reads_the_fast_suites_junit():
    """Owner decision 2026-10-15: no version >= 1.4.0 passes CI until scripts/release_gates.json is done. Both steps
    are bare commands (no `if:`, no `continue-on-error`, no `|| true`), and the twin follows the suite that writes the
    report it reads."""
    lint = [s for s in _steps("lint") if "lint_release_gates.py" in _run(s)]
    assert len(lint) == 1 and set(lint[0]) == {"name", "run"}
    assert _run(lint[0]).strip() == "python3 scripts/lint_release_gates.py"
    steps = _steps("unit-tests")
    twin = [i for i, s in enumerate(steps) if "lint_release_gates.py" in _run(s)]
    suite = [i for i, s in enumerate(steps) if "--cov" in _run(s) and "pytest" in _run(s)]
    assert len(twin) == 1 and len(suite) == 1 and suite[0] < twin[0]
    assert set(steps[twin[0]]) == {"name", "run"}
    assert _run(steps[twin[0]]).strip() == "python scripts/lint_release_gates.py --junit fast-suite.xml"
    assert _run(steps[suite[0]]).count("--junitxml=fast-suite.xml") == 1


MYPY_SCRIPT = ROOT / "scripts/run_mypy_public.sh"


def _mypy_targets() -> list[str]:
    body = MYPY_SCRIPT.read_text()
    block = re.search(r"^FILES=\(\n(.*?)^\)", body, re.MULTILINE | re.DOTALL)
    assert block, "scripts/run_mypy_public.sh must hold its file list in a FILES=( ... ) array"
    return [ln.strip() for ln in block.group(1).splitlines() if ln.strip() and not ln.strip().startswith("#")]


def test_the_mypy_step_runs_the_single_source_script():
    """M65: one list. The unit-tests step installs the pinned mypy and runs the script; no step anywhere runs mypy
    on src/ inline."""
    steps = [s for s in _steps("unit-tests") if re.search(r"\bmypy\b", _commands(s))]
    assert len(steps) == 1, [s.get("name") for s in steps]
    assert _commands(steps[0]).strip() == 'pip install "mypy==1.20.0" bash scripts/run_mypy_public.sh'
    inline = [(job, s.get("name")) for job, s in _all_steps() if re.search(r"\bmypy\s+src/", _commands(s))]
    assert not inline, inline
    body = MYPY_SCRIPT.read_text()
    assert re.search(
        r'^exec mypy "\$\{FILES\[@\]\}" --ignore-missing-imports --follow-imports=silent --warn-unused-ignores$',
        body,
        re.MULTILINE,
    )


# Frozen records that quote an old copy of the list by design, never edited to track it: the pre-diet CLAUDE.md
# snapshot, archived plans, the score cards' captured evidence, and Exp 07's recorded reproduction recipe (a
# five-file subset it ran, part of that experiment's record).
FROZEN = (
    re.compile(r"^docs/lessons/[^/]*pre-diet[^/]*$"),
    re.compile(r"^docs/plans/archive/"),
    re.compile(r"^docs/limits/score_cards/evidence/"),
    re.compile(r"^docs/experiments/protocols/07_imagination_wiring_reproduction\.md$"),
)
WINDOW = 400


def mypy_list_copies(text: str, targets: list[str]) -> list[int]:
    """Offsets where a WINDOW-char span holds three or more distinct targets with `mypy` within WINDOW of it.

    Character windows, not lines, so a copy split over continuation lines (or a YAML block) is still one copy; the
    `mypy` condition keeps other lists of the same files (the swallow lint's MEASUREMENT_PATH, fixture trees) out.
    """
    pat = re.compile("|".join(re.escape(t) for t in sorted(targets, key=len, reverse=True)))
    hits = [(m.start(), m.group()) for m in pat.finditer(text)]
    out: list[int] = []
    for i, (start, _) in enumerate(hits):
        distinct = {g for s, g in hits[i:] if s - start < WINDOW}
        if len(distinct) >= 3 and "mypy" in text[max(0, start - WINDOW) : start + 2 * WINDOW].lower():
            out.append(start)
    return out


def test_no_other_file_carries_the_mypy_file_list():
    """M65: every tracked text file but the script itself and the frozen records is free of a second copy."""
    targets = _mypy_targets()
    assert len(targets) >= 10 and "src/maxim/api.py" in targets and "src/maxim/hivemind/" in targets
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    offenders = []
    for rel in tracked.splitlines():
        if rel == "scripts/run_mypy_public.sh" or any(f.search(rel) for f in FROZEN):
            continue
        try:
            text = (ROOT / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary, or a tracked path absent from this checkout
        if mypy_list_copies(text, targets):
            offenders.append(rel)
    assert not offenders, offenders
    stale = [f.pattern for f in FROZEN if not any(f.search(rel) for rel in tracked.splitlines())]
    assert not stale, f"exemptions that match no tracked file: {stale}"


def test_the_mypy_list_scan_catches_a_split_copy_and_spares_prose():
    targets = _mypy_targets()
    split = "mypy " + " \\\n     ".join(targets[:3]) + " --ignore-missing-imports\n"
    assert mypy_list_copies(split, targets)
    yaml_block = "run: |\n  mypy \\\n" + "".join(f"    {t} \\\n" for t in targets[3:6]) + "    --flag\n"
    assert mypy_list_copies(yaml_block, targets)
    assert not mypy_list_copies("run mypy after changing src/maxim/api.py or src/maxim/session.py", targets)
    assert not mypy_list_copies(" ".join(targets), targets)  # no `mypy` near: some other list of these files
    assert mypy_list_copies("MyPy: " + " ".join(targets[:3]), targets)  # case-insensitive


def test_a_required_check_doc_points_at_the_mypy_script():
    """M65: CLAUDE.md or a docs/agents brief names the script, so the check is findable without the list."""
    docs = [ROOT / "CLAUDE.md", *sorted((ROOT / "docs/agents").glob("*.md"))]
    assert any("scripts/run_mypy_public.sh" in d.read_text(encoding="utf-8") for d in docs)


def test_no_step_name_was_truncated_by_a_yaml_comment():
    """An unquoted `name: ... (x, #940)` parses as `... (x,`: YAML reads ` #` as a comment. Balanced parentheses in
    every parsed step name catch that class (it cut the function-length step's name until it was quoted)."""
    cut = [
        (job, s.get("name"))
        for job, s in _all_steps()
        if str(s.get("name", "")).count("(") != str(s.get("name", "")).count(")")
    ]
    assert not cut, cut


# ── #1318: the pip cache, and the pin that keeps it honest ─────────────────────────────────────────────────────────

_INSTALLS_THE_PACKAGE = re.compile(r"\bpip install\b[^\n]*?(?:-e\s+)?\.(?:\s|$|\[|\")")
# Jobs that install the package where a runner-side pip cache cannot reach, each with its reason.
_UNCACHEABLE = {
    "aarch64-install": "installs inside an arm64 Docker container (/venv/bin/pip), not on the runner",
    "extras-matrix": "R2's lane (#1335, Session S): one extra at floor/latest pins, a clean resolution by design, "
    "and it builds nothing from an sdist",
}


def _installs_the_package(job: str) -> bool:
    steps = _steps(job)
    return any(_INSTALLS_THE_PACKAGE.search(_commands(s)) for s in steps) or any(
        s.get("uses") == "./.github/actions/model-cache-setup" for s in steps
    )


def _first_install(steps: list[dict]) -> int:
    return next(
        i
        for i, s in enumerate(steps)
        if re.search(r"\bpip install\b", _commands(s)) or s.get("uses") == "./.github/actions/model-cache-setup"
    )


def test_every_job_that_installs_the_package_has_its_own_exact_pip_cache():
    """#1318: no job cached pip, and every run rebuilt llama-cpp-python from its sdist (2m50s of a 3m19s install).
    Review fold: setup-python's `cache: pip` key has no job in it, so a 4-minute job saved the shared key first and
    the 38-minute job never got its wheel; restore-keys would bring back stale or unbounded entries; and caching
    pip's downloads would save GBs of CUDA torch per lane into a repo already over its 10 GB cache cap (delta round).
    So: one actions/cache entry per job and Python, of the wheels pip BUILT, keyed on the files that decide the
    install, exact match only."""
    jobs = [job for job in WF["jobs"] if _installs_the_package(job) and job not in _UNCACHEABLE]
    assert {"unit-tests", "model-cache-tests", "slow-tests", "python-compatibility"} <= set(jobs), jobs
    offenders = []
    for job in jobs:
        steps = _steps(job)
        if any(s.get("uses") == "./.github/actions/model-cache-setup" for s in steps):
            steps = ACTION["runs"]["steps"]  # the lanes cache inside their shared setup (no inlined cache)
        caches = [i for i, s in enumerate(steps) if str(s.get("uses", "")).startswith("actions/cache@")]
        spec = steps[caches[0]].get("with", {}) if caches else {}
        key = str(spec.get("key", ""))
        if not (
            caches
            and caches[0] < _first_install(steps)
            and str(spec.get("path", "")).split("#")[0].strip() == "~/.cache/pip/wheels"
            and "${{ github.job }}" in key
            and "hashFiles('pyproject.toml', '.github/constraints/*.txt')" in key
            and re.search(r"-py(3\.12|\$\{\{ matrix\.python-version \}\})-", key)
            and "restore-keys" not in spec
            and "if" not in steps[caches[0]]
        ):
            offenders.append((job, spec))
    assert not offenders, offenders
    shared = [
        (job, s.get("with"))
        for job, spec in WF["jobs"].items()
        for s in spec.get("steps", [])
        if str(s.get("uses", "")).startswith("actions/setup-python") and (s.get("with") or {}).get("cache")
    ]
    assert not shared, f"setup-python's job-less cache key is back: {shared}"


def test_the_lint_job_stays_reproducible_by_prepr():
    """scripts/prepr.py replays the lint job locally and refuses a step it cannot reproduce (a cache is one)."""
    assert not any(str(s.get("uses", "")).startswith("actions/cache") for s in _steps("lint"))


def _llama_extras() -> set[str]:
    import tomllib

    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]

    def pulls(name: str, seen: tuple[str, ...] = ()) -> bool:
        for dep in extras[name]:
            if "llama-cpp-python" in dep:
                return True
            if dep.startswith("pymaxim[") and any(
                pulls(x.strip(), (*seen, name)) for x in dep[8 : dep.index("]")].split(",") if x.strip() not in seen
            ):
                return True
        return False

    return {name for name in extras if pulls(name)}


def test_llama_cpp_python_installs_exactly_pinned_and_portable():
    """#1318: the cache may only ever hold the pinned version (an upgrade is a reviewed edit of the constraints file,
    which is in the cache key), and the wheel it holds must run on any runner CPU (GGML_NATIVE=OFF). Review fold:
    an install through an extra that pulls llama (derived from pyproject) counts too."""
    pins = (ROOT / ".github/constraints/ci-backends.txt").read_text(encoding="utf-8")
    assert re.search(r"^llama-cpp-python==\d+\.\d+\.\d+$", pins, re.MULTILINE), pins
    extras = _llama_extras()
    assert {"llm-llama", "llm-server"} <= extras, extras

    def installs_llama(cmd: str) -> bool:
        if not re.search(r"\bpip install\b", cmd):
            return False
        named = {x.strip() for group in re.findall(r"\.\[([^\]]*)\]", cmd) for x in group.split(",")}
        return "llama-cpp-python" in cmd or bool(named & extras)

    installs = [(where, s) for where, s in _all_steps() if installs_llama(_commands(s))]
    assert installs, "no step installs llama-cpp-python: the pin guards nothing"
    offenders = [
        (where, s.get("name"))
        for where, s in installs
        if "-c .github/constraints/ci-backends.txt" not in _commands(s)
        or "-DGGML_NATIVE=OFF" not in str(s.get("env", {}).get("CMAKE_ARGS", ""))
    ]
    assert not offenders, offenders


def test_release_build_vendors_the_pinned_console_and_audits_it_without_a_waiver():
    """R4 (#1333): CI builds with the real pinned pulse bundle on the release path; the hand-written stub manifest and
    the audit's `--allow-missing-ui-dist` waiver are gone, and the freshness check runs on release PRs."""
    steps = _steps("release-build")
    runs = [_commands(s) for s in steps]
    vendor = [i for i, r in enumerate(runs) if "vendor_console_ui.py --from-pin" in r]
    build = [i for i, r in enumerate(runs) if "python -m build" in r]
    audit = [i for i, r in enumerate(runs) if "audit_release_build.py" in r]
    assert vendor and build and audit and vendor[0] < build[0] < audit[0], runs
    assert not any("ci-stub" in r or "maxim-ui.json" in r for r in runs), "a stub manifest is back"
    assert not any("--allow-missing-ui-dist" in r for r in runs)
    verify = [s for s in steps if "--verify-pin --only-when-releasing" in _commands(s)]
    assert len(verify) == 1 and verify[0].get("if") == "github.event_name == 'pull_request'", verify


def _stub_netns(tmp_path, sudo_mode: str):
    """A PATH where sudo/unshare/setpriv/ip are stubs, so the REAL ci_netns.sh runs on any box. ``sudo_mode``:
    "preserve" passes exactly the names in --preserve-env (both real sudos do); "drop" passes nothing, which is what
    sudo-rs did to ``-E`` (O20)."""
    import os
    import stat
    import sys

    stubs = tmp_path / "stubs"
    stubs.mkdir()
    scripts = {
        "sudo": f"""#!/usr/bin/env bash
kept=()
if [ "{sudo_mode}" = preserve ] && [[ "$1" == --preserve-env=* ]]; then
  IFS=, read -ra names <<< "${{1#--preserve-env=}}"
  for n in "${{names[@]}}"; do [ -n "${{!n+x}}" ] && kept+=("$n=${{!n}}"); done
fi
[[ "$1" == --preserve-env=* || "$1" == -E ]] && shift
exec env -i "${{kept[@]}}" "$@"
""",
        "unshare": '#!/usr/bin/env bash\nwhile [ "$1" != "--" ]; do shift; done; shift; exec "$@"\n',
        "setpriv": '#!/usr/bin/env bash\nwhile [[ "$1" == --* ]]; do shift; done; exec "$@"\n',
        "ip": "#!/bin/sh\nexit 0\n",
    }
    for name, body in scripts.items():
        path = stubs / name
        path.write_text(body)
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    (stubs / "python").symlink_to(sys.executable)
    env = {
        "PATH": f"{stubs}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path),
        # setup-python's interpreter on a CI runner is a shared build: the script passes LD_LIBRARY_PATH through to it.
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
        "MAXIM_TEST_TTS_VOICE_DIR": "/cache/voice",
        "GH_TOKEN": "not-for-the-suite",
        "HF_TOKEN": "not-for-the-suite",
    }
    return env


@pytest.mark.skipif(not __import__("shutil").which("bash"), reason="needs bash")
@pytest.mark.parametrize("sudo_mode", ["preserve", "drop"])
def test_the_netns_arrival_check_passes_a_carried_environment_and_refuses_a_dropped_one(tmp_path, sudo_mode):
    """The behaviour behind the string pins above (architecture review A3): the real script, stubbed below sudo."""
    env = _stub_netns(tmp_path, sudo_mode)
    suite = 'echo "voice=${MAXIM_TEST_TTS_VOICE_DIR:-<unset>} gh=${GH_TOKEN:-<unset>} hf=${HF_TOKEN:-<unset>}"'
    r = subprocess.run(
        ["bash", str(ROOT / "scripts" / "ci_netns.sh"), suite], env=env, capture_output=True, text=True, timeout=60
    )
    if sudo_mode == "preserve":
        assert r.returncode == 0, r.stderr
        assert "voice=/cache/voice gh=<unset> hf=<unset>" in r.stdout  # carried; secret-shaped names kept out
    else:
        assert r.returncode == 3, (r.returncode, r.stdout, r.stderr)
        assert "lost:" in r.stderr and "MAXIM_TEST_TTS_VOICE_DIR" in r.stderr and "voice=" not in r.stdout
