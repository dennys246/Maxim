"""The CI escape paths stay closed (roadmap 1.3.2 item 7, #940; the 2026-09-27 Codex card's Test/CI findings).

Structural pins on `.github/workflows/test.yml`, read PARSED (a comment satisfies nothing):
- no install step may swallow its own failure (`|| echo`, `|| true`);
- the fast suite runs inside a loopback-only network namespace and arms the positive control there
  (tests/unit/test_network_boundary.py);
- both nightly lanes install and run their tests, offline against a warmed model cache, through ONE setup action
  (.github/actions/model-cache-setup, #1117), and each is held to its exact roster (scripts/check_lane_roster.py);
- the claims lint runs in the lint job (roadmap 1.3.2 item 8);
- the secret scan runs in the lint job and, whole-tree, in a nightly job (#1081).
"""

from __future__ import annotations

import re
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


def test_the_shared_setup_action_installs_caches_and_warms():
    """No PR loads the action (both lanes are nightly-only), so its shape is pinned here (design pass): a run step
    without `shell:` is rejected only when a nightly loads it."""
    assert ACTION["runs"]["using"] == "composite"
    steps = ACTION["runs"]["steps"]
    assert all(s.get("shell") == "bash" for s in steps if "run" in s), [s.get("name") for s in steps]
    runs = " ".join(_run(s) for s in steps)
    assert '".[semantic,test,console,sign]"' in runs and "spacy download en_core_web_sm" in runs
    cache = next(s for s in steps if str(s.get("uses", "")).startswith("actions/cache@"))
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
