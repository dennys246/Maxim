"""The CI escape paths stay closed (roadmap 1.3.2 item 7, #940; the 2026-09-27 Codex card's Test/CI findings).

Structural pins on `.github/workflows/test.yml`, read PARSED (a comment satisfies nothing):
- no install step may swallow its own failure (`|| echo`, `|| true`);
- the fast suite runs inside a loopback-only network namespace and arms the positive control there
  (tests/unit/test_network_boundary.py);
- the slow lane installs and runs its tests, offline against a warmed model cache, and is held to its exact
  roster (scripts/check_slow_lane.py);
- the claims lint runs in the lint job (roadmap 1.3.2 item 8).
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

WF = yaml.safe_load((Path(__file__).resolve().parents[2] / ".github/workflows/test.yml").read_text())


def _steps(job: str) -> list[dict]:
    return WF["jobs"][job]["steps"]


def _run(step: dict) -> str:
    return re.sub(r"\s+", " ", str(step.get("run", "")).replace("\\\n", " "))


def _commands(step: dict) -> str:
    """The step's shell without its comment lines (a comment may mention `|| echo` without running it)."""
    lines = [ln for ln in str(step.get("run", "")).splitlines() if not ln.strip().startswith("#")]
    return re.sub(r"\s+", " ", "\n".join(lines).replace("\\\n", " "))


def test_no_install_step_swallows_its_failure():
    offenders = [
        (job, step.get("name"))
        for job, spec in WF["jobs"].items()
        for step in spec.get("steps", [])
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


def test_the_slow_lane_installs_runs_and_checks_its_roster():
    steps = _steps("slow-tests")
    runs = " ".join(_run(s) for s in steps)
    assert '".[semantic,test,console,sign]"' in runs
    lane = next(s for s in steps if '-m "slow"' in _run(s))
    assert lane["env"]["MAXIM_RUN_MODEL_TESTS"] == "1" and lane["env"]["HF_HUB_OFFLINE"] == "1"
    assert "--require-extras=console,sign" in _run(lane) and "--junitxml=slow-results.xml" in _run(lane)
    check = next(s for s in steps if "check_slow_lane.py" in _run(s))
    assert check.get("if") == "always()" and "slow-results.xml" in _run(check)


def test_the_claims_lint_runs_in_the_lint_job():
    assert any(_run(s).strip() == "python3 scripts/lint_claims_sync.py" for s in _steps("lint"))
