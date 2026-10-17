"""scripts/prepr.py runs exactly the CI lint job's steps, and refuses when it cannot reproduce CI (backlog M59).

The selection is read from `.github/workflows/test.yml` at run time; these tests pin that reading against an
INDEPENDENT expected-drop list (exact install lines, no regex), so a lint step added to CI cannot be missed locally
and the skip rule cannot quietly widen. The rest pins each "how could this input lie" defence in prepr's docstring
table: every refusal is exit 2, every downgrade taints the final line.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts import prepr as P

ROOT = Path(__file__).resolve().parents[2]
WF = yaml.safe_load((ROOT / P.WORKFLOW_REL).read_text())
PY = f"{sys.version_info.major}.{sys.version_info.minor}"
MYPY_STEP = "mypy public API surface + hivemind + composition layer"

# The exact install lines of today's workflow, by (job, step). A deliberately different rule from the code's regex:
# adding, removing or rewording an install line in CI must be reviewed here.
EXPECTED_DROPS: dict[tuple[str, str], list[str]] = {
    ("lint", "Ruff lint + format check"): [
        'pip install "ruff==0.14.14"',
        'pip install "pyyaml>=6.0,<7.0"  # scripts/lint_body_rest_neutral.py reads body YAML',
        'pip install "mypy==1.20.0"  # scripts/lint_mypy_ratchet.py; same pin as unit-tests\' mypy step',
    ],
    ("unit-tests", MYPY_STEP): ['pip install "mypy==1.20.0"'],
}


def _expected_script(job: str, step: dict) -> list[str]:
    drops = EXPECTED_DROPS.get((job, step["name"]), [])
    lines = step["run"].splitlines()
    for d in drops:
        assert d in lines, (step["name"], d)
    return [ln for ln in lines if ln not in drops]


def test_the_plan_is_every_lint_run_step_in_order_then_the_mypy_step():
    plan = P.load_plan(ROOT)
    lint = WF["jobs"]["lint"]["steps"]
    expected = [s["name"] for s in lint if "run" in s]
    mypy = [s["name"] for s in WF["jobs"]["unit-tests"]["steps"] if P.MYPY_SCRIPT in str(s.get("run", ""))]
    assert mypy == [MYPY_STEP]
    assert [s.name for s in plan.steps] == expected + mypy
    assert [s.job for s in plan.steps] == ["lint"] * len(expected) + ["unit-tests"]
    # Only the two setup actions are skipped; nothing is install-only today.
    assert sorted(why for _, _, why in plan.skipped) == ["no run: (uses/setup)"] * 2
    assert {s["uses"].split("@")[0] for s in lint if "run" not in s} == {"actions/checkout", "actions/setup-python"}
    assert plan.python_versions == frozenset({"3.12"})


def test_only_the_expected_install_lines_are_dropped():
    plan = P.load_plan(ROOT)
    by_name = {("lint", s["name"]): s for s in WF["jobs"]["lint"]["steps"] if "run" in s}
    by_name.update({("unit-tests", s["name"]): s for s in WF["jobs"]["unit-tests"]["steps"] if "name" in s})
    for step in plan.steps:
        raw = by_name[(step.job, step.name)]
        assert step.script.splitlines() == _expected_script(step.job, raw), step.name
        drops = EXPECTED_DROPS.get((step.job, step.name), [])
        assert len(step.dropped) == len(drops)
        assert all(d.startswith(got) for got, d in zip(step.dropped, drops, strict=True))


def test_ruff_runs_exactly_as_ci_runs_it():
    ruff = [s for s in P.load_plan(ROOT).steps if "ruff check" in s.script]
    assert len(ruff) == 1
    lines = [ln.strip() for ln in ruff[0].script.splitlines()]
    assert "ruff check src/ tests/" in lines
    assert "ruff format --check src/ tests/" in lines
    assert any("ruff==" in d for d in ruff[0].dropped)  # the pin is compared, not lost


def test_the_mypy_step_runs_the_single_source_script():
    mypy = P.load_plan(ROOT).steps[-1]
    assert mypy.script.strip() == f"bash {P.MYPY_SCRIPT}"


def test_the_pr_title_lint_gets_the_title_and_body():
    plan = P.load_plan(ROOT)
    users = [s for s in plan.steps if "lint_fix_touches_tests.py" in s.script]
    assert len(users) == 1
    assert users[0].pr_env == {"PR_TITLE": "title", "PR_BODY": "body"}
    assert "title" in plan.needs
    # github.token (job-level GH_TOKEN) is dropped, never passed through as a literal expression.
    assert all("${{" not in v for s in plan.steps for v in s.env.values())


def _setup() -> list[dict]:
    return [
        {"uses": "actions/checkout@v4", "with": {"fetch-depth": 0}},
        {"name": "py", "uses": "actions/setup-python@v5", "with": {"python-version": PY}},
    ]


def _wf(steps: list[dict], unit_job: dict | None = None, **job: object) -> dict:
    return {
        "jobs": {
            "lint": {"steps": _setup() + steps, **job},
            "unit-tests": {
                "steps": _setup() + [{"name": "m", "run": f'pip install "mypy==1.20.0"\nbash {P.MYPY_SCRIPT}\n'}],
                **(unit_job or {}),
            },
        }
    }


# ── shell semantics (fold 3) ───────────────────────────────────────────────────────────────────────────────────────


def test_shell_matches_githubs_default_and_explicit_bash():
    plan = P.build_plan(_wf([{"name": "a", "run": "true"}, {"name": "b", "run": "true", "shell": "bash"}]))
    by = {s.name: s.shell for s in plan.steps}
    assert by["a"] == ("bash", "-e")  # GitHub: no `shell:` -> `bash -e {0}`, NO pipefail
    assert by["b"] == ("bash", "--noprofile", "--norc", "-eo", "pipefail")  # `shell: bash`


@pytest.mark.parametrize("shell", ["sh", "python", "pwsh", "bash {0}"])
def test_any_other_shell_is_a_plan_error(shell):
    with pytest.raises(P.PlanError):
        P.build_plan(_wf([{"name": "x", "run": "true", "shell": shell}]))


# ── keys prepr cannot reproduce (fold 4) ───────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "step",
    [
        {"name": "x", "run": "true", "if": "always()"},
        {"name": "x", "run": "true", "working-directory": "src"},
        {"name": "x", "run": "true", "continue-on-error": True},
        {"name": "x", "run": "true", "env": {"A": "${{ secrets.X }}"}},
        {"name": "x", "uses": "actions/upload-artifact@v4"},
        {"name": "x", "uses": "astral-sh/ruff-action@v3"},
        {"name": "x", "uses": "actions/checkout@v4", "with": {"fetch-depth": 0, "ref": "other"}},
        {"name": "x", "uses": "actions/checkout@v4", "with": {"fetch-depth": 1}},
        {"name": "x", "uses": "actions/checkout@v4"},
        {"name": "x", "uses": "actions/setup-python@v5", "with": {"python-version": PY, "cache": "pip"}},
        {"name": "x", "uses": "actions/checkout@v4", "with": {"fetch-depth": 0}, "if": "false"},
    ],
)
def test_anything_prepr_cannot_reproduce_is_a_plan_error(step):
    with pytest.raises(P.PlanError):
        P.build_plan(_wf([step]))


@pytest.mark.parametrize("key", ["defaults", "container", "services", "strategy", "if"])
@pytest.mark.parametrize("job", ["lint", "unit-tests"])
def test_job_keys_that_change_step_semantics_are_plan_errors(key, job):
    value: object = {"run": {"shell": "sh"}} if key == "defaults" else "x"
    wf = _wf([{"name": "x", "run": "true"}])
    wf["jobs"][job][key] = value
    with pytest.raises(P.PlanError):
        P.build_plan(wf)


def test_the_live_jobs_only_carry_allowed_keys():
    for job in ("lint", "unit-tests"):
        assert set(WF["jobs"][job]) <= P.JOB_KEYS, job


def test_workflow_defaults_and_unknown_job_env_are_plan_errors():
    wf = _wf([{"name": "x", "run": "true"}])
    wf["defaults"] = {"run": {"shell": "sh"}}
    with pytest.raises(P.PlanError):
        P.build_plan(wf)
    with pytest.raises(P.PlanError):
        P.build_plan(_wf([{"name": "x", "run": "true"}], env={"T": "${{ github.sha }}"}))


def test_no_setup_python_is_a_plan_error():
    wf = _wf([{"name": "x", "run": "true"}])
    for job in ("lint", "unit-tests"):
        wf["jobs"][job]["steps"] = [s for s in wf["jobs"][job]["steps"] if "setup-python" not in str(s.get("uses"))]
    with pytest.raises(P.PlanError, match="python version"):
        P.build_plan(wf)


# ── install lines (fold 5) ─────────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "line",
    [
        "pip install x",
        "pip3 install x==2",
        "pip3.12 install x",
        "python -m pip install x",
        "python3 -m pip install x",
        "python3.12 -m pip install x",
        "uv pip install x",
        "sudo pip install x",
        "sudo -H pip3 install x",
        "pip -q install x",
        "python3 -m pip --quiet --no-input install x",
        'pip install "x>=1"  # a comment; with | separators && in it',
    ],
)
def test_plain_installs_are_dropped(line):
    script, dropped = P.strip_installs(f"{line}\nruff check src/\n")
    assert script == "ruff check src/\n"
    assert len(dropped) == 1 and "#" not in dropped[0]


@pytest.mark.parametrize(
    "line",
    [
        "pip install x && ruff check src/",
        "pip install x; ruff check src/",
        "pip install x || true",
        "pip install x | tee log",
        "pip install x &",
        "pip install `cat reqs`",
        "pip install $(cat reqs)",
        "uv pip install x && true",
        "cd d && pip install x",
        "true; python3 -m pip install x",
        "if true; then pip install x; fi",
        "pip -q install x && true",
    ],
)
def test_compound_installs_are_plan_errors(line):
    with pytest.raises(P.PlanError):
        P.strip_installs(f"{line}\n")


def test_skip_rule_install_only_steps_and_continuations():
    wf = _wf(
        [
            {"name": "deps", "run": "# comment\npip install a \\\n  b==1.0\npython -m pip install c\n"},
            {"name": "lint", "run": "pip3 install x==2\necho pip install y\nruff check src/\n", "env": {"K": "v"}},
        ]
    )
    plan = P.build_plan(wf)
    assert [s.name for s in plan.steps] == ["lint", "m"]
    assert ("lint", "deps", "install-only") in plan.skipped
    lint = plan.steps[0]
    assert lint.script == "echo pip install y\nruff check src/\n"  # a pip install that is not the command stays
    assert lint.dropped == ("pip3 install x==2",)
    assert lint.env == {"K": "v"}
    script, dropped = P.strip_installs("pip install a \\\n  b==1.0\nnext\n")
    assert script == "next\n" and dropped == ("pip install a b==1.0",)


def test_the_mypy_step_must_exist_exactly_once():
    wf = _wf([{"name": "x", "run": "true"}])
    wf["jobs"]["unit-tests"]["steps"] = [{"name": "m", "run": "mypy src/"}]
    with pytest.raises(P.PlanError):
        P.build_plan(wf)


# ── pins (fold 8) ──────────────────────────────────────────────────────────────────────────────────────────────────


def _pin_plan() -> P.Plan:
    return P.build_plan(
        _wf([{"name": "r", "run": 'pip install "ruff==0.14.14" "pyyaml>=6" "rich==1"\nruff check src/\n'}])
    )


def test_ruff_and_mypy_pin_mismatches_are_errors_other_pins_warn():
    def version(pkg: str) -> str:
        return "2" if pkg == "rich" else "0"

    errors, warns = P.check_pins(_pin_plan(), "/bin", version, lambda tool, path: {"ruff": "0.13.0"}.get(tool))
    assert any("ruff 0.13.0" in e and "0.14.14" in e for e in errors)
    assert any("mypy NOT FOUND" in e for e in errors)
    assert warns == ["WARN: rich 2 installed, CI pins 1; its findings may differ from CI's"]
    errors, _ = P.check_pins(
        _pin_plan(), "/bin", version, lambda tool, path: {"ruff": "0.14.14", "mypy": "1.20.0"}[tool]
    )
    assert errors == []


def test_tool_version_reads_the_binary_on_the_given_path(tmp_path):
    fake = tmp_path / "ruff"
    fake.write_text("#!/bin/sh\necho 'ruff 9.9.9'\n")
    fake.chmod(0o755)
    assert P.tool_version("ruff", str(tmp_path)) == "9.9.9"
    assert P.tool_version("ruff", str(tmp_path / "nowhere")) is None


# ── PR text (fold 6) ───────────────────────────────────────────────────────────────────────────────────────────────

BRANCH = "feature"


def OWNER() -> str:
    return "dennys246"


def _pr(**kw: object) -> dict:
    return {
        "number": 7,
        "state": "OPEN",
        "headRefName": BRANCH,
        "isCrossRepository": False,
        "headRepositoryOwner": {"login": "dennys246"},
        "title": "gh",
        "body": "b",
        **kw,
    }


def test_pr_text_resolution():
    assert P.resolve_pr_text(None, None, lambda: _pr(), BRANCH, OWNER) == ("gh", "b", False)
    assert P.resolve_pr_text("gh", "b", lambda: _pr(), BRANCH, OWNER) == (
        "gh",
        "b",
        False,
    )  # typed, but agrees with the PR
    assert P.resolve_pr_text("gh", None, lambda: _pr(), BRANCH, OWNER) == ("gh", "b", False)
    assert P.resolve_pr_text(None, None, lambda: None, BRANCH, OWNER) == (None, None, False)
    assert P.resolve_pr_text("t", None, lambda: None, BRANCH, OWNER) == ("t", "", True)  # no PR: typed text taints


@pytest.mark.parametrize("title, body", [("other", None), (None, "other"), ("gh", "other")])
def test_typed_text_that_differs_from_the_open_pr_refuses(title, body):
    """A1: CI reads the PR's text; a hand-typed stand-in must never silently replace it."""
    with pytest.raises(P.Refusal, match="differ from open PR #7's text"):
        P.resolve_pr_text(title, body, lambda: _pr(), BRANCH, OWNER)


@pytest.mark.parametrize("state", ["CLOSED", "MERGED", None])
def test_a_pr_that_is_not_open_is_no_pr(state):
    assert P.resolve_pr_text(None, None, lambda: _pr(state=state), BRANCH, OWNER) == (None, None, False)


def test_an_open_prs_text_is_used_wherever_its_head_is_but_only_for_this_branch():
    """S1: the PR's text does not depend on its head commit (unpushed work still gets the PR's text); the guard
    against gh resolving the wrong PR is the branch name."""
    assert P.resolve_pr_text(None, None, lambda: _pr(headRefOid="0" * 40), BRANCH, OWNER) == ("gh", "b", False)
    for title, body in ((None, None), ("gh", "b")):
        with pytest.raises(P.Refusal, match="not this branch's PR"):
            P.resolve_pr_text(title, body, lambda: _pr(headRefName="other"), BRANCH, OWNER)
    with pytest.raises(P.Refusal, match="not this branch's PR"):
        P.resolve_pr_text(None, None, lambda: _pr(), "", OWNER)  # detached HEAD


def test_body_alone_follows_the_same_rules():
    assert P.resolve_pr_text(None, "b", lambda: _pr(), BRANCH, OWNER) == ("gh", "b", False)
    with pytest.raises(P.Refusal, match="--body differ"):
        P.resolve_pr_text(None, "other", lambda: _pr(), BRANCH, OWNER)
    assert P.resolve_pr_text(None, "b", lambda: None, BRANCH, OWNER) == (
        None,
        "b",
        True,
    )  # no PR: typed, title still owed


def test_typed_text_is_compared_after_crlf_and_trailing_space_normalization():
    pr = _pr(title="feat: x ", body="line one\r\nline two  \r\n")
    assert P.resolve_pr_text("feat: x", "line one\nline two", lambda: pr, BRANCH, OWNER) == (
        "feat: x ",
        "line one\r\nline two  \r\n",  # CI's raw text is what is used (and scanned)
        False,
    )
    with pytest.raises(P.Refusal):
        P.resolve_pr_text("feat: x", "line one\nline 2", lambda: pr, BRANCH, OWNER)


def test_gh_is_asked_for_state_and_head(monkeypatch):
    seen: list[list[str]] = []

    def fake_run(cmd, **kw):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout='{"state": "OPEN"}', stderr="")

    monkeypatch.setattr(P.shutil, "which", lambda name, **kw: "/x/gh")
    monkeypatch.setattr(P.subprocess, "run", fake_run)
    assert P.gh_pr_view(ROOT) == {"state": "OPEN"}
    assert seen[0][-1] == "number,state,headRefName,isCrossRepository,headRepositoryOwner,title,body"


# ── environment (event vars, BASH_ENV) ─────────────────────────────────────────────────────────────────────────────


def test_event_and_runner_variables_are_removed_never_faked():
    step = P.Step("lint", "x", "true\n", {"K": "v"}, {"PR_TITLE": "title"})
    base = {
        "GITHUB_EVENT_NAME": "pull_request",
        "GITHUB_EVENT_PATH": "/e.json",
        "GITHUB_ACTIONS": "true",
        "GITHUB_REPOSITORY": "a/b",
        "RUNNER_OS": "Linux",
        "BASH_ENV": "/evil",
        "ENV": "/evil",
        "PYTHONPATH": "/elsewhere/src",
        "MYPYPATH": "/elsewhere",
        "MAXIM_SUBSTRATE_PATH": "1",
        "PATH": "/bin",
    }
    env = P.step_env(step, {"title": "feat: x", "body": ""}, base=base)
    assert not [k for k in env if k.startswith(("GITHUB_", "RUNNER_", "MAXIM_"))]
    assert not {"BASH_ENV", "ENV", "PYTHONPATH", "MYPYPATH"} & set(env)
    assert env["K"] == "v" and env["PR_TITLE"] == "feat: x"
    assert env["PATH"].startswith(str(Path(P.sys.executable).parent))


# ── git preflight (folds 1 and 2) ──────────────────────────────────────────────────────────────────────────────────


def _git(root: Path, *args: str) -> str:
    env = dict(
        os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t"
    )
    return subprocess.run(["git", *args], cwd=root, env=env, capture_output=True, text=True, check=True).stdout


def _commit(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", rel)


def make_repo(root: Path, wf: dict | None = None, files: dict[str, str] | None = None) -> Path:
    """A repo whose origin/main is the base commit and whose `feature` branch is one commit ahead of it."""
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "commit.gpgsign", "false")
    _commit(root, P.WORKFLOW_REL, yaml.safe_dump(wf or _wf([{"name": "ok", "run": "true"}])))
    _git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    _git(root, "checkout", "-q", "-b", "feature")
    for rel, text in (files or {}).items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    _commit(root, P.MYPY_SCRIPT, "exit 0\n")
    return root


def _preflight(root: Path, **kw: bool) -> tuple[str, str, list[str]]:
    args = {"fetch": False, "allow_behind": False, "allow_dirty": False, **kw}
    return P.git_preflight(root, **args)


def test_preflight_clean_and_up_to_date(tmp_path):
    root = make_repo(tmp_path / "r")
    head, mb, taints = _preflight(root)
    assert head == _git(root, "rev-parse", "HEAD").strip()
    assert mb == _git(root, "rev-parse", "origin/main").strip()
    assert taints == ["--no-fetch (origin/main may be stale)"]


def test_preflight_refuses_a_branch_behind_origin_main(tmp_path):
    root = make_repo(tmp_path / "r")
    _git(root, "checkout", "-q", "main")
    _commit(root, "later.txt", "x")
    _git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    _git(root, "checkout", "-q", "feature")
    with pytest.raises(P.Refusal, match="merge or rebase onto origin/main"):
        _preflight(root)
    _, _, taints = _preflight(root, allow_behind=True)
    assert any(t.startswith("--allow-behind (HEAD is 1 commit(s) behind") for t in taints)


@pytest.mark.parametrize("change", ["untracked", "modified", "staged"])
def test_preflight_refuses_a_dirty_tree(tmp_path, change):
    root = make_repo(tmp_path / "r")
    if change == "untracked":
        (root / "new.txt").write_text("x")
    else:
        (root / P.MYPY_SCRIPT).write_text("exit 1\n")
        if change == "staged":
            _git(root, "add", "-A")
    with pytest.raises(P.Refusal, match="--allow-dirty"):
        _preflight(root)
    _, _, taints = _preflight(root, allow_dirty=True)
    assert any(t.startswith("--allow-dirty (1 uncommitted") for t in taints)


def test_preflight_refuses_when_fetch_fails(tmp_path):
    root = make_repo(tmp_path / "r")  # no `origin` remote: the fetch fails
    with pytest.raises(P.Refusal, match="git fetch origin main failed"):
        _preflight(root, fetch=True)


def test_preflight_fetches_and_then_passes(tmp_path):
    upstream = make_repo(tmp_path / "up")
    _git(upstream, "checkout", "-q", "main")
    root = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(upstream), str(root)], check=True, capture_output=True)
    _git(root, "config", "commit.gpgsign", "false")
    _git(root, "checkout", "-q", "-b", "feature")
    _commit(root, "x.txt", "x")
    _, _, taints = _preflight(root, fetch=True)
    assert taints == []


def test_preflight_refuses_a_shallow_clone(tmp_path):
    upstream = make_repo(tmp_path / "up")
    root = tmp_path / "shallow"
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1", "--branch", "feature", f"file://{upstream}", str(root)],
        check=True,
        capture_output=True,
    )
    assert _git(root, "rev-parse", "--is-shallow-repository").strip() == "true"
    with pytest.raises(P.Refusal, match="shallow"):
        _preflight(root)


def test_preflight_refuses_without_a_merge_base(tmp_path):
    root = make_repo(tmp_path / "r")
    _git(root, "checkout", "-q", "--orphan", "other")
    _commit(root, "o.txt", "o")
    _git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    _git(root, "checkout", "-q", "feature")
    with pytest.raises(P.Refusal, match="no merge-base"):
        _preflight(root)
    _git(root, "update-ref", "-d", "refs/remotes/origin/main")
    with pytest.raises(P.Refusal, match="no merge-base"):
        _preflight(root)


# ── main (folds 6, 7, 8 and the final line) ────────────────────────────────────────────────────────────────────────


@pytest.fixture
def quiet(monkeypatch):
    """Pins match, no platform taint, no gh: what is left is what each test sets up."""
    monkeypatch.setattr(P, "tool_version", lambda tool, path: {"ruff": "0.14.14", "mypy": "1.20.0"}.get(tool))
    monkeypatch.setattr(P, "platform_note", lambda: None)
    monkeypatch.setattr(P, "gh_pr_view", lambda root=P.REPO_ROOT: None)
    monkeypatch.setattr(P, "scan_pr_text", lambda root, base, head, text: 0)
    monkeypatch.setattr(P, "repo_owner", lambda root=P.REPO_ROOT: "dennys246")


def _final(out: str) -> str:
    return [ln for ln in out.splitlines() if ln.startswith("prepr: ")][-1]


PR_STEP = {
    "name": "title",
    "run": 'test "$PR_TITLE" = "feat: t"\n',
    "env": {"PR_TITLE": "${{ github.event.pull_request.title }}"},
}


def test_main_requires_a_title_when_no_pr_exists(tmp_path, quiet, capsys):
    root = make_repo(tmp_path / "r", _wf([PR_STEP]))
    assert P.main(["--no-fetch"], root=root) == 2
    assert "--title" in capsys.readouterr().err


def test_main_runs_steps_with_githubs_shells_and_reports(tmp_path, quiet, capsys):
    root = make_repo(
        tmp_path / "r",
        _wf(
            [
                {**PR_STEP, "run": 'pip install "nosuchpkg-prepr==1"\n' + PR_STEP["run"]},
                {"name": "bad", "run": "false\necho never\n"},
                {"name": "no-pipefail", "run": "false | true\n"},
                {"name": "pipefail", "run": "false | true\n", "shell": "bash"},
            ]
        ),
    )
    rc = P.main(["--no-fetch", "--title", "feat: t"], root=root)
    out = capsys.readouterr().out
    assert rc == 1
    assert re.search(r"PASS .*\[lint\] title", out)
    assert re.search(r"FAIL .*\[lint\] bad", out) and "never" not in out  # bash -e stops the step
    assert re.search(r"PASS .*\[lint\] no-pipefail", out)  # `bash -e {0}`: GitHub's default has NO pipefail
    assert re.search(r"FAIL .*\[lint\] pipefail", out)  # `shell: bash` adds -o pipefail
    assert re.search(r"PASS .*\[unit-tests\] m", out)
    assert "nosuchpkg-prepr==1 is pinned in CI but not installed" in out
    assert "3/5 steps passed" in out
    assert _final(out).startswith("prepr: FAIL (2): bad; pipefail (tainted: --no-fetch")
    assert "PR text typed by hand" in _final(out)


def test_main_pass_is_bare_only_when_nothing_was_downgraded(tmp_path, quiet, capsys, monkeypatch):
    root = make_repo(tmp_path / "r", _wf([PR_STEP]))
    monkeypatch.setattr(P, "git_preflight", lambda root, **kw: ("h" * 40, "b" * 40, []))
    monkeypatch.setattr(P, "gh_pr_view", lambda root=P.REPO_ROOT: _pr(title="feat: t"))
    assert P.main([], root=root) == 0
    assert _final(capsys.readouterr().out) == "prepr: PASS"
    # The platform is a NOTE on its own line, not an operator taint: exit 0 and a bare PASS.
    monkeypatch.setattr(P, "platform_note", lambda: "platform darwin")
    assert P.main([], root=root) == 0
    out = capsys.readouterr().out
    assert _final(out) == "prepr: PASS" and "NOTE: platform darwin" in out
    # Hand-typed text with no PR at HEAD is an operator downgrade: exit 3.
    monkeypatch.setattr(P, "gh_pr_view", lambda root=P.REPO_ROOT: None)
    assert P.main(["--title", "feat: t"], root=root) == 3
    assert (
        _final(capsys.readouterr().out)
        == "prepr: TAINTED-PASS (tainted: PR text typed by hand (no open PR to read it from))"
    )


def test_main_refuses_a_dirty_tree_and_taints_with_allow_dirty(tmp_path, quiet, capsys):
    root = make_repo(tmp_path / "r")
    (root / "scratch.txt").write_text("x")
    assert P.main(["--no-fetch"], root=root) == 2
    assert "REFUSED" in capsys.readouterr().out
    assert P.main(["--no-fetch", "--allow-dirty"], root=root) == 3
    final = _final(capsys.readouterr().out)
    assert final.startswith("prepr: TAINTED-PASS (") and "--allow-dirty (1 uncommitted/untracked path(s))" in final


def test_no_pr_skips_only_the_pr_steps_and_says_so(tmp_path, quiet, capsys):
    root = make_repo(tmp_path / "r", _wf([PR_STEP, {"name": "other", "run": "true"}]))
    assert P.main(["--no-fetch", "--no-pr"], root=root) == 3
    out = capsys.readouterr().out
    assert "SKIP  (no-pr) title" in out and re.search(r"PASS .*\[lint\] other", out)
    assert _final(out).startswith("prepr: TAINTED-PASS (tainted: --no-fetch")
    assert "--no-pr (1 step(s) reading the PR text skipped; no PR-text secret scan)" in _final(out)
    assert P.main(["--no-fetch", "--no-pr", "--title", "x"], root=root) == 2


def test_main_reads_this_branchs_open_pr_and_refuses_another(tmp_path, quiet, capsys, monkeypatch):
    root = make_repo(tmp_path / "r", _wf([PR_STEP]))
    monkeypatch.setattr(P, "gh_pr_view", lambda root=P.REPO_ROOT: _pr(headRefName="other", title="feat: t"))
    assert P.main(["--no-fetch"], root=root) == 2
    assert "not this branch's PR" in capsys.readouterr().err
    monkeypatch.setattr(P, "gh_pr_view", lambda root=P.REPO_ROOT: _pr(headRefOid="0" * 40, title="feat: t"))
    assert P.main(["--no-fetch"], root=root) == 3  # --no-fetch is the only taint; the PR's head is irrelevant
    assert _final(capsys.readouterr().out) == "prepr: TAINTED-PASS (tainted: --no-fetch (origin/main may be stale))"
    assert P.main(["--no-fetch", "--title", "feat: other"], root=root) == 2  # A1: typed text != the open PR


def test_main_refuses_a_pin_mismatch_unless_allowed(tmp_path, quiet, capsys, monkeypatch):
    root = make_repo(tmp_path / "r")
    monkeypatch.setattr(P, "tool_version", lambda tool, path: "0.0.1")
    assert P.main(["--no-fetch"], root=root) == 2
    assert "mypy 0.0.1" in capsys.readouterr().err
    assert P.main(["--no-fetch", "--allow-pin-mismatch"], root=root) == 3
    assert "--allow-pin-mismatch (mypy 0.0.1" in _final(capsys.readouterr().out)


def test_main_refuses_another_python_unless_allowed(tmp_path, quiet, capsys):
    wf = _wf([{"name": "ok", "run": "true"}])
    for job in ("lint", "unit-tests"):
        wf["jobs"][job]["steps"][1]["with"]["python-version"] = "3.99"
    root = make_repo(tmp_path / "r", wf)
    assert P.main(["--no-fetch"], root=root) == 2
    assert "3.99" in capsys.readouterr().err
    assert P.main(["--no-fetch", "--allow-python-mismatch"], root=root) == 3
    assert "--allow-python-mismatch" in _final(capsys.readouterr().out)


def test_list_prints_the_plan(capsys):
    assert P.main(["--list"], root=ROOT) == 0
    out = capsys.readouterr().out
    assert "RUN   [lint] Ruff lint + format check [bash -e]" in out
    assert f"RUN   [unit-tests] {P.load_plan(ROOT).steps[-1].name}" in out


def test_prepr_sh_refuses_without_pyyaml(tmp_path):
    fake = tmp_path / "python3"
    fake.write_text('#!/bin/sh\nif [ "$1" = "-c" ]; then exit 1; fi\necho ran\n')
    fake.chmod(0o755)
    r = subprocess.run(
        ["bash", str(ROOT / "scripts/prepr.sh"), "--list"],
        env={**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"},
        capture_output=True,
        text=True,
    )
    assert r.returncode == 2 and "PyYAML" in r.stderr and "ran" not in r.stdout


# ── round 2: E1, E2, A2, N1, exit-code docs ────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("run", ['echo "${{ github.sha }}"\n', "test -n '${{ secrets.X }}'\n"])
def test_an_expression_in_a_run_body_is_a_plan_error(run):
    """E1: CI substitutes `${{ }}` before bash sees the script; prepr cannot."""
    with pytest.raises(P.PlanError, match="run body"):
        P.build_plan(_wf([{"name": "x", "run": run}]))


@pytest.mark.parametrize(
    "line",
    [
        'pip install "ruff>=0.14"',
        "pip install ruff",
        "pip install mypy~=1.20",
        "pip install ruff == 0.14.14",
        'pip install "ruff ==0.14.14"',
        "pip install -r reqs.txt",
        "pip install -rreqs.txt",
        "pip install --requirement=reqs.txt",
        "pip install -c constraints.txt pyyaml",
    ],
)
def test_a_strict_tool_without_an_exact_pin_is_a_plan_error(line):
    """E2: the pin check needs `tool==X`; anything else leaves CI's ruff/mypy version unknowable."""
    with pytest.raises(P.PlanError):
        P.build_plan(_wf([{"name": "x", "run": f"{line}\nruff check src/\n"}]))


@pytest.mark.parametrize(
    "line", ['pip install "ruff==0.14.14"', 'pip install "mypy[reports]==1.20.0" types-requests mypy-extensions']
)
def test_exact_strict_pins_and_other_packages_are_accepted(line):
    assert P.build_plan(_wf([{"name": "x", "run": f"{line}\nruff check src/\n"}]))


@pytest.mark.parametrize("cmd", ["python3 scripts/lint_secrets.py", "python3 -m scripts.lint_secrets"])
def test_main_fails_on_a_secret_in_the_pr_text(tmp_path, quiet, capsys, monkeypatch, cmd):
    """A2: the PR-text scan runs when the plan has the secret-scan step, and a finding fails the run."""
    root = make_repo(tmp_path / "r", _wf([{"name": "secrets", "run": cmd}]), {"scripts/lint_secrets.py": ""})
    seen: list[str] = []

    def scan(root, base, head_, text):
        seen.append(text)
        return 1

    monkeypatch.setattr(P, "scan_pr_text", scan)
    monkeypatch.setattr(P, "gh_pr_view", lambda root=P.REPO_ROOT: _pr(title="t", body="b"))
    assert P.main(["--no-fetch"], root=root) == 1
    assert seen == ["t\nb"]
    assert _final(capsys.readouterr().out).startswith(f"prepr: FAIL (1): {P.PR_SECRET_SCAN}")


def test_scan_pr_text_uses_lint_secrets_own_range_scan(capsys):
    """A2, against the real lint: HEAD..HEAD is an empty range, so only the PR text is scanned."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    assert P.scan_pr_text(ROOT, head, head, "feat: x\nan ordinary body") == 0
    key = "AKIA" + "Q7ZB3K9XW2M4P6RT"  # built at run time so the secret scan never sees a literal in this file
    assert P.scan_pr_text(ROOT, head, head, f"feat: x\nleaked {key}") == 1
    assert ":pr-text" in capsys.readouterr().err


def test_preflight_refuses_ignored_python_files_in_the_scanned_roots(tmp_path):
    """N1: lint_orphan_modules globs src/ tests/ scripts/; a gitignored *.py there exists here and not in CI."""
    root = make_repo(tmp_path / "r")
    _commit(root, ".gitignore", "gen_*.py\n*.log\n")
    (root / "docs").mkdir()
    (root / "docs/gen_ok.py").write_text("")  # outside the scanned roots
    (root / "src").mkdir()
    (root / "src/x.log").write_text("")  # ignored, not *.py
    _preflight(root)
    (root / "src/gen_orphan.py").write_text("")
    with pytest.raises(P.Refusal, match="1 gitignored"):
        _preflight(root)
    _, _, taints = _preflight(root, allow_dirty=True)
    assert any("gitignored *.py" in t for t in taints)


def test_exit_codes_are_documented_in_the_docstring_and_help(capsys):
    assert P.EXIT_CODES in (P.__doc__ or "")
    for code, word in (("0", "PASS"), ("1", "FAIL"), ("2", "REFUSED"), ("3", "TAINTED-PASS")):
        assert re.search(rf"^  {code}  prepr: {word} ", P.EXIT_CODES, re.M)
    with pytest.raises(SystemExit):
        P.main(["--help"])
    assert "3  prepr: TAINTED-PASS" in capsys.readouterr().out


# ── round 3: S2 classification, refusal paths, editable installs, scrub ────────────────────────────────────────────

SECRET_STEP = {
    "name": "secrets",
    "run": "# scans tests/unit/test_lint_secrets.py too\npython3 scripts/lint_secrets.py\n",
}


@pytest.mark.parametrize(
    "run, scan",
    [
        ("python3 scripts/lint_secrets.py", True),
        ("python scripts/lint_secrets.py", True),
        ("python3 -m scripts.lint_secrets", True),
        ("python3 scripts/lint_secrets.py --all", False),
        ("python3 scripts/lint_secrets.py --staged", False),
        ("# python3 scripts/lint_secrets.py is in a comment\ntrue", False),
        ("true", False),
    ],
)
def test_the_secret_scan_step_is_classified(run, scan):
    assert P.is_secret_scan(run) is scan


@pytest.mark.parametrize(
    "run",
    [
        "echo scripts/lint_secrets.py",
        "./scripts/lint_secrets.py",
        "bash -c 'python3 scripts/lint_secrets.py'",
        "python3 scripts/lint_secrets.py || true",
        "python3 scripts/lint_secrets_v2.py",
    ],
)
def test_an_unclassifiable_lint_secrets_mention_is_a_plan_error(run):
    with pytest.raises(P.PlanError, match="cannot classify"):
        P.build_plan(_wf([{"name": "x", "run": run}]))


def test_the_real_workflow_selects_exactly_one_secret_scan_step():
    """S2: so the PR-text scan is actually triggered on today's workflow (and the title is required for it)."""
    scans = [s.name for s in P.load_plan(ROOT).steps if s.secret_scan]
    assert len(scans) == 1 and scans[0].startswith("Secret scan"), scans
    plan = P.build_plan(_wf([SECRET_STEP]))
    assert plan.needs == frozenset({"title"})


FAKE_SECRETS = {
    "git": "class GitUnavailable(Exception): pass\nclass ConfigError(Exception): pass\n"
    "def scan_range(*a, **k): raise GitUnavailable('no base')\ndef _report(s, l): return 0\n",
    "config": "class GitUnavailable(Exception): pass\nclass ConfigError(Exception): pass\n"
    "def scan_range(*a, **k): raise ConfigError('bad PATTERNS')\ndef _report(s, l): return 0\n",
    "renamed": "class GitMissing(Exception): pass\nclass ConfigError(Exception): pass\n"
    "def scan_range(*a, **k): raise GitMissing('x')\ndef _report(s, l): return 0\n",
    "import": "raise ImportError('broken import')\n",
}


@pytest.mark.parametrize(
    "kind, needle",
    [
        ("git", "GitUnavailable: no base"),
        ("config", "ConfigError: bad PATTERNS"),
        ("renamed", "lacks ['GitUnavailable']"),
        ("import", "ImportError: broken"),
    ],
)
def test_main_refuses_when_the_pr_text_scan_cannot_run(tmp_path, monkeypatch, capsys, kind, needle):
    """Executor/Wire: every way the PR-text scan cannot run is exit 2 with a message, never a traceback or exit 1."""
    monkeypatch.setattr(P, "tool_version", lambda tool, path: {"ruff": "0.14.14", "mypy": "1.20.0"}.get(tool))
    monkeypatch.setattr(P, "platform_note", lambda: None)
    monkeypatch.setattr(P, "gh_pr_view", lambda root=P.REPO_ROOT: _pr(title="t"))
    monkeypatch.setattr(P, "repo_owner", lambda root=P.REPO_ROOT: "dennys246")
    root = make_repo(tmp_path / "r", _wf([SECRET_STEP]), {"scripts/lint_secrets.py": FAKE_SECRETS[kind]})
    assert P.main(["--no-fetch"], root=root) == 2
    captured = capsys.readouterr()
    assert needle in captured.err and _final(captured.out) == "prepr: REFUSED (exit 2)"


@pytest.mark.parametrize(
    "line",
    [
        "pip install -e .",
        "pip install -e.",
        "pip install --editable=.",
        'pip install ".[dev]"',
        "pip install .",
        "pip install ./pkg",
        "pip install /abs/pkg",
        "pip install git+https://example.invalid/r.git",
        "pip install https://example.invalid/x.tar.gz",
        "pip install dist/x-1.0-py3-none-any.whl",
    ],
)
def test_editable_vcs_and_path_installs_are_plan_errors(line):
    with pytest.raises(P.PlanError, match="path/VCS/editable"):
        P.build_plan(_wf([{"name": "x", "run": f"{line}\nruff check src/\n"}]))


def test_scrubbing_virtual_env_keeps_the_steps_python3_in_this_environment():
    """VIRTUAL_ENV is scrubbed; python3 still resolves (PATH) to this interpreter's environment (its pyvenv.cfg)."""
    step = P.Step("lint", "x", "true\n")
    base = dict(os.environ, VIRTUAL_ENV="/nowhere", PYTHONWARNINGS="error", PYTHONSTARTUP="/x", PYTHONUSERBASE="/u")
    env = P.step_env(step, {"title": "", "body": ""}, base=base)
    assert not {"VIRTUAL_ENV", "PYTHONWARNINGS", "PYTHONSTARTUP", "PYTHONHOME", "PYTHONUSERBASE"} & set(env)
    r = subprocess.run(["python3", "-c", "import sys; print(sys.prefix)"], env=env, capture_output=True, text=True)
    assert r.stdout.strip() == sys.prefix


# ── round 4: forks, payload readers, lint_secrets args, branch name ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "pr, needle",
    [
        (_pr(isCrossRepository=True), "fork"),
        (_pr(isCrossRepository=None), "fork"),
        (_pr(headRepositoryOwner={"login": "someone-else"}), "owned by 'someone-else'"),
        (_pr(headRepositoryOwner=None), "owned by '?'"),
    ],
)
def test_a_fork_or_foreign_owner_pr_of_the_same_branch_name_refuses(pr, needle):
    with pytest.raises(P.Refusal, match=needle):
        P.resolve_pr_text(None, None, lambda: pr, BRANCH, OWNER)
    with pytest.raises(P.Refusal, match="origin by '\\?'"):
        P.resolve_pr_text(None, None, lambda: _pr(), BRANCH, lambda: None)  # origin owner unreadable
    assert (
        P.resolve_pr_text(None, None, lambda: _pr(headRepositoryOwner={"login": "Dennys246"}), BRANCH, OWNER)[2]
        is False
    )


@pytest.mark.parametrize(
    "url, owner",
    [
        ("https://github.com/dennys246/Maxim.git", "dennys246"),
        ("git@github.com:dennys246/Maxim.git", "dennys246"),
        ("https://github.com/dennys246/Maxim", "dennys246"),
        ("ssh://git@github.com/dennys246/Maxim.git", "dennys246"),
        ("ssh://git@github.com:22/dennys246/Maxim", "dennys246"),
        ("https://gitlab.com/dennys246/Maxim.git", None),
        ("https://evil.com/github.com/dennys246/Maxim.git", None),  # host anchored
        ("https://github.com.evil.com/dennys246/Maxim.git", None),
        ("git@github-work:dennys246/Maxim.git", None),  # host alias: refused (follow-up)
        (None, None),  # no remote named origin
    ],
)
def test_repo_owner_parses_origin_or_refuses_naming_it(tmp_path, url, owner):
    root = make_repo(tmp_path / "r")
    if url is not None:
        _git(root, "remote", "add", "origin", url)
    if owner is not None:
        assert P.repo_owner(root) == owner
        return
    with pytest.raises(P.Refusal, match="accepted forms") as e:
        P.repo_owner(root)
    assert (url or "no remote named origin") in str(e.value) and "git remote set-url" in str(e.value)


def test_current_branch_is_the_full_name_and_empty_when_detached(tmp_path):
    root = make_repo(tmp_path / "r")
    _git(root, "tag", "feature")  # a tag sharing the branch's name: --abbrev-ref would say "heads/feature"
    assert P.current_branch(root) == "feature"
    _git(root, "checkout", "-q", "--detach")
    assert P.current_branch(root) == ""


def _pr_payload_readers(paths: list[Path]) -> set[str]:
    import ast

    out: set[str] = set()
    for f in paths:
        for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            sub = isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and n.slice.value == "pull_request"
            get = (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "get"
                and bool(n.args)
                and isinstance(n.args[0], ast.Constant)
                and n.args[0].value == "pull_request"
            )
            if sub or get:
                out.add(f.relative_to(ROOT).as_posix())
    return out


def test_lint_secrets_is_the_only_script_reading_the_pr_event_payload():
    """prepr reproduces exactly one pull_request-payload read (lint_secrets' PR-text scan, by a direct call). A new
    script reading `event["pull_request"]` would see nothing under prepr's local mode, so it must be wired in."""
    readers = _pr_payload_readers(sorted((ROOT / "scripts").rglob("*.py")))
    assert readers == {"scripts/lint_secrets.py"}, (
        f"{sorted(readers)}: a script reading the pull_request event payload other than lint_secrets.py is "
        "invisible to prepr (local mode) -- teach prepr to reproduce it, then update this set"
    )


@pytest.mark.parametrize("run", ["python3 scripts/lint_secrets.py --repo .", "python3 scripts/lint_secrets.py --al"])
def test_lint_secrets_arguments_other_than_exact_flags_are_plan_errors(run):
    with pytest.raises(P.PlanError, match="exact flags only"):
        P.build_plan(_wf([{"name": "x", "run": run}]))


def test_a_test_step_naming_test_lint_secrets_is_not_a_secret_scan():
    assert P.is_secret_scan("python -m pytest tests/unit/test_lint_secrets.py -q") is False


@pytest.mark.parametrize(
    "line",
    ["pip install -i https://example.invalid/simple ruff==0.14.14", "pip install --extra-index-url=x mypy==1.20.0"],
)
def test_another_package_index_is_a_plan_error_with_its_own_message(line):
    with pytest.raises(P.PlanError, match="another package index"):
        P.build_plan(_wf([{"name": "x", "run": f"{line}\nruff check src/\n"}]))
