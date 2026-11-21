"""scripts/lint_coverage.py --reuse-decision / --stamp / --missing-floors (#1640).

The decision is driven on a real fixture git repo shaped like CI's pull-request checkout: ``main``, a ``feature``
branch whose old head C1 changed code and whose new head C2 changes only ``scripts/coverage_floors.json``, a
self-``origin`` (so ``git fetch origin <sha>`` is real), and HEAD = the merge of ``main`` and C2, detached (CI's
``refs/pull/N/merge``). The GitHub API and the artifact download are injected fakes: nothing here calls GitHub.
Every refusal must still exit 0 with ``reuse=false`` in ``$GITHUB_OUTPUT``.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import lint_coverage as L

A = "src/maxim/alpha/a.py"
NOW = dt.datetime(2026, 11, 10, 12, 0, tzinfo=dt.timezone.utc)
REPO = "o/r"
JUNIT = '<testsuites><testsuite name="s"><testcase classname="tests.t" name="test_x"/></testsuite></testsuites>\n'


def _git(root: Path, *args: str) -> str:
    env = dict(
        os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t"
    )
    return subprocess.run(["git", *args], cwd=root, env=env, capture_output=True, text=True, check=True).stdout


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _commit(root: Path, msg: str = "c") -> str:
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", msg)
    return _git(root, "rev-parse", "HEAD").strip()


def _floors(alpha=(80.0, 2), **more) -> str:
    def f(v):
        return {"percent": v[0], "missing": v[1]} if v is not None else {"percent": None, "missing": None}

    pk = {"maxim/alpha": alpha, **more}
    return json.dumps(
        {
            "floors_format_version": 2,
            "band": 1.0,
            "min_package_statements": 200,
            "overall": f((70.0, 5)),
            "packages": {k: f(v) for k, v in pk.items()},
            "files": {},
            "retired_files": [],
        },
        indent=1,
    )


def _coverage_json() -> str:
    lines = list(range(1, 11))
    entry = {
        "executed_lines": lines[:8],
        "missing_lines": lines[8:],
        "excluded_lines": [],
        "summary": {"covered_lines": 8, "num_statements": 10, "num_branches": 0, "covered_branches": 0},
    }
    meta = {"format": 3, "version": L.COVERAGE_VERSION, "timestamp": "t", "branch_coverage": True}
    return json.dumps({"meta": meta, "files": {A: entry}, "totals": {}})


def _merge_checkout(root: Path, head: str) -> None:
    _git(root, "checkout", "-q", "--detach", "main")
    _git(root, "merge", "-q", "--no-ff", "-m", "merge", head)


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        _git(root, "init", "-q", "-b", "main")
        _git(root, "config", "commit.gpgsign", "false")
        _git(root, "config", "uploadpack.allowAnySHA1InWant", "true")  # origin is this repo: fetch any sha
        _write(root, ".gitignore", "coverage.json\nfast-suite.xml\nmeasured-tree.json\ncoverage-reused-from.json\n")
        _write(root, "src/maxim/__init__.py", "")
        _write(root, "src/maxim/alpha/__init__.py", "")
        _write(root, A, "".join(f"a{i} = {i}\n" for i in range(10)))
        _write(root, ".github/workflows/test.yml", "on: push\n")
        _write(root, L.FLOORS_REL, _floors())
        self.base = _commit(root, "base")
        _git(root, "checkout", "-q", "-b", "feature")
        _write(root, A, "".join(f"a{i} = {i * 2}\n" for i in range(10)))  # C1: a code change
        self.c1 = _commit(root, "c1")
        _write(root, L.FLOORS_REL, _floors(alpha=(81.0, 2)))  # C2: floors only, a raise
        self.c2 = _commit(root, "c2")
        _git(root, "checkout", "-q", "-b", "forged", self.c1)  # an intermediate push that edited the workflow
        _write(root, ".github/workflows/test.yml", "on: push  # steps rewritten to `true`\n")
        _write(root, L.FLOORS_REL, _floors(alpha=(81.0, 2)))
        self.forged = _commit(root, "forged")
        _git(root, "remote", "add", "origin", str(root))
        _git(root, "fetch", "-q", "origin")
        _merge_checkout(root, self.c1)  # what C1's run measured
        self.c1_tree = L.measured_tree_id(root)
        _merge_checkout(root, self.c2)  # this run's checkout: HEAD^2 == C2
        self.runs: list[dict] = [self.run(1, self.c1)]
        self.steps: dict[int, list[dict]] = {}
        self.artifacts: dict[int, list[dict]] = {}
        self.files: dict[int, dict[str, str]] = {}
        self.calls: list[str] = []

    def run(self, run_id: int, sha: str, *, attempt: int = 1, hours: float = 1.0, **over) -> dict:
        started = (NOW - dt.timedelta(hours=hours)).isoformat().replace("+00:00", "Z")
        return {
            "id": run_id,
            "head_sha": sha,
            "run_attempt": attempt,
            "event": "pull_request",
            "path": ".github/workflows/test.yml",
            "head_branch": "feature",
            "head_repository": {"full_name": REPO},
            "run_started_at": started,
            **over,
        }

    def ok_steps(self) -> list[dict]:
        names = ["Set up job", "Re-use?", "Stamp", L.RUN_SUITE_STEP, L.MEMORYHUB_STEP, "Coverage ratchet"]
        steps = [{"name": n, "number": i + 1, "conclusion": "success"} for i, n in enumerate(names)]
        steps[-1]["conclusion"] = "failure"  # the ratchet failed (bookkeeping): the case #1640 exists for
        return steps

    def artifact_files(self, run: dict, **over: str | None) -> dict[str, str]:
        stamp = {
            "algo": L.STAMP_ALGO,
            "tree_id": self.c1_tree,
            "head_sha": run["head_sha"],
            "merge_sha": "m" * 40,
            "run_id": str(run["id"]),
            "attempt": str(run["run_attempt"]),
        }
        out = {L.COVERAGE_JSON: _coverage_json(), L.JUNIT_XML: JUNIT, L.STAMP_JSON: json.dumps(stamp)}
        out.update(over)
        return {k: v for k, v in out.items() if v is not None}

    def api(self, path: str) -> object:
        self.calls.append(path)
        if "/actions/workflows/test.yml/runs?" in path:
            assert path.startswith(f"repos/{REPO}/") and "event=pull_request" in path and "branch=feature" in path
            return {"workflow_runs": self.runs}
        run_id = int(path.split("/actions/runs/", 1)[1].split("/", 1)[0])
        run = next(r for r in self.runs if r["id"] == run_id)
        if "/jobs?" in path:
            assert f"/attempts/{run['run_attempt']}/" in path  # the LATEST attempt
            return {"jobs": [{"name": "unit-tests", "steps": self.steps.get(run_id, self.ok_steps())}]}
        if path.endswith("/artifacts?per_page=100"):
            default = [{"name": f"coverage-json-{run['run_attempt']}", "expired": False}]
            return {"artifacts": self.artifacts.get(run_id, default)}
        raise AssertionError(path)

    def download(self, run_id: str, name: str, dest: Path) -> Path:
        run = next(r for r in self.runs if r["id"] == int(run_id))
        dest.mkdir(parents=True, exist_ok=True)
        for rel, text in self.files.get(int(run_id), self.artifact_files(run)).items():
            (dest / rel).write_text(text, encoding="utf-8")
        return dest / L.COVERAGE_JSON

    def env(self, **over: str) -> dict[str, str]:
        e = {
            "GITHUB_EVENT_NAME": "pull_request",
            "GITHUB_RUN_ATTEMPT": "1",
            "GITHUB_REPOSITORY": REPO,
            "GITHUB_HEAD_REF": "feature",
            "GITHUB_RUN_ID": "999",
            "GITHUB_OUTPUT": str(self.root.parent / "gh_output"),
            "GITHUB_STEP_SUMMARY": str(self.root.parent / "gh_summary"),
        }
        e.update(over)
        return e

    def decide(self, capsys, *, api=None, download=None, **env: str) -> tuple[int, str, str]:
        out_file = self.root.parent / "gh_output"
        out_file.unlink(missing_ok=True)
        rc = L.reuse_decision_main(
            self.root, env=self.env(**env), api=api or self.api, download=download or self.download, now=NOW
        )
        printed = capsys.readouterr().out
        return rc, out_file.read_text(encoding="utf-8") if out_file.exists() else "", printed


@pytest.fixture
def fx(tmp_path: Path, monkeypatch) -> Fixture:
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)  # base_ref reads it: CI's own event must not leak in
    root = tmp_path / "repo"
    root.mkdir()
    return Fixture(root)


def _refused(fx: Fixture, capsys, needle: str, **kw) -> str:
    rc, output, printed = fx.decide(capsys, **kw)
    assert rc == 0 and output == "reuse=false\n", (output, printed)
    assert "coverage reuse: NO" in printed and needle in printed, printed
    assert not (fx.root / L.COVERAGE_JSON).exists() and not (fx.root / L.REUSED_FROM_JSON).exists()
    return printed


# ── the positive path ────────────────────────────────────────────────────────────────────────────────────────────


def test_a_floors_only_raise_reuses_the_verified_run_and_places_its_files(fx, capsys):
    assert L.measured_tree_id(fx.root) == fx.c1_tree  # the floors file is the only difference, and it is excluded
    rc, output, printed = fx.decide(capsys)
    assert rc == 0 and output == "reuse=true\n", printed
    assert "re-using run 1 attempt 1" in printed
    assert json.loads((fx.root / L.COVERAGE_JSON).read_text()) == json.loads(_coverage_json())
    assert (fx.root / L.JUNIT_XML).read_text() == JUNIT
    pointer = json.loads((fx.root / L.REUSED_FROM_JSON).read_text())
    assert pointer == {"run_id": 1, "attempt": 1, "head_sha": fx.c1, "tree_id": fx.c1_tree}
    assert "from run 1 attempt 1" in (fx.root.parent / "gh_summary").read_text()
    assert L.load_coverage(fx.root)  # the gate's own read accepts what was placed


def test_a_null_floor_added_is_allowed(fx, capsys):
    _git(fx.root, "checkout", "-q", "feature")
    _write(fx.root, L.FLOORS_REL, _floors(alpha=(81.0, 2), **{"maxim/new": None}))
    c3 = _commit(fx.root, "null")
    _git(fx.root, "fetch", "-q", "origin")
    _merge_checkout(fx.root, c3)
    rc, output, printed = fx.decide(capsys)
    assert output == "reuse=true\n", printed


def test_the_newest_qualifying_run_is_chosen_past_a_reuse_run_and_this_run(fx, capsys):
    """The newest run is this run itself (in progress) and the next a re-use run (its suite skipped)."""
    fx.runs = [fx.run(999, fx.c2, hours=0.1), fx.run(2, fx.c1, hours=0.5), fx.run(1, fx.c1, hours=2)]
    fx.steps[2] = [dict(s, conclusion="skipped") if s["name"] == L.RUN_SUITE_STEP else s for s in fx.ok_steps()]
    rc, output, printed = fx.decide(capsys)
    assert output == "reuse=true\n" and "re-using run 1 " in printed and "run 2 does not qualify" in printed
    assert not any("/runs/999/" in c for c in fx.calls)


# ── when re-use is allowed at all ────────────────────────────────────────────────────────────────────────────────


def test_a_push_always_measures(fx, capsys):
    _refused(fx, capsys, "only a pull_request run re-uses", GITHUB_EVENT_NAME="push")


@pytest.mark.parametrize("attempt", ["2", "3", ""])
def test_a_rerun_always_measures(fx, capsys, attempt):
    _refused(fx, capsys, "a re-run always measures", GITHUB_RUN_ATTEMPT=attempt)


@pytest.mark.parametrize(
    "floors, needle",
    [
        (_floors(alpha=(79.0, 2)), "not a raise"),  # a lowered percent
        (_floors(alpha=(81.0, 3)), "not a raise"),  # a raised missing pin
        (_floors(alpha=(81.0, 2), **{"maxim/new": (50.0, 9)}), "a NEW non-null floor"),
        (_floors(alpha=None), "set back to null"),
    ],
)
def test_a_floors_diff_that_is_not_raise_only_measures(fx, capsys, floors, needle):
    _git(fx.root, "checkout", "-q", "feature")
    _write(fx.root, L.FLOORS_REL, floors)
    c3 = _commit(fx.root, "floors")
    _merge_checkout(fx.root, c3)
    _refused(fx, capsys, needle)


def test_the_raise_only_rule_is_judged_against_the_base_not_the_source_run():
    """Delta fold: lower in push 1, half-raise in push 2 is still a lowering against the base."""
    base = L.parse_floors(_floors(alpha=(80.0, 2)))
    assert L.raise_only_problems(base, L.parse_floors(_floors(alpha=(80.0, 2)))) == []
    assert L.raise_only_problems(base, L.parse_floors(_floors(alpha=(85.0, 1)))) == []
    assert L.raise_only_problems(base, L.parse_floors(_floors(alpha=(79.5, 2))))  # half back up from 79: still low
    assert L.raise_only_problems(base, L.parse_floors(_floors(alpha=(80.0, 2), **{"maxim/x": None}))) == []
    removed = json.loads(_floors())
    del removed["packages"]["maxim/alpha"]
    assert "floor removed" in L.raise_only_problems(base, L.parse_floors(json.dumps(removed)))[0]


# ── which run qualifies ──────────────────────────────────────────────────────────────────────────────────────────


def test_the_forging_scenario_a_matching_stamp_with_a_head_diff_beyond_the_floors_file_measures(fx, capsys):
    """Fold 1: an unreviewed intermediate push ran `true` under the real step names and stamped the next tree's id.
    Its stamp matches, its steps are all success, but its head differs from this PR head in the workflow."""
    fx.runs = [fx.run(1, fx.forged)]
    printed = _refused(fx, capsys, "beyond scripts/coverage_floors.json")
    assert ".github/workflows/test.yml" in printed


def test_a_run_older_than_48_hours_does_not_qualify(fx, capsys):
    fx.runs = [fx.run(1, fx.c1, hours=48.5)]
    _refused(fx, capsys, "more than 2 days")


def test_a_run_of_47_hours_qualifies(fx, capsys):
    fx.runs = [fx.run(1, fx.c1, hours=47)]
    assert fx.decide(capsys)[1] == "reuse=true\n"


@pytest.mark.parametrize(
    "over, needle",
    [
        ({"head_repository": {"full_name": "fork/r"}}, "a fork or another repository"),
        ({"head_branch": "other"}, "another branch"),
        ({"event": "push"}, "not a Tests pull_request run"),
        ({"path": ".github/workflows/other.yml"}, "not a Tests pull_request run"),
        ({"head_sha": "zz"}, "no valid head_sha"),
        ({"head_sha": "1" * 40}, "could not be fetched"),
        ({"run_started_at": "yesterday"}, "unreadable run_started_at"),
        ({"run_started_at": "2026-11-10T11:00:00"}, "carries no timezone"),
    ],
)
def test_a_run_from_elsewhere_does_not_qualify(fx, capsys, over, needle):
    fx.runs = [fx.run(1, fx.c1, **over)]
    _refused(fx, capsys, needle)


@pytest.mark.parametrize(
    "step, conclusion, needle",
    [
        (L.RUN_SUITE_STEP, "failure", "is 'failure', not success"),
        (L.RUN_SUITE_STEP, "skipped", "is 'skipped', not success"),
        (L.MEMORYHUB_STEP, "failure", "is 'failure', not success"),
        ("Stamp", "failure", "is 'failure', not success"),
    ],
)
def test_every_step_through_the_suite_and_memoryhub_must_succeed(fx, capsys, step, conclusion, needle):
    fx.steps[1] = [dict(s, conclusion=conclusion) if s["name"] == step else s for s in fx.ok_steps()]
    _refused(fx, capsys, needle)


def test_a_run_without_the_memoryhub_step_does_not_qualify(fx, capsys):
    fx.steps[1] = [dict(s, conclusion="success") for s in fx.ok_steps() if s["name"] != L.MEMORYHUB_STEP]
    _refused(fx, capsys, "did not both run")
    fx.steps[1] = fx.ok_steps()[:3]  # the attempt ended before its suite ran
    _refused(fx, capsys, "did not both run")


def test_the_latest_attempts_artifact_is_required(fx, capsys):
    fx.runs = [fx.run(1, fx.c1, attempt=2)]
    fx.artifacts[1] = [{"name": "coverage-json-1", "expired": False}]  # attempt 1's file only
    _refused(fx, capsys, "no unexpired, finalized artifact 'coverage-json-2'")


def test_an_expired_artifact_does_not_qualify(fx, capsys):
    fx.artifacts[1] = [{"name": "coverage-json-1", "expired": True}]
    _refused(fx, capsys, "no unexpired, finalized artifact")


@pytest.mark.parametrize("missing", [L.COVERAGE_JSON, L.JUNIT_XML, L.STAMP_JSON])
def test_an_artifact_lacking_a_file_does_not_qualify(fx, capsys, missing):
    fx.files[1] = fx.artifact_files(fx.runs[0], **{missing: None})
    _refused(fx, capsys, "its artifact lacks")


@pytest.mark.parametrize(
    "field, value, needle",
    [
        ("tree_id", "ls-tree-r-z-minus-floors-sha256-v1:" + "0" * 64, "is not this tree's"),  # main moved
        ("algo", "v0", "is not this tree's"),
        ("head_sha", "2" * 40, "is not this tree's"),
        ("run_id", "7", "is not this tree's"),
        ("attempt", "2", "is not this tree's"),
    ],
)
def test_a_stamp_of_another_tree_or_run_does_not_qualify(fx, capsys, field, value, needle):
    stamp = json.loads(fx.artifact_files(fx.runs[0])[L.STAMP_JSON])
    stamp[field] = value
    fx.files[1] = fx.artifact_files(fx.runs[0], **{L.STAMP_JSON: json.dumps(stamp)})
    _refused(fx, capsys, needle)


@pytest.mark.parametrize(
    "rel, text, needle",
    [
        (L.COVERAGE_JSON, _coverage_json().replace(L.COVERAGE_VERSION, "7.0.0"), "does not validate"),
        (L.COVERAGE_JSON, "{not json", "does not validate"),
        (L.COVERAGE_JSON, _coverage_json().replace(A, "src/maxim/alpha/gone.py"), "names files this checkout lacks"),
        (L.JUNIT_XML, "<testsuites><oops", "does not validate"),
        (L.JUNIT_XML, "<testsuites/>", "holds no testcase"),
        (L.STAMP_JSON, "[]", "is not this tree's"),
    ],
)
def test_an_artifact_that_does_not_validate_does_not_qualify(fx, capsys, rel, text, needle):
    fx.files[1] = fx.artifact_files(fx.runs[0], **{rel: text})
    _refused(fx, capsys, needle)


# ── every doubt is a full run, never a red job ───────────────────────────────────────────────────────────────────


def test_an_api_error_is_a_full_run(fx, capsys):
    def api(path):
        raise L.GitUnavailable("gh api: HTTP 502")

    _refused(fx, capsys, "HTTP 502", api=api)


def test_a_candidates_own_download_or_api_error_refuses_only_that_candidate(fx, capsys):
    fx.runs = [fx.run(2, fx.c1, hours=0.5), fx.run(1, fx.c1, hours=2)]

    def download(run_id, name, dest):
        if run_id == "2":
            raise L.GitUnavailable("gh run download 2: HTTP 500")
        return fx.download(run_id, name, dest)

    rc, output, printed = fx.decide(capsys, download=download)
    assert output == "reuse=true\n" and "run 2 does not qualify: GitUnavailable" in printed, printed

    def api(path):
        if "/runs/2/" in path:
            raise L.GitUnavailable("gh api: HTTP 502")
        return fx.api(path)

    rc, output, printed = fx.decide(capsys, api=api)
    assert output == "reuse=true\n" and "HTTP 502" in printed, printed


def test_a_head_is_fetched_only_when_absent_and_at_most_once(fx, capsys, monkeypatch):
    fx.runs = [fx.run(3, "1" * 40, hours=0.2), fx.run(2, "1" * 40, hours=0.5), fx.run(1, fx.c1, hours=2)]
    fetches: list[tuple[str, ...]] = []
    real_git = L.git

    def spy(root, *args, **kw):
        if "fetch" in args:
            fetches.append(args)
        return real_git(root, *args, **kw)

    monkeypatch.setattr(L, "git", spy)
    rc, output, printed = fx.decide(capsys)
    assert output == "reuse=true\n", printed
    assert len(fetches) == 1 and fetches[0][-1] == "1" * 40  # c1 is in the clone: never fetched; 111... once
    assert "already failed" in printed


def test_the_decision_stops_at_its_deadline(fx, capsys):
    ticks = iter([0.0, L.REUSE_DEADLINE_SECONDS + 1])
    out_file = fx.root.parent / "gh_output"
    rc = L.reuse_decision_main(
        fx.root, env=fx.env(), api=fx.api, download=fx.download, now=NOW, clock=lambda: next(ticks)
    )
    printed = capsys.readouterr().out
    assert rc == 0 and out_file.read_text() == "reuse=false\n"
    assert "300 s decision deadline passed after 0 candidate(s)" in printed


def test_an_unexpected_error_is_a_full_run_and_exit_0(fx, capsys):
    def download(run_id, name, dest):
        raise RuntimeError("boom")

    _refused(fx, capsys, "unexpected RuntimeError: boom", download=download)


def test_a_head_that_is_not_a_merge_commit_measures(fx, capsys):
    _git(fx.root, "checkout", "-q", "feature")
    _refused(fx, capsys, "not a pull-request merge commit")


def test_no_earlier_run_measures(fx, capsys):
    fx.runs = []
    _refused(fx, capsys, "none of the 0 earlier Tests run(s)")


def test_a_failure_while_placing_leaves_nothing_behind(fx, capsys, monkeypatch):
    def bad_load(root, path=None):
        if path is None:
            raise L.MeasurementError("strict read failed")
        return {A: None}

    monkeypatch.setattr(L, "load_coverage", bad_load)
    _refused(fx, capsys, "strict read failed")
    assert not (fx.root / L.JUNIT_XML).exists()


def test_reuse_decision_via_main_without_github_env_is_a_full_run(fx, capsys, monkeypatch):
    for k in ("GITHUB_EVENT_NAME", "GITHUB_OUTPUT", "GITHUB_RUN_ATTEMPT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(L, "REPO_ROOT", fx.root)
    assert L.main(["--reuse-decision"]) == 0
    assert "coverage reuse: NO" in capsys.readouterr().out


# ── --stamp ──────────────────────────────────────────────────────────────────────────────────────────────────────


def test_stamp_names_the_pr_head_and_the_tree(fx, capsys):
    env = {"GITHUB_EVENT_NAME": "pull_request", "GITHUB_RUN_ID": "5", "GITHUB_RUN_ATTEMPT": "1"}
    assert L.stamp_main(fx.root, env) == 0
    doc = json.loads((fx.root / L.STAMP_JSON).read_text())
    assert doc["head_sha"] == fx.c2 and doc["merge_sha"] == _git(fx.root, "rev-parse", "HEAD").strip()
    assert doc["tree_id"] == fx.c1_tree and doc["algo"] == L.STAMP_ALGO
    assert (doc["run_id"], doc["attempt"]) == ("5", "1")
    _git(fx.root, "checkout", "-q", "feature")  # a push: HEAD is the measured commit
    assert L.stamp_main(fx.root, {"GITHUB_EVENT_NAME": "push"}) == 0
    assert json.loads((fx.root / L.STAMP_JSON).read_text())["head_sha"] == fx.c2


def test_stamp_never_turns_a_full_run_red(tmp_path, capsys):
    """Exec 5: a git or OS error warns and exits 0 with NO stamp (that run is then no source: fail-closed)."""
    (tmp_path / L.STAMP_JSON).write_text("stale")
    assert L.stamp_main(tmp_path, {"GITHUB_EVENT_NAME": "pull_request"}) == 0  # not a git repo
    assert not (tmp_path / L.STAMP_JSON).exists()
    assert "--stamp WARNING" in capsys.readouterr().err


def test_stamp_on_a_pull_request_without_a_merge_head_writes_nothing(fx, capsys):
    _git(fx.root, "checkout", "-q", "feature")  # one parent: no HEAD^2
    assert L.stamp_main(fx.root, {"GITHUB_EVENT_NAME": "pull_request"}) == 0
    assert not (fx.root / L.STAMP_JSON).exists()


def test_the_tree_id_ignores_only_the_floors_file(fx):
    before = L.measured_tree_id(fx.root)
    _write(fx.root, L.FLOORS_REL, _floors(alpha=(99.0, 0)))
    _git(fx.root, "commit", "-q", "-am", "floors")
    assert L.measured_tree_id(fx.root) == before
    _write(fx.root, "src/maxim/alpha/b.py", "b = 1\n")
    _commit(fx.root, "code")
    assert L.measured_tree_id(fx.root) != before


# ── --missing-floors (prepr) ─────────────────────────────────────────────────────────────────────────────────────

PARITY_SRC = {
    "src/maxim/__init__.py": '"""Package docstring."""\n',
    "src/maxim/top.py": "import os\n\nX = 1; Y = 2\n\n\ndef f(a,\n      b):\n    return (a +\n            b)\n",
    "src/maxim/alpha/__init__.py": "",
    "src/maxim/alpha/a.py": (
        '"""Module docstring."""\n'
        "from __future__ import annotations\n\n"
        "from typing import TYPE_CHECKING\n\n"
        "if TYPE_CHECKING:\n    from os import PathLike\n\n"
        "import functools\n\n\n"
        "@functools.lru_cache\n"
        "def g(x):\n"
        '    """Docstring."""\n'
        "    try:\n        return x + 1\n    except TypeError:\n        return None\n    finally:\n        pass\n\n\n"
        "def hidden():  # pragma: no cover\n    return 1\n\n\n"
        "class C:\n    a = [i for i in range(3)]\n\n    def m(self):\n        if self.a:\n            return 1\n"
        "        else:\n            return 2\n\n\n"
        "def stub():\n    raise NotImplementedError\n\n\n"
        'if __name__ == "__main__":\n    g(1)\n'
    ),
    "src/maxim/beta/__init__.py": "",
    "src/maxim/beta/b.py": "def h():\n    while True:\n        break\n    return [\n        1,\n        2,\n    ]\n",
    "src/maxim/beta/sub/__init__.py": "",
    "src/maxim/beta/sub/c.py": "Z = {\n    'a': 1,\n}\n",
    "src/maxim/ns/x.py": "N = 1\n",  # no __init__.py and never imported: coverage does not report it
    "src/maxim/omitted.py": "O = 1\nP = 2\n",
}


def _parity_tree(root: Path) -> None:
    omit = [*L.FIXED_OMIT, "src/maxim/omitted.py"]
    _write(
        root,
        "pyproject.toml",
        '[project]\nname = "fixture"\n\n'
        f'[tool.coverage.run]\nsource = ["src/maxim"]\nomit = {json.dumps(omit)}\nbranch = true\n\n'
        f"[tool.coverage.report]\nexclude_lines = {json.dumps(L.CANONICAL_EXCLUDE_LINES)}\nshow_missing = true\n",
    )
    for rel, text in PARITY_SRC.items():
        _write(root, rel, text)
    _write(root, "run_it.py", "import maxim.alpha.a\nimport maxim.top\n")  # beta is never imported


def test_missing_floors_counts_statements_exactly_as_ci_coverage_reports_them(tmp_path):
    """The parity test: coverage itself measures the fixture tree (a subprocess, like CI's suite), and every
    package's num_statements summed must equal --missing-floors' count, executed and unexecuted files alike."""
    root = tmp_path / "tree"
    _parity_tree(root)
    env = {k: v for k, v in os.environ.items() if not k.startswith("COV")}
    env["PYTHONPATH"] = str(root / "src")
    for args in (["run", "run_it.py"], ["json", "-o", "coverage.json"]):
        r = subprocess.run([sys.executable, "-m", "cover" + "age", *args], cwd=root, env=env, capture_output=True)
        assert r.returncode == 0, r.stderr.decode()
    files = json.loads((root / "coverage.json").read_text())["files"]
    ci: dict[str, int] = {}
    for rel, f in files.items():
        ci[L.package_of(rel)] = ci.get(L.package_of(rel), 0) + f["summary"]["num_statements"]
    assert "src/maxim/beta/b.py" in files and "src/maxim/ns/x.py" not in files  # the fixture tests both rules
    ours = L.package_statements(root, {"src/maxim/omitted.py"}, L.coverage_statement_counter(root))
    assert ours == ci
    assert ours["maxim/alpha"] != sum(  # the AST count the split rule uses would NOT have matched
        L.statement_count((root / r).read_text()) for r in PARITY_SRC if r.startswith("src/maxim/alpha/")
    )


def test_missing_floors_main_lists_a_package_over_n_without_a_floor(tmp_path, monkeypatch, capsys):
    root = tmp_path / "tree"
    _parity_tree(root)
    _write(root, L.FLOORS_REL, _floors())  # maxim/alpha floored; beta and <root> are not
    _write(root, L.EXCLUSIONS_REL, json.dumps({"exclusions_format_version": 1, "exclusions": [], "ledger": []}))
    monkeypatch.setattr(L, "MIN_PACKAGE_STATEMENTS", 3)
    counts = {"maxim/alpha": 100, "maxim/beta": 3, L.ROOT_PACKAGE: 2}
    seen: list[str] = []

    def count(rel: str) -> int:
        seen.append(rel)
        pkg = L.package_of(rel)
        return counts[pkg] if rel.endswith(("a.py", "b.py", "top.py")) else 0

    assert L.missing_floors_main(root, count) == 1
    err = capsys.readouterr().err
    assert "maxim/beta: 3 statements (>= 3) and no floor" in err  # at N exactly: CI fails it
    assert "maxim/alpha" not in err and "<root>" not in err  # floored (hysteresis), and under N
    assert "src/maxim/ns/x.py" not in seen  # not reported by coverage, never counted
    _write(root, L.FLOORS_REL, _floors(**{"maxim/beta": None}))  # a committed null is the NEW-floor path
    assert L.missing_floors_main(root, count) == 0


def test_missing_floors_main_cannot_count_without_its_files(tmp_path, capsys):
    assert L.missing_floors_main(tmp_path, lambda rel: 0) == 2
    assert "cannot count" in capsys.readouterr().err
