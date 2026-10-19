"""scripts/check_release_tag.py: the publish workflow's own gates (#1372; design pass SF1, SF2, SF3, SF8, SF9).

Every GitHub API and git call is faked; each refusal below is a way a tag push could otherwise publish the wrong
code, or publish without the owner's approval."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import check_release_tag as C

REPO = "dennys246/Maxim"
SHA = "c" * 40
TAG_OBJ = "t" * 40
EMAIL = "owner@example.org"


def _root(tmp_path: Path, pyproject: str = "1.3.3", init: str = "1.3.3") -> Path:
    (tmp_path / "src" / "maxim").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(f'[project]\nname = "pymaxim"\nversion = "{pyproject}"\n')
    (tmp_path / "src" / "maxim" / "__init__.py").write_text(f'__version__ = "{init}"\n')
    return tmp_path


def _api(*, ref_type="tag", verified=True, reason="valid", email=EMAIL, target=SHA, tag_obj=TAG_OBJ):
    def api(path):
        if path == f"repos/{REPO}/git/ref/tags/v1.3.3":
            return {"object": {"type": ref_type, "sha": tag_obj if ref_type == "tag" else SHA}}
        if path == f"repos/{REPO}/git/tags/{tag_obj}":
            return {
                "verification": {"verified": verified, "reason": reason},
                "tagger": {"email": email},
                "object": {"type": "commit", "sha": target},
            }
        raise AssertionError(path)

    return api


def _git(returncode=0):
    return lambda *a: subprocess.CompletedProcess(a, returncode, "", "boom" if returncode > 1 else "")


def _tag(tmp_path, api=None, run_git=None, **root):
    return C.tag_problems(
        "v1.3.3",
        SHA,
        repo=REPO,
        tagger_email=EMAIL,
        api=api or _api(),
        run_git=run_git or _git(),
        root=_root(tmp_path, **root),
    )


def test_the_release_tag_passes(tmp_path):
    assert _tag(tmp_path) == ([], TAG_OBJ)


@pytest.mark.parametrize("root", [{"pyproject": "1.3.2"}, {"init": "1.3.2"}])
def test_a_tag_that_does_not_name_the_trees_version_is_refused(tmp_path, root):
    """SF3: lint_version_sync compares the repo's strings to each other; nothing compared them to the tag."""
    assert any("must all name one version" in p for p in _tag(tmp_path, **root)[0])


def test_a_lightweight_tag_is_refused(tmp_path):
    problems, tag_sha = _tag(tmp_path, api=_api(ref_type="commit"))
    assert tag_sha is None and any("not an annotated tag" in p for p in problems)


@pytest.mark.parametrize("kw", [{"verified": False}, {"reason": "unknown_key"}, {"email": "someone@else.org"}])
def test_an_unsigned_or_foreign_tag_is_refused(tmp_path, kw):
    assert _tag(tmp_path, api=_api(**kw))[0]


def test_a_tag_that_peels_to_another_commit_is_refused(tmp_path):
    assert any("this run checked out" in p for p in _tag(tmp_path, api=_api(target="d" * 40))[0])


def test_a_commit_not_on_main_is_refused(tmp_path):
    assert any("not on origin/main" in p for p in _tag(tmp_path, run_git=_git(1))[0])


def test_an_unreadable_ancestry_is_unverified(tmp_path):
    with pytest.raises(C.Unverified):
        _tag(tmp_path, run_git=_git(128))


def test_still_passes_when_nothing_moved():
    assert C.still_problems("v1.3.3", TAG_OBJ, SHA, repo=REPO, api=_api()) == []


def test_a_re_tag_during_the_approval_wait_is_refused():
    """SF2: run 1 waiting at the gate must not publish after the owner re-tagged for run 2."""
    moved = _api(tag_obj="n" * 40, target="e" * 40)
    problems = C.still_problems("v1.3.3", TAG_OBJ, SHA, repo=REPO, api=moved)
    assert any("re-tagged" in p for p in problems) and any("no longer peels" in p for p in problems)


def _env_api(env, policies=None):
    def api(path):
        if path.endswith("/deployment-branch-policies"):
            return {"branch_policies": policies if policies is not None else [{"name": "v*", "type": "tag"}]}
        return env

    return api


GOOD_ENV = {
    "protection_rules": [{"type": "required_reviewers", "reviewers": [{"type": "User", "reviewer": {"login": "o"}}]}],
    "deployment_branch_policy": {"protected_branches": False, "custom_branch_policies": True},
}


def test_a_protected_pypi_environment_passes():
    assert C.environment_problems(repo=REPO, api=_env_api(GOOD_ENV)) == []


@pytest.mark.parametrize(
    "env,policies,expect",
    [
        ({**GOOD_ENV, "protection_rules": []}, None, "no required reviewer"),
        ({**GOOD_ENV, "deployment_branch_policy": None}, None, "not limited"),
        (GOOD_ENV, [{"name": "*", "type": "branch"}], "expected exactly"),
        (GOOD_ENV, [{"name": "v*", "type": "tag"}, {"name": "main", "type": "branch"}], "expected exactly"),
    ],
)
def test_an_unprotected_pypi_environment_is_refused(env, policies, expect):
    """SF1: GitHub re-creates a missing environment unprotected, and PyPI checks only its name."""
    assert any(expect in p for p in C.environment_problems(repo=REPO, api=_env_api(env, policies)))


def test_release_notes_must_exist_and_use_absolute_links(tmp_path):
    """SF8: checked before the irreversible upload, not by the post-publication audit."""
    ann = tmp_path / "docs" / "announcements"
    ann.mkdir(parents=True)
    assert any("no release notes" in p for p in C.notes_problems("1.3.3", tmp_path))
    (ann / "release_1_3_3.md").write_text("See [the guide](../publication_guide.md).\n")
    assert any("repo-relative" in p for p in C.notes_problems("1.3.3", tmp_path))
    (ann / "release_1_3_3.md").write_text("See [the guide](https://github.com/dennys246/Maxim/blob/main/x.md).\n")
    assert C.notes_problems("1.3.3", tmp_path) == []


def test_an_api_failure_is_unverified_not_a_pass(monkeypatch, capsys):
    """Through the real gh_api: a non-zero `gh` exit is Unverified, and main maps it to 2."""
    failed = subprocess.CompletedProcess(("gh",), 1, "", "HTTP 404")
    monkeypatch.setattr(C.subprocess, "run", lambda *a, **k: failed)
    assert C.main(["environment", "--repo", REPO]) == 2
    assert "Unverified" in capsys.readouterr().err


def test_a_tag_naming_another_version_than_a_consistent_tree_is_refused(tmp_path):
    """SF3, the tag clause alone: pyproject and __init__ agree with each other, not with the tag."""
    assert any("must all name one version" in p for p in _tag(tmp_path, pyproject="1.3.2", init="1.3.2")[0])


def test_a_tag_pointing_at_a_non_commit_is_refused(tmp_path):
    api = _api()

    def tree_target(path):
        out = api(path)
        if "/git/tags/" in path:
            out["object"] = {"type": "tree", "sha": SHA}
        return out

    assert any("points at tree" in p for p in _tag(tmp_path, api=tree_target)[0])


def test_a_re_tag_to_a_lightweight_tag_during_the_wait_is_refused():
    assert C.still_problems("v1.3.3", TAG_OBJ, SHA, repo=REPO, api=_api(ref_type="commit"))


@pytest.mark.parametrize(
    "argv,fake,rc",
    [
        (["still", "--tag", "v1.3.3", "--tag-object", TAG_OBJ, "--sha", SHA, "--repo", REPO], "still_problems", 1),
        (["environment", "--repo", REPO], "environment_problems", 1),
        (["notes", "--version", "1.3.3"], "notes_problems", 1),
    ],
)
def test_main_exits_1_on_any_problem(monkeypatch, argv, fake, rc):
    """SF-d: a subcommand that found problems must not pass its workflow step."""
    monkeypatch.setattr(C, fake, lambda *a, **k: ["a problem"])
    assert C.main(argv) == rc


def test_main_prints_the_tag_object_for_the_workflow_output(monkeypatch, capsys):
    monkeypatch.setattr(C, "tag_problems", lambda *a, **k: ([], TAG_OBJ))
    argv = ["tag", "--tag", "v1.3.3", "--sha", SHA, "--repo", REPO, "--tagger-email", EMAIL]
    assert C.main(argv) == 0
    assert f"tag_object={TAG_OBJ}" in capsys.readouterr().out.splitlines()
    monkeypatch.setattr(C, "tag_problems", lambda *a, **k: (["bad"], TAG_OBJ))
    assert C.main(argv) == 1


def _rules_api(rulesets):
    def api(path):
        if path.endswith("/rulesets"):
            return [{"id": i, "target": r["target"], "enforcement": r["enforcement"]} for i, r in enumerate(rulesets)]
        return rulesets[int(path.rsplit("/", 1)[1])]

    return api


def _ruleset(target="tag", enforcement="active", include=("refs/tags/v*",), rules=("creation", "update", "deletion")):
    return {
        "target": target,
        "enforcement": enforcement,
        "conditions": {"ref_name": {"include": list(include)}},
        "rules": [{"type": t} for t in rules],
    }


def test_a_v_tag_ruleset_passes_beside_others():
    rulesets = [
        _ruleset(target="branch", include=("~DEFAULT_BRANCH",)),
        _ruleset(include=("refs/tags/o19/**",)),
        _ruleset(),
    ]
    assert C.ruleset_problems(repo=REPO, api=_rules_api(rulesets)) == []


@pytest.mark.parametrize(
    "ruleset",
    [
        _ruleset(enforcement="evaluate"),
        _ruleset(include=("refs/tags/o19/**",)),
        _ruleset(rules=("creation", "deletion")),
        _ruleset(target="branch"),
    ],
)
def test_no_effective_v_tag_ruleset_is_refused(ruleset):
    """Review S1: the ruleset is #1372's own Done-when item and had no check."""
    assert C.ruleset_problems(repo=REPO, api=_rules_api([ruleset]))
