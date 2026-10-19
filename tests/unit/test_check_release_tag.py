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


def test_an_api_failure_is_unverified_not_a_pass(monkeypatch):
    def down(_path):
        raise C.Unverified("gh: HTTP 404")

    monkeypatch.setattr(C, "gh_api", down)
    monkeypatch.setattr(C, "environment_problems", lambda **kw: C.environment_problems(api=down, **kw))
    assert C.main(["environment", "--repo", REPO]) == 2
