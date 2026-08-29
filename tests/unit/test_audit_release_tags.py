"""Release-object audit (scripts/audit_release_tags.py --check-releases, roadmap 1.1.x item 16.1).

No network: `_gh_release` and `_pypi_files` are monkeypatched. Verified to fail on the
pre-fix script, which had no release-object audit at all (AttributeError).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import audit_release_tags as A

PYPI = {"pymaxim-1.2.3-py3-none-any.whl": "a" * 64, "pymaxim-1.2.3.tar.gz": "b" * 64}


def _release(assets: list[tuple[str, str]], body: str = "See https://example.com") -> dict:
    return {"tagName": "v1.2.3", "body": body, "assets": [{"name": n, "digest": f"sha256:{d}"} for n, d in assets]}


@pytest.fixture(autouse=True)
def _gh_probe_ok(monkeypatch):
    """The repo-level positive control is network; the unit tests exercise the logic."""
    monkeypatch.setattr(A, "_gh_probe", lambda: None)


@pytest.fixture
def gh(monkeypatch):
    state: dict = {"release": _release([(n, d) for n, d in PYPI.items()])}
    monkeypatch.setattr(A, "_gh_release", lambda tag: state["release"])
    monkeypatch.setattr(A, "_git", lambda *args: "v1.2.3\n" if args[0] == "tag" else "")
    return state


def test_complete_release_has_no_problems(gh) -> None:
    assert A.release_problems("1.2.3", PYPI) == []


def test_missing_release_object_is_a_problem(gh) -> None:
    gh["release"] = None
    problems = A.release_problems("1.2.3", PYPI)
    assert len(problems) == 1 and "no GitHub Release" in problems[0]


def test_zero_assets_is_a_problem(gh) -> None:
    gh["release"] = _release([])
    problems = A.release_problems("1.2.3", PYPI)
    assert any("0 asset(s)" in p for p in problems)
    assert sum("missing the PyPI file" in p for p in problems) == 2


def test_sha256_mismatch_is_a_problem(gh) -> None:
    gh["release"] = _release([("pymaxim-1.2.3-py3-none-any.whl", "c" * 64), ("pymaxim-1.2.3.tar.gz", "b" * 64)])
    problems = A.release_problems("1.2.3", PYPI)
    assert any("NOT the published artifact" in p for p in problems)


def test_asset_without_digest_cannot_be_verified(gh) -> None:
    gh["release"] = {"tagName": "v1.2.3", "body": "", "assets": [{"name": n} for n in PYPI]}
    problems = A.release_problems("1.2.3", PYPI)
    assert sum("exposes no digest" in p for p in problems) == 2


def test_relative_links_in_notes_are_a_problem(gh) -> None:
    gh["release"] = _release(list(PYPI.items()), body="[changelog](../../CHANGELOG.md) and [ok](https://x.dev)")
    problems = A.release_problems("1.2.3", PYPI)
    assert any("repo-relative link" in p and "404" in p for p in problems)


def test_missing_tag_is_a_problem(gh, monkeypatch) -> None:
    monkeypatch.setattr(A, "_git", lambda *args: "")
    assert any("there is no v1.2.3 tag" in p for p in A.release_problems("1.2.3", PYPI))


def test_notes_source_file_is_checked(gh, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(A, "REPO", tmp_path)
    (tmp_path / A.ANNOUNCEMENTS).mkdir(parents=True)
    (tmp_path / A.ANNOUNCEMENTS / "release_1_2_3.md").write_text("[exp](../experiments/52_nurture.md)\n")
    assert any("--notes-file" in p for p in A.release_problems("1.2.3", PYPI))


def test_audit_grandfathers_by_explicit_list_and_flags_stale_entries(gh, monkeypatch, capsys) -> None:
    monkeypatch.setattr(A, "_pypi_files", lambda timeout=30.0: {"1.2.3": PYPI})
    gh["release"] = _release([])  # failing
    assert A.audit_releases({"1.2.3": "the incident"}) == 0
    out = capsys.readouterr().out
    assert "GRANDFATHERED (still failing)" in out and "the incident" in out
    gh["release"] = _release(list(PYPI.items()))  # now clean → the entry is stale
    assert A.audit_releases({"1.2.3": "the incident"}) == 1
    assert "now PASSES" in capsys.readouterr().err
    assert A.audit_releases({"9.9.9": "gone"}) == 1
    assert "PyPI does not serve" in capsys.readouterr().err


def test_repo_level_gh_failure_is_exit_2_not_a_grandfathered_pass(monkeypatch) -> None:
    """A 404 for the whole repo used to read as 'no Release on every tag' — with every
    version grandfathered that is a green run reporting nothing (the review BLOCKER)."""

    def boom() -> None:
        raise A.AuditError("HTTP 404: Not Found")

    monkeypatch.setattr(A, "_gh_probe", boom)
    monkeypatch.setattr(A, "_pypi_files", lambda timeout=30.0: {"1.2.3": PYPI})
    assert A.audit_releases({"1.2.3": "the incident"}) == 2


def test_gh_release_only_swallows_the_literal_release_not_found(monkeypatch) -> None:
    calls = {}

    class _Proc:
        returncode = 1
        stdout = ""

        def __init__(self, stderr: str) -> None:
            self.stderr = stderr

    monkeypatch.setattr(A.subprocess, "run", lambda *a, **k: _Proc(calls["stderr"]))
    calls["stderr"] = "release not found"
    assert A._gh_release("v1.2.3") is None
    calls["stderr"] = "HTTP 404: Not Found (repository not found)"
    with pytest.raises(A.AuditError):
        A._gh_release("v1.2.3")


def test_network_failure_is_exit_2_not_pass(monkeypatch) -> None:
    def boom(timeout: float = 30.0):
        raise A.AuditError("network down")

    monkeypatch.setattr(A, "_pypi_files", boom)
    assert A.audit_releases({}) == 2


def test_empty_pypi_is_exit_2_not_pass(monkeypatch) -> None:
    monkeypatch.setattr(A, "_pypi_files", lambda timeout=30.0: {})
    assert A.audit_releases({}) == 2


def test_grandfather_keys_parse_as_versions_and_carry_a_reason() -> None:
    import re as _re

    for v, reason in A.GRANDFATHERED_RELEASES.items():
        assert _re.fullmatch(r"\d+\.\d+\.\d+(rc\d+)?", v), v
        assert len(reason.strip()) > 30, f"{v}: a grandfather entry needs a real reason, not a label"


# Backfills describe releases already published with the dead links; they are
# grandfathered in the audit and are not re-published from these files.
_BACKFILLS = {"release_1_0_9_backfill.md", "release_1_1_0rc1_backfill.md"}


def test_every_publishable_announcement_source_has_no_relative_links() -> None:
    """Whatever `gh release create --notes-file` publishes NEXT must be clean — checked for
    every release_*.md, not just 1.1.0, so release_1_1_1.md cannot ship with dead links."""
    sources = [p for p in (A.REPO / A.ANNOUNCEMENTS).glob("release_*.md") if p.name not in _BACKFILLS]
    assert sources, "no release notes sources found — the assertion would be vacuous"
    for src in sources:
        assert A._REL_LINK.findall(src.read_text()) == [], f"{src.name} carries repo-relative links"
