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


def test_network_failure_is_exit_2_not_pass(monkeypatch) -> None:
    def boom(timeout: float = 30.0):
        raise A.AuditError("network down")

    monkeypatch.setattr(A, "_pypi_files", boom)
    assert A.audit_releases({}) == 2


def test_empty_pypi_is_exit_2_not_pass(monkeypatch) -> None:
    monkeypatch.setattr(A, "_pypi_files", lambda timeout=30.0: {})
    assert A.audit_releases({}) == 2


def test_grandfather_reasons_are_real_versions() -> None:
    for v in A.GRANDFATHERED_RELEASES:
        assert A.GRANDFATHERED_RELEASES[v].strip(), v


def test_announcement_notes_sources_have_no_relative_links() -> None:
    """The next release's --notes-file must publish clean; the 1.1.0 source was rewritten 2026-08-29."""
    src = A.REPO / A.ANNOUNCEMENTS / "release_1_1_0.md"
    assert src.exists()
    assert A._REL_LINK.findall(src.read_text()) == []
