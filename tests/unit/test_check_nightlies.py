"""The release procedure reads the nightlies (roadmap 1.3.x): scripts/check_nightlies.py.

The rule fails CLOSED -- no run, a stale run, no nightly jobs, or an unreadable API is never a pass."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from scripts import check_nightlies as N

NOW = datetime(2026, 9, 27, 12, tzinfo=timezone.utc)
FRESH = {"databaseId": 1, "createdAt": "2026-09-27T09:00:00Z"}
DAY = timedelta(hours=48)


def _jobs(**conclusions: str) -> list[dict]:
    return [{"name": name, "conclusion": c} for name, c in conclusions.items()]


def test_every_nightly_green_and_recent_is_green() -> None:
    jobs = _jobs(**{"Model-cache tests (nightly)": "success", "Slow tests (nightly)": "success", "lint": "failure"})
    assert N.nightly_verdict(FRESH, jobs, now=NOW, max_age=DAY) == []  # only (nightly) jobs are read


def test_a_red_nightly_refuses() -> None:
    jobs = _jobs(**{"Model-cache tests (nightly)": "failure", "Slow tests (nightly)": "success"})
    assert N.nightly_verdict(FRESH, jobs, now=NOW, max_age=DAY) == ["Model-cache tests (nightly): failure"]


@pytest.mark.parametrize("conclusion", ["cancelled", "skipped", None])
def test_anything_but_success_refuses(conclusion) -> None:
    jobs = [{"name": "Slow tests (nightly)", "conclusion": conclusion, "status": "completed"}]
    assert N.nightly_verdict(FRESH, jobs, now=NOW, max_age=DAY)


def test_no_run_a_stale_run_or_no_nightly_jobs_refuses() -> None:
    green = _jobs(**{"Slow tests (nightly)": "success"})
    assert N.nightly_verdict(None, [], now=NOW, max_age=DAY)
    stale = {"databaseId": 2, "createdAt": "2026-09-20T09:00:00Z"}
    assert any("older than" in p for p in N.nightly_verdict(stale, green, now=NOW, max_age=DAY))
    assert any(
        "no '(nightly)' jobs" in p for p in N.nightly_verdict(FRESH, _jobs(lint="success"), now=NOW, max_age=DAY)
    )


def test_a_tagged_version_is_not_a_release(monkeypatch) -> None:
    monkeypatch.setattr(N, "pyproject_version", lambda: "1.3.0")
    monkeypatch.setattr(N, "is_tagged", lambda v: True)
    monkeypatch.setattr(N, "latest_nightly", lambda: pytest.fail("must not read the nightlies"))
    assert N.main(["--only-when-releasing"]) == 0


def test_a_release_with_a_red_nightly_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(N, "pyproject_version", lambda: "1.3.1")
    monkeypatch.setattr(N, "is_tagged", lambda v: False)
    now = datetime.now(timezone.utc).isoformat()
    monkeypatch.setattr(
        N, "latest_nightly", lambda: ({"databaseId": 3, "createdAt": now}, _jobs(**{"Slow tests (nightly)": "failure"}))
    )
    assert N.main(["--only-when-releasing"]) == 1


def test_an_unreadable_api_is_unverified_not_green(monkeypatch) -> None:
    def boom():
        raise N.CheckError("gh: not authenticated")

    monkeypatch.setattr(N, "latest_nightly", boom)
    assert N.main([]) == 2


def test_the_release_build_job_runs_it() -> None:
    workflow = (N.REPO / ".github" / "workflows" / "test.yml").read_text()
    job = workflow[workflow.index("  release-build:") : workflow.index("  unit-tests:")]
    assert "scripts/check_nightlies.py --only-when-releasing" in job
    assert "actions: read" in job
