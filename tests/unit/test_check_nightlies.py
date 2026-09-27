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
    monkeypatch.setattr(N, "main_head", lambda: "abc")
    now = datetime.now(timezone.utc).isoformat()
    monkeypatch.setattr(
        N,
        "latest_nightly",
        lambda: ({"databaseId": 3, "createdAt": now, "headSha": "abc"}, _jobs(**{"Slow tests (nightly)": "failure"})),
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


def test_a_cancelled_run_is_skipped_for_the_newest_run_that_ran(monkeypatch) -> None:
    """A run superseded in main's concurrency queue completes `cancelled` without running a job, and a
    push run's nightly jobs are skipped -- neither says anything about the nightlies."""
    runs = [
        {"databaseId": 10, "createdAt": "2026-09-27T10:00:00Z", "conclusion": "success", "event": "push"},
        {"databaseId": 9, "createdAt": "2026-09-27T09:00:00Z", "conclusion": "cancelled", "event": "schedule"},
        {"databaseId": 8, "createdAt": "2026-09-26T09:00:00Z", "conclusion": "success", "event": "schedule"},
    ]
    viewed = []

    def gh(*args):
        if args[:2] == ("run", "list"):
            return runs
        viewed.append(args[2])
        return {"jobs": [{"name": "Slow tests (nightly)", "conclusion": "success"}]}

    monkeypatch.setattr(N, "_gh", gh)
    run, jobs = N.latest_nightly()
    assert run["databaseId"] == 8 and viewed == ["8"] and jobs


def test_a_green_nightly_of_an_older_commit_refuses() -> None:
    """Green before the last merge says nothing about the code being released."""
    run = {**FRESH, "headSha": "old"}
    jobs = _jobs(**{"Slow tests (nightly)": "success"})
    problems = N.nightly_verdict(run, jobs, now=NOW, max_age=DAY, main_sha="new")
    assert problems and "gh workflow run test.yml --ref main" in problems[0]
    assert N.nightly_verdict({**run, "headSha": "new"}, jobs, now=NOW, max_age=DAY, main_sha="new") == []


def test_a_dispatched_run_counts(monkeypatch) -> None:
    """After a fix the operator dispatches a run on main; it is read like a scheduled one."""
    runs = [
        {"databaseId": 11, "createdAt": "2026-09-27T11:00:00Z", "conclusion": "success", "event": "workflow_dispatch"}
    ]
    monkeypatch.setattr(N, "_gh", lambda *a: runs if a[:2] == ("run", "list") else {"jobs": []})
    assert N.latest_nightly()[0]["databaseId"] == 11


def test_green_at_main_passes(monkeypatch) -> None:
    now = datetime.now(timezone.utc).isoformat()
    monkeypatch.setattr(N, "main_head", lambda: "abc")
    monkeypatch.setattr(
        N,
        "latest_nightly",
        lambda: ({"databaseId": 4, "createdAt": now, "headSha": "abc"}, _jobs(**{"Slow tests (nightly)": "success"})),
    )
    assert N.main([]) == 0
