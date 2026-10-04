"""A fake GitHub push for the diff-scoped lints (#1089): the event file, the env, and a stand-in for ``gh api``.

``scripts/_lint_git.py::push_base`` reads the push's ``before``/``after`` from ``GITHUB_EVENT_PATH`` and asks the
Actions API which first-parent ancestor last had a green ``lint`` job; ``push_units`` asks which merged PR each
landed commit belongs to. ``fake_push`` serves both from plain dicts, so a test states the history it means.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from scripts import _lint_git

REPO = "owner/repo"


def rev(root: Path, ref: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", ref], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()


def fake_push(
    monkeypatch,
    root: Path,
    *,
    before: str | None = None,
    after: str | None = None,
    green: set[str] | None = None,
    prs: dict[str, dict[str, Any]] | None = None,
    pr_commit_dates: dict[int, str] | None = None,
) -> list[str]:
    """Make ``root``'s HEAD a push to main. ``before`` defaults to ``HEAD^``; ``green`` (default: ``{before}``)
    are the commits whose push run's lint job succeeded; ``prs`` maps a landed commit to its merged PR
    ({number, title, body}). Returns the API paths requested, in order."""
    before = before or rev(root, "HEAD^")
    after = after or rev(root, "HEAD")
    green = {before} if green is None else green
    prs = prs or {}
    event = root.parent / f"push-event-{after[:8]}.json"
    event.write_text(json.dumps({"before": before, "after": after}))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setenv("GITHUB_REPOSITORY", REPO)
    monkeypatch.setenv("GITHUB_JOB", "lint")
    monkeypatch.setenv("GITHUB_WORKFLOW_REF", f"{REPO}/.github/workflows/test.yml@refs/heads/main")
    calls: list[str] = []
    run_ids = {sha: 1000 + i for i, sha in enumerate(sorted(green))}

    def api(path: str) -> Any:
        calls.append(path)
        if "/actions/workflows/test.yml/runs" in path:
            return {"workflow_runs": [{"id": rid, "head_sha": sha} for sha, rid in run_ids.items()]}
        if "/actions/runs/" in path and path.endswith("/jobs?per_page=100"):
            return {"jobs": [{"name": "lint", "conclusion": "success"}]}
        if path.startswith(f"repos/{REPO}/commits/") and path.endswith("/pulls"):
            sha = path.split("/")[-2]
            pr = prs.get(sha)
            return [] if pr is None else [{**pr, "merged_at": "2026-10-04T00:00:00Z", "base": {"ref": "main"}}]
        if "/pulls/" in path and "/commits" in path:
            number = int(path.split("/pulls/")[1].split("/")[0])
            date = (pr_commit_dates or {}).get(number, "2026-10-04T00:00:00Z")
            return [{"commit": {"committer": {"date": date}}}]
        raise AssertionError(f"unexpected API path {path}")

    # The lints import the helper as top-level ``_lint_git`` (scripts/ on sys.path) and the tests as
    # ``scripts._lint_git``: two module objects, so every loaded copy gets the stand-in.
    import importlib
    import sys

    monkeypatch.syspath_prepend(str(Path(_lint_git.__file__).parent))
    importlib.import_module("_lint_git")  # load it if no lint has yet

    for name in ("_lint_git", "scripts._lint_git"):
        if name in sys.modules:
            monkeypatch.setattr(sys.modules[name], "gh_api", api)
    return calls
