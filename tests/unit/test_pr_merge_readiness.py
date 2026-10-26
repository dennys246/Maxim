"""Guard tests for scripts/pr_merge_readiness.py — the mechanical form of the green-PR invariant.

The invariant ("a PR showing all-green may have run NO tests"; "BLOCKED while every
required context is green means look at a surface `gh pr checks` does not render") carried
"no automated test; the mechanically checkable form is tracked follow-up work" from the day
it was written. These tests are that guard.

Two classes carry the weight:

- `TestReplaysPr654` replays the incident stage by stage.
- `TestNeverAssertsAbsenceOverAnUnsettledSnapshot` pins the fix for the review finding
  that this tool ORIGINALLY reproduced the very error it exists to prevent — asserting
  `required-check-absent` (with a fabricated cause) moments after a push, when the checks
  simply had not been created yet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from pr_merge_readiness import (  # noqa: E402
    BLOCKED,
    READY,
    UNSETTLED,
    evaluate,
)

_REQUIRED = [
    {"name": "unit-tests", "status": "completed", "conclusion": "success", "output_title": None},
    {"name": "lint", "status": "completed", "conclusion": "success", "output_title": None},
    {"name": "Release build (wheel contents + version)", "status": "completed", "conclusion": "success"},
    {"name": "CodeQL", "status": "completed", "conclusion": "success", "output_title": "No new alerts"},
]


def _eval(*, pr=None, checks=None, alerts=None, rules=None, contexts=None, errors=()):
    return evaluate(
        pr=pr or {"mergeStateStatus": "CLEAN", "mergeable": "MERGEABLE"},
        checks=checks if checks is not None else list(_REQUIRED),
        alerts=alerts or [],
        ruleset_rules=rules or [],
        required_contexts=contexts,
        fetch_errors=tuple(errors),
    )


def _codes(result):
    return {f.code for f in result.findings}


class TestReady:
    def test_all_required_present_and_green(self):
        assert _eval().exit_code == READY
        assert "READY" in _eval().render()


class TestMergeStateIsConsulted:
    """Review BLOCKER B1: `mergeable` was fetched, printed, and never used."""

    @pytest.mark.parametrize(
        "pr",
        [
            {"mergeStateStatus": "DIRTY", "mergeable": "CONFLICTING"},
            {"mergeStateStatus": "DRAFT", "mergeable": "MERGEABLE"},
            {"mergeStateStatus": "BEHIND", "mergeable": "MERGEABLE"},
            {"mergeStateStatus": "UNSTABLE", "mergeable": "MERGEABLE"},
        ],
    )
    def test_non_clean_states_are_never_ready(self, pr):
        result = _eval(pr=pr)
        assert result.exit_code == BLOCKED, f"{pr} must not report READY"

    def test_conflicting_is_named_explicitly(self):
        """The lesson's own headline case (PR #576) must not read as 'ready to merge'."""
        result = _eval(pr={"mergeStateStatus": "DIRTY", "mergeable": "CONFLICTING"})
        assert "merge-conflict" in _codes(result)
        assert result.exit_code == BLOCKED


class TestNeverAssertsAbsenceOverAnUnsettledSnapshot:
    """Review BLOCKER B2 — the tool reproducing the #654 error inside itself."""

    def test_missing_required_check_while_others_run_is_not_blocking(self):
        """Moments after a push, required checks legitimately do not exist yet."""
        checks = [{"name": "CodeQL", "status": "queued", "conclusion": None}]
        result = _eval(checks=checks)
        assert result.exit_code == UNSETTLED, "absence cannot be asserted mid-flight"
        assert "required-check-not-yet-present" in _codes(result)
        assert "required-check-absent" not in _codes(result)

    def test_absence_becomes_blocking_once_everything_is_settled(self):
        checks = [c for c in _REQUIRED if c["name"] != "unit-tests"]
        result = _eval(checks=checks)
        assert result.exit_code == BLOCKED
        assert "required-check-absent" in _codes(result)

    def test_a_settled_failure_still_outranks_in_flight(self):
        """A positive fact (it failed) is not invalidated by other jobs still running."""
        checks = [
            *_REQUIRED,
            {"name": "Analyze (python)", "status": "in_progress", "conclusion": None},
            {"name": "some-check", "status": "completed", "conclusion": "failure", "output_title": "boom"},
        ]
        assert _eval(checks=checks).exit_code == BLOCKED

    def test_no_fabricated_cause_in_the_unsettled_message(self):
        """The original text asserted 'a CONFLICTING PR produces no merge commit'."""
        result = _eval(checks=[{"name": "CodeQL", "status": "queued", "conclusion": None}])
        joined = result.render().lower()
        assert "conflicting" not in joined


class TestSkippedIsNotPassing:
    """Review HIGH H2: a skipped required job satisfies the gate without running."""

    def test_skipped_required_check_blocks(self):
        checks = [dict(c, conclusion="skipped") for c in _REQUIRED]
        result = _eval(checks=checks)
        assert result.exit_code == BLOCKED
        assert "required-check-skipped" in _codes(result)

    def test_skipped_non_required_check_is_ignored(self):
        checks = [*_REQUIRED, {"name": "Slow tests (nightly)", "status": "completed", "conclusion": "skipped"}]
        assert _eval(checks=checks).exit_code == READY


class TestUnreadableSurfaceIsNotSilence:
    """Review HIGH H4: a failed fetch previously became 'no alerts found'."""

    def test_fetch_error_withholds_a_ready_verdict(self):
        result = _eval(errors=("could not read code-scanning alerts (403)",))
        assert result.exit_code == UNSETTLED
        assert "surface-unverified" in _codes(result)
        assert "NOT checked" in result.render()


class TestFailingCheckQuotesItsOwnOutput:
    def test_output_title_is_surfaced(self):
        checks = [
            *_REQUIRED,
            {
                "name": "CodeQL",
                "status": "completed",
                "conclusion": "failure",
                "output_title": "2 new alerts including 2 high severity security vulnerabilities",
            },
        ]
        result = _eval(checks=checks)
        assert result.exit_code == BLOCKED
        assert any("2 high severity security vulnerabilities" in f.detail for f in result.findings)

    def test_stale_conclusion_blocks(self):
        checks = [*_REQUIRED, {"name": "x", "status": "completed", "conclusion": "stale", "output_title": None}]
        assert _eval(checks=checks).exit_code == BLOCKED


class TestCodeScanningAlerts:
    def test_open_alert_blocks_and_names_rule_and_location(self):
        alerts = [
            {
                "number": 94,
                "rule_id": "py/clear-text-logging-sensitive-data",
                "severity": "high",
                "path": "src/maxim/hivemind/hive_cli.py",
                "line": 139,
            }
        ]
        result = _eval(alerts=alerts)
        assert result.exit_code == BLOCKED
        finding = next(f for f in result.findings if f.code == "code-scanning-alert")
        assert "py/clear-text-logging-sensitive-data" in finding.detail
        assert "hive_cli.py:139" in finding.detail
        assert "written reason" in finding.detail


class TestRequiredContextsPreferredOverPatterns:
    """Review MODERATE M1: substring patterns are fragile; ruleset contexts are exact."""

    def test_exact_contexts_catch_a_lookalike_check_name(self):
        checks = [
            {"name": "pre-commit lint (advisory)", "status": "completed", "conclusion": "success"},
            {"name": "unit-tests-nightly", "status": "completed", "conclusion": "success"},
        ]
        result = _eval(checks=checks, contexts=("unit-tests", "lint"))
        assert result.exit_code == BLOCKED
        assert "required-check-absent" in _codes(result)

    def test_pattern_fallback_still_works_without_contexts(self):
        assert _eval(checks=list(_REQUIRED), contexts=None).exit_code == READY


class TestBlockedWithoutExplanation:
    def test_unexplained_block_is_itself_reported(self):
        result = _eval(pr={"mergeStateStatus": "BLOCKED", "mergeable": "MERGEABLE"})
        assert result.exit_code == BLOCKED
        assert "blocked-unexplained" in _codes(result)

    def test_a_known_cause_suppresses_the_unexplained_finding(self):
        result = _eval(
            pr={"mergeStateStatus": "BLOCKED", "mergeable": "MERGEABLE"},
            alerts=[{"number": 1, "rule_id": "r", "severity": "high", "path": "p", "line": 1}],
        )
        assert "blocked-unexplained" not in _codes(result)
        assert result.exit_code == BLOCKED  # pin the code, not just the finding's absence

    def test_unsettled_also_suppresses_the_unexplained_finding(self):
        result = _eval(
            pr={"mergeStateStatus": "BLOCKED", "mergeable": "MERGEABLE"},
            checks=[*_REQUIRED, {"name": "x", "status": "in_progress", "conclusion": None}],
        )
        assert "blocked-unexplained" not in _codes(result)
        assert result.exit_code == UNSETTLED


class TestReplaysPr654:
    """Stage-by-stage replay of the real incident this tool exists to prevent."""

    def test_stage_2_neutral_check_while_python_still_running(self):
        """The moment the first wasted push was made. No verdict may be rendered."""
        checks = [
            *_REQUIRED[:3],
            {"name": "Analyze (actions)", "status": "completed", "conclusion": "success"},
            {"name": "Analyze (python)", "status": "in_progress", "conclusion": None},
            {
                "name": "CodeQL",
                "status": "completed",
                "conclusion": "neutral",
                "output_title": "1 configuration not found",
            },
        ]
        result = _eval(
            pr={"mergeStateStatus": "BLOCKED", "mergeable": "MERGEABLE"}, checks=checks, rules=["code_scanning"]
        )
        assert result.exit_code == UNSETTLED, "must refuse a verdict while a sub-job is running"
        assert not any(f.severity == "BLOCKING" for f in result.findings)
        # the neutral row's own text must be shown rather than interpreted
        assert any(f.code == "check-neutral" and "1 configuration not found" in f.detail for f in result.findings)

    def test_stage_4_settled_reveals_the_real_cause(self):
        checks = [
            *_REQUIRED[:3],
            {"name": "Analyze (python)", "status": "completed", "conclusion": "success"},
            {
                "name": "CodeQL",
                "status": "completed",
                "conclusion": "failure",
                "output_title": "2 new alerts including 2 high severity security vulnerabilities",
            },
        ]
        alerts = [
            {
                "number": 94,
                "rule_id": "py/clear-text-logging-sensitive-data",
                "severity": "high",
                "path": "src/maxim/hivemind/hive_cli.py",
                "line": 139,
            },
            {
                "number": 95,
                "rule_id": "py/clear-text-logging-sensitive-data",
                "severity": "high",
                "path": "src/maxim/hivemind/hive_cli.py",
                "line": 101,
            },
        ]
        result = _eval(
            pr={"mergeStateStatus": "BLOCKED", "mergeable": "MERGEABLE"},
            checks=checks,
            alerts=alerts,
            rules=["code_scanning"],
        )
        assert result.exit_code == BLOCKED
        rendered = result.render()
        assert "code-scanning-alert" in rendered
        assert "2 high severity security vulnerabilities" in rendered
        assert "ruleset-gate" in rendered  # the surface `gh pr checks` never renders

    def test_stage_5_after_dismissal_it_is_ready(self):
        result = _eval(pr={"mergeStateStatus": "CLEAN", "mergeable": "MERGEABLE"}, rules=["code_scanning"])
        assert result.exit_code == READY


def _run(name, conclusion, *, started_at, run_id, status="completed"):
    return {
        "name": name,
        "status": status,
        "conclusion": conclusion,
        "output_title": None,
        "started_at": started_at,
        "id": run_id,
    }


class TestOnlyTheLatestRunPerCheckIsJudged:
    """#1442: editing a PR's description starts a second `Tests` run on the same head
    commit and the workflow's `concurrency` block cancels the first. GitHub's merge gate
    reads the latest run per check name; the script read every run and printed one
    `check-failed … cancelled` row per job of the superseded run (seen on #1433)."""

    @pytest.mark.xfail(strict=True, reason="#1442 red gate: superseded runs are still judged")
    def test_a_superseded_cancelled_run_under_a_later_success_is_not_blocking(self):
        checks = [
            _run("unit-tests", "cancelled", started_at="2026-10-23T10:00:00Z", run_id=1),
            *_REQUIRED,
        ]
        checks[1] = {**checks[1], "started_at": "2026-10-23T10:05:00Z", "id": 2}
        result = _eval(checks=checks)
        assert "check-failed" not in _codes(result)
        assert result.exit_code == READY

    def test_a_latest_cancelled_run_is_still_blocking(self):
        checks = [
            {**_REQUIRED[0], "started_at": "2026-10-23T10:00:00Z", "id": 1},
            _run("unit-tests", "cancelled", started_at="2026-10-23T10:05:00Z", run_id=2),
            *_REQUIRED[1:],
        ]
        result = _eval(checks=checks)
        assert "check-failed" in _codes(result)
        assert result.exit_code == BLOCKED

    def test_an_older_success_does_not_hide_a_newer_failure(self):
        checks = [
            *_REQUIRED[1:],
            _run("unit-tests", "failure", started_at="2026-10-23T10:05:00Z", run_id=2),
            {**_REQUIRED[0], "started_at": "2026-10-23T10:00:00Z", "id": 1},
        ]
        result = _eval(checks=checks)
        assert "check-failed" in _codes(result)
        assert result.exit_code == BLOCKED

    @pytest.mark.xfail(strict=True, reason="#1442 red gate: superseded runs are still judged")
    def test_a_queued_rerun_without_started_at_is_the_latest(self):
        """A queued run has `started_at: null`; it is newer than any started run, so the
        verdict waits on it rather than reporting the old run's failure."""
        checks = [
            _run("unit-tests", "failure", started_at="2026-10-23T10:00:00Z", run_id=1),
            _run("unit-tests", None, started_at=None, run_id=2, status="queued"),
            *_REQUIRED[1:],
        ]
        result = _eval(checks=checks)
        assert "check-failed" not in _codes(result)
        assert result.exit_code == UNSETTLED

    @pytest.mark.xfail(strict=True, reason="#1442 red gate: superseded runs are still judged")
    def test_same_start_time_breaks_the_tie_on_id(self):
        checks = [
            _run("unit-tests", "success", started_at="2026-10-23T10:00:00Z", run_id=2),
            _run("unit-tests", "cancelled", started_at="2026-10-23T10:00:00Z", run_id=1),
            *_REQUIRED[1:],
        ]
        assert _eval(checks=checks).exit_code == READY


class TestIoShell:
    """Review MODERATE M7: every earlier bug lived in the untested shell."""

    def test_slurped_check_runs_flattens_every_page(self, monkeypatch):
        """`--paginate --slurp` yields a LIST of pages; dropping pages 2+ would
        manufacture a false `required-check-absent`."""
        import pr_merge_readiness as mod

        pages = [
            {"check_runs": [{"name": "unit-tests", "status": "completed", "conclusion": "success", "output": {}}]},
            {"check_runs": [{"name": "lint", "status": "completed", "conclusion": "success", "output": {}}]},
        ]
        monkeypatch.setattr(mod, "_gh_json", lambda *a, **k: pages)
        runs = mod._slurped_check_runs("o/r", "sha")
        assert [r["name"] for r in runs] == ["unit-tests", "lint"]

    @pytest.mark.xfail(strict=True, reason="#1442 red gate: started_at/id are not slurped")
    def test_slurped_check_runs_carry_started_at_and_id(self, monkeypatch):
        """The latest-run filter orders on these two fields; dropping them in the shell
        would leave the pure filter judging every run as equally old."""
        import pr_merge_readiness as mod

        page = {"check_runs": [{"name": "lint", "id": 7, "started_at": "2026-10-23T10:00:00Z", "output": {}}]}
        monkeypatch.setattr(mod, "_gh_json", lambda *a, **k: [page])
        (run,) = mod._slurped_check_runs("o/r", "sha")
        assert (run["id"], run["started_at"]) == (7, "2026-10-23T10:00:00Z")

    def test_slurped_check_runs_tolerates_a_single_object(self, monkeypatch):
        import pr_merge_readiness as mod

        monkeypatch.setattr(mod, "_gh_json", lambda *a, **k: {"check_runs": [{"name": "lint", "output": {}}]})
        assert [r["name"] for r in mod._slurped_check_runs("o/r", "sha")] == ["lint"]

    def test_gh_json_raises_on_empty_output(self, monkeypatch):
        """Previously returned None, which callers then subscripted → traceback."""
        import pr_merge_readiness as mod

        class _P:
            returncode = 0
            stdout = ""
            stderr = ""

        monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: _P())
        with pytest.raises(RuntimeError, match="no output"):
            mod._gh_json(["api", "x"])

    def test_main_returns_tool_error_when_fetch_fails(self, monkeypatch):
        import pr_merge_readiness as mod

        def _boom(*a, **k):
            raise RuntimeError("gh exploded")

        monkeypatch.setattr(mod, "_fetch", _boom)
        assert mod.main(["654"]) == mod.TOOL_ERROR


class TestRendering:
    def test_blocking_sorts_before_info(self):
        result = _eval(
            pr={"mergeStateStatus": "BLOCKED", "mergeable": "MERGEABLE"},
            alerts=[{"number": 1, "rule_id": "r", "severity": "high", "path": "p", "line": 1}],
            rules=["code_scanning"],
        )
        assert result.render().splitlines()[0].startswith("[BLOCKING]")


@pytest.mark.parametrize("state", ["queued", "in_progress", "pending", "waiting", "requested"])
def test_every_unsettled_status_withholds_a_verdict(state):
    checks = [*_REQUIRED, {"name": "something", "status": state, "conclusion": None}]
    assert _eval(checks=checks).exit_code == UNSETTLED


class TestClosedOrMergedPr:
    """Live-run finding: a merged PR reported UNSETTLED with UNKNOWN merge fields."""

    @pytest.mark.parametrize("state", ["MERGED", "CLOSED"])
    def test_closed_pr_is_reported_plainly(self, state):
        result = _eval(pr={"state": state, "mergeable": "UNKNOWN", "mergeStateStatus": "UNKNOWN"})
        assert result.exit_code == READY
        assert "already-closed" in _codes(result)
        assert state in result.render()

    def test_open_pr_is_still_evaluated(self):
        result = _eval(pr={"state": "OPEN", "mergeStateStatus": "DIRTY", "mergeable": "CONFLICTING"})
        assert result.exit_code == BLOCKED


class TestBranchRulesFallback:
    """Live-run finding: /rules/branch 404s on some repos — a 404 is not 'no rules'."""

    def test_falls_back_to_active_rulesets(self, monkeypatch):
        import pr_merge_readiness as mod

        def fake(args, **kw):
            path = args[1]
            if "/rules/branch/" in path:
                raise RuntimeError("gh: Not Found (HTTP 404)")
            if path.endswith("/rulesets"):
                return [{"id": 1}]
            return {
                "enforcement": "active",
                "rules": [
                    {"type": "code_scanning"},
                    {
                        "type": "required_status_checks",
                        "parameters": {"required_status_checks": [{"context": "unit-tests"}]},
                    },
                ],
            }

        monkeypatch.setattr(mod, "_gh_json", fake)
        rules, contexts, err = mod._branch_rules("o/r", "main")
        assert err is None
        assert "code_scanning" in rules
        assert contexts == ["unit-tests"]

    def test_inactive_rulesets_are_ignored(self, monkeypatch):
        import pr_merge_readiness as mod

        def fake(args, **kw):
            path = args[1]
            if "/rules/branch/" in path:
                raise RuntimeError("404")
            if path.endswith("/rulesets"):
                return [{"id": 1}]
            return {"enforcement": "disabled", "rules": [{"type": "code_scanning"}]}

        monkeypatch.setattr(mod, "_gh_json", fake)
        rules, _, err = mod._branch_rules("o/r", "main")
        assert rules == [] and err is None

    def test_both_paths_failing_reports_unverified(self, monkeypatch):
        import pr_merge_readiness as mod

        def boom(*a, **k):
            raise RuntimeError("nope")

        monkeypatch.setattr(mod, "_gh_json", boom)
        _, _, err = mod._branch_rules("o/r", "main")
        assert err is not None and "could not read" in err
