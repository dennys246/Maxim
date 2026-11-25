#!/usr/bin/env python3
"""Answer "why is this PR not mergeable?" in one command — every gating surface at once.

CLAUDE.md's green-PR invariant says a PR page showing all-green may be lying: a check
that never ran is indistinguishable from one that passed, and a merge can be BLOCKED by
surfaces `gh pr checks` does not render at all. That invariant carried "the mechanically
checkable form is tracked follow-up work" from the day it was written. This is that form.

It exists because the process rule alone failed (2026-09-06, PR #654). The gating surface
was found correctly; the mistake came next — reaching for a remembered remedy before
reading the instrument. A CodeQL check read ``neutral — "1 configuration not found"``,
which matches the documented "default setup needs a PUSH" variant, so a push was made
**while `Analyze (python)` was still `in_progress`**. Two cycles were wasted before the
check's own summary — *"2 new alerts including 2 high severity security vulnerabilities"*,
one API call away the whole time — revealed a code-scanning ALERT.

Two design rules follow, and the first review of this script found the tool violating
both, so they are stated as rules rather than intentions:

1. **Never assert a NEGATIVE over an unsettled snapshot.** "Required check absent" is a
   claim about something not existing; it is only sound once nothing is in flight.
   Right after a push, checks legitimately do not exist yet. A settled *failure* of
   a check's latest run is a positive fact and does outrank in-flight; an absence does not.
   (A run superseded by a newer run of the same check is not judged at all, #1442.)
2. **Report what each surface SAYS, never a remembered cause.** No list of known causes
   is encoded here — the list is always missing the next variant. A surface that could
   not be read is reported as UNVERIFIED, never as "nothing found".

Usage:
    python scripts/pr_merge_readiness.py 654
    python scripts/pr_merge_readiness.py 654 --repo owner/name

**The suite of record (M64, owner decisions 2026-10-16 and 2026-11-25).** CI's latest
`unit-tests` run on the PR's head SHA is the final full test suite; no local full suite is
owed before a merge. So READY also asserts: a `unit-tests` check run posted by the app
pinned in `scripts/repo_settings_expected.json` exists on the head, its newest run (by
check-run id, which GitHub assigns in creation order) settled as exactly `success`, and the
head did not move while the checks were read. A floors-only push that re-used an earlier
measurement (#1640) is named as such. Approach note and its design review:
docs/plans/ci_suite_of_record_m64_approach.md.

Exit codes: 0 ready · 1 blocked (reasons printed) · 2 unsettled/unverified (no verdict —
re-run when it settles) · 3 tool error.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Fallback only — the effective ruleset contexts are preferred when readable.
# CodeQL is included per bugs ledger D63, whose durable fix names it explicitly.
REQUIRED_CHECK_PATTERNS: tuple[str, ...] = ("unit-tests", "lint", "release build", "codeql")

# M64: the check whose green run on the head SHA is the final full suite, the settings file
# that pins which app may post it, and the step a floors-only re-use (#1640) skips.
SUITE_OF_RECORD = "unit-tests"
_EXPECTED_SETTINGS = Path(__file__).resolve().parent / "repo_settings_expected.json"
_FAST_SUITE_STEP = "Run required fast suite"

_UNSETTLED = frozenset({"queued", "in_progress", "pending", "waiting", "requested"})
_BAD_CONCLUSIONS = frozenset({"failure", "timed_out", "cancelled", "action_required", "stale", "startup_failure"})
# Merge states that are NOT ready. CLEAN/HAS_HOOKS/UNKNOWN are the mergeable-ish ones;
# everything else here is a distinct, actionable reason the PR cannot merge today.
_BAD_MERGE_STATES: dict[str, str] = {
    "DIRTY": "the branch has merge conflicts — resolve and push (a CONFLICTING PR also suppresses the whole "
    "pull_request-triggered workflow, so its green rows may be stale)",
    "DRAFT": "the PR is a draft — mark it ready for review",
    "BEHIND": "the branch is behind its base and the base requires up-to-date branches — update it",
    "BLOCKED": "a required review, gate, or ruleset is unsatisfied",
    "UNSTABLE": "a non-required check is failing",
}

READY, BLOCKED, UNSETTLED, TOOL_ERROR = 0, 1, 2, 3


@dataclass(frozen=True)
class Finding:
    """One reason the PR is not ready, or one thing worth knowing."""

    severity: str  # "BLOCKING" | "UNSETTLED" | "INFO"
    code: str
    detail: str


@dataclass(frozen=True)
class Readiness:
    exit_code: int
    findings: list[Finding] = field(default_factory=list)

    def render(self) -> str:
        ready = "READY: no gating surface reports a problem."
        if not self.findings:
            return ready
        order = {"BLOCKING": 0, "UNSETTLED": 1, "INFO": 2}
        lines = [
            f"[{f.severity}] {f.code}: {f.detail}"
            for f in sorted(self.findings, key=lambda f: (order.get(f.severity, 9), f.code))
        ]
        # A READY verdict always carries the suite-of-record INFO row (M64), so say READY first.
        return "\n".join([ready, *lines] if self.exit_code == READY else lines)


def _latest_run_per_check(checks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Keep only the newest run of each check, as GitHub's merge gate does (#1442).

    Returns the survivors (input order kept) and how many superseded runs were dropped.

    A head commit can carry several runs of one check: editing a PR's description starts a
    second workflow run and the `concurrency` block cancels the first. The cancelled run is
    superseded, not failed. The check-runs API's default ``filter=latest`` does NOT collapse
    these: it collapses re-runs inside one check suite, and each workflow run is its own suite.

    A check is its name plus the app that posted it (a ruleset context can pin the app), so a
    same-named check from another app is judged on its own, never hidden. "Newest" orders on
    `started_at`, then `id`. An unsettled run with no `started_at` yet counts as the newest, so
    a verdict waits on a pending re-run rather than repeating the old run's result; a SETTLED
    run with no `started_at` (cancelled before it began) counts as the oldest.
    """

    def _age(c: dict[str, Any]) -> tuple[int, str, int]:
        started = c.get("started_at")
        pending = not started and str(c.get("status", "")).lower() in _UNSETTLED
        run_id = c.get("id")
        return (1 if pending else 0, str(started or ""), run_id if isinstance(run_id, int) else 0)

    newest: dict[tuple[str, Any], dict[str, Any]] = {}
    for c in checks:
        key = (str(c.get("name", "")), c.get("app_id"))
        if key not in newest or _age(c) >= _age(newest[key]):
            newest[key] = c
    kept = {id(c) for c in newest.values()}
    return [c for c in checks if id(c) in kept], len(checks) - len(kept)


def _required_missing(
    checks: list[dict[str, Any]],
    required_contexts: tuple[str, ...] | None,
    required_patterns: tuple[str, ...],
) -> list[str]:
    """Names/patterns with no corresponding check run. Exact match when contexts are known."""
    if required_contexts:
        present = {str(c.get("name", "")) for c in checks}
        return [ctx for ctx in required_contexts if ctx not in present]
    joined = " | ".join(str(c.get("name", "")).lower() for c in checks)
    return [p for p in required_patterns if p.lower() not in joined]


def pinned_suite_app_id(path: Path = _EXPECTED_SETTINGS) -> int | None:
    """The app id the expected settings pin for the `unit-tests` context, or None if unreadable.

    The expected-settings file, not live protection: reading protection needs admin scope, and
    the nightly `scripts/check_repo_settings.py` fails when the two drift.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    checks = ((data.get("protection") or {}).get("required_status_checks") or {}).get("checks") or []
    for chk in checks:
        if chk.get("context") == SUITE_OF_RECORD and isinstance(chk.get("app_id"), int):
            return int(chk["app_id"])
    return None


def suite_of_record_run(checks: list[dict[str, Any]], app_id: int | None) -> dict[str, Any] | None:
    """The newest `unit-tests` run from the pinned app, by check-run id.

    Ids, not `started_at`: a run cancelled before it began has no start time but is still the
    newest run GitHub's gate judges (design review finding 6), and a re-run gets a new id.
    """
    if app_id is None:
        return None
    runs = [c for c in checks if c.get("name") == SUITE_OF_RECORD and c.get("app_id") == app_id]
    if not runs:
        return None
    return max(runs, key=lambda c: c.get("id") if isinstance(c.get("id"), int) else 0)


def _suite_of_record_findings(
    raw_checks: list[dict[str, Any]],
    *,
    app_id: int | None,
    head_sha: str,
    reuse: str,
    anything_unsettled: bool,
) -> list[Finding]:
    """M64: CI's `unit-tests` run on the head is the final full suite, so READY must prove it ran green."""
    if app_id is None:
        return [
            Finding(
                "UNSETTLED",
                "surface-unverified",
                "cannot pin which app posts the suite of record: scripts/repo_settings_expected.json is "
                "unreadable or has no unit-tests entry — the final full suite was NOT checked",
            )
        ]
    run = suite_of_record_run(raw_checks, app_id)
    if run is None:
        if anything_unsettled:
            return [
                Finding(
                    "UNSETTLED",
                    "suite-of-record-not-yet-present",
                    f"no {SUITE_OF_RECORD} run from app {app_id} on this head yet, and checks are still in flight",
                )
            ]
        return [
            Finding(
                "BLOCKING",
                "suite-of-record-absent",
                f"no {SUITE_OF_RECORD} run from app {app_id} ran on this head — the final full suite never ran "
                "(a same-named check from another app does not count)",
            )
        ]
    status = str(run.get("status", "")).lower()
    if status in _UNSETTLED:
        return [Finding("UNSETTLED", "suite-of-record-running", f"{SUITE_OF_RECORD} run {run.get('id')} is {status}")]
    conclusion = str(run.get("conclusion") or "").lower()
    if conclusion != "success":
        return [
            Finding(
                "BLOCKING",
                "suite-of-record-not-green",
                f"the newest {SUITE_OF_RECORD} run ({run.get('id')}) ended {conclusion or 'without a conclusion'}; "
                "only `success` counts as the final full suite",
            )
        ]
    findings = [
        Finding(
            "INFO",
            "suite-of-record",
            f"{SUITE_OF_RECORD} run {run.get('id')} passed on head {head_sha[:12] or '(unknown)'} — "
            "this is the final full suite (M64)",
        )
    ]
    if reuse == "reused":
        findings.append(
            Finding(
                "INFO",
                "suite-of-record-reused",
                "this run re-used an earlier run's measurement (a floors-only push, #1640): its tests ran on a "
                "commit that differs from this head only in scripts/coverage_floors.json",
            )
        )
    elif reuse.startswith("unknown"):
        findings.append(Finding("INFO", "suite-of-record-reuse-unknown", f"could not tell whether it re-used: {reuse}"))
    return findings


def evaluate(
    *,
    pr: dict[str, Any],
    checks: list[dict[str, Any]],
    alerts: list[dict[str, Any]],
    ruleset_rules: list[str],
    required_patterns: tuple[str, ...] = REQUIRED_CHECK_PATTERNS,
    required_contexts: tuple[str, ...] | None = None,
    fetch_errors: tuple[str, ...] = (),
    suite_app_id: int | None,
    head_sha: str = "",
    suite_reuse: str = "",
) -> Readiness:
    """Pure verdict over already-fetched data (so this is testable without network).

    ``suite_app_id`` is required (keyword-only, no default): a caller that forgets it must
    get a TypeError, not a verdict with the suite-of-record step silently skipped.
    """
    findings: list[Finding] = []
    raw_checks = checks
    checks, superseded = _latest_run_per_check(checks)
    if superseded:
        findings.append(
            Finding(
                "INFO",
                "superseded-runs-ignored",
                f"{superseded} older run(s) of a check that ran again on this commit were ignored; "
                "the latest run of each check is judged, as GitHub's merge gate does",
            )
        )

    # ── 0. A closed/merged PR has no merge to gate. Say so plainly rather than
    #      reporting UNKNOWN merge fields as if they were an unsettled diagnosis.
    pr_state = str(pr.get("state", "") or "").upper()
    if pr_state in {"MERGED", "CLOSED"}:
        return Readiness(READY, [Finding("INFO", "already-closed", f"this PR is {pr_state} — nothing left to gate")])

    # ── 1. Is anything still moving? Gates the negative claims below. ──────────
    unsettled = [c for c in checks if str(c.get("status", "")).lower() in _UNSETTLED]
    for c in unsettled:
        findings.append(Finding("UNSETTLED", "check-running", f"{c.get('name')} is {c.get('status')}"))

    # ── 2. A surface we could not read is UNVERIFIED, never "nothing found". ──
    for err in fetch_errors:
        findings.append(Finding("UNSETTLED", "surface-unverified", f"{err} — this gating surface was NOT checked"))

    # ── 3. The merge state itself. `mergeable` is consulted, not just printed. ─
    state = str(pr.get("mergeStateStatus", "") or "").upper()
    mergeable = str(pr.get("mergeable", "") or "").upper()
    if mergeable == "CONFLICTING":
        findings.append(Finding("BLOCKING", "merge-conflict", "mergeable=CONFLICTING — resolve conflicts and push"))
    if state in _BAD_MERGE_STATES and not (state == "BLOCKED"):
        findings.append(Finding("BLOCKING", f"merge-state-{state.lower()}", _BAD_MERGE_STATES[state]))
    if "UNKNOWN" in (state, mergeable):
        # GitHub is still computing it, so BEHIND (strict protection) cannot be ruled out yet.
        findings.append(
            Finding(
                "UNSETTLED",
                "merge-state-unknown",
                f"mergeStateStatus={state or '?'} mergeable={mergeable or '?'}: GitHub has not finished "
                "computing whether this PR can merge (a BEHIND branch would read this way) — re-run shortly",
            )
        )

    # ── 4. Required checks PRESENT — a negative claim, so gate it on settledness.
    missing = _required_missing(checks, required_contexts, required_patterns)
    source = "ruleset-required contexts" if required_contexts else "fallback name patterns"
    for name in missing:
        if unsettled:
            findings.append(
                Finding(
                    "UNSETTLED",
                    "required-check-not-yet-present",
                    f"no check matching {name!r} yet ({source}) — but other checks are still in flight, so "
                    "absence cannot be asserted; re-run when everything settles",
                )
            )
        else:
            findings.append(
                Finding(
                    "BLOCKING",
                    "required-check-absent",
                    f"no check matching {name!r} ran ({source}) — an absent check is not a passing one",
                )
            )

    # A skipped required check is PRESENT but did not run: the invariant's own sentence.
    for c in checks:
        if str(c.get("conclusion") or "").lower() != "skipped":
            continue
        label = str(c.get("name", ""))
        is_required = (required_contexts and label in required_contexts) or (
            not required_contexts and any(p.lower() in label.lower() for p in required_patterns)
        )
        if is_required:
            findings.append(
                Finding(
                    "BLOCKING",
                    "required-check-skipped",
                    f"{label} was SKIPPED — it satisfies the merge gate without having run "
                    "(a mechanism that does not run looks exactly like one that ran and found nothing)",
                )
            )

    # ── 5. Failing checks — always quote the check's OWN output. ───────────────
    for c in checks:
        conclusion = str(c.get("conclusion") or "").lower()
        title = c.get("output_title") or "(no output title)"
        if conclusion in _BAD_CONCLUSIONS:
            findings.append(Finding("BLOCKING", "check-failed", f"{c.get('name')} → {conclusion}: {title}"))
        elif conclusion == "neutral":
            findings.append(
                Finding(
                    "INFO",
                    "check-neutral",
                    f"{c.get('name')} → neutral: {title} (neutral commonly means 'not finished yet' or "
                    "'nothing to report', NOT a failure — do not act on it while anything is in flight)",
                )
            )

    # ── 5b. The suite of record (M64): CI's unit-tests run IS the final full suite.
    findings.extend(
        _suite_of_record_findings(
            raw_checks,
            app_id=suite_app_id,
            head_sha=head_sha,
            reuse=suite_reuse,
            anything_unsettled=bool(unsettled),
        )
    )

    # ── 6. Code-scanning alerts gate via a ruleset and render as no check row. ─
    for a in alerts:
        findings.append(
            Finding(
                "BLOCKING",
                "code-scanning-alert",
                f"#{a.get('number')} {a.get('severity')} {a.get('rule_id')} at "
                f"{a.get('path')}:{a.get('line')} — open on this ref (it may predate the PR). Fix it, or "
                "dismiss it as a false positive WITH a written reason; a ruleset code_scanning rule blocks on it",
            )
        )

    # ── 7. Name the gates that never appear as check rows. ────────────────────
    for rule in sorted(set(ruleset_rules)):
        if rule in {"code_scanning", "required_status_checks", "pull_request", "required_signatures"}:
            findings.append(
                Finding("INFO", "ruleset-gate", f"an effective ruleset {rule} rule gates this merge (not a check row)")
            )

    blocking = [f for f in findings if f.severity == "BLOCKING"]
    unsettled_findings = [f for f in findings if f.severity == "UNSETTLED"]

    # ── 8. BLOCKED with nothing to show for it is itself the finding. ─────────
    if state == "BLOCKED" and not blocking and not unsettled_findings:
        findings.append(
            Finding(
                "BLOCKING",
                "blocked-unexplained",
                "mergeStateStatus is BLOCKED but no check, alert, or readable ruleset rule explains it — "
                "look for required reviews, CODEOWNERS, or an org-level ruleset "
                "(`gh api repos/<o>/<r>/rules/branch/<base>`)",
            )
        )
        blocking = [f for f in findings if f.severity == "BLOCKING"]

    if blocking:
        return Readiness(BLOCKED, findings)
    if unsettled_findings:
        return Readiness(UNSETTLED, findings)
    return Readiness(READY, findings)


# ── thin I/O shell ────────────────────────────────────────────────────────────


def _gh_json(args: list[str], *, timeout: float = 60.0) -> Any:
    proc = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} failed: {proc.stderr.strip()}")
    out = proc.stdout.strip()
    if not out:
        raise RuntimeError(f"gh {' '.join(args)} returned no output")
    return json.loads(out)


def _slurped_check_runs(slug: str, sha: str) -> list[dict[str, Any]]:
    """All check runs across pages. `--slurp` wraps each page, so flatten every page."""
    pages = _gh_json(["api", f"repos/{slug}/commits/{sha}/check-runs", "--paginate", "--slurp"])
    if isinstance(pages, dict):  # defensive: a single un-slurped object
        pages = [pages]
    runs: list[dict[str, Any]] = []
    for page in pages or []:
        runs.extend((page or {}).get("check_runs", []) or [])
    return [
        {
            "name": c.get("name"),
            "id": c.get("id"),
            "started_at": c.get("started_at"),
            "app_id": (c.get("app") or {}).get("id"),
            "status": c.get("status"),
            "conclusion": c.get("conclusion"),
            "output_title": (c.get("output") or {}).get("title"),
            "details_url": c.get("details_url"),
        }
        for c in runs
    ]


def _suite_reuse(slug: str, run: dict[str, Any] | None) -> str:
    """'measured', 'reused', '' (nothing to look up) or 'unknown: <why>' for the suite-of-record run.

    Read from the run's own job (exact per attempt), not the run's artifact list, which keeps
    an attempt-1 `coverage-reused-from-1` listed after an attempt-2 re-run that measured.
    A failed read is reported, never a blocker: re-use is a fact worth stating, not a gate.
    """
    if run is None or str(run.get("conclusion") or "").lower() != "success":
        return ""
    match = re.search(r"/job/(\d+)", str(run.get("details_url") or ""))
    if not match:
        return "unknown: the unit-tests check run names no job"
    try:
        job = _gh_json(["api", f"repos/{slug}/actions/jobs/{match.group(1)}"]) or {}
    except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        return f"unknown: could not read job {match.group(1)} ({exc})"
    for step in job.get("steps") or []:
        if step.get("name") == _FAST_SUITE_STEP:
            conclusion = str(step.get("conclusion") or "").lower()
            if conclusion == "skipped":
                return "reused"
            if conclusion == "success":
                return "measured"
            return f"unknown: the {_FAST_SUITE_STEP!r} step ended {conclusion or 'without a conclusion'}"
    return f"unknown: job {match.group(1)} has no {_FAST_SUITE_STEP!r} step"


def _fetch(pr_number: int, repo: str | None) -> dict[str, Any]:
    """Every gating surface, returned as `evaluate`'s keyword arguments."""
    repo_args = ["--repo", repo] if repo else []
    pr = _gh_json(
        ["pr", "view", str(pr_number), *repo_args, "--json", "mergeable,mergeStateStatus,headRefOid,baseRefName,state"]
    )
    slug = repo or _gh_json(["repo", "view", "--json", "nameWithOwner"])["nameWithOwner"]
    errors: list[str] = []

    checks = _slurped_check_runs(slug, pr["headRefOid"])
    suite_app_id = pinned_suite_app_id()
    suite_reuse = _suite_reuse(slug, suite_of_record_run(checks, suite_app_id))
    # A push between the two reads would make READY name checks of a stale head (design review finding 1).
    head_now = (_gh_json(["pr", "view", str(pr_number), *repo_args, "--json", "headRefOid"]) or {}).get("headRefOid")
    if head_now != pr["headRefOid"]:
        errors.append(f"the PR head moved from {pr['headRefOid']} to {head_now} while its checks were read")

    alerts: list[dict[str, Any]] = []
    try:
        raw = _gh_json(["api", f"repos/{slug}/code-scanning/alerts?pr={pr_number}&state=open", "--paginate", "--slurp"])
        flat = [a for page in (raw or []) for a in (page or [])] if isinstance(raw, list) else []
        alerts = [
            {
                "number": a.get("number"),
                "rule_id": (a.get("rule") or {}).get("id"),
                "severity": (a.get("rule") or {}).get("security_severity_level")
                or (a.get("rule") or {}).get("severity"),
                "path": ((a.get("most_recent_instance") or {}).get("location") or {}).get("path"),
                "line": ((a.get("most_recent_instance") or {}).get("location") or {}).get("start_line"),
            }
            for a in flat
        ]
    except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        errors.append(f"could not read code-scanning alerts ({exc})")

    rules, contexts, rule_err = _branch_rules(slug, pr.get("baseRefName") or "")
    if rule_err:
        errors.append(rule_err)

    return {
        "pr": pr,
        "checks": checks,
        "alerts": alerts,
        "ruleset_rules": rules,
        "required_contexts": tuple(contexts) or None,
        "fetch_errors": tuple(errors),
        "suite_app_id": suite_app_id,
        "head_sha": pr["headRefOid"],
        "suite_reuse": suite_reuse,
    }


def _branch_rules(slug: str, base: str) -> tuple[list[str], list[str], str | None]:
    """Effective rules for ``base``, falling back to the repo's active rulesets.

    ``/rules/branch/{branch}`` is the precise answer (it honours enforcement and ref
    conditions) but 404s on some repos/plans, so a 404 there is NOT "no rules" — it
    falls back to enumerating active rulesets. Only when BOTH fail is the surface
    reported unverified, because silently returning [] would tell the reader that a
    gate they cannot see does not exist.
    """
    rules: list[str] = []
    contexts: list[str] = []
    try:
        for rule in _gh_json(["api", f"repos/{slug}/rules/branch/{base}"]) or []:
            rtype = rule.get("type")
            if rtype:
                rules.append(rtype)
            if rtype == "required_status_checks":
                for chk in (rule.get("parameters") or {}).get("required_status_checks", []) or []:
                    if chk.get("context"):
                        contexts.append(chk["context"])
        return rules, contexts, None
    except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass  # not fatal — fall back below

    try:
        for rs in _gh_json(["api", f"repos/{slug}/rulesets"]) or []:
            detail = _gh_json(["api", f"repos/{slug}/rulesets/{rs['id']}"]) or {}
            if str(detail.get("enforcement", "")).lower() != "active":
                continue
            for rule in detail.get("rules", []) or []:
                rtype = rule.get("type")
                if rtype:
                    rules.append(rtype)
                if rtype == "required_status_checks":
                    for chk in (rule.get("parameters") or {}).get("required_status_checks", []) or []:
                        if chk.get("context"):
                            contexts.append(chk["context"])
        return rules, contexts, None
    except (RuntimeError, KeyError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        return rules, contexts, f"could not read branch rules or rulesets ({exc})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report every surface gating a PR's merge.")
    parser.add_argument("pr", type=int, help="PR number")
    parser.add_argument("--repo", default=None, help="owner/name (defaults to the current repo)")
    args = parser.parse_args(argv)

    try:
        surfaces = _fetch(args.pr, args.repo)
    except (RuntimeError, KeyError, TypeError, ValueError, subprocess.TimeoutExpired, OSError) as exc:
        print(f"error: could not read PR state: {exc}", file=sys.stderr)
        return TOOL_ERROR

    result = evaluate(**surfaces)
    pr = surfaces["pr"]
    print(
        f"PR #{args.pr}: {pr.get('state')} / mergeable={pr.get('mergeable')} / {pr.get('mergeStateStatus')}"
        f" / head {pr.get('headRefOid')}"
    )
    print(result.render())
    if result.exit_code == READY and str(pr.get("state", "")).upper() == "OPEN":
        print(f"\nThis verdict is for head {pr.get('headRefOid')}; a later push needs a fresh run.")
    if result.exit_code == UNSETTLED:
        print("\nNo verdict: something is still in flight or unverified. Re-run when it settles.")
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
