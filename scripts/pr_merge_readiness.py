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
   Right after a push, checks legitimately do not exist yet. A settled *failure* is a
   positive fact and does outrank in-flight; an absence does not.
2. **Report what each surface SAYS, never a remembered cause.** No list of known causes
   is encoded here — the list is always missing the next variant. A surface that could
   not be read is reported as UNVERIFIED, never as "nothing found".

Usage:
    python scripts/pr_merge_readiness.py 654
    python scripts/pr_merge_readiness.py 654 --repo owner/name

Exit codes: 0 ready · 1 blocked (reasons printed) · 2 unsettled/unverified (no verdict —
re-run when it settles) · 3 tool error.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

# Fallback only — the effective ruleset contexts are preferred when readable.
# CodeQL is included per bugs ledger D63, whose durable fix names it explicitly.
REQUIRED_CHECK_PATTERNS: tuple[str, ...] = ("unit-tests", "lint", "release build", "codeql")

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
        if not self.findings:
            return "READY: no gating surface reports a problem."
        order = {"BLOCKING": 0, "UNSETTLED": 1, "INFO": 2}
        return "\n".join(
            f"[{f.severity}] {f.code}: {f.detail}"
            for f in sorted(self.findings, key=lambda f: (order.get(f.severity, 9), f.code))
        )


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


def evaluate(
    *,
    pr: dict[str, Any],
    checks: list[dict[str, Any]],
    alerts: list[dict[str, Any]],
    ruleset_rules: list[str],
    required_patterns: tuple[str, ...] = REQUIRED_CHECK_PATTERNS,
    required_contexts: tuple[str, ...] | None = None,
    fetch_errors: tuple[str, ...] = (),
) -> Readiness:
    """Pure verdict over already-fetched data (so this is testable without network)."""
    findings: list[Finding] = []

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
            "status": c.get("status"),
            "conclusion": c.get("conclusion"),
            "output_title": (c.get("output") or {}).get("title"),
        }
        for c in runs
    ]


def _fetch(
    pr_number: int, repo: str | None
) -> tuple[dict, list[dict], list[dict], list[str], tuple[str, ...], list[str]]:
    repo_args = ["--repo", repo] if repo else []
    pr = _gh_json(
        ["pr", "view", str(pr_number), *repo_args, "--json", "mergeable,mergeStateStatus,headRefOid,baseRefName,state"]
    )
    slug = repo or _gh_json(["repo", "view", "--json", "nameWithOwner"])["nameWithOwner"]
    errors: list[str] = []

    checks = _slurped_check_runs(slug, pr["headRefOid"])

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

    return pr, checks, alerts, rules, tuple(contexts), errors


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
        pr, checks, alerts, rules, contexts, errors = _fetch(args.pr, args.repo)
    except (RuntimeError, KeyError, TypeError, ValueError, subprocess.TimeoutExpired, OSError) as exc:
        print(f"error: could not read PR state: {exc}", file=sys.stderr)
        return TOOL_ERROR

    result = evaluate(
        pr=pr,
        checks=checks,
        alerts=alerts,
        ruleset_rules=rules,
        required_contexts=contexts or None,
        fetch_errors=tuple(errors),
    )
    print(f"PR #{args.pr}: {pr.get('state')} / mergeable={pr.get('mergeable')} / {pr.get('mergeStateStatus')}")
    print(result.render())
    if result.exit_code == UNSETTLED:
        print("\nNo verdict: something is still in flight or unverified. Re-run when it settles.")
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
