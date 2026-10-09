#!/usr/bin/env python3
"""The merge gate's settings stay as committed (closes mechanization backlog M6 and D63's enforcement half).

GitHub itself refuses a merge while a required check is absent ("Expected — waiting for status"), given strict
required checks, admins enforced and no ruleset bypass. So the risk M6/D63 still carried was SETTINGS DRIFT: a removed
required check, admins un-enforced, a new ruleset, CodeQL losing a language. Owner decisions 2026-10-06: detect the
drift, don't duplicate the block. The settings are pinned in ``scripts/repo_settings_expected.json`` (a deliberate
change edits it in a reviewed PR; ``--snapshot`` rewrites it from the live state).

**Live check** (nightly job ``Repo settings (nightly)``, read by the release gate; token: ``SETTINGS_READ_TOKEN``, a
fine-grained PAT with Administration: Read-only, held in the ``settings-check`` environment limited to ``main`` so no PR
run can read it):

- ``default_branch`` is ``main`` (a ``~DEFAULT_BRANCH`` ruleset follows the default branch);
- the whole classic protection object on ``main`` (``url`` fields stripped): required contexts pinned per app,
  strict, admins enforced, reviews, signatures, linear history, force-push and deletion;
- the EFFECTIVE rules on ``main`` (``/rules/branches/main``, every ruleset that applies, globs and excludes resolved
  by GitHub) and each one's ruleset ``enforcement``;
- CodeQL default setup: state, languages, query suite;
- ``security_and_analysis`` (#1081, owner decision D3 2026-10-08): secret scanning, push protection (the only layer
  that runs BEFORE a secret is published) and non-provider patterns stay on, so none can be switched off silently;
  an absent object (token scope) is exit 2;
- the ``settings-check`` environment exists and is fenced to ``main`` (GitHub creates a missing environment UNFENCED
  the first time a job names it, so the owner creates it first and the check pins the fence);
- **no ruleset bypass was USED on ``main``** in the last month (paginated) (``rule-suites`` with ``rule_suite_result=bypass``),
  except entries in ``acknowledged_bypasses``. The configured ``bypass_actors`` list needs Administration WRITE to
  read (adversarial pass), so the check watches for its use instead.

**Static check** (``--static``, lint job, no token): each pinned required context is the display name of exactly one
job across ``.github/workflows/*.yml``, and that job has no job-level ``if:`` (a skipped required check counts as
passing, and a second same-named job could satisfy it).

Exit 0 clean; 1 drift; 2 cannot verify (no token, a non-200 read, a missing field). Never a pass on an unreadable read.

Residuals (stated): a snapshot cannot see weaken → merge → restore inside a day, and a user-owned repo has no
audit-log API to cover it; the expected file can be weakened by a reviewed PR (this catches drift outside a PR, not
a reviewed decision), and that includes ``acknowledged_bypasses``: the admin who bypassed can acknowledge it, one
suite id per entry, in a reviewed PR; a bypass older than a month when the nightly has been red that long is missed;
CodeQL default setup analyses PRs to the default branch only; tag rulesets (``o19-markers``) are outside the pin,
since this guards the merge gate on ``main``.

Regression guard: tests/unit/test_check_repo_settings.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED = REPO_ROOT / "scripts" / "repo_settings_expected.json"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
BRANCH = "main"
ENVIRONMENT = "settings-check"
BYPASS_PAGE = 100


class CannotVerify(Exception):
    """A read failed or a field is missing: exit 2, never a pass."""


class Unprotected(Exception):
    """GitHub says the branch has no classic protection at all: drift."""


class NotFound(CannotVerify):
    """HTTP 404. Fail-closed by default (it subclasses CannotVerify); only the environment read treats it as drift."""


def gh_api(path: str) -> Any:
    r = subprocess.run(["gh", "api", path], capture_output=True, text=True, timeout=60, check=False)
    if r.returncode != 0:
        if "Branch not protected" in r.stderr:
            raise Unprotected(path)
        if "(HTTP 404)" in r.stderr:
            raise NotFound(f"gh api {path}: 404")
        raise CannotVerify(f"gh api {path}: {r.stderr.strip() or r.returncode}")
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError as exc:
        raise CannotVerify(f"gh api {path}: unreadable JSON ({exc})") from exc


def _strip_urls(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: _strip_urls(v) for k, v in o.items() if not k.endswith("url")}
    if isinstance(o, list):
        return [_strip_urls(x) for x in o]
    return o


def _field(d: Any, key: str, where: str) -> Any:
    if not isinstance(d, dict) or key not in d:
        raise CannotVerify(f"{where}: no `{key}` in the response (token scope?)")
    return d[key]


def observe(repo: str, api: Callable[[str], Any] = gh_api) -> dict[str, Any]:
    """The live settings, normalised for comparison. Raises CannotVerify on any unreadable part."""
    repo_obj = api(f"repos/{repo}")
    out: dict[str, Any] = {"default_branch": _field(repo_obj, "default_branch", "repo")}
    # GitHub omits (or nulls) this object for a token without admin read: that is "cannot verify", never a pass.
    security = _field(repo_obj, "security_and_analysis", "repo")
    if not isinstance(security, dict):
        raise CannotVerify("repo: `security_and_analysis` is not an object (token scope?)")
    out["security_and_analysis"] = security
    try:
        out["protection"] = _strip_urls(api(f"repos/{repo}/branches/{BRANCH}/protection"))
    except Unprotected:
        out["protection"] = "UNPROTECTED"
    rules = api(f"repos/{repo}/rules/branches/{BRANCH}?per_page=100")
    if not isinstance(rules, list):
        raise CannotVerify("effective rules: not a list")
    out["effective_rules"] = sorted(
        ({"type": r.get("type"), "parameters": r.get("parameters"), "ruleset_id": r.get("ruleset_id")} for r in rules),
        key=lambda r: (str(r["type"]), int(r["ruleset_id"] or 0)),
    )
    out["ruleset_enforcement"] = {
        str(rid): _field(api(f"repos/{repo}/rulesets/{rid}"), "enforcement", f"ruleset {rid}")
        for rid in sorted({r["ruleset_id"] for r in out["effective_rules"] if r["ruleset_id"]})
    }
    setup = api(f"repos/{repo}/code-scanning/default-setup")
    out["codeql_default_setup"] = {
        "state": _field(setup, "state", "code-scanning default setup"),
        "languages": sorted(_field(setup, "languages", "code-scanning default setup") or []),
        "query_suite": _field(setup, "query_suite", "code-scanning default setup"),
    }
    out["token_environment"] = _environment_fence(repo, api)
    suites: list[Any] = []
    for page in range(1, 101):
        batch = api(
            f"repos/{repo}/rulesets/rule-suites?ref=refs/heads/{BRANCH}&rule_suite_result=bypass"
            f"&time_period=month&per_page={BYPASS_PAGE}&page={page}"
        )
        if not isinstance(batch, list):
            raise CannotVerify("rule suites: not a list")
        suites.extend(batch)
        if len(batch) < BYPASS_PAGE:
            break
    else:
        raise CannotVerify("rule suites: more than 100 pages")
    out["bypasses"] = [
        {"id": s.get("id"), "actor": s.get("actor_name"), "at": s.get("pushed_at"), "sha": s.get("after_sha")}
        for s in suites
    ]
    return out


def _environment_fence(repo: str, api: Callable[[str], Any]) -> Any:
    """The token's environment and its deployment-branch fence. GitHub auto-creates a missing environment, UNFENCED, the
    first time a job references it, so a missing or unfenced environment is drift, not "cannot verify"."""
    try:
        env = api(f"repos/{repo}/environments/{ENVIRONMENT}")
    except NotFound:
        return "MISSING"
    policy = _field(env, "deployment_branch_policy", f"environment {ENVIRONMENT}")
    fence: dict[str, Any] = {"deployment_branch_policy": policy}
    if isinstance(policy, dict) and policy.get("custom_branch_policies"):
        listed = api(f"repos/{repo}/environments/{ENVIRONMENT}/deployment-branch-policies?per_page=100")
        fence["branch_policies"] = sorted(
            (
                {"name": b.get("name"), "type": b.get("type")}
                for b in _field(listed, "branch_policies", "branch policies")
            ),
            key=lambda b: (str(b["type"]), str(b["name"])),
        )
    return fence


PINNED = (
    "default_branch",
    "protection",
    "effective_rules",
    "ruleset_enforcement",
    "codeql_default_setup",
    "token_environment",
    "security_and_analysis",
)


def _diff(key: str, exp: Any, obs: Any) -> list[str]:
    """Field by field through dicts, so a key GitHub adds or drops is named, not buried in a whole-object dump."""
    if isinstance(exp, dict) and isinstance(obs, dict):
        out = []
        for k in sorted(set(exp) | set(obs)):
            if k not in exp:
                out.append(
                    f"{key}.{k} drifted: a NEW GitHub field (review it, then pin it): {json.dumps(obs[k], sort_keys=True)}"
                )
            elif k not in obs:
                out.append(f"{key}.{k} drifted: the field is GONE (expected {json.dumps(exp[k], sort_keys=True)})")
            else:
                out.extend(_diff(f"{key}.{k}", exp[k], obs[k]))
        return out
    if exp == obs:
        return []
    return [
        f"{key} drifted:\n    expected {json.dumps(exp, sort_keys=True)}\n    observed {json.dumps(obs, sort_keys=True)}"
    ]


def compare(expected: dict[str, Any], observed: dict[str, Any]) -> list[str]:
    out = []
    for key in PINNED:
        out.extend(_diff(key, expected.get(key), observed.get(key)))
    acked = {str(a.get("id")) for a in expected.get("acknowledged_bypasses", []) if isinstance(a, dict)}
    for b in observed.get("bypasses", []):
        if str(b["id"]) not in acked:
            out.append(f"a ruleset was BYPASSED on {BRANCH}: suite {b['id']} by {b['actor']} at {b['at']} ({b['sha']})")
    return out


def static_problems(contexts: list[str], workflows: Path = WORKFLOWS) -> list[str]:
    """Each required context is exactly one job's display name, and that job cannot be skipped by a job-level `if:`."""
    jobs: list[tuple[str, str, dict]] = []
    for wf in sorted(workflows.glob("*.y*ml")):
        doc = yaml.safe_load(wf.read_text(encoding="utf-8")) or {}
        for jid, job in (doc.get("jobs") or {}).items():
            if isinstance(job, dict):
                jobs.append((wf.name, jid, job))
    out = []
    for ctx in contexts:
        hits = [(wf, jid, job) for wf, jid, job in jobs if str(job.get("name", jid)) == ctx]
        if len(hits) != 1:
            out.append(
                f"required check {ctx!r} is the name of {len(hits)} jobs (must be exactly one): {hits and [h[:2] for h in hits]}"
            )
            continue
        wf, jid, job = hits[0]
        if "if" in job:
            out.append(f"required check {ctx!r} ({wf}::{jid}) has a job-level `if:`: a skipped required check passes")
        for dep in _skippable_needs(wf, jid, jobs):
            out.append(
                f"required check {ctx!r} ({wf}::{jid}) needs {dep}, which has a job-level `if:`: skipped with it"
            )
        if job.get("strategy", {}).get("matrix") if isinstance(job.get("strategy"), dict) else False:
            out.append(f"required check {ctx!r} ({wf}::{jid}) is a matrix job: its check names carry the matrix values")
    return out


def _skippable_needs(wf: str, jid: str, jobs: list[tuple[str, str, dict]]) -> list[str]:
    """Every job the given one transitively `needs:` that has a job-level `if:` (a skipped need skips the dependant)."""
    same = {j: job for w, j, job in jobs if w == wf}
    seen: set[str] = set()
    stack = [jid]
    out = []
    while stack:
        needs = same.get(stack.pop(), {}).get("needs") or []
        for dep in [needs] if isinstance(needs, str) else needs:
            if dep in seen:
                continue
            seen.add(dep)
            if "if" in same.get(dep, {}):
                out.append(dep)
            stack.append(dep)
    return sorted(out)


def _expected() -> dict[str, Any]:
    return json.loads(EXPECTED.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    repo = os.environ.get("GITHUB_REPOSITORY", "dennys246/Maxim")
    try:
        expected = _expected()
        contexts = list(expected["protection"]["required_status_checks"]["contexts"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {EXPECTED.name} unreadable: {exc}", file=sys.stderr)
        return 2
    if argv == ["--static"]:
        problems = static_problems(contexts)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print(f"repo settings (static): {len(contexts)} required checks, {len(problems)} problem(s)")
        return 1 if problems else 0
    if not os.environ.get("GH_TOKEN"):
        print(
            "ERROR: no GH_TOKEN: the settings check needs SETTINGS_READ_TOKEN (fine-grained PAT, Administration: "
            "Read-only, this repo only) in the `settings-check` environment. Cannot verify, so this is not a pass.",
            file=sys.stderr,
        )
        return 2
    try:
        observed = observe(repo)
    except CannotVerify as exc:
        print(f"ERROR: cannot verify the repo settings: {exc}", file=sys.stderr)
        return 2
    if argv == ["--snapshot"]:
        snap = {k: observed[k] for k in PINNED}
        snap["acknowledged_bypasses"] = expected.get("acknowledged_bypasses", [])
        EXPECTED.write_text(json.dumps(snap, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {EXPECTED.relative_to(REPO_ROOT)}: review the diff before committing")
        return 0
    problems = compare(expected, observed) + static_problems(contexts)
    if problems:
        print(
            "repo settings DRIFTED (a deliberate change edits scripts/repo_settings_expected.json in a PR):",
            file=sys.stderr,
        )
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print(
        f"repo settings: as committed ({len(contexts)} required checks, {len(observed['effective_rules'])} rules on {BRANCH})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
