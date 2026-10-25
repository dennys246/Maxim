#!/usr/bin/env python3
"""The publish workflow's own gates (#1372): is this tag the release, is the publish gate in place, are the notes ready?

`.github/workflows/release.yml` publishes pymaxim to PyPI from a pushed `v*` tag, through trusted publishing (OIDC,
no stored token), after the owner approves the `pypi` environment. These are the checks it runs that no existing
script did (design pass on docs/plans/archive/release_pipeline.md, folds SF1, SF2, SF3, SF8, SF9):

``tag``          the tag names pyproject's and ``__init__``'s version; it is an ANNOTATED tag object (a lightweight
                 tag is refused), GitHub-verified (``verification.verified`` with reason ``valid``) and signed by the
                 expected tagger email; it peels to the commit the run checked out; that commit is on ``origin/main``.
                 Prints the tag-object sha for ``still``.
``still``        after the approval pause: the tag still resolves to the same tag object and commit. An owner-bypass
                 re-tag while an old run waits at the gate would otherwise publish the superseded commit (SF2).
``environment``  the ``pypi`` environment exists with required reviewers and deploys only from ``v*`` tags. GitHub
                 re-creates a missing environment UNPROTECTED the first time a job names it, and PyPI's trusted
                 publisher checks only the environment's name, so this runs in ``build``, before ``publish`` can be
                 scheduled (SF1).
``ruleset``      an active tag ruleset covers ``refs/tags/v*`` and restricts creating, updating and deleting those tags,
                 so only the owner (bypass) can push a release tag. #1372's own "Done when" item, checked where it is
                 relied on (architecture review S1).
``notes``        ``docs/announcements/release_<version>.md`` exists and has no repo-relative links, checked BEFORE the
                 irreversible upload rather than after it (SF8).

Exits: 0 pass; 1 refused; 2 could not check (an API or git failure is never a pass).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_release_tags import _REL_LINK  # noqa: E402

_INIT_VERSION = re.compile(r'^__version__\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)

Api = Callable[[str], Any]
Git = Callable[..., subprocess.CompletedProcess[str]]


class Unverified(Exception):
    """An input could not be read: exit 2."""


def gh_api(path: str) -> Any:
    proc = subprocess.run(("gh", "api", path), capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise Unverified(f"gh api {path}: {proc.stderr.strip() or proc.returncode}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise Unverified(f"gh api {path}: unreadable JSON ({exc})") from exc


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(("git", *args), cwd=REPO, capture_output=True, text=True, check=False)


def repo_versions(root: Path = REPO) -> tuple[str, str | None]:
    pyproject = str(tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"])
    m = _INIT_VERSION.search((root / "src" / "maxim" / "__init__.py").read_text())
    return pyproject, (m.group(1) if m else None)


def resolve_tag(tag: str, *, repo: str, api: Api) -> tuple[str, dict[str, Any]]:
    """(tag-object sha, tag object). Refuses a lightweight tag (its ref points straight at a commit)."""
    ref = api(f"repos/{repo}/git/ref/tags/{tag}")
    obj = (ref or {}).get("object") or {}
    if obj.get("type") != "tag":
        raise LookupError(f"{tag} is a {obj.get('type') or 'missing'} ref, not an annotated tag object")
    return str(obj["sha"]), api(f"repos/{repo}/git/tags/{obj['sha']}")


def tag_problems(
    tag: str, sha: str, *, repo: str, tagger_email: str, api: Api = gh_api, run_git: Git = git, root: Path = REPO
) -> tuple[list[str], str | None]:
    problems: list[str] = []
    pyproject, init = repo_versions(root)
    if tag != f"v{pyproject}" or init != pyproject:
        problems.append(f"tag {tag} vs pyproject {pyproject} vs __init__ {init}: they must all name one version")
    try:
        tag_sha, tag_obj = resolve_tag(tag, repo=repo, api=api)
    except LookupError as exc:
        return [*problems, str(exc)], None
    verification = tag_obj.get("verification") or {}
    if verification.get("verified") is not True or verification.get("reason") != "valid":
        problems.append(f"{tag} is not a GitHub-verified signed tag ({verification.get('reason')!r})")
    if (tag_obj.get("tagger") or {}).get("email") != tagger_email:
        problems.append(f"{tag} was tagged by {(tag_obj.get('tagger') or {}).get('email')!r}, not {tagger_email!r}")
    target = tag_obj.get("object") or {}
    if target.get("type") != "commit" or target.get("sha") != sha:
        problems.append(f"{tag} points at {target.get('type')} {target.get('sha')}, but this run checked out {sha}")
    ancestor = run_git("merge-base", "--is-ancestor", sha, "origin/main")
    if ancestor.returncode == 1:
        problems.append(f"{sha[:12]} is not on origin/main: a release tag names reviewed, merged code")
    elif ancestor.returncode != 0:
        raise Unverified(f"git merge-base: {ancestor.stderr.strip() or ancestor.returncode}")
    return problems, tag_sha


def still_problems(tag: str, tag_sha: str, sha: str, *, repo: str, api: Api = gh_api) -> list[str]:
    try:
        now_sha, tag_obj = resolve_tag(tag, repo=repo, api=api)
    except LookupError as exc:
        return [str(exc)]
    problems = []
    if now_sha != tag_sha:
        problems.append(f"{tag} was re-tagged during the approval wait ({tag_sha[:12]} -> {now_sha[:12]})")
    if (tag_obj.get("object") or {}).get("sha") != sha:
        problems.append(f"{tag} no longer peels to {sha[:12]}: this run's build is not the tag's code")
    return problems


def environment_problems(*, repo: str, name: str = "pypi", api: Api = gh_api) -> list[str]:
    env = api(f"repos/{repo}/environments/{name}")
    problems = []
    reviewers = [
        r for rule in env.get("protection_rules") or [] if rule.get("type") == "required_reviewers"
        for r in rule.get("reviewers") or []
    ]  # fmt: skip
    if not reviewers:
        problems.append(f"the {name} environment has no required reviewer: a tag push would publish unapproved")
    policy = env.get("deployment_branch_policy") or {}
    if policy.get("custom_branch_policies") is not True:
        problems.append(f"the {name} environment is not limited to selected tags")
    else:
        rules = api(f"repos/{repo}/environments/{name}/deployment-branch-policies").get("branch_policies") or []
        got = sorted((str(r.get("name")), str(r.get("type"))) for r in rules)
        if got != [("v*", "tag")]:
            problems.append(f"the {name} environment deploys from {got}, expected exactly [('v*', 'tag')]")
    return problems


RULESET_RULES = {"creation", "update", "deletion"}


def ruleset_problems(*, repo: str, api: Api = gh_api) -> list[str]:
    for summary in api(f"repos/{repo}/rulesets") or []:
        if summary.get("target") != "tag" or summary.get("enforcement") != "active":
            continue
        ruleset = api(f"repos/{repo}/rulesets/{summary['id']}")
        include = ((ruleset.get("conditions") or {}).get("ref_name") or {}).get("include") or []
        rules = {r.get("type") for r in ruleset.get("rules") or []}
        if "refs/tags/v*" in include and RULESET_RULES <= rules:
            return []
    return [
        "no active tag ruleset covers refs/tags/v* restricting creation, update and deletion: "
        "anyone with push access could tag a release (docs/plans/archive/release_pipeline.md, owner setup step 2)"
    ]


def notes_problems(version: str, root: Path = REPO) -> list[str]:
    notes = root / "docs" / "announcements" / f"release_{version.replace('.', '_')}.md"
    if not notes.is_file():
        return [f"no release notes at {notes.relative_to(root)}: write them before tagging"]
    links = _REL_LINK.findall(notes.read_text(encoding="utf-8"))
    return [f"{notes.relative_to(root)} has repo-relative links (404 on the Releases page): {links}"] if links else []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tag")
    t.add_argument("--tag", required=True)
    t.add_argument("--sha", required=True)
    t.add_argument("--repo", required=True)
    t.add_argument("--tagger-email", required=True)
    s = sub.add_parser("still")
    s.add_argument("--tag", required=True)
    s.add_argument("--tag-object", required=True)
    s.add_argument("--sha", required=True)
    s.add_argument("--repo", required=True)
    e = sub.add_parser("environment")
    e.add_argument("--repo", required=True)
    r = sub.add_parser("ruleset")
    r.add_argument("--repo", required=True)
    n = sub.add_parser("notes")
    n.add_argument("--version", required=True)
    args = ap.parse_args(argv)
    tag_sha = None
    try:
        if args.cmd == "tag":
            problems, tag_sha = tag_problems(args.tag, args.sha, repo=args.repo, tagger_email=args.tagger_email)
        elif args.cmd == "still":
            problems = still_problems(args.tag, args.tag_object, args.sha, repo=args.repo)
        elif args.cmd == "environment":
            problems = environment_problems(repo=args.repo)
        elif args.cmd == "ruleset":
            problems = ruleset_problems(repo=args.repo)
        else:
            problems = notes_problems(args.version)
    except (Unverified, KeyError, TypeError, AttributeError, OSError) as exc:
        print(f"UNVERIFIED: {exc!r}", file=sys.stderr)
        return 2
    if problems:
        print(f"REFUSED ({args.cmd}):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    if tag_sha:
        print(f"tag_object={tag_sha}")
    print(f"{args.cmd}: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
