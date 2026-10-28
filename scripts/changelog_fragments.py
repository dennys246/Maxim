#!/usr/bin/env python3
"""CHANGELOG fragments (M83 step 3): ``check`` is the versioning lint, ``assemble`` is the release cut.

Each ``src/`` change adds one file, ``changelog.d/<slug>.<kind>.md``, instead of a line under
``CHANGELOG.md``'s ``## [Unreleased]``. Every PR appended at the top of that one section, so it was the commonest
conflict between parallel PRs (2026-10-27: #1502, #1506 and step 0b in one day). The bump PR runs ``assemble``,
which folds every fragment, plus anything still under ``[Unreleased]``, into the new version section.

This replaces ``lint_unreleased_on_src_change.py`` and ``lint_unreleased_declared.py``. Those enforced nearly the
same rule as two CI steps with different scopes and different release-transaction tests. The design and its four
adversarial rounds: ``docs/plans/changelog_fragments_m83_approach.md``. The finding ids (S1-S5, R1-R5, F1-F3)
cited below are that note's.

RULES (``check``, against the merge-base with origin/main, or on a push one first-parent unit at a time)
-------------------------------------------------------------------------------------------------------------
1. A diff touching any path under ``src/`` (S5) ADDS a valid fragment. Under ``-M -C --find-copies-harder`` a
   rename (``R``) never counts (S1); an add (``A``) or a copy (``C``) counts unless it repeats another fragment:
   an exact copy at any length, or 0.9 or more similar for bodies of 80+ characters (``autojunk=False``; R2, F2).
   A version bump is exempt (rule 3 applies instead).
2. Outside a bump, no fragment is deleted or renamed away (S2). Rewording one is fine.
3. A bump (the ``pyproject.toml`` version changed) consumes every fragment exactly once (S3, S4, R4, F1): the new
   section's ``<!-- changelog.d/<file> -->`` markers cover the required set, stay inside the permitted set, appear
   once each, and are each followed by a ``- `` entry (R1, F3). No other ``<!--`` is in the section, no fragment
   is left in ``changelog.d/``, and the version has one heading (R3).
4. Outside a bump, ``[Unreleased]`` gains no line (comment lines count; N5, n8), and no ``## [`` heading is added.
5. Every fragment is a regular file directly in ``changelog.d/``, with a valid name and kind, at least one
   content line, starting with ``- ``, with no heading and no ``<!--`` (n1, n6).

On a pull request "could not check" is a hard error (exit 2), never a silent skip: the D37 rule in
``scripts/_lint_git.py``. A push is judged per merged PR (``push_units``), so a rebase merge is one unit.

Residuals, stated: a deliberately rewritten copy of another entry is not detectable; a short near-copy (under 80
characters) passes; and ``assemble`` does not parse code fences, so a column-0 ``## `` line inside a fence under
``[Unreleased]`` would split the section (none exists today).

Exits: 0 clean; 1 violations; 2 could not check.
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lint_git import GitUnavailable, base_ref, git, must_not_skip, push_units, show  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

CHANGELOG = "CHANGELOG.md"
PYPROJECT = "pyproject.toml"
FRAG_DIR = "changelog.d"
README = f"{FRAG_DIR}/README.md"
SRC_PREFIX = "src/"

# Assembly order: Breaking first, then Keep-a-Changelog's order of interest to a reader.
KINDS = ("breaking", "security", "added", "changed", "deprecated", "removed", "fixed")
HEADING_OF = {k: k.capitalize() for k in KINDS}

SIMILARITY = 0.9
SIMILARITY_FLOOR = 80  # normalized characters; shorter entries look alike without being copies (F2)

_NAME_RE = re.compile(rf"^(?P<slug>[a-z0-9][a-z0-9-]*)\.(?P<kind>{'|'.join(KINDS)})\.md$")
_ANY_KIND_NAME_RE = re.compile(r"^[^/]+\.(?P<kind>[^./]+)\.md$")
_MARKER_RE = re.compile(rf"^<!-- {re.escape(FRAG_DIR)}/(?P<name>[^ ]+) -->$")
_VERSION_RE = re.compile(r'^version\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)
_UNRELEASED_RE = re.compile(r"^##\s*\[Unreleased\]\s*$", re.IGNORECASE)
_VERSION_HEADING_RE = re.compile(r"^##\s*\[(?P<version>\d+\.\d+\.\d+[^\]]*)\]")
_H2_RE = re.compile(r"^##\s")
_H3_RE = re.compile(r"^###\s+(?P<text>.+?)\s*$")
_ANY_HEADING_RE = re.compile(r"^#{1,6}\s")
_RELEASE_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


# ── parsing ──────────────────────────────────────────────────────────────────


def _section_bounds(lines: list[str], match) -> tuple[int, int] | None:
    """(heading index, end index) of the first ``##`` section whose heading ``match`` accepts."""
    for i, line in enumerate(lines):
        if match(line):
            end = next((j for j in range(i + 1, len(lines)) if _H2_RE.match(lines[j])), len(lines))
            return i, end
    return None


def unreleased_lines(changelog_text: str) -> list[str]:
    """Non-blank lines under ``## [Unreleased]``, headings and comment lines included (n8)."""
    lines = changelog_text.splitlines()
    bounds = _section_bounds(lines, _UNRELEASED_RE.match)
    if bounds is None:
        return []
    return [line for line in lines[bounds[0] + 1 : bounds[1]] if line.strip()]


def version_headings(changelog_text: str) -> list[str]:
    """Every ``## [X.Y.Z...]`` heading's version, in file order, duplicates kept."""
    return [m.group("version") for m in map(_VERSION_HEADING_RE.match, changelog_text.splitlines()) if m]


def version_section(changelog_text: str, version: str) -> list[str] | None:
    """The body lines (heading excluded) of ``## [version]``, or None when there is no such section."""
    lines = changelog_text.splitlines()
    bounds = _section_bounds(
        lines, lambda line: (m := _VERSION_HEADING_RE.match(line)) and m.group("version") == version
    )
    return None if bounds is None else lines[bounds[0] + 1 : bounds[1]]


def _version_of(pyproject_text: str) -> str | None:
    m = _VERSION_RE.search(pyproject_text)
    return m.group(1) if m else None


def normalize(body: str) -> str:
    return " ".join(body.lower().split())


def similarity(a: str, b: str) -> float:
    """F2: ``autojunk=False``, because the default treats frequent characters in long text as junk, and a
    2,257-character real bullet with one word changed then scored 0.595."""
    return difflib.SequenceMatcher(None, normalize(a), normalize(b), autojunk=False).ratio()


def fragment_problems(rel: str, mode: str, body: str) -> list[str]:
    """Rule 5 for one tree entry under ``changelog.d/`` (README excluded by the caller)."""
    name = rel[len(FRAG_DIR) + 1 :]
    if "/" in name:
        return [f"{rel}: fragments live directly in {FRAG_DIR}/, not in a subdirectory"]
    if mode != "100644" and mode != "100755":
        return [f"{rel}: not a regular file (git mode {mode}); a symlink is not a fragment"]
    m = _NAME_RE.match(name)
    if m is None:
        kind = _ANY_KIND_NAME_RE.match(name)
        if kind and kind.group("kind") not in KINDS:
            return [f"{rel}: unknown kind {kind.group('kind')!r}; use one of {', '.join(KINDS)}"]
        return [f"{rel}: the name must be <slug>.<kind>.md, with the slug matching [a-z0-9][a-z0-9-]*"]
    problems = []
    content = [line for line in body.splitlines() if line.strip()]
    if not content:
        return [f"{rel}: empty; a fragment holds at least one '- ' entry"]
    if not content[0].startswith("- "):
        problems.append(f"{rel}: must start with a '- ' bullet")
    if any(_ANY_HEADING_RE.match(line) for line in content):
        problems.append(f"{rel}: has a heading; the kind comes from the file name, so the body holds bullets only")
    if "<!--" in body:
        problems.append(f"{rel}: holds '<!--'; comments are reserved for assemble's markers")
    return problems


# ── git reads ────────────────────────────────────────────────────────────────


def _tree(repo_root: Path, rev: str) -> dict[str, str]:
    """{path: git mode} for every entry under changelog.d/ at ``rev``."""
    out: dict[str, str] = {}
    for entry in git(repo_root, "ls-tree", "-r", "-z", rev, "--", f"{FRAG_DIR}/").split("\0"):
        if entry:
            meta, _, path = entry.partition("\t")
            out[path] = meta.split()[0]
    return out


def _fragments_at(repo_root: Path, rev: str) -> set[str]:
    return {p for p in _tree(repo_root, rev) if p != README}


def _name_status(repo_root: Path, base: str, head: str, *, copies: bool = True) -> list[list[str]]:
    """Name-status lines under changelog.d/ with rename detection, and copy detection unless ``copies`` is off
    (S1, R2). The rename and copy flags are explicit, so the user's diff.renames setting cannot change the answer.
    ``-z`` output, so a path git would quote (a ``"``, a backslash, a control character) is read as itself."""
    detect = ["-M", "-C", "--find-copies-harder"] if copies else ["-M"]
    out = git(
        repo_root,
        "-c",
        "core.quotePath=false",
        "-c",
        "diff.renames=true",
        "diff",
        "--name-status",
        "-z",
        *detect,
        base,
        head,
        "--",
        f"{FRAG_DIR}/",
    )
    tokens = out.split("\0")
    entries: list[list[str]] = []
    i = 0
    while i < len(tokens) and tokens[i]:
        status = tokens[i]
        width = 2 if status[0] in "RC" else 1
        entries.append([status, *tokens[i + 1 : i + 1 + width]])
        i += 1 + width
    return entries


def _changed_paths(repo_root: Path, base: str, head: str) -> list[str]:
    """Every path the diff touches, read with ``-z``: a quoted ``src/`` path must not slip past rule 1 (code round)."""
    out = git(repo_root, "diff", "--name-only", "-z", "--no-renames", base, head)
    return [p for p in out.split("\0") if p]


def _required_fragments(repo_root: Path, base: str, head: str) -> set[str]:
    """Every fragment the release must carry: the base's, plus each one a first-parent unit added, minus each one a
    later unit renamed away (its new name replaces it). Walked unit by unit, because on a pull request's branch the
    units are single commits, and a fragment renamed during review would otherwise leave its first name owed
    (found by the F1 test). A deletion keeps the name owed: deleting is how assemble consumes."""
    owed = _fragments_at(repo_root, base)
    for sha in git(repo_root, "rev-list", "--first-parent", "--reverse", f"{base}..{head}").split():
        for parts in _name_status(repo_root, f"{sha}^1", sha, copies=False):
            if parts[0] == "A" and parts[1] != README:
                owed.add(parts[1])
            elif parts[0].startswith("R"):
                owed.discard(parts[1])
                if parts[2] != README:
                    owed.add(parts[2])
    return owed


def _added_names(repo_root: Path, rev_range: str) -> set[str]:
    """Every fragment name ADDED by any commit in a range, merged branches included, with ``--no-renames`` so a
    name that existed only briefly (renamed during review) is still known (F1)."""
    out = git(
        repo_root,
        "log",
        "-z",
        "--no-renames",
        "--diff-filter=A",
        "--name-only",
        "--format=",
        rev_range,
        "--",
        f"{FRAG_DIR}/",
    )
    names = (p.strip("\n") for p in out.split("\0"))
    return {p for p in names if p and p != README}


# ── the rules ────────────────────────────────────────────────────────────────


def _bump_problems(repo_root: Path, base: str, head: str, version: str, head_log: str) -> list[str]:
    """Rule 3. The required set comes from :func:`_required_fragments`; the permitted set adds every name added
    anywhere in the range, so a marker for a name renamed during review is not refused (F1)."""
    required = _required_fragments(repo_root, base, head)
    permitted = required | _added_names(repo_root, f"{base}..{head}")
    problems = []
    if version_headings(head_log).count(version) > 1:
        problems.append(f"{CHANGELOG} has more than one heading for {version}; assemble merges into the existing one")
    section = version_section(head_log, version)
    if section is None:
        return problems + [f"{CHANGELOG} has no '## [{version}]' section for the bumped version"]
    seen: dict[str, int] = {}
    for i, line in enumerate(section):
        m = _MARKER_RE.match(line.strip())
        if m is None:
            if "<!--" in line:
                problems.append(f"the {version} section holds a comment that is not a marker: {line.strip()!r}")
            continue
        rel = f"{FRAG_DIR}/{m.group('name')}"
        seen[rel] = seen.get(rel, 0) + 1
        following = next((s for s in section[i + 1 :] if s.strip()), "")
        if not following.startswith("- "):
            problems.append(f"marker for {rel} has no entry after it; its text was deleted")
        if rel not in permitted:
            problems.append(f"marker for {rel} names no fragment this release consumed")
    problems += [
        f"marker for {rel} appears {n} times; each fragment is assembled once" for rel, n in seen.items() if n > 1
    ]
    problems += [
        f"{rel} is missing from the {version} section (no marker); it would drop out of the release"
        for rel in sorted(required - set(seen))
    ]
    left = sorted(_fragments_at(repo_root, head))
    if left:
        problems.append(f"fragments left in {FRAG_DIR}/ after the bump (run assemble again): {', '.join(left)}")
    return problems


def violations(repo_root: Path, base: str, head: str = "HEAD") -> list[str]:
    """Every rule, for one unit: a pull request's whole diff, or one first-parent commit of a push."""
    head = git(repo_root, "rev-parse", "--verify", f"{head}^{{commit}}").strip()
    base_log, head_log = show(repo_root, base, CHANGELOG), show(repo_root, head, CHANGELOG)
    bumped = _version_of(show(repo_root, base, PYPROJECT) or "") != (
        version := _version_of(show(repo_root, head, PYPROJECT) or "")
    )
    problems: list[str] = []

    tree = _tree(repo_root, head)
    for rel, mode in sorted(tree.items()):
        if rel != README:
            body = show(repo_root, head, rel) if mode in ("100644", "100755") else ""
            problems += fragment_problems(rel, mode, body)

    status = _name_status(repo_root, base, head)
    # A copy (C) is an addition too: its path is new, and the duplicate check below decides whether it is this
    # PR's own entry. Only a rename (R) is never an addition (S1).
    added = [parts[-1] for parts in status if parts[0] == "A" or parts[0].startswith("C")]
    added = [rel for rel in added if rel != README]

    if bumped and version:
        problems += _bump_problems(repo_root, base, head, version, head_log)
    else:
        if any(p.startswith(SRC_PREFIX) for p in _changed_paths(repo_root, base, head)):
            if not any(_NAME_RE.match(rel[len(FRAG_DIR) + 1 :]) for rel in added):
                problems.append(
                    f"this diff changes src/ but adds no new fragment: add {FRAG_DIR}/<slug>.<kind>.md (kinds: "
                    f"{', '.join(KINDS)}; format in {README}). A renamed or copied fragment does not count."
                )
        for parts in status:
            if parts[0] == "D" and parts[1] != README:
                problems.append(f"{parts[1]} was deleted outside a release; only the bump's assemble removes fragments")
            elif parts[0].startswith("R") and parts[1] != README:
                problems.append(f"{parts[1]} was renamed to {parts[2]} outside a release; a fragment keeps its name")
        before, after = unreleased_lines(base_log), unreleased_lines(head_log)
        if len(after) > len(before):
            problems.append(
                f"'## [Unreleased]' grew ({len(before)} -> {len(after)} lines); entries go in {FRAG_DIR}/ now, "
                "so parallel PRs stop conflicting on CHANGELOG.md"
            )
        if len(version_headings(head_log)) > len(version_headings(base_log)):
            problems.append("a new '## [' version heading outside a release; the bump's assemble writes it")

    problems += _duplicate_problems(repo_root, head, tree, added)
    return problems


def _duplicate_problems(repo_root: Path, head: str, tree: dict[str, str], added: list[str]) -> list[str]:
    """An added fragment that repeats another one is not this PR's entry (S1, R2, F2). An exact copy is refused at
    any length; a near copy only at 80+ normalized characters, where it cannot be a coincidence."""
    problems = []
    for rel in added:
        if tree.get(rel) not in ("100644", "100755"):
            continue
        body = show(repo_root, head, rel)
        for other in sorted(tree):
            if other in (rel, README) or tree[other] not in ("100644", "100755"):
                continue
            other_body = show(repo_root, head, other)
            exact = normalize(body) == normalize(other_body)
            score = 1.0 if exact else similarity(body, other_body)
            if exact or (len(normalize(body)) >= SIMILARITY_FLOOR and score >= SIMILARITY):
                problems.append(f"{rel} is {score:.2f} similar to {other}; write this PR's own entry")
    return problems


def range_violations(repo_root: Path, base: str, *, api: Callable[[str], Any] | None = None) -> list[str]:
    """A push: judge each thing that LANDED on its own, so a bump and a later PR that share one push range (a
    cancelled run between them) are each judged as what they are (F1, N6).

    What landed comes from ``_lint_git.push_units``, grouped by merged PR: a rebase-merged PR lands as several
    first-parent commits ("change src/", then "add the fragment"), and judging those one at a time would turn
    ``main`` red on a PR that passed (code round SF1). A direct push, with no PR, is judged commit by commit."""
    groups: list[list[str]] = []
    last_pr: object = None
    for unit in push_units(repo_root, base, api=api):
        pr = unit.pr["number"] if unit.pr else None
        if groups and pr is not None and pr == last_pr:
            groups[-1].append(unit.sha)
        else:
            groups.append([unit.sha])
        last_pr = pr
    problems: list[str] = []
    for shas in groups:
        parent = git(repo_root, "rev-parse", f"{shas[0]}^1").strip()
        label = shas[-1][:12] if len(shas) == 1 else f"{shas[0][:12]}..{shas[-1][:12]}"
        problems += [f"{label}: {p}" for p in violations(repo_root, parent, shas[-1])]
    return problems


def leftover_fragment_problems(repo_root: Path, version: str) -> list[str]:
    """The tag-time belt (N4, R5), run by ``check_release_tag.py notes``: fragments at the commit being released
    are entries missing from its section. It applies only while ``v<version>`` is untagged, or on a tag push, so a
    rehearsal of ``main`` before the bump (pyproject still names a tagged release) is not failed by the cycle's
    pending fragments. An unreadable tag list counts as untagged, the strict path."""
    on_tag_push = os.environ.get("GITHUB_REF", "").startswith("refs/tags/")
    try:
        tagged = bool(git(repo_root, "tag", "--list", f"v{version}").strip())
    except GitUnavailable:
        tagged = False
    if tagged and not on_tag_push:
        return []
    readme = Path(README).name
    left = sorted(p.name for p in (repo_root / FRAG_DIR).glob("*") if p.name != readme)
    return (
        [
            f"{FRAG_DIR}/ still holds fragments at the release commit, so they are missing from {version}: {', '.join(left)}"
        ]
        if left
        else []
    )


# ── assemble ─────────────────────────────────────────────────────────────────


@dataclass
class _Block:
    heading: str | None  # the ### line verbatim; None for the preamble
    kind: str | None  # the kind a PLAIN heading maps to; None for a subtitle or unknown heading
    family: str | None  # the kind a subtitle heading belongs to ("### Changed — x" -> "changed")
    lines: list[str]


def _blocks(lines: list[str]) -> list[_Block]:
    blocks = [_Block(None, None, None, [])]
    for line in lines:
        m = _H3_RE.match(line)
        if m:
            text = m.group("text")
            word = text.split()[0].lower()
            family = word if word in KINDS else None
            plain = family is not None and text.lower() == word
            blocks.append(_Block(line, family if plain else None, family, []))
        else:
            blocks[-1].lines.append(line)
    return blocks


def _split_marked(lines: list[str]) -> tuple[list[str], list[list[str]]]:
    """(unmarked lines before the first marker, [marker chunk, ...]). A chunk runs to the next marker."""
    unmarked: list[str] = []
    chunks: list[list[str]] = []
    for line in lines:
        if _MARKER_RE.match(line.strip()):
            chunks.append([line])
        elif chunks:
            chunks[-1].append(line)
        else:
            unmarked.append(line)
    return unmarked, chunks


def _trim(lines: list[str]) -> list[str]:
    while lines and not lines[0].strip():
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    return lines


def assemble_text(changelog_text: str, *, version: str, heading: str, fragments: list[tuple[str, str, str]]) -> str:
    """Fold ``[Unreleased]`` and ``fragments`` ([(file name, kind, body)], already ordered) into ``## [version]``.

    Merges into an existing section (a re-run is the normal path, R3) and moves everything under
    ``[Unreleased]`` every time (N2). Within each kind block, unmarked text comes first, then the marked chunks
    already there, then the new fragments (F3), so a marker is never followed by someone else's unmarked bullet.
    """
    lines = changelog_text.splitlines()
    unrel = _section_bounds(lines, _UNRELEASED_RE.match)
    if unrel is None:
        raise ValueError(f"{CHANGELOG} has no '## [Unreleased]' heading")
    moved = lines[unrel[0] + 1 : unrel[1]]
    lines = lines[: unrel[0] + 1] + [""] + lines[unrel[1] :]

    existing = _section_bounds(
        lines, lambda line: (m := _VERSION_HEADING_RE.match(line)) and m.group("version") == version
    )
    # The heading always comes from the arguments, so a re-run with a corrected date or name updates it rather
    # than silently keeping the old one (the guide's Date rule can move the date; code round SF2).
    head_line = heading
    if existing is not None:
        body = lines[existing[0] + 1 : existing[1]]
        lines = lines[: existing[0]] + lines[existing[1] :]
    else:
        body = []

    old, new = _blocks(body), _blocks(moved)
    preamble = _trim(old[0].lines) + ([""] if old[0].lines and new[0].lines else []) + _trim(new[0].lines)
    out: list[str] = [head_line, ""]
    if _trim(preamble):
        out += _trim(preamble) + [""]
    for kind in KINDS:
        unmarked: list[str] = []
        chunks: list[list[str]] = []
        for block in old[1:] + new[1:]:
            if block.kind == kind:
                u, c = _split_marked(block.lines)
                unmarked += _trim(u) + ([""] if _trim(u) else [])
                chunks += c
        for name, frag_kind, frag_body in fragments:
            if frag_kind == kind:
                chunks.append([f"<!-- {FRAG_DIR}/{name} -->", *frag_body.rstrip("\n").splitlines()])
        content = _trim(unmarked)
        for chunk in chunks:
            content += ([""] if content else []) + _trim(chunk)
        if content:
            out += [f"### {HEADING_OF[kind]}", "", *content, ""]
        for block in old[1:] + new[1:]:
            if block.kind is None and block.family == kind and _trim(block.lines):
                out += [block.heading or "", "", *_trim(block.lines), ""]
    for block in old[1:] + new[1:]:
        if block.kind is None and block.family is None and _trim(block.lines):
            out += [block.heading or "", "", *_trim(block.lines), ""]

    # Collapse blank runs only in the section written here: released history is never touched (code round SF2).
    out = [line for i, line in enumerate(out) if line.strip() or i == 0 or out[i - 1].strip()]
    first_version = next((i for i, line in enumerate(lines) if _VERSION_HEADING_RE.match(line)), len(lines))
    lines = lines[:first_version] + out + lines[first_version:]
    return "\n".join(lines).rstrip("\n") + "\n"


def ordered_fragments(repo_root: Path) -> list[tuple[str, str, str]]:
    """[(name, kind, body)] in merge order: the position on origin/main's first-parent chain of the commit that added
    each one (n5; position, not commit time, which a rebase can reorder), newest add winning for a re-added name.
    Fragments not yet on main sort last, and ties go by name (N1). Bodies are read from the index, the set
    ``git ls-files`` named, so a staged fragment missing on disk cannot crash the cut."""
    when: dict[str, int] = {}
    try:
        out = git(
            repo_root,
            "log",
            "origin/main",
            "--first-parent",
            "--diff-merges=first-parent",
            "--no-renames",
            "--diff-filter=A",
            "--name-only",
            "--format=@",
            "--",
            f"{FRAG_DIR}/",
        )
    except GitUnavailable:
        out = ""
    position = 0  # 0 is the newest commit; a larger number is older
    for line in out.splitlines():
        if line == "@":
            position += 1
        elif line.strip():
            when.setdefault(Path(line).name, position)
    names = [p for p in git(repo_root, "ls-files", "-z", "--", f"{FRAG_DIR}/").split("\0") if p and p != README]
    frags = []
    for rel in names:
        name = rel[len(FRAG_DIR) + 1 :]
        m = _NAME_RE.match(name)
        if m is None:
            raise ValueError(f"{rel}: not a valid fragment name; fix it before assembling")
        frags.append((name, m.group("kind"), git(repo_root, "show", f":{rel}")))
    return sorted(frags, key=lambda f: (-when[f[0]] if f[0] in when else sys.maxsize, f[0]))


def assemble(repo_root: Path, *, version: str, date: str, name: str) -> list[str]:
    """Write the section and ``git rm`` the fragments. Returns the fragment names consumed.

    Refuses a version that is not ``X.Y.Z`` or differs from ``pyproject.toml``'s: the bump comes first, and a
    ``v1.2.0`` typo would write a heading no other gate reads (code round N5)."""
    pyproject = _version_of((repo_root / PYPROJECT).read_text(encoding="utf-8"))
    if not _RELEASE_VERSION_RE.match(version) or version != pyproject:
        raise ValueError(
            f"--version {version!r} must be X.Y.Z and equal {PYPROJECT}'s version ({pyproject!r}); bump first"
        )
    # Preflight before anything is written: `git rm` refuses a fragment with uncommitted edits, and it used to do so
    # AFTER CHANGELOG.md was rewritten, leaving a half-made cut whose re-run doubled every marker (delta round).
    # Only an added or modified fragment blocks it; a deletion staged by an earlier run is assemble's own work.
    dirty = [
        p
        for p in git(repo_root, "diff", "--name-only", "-z", "--diff-filter=AM", "HEAD", "--", f"{FRAG_DIR}/").split(
            "\0"
        )
        if p and p != README
    ]
    if dirty:
        raise ValueError(f"uncommitted changes to {', '.join(dirty)}; commit or discard them before assembling")
    frags = ordered_fragments(repo_root)
    path = repo_root / CHANGELOG
    heading = f'## [{version}] - {date} — "{name}"'
    path.write_text(
        assemble_text(path.read_text(encoding="utf-8"), version=version, heading=heading, fragments=frags),
        encoding="utf-8",
    )
    if frags:
        git(repo_root, "rm", "-q", "--", *(f"{FRAG_DIR}/{f[0]}" for f in frags))
    return [f[0] for f in frags]


# ── entry point ──────────────────────────────────────────────────────────────


def _check() -> int:
    try:
        base = base_ref(REPO_ROOT)
    except GitUnavailable as exc:
        if must_not_skip(str(exc)):
            return 2
        print(f"INFO: no base ref available; skipping the changelog-fragments check ({exc})")
        return 0
    try:
        fails = (
            range_violations(REPO_ROOT, base)
            if os.environ.get("GITHUB_EVENT_NAME") == "push"
            else violations(REPO_ROOT, base)
        )
    except GitUnavailable as exc:
        if must_not_skip(str(exc)):
            return 2
        print(f"INFO: changelog-fragments check skipped mid-run; git could not answer ({exc})")
        return 0
    except (OSError, UnicodeDecodeError) as exc:
        print(f"ERROR: changelog-fragments check could not run: {exc}", file=sys.stderr)
        return 2
    if fails:
        print("changelog-fragments check FAILED:", file=sys.stderr)
        for f in fails:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("changelog-fragments check: clean")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="the versioning lint (CI lint job)")
    a = sub.add_parser("assemble", help="the release cut, run in the bump PR")
    a.add_argument("--version", required=True)
    a.add_argument(
        "--date", required=True, help="the planned UTC release date; correct it after upload per the guide's Date rule"
    )
    a.add_argument("--name", required=True, help="the release name")
    args = ap.parse_args(argv)
    if args.cmd == "check":
        return _check()
    try:
        consumed = assemble(REPO_ROOT, version=args.version, date=args.date, name=args.name)
    except ValueError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    except GitUnavailable as exc:
        print(f"ERROR: assemble could not run git: {exc}", file=sys.stderr)
        return 2
    print(f"assembled {len(consumed)} fragment(s) into {args.version}: {', '.join(consumed) or '(none)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
