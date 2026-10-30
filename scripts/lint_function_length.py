#!/usr/bin/env python3
"""ONE function-length ratchet over ``src/maxim`` (roadmap 1.3.2 item 6; #940 item 1).

**The rule** (owner decisions 2026-10-04): no function over 200 lines may grow, and no new
function may exceed 200 lines. Every function in ``src/maxim/**/*.py`` longer than
``THRESHOLD`` is pinned in ``src/maxim/utils/function_length_baseline.json`` at its exact
span, and the pin is compared by STRICT EQUALITY: a function that shrinks fails until its
pin is lowered in the same commit, so the file never overstates the debt. A pin may only be
RAISED (or a new function over 200 pinned) through a committed, machine-readable exception
entry that is new in the same diff.

Identity is ``(file, qualname)``: ``file`` is repo-relative (``src/maxim/...``), ``qualname``
is Python's form (``Class.method``, ``outer.<locals>.inner``). Span is
``end_lineno - lineno + 1`` with decorators excluded (the measure since 2026-08-29). A
nested function's lines also count inside its parent's span, so it is charged TWICE: growing a
pinned ``outer.<locals>.inner`` raises both pins and needs an exception for each (the case until the orchestrator's
§11 slice, 2026-10-30: ``start_simulation_mode.<locals>._stall_detector``).

**Every run** (rules 1-3, plus the file's own shape):

1. every function over ``THRESHOLD`` has an entry ("unpinned" otherwise);
2. every entry names a function that exists and is over ``THRESHOLD`` ("orphan: renamed or
   moved" / "remove the entry");
3. every entry's pin equals the measured span ("grew" / "shrank: lower the pin");
   plus: a qualname defined more than once in a file fails as "ambiguous" when any definition
   is over the threshold, and an entry on ANY ambiguous qualname fails whatever its spans;
   ``threshold`` must equal the hard-coded ``THRESHOLD``; ``baseline_format_version`` must be 2; unknown keys fail; every exception is well formed
   (``ref`` is ``#NNN`` or a github.com PR/issue URL, ``reason`` non-empty).

**Diff-scoped** (against the merge-base copy of the baseline, read with ``git show``):

- a pin raised ``a -> b`` needs a NEW exception ``{file, qualname, from: a, to: b}``, where ``a``
  is the lower of the base pin and the def's measured span at the merge-base (so repairing a
  drifted pin needs nothing, and is not a pin DROP a ``split_from`` can claim); a kept pin
  whose key was NOT one def at the merge-base
  (ambiguous or missing) is accepted only by a bare ``{from: null, to: pin}`` exception;
- an entry with no base counterpart is one of:

  * a free MOVE: a base entry removed in the same diff whose function no longer exists at
    HEAD under its old key, whose normalized AST (``ast.dump`` without positions, def name
    blanked) is identical, and whose pin is not exceeded;
  * a recorded move: an exception with ``moved_from: {file, qualname}`` and ``from`` = the
    removed entry's base pin; the old key must be gone at HEAD, and a move to a smaller pin
    counts as a pin DROP. Each removed entry pays for at most ONE move, free or recorded;
  * a split piece: an exception with ``split_from: {file, qualname}`` and ``from: null``; the
    named base entry's pin must have dropped, or the entry been removed, in the same diff;
  * a new function: an exception with a bare ``from: null`` — allowed ONLY in a diff that
    lowers or removes no pin (other than by a move), so a decomposition cannot leave a piece's
    origin unrecorded (fail closed);
- ``exceptions`` is append-only: the base list, as parsed records, is an exact prefix of
  HEAD's list; every NEW exception must be used by a change in the same diff ("unused
  exception" otherwise — no pre-approving a later raise);
- ``threshold`` may not change from base.

**Rule T, "touch it, shrink it"** (owner decisions 2026-10-16; adversarial design pass folded). A diff that adds or
changes lines inside a function pinned at the merge-base must leave it smaller by AT LEAST that many lines:
``span_HEAD <= min(base pin, base span) - touched``. ``touched`` counts the HEAD def's lines, decorators included,
that match no line of its base def (``difflib.SequenceMatcher(autojunk=False)`` over the two def texts, never git's
diff output, so ``.gitattributes`` cannot hide an edit), each def's own indent removed, so a move between nesting
levels touches nothing. A pure deletion touches nothing. It applies to every base entry whose def still exists,
including one now at 200 lines or fewer and dropped from the baseline. It follows a free or recorded move to the new
key, comparing with the new name normalized, so a rename touches nothing; a pinned def gone at HEAD with no such move
is followed to the new def in a changed file that shares the most lines with it, when at least half of that def's
lines came from it (a rename that landed at 200 or fewer). A pin raised through a new exception (not a
``moved_from`` one) is its own record, whatever it touched. Otherwise the diff adds a ``touch_exemptions`` record
``{file, qualname, touched, from, to, date, ref, reason}`` whose numbers match exactly (one record, one change).
``touch_exemptions`` is optional (``[]``), append-only, and an unused new record fails on a pull request. A push
only reports it, because a push can bundle a PR that added the record with one that reverted its change.
Residuals (mechanization backlog M67): deleting comments, blank lines or a docstring counts as shrink (review judges
that); touched counts lines, not tokens, so wrapping a block in ``try:``/``with`` touches every line re-indented
(extract the block instead); decorators count as touched but sit outside the span, so a decorator edit needs a body
shrink; an edit inside a nested pinned function is charged to it and to every pinned function around it, as a raise
is; a removed def rewritten past half under a new name is judged as new code, and half is a hard edge; a def renamed into an ambiguous qualname is not followed; a local
run follows only committed renames; exemptions are self-approved.

**The merge-base baseline must be format 2** (on ``main`` since #1090): a base with no baseline
at ``BASELINE_REL`` (it moved or was deleted) or with any other format fails the diff rules; the
v1 migration path was deleted with #1089.

**When the diff rules run:** on ``pull_request`` they are the gate, and a missing merge-base
(or a base file git cannot read) is an error (``_lint_git.must_not_skip``; exit 2). On a
``push`` to main they run against the last push whose lint job passed (``_lint_git.push_base``,
#1089), with the same exit-2 rule, so a change that reached main without a PR is still judged.
Locally they run when a merge-base exists and are skipped with an INFO line otherwise. Any
other CI event (schedule, dispatch) runs rules 1-3 only.
The lint prints every pinned span and the totals on every run, so the count lives in CI output.

The baseline is CI lint data, not runtime persistence: ``_format_version``/CC3 do not apply
(the ``architecture_baseline.json`` precedent), and the evidence gate's ``SUBJECT_EXCLUDED``
pins its path. ``history`` holds the pre-v2 prose record of raises and tightenings; the lint
ignores it. There is no writer: edit the file by hand with the printed span.

**Residuals (what this does not see):** code moved into module-level statements or a class
body; code moved out of ``src/maxim``; decorators (excluded from the span); ``split_from``
checks that the source's pin DROPPED, not by how much (a 450-line piece split from a 600-line
function that drops by 1 passes — the size of the transfer is for review to judge); and
exceptions are SELF-approved — an exception makes a raise visible and reviewable, it does not
prove anyone approved it.

**Baseline edits per decomposition slice** (worked example): ``big`` (600) loses ``part``
(450) and becomes 140 → remove ``big``'s entry (or, were it still over 200, lower its pin to
the printed span), add an entry ``part: 450`` and an exception
``{file, qualname: part, from: null, to: 450, split_from: {file, qualname: big}, date, ref,
reason}``. A piece of 200 lines or fewer needs nothing.

Regression guard: tests/unit/test_lint_function_length.py drives ``main()`` on fixture git
repos for every rule above, and checks this checkout is clean (rules 1-3).
"""

from __future__ import annotations

import ast
import copy
import difflib
import json
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lint_git import GitUnavailable, base_ref, changed_files, must_not_skip, show  # noqa: E402
from _lint_allowance import append_only_problem, ref_ok  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SCOPE = "src/maxim"
BASELINE_REL = "src/maxim/utils/function_length_baseline.json"
THRESHOLD = 200  # owner decision 2026-10-04; the baseline's `threshold` must equal it
FORMAT_VERSION = 2

_TOP_KEYS = {"_comment", "baseline_format_version", "threshold", "entries", "exceptions", "history", "touch_exemptions"}
_TOUCH_KEYS = {"file", "qualname", "touched", "from", "to", "date", "ref", "reason"}
_ENTRY_KEYS = {"file", "qualname", "lines"}
_EXC_REQUIRED = {"file", "qualname", "from", "to", "date", "ref", "reason"}
_EXC_OPTIONAL = {"split_from", "moved_from"}
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

Key = tuple[str, str]  # (file, qualname)


class BaselineError(ValueError):
    """The baseline file is malformed — a failure, never a skip."""


# ── measurement ──────────────────────────────────────────────────────────────


def defs_in(text: str, rel: str = "<text>") -> dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Every def in ``text`` keyed by Python-style qualname, in source order. SyntaxError propagates."""
    tree = ast.parse(text, filename=rel)
    out: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = prefix + child.name
                out.setdefault(q, []).append(child)
                walk(child, q + ".<locals>.")
            elif isinstance(child, ast.ClassDef):
                walk(child, prefix + child.name + ".")
            else:
                walk(child, prefix)

    walk(tree, "")
    return out


def span(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Inclusive line count, def line to last body line; decorators excluded."""
    assert node.end_lineno is not None
    return node.end_lineno - node.lineno + 1


def normalized(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """The def's AST without positions and with its name blanked — equal iff the body is identical."""
    n = copy.copy(node)
    n.name = ""
    return ast.dump(n, include_attributes=False)


@dataclass
class Measure:
    spans: dict[Key, int] = field(default_factory=dict)  # unambiguous defs only
    ambiguous: dict[Key, list[int]] = field(default_factory=dict)  # every multiply-defined qualname
    nodes: dict[Key, ast.FunctionDef | ast.AsyncFunctionDef] = field(default_factory=dict)
    n_defs: int = 0


def measure_text(rel: str, text: str, m: Measure) -> None:
    for q, nodes in defs_in(text, rel).items():
        m.n_defs += len(nodes)
        if len(nodes) > 1:
            m.ambiguous[(rel, q)] = [span(n) for n in nodes]
        else:
            m.spans[(rel, q)] = span(nodes[0])
            m.nodes[(rel, q)] = nodes[0]


def measure_tree(root: Path) -> tuple[Measure, list[str]]:
    m, errors = Measure(), []
    for path in sorted((root / SCOPE).rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        try:
            measure_text(rel, path.read_text(encoding="utf-8"), m)
        except (SyntaxError, UnicodeDecodeError, ValueError) as e:
            errors.append(f"{rel}: cannot parse ({e}) — the ratchet fails rather than skip a file")
    return m, errors


def measure_at_base(root: Path, base: str, rels: set[str]) -> Measure:
    """Spans of the given files at ``base`` (absent files contribute nothing)."""
    m = Measure()
    for rel in sorted(rels):
        text = show(root, base, rel)
        if text:
            try:
                measure_text(rel, text, m)
            except (SyntaxError, ValueError):
                continue  # unparsable at base: nothing there can justify a HEAD entry
    return m


# ── baseline ─────────────────────────────────────────────────────────────────


@dataclass
class Baseline:
    version: int
    threshold: int | None
    entries: dict[Key, int]
    exceptions: list[dict]
    touch_exemptions: list[dict] = field(default_factory=list)


def _key_obj(v: object, what: str) -> Key:
    if not (isinstance(v, dict) and set(v) == {"file", "qualname"} and all(isinstance(x, str) for x in v.values())):
        raise BaselineError(f"{what} must be {{'file': str, 'qualname': str}}, got {v!r}")
    return (v["file"], v["qualname"])


def parse_baseline(text: str) -> Baseline:
    """Parse the baseline; anything but format 2 is refused, at HEAD and at the merge-base alike."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise BaselineError(f"not JSON: {e}") from e
    if not isinstance(data, dict):
        raise BaselineError("root must be an object")
    version = data.get("baseline_format_version")
    if version != FORMAT_VERSION:
        raise BaselineError(f"baseline_format_version must be {FORMAT_VERSION}, got {version!r}")
    unknown = set(data) - _TOP_KEYS
    if unknown or not {"threshold", "entries", "exceptions"} <= set(data):
        raise BaselineError(f"top-level keys must be {sorted(_TOP_KEYS)} (unknown: {sorted(unknown)})")
    threshold = data["threshold"]
    if not isinstance(threshold, int) or isinstance(threshold, bool):
        raise BaselineError(f"threshold must be an int, got {threshold!r}")
    if not isinstance(data.get("history", []), list) or not all(isinstance(h, str) for h in data.get("history", [])):
        raise BaselineError("history must be a list of strings")
    entries: dict[Key, int] = {}
    if not isinstance(data["entries"], list):
        raise BaselineError("entries must be a list")
    for e in data["entries"]:
        if not isinstance(e, dict) or set(e) != _ENTRY_KEYS:
            raise BaselineError(f"entry must have exactly {sorted(_ENTRY_KEYS)}: {e!r}")
        if not (isinstance(e["file"], str) and isinstance(e["qualname"], str)):
            raise BaselineError(f"entry file/qualname must be strings: {e!r}")
        if not isinstance(e["lines"], int) or isinstance(e["lines"], bool) or e["lines"] <= 0:
            raise BaselineError(f"entry lines must be a positive int: {e!r}")
        k = (e["file"], e["qualname"])
        if k in entries:
            raise BaselineError(f"duplicate entry {k[0]}::{k[1]}")
        entries[k] = e["lines"]
    if not isinstance(data["exceptions"], list):
        raise BaselineError("exceptions must be a list")
    for x in data["exceptions"]:
        _check_exception(x)
    touch = data.get("touch_exemptions", [])
    if not isinstance(touch, list):
        raise BaselineError("touch_exemptions must be a list")
    for x in touch:
        _check_touch_exemption(x)
    return Baseline(version, threshold, entries, list(data["exceptions"]), list(touch))


def _check_touch_exemption(x: object) -> None:
    """Rule T's escape hatch: bound to ONE change by its exact ``touched``/``from``/``to`` (design pass, 2026-10-16)."""
    if not isinstance(x, dict) or set(x) != _TOUCH_KEYS:
        raise BaselineError(f"touch exemption must have exactly {sorted(_TOUCH_KEYS)}: {x!r}")
    if not (isinstance(x["file"], str) and isinstance(x["qualname"], str)):
        raise BaselineError(f"touch exemption file/qualname must be strings: {x!r}")
    for n in ("touched", "from", "to"):
        if not isinstance(x[n], int) or isinstance(x[n], bool) or x[n] < 0:
            raise BaselineError(f"touch exemption `{n}` must be a non-negative int: {x!r}")
    if not (isinstance(x["date"], str) and _DATE_RE.fullmatch(x["date"])):
        raise BaselineError(f"touch exemption `date` must be YYYY-MM-DD: {x!r}")
    if not ref_ok(x["ref"]):
        raise BaselineError(f"touch exemption `ref` must be '#NNN' or a github.com PR/issue URL: {x!r}")
    if not (isinstance(x["reason"], str) and x["reason"].strip()):
        raise BaselineError(f"touch exemption `reason` must be non-empty: {x!r}")


def _check_exception(x: object) -> None:
    if not isinstance(x, dict):
        raise BaselineError(f"exception must be an object: {x!r}")
    keys = set(x)
    if not _EXC_REQUIRED <= keys or keys - _EXC_REQUIRED - _EXC_OPTIONAL:
        raise BaselineError(f"exception keys must be {sorted(_EXC_REQUIRED)} + optional {sorted(_EXC_OPTIONAL)}: {x!r}")
    if len(keys & _EXC_OPTIONAL) > 1:
        raise BaselineError(f"exception may carry split_from OR moved_from, not both: {x!r}")
    if not (isinstance(x["file"], str) and isinstance(x["qualname"], str)):
        raise BaselineError(f"exception file/qualname must be strings: {x!r}")
    if not (x["from"] is None or (isinstance(x["from"], int) and not isinstance(x["from"], bool))):
        raise BaselineError(f"exception `from` must be an int or null: {x!r}")
    if not isinstance(x["to"], int) or isinstance(x["to"], bool) or x["to"] <= 0:
        raise BaselineError(f"exception `to` must be a positive int: {x!r}")
    if not (isinstance(x["date"], str) and _DATE_RE.fullmatch(x["date"])):
        raise BaselineError(f"exception `date` must be YYYY-MM-DD: {x!r}")
    if not ref_ok(x["ref"]):
        raise BaselineError(f"exception `ref` must be '#NNN' or a github.com PR/issue URL: {x!r}")
    if not (isinstance(x["reason"], str) and x["reason"].strip()):
        raise BaselineError(f"exception `reason` must be non-empty: {x!r}")
    for opt in _EXC_OPTIONAL & keys:
        _key_obj(x[opt], f"exception `{opt}`")


# ── rules ────────────────────────────────────────────────────────────────────


def _fmt(k: Key) -> str:
    return f"{k[0]}::{k[1]}"


def head_rules(m: Measure, b: Baseline) -> list[str]:
    """Rules 1-3 + ambiguity + threshold equality: everything that needs no git."""
    out: list[str] = []
    if b.threshold != THRESHOLD:
        out.append(
            f"baseline threshold is {b.threshold}, the lint's THRESHOLD is {THRESHOLD} (owner decision; may not change)"
        )
    for k, spans in sorted(m.ambiguous.items()):
        if max(spans) > THRESHOLD:
            out.append(
                f"{_fmt(k)}: ambiguous — defined {len(spans)}x (spans {spans}) with one over {THRESHOLD}; rename one"
            )
    for k, s in sorted(m.spans.items()):
        if s > THRESHOLD and k not in b.entries:
            out.append(
                f"{_fmt(k)}: {s} lines, over {THRESHOLD} and unpinned (a new function over {THRESHOLD}, or a deleted entry)"
            )
    for k, pin in sorted(b.entries.items()):
        if k in m.ambiguous:
            # Fail whatever the spans: a pin on a qualname that names several defs measures nothing, and a
            # pin (+ exception) on short conditional defs would pre-approve a later single long def.
            out.append(f"{_fmt(k)}: cannot pin an ambiguous qualname (defined {len(m.ambiguous[k])}x) — rename one")
            continue
        if k not in m.spans:
            out.append(
                f"{_fmt(k)}: orphan entry — no such function (renamed or moved: update the baseline deliberately)"
            )
            continue
        s = m.spans[k]
        if pin <= THRESHOLD:
            out.append(f"{_fmt(k)}: pinned at {pin}, not over {THRESHOLD} — remove the entry")
        elif s > pin:
            out.append(
                f"{_fmt(k)}: grew to {s} lines (pin {pin}). Add code by EXTRACTING; a reviewed raise needs a new "
                "exception entry in the same diff"
            )
        elif s < pin:
            out.append(
                f"{_fmt(k)}: shrank to {s} lines (pin {pin}) — lower the pin to {s}"
                + (" (or remove the entry: it is no longer over the threshold)" if s <= THRESHOLD else "")
            )
    return out


def _same(x: dict, k: Key, frm: int | None, to: int, kind: str | None, src: Key | None) -> bool:
    if (x["file"], x["qualname"], x["from"], x["to"]) != (k[0], k[1], frm, to):
        return False
    present = [o for o in _EXC_OPTIONAL if o in x]
    if kind is None:
        return not present
    return present == [kind] and _key_obj(x[kind], kind) == src


def diff_rules(
    root: Path, base: str, m: Measure, head: Baseline, base_text: str, *, report_unused_touch: bool = True
) -> list[str]:
    out: list[str] = []
    if not base_text:
        # v2 has been on main since #1090, so a merge-base without the baseline means it moved or was deleted.
        return [f"{BASELINE_REL} is absent at the merge-base (moved or deleted?) — cannot verify pin changes"]
    try:
        bb = parse_baseline(base_text)
    except BaselineError as e:
        return [f"merge-base baseline unreadable ({e}) — cannot verify pin changes"]

    if head.threshold != bb.threshold:
        out.append(f"threshold changed {bb.threshold} -> {head.threshold} (owner-set; may not change)")
    n_old = len(bb.exceptions)
    if problem := append_only_problem(base=bb.exceptions, head=head.exceptions, what="exceptions are"):
        out.append(problem)
        return out
    new_exc = head.exceptions[n_old:]
    used = [False] * len(new_exc)

    def claim(k: Key, frm: int | None, to: int, kind: str | None = None, src: Key | None = None) -> bool:
        for i, x in enumerate(new_exc):
            if not used[i] and _same(x, k, frm, to, kind, src):
                used[i] = True
                return True
        return False

    removed = {k for k in bb.entries if k not in head.entries}
    consumed: set[Key] = set()  # removed base entries already claimed by ONE move (free or moved_from)
    raised_or_kept = {k: p for k, p in head.entries.items() if k in bb.entries}
    new_entries = {k: p for k, p in head.entries.items() if k not in bb.entries}
    base_nodes = measure_at_base(root, base, {r[0] for r in removed} | {k[0] for k in raised_or_kept})

    raised_recorded: set[Key] = set()  # rule T: a recorded raise is its own record of the change
    for k, pin in sorted(raised_or_kept.items()):
        old = bb.entries[k]
        bs = base_nodes.spans.get(k)
        if bs is None:
            # No single def under this key at the merge-base (ambiguous or missing — e.g. a pin on an ambiguous
            # qualname that slipped in before the every-run check existed): its "kept" status proves nothing, so
            # only a bare {from: null, to: pin} claim accepts it.
            if not claim(k, None, pin):
                out.append(
                    f"{_fmt(k)}: its merge-base pin {old} did not match a single def at the merge-base, so it is "
                    'treated as new: record an exception with "from": null'
                )
            continue
        # A drifted base (pin != measured span) is judged against what was really there: the lower of the two.
        old = min(bb.entries[k], bs)
        if pin > old and claim(k, old, pin):
            raised_recorded.add(k)
        elif pin > old:
            out.append(
                f"{_fmt(k)}: pin raised {old} -> {pin} without a new exception "
                f'{{"file", "qualname", "from": {old}, "to": {pin}, "date", "ref", "reason"}}'
            )

    # Pass 1: moves. A FREE move needs the removed key to be GONE at HEAD (not merely unpinned: a copy whose
    # original shrank is new debt) and an identical normalized AST; each removed entry pays for ONE move.
    unresolved: dict[Key, int] = {}
    shrunk_moves: set[Key] = set()
    move_of: dict[Key, Key] = {}  # removed base key -> the HEAD key it moved to (rule T follows a move)
    for k, pin in sorted(new_entries.items()):
        node = m.nodes.get(k)
        free = next(
            (
                r
                for r in sorted(removed - consumed)
                if r not in m.spans
                and r not in m.ambiguous
                and r in base_nodes.nodes
                and node is not None
                and normalized(base_nodes.nodes[r]) == normalized(node)
                and pin <= bb.entries[r]
            ),
            None,
        )
        if free is None:
            # A RECORDED move also needs the old key gone at HEAD (else it is a split wearing a move's label), and
            # one that shrinks counts as a DROP below, so it cannot unlock a bare `from: null` for a sibling piece.
            free = next(
                (
                    r
                    for r in sorted(removed - consumed)
                    if r not in m.spans and r not in m.ambiguous and claim(k, bb.entries[r], pin, "moved_from", r)
                ),
                None,
            )
            if free is not None and pin < bb.entries[free]:
                shrunk_moves.add(free)
        if free is not None:
            consumed.add(free)
            move_of[free] = k
        else:
            unresolved[k] = pin

    # Pass 2: splits and new functions. When this diff lowers or removes any pin (other than by a move), a new
    # entry over the threshold must say where its debt came from: a bare `from: null` is refused (fail closed).
    # A pin lowered only to repair drift (base pin above the function's measured base span) lost no lines, so it
    # is judged against min(base pin, base span), like a raise is (#1089 item 3).
    def _base_floor(r: Key) -> int:
        bs = base_nodes.spans.get(r)
        return bb.entries[r] if bs is None else min(bb.entries[r], bs)

    drops = sorted(
        r
        for r in bb.entries
        if (r in head.entries and head.entries[r] < _base_floor(r)) or r in (removed - consumed) | shrunk_moves
    )
    for k, pin in sorted(unresolved.items()):
        split = next(
            (
                i
                for i, x in enumerate(new_exc)
                if not used[i]
                and "split_from" in x
                and _same(x, k, None, pin, "split_from", _key_obj(x["split_from"], "split_from"))
            ),
            None,
        )
        if split is not None:
            used[split] = True
            src = _key_obj(new_exc[split]["split_from"], "split_from")
            if src not in drops or src in consumed:  # a moved entry cannot also be a split source
                out.append(
                    f"{_fmt(k)}: split_from {_fmt(src)}, but that base entry's pin did not drop and it was not "
                    "removed in this diff (or it was not pinned at base)"
                )
            continue
        if claim(k, None, pin):
            if drops:
                out.append(
                    f"{_fmt(k)}: new entry at {pin} with a bare from-null exception in a diff that lowers or removes "
                    f"pins ({', '.join(_fmt(r) for r in drops)}) — record split_from or moved_from"
                )
            continue
        out.append(
            f"{_fmt(k)}: new entry at {pin} lines (over {THRESHOLD}) needs a new exception with "
            '"from": null (+ "split_from"/"moved_from" when it came from a pinned function)'
        )

    out += touch_rule(root, base, m, head, bb, base_nodes, move_of, raised_recorded, report_unused_touch)

    for i, x in enumerate(new_exc):
        if not used[i]:
            out.append(
                f"unused exception for {x['file']}::{x['qualname']} ({x['from']} -> {x['to']}): no matching change in this diff"
            )
    return out


def _def_lines(text: str, node: ast.FunctionDef | ast.AsyncFunctionDef, as_name: str | None = None) -> list[str]:
    """A def's source lines, decorators INCLUDED (editing a decorator is touching the function), with the def's own
    indent removed, so a move between nesting levels (method to module, ``<locals>`` to module) is not an edit, and
    with the def renamed to ``as_name`` when given, so a move's new name is not one either (rule T)."""
    start = min([d.lineno for d in node.decorator_list] + [node.lineno])
    c = node.col_offset
    lines = [ln[c:] if not ln[:c].strip() else ln for ln in text.splitlines()[start - 1 : node.end_lineno]]
    if as_name is not None and as_name != node.name:
        i = node.lineno - start
        lines[i] = lines[i].replace(f"def {node.name}(", f"def {as_name}(", 1)
    return lines


def touched_lines(base_lines: list[str], head_lines: list[str]) -> int:
    """Lines of the HEAD def that match no line of its base def, in order. ``autojunk=False``: difflib's default
    treats lines frequent in a 200+-line sequence (blank lines, ``)``) as junk, exactly the sizes judged here."""
    sm = difflib.SequenceMatcher(None, base_lines, head_lines, autojunk=False)
    return len(head_lines) - sum(b.size for b in sm.get_matching_blocks())


def touch_rule(
    root: Path,
    base: str,
    m: Measure,
    head: Baseline,
    bb: Baseline,
    base_nodes: Measure,
    move_of: dict[Key, Key],
    raised_recorded: set[Key],
    report_unused: bool,
) -> list[str]:
    """Rule T, "touch it, shrink it" (owner decisions 2026-10-16): a diff that adds or changes lines inside a function
    pinned at the merge-base leaves it smaller by at least that many lines, or carries a matching touch exemption."""
    out: list[str] = []
    if problem := append_only_problem(
        base=bb.touch_exemptions, head=head.touch_exemptions, what="touch_exemptions are"
    ):
        return [problem]
    new_touch = head.touch_exemptions[len(bb.touch_exemptions) :]
    used = [False] * len(new_touch)
    base_texts: dict[str, str] = {}
    head_texts: dict[str, str] = {}

    def text_at(rel: str, at_base: bool) -> str:
        cache = base_texts if at_base else head_texts
        if rel not in cache:
            cache[rel] = show(root, base, rel) if at_base else (root / rel).read_text(encoding="utf-8")
        return cache[rel]

    new_defs: list[Key] | None = None  # HEAD defs under a qualname their file did not have at base (computed once)
    for r in sorted(bb.entries):
        base_node = base_nodes.nodes.get(r)
        if base_node is None:
            continue  # not one def at base: the other diff rules judge it
        base_lines = _def_lines(text_at(r[0], True), base_node)
        target = r if r in m.nodes else move_of.get(r)
        if target is None:
            # Gone at HEAD and not a move the other rules recognised (a renamed def that landed at 200 lines or
            # fewer has no entry to move to): its successor is the new def that carries most of its lines.
            if new_defs is None:
                new_defs = _new_defs(root, base, m, set(move_of.values()) | set(bb.entries))
            target = _successor(base_lines, base_node.name, new_defs, m, lambda rel: text_at(rel, False))
            if target is None:
                continue  # deleted, or rewritten past half: nothing at HEAD is this function
        if target not in m.nodes:
            continue  # several defs share the name at HEAD: residual (rules 1-3 judge only an ambiguous one over 200)
        node = m.nodes[target]
        touched = touched_lines(base_lines, _def_lines(text_at(target[0], False), node, base_node.name))
        floor = min(bb.entries[r], span(base_node))
        now = m.spans[target]
        if touched == 0 or now <= floor - touched or target in raised_recorded:
            continue
        hit = next(
            (
                i
                for i, x in enumerate(new_touch)
                if not used[i]
                and (x["file"], x["qualname"], x["touched"], x["from"], x["to"]) == (*target, touched, floor, now)
            ),
            None,
        )
        if hit is not None:
            used[hit] = True
            continue
        out.append(
            f"{_fmt(target)}: this diff edits {touched} line(s) inside a function pinned at {floor}, which must end at "
            f"{floor - touched} or fewer (it is {now}). Move code out of it, or record a touch exemption "
            f'{{"file", "qualname", "touched": {touched}, "from": {floor}, "to": {now}, "date", "ref", "reason"}}'
        )
    if report_unused:
        out += [
            f"unused touch exemption for {x['file']}::{x['qualname']} (touched {x['touched']}, {x['from']} -> "
            f"{x['to']}): no matching change in this diff"
            for i, x in enumerate(new_touch)
            if not used[i]
        ]
    return out


def _new_defs(root: Path, base: str, m: Measure, taken: set[Key]) -> list[Key]:
    """Unambiguous HEAD defs, in files this diff changed (base..HEAD, so a local run judges committed renames only),
    whose qualname their file (at its base path, following a file rename) did not have at the merge-base or whose
    text changed since, so a rename into a name that already existed is followed too; ``taken`` (keys a recognised
    move claimed, and pinned keys, which are judged as themselves) excluded."""
    out: list[Key] = []
    for rel, old in changed_files(root, base, SCOPE):
        text = show(root, base, old) if old else ""
        try:
            had = defs_in(text, old) if old else {}
        except (SyntaxError, ValueError):
            had = {}
        head_text = (root / rel).read_text(encoding="utf-8")
        for k in m.nodes:
            if k[0] != rel or k in taken or k not in m.spans:
                continue
            was = had.get(k[1], [])
            if len(was) != 1 or _def_lines(text, was[0]) != _def_lines(head_text, m.nodes[k]):
                out.append(k)
    return sorted(out)


def _successor(
    base_lines: list[str], base_name: str, candidates: list[Key], m: Measure, head_text: Callable[[str], str]
) -> Key | None:
    """The candidate sharing the most lines with a removed pinned def, if at least half of its own lines came from
    it (a rename with edits); ``None`` when no candidate is mostly that def (a deletion, or new code). Lines are
    compared whitespace-stripped HERE ONLY, so re-indenting the body cannot make a renamed def look like new code;
    the successor's ``touched`` is still counted on the def texts. Half is a hard edge (residual)."""
    stripped = [ln.strip() for ln in base_lines]
    best: tuple[int, Key] | None = None
    for k in candidates:
        lines = [ln.strip() for ln in _def_lines(head_text(k[0]), m.nodes[k], base_name)]
        shared = len(lines) - touched_lines(stripped, lines)
        if 2 * shared >= len(lines) and (best is None or shared > best[0]):
            best = (shared, k)
    return None if best is None else best[1]


# ── main ─────────────────────────────────────────────────────────────────────


def check_head(root: Path) -> tuple[list[str], Measure | None, Baseline | None]:
    """Rules 1-3 on ``root``'s working tree (no git)."""
    m, failures = measure_tree(root)
    try:
        b = parse_baseline((root / BASELINE_REL).read_text(encoding="utf-8"))
    except (OSError, BaselineError) as e:
        return failures + [f"{BASELINE_REL}: {e}"], m, None
    return failures + head_rules(m, b), m, b


def main() -> int:
    started = time.monotonic()
    root = REPO_ROOT
    event = os.environ.get("GITHUB_EVENT_NAME")
    failures, m, b = check_head(root)

    if b is not None and m is not None:
        print(f"function-length ratchet (threshold {THRESHOLD}; span = def line to last line, decorators excluded):")
        for k, pin in sorted(b.entries.items()):
            s = m.spans.get(k)
            print(f"  {_fmt(k)} = {s if s is not None else 'NOT FOUND'} (pin {pin})")
        print(
            f"function-length ratchet: {len(b.entries)} pinned over {THRESHOLD}, "
            f"{sum(b.entries.values())} pinned lines, {m.n_defs} defs measured, {len(b.exceptions)} exceptions"
        )

    if b is not None and m is not None and event in (None, "", "pull_request", "push"):
        try:
            base = base_ref(root)
            base_text = show(root, base, BASELINE_REL)
        except GitUnavailable as e:
            if must_not_skip(str(e)):
                return 2
            print(f"INFO: no merge-base available; diff-scoped rules skipped ({e})")
        else:
            try:
                failures += diff_rules(root, base, m, b, base_text, report_unused_touch=event != "push")
            except GitUnavailable as e:  # a base file read failed mid-check: fail closed, never a traceback
                print(f"ERROR: git could not read the merge-base during the diff-scoped rules ({e})", file=sys.stderr)
                return 2
    elif event:
        print(f"function-length ratchet: {event} event — rules 1-3 only")

    print(f"function-length ratchet: {time.monotonic() - started:.1f}s")
    if failures:
        print("function-length ratchet FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("function-length ratchet: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
