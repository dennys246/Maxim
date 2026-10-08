#!/usr/bin/env python3
"""The ledger evidence gate (M1b PR 5b-1; spec: docs/plans/m1b_ledger_evidence_gate.md, "PR 5b build spec").

A ledger row at a positive status or PARTIAL, when it changes, must rest on evidence that is ESTABLISHED: stamped,
not a typed abort, not mock, the code on main, one known code tree (judged by ``scripts/_evidence_records.py``). A raise, a move between positive tokens or a date
change additionally needs NEW support: a newly cited, stamped VERDICT that the merge-base pass table
(``docs/experiments/evidence_pass_table.json``) lets support this row at its new token and its scope (the qualifier
head, #1141). Records committed before
M1a are LEGACY (``docs/experiments/evidence_legacy.json``): judged as such, never new support. Owner-named
overrides live in ``docs/experiments/evidence_exceptions.json``: only clauses already on main act, append-only.

Diff-scoped against the merge-base with origin/main. It reads committed bytes (git objects at HEAD); uncommitted
changes (untracked files included) under the data root, in the ledger, the pass table, the legacy snapshot or the
exceptions file fail. It catches forgetting, not evasion: an author can still cite a clean
but irrelevant record (review is the check).

    python scripts/lint_evidence_gate.py              # the gate (CI lint job)
    python scripts/lint_evidence_gate.py --json       # machine-readable
    python scripts/lint_evidence_gate.py --write-legacy   # (re)generate the legacy snapshot from the rules

Exits: 0 clean; 1 violations; 2 the base could not be read on a pull request.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
REPO_ROOT = SCRIPTS_DIR.parent

import _ledger as L  # noqa: E402
import _lint_git  # noqa: E402
import lint_prereg_precedes_data as P  # noqa: E402
from _evidence_records import (  # noqa: E402  (re-exported: the gate's record vocabulary)
    DATA_ROOT,
    ESTABLISHED,
    EXCEPTED,
    LEGACY,
    NOT_ESTABLISHED,
    O19_JUDGE,
    O19_KINDS,
    SUPPORT_KINDS,
    Ctx,
    COMPLETE_RULES,
    MALFORMED,
    GateError,
    Judgement,
    Repo,
    decompressed,
    judge_entry,
    o19_history_problems,
    o19_judge_edit_problems,
    o19_closed_data_problems,
    o19_succession_problems,
    o19_table_problems,
    sha256,
    str_field,
    unjudged,
    watched_paths,
)

PASS_TABLE = "docs/experiments/evidence_pass_table.json"
LEGACY_SNAPSHOT = "docs/experiments/evidence_legacy.json"
EXCEPTIONS = "docs/experiments/evidence_exceptions.json"

# M1a (#999) merged at this commit time: a record first committed before it, carrying no `record_kind`, is legacy.
M1A_CUTOFF = 1790721676  # 2026-09-29T22:41:16Z, `git show -s --format=%ct fc9f19d0`
LEAVING_UNCHECKED = frozenset({"STALE", "BROKEN"})


# ── support, exceptions, the legacy snapshot ─────────────────────────────────────────────────────────────


def require_met(record: dict, require: dict) -> bool:
    for dotted, want in require.items():
        node = record
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return False
            node = node[part]
        if type(node) is not type(want) or node != want:
            return False
    return True


def row_scope(row: L.Row) -> str | None:
    """The scope a row claims at its status: its qualifier's head (``narrow``, ``rung A``), or None when unqualified.
    An empty qualifier ``()`` gives ``""``, a head no pass-table entry may list (#1141 design pass N2)."""
    return None if row.qualifier is None else L.qualifier_head(row.qualifier)


def support_problem(
    j: Judgement, row_id: str, token: str, table: dict, after: float, *, ctx: Ctx, scope: str | None
) -> str | None:
    """Why this newly cited record does NOT supply new support for moving ``row_id`` to ``token`` at ``scope`` (the
    row's HEAD qualifier head, None = unqualified; required, never defaulted: a default None would match every
    unqualified entry) (None = it does). An O19 verdict additionally obeys the campaign-succession rules (#1059), read
    through the gate's ``ctx``."""
    if j.status != ESTABLISHED:
        return f"{j.path} is {j.status}"
    if j.kind not in SUPPORT_KINDS or j.record is None:
        return f"{j.path} is a {j.kind}, not a verdict (only a stamped verdict supplies new support)"
    kind = str_field(j.record, "kind")
    entry = table.get(kind)
    if not isinstance(entry, dict):
        return f"{j.path}: verdict kind {kind!r} is not in the merge-base pass table"
    if row_id not in (entry.get("rows") or []):
        return f"{j.path}: {kind} may not support {row_id}"
    if j.record.get("verdict") not in ((entry.get("targets") or {}).get(token) or []):
        return f"{j.path}: verdict {j.record.get('verdict')!r} does not support {token}"
    # #1141: the KIND implies the scope it supports, read from the merge-base; exact match, no subsumption. A base
    # entry without `scopes` supports nothing; a head outside the vocabulary matches nothing.
    scopes = entry.get("scopes")
    if not isinstance(scopes, list):
        return f"{j.path}: {kind} declares no `scopes` in the merge-base pass table (it supports nothing until it does)"
    if scope is not None and not L.SCOPE_HEAD.match(scope):
        return f"{j.path}: the row's qualifier head {scope!r} is not a scope word (narrow, rung X), so no verdict supports it"
    if scope not in scopes:
        shown = "null (the unqualified claim)" if scope is None else repr(scope)
        return (
            f"{j.path}: {kind} supports scopes {scopes}, not {shown} (a verdict re-supporting another scope is not "
            "support for this one; a new scope is a reviewed pass-table change on main first)"
        )
    if not require_met(j.record, entry.get("require") or {}):
        return f"{j.path}: {kind}'s required fields {entry.get('require')} do not hold"
    if j.prereg != "PASS" or j.data_prereg != "PASS":
        return (
            f"{j.path}: prereg status {j.prereg} / its data's {j.data_prereg} "
            "(new support and the data it judges must both be pre-registered and PASS)"
        )
    if j.allowed_dirty:
        return f"{j.path}: allowed-dirty data is never the sole new support"
    if j.time is None or j.time <= after:
        return f"{j.path}: its runs (ts {j.time}) are not after the previous status was set ({after})"
    if kind in O19_KINDS:
        problems = o19_succession_problems(j, token, ctx)
        if problems:
            return f"{j.path}: " + "; ".join(problems)
    return None


EXCEPTION_FIELDS = ("id", "kind", "row", "to", "to_date", "path", "sha256", "owner", "reason", "date")
SUPERSEDED_FIELDS = ("id", "kind", "row", "by", "to_date", "owner", "reason", "date")  # lint_ledger_format's (D3)


def pass_table_problems(table, where: str, *, at_base: bool = False) -> list[str]:
    """The pass table's shape: ``kind -> {rows: [ID], targets: {TOKEN: [verdict]}, require?: {dotted: value}}``."""
    if not isinstance(table, dict):
        return [f"{PASS_TABLE} at {where} is not an object"]
    out = []
    for kind, entry in table.items():
        if kind.startswith("_"):
            continue  # `_comment`
        rows = entry.get("rows") if isinstance(entry, dict) else None
        targets = entry.get("targets") if isinstance(entry, dict) else None
        require = entry.get("require", {}) if isinstance(entry, dict) else None
        if (
            not isinstance(rows, list)
            or not all(isinstance(r, str) and L.ID_RE.match(r) for r in rows)
            or not isinstance(targets, dict)
            or not all(
                t in L.RANK and isinstance(v, list) and all(isinstance(x, str) for x in v) for t, v in targets.items()
            )
            or not isinstance(require, dict)
            or set(entry) - {"rows", "targets", "require", "note", "complete", "scopes"}
        ):
            out.append(f"{PASS_TABLE} at {where}: entry {kind!r} is malformed (rows / targets / require)")
        elif "REPRODUCED" in targets and kind not in O19_KINDS:
            # #1059 S5: REPRODUCED is a successor O19 campaign's label; no other kind has campaigns.
            out.append(f"{PASS_TABLE} at {where}: entry {kind!r} targets REPRODUCED, which only an O19 kind may")
        elif scopes_problem(kind, entry, at_base=at_base):
            out.append(f"{PASS_TABLE} at {where}: entry {kind!r}: {scopes_problem(kind, entry, at_base=at_base)}")
        elif complete_problem(kind, entry.get("complete"), at_base=at_base):
            problem = complete_problem(kind, entry.get("complete"), at_base=at_base)
            out.append(f"{PASS_TABLE} at {where}: entry {kind!r}: {problem}")
    return out


def scopes_problem(kind: str, entry: dict, *, at_base: bool = False) -> str | None:
    """`scopes` (#1141): a non-empty list of unique qualifier heads, each null (the unqualified claim) or a string. At
    HEAD it is REQUIRED, each string a scope word (``_ledger.SCOPE_HEAD``), and a kind with a complete-run rule lists
    exactly one: its frozen sets bind a run's completeness, not which rung it ran, so a second scope on it would let
    an old-scope re-run support the new one (design pass S1). At the MERGE-BASE only the structure is checked: absent
    supplies no support, and a string outside today's vocabulary is inert (``support_problem`` never matches it), so
    narrowing the vocabulary cannot red every PR (design pass S3)."""
    if "scopes" not in entry:
        return None if at_base else "needs `scopes` (the qualifier heads its verdicts support; null = unqualified)"
    scopes = entry["scopes"]
    if (
        not isinstance(scopes, list)
        or not scopes
        or not all(s is None or isinstance(s, str) for s in scopes)
        or len(set(scopes)) != len(scopes)
    ):
        return "`scopes` must be a non-empty list of unique null-or-string qualifier heads"
    if at_base:
        return None
    bad = [s for s in scopes if s is not None and not L.SCOPE_HEAD.match(s)]
    if bad:
        return f"`scopes` {bad} are not scope words (narrow, rung X; null = unqualified)"
    if kind in COMPLETE_RULES and len(scopes) != 1:
        return "a kind with a complete-run rule supports exactly one scope (a new scope on a ruled kind is a new kind)"
    return None


def scope_change_notes(base_table, head_table) -> list[str]:
    """The pass-table change that admits a new scope is the review point for #1141's rule, so make it loud: name every
    scope a kind gains, and say when its `require` / `complete` did not change with it (nothing then binds a verdict
    of that kind to the new scope but review)."""
    if not isinstance(base_table, dict) or not isinstance(head_table, dict):
        return []
    out: list[str] = []
    first: list[str] = []
    for kind, entry in head_table.items():
        if kind.startswith("_") or not isinstance(entry, dict) or not isinstance(entry.get("scopes"), list):
            continue
        old = base_table.get(kind) if isinstance(base_table.get(kind), dict) else {}
        if old and "scopes" not in old:
            first.append(kind)  # the bootstrap (#1141): the kind's existing support, now declared
            continue
        old_scopes = old.get("scopes") if isinstance(old.get("scopes"), list) else []
        added = [s for s in entry["scopes"] if s not in old_scopes]
        if not added:
            continue
        unbound = old and entry.get("require") == old.get("require") and entry.get("complete") == old.get("complete")
        out.append(
            f"{PASS_TABLE}: {kind} now supports scope(s) {added}: review decides how its verdicts are bound to them"
            + (
                " (its require / complete did not change: the new scope is not bound to its verdicts)"
                if unbound
                else ""
            )
        )
    if first:
        out.append(f"{PASS_TABLE}: {len(first)} kind(s) declare their first `scopes` (bootstrap): {', '.join(first)}")
    return out


def ruled_scope_problems(base_table, head_table) -> list[str]:
    """A kind with a complete-run rule keeps the scope main gave it (#1141 review): its frozen sets bind completeness,
    not which rung ran, so swapping ``["rung A"]`` for ``["rung B"]`` would let a rung-A re-run support rung B. A new
    scope on a ruled kind is a new kind. Deleting the entry is refused too (delta round): a later PR could re-add it
    with another scope, and the base it is judged against would hold none. Retire one by emptying its ``targets``.
    (A base entry without ``scopes`` is the bootstrap: nothing to keep.)"""
    if not isinstance(base_table, dict) or not isinstance(head_table, dict):
        return []
    out = []
    for kind, old in base_table.items():
        if kind not in COMPLETE_RULES or not isinstance(old, dict) or "scopes" not in old:
            continue
        entry = head_table.get(kind)
        if not isinstance(entry, dict):
            out.append(
                f"{PASS_TABLE}: {kind} has a complete-run rule and scopes {old['scopes']} on main, so its entry may not "
                "be deleted (retire it by emptying its targets, keeping scopes and complete)"
            )
        elif entry.get("scopes") != old["scopes"]:
            out.append(
                f"{PASS_TABLE}: {kind} has a complete-run rule, so its scopes {old['scopes']} may not change "
                "(a new scope on a ruled kind is a new kind)"
            )
    return out


def complete_problem(kind: str, complete, *, at_base: bool = False) -> str | None:
    """A kind with a code-owned complete-run rule MUST carry its `complete` block at HEAD (and no other kind may): a
    missing block would read as "no completeness check". At the MERGE-BASE a ruled kind may still lack its block (the
    PR adding the rule cannot also have added the block on main); the record judges refuse its verdicts meanwhile.
    A block whose kind has NO rule fails at both: dropping a rule is caught."""
    rule = COMPLETE_RULES.get(kind)
    if rule is None:
        return None if complete is None else "carries a `complete` block, but no complete-run rule exists for it"
    if complete is None and at_base:
        return None
    if not isinstance(complete, dict) or complete.get("rule") != rule:
        return f"needs a `complete` block with rule {rule!r}"
    if rule == "exp53_runs":
        manifest = complete.get("manifest")
        if set(complete) != {"rule", "start", "manifest"} or not isinstance(complete.get("start"), dict):
            return "`complete` must be exactly {rule, start, manifest} with `start` an object"
        if not isinstance(manifest, str) or not manifest.startswith(DATA_ROOT + "/") or ".." in manifest.split("/"):
            return "`complete.manifest` must be a path under the data root"
        return None
    sets = complete.get("sets")
    if set(complete) != {"rule", "sets"} or not isinstance(sets, dict) or not sets:
        return "`complete` must be exactly {rule, sets} with `sets` a non-empty object"
    for arm, seeds in sets.items():
        if (
            not isinstance(seeds, list)
            or not seeds
            or not all(type(x) is int for x in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            return f"`complete.sets.{arm}` must be a non-empty list of distinct integers"
    return None


def load_json(repo: Repo, ref: str, path: str, default):
    raw = repo.blob(ref, path)
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise GateError(f"{path} at {ref[:12]} is not JSON") from exc


def exceptions_problems(base_list, head_list, repo: Repo | None = None) -> list[str]:
    out = []
    if not isinstance(head_list, list) or not isinstance(base_list, list):
        return [f"{EXCEPTIONS} must be a JSON list"]
    head_set = [json.dumps(e, sort_keys=True) for e in head_list]
    for e in base_list:
        if json.dumps(e, sort_keys=True) not in head_set:
            label = e.get("id") if isinstance(e, dict) else e
            out.append(f"{EXCEPTIONS}: an entry on main was edited or removed (append-only): {label!r}"[:200])
    for e in head_list:
        if not isinstance(e, dict) or e.get("kind") not in ("ledger", "prereg", "superseded"):
            out.append(f"{EXCEPTIONS}: an entry is not a ledger/prereg/superseded exception: {e!r}"[:200])
        elif e["kind"] == "superseded" and (
            any(not e.get(f) for f in SUPERSEDED_FIELDS) or "from" not in e  # `from: null` = a new row
        ):
            # Read by lint_ledger_format.py (D3): a row entering SUPERSEDED, or re-pointing it, without its successor
            # reaching a positive status in the same diff.
            out.append(f"{EXCEPTIONS}: superseded exception {e.get('id')!r} lacks a required field")
        elif e["kind"] == "ledger" and (
            any(not e.get(f) for f in EXCEPTION_FIELDS) or "from" not in e  # `from: null` = a new row
        ):
            out.append(f"{EXCEPTIONS}: ledger exception {e.get('id')!r} lacks a required field")
        elif (
            e["kind"] == "ledger"
            and "to_qualifier" in e
            and not (e["to_qualifier"] is None or (isinstance(e["to_qualifier"], str) and e["to_qualifier"]))
        ):
            out.append(
                f"{EXCEPTIONS}: ledger exception {e.get('id')!r}: to_qualifier must be a non-empty string or null"
            )
        elif e["kind"] == "prereg":
            # The pin's form is checked against HEAD only for a NEW clause: one already on main whose entry later
            # changed shape is inert (the prereg lint notes it), never a permanent failure of every PR.
            root = repo.root if repo is not None and e not in base_list else None
            problem = P.prereg_exception_problem(e, root)
            if problem:
                out.append(f"{EXCEPTIONS}: prereg exception {e.get('id')!r} {problem}")
    ids = [e.get("id") for e in head_list if isinstance(e, dict)]
    if not all(isinstance(i, str) and i for i in ids) or len(set(ids)) != len(ids):
        out.append(f"{EXCEPTIONS}: every entry needs a unique string id")
    return out


def legacy_problems(repo: Repo, head_snap, base_snap) -> list[str]:
    out = []
    if not isinstance(head_snap, dict):
        return [f"{LEGACY_SNAPSHOT} must be a JSON object"]
    if base_snap is not None and isinstance(base_snap, dict):
        for key in sorted(set(head_snap) - set(base_snap)):
            out.append(f"{LEGACY_SNAPSHOT}: {key} was added (the snapshot only shrinks)")
    tree = repo.tree("HEAD")
    for path, digest in sorted(head_snap.items()):
        if path not in tree:
            is_dir = any(p.startswith(path.rstrip("/") + "/") for p in tree)
            out.append(
                f"{LEGACY_SNAPSHOT}: {path} is a directory, not a file (keys name files)"
                if is_dir
                else f"{LEGACY_SNAPSHOT}: {path} is gone (remove its key)"
            )
            continue
        data = repo.blob("HEAD", path) or b""
        if sha256(data) != digest:
            out.append(f"{LEGACY_SNAPSHOT}: {path} changed since it was snapshotted (a changed record is new data)")
            continue
        first = repo.first_commit_time(path, "HEAD")
        if first is None or first >= M1A_CUTOFF:
            out.append(f"{LEGACY_SNAPSHOT}: {path} was first committed after M1a")
        try:
            body = decompressed(path, data)
        except MALFORMED:
            body = data
        if b'"record_kind"' in body:
            out.append(f"{LEGACY_SNAPSHOT}: {path} carries a record_kind (it is not legacy)")
    return out


def generate_legacy(repo: Repo) -> dict[str, str]:
    snap = {}
    for path, (mode, _oid) in sorted(repo.tree("HEAD").items()):
        if not path.startswith(DATA_ROOT + "/") or mode in ("120000", "160000"):
            continue  # a symlink or a submodule is never a record
        first = repo.first_commit_time(path, "HEAD")
        if first is None or first >= M1A_CUTOFF:
            continue
        data = repo.blob("HEAD", path) or b""
        try:
            body = decompressed(path, data)
        except MALFORMED:
            body = data
        if b'"record_kind"' not in body:
            snap[path] = sha256(data)
    return snap


def status_set_time(repo: Repo, base: str, row_id: str, token: str, date: str) -> float:
    """The commit time at which the base's status line of ``row_id`` was first set (first-parent history)."""
    commits = repo.git("log", "--first-parent", "--format=%H %ct", base, "--", L.LEDGER_PATH).splitlines()
    when = None
    for line in commits:
        sha, ct = line.split()
        rows, _ = L.parse((repo.blob(sha, L.LEDGER_PATH) or b"").decode("utf-8", errors="replace"))
        row = next((r for r in rows if r.id == row_id), None)
        if row is None or (row.token, row.date) != (token, date):
            break
        when = float(ct)
    return when if when is not None else float(repo.commit_time(base))


# ── the gate ─────────────────────────────────────────────────────────────────────────────────────────────


@dataclass
class RowResult:
    id: str
    triggers: list[str]
    failures: list[str]
    notes: list[str]


def qualifier_widens(old: L.Row | None, row: L.Row) -> bool:
    """A qualifier change that can WIDEN the claim (#1108, owner decision 2026-10-06): the qualifier is removed, or its
    scope word changes (``narrow`` -> ``rung A``, ``rung A`` -> ``rung B``). It needs NEW support, like a raise or a
    re-date. Detail appended under the same scope word, or a qualifier added to an unqualified row, is judged only;
    text cannot prove an appended clause narrows, so a same-head widening (``rung A, and rung B``) is the stated
    residual that review checks."""
    if old is None:
        return False
    if old.qualifier_unparsed:
        return True  # #1141 review: an unparsed base qualifier cannot show that the change keeps its scope
    if not old.qualifier:
        return False
    return row.qualifier is None or L.qualifier_head(row.qualifier) != L.qualifier_head(old.qualifier)


def triggers_for(row: L.Row, old: L.Row | None, changed: set[str], watched: list[str]) -> list[str]:
    out = []
    if old is None:
        return ["new row"]
    if L.is_raise(old.token, row.token):
        out.append(f"raise {old.token} -> {row.token}")
    elif old.token != row.token and {old.token, row.token} <= L.POSITIVE:
        out.append(f"move {old.token} -> {row.token}")
    if old.date != row.date:
        out.append(f"date {old.date} -> {row.date}")
    if {e.path for e in old.evidence} != {e.path for e in row.evidence}:
        out.append("Evidence changed")
    if old.claim != row.claim:
        out.append("claim changed")
    # Any change but an exact match is judged (#1108: the old substring test let `(rung A)` -> `(rung A and B)` pass
    # unjudged). Compared raw: a ledger row is one table line, so a whitespace change inside the parentheses is an edit.
    if old.qualifier != row.qualifier:
        if old.qualifier is None:
            out.append("qualifier added")
        elif row.qualifier is None:
            out.append("qualifier removed")
        else:
            out.append("qualifier changed")
        if qualifier_widens(old, row):
            out.append("qualifier widened")
    for prefix in watched:
        if any(c == prefix or c.startswith(prefix.rstrip("/") + "/") for c in changed):
            out.append(f"{prefix} changed")
    return out


def active_exceptions(base_exc, row: L.Row, old: L.Row | None) -> list[dict]:
    """The ledger exceptions that act on ``row`` now. Only clauses already on main act (an exception added by this
    change is reviewed first, used after), and only for the row's CURRENT transition: HEAD's token and date are the
    clause's ``to`` / ``to_date``, and this change performs its ``from -> to`` or the base already sits there. A clause
    for a transition that is no longer current is inert history."""
    out = []
    for e in base_exc if isinstance(base_exc, list) else []:
        if not (isinstance(e, dict) and e.get("kind") == "ledger" and e.get("row") == row.id):
            continue
        if not (isinstance(e.get("id"), str) and isinstance(e.get("path"), str)):
            continue  # malformed (reported by exceptions_problems): never acts
        if (e.get("to"), e.get("to_date")) != (row.token, row.date):
            continue
        performs = e.get("from") == (old.token if old else None)
        settled = old is not None and (old.token, old.date) == (row.token, row.date)
        if performs or settled:
            out.append(e)
    return out


def pinned(repo: Repo, clause: dict) -> bool:
    """The clause's ``sha256`` is the bytes of the FILE it names at HEAD (absent, or not a file: never pinned)."""
    path = clause.get("path")
    if not isinstance(path, str) or repo.kind("HEAD", path) != "blob":
        return False
    return clause.get("sha256") == sha256(repo.blob("HEAD", path) or b"")


def judged_class(row: L.Row) -> bool:
    """Positive rows, PARTIAL rows (on every change: owner decision 2026-10-01) and RE-VALIDATED-BY-TESTS (noted:
    named, not checked)."""
    return row.token in L.POSITIVE or row.token in ("PARTIAL", L.BY_TESTS)


def gate(
    repo_root: Path = REPO_ROOT, *, base: str | None = None, prereg: dict[str, str] | None = None
) -> tuple[list[str], list[str], list[RowResult]]:
    """(failures, notes, per-row results)."""
    repo = Repo(repo_root)
    base = base or _lint_git.base_ref(repo.root)
    failures: list[str] = []
    notes: list[str] = []
    dirty = repo.dirty([L.LEDGER_PATH, DATA_ROOT, PASS_TABLE, LEGACY_SNAPSHOT, EXCEPTIONS])
    if dirty:
        failures.append(f"uncommitted changes in gated paths (uncommitted evidence is not evidence): {dirty[:5]}")
    head_rows, problems = L.parse((repo.blob("HEAD", L.LEDGER_PATH) or b"").decode("utf-8"))
    base_rows = {r.id: r for r in L.parse((repo.blob(base, L.LEDGER_PATH) or b"").decode("utf-8"))[0]}
    if problems:
        failures += [f"ledger: {p}" for p in problems]
    table = load_json(repo, base, PASS_TABLE, {})
    failures += pass_table_problems(table, "the merge-base", at_base=True)
    if not isinstance(table, dict):
        table = {}
    head_table = load_json(repo, "HEAD", PASS_TABLE, {})
    failures += pass_table_problems(head_table, "HEAD")
    failures += ruled_scope_problems(table, head_table)
    notes += scope_change_notes(table, head_table)
    head_snap = load_json(repo, "HEAD", LEGACY_SNAPSHOT, None)
    base_snap = load_json(repo, base, LEGACY_SNAPSHOT, None)
    if head_snap is None:
        failures.append(f"{LEGACY_SNAPSHOT} is missing")
        head_snap = {}
    failures += legacy_problems(repo, head_snap, base_snap)
    base_exc = load_json(repo, base, EXCEPTIONS, [])
    head_exc = load_json(repo, "HEAD", EXCEPTIONS, [])
    failures += exceptions_problems(base_exc, head_exc, repo)
    if prereg is None:
        try:
            envelope = P.classify_all(repo.root, base)
        except P.LintError as exc:
            raise GateError(f"the prereg classification could not run: {exc}") from exc
        failures += [f"prereg lint: {f}" for f in envelope.get("failures") or []]
        prereg = {e["entry"]: e["status"] for e in envelope.get("entries") or []}
    ctx = Ctx(repo=repo, base=base, ref="HEAD", legacy=head_snap, prereg=prereg, table=table)
    changed = repo.changed(base)
    # #1050 half B: a judge edit may not change any existing O19 verdict, cited by a row or not (strict).
    if O19_JUDGE in changed:
        failures += o19_judge_edit_problems(ctx)
        # #1059 D1/S3: the kind -> experiment map and every frozen campaign entry are append-only.
        failures += o19_table_problems(repo, base)
    # #1059: a closed campaign's data directory is immutable (every diff, whatever else it touches).
    failures += o19_closed_data_problems(ctx, changed)
    # #1050 N2: every judge main ever held still loads through this (possibly edited) gate.
    failures += o19_history_problems(repo, base)
    results = []
    for row in head_rows:
        old = base_rows.get(row.id)
        if row.token is None:
            continue
        if old and old.token in LEAVING_UNCHECKED and row.token != old.token and not judged_class(row):
            notes.append(f"{row.id}: leaves {old.token} for {row.token} (not judged: review decides)")
        if row.token == L.BY_TESTS and (old is None or old.token != L.BY_TESTS):
            # A base row with no parsed token (its parse problems are reported elsewhere) is named as such (#1037).
            entered = (f"from {old.token}" if old.token else "from an unparsed status") if old else "as a new row"
            notes.append(f"{row.id}: enters RE-VALIDATED-BY-TESTS {entered} (not judged: review decides)")
        if not judged_class(row):
            continue
        judgements = [judge_entry(e.path, ctx) for e in row.evidence]
        watched = [e.path for e in row.evidence] + [p for j in judgements for p in watched_paths(j)]
        trig = triggers_for(row, old, changed, watched)
        if not trig:
            continue
        res = RowResult(id=row.id, triggers=trig, failures=[], notes=[])
        results.append(res)
        if row.token == L.BY_TESTS:
            res.notes.append("RE-VALIDATED-BY-TESTS: named, not checked")
            notes += [f"{row.id}: {n}" for n in res.notes]
            continue
        if row.qualifier_unparsed:
            res.failures.append(
                "the Status line's parenthesised qualifier was not parsed (nested parentheses?): the row would read as "
                "unqualified, the full claim"
            )
        # A clause acts only when pinned to the cited FILE's bytes (a session directory is never pinned).
        active = [e for e in active_exceptions(base_exc, row, old) if pinned(repo, e)]
        excepted: set[str] = set()
        for j in judgements:
            exc = next((e for e in active if e.get("path") == j.path), None)
            if j.status == NOT_ESTABLISHED and exc is not None:
                j.status = EXCEPTED
                excepted.add(exc.get("id"))
            if j.status == NOT_ESTABLISHED:
                res.failures += [f"{j.path}: {r}" for r in (j.reasons or ["not established"])]
            elif j.status != ESTABLISHED:
                res.notes.append(f"{j.path}: {j.status}")
        # A judged row that changes rests on something (owner decision 2026-10-01): an Evidence-less PARTIAL row
        # cannot be rewritten with nothing to judge.
        if not any(j.status in (ESTABLISHED, LEGACY, EXCEPTED) for j in judgements):
            res.failures.append(
                "cites no ESTABLISHED, LEGACY or EXCEPTED record (a judged row that changes rests on one)"
            )
        widened = qualifier_widens(old, row)
        needs_support = old is None or L.needs_new_date(old.token, row.token) or old.date != row.date or widened
        if needs_support:
            base_paths = {e.path for e in old.evidence} if old else set()
            after = status_set_time(repo, base, row.id, old.token, old.date) if old else float(repo.commit_time(base))
            new = [j for j in judgements if j.path not in base_paths]
            scope = row_scope(row)
            problems_new = [support_problem(j, row.id, row.token, table, after, ctx=ctx, scope=scope) for j in new]
            cited = {e.path for e in row.evidence}
            supporting = [e for e in active if e.get("from") == (old.token if old else None) and e.get("path") in cited]
            if widened:
                # For a pure widening every clause at the current (to, to_date) reads as "settled", so a re-date clause
                # would support every later widening. Only a clause bound to THIS qualifier does (design pass, #1108).
                supporting = [e for e in supporting if "to_qualifier" in e and e["to_qualifier"] == row.qualifier]
            excepted |= {e.get("id") for e in supporting}
            if not any(p is None for p in problems_new) and not supporting:
                hint = (
                    " (a widened qualifier: an owner exception supplies it only with `to_qualifier` naming the new one)"
                    if widened
                    else ""
                )
                res.failures.append(
                    "no NEW support: " + ("; ".join(p for p in problems_new if p) or "no newly cited record") + hint
                )
        cited_paths = {j.path for j in judgements}
        for e in active:
            if e.get("id") not in excepted and e.get("path") in cited_paths:
                res.notes.append(
                    f"{e.get('path')}: exception {e.get('id')!r} is stale (the record is judged without it)"
                )
        for e in active_exceptions(base_exc, row, old):
            if e.get("path") in cited_paths and not pinned(repo, e):
                res.notes.append(f"{e.get('path')}: exception {e.get('id')!r} does not pin the cited bytes (inert)")
        if old and {e.path for e in old.evidence} - {e.path for e in row.evidence}:
            base_ctx = Ctx(
                repo=repo,
                base=base,
                ref=base,
                legacy=load_json(repo, base, LEGACY_SNAPSHOT, {}) or {},
                prereg=prereg,
                table=table,
            )
            at_base = [judge_entry(e.path, base_ctx) for e in old.evidence]
            # A base record the judges could not judge (a gate defect) keeps the ratchet ON (fail closed), and says why.
            could_not = [b for b in at_base if unjudged(b)]
            res.notes += [f"{b.path} at the merge-base could not be judged: {b.reasons[-1]}" for b in could_not]
            if not any(j.status == ESTABLISHED for j in judgements):
                if any(b.status == ESTABLISHED for b in at_base):
                    res.failures.append("Evidence removed: the row had an ESTABLISHED record on main and keeps none")
                elif could_not:
                    res.failures.append(
                        "Evidence removed while a base record could not be judged: keep or add an ESTABLISHED record"
                    )
        failures += [f"{row.id}: {f}" for f in res.failures]
        notes += [f"{row.id}: {n}" for n in res.notes]
    return failures, notes, results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--base", help="diff against this commit (default: the merge-base with origin/main)")
    ap.add_argument("--write-legacy", action="store_true", help="write the legacy snapshot from the rules")
    args = ap.parse_args(argv)
    if args.write_legacy:
        snap = generate_legacy(Repo(REPO_ROOT))
        (REPO_ROOT / LEGACY_SNAPSHOT).write_text(json.dumps(snap, indent=1, sort_keys=True) + "\n")
        print(f"wrote {LEGACY_SNAPSHOT}: {len(snap)} legacy records")
        return 0
    try:
        failures, notes, results = gate(REPO_ROOT, base=args.base)
    except _lint_git.GitUnavailable as exc:
        if _lint_git.must_not_skip(f"evidence gate: {exc}"):
            return 2
        print(f"SKIP evidence gate (no merge-base: {exc})", file=sys.stderr)
        return 0
    except GateError as exc:
        print(f"evidence gate FAILED: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"failures": failures, "notes": notes,
                          "rows": [r.__dict__ for r in results]}, indent=2))  # fmt: skip
    else:
        for n in notes:
            print(f"NOTE {n}")
        if failures:
            print(f"evidence gate FAILED ({len(failures)}):", file=sys.stderr)
            for f in failures:
                print(f"  {f}", file=sys.stderr)
        else:
            print(f"evidence gate: clean ({len(results)} triggered row(s) judged)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
