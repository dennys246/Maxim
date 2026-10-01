"""Judging cited records for the ledger evidence gate (M1b PR 5b; spec: docs/plans/m1b_ledger_evidence_gate.md,
"PR 5b build spec"). ``scripts/lint_evidence_gate.py`` owns the ledger side (triggers, support, exceptions, the
legacy snapshot); this module owns what a cited record IS and whether it is established.

Imports nothing from ``maxim`` (``_provenance`` loads ``src/maxim/utils/code_tree.py`` by path, stdlib-only). Records are read from git objects (committed bytes), never the working
tree. Every judge refuses a malformed input (NOT-ESTABLISHED with a reason) rather than raising: a gate that crashes
on a hand-edited file judges nothing.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _ledger as L  # noqa: E402
import _lint_git  # noqa: E402
from _provenance import provenance_digest  # noqa: E402

DATA_ROOT = L.DATA_ROOT
O19_JUDGE = "scripts/o19_verdict.py"
O19_KINDS = frozenset({"exp10_verdict", "exp09_verdict"})
# A run's finish reasons that are citable (pinned against simulation/sim_types.py's failure set by a test).
FINISH_OK = frozenset({"completed", "max_turns", "complete", "all_encounters_complete"})
FINISH_OK_PREFIX = "campaign_end:"
SKEW_S = 300  # a unit's ts may precede its own executed commit's committer time by at most this (clock skew)
HEX40 = re.compile(r"^[0-9a-f]{40}$")

ESTABLISHED, LEGACY, EXCEPTED, NOT_ESTABLISHED = "ESTABLISHED", "LEGACY", "EXCEPTED", "NOT-ESTABLISHED"
SUPPORT_KINDS = frozenset({"verdict"})  # only a stamped verdict supplies NEW support (owner decision 2026-09-30)
NON_SUPPORT_KINDS = frozenset({"instrument_check", "diagnosis", "harness_header", "harness_demo"})
SINGLE_DOCUMENT_KINDS = frozenset({"verdict", "instrument_check", "diagnosis"})
BAD_PREREG = frozenset({"FAIL", "NON_GATED", "NOT_GOVERNED"})
# Errors the judges' own type guards should make impossible: on main's evidence one means "could not be judged".
CODE_ERRORS = ("TypeError", "AttributeError", "KeyError", "IndexError")
# What a malformed record raises while being read; each becomes a refusal, never a crash.
MALFORMED = (TypeError, AttributeError, ValueError, KeyError, IndexError, EOFError, zlib.error, OSError, UnicodeError)


class GateError(Exception):
    """The gate could not read something it needs: the row fails, never passes."""


# ── git access (committed bytes only) ─────────────────────────────────────────────────────────────────────


class Repo:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._blobs: dict[tuple[str, str], bytes | None] = {}
        self._cache: dict[str, object] = {}

    def git(self, *args: str) -> str:
        return _lint_git.git(self.root, *args)

    def kind(self, ref: str, path: str) -> str | None:
        """The object type at ``ref:path`` (``blob``, ``tree``, ``commit``), or None when absent there."""
        out = subprocess.run(["git", "cat-file", "-t", f"{ref}:{path}"], cwd=self.root, capture_output=True)
        return out.stdout.decode().strip() if out.returncode == 0 else None

    def blob(self, ref: str, path: str) -> bytes | None:
        """The file's bytes at ``ref``; None ONLY when it is absent there. A non-file object (a directory) is a
        GateError; a read failure of a file raises GitUnavailable."""
        key = (ref, path)
        if key not in self._blobs:
            kind = self.kind(ref, path)
            if kind is None:
                self._blobs[key] = None
            elif kind != "blob":
                raise GateError(f"{ref}:{path} is a {kind}, not a file")
            else:
                out = subprocess.run(["git", "cat-file", "blob", f"{ref}:{path}"], cwd=self.root, capture_output=True)
                if out.returncode != 0:
                    raise _lint_git.GitUnavailable(f"git cat-file {ref}:{path}: {out.stderr.decode().strip()}")
                self._blobs[key] = out.stdout
        return self._blobs[key]

    def tree(self, ref: str) -> dict[str, tuple[str, str]]:
        """``path -> (mode, blob id)`` for every file under the data root and scripts/ at ``ref``."""
        key = f"tree:{ref}"
        if key not in self._cache:
            files: dict[str, tuple[str, str]] = {}
            for line in self.git("ls-tree", "-r", ref, "--", DATA_ROOT, "scripts").splitlines():
                meta, _, path = line.partition("\t")
                mode, _type, oid = meta.split()
                files[path] = (mode, oid)
            self._cache[key] = files
        return self._cache[key]  # type: ignore[return-value]

    def is_ancestor(self, commit: str, of: str) -> bool:
        out = subprocess.run(["git", "merge-base", "--is-ancestor", commit, of], cwd=self.root, capture_output=True)
        return out.returncode == 0

    def commit_time(self, ref: str) -> int:
        key = f"ct:{ref}"
        if key not in self._cache:
            self._cache[key] = int(self.git("show", "-s", "--format=%ct", ref).strip())
        return self._cache[key]  # type: ignore[return-value]

    def first_commit_time(self, path: str, ref: str) -> int | None:
        """When ``path`` was first added under the data root on ``ref``'s history (one history pass, cached)."""
        key = f"added:{ref}"
        if key not in self._cache:
            added: dict[str, int] = {}
            when = 0
            for line in self.git(
                "log", "--diff-filter=A", "--name-only", "--format=@%ct", ref, "--", DATA_ROOT
            ).splitlines():
                if line.startswith("@"):
                    when = int(line[1:])
                elif line.strip():
                    added[line.strip()] = when  # newest first: the last write is the oldest add
            self._cache[key] = added
        return self._cache[key].get(path)  # type: ignore[union-attr]

    def changed(self, base: str) -> set[str]:
        """Every path changed between ``base`` and HEAD, renames as a delete plus an add (both names)."""
        out = self.git("diff", "--name-only", "--no-renames", "-z", base, "HEAD")
        return {p for p in out.split("\0") if p}

    def dirty(self, paths: list[str]) -> list[str]:
        out = self.git("status", "--porcelain", "--untracked-files=all", "--", *paths)
        return [ln[3:] for ln in out.splitlines() if ln.strip()]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decompressed(path: str, data: bytes) -> bytes:
    return gzip.decompress(data) if path.endswith(".gz") else data


def json_lines(data: bytes) -> list[dict]:
    rows = []
    for i, raw in enumerate(data.decode("utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            rec = json.loads(raw)
        except ValueError as exc:
            raise GateError(f"line {i} is not JSON") from exc
        if not isinstance(rec, dict):
            raise GateError(f"line {i} is not a JSON object")
        rows.append(rec)
    return rows


def kind_of(record: dict) -> str:
    """A record's ``record_kind`` as a string; any other shape is a kind no rule accepts (never hashed raw)."""
    kind = record.get("record_kind")
    return kind if isinstance(kind, str) else f"<{type(kind).__name__}>"


def str_field(record: dict, key: str) -> str | None:
    """``record[key]`` when it is a string, else None: a record value is never hashed in another shape."""
    value = record.get(key)
    return value if isinstance(value, str) else None


def unknown(value) -> bool:
    return not isinstance(value, str) or not value or value == "unknown" or value.startswith("unknown:")


def as_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


# ── judgements ────────────────────────────────────────────────────────────────────────────────────────────


@dataclass
class Judgement:
    path: str
    status: str = NOT_ESTABLISHED
    kind: str | None = None
    reasons: list[str] = field(default_factory=list)
    time: float | None = None  # min ts over counted units
    allowed_dirty: bool = False
    record: dict | None = None  # a verdict's own JSON (support candidates)
    prereg: str | None = None
    data_prereg: str | None = None  # a verdict's data entry's prereg status

    def fail(self, why: str) -> Judgement:
        self.status = NOT_ESTABLISHED
        self.reasons.append(why)
        return self


@dataclass
class Ctx:
    repo: Repo
    base: str
    ref: str  # where records are read (HEAD; the base for the Evidence-removal ratchet)
    legacy: dict[str, str]
    prereg: dict[str, str]  # top-level data entry -> classify_all status

    def prereg_status(self, path: str) -> str | None:
        best = None
        for entry, status in self.prereg.items():
            if path == entry or path.startswith(entry + "/"):
                if best is None or len(entry) > len(best[0]):
                    best = (entry, status)
        return best[1] if best else None


def judge_provenance(prov, j: Judgement, ctx: Ctx, label: str, *, allow_dirty_ok: bool = True) -> str | None:
    """Check a provenance block; returns its code tree, or None after failing ``j``."""
    if not isinstance(prov, dict):
        j.fail(f"{label}: no provenance block")
        return None
    executed = prov.get("executed_git_hash")
    if not isinstance(executed, str) or not HEX40.match(executed):
        j.fail(f"{label}: executed_git_hash {executed!r} is not a full commit id")
        return None
    if not ctx.repo.is_ancestor(executed, ctx.base):
        j.fail(f"{label}: executed {executed[:12]} is not on main (an ancestor of the merge-base)")
    tree = prov.get("code_tree_sha256")
    if unknown(tree):
        j.fail(f"{label}: code_tree_sha256 is {tree!r}")
        return None
    dirty = prov.get("working_tree_dirty_src_scripts")
    if dirty is not False:
        if dirty is True and allow_dirty_ok and prov.get("allow_dirty") is True:
            j.allowed_dirty = True
        else:
            j.fail(f"{label}: working_tree_dirty_src_scripts is {dirty!r} without a granted allow_dirty")
    return tree


def check_skew(ts, executed, j: Judgement, ctx: Ctx, label: str) -> None:
    """A counted unit cannot have run before the commit IT names (allowing clock skew). Judged per unit against its
    own ``executed_git_hash``: never a verdict writer's, a header's or a failed row's later commit."""
    if not isinstance(ts, (int, float)) or not isinstance(executed, str) or not HEX40.match(executed):
        return  # a missing ts / hash is refused by its own rule
    if not ctx.repo.is_ancestor(executed, ctx.base):
        return  # a commit not on main is refused by its own rule
    when = ctx.repo.commit_time(executed)
    if ts < when - SKEW_S:
        j.fail(f"{label}: ran at ts {ts:.0f}, before the commit it claims to have run on ({executed[:12]}, {when})")


def finish_ok(reason) -> bool:
    return isinstance(reason, str) and (reason in FINISH_OK or reason.startswith(FINISH_OK_PREFIX))


def judge_sim_fields(sim: dict, j: Judgement, ctx: Ctx, label: str, tree: str | None) -> None:
    """A sim report's (or a harness row's echoed) evidence fields. A dirty sim is refused outright (stricter than
    the base design's bound-allowance path, which no harness writes)."""
    if not finish_ok(sim.get("finish_reason")):
        j.fail(f"{label}: finish_reason {sim.get('finish_reason')!r} is not citable")
    if sim.get("code_changed_during_run") is not False:
        j.fail(f"{label}: code_changed_during_run is {sim.get('code_changed_during_run')!r}")
    if sim.get("working_tree_dirty_src_scripts") is not False:
        j.fail(f"{label}: the sim ran on a dirty tree")
    start, end = sim.get("code_tree_sha256"), sim.get("end_code_tree_sha256")
    if unknown(start) or start != end or (tree is not None and start != tree):
        j.fail(f"{label}: code tree {start!r} / end {end!r} / harness {tree!r} are not one known tree")
    if not isinstance(sim.get("configured_n_ctx"), int):
        j.fail(f"{label}: no configured_n_ctx")
    if not sim.get("language_profile"):
        j.fail(f"{label}: no language_profile")
    for key in sorted(k for k in sim if isinstance(k, str) and k.endswith("_profile") and sim[k]):
        role = key[: -len("_profile")]
        if not isinstance(sim.get(f"{role}_router_n_ctx"), int):
            j.fail(f"{label}: {role} ran without a stamped {role}_router_n_ctx")
    if not isinstance(sim.get("ts"), (int, float)):
        j.fail(f"{label}: no ts")
    resume = sim.get("resume")
    if resume is not None and not (isinstance(resume, dict) and resume.get("resume_loaded") is True):
        j.fail(f"{label}: resumed, but resume_loaded is not true")
    executed = sim.get("executed_git_hash")
    if not isinstance(executed, str) or not HEX40.match(executed) or not ctx.repo.is_ancestor(executed, ctx.base):
        j.fail(f"{label}: executed {executed!r} is not a full commit on main")
    else:
        check_skew(sim.get("ts"), executed, j, ctx, label)


def judge_sim_report(report, j: Judgement, ctx: Ctx) -> None:
    if not isinstance(report, dict) or report.get("record_kind") != "sim_report":
        j.fail("report.json is not a stamped sim_report")
        return
    flat = {**as_dict(report.get("provenance")), "finish_reason": report.get("finish_reason"), "ts": report.get("ts")}
    judge_sim_fields(flat, j, ctx, "sim_report", None)
    if isinstance(report.get("ts"), (int, float)):
        j.time = float(report["ts"])


def counted_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("record_kind") == "harness_row" and r.get("status") != "failed"]


def judge_row_file(rows: list[dict], j: Judgement, ctx: Ctx, *, scope_rows: list[dict] | None = None) -> None:
    """A rows file: only harness rows and headers; no mock; each non-failed row established; one tree."""
    kinds = {kind_of(r) for r in rows}
    if not kinds <= {"harness_row", "harness_header"}:
        j.fail(f"a rows file holds other kinds {sorted(map(str, kinds - {'harness_row', 'harness_header'}))}")
        return
    if any(r.get("mock") is not False for r in rows):
        j.fail("a line is mock or does not say (a mock line sinks the whole file)")
        return
    trees = set()
    for i, r in enumerate(rows, 1):
        if r.get("record_kind") == "harness_header" or r.get("status") == "failed":
            judge_provenance(r.get("provenance"), j, ctx, f"line {i} provenance")
            continue
        tree = judge_provenance(r.get("provenance"), j, ctx, f"row {i}")
        trees.add(tree)
        if isinstance(r.get("ts"), (int, float)):
            check_skew(r["ts"], as_dict(r.get("provenance")).get("executed_git_hash"), j, ctx, f"row {i}")
        family = as_dict(r.get("provenance")).get("harness_family")
        sims = r.get("sims")
        if family not in ("spawning", "in_process"):
            j.fail(f"row {i}: harness_family {family!r} is neither spawning nor in_process")
        elif family == "spawning":
            if not isinstance(sims, list) or not sims:
                j.fail(f"row {i}: a spawning harness row names no sims")
                continue
            for s, sim in enumerate(sims):
                if isinstance(sim, dict):
                    judge_sim_fields(sim, j, ctx, f"row {i} sim {s}", tree)
                else:
                    j.fail(f"row {i} sim {s}: not an object")
        elif sims:
            j.fail(f"row {i}: an in-process row carries sims (its sims would go unjudged)")
    if len(trees) > 1:
        j.fail(f"the rows ran on {len(trees)} code trees (one per file)")
    units = counted_rows(scope_rows if scope_rows is not None else rows)
    known = [t for t in (unit_time(r) for r in units) if t is not None]
    j.time = min(known) if known else None


def unit_time(row: dict) -> float | None:
    if isinstance(row.get("ts"), (int, float)):
        return float(row["ts"])
    sims = row.get("sims") if isinstance(row.get("sims"), list) else []
    times = [s["ts"] for s in sims if isinstance(s, dict) and isinstance(s.get("ts"), (int, float))]
    return float(min(times)) if times else None


def judge_event_file(lines: list[dict], j: Judgement, ctx: Ctx) -> set[str]:
    """An evidence event log, judged per ``log_run_id`` group. Returns the ids of the groups that COUNT (exactly one
    terminal ``ok``); a failed group is excluded, a mock line sinks the file."""
    kinds = {kind_of(r) for r in lines}
    if not kinds <= {"harness_event", "harness_run_end"}:
        j.fail(f"an event log holds other kinds {sorted(map(str, kinds - {'harness_event', 'harness_run_end'}))}")
        return set()
    if any(r.get("mock") is not False for r in lines):
        j.fail("a line is mock or does not say (a mock line sinks the whole file)")
        return set()
    if any(not isinstance(r.get("log_run_id"), str) or not r.get("log_run_id") for r in lines):
        j.fail("a line carries no string log_run_id")
        return set()
    groups: dict[str, list[dict]] = {}
    for r in lines:
        groups.setdefault(r["log_run_id"], []).append(r)
    counted: set[str] = set()
    trees, times = set(), []
    for gid, group in groups.items():
        terminals = [r for r in group if r.get("record_kind") == "harness_run_end"]
        if len(terminals) != 1 or terminals[0].get("status") != "ok":
            continue  # a failed / aborted / unterminated run: excluded, never counted
        blocks = [r["provenance"] for r in group if isinstance(r.get("provenance"), dict)]
        if not blocks or len({json.dumps(b, sort_keys=True) for b in blocks}) != 1:
            j.fail(f"group {gid[:8]}: not exactly one provenance block")
            continue
        block = blocks[0]
        if block.get("harness_family") != "in_process":
            j.fail(f"group {gid[:8]}: the block is not an in-process harness's")
        d = provenance_digest(block)
        if any(r.get("provenance_sha256") != d for r in group):
            j.fail(f"group {gid[:8]}: a line's provenance_sha256 does not match the block")
        tree = judge_provenance(block, j, ctx, f"group {gid[:8]}")
        if tree is not None and terminals[0].get("end_code_tree_sha256") != tree:
            j.fail(f"group {gid[:8]}: the run ended on another code tree")
        body = [r for r in group if r.get("record_kind") == "harness_event"]
        if not body:
            j.fail(f"group {gid[:8]}: a terminal with no events")
            continue
        counted.add(gid)
        trees.add(tree)
        group_times = [float(r["ts"]) for r in body if isinstance(r.get("ts"), (int, float))]
        if group_times:
            check_skew(min(group_times), block.get("executed_git_hash"), j, ctx, f"group {gid[:8]}")
        times += group_times
    if not counted:
        j.fail("no run group in the log ended ok")
    if len(trees) > 1:
        j.fail(f"the runs ran on {len(trees)} code trees (one per file)")
    j.time = min(times) if times else None
    return counted


def judge_non_support(rec: dict, j: Judgement, ctx: Ctx) -> None:
    kind = rec.get("record_kind")
    prov = rec.get("code_provenance") if kind == "diagnosis" else rec.get("provenance")
    judge_provenance(prov, j, ctx, f"{kind} provenance")
    if rec.get("mock") is not False:
        j.fail(f"{kind}: mock is {rec.get('mock')!r}")
    if kind == "instrument_check" and (rec.get("status") != "ok" or rec.get("pass") is not True):
        j.fail("instrument_check: did not end ok and pass at its frozen parameters")


def select_scope(scope, rows: list[dict]) -> list[dict] | None:
    """The rows a verdict's scope selects, or None when the scope is malformed (unknown keys, wrong types)."""
    if not isinstance(scope, dict) or not scope:
        return None
    if scope == {"all_rows": True}:
        return rows
    if set(scope) - {"run_ids", "campaign_id"}:
        return None
    run_ids, campaign = scope.get("run_ids"), scope.get("campaign_id")
    if run_ids is not None and (not isinstance(run_ids, list) or not all(isinstance(x, str) for x in run_ids)):
        return None
    if campaign is not None and not isinstance(campaign, str):
        return None
    return [
        r
        for r in rows
        if (run_ids is None or r.get("run_id") in run_ids) and (campaign is None or r.get("campaign_id") == campaign)
    ]


def judge_verdict(rec: dict, j: Judgement, ctx: Ctx) -> None:
    j.record = rec
    data_rel = rec.get("data")
    if not isinstance(data_rel, str) or not data_rel.startswith(DATA_ROOT + "/") or ".." in data_rel.split("/"):
        j.fail(f"verdict data {data_rel!r} is not a path under {DATA_ROOT}/")
        return
    j.data_prereg = ctx.prereg_status(data_rel)
    if j.data_prereg in BAD_PREREG:
        j.fail(f"verdict data {data_rel}: prereg status {j.data_prereg}")
    entry = ctx.repo.tree(ctx.ref).get(data_rel)
    if entry is None or entry[0] == "120000":
        j.fail(f"verdict data {data_rel} is not a tracked file (or is a symlink)")
        return
    data = ctx.repo.blob(ctx.ref, data_rel) or b""
    if sha256(data) != rec.get("data_sha256"):
        j.fail(f"verdict data {data_rel}: sha256 differs from data_sha256")
        return
    rows = json_lines(decompressed(data_rel, data))
    scoped = select_scope(rec.get("scope"), rows)
    if scoped is None:
        j.fail(f"verdict scope {rec.get('scope')!r} is malformed")
        return
    if rec.get("mock") is not False or any(r.get("mock") is not False for r in rows):
        j.fail("verdict over mock (or unmarked) rows")
        return
    kinds = {kind_of(r) for r in rows}
    sub = Judgement(path=data_rel, status=ESTABLISHED)
    if kinds <= {"harness_row", "harness_header"}:
        judge_row_file(rows, sub, ctx, scope_rows=scoped)
        units = counted_rows(scoped)
    elif kinds <= {"harness_event", "harness_run_end"}:
        counted = judge_event_file(rows, sub, ctx)
        units = [r for r in scoped if kind_of(r) == "harness_event" and str_field(r, "log_run_id") in counted]
    else:
        j.fail(f"verdict data {data_rel} holds unknown kinds")
        return
    j.reasons += [f"data: {r}" for r in sub.reasons]
    if sub.status != ESTABLISHED:
        j.status = NOT_ESTABLISHED
    if not units:
        j.fail("the verdict's scope selects no counted unit")
    # One code tree per scope follows from one per file: a verdict judges exactly one data file.
    known = [t for t in (unit_time(u) for u in units) if t is not None]
    j.time = min(known) if known else None
    j.allowed_dirty = sub.allowed_dirty
    judge_provenance(rec.get("provenance"), j, ctx, "verdict provenance", allow_dirty_ok=False)
    if str_field(rec, "kind") in O19_KINDS:
        judge_o19(rec, rows, data_rel, j, ctx)


def judge_o19(rec: dict, rows: list[dict], data_rel: str, j: Judgement, ctx: Ctx) -> None:
    """The O19 re-run verdicts: the measured session files re-hashed, and the merge-base judge re-run."""
    data_dir = data_rel.rsplit("/", 1)[0]
    tree = ctx.repo.tree(ctx.ref)
    for i, r in enumerate(rows, 1):
        if r.get("record_kind") != "harness_row" or r.get("status") != "ok":
            continue
        session, files = r.get("session_id"), r.get("files")
        if not isinstance(session, str) or not session or "/" in session or session in (".", ".."):
            j.fail(f"row {i}: session_id {session!r} is not one plain path component")
            continue
        if not isinstance(files, dict):
            j.fail(f"row {i}: files is not a mapping")
            continue
        for name, digest in files.items():
            if not isinstance(name, str) or "/" in name or name in (".", ".."):
                j.fail(f"row {i}: file name {name!r} is not plain")
                continue
            plain, packed = f"{data_dir}/{session}/{name}", f"{data_dir}/{session}/{name}.gz"
            present = [p for p in (plain, packed) if p in tree]
            if len(present) != 1:
                j.fail(f"row {i}: {session}/{name} is {'missing' if not present else 'present both plain and .gz'}")
                continue
            if tree[present[0]][0] == "120000":
                j.fail(f"row {i}: {present[0]} is a symlink")
                continue
            data = decompressed(present[0], ctx.repo.blob(ctx.ref, present[0]) or b"")
            if sha256(data) != digest:
                j.fail(f"row {i}: {present[0]} differs from its row's SHA-256")
    if rec.get("apparatus_checked") is not True:
        j.fail("O19 verdict: the apparatus (markers, ruleset, landing) was not checked")
        return
    bound = as_dict(rec.get("bound_files")).get(O19_JUDGE)
    source = ctx.repo.blob(ctx.base, O19_JUDGE)
    base_blob = ctx.repo.git("rev-parse", f"{ctx.base}:{O19_JUDGE}").strip() if source is not None else None
    if not bound or bound != base_blob or source is None or sha256(source) != rec.get("verdict_source_sha256"):
        j.fail("O19 verdict: the merge-base judge is not the one that wrote the verdict")
        return
    before = list(sys.path)
    try:  # the merge-base module's later calls (attempts_from_rows, judge) run under a restored sys.path too
        rejudged = rejudge_o19(source, rec, rows, data_dir, ctx)
    finally:
        sys.path[:] = before
    if isinstance(rejudged, str):
        j.fail(f"O19 verdict: re-judge refused: {rejudged}")
        return

    def projection(out: dict) -> list:
        attempts = out.get("attempts") if isinstance(out.get("attempts"), list) else []
        return [(as_dict(a).get("run_id"), as_dict(a).get("k"), as_dict(a).get("complete")) for a in attempts]

    if (rejudged.get("verdict"), rejudged.get("deciding_attempt"), projection(rejudged)) != (
        rec.get("verdict"),
        rec.get("deciding_attempt"),
        projection(rec),
    ):
        j.fail("O19 verdict: re-judging the bound bytes gives a different result")


def rejudge_o19(source: bytes, rec: dict, rows: list[dict], data_dir: str, ctx: Ctx):
    """Run the MERGE-BASE ``o19_verdict.judge`` on the bound bytes. Returns its output dict, or a refusal reason."""
    markers = as_dict(rec.get("apparatus")).get("markers")
    if not isinstance(markers, list) or not markers or not all(isinstance(m, dict) for m in markers):
        return "the verdict records no start markers"
    if not all(isinstance(m.get("k"), int) and isinstance(m.get("run_id"), str) for m in markers):
        return "a start marker has no integer k or string run_id"
    ordered_markers = sorted(markers, key=lambda m: m["k"])  # stable over the stamped order
    if [m["k"] for m in ordered_markers] != list(range(1, len(ordered_markers) + 1)):
        return "marker k values are not 1..n"
    with tempfile.TemporaryDirectory() as tmp:
        mod_path = Path(tmp) / "o19_verdict_mergebase.py"
        mod_path.write_bytes(source)
        before = list(sys.path)
        try:
            spec = importlib.util.spec_from_file_location("_o19_mergebase", mod_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as exc:  # noqa: BLE001 — a load failure is a refusal, never a crash
            return f"load: {type(exc).__name__}: {exc}"
        finally:
            sys.path[:] = before
        exp = {"exp10_verdict": "10", "exp09_verdict": "09"}[rec["kind"]]
        try:
            attempts = mod.attempts_from_rows(rows)
        except Exception as exc:  # noqa: BLE001 — the judge's own Refusal included
            return f"{type(exc).__name__}: {exc}"
        if set(attempts) - {m["run_id"] for m in ordered_markers}:
            return "rows name attempts with no start marker"
        ordered = [{"run_id": m["run_id"], "k": m["k"], "rows": attempts.get(m["run_id"], [])} for m in ordered_markers]
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            for path, (mode, _oid) in ctx.repo.tree(ctx.ref).items():
                if path.startswith(data_dir + "/") and mode not in ("120000", "160000"):
                    dest = root_path / path[len(data_dir) + 1 :]
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(ctx.repo.blob(ctx.ref, path) or b"")
            try:
                out = mod.judge(exp, ordered, root_path)
            except Exception as exc:  # noqa: BLE001 — the judge's own Refusal included
                return f"{type(exc).__name__}: {exc}"
            return out if isinstance(out, dict) else "the judge returned no result"


def unjudged(j: Judgement) -> bool:
    """The record could not be judged (a judge-code error, not bad bytes: a corrupt file is simply refused)."""
    return any(r.startswith(tuple(f"malformed: {e}:" for e in CODE_ERRORS)) for r in j.reasons)


def watched_paths(j: Judgement) -> list[str]:
    """What else a cited record rests on, whose change re-judges its row: a verdict's data file, and an O19
    verdict's whole data directory (its session files)."""
    data = j.record.get("data") if j.record else None
    if not isinstance(data, str):
        return []
    out = [data]
    if str_field(j.record, "kind") in O19_KINDS:
        out.append(data.rsplit("/", 1)[0] + "/")
    return out


def judge_entry(path: str, ctx: Ctx) -> Judgement:
    """Judge one cited path at ``ctx.ref``. A malformed record is a refusal, never an exception."""
    j = Judgement(path=path)
    try:
        _judge_entry(path, ctx, j)
    except (GateError, *MALFORMED) as exc:
        j.fail(f"malformed: {type(exc).__name__}: {exc}"[:300])
    return j


def _judge_entry(path: str, ctx: Ctx, j: Judgement) -> None:
    j.prereg = ctx.prereg_status(path)
    if j.prereg in BAD_PREREG:  # before anything else, legacy included
        j.fail(f"prereg status {j.prereg}")
        return
    tree = ctx.repo.tree(ctx.ref)
    if path in tree:
        data = ctx.repo.blob(ctx.ref, path) or b""
        if ctx.legacy.get(path) == sha256(data):
            j.status, j.kind = LEGACY, "legacy"
            return
        if tree[path][0] == "120000":
            j.fail("a symlink is not evidence")
            return
        body = decompressed(path, data)
        j.status = ESTABLISHED
        try:
            single = json.loads(body.decode("utf-8"))  # one JSON document; JSONL fails here and is read by lines
        except ValueError:
            single = None
        # A single-document record is one of these kinds; anything else (a one-line rows file or event log is valid
        # JSON too) is read line by line.
        if isinstance(single, dict) and kind_of(single) in SINGLE_DOCUMENT_KINDS:
            j.kind = single["record_kind"]
            if j.kind == "verdict":
                judge_verdict(single, j, ctx)
            else:
                judge_non_support(single, j, ctx)
            return
        lines = json_lines(body)
        kinds = {kind_of(r) for r in lines}
        if kinds and kinds <= {"harness_row", "harness_header"}:
            j.kind = "harness_row"
            judge_row_file(lines, j, ctx)
        elif kinds and kinds <= {"harness_event", "harness_run_end"}:
            j.kind = "harness_event"
            judge_event_file(lines, j, ctx)
        elif kinds == {"harness_demo"}:
            j.kind = "harness_demo"
            for r in lines:  # every line: a mock or unstamped one sinks the file
                judge_non_support(r, j, ctx)
        else:
            j.fail(f"lines of unknown or mixed kinds {sorted(map(str, kinds))}")
    elif f"{path}/report.json" in tree:
        j.kind, j.status = "sim_report", ESTABLISHED
        judge_sim_report(json.loads(ctx.repo.blob(ctx.ref, f"{path}/report.json") or b""), j, ctx)
    else:
        j.fail("not a tracked file or session directory")
