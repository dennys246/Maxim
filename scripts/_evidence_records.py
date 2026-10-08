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
from dataclasses import dataclass, field, replace
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _ledger as L  # noqa: E402
import _lint_git  # noqa: E402
from _provenance import provenance_digest  # noqa: E402

DATA_ROOT = L.DATA_ROOT
O19_JUDGE = "scripts/o19_verdict.py"
O19_RERUN = "scripts/o19_rerun.py"
# What the gate calls on a loaded judge (any version on main's history; every gate run loads every one of them,
# o19_history_problems). Every attribute _rejudge_with uses MUST be listed here, or that check cannot see a gate edit
# that an old judge fails.
# ``protocol_problems`` arrived with campaign succession: required only when the judge's table has a successor.
O19_INTERFACE = ("PROTOCOL", "rows_path", "attempts_from_rows", "judge", "MARKER_NAMESPACE", "MAX_ATTEMPTS")
O19_KINDS = frozenset({"exp10_verdict", "exp09_verdict", "exp63_verdict"})
# A run's finish reasons that are citable (pinned against simulation/sim_types.py's failure set by a test).
FINISH_OK = frozenset({"completed", "max_turns", "complete", "all_encounters_complete"})
FINISH_OK_PREFIX = "campaign_end:"
SKEW_S = 300  # a unit's ts may precede its own executed commit's committer time by at most this (clock skew)
HEX40 = re.compile(r"^[0-9a-f]{40}$")

ESTABLISHED, LEGACY, EXCEPTED, NOT_ESTABLISHED = "ESTABLISHED", "LEGACY", "EXCEPTED", "NOT-ESTABLISHED"
SUPPORT_KINDS = frozenset({"verdict"})  # only a stamped verdict supplies NEW support (owner decision 2026-09-30)
NON_SUPPORT_KINDS = frozenset({"instrument_check", "diagnosis", "harness_header", "harness_demo"})
SINGLE_DOCUMENT_KINDS = frozenset({"verdict", "instrument_check", "diagnosis"})
# The prereg lint's statuses a cited record may carry (an ALLOW-list, M1b 5b-2: anything else, an unknown or a new
# status included, refuses). New support additionally requires PASS.
PREREG_OK = frozenset({"PASS", "GRANDFATHERED", "UNGOVERNED_RERUN", "OUT_OF_SCOPE", "EXCEPTED"})
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

    def blob_id(self, ref: str, path: str) -> str | None:
        """The blob id at ``ref:path``; None when absent there or not a file."""
        if self.kind(ref, path) != "blob":
            return None
        return self.git("rev-parse", f"{ref}:{path}").strip()

    def first_parents(self, ref: str) -> frozenset[str]:
        """Every commit on ``ref``'s first-parent history (what landed on main, not a merged branch's commits)."""
        key = f"fp:{ref}"
        if key not in self._cache:
            self._cache[key] = frozenset(self.git("rev-list", "--first-parent", ref).split())
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
    # An O19 verdict re-judged by its bound judge: {"rejudged": its output, "protocol": its PROTOCOL, "harness_env":
    # its HARNESS_ENV, "model": its model pins} (JSON-plain), what the succession rules read (#1059). None until the re-judge agrees.
    o19: dict | None = None

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
    table: dict  # the MERGE-BASE pass table (complete-run rules read its `complete`): required, never defaulted

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
    if j.data_prereg not in PREREG_OK:
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
    j_counted: set[str] = set()
    if kinds <= {"harness_row", "harness_header"}:
        judge_row_file(rows, sub, ctx, scope_rows=scoped)
        units = counted_rows(scoped)
    elif kinds <= {"harness_event", "harness_run_end"}:
        counted = judge_event_file(rows, sub, ctx)
        j_counted = counted
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
    judge_complete(rec, rows, scoped, j_counted, j, ctx)
    if str_field(rec, "kind") in O19_KINDS:
        if j.path != data_rel.rsplit("/", 1)[0] + "/verdict.json":  # half B re-judges every one it can find (#1050)
            j.fail(f"O19 verdict: {j.path} is not its rows' verdict.json ({data_rel.rsplit('/', 1)[0]}/verdict.json)")
            return
        judge_o19(rec, rows, data_rel, j, ctx)


# ── complete-run rules (M1b PR 5b-2) ─────────────────────────────────────────────────────────────────────
# The gate re-derives a verdict's run STRUCTURE from the bound bytes; the verdict's VALUE is trusted from its
# stamped writer (owner decision 2026-10-01). Which rule a kind gets is code-owned; the frozen sets / pins it checks
# come from the MERGE-BASE pass table's `complete` block (a PR cannot loosen its own sets/pins; the rule code itself,
# like all gate code, is review's).
COMPLETE_RULES = {
    "exp53_verdict": "exp53_runs",
    "exp60_verdict": "seeds_per_run_arm",
    "exp61_verdict": "campaign_pairs",
    "exp62_verdict": "campaign_seeds",
}


def judge_complete(rec: dict, rows: list[dict], scoped: list[dict], counted: set[str], j: Judgement, ctx: Ctx):
    rule = COMPLETE_RULES.get(str_field(rec, "kind") or "")
    if rule is None:
        return
    spec = as_dict(ctx.table.get(rec["kind"])).get("complete")
    if not isinstance(spec, dict) or spec.get("rule") != rule:
        j.fail(f"complete-run: the merge-base pass table carries no `{rule}` rule for {rec['kind']}")
        return
    RULES[rule](rec, rows, scoped, counted, spec, j, ctx)


def _keyed(r: dict, seed_field: str, sets: dict) -> tuple[str, int] | None:
    arm, seed = str_field(r, "arm"), r.get(seed_field)
    if arm not in sets or type(seed) is not int or seed not in sets[arm]:
        return None
    return arm, seed


def _clean(r: dict) -> bool:
    return r.get("refusal") is None  # the analyzers' predicate: any non-null refusal (even "") is refused


def seeds_per_run_arm(rec, rows, scoped, counted, spec, j: Judgement, ctx: Ctx) -> None:
    """Exp 60: every run in the FILE whose (arm, seed) keys equal one arm's frozen seeds (refused rows included) is
    a complete run; exactly one per arm, and the scope is exactly those runs (no cherry-picking a later run)."""
    sets = spec["sets"]
    by_run: dict[str, list[tuple[str, int]]] = {}
    for i, r in enumerate(rows, 1):
        if kind_of(r) != "harness_row":
            continue
        rid, key = str_field(r, "run_id"), _keyed(r, "seed", sets)
        if rid is None or key is None:
            j.fail(f"complete-run: row {i} has no string run_id or an (arm, seed) outside the frozen sets")
            return
        by_run.setdefault(rid, []).append(key)
    complete: dict[str, list[str]] = {}
    for rid, keys in by_run.items():
        arms = {a for a, _ in keys}
        if len(keys) != len(set(keys)) or len(arms) != 1:
            j.fail(f"complete-run: run {rid} repeats an (arm, seed) or spans {len(arms)} arms")
            return
        arm = next(iter(arms))
        if {s for _, s in keys} == set(sets[arm]):
            complete.setdefault(arm, []).append(rid)
    for arm in sets:
        if len(complete.get(arm, [])) != 1:
            j.fail(
                f"complete-run: {len(complete.get(arm, []))} complete runs of arm {arm} in the file (need exactly 1)"
            )
    run_ids = as_dict(rec.get("scope")).get("run_ids")
    if run_ids is not None and (len(run_ids) != len(set(run_ids)) or set(run_ids) - set(by_run)):
        j.fail("complete-run: the scope repeats a run_id or names one with no rows")
    scoped_runs = {str_field(r, "run_id") for r in scoped if kind_of(r) == "harness_row"}
    if scoped_runs != {rid for rids in complete.values() for rid in rids}:
        j.fail("complete-run: the scope is not exactly the file's complete runs (one per arm)")


def _campaign_rule(seed_field: str, keyed_kind: str, allowed: frozenset, required: frozenset):
    def rule(rec, rows, scoped, counted, spec, j: Judgement, ctx: Ctx) -> None:
        """Exp 61 / 62: one campaign per FILE; the keyed rows' (arm, seed) equal the frozen sets exactly (refused rows
        present, never a duplicate CLEAN key); every row of an allowed kind; the required kinds present."""
        sets = spec["sets"]
        campaigns = {str_field(r, "campaign_id") or f"<{r.get('campaign_id')!r}>" for r in rows}
        if len(campaigns) != 1:
            j.fail(f"complete-run: the data file holds {len(campaigns)} campaigns (one per file)")
            return
        (campaign,) = campaigns
        scope = as_dict(rec.get("scope"))
        named = rec.get("campaign_id")
        # `verdict` with no --campaign-id writes campaign_id null over all_rows: the file's one campaign is proven above.
        if scope.get("campaign_id", campaign) != campaign or not (
            named == campaign or (named is None and "all_rows" in scope)
        ):
            j.fail("complete-run: the verdict's campaign_id / scope is not the file's campaign")
        present: dict[tuple[str, int], int] = {}
        kinds = set()
        for i, r in enumerate(rows, 1):
            kind = str_field(r, "kind")
            if kind not in allowed:
                j.fail(f"complete-run: row {i} kind {r.get('kind')!r} is not one of {sorted(allowed)}")
                return
            kinds.add(kind)
            if kind != keyed_kind:
                continue
            key = _keyed(r, seed_field, sets)
            if key is None:
                j.fail(f"complete-run: row {i} (arm, {seed_field}) is outside the frozen sets")
                return
            present.setdefault(key, 0)
            if _clean(r):
                present[key] += 1
        want = {(a, s) for a, seeds in sets.items() for s in seeds}
        if set(present) != want:
            j.fail(f"complete-run: {len(want - set(present))} frozen key(s) missing from the campaign")
        if any(n > 1 for n in present.values()):
            j.fail("complete-run: a frozen key has two clean rows")
        if required - kinds:
            j.fail(f"complete-run: the campaign has no {sorted(required - kinds)} row")

    return rule


campaign_pairs = _campaign_rule(
    "pair_seed", "receiver", frozenset({"receiver", "apparatus", "donor", "anti_vacuity"}), frozenset({"anti_vacuity"})
)
campaign_seeds = _campaign_rule(
    "seed", "row", frozenset({"row", "replay", "apparatus"}), frozenset({"replay", "apparatus"})
)


def exp53_runs(rec, rows, scoped, counted, spec, j: Judgement, ctx: Ctx) -> None:
    """Exp 53: one experiment and no debug run per file; exactly one gate-I phase-1 run and one complete phase-2
    primary run, both scoped and named by `runs_used`; each run pinned (start fields, agents to the merge-base
    manifest) and bound to one ok log group of its own."""
    events = [r for r in rows if kind_of(r) == "harness_event"]
    starts = [r for r in events if r.get("event") == "start"]
    pins = spec["start"]
    if any(r.get("only") is not None or r.get("experiment") != pins.get("experiment") for r in starts):
        j.fail("complete-run: the file holds a debug (--only) run or another experiment")
        return
    runs: dict[str, list[dict]] = {}
    for r in events:
        rid = str_field(r, "run_id")
        if rid is not None:
            runs.setdefault(rid, []).append(r)

    def first(rid: str, event: str) -> list[dict]:
        return [r for r in runs[rid] if r.get("event") == event]

    phase1 = [
        rid
        for rid in runs
        if any(not g.get("informative") for g in first(rid, "gate_I"))
        and [s.get("phase") for s in first(rid, "start")] == [1]
    ]
    primary = [
        rid
        for rid in runs
        if [s.get("phase") for s in first(rid, "start")] == [2]
        and [e.get("status") for e in first(rid, "run_end")] == ["complete"]
        and any(t.get("condition") == "primary" for t in first(rid, "trial"))
    ]
    if len(phase1) != 1 or len(primary) != 1:
        j.fail(
            f"complete-run: {len(phase1)} gate-I phase-1 run(s), {len(primary)} complete primary run(s) (need 1 each)"
        )
        return
    used = rec.get("runs_used")
    if not isinstance(used, dict) or set(used) != {"phase1", "primary", "secondary"}:
        j.fail("complete-run: runs_used is not {phase1, primary, secondary}")
        return
    secondary = used["secondary"]
    if (
        used["phase1"] != phase1[0]
        or used["primary"] != primary[0]
        or not (secondary is None or (isinstance(secondary, str) and secondary in runs))
    ):
        j.fail("complete-run: runs_used does not name the file's phase-1 and primary runs")
        return
    if secondary is not None and not (
        [s.get("phase") for s in first(secondary, "start")] == [2]
        and [e.get("status") for e in first(secondary, "run_end")] == ["complete"]
        and any(t.get("condition") == "secondary" for t in first(secondary, "trial"))
    ):
        j.fail("complete-run: runs_used.secondary is not a complete phase-2 run with secondary trials")
    named = {phase1[0], primary[0]} | ({secondary} if secondary else set())
    if set(as_dict(rec.get("scope")).get("run_ids") or []) != named:
        j.fail("complete-run: the scope is not exactly the runs runs_used names")
    if [g.get("verdict") for g in first(phase1[0], "gate_I")] != ["PASS"]:
        j.fail("complete-run: the phase-1 run's gate I is not PASS")
    if not first(phase1[0], "start")[0].get("ts", 0) < first(primary[0], "start")[0].get("ts", 0):
        j.fail("complete-run: phase 2 does not follow its phase-1 run")
    for rid in sorted(named):
        _exp53_run_pinned(rid, runs[rid], rows, counted, spec, j, ctx)


def _exp53_run_pinned(rid: str, lines: list[dict], rows: list[dict], counted: set[str], spec, j, ctx: Ctx) -> None:
    (start,) = [r for r in lines if r.get("event") == "start"] or [{}]
    for field_, want in spec["start"].items():
        if start.get(field_) != want or type(start.get(field_)) is not type(want):
            j.fail(f"complete-run: run {rid} start.{field_} is {start.get(field_)!r}, pinned {want!r}")
    if any(r.get("dry_run") is not False for r in lines):
        j.fail(f"complete-run: run {rid} has a dry-run line")
    groups = {str_field(r, "log_run_id") for r in lines}
    others = {str_field(r, "run_id") for r in rows if str_field(r, "log_run_id") in groups} - {None, rid}
    if len(groups) != 1 or not groups <= counted or others:
        j.fail(f"complete-run: run {rid} is not alone in exactly one log group that ended ok")
    raw = ctx.repo.blob(ctx.base, spec["manifest"])
    if raw is None:
        j.fail(f"complete-run: the manifest {spec['manifest']} is not at the merge-base")
        return
    manifest = json.loads(raw)
    entries = [e for e in as_dict(manifest).get("agents") or [] if isinstance(e, dict)]
    by_sha = {(e.get("nac_sha256"), e.get("ec_sha256")): e for e in entries}
    must = {e.get("label") for e in entries if not e.get("exploratory")}
    loads: dict[str, tuple] = {}
    per_condition: dict[object, list[str]] = {}
    for r in lines:
        if r.get("event") != "agent_load":
            continue
        e = by_sha.get((r.get("nac_sha256"), r.get("ec_sha256")))
        ident = (r.get("agent"), r.get("arm"), r.get("seed"), r.get("exploratory_agent"))
        if e is None or ident != (e.get("label"), e.get("arm"), e.get("seed"), e.get("exploratory")):
            j.fail(f"complete-run: run {rid} loads {r.get('agent')!r} unlike its merge-base manifest entry")
            return
        if loads.setdefault(r["agent"], ident[1:4:2]) != ident[1:4:2]:
            j.fail(f"complete-run: run {rid} loads {r['agent']!r} twice differently")
            return
        per_condition.setdefault(r.get("condition"), []).append(r["agent"])
    for cond, labels in per_condition.items():
        if len(labels) != len(set(labels)) or {a for a in labels if not loads[a][1]} != must:
            j.fail(f"complete-run: run {rid} ({cond or 'phase 1'}) does not load each manifest agent exactly once")
    if not per_condition:
        j.fail(f"complete-run: run {rid} loads no agent")
    for r in lines:
        if r.get("event") not in ("trial", "probe"):
            continue
        # An invalid placement is logged without arm / exploratory flag (the analyzer drops it): its agent must
        # still be one the run loaded.
        if r.get("agent") not in loads or (
            r.get("invalid") is not True and (r.get("arm"), r.get("exploratory_agent")) != loads[r["agent"]]
        ):
            j.fail(f"complete-run: run {rid} has a {r['event']} for {r.get('agent')!r} unlike its load")
            return


RULES = {
    "exp53_runs": exp53_runs,
    "seeds_per_run_arm": seeds_per_run_arm,
    "campaign_pairs": campaign_pairs,
    "campaign_seeds": campaign_seeds,
}


def judge_o19(rec: dict, rows: list[dict], data_rel: str, j: Judgement, ctx: Ctx) -> None:
    """The O19 re-run verdicts: the measured session files re-hashed, and the BOUND judge (the one bound to the
    verdict's data, #1050) re-run."""
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
    source = bound_judge(rec, data_rel, rows, j, ctx)
    if source is None:
        return
    capture: dict = {}
    rejudged = rejudge_o19(
        source, rec, rows, data_dir, ctx, bound_files=as_dict(rec.get("bound_files")), capture=capture
    )
    if isinstance(rejudged, str):
        j.fail(f"O19 verdict: re-judge refused: {rejudged}")
        return
    diff = o19_difference(rejudged, rec)
    if diff:
        j.fail(f"O19 verdict: re-judging with the bound judge gives a different {diff}")
        return
    try:
        j.o19 = {"rejudged": _plain(rejudged), **capture}
    except (TypeError, ValueError) as exc:  # a bound judge's output JSON cannot hold supports nothing (j.o19 None)
        j.reasons.append(f"O19 verdict: the bound judge's output is not plain JSON ({exc})")


def bound_judge(rec: dict, data_rel: str, rows: list[dict], j: Judgement, ctx: Ctx) -> bytes | None:
    """The judge that wrote the verdict, bound to its DATA (#1050): the verdict was written at ``verdict_commit``, a
    commit on the merge-base's first-parent history that is the verdict's own executed commit and holds the judged
    rows bytes; at it and at every commit an attempt ran on, each bound path is its bound blob. Returns the bound
    judge's bytes, or None after failing ``j`` (every miss is its own reason)."""
    n = len(j.reasons)
    bound = rec.get("bound_files")
    if not isinstance(bound, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in bound.items()):
        j.fail("O19 verdict: bound_files is not a path -> blob id mapping")
        return None
    if not isinstance(bound.get(O19_JUDGE), str) or not HEX40.match(bound[O19_JUDGE]):
        j.fail(f"O19 verdict: bound_files names no {O19_JUDGE} blob")
        return None
    vc = rec.get("verdict_commit")
    if not isinstance(vc, str) or not HEX40.match(vc):
        j.fail(f"O19 verdict: verdict_commit {vc!r} is not a full commit id")
        return None
    if vc not in ctx.repo.first_parents(ctx.base):
        j.fail(f"O19 verdict: verdict_commit {vc[:12]} is not on the merge-base's first-parent history")
    if vc != as_dict(rec.get("provenance")).get("executed_git_hash"):
        j.fail(f"O19 verdict: verdict_commit {vc[:12]} is not the verdict's own executed commit")
    # Every commit an attempt ran on: each row's (a row without one is refused by the rows rules) and each start
    # marker's peeled commit (an attempt that left no rows still ran there).
    markers = as_dict(rec.get("apparatus")).get("markers")
    marker_list = [m for m in (markers if isinstance(markers, list) else []) if isinstance(m, dict)]
    peeled = {m.get("run_id"): m.get("peeled") for m in marker_list}
    for m in marker_list:  # the writer stamps it on every marker; an attempt without rows is bound only through it
        if not isinstance(m.get("peeled"), str) or not HEX40.match(m["peeled"]):
            j.fail(f"O19 verdict: start marker {m.get('ref')!r} records no peeled commit")
    for r in rows:
        run_id = as_dict(r.get("provenance")).get("harness_run_id")
        ran = as_dict(r.get("provenance")).get("executed_git_hash")
        if run_id in peeled and ran != peeled[run_id]:
            j.fail(f"O19 verdict: attempt {str(run_id)[:12]} ran on {str(ran)[:12]}, not its marker's commit")
    executed = {as_dict(r.get("provenance")).get("executed_git_hash") for r in rows}
    executed |= set(peeled.values())
    executed.discard(None)
    commits = [("verdict_commit", vc)]
    for c in sorted(executed, key=str):
        if not isinstance(c, str) or not HEX40.match(c):
            j.fail(f"O19 verdict: executed commit {c!r} is not a full commit id")
        elif not ctx.repo.is_ancestor(c, ctx.base):
            j.fail(f"O19 verdict: executed commit {c[:12]} is not on main (an ancestor of the merge-base)")
        else:
            commits.append(("executed commit", c))
    for path, want in sorted(bound.items()):
        for label, c in commits:
            have = ctx.repo.blob_id(c, path)
            if have != want:
                j.fail(f"O19 verdict: bound {path} is {have and have[:12]} at {label} {c[:12]}, not {want[:12]}")
    data = ctx.repo.blob(vc, data_rel) if ctx.repo.kind(vc, data_rel) == "blob" else None
    if data is None or sha256(data) != rec.get("data_sha256"):
        j.fail(f"O19 verdict: {data_rel} at verdict_commit {vc[:12]} is not the data_sha256 bytes")
    source = ctx.repo.blob(vc, O19_JUDGE) if ctx.repo.kind(vc, O19_JUDGE) == "blob" else None
    if source is None or sha256(source) != rec.get("verdict_source_sha256"):
        j.fail("O19 verdict: verdict_source_sha256 is not the bound judge's SHA-256")
    return source if len(j.reasons) == n else None


def o19_difference(rejudged: dict, rec: dict) -> str | None:
    """The first field a re-judged O19 result disagrees with the record on (None = the same result)."""

    def projection(out: dict) -> list:
        attempts = out.get("attempts") if isinstance(out.get("attempts"), list) else []
        return [(as_dict(a).get("run_id"), as_dict(a).get("k"), as_dict(a).get("complete")) for a in attempts]

    def plain(value):  # the record went through JSON; the re-judged dict may hold tuples
        return json.loads(json.dumps(value, sort_keys=True))

    for name, get in (
        ("verdict", lambda o: o.get("verdict")),
        ("deciding_attempt", lambda o: o.get("deciding_attempt")),
        ("attempts", projection),
        ("gates", lambda o: plain(o.get("gates"))),
        # #1079 S1: the STRUCTURED within-campaign bar ([run_id, k, status]); absent on every pre-#1079 judge's output
        # and on the four pre-#1079 campaigns' (None == None). Its prose twin (within_campaign_leak_notes) may embed
        # host paths and exception text: the writer prints it and never persists it, and it is never compared.
        ("within_campaign_leaks", lambda o: plain(o.get("within_campaign_leaks"))),
    ):
        if get(rejudged) != get(rec):
            return name
    return None


def load_o19_judge(source: bytes):
    """Load an ``o19_verdict.py`` blob from a temporary file (``sys.path`` restored after). Returns the module, or a
    refusal reason when it does not load or lacks what the gate calls (``O19_INTERFACE``)."""
    with tempfile.TemporaryDirectory() as tmp:
        mod_path = Path(tmp) / "o19_verdict_bound.py"
        mod_path.write_bytes(source)
        before = list(sys.path)
        try:
            spec = importlib.util.spec_from_file_location("_o19_bound", mod_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as exc:  # noqa: BLE001 — a load failure is a refusal, never a crash
            return f"load: {type(exc).__name__}: {exc}"
        finally:
            sys.path[:] = before
    missing = [name for name in O19_INTERFACE if not hasattr(mod, name)]
    if missing:
        return f"the judge lacks {missing}"
    if not isinstance(mod.PROTOCOL, dict):
        return "the judge's PROTOCOL is not a campaign table"
    has_successor = any(isinstance(p, dict) and "supersedes" in p for p in mod.PROTOCOL.values())
    if has_successor and not callable(getattr(mod, "protocol_problems", None)):
        return "the judge's campaign table has a successor but no protocol_problems"
    return mod


def rejudge_o19(
    source: bytes,
    rec: dict,
    rows: list[dict],
    data_dir: str,
    ctx: Ctx,
    *,
    bound_files: dict | None = None,
    ref: str | None = None,
    capture: dict | None = None,
):
    """Run the given ``o19_verdict`` bytes' ``judge`` on the record's bytes (half A: the bound judge; half B: an
    edited HEAD judge). With ``bound_files``, they must be exactly the judge, the harness and the campaign's prereg
    as this judge's table names it. Returns the judge's output dict, or a refusal reason. ``capture`` receives the
    loaded judge's ``protocol`` and ``harness_env`` (JSON-plain; None when it has none), for the succession rules."""
    markers = as_dict(rec.get("apparatus")).get("markers")
    if not isinstance(markers, list) or not markers or not all(isinstance(m, dict) for m in markers):
        return "the verdict records no start markers"
    if not all(isinstance(m.get("k"), int) and isinstance(m.get("run_id"), str) for m in markers):
        return "a start marker has no integer k or string run_id"
    ordered_markers = sorted(markers, key=lambda m: m["k"])  # stable over the stamped order
    if [m["k"] for m in ordered_markers] != list(range(1, len(ordered_markers) + 1)):
        return "marker k values are not 1..n"
    before = list(sys.path)
    try:  # the module's later calls (attempts_from_rows, judge) run under a restored sys.path too
        mod = load_o19_judge(source)
        if isinstance(mod, str):
            return mod
        if capture is not None:
            try:
                capture["protocol"] = _plain(mod.PROTOCOL)
                capture["harness_env"] = _plain(getattr(mod, "HARNESS_ENV", None))
                capture["model"] = _model_of(mod)
            except (TypeError, ValueError) as exc:
                return f"the judge's campaign table is not plain JSON ({exc})"
        return _rejudge_with(mod, rec, rows, data_dir, ctx, ordered_markers, bound_files, ref or ctx.ref)
    finally:
        sys.path[:] = before


def _rejudge_with(mod, rec: dict, rows: list[dict], data_dir: str, ctx: Ctx, ordered_markers, bound_files, ref: str):
    # ``ref``: where the record's rows, session files and predecessor closure are read (half B: main's copy of a
    # landed record, so a PR cannot edit the judge and its inputs together).
    # The campaign the verdict judged (#1042 follow-up, campaign 2): the record's ``experiment`` names a key of
    # the judge's campaign table, of the record's kind, whose rows file and markers the record must be.
    exp = rec.get("experiment")
    protocol = mod.PROTOCOL
    if not isinstance(exp, str) or exp not in protocol or not isinstance(protocol[exp], dict):
        return f"the verdict's campaign {exp!r} is not in the judge's campaign table"
    if protocol[exp].get("kind") != rec.get("kind"):
        return f"campaign {exp} is not of kind {rec.get('kind')!r}"
    if bound_files is not None:
        want = {O19_JUDGE, O19_RERUN, protocol[exp].get("prereg")}
        if set(bound_files) != want:
            return f"bound_files names {sorted(bound_files)}, not exactly {sorted(map(str, want))}"
    if rec.get("data") != mod.rows_path(exp):
        return f"the verdict's data {rec.get('data')!r} is not campaign {exp}'s rows file {mod.rows_path(exp)!r}"
    if len(ordered_markers) > mod.MAX_ATTEMPTS:
        return f"{len(ordered_markers)} start markers: more than {mod.MAX_ATTEMPTS} attempts"
    for m in ordered_markers:
        if m.get("ref") != f"{mod.MARKER_NAMESPACE}/{exp}/attempt-{m['k']}-{m['run_id']}":
            return f"start marker {m.get('ref')!r} is not campaign {exp}'s attempt {m['k']}"
    if callable(getattr(mod, "protocol_problems", None)) and (problems := mod.protocol_problems()):
        return f"the judge's campaign table is unsound: {problems}"
    sup = protocol[exp].get("supersedes")
    if sup is not None:  # the closure's content; its timing against the markers is the verdict's check
        closure = ctx.repo.blob(ctx.base, sup["verdict"])  # one trust root with the predecessor data: main's copy
        if closure is None or sha256(closure) != sup["verdict_sha256"]:
            return f"campaign {exp}'s predecessor closure {sup['verdict']} is not the pinned verdict"
        if as_dict(as_dict(rec.get("apparatus")).get("succession")).get("key") != sup["key"]:
            return f"the verdict does not record campaign {exp}'s succession from {sup['key']}"
    try:
        attempts = mod.attempts_from_rows(rows)
    except Exception as exc:  # noqa: BLE001 — the judge's own Refusal included
        return f"{type(exc).__name__}: {exc}"
    if set(attempts) - {m["run_id"] for m in ordered_markers}:
        return "rows name attempts with no start marker"
    # #1079 (exec S1): every row is bound to ITS start marker by the fields the harness stamps on it (``attempt_k``,
    # ``marker``), so a record cannot drop an earlier marker and renumber the deciding attempt to k = 1 (which would
    # make the within-campaign trigger vacuous) without forging the rows too.
    by_run = {m["run_id"]: m for m in ordered_markers}
    for r in rows:
        m = by_run.get(as_dict(r.get("provenance")).get("harness_run_id"))
        if r.get("record_kind") != "harness_row" or m is None:
            continue
        k = r.get("attempt_k")
        # The marker clause is belt-and-braces: the ref is already forced to <ns>/<exp>/attempt-<k>-<run_id> above,
        # and the row's run id is the marker's, so the k comparison carries the binding.
        if isinstance(k, bool) or k != m["k"] or r.get("marker") != m.get("ref"):
            return (
                f"a row of attempt {str(m['run_id'])[:12]} carries attempt_k {k!r} and marker {r.get('marker')!r}, "
                f"not its start marker's ({m['k']}, {m.get('ref')!r})"
            )[:400]
    ordered = [{"run_id": m["run_id"], "k": m["k"], "rows": attempts.get(m["run_id"], [])} for m in ordered_markers]
    # The data root mirrors the data directory's parent: the campaign's own directory, and each campaign it succeeds
    # (a successor's judge reads its predecessors' committed phases beside its own, #1059). A judge that reads only
    # its own directory sees the same bytes as before.
    dirs = [data_dir]
    seen, cur = {exp}, protocol[exp].get("supersedes")
    while isinstance(cur, dict) and cur.get("key") in protocol and cur["key"] not in seen:
        seen.add(cur["key"])
        dirs.append(mod.rows_path(cur["key"]).rsplit("/", 1)[0])
        cur = protocol[cur["key"]].get("supersedes")
    # The campaign's own directory is read at ``ref``; every predecessor's at the MERGE-BASE (main's copy of a closed
    # campaign: a PR cannot edit a predecessor's rows to empty a leak), and the judge checks those bytes against the
    # pinned closure besides.
    with tempfile.TemporaryDirectory() as root:
        root_path = Path(root)
        (root_path / data_dir.rsplit("/", 1)[-1]).mkdir()
        for directory in dirs:
            at = ref if directory == data_dir else ctx.base
            for path, (mode, _oid) in ctx.repo.tree(at).items():
                if path.startswith(directory + "/") and mode not in ("120000", "160000"):
                    dest = root_path / directory.rsplit("/", 1)[-1] / path[len(directory) + 1 :]
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(ctx.repo.blob(at, path) or b"")
        try:
            out = mod.judge(exp, ordered, root_path / data_dir.rsplit("/", 1)[-1])
        except Exception as exc:  # noqa: BLE001 — the judge's own Refusal included
            return f"{type(exc).__name__}: {exc}"
        return out if isinstance(out, dict) else "the judge returned no result"


def _plain(value):
    """``value`` as JSON holds it (tuples read as lists), for comparing a judge's tables across versions."""
    return json.loads(json.dumps(value, sort_keys=True))


# ── #1059: a successor campaign supports REPRODUCED only on byte-identical subject code (strict gate) ─────────
# The subject (owner decision 2026-10-04): what the sim runs and reads. Compared as `git ls-tree` listings (mode, type,
# object id, path) at every executed commit of the campaign chain; literal pathspecs, so no rename, textconv or
# attribute magic applies. Installed library versions, llama.cpp, the model and encoder weights and `.python-version`
# lie outside it: a disclosed gap (mechanization backlog M28). scripts/exp44/capture_paired_prompts.py is not subject:
# the orchestrator loads it only under MAXIM_EXP44_CAPTURE_LOG, which no O19 sim gets (C4 pins its env).
SUBJECT_PATHS = (
    "src/maxim",
    "pyproject.toml",
    "uv.lock",
    "poetry.lock",
    "pdm.lock",
    "Pipfile.lock",
    "scenarios",
    "data",
)
SUBJECT_EXCLUDED = frozenset({"src/maxim/utils/function_length_baseline.json"})  # non-runtime lint data
RESUME_FLAG = "--resume-sim"  # the one argv pair that legitimately differs between campaigns (a session id)


MODEL_FIELDS = ("MODEL_PROFILE", "MODEL_PROFILE_STAMPED", "MODEL_GGUF", "N_CTX")  # set via `maxim config`: no argv/env


def _model_of(mod) -> dict:
    """A judge's model pins (absent ones as None): compared across a campaign chain like its phases (S6)."""
    return _plain({name: getattr(mod, name, None) for name in MODEL_FIELDS})


def base_o19_judge(ctx: Ctx):
    """The merge-base's ``o19_verdict.py``, loaded (cached per gate run); a refusal string when absent or unloadable."""
    key = f"o19-base:{ctx.base}"
    if key not in ctx.repo._cache:
        source = ctx.repo.blob(ctx.base, O19_JUDGE)
        before = list(sys.path)
        try:
            ctx.repo._cache[key] = "no judge at the merge-base" if source is None else load_o19_judge(source)
        finally:
            sys.path[:] = before
    return ctx.repo._cache[key]


def subject_listing(repo: Repo, commit: str) -> list[str] | None:
    """The subject's ``mode type oid\tpath`` entries at ``commit`` (exclusions dropped), or None when unreadable."""
    out = subprocess.run(
        ["git", "--literal-pathspecs", "ls-tree", "-r", "-z", "--full-tree", commit, "--", *SUBJECT_PATHS],
        cwd=repo.root,
        capture_output=True,
    )
    if out.returncode != 0:
        return None
    entries = out.stdout.decode("utf-8", errors="surrogateescape").split("\0")
    return sorted(e for e in entries if e and e.partition("\t")[2] not in SUBJECT_EXCLUDED)


def _argv_key(argv) -> str:
    """A recorded argv with only the ``--resume-sim <session>`` pair dropped, as comparable JSON."""
    if not isinstance(argv, list):
        return json.dumps(argv)
    out, i = [], 0
    while i < len(argv):
        if argv[i] == RESUME_FLAG:
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return json.dumps(out)


def _campaign_record(ctx: Ctx, path: str, pin: str | None) -> tuple[dict, list[dict]] | str:
    """A campaign's verdict (as main holds it, pinned by SHA-256 when ``pin``) and its data's rows, or a refusal."""
    raw = ctx.repo.blob(ctx.base, path) if ctx.repo.kind(ctx.base, path) == "blob" else None
    if raw is None or (pin is not None and sha256(raw) != pin):
        return f"{path} is not the pinned verdict on main"
    try:
        rec = json.loads(raw.decode("utf-8"))
        data_rel = rec.get("data")
        data = ctx.repo.blob(ctx.base, data_rel) if isinstance(data_rel, str) else None
        if data is None or sha256(data) != rec.get("data_sha256"):
            return f"{path}: its rows are not its data_sha256 bytes on main"
        return rec, json_lines(decompressed(data_rel, data))
    except (GateError, *MALFORMED) as exc:
        return f"{path} cannot be read ({type(exc).__name__})"


def _executed(rec: dict, rows: list[dict]) -> set:
    """Every commit a campaign's attempts ran on: its rows' executed commits and its markers' peeled commits (never
    the verdict writer's own ``provenance.executed_git_hash``, #1059 S1)."""
    found = {as_dict(r.get("provenance")).get("executed_git_hash") for r in rows if kind_of(r) == "harness_row"}
    markers = as_dict(rec.get("apparatus")).get("markers")
    found |= {as_dict(m).get("peeled") for m in (markers if isinstance(markers, list) else [])}
    found.discard(None)
    return found


def _closure_problems(
    ctx: Ctx, pred: str, crec: dict, crows: list[dict], key: str, entry: dict, base_protocol: dict
) -> str | dict:
    """Predecessor ``pred``'s pinned closure (``crec``, its rows ``crows``, both as main holds them) judged at the gate
    (#1078, #1081 item 5): established by ``judge_o19`` through ITS bound judge at the merge-base (verdict_commit on
    main's first-parent history, every bound blob the same there and at each executed commit, ``data_sha256``,
    ``verdict_source_sha256``, the markers bound to the rows, the session files re-hashed, and the bound judge's
    re-run equal to the record), a real ABORT of ``pred``, and of the cited campaign ``key``'s experiment and kind (by
    the merge-base table, never the record alone). Returns the bound judge's ``{protocol, harness_env, model}`` for
    S6, or a refusal. Never ``judge_entry``: its scope rule refuses every ABORT that counted no unit (campaign 10's
    real closure), which would bar every successor of a phase-0 abort."""
    jp = Judgement(path=f"{pred}'s closure")
    try:
        judge_o19(
            crec, crows, crec.get("data") if isinstance(crec.get("data"), str) else "", jp, replace(ctx, ref=ctx.base)
        )
    except (GateError, *MALFORMED) as exc:
        jp.fail(f"malformed: {type(exc).__name__}: {exc}")
    if jp.reasons or not isinstance(jp.o19, dict):
        reasons = list(dict.fromkeys(jp.reasons))[:3] or ["its bound judge gave no result"]
        return f"its closure is not established: {reasons}"[:500]
    pentry = as_dict(base_protocol.get(pred))
    if (pentry.get("experiment"), pentry.get("kind")) != (entry.get("experiment"), entry.get("kind")):
        return f"its entry in the merge-base table is not campaign {key}'s experiment and kind"
    if crec.get("verdict") != "ABORT":
        return (
            f"its closure verdict is {crec.get('verdict')!r}, not an ABORT (only an ABORT may be succeeded; PASS, FAIL"
            " and NOT SHOWN are terminal)"
        )
    if crec.get("experiment") != pred:
        return f"its closure names campaign {crec.get('experiment')!r}, not {pred}"
    if str_field(crec, "kind") != entry.get("kind"):
        return f"its closure is of kind {crec.get('kind')!r}, not {entry.get('kind')!r}"
    if crec.get("mock") is not False:
        return "its closure is a mock verdict (or does not say)"
    if any(not isinstance(r, dict) or r.get("mock") is not False for r in crows):
        return "its closure judged a mock row (or one that does not say)"
    cap = jp.o19
    return {"protocol": cap.get("protocol"), "harness_env": cap.get("harness_env"), "model": cap.get("model")}


def o19_within_campaign_problems(cap: dict, key) -> list[str]:
    """#1079 (owner decision 2026-10-08, REFUSE): why the cited verdict's DECIDING attempt may not supply support
    because an earlier attempt of the same campaign leaked a FAILED gate ([] = it may). Reads only the BOUND judge's
    re-run (``cap["rejudged"]``, equal to the record by ``o19_difference``), never the record's own field.

    - The trigger is structural (design pass S2): the deciding attempt's k is not 1. Markers are forced to k = 1..n
      (``rejudge_o19``) and an attempt after the deciding one refuses in the judge, so k == 1 means nothing earlier
      exists, committed or not. It is vacuous for every pre-#1079 campaign (10 and 10c2 decide nothing; 09 and 63
      decide at k = 1), so their old bound judges never need the field; the gate holds no key list.
    - Triggered, the bound judge must emit ``within_campaign_leaks`` (else the verdict supports nothing: a new key a
      later edit put in ``PRE_1079_KEYS`` fails closed here), and it must be empty.
    - D1 (a rowless earlier marker bars) is WRITER-enforced and gate-TRUSTED (design pass S3): the writer reads the
      markers from origin (``ls-remote``); the gate sees only the record's ``apparatus.markers``, but ``_rejudge_with``
      binds every row to its marker (``attempt_k``, ``marker``), so a record that drops a rowless marker and renumbers
      k is refused unless its rows are forged too: that residue is the forged-verdict class (reproduction.md section
      12), with a gate-side check of the rows history and the markers against fetched tags owed (#1168)."""
    rejudged = as_dict(cap.get("rejudged"))
    deciding = rejudged.get("deciding_attempt")
    if deciding is None:
        return []  # an ABORT supports nothing (the pass table refuses it first)
    attempts = rejudged.get("attempts") if isinstance(rejudged.get("attempts"), list) else []
    k = next((as_dict(a).get("k") for a in attempts if as_dict(a).get("run_id") == deciding), None)
    if k == 1 and not isinstance(k, bool):
        return []  # nothing came before the deciding attempt, so nothing can have leaked
    leaks = rejudged.get("within_campaign_leaks")
    if not isinstance(leaks, list):
        return [
            f"campaign {key}: its deciding attempt is attempt {k!r}, but its bound judge does not compute the "
            "within-campaign leaked-gate bar (#1079): the verdict supports nothing"
        ]
    if leaks:
        return [
            f"campaign {key}: a FAILED gate leaked into an earlier attempt of this campaign (#1079): {leaks[:3]}"[:400]
        ]
    return []


def o19_succession_problems(j: Judgement, token: str, ctx: Ctx) -> list[str]:
    """Why this O19 verdict may not support ``token`` under the campaign-succession rules (#1059; [] = it may).

    - S2: whether the campaign HAS a predecessor is read from the BOUND judge's table, whose entry for the campaign
      must equal the merge-base table's (never the record's own fields).
    - #1079: no earlier attempt of the cited campaign leaked a FAILED gate (:func:`o19_within_campaign_problems`),
      for a root or a successor campaign and every token.
    - A ROOT campaign's verdict never supports REPRODUCED; a SUCCESSOR's supports no positive token but REPRODUCED
      (owner decision 2026-10-02), and for every token it supports (PARTIAL included):
    - D2: its bound judge emits the leaked-gate bar, and it is empty (a bound judge without it: supports nothing);
    - S1/S4: every commit any campaign of the chain ran on (each predecessor's pinned closure verdict's rows and
      markers, the successor's own, aborted attempts included) exists, is on main, and holds one subject listing;
    - #1078: each predecessor's pinned closure is judged here (:func:`_closure_problems`: established through its
      own bound judge, a real ABORT of that campaign, of this campaign's experiment and kind), and it was on main at
      every commit the campaign pinning it ran on;
    - S6: the bound judges' phases, HARNESS_ENV and model pins (``MODEL_FIELDS``), and every recorded sim argv (less ``--resume-sim <id>``) and
      MAXIM_* env per phase, are the root's."""
    rec = j.record or {}
    if str_field(rec, "kind") not in O19_KINDS:
        return []
    cap = j.o19
    if not isinstance(cap, dict) or not isinstance(cap.get("protocol"), dict):
        return ["the O19 verdict was not re-judged by its bound judge, so its campaign cannot be placed"]
    key = rec.get("experiment")
    entry = cap["protocol"].get(key)
    base_mod = base_o19_judge(ctx)
    if isinstance(base_mod, str):
        return [f"the merge-base campaign table cannot be read: {base_mod}"]
    base_protocol = _plain(base_mod.PROTOCOL)
    if not isinstance(entry, dict) or base_protocol.get(key) != entry:
        return [f"campaign {key}'s entry in the merge-base campaign table is not its bound judge's (S2)"]
    if within := o19_within_campaign_problems(cap, key):  # #1079: root and successor campaigns, every token
        return within
    if not isinstance(entry.get("supersedes"), dict):
        if token == "REPRODUCED":
            return [f"campaign {key} is a root campaign: only a successor campaign's verdict supports REPRODUCED"]
        return []
    if token in L.POSITIVE and token != "REPRODUCED":
        return [f"campaign {key} is a successor campaign: its verdict supports REPRODUCED, never {token}"]
    leaked = as_dict(cap.get("rejudged")).get("leaked_gates")
    if not isinstance(leaked, list):
        return [
            f"campaign {key}'s bound judge does not compute the leaked-gate bar: a successor verdict supports nothing"
        ]
    if leaked:
        return [f"a FAILED gate leaked into a predecessor campaign's committed phases: {leaked[:3]}"[:400]]
    # The chain, from the merge-base table: (campaign, its verdict path, the pin) from the successor to the root.
    chain: list[tuple[str, str, str | None]] = [(key, j.path, None)]
    seen, cur = {key}, entry
    while isinstance(cur.get("supersedes"), dict):
        sup = cur["supersedes"]
        pred = sup.get("key")
        if pred in seen or not isinstance(base_protocol.get(pred), dict):
            return [f"campaign {key}'s succession chain is broken at {pred!r}"]
        seen.add(pred)
        chain.append((pred, sup.get("verdict"), sup.get("verdict_sha256")))
        cur = base_protocol[pred]
    problems: list[str] = []
    commits: dict[str, str] = {}  # commit -> the campaign that ran on it
    argv: dict[object, set[str]] = {}
    env: dict[object, set[str]] = {}
    tables: dict[str, tuple[str, str]] = {}  # campaign -> (its bound phases, its bound HARNESS_ENV)
    succ_executed: set = set()  # the commits the campaign that pins this one ran on
    chain_key = key  # that campaign
    for campaign, path, pin in chain:
        if campaign == key:
            rows = json_lines(decompressed(rec["data"], ctx.repo.blob(ctx.ref, rec["data"]) or b""))
            got: tuple[dict, list[dict]] | str = (rec, rows)
            bound = {"protocol": cap["protocol"], "harness_env": cap.get("harness_env"), "model": cap.get("model")}
        else:
            # The pinned closure must be the one in the campaign's own data directory: that is the directory the
            # freeze (o19_closed_data_problems), the history check and the leaked-gate bar read (#1059 delta review).
            own = f"{base_mod.rows_path(campaign).rsplit('/', 1)[0]}/verdict.json"
            if path != own:
                got = f"its pinned closure {path!r} is not its own data directory's {own!r}"
            else:
                got = _campaign_record(ctx, path, pin)
            history = closed_history_problem(ctx, campaign)
            if history and not isinstance(got, str):
                got = history
            bound = None
        if isinstance(got, str):
            problems.append(f"campaign {campaign}: {got}")
            succ_executed = set()
            continue
        crec, crows = got
        if bound is None:  # a predecessor: its closure judged through ITS bound judge, which is what IT ran (#1078)
            # S3 (design pass): the pinned closure was on main before the campaign that pins it ran: at every commit
            # that campaign ran on (each on main, checked below), the closure's path holds the pinned bytes.
            for c in sorted(c for c in succ_executed if isinstance(c, str) and HEX40.match(c)):
                at = ctx.repo.blob(c, path) if ctx.repo.kind(c, path) == "blob" else None
                if at is None or sha256(at) != pin:
                    problems.append(
                        f"campaign {campaign}: {path} is not the pinned closure at campaign {chain_key}'s executed "
                        f"commit {c[:12]}: the closure must reach main before its successor runs"
                    )
            closure = _closure_problems(ctx, campaign, crec, crows, key, entry, base_protocol)
            if isinstance(closure, str):
                problems.append(f"campaign {campaign}: {closure}")
                succ_executed = set()
                continue
            bound = closure
        phases = as_dict(bound["protocol"].get(campaign)).get("phases")
        tables[campaign] = (
            json.dumps(phases, sort_keys=True),
            json.dumps(bound["harness_env"], sort_keys=True),
            json.dumps(bound["model"], sort_keys=True),
        )
        executed = _executed(crec, crows)
        succ_executed, chain_key = executed, campaign
        if not executed:
            problems.append(f"campaign {campaign}: no executed commit is recorded")
        for c in executed:
            commits.setdefault(c, campaign)
        for r in crows:
            if kind_of(r) == "harness_row" and "sim_argv" in r:
                argv.setdefault(r.get("phase_index"), set()).add(_argv_key(r.get("sim_argv")))
                env.setdefault(r.get("phase_index"), set()).add(json.dumps(r.get("sim_env"), sort_keys=True))
    if problems:
        return problems
    root = chain[-1][0]
    for campaign, table in tables.items():
        if table != tables[root]:
            problems.append(
                f"campaign {campaign}'s bound phases, HARNESS_ENV or model pins are not root campaign {root}'s (S6)"
            )
    for index in sorted(set(argv) | set(env), key=str):
        if len(argv.get(index, set())) > 1 or len(env.get(index, set())) > 1:
            problems.append(f"phase {index}: the recorded sim argv or MAXIM_* env differs across the campaigns (S6)")
    listings: dict[str, list[str]] = {}
    for c, campaign in sorted(commits.items(), key=lambda kv: str(kv[0])):
        if not isinstance(c, str) or not HEX40.match(c):
            problems.append(f"campaign {campaign}: executed commit {c!r} is not a full commit id")
            continue
        # Defence in depth since #1078: the successor's own judge_o19 (bound_judge) and each predecessor's
        # _closure_problems refuse such a commit first, so these two branches are not reached from O19 input.
        exists = subprocess.run(["git", "cat-file", "-e", f"{c}^{{commit}}"], cwd=ctx.repo.root, capture_output=True)
        if exists.returncode != 0:
            problems.append(f"campaign {campaign}: executed commit {c[:12]} does not exist here")
            continue
        if not ctx.repo.is_ancestor(c, ctx.base):
            problems.append(f"campaign {campaign}: executed commit {c[:12]} is not on main")
            continue
        listing = subject_listing(ctx.repo, c)
        if listing is None:
            problems.append(f"campaign {campaign}: the subject at {c[:12]} cannot be listed")
            continue
        listings[c] = listing
    distinct = {json.dumps(v) for v in listings.values()}
    if len(distinct) > 1:
        first, *rest = sorted(listings.items())
        for c, listing in rest:
            if listing != first[1]:
                path = sorted(set(listing) ^ set(first[1]))[0].partition("\t")[2]
                problems.append(
                    f"the subject differs between executed commits {first[0][:12]} ({commits[first[0]]}) and "
                    f"{c[:12]} ({commits[c]}), first at {path}: REPRODUCED needs byte-identical subject code"
                )
                break
    return problems


def o19_table_problems(repo: Repo, base: str) -> list[str]:
    """A HEAD ``o19_verdict.py`` against the merge-base's (#1059): D1, each verdict kind belongs to one experiment
    and that map is append-only; S3, a campaign entry is FROZEN (kept, unedited) once its rows or verdict exist on
    main or another entry names it in ``supersedes``, so "has a predecessor" cannot be laundered by an edit; and an
    entry's ``supersedes`` object, once on main, is frozen itself (no re-pin to a rewritten closure); and #1077, the
    campaign count against the merge-base's cap (:func:`_o19_campaign_count_problems`)."""
    base_src, head_src = repo.blob(base, O19_JUDGE), repo.blob("HEAD", O19_JUDGE)
    if base_src is None or head_src is None:
        return []  # no table on one side: a deleted judge while verdicts exist fails half B
    before = list(sys.path)
    try:
        base_mod, head_mod = load_o19_judge(base_src), load_o19_judge(head_src)
    finally:
        sys.path[:] = before
    if isinstance(head_mod, str):
        return [f"{O19_JUDGE} at HEAD does not load through the gate: {head_mod}"]
    if isinstance(base_mod, str):
        return []  # the history check names a base judge that does not load
    base_p, head_p = _plain(base_mod.PROTOCOL), _plain(head_mod.PROTOCOL)
    out: list[str] = []

    def kinds(protocol: dict) -> dict[str, set]:
        found: dict[str, set] = {}
        for p in protocol.values():
            found.setdefault(str(as_dict(p).get("kind")), set()).add(as_dict(p).get("experiment"))
        return found

    head_kinds, base_kinds = kinds(head_p), kinds(base_p)
    for kind, exps in sorted(head_kinds.items()):
        if len(exps) != 1:
            out.append(f"{O19_JUDGE}: verdict kind {kind} belongs to {len(exps)} experiments (D1: exactly one)")
        elif kind in base_kinds and exps != base_kinds[kind]:
            out.append(f"{O19_JUDGE}: verdict kind {kind} moved to another experiment (D1: the map is append-only)")
    out += _o19_campaign_count_problems(base_mod, head_mod, head_p)
    named = {as_dict(as_dict(p).get("supersedes")).get("key") for table in (base_p, head_p) for p in table.values()}
    tree = repo.tree(base)
    for key, entry in sorted(base_p.items()):
        rows_rel = base_mod.rows_path(key)
        landed = rows_rel in tree or rows_rel.rsplit("/", 1)[0] + "/verdict.json" in tree
        if (landed or key in named) and head_p.get(key) != entry:
            out.append(
                f"{O19_JUDGE}: campaign {key} is frozen (its data is on main or a successor names it) and was "
                "removed or edited (S3: the campaign table is append-only)"
            )
        # #1059 delta review: a successor's pin chain is immutable once on main. Only the `supersedes` object is
        # frozen (a not-yet-run successor's prereg and phases may still be amended before data, as Exp 09's were).
        sup = as_dict(entry).get("supersedes")
        if sup is not None and as_dict(head_p.get(key)).get("supersedes") != sup:
            out.append(
                f"{O19_JUDGE}: campaign {key}'s `supersedes` (its predecessor and the pinned closure SHA-256) is on "
                "main and was removed or edited: a re-pin could point a successor at a rewritten closure (frozen)"
            )
    return out


def _cap_problem(mod, where: str) -> tuple[int | None, str | None]:
    """A judge's ``MAX_CAMPAIGNS`` as the cap, or why it is not one (an int, not a bool, >= 1). Absent at the
    merge-base means a pre-succession judge: a cap of 1. Read with ``getattr``: it is NOT in ``O19_INTERFACE``
    (the oldest judge on main's history has none)."""
    cap = getattr(mod, "MAX_CAMPAIGNS", None)
    if cap is None and where == "the merge-base":
        return 1, None
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
        return None, f"{O19_JUDGE} at {where}: MAX_CAMPAIGNS {cap!r} is not an integer >= 1"
    return cap, None


def _o19_campaign_count_problems(base_mod, head_mod, head_p: dict) -> list[str]:
    """#1077: the campaign cap is read from the MERGE-BASE judge, so one PR cannot raise it and add the campaign
    that uses it (a raise is its own PR; a later PR whose merge-base holds the raise may use it). HEAD's campaigns
    are counted per experiment and per chain root (design pass S1), and HEAD's own cap must be well-formed and hold
    HEAD's table (S2: a malformed or too-low cap cannot land and later refuse every judge edit). Continuity is
    gate-owned too (S1): every HEAD successor's experiment and kind are its predecessor's, never trusted to HEAD's
    ``protocol_problems``."""
    out: list[str] = []
    by_experiment: dict[str, list[str]] = {}
    by_root: dict[str, list[str]] = {}
    for key, entry in sorted(head_p.items()):
        entry = as_dict(entry)
        exp = entry.get("experiment")
        if not isinstance(exp, str):
            out.append(f"{O19_JUDGE}: campaign {key}'s experiment {exp!r} is not a string")
        else:
            by_experiment.setdefault(exp, []).append(key)
        root, seen = key, {key}
        while isinstance(as_dict(head_p.get(root)).get("supersedes"), dict):
            prev = head_p[root]["supersedes"].get("key")
            pentry = head_p.get(prev) if isinstance(prev, str) else None
            if not isinstance(pentry, dict) or prev in seen:
                out.append(
                    f"{O19_JUDGE}: campaign {root} supersedes {prev!r}, which is not a campaign in the table (or the chain cycles)"
                )
                break
            if (pentry.get("experiment"), pentry.get("kind")) != (
                head_p[root].get("experiment"),
                head_p[root].get("kind"),
            ):
                out.append(f"{O19_JUDGE}: campaign {root} supersedes {prev}, another experiment or kind (S1)")
            seen.add(prev)
            root = prev
        by_root.setdefault(root, []).append(key)
    out = sorted(set(out), key=out.index)
    base_cap, problem = _cap_problem(base_mod, "the merge-base")
    if problem:
        out.append(problem)
    head_cap, problem = _cap_problem(head_mod, "HEAD")
    if problem:
        out.append(problem)
    largest = max((len(v) for v in (*by_experiment.values(), *by_root.values())), default=0)
    if head_cap is not None and head_cap < largest:
        out.append(f"{O19_JUDGE} at HEAD: MAX_CAMPAIGNS {head_cap} is below its own table's {largest} campaigns")
    if base_cap is not None:
        for label, groups in (("experiment", by_experiment), ("the chain rooted at campaign", by_root)):
            for name, keys in sorted(groups.items()):
                if len(keys) > base_cap:
                    out.append(
                        f"{O19_JUDGE}: {label} {name} has {len(keys)} campaigns at HEAD ({keys}), more than the "
                        f"merge-base's MAX_CAMPAIGNS {base_cap} (#1077: raise the cap in its own PR first)"
                    )
    return out


def closed_campaign_dir(ctx: Ctx, key: str) -> str | None:
    """Campaign ``key``'s data directory when the merge-base holds its verdict (the campaign is CLOSED), else None."""
    base_mod = base_o19_judge(ctx)
    if isinstance(base_mod, str):
        return None
    directory = base_mod.rows_path(key).rsplit("/", 1)[0]
    return directory if ctx.repo.kind(ctx.base, f"{directory}/verdict.json") == "blob" else None


def o19_closed_data_problems(ctx: Ctx, changed: set[str]) -> list[str]:
    """#1059 delta review: a CLOSED campaign's data directory (its verdict on main) is immutable: any add, edit,
    delete or rename under it fails, with no exception path (the owner's strict stance on O19). Its rows and closure
    are what a successor's leaked-gate bar and subject check read."""
    base_mod = base_o19_judge(ctx)
    if isinstance(base_mod, str):
        return []
    out = []
    for key in sorted(base_mod.PROTOCOL):
        directory = closed_campaign_dir(ctx, key)
        hits = sorted(c for c in changed if directory and c.startswith(directory + "/"))
        if hits:
            out.append(
                f"{directory}: campaign {key} is closed (its verdict is on main), so its data directory is immutable;"
                f" this change touches {hits[:3]} (a successor's leaked-gate bar and subject check read these bytes)"
            )
    return out


def closed_history_problem(ctx: Ctx, key: str) -> str | None:
    """Why campaign ``key``'s data directory on main is not as its closure left it (None = untouched since): no
    first-parent commit after the one that added its ``verdict.json`` may touch the directory. Catches a rewrite that
    reached main before the immutability rule ran (or around it). Runs only when a successor verdict is newly cited
    (through ``support_problem``); rows already citing one are held by ``o19_closed_data_problems`` from then on.
    Needs git >= 2.31 for first-parent ``--diff-filter=A`` on merges; on older git ``added`` is empty and this
    refuses (fail closed)."""
    base_mod = base_o19_judge(ctx)
    if isinstance(base_mod, str):
        return f"campaign {key}: the merge-base campaign table cannot be read"
    directory = base_mod.rows_path(key).rsplit("/", 1)[0]
    touched = ctx.repo.git("log", "--first-parent", "--format=%H", ctx.base, "--", directory).split()
    added = ctx.repo.git(
        "log", "--first-parent", "--diff-filter=A", "--format=%H", ctx.base, "--", f"{directory}/verdict.json"
    ).split()
    if not added:
        return f"campaign {key}: its closure verdict never landed on main"
    after = touched[: touched.index(added[-1])] if added[-1] in touched else touched
    if after:
        return (
            f"campaign {key}: {directory} changed on main after its closure landed ({after[0][:12]}); a closed "
            "campaign's data is immutable, so this chain supports nothing"
        )
    return None


def o19_judge_edit_problems(ctx: Ctx) -> list[str]:
    """Half B of #1050: a diff touching ``scripts/o19_verdict.py`` re-judges EVERY O19 verdict in the HEAD tree with
    the HEAD script; any different result, refusal, or a deleted script while one exists fails (strict, owner
    decision 2026-10-03: no exception path; a judge change scopes new behaviour to new campaign keys)."""
    out: list[str] = []
    records: list[tuple[str, dict]] = []
    # Every verdict.json main holds or this diff adds: one deleted or renamed in the same diff is still re-judged.
    trees = {**ctx.repo.tree("HEAD"), **ctx.repo.tree(ctx.base)}
    for path, (mode, _oid) in sorted(trees.items()):
        if not path.startswith(DATA_ROOT + "/") or path.rsplit("/", 1)[-1] != "verdict.json":
            continue
        if mode in ("120000", "160000"):
            out.append(f"{path}: not a file")
            continue
        # The record as main holds it when it is there (a PR may not edit the judge and rewrite a landed verdict to
        # match); a verdict new in this diff is judged as HEAD holds it.
        ref = ctx.base if ctx.repo.kind(ctx.base, path) == "blob" else "HEAD"
        try:
            rec = json.loads((ctx.repo.blob(ref, path) or b"").decode("utf-8"))
        except (ValueError, UnicodeError):
            out.append(f"{path}: not JSON, so not known to be a non-O19 record")
            continue
        if isinstance(rec, dict) and str_field(rec, "kind") in O19_KINDS:
            records.append((path, ref, rec))
    source = ctx.repo.blob("HEAD", O19_JUDGE)
    if records and source is None:
        out.append(f"{O19_JUDGE} is deleted while O19 verdicts exist ({[p for p, _, _ in records]})")
        records = []
    for path, ref, rec in records:
        data_rel = rec.get("data")
        if not isinstance(data_rel, str) or not data_rel.startswith(DATA_ROOT + "/") or ".." in data_rel.split("/"):
            out.append(f"{path}: its data {data_rel!r} is not under {DATA_ROOT}/")
            continue
        try:
            rows = json_lines(decompressed(data_rel, ctx.repo.blob(ref, data_rel) or b""))
            rejudged = rejudge_o19(source, rec, rows, data_rel.rsplit("/", 1)[0], ctx, ref=ref)
            diff = None if isinstance(rejudged, str) else o19_difference(rejudged, rec)
        except (GateError, TypeError, *MALFORMED) as exc:
            out.append(f"{path}: {type(exc).__name__}: {exc}"[:300])
            continue
        if isinstance(rejudged, str):
            out.append(f"{path}: the edited judge refuses it: {rejudged}"[:300])
        elif diff:
            out.append(f"{path}: {diff} was {rec.get(diff)!r}, the edited judge gives {rejudged.get(diff)!r}"[:400])
    return [
        f"a judge edit changes an existing O19 verdict: {p} (strict, #1050: scope new judge behaviour to a new"
        " campaign key; there is no exception path)"
        for p in out
    ]


def o19_history_problems(repo: Repo, base: str) -> list[str]:
    """Every ``o19_verdict.py`` the merge-base's first-parent history ever held still loads through the gate with
    the interface it calls (#1050 N2): the gate re-judges a verdict with the judge that wrote it, so a GATE edit that
    an old judge cannot satisfy would strand that judge's verdicts. Needs the full history (the lint job has it)."""
    if repo.git("rev-parse", "--is-shallow-repository").strip() != "false":
        return [f"{O19_JUDGE}'s history cannot be checked in a shallow clone (#1050: fetch the full history)"]
    out = []
    for commit in repo.git("log", "--first-parent", "--format=%H", base, "--", O19_JUDGE).split():
        source = repo.blob(commit, O19_JUDGE)
        if source is None:
            continue  # the commit that deleted it (a deleted judge while verdicts exist fails half B)
        before = list(sys.path)
        try:
            mod = load_o19_judge(source)
        finally:
            sys.path[:] = before
        if isinstance(mod, str):
            out.append(f"{O19_JUDGE} at {commit[:12]} no longer loads through the gate: {mod}")
    return out


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
    if j.prereg not in PREREG_OK:  # before anything else, legacy included
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
