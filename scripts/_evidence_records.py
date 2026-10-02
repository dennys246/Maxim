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
        # The campaign the verdict judged (#1042 follow-up, campaign 2): the record's ``experiment`` names a key of
        # the merge-base campaign table, of the record's kind, whose rows file and markers the record must be.
        exp = rec.get("experiment")
        protocol = getattr(mod, "PROTOCOL", None)
        if not isinstance(exp, str) or not isinstance(protocol, dict) or exp not in protocol:
            return f"the verdict's campaign {exp!r} is not in the merge-base campaign table"
        if protocol[exp].get("kind") != rec.get("kind"):
            return f"campaign {exp} is not of kind {rec.get('kind')!r}"
        if rec.get("data") != mod.rows_path(exp):
            return f"the verdict's data {rec.get('data')!r} is not campaign {exp}'s rows file {mod.rows_path(exp)!r}"
        if len(ordered_markers) > mod.MAX_ATTEMPTS:
            return f"{len(ordered_markers)} start markers: more than {mod.MAX_ATTEMPTS} attempts"
        for m in ordered_markers:
            if m.get("ref") != f"{mod.MARKER_NAMESPACE}/{exp}/attempt-{m['k']}-{m['run_id']}":
                return f"start marker {m.get('ref')!r} is not campaign {exp}'s attempt {m['k']}"
        if problems := mod.protocol_problems():
            return f"the merge-base campaign table is unsound: {problems}"
        sup = protocol[exp].get("supersedes")
        if sup is not None:  # the closure's content; its timing against the markers is the verdict's check
            closure = ctx.repo.blob(ctx.ref, sup["verdict"])
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
