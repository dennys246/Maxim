#!/usr/bin/env python3
"""The ledger evidence gate (M1b PR 5b-1; spec: docs/plans/m1b_ledger_evidence_gate.md, "PR 5b build spec").

A ledger row moving to (or staying at) a positive status must rest on evidence that is ESTABLISHED: stamped, not a
typed abort, not mock, the code on main, one known code tree. A raise, a move between positive tokens or a date
change additionally needs NEW support: a newly cited, stamped VERDICT that the merge-base pass table
(``docs/experiments/evidence_pass_table.json``) lets support this row at its new token. Records committed before
M1a are LEGACY (``docs/experiments/evidence_legacy.json``): judged as such, never new support. Owner-named
overrides live in ``docs/experiments/evidence_exceptions.json``, read from the merge-base, append-only.

Diff-scoped against the merge-base with origin/main. It reads committed bytes (git objects at HEAD); a cited path
or the ledger with uncommitted changes fails. It catches forgetting, not evasion: an author can still cite a clean
but irrelevant record (review is the check).

    python scripts/lint_evidence_gate.py              # the gate (CI lint job)
    python scripts/lint_evidence_gate.py --json       # machine-readable
    python scripts/lint_evidence_gate.py --write-legacy   # (re)generate the legacy snapshot from the rules

Exits: 0 clean; 1 violations; 2 the base could not be read on a pull request.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
REPO_ROOT = SCRIPTS_DIR.parent

import _ledger as L  # noqa: E402
import _lint_git  # noqa: E402

DATA_ROOT = L.DATA_ROOT
PASS_TABLE = "docs/experiments/evidence_pass_table.json"
LEGACY_SNAPSHOT = "docs/experiments/evidence_legacy.json"
EXCEPTIONS = "docs/experiments/evidence_exceptions.json"
O19_JUDGE = "scripts/o19_verdict.py"
O19_KINDS = frozenset({"exp10_verdict", "exp09_verdict"})

# M1a (#999) merged at this commit time: a record first committed before it, carrying no `record_kind`, is legacy.
M1A_CUTOFF = 1790721676  # 2026-09-29T22:41:16Z, `git show -s --format=%ct fc9f19d0`
# A run's finish reasons that are citable (pinned against simulation/sim_types.py's failure set by a test).
FINISH_OK = frozenset({"completed", "max_turns", "complete", "all_encounters_complete"})
FINISH_OK_PREFIX = "campaign_end:"
SKEW_S = 300  # a record's ts may precede its executed commit's committer time by at most this (clock skew)
HEX40 = re.compile(r"^[0-9a-f]{40}$")

ESTABLISHED, LEGACY, EXCEPTED, NOT_ESTABLISHED = "ESTABLISHED", "LEGACY", "EXCEPTED", "NOT-ESTABLISHED"
SUPPORT_KINDS = frozenset({"verdict"})  # only a stamped verdict supplies NEW support (owner decision 2026-09-30)
NON_SUPPORT_KINDS = frozenset({"instrument_check", "diagnosis", "harness_header", "harness_demo"})
SINGLE_DOCUMENT_KINDS = frozenset({"verdict", "instrument_check", "diagnosis"})


class GateError(Exception):
    """The gate could not read something it needs (a git object, a JSON file): the row fails, never passes."""


# ── git access (committed bytes only) ─────────────────────────────────────────────────────────────────────


class Repo:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._blobs: dict[tuple[str, str], bytes | None] = {}
        self._trees: dict[str, dict[str, tuple[str, str]]] = {}

    def git(self, *args: str) -> str:
        return _lint_git.git(self.root, *args)

    def blob(self, ref: str, path: str) -> bytes | None:
        key = (ref, path)
        if key not in self._blobs:
            out = subprocess.run(["git", "cat-file", "blob", f"{ref}:{path}"], cwd=self.root, capture_output=True)
            self._blobs[key] = out.stdout if out.returncode == 0 else None
        return self._blobs[key]

    def tree(self, ref: str) -> dict[str, tuple[str, str]]:
        """``path -> (mode, blob id)`` for every file under the data root and scripts/ at ``ref``."""
        if ref not in self._trees:
            out = self.git("ls-tree", "-r", ref, "--", DATA_ROOT, "scripts")
            files: dict[str, tuple[str, str]] = {}
            for line in out.splitlines():
                meta, _, path = line.partition("\t")
                mode, _type, oid = meta.split()
                files[path] = (mode, oid)
            self._trees[ref] = files
        return self._trees[ref]

    def is_ancestor(self, commit: str, of: str) -> bool:
        out = subprocess.run(["git", "merge-base", "--is-ancestor", commit, of], cwd=self.root, capture_output=True)
        return out.returncode == 0

    def commit_time(self, ref: str) -> int:
        return int(self.git("show", "-s", "--format=%ct", ref).strip())

    def first_commit_time(self, path: str, ref: str) -> int | None:
        """When ``path`` was first added under the data root on ``ref``'s history (one history pass, cached)."""
        key = f"__added__{ref}"
        if key not in self._trees:
            added: dict[str, int] = {}
            when = 0
            for line in self.git(
                "log", "--diff-filter=A", "--name-only", "--format=@%ct", ref, "--", DATA_ROOT
            ).splitlines():
                if line.startswith("@"):
                    when = int(line[1:])
                elif line.strip():
                    added[line.strip()] = when  # log runs newest first: the last write is the oldest add
            self._trees[key] = added  # type: ignore[assignment]
        return self._trees[key].get(path)  # type: ignore[union-attr]

    def changed(self, base: str) -> set[str]:
        return set(self.git("diff", "--name-only", base, "HEAD").split("\n")) - {""}

    def dirty(self, paths: list[str]) -> list[str]:
        if not paths:
            return []
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


def unknown(value) -> bool:
    return not isinstance(value, str) or not value or value == "unknown" or value.startswith("unknown:")


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


def finish_ok(reason) -> bool:
    return isinstance(reason, str) and (reason in FINISH_OK or reason.startswith(FINISH_OK_PREFIX))


def judge_sim_fields(sim: dict, j: Judgement, ctx: Ctx, label: str, tree: str | None) -> None:
    """A sim report's (or a harness row's echoed) evidence fields."""
    if not finish_ok(sim.get("finish_reason")):
        j.fail(f"{label}: finish_reason {sim.get('finish_reason')!r} is not citable")
    if sim.get("code_changed_during_run") is not False:
        j.fail(f"{label}: code_changed_during_run is {sim.get('code_changed_during_run')!r}")
    if sim.get("working_tree_dirty_src_scripts") is not False:
        j.fail(f"{label}: the sim ran on a dirty tree")
    start, end = sim.get("code_tree_sha256"), sim.get("end_code_tree_sha256")
    if unknown(start) or start != end or (tree is not None and start != tree):
        j.fail(f"{label}: code tree {start!r} / end {end!r} / harness {tree!r} are not one known tree")
    if sim.get("configured_n_ctx") is None or not any(k.endswith("_profile") and sim.get(k) for k in sim):
        j.fail(f"{label}: no stamped model profile and context")
    resume = sim.get("resume")
    if resume is not None and not (isinstance(resume, dict) and resume.get("resume_loaded") is True):
        j.fail(f"{label}: resumed, but resume_loaded is not true")
    executed = sim.get("executed_git_hash")
    if not isinstance(executed, str) or not HEX40.match(executed) or not ctx.repo.is_ancestor(executed, ctx.base):
        j.fail(f"{label}: executed {executed!r} is not a full commit on main")


def judge_sim_report(report: dict, j: Judgement, ctx: Ctx) -> None:
    if report.get("record_kind") != "sim_report":
        j.fail("report.json is not a stamped sim_report")
        return
    prov = report.get("provenance") or {}
    flat = {**prov, "finish_reason": report.get("finish_reason")}
    judge_sim_fields(flat, j, ctx, "sim_report", None)
    if not isinstance(report.get("ts"), (int, float)):
        j.fail("sim_report has no ts")
    else:
        j.time = float(report["ts"])


def counted_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("record_kind") == "harness_row" and r.get("status") != "failed"]


def judge_row_file(rows: list[dict], j: Judgement, ctx: Ctx, *, scope_rows: list[dict] | None = None) -> None:
    """A rows file: only harness rows and headers; no mock; each non-failed row established; one tree."""
    kinds = {r.get("record_kind") for r in rows}
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
        if (r.get("provenance") or {}).get("harness_family") == "spawning":
            sims = r.get("sims") or []
            if not sims:
                j.fail(f"row {i}: a spawning harness row names no sims")
            for s, sim in enumerate(sims):
                if isinstance(sim, dict):
                    judge_sim_fields(sim, j, ctx, f"row {i} sim {s}", tree)
                else:
                    j.fail(f"row {i} sim {s}: not an object")
    if len(trees) > 1:
        j.fail(f"the rows ran on {len(trees)} code trees (one per file)")
    units = counted_rows(scope_rows if scope_rows is not None else rows)
    times = [unit_time(r) for r in units]
    known = [t for t in times if t is not None]
    j.time = min(known) if known else None


def unit_time(row: dict) -> float | None:
    if isinstance(row.get("ts"), (int, float)):
        return float(row["ts"])
    sims = [s.get("ts") for s in row.get("sims") or [] if isinstance(s, dict) and isinstance(s.get("ts"), (int, float))]
    return float(min(sims)) if sims else None


def judge_event_file(lines: list[dict], j: Judgement, ctx: Ctx) -> None:
    """An evidence event log: judged per ``log_run_id`` group; a failed group is excluded, a mock line sinks it."""
    kinds = {r.get("record_kind") for r in lines}
    if not kinds <= {"harness_event", "harness_run_end"}:
        j.fail(f"an event log holds other kinds {sorted(map(str, kinds - {'harness_event', 'harness_run_end'}))}")
        return
    if any(r.get("mock") is not False for r in lines):
        j.fail("a line is mock or does not say (a mock line sinks the whole file)")
        return
    if any(not r.get("log_run_id") for r in lines):
        j.fail("a line carries no log_run_id")
        return
    import importlib  # noqa: PLC0415

    sys.path.insert(0, str(SCRIPTS_DIR / "orient_backbone"))
    try:
        digest = importlib.import_module("live_common").provenance_digest
    finally:
        sys.path.pop(0)
    groups: dict[str, list[dict]] = {}
    for r in lines:
        groups.setdefault(r["log_run_id"], []).append(r)
    counted, trees, times = 0, set(), []
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
        d = digest(block)
        if any(r.get("provenance_sha256") != d for r in group):
            j.fail(f"group {gid[:8]}: a line's provenance_sha256 does not match the block")
        tree = judge_provenance(block, j, ctx, f"group {gid[:8]}")
        if tree is not None and terminals[0].get("end_code_tree_sha256") != tree:
            j.fail(f"group {gid[:8]}: the run ended on another code tree")
        body = [r for r in group if r.get("record_kind") == "harness_event"]
        if not body:
            j.fail(f"group {gid[:8]}: a terminal with no events")
            continue
        counted += 1
        trees.add(tree)
        times += [float(r["ts"]) for r in body if isinstance(r.get("ts"), (int, float))]
    if counted == 0:
        j.fail("no run group in the log ended ok")
    if len(trees) > 1:
        j.fail(f"the runs ran on {len(trees)} code trees (one per file)")
    j.time = min(times) if times else None


def judge_non_support(rec: dict, j: Judgement, ctx: Ctx) -> None:
    kind = rec.get("record_kind")
    prov = rec.get("code_provenance") if kind == "diagnosis" else rec.get("provenance")
    judge_provenance(prov, j, ctx, f"{kind} provenance")
    if rec.get("mock") is not False:
        j.fail(f"{kind}: mock is {rec.get('mock')!r}")
    if kind == "instrument_check" and (rec.get("status") != "ok" or rec.get("pass") is not True):
        j.fail("instrument_check: did not end ok and pass at its frozen parameters")


def judge_verdict(rec: dict, j: Judgement, ctx: Ctx) -> None:
    j.record = rec
    data_rel = rec.get("data")
    if not isinstance(data_rel, str) or data_rel.startswith("/") or ".." in data_rel.split("/"):
        j.fail(f"verdict data {data_rel!r} is not a repo-relative path")
        return
    entry = ctx.repo.tree(ctx.ref).get(data_rel)
    if entry is None or entry[0] == "120000":
        j.fail(f"verdict data {data_rel} is not a tracked file (or is a symlink)")
        return
    data = ctx.repo.blob(ctx.ref, data_rel) or b""
    if sha256(data) != rec.get("data_sha256"):
        j.fail(f"verdict data {data_rel}: sha256 differs from data_sha256")
        return
    try:
        rows = json_lines(decompressed(data_rel, data))
    except (GateError, OSError, UnicodeDecodeError) as exc:
        j.fail(f"verdict data {data_rel}: {exc}")
        return
    scope = rec.get("scope")
    if not isinstance(scope, dict) or not scope:
        j.fail("verdict has no scope")
        return
    if scope == {"all_rows": True}:
        scoped = rows
    else:
        run_ids, campaign = scope.get("run_ids"), scope.get("campaign_id")
        scoped = [
            r
            for r in rows
            if (run_ids is None or r.get("run_id") in run_ids)
            and (campaign is None or r.get("campaign_id") == campaign)
        ]
    if rec.get("mock") is not False or any(r.get("mock") is not False for r in rows):
        j.fail("verdict over mock (or unmarked) rows")
        return
    kinds = {r.get("record_kind") for r in rows}
    sub = Judgement(path=data_rel)
    sub.status = ESTABLISHED
    if kinds <= {"harness_row", "harness_header"}:
        judge_row_file(rows, sub, ctx, scope_rows=scoped)
        units = counted_rows(scoped)
    elif kinds <= {"harness_event", "harness_run_end"}:
        judge_event_file(rows, sub, ctx)
        units = scoped
    else:
        j.fail(f"verdict data {data_rel} holds unknown kinds")
        return
    j.reasons += [f"data: {r}" for r in sub.reasons]
    if sub.status != ESTABLISHED:
        j.status = NOT_ESTABLISHED
    if not units:
        j.fail("the verdict's scope selects no counted unit")
    trees = {
        (u.get("provenance") or {}).get("code_tree_sha256") for u in units if isinstance(u.get("provenance"), dict)
    }
    if len(trees) > 1:
        j.fail(f"the verdict's scope spans {len(trees)} code trees")
    j.time, j.allowed_dirty = sub.time, sub.allowed_dirty
    judge_provenance(rec.get("provenance"), j, ctx, "verdict provenance", allow_dirty_ok=False)
    if rec.get("kind") in O19_KINDS:
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
    bound = (rec.get("bound_files") or {}).get(O19_JUDGE)
    base_blob = (
        ctx.repo.git("rev-parse", f"{ctx.base}:{O19_JUDGE}").strip() if ctx.repo.blob(ctx.base, O19_JUDGE) else None
    )
    source = ctx.repo.blob(ctx.base, O19_JUDGE)
    if not bound or bound != base_blob or source is None or sha256(source) != rec.get("verdict_source_sha256"):
        j.fail("O19 verdict: the merge-base judge is not the one that wrote the verdict")
        return
    rejudged = rejudge_o19(source, rec, rows, data_dir, ctx)
    if isinstance(rejudged, str):
        j.fail(f"O19 verdict: re-judge refused: {rejudged}")
        return

    def projection(out: dict) -> list:
        return [(a.get("run_id"), a.get("k"), a.get("complete")) for a in out.get("attempts") or []]

    if (rejudged.get("verdict"), rejudged.get("deciding_attempt"), projection(rejudged)) != (
        rec.get("verdict"),
        rec.get("deciding_attempt"),
        projection(rec),
    ):
        j.fail("O19 verdict: re-judging the bound bytes gives a different result")


def rejudge_o19(source: bytes, rec: dict, rows: list[dict], data_dir: str, ctx: Ctx):
    """Run the MERGE-BASE ``o19_verdict.judge`` on the bound bytes. Returns its output dict, or a reason."""
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
        markers = (rec.get("apparatus") or {}).get("markers")
        if not isinstance(markers, list) or not markers:
            return "the verdict records no start markers"
        try:
            attempts = mod.attempts_from_rows(rows)
        except Exception as exc:  # noqa: BLE001
            return f"{type(exc).__name__}: {exc}"
        marked = [m.get("run_id") for m in sorted(markers, key=lambda m: m.get("k", 0))]
        if set(attempts) - set(marked):
            return "rows name attempts with no start marker"
        ordered = [{"run_id": rid, "k": i + 1, "rows": attempts.get(rid, [])} for i, rid in enumerate(marked)]
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            for path, (mode, _oid) in ctx.repo.tree(ctx.ref).items():
                if path.startswith(data_dir + "/") and mode != "120000":
                    dest = root_path / path[len(data_dir) + 1 :]
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(ctx.repo.blob(ctx.ref, path) or b"")
            try:
                return mod.judge(exp, ordered, root_path)
            except Exception as exc:  # noqa: BLE001 — the judge's own Refusal included
                return f"{type(exc).__name__}: {exc}"


def judge_entry(path: str, ctx: Ctx) -> Judgement:
    """Judge one cited path at ``ctx.ref``."""
    j = Judgement(path=path)
    j.prereg = ctx.prereg_status(path)
    tree = ctx.repo.tree(ctx.ref)
    if path in tree:
        data = ctx.repo.blob(ctx.ref, path) or b""
        if ctx.legacy.get(path) == sha256(data):
            j.status, j.kind = LEGACY, "legacy"
            return j
        if tree[path][0] == "120000":
            return j.fail("a symlink is not evidence")
        try:
            body = decompressed(path, data)
        except OSError as exc:
            return j.fail(f"cannot decompress: {exc}")
        j.status = ESTABLISHED
        try:
            single = json.loads(body.decode("utf-8"))  # one JSON document; JSONL fails here and is read by lines
        except (UnicodeDecodeError, ValueError):
            single = None
        # A single-document record is one of these kinds; anything else (a one-line rows file or event log is valid
        # JSON too) is read line by line.
        if isinstance(single, dict) and single.get("record_kind") in SINGLE_DOCUMENT_KINDS:
            j.kind = single["record_kind"]
            if j.kind == "verdict":
                judge_verdict(single, j, ctx)
            else:
                judge_non_support(single, j, ctx)
        else:
            try:
                lines = json_lines(body)
            except (GateError, UnicodeDecodeError) as exc:
                return j.fail(str(exc))
            kinds = {r.get("record_kind") for r in lines}
            if kinds and kinds <= {"harness_row", "harness_header"}:
                j.kind = "harness_row"
                judge_row_file(lines, j, ctx)
            elif kinds and kinds <= {"harness_event", "harness_run_end"}:
                j.kind = "harness_event"
                judge_event_file(lines, j, ctx)
            elif kinds == {"harness_demo"}:
                j.kind = "harness_demo"
                for r in lines[:1]:
                    judge_non_support(r, j, ctx)
            else:
                j.fail(f"lines of unknown or mixed kinds {sorted(map(str, kinds))}")
    elif f"{path}/report.json" in tree:
        j.kind, j.status = "sim_report", ESTABLISHED
        try:
            judge_sim_report(json.loads(ctx.repo.blob(ctx.ref, f"{path}/report.json") or b""), j, ctx)
        except ValueError:
            j.fail("report.json is not JSON")
    else:
        j.fail("not a tracked file or session directory")
    if j.prereg in ("FAIL", "NON_GATED", "NOT_GOVERNED"):
        j.fail(f"prereg status {j.prereg}")
    return j


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


def support_problem(j: Judgement, row_id: str, token: str, table: dict, after: float) -> str | None:
    """Why this newly cited record does NOT supply new support for moving ``row_id`` to ``token`` (None = it does)."""
    if j.status != ESTABLISHED:
        return f"{j.path} is {j.status}"
    if j.kind not in SUPPORT_KINDS or j.record is None:
        return f"{j.path} is a {j.kind}, not a verdict (only a stamped verdict supplies new support)"
    kind = j.record.get("kind")
    entry = table.get(kind)
    if not isinstance(entry, dict):
        return f"{j.path}: verdict kind {kind!r} is not in the merge-base pass table"
    if row_id not in (entry.get("rows") or []):
        return f"{j.path}: {kind} may not support {row_id}"
    if j.record.get("verdict") not in ((entry.get("targets") or {}).get(token) or []):
        return f"{j.path}: verdict {j.record.get('verdict')!r} does not support {token}"
    if not require_met(j.record, entry.get("require") or {}):
        return f"{j.path}: {kind}'s required fields {entry.get('require')} do not hold"
    if j.prereg != "PASS":
        return f"{j.path}: prereg status {j.prereg} (new support must be pre-registered and PASS)"
    if j.allowed_dirty:
        return f"{j.path}: allowed-dirty data is never the sole new support"
    if j.time is None or j.time <= after:
        return f"{j.path}: its runs (ts {j.time}) are not after the previous status was set ({after})"
    return None


EXCEPTION_FIELDS = ("id", "kind", "row", "from", "to", "to_date", "path", "sha256", "owner", "reason", "date")


def load_json(repo: Repo, ref: str, path: str, default):
    raw = repo.blob(ref, path)
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise GateError(f"{path} at {ref[:12]} is not JSON") from exc


def exceptions_problems(base_list, head_list) -> list[str]:
    out = []
    if not isinstance(head_list, list) or not isinstance(base_list, list):
        return [f"{EXCEPTIONS} must be a JSON list"]
    head_set = [json.dumps(e, sort_keys=True) for e in head_list]
    for e in base_list:
        if json.dumps(e, sort_keys=True) not in head_set:
            out.append(f"{EXCEPTIONS}: an entry on main was edited or removed (append-only): {e.get('id')!r}")
    for e in head_list:
        if not isinstance(e, dict) or e.get("kind") not in ("ledger", "prereg"):
            out.append(f"{EXCEPTIONS}: an entry is not a ledger/prereg exception: {e!r}"[:200])
        elif e["kind"] == "ledger" and any(not e.get(f) for f in EXCEPTION_FIELDS):
            out.append(f"{EXCEPTIONS}: ledger exception {e.get('id')!r} lacks a required field")
    return out


def legacy_problems(repo: Repo, base: str, head_snap, base_snap) -> list[str]:
    out = []
    if not isinstance(head_snap, dict):
        return [f"{LEGACY_SNAPSHOT} must be a JSON object"]
    if base_snap is not None and isinstance(base_snap, dict):
        for key in sorted(set(head_snap) - set(base_snap)):
            out.append(f"{LEGACY_SNAPSHOT}: {key} was added (the snapshot only shrinks)")
    tree = repo.tree("HEAD")
    for path, digest in sorted(head_snap.items()):
        if path not in tree:
            out.append(f"{LEGACY_SNAPSHOT}: {path} is gone (remove its key)")
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
        except OSError:
            body = data
        if b'"record_kind"' in body:
            out.append(f"{LEGACY_SNAPSHOT}: {path} carries a record_kind (it is not legacy)")
    return out


def generate_legacy(repo: Repo) -> dict[str, str]:
    snap = {}
    for path, (mode, _oid) in sorted(repo.tree("HEAD").items()):
        if not path.startswith(DATA_ROOT + "/") or mode == "120000":
            continue
        first = repo.first_commit_time(path, "HEAD")
        if first is None or first >= M1A_CUTOFF:
            continue
        data = repo.blob("HEAD", path) or b""
        try:
            body = decompressed(path, data)
        except OSError:
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
    if old.qualifier and (row.qualifier is None or old.qualifier not in row.qualifier):
        out.append("qualifier removed or rewritten")
    for prefix in watched:
        if any(c == prefix or c.startswith(prefix.rstrip("/") + "/") for c in changed):
            out.append(f"{prefix} changed")
    return out


def judged_class(row: L.Row, old: L.Row | None) -> bool:
    if row.token in L.POSITIVE:
        return True
    return row.token == "PARTIAL" and L.is_raise(old.token if old else None, "PARTIAL")


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
    if not isinstance(table, dict):
        failures.append(f"{PASS_TABLE} at the merge-base is not an object")
        table = {}
    head_snap = load_json(repo, "HEAD", LEGACY_SNAPSHOT, None)
    base_snap = load_json(repo, base, LEGACY_SNAPSHOT, None)
    if head_snap is None:
        failures.append(f"{LEGACY_SNAPSHOT} is missing")
        head_snap = {}
    failures += legacy_problems(repo, base, head_snap, base_snap)
    base_exc = load_json(repo, base, EXCEPTIONS, [])
    head_exc = load_json(repo, "HEAD", EXCEPTIONS, [])
    failures += exceptions_problems(base_exc, head_exc)
    if prereg is None:
        import lint_prereg_precedes_data as P  # noqa: PLC0415

        envelope = P.classify_all(repo.root)
        failures += [f"prereg lint: {f}" for f in envelope.get("failures") or []]
        prereg = {e["entry"]: e["status"] for e in envelope.get("entries") or []}
    ctx = Ctx(repo=repo, base=base, ref="HEAD", legacy=head_snap, prereg=prereg)
    changed = repo.changed(base)
    results = []
    for row in head_rows:
        old = base_rows.get(row.id)
        if row.token is None or not judged_class(row, old):
            continue
        judgements = [judge_entry(e.path, ctx) for e in row.evidence]
        watched = [e.path for e in row.evidence]
        for j in judgements:
            if j.record and isinstance(j.record.get("data"), str):
                watched.append(j.record["data"])
                if j.record.get("kind") in O19_KINDS:
                    data_dir = j.record["data"].rsplit("/", 1)[0]
                    watched.append(data_dir + "/")
        trig = triggers_for(row, old, changed, watched)
        if not trig:
            continue
        res = RowResult(id=row.id, triggers=trig, failures=[], notes=[])
        results.append(res)
        if row.token == L.BY_TESTS:
            res.notes.append("RE-VALIDATED-BY-TESTS: named, not checked")
            continue
        active = [e for e in head_exc if isinstance(e, dict) and e.get("kind") == "ledger" and e.get("row") == row.id]
        for j in judgements:
            exc = next((e for e in active if e.get("path") == j.path), None)
            if j.status == NOT_ESTABLISHED and exc and exc.get("sha256") == sha256(repo.blob("HEAD", j.path) or b""):
                j.status = EXCEPTED
            if j.status == NOT_ESTABLISHED:
                res.failures += [f"{j.path}: {r}" for r in (j.reasons or ["not established"])]
            elif j.status != ESTABLISHED:
                res.notes.append(f"{j.path}: {j.status}")
        needs_support = old is None or L.needs_new_date(old.token, row.token) or old.date != row.date
        if needs_support:
            base_paths = {e.path for e in old.evidence} if old else set()
            after = status_set_time(repo, base, row.id, old.token, old.date) if old else float(repo.commit_time(base))
            new = [j for j in judgements if j.path not in base_paths]
            problems_new = [support_problem(j, row.id, row.token, table, after) for j in new]
            cited = {e.path for e in row.evidence}
            first_clause = any(
                e.get("from") == (old.token if old else None)
                and e.get("to") == row.token
                and e.get("to_date") == row.date
                and e.get("path") in cited
                and e.get("sha256") == sha256(repo.blob("HEAD", e["path"]) or b"")
                for e in active
            )
            if not any(p is None for p in problems_new) and not first_clause:
                res.failures.append(
                    "no NEW support: " + ("; ".join(p for p in problems_new if p) or "no newly cited record")
                )
        if old and {e.path for e in old.evidence} - {e.path for e in row.evidence} and row.token in L.POSITIVE:
            base_ctx = Ctx(
                repo=repo, base=base, ref=base, legacy=load_json(repo, base, LEGACY_SNAPSHOT, {}) or {}, prereg=prereg
            )
            had = any(judge_entry(e.path, base_ctx).status == ESTABLISHED for e in old.evidence)
            if had and not any(j.status == ESTABLISHED for j in judgements):
                res.failures.append("Evidence removed: the row had an ESTABLISHED record on main and keeps none")
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
