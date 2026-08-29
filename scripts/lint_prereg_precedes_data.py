#!/usr/bin/env python3
"""Pre-registration precedes data, checked by CI (roadmap 1.1.x item 16.8).

The rule (docs/agents/simulation-experiments.md §3, full history in
docs/lessons/experiment-prereg-precedes-data.md): a GATED experiment record —
anything under ``docs/experiments/data/`` — is only evidence of a frozen gate
if its pre-registration (and every amendment governing that data) was ON
``main`` before the first record was written, and the record came from a
clean tree (or says it did not). Exp 53/53b (2026-08-26, release day) broke
this — the 53 prereg reached ``main`` two hours AFTER the first record, the
53b prereg only in the squash that also landed its data — and got an EARNED
ledger row the same afternoon the tag was placed. This lint turns "was the
gate frozen before the data" from a self-attestation into something CI
answers.

What it checks, for every experiment token that has a pre-registration:

* **Map.** Pre-registrations are every ``protocols/*preregistration*.md``,
  plus any such file a result doc ``docs/experiments/*.md`` links. A prereg's
  token is its filename up to the first ``_`` with a leading ``exp`` stripped
  (``exp53b_…`` → ``53b``, ``h1_healthy…`` → ``h1``). A data entry (file or
  directory directly under ``docs/experiments/data/``) has the same token
  rule (``53b_cross…`` → ``53b``, ``44b_pilot/`` → ``44b``). A data token is
  governed by the preregs with the same token AND, for a lettered token, by
  its numeric parent's (``53b`` data is governed by the ``53b`` prereg and
  the ``53`` prereg — 53b is 53's declared-delta follow-up). Data whose
  token has no prereg is out of scope (it never claimed a frozen gate).
* **Non-gated entries** are skipped by name: ``dry_run`` / ``dryrun`` /
  ``nonfrozen`` — harness shakedowns that legitimately predate the prereg's
  landing and are never cited as evidence.
* **Time the prereg REACHED the ref:** ``git log --first-parent <ref>
  --diff-filter=A --format=%ct -- <prereg>``. ``--first-parent`` is
  load-bearing: without it a merge-committed PR reports the file's BRANCH
  commit time, and the brief's rule (3) mandates merge commits for data /
  protocol PRs — both review lenses caught this on the first draft. A prereg
  missing from the ref = FAIL. A prereg renamed after its data is a FAIL by
  construction (the rename is when the new path reached the ref).
* **Time each PRE-DATA amendment reached the ref:** every line starting
  ``**Amendment N`` must be a header of the shape ``**Amendment N — <date>,
  PRE-DATA|POST-DATA, …**``; anything else, or an unclassified amendment, is
  a FAIL (a mis-formatted header must not silently drop out of the check).
  PRE-DATA amendments are judged by the first commit on the ref's
  first-parent chain that introduced their header; POST-DATA ones are
  reported, not judged.
* **Time of the data:** the minimum top-level ``ts`` across the entry's
  JSON/JSONL records (epoch seconds, or ISO-8601 WITH an offset; a naive ISO
  string is a FAIL for non-grandfathered entries because its zone is
  unknowable — the one producer, the 44b pilot harness, now writes epoch).
  Files without any ``ts`` (the Exp 52/54 campaign rows, JSON inputs) fall
  back to the entry's first-parent commit time on the ref, then on HEAD (a
  PR branch), then "now" for an uncommitted file. **Every fallback is MORE
  LENIENT than the true first-write time** — a later data time makes
  ``prereg < data`` easier — so the check degrades to commit granularity
  ("data committed after the prereg was on main") and says so with a NOTE
  per entry. A ``.jsonl`` entry first committed after 2026-08-29 with no
  ``ts`` at all is a FAIL: every harness now stamps one.
* **Assertion:** prereg time < data time, and every PRE-DATA amendment time
  < data time. Strict: the same commit fails (that is exactly the squash
  failure mode). A NOTE also counts commits that touched the prereg on the
  ref after the data's first ts (in-place edits without an amendment
  header are otherwise invisible — see "not caught").
* **Clean-tree attestation:** a record (top level or under ``provenance``)
  carrying ``working_tree_dirty_src_scripts: true`` without ``allow_dirty:
  true`` is a FAIL — the harness refused-or-allowed door (item 16.7) covers
  the write path; this covers the record after the fact, including one
  written to ``/tmp`` and moved in. Any ``allow_dirty: true`` must be
  echoed by a result doc for the experiment (the write-up cannot omit it).
* The lint refuses to pass vacuously: a shallow repository, a missing ref,
  or ZERO governed entries is exit 2, never a pass.

**Grandfathered entries** (``GRANDFATHERED`` below) are listed by path WITH
the reason — the rule is not weakened for them — and reported as
``GRANDFATHERED (still failing)``; an entry that starts passing (history
rewritten) or names a missing file fails the lint as stale, so the list
cannot outlive its reason.

**Catches forgetting, not evasion** (house convention for heuristic lints):
``ts`` and the dirty flag are harness-self-reported; a prereg's frozen gates
edited in place after data with no amendment header are only visible through
the post-data-commit NOTE; a POST-DATA label written where PRE-DATA was true
is trusted; a record written elsewhere without provenance and moved in is
only caught if it carries the dirty flag. It is a forcing function for the
honest author, not a security boundary.

Exits: 0 clean; 1 violations (stderr); 2 cannot check (shallow repo, git
failure, missing ref, nothing governed).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REF = "origin/main"

EXPERIMENTS_DIR = Path("docs/experiments")
DATA_DIR = EXPERIMENTS_DIR / "data"
PROTOCOLS_DIR = EXPERIMENTS_DIR / "protocols"

# Gated .jsonl entries first committed on or after this date must carry `ts`.
TS_REQUIRED_FROM = datetime(2026, 8, 29, tzinfo=timezone.utc).timestamp()

# Explicit, reasoned exceptions — reported as still-failing on every run.
GRANDFATHERED: dict[str, str] = {
    "docs/experiments/data/53_cross_context_readout.jsonl": (
        "Exp 53 original (2026-08-26 release day): first record 15:04Z from a DIRTY tree at 68f9026e (stamped, no "
        "allow_dirty — the flag did not exist); its pre-registration reached main at 17:05Z (#550), two hours AFTER "
        "the data. Disclosed in docs/experiments/53_cross_context_readout.md and docs/lessons/"
        "experiment-prereg-precedes-data.md; the ledger row rests on the R1 replication "
        "(53b_cross_context_readout_replication_2026-08-28.jsonl), which passes every rule here."
    ),
    "docs/experiments/data/53b_cross_context_readout.jsonl": (
        "Exp 53b original (2026-08-26): first record 15:57Z from the same DIRTY tree; its pre-registration's only "
        "appearance on main is the squash 617b1625 (#551, 18:27Z) that also landed this file — same commit, so the "
        "freeze-before-data evidence does not exist. Disclosed in the result doc + lesson; superseded by R1."
    ),
    "docs/experiments/data/53_agents": (
        "Exp 53 inputs (the nursery agents' nac/ec state, sha256-pinned by the manifest): landed in the same "
        "squash 617b1625 as PRE-DATA amendments 1–2 of the 53 pre-registration, so 'amendment before data' "
        "cannot be shown for the original run. R1 (2026-08-28) re-read these same files from a clean tree."
    ),
    "docs/experiments/data/53_agents_manifest.json": (
        "Exp 53 manifest — same squash 617b1625 as amendments 1–2 (see 53_agents)."
    ),
    "docs/experiments/data/44b_pilot": (
        "Exp 44b PILOT (instrument shakedown, docs/experiments/44b_pilot.md — explicitly not the confirmatory "
        "run and not evidence for the 44b gates): campaign_start `2026-08-10T11:53:49` is a NAIVE local time "
        "(−0600, i.e. 17:53Z) and precedes the pre-registration commit 1667ad19 (12:10:05 −0600) by 16 minutes "
        "— the prereg was on a branch, not on main, when the pilot began. Surfaced by this lint 2026-08-29; the "
        "confirmatory run has not happened; the harness now writes epoch `ts`."
    ),
    "docs/experiments/data/h1_partc_big_block.jsonl": (
        "H1 Part C `_big` block (2026-08-24, run 20260824T213320Z at b01a6589): stamped "
        "working_tree_dirty_src_scripts: true with no allowance (the --allow-dirty door did not exist). Cited by "
        "the Exp 45 EARNED ledger row (behavioral_graduation_candidates.md) as delivered-shift evidence — surfaced "
        "by the 2026-08-29 review round; disclosed on that row. Re-run under the refusal rule is owed (item 16.7)."
    ),
    "docs/experiments/data/54_targets.json": (
        "Exp 54 sweep-declared Phase B targets (2026-08-26, 93887e6e): provenance block stamped "
        "working_tree_dirty_src_scripts: true with no allowance. A gated INPUT to Phase B/C; disclosed in "
        "docs/experiments/54_nurture_reachy_body.md (2026-08-29). Phase B/C must re-declare from a clean tree "
        "or run with --allow-dirty and echo it."
    ),
}

NON_GATED_MARKERS = ("dry_run", "dryrun", "nonfrozen")
_PREREG_LINK = re.compile(r"\(([^)\s\[(]*protocols/[^)\s\[(]*preregistration[^)\s\[(]*\.md)(?:#[^)]*)?\)")
_AMENDMENT_LINE = re.compile(r"^\*\*Amendment\s+(\d+)\b.*$", re.M)
# The bold header may wrap across lines: `**Amendment N — <date>, PRE-DATA, …**`
_AMENDMENT_HEADER = re.compile(r"^\*\*Amendment\s+(\d+)\s+—(.*?)\*\*", re.M | re.S)
_TS_KEY = "ts"


class LintError(RuntimeError):
    """The check cannot be performed (exit 2) — never a pass."""


def _git(repo_root: Path, *args: str) -> str:
    r = subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise LintError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def token_of(name: str) -> str:
    """``exp53b_cross…`` → ``53b``; ``44b_pilot`` → ``44b``; ``h1_doa_sweep.jsonl`` → ``h1``."""
    stem = name.split("/")[-1]
    if stem.startswith("exp") and len(stem) > 3 and stem[3].isdigit():
        stem = stem[3:]
    head = stem.split("_", 1)[0]
    return head.split(".", 1)[0]


def parent_token(token: str) -> str | None:
    """``53b`` → ``53``; ``53`` → None; ``h1`` → None."""
    m = re.fullmatch(r"(\d+)[a-z]+", token)
    return m.group(1) if m else None


def prereg_map(repo_root: Path) -> tuple[dict[str, set[Path]], dict[str, set[Path]], list[str]]:
    """(token → preregs, token → result docs naming a prereg, notes)."""
    preregs: dict[str, set[Path]] = {}
    docs: dict[str, set[Path]] = {}
    notes: list[str] = []
    for prereg in sorted((repo_root / PROTOCOLS_DIR).glob("*preregistration*.md")):
        preregs.setdefault(token_of(prereg.name), set()).add(prereg.relative_to(repo_root))
    for doc in sorted((repo_root / EXPERIMENTS_DIR).glob("*.md")):
        text = doc.read_text(errors="replace")
        for m in _PREREG_LINK.finditer(text):
            target = (doc.parent / m.group(1)).resolve()
            try:
                rel = target.relative_to(repo_root)
            except ValueError:
                notes.append(
                    f"{doc.relative_to(repo_root)}: prereg link resolves outside the repo — ignored: {m.group(1)}"
                )
                continue
            tok = token_of(rel.name)
            preregs.setdefault(tok, set()).add(rel)
            docs.setdefault(tok, set()).add(doc.relative_to(repo_root))
    return preregs, docs, notes


def first_commit_time(repo_root: Path, ref: str, path: Path) -> int | None:
    """Epoch of the first commit on ``ref``'s FIRST-PARENT chain that added ``path`` (None: not on ref)."""
    out = _git(repo_root, "log", "--first-parent", ref, "--diff-filter=A", "--format=%ct", "--", str(path)).split()
    return int(out[-1]) if out else None


def amendment_times(repo_root: Path, ref: str, prereg: Path) -> list[tuple[int, str, bool, int | None]]:
    """(number, header, pre_data, epoch-on-ref) per amendment; raises LintError on a malformed header."""
    full = repo_root / prereg
    if not full.exists():
        raise LintError(f"pre-registration {prereg} is named but does not exist in the working tree")
    text = full.read_text(errors="replace")
    out = []
    for line_m in _AMENDMENT_LINE.finditer(text):
        header_m = _AMENDMENT_HEADER.match(text, line_m.start())
        kinds = re.findall(r"\b(PRE-DATA|POST-DATA)\b", header_m.group(2)) if header_m else []
        if header_m is None or len(set(kinds)) != 1:
            raise LintError(
                f"{prereg}: amendment header is not of the shape `**Amendment N — <date>, PRE-DATA|POST-DATA, …**`: "
                f"{line_m.group(0)[:90]!r} — an unclassified amendment cannot be judged and must not drop out"
            )
        num, kind = int(header_m.group(1)), kinds[0]
        needle = f"**Amendment {num}"
        log = _git(
            repo_root, "log", "--first-parent", ref, "--reverse", "--format=%ct", "-S", needle, "--", str(prereg)
        )
        out.append((num, header_m.group(0), kind == "PRE-DATA", int(log.split()[0]) if log.split() else None))
    return out


def commits_after(repo_root: Path, ref: str, path: Path, after: float) -> int:
    """Commits on the ref's first-parent chain touching ``path`` with committer time > ``after``."""
    out = _git(repo_root, "log", "--first-parent", ref, "--format=%ct", "--", str(path)).split()
    return sum(1 for t in out if int(t) > after)


def _parse_ts(value: object) -> tuple[float | None, bool]:
    """(epoch, naive) from a float `ts` or an ISO-8601 string."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value), False
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None, False
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc).timestamp(), True
        return dt.timestamp(), False
    return None, False


def _records(path: Path):
    text = path.read_text(errors="replace")
    if path.suffix == ".jsonl":
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
    elif path.suffix == ".json":
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return
        if isinstance(obj, list):
            yield from (o for o in obj if isinstance(o, dict))
        elif isinstance(obj, dict):
            yield obj


def _dirty_flag(rec: dict) -> bool:
    prov = rec.get("provenance") if isinstance(rec.get("provenance"), dict) else {}
    return rec.get("working_tree_dirty_src_scripts") is True or prov.get("working_tree_dirty_src_scripts") is True


def _allow_flag(rec: dict) -> bool:
    prov = rec.get("provenance") if isinstance(rec.get("provenance"), dict) else {}
    return rec.get("allow_dirty") is True or prov.get("allow_dirty") is True


class DataFacts:
    __slots__ = ("first_ts", "naive", "allow_dirty", "dirty_unallowed", "has_records")

    def __init__(self) -> None:
        self.first_ts: float | None = None
        self.naive = False
        self.allow_dirty = False
        self.dirty_unallowed = 0
        self.has_records = False


def data_facts(entry: Path) -> DataFacts:
    facts = DataFacts()
    files = [entry] if entry.is_file() else sorted(p for p in entry.rglob("*") if p.suffix in (".json", ".jsonl"))
    for f in files:
        for rec in _records(f):
            facts.has_records = True
            if _allow_flag(rec):
                facts.allow_dirty = True
            elif _dirty_flag(rec):
                facts.dirty_unallowed += 1
            ts, naive = _parse_ts(rec.get(_TS_KEY))
            if ts is not None:
                facts.naive |= naive
                if facts.first_ts is None or ts < facts.first_ts:
                    facts.first_ts = ts
    return facts


def data_time(repo_root: Path, ref: str, rel: Path, facts: DataFacts) -> tuple[float, str, bool]:
    """(first-write time, how it was established, fallback?)."""
    if facts.first_ts is not None:
        return facts.first_ts, "first ts", False
    t = first_commit_time(repo_root, ref, rel)
    if t is not None:
        return float(t), f"first commit on {ref} (no ts field — commit granularity, LENIENT)", True
    t = first_commit_time(repo_root, "HEAD", rel)
    if t is not None:
        return float(t), "first commit on HEAD (not on the ref yet; no ts field — LENIENT)", True
    return time.time(), "now (uncommitted; no ts field — LENIENT)", True


def _fmt(t: float | int) -> str:
    return datetime.fromtimestamp(float(t), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def lint(repo_root: Path = REPO_ROOT, ref: str = DEFAULT_REF, *, grandfathered: dict[str, str] | None = None) -> int:
    repo_root = Path(repo_root).resolve()
    grandfathered = GRANDFATHERED if grandfathered is None else grandfathered
    try:
        if _git(repo_root, "rev-parse", "--is-shallow-repository").strip() == "true":
            raise LintError("shallow repository — fetch full history (git fetch --unshallow) before running this lint")
        _git(repo_root, "rev-parse", "--verify", "--quiet", ref)
    except LintError as exc:
        print(f"ERROR: cannot check prereg-precedes-data: {exc}", file=sys.stderr)
        return 2

    failures: list[str] = []
    notes: list[str] = []
    checked = 0
    try:
        preregs, docs, map_notes = prereg_map(repo_root)
        notes.extend(map_notes)
        data_root = repo_root / DATA_DIR
        for entry in sorted(data_root.iterdir()) if data_root.exists() else []:
            rel = entry.relative_to(repo_root)
            if any(mk in entry.name for mk in NON_GATED_MARKERS):
                continue
            tok = token_of(entry.name)
            governing: set[Path] = set(preregs.get(tok, ()))
            parent = parent_token(tok)
            if parent:
                governing |= preregs.get(parent, set())
            if not governing:
                continue
            checked += 1
            key = rel.as_posix()
            facts = data_facts(entry)
            when, how, fallback = data_time(repo_root, ref, rel, facts)
            problems: list[str] = []
            if fallback:
                notes.append(f"{rel}: no `ts` in any record — judged at commit granularity ({how})")
                added = first_commit_time(repo_root, ref, rel)
                if entry.suffix == ".jsonl" and facts.has_records and added is not None and added >= TS_REQUIRED_FROM:
                    problems.append(
                        "a .jsonl record file committed after 2026-08-29 must carry epoch `ts` on its records"
                    )
            if facts.naive:
                problems.append(
                    "`ts` is a naive ISO-8601 string (no UTC offset) — its zone is unknowable; write epoch seconds"
                )
            for prereg in sorted(governing):
                p_time = first_commit_time(repo_root, ref, prereg)
                if p_time is None:
                    problems.append(f"pre-registration {prereg} is not on {ref}")
                    continue
                if not p_time < when:
                    problems.append(
                        f"pre-registration {prereg} reached {ref} at {_fmt(p_time)}, not before the data ({_fmt(when)}, {how})"
                    )
                for num, _header, pre_data, a_time in amendment_times(repo_root, ref, prereg):
                    if not pre_data:
                        notes.append(f"{rel}: amendment {num} of {prereg.name} is POST-DATA — reported, not judged")
                        continue
                    if a_time is None:
                        problems.append(f"PRE-DATA amendment {num} of {prereg} is not on {ref}")
                    elif not a_time < when:
                        problems.append(
                            f"PRE-DATA amendment {num} of {prereg} reached {ref} at {_fmt(a_time)}, "
                            f"not before the data ({_fmt(when)}, {how})"
                        )
                later = commits_after(repo_root, ref, prereg, when)
                if later:
                    notes.append(
                        f"{rel}: {prereg.name} was touched by {later} commit(s) on {ref} after the data's first ts "
                        "— in-place edits without an amendment header are not judged; check them"
                    )
            if facts.dirty_unallowed:
                problems.append(
                    f"{facts.dirty_unallowed} record(s) stamp working_tree_dirty_src_scripts: true without "
                    "allow_dirty: true — a gated record from a dirty tree is refused or explicitly allowed, never silent"
                )
            if facts.allow_dirty:
                result_docs = docs.get(tok, set()) | (docs.get(parent, set()) if parent else set())
                echoed = any("allow_dirty" in (repo_root / d).read_text(errors="replace") for d in result_docs)
                if not echoed:
                    problems.append(
                        "records carry allow_dirty: true but no result doc for this experiment mentions "
                        "`allow_dirty` — the write-up must echo the harness's dirty-tree allowance"
                    )
            if problems and key in grandfathered:
                notes.append(f"{rel}: GRANDFATHERED (still failing) — {grandfathered[key]}")
                for p in problems:
                    notes.append(f"    {p}")
            elif problems:
                failures.append(f"{rel}:")
                failures.extend(f"    {p}" for p in problems)
            elif key in grandfathered:
                failures.append(f"{rel}: listed as GRANDFATHERED but now PASSES — remove the stale entry")
        for key in grandfathered:
            if not (repo_root / key).exists():
                failures.append(f"{key}: GRANDFATHERED entry names a file that no longer exists — remove it")
        if checked == 0:
            raise LintError(
                "zero governed data entries — the prereg map or the data glob is broken; refusing to pass vacuously"
            )
    except LintError as exc:
        print(f"ERROR: cannot check prereg-precedes-data: {exc}", file=sys.stderr)
        return 2

    for n in notes:
        print(f"NOTE: {n}")
    if failures:
        print("prereg-precedes-data lint FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        print(
            "\nA gated record's pre-registration (and each PRE-DATA amendment) must be ON main before the "
            "first record's ts, and the record must come from a clean tree or say allow_dirty — merge the prereg as "
            "its own PR first, then run (docs/agents/simulation-experiments.md §3; "
            "docs/lessons/experiment-prereg-precedes-data.md).",
            file=sys.stderr,
        )
        return 1
    print(
        f"prereg-precedes-data lint: clean — {checked} governed data entr{'y' if checked == 1 else 'ies'} "
        f"checked against {ref} (first-parent), {len(grandfathered)} grandfathered by explicit list (see GRANDFATHERED)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", default=DEFAULT_REF, help="the ref that counts as 'on main' (default: origin/main)")
    args = ap.parse_args(argv)
    return lint(REPO_ROOT, args.ref)


if __name__ == "__main__":
    sys.exit(main())
