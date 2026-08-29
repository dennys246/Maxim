#!/usr/bin/env python3
"""Pre-registration precedes data, checked by CI (roadmap 1.1.x item 16.8).

The rule (docs/agents/simulation-experiments.md §3, full history in
docs/lessons/experiment-prereg-precedes-data.md): a GATED experiment record —
anything under ``docs/experiments/data/`` — is only evidence of a frozen gate
if its pre-registration (and every amendment governing that data) was ON
``main`` before the first record was written. Exp 53/53b (2026-08-26, release
day) broke this twice — the 53 prereg reached ``main`` two hours AFTER the
first record, the 53b prereg only in the squash that also landed its data —
and got an EARNED ledger row the same afternoon the tag was placed. This
lint turns "was the gate frozen before the data" from a self-attestation into
something CI answers.

What it checks, for every experiment token that has a pre-registration:

* **Map.** Result docs ``docs/experiments/*.md`` name pre-registrations by
  linking ``protocols/*preregistration*.md``. A prereg's token is its filename
  up to the first ``_`` with a leading ``exp`` stripped (``exp53b_…`` → ``53b``,
  ``h1_healthy…`` → ``h1``). A data entry (file or directory directly under
  ``docs/experiments/data/``) has the same token rule (``53b_cross…`` →
  ``53b``, ``44b_pilot/`` → ``44b``). A data token is governed by the preregs
  with the same token AND, for a lettered token, by its numeric parent's
  (``53b`` data is governed by the ``53b`` prereg and the ``53`` prereg —
  53b is 53's declared-delta follow-up). Data whose token has no prereg is
  out of scope (it never claimed a frozen gate).
* **Non-gated entries** are skipped by name: ``dry_run`` / ``dryrun`` /
  ``nonfrozen`` — harness shakedowns that legitimately predate the prereg's
  landing and are never cited as evidence.
* **Time of the prereg on main:** ``git log <ref> --diff-filter=A
  --format=%ct -- <prereg>`` (its FIRST appearance on the ref). Missing from
  the ref = FAIL ("prereg not on main").
* **Time of each PRE-DATA amendment on main:** for every ``**Amendment N —
  …PRE-DATA…**`` header in the prereg, the first commit on the ref whose
  diff introduced that header (``git log --reverse -S``). Amendments not
  marked PRE-DATA (post-data relabels the amendment rule allows) are
  reported, not judged — they do not govern frozen gates.
* **Time of the data:** the minimum top-level ``ts`` across the entry's
  JSON/JSONL records (float epoch or ISO-8601; directories are walked). Files
  without any ``ts`` (Exp 52/54 campaign rows) fall back to the entry's own
  first-appearance commit time on the ref, then on HEAD (a PR branch), then
  "now" for an uncommitted file — every fallback is a LATER bound than the
  true first write, so the check can only get stricter, never vacuous.
* **Assertion:** prereg time < data time, and every PRE-DATA amendment time <
  data time. Strict: the same commit fails (that is exactly the squash
  failure mode).
* **allow_dirty echo:** a data file any of whose records carries
  ``allow_dirty: true`` (a harness run with ``--allow-dirty``, item 16.7) must
  have its result doc mention ``allow_dirty`` — the write-up cannot omit it.

**Grandfathered entries** (``GRANDFATHERED`` below) are the ORIGINAL Exp 53 /
53b files: they fail this rule, the failure is recorded in the lesson and the
result doc, and the ledger row rests on the R1 replication (which passes).
They are listed by path WITH the reason — the rule is not weakened for them
— and the lint reports them as ``GRANDFATHERED (still failing)``; if one ever
starts passing (history rewritten), the entry becomes stale and the lint says
so, so the list cannot outlive its reason.

Needs full history of the ref (a shallow ``--depth=1`` clone makes every file
look freshly added and the check would pass vacuously) — a shallow repository
is exit 2, never a pass.

Exits: 0 clean; 1 violations (stderr); 2 cannot check (shallow repo, git
failure, missing ref).
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

# Original Exp 53 / 53b records (2026-08-26, release day) — the incident itself.
GRANDFATHERED: dict[str, str] = {
    "docs/experiments/data/53_cross_context_readout.jsonl": (
        "Exp 53 original: first record 15:04Z at a dirty 68f9026e; its pre-registration reached main "
        "at 17:05Z (#550), two hours AFTER the data. Disclosed in docs/experiments/53_cross_context_readout.md "
        "and docs/lessons/experiment-prereg-precedes-data.md; the ledger row rests on the R1 replication "
        "(53b_cross_context_readout_replication_2026-08-28.jsonl), which passes this rule."
    ),
    "docs/experiments/data/53b_cross_context_readout.jsonl": (
        "Exp 53b original: first record 15:57Z; its pre-registration's only appearance on main is the squash "
        "617b1625 (#551, 18:27Z) that also landed this file — same commit, so the freeze-before-data evidence "
        "does not exist. Disclosed in the result doc + lesson; superseded as evidence by the R1 replication."
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
        "run and not evidence for the 44b gates): campaign_start 2026-08-10 11:53:49 local (naive ts) precedes "
        "the pre-registration commit 1667ad19 (12:10:05 -0600) by 16 minutes — the prereg was on a branch, not "
        "on main, when the pilot began. Surfaced by this lint 2026-08-29; the confirmatory run has not happened."
    ),
}

NON_GATED_MARKERS = ("dry_run", "dryrun", "nonfrozen")
_PREREG_LINK = re.compile(r"\(([^)\s\[(]*protocols/[^)\s\[(]*preregistration[^)\s\[(]*\.md)(?:#[^)]*)?\)")
_AMENDMENT_HEADER = re.compile(r"^\*\*Amendment\s+(\d+)\s+—[^*]*\*\*", re.M)
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


def prereg_map(repo_root: Path) -> tuple[dict[str, set[Path]], dict[str, set[Path]]]:
    """(token → preregs named by result docs, token → result docs naming a prereg)."""
    preregs: dict[str, set[Path]] = {}
    docs: dict[str, set[Path]] = {}
    for doc in sorted((repo_root / EXPERIMENTS_DIR).glob("*.md")):
        text = doc.read_text(errors="replace")
        for m in _PREREG_LINK.finditer(text):
            target = (doc.parent / m.group(1)).resolve()
            try:
                rel = target.relative_to(repo_root)
            except ValueError:
                continue
            tok = token_of(rel.name)
            preregs.setdefault(tok, set()).add(rel)
            docs.setdefault(tok, set()).add(doc.relative_to(repo_root))
    return preregs, docs


def first_commit_time(repo_root: Path, ref: str, path: Path) -> int | None:
    """Epoch of the FIRST commit on ``ref`` that added ``path`` (None: not on ref)."""
    out = _git(repo_root, "log", ref, "--diff-filter=A", "--format=%ct", "--", str(path)).split()
    return int(out[-1]) if out else None


def amendment_times(repo_root: Path, ref: str, prereg: Path) -> list[tuple[int, str, bool, int | None]]:
    """(number, header, pre_data, epoch-on-ref) for each amendment header in the prereg."""
    text = (repo_root / prereg).read_text(errors="replace")
    out = []
    for m in _AMENDMENT_HEADER.finditer(text):
        header = m.group(0)
        pre_data = "PRE-DATA" in header.upper()
        needle = header.split("—", 1)[0].strip()  # "**Amendment N" — stable across header edits
        log = _git(repo_root, "log", ref, "--reverse", "--format=%ct", "-S", needle, "--", str(prereg)).split()
        out.append((int(m.group(1)), header, pre_data, int(log[0]) if log else None))
    return out


NAIVE_SEEN: set[str] = set()


def _parse_ts(value: object, *, where: str = "") -> float | None:
    """Epoch from a float `ts` or an ISO-8601 string. A NAIVE ISO string (no offset) is
    read as UTC and reported — the harness should write epoch seconds (JsonlLog does)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if dt.tzinfo is None:
            NAIVE_SEEN.add(where)  # reported once per file, repo-relative
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    return None


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


def data_first_ts(entry: Path) -> tuple[float | None, bool]:
    """(min top-level ts across the entry's records, any record has allow_dirty: true)."""
    files = [entry] if entry.is_file() else sorted(p for p in entry.rglob("*") if p.suffix in (".json", ".jsonl"))
    first: float | None = None
    allow_dirty = False
    for f in files:
        for rec in _records(f):
            if rec.get("allow_dirty") is True:
                allow_dirty = True
            ts = _parse_ts(rec.get(_TS_KEY), where=f.as_posix().split("docs/experiments/", 1)[-1])
            if ts is not None and (first is None or ts < first):
                first = ts
    return first, allow_dirty


def data_time(repo_root: Path, ref: str, rel: Path) -> tuple[float, str]:
    """The data's first-write time and how it was established (ts | ref-add | head-add | now)."""
    ts, _ = data_first_ts(repo_root / rel)
    if ts is not None:
        return ts, "first ts"
    t = first_commit_time(repo_root, ref, rel)
    if t is not None:
        return float(t), f"first commit on {ref} (no ts field)"
    t = first_commit_time(repo_root, "HEAD", rel)
    if t is not None:
        return float(t), "first commit on HEAD (not yet on the ref; no ts field)"
    return time.time(), "now (uncommitted; no ts field)"


def _fmt(t: float | int) -> str:
    return datetime.fromtimestamp(float(t), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def lint(repo_root: Path = REPO_ROOT, ref: str = DEFAULT_REF, *, grandfathered: dict[str, str] | None = None) -> int:
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
        preregs, docs = prereg_map(repo_root)
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
            when, how = data_time(repo_root, ref, rel)
            problems: list[str] = []
            for prereg in sorted(governing):
                p_time = first_commit_time(repo_root, ref, prereg)
                if p_time is None:
                    problems.append(f"pre-registration {prereg} is not on {ref}")
                    continue
                if not p_time < when:
                    problems.append(
                        f"pre-registration {prereg} reached {ref} at {_fmt(p_time)}, "
                        f"not before the data ({_fmt(when)}, {how})"
                    )
                for num, header, pre_data, a_time in amendment_times(repo_root, ref, prereg):
                    if not pre_data:
                        notes.append(f"{rel}: amendment {num} of {prereg.name} is not marked PRE-DATA — not judged")
                        continue
                    if a_time is None:
                        problems.append(f"PRE-DATA amendment {num} of {prereg} is not on {ref}")
                    elif not a_time < when:
                        problems.append(
                            f"PRE-DATA amendment {num} of {prereg} reached {ref} at {_fmt(a_time)}, "
                            f"not before the data ({_fmt(when)}, {how})"
                        )
            _, allow_dirty = data_first_ts(entry)
            if allow_dirty:
                result_docs = docs.get(tok, set()) | (docs.get(parent, set()) if parent else set())
                echoed = any("allow_dirty" in (repo_root / d).read_text(errors="replace") for d in result_docs)
                if not echoed:
                    problems.append(
                        "records carry allow_dirty: true but no result doc for this experiment mentions "
                        "`allow_dirty` — the write-up must echo the harness's dirty-tree allowance"
                    )
            key = rel.as_posix()
            if problems and key in grandfathered:
                notes.append(f"{rel}: GRANDFATHERED (still failing) — {grandfathered[key]}")
                for p in problems:
                    notes.append(f"    {p}")
            elif problems:
                failures.append(f"{rel}:")
                failures.extend(f"    {p}" for p in problems)
            elif key in grandfathered:
                failures.append(
                    f"{rel}: listed as GRANDFATHERED but now PASSES — remove the stale entry (its reason no longer holds)"
                )
        for key in grandfathered:
            if not (repo_root / key).exists():
                failures.append(f"{key}: GRANDFATHERED entry names a file that no longer exists — remove it")
    except LintError as exc:
        print(f"ERROR: cannot check prereg-precedes-data: {exc}", file=sys.stderr)
        return 2

    for n in notes:
        print(f"NOTE: {n}")
    for w in sorted(NAIVE_SEEN):
        print(f"NOTE: {w}: naive ISO-8601 `ts` (no UTC offset) read as UTC — write epoch seconds instead")
    if failures:
        print("prereg-precedes-data lint FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        print(
            "\nA gated record's pre-registration (and each PRE-DATA amendment) must be ON main before the "
            "first record's ts — merge the prereg as its own PR first, then run (docs/agents/"
            "simulation-experiments.md §3; docs/lessons/experiment-prereg-precedes-data.md).",
            file=sys.stderr,
        )
        return 1
    print(
        f"prereg-precedes-data lint: clean — {checked} governed data entr{'y' if checked == 1 else 'ies'} "
        f"checked against {ref}, {len(grandfathered)} grandfathered by explicit list (see GRANDFATHERED)"
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
