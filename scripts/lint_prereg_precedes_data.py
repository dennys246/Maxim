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
  every flat ``docs/experiments/*_prereg.md`` (the 1.3-era layout: Exp 58–62,
  R2, R3 — added 1.3.1, when none of 1.3.0's own experiments was governed),
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
  landing and are never cited as evidence — and by type: a ``.py`` file is an
  analysis instrument that lives beside the records, not a record (three of
  them — two Exp 60 oxygen checks and the Exp 62 cross-pool replay — landed
  in the same commit as their prereg).
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

**Re-runs (M1b PR 4, 2026-09-30).**

* **Recognising a re-run.** A data entry is a re-run when:
  - its name starts ``rerun_``;
  - its token is ``NNd<MM>`` (a re-run of NN under a fix: ``42d53`` → parent ``42``; ``53bd53`` → ``53b`` → ``53``);
  - a governed entry's name carries a replication / re-baseline / rerun word; or
  - a pre-registration declares it (a scoped amendment, a ``**Scope:**`` line, or an ``RB-<n>`` section).
* **Tokens.** Tokens strip ``rerun_`` and leading zeros (``rerun_exp09_…`` → ``9``).
* **Declaring a re-run.** A re-run needs its own PRE-DATA declaration on the ref before its data:
  - an amendment scoped to it: ``**Amendment N — <date>, PRE-DATA, for `<entry>`[, `<entry>`] …**``, where
    ``for`` sits right after the class and the entries are backticked; or
  - a re-run pre-registration's ``**Scope:**`` line (``docs/experiments/protocols/TEMPLATE_rerun.md``; the
    ``exp<NN>_rerun_…`` filename marks one, and a Scope line in any other pre-registration FAILS, since it would
    otherwise take the original experiment's data out of the lint). A declared entry must be the
    pre-registration's own experiment's data.
* **Which data a declaration governs.**
  - A pre-registration with a Scope line governs only the entries it names.
  - A scoped amendment is judged only against its entries. An unscoped one is still judged against all of the
    token's data, and its failure message suggests scoping it.
* **The explicit forms need a pre-registration.** A ``rerun_`` or ``NNd<MM>`` entry with no governing
  pre-registration fails unless listed in ``UNGOVERNED_RERUNS``. A word-only match with no pre-registration is
  out of scope, as its original is.
* **Timing.** Declarations are timed by walking the pre-registration's first-parent history, not by
  ``git log -S`` on a header needle. The time of (amendment N, entry) is the first commit where N is PRE-DATA
  and its scope names the entry. So re-scoping or a POST→PRE relabel after the data dates from the edit. A
  historical header in an older shape counts by the class it names (``(pre-data; …)``); only the current text
  must meet the grammar.
* **``ts`` and session directories.** A re-run first committed after the freeze carries ``ts`` on every
  record. A directory of session ``report.json`` files is timed from those reports only.
* **The allow_dirty echo.** It now needs a result doc that names the entry (on a path boundary, or by a
  ``harness_run_id`` from its records), in the paragraph that mentions ``allow_dirty``.
* **Frozen exception lists.** ``GRANDFATHERED`` / ``UNGOVERNED_RERUNS`` / ``NOT_GOVERNED`` may only SHRINK
  against the ref's copy of this file, and may only name data first committed before ``EXCEPTIONS_FROZEN``. A
  new exception goes through the committed exceptions file (M1b decision 1), never into this script beside the
  data it excuses.
* **``classify``.** The ``--json`` document is ``{"_format_version", "entries", "failures"}``; per entry it returns PASS | FAIL | GRANDFATHERED | UNGOVERNED_RERUN | NOT_GOVERNED |
  OUT_OF_SCOPE | NON_GATED, whether the entry is a re-run, and the problems. ``--json`` prints it: the surface
  the M1b evidence gate reads.

**Grandfathered entries** (``GRANDFATHERED`` below) are listed by path WITH
the reason — the rule is not weakened for them — and reported as
``GRANDFATHERED (still failing)``; an entry that starts passing (history
rewritten) or names a missing file fails the lint as stale, so the list
cannot outlive its reason.

**Not-governed entries** (``NOT_GOVERNED`` below) are the token rule's
false matches: an entry whose token collides with a prereg that does not
govern it (``r2`` names a ladder RUNG, and the R2 drive-premise check predates
— and is not — the R2 learned-bias experiment). Listed by path with the
reason, reported as ``NOT GOVERNED``, and stale-checked like the grandfather
list: an entry that no longer matches any prereg, or names a missing file,
fails the lint.

**Two classes only:** an amendment is PRE-DATA or POST-DATA for ALL of an
experiment's data. One that is POST-DATA for some records and PRE-DATA for
others (Exp 60's Amendment 2, the freeze: after the apparatus and gate data,
before the trials) is judged by the class its bold header names — here
POST-DATA, reported not judged. That it preceded the trials was checked by
hand (2026-09-27: ~3.5 min before the first trial ``ts``).

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

# M1b PR 4 (2026-09-30): the exception lists below are FROZEN. They may only shrink against the ref, and may
# only name data first committed on the ref before this instant; a new exception goes through the committed
# exceptions file (M1b decision 1), never into this script in the same PR as the data it excuses.
EXCEPTIONS_FROZEN = datetime(2026, 9, 30, 16, 0, tzinfo=timezone.utc).timestamp()

_RERUN_REASON = (
    "re-run under the original frozen gates before the M1b PR 4 rule (2026-09-30) required a re-run's own "
    "scoped PRE-DATA declaration; nothing declared it in advance"
)

# Explicit, reasoned exceptions — reported as still-failing on every run.
GRANDFATHERED: dict[str, str] = {
    "docs/experiments/data/52d53_phaseB_embodied.jsonl": f"Exp 52 post-D53 re-validation (2026-09-02): {_RERUN_REASON}.",
    "docs/experiments/data/53d53_cross_context_readout.jsonl": (
        f"Exp 53 post-D53 hardware re-run (2026-09-02): {_RERUN_REASON}."
    ),
    "docs/experiments/data/53d53_phase2_aborted_run.jsonl": (
        f"Exp 53 post-D53 phase-2 run, aborted (2026-09-02; disclosed, never evidence): {_RERUN_REASON}."
    ),
    "docs/experiments/data/53b_cross_context_readout_replication_2026-08-28.jsonl": (
        f"Exp 53 R1 replication (2026-08-28), re-reading the original's agent files from a clean tree: {_RERUN_REASON}."
    ),
    "docs/experiments/data/exp56_rebaseline_1204": (
        "Exp 56 re-baseline on Paper 1.20.4 (2026-09-19): declared in the pre-registration's RB-1 section before its "
        "data, but not as a scoped PRE-DATA amendment or Scope line (the grammar did not exist yet)."
    ),
    "docs/experiments/data/53_cross_context_readout.jsonl": (
        "Exp 53 original (2026-08-26 release day): first record 15:04Z from a DIRTY tree at 68f9026e (stamped, no "
        "allow_dirty — the flag did not exist); its pre-registration reached main at 17:05Z (#550), two hours AFTER "
        "the data. Disclosed in docs/experiments/53_cross_context_readout.md and docs/lessons/"
        "experiment-prereg-precedes-data.md; the ledger row rests on the R1 replication "
        "(53b_cross_context_readout_replication_2026-08-28.jsonl) — itself GRANDFATHERED since M1b PR 4 (2026-09-30) "
        "as a re-run with no scoped PRE-DATA declaration."
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

# Re-runs of experiments with NO pre-registration (M1b PR 4). Frozen like GRANDFATHERED; the next re-run of
# any of these lands a re-run pre-registration first (docs/experiments/protocols/TEMPLATE_rerun.md).
UNGOVERNED_RERUNS: dict[str, str] = {
    "docs/experiments/data/42d53_results.jsonl": (
        "Exp 42 post-D53 re-validation (2026-09-01): Exp 42 never had a pre-registration file (its design lives in "
        "the result doc); rows rebuilt from sandboxes (ledger T1-6 caveat)."
    ),
    "docs/experiments/data/42d53_results_gateoff.jsonl": "Exp 42 post-D53 gate-off arm (2026-09-01): as 42d53_results.",
    "docs/experiments/data/rerun_exp09_2026-09-24": (
        "Exp 09 re-run (2026-09-24): Exp 09 has no pre-registration; the run ended planning_failed and backs no "
        "status (ledger T3-9 STALE since 2026-09-30)."
    ),
    "docs/experiments/data/rerun_exp10_2026-09-27": (
        "Exp 10 re-run (2026-09-27): Exp 10 has no pre-registration; every session ended planning_failed and it "
        "backs no status (ledger T1-1 STALE since 2026-09-30). The complete re-run (outstanding O19) lands a "
        "re-run pre-registration first."
    ),
}

# Token-collision exceptions: the entry matches a prereg's token but is not that experiment's data.
NOT_GOVERNED: dict[str, str] = {
    "docs/experiments/data/r2_drive_premise.json": (
        "The R2 drive-premise CHECK (2026-09-07, PREMISE-NULL; docs/experiments/r2_drive_premise_check.md): a "
        "premise probe run with no pre-registration. Its token `r2` is the ladder rung, shared with the later "
        "R2 LEARNED-BIAS preregs (r2_learned_bias_prereg.md 2026-09-12, _v2 2026-09-13), which do not govern it "
        "and which ran no live data (closed offline, 2026-09-12)."
    ),
}

NON_GATED_MARKERS = ("dry_run", "dryrun", "nonfrozen")
NON_RECORD_SUFFIXES = (".py",)
RERUN_PREFIX = "rerun_"
_RERUN_WORDS = re.compile(r"(?<![a-z])(?:rerun|replicat|rebaseline)", re.I)
# `**Scope:** \`<entry>\`[, …]` at a line start: the entries a re-run pre-registration governs (anything else
# after `**Scope:**` is prose and ignored — result docs use it that way).
_SCOPE_LINE = re.compile(r"^(?:>\s*)?\*\*Scope:\*\*\s*(`[^`]+`(?:\s*(?:,\s*and\b|,|;|\band\b)\s*`[^`]+`)*)", re.M)
_ENTRY_LIST = re.compile(r"`[^`]+`(?:\s*(?:,\s*and\b|,|;|\band\b)\s*`[^`]+`)*")
_RB_START = re.compile(r"^\s*(?:#+\s*|[-*]\s*)?\*{0,2}RB-\d+\b")
_SECTION_END = re.compile(r"^\s*(?:#|[-*]\s+\*\*)")
_PREREG_LINK = re.compile(
    r"\(([^)\s\[(]*(?:protocols/[^)\s\[(]*preregistration[^)\s\[(]*|[^)\s\[(/]*_prereg)\.md)(?:#[^)]*)?\)"
)
# A header may sit in a blockquote (`> **Amendment 2 — …**`, Exp 60's freeze): it must not drop out.
_AMENDMENT_LINE = re.compile(r"^(?:>\s*)?\*\*Amendment\s+(\d+)\b.*$", re.M)
# The bold header may wrap across lines: `**Amendment N — <date>, PRE-DATA, …**`
_AMENDMENT_HEADER = re.compile(r"^(?:>\s*)?\*\*Amendment\s+(\d+)\s+—(.*?)\*\*", re.M | re.S)
_TS_KEY = "ts"


class LintError(RuntimeError):
    """The check cannot be performed (exit 2) — never a pass."""


def _git(repo_root: Path, *args: str, check: bool = True) -> str:
    r = subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, timeout=120)
    if r.returncode != 0 and not check:
        return ""
    if r.returncode != 0:
        raise LintError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def token_of(name: str) -> str:
    """``exp53b_cross…`` → ``53b``; ``44b_pilot`` → ``44b``; ``h1_doa_sweep.jsonl`` → ``h1``;
    ``rerun_exp09_2026-09-24`` → ``9`` (M1b PR 4: the ``rerun_`` prefix and leading zeros are not the token)."""
    stem = name.split("/")[-1]
    if stem.startswith(RERUN_PREFIX):
        stem = stem[len(RERUN_PREFIX) :]
    if stem.startswith("exp") and len(stem) > 3 and stem[3].isdigit():
        stem = stem[3:]
    head = stem.split("_", 1)[0].split(".", 1)[0]
    m = re.fullmatch(r"0+(\d.*)", head)
    return m.group(1) if m else head


def parent_token(token: str) -> str | None:
    """``53b`` → ``53``; ``42d53`` → ``42`` (a re-run of 42 under the D53 fix); ``53bd53`` → ``53b``;
    ``53`` → None; ``h1`` → None."""
    m = re.fullmatch(r"(\d+[a-z]*)d\d+", token)
    if m:
        return m.group(1)
    m = re.fullmatch(r"(\d+)[a-z]+", token)
    return m.group(1) if m else None


def parent_chain(token: str) -> list[str]:
    """Every ancestor: ``53bd53`` → [``53b``, ``53``]."""
    out: list[str] = []
    cur = parent_token(token)
    while cur and cur not in out:
        out.append(cur)
        cur = parent_token(cur)
    return out


def is_rerun_prereg(prereg: Path) -> bool:
    """The re-run pre-registration filename form (protocols/TEMPLATE_rerun.md): ``exp<NN>_rerun_…``."""
    return "_rerun_" in prereg.name


def owns(prereg: Path, entry: str) -> bool:
    """Whether ``prereg``'s experiment is the entry's own (its token or an ancestor of it)."""
    tok = token_of(entry)
    return token_of(prereg.name) in {tok, *parent_chain(tok)}


def rerun_by_name(name: str) -> bool:
    """A re-run by its name: ``rerun_…``, a ``NNd<MM>`` token, or a replication / re-baseline / rerun word."""
    return (
        name.startswith(RERUN_PREFIX)
        or bool(re.fullmatch(r"\d+[a-z]*d\d+", token_of(name)))
        or bool(_RERUN_WORDS.search(name))
    )


class Declarations:
    """What one version of a pre-registration declares: its amendments (number, PRE-DATA?, scope or None for
    unscoped), its ``**Scope:**`` entries (a re-run pre-registration governs only these), and the entries its
    ``RB-<n>`` re-baseline sections name."""

    __slots__ = ("amendments", "scope", "rb_entries")

    def __init__(self) -> None:
        self.amendments: list[tuple[int, bool, frozenset[str] | None]] = []
        self.scope: set[str] = set()
        self.rb_entries: set[str] = set()


def _normalise_entry(raw: str) -> str:
    """A declared entry as a top-level name under the data directory: whitespace from a wrapped line removed, a
    trailing ``/`` and a ``docs/experiments/data/`` or ``data/`` prefix dropped."""
    name = "".join(raw.split()).rstrip("/")
    for prefix in (f"{DATA_DIR.as_posix()}/", "data/"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name


def _entry_list(text: str) -> list[str] | None:
    """Backticked entry names separated by ``,`` / ``;`` / ``and`` / ``, and``; None when ``text`` does not open
    with one. Raises LintError for a name that cannot be a top-level data entry."""
    m = _ENTRY_LIST.match(text)
    if not m:
        return None
    names = [_normalise_entry(n) for n in re.findall(r"`([^`]+)`", m.group(0))]
    bad = [n for n in names if not n or "/" in n]
    if bad:
        raise LintError(f"declared entry {bad[0]!r} is not a top-level name under {DATA_DIR}/")
    return names


def parse_declarations(text: str, label: str, *, strict: bool = True) -> Declarations:
    """Raises LintError on a malformed amendment header or an unparseable positional scope. ``strict=False`` reads
    a HISTORICAL version: a header of any shape counts by the class it names (``(pre-data; …)`` included) and
    one naming none is skipped — only the current text must meet the grammar, but a class declared earlier in
    an older shape still dates from when it was declared."""
    d = Declarations()
    for line_m in _AMENDMENT_LINE.finditer(text):
        header_m = _AMENDMENT_HEADER.match(text, line_m.start())
        if not strict:
            loose = re.match(r"(?:>\s*)?\*\*Amendment\s+(\d+)\b(.*?)\*\*", text[line_m.start() :], re.S)
            # The class must be the header's FIRST clause: `(pre-data; …)` or `— <date>, PRE-DATA`. A bold prose
            # line that merely mentions pre-data ("drafted as pre-data but not in force") declares nothing.
            lead = (
                re.match(
                    r"\s*(?:\(\s*(pre-data|post-data)\b|—\s*[^,*]*,\s*(pre-data|post-data)\b)", loose.group(2), re.I
                )
                if loose
                else None
            )
            if lead:
                body = " ".join(loose.group(2).split())
                scope = None
                fm = re.search(r"\b(?:PRE|POST)-DATA\s*,\s*for\s+", body, re.I)
                try:
                    names = _entry_list(body[fm.end() :]) if fm else None
                except LintError:
                    names = None
                if names:
                    scope = frozenset(names)
                kind = (lead.group(1) or lead.group(2)).upper()
                d.amendments.append((int(loose.group(1)), kind == "PRE-DATA", scope))
            continue
        body = " ".join(header_m.group(2).split()) if header_m else ""
        kinds = re.findall(r"\b(PRE-DATA|POST-DATA)\b", body)
        if header_m is None or len(set(kinds)) != 1:
            raise LintError(
                f"{label}: amendment header is not of the shape `**Amendment N — <date>, PRE-DATA|POST-DATA, …**`: "
                f"{line_m.group(0)[:90]!r} — an unclassified amendment cannot be judged and must not drop out"
            )
        after = body[re.search(r"\b(?:PRE|POST)-DATA\b", body).end() :]
        scope: frozenset[str] | None = None
        fm = re.match(r"\s*,\s*for\s+", after)
        if fm:
            names = _entry_list(after[fm.end() :])
            if not names:
                raise LintError(
                    f"{label}: amendment {header_m.group(1)} has `for` after its class but no backticked entry "
                    "(`for \\`<entry>\\`[, \\`<entry>\\`]`) — a scope that does not parse cannot be judged"
                )
            scope = frozenset(names)
        d.amendments.append((int(header_m.group(1)), kinds[0] == "PRE-DATA", scope))
    for m in _SCOPE_LINE.finditer(text):
        try:
            d.scope.update(_entry_list(m.group(1)) or [])
        except LintError:
            if strict:
                raise
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if _RB_START.match(line):
            j = i + 1
            block = [line]
            while j < len(lines) and not _SECTION_END.match(lines[j]):
                block.append(lines[j])
                j += 1
            d.rb_entries.update(re.findall(r"data/([\w.\-]+)", "\n".join(block)))
    return d


def declaration_history(repo_root: Path, ref: str, prereg: Path) -> dict[str, dict]:
    """First times on ``ref``'s first-parent chain (M1b PR 4 — replaces ``git log -S``, which an edited header
    defeats): ``unscoped[N]`` = the first commit where amendment N is PRE-DATA and unscoped; ``scoped[(N, e)]`` =
    the first where N is PRE-DATA and its scope names entry e; ``scope[e]`` = the first where a Scope line names e.
    Re-scoping, a POST→PRE label flip, or an entry added to a scope later all get the LATER time."""
    out: dict[str, dict] = {"unscoped": {}, "scoped": {}, "scope": {}}
    log = _git(repo_root, "log", "--first-parent", "--reverse", "--format=%H %ct", ref, "--", str(prereg))
    for line in log.split("\n"):
        if not line.strip():
            continue
        sha, ct = line.split()
        present = _git(repo_root, "cat-file", "-t", f"{sha}:{prereg.as_posix()}", check=False).strip() == "blob"
        text = _git(repo_root, "show", f"{sha}:{prereg.as_posix()}") if present else ""
        decl = parse_declarations(text, f"{prereg}@{sha[:8]}", strict=False)
        now: dict[str, set] = {"unscoped": set(), "scoped": set(), "scope": set(decl.scope)}
        for num, pre, scope in decl.amendments:
            if pre and scope is None:
                now["unscoped"].add(num)
            elif pre:
                now["scoped"].update((num, e) for e in scope)
        # Only the CURRENT contiguous run counts: a declaration that disappears, or flips to POST-DATA, and comes
        # back later dates from its return (removed-then-re-added, PRE→POST→PRE).
        for kind, keys in now.items():
            for gone in set(out[kind]) - keys:
                del out[kind][gone]
            for k in keys:
                out[kind].setdefault(k, int(ct))
    return out


def prereg_map(repo_root: Path) -> tuple[dict[str, set[Path]], dict[str, set[Path]], list[str]]:
    """(token → preregs, token → result docs naming a prereg, notes)."""
    preregs: dict[str, set[Path]] = {}
    docs: dict[str, set[Path]] = {}
    notes: list[str] = []
    for prereg in sorted((repo_root / PROTOCOLS_DIR).glob("*preregistration*.md")):
        preregs.setdefault(token_of(prereg.name), set()).add(prereg.relative_to(repo_root))
    # The 1.3-era layout: `docs/experiments/<token>_…_prereg.md`, flat beside the result docs (Exp 58-62,
    # R2, R3). Reading only protocols/ left every 1.3.0 experiment ungoverned (roadmap 1.3.1).
    for prereg in sorted((repo_root / EXPERIMENTS_DIR).glob("*_prereg.md")):
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
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj
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
    __slots__ = ("first_ts", "naive", "allow_dirty", "dirty_unallowed", "has_records", "missing_ts", "run_ids")

    def __init__(self) -> None:
        self.first_ts: float | None = None
        self.naive = False
        self.allow_dirty = False
        self.dirty_unallowed = 0
        self.has_records = False
        self.missing_ts = 0  # records (session reports) with no `ts`
        self.run_ids: set[str] = set()


def data_facts(entry: Path) -> DataFacts:
    """A directory holding session ``report.json`` files is judged by those reports ONLY — its time, dirty flag,
    allowance and run ids (restored agent state and harness rows beside them are not read here; binding a sim to
    the harness row that allowed it is the evidence gate's job, M1b decision 4); otherwise every JSON / JSONL file
    under the entry."""
    facts = DataFacts()
    if entry.is_file():
        files = [entry]
    else:
        reports = sorted(entry.rglob("report.json"))
        files = reports or sorted(p for p in entry.rglob("*") if p.suffix in (".json", ".jsonl"))
    for f in files:
        for rec in _records(f):
            facts.has_records = True
            if _allow_flag(rec):
                facts.allow_dirty = True
            elif _dirty_flag(rec):
                facts.dirty_unallowed += 1
            prov = rec.get("provenance") if isinstance(rec.get("provenance"), dict) else {}
            for rid in (rec.get("harness_run_id"), prov.get("harness_run_id")):
                if isinstance(rid, str) and rid:
                    facts.run_ids.add(rid)
            ts, naive = _parse_ts(rec.get(_TS_KEY))
            if ts is None:
                facts.missing_ts += 1
                continue
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


class _Index:
    """Everything the per-entry judgement consults, built once per run."""

    def __init__(self, repo_root: Path, ref: str) -> None:
        self.repo_root, self.ref = repo_root, ref
        by_token, self.docs_by_token, self.notes = prereg_map(repo_root)
        self.decl: dict[Path, Declarations] = {}
        for preregs in by_token.values():
            for prereg in preregs:
                full = repo_root / prereg
                # A linked prereg that does not exist fails only when data it governs is judged (as before).
                self.decl[prereg] = (
                    parse_declarations(full.read_text(errors="replace"), str(prereg))
                    if full.exists()
                    else Declarations()
                )
        # A RE-RUN pre-registration (the template's filename form, `exp<NN>_rerun_…`) governs ONLY the entries its
        # `**Scope:**` line names: it must not retroactively govern the original run's data. The narrowing is keyed
        # on the filename, never on the Scope line alone — a Scope line added to an ORIGINAL pre-registration would
        # otherwise silently take that experiment's data out of the lint; there it is a failure (``self.problems``).
        self.problems: list[str] = []
        self.by_token = {t: {p for p in ps if not is_rerun_prereg(p)} for t, ps in by_token.items()}
        self.scoped: dict[str, set[Path]] = {}
        for prereg, d in self.decl.items():
            if d.scope and not is_rerun_prereg(prereg):
                self.problems.append(
                    f"{prereg}: a `**Scope:**` line belongs only in a re-run pre-registration (`exp<NN>_rerun_…`, "
                    "protocols/TEMPLATE_rerun.md); in an original pre-registration declare a re-run with a scoped amendment"
                )
                continue
            if is_rerun_prereg(prereg) and not d.scope:
                self.problems.append(f"{prereg}: a re-run pre-registration names its entries in a `**Scope:**` line")
            for e in d.scope:
                self.scoped.setdefault(e, set()).add(prereg)
        self.docs_by_prereg: dict[Path, set[Path]] = {}
        for doc in sorted((repo_root / EXPERIMENTS_DIR).glob("*.md")):
            for m in _PREREG_LINK.finditer(doc.read_text(errors="replace")):
                try:
                    rel = (doc.parent / m.group(1)).resolve().relative_to(repo_root)
                except ValueError:
                    continue
                self.docs_by_prereg.setdefault(rel, set()).add(doc.relative_to(repo_root))
        self._history: dict[Path, dict[str, dict]] = {}

    def history(self, prereg: Path) -> dict[str, dict]:
        if prereg not in self._history:
            self._history[prereg] = declaration_history(self.repo_root, self.ref, prereg)
        return self._history[prereg]

    def governing(self, name: str) -> set[Path]:
        tok = token_of(name)
        out = set(self.scoped.get(name, ()))
        for t in [tok, *parent_chain(tok)]:
            out |= self.by_token.get(t, set())
        return out

    def is_rerun(self, name: str) -> bool:
        return rerun_by_name(name) or name in self.scoped or any(name in d.rb_entries for d in self.decl.values())

    @staticmethod
    def explicit_rerun(name: str) -> bool:
        """The re-run forms that need a pre-registration even when the experiment has none: the ``rerun_``
        prefix and the ``NNd<MM>`` token. A replication / re-baseline word alone on an ungoverned entry is out of
        scope, as its original is."""
        return name.startswith(RERUN_PREFIX) or bool(re.fullmatch(r"\d+[a-z]*d\d+", token_of(name)))


def _names_entry(text: str, name: str) -> list[str]:
    """The paragraphs of ``text`` that name ``name`` on a path boundary (``53d53_x`` does not name ``53_x``)."""
    pat = re.compile(r"(?<![\w.\-])" + re.escape(name) + r"(?![\w\-]|\.\w)")
    return [para for para in re.split(r"\n\s*\n", text) if pat.search(para)]


def classify(
    ix: _Index,
    entry: Path,
    *,
    grandfathered: dict[str, str],
    not_governed: dict[str, str],
    ungoverned_reruns: dict[str, str],
) -> dict:
    """One data entry's judgement: ``status`` PASS | FAIL | GRANDFATHERED | UNGOVERNED_RERUN | NOT_GOVERNED |
    OUT_OF_SCOPE | NON_GATED, ``rerun``, ``problems`` and ``notes``. The surface M1b PR 5's evidence gate reads
    (also printed per entry by ``--json``)."""
    repo_root, ref = ix.repo_root, ix.ref
    rel = entry.relative_to(repo_root)
    key, name = rel.as_posix(), entry.name
    out: dict = {"entry": key, "status": "PASS", "rerun": False, "problems": [], "notes": []}
    if any(mk in name for mk in NON_GATED_MARKERS) or entry.suffix in NON_RECORD_SUFFIXES:
        out["status"] = "NON_GATED"
        return out
    governing = ix.governing(name)
    rerun = out["rerun"] = ix.is_rerun(name)
    if key in not_governed:
        if governing:
            out["status"] = "NOT_GOVERNED"
            out["notes"].append(f"NOT GOVERNED (token collision) — {not_governed[key]}")
        else:
            out["status"] = "FAIL"
            out["problems"].append("listed as NOT_GOVERNED but matches no prereg — remove the stale entry")
        return out
    if not governing:
        if not ix.explicit_rerun(name):
            out["status"] = "OUT_OF_SCOPE"
            return out
        if key in ungoverned_reruns:
            out["status"] = "UNGOVERNED_RERUN"
            out["notes"].append(f"UNGOVERNED RE-RUN (listed) — {ungoverned_reruns[key]}")
        else:
            out["status"] = "FAIL"
            out["problems"].append(
                "a re-run whose experiment has no pre-registration: land a re-run pre-registration "
                "(docs/experiments/protocols/TEMPLATE_rerun.md) naming this entry in a `**Scope:**` line before the data"
            )
        return out
    if key in ungoverned_reruns:
        out["status"] = "FAIL"
        out["problems"].append("listed as UNGOVERNED_RERUNS but a pre-registration governs it — remove the stale entry")
        return out
    facts = data_facts(entry)
    when, how, fallback = data_time(repo_root, ref, rel, facts)
    problems, notes = out["problems"], out["notes"]
    added = first_commit_time(repo_root, ref, rel)
    out.update(
        data_time=when,
        data_time_source="fallback" if fallback else "ts",
        first_committed=added,
        run_ids=sorted(facts.run_ids),
        allow_dirty=facts.allow_dirty,
        dirty_unallowed=facts.dirty_unallowed,
        governing=sorted(p.as_posix() for p in governing),
        declared_by=[],
    )
    if fallback:
        notes.append(f"no `ts` in any record — judged at commit granularity ({how})")
        if entry.suffix == ".jsonl" and facts.has_records and added is not None and added >= TS_REQUIRED_FROM:
            problems.append("a .jsonl record file committed after 2026-08-29 must carry epoch `ts` on its records")
    if rerun and (added is None or added >= EXCEPTIONS_FROZEN) and (fallback or facts.missing_ts):
        problems.append("a re-run recorded after the M1b PR 4 freeze carries `ts` on every record (every report.json)")
    if facts.naive:
        problems.append("`ts` is a naive ISO-8601 string (no UTC offset) — its zone is unknowable; write epoch seconds")
    declared = False
    for prereg in sorted(governing):
        if not (repo_root / prereg).exists():
            raise LintError(f"pre-registration {prereg} is named but does not exist in the working tree")
        p_time = first_commit_time(repo_root, ref, prereg)
        if p_time is None:
            problems.append(f"pre-registration {prereg} is not on {ref}")
            continue
        if not p_time < when:
            problems.append(
                f"pre-registration {prereg} reached {ref} at {_fmt(p_time)}, not before the data ({_fmt(when)}, {how})"
            )
        hist = ix.history(prereg)
        for num, pre, scope in ix.decl[prereg].amendments:
            if scope is not None and name not in scope:
                continue
            if not pre:
                notes.append(f"amendment {num} of {prereg.name} is POST-DATA — reported, not judged")
                continue
            a_time = hist["unscoped"].get(num) if scope is None else hist["scoped"].get((num, name))
            if a_time is None:
                problems.append(f"PRE-DATA amendment {num} of {prereg} is not on {ref} in its current form")
            elif not a_time < when:
                hint = "" if scope is not None else " — if it was written for a re-run, scope it: `for \\`<entry>\\``"
                problems.append(
                    f"PRE-DATA amendment {num} of {prereg} reached {ref} at {_fmt(a_time)}, "
                    f"not before the data ({_fmt(when)}, {how}){hint}"
                )
            elif scope is not None:
                declared = True
                out["declared_by"].append(f"{prereg.as_posix()}#amendment-{num}")
        if name in ix.decl[prereg].scope:
            s_time = hist["scope"].get(name)
            if s_time is None:
                problems.append(f"{prereg} names this entry in its Scope line, but not on {ref}")
            elif not s_time < when:
                problems.append(f"{prereg}'s Scope line named this entry on {ref} at {_fmt(s_time)}, after the data")
            else:
                declared = True
                out["declared_by"].append(f"{prereg.as_posix()}#scope")
        later = commits_after(repo_root, ref, prereg, when)
        if later:
            notes.append(
                f"{prereg.name} was touched by {later} commit(s) on {ref} after the data's first ts "
                "— in-place edits without an amendment header are not judged; check them"
            )
    if rerun and not declared:
        problems.append(
            "a re-run needs its own PRE-DATA declaration before its data: an amendment scoped to it "
            "(`**Amendment N — <date>, PRE-DATA, for \\`<entry>\\`, …**`) or a re-run pre-registration's `**Scope:**` line"
        )
    if facts.dirty_unallowed:
        problems.append(
            f"{facts.dirty_unallowed} record(s) stamp working_tree_dirty_src_scripts: true without "
            "allow_dirty: true — a gated record from a dirty tree is refused or explicitly allowed, never silent"
        )
    if facts.allow_dirty:
        docs = set().union(*(ix.docs_by_prereg.get(p, set()) for p in governing))
        echoed = False
        for d in docs:
            text = (repo_root / d).read_text(errors="replace")
            paras = _names_entry(text, name) + [p for rid in facts.run_ids for p in _names_entry(text, rid)]
            echoed |= any("allow_dirty" in para for para in paras)
        if not echoed:
            problems.append(
                "records carry allow_dirty: true but no result doc of this experiment names the entry (or its "
                "harness_run_id) in a paragraph mentioning `allow_dirty` — the write-up must echo the allowance"
            )
    if problems and key in grandfathered:
        out["status"] = "GRANDFATHERED"
        notes.append(f"GRANDFATHERED (still failing) — {grandfathered[key]}")
        notes.extend(f"    {p}" for p in problems)
        out["problems"] = []
    elif problems:
        out["status"] = "FAIL"
    elif key in grandfathered:
        out["status"] = "FAIL"
        out["problems"] = ["listed as GRANDFATHERED but now PASSES — remove the stale entry"]
    return out


def _exception_keys_at_ref(repo_root: Path, ref: str) -> dict[str, set[str]] | None:
    """The exception lists as they stand on the ref (None: the ref's lint predates the M1b PR 4 freeze)."""
    import ast

    rel = "scripts/lint_prereg_precedes_data.py"
    if _git(repo_root, "cat-file", "-t", f"{ref}:{rel}", check=False).strip() != "blob":
        return {}  # the file moved or is gone on the ref: every current key counts as new (fail closed)
    text = _git(repo_root, "show", f"{ref}:{rel}")
    if "EXCEPTIONS_FROZEN" not in text:
        return None  # the ref predates the freeze (this PR's own run)
    out: dict[str, set[str]] = {}
    for node in ast.parse(text).body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Dict):
                out[target.id] = {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    return out


def frozen_list_problems(
    repo_root: Path, ref: str, lists: dict[str, dict[str, str]], frozen_at: float | None
) -> list[str]:
    """The exception lists may only SHRINK against the ref, and may only name entries first committed on the ref
    before the freeze: a new exception goes through the committed exceptions file (M1b decision 1), never
    into this script in the same PR as the data it excuses."""
    if frozen_at is None:
        return []
    out: list[str] = []
    at_ref = _exception_keys_at_ref(repo_root, ref)
    for list_name, entries in lists.items():
        for key in entries:
            if at_ref is not None and key not in at_ref.get(list_name, set()):
                out.append(f"{key}: added to {list_name} after the freeze — exceptions go through the exceptions file")
                continue
            added = first_commit_time(repo_root, ref, Path(key))
            if added is None or added >= frozen_at:
                out.append(f"{key}: {list_name} may only name data first committed on {ref} before the freeze")
                continue
            # The exception covers the data as it was: any later change under the path (a new session in a
            # listed directory, rows appended to a listed file) is new data and is not excused.
            touched = [
                int(t)
                for ref_ in (ref, "HEAD")
                for t in _git(repo_root, "log", "--format=%ct", ref_, "--", key).split()
            ]
            dirty = _git(repo_root, "status", "--porcelain", "--", key).strip()
            if (touched and max(touched) >= frozen_at) or dirty:
                out.append(f"{key}: {list_name} excuses the data as it stood at the freeze; it has changed since")
    return out


JSON_FORMAT_VERSION = "1.0"
# Every status classify() can return. A consumer (the M1b evidence gate) fails closed on any other value: the
# set will grow (e.g. an EXCEPTED status once the exceptions file is read here).
STATUSES = ("PASS", "FAIL", "GRANDFATHERED", "UNGOVERNED_RERUN", "NOT_GOVERNED", "OUT_OF_SCOPE", "NON_GATED")


def _envelope(results: list[dict], failures: list[str]) -> dict:
    """The ``--json`` document: per-entry classifications plus the failures no single entry carries."""
    return {"_format_version": JSON_FORMAT_VERSION, "entries": results, "failures": failures}


def classify_all(repo_root: Path = REPO_ROOT, ref: str = DEFAULT_REF) -> dict:
    """The ``--json`` envelope as a value, for the evidence gate: every data entry classified with the real
    exception lists. Raises LintError when the check cannot run."""
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        code = lint(repo_root, ref, as_json=True)
    if code == 2:
        raise LintError("cannot classify: the prereg lint could not run (see `--json` on the command line)")
    return json.loads(buf.getvalue())


def lint(
    repo_root: Path = REPO_ROOT,
    ref: str = DEFAULT_REF,
    *,
    grandfathered: dict[str, str] | None = None,
    not_governed: dict[str, str] | None = None,
    ungoverned_reruns: dict[str, str] | None = None,
    frozen_at: float | None = None,
    as_json: bool = False,
) -> int:
    repo_root = Path(repo_root).resolve()
    use_real = grandfathered is None and not_governed is None and ungoverned_reruns is None
    grandfathered = GRANDFATHERED if grandfathered is None else grandfathered
    not_governed = NOT_GOVERNED if not_governed is None else not_governed
    ungoverned_reruns = UNGOVERNED_RERUNS if ungoverned_reruns is None else ungoverned_reruns
    if frozen_at is None and use_real:
        frozen_at = EXCEPTIONS_FROZEN
    try:
        if _git(repo_root, "rev-parse", "--is-shallow-repository").strip() == "true":
            raise LintError("shallow repository — fetch full history (git fetch --unshallow) before running this lint")
        _git(repo_root, "rev-parse", "--verify", "--quiet", ref)
    except LintError as exc:
        print(f"ERROR: cannot check prereg-precedes-data: {exc}", file=sys.stderr)
        return 2

    failures: list[str] = []
    notes: list[str] = []
    results: list[dict] = []
    checked = 0
    try:
        ix = _Index(repo_root, ref)
        notes.extend(ix.notes)
        failures.extend(ix.problems)
        data_root = repo_root / DATA_DIR
        names = {e.name for e in data_root.iterdir()} if data_root.exists() else set()
        for prereg, d in sorted(ix.decl.items()):
            named = set(d.scope) | {e for _n, _p, sc in d.amendments if sc for e in sc}
            for e in sorted(named):
                if e not in names:
                    notes.append(f"{prereg.name} declares `{e}`, which is not in {DATA_DIR} (yet)")
                elif not owns(prereg, e):
                    failures.append(f"{prereg}: declares `{e}`, which is not its experiment's data (token mismatch)")
        for entry in sorted(data_root.iterdir()) if data_root.exists() else []:
            r = classify(
                ix, entry, grandfathered=grandfathered, not_governed=not_governed, ungoverned_reruns=ungoverned_reruns
            )
            results.append(r)
            if r["status"] not in ("NON_GATED", "OUT_OF_SCOPE", "NOT_GOVERNED", "UNGOVERNED_RERUN"):
                checked += 1
            notes.extend(f"{r['entry']}: {n}" for n in r["notes"])
            if r["status"] == "FAIL":
                failures.append(f"{r['entry']}:")
                failures.extend(f"    {p}" for p in r["problems"])
        for list_name, entries in (
            ("GRANDFATHERED", grandfathered),
            ("NOT_GOVERNED", not_governed),
            ("UNGOVERNED_RERUNS", ungoverned_reruns),
        ):
            for key in entries:
                if not (repo_root / key).exists():
                    failures.append(f"{key}: {list_name} entry names a file that no longer exists — remove it")
        failures.extend(
            frozen_list_problems(
                repo_root,
                ref,
                {"GRANDFATHERED": grandfathered, "NOT_GOVERNED": not_governed, "UNGOVERNED_RERUNS": ungoverned_reruns},
                frozen_at,
            )
        )
        if checked == 0:
            raise LintError(
                "zero governed data entries — the prereg map or the data glob is broken; refusing to pass vacuously"
            )
    except LintError as exc:
        print(f"ERROR: cannot check prereg-precedes-data: {exc}", file=sys.stderr)
        return 2

    if as_json:
        print(json.dumps(_envelope(results, failures), indent=2))
        return 1 if failures else 0
    for n in notes:
        print(f"NOTE: {n}")
    if failures:
        print("prereg-precedes-data lint FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        print(
            "\nA gated record's pre-registration (and each PRE-DATA amendment) must be ON main before the "
            "first record's ts, a re-run needs its own scoped PRE-DATA declaration, and the record must come from a "
            "clean tree or say allow_dirty — merge the prereg as its own PR first, then run "
            "(docs/agents/simulation-experiments.md §3; docs/lessons/experiment-prereg-precedes-data.md).",
            file=sys.stderr,
        )
        return 1
    print(
        f"prereg-precedes-data lint: clean — {checked} governed data entr{'y' if checked == 1 else 'ies'} "
        f"checked against {ref} (first-parent), {len(grandfathered)} grandfathered, {len(ungoverned_reruns)} "
        f"ungoverned re-runs and {len(not_governed)} not governed, by explicit frozen lists"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", default=DEFAULT_REF, help="the ref that counts as 'on main' (default: origin/main)")
    ap.add_argument("--json", action="store_true", help="print each data entry's classification as JSON")
    args = ap.parse_args(argv)
    return lint(REPO_ROOT, args.ref, as_json=args.json)


if __name__ == "__main__":
    sys.exit(main())
