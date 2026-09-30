#!/usr/bin/env python3
"""Every harness that produces experiment records must run the provenance guard.

Three families, three doors, one rule (a result whose code-under-test cannot be
established is not a validation):

1. **Sub-sim spawners** (the Exp 42b retraction, CLAUDE.md first lesson): a
   ``scripts/**/*.py`` that spawns the maxim runtime (subprocess + a maxim
   invocation pattern) must reference ``assert_repo_interpreter`` — the
   sub-sims imported a DIFFERENT checkout than the one under test, silently,
   with authoritative-looking JSONL. Since 2026-08-29 it must ALSO run the
   gated-record preflight (``preflight_gated_record[_or_exit]`` or
   ``executed_code_provenance(..., out_path=...)``): item 16.7 covers every
   harness writing under ``docs/experiments/data/``, not only the in-process
   family — the first draft of this widening left the Exp 52/54 flagship
   harness (``benchmark_cradle_mother.py``) unrefused.

2. **In-process harnesses** (the Exp 53/53b release-day incident, roadmap
   1.1.x item 16.7, docs/lessons/experiment-prereg-precedes-data.md): a
   ``scripts/orient_*/**/*.py`` that writes records (``json.dump(``,
   ``write_text(``, ``open(..., "w"/"a")``) must reference the gated-record
   preflight — ``preflight_gated_record`` / ``in_process_code_provenance``
   (``scripts/_provenance.py``) or the family's guarded writer ``JsonlLog(``
   (``scripts/orient_backbone/live_common.py``, whose constructor runs the
   preflight). The lint also asserts that ``live_common.JsonlLog`` still
   references the preflight, so the delegation cannot rot silently. The
   Exp 53 harness *stamped* ``working_tree_dirty_src_scripts: true`` into
   every start record and kept going — stamping is detection, refusing is
   enforcement, and this family was outside family 1's regex.

3. **Gated-path writers anywhere under ``scripts/``** (added 2026-08-30, the
   1.1.2 review round): any ``scripts/**/*.py`` that names
   ``docs/experiments/data`` and writes records must run one of the sanctioned
   guards. Families 1 and 2 both key on HOW a harness runs — it spawns
   ``maxim``, or it lives under ``scripts/orient_*/`` — so
   ``scripts/fail_loud_stage2.py`` matched neither while writing a new
   artifact into the gated tree, and it hand-rolled a dirty check that only
   STAMPED the flag. This family keys on WHERE records land, which is what the
   rule is actually about, and accepts every guard form the other two accept.
   A file already flagged by family 1 or 2 is not reported twice.
   Note it keys on the LITERAL string ``docs/experiments/data``: a harness
   assembling the path from segments escapes it, per the convention below.

Since 2026-09-29 (M1b, #1003) family 1 must also USE the stamp, not only run the preflight, checked on the
AST: it calls ``executed_code_provenance(..., out_path=...)`` (the preflight alone was how Exp 37 and Exp 41
discarded their provenance) and names ``"record_kind"``; and a harness that spawns ``"--sim"`` sets
``env["MAXIM_HARNESS_RUN_ID"]`` literally, calls ``harness_run_id(``, finds its report through
``find_own_report``/``spawn_evidence`` (never the newest directory) and echoes it with ``sim_evidence(``.
Since M1b PR 5a a ``"--sim"`` spawner may not call ``in_process_code_provenance``: the provenance block's
``harness_family`` (``"in_process"`` / ``"spawning"``, stamped by ``_provenance`` itself) tells the evidence gate
whether to judge a row by its own provenance or by the sims it echoes, so a spawner claiming ``in_process`` would
be judged without its sims.
Presence checks: one literal covers a file, and they catch forgetting, not a wrong argument.

False positives (a script whose match is not a record write / sub-sim spawn)
opt out with a line containing ``# provenance-exempt:`` followed by the reason.

This lint catches FORGETTING, not evasion (house convention for heuristic
lints): a docstring mention of the guard name counts as compliance, one
exempt marker exempts the whole file, and ``shell=True`` string spawns,
``os.system`` writes, ``shutil.copy``/``move``, ``atomic_write_json``,
``.save(`` and hand-rolled ``os.replace`` escape the regexes. It is a forcing
function for the honest author, not a security boundary.

Exits: 0 clean; 1 violations (stderr).
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# A spawn of the maxim runtime, in list-arg or console-script form:
#   [sys.executable, "-m", "maxim", ...]   /  ["maxim", "--sim", ...]
#   shutil.which("maxim") / .venv/bin/maxim used as argv[0]
_MAXIM_SPAWN = re.compile(
    r"""(
        -m['"],\s*['"]maxim['"]            # [..., "-m", "maxim", ...]
      | ['"]maxim['"],\s*['"]--            # ["maxim", "--sim"/"--goal"...]
      | which\(['"]maxim['"]\)             # shutil.which("maxim")
      | bin/maxim['"]                      # .venv/bin/maxim path literal
    )""",
    re.VERBOSE,
)

# A record write in the in-process family.
_RECORD_WRITE = re.compile(
    r"""(
        json\.dump\(                        # json.dump(obj, fh)
      | \.write_text\(                      # Path.write_text(...)
      | \bopen\([^)\n]*['"][wa]b?['"]       # open(path, "w") / "a" / "wb" / "ab"
      | \bopen\([^)\n]*mode=['"][wa]b?['"]  # open(path, mode="w")
      | \.open\(['"][wa]b?['"]              # Path.open("w")
    )""",
    re.VERBOSE,
)
_SPAWNER_GATE = re.compile(r"preflight_gated_record|executed_code_provenance\([^)]*out_path=", re.S)
IN_PROCESS_FAMILY_GLOB = "orient_*/**/*.py"
GUARDED_WRITER = Path("scripts/orient_backbone/live_common.py")
_IN_PROCESS_GUARDS = ("preflight_gated_record", "in_process_code_provenance", "JsonlLog(")
EXEMPT_MARKER = "# provenance-exempt:"

# Family 3 — ANY script that names the gated data directory and writes records.
#
# Added 2026-08-30 after `scripts/fail_loud_stage2.py` (1.1.2 Cluster A) wrote a
# new artifact into docs/experiments/data/ while escaping both existing
# families: it spawns no `maxim` (so not Family 1) and lives at the top level of
# scripts/ rather than under orient_*/ (so not Family 2). It hand-rolled its own
# dirty-tree check and only STAMPED the flag — detection, not enforcement, the
# exact Exp 53/53b shape — and its own review round caught the resulting artifact
# claiming a clean tree over a `dirty: true` stamp.
#
# The families above are keyed on HOW a harness runs; this one is keyed on WHERE
# it writes, which is what the rule is actually about.
GATED_DIR_REFERENCE = re.compile(r"docs/experiments/data")


def _names_called(tree: ast.AST) -> dict[str, list[ast.Call]]:
    """Every call in ``tree`` by the called name (``f(...)`` and ``mod.f(...)`` both key ``f``)."""
    out: dict[str, list[ast.Call]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
            if name:
                out.setdefault(name, []).append(node)
    return out


def _stamps_provenance(calls: dict[str, list[ast.Call]]) -> bool:
    """``executed_code_provenance(..., out_path=...)``: the call that both refuses a dirty gated write and
    returns the block the harness stamps."""
    return any(any(k.arg == "out_path" for k in c.keywords) for c in calls.get("executed_code_provenance", []))


def _string_constants(tree: ast.AST) -> set[str]:
    return {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def _sets_run_id_env(tree: ast.AST) -> bool:
    """``<env>["MAXIM_HARNESS_RUN_ID"] = ...`` somewhere (the explicit hand-off to each sub-sim)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (
                    isinstance(t, ast.Subscript)
                    and isinstance(t.slice, ast.Constant)
                    and t.slice.value == "MAXIM_HARNESS_RUN_ID"
                ):
                    return True
    return False


def _spawns_sim(tree: ast.AST) -> bool:
    """A ``"--sim"`` inside a list literal: a command line that runs a simulation (not an argparse flag)."""
    return any(
        isinstance(n, ast.List) and any(isinstance(e, ast.Constant) and e.value == "--sim" for e in n.elts)
        for n in ast.walk(tree)
    )


def _stamp_failures(rel: Path, tree: ast.AST) -> list[str]:
    """Family 1's #1003 checks: the provenance block is stamped, records are typed, and a sim spawner binds
    its sims by run id."""
    calls = _names_called(tree)
    out: list[str] = []
    if not _stamps_provenance(calls):
        out.append(
            f"{rel}: spawns maxim sub-sims but never stamps executed_code_provenance(..., out_path=...) — "
            "the preflight alone refuses a dirty tree but leaves every record unable to say which code ran "
            "(#1003: Exp 37 and Exp 41 discarded it); stamp the returned block into every record"
        )
    if "record_kind" not in _string_constants(tree):
        out.append(f'{rel}: spawns maxim sub-sims but its records carry no "record_kind" (M1b, #1003)')
    if _spawns_sim(tree):
        missing = [
            what
            for what, ok in (
                ('env["MAXIM_HARNESS_RUN_ID"] = run_id on the spawn env', _sets_run_id_env(tree)),
                ("harness_run_id()", "harness_run_id" in calls),
                ("find_own_report()/spawn_evidence()", "find_own_report" in calls or "spawn_evidence" in calls),
                ("sim_evidence()", "sim_evidence" in calls),
            )
            if not ok
        ]
        if missing:
            out.append(
                f"{rel}: spawns `maxim --sim` without binding its sims by harness run id — missing "
                f"{', '.join(missing)} (M1b, #1003: find the report the spawn wrote, never the newest directory, "
                "and echo its evidence into the row)"
            )
        if "in_process_code_provenance" in calls:
            out.append(
                f"{rel}: spawns `maxim --sim` but calls in_process_code_provenance — that stamps "
                'harness_family: "in_process", which the evidence gate judges by the row\'s own provenance '
                "instead of the sims it spawned (M1b PR 5a); a spawner stamps executed_code_provenance only"
            )
    return out


def lint(repo_root: Path = REPO_ROOT) -> list[str]:
    """Return the violation messages for the scripts tree under ``repo_root``."""
    failures: list[str] = []
    # One defect, one message. Family 3 overlaps both earlier families by
    # design (it keys on WHERE records land, they key on HOW the harness
    # runs), so without this an orient_*/ violator reported twice and an
    # author reading the failure list would over-count.
    flagged: set[Path] = set()
    scripts = repo_root / "scripts"

    # Family 1 — sub-sim spawners.
    for path in sorted(scripts.rglob("*.py")):
        rel = path.relative_to(repo_root)
        if rel.as_posix() == "scripts/_provenance.py":
            continue  # the guard itself: its docstring quotes the spawn shape it exists to guard
        text = path.read_text(errors="replace")
        if "subprocess" not in text or not _MAXIM_SPAWN.search(text):
            continue
        if EXEMPT_MARKER in text:
            continue
        if "assert_repo_interpreter" not in text:
            failures.append(
                f"{rel}: spawns maxim sub-sims without the provenance preflight — "
                "call scripts/_provenance.py::assert_repo_interpreter before the "
                "first spawn (exit 3 on mismatch) and stamp executed_code_provenance "
                "into every run record, or mark a false positive with "
                f"'{EXEMPT_MARKER} <reason>' (Exp 42b lesson)"
            )
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            failures.append(f"{rel}: spawns maxim sub-sims but does not parse — cannot check its provenance")
            flagged.add(path)
            continue
        stamp_failures = _stamp_failures(rel, tree)
        if stamp_failures:
            failures.extend(stamp_failures)
            flagged.add(path)

    # Family 2 — in-process record writers. The guarded writer is the delegate,
    # so it must itself reference the preflight (positive control on the delegation).
    writer = repo_root / GUARDED_WRITER
    writer_text = writer.read_text(errors="replace") if writer.exists() else ""
    writer_guarded = "preflight_gated_record" in writer_text and "class JsonlLog" in writer_text
    if not writer_guarded:
        failures.append(
            f"{GUARDED_WRITER}: JsonlLog no longer references preflight_gated_record — the in-process "
            "family's record writer must run the gated-record preflight in its constructor (item 16.7)"
        )
    for path in sorted(scripts.glob(IN_PROCESS_FAMILY_GLOB)):
        rel = path.relative_to(repo_root)
        if path == writer:
            continue
        text = path.read_text(errors="replace")
        if not _RECORD_WRITE.search(text):
            continue
        if EXEMPT_MARKER in text:
            continue
        if any(g in text for g in _IN_PROCESS_GUARDS):
            continue
        failures.append(
            f"{rel}: in-process harness writes records without the gated-record preflight — "
            "write through live_common.JsonlLog (guarded) or call scripts/_provenance.py::"
            "preflight_gated_record[_or_exit] / in_process_code_provenance with the output path "
            "(exit 3 on a dirty tree unless --allow-dirty, which stamps allow_dirty: true), or mark a "
            f"false positive with '{EXEMPT_MARKER} <reason>' (Exp 53/53b lesson, item 16.7)"
        )
        flagged.add(path)

    # Family 3 — gated-path writers anywhere under scripts/, keyed on WHERE the
    # records land rather than on how the harness runs.
    for path in sorted(scripts.rglob("*.py")):
        rel = path.relative_to(repo_root)
        if path == writer or path.name == "_provenance.py" or path in flagged:
            continue
        text = path.read_text(errors="replace")
        if not GATED_DIR_REFERENCE.search(text):
            continue
        if not _RECORD_WRITE.search(text):
            continue
        if EXEMPT_MARKER in text:
            continue
        # Any of the three sanctioned guard forms counts: the in-process
        # preflight, the delegating JsonlLog writer, or the spawner's
        # `executed_code_provenance(..., out_path=)` — which is how
        # scripts/exp44/campaign.py is (correctly) guarded. Family 3 keys on
        # WHERE records land, not on which family a harness belongs to, so it
        # must accept every form the other two families accept.
        if any(g in text for g in _IN_PROCESS_GUARDS) or _SPAWNER_GATE.search(text):
            continue
        try:
            if _stamps_provenance(_names_called(ast.parse(text))):
                continue  # the spawner's form, however its arguments wrap
        except SyntaxError:
            pass
        failures.append(
            f"{rel}: writes records and names docs/experiments/data/ but runs no gated-record "
            "preflight — call scripts/_provenance.py::preflight_gated_record[_or_exit] with the "
            "output path (exit 3 on a dirty tree unless --allow-dirty, which stamps "
            f"allow_dirty: true), or mark a false positive with '{EXEMPT_MARKER} <reason>'. "
            "Stamping a dirty flag is detection; refusing to write is enforcement (item 16.7)"
        )
    return failures


def main() -> int:
    failures = lint()
    if failures:
        print("harness-provenance lint FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    print("harness-provenance lint: clean (sub-sim spawners + in-process record writers + gated-path writers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
