#!/usr/bin/env python
"""Fail-loud Stage 2 — measure which instrumented swallow sites actually fire.

WHY THIS EXISTS
---------------
`docs/plans/measurement_path_fail_loud.md` Stage 1 (PR #487) replaced every
silent swallow in the measurement path with a zero-arg
``log_swallowed_exception()`` call, which emits a structured
``swallowed_exception`` event into the ``MAXIM_LOG_FILE`` JSONL. Stage 2 is the
measurement: run the real paths, see which of those sites FIRE, and freeze the
answer as a baseline.

The baseline is not decoration. `docs/plans/god_function_decomposition.md`
states a per-PR behaviour gate — "zero new ``swallowed_exception`` firings vs
the Stage-2 baseline" — and until this ran, that gate cited an artifact that
did not exist. An extraction verified against a nonexistent baseline is the
"assert an enforcement artifact that does not exist" failure the 2026-08-27
score card named.

WHAT THE COMPARISON KEY IS, AND WHY IT IS NOT THE SITE STRING
-------------------------------------------------------------
``log_swallowed_exception()`` derives its site from the caller's frame:
``basename:function:lineno``. That string is EXACTLY the wrong key for this
gate, because the gate's whole purpose is to survive a refactor that moves
code — extraction changes every line number in the file and renames the
enclosing function (``run_agentic_loop`` -> ``_loop_<slug>``). Keying on it
would report 100% "new" sites on a purely mechanical move and 0% signal.

So the gate keys on ``(basename, exc_type)`` and compares two things:

1. **New pairs** — a ``(file, exception type)`` that fired in the candidate
   capture but not in the baseline. This is the fail condition.
2. **Per-pair count increase** — the same pair firing strictly more often.
   Reported, and fails under ``--strict-counts``.

The deliberate blind spot, stated rather than hidden: a NEW swallow of an
exception type that already fires in that same file is invisible to key (1)
and only shows up as a count increase in key (2). Narrowing the key any
further costs refactor-robustness, which is the property the gate exists for.
The raw site strings are kept in the artifact for human reading and for
attributing a firing once the gate has flagged it.

USAGE
-----
    # the denominator: every Stage-1 instrumented site in the tree
    python scripts/fail_loud_stage2.py inventory

    # freeze a baseline from one or more captures
    python scripts/fail_loud_stage2.py baseline \
        --capture generative=/tmp/gen.jsonl --capture substrate=/tmp/sub.jsonl \
        --out docs/experiments/data/fail_loud_stage2/baseline.json

    # gate a candidate capture against it (exit 1 on a new pair)
    python scripts/fail_loud_stage2.py check \
        --capture generative=/tmp/gen2.jsonl \
        --baseline docs/experiments/data/fail_loud_stage2/baseline.json
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _provenance import preflight_gated_record_or_exit  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src" / "maxim"
DEFAULT_BASELINE = REPO_ROOT / "docs" / "experiments" / "data" / "fail_loud_stage2" / "baseline.json"

# The Stage-1 helper name. Only the ZERO-ARG form is a Stage-1 instrumented
# swallow; the legacy explicit form (`log_swallowed_exception(e, operation=...)`)
# is a different, pre-existing debug channel and is not part of this inventory.
HELPER = "log_swallowed_exception"


def _git_hash(cwd: Path) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=30)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# NOTE: there is deliberately no local dirty-tree helper here. The first cut of
# this script had one, and having a second implementation of the repo's
# provenance rule is how it ended up merely STAMPING the flag instead of
# refusing the write. `_provenance.py::preflight_gated_record` is the single
# source of truth; see cmd_baseline.


def enclosing_function(tree: ast.AST, lineno: int) -> str:
    """Innermost function containing `lineno`, or "<module>"."""
    best: tuple[int, str] | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = node.end_lineno or node.lineno
            if node.lineno <= lineno <= end:
                span = end - node.lineno
                if best is None or span < best[0]:
                    best = (span, node.name)
    return best[1] if best else "<module>"


def inventory_sites(src_root: Path = SRC_ROOT) -> list[dict[str, object]]:
    """Every zero-arg `log_swallowed_exception()` call site under `src_root`."""
    sites: list[dict[str, object]] = []
    for path in sorted(src_root.rglob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name != HELPER or node.args or node.keywords:
                continue
            try:
                rel = str(path.relative_to(REPO_ROOT))
            except ValueError:
                # `src_root` outside the repo (tests pass a tmp tree).
                rel = str(path)
            sites.append(
                {
                    "file": rel,
                    "basename": path.name,
                    "function": enclosing_function(tree, node.lineno),
                    "line": node.lineno,
                }
            )
    return sites


def _parse_capture(path: Path) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Extract swallowed_exception firings from a MAXIM_LOG_FILE JSONL.

    Accepts a plain `.jsonl` or a gzipped one; the sha256 is always taken over
    the DECOMPRESSED bytes, so committing a capture gzipped does not change its
    recorded digest. Compression is detected by the gzip MAGIC BYTES, not by
    the `.gz` suffix — a gzipped capture named `.jsonl` (a `gzip -c` redirect,
    a rotated `.gz.1`) would otherwise take the plain branch, fail every
    `json.loads`, and read as zero firings.

    Returns (firings, capture_meta). Malformed lines are counted, not skipped
    silently — a capture we cannot fully parse is a capture we cannot trust.
    """
    firings: list[dict[str, str]] = []
    total = 0
    unparsable = 0
    structured = 0
    digest = hashlib.sha256()
    with path.open("rb") as probe:
        gzipped = probe.read(2) == b"\x1f\x8b"
    opener = gzip.open if gzipped else open
    with opener(path, "rb") as handle:
        for raw in handle:
            digest.update(raw)
            total += 1
            try:
                record = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                unparsable += 1
                continue
            if not isinstance(record, dict):
                continue
            # Two real emitted shapes, both from LogRecord in
            # utils/structured_logging.py. The JSONL handler uses
            # `to_compact()`, which keys the event as "e" and FLATTENS `data`
            # to the top level; `to_verbose()` keys it "event" with a nested
            # "data". A parser written against only the `extra={"event":...,
            # "data":...}` call-site shape reads neither and reports a silent
            # zero — which is indistinguishable from "nothing fired".
            if "e" in record or "event" in record:
                structured += 1
            if record.get("e") == "swallowed_exception":
                data = record
            elif record.get("event") == "swallowed_exception":
                data = record.get("data") or {}
                if not isinstance(data, dict):
                    data = {}
            else:
                continue
            site = str(data.get("site") or "")
            firings.append(
                {
                    "site": site,
                    "basename": site.split(":")[0] if site else "",
                    "function": site.split(":")[1] if site.count(":") >= 2 else "",
                    "exc_type": str(data.get("exc_type") or "None"),
                    "exc": str(data.get("exc") or ""),
                }
            )
    meta = {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "lines": total,
        "unparsable_lines": unparsable,
        "structured_records": structured,
        "firings": len(firings),
    }
    return firings, meta


def _pair_counts(firings: list[dict[str, str]]) -> Counter:
    return Counter((f["basename"], f["exc_type"]) for f in firings)


def _load_captures(specs: list[str]) -> tuple[dict[str, list[dict[str, str]]], dict[str, dict]]:
    by_mode: dict[str, list[dict[str, str]]] = {}
    metas: dict[str, dict] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--capture expects MODE=PATH, got: {spec!r}")
        mode, _, raw_path = spec.partition("=")
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise SystemExit(f"capture not found for mode {mode!r}: {path}")
        if mode in by_mode:
            # Silently keeping only the last one always loses firings, i.e.
            # always moves the gate toward a pass.
            raise SystemExit(f"--capture mode {mode!r} given twice; use distinct mode names")
        firings, meta = _parse_capture(path)
        by_mode[mode] = firings
        metas[mode] = meta
    if not by_mode:
        raise SystemExit("no --capture given")
    return by_mode, metas


def usable_capture_problems(metas: dict[str, dict], *, min_lines: int, max_unparsable_fraction: float) -> list[str]:
    """Reasons a capture set cannot support a verdict.

    A gate that reads an EMPTY capture and reports "no new firings" has
    measured nothing and says it passed — the same vacuous-guard shape
    `check_slow_lane.py` exists to prevent for the slow lane. The failure modes
    are ordinary: `MAXIM_LOG_FILE` unset, pointed at a rotated-away path, or a
    sim that died before logging.
    """
    problems: list[str] = []
    for mode, meta in sorted(metas.items()):
        lines = int(meta["lines"])
        unparsable = int(meta["unparsable_lines"])
        structured = int(meta["structured_records"])
        if lines <= 0 or lines < min_lines:
            problems.append(
                f"capture {mode!r} has {lines} line(s) (minimum {min_lines}) — "
                f"an empty capture cannot support a verdict. Was MAXIM_LOG_FILE set?"
            )
            continue
        fraction = unparsable / lines
        if fraction > max_unparsable_fraction:
            problems.append(
                f"capture {mode!r}: {unparsable}/{lines} lines unparsable "
                f"({fraction:.0%} > {max_unparsable_fraction:.0%}) — not a JSONL capture we can trust"
            )
            continue
        # Line count and JSON-parseability are SHAPE, not provenance. Any
        # well-formed JSONL passes both — including
        # ~/.maxim/util/lane_decisions.jsonl, which has thousands of parseable
        # lines and no swallow events, and would therefore read as a clean
        # pass. Since the committed baseline is all-zeros, "your capture had no
        # swallow events" is the gate's ONLY pass condition, which is exactly
        # what pointing it at the wrong file produces. Require the capture to
        # carry the structured-event field every maxim log record has.
        if structured == 0:
            problems.append(
                f"capture {mode!r}: {lines} parseable line(s) but NONE carry a structured event "
                f'key ("e" or "event") — this does not look like a MAXIM_LOG_FILE capture. '
                f"Pointing --capture at the wrong JSONL reads as a clean pass, so it is refused."
            )
    return problems


def cmd_inventory(args: argparse.Namespace) -> int:
    sites = inventory_sites()
    if args.json:
        print(json.dumps(sites, indent=2))
        return 0
    per_file = Counter(s["file"] for s in sites)
    print(f"Stage-1 instrumented (zero-arg) swallow sites: {len(sites)} in {len(per_file)} files")
    for file, count in sorted(per_file.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {count:3d}  {file}")
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    # This writes under docs/experiments/data/, which is a GATED path. Route
    # through the canonical provenance surface rather than re-deriving one:
    # exit 3 on a dirty src/scripts tree unless --allow-dirty, which then
    # stamps `allow_dirty: true` into the artifact so a write-up cannot omit
    # it. The first cut of this script hand-rolled the dirty check and only
    # STAMPED the flag — detection, not enforcement, which is precisely the
    # Exp 53/53b failure this repo already paid for once.
    provenance = preflight_gated_record_or_exit(REPO_ROOT, args.out, allow_dirty=args.allow_dirty)

    by_mode, metas = _load_captures(args.capture)
    sites = inventory_sites()

    all_firings = [f for mode in by_mode.values() for f in mode]
    pairs = _pair_counts(all_firings)

    fired_sites = sorted({f["site"] for f in all_firings if f["site"]})

    problems = usable_capture_problems(
        metas, min_lines=args.min_lines, max_unparsable_fraction=args.max_unparsable_fraction
    )
    if problems:
        print("refusing to write a baseline from captures that cannot support one:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 2

    payload = {
        "_format_version": "1.0",
        "artifact": "fail_loud_stage2_baseline",
        "plan": "docs/plans/measurement_path_fail_loud.md",
        "generated_utc": args.generated_utc,
        "git_hash": _git_hash(REPO_ROOT),
        "working_tree_dirty_src_scripts": provenance["working_tree_dirty_src_scripts"],
        "allow_dirty": provenance["allow_dirty"],
        "comparison_key": "(basename, exc_type)",
        "instrumented_site_count": len(sites),
        "instrumented_sites": [
            f"{s['basename']}:{s['function']}:{s['line']}" for s in sorted(sites, key=lambda s: (s["file"], s["line"]))
        ],
        "captures": metas,
        "modes": {mode: len(firings) for mode, firings in by_mode.items()},
        "total_firings": len(all_firings),
        "fired_pairs": [
            {"basename": basename, "exc_type": exc_type, "count": count}
            for (basename, exc_type), count in sorted(pairs.items())
        ],
        "fired_sites_raw": fired_sites,
        # DISTINCT sites observed firing, not "instrumented minus fired".
        # An earlier form of this field tried to publish an unfired count by
        # intersecting site strings, and got it wrong in a way that always
        # overstated dead instrumentation: `basename:function:line` rsplit on
        # 2 colons leaves only the BASENAME, so it subtracted "files that
        # fired" from "number of sites" and could never report below
        # 50 - 13 = 37 even if every site fired. The honest position is that a
        # fired site cannot be matched back to an inventory entry across a
        # refactor at all (line numbers move, functions get renamed — the very
        # reason the gate keys on (basename, exc_type)), so no unfired count is
        # published. Compare `instrumented_site_count` against this.
        "distinct_fired_site_count": len(fired_sites),
        "notes": args.note or [],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(f"  instrumented sites : {len(sites)}")
    print(f"  total firings      : {len(all_firings)}")
    print(f"  distinct (file,exc): {len(pairs)}")
    for (basename, exc_type), count in sorted(pairs.items()):
        print(f"    {count:4d}  {basename}  {exc_type}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    baseline_path = Path(args.baseline)
    if not baseline_path.is_file():
        print(f"FAIL: baseline not found: {baseline_path}", file=sys.stderr)
        print("      Run `fail_loud_stage2.py baseline` first — this gate", file=sys.stderr)
        print("      must not pass by citing an artifact that does not exist.", file=sys.stderr)
        return 2
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    base_pairs = Counter(
        {(row["basename"], row["exc_type"]): int(row["count"]) for row in baseline.get("fired_pairs", [])}
    )

    by_mode, metas = _load_captures(args.capture)

    # A gate that reads an empty capture and reports "no new firings" has
    # measured nothing and says it passed. Refuse before comparing.
    problems = usable_capture_problems(
        metas, min_lines=args.min_lines, max_unparsable_fraction=args.max_unparsable_fraction
    )
    if problems:
        print("FAIL: the captures cannot support a verdict:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 2

    # Deleting instrumentation must not read as "fewer firings, therefore
    # green". Rewriting `except Exception: log_swallowed_exception()` into a
    # plain `logger.debug(...)` passes lint_no_silent_swallows (it is
    # handled-and-logged, and that ratchet only blocks counts going UP) and
    # would silently de-instrument the measurement path.
    baseline_sites = int(baseline.get("instrumented_site_count", 0))
    current_sites = len(inventory_sites())
    print(f"instrumented sites: baseline {baseline_sites}, now {current_sites}")
    if not baseline_sites:
        print(
            "WARNING: baseline carries no instrumented_site_count — the de-instrumentation "
            "check cannot run against it. Re-generate the baseline with the current tool."
        )
    if baseline_sites and current_sites < baseline_sites:
        print(
            f"FAIL: instrumented swallow sites fell {baseline_sites} -> {current_sites}. "
            "The measurement path lost instrumentation; a lower firing count is not "
            "evidence of anything. Re-baseline deliberately if the removal is intended.",
            file=sys.stderr,
        )
        return 1

    cand_pairs = _pair_counts([f for mode in by_mode.values() for f in mode])

    new_pairs = [p for p in cand_pairs if p not in base_pairs]
    grown = [(p, base_pairs[p], cand_pairs[p]) for p in cand_pairs if p in base_pairs and cand_pairs[p] > base_pairs[p]]
    gone = [p for p in base_pairs if p not in cand_pairs]

    print(f"baseline : {sum(base_pairs.values())} firings over {len(base_pairs)} (file,exc) pairs")
    print(f"candidate: {sum(cand_pairs.values())} firings over {len(cand_pairs)} (file,exc) pairs")
    for pair in sorted(gone):
        print(f"  gone (ok): {pair[0]}  {pair[1]}")
    for pair, before, after in sorted(grown):
        print(f"  GREW: {pair[0]}  {pair[1]}  {before} -> {after}")
    for pair in sorted(new_pairs):
        print(f"  NEW : {pair[0]}  {pair[1]}  x{cand_pairs[pair]}")

    failed = bool(new_pairs) or (bool(grown) and args.strict_counts)
    if failed:
        print("\nFAIL: new swallowed_exception firings vs the Stage-2 baseline.", file=sys.stderr)
        print("Per docs/plans/god_function_decomposition.md this blocks the PR.", file=sys.stderr)
        return 1
    print("\nOK: no new (file, exception-type) swallow firings vs baseline.")
    return 0


def _add_usability_args(parser: argparse.ArgumentParser) -> None:
    """Thresholds below which a capture cannot support a verdict at all."""
    parser.add_argument(
        "--min-lines",
        type=int,
        default=100,
        help="refuse a capture with fewer log lines than this (default 100). A real run of "
        "either mode produces thousands; anything near zero means the capture did not happen.",
    )
    parser.add_argument(
        "--max-unparsable-fraction",
        type=float,
        default=0.05,
        help="refuse a capture with more than this fraction of non-JSON lines (default 0.05)",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_inv = sub.add_parser("inventory", help="list every Stage-1 instrumented swallow site")
    p_inv.add_argument("--json", action="store_true")
    p_inv.set_defaults(func=cmd_inventory)

    p_base = sub.add_parser("baseline", help="freeze a Stage-2 baseline from captures")
    p_base.add_argument("--capture", action="append", required=True, metavar="MODE=PATH")
    p_base.add_argument("--out", default=str(DEFAULT_BASELINE))
    p_base.add_argument("--generated-utc", default=os.environ.get("STAGE2_GENERATED_UTC", ""))
    p_base.add_argument("--note", action="append")
    p_base.add_argument(
        "--allow-dirty",
        action="store_true",
        help="write the baseline from a dirty src/scripts tree; stamps allow_dirty: true into the artifact",
    )
    _add_usability_args(p_base)
    p_base.set_defaults(func=cmd_baseline)

    p_check = sub.add_parser("check", help="gate a capture against the baseline")
    p_check.add_argument("--capture", action="append", required=True, metavar="MODE=PATH")
    p_check.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    p_check.add_argument(
        "--strict-counts", action="store_true", help="also fail when an existing pair fires more often"
    )
    _add_usability_args(p_check)
    p_check.set_defaults(func=cmd_check)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
