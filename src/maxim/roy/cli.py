"""CLI dispatcher for ``maxim roy <subcommand>``.

Roy harness — persona-convergence crucible utilities. Mirrors the
``maxim bench`` and ``maxim doctor`` dispatcher pattern: positional
subcommand, argparse-driven flags inside each handler, returns an int
exit code.

Subcommands:

- ``diff <session_a> <session_b> [--json]``: R2. Substrate divergence
  analysis between two ``sim_reports`` session directories.
- ``run <iteration_spec.yaml> [--dry-run]``: R3. Three-arm iteration
  runner that primes arm A, evaluates all three arms on the same
  test scenario, and writes pairwise substrate diffs to
  ``~/.maxim/roy/<iteration_name>/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from maxim.analysis.substrate_diff import substrate_diff, substrate_diff_to_json


def _resolve_session_dir(arg: str) -> Path:
    """Accept a session id (resolved against ``~/.maxim/sim_reports/``)
    or an explicit path. Returns the path as-is if it exists, otherwise
    falls back to the sim_reports default location."""
    p = Path(arg).expanduser()
    if p.is_dir():
        return p
    try:
        from maxim.utils.paths import sim_reports

        candidate = sim_reports() / arg
        if candidate.is_dir():
            return candidate
    except Exception:
        pass
    return p


def _run_diff(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="maxim roy diff",
        description=(
            "Compare two sim_reports session directories and print a "
            "substrate divergence report (NAc / EC / Hippocampus / ATL)."
        ),
    )
    parser.add_argument("session_a", help="Session id under ~/.maxim/sim_reports/ or a path")
    parser.add_argument("session_b", help="Session id under ~/.maxim/sim_reports/ or a path")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON to stdout instead of a text report.",
    )
    args = parser.parse_args(argv)

    dir_a = _resolve_session_dir(args.session_a)
    dir_b = _resolve_session_dir(args.session_b)

    if not dir_a.is_dir():
        print(f"error: session_a not found: {dir_a}", file=sys.stderr)
        return 2
    if not dir_b.is_dir():
        print(f"error: session_b not found: {dir_b}", file=sys.stderr)
        return 2

    diff = substrate_diff(dir_a, dir_b)

    if args.json:
        print(json.dumps(substrate_diff_to_json(diff), indent=2, sort_keys=True))
    else:
        print(diff)
    return 0


def _run_run(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="maxim roy run",
        description=(
            "Run a Roy three-arm iteration: priming + held-out test "
            "across arms A (primed, neutral), B (blank, persona-injected) "
            "and C (blank, neutral), then compute pairwise substrate "
            "divergence. Writes result.json + summary.md to "
            "~/.maxim/roy/<iteration_name>/."
        ),
    )
    parser.add_argument("spec", help="Path to the iteration spec YAML")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the spec and print a one-line summary without running any sims.",
    )
    args = parser.parse_args(argv)

    spec_path = Path(args.spec).expanduser()
    if not spec_path.exists():
        print(f"error: spec not found: {spec_path}", file=sys.stderr)
        return 2

    from maxim.simulation.roy_runner import run_roy_iteration, validate_spec

    if args.dry_run:
        try:
            spec = validate_spec(spec_path)
        except (ValueError, FileNotFoundError) as e:
            print(f"error: invalid spec: {e}", file=sys.stderr)
            return 2
        priming_label = "<inline>" if spec.priming_inline is not None else spec.priming_path.name
        print(
            f"OK: {spec.name} — priming={priming_label} "
            f"test={Path(spec.test_scenario.fixture).name}:{spec.test_scenario.turns}turns "
            f"arms={','.join(f'{k}({spec.arms[k].substrate}/{spec.arms[k].system_prompt})' for k in ('a', 'b', 'c'))}"
        )
        return 0

    try:
        result = run_roy_iteration(spec_path)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    # Print the markdown summary the runner already wrote to disk so
    # the operator sees the per-arm + per-pair outcome inline.
    summary_path = Path(result.artifact_dir) / "summary.md"
    if summary_path.exists():
        print(summary_path.read_text())
    else:
        print(f"Iteration {result.name} complete: artifact_dir={result.artifact_dir}")

    if result.aborted_at is not None:
        return 1
    arms_failed = sum(1 for a in result.arms.values() if a.error is not None)
    return 1 if arms_failed == len(result.arms) else 0


def run_roy_subcommand(argv: Sequence[str]) -> int:
    """Dispatch ``maxim roy <subcommand> [args]``."""
    if not argv:
        print(
            "usage: maxim roy <subcommand> [args]\n\n"
            "subcommands:\n"
            "  diff <session_a> <session_b> [--json]   "
            "substrate divergence analysis between two session dirs\n"
            "  run <iteration_spec.yaml> [--dry-run]   "
            "run a three-arm Roy iteration\n",
            file=sys.stderr,
        )
        return 2

    subcommand, *rest = argv
    if subcommand == "diff":
        return _run_diff(rest)
    if subcommand == "run":
        return _run_run(rest)

    print(
        f"unknown roy subcommand: {subcommand!r}\nknown subcommands: diff, run",
        file=sys.stderr,
    )
    return 2
