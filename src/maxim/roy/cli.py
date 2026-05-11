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
- ``log <iteration_id> [--plan <path>] [--dry-run]``: R4. Generate
  the reproduction-protocol runbook plus an iteration-log entry on
  the appropriate plan doc from a recorded ``result.json``.
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


def _run_log(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="maxim roy log",
        description=(
            "Generate the reproduction-protocol runbook + iteration-log "
            "entry for a recorded Roy iteration. Writes "
            "docs/experiments/protocols/roy_<id>.md and appends to the "
            "auto-detected plan doc's '## Iteration log' section. "
            "Both artifacts are drafts — operator review required before commit."
        ),
    )
    parser.add_argument("iteration_id", help="Iteration name (matches ~/.maxim/roy/<id>/)")
    parser.add_argument(
        "--plan",
        help=(
            "Explicit plan-doc path for the iteration-log append. "
            "Auto-detected from the iteration name prefix when omitted "
            "(roy-cradle-* → grounded_language_acquisition.md, "
            "roy-{0..5,cautious,adversarial,explorer,collector,hider}-* → persona_convergence_crucible.md)."
        ),
    )
    parser.add_argument(
        "--protocol",
        help="Explicit protocol output path. Defaults to docs/experiments/protocols/roy_<id>.md.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print both artifacts to stdout instead of writing them.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Skip the iteration-log append; only write the protocol runbook.",
    )
    parser.add_argument(
        "--no-protocol",
        action="store_true",
        help="Skip the protocol runbook; only append the iteration-log entry.",
    )
    parser.add_argument(
        "--artifact-root",
        help="Override the artifact root (default ~/.maxim/roy).",
    )
    args = parser.parse_args(argv)

    from maxim.analysis.roy_log import (
        default_protocol_path,
        detect_plan_path,
        generate_iteration_log_entry,
        generate_protocol,
        load_iteration_result,
    )

    artifact_root = Path(args.artifact_root).expanduser() if args.artifact_root else None
    try:
        loaded = load_iteration_result(args.iteration_id, artifact_root=artifact_root)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    # Resolve protocol path
    if args.protocol:
        protocol_path = Path(args.protocol).expanduser()
    else:
        protocol_path = default_protocol_path(args.iteration_id)
        if protocol_path is None and not args.no_protocol:
            print(
                "error: cannot locate repo root for default protocol path; pass --protocol <path> or --no-protocol.",
                file=sys.stderr,
            )
            return 2

    # Resolve plan path
    plan_path: Path | None = None
    if not args.no_log:
        if args.plan:
            plan_path = Path(args.plan).expanduser()
        else:
            plan_path = detect_plan_path(args.iteration_id)
            if plan_path is None:
                print(
                    f"error: cannot auto-detect plan doc for iteration {args.iteration_id!r}; "
                    f"pass --plan <path> or --no-log.",
                    file=sys.stderr,
                )
                return 2
        if not plan_path.is_file():
            print(f"error: plan doc not found: {plan_path}", file=sys.stderr)
            return 2

    # Generate protocol
    if not args.no_protocol and protocol_path is not None:
        try:
            rendered = generate_protocol(loaded, protocol_path, dry_run=args.dry_run)
        except (FileExistsError, OSError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        if args.dry_run:
            print(f"--- protocol ({protocol_path}) ---")
            print(rendered)
        else:
            print(f"wrote protocol: {protocol_path}")

    # Generate iteration-log entry
    if not args.no_log and plan_path is not None:
        try:
            block = generate_iteration_log_entry(loaded, plan_path, dry_run=args.dry_run)
        except (ValueError, FileNotFoundError, OSError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        if args.dry_run:
            print(f"--- iteration log entry ({plan_path}) ---")
            print(block)
        else:
            print(f"appended iteration-log entry to: {plan_path}")

    return 0


def run_roy_subcommand(argv: Sequence[str]) -> int:
    """Dispatch ``maxim roy <subcommand> [args]``."""
    if not argv:
        print(
            "usage: maxim roy <subcommand> [args]\n\n"
            "subcommands:\n"
            "  diff <session_a> <session_b> [--json]   "
            "substrate divergence analysis between two session dirs\n"
            "  run <iteration_spec.yaml> [--dry-run]   "
            "run a three-arm Roy iteration\n"
            "  log <iteration_id> [--plan PATH] [--dry-run]   "
            "generate protocol runbook + iteration-log entry\n",
            file=sys.stderr,
        )
        return 2

    subcommand, *rest = argv
    if subcommand == "diff":
        return _run_diff(rest)
    if subcommand == "run":
        return _run_run(rest)
    if subcommand == "log":
        return _run_log(rest)

    print(
        f"unknown roy subcommand: {subcommand!r}\nknown subcommands: diff, run, log",
        file=sys.stderr,
    )
    return 2
