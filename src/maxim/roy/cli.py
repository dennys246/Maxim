"""CLI dispatcher for ``maxim roy <subcommand>``.

Roy harness session R2 of 5 — substrate divergence analysis between
two sim_reports session directories. Mirrors the ``maxim bench`` and
``maxim doctor`` dispatcher pattern: positional subcommand,
argparse-driven flags inside each handler, returns an int exit code.
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


def run_roy_subcommand(argv: Sequence[str]) -> int:
    """Dispatch ``maxim roy <subcommand> [args]``."""
    if not argv:
        print(
            "usage: maxim roy <subcommand> [args]\n\n"
            "subcommands:\n"
            "  diff <session_a> <session_b> [--json]   "
            "substrate divergence analysis between two session dirs\n",
            file=sys.stderr,
        )
        return 2

    subcommand, *rest = argv
    if subcommand == "diff":
        return _run_diff(rest)

    print(
        f"unknown roy subcommand: {subcommand!r}\nknown subcommands: diff",
        file=sys.stderr,
    )
    return 2
