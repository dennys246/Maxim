"""``maxim substrate`` CLI verbs (Hivemind shareability, PR D).

v1_refinement.md §B5 PR D. Operator-visible CLI entry points for the
substrate-bundle round trip. Thin wrappers over
:mod:`maxim.hivemind.bundle` — they parse argv, locate the persisted
NAc + EC files, call :func:`compose_bundle` / :func:`extract_bundle`,
and print a short summary.

Subcommands:

- ``maxim substrate export <output.zip> --session <session_id>
   --contributor-id <id> [--domain X] [--no-identity-filter]``

- ``maxim substrate import <input.zip> --output-dir <dir>``

- ``maxim substrate inspect <input.zip>`` — print manifest only.

The 1.0 ``import`` verb does NOT auto-merge into a live system. It
extracts the bundle so the user (or 1.1+ Oasis software) can decide
what to do with the extracted dicts — typically pass them through
``nac_merge`` / ``ec_merge`` (PR B) and re-load into a live system.
This keeps the CLI side-effect-free at the bio-stack layer.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from maxim.hivemind.bundle import (
    BUNDLE_SCHEMA_VERSION,
    compose_bundle,
    extract_bundle,
    read_bundle_manifest,
)

logger = logging.getLogger(__name__)


def _expand_session_dir(session: str) -> Path:
    """Resolve a ``--session`` argument to an absolute persistence dir.

    Accepts:

    - An absolute or relative path to a directory containing ``aut_nac.json``
      and/or ``aut_ec.json`` (used for sessions outside ``~/.maxim/``).
    - A bare session ID, resolved against ``~/.maxim/sessions/{id}/``.
    """
    candidate = Path(session).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    fallback = Path.home() / ".maxim" / "sessions" / session
    return fallback.resolve()


def _read_optional_json(path: Path) -> dict | None:
    """Read ``path`` and return parsed JSON, or ``None`` if absent."""
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _run_export(args: argparse.Namespace) -> int:
    session_dir = _expand_session_dir(args.session)
    if not session_dir.is_dir():
        print(f"error: session directory not found: {session_dir}", file=sys.stderr)
        return 2

    nac_state = _read_optional_json(session_dir / "aut_nac.json")
    ec_payload = _read_optional_json(session_dir / "aut_ec.json")
    ec_substrate_nodes = ec_payload.get("substrate_nodes") if isinstance(ec_payload, dict) else None

    if nac_state is None and ec_substrate_nodes is None:
        print(
            f"error: no aut_nac.json or aut_ec.json in {session_dir}; nothing to export",
            file=sys.stderr,
        )
        return 2

    output_path = Path(args.output).expanduser().resolve()
    try:
        manifest = compose_bundle(
            nac_state=nac_state,
            ec_substrate_nodes=ec_substrate_nodes,
            output_path=output_path,
            contributor_id=args.contributor_id,
            domain=args.domain,
            apply_identity_filter=not args.no_identity_filter,
            identity_threshold=args.identity_threshold,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    n_slices = len(manifest.get("contents", {}))
    print(
        f"composed bundle at {output_path}\n"
        f"  contributor: {manifest['contributor_id']}\n"
        f"  domain:      {manifest.get('domain')}\n"
        f"  slices:      {n_slices}\n"
        f"  identity_filter: {manifest.get('identity_filter_applied')}"
    )
    return 0


def _run_import(args: argparse.Namespace) -> int:
    bundle_path = Path(args.input).expanduser().resolve()
    if not bundle_path.is_file():
        print(f"error: bundle file not found: {bundle_path}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir).expanduser().resolve()
    try:
        manifest = extract_bundle(bundle_path, output_dir)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f"extracted bundle to {output_dir}\n"
        f"  contributor: {manifest.get('contributor_id')}\n"
        f"  domain:      {manifest.get('domain')}\n"
        f"  schema_version: {manifest.get('schema_version')}\n"
        f"  slices:      {sorted(manifest.get('contents', {}).keys())}"
    )
    print("(use maxim.hivemind.nac_merge / ec_merge to merge into a live system)")
    return 0


def _run_inspect(args: argparse.Namespace) -> int:
    bundle_path = Path(args.input).expanduser().resolve()
    if not bundle_path.is_file():
        print(f"error: bundle file not found: {bundle_path}", file=sys.stderr)
        return 2

    try:
        manifest = read_bundle_manifest(bundle_path)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="maxim substrate", description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    p_export = sub.add_parser("export", help="Compose a substrate bundle from a session")
    p_export.add_argument("output", help="Path to write the .zip bundle to")
    p_export.add_argument(
        "--session",
        required=True,
        help="Session ID (resolved under ~/.maxim/sessions/{id}/) or a path to a session directory",
    )
    p_export.add_argument(
        "--contributor-id",
        required=True,
        help="Opaque ID identifying this Maxim. Must NOT start with '_' (reserved namespace).",
    )
    p_export.add_argument(
        "--domain",
        default=None,
        help='Substrate-domain tag scoping the bundle (e.g. "combat", "cooking"). Default: undomained.',
    )
    p_export.add_argument(
        "--no-identity-filter",
        action="store_true",
        help="Skip the identity-bearing-pattern quarantine (use for trusted-internal backups only).",
    )
    p_export.add_argument(
        "--identity-threshold",
        type=int,
        default=2,
        help="Identity-bearing heuristic threshold (default 2 — bundle-stricter than the heuristic default).",
    )
    p_export.set_defaults(func=_run_export)

    p_import = sub.add_parser("import", help="Extract a substrate bundle to a directory")
    p_import.add_argument("input", help="Path to the .zip bundle")
    p_import.add_argument(
        "--output-dir",
        required=True,
        help="Directory to extract the bundle to (created if absent).",
    )
    p_import.set_defaults(func=_run_import)

    p_inspect = sub.add_parser("inspect", help="Print the bundle manifest without extracting")
    p_inspect.add_argument("input", help="Path to the .zip bundle")
    p_inspect.set_defaults(func=_run_inspect)

    return parser


def run_substrate_subcommand(argv: list[str]) -> int:
    """Entry point dispatched from ``maxim.cli.main`` for ``maxim substrate ...``.

    Returns process exit code. Bundle schema version supported by this
    build is :data:`maxim.hivemind.bundle.BUNDLE_SCHEMA_VERSION`
    (currently ``1``).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "run_substrate_subcommand",
]
