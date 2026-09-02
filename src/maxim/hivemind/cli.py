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

- ``maxim substrate merge-nac <source_nac.json> [--into <target_nac.json>]
   --source-id <id> [--target-id <id>]`` — one-shot MERGE of a trained
   NAc policy file into a runtime ``nac.json`` (live_audio_orient_wiring.md
   Stage 4b). A merge, never a replace: ``NAc.load_safe`` would clobber
   the runtime's other learning. ONE-SHOT by design — re-running the same
   import double-counts Welford observations (which is why this is a CLI
   verb the operator invokes consciously, not a boot-time flag). The
   policy-meta sidecar (``*.meta.json`` — bin boundary, gain,
   action_deltas) travels with the import; a target sidecar that
   DISAGREES with the source's aborts before any mutation (merging
   policies trained in different state spaces silently corrupts both).

The 1.0 ``import`` verb does NOT auto-merge into a live system. It
extracts the bundle so the user (or 1.1+ Oasis software) can decide
what to do with the extracted dicts — typically pass them through
``substrate_merge`` and re-load into a live system via
``EC.ingest_substrate_nodes`` + ``NAc.load_state``.
This keeps the CLI side-effect-free at the bio-stack layer.

**Do not hand-compose ``nac_merge`` + ``ec_merge`` here**, which is what
this paragraph said until 2026-09-02: that sequence merges the two
slices INDEPENDENTLY and discards the EC alignment, so the donor's
``cluster_reward_bias`` keys name clusters the receiver has no node for
and the merged want reads out as exactly 0.0 while the bias dict grows
(D43). A bundle carries both slices precisely so the aligned merge is
possible; ``substrate_merge`` is that composition.
``merge-nac`` is the deliberate exception: a file-level merge verb that
still never touches a LIVE bio-stack (it rewrites the persisted JSON;
the runtime picks it up at next boot via the Stage-4a load path).
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
    # Encode-time encoder stamps (artifact stamping, 1.1 item 7) — read
    # from the payload the writing system produced, NEVER fabricated here:
    # this CLI's own encoder singleton need not match the one that wrote
    # the session, and stamping the wrong provenance is exactly the
    # calibration leak the stamp exists to prevent. Pre-stamping payloads
    # carry None (honest unknown).
    ec_encoder_provenance = ec_payload.get("encoder_provenance") if isinstance(ec_payload, dict) else None

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
            encoder_provenance=ec_encoder_provenance,
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
    print(
        "(use maxim.hivemind.substrate_merge to merge into a live system, then\n"
        " EC.ingest_substrate_nodes + NAc.load_state to apply the result)"
    )
    return 0


def _meta_sidecar_path(nac_path: Path) -> Path:
    """``foo/nac.json`` → ``foo/nac.meta.json`` (the policy-meta sidecar).

    Mirrors ``scripts/orient_backbone/live_common.py::_meta_path``: the
    state-space definition (bin boundary, gain, action_deltas) travels
    WITH the policy, because a bin name like ``near_left`` means nothing
    without the boundary that produced it.
    """
    name = nac_path.name
    stem = name[:-5] if name.endswith(".json") else name
    return nac_path.with_name(stem + ".meta.json")


def _run_merge_nac(args: argparse.Namespace) -> int:
    source_path = Path(args.source).expanduser().resolve()
    if not source_path.is_file():
        print(f"error: source NAc file not found: {source_path}", file=sys.stderr)
        return 2

    if args.into:
        target_path = Path(args.into).expanduser().resolve()
    else:
        from maxim.utils.paths import user_memory

        target_path = user_memory() / "nac.json"

    # ── Fail fast BEFORE any mutation ────────────────────────────────
    # State-space compatibility: a target trained at a different bin
    # boundary/gain would silently mis-bin every lookup post-merge.
    # Corrupt inputs (truncated JSON, list-rooted files) get the same
    # clean rc=2 contract as every other failure — never a traceback.
    def _read_dict_or_none(path: Path, label: str) -> "dict | None | int":
        try:
            data = _read_optional_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"error: cannot read {label} ({path}): {exc}", file=sys.stderr)
            return 2
        if data is not None and not isinstance(data, dict):
            print(f"error: {label} is not a JSON object ({path}). Nothing was written.", file=sys.stderr)
            return 2
        return data

    src_meta = _read_dict_or_none(_meta_sidecar_path(source_path), "source policy-meta sidecar")
    if src_meta == 2:
        return 2
    tgt_meta = _read_dict_or_none(_meta_sidecar_path(target_path), "target policy-meta sidecar")
    if tgt_meta == 2:
        return 2

    def _meta_essence(meta: "dict | None") -> "dict | None":
        # Compare state-space CONTENT only — the CC1 ``_format_version``
        # stamp is bookkeeping, and comparing it would false-abort
        # imports whose sidecars differ only by stamping history.
        if meta is None:
            return None
        return {k: v for k, v in meta.items() if k != "_format_version"}

    if src_meta is not None and tgt_meta is not None and _meta_essence(src_meta) != _meta_essence(tgt_meta):
        print(
            "error: policy-meta sidecars disagree — the two NAc files were "
            "trained in different state spaces; merging would silently "
            "corrupt both. Nothing was written.\n"
            f"  source meta ({_meta_sidecar_path(source_path)}): {json.dumps(_meta_essence(src_meta), sort_keys=True)}\n"
            f"  target meta ({_meta_sidecar_path(target_path)}): {json.dumps(_meta_essence(tgt_meta), sort_keys=True)}",
            file=sys.stderr,
        )
        return 2

    try:
        source_state = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot read source NAc file: {exc}", file=sys.stderr)
        return 2
    if not isinstance(source_state, dict):
        print(f"error: source NAc file is not a JSON object ({source_path}).", file=sys.stderr)
        return 2
    target_state = _read_dict_or_none(target_path, "target NAc file")
    if target_state == 2:
        return 2
    target_state = target_state or {}
    target_existed = target_path.is_file()

    # The persisted files carry the CC1 ``_format_version`` stamp; the
    # merge consumes pure ``NAc.dump()`` shapes, and ``with_format_version``
    # fails loudly on a conflicting leftover stamp — strip both.
    source_state.pop("_format_version", None)
    target_state.pop("_format_version", None)

    from maxim.hivemind.merge import NAC_KEY_SEP, nac_merge

    try:
        merged = nac_merge(
            target_state,
            source_state,
            left_source=args.target_id,
            right_source=args.source_id,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # D43 guard: `merge-nac` is scoped to SAME-substrate policy import — a
    # trained policy folded into the runtime NAc of the same body, where the
    # encoder is shared and `cluster_reward_bias`'s (agent, cluster, tool)
    # keys therefore match by construction. That use is correct and stays
    # correct.
    #
    # Across DIFFERENT substrates it is not, and it fails silently: cluster
    # ids are `uuid4()` per substrate, so nothing aligns, the donor's biases
    # land under ids the target has no node for, and the summary below prints
    # a LARGER "cluster biases" count than before — `len()` is the union,
    # maximal exactly when nothing matched. The success line moves the wrong
    # way. Detect the condition (donor has biases, zero cluster overlap) and
    # say so rather than reporting a clean merge.
    def _cluster_ids(state: dict) -> set:
        out = set()
        for key in state.get("cluster_reward_bias") or {}:
            parts = str(key).split(NAC_KEY_SEP)
            if len(parts) == 3:
                out.add(parts[1])
        return out

    src_clusters = _cluster_ids(source_state)
    tgt_clusters = _cluster_ids(target_state)
    if src_clusters and tgt_clusters and not (src_clusters & tgt_clusters):
        print(
            "warning: the two NAc files share NO cluster ids "
            f"({len(src_clusters)} in source, {len(tgt_clusters)} in target, 0 in common).\n"
            "  This is the signature of a CROSS-SUBSTRATE merge, which merge-nac "
            "cannot do correctly: cluster ids are per-substrate uuid4, so every\n"
            "  merged cluster bias below will name a cluster this target has no "
            "node for and will read out as exactly 0.0 (D43). The bias count\n"
            "  in the summary will still GROW — it is the union size, not a "
            "measure of transfer.\n"
            "  For a cross-substrate merge use maxim.hivemind.substrate_merge, "
            "which aligns the two ECs first and re-keys the biases through the\n"
            "  resulting id map; it needs both sides' EC substrate_nodes, which "
            "a substrate bundle carries and a bare nac.json does not.",
            file=sys.stderr,
        )

    # Preserve the decay clock (pre-merge review fold): ``nac_merge``
    # rebuilds from a fixed field list and drops ``saved_at``; without it
    # the next boot's ``load_safe(apply_decay=True)`` finds no stamp and
    # skips wall-clock decay for ALL runtime biases once. The TARGET's
    # stamp is the right clock for the pre-existing state (the trained
    # policy is frozen — extra decay on it is not owed either way).
    saved_at = target_state.get("saved_at") or source_state.get("saved_at")
    if saved_at:
        merged["saved_at"] = saved_at

    from maxim.decisions.nac import _NAC_FORMAT_VERSION
    from maxim.utils.atomic_io import atomic_write_json, atomic_write_text
    from maxim.utils.format_version import with_format_version

    # Pre-merge backup of the target (the one destructive-ish step) —
    # atomic, so a crash mid-backup can't leave a truncated .bak as the
    # only rollback artifact.
    if target_existed:
        backup_path = target_path.with_name(target_path.name + ".pre-merge.bak")
        try:
            atomic_write_text(str(backup_path), target_path.read_text(encoding="utf-8"))
        except OSError as exc:
            print(f"error: cannot write pre-merge backup ({backup_path}): {exc}. Nothing was written.", file=sys.stderr)
            return 2

    target_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(str(target_path), with_format_version(merged, version=_NAC_FORMAT_VERSION))

    # The sidecar travels with the import (make-the-definition-travel).
    # Stamped per CC1 — every persisted JSON Maxim writes carries
    # ``_format_version``; the equality gate above compares stamp-stripped
    # essence, so stamping cannot false-abort a legitimate re-import.
    if src_meta is not None:
        atomic_write_json(str(_meta_sidecar_path(target_path)), with_format_version(dict(src_meta)))

    print(
        f"merged {source_path.name} into {target_path}\n"
        f"  links:           {len(merged.get('links', {}))} event signatures\n"
        f"  cluster biases:  {len(merged.get('cluster_reward_bias', {}))}\n"
        f"  reward biases:   {len(merged.get('reward_bias', {}))}\n"
        f"  observations:    {merged.get('total_observations', 0)}\n"
        f"  policy meta:     {'copied' if src_meta is not None else 'none'}\n"
        + (f"  backup:          {target_path.name}.pre-merge.bak\n" if target_existed else "")
        + "NOTE: one-shot — re-running the same import double-counts Welford observations."
    )
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

    p_merge_nac = sub.add_parser(
        "merge-nac",
        help="One-shot MERGE of a trained NAc policy file into a runtime nac.json (never a replace)",
    )
    p_merge_nac.add_argument("source", help="Path to the trained policy NAc JSON (NAc.save output)")
    p_merge_nac.add_argument(
        "--into",
        default=None,
        help="Target nac.json to merge into (default: ~/.maxim/memory/nac.json — the runtime NAc's Stage-4a persistence path). Created if absent.",
    )
    p_merge_nac.add_argument(
        "--source-id",
        required=True,
        help="Opaque contributor ID for the imported policy (e.g. 'reachy-orient-45c'). Must NOT start with '_'.",
    )
    p_merge_nac.add_argument(
        "--target-id",
        default="runtime",
        help="Contributor ID for the existing runtime state (default: 'runtime').",
    )
    p_merge_nac.set_defaults(func=_run_merge_nac)

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
