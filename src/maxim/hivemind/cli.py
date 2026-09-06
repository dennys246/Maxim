"""``maxim substrate`` CLI verbs (Hivemind shareability, PR D).

v1_refinement.md §B5 PR D. Operator-visible CLI entry points for the
substrate-bundle round trip. Thin wrappers over
:mod:`maxim.hivemind.bundle` — they parse argv, locate the persisted
NAc + EC files, call :func:`compose_bundle` / :func:`extract_bundle`,
and print a short summary.

Subcommands:

- ``maxim substrate export <output.zip> --session <session_id>
   --contributor-id <id> [--domain X] [--no-identity-filter]
   [--body-ref NAME] [--body-yaml PATH] [--affordance-namespace NS]``
   — gate 7: declare the body the biases were learned on; ``--body-yaml``
   derives ``capability_map`` via the real tool-naming path.

- ``maxim substrate import <input.zip> --output-dir <dir>
   [--receiver-body NAME] [--allow-unverified-body]`` — gate 7:
   ``--receiver-body`` refuses cross-body/undeclared bundles before
   anything is written.

- ``maxim substrate inspect <input.zip>`` — print manifest only.

- ``maxim substrate ingest <input.zip> --session <receiver> --trust <id>
   --receiver-body NAME [--inherent-trust <id>] [--allow-unstamped-geometry]
   [--receiver-agent-id ID] [--force-digest] [--apply]`` — the 1.2 Oasis
   ingestion path: the V1–V10 receiver validation contract
   (docs/plans/sharing_threat_model.md §5) + the aligned ``substrate_merge``
   (with the tighten-only negative-valence clamp at its seam) + the V8
   ingestion journal. Dry-run by default. THE way foreign substrate enters
   a receiver; ``merge-nac`` (below) is trusted-local only, by design.

- ``maxim substrate invalidate --session <id> [--modality M
   --drop-geometry TAG] [--apply]`` — gate 1 migrate half: census with no
   geometry named; otherwise drop stale-geometry EC nodes, prune the NAc
   biases keyed on them, tombstone everything removed. Dry-run by default.

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
    assert_bundle_body_compatible,
    compose_bundle,
    extract_bundle,
    read_bundle_manifest,
)
from maxim.utils.optional_deps import OptionalDependencyError

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
    """Read ``path`` and return parsed JSON, or ``None`` if absent.

    Malformed JSON raises :class:`ValueError` naming the file — callers turn
    that into the CLI's rc=2 contract (a traceback on a corrupt session file
    is the shape the 2026-09-04 review round removed).
    """
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON in {path}: {exc}") from exc


class _MergeInputError(Exception):
    """A merge-nac input failed to read/validate; the message is already
    printed — the caller just returns the rc=2 contract. Replaces the old
    int-2-in-a-union sentinel (gate 8 hivemind mypy: a sentinel that shares
    a union with real data is exactly the shape a type checker exists to
    forbid)."""


def _run_export(args: argparse.Namespace) -> int:
    session_dir = _expand_session_dir(args.session)
    if not session_dir.is_dir():
        print(f"error: session directory not found: {session_dir}", file=sys.stderr)
        return 2

    try:
        nac_state = _read_optional_json(session_dir / "aut_nac.json")
        ec_payload = _read_optional_json(session_dir / "aut_ec.json")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
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

    # Gate 7 (typed bundles): the operator declares the body the biases were
    # learned on; the capability map is DERIVED from the body YAML via the
    # same tool-naming path that produced the keys, never hand-authored.
    # Neither is inferred when undeclared — an unverifiable bundle stays
    # honestly unverifiable and is refused downstream by default.
    body_ref: str | None = args.body_ref
    if body_ref is not None and not body_ref.strip():
        # An empty body_ref can never match any receiver (receiver_body must be
        # non-empty) yet would suppress the ships-unverifiable warning below.
        print("error: --body-ref must be a non-empty string (omit it to ship undeclared)", file=sys.stderr)
        return 2
    capability_map: dict[str, str] | None = None
    if args.body_yaml is not None:
        from maxim.embodiment.spec import load_spec
        from maxim.embodiment.tool_bridge import derive_capability_map

        # Broad catch, handled loudly: this is the CLI's rc=2 contract for
        # operator input. The realistic failures span yaml.YAMLError,
        # ConfigurationError, KeyError from malformed drive specs, and
        # ValueError from an unresolvable tool-name collision (executor-lens
        # fold) — none of which should escape as a traceback.
        try:
            spec = load_spec(args.body_yaml)
            capability_map = derive_capability_map(spec.root_entity)
        except Exception as exc:
            print(f"error: --body-yaml failed to load: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        if body_ref is None:
            body_ref = spec.name
        if not capability_map:
            print(
                "note: --body-yaml derived 0 capability keys (no modulator affordances in the "
                "body spec) — the bundle ships an empty capability_map.",
                file=sys.stderr,
            )

    output_path = Path(args.output).expanduser().resolve()
    try:
        signer = None
        if getattr(args, "sign", False):
            from maxim.hivemind.signing import load_or_create_signer, public_key_path

            signer = load_or_create_signer(signer_identity=args.signer_id or args.contributor_id)

        manifest = compose_bundle(
            nac_state=nac_state,
            ec_substrate_nodes=ec_substrate_nodes,
            output_path=output_path,
            contributor_id=args.contributor_id,
            domain=args.domain,
            apply_identity_filter=not args.no_identity_filter,
            identity_threshold=args.identity_threshold,
            signer=signer,
            encoder_provenance=ec_encoder_provenance,
            body_ref=body_ref,
            affordance_namespace=args.affordance_namespace,
            capability_map=capability_map,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OptionalDependencyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    n_slices = len(manifest.get("contents", {}))
    print(
        f"composed bundle at {output_path}\n"
        f"  contributor: {manifest['contributor_id']}\n"
        f"  domain:      {manifest.get('domain')}\n"
        f"  slices:      {n_slices}\n"
        f"  identity_filter: {manifest.get('identity_filter_applied')}\n"
        f"  body_ref:    {manifest.get('body_ref')}"
        + (
            f"\n  capability_map: {len(manifest.get('capability_map') or {})} keys"
            if capability_map is not None
            else ""
        )
    )
    if signer is not None:
        print(
            f"  signed:      ed25519 as {manifest.get('signer_identity')!r}\n"
            f"  public key:  {public_key_path()} (share this so receivers can --trust-key)"
        )
    if body_ref is None:
        print(
            "note: no --body-ref/--body-yaml given — bundle ships body_ref: null and "
            "will be REFUSED by body-checking receivers (BundleBodyUnverifiable).",
            file=sys.stderr,
        )
    return 0


def _run_import(args: argparse.Namespace) -> int:
    bundle_path = Path(args.input).expanduser().resolve()
    if not bundle_path.is_file():
        print(f"error: bundle file not found: {bundle_path}", file=sys.stderr)
        return 2

    # Gate 7 (typed bundles): refuse a cross-body bundle BEFORE anything is
    # written to disk. Undeclared bodies are unverifiable, not compatible —
    # --allow-unverified-body accepts that risk explicitly.
    if args.receiver_body is not None:
        import zipfile as _zipfile

        try:
            manifest = read_bundle_manifest(bundle_path)
        except (ValueError, OSError, KeyError, _zipfile.BadZipFile) as exc:
            print(f"error: cannot read bundle manifest: {exc}", file=sys.stderr)
            return 2
        try:
            # ValueError covers both refusal classes (they subclass it) AND
            # the non-empty-receiver_body validation — an unset shell var in
            # --receiver-body "$BODY" must report, not traceback.
            assert_bundle_body_compatible(
                manifest,
                receiver_body=args.receiver_body,
                allow_unverified=args.allow_unverified_body,
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    elif args.allow_unverified_body:
        # A silent no-op flag is the shape this codebase pushes into errors:
        # the operator believes risk was "accepted" on a path that never checked.
        print("error: --allow-unverified-body is meaningless without --receiver-body", file=sys.stderr)
        return 2
    else:
        print(
            "note: body compatibility NOT checked (pass --receiver-body to refuse "
            "cross-body bundles; their biases merge silently as zero — D43 barrier 3).",
            file=sys.stderr,
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    import zipfile as _zipfile

    try:
        manifest = extract_bundle(bundle_path, output_dir)
    except (ValueError, OSError, _zipfile.BadZipFile) as exc:
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


def _run_invalidate(args: argparse.Namespace) -> int:
    """Gate 1's migrate half — invalidate stale-geometry EC nodes in a session.

    EC stores centroids, not raw readings, so a stale-geometry node cannot be
    re-encoded in place: migration is loud invalidation (nodes removed, the
    NAc biases keyed on them pruned in the same operation, everything removed
    recorded verbatim in a tombstone sidecar), and live re-encoding happens
    organically as new readings arrive in the live geometry. Dry-run by
    default; ``--apply`` writes.
    """
    from maxim.hivemind.merge import invalidate_stale_geometry_nodes, prune_nac_cluster_biases

    session_dir = _expand_session_dir(args.session)
    if not session_dir.is_dir():
        print(f"error: session directory not found: {session_dir}", file=sys.stderr)
        return 2

    ec_path = session_dir / "aut_ec.json"
    nac_path = session_dir / "aut_nac.json"
    try:
        ec_payload = _read_optional_json(ec_path)
        nac_state = _read_optional_json(nac_path)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    nodes = ec_payload.get("substrate_nodes") if isinstance(ec_payload, dict) else None
    if not isinstance(ec_payload, dict) or not isinstance(nodes, dict):
        print(f"error: no substrate_nodes in {ec_path}; nothing to invalidate", file=sys.stderr)
        return 2

    # No --drop-geometry: print the per-modality geometry census so the
    # operator can see what is stale, and stop. Never guess which geometry
    # is the live one — the operator names the one to drop. A given
    # --modality FILTERS the census (a silently ignored flag is the shape
    # this CLI turns into behavior or an error, never a no-op).
    if args.drop_geometry is None:
        if args.apply:
            print("error: --apply requires --modality and --drop-geometry", file=sys.stderr)
            return 2
        census: dict[str, dict[str, int]] = {}
        malformed = 0
        for node in nodes.values():
            if not isinstance(node, dict):
                malformed += 1
                continue
            mod = str(node.get("modality", "?"))
            if args.modality is not None and mod != args.modality:
                continue
            geom = node.get("geometry")
            by_geom = census.setdefault(mod, {})
            key = "(unstamped)" if geom is None else str(geom)
            by_geom[key] = by_geom.get(key, 0) + 1
        scope = f" (modality {args.modality!r})" if args.modality is not None else ""
        print(f"geometry census for {ec_path}{scope} ({len(nodes)} nodes):")
        for mod in sorted(census):
            for geom, count in sorted(census[mod].items()):
                print(f"  {mod:12s} {geom}: {count}")
        if malformed:
            print(f"  ({malformed} malformed non-dict node entr{'y' if malformed == 1 else 'ies'} skipped)")
        print("(re-run with --modality and --drop-geometry to invalidate a stale geometry)")
        return 0

    if args.modality is None:
        print("error: --drop-geometry requires --modality", file=sys.stderr)
        return 2

    kept, removed_ids = invalidate_stale_geometry_nodes(nodes, modality=args.modality, drop_geometry=args.drop_geometry)
    if not removed_ids:
        print(f"nothing to invalidate: no {args.modality!r} node is stamped with geometry {args.drop_geometry!r}")
        return 0

    pruned_entries: dict[str, dict] = {}
    pruned_count = 0
    new_nac = None
    if isinstance(nac_state, dict):
        new_nac, pruned_count = prune_nac_cluster_biases(nac_state, set(removed_ids))
        for field in ("cluster_reward_bias", "cluster_reward_source", "reward_bias"):
            old_field = nac_state.get(field)
            new_field = new_nac.get(field)
            if isinstance(old_field, dict) and isinstance(new_field, dict):
                dropped = {k: v for k, v in old_field.items() if k not in new_field}
                if dropped:
                    pruned_entries[field] = dropped

    print(
        f"invalidate {args.modality!r} nodes with geometry {args.drop_geometry!r}:\n"
        f"  EC nodes removed:    {len(removed_ids)} of {len(nodes)}\n"
        f"  NAc biases pruned:   {pruned_count}" + ("" if nac_state is not None else "  (no aut_nac.json in session)")
    )

    if not args.apply:
        print("DRY RUN — nothing written. Re-run with --apply to invalidate.")
        return 0

    from datetime import datetime, timezone

    from maxim.utils.atomic_io import atomic_write_json
    from maxim.utils.format_version import with_format_version

    # Tombstone FIRST: everything removed, verbatim, so the deletion is
    # auditable and hand-reversible. Then NAc BEFORE EC: if the NAc write
    # fails mid-way, pruned biases whose nodes still exist is benign; the
    # reverse (nodes gone, biases dangling) is the D2 shape this verb
    # exists to eliminate. Microsecond stamp + refuse-if-exists: the audit
    # record must never be silently clobbered by a same-second re-run.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    tombstone_path = session_dir / f"aut_ec.invalidated.{stamp}.json"
    if tombstone_path.exists():
        print(f"error: tombstone already exists: {tombstone_path}", file=sys.stderr)
        return 2
    atomic_write_json(
        str(tombstone_path),
        with_format_version(
            {
                "reason": "gate1-geometry-invalidation",
                "modality": args.modality,
                "drop_geometry": args.drop_geometry,
                "removed_nodes": {nid: nodes[nid] for nid in removed_ids},
                "pruned_nac_entries": pruned_entries,
            }
        ),
    )
    if new_nac is not None:
        atomic_write_json(str(nac_path), new_nac)
    ec_payload["substrate_nodes"] = kept
    atomic_write_json(str(ec_path), ec_payload)
    print(f"applied. Tombstone: {tombstone_path}")
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
    def _read_dict_or_fail(path: Path, label: str) -> "dict | None":
        try:
            data = _read_optional_json(path)
        except (OSError, ValueError) as exc:
            # ValueError covers _read_optional_json's malformed-JSON wrap
            # (and json.JSONDecodeError, which subclasses it).
            print(f"error: cannot read {label} ({path}): {exc}", file=sys.stderr)
            raise _MergeInputError from exc
        if data is not None and not isinstance(data, dict):
            print(f"error: {label} is not a JSON object ({path}). Nothing was written.", file=sys.stderr)
            raise _MergeInputError
        return data

    try:
        src_meta = _read_dict_or_fail(_meta_sidecar_path(source_path), "source policy-meta sidecar")
        tgt_meta = _read_dict_or_fail(_meta_sidecar_path(target_path), "target policy-meta sidecar")
    except _MergeInputError:
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
    try:
        target_state_opt = _read_dict_or_fail(target_path, "target NAc file")
    except _MergeInputError:
        return 2
    target_state: dict = target_state_opt or {}
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


def _resolve_receiver_pair(session_dir: Path) -> tuple[Path, Path] | None:
    """Locate the receiver's NAc/EC pair in either persistence layout.

    Sessions persist ``aut_nac.json``/``aut_ec.json``; agents persist
    ``nac.json``/``ec.json`` (bio_stack). Detected as a PAIR — the NAc+EC
    pair invariant means ingesting into half a receiver would mint the
    D2 dangling shape.
    """
    for nac_name, ec_name in (("aut_nac.json", "aut_ec.json"), ("nac.json", "ec.json")):
        nac_p, ec_p = session_dir / nac_name, session_dir / ec_name
        if nac_p.is_file() or ec_p.is_file():
            return nac_p, ec_p
    return None


def _run_ingest(args: argparse.Namespace) -> int:
    """The 1.2 Oasis ingestion verb — V1–V10 validated merge into a receiver.

    docs/plans/oasis_ingestion_contract.md is the design record; the
    pipeline lives in :func:`maxim.hivemind.ingest.ingest_bundle`. Dry-run
    by default (the ``invalidate`` precedent); ``--apply`` writes with a
    pre-ingest backup (the ``merge-nac`` precedent). MUST NOT run against
    a receiver a live session currently owns — the runtime persists at
    session end and would clobber the ingest (contract §1).
    """
    import zipfile as _zipfile

    from maxim.hivemind.ingest import IngestionJournal, IngestRefused, ingest_bundle

    bundle_path = Path(args.input).expanduser().resolve()
    if not bundle_path.is_file():
        print(f"error: bundle file not found: {bundle_path}", file=sys.stderr)
        return 2

    session_dir = _expand_session_dir(args.session)
    if not session_dir.is_dir():
        print(f"error: receiver directory not found: {session_dir}", file=sys.stderr)
        return 2
    pair = _resolve_receiver_pair(session_dir)
    if pair is None:
        print(
            f"error: no NAc/EC pair (aut_nac.json/aut_ec.json or nac.json/ec.json) in {session_dir}; "
            "a receiver must exist — create one with maxim.create.agent() and shut it down first.",
            file=sys.stderr,
        )
        return 2
    nac_path, ec_path = pair

    try:
        receiver_nac = _read_optional_json(nac_path)
        ec_payload = _read_optional_json(ec_path)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if receiver_nac is not None:
        receiver_nac.pop("_format_version", None)
    receiver_ec_nodes = ec_payload.get("substrate_nodes") if isinstance(ec_payload, dict) else None
    if ec_payload is not None and not isinstance(receiver_ec_nodes, dict):
        print(f"error: {ec_path} has no substrate_nodes object", file=sys.stderr)
        return 2

    journal_path = session_dir / "substrate_ingest_journal.json"
    try:
        journal = IngestionJournal(journal_path)
    except (ValueError, OSError) as exc:
        print(f"error: cannot read ingestion journal ({journal_path}): {exc}", file=sys.stderr)
        return 2

    trusted_keys: dict[str, str] = {}
    for entry in getattr(args, "trust_key", []) or []:
        identity, sep, pubkey = entry.partition("=")
        if not sep or not identity or not pubkey:
            print(f"error: --trust-key must be IDENTITY=PUBKEY_B64, got {entry!r}", file=sys.stderr)
            return 2
        trusted_keys[identity] = pubkey

    try:
        report = ingest_bundle(
            bundle_path,
            receiver_nac=receiver_nac,
            receiver_ec_nodes=receiver_ec_nodes,
            receiver_body=args.receiver_body,
            trusted_sources=frozenset(args.trust),
            inherent_trusted_sources=frozenset(args.inherent_trust or []),
            journal=journal,
            receiver_agent_id=args.receiver_agent_id,
            allow_unverified_body=args.allow_unverified_body,
            allow_unstamped_geometry=args.allow_unstamped_geometry,
            force_digest=args.force_digest,
            require_signed=getattr(args, "require_signed", False),
            trusted_keys=trusted_keys,
        )
    except (IngestRefused, ValueError, OSError, _zipfile.BadZipFile) as exc:
        # IngestRefused and the gate-7 refusals subclass ValueError; every
        # refusal is the rc=2 contract, never a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f"validated bundle {bundle_path.name} (digest {report.digest[:12]}…)\n"
        f"  contributor:       {report.contributor_id}\n"
        f"  body_ref:          {report.manifest.get('body_ref')}\n"
        f"  biases rekeyed:    {report.biases_rekeyed} (dropped: {report.biases_dropped})\n"
        f"  biases tightened:  {report.biases_tightened}\n"
        f"  inherent admitted: {report.inherent_keys_admitted}\n"
        f"  donor EC nodes:    {len(report.id_map)}"
    )
    for note in report.notes:
        print(f"  note: {note}")
    if report.valence_entries:
        print("  percept valences by entity class (V4 report):")
        for entity_class, valence in sorted(report.valence_entries.items()):
            print(f"    {entity_class}: {valence:+.3f}")

    if not args.apply:
        print("DRY RUN — nothing written. Re-run with --apply to ingest.")
        return 0

    from maxim.decisions.nac import _NAC_FORMAT_VERSION
    from maxim.utils.atomic_io import atomic_write_json, atomic_write_text
    from maxim.utils.format_version import with_format_version

    # Pre-ingest backups (merge-nac precedent), then journal → EC → NAc.
    # Journal first: a crash after it leaves a bundle marked ingested with
    # no state change, recoverable via --force-digest; state-before-journal
    # would let a replay double-count (row J). EC before NAc: merged NAc
    # biases naming EC nodes not yet on disk is the D2 dangling shape;
    # nodes-without-biases is benign (contract §3.12).
    for path in (nac_path, ec_path):
        if path.is_file():
            backup_path = path.with_name(path.name + ".pre-ingest.bak")
            try:
                atomic_write_text(str(backup_path), path.read_text(encoding="utf-8"))
            except OSError as exc:
                print(f"error: cannot write backup ({backup_path}): {exc}. Nothing was written.", file=sys.stderr)
                return 2

    journal.record(report.journal_entry)
    try:
        journal.save()
    except OSError as exc:
        print(f"error: cannot write ingestion journal: {exc}. Nothing was written.", file=sys.stderr)
        return 2

    if ec_payload is None:
        # A freshly-minted EC file must carry the CC1 stamp itself — the
        # splice branch below preserves the stamp the writing EC put there
        # (arch-lens finding 5).
        ec_payload = with_format_version({"substrate_nodes": {}})
    ec_payload["substrate_nodes"] = report.ec_nodes
    try:
        atomic_write_json(str(ec_path), ec_payload)
        atomic_write_json(str(nac_path), with_format_version(dict(report.nac), version=_NAC_FORMAT_VERSION))
    except OSError as exc:
        # The journal entry is already durable — recoverable via
        # --force-digest once the disk trouble is resolved; a traceback is
        # not the rc=2 contract (executor-lens finding 6).
        print(
            f"error: cannot write receiver state: {exc}. The journal already records this digest; "
            "re-run with --force-digest after resolving.",
            file=sys.stderr,
        )
        return 2

    print(
        f"applied. Receiver updated: {nac_path.name} + {ec_path.name}\n"
        f"  journal: {journal_path.name} ({len(journal.entries)} entries)\n"
        "NOTE: pick up on the receiver's next boot; never ingest into a receiver a live session owns."
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


def _run_keygen(args: argparse.Namespace) -> int:
    """Mint (if absent) and print the local ed25519 bundle-signing public key."""
    try:
        from maxim.hivemind.signing import load_or_create_signer, public_key_path, signing_key_path

        signer = load_or_create_signer(signer_identity=args.signer_id)
    except OptionalDependencyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"signer_identity: {signer.signer_identity}\n"
        f"public_key:      {signer.public_key_b64}\n"
        f"private key:     {signing_key_path()} (0600; never share)\n"
        f"public key file: {public_key_path()}\n"
        f"share as:        --trust-key {signer.signer_identity}={signer.public_key_b64}"
    )
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
    p_export.add_argument(
        "--body-ref",
        default=None,
        help=(
            "Gate 7: the body the biases were learned on (e.g. 'reachy_mini'). "
            "Undeclared bundles ship body_ref: null and are refused by body-checking receivers."
        ),
    )
    p_export.add_argument(
        "--body-yaml",
        default=None,
        help=(
            "Gate 7: path to the body's embodiment YAML. Derives manifest.capability_map "
            "(tool:<name> -> <modulator>/<affordance>) via the real tool-naming path, and "
            "defaults --body-ref to the spec's name."
        ),
    )
    p_export.add_argument(
        "--affordance-namespace",
        default=None,
        help="Gate 7: name of the vocabulary the bundle's tool signatures live in.",
    )
    p_export.add_argument(
        "--sign",
        action="store_true",
        help=(
            "Slice A (1.2 P2P): sign the bundle with the persisted ed25519 key "
            "(minted on first use under ~/.config/maxim/). Requires the [sign] extra."
        ),
    )
    p_export.add_argument(
        "--signer-id",
        default=None,
        help="signer_identity written to the manifest (defaults to --contributor-id).",
    )
    p_export.set_defaults(func=_run_export)

    p_import = sub.add_parser("import", help="Extract a substrate bundle to a directory")
    p_import.add_argument("input", help="Path to the .zip bundle")
    p_import.add_argument(
        "--output-dir",
        required=True,
        help="Directory to extract the bundle to (created if absent).",
    )
    p_import.add_argument(
        "--receiver-body",
        default=None,
        help=(
            "Gate 7: this receiver's body name. Refuses a bundle learned on a different "
            "body (BundleBodyMismatch) or an undeclared one (BundleBodyUnverifiable) "
            "before anything is written."
        ),
    )
    p_import.add_argument(
        "--allow-unverified-body",
        action="store_true",
        help="Accept a bundle with no declared body_ref despite --receiver-body (explicit risk).",
    )
    p_import.set_defaults(func=_run_import)

    p_invalidate = sub.add_parser(
        "invalidate",
        help="Gate 1 migrate half: drop stale-geometry EC nodes + the NAc biases keyed on them (dry-run by default)",
    )
    p_invalidate.add_argument(
        "--session",
        required=True,
        help="Session ID (resolved under ~/.maxim/sessions/{id}/) or a path to a session directory",
    )
    p_invalidate.add_argument(
        "--modality",
        default=None,
        help="Modality whose stale nodes to invalidate (e.g. 'world').",
    )
    p_invalidate.add_argument(
        "--drop-geometry",
        default=None,
        help=(
            "The stale geometry tag to drop, by name (copy it from the census this command "
            "prints when run without this flag, or from the EC mismatch warning). Unstamped "
            "nodes are never touched."
        ),
    )
    p_invalidate.add_argument(
        "--apply",
        action="store_true",
        help="Actually write (default is a dry-run report). Writes a tombstone sidecar first.",
    )
    p_invalidate.set_defaults(func=_run_invalidate)

    p_ingest = sub.add_parser(
        "ingest",
        help="Validate (V1-V10) + merge a foreign bundle into a receiver's NAc/EC pair (dry-run by default)",
    )
    p_ingest.add_argument("input", help="Path to the .zip bundle")
    p_ingest.add_argument(
        "--session",
        required=True,
        help="Receiver: session ID (under ~/.maxim/sessions/) or a path to a session/agent directory. Must be AT REST.",
    )
    p_ingest.add_argument(
        "--trust",
        action="append",
        required=True,
        metavar="CONTRIBUTOR_ID",
        help=(
            "Operator-attested contributor id (V1 front door; repeatable). A bundle whose "
            "manifest.contributor_id is not listed is refused — never admitted-with-clamps."
        ),
    )
    p_ingest.add_argument(
        "--inherent-trust",
        action="append",
        default=None,
        metavar="CONTRIBUTOR_ID",
        help=(
            "Queen-attested contributor id (repeatable). Only these may ship inherent-class bias "
            "keys; anyone else's inherent claim refuses the bundle (privilege-escalation guard)."
        ),
    )
    p_ingest.add_argument(
        "--receiver-body",
        required=True,
        help="This receiver's body name (gate 7 refusal always runs on the ingest path).",
    )
    p_ingest.add_argument(
        "--allow-unverified-body",
        action="store_true",
        help="Accept a bundle with no declared body_ref (explicit risk, gate 7).",
    )
    p_ingest.add_argument(
        "--allow-unstamped-geometry",
        action="store_true",
        help=(
            "V3 legacy override: admit foreign EC nodes without geometry stamps (e.g. the "
            "SHA-manifested 53_agents archive, which predates stamping). Default is refusal."
        ),
    )
    p_ingest.add_argument(
        "--receiver-agent-id",
        default=None,
        help=(
            "Normalize donor agent ids to this receiver's own at the ingestion boundary "
            "(the agent name the receiver runs under). Omit only when donor and receiver "
            "genuinely share an agent id."
        ),
    )
    p_ingest.add_argument(
        "--force-digest",
        action="store_true",
        help="Re-ingest a bundle the journal has already seen (eyes-open replay; V8).",
    )
    p_ingest.add_argument(
        "--apply",
        action="store_true",
        help="Actually write (default is a dry-run report). Backs up the pair, journals first.",
    )
    p_ingest.add_argument(
        "--require-signed",
        action="store_true",
        help=(
            "Slice A (1.2 P2P): refuse any bundle without a valid ed25519 signature from a "
            "--trust-key signer. The Queen-tier default; experimental-tier ingests omit it."
        ),
    )
    p_ingest.add_argument(
        "--trust-key",
        action="append",
        default=[],
        metavar="IDENTITY=PUBKEY_B64",
        help="Trust a signer: <signer_identity>=<base64 public key>. Repeatable.",
    )
    p_ingest.set_defaults(func=_run_ingest)

    p_inspect = sub.add_parser("inspect", help="Print the bundle manifest without extracting")
    p_inspect.add_argument("input", help="Path to the .zip bundle")
    p_inspect.set_defaults(func=_run_inspect)

    p_keygen = sub.add_parser(
        "keygen",
        help="Mint (if absent) and print the local ed25519 bundle-signing public key (Slice A)",
    )
    p_keygen.add_argument(
        "--signer-id",
        required=True,
        help="signer_identity to bind the key to (the label receivers --trust-key).",
    )
    p_keygen.set_defaults(func=_run_keygen)

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
    (currently ``2`` — gate 7 typed bundles).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "run_substrate_subcommand",
]
