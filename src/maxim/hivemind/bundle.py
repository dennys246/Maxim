"""Substrate snapshot bundle format (Hivemind shareability, PR D).

v1_refinement.md §B5 PR D. Composes a portable, versioned, optionally
signed archive of one Maxim's NAc + EC substrate, suitable for exchange
between Oases and substrate-primary Maxims. The 1.1 Oasis software will
build on this format; the 1.2 P2P Hivemind protocol will exchange these
bundles between peers.

The format is a ZIP containing JSON files at the bundle root:

    maxim-substrate.zip
    ├── manifest.json   # _format_version, contributor_id, domain, signature
    ├── nac.json        # NAc.dump() output (optionally identity-filtered)
    └── ec.json         # EC substrate_nodes slice (optionally domain-filtered)

Hippocampus episodes are NEVER included by construction — the
"hippocampus-episodes-stay-local" rule from ``maxim_hivemind.md`` is the
load-bearing privacy invariant. ATL, reflexes, and cerebellum payloads
are reserved for 1.1 (Phase B5 spec mentions them; this 1.0 ship
includes only NAc + EC because that's what PRs A/B/C give us merge
math for).

Manifest signature slot
-----------------------

The ``signature``, ``signature_algorithm``, and ``signer_identity``
fields are reserved (per the 2026-05-30 design decision: "Reserve
signature field in manifest, no verification yet" + the CC13 auth
format-freeze, which added ``signer_identity``). At 1.0 they are always
``None`` — the slots exist so 1.1+ verification can land WITHOUT bumping
the bundle's ``_format_version`` and breaking 1.0 bundles. Callers that
want signing build their own ZIP with a populated ``signature`` field
and a custom verifier; this module does NOT validate.

The recognized ``signature_algorithm`` vocabulary (``ed25519``,
``ed25519-pgp``, ``webauthn``, ``pkcs7``, reserved ``hsm:*`` / ``kms:*``
/ ``vendor:*`` prefixes, ...) is published in
``docs/user/hivemind_bundle_format.md`` so the 1.2 P2P protocol's
heterogeneous producers and consumers share a string vocabulary. The
registry is documentation-only at 1.0 (no validator), consistent with
the no-verification-yet decision. ``signer_identity`` is the reserved
"who claims to have signed this" string, parallel to ``contributor_id``,
so 1.1+ can bind a verified identity to the claimed contributor without
retrofitting the manifest shape.

Format version contract
-----------------------

The ``manifest.json`` root carries ``_format_version`` per the CC1 1.0
freeze invariant. Older bundles will fail to load via
:func:`extract_bundle` — there is no 0.x bundle in the wild because
this is a 1.0 feature.

Identity filter
---------------

When ``apply_identity_filter=True`` (the default), :func:`compose_bundle`
routes the NAc links through
:func:`maxim.hivemind.identity.filter_identity_bearing_links` and drops
EC nodes whose ``domain`` is the
:data:`maxim.hivemind.identity.IDENTITY_DOMAIN_MARKER`. The identity
threshold defaults to ``2`` per the PR C review fold (game-substrate
contexts over-flag at threshold=1 because every "Dragon"/"Goblin" trips
the proper-noun signal).
"""

from __future__ import annotations

import contextlib
import copy
import datetime as _dt
import json
import logging
import os
import re
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from maxim.hivemind.identity import (
    IDENTITY_DOMAIN_MARKER,
    filter_identity_bearing_links,
    is_identity_bearing,
)
from maxim.hivemind.merge import NAC_KEY_SEP, _merge_link_pair, _merge_welford, _validate_source
from maxim.utils.atomic_io import atomic_write_text
from maxim.utils.format_version import FORMAT_VERSION, check_format_version

logger = logging.getLogger(__name__)

# Schema version for the bundle envelope itself. Separate from the
# bio-system payload ``_format_version`` — bumping this would require
# a migration registered alongside the bump.
BUNDLE_SCHEMA_VERSION: int = 2

# Bundle-level kind marker for the manifest.
BUNDLE_KIND: str = "substrate_bundle"

# Default identity-filter threshold for the bundle composer. PR C
# Architecture-lens review flagged game-substrate over-flagging at
# threshold=1; bundles default to the stricter 2 so generic creature
# tokens like "Dragon" ride along.
_DEFAULT_BUNDLE_IDENTITY_THRESHOLD: int = 2


# ─────────────────────────────────────────────────────────────────────────
# Migration registry (Architecture review IMPORTANT fold)
#
# Pre-emptive seam matching the ``memory/snapshot.py`` envelope-migration
# pattern. Empty at 1.0 — bumping ``BUNDLE_SCHEMA_VERSION`` to 2 in 1.1
# registers a single ``v1 → v2`` function via the decorator below and
# the extract path upgrades 1.0-shaped bundles transparently. Reserving
# the seam now avoids a painful retrofit when 1.1 first needs it.
# ─────────────────────────────────────────────────────────────────────────


MigrationFn = Callable[[dict[str, Any]], dict[str, Any]]

_BUNDLE_MIGRATIONS: dict[int, MigrationFn] = {}


def register_bundle_migration(from_version: int) -> Callable[[MigrationFn], MigrationFn]:
    """Decorator: register a bundle-manifest migration ``from_version → from_version+1``.

    Bumping :data:`BUNDLE_SCHEMA_VERSION` requires registering a
    matching migration so older bundles upgrade transparently on
    extract. The registry follows the same shape as the
    ``memory/snapshot.py`` envelope-migration registry.
    """

    def _decorator(fn: MigrationFn) -> MigrationFn:
        if from_version in _BUNDLE_MIGRATIONS:
            raise ValueError(f"bundle migration from version {from_version} already registered")
        _BUNDLE_MIGRATIONS[from_version] = fn
        return fn

    return _decorator


@register_bundle_migration(1)
def _v1_to_v2_typed_bundle(manifest: dict[str, Any]) -> dict[str, Any]:
    """v1 → v2: gate 7 typed bundles — stamp the body/namespace fields.

    A v1 bundle predates the typed contract, so its body of origin is
    genuinely unknown. The fields are stamped as ``None``, which
    :func:`assert_bundle_body_compatible` treats as "unverifiable" rather
    than "compatible" — an old bundle cannot silently pass a body check it
    was never subject to.
    """
    out = dict(manifest)
    out.setdefault("body_ref", None)
    out.setdefault("affordance_namespace", None)
    out.setdefault("capability_map", {})
    out["schema_version"] = 2
    return out


def migrate_bundle_envelope(manifest: dict[str, Any], *, target_version: int | None = None) -> dict[str, Any]:
    """Upgrade a bundle manifest through the migration chain.

    At v1 with an empty registry this is a no-op deep-copy (callers may
    mutate the returned dict without aliasing the input). Bumping
    ``BUNDLE_SCHEMA_VERSION`` to 2 + registering a ``v1`` migration is
    the only change 1.1 needs to make for older bundles to load.
    """
    if target_version is None:
        target_version = BUNDLE_SCHEMA_VERSION
    if not isinstance(manifest, dict):
        raise ValueError(f"bundle manifest must be dict, got {type(manifest).__name__}")
    version = manifest.get("schema_version")
    if not isinstance(version, int):
        # Defer the validation error to extract_bundle's existing branch;
        # passing through unchanged keeps a single point of error reporting.
        return copy.deepcopy(manifest)
    if version > target_version:
        return copy.deepcopy(manifest)

    current = copy.deepcopy(manifest)
    while True:
        cur_version = current["schema_version"]
        if cur_version == target_version:
            return current
        migration = _BUNDLE_MIGRATIONS.get(cur_version)
        if migration is None:
            raise ValueError(
                f"bundle envelope schema_version {cur_version} has no migration to {cur_version + 1} "
                f"(registry keys: {sorted(_BUNDLE_MIGRATIONS.keys())})"
            )
        current = migration(current)
        if not isinstance(current, dict) or current.get("schema_version") != cur_version + 1:
            raise ValueError(f"bundle migration {cur_version}→{cur_version + 1} returned an invalid envelope")


@contextlib.contextmanager
def isolated_bundle_migrations() -> Iterator[None]:
    """Context manager: snapshot + clear + restore the bundle-migration registry.

    Test-only helper matching ``memory/snapshot.py::isolated_migrations``.
    Tests that need to register synthetic migrations wrap setup in this
    context so they start from a clean registry and the pre-test
    registry is restored on exit.
    """
    saved = dict(_BUNDLE_MIGRATIONS)
    _BUNDLE_MIGRATIONS.clear()
    try:
        yield
    finally:
        _BUNDLE_MIGRATIONS.clear()
        _BUNDLE_MIGRATIONS.update(saved)


def _utc_now_iso() -> str:
    """ISO 8601 timestamp in UTC with second-level resolution."""
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


# ─────────────────────────────────────────────────────────────────────────
# NAc content scrub — model-generated text must not ship
#
# The NAc dump carries locally-scoped text on several surfaces (privacy
# audit + two-lens review, PR #506; each finding independently
# cross-confirmed there):
#
# 1. ``event_context`` — tool_dispatch.py sets ``ctx["goal"]`` to the
#    LLM's own ``reasoning[:100]`` (model-generated text, verbatim).
# 2. ``outcome_signature`` — ``f"{success|failure}:{outcome_summary}"``
#    where outcome_summary is raw tool output / error text (paths,
#    hostnames, credentials). The same strings key ``outcome_index``.
# 3. ``memory_ids`` — hippocampus episode IDs. Episodes NEVER ship
#    (the load-bearing privacy invariant above); their IDs don't either.
#    ``percept_refs`` (percept IDs + content hashes) are the same
#    reference class and get the same treatment.
# 4. ``goal_reward_bias`` keys — verbatim goal strings (operator
#    ``--goal`` free text or LLM-proposed goal descriptions) via
#    ``credit_goal``. Goals are session/operator-specific; a foreign
#    substrate cannot match them, so the field has no cross-org
#    transfer value. Dropped entirely.
# 5. Event signatures — ``build_tool_signature`` emits
#    ``tool:use:<action>`` where ``<action>`` is a verbatim LLM tool
#    parameter (arbitrary free text). The same signature string ships
#    through FOUR surfaces: ``links`` keys, the per-link
#    ``event_signature`` field, ``event_outcome_welford`` keys, and the
#    tsig third of ``cluster_reward_bias`` / ``cluster_reward_source``
#    keys. Identifier-shaped actions (``tool:use:dodge``) are the
#    documented transfer vocabulary and are kept; anything else is
#    truncated to ``tool:use``.
# 6. ``priors`` — zero production producers today, but a verbatim
#    pass-through one ``set_prior`` caller away from re-opening the
#    key-leak class. Dropped (empty in every real run anyway).
#
# The scrub is applied at COMPOSITION, not at capture: everything stays
# fully populated locally (debugging needs it) — only the bundle is
# scrubbed. It runs unconditionally, independent of
# ``apply_identity_filter`` (which drops whole links by event_signature
# key and never inspects fields) — like the hippocampus exclusion, this
# is a privacy invariant by construction, not an option.
#
# No defensive handlers here on purpose: an unexpected shape must raise
# at compose time, not silently ship unscrubbed.
# ─────────────────────────────────────────────────────────────────────────

# ALLOWLIST of event_context keys that may ship in a bundle. Allowlist,
# not denylist — a denylist silently leaks whatever field a future
# producer adds next.
_BUNDLE_EVENT_CONTEXT_ALLOWLIST: frozenset[str] = frozenset({"agent_id"})

# Identifier-shaped token: single short token, no whitespace. Gates the
# ``tool:use:<action>`` tail (``tool:use:dodge`` / ``tool:use:open`` are
# the documented transfer vocabulary; a sentence-shaped action is
# verbatim LLM output) and the ``percept_valences`` entity_class (YAML
# component names like ``rusty_sword``; an imagined entity's LLM-coined
# multi-word name is not).
_IDENTIFIER_TOKEN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

_USE_SIG_PREFIX = "tool:use:"

# Composite-key separator used by NAc.dump() for welford / cluster /
# percept-valence keys.
# Re-exported from merge.py, which owns it (bundle imports merge, not the
# reverse). Kept as a module-local alias so existing references are unchanged.
_NAC_KEY_SEP = NAC_KEY_SEP


def _scrub_event_signature(sig: str) -> str:
    """Truncate ``tool:use:<free text>`` signatures to ``tool:use``.

    Identifier-shaped action tails are kept — they are the cross-entity
    transfer vocabulary the bundle exists to ship. Everything else in
    the signature space is template-generated (``tool:<name>``,
    ``drive:<sensor>``, ``conversation:<channel>``) and passes through.
    """
    if sig.startswith(_USE_SIG_PREFIX):
        action = sig[len(_USE_SIG_PREFIX) :]
        if not _IDENTIFIER_TOKEN.match(action):
            return "tool:use"
    return sig


def _scrub_link_for_bundle(link: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of one CausalLink dict scrubbed for bundle export."""
    scrubbed = dict(link)
    scrubbed["event_context"] = {k: v for k, v in link["event_context"].items() if k in _BUNDLE_EVENT_CONTEXT_ALLOWLIST}
    # Canonical valence-preserving form built from STRUCTURED fields.
    # The review round refuted the first-token-of-outcome_signature
    # draft twice over: (a) only one of the four outcome_signature
    # producers embeds success|failure in its first token, so valence
    # was destroyed for the others, violating nac_merge's design rule
    # #2 (valence-distinct links stay separate — _merge_link_pair's
    # documented precondition); (b) truncation made outcome signatures
    # non-unique within an event's link list, and _merge_link_lists
    # pairs by outcome_signature, silently clobbering all but one link.
    # ``{outcome_type}:{valence}`` is equally free-text-free, unique per
    # valence class, and merge-safe by construction.
    scrubbed["outcome_signature"] = f"{link['outcome_type']}:{link['outcome_valence']}"
    scrubbed["event_signature"] = _scrub_event_signature(link["event_signature"])
    scrubbed["memory_ids"] = []
    scrubbed["percept_refs"] = []
    return scrubbed


def scrub_nac_state_for_bundle(nac_state: dict[str, Any]) -> dict[str, Any]:
    """Scrub a ``NAc.dump()``-shaped state dict for bundle export.

    Pure function: the input is not mutated. See the section comment
    above for the field-by-field rationale. Key collisions introduced
    by signature scrubbing are merged with the same math the hivemind
    merge layer uses (``_merge_link_pair``, parallel-Welford, bias
    mean, source promotion to ``"mixed"``), so the shipped state
    satisfies ``nac_merge``'s pairing invariants — outcome signatures
    stay unique per link list, valence classes stay separate.
    """
    scrubbed = dict(nac_state)

    # links: scrub each link, re-key on the scrubbed event signature,
    # and fold links that now share (event_sig, outcome_sig) via
    # _merge_link_pair — nac_merge pairs by outcome_signature, so
    # shipping duplicates would silently clobber all but one on the
    # receiving side. The canonical outcome signature embeds valence,
    # so same-key folding satisfies _merge_link_pair's same-valence
    # precondition by construction.
    merged_links: dict[str, list[dict[str, Any]]] = {}
    for evt_sig, links in (nac_state.get("links", {}) or {}).items():
        bucket = merged_links.setdefault(_scrub_event_signature(evt_sig), [])
        for link in links:
            scrubbed_link = _scrub_link_for_bundle(link)
            existing = next(
                (b for b in bucket if b["outcome_signature"] == scrubbed_link["outcome_signature"]),
                None,
            )
            if existing is None:
                bucket.append(scrubbed_link)
            else:
                bucket[bucket.index(existing)] = _merge_link_pair(
                    existing,
                    scrubbed_link,
                    left_source=str(existing.get("source") or "local"),
                    right_source=str(scrubbed_link.get("source") or "local"),
                )
    scrubbed["links"] = merged_links

    # outcome_index: rebuilt from the scrubbed links (keys ARE outcome
    # signatures, now canonical) — rebuilding also drops index entries
    # for links the caller filtered out.
    rebuilt_index: dict[str, list[str]] = {}
    for links in merged_links.values():
        for link in links:
            bucket_ids = rebuilt_index.setdefault(link["outcome_signature"], [])
            if link["id"] not in bucket_ids:
                bucket_ids.append(link["id"])
    scrubbed["outcome_index"] = rebuilt_index

    # goal_reward_bias / priors: dropped entirely (see section comment).
    scrubbed["goal_reward_bias"] = {}
    scrubbed["priors"] = {}

    # event_outcome_welford: scrub the signature half of the composite
    # key; parallel-Welford merge on collision. A separator-less key is
    # not a shape NAc.dump() can emit — raise rather than ship a
    # silently-mangled key (same policy the cluster unpacking below
    # enforces by construction).
    merged_welford: dict[str, dict[str, float]] = {}
    for key, state in (nac_state.get("event_outcome_welford", {}) or {}).items():
        if _NAC_KEY_SEP not in key:
            raise ValueError(f"malformed event_outcome_welford key (no separator): {key!r}")
        aid, _, evt_sig = key.partition(_NAC_KEY_SEP)
        new_key = f"{aid}{_NAC_KEY_SEP}{_scrub_event_signature(evt_sig)}"
        if new_key in merged_welford:
            merged_welford[new_key] = _merge_welford({new_key: merged_welford[new_key]}, {new_key: dict(state)})[
                new_key
            ]
        else:
            merged_welford[new_key] = dict(state)
    scrubbed["event_outcome_welford"] = merged_welford

    # percept_valences: keys are {aid}\x1f{entity_class}\x1f{failure_mode}.
    # entity_class is usually a YAML component name (rusty_sword), but
    # imagined entities carry LLM-coined names built from percept noun
    # phrases — potentially user speech. Identifier-shaped classes ship
    # (they are the transfer vocabulary, same line as tool:use actions);
    # anything else is dropped. failure_mode is template vocabulary
    # (drive:hunger:discomfort) and passes through.
    scrubbed["percept_valences"] = {
        key: valence
        for key, valence in (nac_state.get("percept_valences", {}) or {}).items()
        if _IDENTIFIER_TOKEN.match(key.split(_NAC_KEY_SEP, 2)[1])
    }

    # cluster_reward_bias: scrub the tsig third of the key; mean on
    # collision (matches nac_merge's bias semantics).
    merged_cluster: dict[str, list[float]] = {}
    for key, bias in (nac_state.get("cluster_reward_bias", {}) or {}).items():
        aid, cid, tsig = key.split(_NAC_KEY_SEP, 2)
        new_key = _NAC_KEY_SEP.join((aid, cid, _scrub_event_signature(tsig)))
        merged_cluster.setdefault(new_key, []).append(float(bias))
    scrubbed["cluster_reward_bias"] = {k: sum(v) / len(v) for k, v in merged_cluster.items()}

    # cluster_reward_source (present since the S1 provenance fold; older
    # dumps lack it): same key scrub; disagreeing sources promote to
    # "mixed", NAc's own semantics for multi-source accumulation.
    if "cluster_reward_source" in nac_state:
        merged_source: dict[str, str] = {}
        for key, src in (nac_state.get("cluster_reward_source", {}) or {}).items():
            aid, cid, tsig = key.split(_NAC_KEY_SEP, 2)
            new_key = _NAC_KEY_SEP.join((aid, cid, _scrub_event_signature(tsig)))
            if new_key in merged_source and merged_source[new_key] != src:
                merged_source[new_key] = "mixed"
            else:
                merged_source[new_key] = src
        scrubbed["cluster_reward_source"] = merged_source

    return scrubbed


# Absolute filesystem path (POSIX, home-relative, or Windows drive) —
# the shape a local ``EncoderConfig(model_name="/Users/x/models/…")``
# stamps into encode-time provenance. Hub-style model names
# ("paraphrase-mpnet-base-v2") don't match.
_ABS_PATH_PATTERN = re.compile(r"^(/|~[/\\]|[A-Za-z]:[\\/])")

_REDACTED_PATH_MARKER = "[REDACTED_PATH]"


def _redact_paths_in_provenance(value: Any) -> Any:
    """Replace path-shaped strings in a provenance payload with a marker.

    ``EC.record_encoder_provenance`` accepts arbitrary JSON-serializable
    dicts and the manifest carries them verbatim — an operator pointing
    ``model_name`` at a local checkpoint would otherwise ship that
    filesystem path in every bundle. The marker (rather than dropping
    the key) keeps the provenance honest: it shows a local value was
    there without disclosing it. ``None`` stays ``None`` per the
    honest-unknown contract.
    """
    if isinstance(value, str):
        return _REDACTED_PATH_MARKER if _ABS_PATH_PATTERN.match(value) else value
    if isinstance(value, dict):
        return {k: _redact_paths_in_provenance(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_paths_in_provenance(v) for v in value]
    return value


def _filter_ec_nodes_by_domain(
    nodes: dict[str, dict[str, Any]],
    *,
    domain: str | None,
) -> dict[str, dict[str, Any]]:
    """Filter EC substrate_nodes by domain.

    Drops nodes with ``domain == IDENTITY_DOMAIN_MARKER`` unconditionally.
    When ``domain`` is non-None, additionally drops nodes whose ``domain``
    field doesn't match (passes through undomained ``None`` nodes — they
    are generic).
    """
    out: dict[str, dict[str, Any]] = {}
    for nid, nd in nodes.items():
        nd_domain = nd.get("domain")
        # Always drop reserved-identity nodes.
        if nd_domain == IDENTITY_DOMAIN_MARKER:
            continue
        # Per-domain filter: when caller scopes to a specific domain,
        # admit only that domain plus undomained generic nodes.
        if domain is not None and nd_domain is not None and nd_domain != domain:
            continue
        out[nid] = nd
    return out


class BundleBodyMismatch(ValueError):
    """A bundle was learned on a different body than the receiver's.

    Gate 7. Raised by :func:`assert_bundle_body_compatible`. The point is
    LOUDNESS: without the check, a cross-body bundle merges "successfully",
    contributes exactly 0.0 (its tool signatures carry the sender's entity
    name — D43 barrier 3), and reads out as "this agent has learned nothing
    yet". See docs/plans/d43_merge_correctness.md §5a.
    """

    def __init__(self, *, bundle_body: str | None, receiver_body: str) -> None:
        self.bundle_body = bundle_body
        self.receiver_body = receiver_body
        super().__init__(
            f"bundle was learned on body {bundle_body!r} but the receiver is {receiver_body!r}. "
            "Tool signatures are entity-prefixed, so merging this bundle would report success and "
            "contribute exactly 0.0. Re-export from the receiver's body, or adopt a capability "
            "namespace (docs/plans/d43_merge_correctness.md §5a)."
        )


class BundleBodyUnverifiable(ValueError):
    """The bundle predates gate 7 and does not declare the body it came from."""

    def __init__(self, *, receiver_body: str) -> None:
        self.receiver_body = receiver_body
        super().__init__(
            "bundle does not declare `body_ref` (pre-gate-7 bundle, schema v1). Its body of origin "
            f"cannot be established, so compatibility with {receiver_body!r} is UNVERIFIABLE — not "
            "confirmed. Pass allow_unverified=True to accept the risk explicitly."
        )


def assert_bundle_body_compatible(
    manifest: dict[str, Any],
    *,
    receiver_body: str,
    allow_unverified: bool = False,
) -> None:
    """Refuse a bundle whose body of origin differs from the receiver's.

    Gate 7's whole content: make the cross-body case LOUD. Three outcomes —
    match returns silently; mismatch raises :class:`BundleBodyMismatch`;
    a bundle that declares no body raises :class:`BundleBodyUnverifiable`
    unless ``allow_unverified``.

    **Absence is not compatibility.** A v1 bundle migrated to v2 carries
    ``body_ref: None`` because its origin is genuinely unknown, and this
    refuses it by default rather than letting it pass a check it was never
    subject to — the same reasoning as the format-version contract's
    ``"0.x"`` sentinel.
    """
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest must be dict, got {type(manifest).__name__}")
    if not receiver_body:
        raise ValueError("receiver_body must be a non-empty string")
    bundle_body = manifest.get("body_ref")
    if bundle_body is None:
        if allow_unverified:
            return
        raise BundleBodyUnverifiable(receiver_body=receiver_body)
    if str(bundle_body) != str(receiver_body):
        raise BundleBodyMismatch(bundle_body=str(bundle_body), receiver_body=str(receiver_body))


def compose_bundle(
    *,
    nac_state: dict[str, Any] | None,
    ec_substrate_nodes: dict[str, dict[str, Any]] | None,
    output_path: str | Path,
    contributor_id: str,
    domain: str | None = None,
    apply_identity_filter: bool = True,
    identity_threshold: int = _DEFAULT_BUNDLE_IDENTITY_THRESHOLD,
    signature: str | None = None,
    signature_algorithm: str | None = None,
    signer_identity: str | None = None,
    encoder_provenance: dict[str, Any] | None = None,
    body_ref: str | None = None,
    affordance_namespace: str | None = None,
    capability_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Compose a substrate snapshot bundle.

    Pure function: inputs are not mutated. Writes a zip file at
    ``output_path``. Returns the manifest dict.

    Parameters
    ----------
    nac_state
        Output of ``NAc.dump()`` — the shareable NAc state. ``None``
        skips the ``nac.json`` slice.
    ec_substrate_nodes
        The ``substrate_nodes`` slice from ``EC.save()``'s payload
        (``json.loads(ec.json)["substrate_nodes"]``). ``None`` skips
        the ``ec.json`` slice.
    output_path
        Where to write the zip file. Parent directory is created if
        absent.
    contributor_id
        Opaque ID identifying which Maxim composed the bundle. Goes
        into the manifest. Per PR B's reserved-namespace rule, this
        MUST NOT start with the reserved ``_`` prefix.
    domain
        Optional substrate-domain scope (``"combat"``, ``"cooking"``,
        ...). When set, EC nodes whose ``domain`` field disagrees are
        dropped (undomained nodes still ride along).
    apply_identity_filter
        When True (default), drops identity-bearing NAc event
        signatures + reserved-identity-domain EC nodes from the
        bundle. False disables the filter for trusted-internal
        backups.
    identity_threshold
        Threshold passed to
        :func:`maxim.hivemind.identity.is_identity_bearing` when
        filtering. Default 2 — bundle-stricter than the heuristic's
        default of 1, per PR C's game-substrate fold.
    signature, signature_algorithm
        Reserved slots — populate at the caller's discretion. This
        module does NOT compute signatures and does NOT validate
        them at extract time. Default ``None`` / ``None``. Recognized
        ``signature_algorithm`` values are published in
        ``docs/user/hivemind_bundle_format.md`` (the 1.2 P2P verifier
        vocabulary); the registry is documentation-only at 1.0.
    signer_identity
        Reserved slot (CC13 auth format-freeze) — the "who claims to
        have signed this" string, parallel to ``contributor_id``.
        Always ``None`` at 1.0; reserved so 1.1+ bundle verification
        can bind a verified identity to ``contributor_id`` without
        retrofitting the manifest shape. NOT validated here.
    encoder_provenance
        Encode-time encoder stamps from the source EC payload
        (``ec.json``'s ``encoder_provenance`` key — recorded by the
        encoders via ``EC.record_encoder_provenance``, never authored
        post-hoc). Carried into ``manifest["encoder_provenance"]
        ["recorded"]`` verbatim; ``None`` (pre-stamping payloads) is
        carried as ``None`` — an honest "unknown", not a fabricated
        default. Independent of this parameter, the manifest ALWAYS
        carries ``observed_embedding_dims`` derived from the ACTUAL
        arrays in the EC slice at write time (checked truth, per the
        fabric plan's "stamp the realized state, not its name" rule).

        MERGE SEMANTICS (pinned for 1.2 — do not build on the naive
        reading): ``recorded`` describes the COMPOSING substrate's own
        encoders only. A substrate that previously imported foreign
        nodes via ``ec_merge`` ships arrays encoded elsewhere that its
        local stamps do not describe — the 1.2 P2P merge must union
        provenance per-contributor rather than trusting a merged
        substrate's local stamps. ``observed_embedding_dims`` is the
        measured backstop either way, but dims alone cannot distinguish
        a 384-dim fallback from a real 384-dim model.
    """
    # Fold (Executor IMPORTANT): route through the same validator
    # PR B's merge functions use, instead of duplicating the
    # reserved-prefix check inline. This catches the divergence the
    # reviewer flagged: PR B's _validate_source also rejects empty
    # strings and non-string types, which the inline check missed.
    _validate_source(contributor_id, label="contributor_id")

    # Snapshot the input pieces (NAc state + filtered EC nodes).
    bundle_contents: dict[str, str] = {}  # filename -> serialized JSON

    if nac_state is not None:
        filtered_nac = dict(nac_state)
        if apply_identity_filter:
            filtered_links = filter_identity_bearing_links(
                filtered_nac.get("links", {}) or {},
                threshold=identity_threshold,
            )
            filtered_nac["links"] = filtered_links
            # The same event-signature strings the links filter drops
            # also key event_outcome_welford — without this, an
            # identity-quarantined signature ships anyway through its
            # Welford twin (PR #506 audit). cluster_reward_bias tsigs
            # are deliberately NOT filtered here: they are
            # build_tool_signature output (template except tool:use:
            # tails, which the unconditional scrub already truncates),
            # and the identity heuristic needs whitespace-separated
            # tokens it never contains.
            filtered_nac["event_outcome_welford"] = {
                key: state
                for key, state in (filtered_nac.get("event_outcome_welford", {}) or {}).items()
                if not is_identity_bearing(key.partition(_NAC_KEY_SEP)[2], threshold=identity_threshold)
            }
        # Content scrub is UNCONDITIONAL (see the scrub section above) —
        # the AST guard test pins that every nac.json assignment routes
        # through scrub_nac_state_for_bundle inline in this call.
        bundle_contents["nac.json"] = json.dumps(
            scrub_nac_state_for_bundle(filtered_nac), indent=2, sort_keys=True, default=str
        )

    observed_embedding_dims: dict[str, list[int]] = {}
    if ec_substrate_nodes is not None:
        ec_nodes_filtered = _filter_ec_nodes_by_domain(ec_substrate_nodes, domain=domain)
        bundle_contents["ec.json"] = json.dumps(
            {"substrate_nodes": ec_nodes_filtered},
            indent=2,
            sort_keys=True,
            default=str,
        )
        # Artifact stamping (1.1 item 7): dims measured on the ACTUAL
        # arrays being shipped — a per-modality dim SET, so a mixed-space
        # slice (the #467 corruption class) is visible in the manifest
        # rather than discovered at merge time.
        dims_by_modality: dict[str, set[int]] = {}
        for node in ec_nodes_filtered.values():
            modality = str(node.get("modality") or "unknown")
            emb = node.get("embedding") or []
            dims_by_modality.setdefault(modality, set()).add(len(emb))
        observed_embedding_dims = {m: sorted(d) for m, d in sorted(dims_by_modality.items())}

    manifest: dict[str, Any] = {
        "_format_version": FORMAT_VERSION,
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "kind": BUNDLE_KIND,
        "contributor_id": contributor_id,
        "domain": domain,
        "created_at": _utc_now_iso(),
        "identity_filter_applied": bool(apply_identity_filter),
        "identity_threshold": int(identity_threshold) if apply_identity_filter else None,
        "contents": {
            slice_name.removesuffix(".json"): {"file": slice_name} for slice_name in sorted(bundle_contents.keys())
        },
        "encoder_provenance": {
            "observed_embedding_dims": observed_embedding_dims,
            "recorded": _redact_paths_in_provenance(encoder_provenance),
        },
        "signature": signature,
        "signature_algorithm": signature_algorithm,
        "signer_identity": signer_identity,
        # Gate 7 (typed bundles). `body_ref` is the body this substrate was
        # learned on; a receiver checks IT via `assert_bundle_body_compatible`
        # and REFUSES a mismatch, converting a silent cross-body miss (D43
        # barrier 3) into a loud one. `affordance_namespace` names the
        # vocabulary the tool signatures live in — declarative today, no
        # reader yet.
        "body_ref": body_ref,
        "affordance_namespace": affordance_namespace,
        # Forward insurance, and the reason to prefer this over plain gate 7(a):
        # the body-agnostic capability key `(modulator, affordance)` for each
        # body-prefixed tool signature. Bundles carry BOTH keys from day one, so
        # adopting a capability namespace later is a READER-side change with no
        # migration — which is the half `register_bundle_migration` cannot cover,
        # since it migrates the manifest and never the keyed payload.
        # See docs/plans/d43_merge_correctness.md §5a.
        "capability_map": dict(capability_map or {}),
    }
    manifest_json = json.dumps(manifest, indent=2, sort_keys=True, default=str)

    # Atomic write via tmp + os.replace. Zip writing is single-shot;
    # if any step raises we tear down the tmp file.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", manifest_json)
            for filename, content in bundle_contents.items():
                zf.writestr(filename, content)
        os.replace(tmp_path, output_path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError as cleanup_err:
                logger.warning("Failed to clean up %s: %s", tmp_path, cleanup_err)
        raise

    logger.info("Composed substrate bundle at %s (%d slices)", output_path, len(bundle_contents))
    return manifest


def _safe_join(output_dir: Path, name: str) -> Path:
    """Resolve ``output_dir / name`` and reject ZIP-slip escape attempts.

    Pre-merge review CRITICAL (Executor lens): a malicious bundle entry
    named ``../../../etc/passwd`` (or an absolute path, or any name
    containing ``..``) resolves outside ``output_dir`` and would let a
    crafted bundle clobber arbitrary files on extract. The 1.2 P2P
    protocol will exchange bundles between peers, so this is a real
    threat surface even before the import verb is widely used.

    Rejects: empty names, absolute paths, and any resolved target that
    falls outside ``output_dir`` (catches ``..`` traversal + symlink
    escape). Returns the safe resolved Path on success; raises
    ``ValueError`` otherwise.
    """
    if not name:
        raise ValueError("bundle contains an empty path entry; refusing to extract")
    name_path = Path(name)
    if name_path.is_absolute():
        raise ValueError(f"bundle entry {name!r} is an absolute path; refusing to extract")
    candidate = (output_dir / name_path).resolve()
    base = output_dir.resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"bundle entry {name!r} resolves outside output_dir; refusing to extract (ZIP slip)") from exc
    return candidate


def extract_bundle(
    bundle_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Extract a substrate bundle to a directory.

    Validates the manifest's ``kind`` + ``schema_version`` +
    ``_format_version`` (via :func:`check_format_version`). Writes
    ``manifest.json``, ``nac.json``, and ``ec.json`` (whichever are
    present) to ``output_dir``. Returns the parsed manifest dict.

    Does NOT auto-merge into a live NAc / EC — that's the caller's
    decision. The 1.1 Oasis software will wrap this in a pipeline that
    calls ``nac_merge`` / ``ec_merge`` against the extracted dicts; the
    1.0 CLI verb just round-trips the data.

    Every ZIP entry is routed through :func:`_safe_join` before being
    written — absolute paths, ``..`` traversal, and symlink escape are
    all rejected (ZIP-slip CVE class). A malicious bundle with one
    safe slice and one escape slice writes nothing — the safety check
    runs in a pre-validation pass before any disk writes.

    Raises ``ValueError`` on a manifest with wrong kind or unrecognized
    schema_version, or on any ZIP entry that fails the path-safety
    check. Raises ``zipfile.BadZipFile`` on a malformed archive.
    """
    bundle_path = Path(bundle_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(bundle_path, "r") as zf:
        if "manifest.json" not in zf.namelist():
            raise ValueError(f"bundle {bundle_path} missing manifest.json")
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError(f"manifest.json must be a JSON object, got {type(manifest).__name__}")

        # Bundle envelope migration (post-fold scaffolding for 1.1+):
        # routes the manifest through the migration registry before the
        # kind / schema_version validation runs. At 1.0 the registry is
        # empty so this is a no-op; 1.1's first bundle migration just
        # registers a ``v1 → v2`` function and existing 1.0 bundles
        # upgrade transparently.
        manifest = migrate_bundle_envelope(manifest)

        check_format_version(manifest, "substrate_bundle", log=logger)

        if manifest.get("kind") != BUNDLE_KIND:
            raise ValueError(f"manifest kind {manifest.get('kind')!r} != {BUNDLE_KIND!r}")
        schema_v = manifest.get("schema_version")
        if not isinstance(schema_v, int) or schema_v > BUNDLE_SCHEMA_VERSION:
            raise ValueError(
                f"manifest schema_version {schema_v!r} unsupported (this build supports up to {BUNDLE_SCHEMA_VERSION})"
            )

        # Pre-validate EVERY entry path before writing anything — a
        # bundle with one good slice and one ZIP-slip slice writes
        # nothing.
        safe_targets: list[tuple[str, Path]] = []
        for name in sorted(zf.namelist()):
            safe_targets.append((name, _safe_join(output_dir, name)))

        for name, target in safe_targets:
            if name == "manifest.json":
                # Re-serialize the (possibly migrated) manifest for the
                # extracted copy so external readers see the upgraded
                # shape instead of the on-disk legacy one.
                atomic_write_text(str(target), json.dumps(manifest, indent=2, sort_keys=True))
                continue
            content = zf.read(name).decode("utf-8")
            atomic_write_text(str(target), content)

    logger.info("Extracted substrate bundle from %s to %s", bundle_path, output_dir)
    return manifest


def read_bundle_manifest(bundle_path: str | Path) -> dict[str, Any]:
    """Read the manifest from a bundle without extracting it.

    Convenience for CLI ``maxim substrate inspect`` and 1.1 Oasis
    discovery. Validates ``kind`` + ``schema_version`` like
    :func:`extract_bundle`.
    """
    bundle_path = Path(bundle_path)
    with zipfile.ZipFile(bundle_path, "r") as zf:
        if "manifest.json" not in zf.namelist():
            raise ValueError(f"bundle {bundle_path} missing manifest.json")
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))

    if not isinstance(manifest, dict):
        raise ValueError(f"manifest.json must be a JSON object, got {type(manifest).__name__}")
    manifest = migrate_bundle_envelope(manifest)
    check_format_version(manifest, "substrate_bundle", log=logger)
    if manifest.get("kind") != BUNDLE_KIND:
        raise ValueError(f"manifest kind {manifest.get('kind')!r} != {BUNDLE_KIND!r}")
    schema_v = manifest.get("schema_version")
    if not isinstance(schema_v, int) or schema_v > BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"manifest schema_version {schema_v!r} unsupported (this build supports up to {BUNDLE_SCHEMA_VERSION})"
        )
    return manifest


__all__ = [
    "BUNDLE_KIND",
    "BUNDLE_SCHEMA_VERSION",
    "BundleBodyMismatch",
    "BundleBodyUnverifiable",
    "assert_bundle_body_compatible",
    "compose_bundle",
    "extract_bundle",
    "isolated_bundle_migrations",
    "migrate_bundle_envelope",
    "read_bundle_manifest",
    "register_bundle_migration",
    "scrub_nac_state_for_bundle",
]
