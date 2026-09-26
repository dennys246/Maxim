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
import hashlib
import io
import json
import logging
import os
import re
import zipfile
import zlib
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from maxim.hivemind.identity import (
    IDENTITY_DOMAIN_MARKER,
    filter_identity_bearing_links,
    is_identity_bearing,
)
from maxim.hivemind.merge import NAC_KEY_SEP, NODE_ID_CHARSET, _merge_link_pair, _merge_welford, _validate_source
from maxim.hivemind.entry_index import EntryIndexError, build_index, normalize_agent_segment, verify_index
from maxim.utils.optional_deps import OptionalDependencyError
from maxim.hivemind.signing import (
    SIGNATURE_ALGORITHM,
    SIGNATURE_MEMBER,
    SIGNATURE_SCHEME_V2,
    SignedRelease,
    bundle_signing_payload,
    bundle_signing_payload_v2,
    validate_license,
    validate_release_sequence,
    verify_payload,
)
from maxim.utils.atomic_io import atomic_write_text
from maxim.utils.format_version import FORMAT_VERSION, check_format_version

logger = logging.getLogger(__name__)

# Schema version for the bundle envelope itself. Separate from the
# bio-system payload ``_format_version`` — bumping this would require
# a migration registered alongside the bump.
BUNDLE_SCHEMA_VERSION: int = 3

#: The schema an UNSIGNED bundle is written at. The number says what a reader must understand: an
#: unsigned contribution needs nothing schema 3 added (its manifest only drops the null signature slots
#: and may carry a license, both read through ``.get``), so it stays readable by 1.3.x peers and Oasis
#: servers; a signed v2 release is schema 3, which an older reader correctly refuses -- it cannot
#: verify it.
UNSIGNED_BUNDLE_SCHEMA_VERSION: int = 2

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


@register_bundle_migration(2)
def _v2_to_v3_signed_releases(manifest: dict[str, Any]) -> dict[str, Any]:
    """v2 → v3: the release format v2 (docs/plans/oasis_entry_index_v2.md) — a v2 manifest gains nothing.

    A v2 manifest's signature fields (if any) are a v1 signature over the manifest AS STORED: verify
    before migrating (``verify_bundle_zip`` reads the raw manifest), because this migration rewrites
    ``schema_version``, which that signature covers.
    """
    out = dict(manifest)
    out["schema_version"] = 3
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


def scrub_cluster_fear_for_bundle(fear: Any) -> dict[str, float]:
    """The export-side filter for Wire-4 fear (Exp 61): well-formed triple keys only (charset as
    the ingest bound), allowlisted failure mode only, strictly NEGATIVE values clamped to ``[-1, 0)``
    (a zero or positive value is not a fear and would only inflate the receiver's counters);
    anything else is dropped here
    (the receiver's ingest bound REFUSES the same shapes — the two sites agree by
    reading one allowlist). Pure; returns a fresh dict."""
    from maxim.decisions.nac import DEFAULT_CLUSTER_FEAR_FAILURE_MODES  # noqa: PLC0415 — no cycle; lazy by choice

    if not isinstance(fear, dict):
        return {}
    kept: dict[str, float] = {}
    for key, value in fear.items():
        parts = str(key).split(NAC_KEY_SEP)
        if len(parts) != 3 or not all(parts) or not NODE_ID_CHARSET.match(parts[1]):
            continue  # the receiver would refuse the whole bundle (V2/V9); filter by the same rule
        if parts[2] not in DEFAULT_CLUSTER_FEAR_FAILURE_MODES:
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if v != v or v >= 0.0:  # NaN, or not a fear (zero / positive) — a zero fear must not count as one
            continue
        kept[str(key)] = max(-1.0, v)
    return kept


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
    # Wire-4 fear (`cluster_fear`) TRAVELS since Exp 61 (1.3 Phase 2, 2026-09-16;
    # the Exp 58 deferral is discharged). It ships in the exact shape
    # `NAc.dump()` writes — triple keys `agent\x1fcluster\x1ffailure_mode` —
    # clamped to [-1, 0] (fear only; a positive value is not fear) and filtered
    # to the Wire-4 allowlist, the same filter `NAc.record_cluster_fear`
    # applies on the write path. The receiver's ingest bound re-validates all
    # of this (refusal, not strip), applies the social discount, and re-keys
    # through the aligned EC id map; `nac_merge` MIN-folds. Nine sites move
    # together — see `docs/experiments/exp61_shared_fear_prereg.md` §Mechanism.
    scrubbed["cluster_fear"] = scrub_cluster_fear_for_bundle(nac_state.get("cluster_fear"))
    if not scrubbed["cluster_fear"]:
        scrubbed.pop("cluster_fear", None)

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

    # inherent_bias_keys (1.2 poison-resistance slice; older dumps lack it):
    # markers name cluster_reward_bias keys, so the tsig third gets the same
    # scrub — a marker left on the pre-scrub key would exempt nothing after
    # the re-key. Markers whose entry did not survive the scrub are dropped
    # (a marker naming an absent bias is the dangling-half shape).
    if "inherent_bias_keys" in nac_state:
        scrubbed_inherent: set[str] = set()
        for key in nac_state.get("inherent_bias_keys", []) or []:
            aid, cid, tsig = str(key).split(_NAC_KEY_SEP, 2)
            new_key = _NAC_KEY_SEP.join((aid, cid, _scrub_event_signature(tsig)))
            if new_key in scrubbed["cluster_reward_bias"]:
                scrubbed_inherent.add(new_key)
        scrubbed["inherent_bias_keys"] = sorted(scrubbed_inherent)

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


# ─────────────────────────────────────────────────────────────────────────
# Provenance at export (2026-09-25, owner decision (c)).
#
# A receiver's ingest (V1, ``ingest._sweep_provenance_value``) accepts only the
# manifest's own contributor or ``"local"`` as payload provenance. An exporter
# that has ingested other contributors' material holds rows whose ``source`` is
# a donor id or ``"_consensus"`` and whose ``contributors`` lists upstream ids --
# shipped verbatim, the bundle both PUBLISHED those ids and was REFUSED by every
# receiver. So export decides, explicitly:
#
# * default -- an agent contributes ONLY ITS OWN LEARNING: a link or EC node is
#   exported only when its provenance is the exporter's alone (no source, ``"local"``, or its own
#   ``contributor_id`` -- exactly what V1 accepts), re-stamped as the exporter's ``contributor_id``.
#   "Own" is narrower than "learned here": a ``_consensus`` row (local and foreign votes merged, even
#   a local row once folded with foreign material) is dropped, since its own share cannot be
#   separated. Cluster-keyed NAc rows naming a dropped node are dropped with it.
# * ``reauthor=True`` -- RELEASE COMPOSITION (the Queen publishing merged
#   contributions): every row is re-stamped as the author's (``source`` = ``contributors`` = its
#   ``contributor_id``); the release's signature carries the provenance.
#
# Rows with no provenance fields (cluster fear / reward bias, Welford, priors)
# cannot be filtered this way: a received fear folded into a local cluster
# re-exports as the exporter's; with no EC slice (a NAc-only export) cluster rows
# cannot be matched to dropped nodes at all; and a donor's AGENT id can survive in
# the agent segment of welford / percept-valence / reward-bias keys. Stated, not
# hidden. ``reauthor`` is Queen-intent by convention only: it can only claim
# others' learning as the exporter's own (receivers stamp the manifest
# contributor, and inherent trust keys on that id), never impersonate anyone.
# ─────────────────────────────────────────────────────────────────────────

_LOCAL_SOURCE = "local"  # the receiver's accepted self-reference (ingest._SELF_SOURCE)
_CLUSTER_KEYED_NAC_FIELDS = ("cluster_reward_bias", "cluster_reward_source", "cluster_fear")


def _is_own(entry: dict[str, Any], contributor_id: str) -> bool:
    """True when a link/node's provenance is the exporter's alone -- exactly what a receiver's V1
    sweep accepts: no source, ``"local"``, or the exporter's own ``contributor_id``."""
    own = (None, _LOCAL_SOURCE, contributor_id)
    if entry.get("source") not in own:
        return False
    contributors = entry.get("contributors")
    if contributors is None:
        return True
    if not isinstance(contributors, (list, tuple)):  # V1 refuses a non-list outright: never ship one
        return False
    return all(c in own for c in contributors)


def _cluster_of(key: Any) -> str | None:
    """The cluster id of an ``agent␟cluster␟x`` NAc key, or None for another shape."""
    parts = str(key).split(NAC_KEY_SEP)
    return parts[1] if len(parts) == 3 else None


def _reauthored(entry: dict[str, Any], contributor_id: str) -> dict[str, Any]:
    """Stamp a row as the EXPORTER's: its own ``contributor_id`` -- which V1 accepts, and which the
    non-ingest ``substrate import`` + merge path attributes correctly (a ``"local"`` stamp would make
    the IMPORTER read the row as its own). A single contributor, so a later fold stays single."""
    out = dict(entry)
    out["source"] = contributor_id
    out["contributors"] = [contributor_id]
    return out


def provenance_for_export(
    nac_state: dict[str, Any] | None,
    ec_nodes: dict[str, dict[str, Any]] | None,
    *,
    contributor_id: str,
    reauthor: bool,
) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]] | None]:
    """Apply the export provenance rule (see the section comment). Pure; returns fresh dicts."""
    nodes_out: dict[str, dict[str, Any]] | None = None
    dropped_nodes: set[str] = set()
    if ec_nodes is not None:
        nodes_out = {}
        for nid, node in ec_nodes.items():
            if reauthor:
                nodes_out[nid] = _reauthored(node, contributor_id)
            elif _is_own(node, contributor_id):
                nodes_out[nid] = _reauthored(node, contributor_id)
            else:
                dropped_nodes.add(nid)
    nac_out: dict[str, Any] | None = None
    if nac_state is not None:
        nac_out = dict(nac_state)
        links_out: dict[str, list[dict[str, Any]]] = {}
        for sig, links in (nac_state.get("links", {}) or {}).items():
            # Kept rows are re-stamped as the exporter's in BOTH modes: two own links (one "local",
            # one the own contributor id) that the signature scrub later folds would otherwise merge
            # to "_consensus", which V1 refuses (review round).
            kept = [_reauthored(link, contributor_id) for link in links if reauthor or _is_own(link, contributor_id)]
            if kept:
                links_out[sig] = kept
        nac_out["links"] = links_out
        if dropped_nodes:
            for field in _CLUSTER_KEYED_NAC_FIELDS:
                rows = nac_state.get(field)
                if isinstance(rows, dict):
                    nac_out[field] = {k: v for k, v in rows.items() if _cluster_of(k) not in dropped_nodes}
    return nac_out, nodes_out


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
    yet". See docs/plans/archive/d43_merge_correctness.md §5a.
    """

    def __init__(self, *, bundle_body: str | None, receiver_body: str) -> None:
        self.bundle_body = bundle_body
        self.receiver_body = receiver_body
        super().__init__(
            f"bundle was learned on body {bundle_body!r} but the receiver is {receiver_body!r}. "
            "Tool signatures are entity-prefixed, so merging this bundle would report success and "
            "contribute exactly 0.0. Re-export from the receiver's body, or adopt a capability "
            "namespace (docs/plans/archive/d43_merge_correctness.md §5a)."
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
    release: SignedRelease | None = None,
    license: str | None = None,
    encoder_provenance: dict[str, Any] | None = None,
    body_ref: str | None = None,
    affordance_namespace: str | None = None,
    capability_map: dict[str, str] | None = None,
    reauthor: bool = False,
    agent_id: str | None = None,
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
    release
        A :class:`~maxim.hivemind.signing.SignedRelease` makes this a SIGNED v2 release
        (docs/plans/oasis_entry_index_v2.md): the NAc agent segment is normalized, the manifest
        carries ``signer_identity`` / ``release_sequence`` / ``license`` / ``entry_index``, and a
        detached ``signature.json`` signs every other member's raw bytes. ``None`` = unsigned.
    license
        An unsigned bundle's license (SPDX id), or ``None`` (the default: signing is where terms are
        required). A release carries its own, so passing both is refused.
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

    # Provenance first (see "Provenance at export"): an agent ships only its own learning unless
    # this is RELEASE composition (``reauthor=True``), which re-stamps every row as the author's --
    # and a release is SIGNED, enforced here, not only in the CLI.
    if reauthor and release is None:
        raise ValueError("reauthor=True composes a release, and a release must be signed: pass release=")
    if release is not None and license is not None:
        raise ValueError("a release carries its license in release=; do not also pass license=")
    if license is not None:
        validate_license(license)
    if agent_id is not None and release is None:
        raise ValueError("agent_id= selects whose rows a signed release ships; it needs release=")
    nac_state, ec_substrate_nodes = provenance_for_export(
        nac_state, ec_substrate_nodes, contributor_id=contributor_id, reauthor=reauthor
    )

    if nac_state is not None:
        filtered_nac = dict(nac_state)
        if release is not None:
            # A release ships ONE agent's learning under a fixed agent token (the receiver re-keys it):
            # local agent ids never ship, and rows filed under any other agent -- which this agent
            # never reads -- are dropped, never relabelled onto its own keys.
            filtered_nac, foreign_rows = normalize_agent_segment(filtered_nac, own_agent_id=agent_id)
            if foreign_rows:
                logger.warning(
                    "release: dropped %d NAc row(s) filed under other agents (never read by this agent; "
                    "a release ships its own agent's learning only)",
                    foreign_rows,
                )
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
        "schema_version": BUNDLE_SCHEMA_VERSION if release is not None else UNSIGNED_BUNDLE_SCHEMA_VERSION,
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
        "license": release.license if release is not None else license,
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
        # See docs/plans/archive/d43_merge_correctness.md §5a.
        "capability_map": dict(capability_map or {}),
    }

    # Release format v2: the identity, the sequence and the entry index are MANIFEST fields, so the
    # detached signature over the raw member bytes covers them (no field is written after signing).
    if release is not None:
        manifest["signer_identity"] = release.signer.signer_identity
        manifest["release_sequence"] = release.release_sequence
        manifest["entry_index"] = build_index(
            json.loads(bundle_contents["nac.json"]) if "nac.json" in bundle_contents else None,
            json.loads(bundle_contents["ec.json"])["substrate_nodes"] if "ec.json" in bundle_contents else None,
        )

    manifest_json = json.dumps(manifest, indent=2, sort_keys=True, default=str)
    members: dict[str, bytes] = {"manifest.json": manifest_json.encode("utf-8")}
    members.update({name: content.encode("utf-8") for name, content in bundle_contents.items()})
    if release is not None:
        members[SIGNATURE_MEMBER] = json.dumps(
            {
                "signature_scheme": SIGNATURE_SCHEME_V2,
                "signature_algorithm": SIGNATURE_ALGORITHM,
                "signature": release.signer.sign_payload(bundle_signing_payload_v2(members)),
            },
            indent=2,
            sort_keys=True,
        ).encode("utf-8")

    # Atomic write via tmp + os.replace. Zip writing is single-shot;
    # if any step raises we tear down the tmp file.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, data in members.items():
                zf.writestr(name, data)
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


def verify_bundle_signature_parts(
    manifest: Mapping[str, Any],
    slices: Mapping[str, str],
    *,
    trusted_keys: Mapping[str, str],
) -> tuple[bool, str]:
    """Verify a bundle signature from already-read parts (the ingest seam).

    ``slices`` maps on-disk filename (``"nac.json"``/``"ec.json"``) to the
    RAW content string read from the archive — the same keys and bytes
    :func:`compose_bundle` signed. ``trusted_keys`` maps ``signer_identity``
    → base64 public key. Returns ``(verified, reason)``; never raises on a
    bad/absent signature (an unverifiable bundle is a policy decision the
    caller makes, not an exception).
    """
    sig = manifest.get("signature")
    algo = manifest.get("signature_algorithm")
    signer = manifest.get("signer_identity")
    if not sig or not algo or not signer:
        return False, "bundle carries no signature (signature/signature_algorithm/signer_identity absent)"
    if algo != SIGNATURE_ALGORITHM:
        return False, f"unsupported signature_algorithm {algo!r} (this build verifies only {SIGNATURE_ALGORITHM!r})"
    if not isinstance(signer, str) or signer not in trusted_keys:
        return False, f"signer_identity {signer!r} is not among the receiver's trusted keys {sorted(trusted_keys)}"
    # ``signer_identity`` is NOT in the signed payload (it is written after signing), so it is safe only
    # because it selects WHICH key must verify: relabelling a bundle to another identity makes that
    # identity's key fail. The one case where a relabel would still verify is one key registered under
    # two identities -- then the signature cannot bind to either label, so it is refused outright.
    # Compare DECODED key bytes: a 32-byte key has four valid base64 spellings (the last character
    # carries two unused bits, which b64decode ignores), so a string comparison misses the alias.
    import base64
    import binascii

    def _key_bytes(value: str) -> bytes | None:
        try:
            return base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError):
            return None

    signer_key = _key_bytes(trusted_keys[signer])
    aliases = sorted(
        i for i, k in trusted_keys.items() if i != signer and signer_key is not None and _key_bytes(k) == signer_key
    )
    if aliases:
        return False, (
            f"the key trusted for {signer!r} is also trusted as {aliases}: one key under two identities "
            "cannot say which one signed (register each key once)"
        )
    payload = bundle_signing_payload(manifest, slices)
    if not verify_payload(payload, str(sig), trusted_keys[signer]):
        return False, f"ed25519 signature does not verify for signer {signer!r} (tampered or wrong key)"
    return True, f"verified ed25519 signature from {signer!r}"


# ─────────────────────────────────────────────────────────────────────────
# Release format v2 verification (docs/plans/oasis_entry_index_v2.md)
# ─────────────────────────────────────────────────────────────────────────

#: A v2 bundle member name: plain ASCII, no paths, no case games, no cp437/UTF-8 name divergence.
_MEMBER_NAME = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

#: V6 -- maximum ZIP members (3 canonical + signature.json today; headroom for 1.2+ slices).
MAX_BUNDLE_ENTRIES: int = 16
#: V6 -- per-member UNCOMPRESSED size cap, enforced on the ACTUAL decompressed bytes (the central
#: directory's declared size is an attacker assertion; see :func:`bounded_member_read`).
MAX_ENTRY_UNCOMPRESSED_BYTES: int = 64 * 1024 * 1024
#: V6 -- whole-archive uncompressed cap.
MAX_TOTAL_UNCOMPRESSED_BYTES: int = 128 * 1024 * 1024


class MemberReadError(ValueError):
    """A ZIP member that cannot be read within its caps -- corrupt, encrypted, or lying about its size."""


def bounded_member_read(zf: zipfile.ZipFile, name: str, *, max_bytes: int) -> bytes:
    """Decompress one member with the size cap enforced on ACTUAL bytes; raises only
    :class:`MemberReadError` on a bad member (``KeyError`` for a name the archive lacks -- callers
    check membership first).

    The central-directory ``file_size`` is itself an attacker assertion -- a binary-patched header can
    declare 10 bytes over an 800 MB stream and ``zf.read`` inflates the whole thing before the CRC
    check fires (executor-lens finding 3, measured at +1.3 GB RSS). Streaming through ``zf.open``
    with a capped read bounds memory to the cap regardless of what the headers claim.
    """
    try:
        with zf.open(name, "r") as fh:
            data = fh.read(max_bytes + 1)
    except (zipfile.BadZipFile, zlib.error, EOFError, NotImplementedError, RuntimeError, OSError) as exc:
        # A stream that CRC-fails at its declared boundary is corruption or a lying header truncated
        # by the bounded read; an unknown compression or an encrypted member is unreadable -- all are
        # refusals, not tracebacks.
        raise MemberReadError(f"entry {name!r} is corrupt or lies about its size: {exc}") from exc
    if len(data) > max_bytes:
        raise MemberReadError(
            f"entry {name!r} decompresses past {max_bytes} bytes despite its declared size -- lying "
            "central-directory header (zip bomb)"
        )
    return data


#: The index cap a verifier applies by default (the ingest V6 node cap).
_DEFAULT_MAX_ENTRIES = 50_000

_V1_DEPRECATION_WARNED = False


@dataclass(frozen=True)
class BundleVerification:
    """The outcome of :func:`verify_bundle_zip`. Runtime-ephemeral (returned, never persisted as-is).

    ``payload_digest`` is the signed payload's sha256 -- a release's identity for dedup and
    equivocation (stable across re-zips); ``signer_key`` the hex of the decoded public key that
    verified it; ``entry_digests`` the recomputed ``{entry id: digest}`` of a v2 release.
    """

    ok: bool
    reason: str
    scheme: int | None = None
    payload_digest: str | None = None
    signer_identity: str | None = None
    signer_key: str | None = None
    release_sequence: int | None = None
    license: str | None = None
    entry_digests: dict[str, str] = field(default_factory=dict)


def _strict_json(raw: bytes, what: str) -> Any:
    """Parse JSON refusing duplicate keys and non-finite numbers (two parsers must never disagree)."""

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in pairs:
            if k in out:
                raise ValueError(f"{what}: duplicate key {k!r}")
            out[k] = v
        return out

    def no_constants(name: str) -> Any:
        raise ValueError(f"{what}: non-finite number {name}")

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=no_duplicates, parse_constant=no_constants)
    except RecursionError as exc:
        raise ValueError(f"{what}: nests too deeply to parse") from exc


def _key_bytes(value: Any) -> bytes | None:
    import base64
    import binascii

    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError, TypeError):
        return None


def _declared_slice_files(manifest: Mapping[str, Any]) -> dict[str, str]:
    contents = manifest.get("contents") or {}
    if not isinstance(contents, Mapping):
        return {}
    return {
        name: meta["file"]
        for name, meta in contents.items()
        if isinstance(meta, Mapping) and isinstance(meta.get("file"), str)
    }


def verify_bundle_zip(
    zf: zipfile.ZipFile,
    *,
    trusted_keys: Mapping[str, str],
    accept_v1: bool,
    max_entries: int = _DEFAULT_MAX_ENTRIES,
    max_members: int = MAX_BUNDLE_ENTRIES,
    max_member_bytes: int = MAX_ENTRY_UNCOMPRESSED_BYTES,
    max_total_bytes: int = MAX_TOTAL_UNCOMPRESSED_BYTES,
) -> BundleVerification:
    """Verify an open bundle ZIP -- reading the manifest AS STORED, before any envelope migration.

    Scheme v2 (a ``signature.json`` member): the detached signature over every other member's raw
    bytes, the signed ``signer_identity`` / ``release_sequence`` / ``license``, and the entry index
    against the slices. Scheme v1 (signature fields in a schema <= 2 manifest): verified as before,
    only when ``accept_v1``, with a deprecation warning. ``accept_v1`` has no default: whether an
    Oasis still takes legacy bundles is the caller's decision, never an inherited one.

    Never raises on bad input -- every failure is ``ok=False`` with a reason. Every member is read at
    most once, through :func:`bounded_member_read`, under the per-member and total caps; a v2 release
    must consist of EXACTLY its manifest, its declared slices and ``signature.json``, checked before
    any slice (or undeclared member) is decompressed.
    """
    reader = _CappedReader(zf, max_member_bytes=max_member_bytes, max_total_bytes=max_total_bytes)
    try:
        return _verify_bundle_zip(
            zf, reader, trusted_keys=trusted_keys, accept_v1=accept_v1, max_entries=max_entries, max_members=max_members
        )
    except MemberReadError as exc:
        return BundleVerification(False, f"unreadable member: {exc}")


class _CappedReader:
    """Reads each member once, bounded per member and in total (V6 on actual bytes)."""

    def __init__(self, zf: zipfile.ZipFile, *, max_member_bytes: int, max_total_bytes: int) -> None:
        self._zf = zf
        self._max_member = max_member_bytes
        self._max_total = max_total_bytes
        self._cache: dict[str, bytes] = {}
        self._total = 0

    def __call__(self, name: str) -> bytes:
        if name not in self._cache:
            data = bounded_member_read(self._zf, name, max_bytes=self._max_member)
            self._total += len(data)
            if self._total > self._max_total:
                raise MemberReadError(f"the archive decompresses past {self._max_total} bytes in total")
            self._cache[name] = data
        return self._cache[name]


def _verify_bundle_zip(
    zf: zipfile.ZipFile,
    read: _CappedReader,
    *,
    trusted_keys: Mapping[str, str],
    accept_v1: bool,
    max_entries: int,
    max_members: int,
) -> BundleVerification:
    infos = zf.infolist()
    if len(infos) > max_members:
        return BundleVerification(False, f"bundle has {len(infos)} members (cap {max_members})")
    names = [i.filename for i in infos]
    if len(names) != len(set(names)):
        return BundleVerification(False, "duplicate ZIP member names (a reader could see different bytes)")
    bad = [n for n in names if not _MEMBER_NAME.match(n)]
    if bad:
        return BundleVerification(False, f"malformed member name(s) {bad[:3]}")
    if "manifest.json" not in names:
        return BundleVerification(False, "no manifest.json")
    try:
        raw = _strict_json(read("manifest.json"), "manifest.json")
    except MemberReadError:
        raise  # reported as "unreadable member", not as a JSON fault
    except (ValueError, UnicodeDecodeError) as exc:
        return BundleVerification(False, f"manifest.json is not strict JSON: {exc}")
    if not isinstance(raw, dict):
        return BundleVerification(False, "manifest.json is not an object")
    files = _declared_slice_files(raw)

    if SIGNATURE_MEMBER in names:
        return _verify_v2(read, names, raw, files, trusted_keys=trusted_keys, max_entries=max_entries)
    if raw.get("signature"):
        return _verify_v1(read, names, raw, files, trusted_keys=trusted_keys, accept_v1=accept_v1)
    return BundleVerification(False, "bundle carries no signature")


def _trusted_signer(signer: Any, trusted_keys: Mapping[str, str]) -> tuple[str | None, str]:
    """``(key, "")`` for a trusted, un-aliased signer; ``(None, reason)`` otherwise."""
    if not isinstance(signer, str) or signer not in trusted_keys:
        return None, f"signer_identity {signer!r} is not among the receiver's trusted keys {sorted(trusted_keys)}"
    signer_key = _key_bytes(trusted_keys[signer])
    aliases = sorted(
        i for i, k in trusted_keys.items() if i != signer and signer_key is not None and _key_bytes(k) == signer_key
    )
    if aliases:
        return None, (
            f"the key trusted for {signer!r} is also trusted as {aliases}: one key under two identities "
            "cannot say which one signed (register each key once)"
        )
    return trusted_keys[signer], ""


def _verify_v2(
    read: _CappedReader,
    names: list[str],
    raw: dict[str, Any],
    files: dict[str, str],
    *,
    trusted_keys: Mapping[str, str],
    max_entries: int,
) -> BundleVerification:
    try:
        sig_doc = _strict_json(read(SIGNATURE_MEMBER), SIGNATURE_MEMBER)
    except MemberReadError:
        raise  # reported as "unreadable member", not as a JSON fault
    except (ValueError, UnicodeDecodeError) as exc:
        return BundleVerification(False, f"{SIGNATURE_MEMBER} is not strict JSON: {exc}")
    if not isinstance(sig_doc, dict):
        return BundleVerification(False, f"{SIGNATURE_MEMBER} is not an object")
    scheme = sig_doc.get("signature_scheme")
    if isinstance(scheme, bool) or scheme != SIGNATURE_SCHEME_V2:
        return BundleVerification(False, f"unknown signature_scheme {scheme!r} (refused, never read as v1)")
    if sig_doc.get("signature_algorithm") != SIGNATURE_ALGORITHM:
        return BundleVerification(False, f"unsupported signature_algorithm {sig_doc.get('signature_algorithm')!r}")
    if raw.get("schema_version") != 3:
        return BundleVerification(False, f"a v2 signature on a schema {raw.get('schema_version')!r} manifest")
    if SIGNATURE_MEMBER in files.values():
        return BundleVerification(False, f"the manifest declares {SIGNATURE_MEMBER} as a slice")
    # A release is EXACTLY its manifest, its declared slices and its signature: nothing undeclared is
    # decompressed (V7), and a declared slice that is absent is not a release that was signed whole.
    expected = {"manifest.json", SIGNATURE_MEMBER, *files.values()}
    if set(names) != expected:
        extra, missing = sorted(set(names) - expected), sorted(expected - set(names))
        return BundleVerification(False, f"release members differ from its manifest: extra {extra}, missing {missing}")
    signer = raw.get("signer_identity")
    key, reason = _trusted_signer(signer, trusted_keys)
    if key is None:
        return BundleVerification(False, reason, scheme=2)
    payload = bundle_signing_payload_v2({n: read(n) for n in names})
    if not verify_payload(payload, str(sig_doc.get("signature")), key):
        return BundleVerification(False, f"ed25519 signature does not verify for signer {signer!r}", scheme=2)
    digest = hashlib.sha256(payload).hexdigest()
    try:
        sequence = validate_release_sequence(raw.get("release_sequence"))
        license = validate_license(raw.get("license"))
    except ValueError as exc:
        return BundleVerification(False, f"signed manifest field invalid: {exc}", scheme=2, payload_digest=digest)
    try:
        nac = _strict_json(read(files["nac"]), "nac.json") if "nac" in files else None
        ec = _strict_json(read(files["ec"]), "ec.json") if "ec" in files else None
        nodes = ec.get("substrate_nodes") if isinstance(ec, Mapping) else None
        entry_digests = verify_index(raw.get("entry_index"), nac, nodes, max_entries=max_entries)
    except OptionalDependencyError as exc:
        return BundleVerification(
            False, f"cannot check the entry index: {exc.fix_hint}", scheme=2, payload_digest=digest
        )
    except (EntryIndexError, ValueError, UnicodeDecodeError, KeyError) as exc:
        return BundleVerification(False, f"entry index refused: {exc}", scheme=2, payload_digest=digest)
    return BundleVerification(
        True,
        f"verified v2 release {sequence} from {signer!r}",
        scheme=2,
        payload_digest=digest,
        signer_identity=signer,
        signer_key=(_key_bytes(key) or b"").hex(),
        release_sequence=sequence,
        license=license,
        entry_digests=entry_digests,
    )


def _verify_v1(
    read: _CappedReader,
    names: list[str],
    raw: dict[str, Any],
    files: dict[str, str],
    *,
    trusted_keys: Mapping[str, str],
    accept_v1: bool,
) -> BundleVerification:
    global _V1_DEPRECATION_WARNED
    if not accept_v1:
        return BundleVerification(False, "a v1 (legacy) signature, and this Oasis accepts v2 releases only", scheme=1)
    version = raw.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or version > 2:
        return BundleVerification(False, f"a v1 signature on a schema {version!r} manifest (downgrade)", scheme=1)
    try:
        slices = {f: read(f).decode("utf-8") for f in files.values() if f in names}
    except UnicodeDecodeError as exc:
        return BundleVerification(False, f"a v1 slice is not UTF-8: {exc}", scheme=1)
    ok, reason = verify_bundle_signature_parts(raw, slices, trusted_keys=trusted_keys)
    if not ok:
        return BundleVerification(False, reason, scheme=1)
    if not _V1_DEPRECATION_WARNED:
        _V1_DEPRECATION_WARNED = True
        logger.warning("verified a v1 (legacy) bundle signature -- v1 is refused from 2.0; re-sign as a v2 release")
    signer = raw.get("signer_identity")
    if not isinstance(signer, str) or signer not in trusted_keys:  # verify_bundle_signature_parts checked it
        return BundleVerification(False, f"v1 signer {signer!r} is not trusted", scheme=1)
    return BundleVerification(
        True,
        reason,
        scheme=1,
        payload_digest=hashlib.sha256(bundle_signing_payload(raw, slices)).hexdigest(),
        signer_identity=signer,
        signer_key=(_key_bytes(trusted_keys[signer]) or b"").hex(),
    )


def content_payload_digest(
    zf: zipfile.ZipFile,
    *,
    max_member_bytes: int = MAX_ENTRY_UNCOMPRESSED_BYTES,
    max_total_bytes: int = MAX_TOTAL_UNCOMPRESSED_BYTES,
) -> str | None:
    """A bundle's payload identity, computed over EXACTLY what ingest reads -- without verifying anything.

    Ingest reads ``manifest.json`` and the declared slices, never ``signature.json`` or an undeclared
    member, so the identity covers those and nothing else: a schema-3 manifest in the v2 framing
    (``bundle_signing_payload_v2`` over the manifest + declared slices), a schema <= 2 manifest in the
    v1 framing (the manifest minus its signature fields + the declared slices), signed or not. Whenever
    the bundle verifies this equals :attr:`BundleVerification.payload_digest` (verification requires
    exactly that member set), so a stripped signature, an added README or a re-zip leave it unchanged.
    It needs no key and grants no authority: an identity for DEDUP only. ``None`` when the manifest is
    not strict JSON or a declared slice cannot be read within the caps.
    """
    read = _CappedReader(zf, max_member_bytes=max_member_bytes, max_total_bytes=max_total_bytes)
    try:
        names = set(zf.namelist())
        if "manifest.json" not in names:
            return None
        manifest_bytes = read("manifest.json")
        raw = _strict_json(manifest_bytes, "manifest.json")
        if not isinstance(raw, dict):
            return None
        # Every declared slice. (Ingest refuses a manifest declaring signature.json or manifest.json as one;
        # the v2 framing would skip a declared signature.json structurally, the v1 framing hashes it.)
        files = sorted({f for f in _declared_slice_files(raw).values() if f in names})
        if raw.get("schema_version") == 3:
            members = {"manifest.json": manifest_bytes, **{f: read(f) for f in files}}
            return hashlib.sha256(bundle_signing_payload_v2(members)).hexdigest()
        slices = {f: read(f).decode("utf-8") for f in files}
        return hashlib.sha256(bundle_signing_payload(raw, slices)).hexdigest()
    except (ValueError, UnicodeDecodeError):  # MemberReadError is a ValueError; _strict_json maps RecursionError
        return None


def bundle_signature_scheme(bundle_path: str | Path) -> int | None:
    """Which signing scheme a bundle CLAIMS -- ``2`` (a ``signature.json`` member), ``1`` (a
    ``signature`` in the manifest, the same test :func:`verify_bundle_zip` dispatches on), ``None``
    (unsigned). A PRESENCE check for routing and display only -- it verifies nothing and must never
    gate trust (:func:`verify_bundle_zip` does)."""
    with zipfile.ZipFile(bundle_path, "r") as zf:
        if SIGNATURE_MEMBER in zf.namelist():
            return SIGNATURE_SCHEME_V2
        try:
            raw = json.loads(
                bounded_member_read(zf, "manifest.json", max_bytes=MAX_ENTRY_UNCOMPRESSED_BYTES).decode("utf-8")
            )
        except (KeyError, ValueError, UnicodeDecodeError, RecursionError):
            return None
    return 1 if isinstance(raw, dict) and raw.get("signature") else None


def verify_bundle_signature(
    bundle_path: str | Path,
    *,
    trusted_keys: Mapping[str, str],
    accept_v1: bool,
) -> tuple[bool, str]:
    """Open a bundle and verify its signature against ``trusted_keys``.

    Standalone convenience over :func:`verify_bundle_signature_parts` —
    reads the manifest and the RAW declared-slice bytes from the ZIP.
    Returns ``(verified, reason)``.
    """
    with zipfile.ZipFile(bundle_path, "r") as zf:
        result = verify_bundle_zip(zf, trusted_keys=trusted_keys, accept_v1=accept_v1)
    return result.ok, result.reason


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
        # The one manifest reader (bounded, migrated, kind/schema/format validated).
        manifest = _manifest_from_zip(zf, str(bundle_path))
        read = _CappedReader(
            zf, max_member_bytes=MAX_ENTRY_UNCOMPRESSED_BYTES, max_total_bytes=MAX_TOTAL_UNCOMPRESSED_BYTES
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
            content = read(name).decode("utf-8")
            atomic_write_text(str(target), content)

    logger.info("Extracted substrate bundle from %s to %s", bundle_path, output_dir)
    return manifest


def _manifest_from_zip(zf: zipfile.ZipFile, source_label: str) -> dict[str, Any]:
    """Read + validate ``manifest.json`` from an open bundle ZIP (kind/schema/format)."""
    if "manifest.json" not in zf.namelist():
        raise ValueError(f"bundle {source_label} missing manifest.json")
    # Bounded on ACTUAL bytes: this runs before any trust check on the network-facing paths (the
    # Oasis /contribute handler, hive pull, inspect). MemberReadError is a ValueError.
    try:
        manifest = json.loads(
            bounded_member_read(zf, "manifest.json", max_bytes=MAX_ENTRY_UNCOMPRESSED_BYTES).decode("utf-8")
        )
    except RecursionError as exc:
        raise ValueError(f"bundle {source_label}: manifest.json nests too deeply to parse") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest.json must be a JSON object, got {type(manifest).__name__}")
    try:
        manifest = migrate_bundle_envelope(manifest)
    except RecursionError as exc:  # deepcopy of a manifest nested a few hundred deep
        raise ValueError(
            f"bundle {source_label}: manifest.json nests too deeply to migrate (or the caller's stack is already deep)"
        ) from exc
    check_format_version(manifest, "substrate_bundle", log=logger)
    if manifest.get("kind") != BUNDLE_KIND:
        raise ValueError(f"manifest kind {manifest.get('kind')!r} != {BUNDLE_KIND!r}")
    schema_v = manifest.get("schema_version")
    if not isinstance(schema_v, int) or schema_v > BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"manifest schema_version {schema_v!r} unsupported (this build supports up to {BUNDLE_SCHEMA_VERSION})"
        )
    return manifest


def read_bundle_manifest(bundle_path: str | Path) -> dict[str, Any]:
    """Read the manifest from a bundle without extracting it.

    Convenience for CLI ``maxim substrate inspect`` and 1.1 Oasis
    discovery. Validates ``kind`` + ``schema_version`` like
    :func:`extract_bundle`.
    """
    bundle_path = Path(bundle_path)
    with zipfile.ZipFile(bundle_path, "r") as zf:
        return _manifest_from_zip(zf, str(bundle_path))


def read_bundle_manifest_bytes(raw: bytes) -> dict[str, Any]:
    """Read + validate a bundle manifest from raw ZIP bytes, touching no disk.

    The in-memory sibling of :func:`read_bundle_manifest`, for validating a
    received contribution BEFORE it is committed to disk (Oasis ``/contribute``).
    Raises the same ``ValueError`` / ``zipfile.BadZipFile`` on a malformed or
    wrong-kind bundle.
    """
    with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
        return _manifest_from_zip(zf, "<contribution bytes>")


__all__ = [
    "BUNDLE_KIND",
    "BUNDLE_SCHEMA_VERSION",
    "UNSIGNED_BUNDLE_SCHEMA_VERSION",
    "BundleBodyMismatch",
    "BundleBodyUnverifiable",
    "BundleVerification",
    "MAX_BUNDLE_ENTRIES",
    "MAX_ENTRY_UNCOMPRESSED_BYTES",
    "MAX_TOTAL_UNCOMPRESSED_BYTES",
    "MemberReadError",
    "assert_bundle_body_compatible",
    "bounded_member_read",
    "bundle_signature_scheme",
    "compose_bundle",
    "content_payload_digest",
    "extract_bundle",
    "isolated_bundle_migrations",
    "migrate_bundle_envelope",
    "read_bundle_manifest",
    "register_bundle_migration",
    "verify_bundle_zip",
    "scrub_nac_state_for_bundle",
]
