"""Oasis ingestion adapter — the V1–V10 receiver validation contract (1.2).

Implements the frozen receiver validation contract of
``docs/plans/sharing_threat_model.md`` §5 against the step order pinned in
``docs/plans/archive/oasis_ingestion_contract.md`` §3. Every duty ROUTES THROUGH the
existing seams — ``read_bundle_manifest``, ``assert_bundle_body_compatible``,
``filter_identity_bearing_links`` / ``is_identity_bearing``,
``scrub_nac_state_for_bundle``, and ``substrate_merge`` with the reserved
``trusted_sources`` / ``validate_node`` / ``validate_link`` /
``strict_geometry`` parameters — it does not re-implement them (the gate-7
rule generalized, stated in the threat model itself).

The one public pipeline entry is :func:`ingest_bundle`; the one operator
caller is ``maxim substrate ingest`` (``hivemind/cli.py``). A bundle is a ZIP
of attacker-controlled bytes: every manifest field AND every payload field is
an assertion, not a fact. Refusals raise :class:`IngestRefused` carrying the
duty tag; normalizations (caps, clamps, truncations) are recorded in the
returned :class:`IngestReport` notes so the operator sees what admission
changed.

The tighten-only clamp for negative valence does NOT live here — it fires
inside ``substrate_merge`` itself (the roadmap-decided seam), so every
consumer of the aligned merge gets it and no adapter bypass exists. This
module's poison-resistance contribution is the *entry rule* for the inherent
bias class: inherent-class markers are admitted only from Queen-provenance
contributors (``inherent_trusted_sources``), refused loudly from anyone else.
"""

from __future__ import annotations

from maxim.decisions.causal_link import bound_predicted_value
import copy
import hashlib
import io
import json
import logging
import math
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from maxim.hivemind.entry_index import (
    AGENT_TOKEN,
    NON_SITUATION_AGENT_FIELDS,
    EntryIndexError,
    agent_ids,
    is_agent_id,
    keep_agent_rows,
)
from maxim.hivemind.signing import SIGNATURE_MEMBER
from maxim.hivemind.bundle import (
    MAX_BUNDLE_ENTRIES,
    MAX_ENTRY_UNCOMPRESSED_BYTES,
    MAX_TOTAL_UNCOMPRESSED_BYTES,
    BundleVerification,
    MemberReadError,
    assert_bundle_body_compatible,
    bounded_member_read,
    content_payload_digest,
    read_bundle_manifest_bytes,
    scrub_nac_state_for_bundle,
    verify_bundle_zip,
)
from maxim.hivemind.identity import filter_identity_bearing_links, is_identity_bearing
from maxim.hivemind.merge import (
    NODE_ID_CHARSET,
    NAC_KEY_SEP,
    SubstrateMergeResult,
    _validate_source,
    substrate_merge,
)
from maxim.utils.atomic_io import atomic_write_json
from maxim.utils.format_version import check_format_version, with_format_version

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────
# Adapter constants (oasis_ingestion_contract.md §5). The frozen requirement
# (threat model V6/V2) is that these EXIST and refuse loudly; the values are
# adapter decisions.
# ─────────────────────────────────────────────────────────────────────────

# V6 — MAX_BUNDLE_ENTRIES / MAX_ENTRY_UNCOMPRESSED_BYTES / MAX_TOTAL_UNCOMPRESSED_BYTES live in
# bundle.py (imported above) so the one verifier enforces the same caps; they are read from this
# module's globals at call time, so ingest-level overrides reach verify_bundle_zip too.
#: V6 — post-parse structural cap on nodes / links / keys per slice dict.
MAX_NODES_PER_SLICE: int = 50_000
#: V2 (row B) — cap on any foreign evidence count (node ``count``, link
#: ``observation_count``, Welford ``n``): assertion alone must not buy
#: centroid dominance. The taught archive maxes at 518.
MAX_FOREIGN_COUNT: int = 1_000
#: V2 — cap on the (diagnostic) ``total_observations`` counter.
MAX_FOREIGN_TOTAL_OBSERVATIONS: int = 1_000_000
#: V2 (row M) — ``confidence`` max-folds in ``_merge_link_pair``; a foreign
#: 1.0 would be permanent. Foreign confidence is capped below it.
CAP_FOREIGN_CONFIDENCE: float = 0.9
#: Exp 61 (2026-09-16, prereg D1): a fear admitted from a FOREIGN bundle is multiplied by
#: this before the MIN fold — vicarious conditioning is real, drives avoidance without the
#: observer's own US, and is reliably weaker than direct conditioning; a re-export hands the
#: next hop the discount squared (chain attenuation with no provenance field). 0.75 keeps
#: a saturated donor (−1.0 → −0.75) above the strict activation floor (0.5) with margin;
#: `merge-nac` (trusted-local) stays verbatim. There is deliberately NO foreign cap on
#: fear magnitude beyond [-1, 0]: a cap at/below the floor would be a structural null.
FOREIGN_FEAR_DISCOUNT: float = 0.75
#: V2 (row K) — foreign list fields are truncated BELOW the merge's
#: ``[-100:]`` tail-truncation window so they cannot evict local history.
MAX_FOREIGN_DELTAS: int = 50
#: V2 (row B, executor-lens finding 1) — the count cap alone does NOT stop
#: centroid dominance: the cosine match gate is magnitude-invariant while
#: the count-weighted fold is magnitude-SENSITIVE, so a count=1 node with a
#: 1e12-norm embedding still folds and then owns the merged centroid
#: outright. Honest centroids in the shipped archive sit at norm 8–25;
#: 1000 is ~40x headroom while capping the dominance a single node can buy.
MAX_FOREIGN_EMBEDDING_NORM: float = 1000.0

#: Payload provenance values accepted as the composer's honest self-reference
#: (every real exporter stamps ``"local"``); normalized to the manifest
#: contributor at admission (receiver-stamping, V1/V5).
_SELF_SOURCE: str = "local"

#: The Valence enum's serialized vocabulary. Duplicated from
#: ``decisions.causal_link.Valence`` rather than imported to keep the
#: hivemind layer free of internal-module imports (the ``_cosine`` /
#: ``DEFAULT_FROZEN_CENTROID_MODALITIES`` convention); an invalid value
#: would otherwise traceback inside ``CausalLink.from_dict`` at the
#: receiver's NEXT BOOT — long after admission, on the receiver's own
#: load path.
_VALID_VALENCES: frozenset[str] = frozenset({"positive", "negative", "neutral", "unknown"})

#: Per-link fields that must be present as non-empty strings for the merge
#: and the eventual ``CausalLink.from_dict`` load to be well-defined.
_REQUIRED_LINK_STR_FIELDS: tuple[str, ...] = (
    "id",
    "event_type",
    "event_signature",
    "outcome_type",
    "outcome_signature",
)

_JOURNAL_FILE_TYPE: str = "substrate_ingest_journal"


class IngestRefused(ValueError):
    """A bundle failed a receiver validation duty and was refused whole.

    ``duty`` names the threat-model duty (``"V1"``…``"V10"``, ``"gate7"``,
    ``"inherent"``, ``"signature"``) so refusals are auditable against
    ``sharing_threat_model.md`` §5.
    """

    def __init__(self, *, duty: str, reason: str) -> None:
        self.duty = duty
        self.reason = reason
        super().__init__(f"[{duty}] {reason}")


# ─────────────────────────────────────────────────────────────────────────
# V8 — the ingestion journal (dedup + tombstones + durable attribution)
# ─────────────────────────────────────────────────────────────────────────


def _check_release_fields(entry: dict[str, Any], path: Path) -> None:
    """Fail loud on a journal entry whose release-state fields have the wrong type: the ordering rules
    compare them with ``==``, so ``True`` / ``1.0`` / ``"1"`` would silently match or miss (the same
    no-guessing rule the registry applies to its trust fields)."""
    for key in ("release_sequence", "signature_scheme"):
        value = entry.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError(f"ingestion journal {path}: {key} {value!r} is not an integer")
    for key in ("signer_key", "payload_digest"):
        value = entry.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"ingestion journal {path}: {key} {value!r} is not a string")


class IngestionJournal:
    """Per-receiver ingestion journal (threat model V5 + V8).

    Records each admitted bundle's digest + contributor + timestamp (+ the
    per-ingest counts that make a later distrust decision actionable — the
    NAc bias dicts are provenance-free, so the journal, not the state, is
    the durable attribution surface). Re-ingestion of a seen digest is
    refused; a tombstoned contributor's bundles are refused so replays
    cannot resurrect pruned state (row J). Also the slow-poison audit
    surface: repeated near-identical bundles from one contributor are
    visible here even when each merge is individually in-clamp.

    Persisted as a single JSON document via ``atomic_write_json`` +
    ``with_format_version`` (the CC1 contract).
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.entries: list[dict[str, Any]] = []
        self.tombstones: list[dict[str, Any]] = []
        if self.path.is_file():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError(f"ingestion journal {self.path} is not a JSON object")
            check_format_version(data, _JOURNAL_FILE_TYPE, log=logger)
            entries = data.get("entries", [])
            tombstones = data.get("tombstones", [])
            if not isinstance(entries, list) or not isinstance(tombstones, list):
                raise ValueError(f"ingestion journal {self.path} has malformed entries/tombstones")
            self.entries = [e for e in entries if isinstance(e, dict)]
            self.tombstones = [t for t in tombstones if isinstance(t, dict)]
            for e in self.entries:
                _check_release_fields(e, self.path)

    def has_digest(self, digest: str) -> bool:
        """Was a bundle with this identity admitted? Matches the ZIP sha256 (``digest``, every entry)
        OR the signed-payload digest (``payload_digest``, verified entries since release format v2) --
        so a re-zipped release dedups, and a release admitted before the change is not admitted twice."""
        return any(e.get("digest") == digest or e.get("payload_digest") == digest for e in self.entries)

    def equivocation(self, signer_key: str, release_sequence: int, payload_digest: str) -> dict[str, Any] | None:
        """The admitted entry a release EQUIVOCATES against: same signing key and sequence, a different
        signed payload. Keyed by public key (hex), not identity, so a rotated key starts clean and two
        registry names for one key cannot split its history."""
        for e in self.entries:
            if (
                e.get("signer_key") == signer_key
                and e.get("release_sequence") == release_sequence
                and e.get("payload_digest") != payload_digest
            ):
                return e
        return None

    def has_v2_from(self, signer_key: str) -> bool:
        """Has a v2 release from this signing key been admitted? (Then a v1 bundle from it is a downgrade.)"""
        return any(e.get("signer_key") == signer_key and e.get("signature_scheme") == 2 for e in self.entries)

    def is_tombstoned(self, contributor_id: str) -> bool:
        return any(t.get("contributor_id") == contributor_id for t in self.tombstones)

    def add_tombstone(self, contributor_id: str, *, reason: str = "") -> None:
        """Distrust a contributor: every later bundle from them is refused.

        Tombstoning is one-way at this surface (removing one is a hand edit
        of the journal file — deliberate operator friction). The companion
        pruning flow (``prune_nac_cluster_biases`` driven by this journal's
        attribution records) is follow-up 1.2 work named in
        ``oasis_ingestion_contract.md`` §4.
        """
        _validate_source(contributor_id, label="contributor_id")
        if not self.is_tombstoned(contributor_id):
            self.tombstones.append(
                {
                    "contributor_id": contributor_id,
                    "created_at": time.time(),
                    "reason": reason,
                }
            )

    def record(self, entry: dict[str, Any]) -> None:
        self.entries.append(dict(entry))

    def save(self) -> None:
        payload: dict[str, Any] = {
            "entries": list(self.entries),
            "tombstones": list(self.tombstones),
        }
        atomic_write_json(str(self.path), with_format_version(payload))


# ─────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class IngestReport:
    """What one ingestion validated, normalized, and merged.

    Runtime-ephemeral (a function return, never persisted and never on a
    wire) — CC3's forward-compat clause does not apply; the durable record
    is ``journal_entry``, written through :class:`IngestionJournal`.
    """

    manifest: dict[str, Any]
    digest: str
    contributor_id: str
    nac: dict[str, Any]
    ec_nodes: dict[str, dict[str, Any]]
    id_map: dict[str, str]
    biases_rekeyed: int
    biases_dropped: int
    biases_tightened: int
    inherent_keys_admitted: int
    links_dropped_identity: int
    welford_dropped_identity: int
    #: Exp 61 fear transport — the honest indicator for the fear field, same shape as
    #: the bias counts: keys re-keyed onto surviving clusters, keys dropped because the
    #: donor cluster did not survive the aligned merge, and keys that landed BELOW the
    #: read floor (|v| ≤ θ after the discount) and so will never act.
    fear_rekeyed: int = 0
    fear_dropped: int = 0
    fear_below_floor: int = 0
    valence_entries: dict[str, float] = field(default_factory=dict)
    #: Non-situation NAc rows (percept valences, outcome stats, node-keyed reward bias) filed under an
    #: agent this receiver is not, dropped at ingest: they could never be read here.
    foreign_rows_dropped: int = 0
    #: The signature verdict when ``require_signed`` (``None`` otherwise): the payload digest,
    #: signer key, release sequence and entry digests the receiver-state layer keys on.
    verification: BundleVerification | None = None
    undeclared_members: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    journal_entry: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────
# Parse helpers — V2's NaN/Infinity refusal happens AT parse time
# ─────────────────────────────────────────────────────────────────────────


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise IngestRefused(duty="V2", reason=f"non-finite numeric literal {text!r} in payload")
    return value


def _refuse_constant(name: str) -> float:
    # json.loads accepts NaN / Infinity / -Infinity by default; a NaN maps
    # to +hi in _merge_mean_clamped (Python min/max NaN semantics, row M).
    raise IngestRefused(duty="V2", reason=f"JSON constant {name!r} refused (NaN/Infinity poisoning, row M)")


def _loads_strict(raw: bytes, *, slice_name: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8"), parse_float=_finite_float, parse_constant=_refuse_constant)
    except IngestRefused:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IngestRefused(duty="V2", reason=f"malformed JSON in {slice_name}: {exc}") from exc
    except RecursionError as exc:
        raise IngestRefused(duty="V2", reason=f"{slice_name} nests too deeply to parse") from exc


def _require_finite(value: Any, *, where: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        # OverflowError: a hostile INTEGER literal (10**400) never meets
        # parse_float, and float() on it overflows — that is a refusal,
        # not a traceback (arch-lens finding 1).
        raise IngestRefused(duty="V2", reason=f"non-numeric or overflowing value at {where}: {value!r}") from exc
    if not math.isfinite(out):
        raise IngestRefused(duty="V2", reason=f"non-finite value at {where}")
    return out


def _require_in_range(value: Any, lo: float, hi: float, *, where: str) -> float:
    out = _require_finite(value, where=where)
    if out < lo or out > hi:
        raise IngestRefused(duty="V2", reason=f"value {out} at {where} outside [{lo}, {hi}]")
    return out


# ─────────────────────────────────────────────────────────────────────────
# V6 — resource caps from the central directory, before decompression
# ─────────────────────────────────────────────────────────────────────────


def _bounded_zip_read(zf: zipfile.ZipFile, name: str) -> bytes:
    """Decompress one member with the size cap enforced on ACTUAL bytes.

    The central-directory ``file_size`` is itself an attacker assertion — a
    binary-patched header can declare 10 bytes over an 800 MB stream, pass
    :func:`_check_resource_caps`, and then ``zf.read`` inflates the whole
    thing in memory before the CRC check fires (executor-lens finding 3,
    measured at +1.3 GB RSS). The capped reader is
    :func:`bundle.bounded_member_read`, shared with the signature verifier.
    """
    try:
        return bounded_member_read(zf, name, max_bytes=MAX_ENTRY_UNCOMPRESSED_BYTES)
    except MemberReadError as exc:
        raise IngestRefused(duty="V6", reason=str(exc)) from exc


def _check_resource_caps(zf: zipfile.ZipFile) -> None:
    infos = zf.infolist()
    if len(infos) > MAX_BUNDLE_ENTRIES:
        raise IngestRefused(
            duty="V6",
            reason=f"bundle has {len(infos)} entries (cap {MAX_BUNDLE_ENTRIES})",
        )
    total = 0
    for info in infos:
        if info.file_size > MAX_ENTRY_UNCOMPRESSED_BYTES:
            raise IngestRefused(
                duty="V6",
                reason=(
                    f"entry {info.filename!r} declares {info.file_size} uncompressed bytes "
                    f"(cap {MAX_ENTRY_UNCOMPRESSED_BYTES})"
                ),
            )
        total += info.file_size
    if total > MAX_TOTAL_UNCOMPRESSED_BYTES:
        raise IngestRefused(
            duty="V6",
            reason=f"bundle declares {total} total uncompressed bytes (cap {MAX_TOTAL_UNCOMPRESSED_BYTES})",
        )


# ─────────────────────────────────────────────────────────────────────────
# V1 — payload provenance sweep + receiver stamping
# ─────────────────────────────────────────────────────────────────────────


def _sweep_provenance_value(value: Any, *, contributor_id: str, where: str) -> None:
    """Refuse any in-payload provenance id that is not the author's own.

    Row I: contributor/source fields in the payload are attacker bytes.
    Accepted: the manifest's ``contributor_id`` and the honest
    self-reference ``"local"``. Refused: reserved ``_*`` sentinels
    (``"_consensus"`` never reaches ``_validate_source`` on the payload
    path), any other id (trusted-id stuffing / attribution laundering),
    and any multi-party set (a single-author bundle asserting consensus).
    """
    if value is None:
        return
    if not isinstance(value, str):
        raise IngestRefused(duty="V1", reason=f"non-string provenance value at {where}: {value!r}")
    if value.startswith("_"):
        raise IngestRefused(duty="V1", reason=f"reserved sentinel {value!r} in payload provenance at {where}")
    if value not in (contributor_id, _SELF_SOURCE):
        raise IngestRefused(
            duty="V1",
            reason=(
                f"payload provenance at {where} names {value!r}, not the manifest contributor "
                f"{contributor_id!r} (trusted-id stuffing / attribution laundering, row I)"
            ),
        )


def _sweep_and_stamp_provenance(entry: dict[str, Any], *, contributor_id: str, where: str) -> None:
    """Sweep one link/node dict's provenance, then receiver-stamp it in place.

    The admitted tag is stamped BY THE RECEIVER from its own trust decision
    (the V1 door), never copied from the payload.
    """
    _sweep_provenance_value(entry.get("source"), contributor_id=contributor_id, where=f"{where}.source")
    contributors = entry.get("contributors", []) or []
    if not isinstance(contributors, (list, tuple)):
        raise IngestRefused(duty="V1", reason=f"malformed contributors at {where}: {contributors!r}")
    for c in contributors:
        _sweep_provenance_value(c, contributor_id=contributor_id, where=f"{where}.contributors")
    entry["source"] = contributor_id
    entry["contributors"] = [contributor_id]


# ─────────────────────────────────────────────────────────────────────────
# Slice validation / normalization
# ─────────────────────────────────────────────────────────────────────────


#: V9 identifier charset for EC node ids and cluster-id key segments. An
#: honest composer emits uuid4 ids, which this covers with headroom; it
#: refuses the NAC_KEY_SEP byte and every other control/whitespace
#: character by construction, and '#' (the merge's collision-suffix
#: marker — a pre-crafted '#' id masquerades as a prior collision).
_NODE_ID_CHARSET = NODE_ID_CHARSET  # owned by merge.py so the export scrub filters by the same rule


def _check_key_shape(key: str, parts_expected: int, *, where: str) -> list[str]:
    parts = str(key).split(NAC_KEY_SEP)
    if len(parts) != parts_expected:
        raise IngestRefused(
            duty="V9",
            reason=f"malformed composite key at {where} ({len(parts)} segments, expected {parts_expected})",
        )
    if any(not p for p in parts):
        raise IngestRefused(duty="V9", reason=f"empty key segment at {where}")
    return parts


def _validate_nac_payload(
    nac_state: dict[str, Any],
    *,
    contributor_id: str,
    now: float,
    notes: list[str],
) -> dict[str, Any]:
    """The V2/V9/V1 admission pass over a foreign ``nac.json`` slice.

    Returns a normalized deep copy; refusals raise. Bounds every asserted
    number, caps the monotone-fold fields, truncates list fields below the
    merge's tail-truncation windows, sweeps + receiver-stamps provenance,
    and validates key hygiene. The V4 scrub/quarantine re-run happens
    AFTER this pass (:func:`_receiver_scrub`).
    """
    state = copy.deepcopy(nac_state)
    state.pop("_format_version", None)
    # The donor's decay clock is not the receiver's: nac_merge keeps the
    # LATER saved_at, so a far-future stamp would freeze receiver decay
    # (row M). Dropping it keeps the receiver's own clock.
    if state.pop("saved_at", None) is not None:
        notes.append("donor saved_at dropped (receiver keeps its own decay clock)")

    links = state.get("links", {}) or {}
    if not isinstance(links, dict):
        raise IngestRefused(duty="V2", reason="nac.json links is not an object")
    if len(links) > MAX_NODES_PER_SLICE:
        raise IngestRefused(
            duty="V6", reason=f"nac.json declares {len(links)} event signatures (cap {MAX_NODES_PER_SLICE})"
        )

    counts_capped = 0
    confidence_capped = 0
    predicted_values_bounded = 0
    deltas_truncated = 0
    timestamps_clamped = 0

    for evt_sig, link_list in links.items():
        if not isinstance(link_list, list):
            raise IngestRefused(duty="V2", reason=f"links[{evt_sig!r}] is not a list")
        if len(link_list) > MAX_NODES_PER_SLICE:
            raise IngestRefused(
                duty="V6", reason=f"links[{evt_sig!r}] has {len(link_list)} entries (cap {MAX_NODES_PER_SLICE})"
            )
        for i, link in enumerate(link_list):
            if not isinstance(link, dict):
                raise IngestRefused(duty="V2", reason=f"links[{evt_sig!r}][{i}] is not an object")
            where = f"links[{evt_sig!r}][{i}]"
            # Structural shape first: the scrub and the receiver's eventual
            # CausalLink.from_dict both index these fields; a hostile link
            # missing one must refuse HERE, not traceback there.
            for req in _REQUIRED_LINK_STR_FIELDS:
                if not isinstance(link.get(req), str) or not link[req]:
                    raise IngestRefused(duty="V2", reason=f"{where}.{req} missing or not a non-empty string")
            if link.get("outcome_valence") not in _VALID_VALENCES:
                raise IngestRefused(
                    duty="V2",
                    reason=f"{where}.outcome_valence {link.get('outcome_valence')!r} not in {sorted(_VALID_VALENCES)}",
                )
            if not isinstance(link.get("event_context", {}), dict):
                raise IngestRefused(duty="V2", reason=f"{where}.event_context is not an object")
            link.setdefault("event_context", {})
            if not isinstance(link.get("context_factors", {}) or {}, dict):
                raise IngestRefused(duty="V2", reason=f"{where}.context_factors is not an object")
            _sweep_and_stamp_provenance(link, contributor_id=contributor_id, where=where)
            # Refuse garbage, then bound a stray-but-plausible value into the Rescorla-Wagner range
            # [0, 1] -- anything below 0 would make a surprise above 1 on the next update.
            raw_pv = _require_in_range(link.get("predicted_value", 0.5), -1.0, 1.0, where=f"{where}.predicted_value")
            link["predicted_value"] = bound_predicted_value(raw_pv, where=f"{where}.predicted_value")
            if link["predicted_value"] != raw_pv:
                predicted_values_bounded += 1
            conf = _require_in_range(link.get("confidence", 0.5), 0.0, 1.0, where=f"{where}.confidence")
            if conf > CAP_FOREIGN_CONFIDENCE:
                conf = CAP_FOREIGN_CONFIDENCE
                confidence_capped += 1
            link["confidence"] = conf
            n = int(_require_finite(link.get("observation_count", 0), where=f"{where}.observation_count"))
            if n < 0:
                raise IngestRefused(duty="V2", reason=f"negative observation_count at {where}")
            if n > MAX_FOREIGN_COUNT:
                n = MAX_FOREIGN_COUNT
                counts_capped += 1
            link["observation_count"] = n
            last_observed = _require_finite(link.get("last_observed", 0.0), where=f"{where}.last_observed")
            if last_observed > now:
                last_observed = now
                timestamps_clamped += 1
            link["last_observed"] = last_observed
            td = link.get("temporal_delta") or {}
            if not isinstance(td, dict):
                raise IngestRefused(duty="V2", reason=f"{where}.temporal_delta is not an object")
            raw_deltas = td.get("observed_deltas", [])
            if not isinstance(raw_deltas, list):
                raise IngestRefused(duty="V2", reason=f"{where}.observed_deltas is not a list")
            deltas = list(raw_deltas)
            for j, d in enumerate(deltas):
                _require_finite(d, where=f"{where}.observed_deltas[{j}]")
            if len(deltas) > MAX_FOREIGN_DELTAS:
                deltas = deltas[-MAX_FOREIGN_DELTAS:]
                deltas_truncated += 1
            link["temporal_delta"] = {"observed_deltas": deltas}
            raw_history = link.get("prediction_history", []) or []
            if not isinstance(raw_history, list):
                raise IngestRefused(duty="V2", reason=f"{where}.prediction_history is not a list")
            history = list(raw_history)
            for j, h in enumerate(history):
                _require_finite(h, where=f"{where}.prediction_history[{j}]")
            link["prediction_history"] = history[-MAX_FOREIGN_DELTAS:]
            # Episodes never ship; neither do their ids (the compose scrub
            # empties both — the receiver re-enforces before re-scrubbing).
            link["memory_ids"] = []
            link["percept_refs"] = []
            if link.get("last_rpe") is not None:
                link["last_rpe"] = _require_finite(link.get("last_rpe"), where=f"{where}.last_rpe")

    total_obs = int(_require_finite(state.get("total_observations", 0), where="total_observations"))
    if total_obs < 0:
        raise IngestRefused(duty="V2", reason="negative total_observations")
    if total_obs > MAX_FOREIGN_TOTAL_OBSERVATIONS:
        state["total_observations"] = MAX_FOREIGN_TOTAL_OBSERVATIONS
        notes.append(f"total_observations capped at {MAX_FOREIGN_TOTAL_OBSERVATIONS}")
    else:
        state["total_observations"] = total_obs

    from maxim.decisions.nac import DEFAULT_CLUSTER_FEAR_FAILURE_MODES  # noqa: PLC0415 — one allowlist, no cycle

    fear_discounted = 0
    for field_name, lo, hi, parts_expected in (
        ("reward_bias", 0.0, 1.0, 0),
        ("goal_reward_bias", -1.0, 1.0, 0),
        ("cluster_reward_bias", -1.0, 1.0, 3),
        ("percept_valences", -1.0, 1.0, 3),
        # Wire-4 fear travels since Exp 61 (2026-09-16): bounded [-1, 0] (fear only),
        # triple-keyed, cluster id charset-checked like a bias, failure mode REFUSED
        # outside the Wire-4 allowlist (a privilege-shaped field: refusal, not strip —
        # the `inherent` posture), and DISCOUNTED by `FOREIGN_FEAR_DISCOUNT` before the
        # MIN fold (vicarious fear is weaker than direct conditioning; prereg D1).
        ("cluster_fear", -1.0, 0.0, 3),
    ):
        entries = state.get(field_name, {}) or {}
        if not isinstance(entries, dict):
            raise IngestRefused(duty="V2", reason=f"nac.json {field_name} is not an object")
        if len(entries) > MAX_NODES_PER_SLICE:
            raise IngestRefused(duty="V6", reason=f"{field_name} has {len(entries)} keys (cap {MAX_NODES_PER_SLICE})")
        validated: dict[str, float] = {}
        for key, value in entries.items():
            if parts_expected:
                parts = _check_key_shape(str(key), parts_expected, where=f"{field_name}[{key!r}]")
                if field_name in ("cluster_reward_bias", "cluster_fear") and not _NODE_ID_CHARSET.match(parts[1]):
                    raise IngestRefused(
                        duty="V9",
                        reason=(
                            f"cluster id {parts[1]!r} fails the identifier charset "
                            "(no '#' collision masquerade, no control/whitespace)"
                        ),
                    )
                if field_name == "cluster_fear" and parts[2] not in DEFAULT_CLUSTER_FEAR_FAILURE_MODES:
                    raise IngestRefused(
                        duty="V2",
                        reason=(
                            f"cluster_fear failure mode {parts[2]!r} is not in the Wire-4 allowlist "
                            f"{sorted(DEFAULT_CLUSTER_FEAR_FAILURE_MODES)} — a fear no local pain could have written"
                        ),
                    )
            bounded = _require_in_range(value, lo, hi, where=f"{field_name}[{key!r}]")
            if field_name == "cluster_fear":
                if bounded == 0.0:
                    continue  # a zero fear is not a fear — never re-keyed, never counted
                bounded = bounded * FOREIGN_FEAR_DISCOUNT
                fear_discounted += 1
            validated[str(key)] = bounded
        state[field_name] = validated
    if fear_discounted:
        notes.append(
            f"cluster_fear: {fear_discounted} entr{'y' if fear_discounted == 1 else 'ies'} discounted ×{FOREIGN_FEAR_DISCOUNT} (foreign fear is vicarious, prereg Exp 61 D1)"
        )

    welford = state.get("event_outcome_welford", {}) or {}
    if not isinstance(welford, dict):
        raise IngestRefused(duty="V2", reason="event_outcome_welford is not an object")
    if len(welford) > MAX_NODES_PER_SLICE:
        raise IngestRefused(
            duty="V6", reason=f"event_outcome_welford has {len(welford)} keys (cap {MAX_NODES_PER_SLICE})"
        )
    validated_welford: dict[str, dict[str, float]] = {}
    for key, wstate in welford.items():
        _check_key_shape(str(key), 2, where=f"event_outcome_welford[{key!r}]")
        if not isinstance(wstate, dict):
            raise IngestRefused(duty="V2", reason=f"event_outcome_welford[{key!r}] is not an object")
        mean = _require_finite(wstate.get("mean", 0.0), where=f"event_outcome_welford[{key!r}].mean")
        # Outcome rewards are valence-mapped into [0, 1] (_VALENCE_TO_REWARD),
        # so an honest mean cannot leave [-1, 1]; an asserted 1e100 would
        # permanently saturate the receiver's uncertainty interval for the
        # signature (executor-lens finding 4).
        if mean < -1.0 or mean > 1.0:
            raise IngestRefused(
                duty="V2", reason=f"Welford mean {mean} at event_outcome_welford[{key!r}] outside [-1, 1]"
            )
        m2 = _require_finite(wstate.get("m2", 0.0), where=f"event_outcome_welford[{key!r}].m2")
        if m2 < 0.0:
            raise IngestRefused(duty="V2", reason=f"negative Welford m2 at event_outcome_welford[{key!r}]")
        n_w = _require_finite(wstate.get("n", 0.0), where=f"event_outcome_welford[{key!r}].n")
        if n_w < 0.0:
            raise IngestRefused(duty="V2", reason=f"negative Welford n at event_outcome_welford[{key!r}]")
        if n_w > MAX_FOREIGN_COUNT:
            # Capping n without scaling m2 would inflate the implied
            # variance (m2/n); scale proportionally so the asserted
            # variance is preserved under the cap.
            m2 = m2 * (float(MAX_FOREIGN_COUNT) / n_w)
            n_w = float(MAX_FOREIGN_COUNT)
            counts_capped += 1
        # Honest m2 for [0,1]-range rewards is bounded by n·(range/2)²·4 = n;
        # 4·n gives slack while refusing the 1e300 magnitude class (which
        # also overflows _merge_welford's (Δmean)² arithmetic into a
        # traceback — the same refusal-shape family as the huge-int case).
        if m2 > 4.0 * max(n_w, 1.0):
            raise IngestRefused(
                duty="V2",
                reason=f"Welford m2 {m2:.3g} at event_outcome_welford[{key!r}] exceeds 4·n (impossible variance)",
            )
        validated_welford[str(key)] = {"mean": mean, "m2": m2, "n": n_w}
    state["event_outcome_welford"] = validated_welford

    inherent = state.get("inherent_bias_keys", []) or []
    if not isinstance(inherent, list):
        raise IngestRefused(duty="V2", reason="inherent_bias_keys is not a list")
    for key in inherent:
        _check_key_shape(str(key), 3, where=f"inherent_bias_keys[{key!r}]")

    if counts_capped:
        notes.append(f"{counts_capped} foreign counts capped at {MAX_FOREIGN_COUNT} (row B)")
    if confidence_capped:
        notes.append(f"{confidence_capped} confidence values capped at {CAP_FOREIGN_CONFIDENCE} (row M)")
    if predicted_values_bounded:
        notes.append(f"{predicted_values_bounded} predicted_value values bounded to [0, 1]")
    if deltas_truncated:
        notes.append(f"{deltas_truncated} observed_deltas lists truncated to {MAX_FOREIGN_DELTAS} (row K)")
    if timestamps_clamped:
        notes.append(f"{timestamps_clamped} future last_observed timestamps clamped to now (row M)")
    return state


def _validate_ec_payload(
    ec_nodes: dict[str, Any],
    *,
    contributor_id: str,
    manifest: dict[str, Any],
    allow_unstamped_geometry: bool,
    notes: list[str],
) -> dict[str, dict[str, Any]]:
    """The V2/V3/V9 admission pass over a foreign ``ec.json`` nodes slice.

    Returns a normalized deep copy with per-node ``domain`` STRIPPED (row
    N's stamping half: ``ec_merge_aligned``'s ``domain or`` fold must never
    stamp a LOCAL survivor with foreign bytes — inserted foreign nodes are
    re-stamped from ``manifest.domain`` by the caller, after the merge).
    """
    if len(ec_nodes) > MAX_NODES_PER_SLICE:
        raise IngestRefused(duty="V6", reason=f"ec.json declares {len(ec_nodes)} nodes (cap {MAX_NODES_PER_SLICE})")

    counts_capped = 0
    domains_stripped = 0
    out: dict[str, dict[str, Any]] = {}
    dims_by_modality: dict[str, set[int]] = {}
    for nid, node in ec_nodes.items():
        nid_s = str(nid)
        if not _NODE_ID_CHARSET.match(nid_s):
            raise IngestRefused(
                duty="V9",
                reason=(
                    f"node id {nid_s!r} fails the identifier charset (V9: no separator bytes, no "
                    "'#' collision masquerade, no control/whitespace characters; honest ids are uuid4)"
                ),
            )
        if not isinstance(node, dict):
            raise IngestRefused(duty="V2", reason=f"ec.json node {nid_s!r} is not an object")
        node = copy.deepcopy(node)
        where = f"ec.json[{nid_s!r}]"
        embedding = node.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise IngestRefused(duty="V2", reason=f"{where} has no embedding (a node without a centroid cannot merge)")
        sq_sum = 0.0
        for j, v in enumerate(embedding):
            fv = _require_finite(v, where=f"{where}.embedding[{j}]")
            sq_sum += fv * fv
        norm = math.sqrt(sq_sum)
        if norm <= 0.0 or norm > MAX_FOREIGN_EMBEDDING_NORM:
            raise IngestRefused(
                duty="V2",
                reason=(
                    f"{where} embedding L2 norm {norm:.3g} outside (0, {MAX_FOREIGN_EMBEDDING_NORM}] — "
                    "the cosine match gate is magnitude-invariant but the count-weighted fold is not; "
                    "an inflated norm buys centroid dominance the count cap exists to refuse (row B)"
                ),
            )
        modality = str(node.get("modality", ""))
        if not modality:
            raise IngestRefused(duty="V2", reason=f"{where} declares no modality")
        dims_by_modality.setdefault(modality, set()).add(len(embedding))

        count = int(_require_finite(node.get("count", node.get("member_count", 1)), where=f"{where}.count"))
        if count < 0:
            raise IngestRefused(duty="V2", reason=f"negative count at {where}")
        if count > MAX_FOREIGN_COUNT:
            count = MAX_FOREIGN_COUNT
            counts_capped += 1
        node["count"] = count
        node["member_count"] = count

        geometry = node.get("geometry")
        if geometry is None and not allow_unstamped_geometry:
            raise IngestRefused(
                duty="V3",
                reason=(
                    f"{where} carries no geometry stamp; unstamped foreign nodes are refused at "
                    "admission (strict_geometry alone only blocks folds — unstamped nodes still "
                    "insert). Pass allow_unstamped_geometry only for an operator-attested legacy "
                    "archive."
                ),
            )

        domain = node.get("domain")
        if domain is not None:
            if not isinstance(domain, str) or domain.startswith("_"):
                raise IngestRefused(
                    duty="V9",
                    reason=f"{where} declares reserved domain {domain!r} (row N: self-hiding poison)",
                )
            node["domain"] = None
            domains_stripped += 1

        _sweep_and_stamp_provenance(node, contributor_id=contributor_id, where=where)
        out[nid_s] = node

    # V3 measured class: re-measure observed_embedding_dims from the actual
    # arrays and compare with the manifest's stamp. The declared side is
    # attacker bytes too — a malformed shape (an int where a list belongs)
    # is a mismatch, not a traceback (executor-lens finding 2).
    declared = (manifest.get("encoder_provenance") or {}).get("observed_embedding_dims")
    measured = {m: sorted(d) for m, d in sorted(dims_by_modality.items())}
    declared_norm: dict[str, list[int]] | None = None
    if isinstance(declared, dict) and all(isinstance(v, list) for v in declared.values()):
        try:
            declared_norm = {str(m): sorted(int(x) for x in v) for m, v in sorted(declared.items())}
        except (TypeError, ValueError):
            declared_norm = None
    if declared_norm != measured:
        raise IngestRefused(
            duty="V3",
            reason=(
                f"manifest observed_embedding_dims {declared!r} does not match dims measured from "
                f"the actual arrays {measured!r}"
            ),
        )
    if counts_capped:
        notes.append(f"{counts_capped} foreign node counts capped at {MAX_FOREIGN_COUNT} (row B)")
    if domains_stripped:
        notes.append(
            f"{domains_stripped} foreign per-node domains stripped pre-merge (row N: no stamping of "
            "local survivors); inserted nodes are re-stamped from manifest.domain"
        )
    return out


def _receiver_scrub(
    nac_state: dict[str, Any],
    *,
    notes: list[str],
) -> tuple[dict[str, Any], int, int]:
    """V4 — re-run BOTH quarantine and content scrub on receipt.

    Routes through the same functions compose uses
    (``filter_identity_bearing_links`` at the bundle-stricter threshold 2,
    the Welford-key identity filter, ``scrub_nac_state_for_bundle`` — pure,
    runs identically on a received state) — never trusting the sender's
    filter. This is also the payload SHAPE validation the loaders do not
    provide and the prompt-injection gate (row L: free text in merged keys
    reaches prompt annotation).
    """
    links = nac_state.get("links", {}) or {}
    filtered_links = filter_identity_bearing_links(links, threshold=2)
    links_dropped = len(links) - len(filtered_links)
    state = dict(nac_state)
    state["links"] = filtered_links

    welford = state.get("event_outcome_welford", {}) or {}
    filtered_welford = {
        key: wstate
        for key, wstate in welford.items()
        if not is_identity_bearing(key.partition(NAC_KEY_SEP)[2], threshold=2)
    }
    welford_dropped = len(welford) - len(filtered_welford)
    state["event_outcome_welford"] = filtered_welford

    scrubbed = scrub_nac_state_for_bundle(state)
    if links_dropped:
        notes.append(f"{links_dropped} identity-bearing event signatures dropped at receipt (V4)")
    if welford_dropped:
        notes.append(f"{welford_dropped} identity-bearing Welford keys dropped at receipt (V4)")
    return scrubbed, links_dropped, welford_dropped


# ─────────────────────────────────────────────────────────────────────────
# The pipeline
# ─────────────────────────────────────────────────────────────────────────


def ingest_bundle(
    bundle_path: str | Path,
    *,
    receiver_nac: dict[str, Any] | None,
    receiver_ec_nodes: dict[str, dict[str, Any]] | None,
    receiver_body: str,
    trusted_sources: frozenset[str] | set[str],
    journal: IngestionJournal,
    inherent_trusted_sources: frozenset[str] | set[str] = frozenset(),
    receiver_agent_id: str | None = None,
    allow_unverified_body: bool = False,
    allow_unstamped_geometry: bool = False,
    force_digest: bool = False,
    require_signed: bool = False,
    trusted_keys: dict[str, str] | None = None,
    accept_v1: bool = True,
) -> IngestReport:
    """Validate + merge one foreign bundle into receiver state dicts.

    The V1–V10 pipeline in the step order of
    ``oasis_ingestion_contract.md`` §3. Pure with respect to the receiver:
    inputs are not mutated, nothing is written to disk — the caller applies
    ``report.nac`` / ``report.ec_nodes`` and records
    ``report.journal_entry`` through the journal (the CLI's ``--apply``).

    Raises :class:`IngestRefused` (or gate 7's ``BundleBodyMismatch`` /
    ``BundleBodyUnverifiable``) on any duty failure; nothing about the
    receiver changes on refusal.

    ``accept_v1`` (with ``require_signed``): whether a legacy v1 signature is still accepted. It
    defaults to True for direct callers (legacy v1-signed bundles such as the Exp 56/61-era releases
    keep verifying; the harnesses themselves ingest unsigned bundles); ``hive pull``
    passes the registry entry's decision (``--refuse-v1`` when ``accept_v1: false``, the default for
    a newly added Oasis).

    With ``require_signed``, two release-ordering rules are read from ``journal`` (what was ADMITTED,
    keyed by the verified public key): a second payload for an admitted ``(key, sequence)`` is refused
    (``equivocation``), and a v1 bundle from a key whose v2 release was admitted is refused
    (``downgrade``). Dedup (V8) matches the ZIP sha256 or the signed-payload digest -- for an
    unverified bundle, the digest its content implies. Limits: the rules read only VERIFIED entries, so
    a release first admitted unverified (``allow_unsigned``, or no ``require_signed``) seeds neither
    rule; the journal is per session and load-modify-save, so two concurrent ``--apply`` runs on one
    session can each admit half of an equivocating pair.

    The file is read ONCE: the journal digest, the manifest and every verified member come from the
    same in-memory snapshot, so a file swapped mid-ingest cannot pair one bundle's trust decisions
    with another's bytes.
    """
    if receiver_agent_id is not None and not is_agent_id(receiver_agent_id):
        raise ValueError(f"receiver_agent_id {receiver_agent_id!r} is not an agent id (no ':' or \\x1f)")
    bundle_path = Path(bundle_path)
    notes: list[str] = []
    now = time.time()

    raw = bundle_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()

    # 1. V6 — caps from the central directory, before decompression.
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw), "r")
    except zipfile.BadZipFile as exc:
        raise IngestRefused(duty="V6", reason=f"not a readable ZIP archive: {exc}") from exc
    with zf:
        _check_resource_caps(zf)
        # Verify the manifest's TRUE decompressed size before the existing
        # seam re-reads it — the declared sizes checked above are attacker
        # bytes (see _bounded_zip_read).
        if "manifest.json" in zf.namelist():
            _bounded_zip_read(zf, "manifest.json")

        # Signature duty — right after V6 and BEFORE any trust decision (spec order: V6 → signature →
        # index → V1 → gate 7 → V8), so a forged or untrusted bundle is refused as unsigned rather
        # than on whatever later duty it trips. Verified against the manifest AS STORED
        # (verify_bundle_zip reads it raw): the envelope migration rewrites schema_version, which a
        # v1 signature covers. Unsigned/opt-in ingests (require_signed=False) skip this and rely on V1
        # trust alone (the experimental tier).
        verification: BundleVerification | None = None
        if require_signed:
            verification = verify_bundle_zip(
                zf,
                trusted_keys=dict(trusted_keys or {}),
                accept_v1=accept_v1,
                max_members=MAX_BUNDLE_ENTRIES,
                max_member_bytes=MAX_ENTRY_UNCOMPRESSED_BYTES,
                max_total_bytes=MAX_TOTAL_UNCOMPRESSED_BYTES,
            )
            if not verification.ok:
                raise IngestRefused(duty="signature", reason=f"require_signed: {verification.reason}")
            notes.append(f"signature verified ({verification.reason})")

        # 2. Manifest through the existing seam (migration, schema_version,
        # _format_version, kind).
        try:
            manifest = read_bundle_manifest_bytes(raw)
        except zipfile.BadZipFile as exc:
            raise IngestRefused(duty="V6", reason=f"manifest.json is unreadable: {exc}") from exc

        # 3. V1 — the front door: bundle-level trust on the manifest.
        contributor_id = manifest.get("contributor_id")
        if not isinstance(contributor_id, str) or not contributor_id:
            raise IngestRefused(duty="V1", reason=f"manifest contributor_id missing/malformed: {contributor_id!r}")
        _validate_source(contributor_id, label="contributor_id")
        if contributor_id not in trusted_sources:
            raise IngestRefused(
                duty="V1",
                reason=(
                    f"contributor {contributor_id!r} is not operator-attested (trusted: "
                    f"{sorted(trusted_sources)}). Refusal, never admit-with-clamps."
                ),
            )
        domain = manifest.get("domain")
        if domain is not None and (not isinstance(domain, str) or domain.startswith("_")):
            raise IngestRefused(duty="V9", reason=f"manifest domain {domain!r} is reserved/malformed")

        # 4. Gate 7 — body compatibility through the existing function.
        assert_bundle_body_compatible(manifest, receiver_body=receiver_body, allow_unverified=allow_unverified_body)

        # 5. V8 — journal gate: digest dedup + contributor tombstones.
        if journal.is_tombstoned(contributor_id):
            raise IngestRefused(duty="V8", reason=f"contributor {contributor_id!r} is tombstoned (distrusted)")
        # Release ordering rules (release format v2, decision (b)) — derived from the journal of what was
        # ADMITTED, keyed by the verified signing key. Neither is waivable by force_digest: a replay is an
        # operator choice, a second payload under one (key, sequence) is not.
        if verification is not None and verification.signer_key:
            if verification.scheme == 2 and verification.release_sequence is not None:
                clash = journal.equivocation(
                    verification.signer_key, verification.release_sequence, str(verification.payload_digest)
                )
                if clash is not None:
                    raise IngestRefused(
                        duty="equivocation",
                        reason=(
                            f"signer {verification.signer_identity!r} already released sequence "
                            f"{verification.release_sequence} as a different payload "
                            f"({str(clash.get('payload_digest'))[:12]}…, not {str(verification.payload_digest)[:12]}…) "
                            "— one sequence binds to one release"
                        ),
                    )
            if verification.scheme == 1 and journal.has_v2_from(verification.signer_key):
                raise IngestRefused(
                    duty="downgrade",
                    reason=(
                        f"a v1 signature from signer {verification.signer_identity!r}, whose v2 releases this "
                        "receiver has already admitted — a key that signs v2 does not go back to v1"
                    ),
                )
        # Dedup on the ZIP bytes AND on the payload identity -- the VERIFIED payload digest when this
        # ingest verified, otherwise ``content_payload_digest`` (a hash of exactly what ingest reads: the
        # manifest + declared slices; no key, no authority). Equal whenever the bundle verifies, so a
        # re-zipped, re-signed-away or README-padded copy of an admitted release -- on either path -- is
        # the same release and is never merged twice.
        payload_digest = (
            verification.payload_digest
            if verification is not None
            else content_payload_digest(
                zf, max_member_bytes=MAX_ENTRY_UNCOMPRESSED_BYTES, max_total_bytes=MAX_TOTAL_UNCOMPRESSED_BYTES
            )
        )
        if payload_digest is None:
            notes.append(
                "the bundle's payload identity could not be computed (a manifest that is not strict JSON, or a "
                "declared slice that is unreadable or not UTF-8); dedup falls back to these exact ZIP bytes, so a "
                "re-packaged copy would not be recognised"
            )
        seen = [d for d in (digest, payload_digest) if d]
        if any(journal.has_digest(d) for d in seen) and not force_digest:
            raise IngestRefused(
                duty="V8",
                reason=(
                    f"bundle digest {digest[:12]}… was already ingested (as these bytes or as the same signed "
                    "release); re-ingestion sums counts and re-walks the mean fold (row J). Pass force_digest "
                    "for an eyes-open replay."
                ),
            )

        # V10 — capability_map entries are unverifiable claims; the adapter
        # validates shape and otherwise carries them (no reader exists yet).
        capability_map = manifest.get("capability_map", {}) or {}
        if not isinstance(capability_map, dict):
            raise IngestRefused(duty="V10", reason="manifest capability_map is not an object")

        # 6. V7 — declared slices only, read in memory (nothing undeclared
        # reaches a loader; nothing is extracted to disk).
        contents = manifest.get("contents", {}) or {}
        if not isinstance(contents, dict):
            raise IngestRefused(duty="V7", reason="manifest contents is not an object")
        declared_files: dict[str, str] = {}
        for slice_name, meta in contents.items():
            file_name = meta.get("file") if isinstance(meta, dict) else None
            if not isinstance(file_name, str) or not file_name:
                raise IngestRefused(duty="V7", reason=f"manifest contents[{slice_name!r}] declares no file")
            if file_name in ("manifest.json", SIGNATURE_MEMBER):
                # The verifier refuses this for a signed release; the unverified path must too, or it
                # would read the signature member (or the manifest) as a payload slice -- content no
                # identity or signature covers as a slice.
                raise IngestRefused(
                    duty="V7", reason=f"manifest contents[{slice_name!r}] declares {file_name!r}, which is not a slice"
                )
            declared_files[str(slice_name)] = file_name

        namelist = set(zf.namelist())
        missing = [f for f in declared_files.values() if f not in namelist]
        if missing:
            raise IngestRefused(
                duty="V3",
                reason=f"manifest declares slices absent from the archive: {sorted(missing)} (measured-class mismatch)",
            )
        undeclared = sorted(namelist - set(declared_files.values()) - {"manifest.json", SIGNATURE_MEMBER})
        if undeclared:
            notes.append(f"undeclared ZIP members ignored, never read: {undeclared} (V7)")

        donor_nac: dict[str, Any] | None = None
        donor_ec: dict[str, dict[str, Any]] | None = None
        if "nac" in declared_files:
            raw = _bounded_zip_read(zf, declared_files["nac"])
            parsed = _loads_strict(raw, slice_name=declared_files["nac"])
            if not isinstance(parsed, dict):
                raise IngestRefused(duty="V2", reason="nac.json is not a JSON object")
            donor_nac = parsed
            # A v2 release ships its NAc keys under a fixed agent token (entry_index.AGENT_TOKEN), so
            # re-keying to the receiver's own agent id is MANDATORY: without it the rows stay keyed to
            # the token, NAc reads filter by agent id, and every value would silently read 0.0.
            try:
                donor_agents = agent_ids(donor_nac)
            except EntryIndexError as exc:
                raise IngestRefused(duty="V2", reason=f"nac.json: {exc}") from exc
            if receiver_agent_id is None and AGENT_TOKEN in donor_agents:
                raise IngestRefused(
                    duty="rekey",
                    reason=(
                        f"the bundle's NAc rows are keyed to the agent token {AGENT_TOKEN!r} (a v2 release); "
                        "pass receiver_agent_id (--receiver-agent-id) so they re-key to your agent"
                    ),
                )
        if "ec" in declared_files:
            raw = _bounded_zip_read(zf, declared_files["ec"])
            parsed = _loads_strict(raw, slice_name=declared_files["ec"])
            if not isinstance(parsed, dict) or not isinstance(parsed.get("substrate_nodes"), dict):
                raise IngestRefused(duty="V2", reason="ec.json is not an object with substrate_nodes")
            donor_ec = parsed["substrate_nodes"]

    # 7. The payload admission pass (V2 / V9 / V3 / V1 sweep + stamping).
    if donor_nac is not None:
        donor_nac = _validate_nac_payload(donor_nac, contributor_id=contributor_id, now=now, notes=notes)
    if donor_ec is not None:
        donor_ec = _validate_ec_payload(
            donor_ec,
            contributor_id=contributor_id,
            manifest=manifest,
            allow_unstamped_geometry=allow_unstamped_geometry,
            notes=notes,
        )

    # 8. Inherent-class admission — Queen provenance only (the entry rule;
    # coding_habits_oasis.md §4). Refused loudly, never stripped: an
    # inherent-class claim from a non-Queen source is a privilege
    # escalation attempt on the safety floor, not a cleanable field.
    inherent_keys = list((donor_nac or {}).get("inherent_bias_keys", []) or [])
    if inherent_keys and contributor_id not in inherent_trusted_sources:
        raise IngestRefused(
            duty="inherent",
            reason=(
                f"bundle declares {len(inherent_keys)} inherent-class bias keys but contributor "
                f"{contributor_id!r} is not Queen-attested (inherent_trusted_sources). A "
                "locally-learned bias never self-promotes into the safety floor."
            ),
        )

    # 9. V4 — receiver-side quarantine + content scrub, re-run on receipt.
    links_dropped = welford_dropped = foreign_rows_dropped = 0
    valence_entries: dict[str, float] = {}
    if donor_nac is not None:
        donor_nac, links_dropped, welford_dropped = _receiver_scrub(donor_nac, notes=notes)
        # The operator's V4 report shows every valence the donor CLAIMS -- computed before the drop
        # below, so a valence the receiver will not keep is still visible (arch-lens finding 7: a lossy
        # trust report is worse than a longer one); the note says how many were not admitted.
        for key, valence in (donor_nac.get("percept_valences", {}) or {}).items():
            parts = str(key).split(NAC_KEY_SEP)
            if len(parts) == 3:
                # Keyed by entity_class + failure_mode: two failure modes on
                # one entity class must both reach the operator's report.
                valence_entries[f"{parts[1]} ({parts[2]})"] = float(valence)
        # Agent-keyed rows that are NOT situation rows (percept valences, per-tool outcome stats, the
        # node-keyed reward bias) cannot be re-keyed onto a receiver situation, so they keep the agent
        # segment they arrived with -- and NAc reads filter on the reader's own agent id. A row the
        # receiving agent can never read is dropped here, not stored as inert clutter that a later
        # signed export would have to reason about. Kept: rows under the receiver's own id, or (no
        # receiver id given) any real id, exactly the rows a read could reach before.
        donor_nac, foreign_rows_dropped = keep_agent_rows(
            donor_nac,
            # (With no receiver id, a token row cannot get here -- the `rekey` refusal above fires on
            # it first; the `!= AGENT_TOKEN` arm is the belt behind that refusal.)
            lambda agent: agent == receiver_agent_id if receiver_agent_id is not None else agent != AGENT_TOKEN,
            fields=NON_SITUATION_AGENT_FIELDS,
        )
        if foreign_rows_dropped:
            notes.append(
                f"{foreign_rows_dropped} non-situation NAc row(s) (percept valences / outcome stats / node-keyed "
                "reward bias) filed under another agent NOT admitted -- not readable by the receiving agent; "
                "situation rows re-key. The V4 valence report lists what the donor claimed."
            )

    # 10. The merge — through substrate_merge (alignment + re-key + fold +
    # the tighten-only clamp at its decided seam), with the reserved
    # parameters as the belt behind the V1 door.
    # Belt against the dangling-marker escalation (executor-lens finding 5):
    # a RECEIVER-side inherent marker whose bias no longer exists (e.g. left
    # by an older prune) must not ride into the fold, where a foreign bias
    # landing at the same triple would inherit its decay exemption. The
    # primary fix lives in prune_nac_cluster_biases; this catches journals
    # of state pruned before that fix.
    effective_receiver_nac = dict(receiver_nac or {})
    receiver_markers = effective_receiver_nac.get("inherent_bias_keys")
    if isinstance(receiver_markers, list):
        live_bias_keys = set((effective_receiver_nac.get("cluster_reward_bias") or {}).keys())
        kept_markers = [k for k in receiver_markers if str(k) in live_bias_keys]
        if len(kept_markers) != len(receiver_markers):
            notes.append(
                f"{len(receiver_markers) - len(kept_markers)} dangling receiver-side inherent markers "
                "dropped pre-merge (no live bias; escalation belt)"
            )
            effective_receiver_nac["inherent_bias_keys"] = kept_markers

    # trusted_sources here is the BELT behind the V1 door (it filters the
    # DONOR side only): after receiver-stamping, every admitted link/node
    # carries exactly {contributor_id}, so the subset test passes for
    # honest material and drops anything a bug let through unstamped.
    result: SubstrateMergeResult = substrate_merge(
        receiver_nac=effective_receiver_nac,
        receiver_ec=dict(receiver_ec_nodes or {}),
        donor_nac=donor_nac or {},
        donor_ec=donor_ec or {},
        receiver_source=_SELF_SOURCE,
        donor_source=contributor_id,
        receiver_agent_id=receiver_agent_id,
        strict_geometry=True,
        trusted_sources=frozenset({contributor_id}),
    )

    # Row N close-out: re-stamp manifest.domain onto INSERTED foreign nodes
    # only (an id the receiver did not hold before the merge). Folded-into
    # local survivors keep their own domain — foreign bytes never stamp them.
    merged_ec = result.ec_nodes
    if domain is not None and donor_ec is not None:
        receiver_ids = set((receiver_ec_nodes or {}).keys())
        for donor_id, target_id in result.id_map.items():
            if target_id not in receiver_ids and target_id in merged_ec:
                merged_ec[target_id]["domain"] = domain

    admitted_inherent = len(
        set(result.nac.get("inherent_bias_keys", []) or [])
        - set((receiver_nac or {}).get("inherent_bias_keys", []) or [])
    )

    journal_entry: dict[str, Any] = {
        "digest": digest,
        "contributor_id": contributor_id,
        "ingested_at": now,
        "bundle_path": str(bundle_path),
        "body_ref": manifest.get("body_ref"),
        "domain": domain,
        "biases_rekeyed": result.biases_rekeyed,
        "biases_dropped": result.biases_dropped,
        "biases_tightened": result.biases_tightened,
        "fear_rekeyed": result.fear_rekeyed,
        "fear_dropped": result.fear_dropped,
        "fear_below_floor": result.fear_below_floor,
        "fear_discount": FOREIGN_FEAR_DISCOUNT if result.fear_rekeyed else None,
        "inherent_keys_admitted": admitted_inherent,
        "links_dropped_identity": links_dropped,
        "welford_dropped_identity": welford_dropped,
        "foreign_rows_dropped": foreign_rows_dropped,
        "donor_nodes": len(donor_ec or {}),
        "notes": list(notes),
    }
    if payload_digest:
        # Dedup identity (see V8). ``payload_verified`` says whether a signature stood behind it; only
        # verified entries carry the signer fields the ordering rules read, so an unverified entry can
        # never cause (or be) an equivocation or downgrade refusal.
        journal_entry["payload_digest"] = payload_digest
        journal_entry["payload_verified"] = verification is not None
    if verification is not None:
        # The receiver-state record (release format v2): what the ordering rules above read back.
        journal_entry.update(
            {
                "signature_scheme": verification.scheme,
                "signer_key": verification.signer_key,
                "signer_identity": verification.signer_identity,
                "release_sequence": verification.release_sequence,
                "license": verification.license,
            }
        )

    return IngestReport(
        manifest=manifest,
        digest=digest,
        contributor_id=contributor_id,
        nac=result.nac,
        ec_nodes=merged_ec,
        id_map=result.id_map,
        biases_rekeyed=result.biases_rekeyed,
        biases_dropped=result.biases_dropped,
        biases_tightened=result.biases_tightened,
        fear_rekeyed=result.fear_rekeyed,
        fear_dropped=result.fear_dropped,
        fear_below_floor=result.fear_below_floor,
        inherent_keys_admitted=admitted_inherent,
        links_dropped_identity=links_dropped,
        welford_dropped_identity=welford_dropped,
        valence_entries=valence_entries,
        foreign_rows_dropped=foreign_rows_dropped,
        verification=verification,
        undeclared_members=undeclared,
        notes=notes,
        journal_entry=journal_entry,
    )


__all__ = [
    "CAP_FOREIGN_CONFIDENCE",
    "FOREIGN_FEAR_DISCOUNT",
    "IngestReport",
    "IngestRefused",
    "IngestionJournal",
    "MAX_BUNDLE_ENTRIES",
    "MAX_ENTRY_UNCOMPRESSED_BYTES",
    "MAX_FOREIGN_COUNT",
    "MAX_FOREIGN_DELTAS",
    "MAX_FOREIGN_TOTAL_OBSERVATIONS",
    "MAX_NODES_PER_SLICE",
    "MAX_TOTAL_UNCOMPRESSED_BYTES",
    "ingest_bundle",
]
