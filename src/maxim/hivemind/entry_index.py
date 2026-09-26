"""The Queen-signed entry index of a v2 bundle (docs/plans/oasis_entry_index_v2.md).

An **entry** is one situation cluster: the cluster's EC node when the bundle carries one, plus every NAc
row keyed by that cluster id (``cluster_fear``, ``cluster_reward_bias``, ``cluster_reward_source``) and
the ``inherent_bias_keys`` naming those rows. Node-less entries are allowed (a NAc-only lineage). Rows
not keyed by a situation stay whole-bundle and never become entries.

Each entry's **digest** is ``sha256`` over the RFC 8785 (JCS) serialization of its **projection**,
built from the slices as parsed from the signed bytes. The exporter normalizes the agent segment of
every NAc composite key to :data:`AGENT_TOKEN` before signing (so two rows can never collapse onto one
key, and local agent ids never ship); the verifier refuses any other segment.

This module is pure: it computes and checks, it never reads or writes a bundle.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from typing import Any

from maxim.hivemind.merge import NAC_KEY_SEP, NODE_ID_CHARSET
from maxim.utils.optional_deps import require_optional_dependency

#: The agent segment every NAc composite key carries in a v2 bundle. A receiver re-keys it to its own
#: agent id at ingest (``merge.rekey_nac_state(to_agent_id=...)``), which is therefore mandatory.
AGENT_TOKEN = "agent"

#: The NAc fields whose rows belong to a situation cluster (the ``cluster`` is the key's middle part).
CLUSTER_FIELDS: tuple[str, ...] = ("cluster_fear", "cluster_reward_bias", "cluster_reward_source")

#: ``␟``-joined composite-key fields whose FIRST part is an agent id.
_TRIPLE_FIELDS: tuple[str, ...] = (*CLUSTER_FIELDS, "percept_valences")

#: The EC-node fields an entry's projection covers. ``source`` / ``contributors`` are excluded: the
#: exporter re-stamps them, so the same entry from two exporters keeps one digest.
EC_NODE_FIELDS: tuple[str, ...] = ("modality", "embedding", "geometry", "count", "domain")

#: The index format version this build writes and verifies.
INDEX_VERSION = 1

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class EntryIndexError(ValueError):
    """A bundle's situation state and its entry index disagree, or cannot be indexed."""


# ── agent-segment normalization (export side) ──────────────────────────────────────────────────


def _agent_ids(nac: Mapping[str, Any]) -> set[str]:
    ids: set[str] = set()
    for field in _TRIPLE_FIELDS:
        for key in (nac.get(field) or {}).keys():
            ids.add(str(key).split(NAC_KEY_SEP, 1)[0])
    for key in nac.get("inherent_bias_keys") or []:
        ids.add(str(key).split(NAC_KEY_SEP, 1)[0])
    for key in (nac.get("event_outcome_welford") or {}).keys():
        ids.add(str(key).split(NAC_KEY_SEP, 1)[0])
    for key in (nac.get("reward_bias") or {}).keys():
        ids.add(str(key).split(":", 1)[0])
    return ids


def _retoken(key: Any, sep: str) -> str:
    head, _, rest = str(key).partition(sep)
    return f"{AGENT_TOKEN}{sep}{rest}"


def normalize_agent_segment(nac: Mapping[str, Any]) -> dict[str, Any]:
    """Rewrite the agent segment of every NAc composite key to :data:`AGENT_TOKEN`. Pure.

    Refuses (``EntryIndexError``) a state holding rows for more than one local agent id: normalizing
    them would collapse two agents' rows onto one key, and a digest must never cover two values.
    """
    ids = _agent_ids(nac)
    if len(ids - {AGENT_TOKEN}) > 1:
        raise EntryIndexError(
            f"the NAc state holds rows for {len(ids)} agent ids {sorted(ids)}; a signed bundle is one agent's "
            "learning -- export one agent at a time"
        )
    out = dict(nac)
    for field in _TRIPLE_FIELDS:
        if field in nac:
            out[field] = {_retoken(k, NAC_KEY_SEP): v for k, v in (nac.get(field) or {}).items()}
    if "inherent_bias_keys" in nac:
        out["inherent_bias_keys"] = sorted(_retoken(k, NAC_KEY_SEP) for k in nac.get("inherent_bias_keys") or [])
    if "event_outcome_welford" in nac:
        out["event_outcome_welford"] = {
            _retoken(k, NAC_KEY_SEP): v for k, v in (nac.get("event_outcome_welford") or {}).items()
        }
    if "reward_bias" in nac:
        out["reward_bias"] = {_retoken(k, ":"): v for k, v in (nac.get("reward_bias") or {}).items()}
    return out


# ── the projection and its digest ──────────────────────────────────────────────────────────────


def _split_triple(key: Any) -> tuple[str, str, str] | None:
    parts = str(key).split(NAC_KEY_SEP)
    return (parts[0], parts[1], parts[2]) if len(parts) == 3 and all(parts) else None


def _node_count(node: Mapping[str, Any]) -> Any:
    """The ingest fallback: ``count``, else ``member_count``."""
    return node.get("count", node.get("member_count"))


def _check_finite(value: Any, where: str) -> None:
    """JCS cannot represent NaN/Infinity, and a lone surrogate cannot be UTF-8 encoded: refuse both."""
    if isinstance(value, float) and not math.isfinite(value):
        raise EntryIndexError(f"non-finite number at {where}")
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            raise EntryIndexError(f"unencodable string at {where}") from None
    elif isinstance(value, Mapping):
        for k, v in value.items():
            _check_finite(k, where)
            _check_finite(v, f"{where}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _check_finite(v, f"{where}[{i}]")


def jcs(value: Any) -> bytes:
    """RFC 8785 (JCS) serialization, via the ``rfc8785`` package (the ``[sign]`` extra)."""
    require_optional_dependency("rfc8785", feature="Oasis v2 entry index")
    import rfc8785  # noqa: PLC0415 -- optional dependency, loaded through the canonical surface

    _check_finite(value, "projection")
    try:
        return bytes(rfc8785.dumps(value))
    except (TypeError, ValueError) as exc:  # the package's own refusals (e.g. an int beyond IEEE range)
        raise EntryIndexError(f"projection not JCS-serializable: {exc}") from None


def entries(nac: Mapping[str, Any] | None, ec_nodes: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Every entry's projection, keyed by cluster id: EC nodes, plus clusters that only NAc rows name.

    Raises ``EntryIndexError`` on a cluster-keyed row that is not a well-formed triple (such a row can
    belong to no entry, and "no unindexed situation state" must hold).
    """
    nac = nac or {}
    ec_nodes = ec_nodes or {}
    out: dict[str, dict[str, Any]] = {}

    def entry(cid: str) -> dict[str, Any]:
        if cid not in out:
            node = ec_nodes.get(cid)
            out[cid] = {
                "id": cid,
                "ec_node": None
                if node is None
                else {f: (_node_count(node) if f == "count" else node.get(f)) for f in EC_NODE_FIELDS},
                "nac": {field: {} for field in CLUSTER_FIELDS},
                "inherent": [],
            }
        return out[cid]

    for cid in ec_nodes:
        entry(str(cid))
    for field in CLUSTER_FIELDS:
        for key, value in (nac.get(field) or {}).items():
            triple = _split_triple(key)
            if triple is None:
                raise EntryIndexError(f"{field} key {key!r} is not agent\\x1fcluster\\x1fthird")
            _, cid, third = triple
            entry(cid)["nac"][field][f"{cid}{NAC_KEY_SEP}{third}"] = value
    for key in nac.get("inherent_bias_keys") or []:
        triple = _split_triple(key)
        if triple is None:
            raise EntryIndexError(f"inherent_bias_keys entry {key!r} is not a triple")
        _, cid, third = triple
        if key not in (nac.get("cluster_reward_bias") or {}):
            raise EntryIndexError(f"inherent marker {key!r} names no cluster_reward_bias row (dangling)")
        entry(cid)["inherent"].append(f"{cid}{NAC_KEY_SEP}{third}")
    for e in out.values():
        e["inherent"].sort()
    return out


def digest(projection: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(jcs(dict(projection))).hexdigest()


def build_index(nac: Mapping[str, Any] | None, ec_nodes: Mapping[str, Any] | None) -> dict[str, Any]:
    """The ``entry_index`` manifest field for these (already scrubbed and normalized) slices."""
    items = []
    for cid, projection in sorted(entries(nac, ec_nodes).items()):
        node = projection["ec_node"]
        items.append(
            {"id": cid, "modality": None if node is None else node.get("modality"), "digest": digest(projection)}
        )
    return {"version": INDEX_VERSION, "entries": items}


# ── verification (receiver side) ───────────────────────────────────────────────────────────────


def verify_index(
    index: Any,
    nac: Mapping[str, Any] | None,
    ec_nodes: Mapping[str, Any] | None,
    *,
    max_entries: int,
) -> dict[str, str]:
    """Check a signed bundle's ``entry_index`` against its slices. Returns ``{id: digest}`` (recomputed).

    Refuses (``EntryIndexError``): an unknown index version; malformed, duplicate or unsorted ids; a
    malformed digest; an index modality differing from its node's; an agent segment other than
    :data:`AGENT_TOKEN`; a malformed cluster key or dangling inherent marker; any entry the slices hold
    that the index omits, or the reverse; a recomputed digest that differs.
    """
    if not isinstance(index, Mapping) or index.get("version") != INDEX_VERSION:
        raise EntryIndexError(f"entry_index version {getattr(index, 'get', lambda _k: None)('version')!r} unsupported")
    items = index.get("entries")
    if not isinstance(items, list):
        raise EntryIndexError("entry_index.entries is not a list")
    if len(items) > max_entries:
        raise EntryIndexError(f"entry_index lists {len(items)} entries (cap {max_entries})")
    ids: list[str] = []
    claimed: dict[str, Mapping[str, Any]] = {}
    for item in items:
        if not isinstance(item, Mapping):
            raise EntryIndexError("an entry_index entry is not an object")
        eid, dig = item.get("id"), item.get("digest")
        if not isinstance(eid, str) or not NODE_ID_CHARSET.match(eid):
            raise EntryIndexError(f"entry id {eid!r} is malformed")
        if not isinstance(dig, str) or not _DIGEST.match(dig):
            raise EntryIndexError(f"entry {eid!r} digest {dig!r} is malformed")
        if eid in claimed:
            raise EntryIndexError(f"entry id {eid!r} is listed twice")
        ids.append(eid)
        claimed[eid] = item
    if ids != sorted(ids):
        raise EntryIndexError("entry_index entries are not sorted by id")
    agents = _agent_ids(nac or {})
    if agents - {AGENT_TOKEN}:
        raise EntryIndexError(f"agent segment(s) {sorted(agents - {AGENT_TOKEN})} are not the token {AGENT_TOKEN!r}")
    actual = entries(nac, ec_nodes)
    missing, extra = sorted(set(actual) - set(claimed)), sorted(set(claimed) - set(actual))
    if missing:
        raise EntryIndexError(f"situation state not covered by the index (unindexed): {missing[:5]}")
    if extra:
        raise EntryIndexError(f"index lists entries the slices do not hold: {extra[:5]}")
    recomputed: dict[str, str] = {}
    for eid, projection in actual.items():
        node = projection["ec_node"]
        if claimed[eid].get("modality") != (None if node is None else node.get("modality")):
            raise EntryIndexError(f"entry {eid!r} index modality differs from its node's")
        recomputed[eid] = digest(projection)
        if recomputed[eid] != claimed[eid]["digest"]:
            raise EntryIndexError(f"entry {eid!r} digest does not match its slices")
    return recomputed


__all__ = [
    "AGENT_TOKEN",
    "CLUSTER_FIELDS",
    "EC_NODE_FIELDS",
    "INDEX_VERSION",
    "EntryIndexError",
    "build_index",
    "digest",
    "entries",
    "jcs",
    "normalize_agent_segment",
    "verify_index",
]
