"""The Queen-signed entry index of a v2 bundle (docs/plans/oasis_entry_index_v2.md).

An **entry** is one situation cluster: the cluster's EC node when the bundle carries one, plus every NAc
row keyed by that cluster id (``cluster_fear``, ``cluster_reward_bias``, ``cluster_reward_source``) and
the ``inherent_bias_keys`` naming those rows. Node-less entries are allowed (a NAc-only lineage). Rows
not keyed by a situation stay whole-bundle and never become entries.

Each entry's **digest** is ``sha256`` over the RFC 8785 (JCS) serialization of its **projection**,
built from the slices as parsed from the signed bytes. The exporter keeps only its own agent's NAc rows
and rewrites their agent segment to :data:`AGENT_TOKEN` before signing (so two agents' rows can never
collapse onto one key, and local agent ids never ship); the verifier refuses any other segment.

Equal digests mean equal CONTENT, not equal admission: ``source`` / ``contributors`` are outside the
projection, and ingest's V1 sweep reads them -- a per-entry dedup must not treat one as the other.

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
#: agent id at ingest (``merge.rekey_nac_state(to_agent_id=...)``), which is therefore mandatory. In the
#: reserved ``_`` namespace, so it can never be mistaken for a real agent's id.
AGENT_TOKEN = "_agent"

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


# ── shape (hostile input is refused as EntryIndexError, never an AttributeError) ─────────────────


def _mapping_field(nac: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = nac.get(field) or {}
    if not isinstance(value, Mapping):
        raise EntryIndexError(f"NAc field {field!r} is not an object")
    return value


def _list_field(nac: Mapping[str, Any], field: str) -> list[Any]:
    value = nac.get(field) or []
    if not isinstance(value, list):
        raise EntryIndexError(f"NAc field {field!r} is not a list")
    return value


# ── agent-segment normalization (export side) ──────────────────────────────────────────────────


def _links_by_agent(nac: Mapping[str, Any]) -> list[tuple[str, str | None]]:
    """``(event signature, agent id | None)`` per causal link, the id from ``event_context.agent_id``."""
    out: list[tuple[str, str | None]] = []
    links = nac.get("links") or {}
    if not isinstance(links, Mapping):
        raise EntryIndexError("NAc field 'links' is not an object")
    for sig, bucket in links.items():
        if not isinstance(bucket, list):
            raise EntryIndexError(f"links[{sig!r}] is not a list")
        for link in bucket:
            context = link.get("event_context") if isinstance(link, Mapping) else None
            agent = context.get("agent_id") if isinstance(context, Mapping) else None
            out.append((str(sig), agent if isinstance(agent, str) else None))
    return out


def agent_ids(nac: Mapping[str, Any]) -> set[str]:
    """Every agent id a NAc state names: its composite keys' first segment AND its causal links'
    ``event_context.agent_id``. Links count: ``NAc.predict`` matches a link's event context, so a link is
    agent-scoped in practice -- a release's own-rows rule, its token check and its re-key cover them."""
    if not isinstance(nac, Mapping):
        raise EntryIndexError("the NAc state is not an object")
    ids: set[str] = set()
    for field in _TRIPLE_FIELDS:
        for key in _mapping_field(nac, field):
            ids.add(str(key).split(NAC_KEY_SEP, 1)[0])
    for key in _list_field(nac, "inherent_bias_keys"):
        ids.add(str(key).split(NAC_KEY_SEP, 1)[0])
    for key in _mapping_field(nac, "event_outcome_welford"):
        ids.add(str(key).split(NAC_KEY_SEP, 1)[0])
    for key in _mapping_field(nac, "reward_bias"):
        ids.add(str(key).split(":", 1)[0])
    ids.update(agent for _, agent in _links_by_agent(nac) if agent is not None)
    return ids


def is_agent_id(value: Any) -> bool:
    """A usable agent id: a non-empty string holding neither key separator. ``reward_bias`` keys split
    on ``:`` and the composite keys on ``\x1f``, so an id containing either would be misread as a
    different agent (and its own rows dropped as foreign)."""
    return isinstance(value, str) and bool(value) and ":" not in value and NAC_KEY_SEP not in value


def _retoken(key: Any, sep: str) -> str:
    head, _, rest = str(key).partition(sep)
    return f"{AGENT_TOKEN}{sep}{rest}"


#: Agent-keyed fields that are NOT situation rows, so ingest cannot re-key them onto a receiver
#: situation: ``(field, separator)``. ``inherent_bias_keys`` is a list; the rest are objects.
NON_SITUATION_AGENT_FIELDS: tuple[tuple[str, str], ...] = (
    ("percept_valences", NAC_KEY_SEP),
    ("event_outcome_welford", NAC_KEY_SEP),
    ("reward_bias", ":"),
)
_ALL_AGENT_FIELDS: tuple[tuple[str, str], ...] = (
    *((f, NAC_KEY_SEP) for f in CLUSTER_FIELDS),
    ("inherent_bias_keys", NAC_KEY_SEP),
    *NON_SITUATION_AGENT_FIELDS,
)


def keep_agent_rows(
    nac: Mapping[str, Any], keep: Any, *, fields: tuple[tuple[str, str], ...] = _ALL_AGENT_FIELDS
) -> tuple[dict[str, Any], int]:
    """``(nac with only the rows whose agent segment satisfies keep(agent), rows dropped)``. Pure.

    One rule for both ends of a release: an agent's NAc reads filter on its own agent id, so a row
    filed under any other id is never read by it -- the exporter does not ship such rows and a
    receiver does not keep them.
    """
    out = dict(nac)
    dropped = 0
    for field, sep in fields:
        if field not in nac:
            continue
        value = _list_field(nac, field) if field == "inherent_bias_keys" else _mapping_field(nac, field)
        if isinstance(value, list):
            kept_list = [k for k in value if keep(str(k).split(sep, 1)[0])]
            dropped += len(value) - len(kept_list)
            out[field] = kept_list
        else:
            kept = {k: v for k, v in value.items() if keep(str(k).split(sep, 1)[0])}
            dropped += len(value) - len(kept)
            out[field] = kept
    return out, dropped


def _retoken_link(link: Any) -> Any:
    if not isinstance(link, Mapping):
        return link
    context = link.get("event_context")
    if not isinstance(context, Mapping) or "agent_id" not in context:
        return link
    return {**link, "event_context": {**context, "agent_id": AGENT_TOKEN}}


def normalize_agent_segment(nac: Mapping[str, Any], *, own_agent_id: str | None = None) -> tuple[dict[str, Any], int]:
    """Keep ONE agent's rows and rewrite their agent segment to :data:`AGENT_TOKEN`. Pure.

    Returns ``(normalized nac, rows dropped)``. The agent is ``own_agent_id`` when given, else the one
    real agent id the state holds; rows under any other id -- another agent's, or the token itself
    (rows an ingested release left) -- are dropped, never relabelled: relabelling would collapse two
    agents' rows onto one key, and a signed digest must never cover a value its signer did not learn.
    Refuses (``EntryIndexError``) a state with several real agent ids and no ``own_agent_id``, and an
    ``own_agent_id`` that holds no rows while others do.
    """
    ids = agent_ids(nac)
    real = ids - {AGENT_TOKEN}
    if own_agent_id is not None:
        if not is_agent_id(own_agent_id) or own_agent_id == AGENT_TOKEN:
            raise EntryIndexError(f"own_agent_id {own_agent_id!r} is not an agent id")
        if real and own_agent_id not in real:
            raise EntryIndexError(
                f"agent {own_agent_id!r} holds no rows in this NAc state (agent ids present: {sorted(real)})"
            )
        own: str | None = own_agent_id
    elif len(real) > 1:
        raise EntryIndexError(
            f"the NAc state holds rows for {len(real)} agent ids {sorted(real)}: name which one is yours "
            "(agent_id= / --agent-id) -- a signed release ships one agent's learning"
        )
    else:
        own = next(iter(real), None)
    kept, dropped = keep_agent_rows(nac, lambda agent: agent == own)
    out = dict(kept)
    for field, sep in _ALL_AGENT_FIELDS:
        if field not in kept:
            continue
        if field == "inherent_bias_keys":
            out[field] = sorted(_retoken(k, sep) for k in kept[field])
        else:
            out[field] = {_retoken(k, sep): v for k, v in kept[field].items()}
    # A causal link names its agent in ``event_context.agent_id`` -- and it is NOT inert once stored:
    # ``NAc.predict`` matches a link's event context. So links follow the own-rows rule too: a link naming
    # another agent (or the token an earlier ingest left) is dropped and counted; the exporter's own, and
    # links naming no agent, ship -- relabelled to the token, which ingest re-keys to the receiver
    # (``merge.rekey_nac_state``).
    links = kept.get("links")
    if isinstance(links, Mapping):
        shipped: dict[str, Any] = {}
        for sig, bucket in links.items():
            if not isinstance(bucket, list):
                continue
            own_links = []
            for link in bucket:
                context = link.get("event_context") if isinstance(link, Mapping) else None
                agent = context.get("agent_id") if isinstance(context, Mapping) else None
                if agent is not None and agent != own:
                    dropped += 1
                    continue
                own_links.append(_retoken_link(link))
            if own_links:
                shipped[sig] = own_links
        out["links"] = shipped
    return out, dropped


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
    if not isinstance(nac, Mapping) or not isinstance(ec_nodes, Mapping):
        raise EntryIndexError("the NAc state and the EC substrate_nodes must be objects")
    bad_nodes = [cid for cid, node in ec_nodes.items() if not isinstance(node, Mapping)]
    if bad_nodes:
        raise EntryIndexError(f"EC node(s) {sorted(map(str, bad_nodes))[:3]} are not objects")
    reward_rows = _mapping_field(nac, "cluster_reward_bias")
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
        for key, value in _mapping_field(nac, field).items():
            triple = _split_triple(key)
            if triple is None:
                raise EntryIndexError(f"{field} key {key!r} is not agent\\x1fcluster\\x1fthird")
            _, cid, third = triple
            entry(cid)["nac"][field][f"{cid}{NAC_KEY_SEP}{third}"] = value
    for key in _list_field(nac, "inherent_bias_keys"):
        triple = _split_triple(key)
        if triple is None:
            raise EntryIndexError(f"inherent_bias_keys entry {key!r} is not a triple")
        _, cid, third = triple
        if key not in reward_rows:
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
    agents = agent_ids(nac or {})
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
    "agent_ids",
    "build_index",
    "digest",
    "entries",
    "jcs",
    "is_agent_id",
    "keep_agent_rows",
    "normalize_agent_segment",
    "verify_index",
]
