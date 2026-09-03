"""Bayesian aggregation of NAc + EC state dicts for the Hivemind.

v1_refinement.md §B5 PR B. Pure-function utilities that take two
contributors' substrate state dicts (typically ``NAc.dump()`` / the
``substrate_nodes`` slice of ``EntorhinalCortex.save()``) and produce a
merged state that loads back into a live system via the corresponding
``load_state()`` / ``load()`` paths.

Design decisions (locked 2026-05-30 per kickoff Q&A):

1. **Zero prior for unobserved entries** — a contributor that never
   observed a (agent, key) pair contributes no evidence. Its absence
   neither boosts nor decays the merged value. Matches how NAc already
   clamps reward bias to ``[0, max]`` and treats unknown as absent.
2. **Valence-distinct CausalLinks stay separate** — when two
   contributors disagree about the valence of the same event-outcome
   pair (POSITIVE vs NEGATIVE), each side's link is preserved. The
   ``contributors`` tuple on each link records who voted that way.
   Matches ``maxim_hivemind.md``: "Outcome valence preserved as
   separate distributions, not collapsed."
3. **EC centroid merging uses ``member_count``-weighted mean** —
   reuses the existing running-mean math (``ec.py:387-392``). A node
   with 10 members weighs 10× a 1-member node.
4. **Cosine match threshold defaults to 0.44** — matches
   ``ECConfig.pattern_complete_threshold``. Caller can override.

Convention:

- ``left`` / ``right`` — the two input state dicts.
- ``left_source`` / ``right_source`` — opaque contributor IDs
  (e.g. ``"oasis-abc"``, ``"local"``). Required keyword-only.
- A link / node that appeared in only one side keeps its existing
  ``source`` and ``contributors``.
- A link / node that appeared in BOTH sides aggregates: ``source``
  becomes :data:`CONSENSUS_SOURCE`, ``contributors`` becomes the
  order-preserving union of both sides' contributors (each side's
  contributors are either its existing ``contributors`` tuple if
  populated, or ``(its source,)`` if solo).
- All merge functions are PURE: input dicts are not mutated; the
  output is a fresh dict.
- Commutativity: ``nac_merge(a, b, ...) == nac_merge(b, a, ...)`` for
  the merged values (contributor ORDER may differ between the two
  call orderings, which is intentional — the audit trail records the
  observation order).
- Idempotence on identical inputs: ``nac_merge(a, a, ..., left_source=X,
  right_source=X)`` yields per-key values unchanged except ``contributors``
  collapses to ``(X,)`` after de-duplication.

Inputs are assumed to be at the current ``_format_version``. Migration
of older dumps is the bundle-importer's responsibility (PR D), not
this module's.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

# Reserved marker for fan-in links / nodes. Distinguishes "this came
# from a single contributor" (``source = "oasis-x"``) from "this
# aggregated multiple contributors" (``source = CONSENSUS_SOURCE``,
# ``contributors = ("oasis-x", "oasis-y", ...)``). Reserved as a class
# attribute on this module so callers compare against the symbol rather
# than re-typing the literal.
#
# Underscore-prefixed by design — the merge functions reject contributor
# IDs in the reserved ``_*`` namespace at validation time so a user
# can't accidentally (or maliciously) supply ``left_source="_consensus"``
# and shadow the sentinel.
CONSENSUS_SOURCE: str = "_consensus"

# Reserved-namespace prefix for sentinel sources. Any contributor ID
# starting with this string is rejected by ``_validate_source``.
_RESERVED_SOURCE_PREFIX: str = "_"


def _validate_source(source: str, *, label: str) -> None:
    """Reject contributor IDs in the reserved ``_*`` namespace.

    Pre-merge review (PR B Architecture-lens IMPORTANT) flagged the
    risk that a caller supplies ``left_source=CONSENSUS_SOURCE`` and
    shadows the fan-in sentinel. Validating at entry to every public
    merge function is the structural fix.
    """
    if not isinstance(source, str) or not source:
        raise ValueError(f"{label} must be a non-empty string, got {source!r}")
    if source.startswith(_RESERVED_SOURCE_PREFIX):
        raise ValueError(
            f"{label}={source!r} starts with the reserved prefix "
            f"{_RESERVED_SOURCE_PREFIX!r} — contributor IDs must not "
            f"collide with sentinel markers like {CONSENSUS_SOURCE!r}."
        )


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def _link_contributors(link: dict[str, Any], fallback_source: str) -> tuple[str, ...]:
    """Resolve a link's effective contributor list.

    If ``contributors`` is populated, that IS the contributor set
    (the link is already a prior fan-in result). If empty, the link
    is solo-source — the contributor is its ``source`` field, falling
    back to ``fallback_source`` if absent.
    """
    contribs = tuple(link.get("contributors", ()))
    if contribs:
        return contribs
    source = link.get("source", fallback_source)
    return (source,)


def _union_contributors(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    """Order-preserving deduplicated union of two contributor lists."""
    seen: set[str] = set()
    merged: list[str] = []
    for c in (*left, *right):
        if c not in seen:
            seen.add(c)
            merged.append(c)
    return tuple(merged)


def _resolved_source(contributors: tuple[str, ...]) -> str:
    """Pick the ``source`` value for a merged link / node.

    Single contributor → that contributor's ID. Multiple → the
    :data:`CONSENSUS_SOURCE` sentinel.
    """
    if len(contributors) <= 1:
        return contributors[0] if contributors else "local"
    return CONSENSUS_SOURCE


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two dense vectors; 0.0 on zero norm.

    Duplicated from ``maxim.similarity.ec._cosine_similarity`` to keep
    the Hivemind layer free of internal-module imports — same math,
    same edge case.

    DIMENSION MISMATCH IS NOT SIMILARITY (2026-08-06). Vectors of
    different length come from different encoder spaces and are not
    comparable at all. ``zip`` silently truncates to the shorter one, so
    a 384-dim node and a 768-dim node of the same modality tag were
    compared over the first 384 dims and MERGED whenever that partial
    cosine cleared the threshold — a silent cross-space corruption on
    the shipped ``ec_merge`` surface (``ec_merge`` gates on ``modality``
    only, never on dimension, and EC node payloads carry no encoder
    identity). Returning 0.0 makes the pair fall below every threshold,
    so the right-side node inserts as its OWN node instead of
    contaminating a left-side centroid — the non-destructive outcome,
    and consistent with the zero-norm convention above.
    """
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    n_a = sum(x * x for x in a) ** 0.5
    n_b = sum(x * x for x in b) ** 0.5
    if n_a == 0.0 or n_b == 0.0:
        return 0.0
    return dot / (n_a * n_b)


# ─────────────────────────────────────────────────────────────────────────
# CausalLink merge
# ─────────────────────────────────────────────────────────────────────────


def _merge_link_pair(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    left_source: str,
    right_source: str,
) -> dict[str, Any]:
    """Aggregate two CausalLink dicts that share ``(event_sig, outcome_sig)``.

    Caller must pre-check both links share both signatures — this
    function does not re-validate.

    Per design rule #2 (valence-distinct stay separate) this function
    is ONLY called when both sides agree on ``outcome_valence``, since
    valence is part of how ``outcome_signature`` is constructed via
    ``NAc._generate_link_id`` (the link-key embeds valence). If a
    future change decouples the two, the caller (``_merge_link_lists``)
    must skip pairs with different valences.
    """
    n_l = int(left.get("observation_count", 0))
    n_r = int(right.get("observation_count", 0))
    total_n = n_l + n_r

    # Weighted-mean Rescorla-Wagner predicted_value
    pv_l = float(left.get("predicted_value", 0.5))
    pv_r = float(right.get("predicted_value", 0.5))
    if total_n > 0:
        merged_pv = (n_l * pv_l + n_r * pv_r) / total_n
    else:
        merged_pv = (pv_l + pv_r) / 2.0

    # Confidence: cap at the higher of the two — consensus across
    # multiple observers is at least as confident as the more confident
    # individual observation.
    merged_conf = max(float(left.get("confidence", 0.5)), float(right.get("confidence", 0.5)))

    # Memory IDs: union, capped at 50 per the existing CausalLink
    # invariant (``record_observation`` truncates to last 50). Order
    # preserved — left first, then right's novel IDs.
    seen_mids: set[str] = set()
    merged_memory_ids: list[str] = []
    for mid in (*left.get("memory_ids", []), *right.get("memory_ids", [])):
        if mid not in seen_mids:
            seen_mids.add(mid)
            merged_memory_ids.append(mid)
    merged_memory_ids = merged_memory_ids[-50:]

    # Temporal delta: union both sides' ``observed_deltas``, then
    # truncate to the last 100 per the ``TemporalDelta.add_observation``
    # ring-buffer invariant. Pre-fold, only left's deltas survived — the
    # IMPORTANT review finding flagged this as silent data loss on the
    # field that drives B2 oscillator imminence prediction. Concatenation
    # preserves left first; the trailing 100 stays bounded.
    left_td = left.get("temporal_delta") or {}
    right_td = right.get("temporal_delta") or {}
    merged_deltas = list(left_td.get("observed_deltas", [])) + list(right_td.get("observed_deltas", []))
    merged_deltas = merged_deltas[-100:]

    # Contributors
    contribs = _union_contributors(
        _link_contributors(left, left_source),
        _link_contributors(right, right_source),
    )

    # Take left's deterministic fields (id, event_type, signatures,
    # outcome_type, outcome_valence — they're identical by precondition
    # except possibly id, where we pick left's deterministically).
    # imagined: OR — if either side learned this from imagination,
    # the merged link inherits the partial-confidence tag.
    return {
        "id": left["id"],
        "event_type": left["event_type"],
        "event_signature": left["event_signature"],
        "event_context": dict(left.get("event_context", {})),
        "outcome_type": left["outcome_type"],
        "outcome_signature": left["outcome_signature"],
        "outcome_valence": left["outcome_valence"],
        "temporal_delta": {"observed_deltas": merged_deltas},
        "predicted_value": merged_pv,
        "prediction_history": list(left.get("prediction_history", [])),
        "observation_count": total_n,
        "confidence": merged_conf,
        "last_observed": max(
            float(left.get("last_observed", 0.0)),
            float(right.get("last_observed", 0.0)),
        ),
        "memory_ids": merged_memory_ids,
        "context_factors": dict(left.get("context_factors", {})),
        "last_rpe": left.get("last_rpe"),
        "percept_refs": list(left.get("percept_refs", [])),
        "imagined": bool(left.get("imagined", False)) or bool(right.get("imagined", False)),
        "source": _resolved_source(contribs),
        "domain": left.get("domain") or right.get("domain"),
        "contributors": list(contribs),
    }


def _merge_link_lists(
    left_links: list[dict[str, Any]],
    right_links: list[dict[str, Any]],
    *,
    left_source: str,
    right_source: str,
) -> list[dict[str, Any]]:
    """Merge two lists of CausalLink dicts that share the same ``event_signature``.

    Links are paired by ``outcome_signature`` (which already embeds
    valence — see CausalLink class docstring). Pairs are aggregated;
    unique links are preserved as-is per design rule #2 (valence-
    distinct stay separate, but the same valence pair just hasn't
    been observed by both contributors yet).
    """
    by_outcome: dict[str, tuple[dict[str, Any] | None, dict[str, Any] | None]] = {}
    for ld in left_links:
        by_outcome[ld["outcome_signature"]] = (ld, None)
    for rd in right_links:
        outcome_sig = rd["outcome_signature"]
        ld_existing = by_outcome.get(outcome_sig, (None, None))[0]
        by_outcome[outcome_sig] = (ld_existing, rd)

    merged: list[dict[str, Any]] = []
    for ld, rd in by_outcome.values():
        if ld is None and rd is not None:
            merged.append(copy.deepcopy(rd))
        elif rd is None and ld is not None:
            merged.append(copy.deepcopy(ld))
        elif ld is not None and rd is not None:
            merged.append(_merge_link_pair(ld, rd, left_source=left_source, right_source=right_source))
    return merged


# ─────────────────────────────────────────────────────────────────────────
# Scalar field mergers
# ─────────────────────────────────────────────────────────────────────────


def _merge_mean_clamped(
    left: dict[str, float],
    right: dict[str, float],
    *,
    lo: float,
    hi: float,
) -> dict[str, float]:
    """Merge two ``key → float`` dicts by unweighted mean of shared keys.

    Unique-to-one-side keys keep their value (zero-prior rule: the
    other contributor's absence is no evidence). Shared-key values are
    averaged then clamped to ``[lo, hi]``.

    Per-key observation counts are not available for these scalar
    NAc fields (``reward_bias`` etc.), so observation-weighted mean
    is not possible. Unweighted mean is the principled fallback.
    """
    merged: dict[str, float] = {}
    keys = set(left.keys()) | set(right.keys())
    for k in keys:
        if k in left and k in right:
            v = (float(left[k]) + float(right[k])) / 2.0
        elif k in left:
            v = float(left[k])
        else:
            v = float(right[k])
        merged[k] = max(lo, min(hi, v))
    return merged


def _merge_welford(
    left: dict[str, dict[str, float]],
    right: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Parallel-Welford merge for ``{mean, m2, n}`` Welford states.

    Uses Chan's algorithm (numerically stable for unequal-size
    aggregates). Unique-to-one-side keys keep their state. Shared keys
    use::

        n  = n_a + n_b
        mu = (n_a * mu_a + n_b * mu_b) / n
        m2 = m2_a + m2_b + (mu_a - mu_b)**2 * n_a * n_b / n
    """
    merged: dict[str, dict[str, float]] = {}
    keys = set(left.keys()) | set(right.keys())
    for k in keys:
        if k in left and k in right:
            la, lb = left[k], right[k]
            n_a, n_b = float(la.get("n", 0.0)), float(lb.get("n", 0.0))
            n = n_a + n_b
            if n == 0.0:
                merged[k] = {"mean": 0.0, "m2": 0.0, "n": 0.0}
                continue
            mu_a, mu_b = float(la.get("mean", 0.0)), float(lb.get("mean", 0.0))
            m2_a, m2_b = float(la.get("m2", 0.0)), float(lb.get("m2", 0.0))
            mu = (n_a * mu_a + n_b * mu_b) / n
            m2 = m2_a + m2_b + (mu_a - mu_b) ** 2 * n_a * n_b / n
            merged[k] = {"mean": mu, "m2": m2, "n": n}
        elif k in left:
            merged[k] = dict(left[k])
        else:
            merged[k] = dict(right[k])
    return merged


# ─────────────────────────────────────────────────────────────────────────
# Public: NAc state-dict merge
# ─────────────────────────────────────────────────────────────────────────


def nac_merge(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    left_source: str,
    right_source: str,
    max_reward_bias: float = 0.20,
    max_cluster_reward_bias: float = 1.0,
    trusted_sources: frozenset[str] | None = None,
    validate_link: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    """Aggregate two ``NAc.dump()``-shape state dicts.

    Returns a fresh dict that loads cleanly into a live NAc via
    ``NAc.load_state(...)``. Inputs are not mutated. See module
    docstring for the per-field aggregation rules.

    Parameters
    ----------
    left, right
        State dicts produced by ``NAc.dump()`` (or a pre-filtered
        shareable subset — the bundle composer may strip non-shareable
        fields; this function tolerates absence by skipping those keys).
    left_source, right_source
        Opaque contributor IDs. Used to populate ``contributors`` on
        merged links when the per-link contributor list is empty
        (solo-source side). Required keyword-only. Must NOT start with
        the reserved ``_`` prefix (would collide with sentinel markers
        like :data:`CONSENSUS_SOURCE`).
    max_reward_bias
        Upper clamp magnitude for ``reward_bias`` ``[0, max]``,
        ``goal_reward_bias`` ``[-max, +max]`` (signed). Default 0.20
        matches ``NACConfig.max_reward_bias``.
    max_cluster_reward_bias
        Upper clamp magnitude for ``cluster_reward_bias``
        ``[-max, +max]`` (signed; Wire-A primary action-selection
        signal). Default 1.0 matches ``NACConfig.max_cluster_reward_bias``.
        Separate from ``max_reward_bias`` because Wire-A intentionally
        gives cluster bias wider range than per-node reward bias.
    trusted_sources
        Reservation for 1.2 poison-resistance (per
        ``maxim_hivemind.md`` "Poison resistance"). When non-None, only
        links whose entire ``contributors`` set is a subset of
        ``trusted_sources`` are admitted from the right-hand input.
        Default ``None`` — no filter. Reserved as a kwarg now so 1.2
        can wire callers without breaking the merge signature.
    validate_link
        Reservation for 1.2 poison-resistance. When non-None, each
        candidate right-side link is passed to the callable; a falsy
        return value causes the link to be skipped. Default ``None``.

    ``percept_valences`` are clamped to ``[-1.0, 1.0]`` independently
    of either ``max_*`` parameter.
    """
    _validate_source(left_source, label="left_source")
    _validate_source(right_source, label="right_source")

    # Links — merge per event_signature. Sorted iteration so the output
    # dict has deterministic insertion order (NICE review finding for
    # PR D bundle hashing).
    left_links = left.get("links", {}) or {}
    right_links = right.get("links", {}) or {}
    event_keys = sorted(set(left_links.keys()) | set(right_links.keys()))
    merged_links: dict[str, list[dict[str, Any]]] = {}
    for evt_sig in event_keys:
        filtered_right = list(right_links.get(evt_sig, []))
        if trusted_sources is not None:
            filtered_right = [
                ld for ld in filtered_right if set(_link_contributors(ld, right_source)).issubset(trusted_sources)
            ]
        if validate_link is not None:
            filtered_right = [ld for ld in filtered_right if validate_link(ld)]
        merged_links[evt_sig] = _merge_link_lists(
            left_links.get(evt_sig, []),
            filtered_right,
            left_source=left_source,
            right_source=right_source,
        )

    # outcome_index — union of values per key. Sorted for determinism.
    left_idx = left.get("outcome_index", {}) or {}
    right_idx = right.get("outcome_index", {}) or {}
    merged_outcome_index: dict[str, list[str]] = {}
    for k in sorted(set(left_idx.keys()) | set(right_idx.keys())):
        merged_outcome_index[k] = sorted(set(left_idx.get(k, [])) | set(right_idx.get(k, [])))

    # priors — shared keys: keep higher-confidence. Sorted for determinism.
    left_priors = left.get("priors", {}) or {}
    right_priors = right.get("priors", {}) or {}
    merged_priors: dict[str, Any] = {}
    for k in sorted(set(left_priors.keys()) | set(right_priors.keys())):
        if k in left_priors and k in right_priors:
            lp = left_priors[k]
            rp = right_priors[k]
            # Tuples were JSON-encoded as lists; either is fine.
            l_conf = float(lp[1]) if isinstance(lp, list | tuple) and len(lp) >= 2 else 0.0
            r_conf = float(rp[1]) if isinstance(rp, list | tuple) and len(rp) >= 2 else 0.0
            merged_priors[k] = lp if l_conf >= r_conf else rp
        else:
            merged_priors[k] = left_priors.get(k, right_priors.get(k))

    return {
        "version": "1.0",  # legacy payload string — tombstoned, do not bump
        "links": merged_links,
        "outcome_index": merged_outcome_index,
        "priors": merged_priors,
        "total_observations": int(left.get("total_observations", 0)) + int(right.get("total_observations", 0)),
        "reward_bias": _merge_mean_clamped(
            left.get("reward_bias", {}) or {},
            right.get("reward_bias", {}) or {},
            lo=0.0,
            hi=max_reward_bias,
        ),
        # CRITICAL fold: goal_reward_bias is SIGNED (negative = no-go
        # signal for ThoughtGate). Pre-fold the unsigned clamp silently
        # clipped every negative value to 0.
        "goal_reward_bias": _merge_mean_clamped(
            left.get("goal_reward_bias", {}) or {},
            right.get("goal_reward_bias", {}) or {},
            lo=-max_reward_bias,
            hi=max_reward_bias,
        ),
        # CRITICAL fold: cluster_reward_bias uses NACConfig's separate
        # ``max_cluster_reward_bias`` (default 1.0) — 5× wider than
        # ``max_reward_bias`` by intentional Wire-A design. Pre-fold the
        # 0.20 cap silently clipped every primary-signal value > 0.20.
        "cluster_reward_bias": _merge_mean_clamped(
            left.get("cluster_reward_bias", {}) or {},
            right.get("cluster_reward_bias", {}) or {},
            lo=-max_cluster_reward_bias,
            hi=max_cluster_reward_bias,
        ),
        "percept_valences": _merge_mean_clamped(
            left.get("percept_valences", {}) or {},
            right.get("percept_valences", {}) or {},
            lo=-1.0,
            hi=1.0,
        ),
        # D43 step 6 — state the merge used to DELETE.
        #
        # `cluster_reward_source` was absent from this dict entirely, so
        # `NAc.load_state` reset it to {} and every merge wiped the RECEIVER's
        # own credit provenance — local data loss, a different failure class
        # from "the foreign value didn't land". Union with a one-way "mixed"
        # promotion on disagreement, matching what the bundle scrub already
        # does when two contributors disagree about why a bias exists.
        "cluster_reward_source": _merge_credit_sources(
            left.get("cluster_reward_source"), right.get("cluster_reward_source")
        ),
        # `saved_at` was dropped too, and only the CLI patched it back — so
        # every other caller silently lost the decay clock. Keep the LATER of
        # the two: decay is elapsed-time-based, and the younger state is the
        # one whose biases have decayed least.
        "saved_at": _later_saved_at(left.get("saved_at"), right.get("saved_at")),
        "event_outcome_welford": _merge_welford(
            left.get("event_outcome_welford", {}) or {},
            right.get("event_outcome_welford", {}) or {},
        ),
    }


# ─────────────────────────────────────────────────────────────────────────
# Public: EC substrate-nodes merge
# ─────────────────────────────────────────────────────────────────────────


# Default frozen-prototype modalities — MUST match
# ``ECConfig.frozen_centroid_modalities`` (pinned by
# ``test_hivemind_frozen_modalities_match_ec_default``).
# ``ec_merge`` does NOT update the centroid for nodes in these modalities
# (it only sums counts + unions contributors). This preserves the
# bio-fidelity invariant from the EC centroid-drift fix lesson:
# interoceptive embeddings track smooth drive drift and a running-mean
# centroid update across contributors would re-introduce the drift the
# frozen-modality contract was designed to prevent.
#
# ``"audio"`` was MISSING here until 2026-08-06 while ``ECConfig`` has
# carried it since the exteroception seam shipped — so a default-argument
# ``ec_merge`` running-mean-updated audio centroids ACROSS contributors,
# exactly what the local EC forbids for that modality. The docstring
# already claimed the two matched; they did not. The literal is
# duplicated rather than imported to keep this layer free of
# internal-module imports (see ``_cosine``), so a test pins the equality
# instead of the type system.
#: Separator inside NAc's composite dict keys — the ``(agent_id, cluster_id,
#: tool_signature)`` triple and the ``(agent_id, entity_class, failure_mode)``
#: pair are stored as ``\x1f``-joined strings. Defined HERE, in the lowest
#: module of the hivemind package, because ``bundle.py`` already imports from
#: ``merge.py`` — the reverse would cycle.
NAC_KEY_SEP = "\x1f"

DEFAULT_FROZEN_CENTROID_MODALITIES: frozenset[str] = frozenset({"interoception", "audio"})


# ─────────────────────────────────────────────────────────────────────────
# D43 — alignment-preserving EC merge
#
# `ec_merge` historically computed a right→left node alignment internally and
# returned only the merged node set, discarding the map. `nac_merge` then
# folded `cluster_reward_bias` on exact string keys — so a foreign want landed
# under a cluster id that is not a node in the receiver's EC, read out as
# exactly 0.0, and the merge reported success. See
# docs/plans/d43_merge_correctness.md.
#
# THE THRESHOLD IS PER MODALITY, and this is the half no plan document named.
# The 0.44 default is `ECConfig.pattern_complete_threshold`, tuned for
# paraphrase-mpnet TEXT. Sensor modalities cluster at
# `SensorEncoderConfig.pattern_threshold = 0.85`. Applied to `_sensor_embed`
# output, virtually EVERY interoception node pair clears 0.44 — so returning
# the id map without retuning would collapse all of the donor's interoception
# clusters onto whichever receiver node scores highest. That is a CONFIDENTLY
# WRONG map where today there is an honestly missing one: strictly worse.
# ─────────────────────────────────────────────────────────────────────────

#: Modalities whose nodes are produced by ``SensorEncoder`` rather than by a
#: text encoder, and therefore align at the sensor threshold. Anything absent
#: falls back to the ``cosine_threshold`` argument.
SENSOR_MODALITY_THRESHOLDS: dict[str, float] = {
    "interoception": 0.85,
    "audio": 0.85,
}


@dataclass(frozen=True)
class ECMergeResult:
    """Merged nodes PLUS the right→left alignment that produced them.

    ``id_map`` maps a donor (right) node id to the surviving id it folded
    into. A donor node inserted rather than folded maps to the id it was
    inserted under — which may be suffixed on collision — so every right-side
    id has an entry and a re-key never silently drops a bias.
    """

    nodes: dict[str, dict[str, Any]]
    id_map: dict[str, str]


def ec_merge_aligned(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    *,
    left_source: str,
    right_source: str,
    cosine_threshold: float = 0.44,
    modality_thresholds: dict[str, float] | None = None,
    frozen_centroid_modalities: frozenset[str] = DEFAULT_FROZEN_CENTROID_MODALITIES,
    trusted_sources: frozenset[str] | None = None,
    validate_node: Callable[[dict[str, Any]], bool] | None = None,
    strict_geometry: bool = False,
) -> ECMergeResult:
    """Aggregate two ``substrate_nodes``-shape dicts.

    Inputs match the ``substrate_nodes`` slice of ``EC.save()`` payload::

        {node_id: {"embedding": [...], "modality": "text",
                   "count": int, "source": str,
                   "domain": str | None}}

    Output emits BOTH ``"count"`` (on-disk key consumed by
    ``EC.load()``) and ``"member_count"`` (public alias surfaced by
    ``EntorhinalCortex.substrate_node_metadata``) populated with the
    same integer — see :func:`_normalize_node` for the rationale.

    For backward compatibility the function also accepts the
    pre-PR-A-fold ``"count"`` key OR the post-fold ``"member_count"``
    key as inputs.

    Matching rule: for each right-side node, scan left-side nodes of
    the same modality. If the best cosine similarity meets
    ``cosine_threshold``, merge into that left-side node; otherwise
    insert the right-side node as a new entry under its own ID. Merge
    uses ``count``-weighted centroid mean (design rule #3), sums
    counts, unions contributors, and resolves ``source`` via the same
    convention as NAc links.

    Pure function: inputs are not mutated; output is a fresh dict.

    ENCODER-PROVENANCE IS SEVERED AT MERGE (artifact stamping, 1.1
    item 7 — stated so 1.1 Oasis does not inherit it silently): this
    function consumes the bare nodes dict; the bundle's encode-time
    stamps live in ``manifest["encoder_provenance"]`` and do NOT ride
    through here. Consequence: a substrate that imports foreign nodes
    and later RE-EXPORTS will stamp them with the importing session's
    own ``EC._encoder_provenance`` — wrong provenance on right arrays,
    which is worse than the honest ``None``. Per-contributor
    provenance union is 1.2 P2P merge work (see the
    ``compose_bundle`` docstring's MERGE SEMANTICS pin); until then,
    consumers must treat a merged substrate's stamps as describing its
    LOCAL encoders only.

    Parameters
    ----------
    left, right
        Substrate-nodes dicts.
    left_source, right_source
        Opaque contributor IDs. Must NOT start with the reserved ``_``
        prefix. Required keyword-only.
    cosine_threshold
        Match threshold. Default 0.44 matches
        ``ECConfig.pattern_complete_threshold``.
    trusted_sources, validate_node
        Reserved for 1.2 poison resistance; same semantics as
        ``nac_merge``'s parameters of the same shape. Default
        ``None`` — no filter.
    """
    _validate_source(left_source, label="left_source")
    _validate_source(right_source, label="right_source")

    merged: dict[str, dict[str, Any]] = {}
    id_map: dict[str, str] = {}
    _thresholds = dict(SENSOR_MODALITY_THRESHOLDS)
    if modality_thresholds:
        _thresholds.update(modality_thresholds)
    # Seed merged with deep copies of left so iteration mutations don't
    # touch the caller's dict. Sorted iteration keeps the seed phase
    # deterministic for PR D bundle hashing.
    for nid in sorted(left.keys()):
        merged[nid] = copy.deepcopy(_normalize_node(left[nid], fallback_source=left_source))

    for nid_r in sorted(right.keys()):
        nd_r = right[nid_r]
        # Poison-resistance filters before any normalization work.
        if trusted_sources is not None:
            if not set(_node_contributors(nd_r, right_source)).issubset(trusted_sources):
                continue
        if validate_node is not None and not validate_node(nd_r):
            continue
        norm_r = _normalize_node(nd_r, fallback_source=right_source)
        modality_r = norm_r["modality"]
        emb_r = norm_r["embedding"]

        # Find best left-side node of the same modality.
        best_id: str | None = None
        best_sim = -1.0
        # Per-modality threshold (D43). A sensor modality must align at the
        # threshold its clusters were FORMED at, not at the text default.
        thresh_r = _thresholds.get(modality_r, cosine_threshold)
        geom_r = norm_r.get("geometry")
        for nid_l, nd_l in merged.items():
            if nd_l["modality"] != modality_r:
                continue
            # Gate 2 / D4: DIFFERENT ENCODING SPACE IS NOT LOW SIMILARITY.
            # `_cosine` already refuses a dimension mismatch, but a place code
            # adds sensor names while keeping dim=384 and the same "audio"
            # tag, so the length guard never fires and old- and new-geometry
            # nodes fold whenever the partial cosine clears the threshold.
            # Because `audio` is a frozen-centroid modality the centroid never
            # moves, so the corruption is invisible: only counts and
            # contributors inflate. Two nodes that BOTH declare a geometry and
            # declare DIFFERENT ones are incomparable and never fold.
            #
            # An absent tag is unverifiable, not a match — legacy nodes
            # predate the stamp, so they are allowed through (the file still
            # loads) exactly as gate 7 admits an undeclared `body_ref`.
            # `strict_geometry=True` refuses those too.
            geom_l = nd_l.get("geometry")
            if geom_l is not None and geom_r is not None and geom_l != geom_r:
                continue
            if strict_geometry and (geom_l is None or geom_r is None):
                continue
            sim = _cosine(nd_l["embedding"], emb_r)
            if sim >= thresh_r and sim > best_sim:
                best_sim = sim
                best_id = nid_l

        if best_id is None:
            # No match — insert as a new node under its own id. Keep
            # the (already-deep-copied via _normalize_node) dict.
            if nid_r in merged:
                # ID collision across contributors that didn't pattern-
                # match — append a suffix so we don't overwrite.
                suffix = 1
                candidate = f"{nid_r}#{right_source}"
                while candidate in merged:
                    suffix += 1
                    candidate = f"{nid_r}#{right_source}#{suffix}"
                merged[candidate] = norm_r
                id_map[nid_r] = candidate
            else:
                merged[nid_r] = norm_r
                id_map[nid_r] = nid_r
            continue

        # Match — accumulate counts + contributors.
        id_map[nid_r] = best_id
        target = merged[best_id]
        n_l = int(target.get("count", target.get("member_count", 1)))
        n_r = int(norm_r.get("count", norm_r.get("member_count", 1)))
        total_n = n_l + n_r

        # Bio-fidelity fold (Bio-fidelity lens IMPORTANT): for modalities
        # in ``frozen_centroid_modalities`` (default: ``"interoception"``)
        # the centroid is FROZEN — running-mean updates across
        # contributors would re-introduce the centroid drift the
        # frozen-modality contract prevents within a single substrate
        # (see CLAUDE.md "behavioral EC centroid drift" lesson). Keep
        # left's embedding; only counts + contributors accumulate.
        if modality_r in frozen_centroid_modalities:
            pass  # centroid unchanged
        else:
            target["embedding"] = [
                (n_l * float(el) + n_r * float(er)) / total_n for el, er in zip(target["embedding"], emb_r)
            ]

        # Update both keys in lockstep — ``count`` is the on-disk shape
        # ``EC.save()``/``EC.load()`` consume; ``member_count`` is the
        # public ``substrate_node_metadata`` alias (post PR A fold).
        target["count"] = total_n
        target["member_count"] = total_n
        contribs = _union_contributors(
            _node_contributors(target, left_source),
            _node_contributors(norm_r, right_source),
        )
        target["contributors"] = list(contribs)
        target["source"] = _resolved_source(contribs)
        target["domain"] = target.get("domain") or norm_r.get("domain")
        # Same expiry rule as the live path: an unstamped survivor that
        # absorbs a stamped donor adopts its geometry. Without this an
        # unstamped node is a permanent BRIDGE — two donors the direct gate
        # refuses each fold into it and are thereby united through it, and it
        # stays unstamped to do the same on every later merge.
        if target.get("geometry") is None and norm_r.get("geometry") is not None:
            target["geometry"] = norm_r["geometry"]

    return ECMergeResult(nodes=merged, id_map=id_map)


def nac_merge_many(
    states: "list[dict[str, Any]]",
    *,
    sources: "list[str]",
    **kwargs: Any,
) -> dict[str, Any]:
    """Fold N NAc states with EQUAL weight — the N→1 semantics, decided.

    D43 step 5. The semantics were not undecided; they were decided wrong. The
    only shipped N>2 fold is a pairwise left-fold
    (``scripts/orient_substrate/5_operant_creche_federation.py::merge_creche``
    and its twin), and ``_merge_mean_clamped`` is an **unweighted** mean of
    shared keys, which is **not associative**. Folding four contributors
    pairwise gives them weights **1/8, 1/8, 1/4, 1/2** — the last contributor
    takes half the pooled bias. "Most-recent contributor dominates,
    exponentially" is not a federation.

    This folds the bias dicts in ONE pass so every contributor gets 1/N, and
    delegates everything else — links, welford, outcome_index — to the pairwise
    :func:`nac_merge`, which is already order-independent for those fields
    (observation-weighted means and a true parallel Welford).

    Note the internal inconsistency this resolves: within a single
    ``nac_merge`` call, ``total_observations`` sums exactly and
    ``event_outcome_welford`` uses Chan's parallel algorithm, while only the
    four bias dicts used the order-dependent unweighted mean.
    """
    if not states:
        raise ValueError("nac_merge_many requires at least one state")
    if len(states) != len(sources):
        raise ValueError(f"states/sources length mismatch: {len(states)} vs {len(sources)}")
    if len(states) == 1:
        return dict(states[0])

    # Pairwise for the order-independent fields.
    merged = states[0]
    for state, source in zip(states[1:], sources[1:], strict=True):
        merged = nac_merge(merged, state, left_source=sources[0], right_source=source, **kwargs)

    # Then overwrite the bias dicts with a true equal-weight N-way mean.
    for field, lo, hi in (
        ("reward_bias", 0.0, None),
        ("goal_reward_bias", None, None),
        ("cluster_reward_bias", -1.0, 1.0),
        ("percept_valences", -1.0, 1.0),
    ):
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for state in states:
            src = state.get(field)
            if not isinstance(src, dict):
                continue
            for key, value in src.items():
                try:
                    v = float(value)
                except (TypeError, ValueError):
                    continue
                totals[str(key)] = totals.get(str(key), 0.0) + v
                counts[str(key)] = counts.get(str(key), 0) + 1
        folded: dict[str, float] = {}
        for key, total in totals.items():
            # Zero-prior rule, unchanged from _merge_mean_clamped: a key absent
            # from a contributor is NO EVIDENCE, not a zero vote. Divide by the
            # number of contributors that HELD the key, not by N.
            v = total / counts[key]
            if lo is not None:
                v = max(lo, v)
            if hi is not None:
                v = min(hi, v)
            folded[key] = v
        if folded or field in merged:
            merged[field] = folded
    return merged


def _merge_credit_sources(left: Any, right: Any) -> dict[str, str]:
    """Union two ``cluster_reward_source`` maps; disagreement promotes to "mixed".

    One-way, matching ``NAc._note_cluster_reward_source``: once a triple has
    accumulated credit from two different sources, "mixed" is the honest label
    and it never demotes back.
    """
    out: dict[str, str] = {}
    for src in (left, right):
        if not isinstance(src, dict):
            continue
        for key, value in src.items():
            k, v = str(key), str(value)
            if k in out and out[k] != v:
                out[k] = "mixed"
            else:
                out.setdefault(k, v)
    return out


def _later_saved_at(left: Any, right: Any) -> Any:
    """The younger of two ISO timestamps — the state whose biases decayed least."""
    candidates = [t for t in (left, right) if isinstance(t, str) and t]
    if not candidates:
        return left if left is not None else right
    return max(candidates)


def rekey_nac_state(
    nac_state: dict[str, Any],
    id_map: dict[str, str],
    *,
    to_agent_id: str | None = None,
) -> dict[str, Any]:
    """Rewrite a donor NAc state's cluster ids (and optionally its agent id).

    D43 step 3 + 4. `cluster_reward_bias` and `cluster_reward_source` are keyed
    on the triple ``(agent_id, cluster_id, tool_signature)``. Both elements that
    can miss are rewritten here, BEFORE :func:`nac_merge` runs, so the fold sees
    keys that can actually match:

    * **cluster_id** through ``id_map`` — the alignment
      :func:`ec_merge_aligned` computes and used to discard.
    * **agent_id** to ``to_agent_id`` when given. This is deliberately a
      normalisation at the INGESTION BOUNDARY rather than a change to the key
      shape. The agent field exists for per-agent stash discipline *within* one
      substrate; across a merge the contributor identity already lives in the
      bundle manifest's ``contributor_id`` and in each node's ``contributors``
      list, so rewriting it here closes the second axis **without touching a
      single persisted file** — and it fixes ``get_agent_tool_biases``'s
      ``if aid != agent_id: continue`` filter for free.

    A cluster id absent from ``id_map`` is **dropped, not passed through**. A
    donor bias whose cluster did not survive the merge names a node the
    receiver cannot reach; keeping it is what made the union grow while
    contributing exactly 0.0. Dropping is the honest fold.

    Returns a new dict; the input is not mutated.
    """
    out = dict(nac_state)
    for field in ("cluster_reward_bias", "cluster_reward_source"):
        src = nac_state.get(field)
        if not isinstance(src, dict):
            continue
        rekeyed: dict[str, Any] = {}
        for key, value in src.items():
            parts = str(key).split(NAC_KEY_SEP)
            if len(parts) != 3:
                continue
            aid, cid, tsig = parts
            mapped = id_map.get(cid)
            if mapped is None:
                continue  # the donor cluster did not survive — drop, don't fake
            rekeyed[NAC_KEY_SEP.join((to_agent_id or aid, mapped, tsig))] = value
        out[field] = rekeyed
    return out


def ec_merge(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    *,
    left_source: str,
    right_source: str,
    cosine_threshold: float = 0.44,
    modality_thresholds: dict[str, float] | None = None,
    frozen_centroid_modalities: frozenset[str] = DEFAULT_FROZEN_CENTROID_MODALITIES,
    trusted_sources: frozenset[str] | None = None,
    validate_node: Callable[[dict[str, Any]], bool] | None = None,
    strict_geometry: bool = False,
) -> dict[str, dict[str, Any]]:
    """Merged nodes only — the pre-D43 contract, unchanged.

    Kept because a dozen call sites want the nodes and nothing else. **Anything
    that will re-key NAc state must call :func:`ec_merge_aligned` instead** and
    use its ``id_map``; folding bias keys after a node-only merge is exactly
    the D43 defect.
    """
    return ec_merge_aligned(
        left,
        right,
        left_source=left_source,
        right_source=right_source,
        cosine_threshold=cosine_threshold,
        modality_thresholds=modality_thresholds,
        frozen_centroid_modalities=frozen_centroid_modalities,
        trusted_sources=trusted_sources,
        validate_node=validate_node,
        strict_geometry=strict_geometry,
    ).nodes


@dataclass(frozen=True)
class SubstrateMergeResult:
    """What a full substrate merge produced, plus enough to audit it.

    Runtime-ephemeral (a function return, never persisted and never on a
    wire), so CC3's forward-compat clause does not apply — same as
    :class:`ECMergeResult`.

    ``biases_rekeyed`` / ``biases_dropped`` exist because D43's defining
    property was that the merge reported success while contributing nothing.
    A caller that prints ``len(cluster_reward_bias)`` is printing
    ``|left union right|``, which is **maximal exactly when nothing aligns**.
    These two counts are the honest indicator: a merge that dropped every
    donor bias says so.
    """

    nac: dict[str, Any]
    ec_nodes: dict[str, dict[str, Any]]
    id_map: dict[str, str]
    biases_rekeyed: int
    biases_dropped: int


def substrate_merge(
    *,
    receiver_nac: dict[str, Any],
    receiver_ec: dict[str, dict[str, Any]],
    donor_nac: dict[str, Any],
    donor_ec: dict[str, dict[str, Any]],
    receiver_source: str,
    donor_source: str,
    receiver_agent_id: str | None = None,
    **kwargs: Any,
) -> SubstrateMergeResult:
    """Merge a donor substrate INTO a receiver — the whole aligned sequence.

    **This is the function a consumer should call.** D43 shipped the three
    pieces — :func:`ec_merge_aligned`, :func:`rekey_nac_state`,
    :func:`nac_merge` — and left the composition to call sites, of which
    there were then zero: every shipped consumer still called bare
    ``nac_merge``, so the fix was a capability nobody used and a merged want
    still read out as 0.0 through every real path.

    Order is load-bearing and is the reason this exists as one function
    rather than a documented recipe:

    1. **Align first.** ``ec_merge_aligned`` folds the donor's nodes into the
       receiver's and reports which donor id became which surviving id.
    2. **Re-key second, on the donor only.** The receiver's own bias keys
       already name its own clusters; rewriting them would be the bug in
       mirror image.
    3. **Fold last.** ``nac_merge`` now sees keys that can match.

    Doing (3) before (1) is exactly D43. Doing (2) to the receiver corrupts
    state that was correct.

    ``receiver_agent_id`` normalises the donor's ``agent_id`` at this
    ingestion boundary — see :func:`rekey_nac_state` for why that closes the
    second key axis without migrating a single persisted file. Leave it
    ``None`` only when donor and receiver genuinely share an agent id.

    Extra keyword arguments are forwarded to BOTH underlying merges; each
    ignores the ones it does not take, so ``trusted_sources`` reaches both
    while ``max_cluster_reward_bias`` reaches only the NAc fold.
    """
    ec_keys = {
        "cosine_threshold",
        "modality_thresholds",
        "frozen_centroid_modalities",
        "trusted_sources",
        "validate_node",
        "strict_geometry",
    }
    nac_keys = {
        "max_reward_bias",
        "max_cluster_reward_bias",
        "trusted_sources",
        "validate_link",
    }
    unknown = set(kwargs) - ec_keys - nac_keys
    if unknown:
        raise TypeError(f"substrate_merge got unexpected keyword arguments: {sorted(unknown)}")

    aligned = ec_merge_aligned(
        receiver_ec,
        donor_ec,
        left_source=receiver_source,
        right_source=donor_source,
        **{k: v for k, v in kwargs.items() if k in ec_keys},
    )

    def _bias_count(state: dict[str, Any]) -> int:
        biases = state.get("cluster_reward_bias")
        return len(biases) if isinstance(biases, dict) else 0

    before = _bias_count(donor_nac)
    rekeyed_donor = rekey_nac_state(donor_nac, aligned.id_map, to_agent_id=receiver_agent_id)
    after = _bias_count(rekeyed_donor)

    merged_nac = nac_merge(
        receiver_nac,
        rekeyed_donor,
        left_source=receiver_source,
        right_source=donor_source,
        **{k: v for k, v in kwargs.items() if k in nac_keys},
    )

    return SubstrateMergeResult(
        nac=merged_nac,
        ec_nodes=aligned.nodes,
        id_map=aligned.id_map,
        biases_rekeyed=after,
        biases_dropped=before - after,
    )


def _normalize_node(node: dict[str, Any], *, fallback_source: str) -> dict[str, Any]:
    """Return a deep-copied node dict with canonical keys.

    Output shape matches ``EC.save()``'s on-disk per-node dict so a
    merged dict can be reloaded via ``EC.load()`` without an additional
    key-rename step:

    - ``count`` — the on-disk key used by both ``EC.save()`` and
      ``EC.load()`` (``similarity/ec.py``). Public callers reading
      from ``substrate_node_metadata`` see this same value under the
      ``member_count`` alias — both are populated.
    - ``member_count`` — the alias the PR A architecture-fold
      established for the public ``substrate_node_metadata`` API.

    Both keys are emitted with the same integer value so a downstream
    consumer that reads either gets the same answer. The input accepts
    either key (``count`` legacy, ``member_count`` post-fold) and the
    merger handles both transparently — but the output is
    fully-redundant by design to avoid silent round-trip key drift.
    """
    count_val = int(node.get("count", node.get("member_count", 1)))
    out: dict[str, Any] = {
        "embedding": list(node.get("embedding", [])),
        "modality": str(node.get("modality", "")),
        "count": count_val,
        "member_count": count_val,
        "source": str(node.get("source", fallback_source)),
        "domain": node.get("domain"),
        # Gate 2 / D4. Carried through the merge because it is what makes two
        # embeddings comparable at all; dropping it here would make every
        # merged node look unstamped and silently disable the geometry gate.
        "geometry": node.get("geometry"),
        "contributors": list(node.get("contributors", ())),
    }
    return out


def _node_contributors(node: dict[str, Any], fallback_source: str) -> tuple[str, ...]:
    """Same shape as :func:`_link_contributors` for EC nodes."""
    contribs = tuple(node.get("contributors", ()))
    if contribs:
        return contribs
    return (str(node.get("source", fallback_source)),)


__all__ = [
    "CONSENSUS_SOURCE",
    "ECMergeResult",
    "SubstrateMergeResult",
    "substrate_merge",
    "ec_merge_aligned",
    "rekey_nac_state",
    "nac_merge_many",
    "NAC_KEY_SEP",
    "SENSOR_MODALITY_THRESHOLDS",
    "ec_merge",
    "nac_merge",
]
