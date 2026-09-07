"""Exp 57 dose-response ladder — the science core (reuses Exp 56 wholesale).

Pre-registration:
``docs/experiments/protocols/exp57_dose_response_ladder_preregistration.md``
(FROZEN; this module implements it — the analyzer's constants are the verdict
authority). Everything sensor/teacher/export/ingest routes through
``scripts/exp56/common.py`` (imported as :data:`C`); this file adds only the
LADDER-specific assembly:

* the G = 4 contingencies (the four FROZEN Exp 56 slots), each with a
  per-cohort permutation mapping ``slot_index -> target affordance``;
* per-contributor INDEPENDENT seed AND presentation order (the coverage-
  widening requirement — byte-identical contributors force a flat curve);
* per-trial ``(NAc, EC)`` snapshots of a training contributor;
* :func:`fold_snapshots` — the LEFT-ASSOCIATIVE fold through the REAL 1.2
  ingest path (``export`` -> ``ingest`` sequentially into an accumulating
  receiver), which gives the last contributor weight 1/2 (NOT the equal-weight
  ``nac_merge_many`` 1/N, which is off the foreign path — see the fold weights
  comment on :func:`fold_snapshots`);
* :func:`coverage` — the DETERMINISTIC dependent measure, read from
  ``NAc_RECOMMEND`` provenance (a contingency is covered iff the LEARNED-BIAS
  component makes the taught affordance the argmax), not the epsilon-greedy
  emitted action;
* :func:`tau` — trials-to-criterion with the sliding-window criterion and the
  right-censoring sentinel.

FROZEN GATE constants (rungs, delta_eff, p, cohort count) live in
``scripts/analyze_exp57.py``; the APPARATUS constants (K_max, C, W) are set by
the Phase-0 amendment and passed in as parameters (design targets: C = 3/4,
W = 3, K_max so N = 1 sits ~1/2).
"""

from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

# Reuse the Exp 56 apparatus wholesale (the prereg's S1 reuse-by-reference).
from exp56 import common as C

# ── the ladder constants (carried from the frozen design) ────────────────

#: G distinct taught contingencies = the four FROZEN Exp 56 slots (prereg
#: §Apparatus: "G = 4, matching the 4 FROZEN slots"). A raised G would require
#: ADDING slots by amendment; it is not reopened here.
CONTINGENCY_SLOTS: list[dict[str, float]] = list(C.FROZEN["contingency_slots"])
G: int = len(CONTINGENCY_SLOTS)

#: Design-target apparatus constants (frozen by the Phase-0 amendment, NOT
#: here). Exposed as defaults so the harness runs; the confirmatory campaign
#: passes the Phase-0-set values explicitly.
CRITERION_TARGET: float = 3.0 / 4.0  # C = 3 of 4
WINDOW_TARGET: int = 3  # W

#: Right-censoring sentinel offset: tau = K_max + 1 when criterion is never
#: sustained within the per-agent budget (prereg §DV).
CENSOR_OFFSET: int = 1

#: The d1 value the coverage probe injects so the L12 zero-prior assertion is
#: NON-vacuous — above recommend_action's 0.5 drive activation floor
#: (decisions/nac.py). Bench d1 has no affinity/name-match so the drive
#: component stays 0 (coverage unchanged); a regression fires the assertion.
DRIVE_PROBE_LEVEL: float = 1.0


# ── per-cohort / per-contributor seeding (the coverage-widening fix) ──────


def cohort_slot_to_target(cohort_seed: int) -> dict[int, str]:
    """Map each of the G slots to a distinct taught affordance, per cohort.

    The per-cohort permutation makes the affordance alphabet non-load-bearing
    (prereg §Apparatus: "the per-cohort permutation maps slots->affordances so
    the alphabet is not load-bearing"). Deterministic in ``cohort_seed``.
    """
    rng = random.Random(cohort_seed * 2654435761 + 11)
    affs = list(C.AFFORDANCES)
    rng.shuffle(affs)
    return {g: affs[g] for g in range(G)}


def contributor_seeds(cohort_seed: int, rung: int, n: int, *, salt: int = 0) -> list[int]:
    """N INDEPENDENT per-contributor seeds for a (cohort, rung).

    Load-bearing (prereg §Apparatus / S5): each contributor draws an
    independent seed AND presentation order so contributors cover different
    contingencies first and the union can genuinely widen with N. Byte-
    identical contributors would force creche(N) == creche(1) and a flat
    curve. ``salt`` separates the creche / single_matched / creche_none draws
    within one (cohort, rung) so they never collide.
    """
    rng = random.Random((cohort_seed * 100003) ^ (rung * 917_951) ^ (salt * 2_246_822_519) ^ 3)
    return [rng.randrange(1, 1 << 30) for _ in range(n)]


# ── per-contributor schedule (independent order — the widening driver) ────


def contributor_schedule(
    contributor_seed: int,
    *,
    reps_per_cell: int,
    n_contingencies: int = G,
    min_trials: int | None = None,
) -> tuple[list[tuple[int, bool, str]], list[int]]:
    """A contingency-blocked balanced schedule with a PER-CONTRIBUTOR order.

    Each trial is ``(contingency_index, situation_active, affordance)``. The
    contingencies are visited in a per-contributor SHUFFLED order (the
    independent presentation order, §S5), and within each contingency block the
    balanced (situation-state x affordance) cells are shuffled too — both
    driven by ``contributor_seed``. The balanced-exposure STRUCTURE is
    identical across contributors (the link-neutralization); only the seed and
    order differ, so a contributor at a small per-agent budget has decisively
    covered only its FIRST few contingencies and different contributors cover
    different ones first -> the union widens with N.

    One balanced PASS is ``n_contingencies * 2 * |AFFORDANCES| * reps_per_cell``
    trials. ``single_matched`` needs up to ``N * K_max`` trials — more than one
    pass — so when ``min_trials`` is given the schedule REPEATS balanced passes
    (each pass re-shuffling the contingency order and every cell) until it is at
    least that long. Repeating preserves the balanced-exposure structure (each
    cell equally often per pass) and the independent per-contributor order; it
    is the honest way to give one agent "all N*K_max trials" (prereg
    §Conditions, single_matched) rather than silently truncating.

    Returns ``(trials, first_pass_contingency_order)``.
    """
    rng = random.Random(contributor_seed * 104729 + 7)
    first_order: list[int] | None = None
    trials: list[tuple[int, bool, str]] = []
    while True:
        order = list(range(n_contingencies))
        rng.shuffle(order)
        if first_order is None:
            first_order = order
        for g in order:
            block = [(g, state, aff) for state in (False, True) for aff in C.AFFORDANCES for _ in range(reps_per_cell)]
            rng.shuffle(block)
            trials.extend(block)
        if min_trials is None or len(trials) >= min_trials:
            break
    return trials, first_order


# ── per-trial snapshot capture ───────────────────────────────────────────


def ec_substrate_nodes_dict(ec: Any) -> dict[str, dict[str, Any]]:
    """Extract the ``substrate_nodes`` slice as ``EC.save()`` shapes it.

    EC has ``save()``/``load()`` but no ``dumps()``, so this replicates the
    exact per-node dict ``similarity/ec.py::save`` emits (embedding, modality,
    count, source, domain, geometry) — the shape ``ec.json`` carries and the
    ingest adapter consumes.
    """
    out: dict[str, dict[str, Any]] = {}
    for nid, (emb, mod) in ec._substrate_nodes.items():
        out[nid] = {
            "embedding": list(emb),
            "modality": mod,
            "count": ec._substrate_node_counts.get(nid, 1),
            "source": ec._substrate_node_sources.get(nid, "local"),
            "domain": ec._substrate_node_domains.get(nid),
            "geometry": ec._substrate_node_geometries.get(nid),
        }
    return out


def snapshot_session(session: "C.BenchSession") -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """One ``(NAc.dump(), ec_substrate_nodes)`` snapshot of a contributor."""
    return session.aut.bio.nac.dump(), ec_substrate_nodes_dict(session.aut.bio.ec)


def train_contributor_with_snapshots(
    session: "C.BenchSession",
    *,
    world: Any,
    contributor_seed: int,
    slot_to_target: dict[int, str],
    bot_name: str,
    k_max: int,
    reps_per_cell: int,
    settle_s: float = 0.6,
    teach: bool = True,
) -> list[tuple[dict[str, Any], dict[str, dict[str, Any]]]]:
    """Run a contributor's balanced schedule, snapshotting AFTER each trial.

    Mirrors ``C.run_donor_training`` (encode -> execute-and-record -> teacher)
    but (a) generalizes to the G-contingency world and (b) captures a
    ``(NAc, EC)`` snapshot after every trial so the B-phase can fold at any
    per-agent checkpoint t. Returns ``k_max`` snapshots (snapshot i is the
    substrate AFTER trial i+1).

    ``teach=False`` withholds the teacher entirely (the creche_none noise-floor
    construction): the balanced schedule still runs, so the link channel is
    exercised identically, but zero operant credit is minted.
    """
    trials, _order = contributor_schedule(contributor_seed, reps_per_cell=reps_per_cell, min_trials=k_max)
    if len(trials) < k_max:  # fail loud, never silently truncate (verify-the-instrument)
        raise ValueError(
            f"contributor_schedule produced {len(trials)} trials < requested k_max={k_max} "
            f"(reps_per_cell={reps_per_cell}); the schedule must cover N*K_max for single_matched"
        )
    snapshots: list[tuple[dict[str, Any], dict[str, dict[str, Any]]]] = []
    for idx, (g, situation, aff) in enumerate(trials[:k_max]):
        slot = CONTINGENCY_SLOTS[g]
        anchor = slot if situation else C.FROZEN["rest_anchor"]
        C.settle_until_reflected(
            session,
            world,
            bot_name,
            anchor,
            situation=situation,
            slot=slot,
            where=f"contributor trial {idx} (contingency {g})",
            timeout_s=max(5.0, settle_s * 8),
        )
        clusters = session.encode_clusters()
        tool = f"{C.ENTITY_NAME}_{aff}"
        session.execute_and_record(tool, clusters, reasoning="exp57 balanced schedule")
        if teach:
            C.teacher_tick(
                session,
                situation_active=situation,
                executed_aff=aff,
                target_aff=slot_to_target[g],
                arm="creche",
            )
        snapshots.append(snapshot_session(session))
    return snapshots


# ── the fold (LEFT-ASSOCIATIVE, through the REAL 1.2 ingest path) ─────────


def _write_receiver_home(receiver_home: Path) -> None:
    """Create a fresh empty receiver session dir the ingest CLI can read.

    The ingest CLI (``_resolve_receiver_pair``) needs an ``nac.json``/``ec.json``
    pair in the session dir. A fresh empty NAc dump + empty substrate_nodes is
    the cold-start receiver — no bridge required (the B-phase is offline).
    """
    from maxim.decisions.nac import NAc, _NAC_FORMAT_VERSION
    from maxim.utils.format_version import with_format_version

    receiver_home.mkdir(parents=True, exist_ok=True)
    (receiver_home / "nac.json").write_text(
        json.dumps(with_format_version(NAc().dump(), version=_NAC_FORMAT_VERSION), indent=2)
    )
    (receiver_home / "ec.json").write_text(json.dumps(with_format_version({"substrate_nodes": {}}), indent=2))


def _stage_snapshot(stage_dir: Path, nac_state: dict[str, Any], ec_nodes: dict[str, dict[str, Any]]) -> Path:
    """Write a snapshot to the ``aut_nac.json``/``aut_ec.json`` pair the export
    CLI reads."""
    from maxim.utils.format_version import with_format_version

    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "aut_nac.json").write_text(json.dumps(with_format_version(dict(nac_state)), indent=2))
    (stage_dir / "aut_ec.json").write_text(
        json.dumps(with_format_version({"substrate_nodes": dict(ec_nodes)}), indent=2)
    )
    return stage_dir


def fold_snapshots(
    snapshots: list[tuple[dict[str, Any], dict[str, dict[str, Any]]]],
    receiver_agent_id: str,
    *,
    workdir: Path,
    contributor_ids: "list[str] | None" = None,
) -> dict[str, Any]:
    """Fold N contributor snapshots into a fresh receiver, LEFT-ASSOCIATIVELY.

    The fold is the shipped foreign path: for each contributor, compose a
    bundle via ``C.export_bundle`` and ingest it into the ACCUMULATING receiver
    via ``C.ingest_bundle_into`` (which writes the merged state back, so the
    next ingest sees it):

        r = fresh; for c in contributors: r = ingest(c into r)

    Because ``_merge_mean_clamped`` averages PAIRWISE, three-plus contributors
    on one shared key get an ORDER-DEPENDENT convex combination — for three,
    weights (1/4, 1/4, 1/2); for four, (1/8, 1/8, 1/4, 1/2) — the last-folded
    contributor weighted 1/2, bounded above by the max contributor value
    (prereg §"The mechanism"). This is DELIBERATELY NOT ``nac_merge_many``'s
    equal-weight 1/N fold: that one is trusted-local-only, does no EC alignment,
    and is off the foreign ingest path. The ladder measures the shipped foreign
    semantics.

    ``single_matched`` is a 1->1 fold of ONE contributor (trained to N*K_max)
    through this SAME path — call with a one-element ``snapshots`` list.

    Returns the merged ``NAc.dump()``-shape state dict (read back from the
    receiver home). The merged ``ec.json`` is left in ``<workdir>/recv`` for
    the caller to derive contingency clusters from.
    """
    workdir = Path(workdir)
    if contributor_ids is None:
        contributor_ids = [f"contrib-{i}" for i in range(len(snapshots))]
    if len(contributor_ids) != len(snapshots):
        raise ValueError("contributor_ids length must match snapshots")

    receiver_home = workdir / "recv"
    shutil.rmtree(receiver_home, ignore_errors=True)
    _write_receiver_home(receiver_home)

    for i, (nac_state, ec_nodes) in enumerate(snapshots):
        stage = _stage_snapshot(workdir / f"stage_{i}", nac_state, ec_nodes)
        bundle = workdir / f"c{i}.zip"
        if bundle.exists():
            bundle.unlink()
        C.export_bundle(stage, bundle, contributor_id=contributor_ids[i])
        C.ingest_bundle_into(
            receiver_home,
            bundle,
            contributor_id=contributor_ids[i],
            receiver_agent_id=receiver_agent_id,
        )

    merged = json.loads((receiver_home / "nac.json").read_text())
    merged.pop("_format_version", None)
    return merged


# ── the dependent measure (bias-decisive coverage from provenance) ───────


def contingency_covered(provenance: dict[str, Any], taught_tool: str) -> bool:
    """The prereg DV, read DETERMINISTICALLY from ``NAc_RECOMMEND`` provenance.

    A contingency counts as covered iff the LEARNED-BIAS component makes the
    taught affordance the argmax — the Exp 56 ``bias_decisive`` property read as
    a property of the merged SUBSTRATE (the ``best_tool`` argmax from
    provenance), NOT the epsilon-greedy emitted action:

        best_tool == taught  AND  learned_bias > 0  AND  learned_margin > 0
    """
    best = provenance.get("best_tool")
    components = provenance.get("score_components") or {}
    learned = float(components.get("learned_bias", 0.0) or 0.0)
    margin = provenance.get("learned_margin")
    return best == taught_tool and learned > 0.0 and margin is not None and float(margin) > 0.0


def coverage(
    merged_nac: dict[str, Any],
    contingency_clusters: dict[int, dict[str, str]],
    slot_to_target: dict[int, str],
    *,
    receiver_agent_id: str,
    return_detail: bool = False,
) -> "float | tuple[float, dict[int, bool]]":
    """Coverage = (# contingencies covered) / G, read from provenance.

    For each of the G contingencies, load ``merged_nac`` into a BARE NAc
    (``NAc(config=NACConfig())``; ``load_state``), call ``recommend_action`` at
    that contingency's clusters over the ROSTER, capture the ``NAc_RECOMMEND``
    decision provenance, and count it covered iff the learned-bias component
    makes the taught affordance the argmax (:func:`contingency_covered`). Read
    from provenance, NOT the epsilon-greedy emitted action (prereg §DV).
    """
    from maxim.decisions.nac import NAc, NACConfig

    detail: dict[int, bool] = {}
    for g in range(G):
        clusters = contingency_clusters.get(g) or {}
        taught_tool = f"{C.ENTITY_NAME}_{slot_to_target[g]}"
        nac = NAc(config=NACConfig())
        nac.load_state(dict(merged_nac))
        with C.RecommendCapture() as cap:
            nac.recommend_action(
                agent_id=receiver_agent_id,
                available_tools=list(C.ROSTER),
                # L12 probe drive: d1 is set ABOVE the recommend_action 0.5
                # activation floor DELIBERATELY so the zero-prior assertion below
                # is NON-VACUOUS (methodology-lens review). With d1=0 the scorer
                # skips the drive term unconditionally, so asserting drive==0
                # would test the harness's own input, not the substrate. At
                # d1=DRIVE_PROBE_LEVEL the drive term FIRES iff d1 name-matches a
                # tool or has an affinity-table entry — bench d1 has neither, so
                # the component is 0 and coverage (a learned-bias argmax) is
                # unchanged, but a regression (d1 gaining an affinity, or a tool
                # named to contain "d1") now RAISES here instead of silently
                # corrupting the argmax the DV reads.
                current_drives={"d1": DRIVE_PROBE_LEVEL},
                current_clusters=clusters or None,
                min_confidence=C.FROZEN["min_confidence"],
            )
        provenance = dict(cap.events[-1].get("data", {})) if cap.events else {}
        # L12 zero-prior assertion (prereg Phase-0 check 3 / S3), the exp56
        # probe_receiver:911 rule made real HERE (exp57 does not call
        # probe_receiver): a nonzero drive COMPONENT at an active d1 would
        # corrupt the best_tool argmax the DV reads. Raise loudly rather than
        # silently miscount — a can't-fail check is not a check (the D62 lesson).
        _drive = float((provenance.get("score_components") or {}).get("drive", 0.0) or 0.0)
        if abs(_drive) > 1e-9:
            raise AssertionError(
                f"L12 drive-prior leak: score_components['drive']={_drive} != 0 at "
                f"d1={DRIVE_PROBE_LEVEL} (contingency {g}, taught={taught_tool}) — the "
                f"coverage DV is not drive-blind (d1 gained an affinity/name-match)"
            )
        # NOTE (methodology-lens): contingency_covered requires learned_margin
        # not None, which is None only when a single tool scores. The balanced
        # schedule executes all 8 ROSTER tools -> causal links transfer through
        # the fold -> every tool scores -> a runner-up exists -> margin defined.
        # If that ever fails, this reads CONSERVATIVELY as uncovered (undercounts;
        # cannot manufacture a pass). Phase-0 check 5 confirms margins on live data.
        detail[g] = contingency_covered(provenance, taught_tool)
    frac = sum(1 for v in detail.values() if v) / G
    if return_detail:
        return frac, detail
    return frac


# ── contingency clusters from a merged substrate (bridge-free encode) ────


def world_sensors_for_slot(slot: dict[str, float]) -> dict[str, float]:
    """A clean (un-jittered) world-sensor reading for a slot.

    Mirrors ``C.ScriptedBridgeServer._snapshot`` without the seeded jitter, so
    the B-phase can re-encode a contingency against the merged EC WITHOUT a live
    bridge and pattern-complete to the merged node the taught bias is keyed on.
    """
    dist = (slot["x"] ** 2 + slot["z"] ** 2) ** 0.5
    return {
        "y_altitude": float(slot["y"]),
        "distance_from_spawn": dist,
        "speed": 0.0,
        "on_ground": 1.0,
        "time_of_day": 0.25,
    }


_WORLD_RANGES_CACHE: "dict[str, tuple[float, float]] | None" = None


def world_ranges() -> dict[str, tuple[float, float]]:
    """Declared world-sensor ranges from the bench body (single source)."""
    global _WORLD_RANGES_CACHE
    if _WORLD_RANGES_CACHE is None:
        from maxim.embodiment.component_registry import ComponentRegistry

        entity = ComponentRegistry().instantiate(C.BODY_REF)
        names = [n for n, s in entity.sensors.items() if (s.reading_schema or {}).get("modality") == "world"]
        out: dict[str, tuple[float, float]] = {}
        for name in names:
            schema = getattr(entity.sensors[name], "reading_schema", {}) or {}
            rng = schema.get("range")
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                out[name] = (float(rng[0]), float(rng[1]))
        _WORLD_RANGES_CACHE = out
    return _WORLD_RANGES_CACHE


def contingency_clusters_from_ec_nodes(
    ec_nodes: dict[str, dict[str, Any]],
    *,
    receiver_agent_id: str,
) -> dict[int, dict[str, str]]:
    """Encode each contingency against the merged substrate -> cluster ids.

    Builds a throwaway EC from the merged nodes, wraps a production
    ``SensorEncoder``, and encodes each slot's clean world reading. A slot the
    merged substrate covers pattern-completes to its merged node (the id the
    taught bias is keyed on); an uncovered slot returns an empty cluster set
    (no bias -> not covered). This is the real consumer's encode path, run
    bridge-free and deterministically.
    """
    from maxim.similarity.ec import EntorhinalCortex
    from maxim.similarity.encoder import SensorEncoder, SensorEncoderConfig

    ec = EntorhinalCortex()
    ec.ingest_substrate_nodes(dict(ec_nodes))
    encoder = SensorEncoder(ec=ec, config=SensorEncoderConfig())
    ranges = world_ranges()
    out: dict[int, dict[str, str]] = {}
    for g, slot in enumerate(CONTINGENCY_SLOTS):
        sensors = {k: v for k, v in world_sensors_for_slot(slot).items() if k in ranges}
        cid = encoder.encode_sensors(agent_id=receiver_agent_id, sensors=sensors, modality="world", ranges=ranges)
        out[g] = {"world": cid} if cid else {}
    return out


# ── trials-to-criterion (sliding window + right-censoring) ────────────────


def tau(coverage_series: list[float], c_crit: float, w: int, k_max: int) -> int:
    """Smallest t such that the last W checkpoints all satisfy d >= C.

    ``coverage_series`` is the per-checkpoint coverage d(t) for t = 1..K_max
    (index 0 is trial 1). tau is the smallest 1-indexed t at which the sliding
    window of the last W checkpoints are ALL >= ``c_crit`` (sustained, so a
    single noisy checkpoint cannot trip it). If no such t <= K_max exists, tau
    is RIGHT-CENSORED at the sentinel ``k_max + CENSOR_OFFSET`` (prereg §DV).
    """
    for i in range(w - 1, len(coverage_series)):
        window = coverage_series[i - w + 1 : i + 1]
        if all(d >= c_crit for d in window):
            return i + 1  # 1-indexed trial number
    return k_max + CENSOR_OFFSET


def is_censored(tau_value: int, k_max: int) -> bool:
    """True when a tau value is the right-censoring sentinel."""
    return tau_value >= k_max + CENSOR_OFFSET


# ── the anti-vacuity kit (coverage-level; the D62 shape generalized) ─────


def noop_coverage_kit(
    *,
    snapshots: list[tuple[dict[str, Any], dict[str, dict[str, Any]]]],
    contributor_ids: list[str],
    receiver_agent_id: str,
    slot_to_target: dict[int, str],
    workdir: Path,
) -> dict[str, Any]:
    """Re-run one recorded fold under no-op merge variants and read coverage.

    The D62 kit generalized to coverage (a gate that cannot fail is not a gate):

    * ``receiver_unchanged`` — ``substrate_merge`` replaced by a variant that
      returns the receiver untouched. MUST collapse coverage to the floor
      (no donor state arrives).
    * ``empty_fold`` — merge returns empty state. MUST collapse to the floor.
    * ``rekeyed_alone`` — ONE contributor, re-keyed, no averaging partner;
      RECORDED and expected to persist (D62 / Exp 56 amendment 1: on a fresh
      receiver, donor-alone is equivalent to the real merge, so asserting
      collapse there would test a recipe, not the gate).

    ``kit_pass`` is the two must-collapse variants both collapsing to coverage
    ``0.0``.
    """
    from unittest import mock as _mock

    import maxim.hivemind.ingest as ingest_mod
    from maxim.hivemind.merge import (
        SubstrateMergeResult,
        ec_merge_aligned,
        rekey_nac_state,
    )

    workdir = Path(workdir)

    def _variant(kind: str):
        def fake(**kwargs: Any) -> SubstrateMergeResult:
            if kind == "receiver_unchanged":
                return SubstrateMergeResult(
                    nac=dict(kwargs["receiver_nac"]),
                    ec_nodes=dict(kwargs["receiver_ec"]),
                    id_map={},
                    biases_rekeyed=0,
                    biases_dropped=0,
                )
            if kind == "empty_fold":
                return SubstrateMergeResult(nac={}, ec_nodes={}, id_map={}, biases_rekeyed=0, biases_dropped=0)
            # rekeyed_alone: the donor re-keyed with no averaging partner.
            aligned = ec_merge_aligned(
                kwargs["receiver_ec"],
                kwargs["donor_ec"],
                left_source=kwargs["receiver_source"],
                right_source=kwargs["donor_source"],
                strict_geometry=True,
            )
            donor_only = rekey_nac_state(
                kwargs["donor_nac"], aligned.id_map, to_agent_id=kwargs.get("receiver_agent_id")
            )
            return SubstrateMergeResult(
                nac=donor_only,
                ec_nodes=aligned.nodes,
                id_map=aligned.id_map,
                biases_rekeyed=0,
                biases_dropped=0,
            )

        return fake

    out: dict[str, Any] = {}
    for kind in ("receiver_unchanged", "empty_fold", "rekeyed_alone"):
        variant_dir = workdir / f"noop_{kind}"
        shutil.rmtree(variant_dir, ignore_errors=True)
        with _mock.patch.object(ingest_mod, "substrate_merge", side_effect=_variant(kind)):
            merged = fold_snapshots(
                snapshots if kind != "rekeyed_alone" else snapshots[:1],
                receiver_agent_id,
                workdir=variant_dir,
                contributor_ids=(contributor_ids if kind != "rekeyed_alone" else contributor_ids[:1]),
            )
            clusters = contingency_clusters_from_ec_nodes(
                json.loads((variant_dir / "recv" / "ec.json").read_text()).get("substrate_nodes", {}),
                receiver_agent_id=receiver_agent_id,
            )
        cov = coverage(merged, clusters, slot_to_target, receiver_agent_id=receiver_agent_id)
        out[kind] = {"coverage": cov}
    out["kit_pass"] = (out["receiver_unchanged"]["coverage"] == 0.0) and (out["empty_fold"]["coverage"] == 0.0)
    out["ts"] = time.time()
    return out
