"""R1 — cross-layout generalization: an OFFLINE STRUCTURAL NULL.

The Minecraft survival ladder's R1 rung (docs/plans/minecraft_benchmark.md
Part II) asks whether Exp 56's shared want carried a generalizable REPRESENTATION
or a cached ASSOCIATION: does a want taught at layout S1 fire when the receiver
meets the contingency at a layout it never trained on?

The answer is knowable from the substrate mechanism, WITHOUT a live campaign
(the R2 precedent): the cluster-keyed learned-bias readout is an EXACT-key dict
lookup —

    NAc.cluster_reward_bias -> self._cluster_reward_bias.get(
        (agent_id, cluster_id, tool_signature), 0.0)

— with NO similarity/neighbour consultation, and a world layout maps to a
cluster id by a 0.85 cosine threshold. So a layout distinct enough to BE a
different situation (cos < 0.85 -> a different cluster id) necessarily MISSES the
key the want was taught on -> learned_bias 0 -> no transfer. The only layout that
fires is one that collapses to S1's own cluster (cos >= 0.85), which is not a
different layout at all. "Generalizes" and "is a genuinely different layout" are
the same 0.85 comparison with opposite sign; they cannot both hold. A live
cross-layout campaign would only re-confirm this.

This probe demonstrates it END-TO-END through the REAL ingest + recommend_action
path (never asserting on dict internals — the D44 rule): a want taught at S1,
ingested into a fresh receiver, reads out learned-bias-decisive at S1 (control)
and at a same-cluster perturbation (vacuous), but COLLAPSES at a genuinely
distinct layout. Verdict `CACHE-CONFIRMED` = fires at S1, collapses at the
distinct layout. A null ships as a null (Exp 53 / R2 shape).

    export PYTHONPATH="$PWD/src"   # if running from a worktree
    python scripts/r1_cross_layout_probe.py --write-experiment-results
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _provenance import evidence_out_paths, executed_code_provenance  # noqa: E402

from exp56 import common as C  # noqa: E402
from exp57 import common57 as X  # noqa: E402

# Two GENUINELY distinct layouts + one vacuous same-cluster perturbation. Chosen
# so the encoder assigns S1 and S_DISTINCT different world clusters (verified in
# check 0 below) — the interesting case R1 exists to probe. S_SAME (a 1-block
# nudge) stays in S1's cluster, illustrating the only way "transfer" occurs.
S1 = {"x": 88, "y": 112, "z": 0}
S_DISTINCT = {"x": 10, "y": 64, "z": 0}
S_SAME = {"x": 88, "y": 112, "z": 1}
TARGET_AFF = "aff_c"
TARGET_TOOL = f"{C.ENTITY_NAME}_{TARGET_AFF}"
DONOR_AGENT = "r1-donor"
RECV_AGENT = "r1-recv"
UNIFORM_BASELINE = 0.1  # small S1-cluster bias on every non-target tool (the
#: "all tools score so learned_margin is defined" role the balanced schedule's
#: signal plays in the live apparatus — see NOTE in _donor_snapshot).


def _ec_for(nodes: dict) -> object:
    from maxim.similarity.ec import EntorhinalCortex

    ec = EntorhinalCortex()
    ec.ingest_substrate_nodes(dict(nodes))
    return ec


def _encode_slot(ec: object, slot: dict, agent_id: str) -> str:
    from maxim.similarity.encoder import SensorEncoder, SensorEncoderConfig

    ranges = X.world_ranges()
    enc = SensorEncoder(ec=ec, config=SensorEncoderConfig())
    sensors = {k: v for k, v in X.world_sensors_for_slot(slot).items() if k in ranges}
    return enc.encode_sensors(agent_id=agent_id, sensors=sensors, modality="world", ranges=ranges)


def _donor_snapshot() -> tuple[dict, dict, str]:
    """A donor taught the want at S1: one world node at S1's reading + a
    cluster-keyed operant bias on the target there (0.8), plus a small uniform
    cluster-keyed bias on every OTHER ROSTER tool AT S1 (0.1).

    NOTE: the uniform baseline stands in for the live apparatus's balanced-
    schedule signal — its only role is to make every tool SCORE at S1 so
    ``learned_margin`` is defined (a runner-up exists) and the target is
    learned-bias-DECISIVE there. It is keyed on S1's cluster exactly like the
    taught want, so it too fails to fire at any distinct layout — which is the
    whole point: NOTHING keyed on S1's cluster reaches a different cluster. The
    learned-bias channel is the only situation-specific readout, and every part
    of it is an exact-key S1 lookup.
    """
    from maxim.similarity.ec import EntorhinalCortex

    ec = EntorhinalCortex()
    cid_s1 = _encode_slot(ec, S1, DONOR_AGENT)
    donor_ec = X.ec_substrate_nodes_dict(ec)
    crb: dict[str, float] = {}
    src: dict[str, str] = {}
    for aff in C.AFFORDANCES:
        tool = f"{C.ENTITY_NAME}_{aff}"
        key = f"{DONOR_AGENT}\x1f{cid_s1}\x1ftool:{tool}"
        crb[key] = 0.8 if tool == TARGET_TOOL else UNIFORM_BASELINE
        src[key] = "operant"
    nac = {
        "version": "1.0",
        "links": {},
        "outcome_index": {},
        "priors": {},
        "total_observations": 0,
        "reward_bias": {},
        "goal_reward_bias": {},
        "cluster_reward_bias": crb,
        "cluster_reward_source": src,
        "inherent_bias_keys": [],
        "percept_valences": {},
        "event_outcome_welford": {},
    }
    return nac, donor_ec, cid_s1


def _decisive_at(merged_nac: dict, merged_ec_nodes: dict, slot: dict) -> dict:
    """Read out learned-bias-decisiveness for the taught target at ``slot``,
    through the REAL consumer: encode the slot against the merged EC to get its
    cluster, then recommend_action at that cluster (bias-decisive from
    NAc_RECOMMEND provenance, never the emitted action)."""
    from maxim.decisions.nac import NAc, NACConfig

    cid = _encode_slot(_ec_for(merged_ec_nodes), slot, RECV_AGENT)
    nac = NAc(config=NACConfig())
    nac.load_state(dict(merged_nac))
    with C.RecommendCapture() as cap:
        nac.recommend_action(
            agent_id=RECV_AGENT,
            available_tools=list(C.ROSTER),
            current_drives={"d1": X.DRIVE_PROBE_LEVEL},
            current_clusters={"world": cid} if cid else None,
            min_confidence=C.FROZEN["min_confidence"],
        )
    prov = dict(cap.events[-1].get("data", {})) if cap.events else {}
    return {
        "cluster_id": cid,
        "bias_decisive": X.contingency_covered(prov, TARGET_TOOL),
        "best_tool": prov.get("best_tool"),
        "learned_bias": (prov.get("score_components") or {}).get("learned_bias"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/experiments/data/r1_cross_layout.json")
    ap.add_argument("--write-experiment-results", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args(argv)

    out_path = evidence_out_paths(
        REPO_ROOT, [args.out], write_experiment_results=args.write_experiment_results, allow_dirty=args.allow_dirty
    )[0]
    prov = executed_code_provenance(REPO_ROOT, "maxim", out_path=out_path, allow_dirty=args.allow_dirty)

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        donor_nac, donor_ec, cid_s1_donor = _donor_snapshot()
        merged = X.fold_snapshots([(donor_nac, donor_ec)], RECV_AGENT, workdir=Path(td), contributor_ids=["r1-donor"])
        merged_ec_nodes = json.loads((Path(td) / "recv" / "ec.json").read_text()).get("substrate_nodes", {})

        at_s1 = _decisive_at(merged, merged_ec_nodes, S1)
        at_distinct = _decisive_at(merged, merged_ec_nodes, S_DISTINCT)
        at_same = _decisive_at(merged, merged_ec_nodes, S_SAME)

    # check 0: the distinct layout really is a different cluster (else vacuous);
    # the same-perturbation layout really collapses to S1's cluster.
    layouts_distinct = at_distinct["cluster_id"] != at_s1["cluster_id"]
    same_is_same = at_same["cluster_id"] == at_s1["cluster_id"]

    cache_confirmed = (
        at_s1["bias_decisive"]  # control: fires at the trained layout
        and not at_distinct["bias_decisive"]  # collapses at a genuinely distinct layout
        and layouts_distinct  # and that layout was genuinely distinct (non-vacuous)
    )
    record = {
        "ts": time.time(),
        "rung": "R1",
        "body_ref": C.BODY_REF,
        "provenance": prov,
        "target_tool": TARGET_TOOL,
        "layouts": {"S1_trained": S1, "S_distinct": S_DISTINCT, "S_same_cluster": S_SAME},
        "readout": {"at_S1": at_s1, "at_S_distinct": at_distinct, "at_S_same_cluster": at_same},
        "layouts_distinct": layouts_distinct,
        "same_perturbation_stays_in_cluster": same_is_same,
        "cache_confirmed": cache_confirmed,
        "verdict": "CACHE-CONFIRMED" if cache_confirmed else "INCONCLUSIVE",
        "mechanism": (
            "cluster_reward_bias is an exact-key dict.get with no similarity channel; "
            "a distinct layout (cos<0.85) gets a distinct cluster id -> exact-key miss -> "
            "learned_bias 0. Generalization would require cos>=0.85, i.e. the same cluster."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2) + "\n")

    print(json.dumps({k: record[k] for k in ("rung", "verdict", "cache_confirmed", "layouts_distinct")}, indent=2))
    print(f"  at S1 (trained):        decisive={at_s1['bias_decisive']} cluster={at_s1['cluster_id'][:8]}")
    print(
        f"  at S_distinct (unseen): decisive={at_distinct['bias_decisive']} "
        f"cluster={at_distinct['cluster_id'][:8]} (distinct from S1: {layouts_distinct})"
    )
    print(
        f"  at S_same (perturbed):  decisive={at_same['bias_decisive']} "
        f"cluster={at_same['cluster_id'][:8]} (same as S1: {same_is_same})"
    )
    print(f"  record -> {out_path}")
    # A structural null is a valid result and exits 0; only an apparatus error is nonzero.
    return 0


if __name__ == "__main__":
    sys.exit(main())
