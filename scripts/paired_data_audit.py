"""Paired-data audit — the entry condition of the grounded-language + JEPA-projection line.

``docs/plans/roadmap_1_4.md`` §Parallel lines: before either plan revives, count the
(sensor vector, text percept) pairs the survival world actually yields — per run and per
situation — how many DISTINCT texts there are, how many are TEMPLATED game strings rather
than language, and what the sensor side looks like when a text arrives. The audit commits
to nothing: it reports numbers and sets no pass/fail gate (none was pre-registered).

Read-only over committed traces. A "paired" trace is a JSONL file carrying both world
``snapshot`` records and bridge ``event`` records (the shape
``scripts/l11_real_trace_remeasure.py capture`` writes). Each event is paired with the
snapshot written in the same capture tick (same ``ts``), else the latest earlier one.
The sensor side is replayed through the REAL ``SensorEncoder`` + ``EntorhinalCortex``
(world modality, declared ranges, A4 gain) — the consumer a projection would train on.

Known-answer check: a ``damage`` text restates ``bot.health`` at emit time, so if the
pairing is right the parsed number must match the paired snapshot's ``health`` — or the one
before it, since the text can carry the pre-hit value. A low match rate on the
paired-or-previous count means the pairing, not the corpus, is broken.

    python scripts/paired_data_audit.py                  # scan every committed trace
    python scripts/paired_data_audit.py --json out.json  # also write the numbers
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "docs" / "experiments" / "data"
NUM = re.compile(r"-?\d+(?:\.\d+)?")
WORD = re.compile(r"[a-z]+")
HEALTH_TOL = 1.0  # one heart-half: the damage event can precede the tick's snapshot by < cadence


def _load(path: Path) -> tuple[list[dict], list[dict]]:
    snaps, events = [], []
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            if rec.get("kind") == "snapshot" and isinstance(rec.get("state"), dict):
                snaps.append(rec)
            elif rec.get("kind") == "event" and isinstance(rec.get("text"), str) and "event_kind" in rec:
                events.append(rec)
    return snaps, events


def _template(texts: list[str]) -> tuple[list[str], Counter]:
    """Tokens shared by every text of a kind = the template; the rest are fillers."""
    token_lists = [NUM.sub("<N>", t.lower()).split() for t in texts]
    fixed = set(token_lists[0]).intersection(*token_lists[1:]) if token_lists else set()
    fillers: Counter = Counter()
    for toks in token_lists:
        fillers[" ".join(t for t in toks if t not in fixed)] += 1
    return [t for t in token_lists[0] if t in fixed], fillers


def _world_nodes(snaps: list[dict]) -> tuple[list[str], list[np.ndarray]]:
    """Replay every snapshot through the real world encoder.

    Returns the EC node id per snapshot (what the substrate keys on) and the unit-norm raw
    gained vector per snapshot (what a projection would train on).
    """
    sys.path.insert(0, str(REPO / "scripts"))
    from l11_real_trace_remeasure import _declared_world_ranges

    from maxim.similarity.ec import ECConfig, EntorhinalCortex
    from maxim.similarity.encoder import SensorEncoder, SensorEncoderConfig, _sensor_embed

    ranges = _declared_world_ranges()
    config = SensorEncoderConfig()
    encoder = SensorEncoder(ec=EntorhinalCortex(ECConfig()), config=config)
    ids, vecs, last = [], [], "rest"
    for rec in snaps:
        state = {k: float(v) for k, v in rec["state"].items() if k in ranges}
        node = encoder.encode_sensors(agent_id="audit", sensors=state, modality="world", ranges=ranges)
        # None = the gained body rests at neutral (designed: "a body at rest encodes nothing").
        if node is not None:
            last = node
        elif encoder.last_encode_was_designed_rest(agent_id="audit", modality="world"):
            last = "rest"
        ids.append(last)
        v = np.asarray(
            _sensor_embed(state, ranges=ranges, dim=config.embedding_dim, gain_exponent=config.gain_exponent)
        )
        norm = float(np.linalg.norm(v))
        vecs.append(v / norm if norm else v)
    return ids, vecs


def _kind_separability(pairs: list[tuple[str, np.ndarray]]) -> dict:
    """Leave-one-out nearest-centroid: can the raw sensor vector at the event name its kind?

    Compared against always guessing the majority kind. Identity check: a unit vector
    against itself must score cosine 1.0, else the vectors are not what we think.
    """
    kinds = sorted({k for k, _ in pairs})
    sums = {k: sum(v for kk, v in pairs if kk == k) for k in kinds}
    counts = Counter(k for k, _ in pairs)
    correct = 0
    for k, v in pairs:
        best, best_cos = None, -2.0
        for kk in kinds:
            n = counts[kk] - (kk == k)
            if n == 0:
                continue
            c = (sums[kk] - (v if kk == k else 0)) / n
            cos = float(v @ c / (np.linalg.norm(c) or 1.0))
            if cos > best_cos:
                best, best_cos = kk, cos
        correct += best == k
    return {
        "identity_cosine": round(float(pairs[0][1] @ pairs[0][1]), 6) if pairs else None,
        "loo_nearest_centroid_correct": correct,
        "majority_baseline_correct": max(counts.values()) if counts else 0,
        "n": len(pairs),
        "caveat": "single snapshot at the event, not the before->after transition",
    }


def audit_trace(path: Path) -> dict:
    snaps, events = _load(path)
    nodes, vecs = _world_nodes(snaps)
    kind_vecs: list[tuple[str, np.ndarray]] = []
    by_ts = {s["ts"]: i for i, s in enumerate(snaps)}
    kinds: dict[str, list[str]] = defaultdict(list)
    pair_nodes: dict[str, Counter] = defaultdict(Counter)
    node_kinds: dict[str, set] = defaultdict(set)
    health_checked = health_match = health_match_or_previous = unpaired = 0
    for ev in events:
        kind = ev["event_kind"]
        kinds[kind].append(ev["text"])
        i = by_ts.get(ev["ts"])
        if i is None:
            earlier = [j for j, s in enumerate(snaps) if s["ts"] <= ev["ts"]]
            i = earlier[-1] if earlier else None
        if i is None:
            unpaired += 1
            continue
        pair_nodes[kind][nodes[i]] += 1
        kind_vecs.append((kind, vecs[i]))
        node_kinds[nodes[i]].add(kind)
        if kind == "damage" and (m := NUM.search(ev["text"])):
            said = float(m.group())
            health_checked += 1
            at_pair = abs(said - float(snaps[i]["state"].get("health", -99))) <= HEALTH_TOL
            before = i > 0 and abs(said - float(snaps[i - 1]["state"].get("health", -99))) <= HEALTH_TOL
            health_match += at_pair
            health_match_or_previous += at_pair or before
    per_kind = {}
    for kind, texts in sorted(kinds.items()):
        template, fillers = _template(texts)
        per_kind[kind] = {
            "events": len(texts),
            "distinct_texts": len(set(texts)),
            "template": " ".join(template),
            "distinct_fillers": len(fillers),
            "top_fillers": fillers.most_common(5),
            "world_nodes_at_event": len(pair_nodes[kind]),
            "rest_at_event": pair_nodes[kind].get("rest", 0),
        }
    all_texts = [t for ts in kinds.values() for t in ts]
    vocab = Counter(w for t in all_texts for w in WORD.findall(t.lower()))
    return {
        "trace": str(path.relative_to(REPO)),
        "snapshots": len(snaps),
        "events": len(events),
        "unpaired_events": unpaired,
        "sensor_keys": sorted(snaps[0]["state"]) if snaps else [],
        "distinct_texts": len(set(all_texts)),
        "distinct_templates": len(per_kind),
        "vocabulary_word_types": len(vocab),
        "world_nodes_whole_trace": len(set(nodes) - {"rest"}),
        "world_nodes_hosting_events": len(set(node_kinds) - {"rest"}),
        "world_nodes_hosting_2plus_kinds": sum(1 for n, k in node_kinds.items() if n != "rest" and len(k) > 1),
        # mineflayer fires entityHurt before the health update lands, so a damage text can
        # report the PRE-hit health — i.e. match the snapshot one tick before its pair.
        "known_answer_damage_health": {
            "checked": health_checked,
            "matched_paired_snapshot": health_match,
            "matched_paired_or_previous": health_match_or_previous,
        },
        "raw_vector_kind_separability": _kind_separability(kind_vecs),
        "per_kind": per_kind,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", type=Path, help="also write the report here (outside the gated data tree)")
    args = ap.parse_args()
    candidates = sorted(DATA_DIR.rglob("*.jsonl"))
    paired = [p for p in candidates if all(_load(p))]
    report = {
        "jsonl_files_scanned": len(candidates),
        "paired_traces": [audit_trace(p) for p in paired],
    }
    print(json.dumps(report, indent=2, default=list))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=list) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
