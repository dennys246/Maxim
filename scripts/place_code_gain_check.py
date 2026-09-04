#!/usr/bin/env python3
"""Does the A4 gain help or hurt the audio place code? (1.1.4 PR 1, plan decision D3.)

The A4 nonlinear gain lives in ``_sensor_embed`` and would, if applied
globally, also reshape the place-coded audio channel — Gaussian population
activations whose INTERMEDIATE values (the overlap between adjacent cells)
carry the interpolation that Exp 46 validated at 6/6 direction separation. A
p=3 gain crushes intermediate activations (0.7 → weight 0.064), so the prior
is that gain hurts here; this measures it, per the house rule that membership
in ``SensorEncoderConfig.gain_modalities`` is measured, not assumed.

**Metric, frozen before the run:** over the 7 canonical cell-center
directions, at the shipped 0.85 threshold against the real
``EntorhinalCortex`` (audio frozen-centroid, real register protocol):

  separation   distinct clusters over the 7 directions          (want 7/7)
  stability    fraction of ±0.03 jittered readings completing
               onto their direction's cluster, 20 per direction (want ~1.0)

Both arms identical apart from ``gain_exponent``. A gain arm that loses
either criterion relative to no-gain keeps audio OUT of ``gain_modalities``.

Also measured, same record: the RAW-scalar audio channel (place code OFF,
the default) under gain — a single-sensor channel is scale-invariant under
cosine EXCEPT that the CENTERED reading (azimuth 0 → normalized 0.5) embeds
to the zero vector, deleting the "sound dead ahead" cluster. Reported as the
fraction of directions that produce a nonzero embedding.

Usage
-----
    python scripts/place_code_gain_check.py --json docs/experiments/data/place_code_gain_check_2026-09-03.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from maxim.similarity.ec import ECConfig, EntorhinalCortex  # noqa: E402
from maxim.similarity.encoder import _sensor_embed  # noqa: E402
from maxim.similarity.place_code import place_code, place_code_ranges  # noqa: E402

DIRECTIONS = (-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9)
JITTER_SD = 0.03
JITTERS_PER_DIRECTION = 20
THRESHOLD = 0.85  # SensorEncoderConfig.pattern_threshold, the shipped value


def run_arm(gain_exponent: float | None, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    ranges = place_code_ranges(prefix="azdir")
    ec = EntorhinalCortex(ECConfig(frozen_centroid_modalities=frozenset({"audio"})))
    ids: dict[float, str] = {}
    for d in DIRECTIONS:
        vec = _sensor_embed(place_code(d, prefix="azdir"), ranges=ranges, gain_exponent=gain_exponent)
        res = ec.pattern_complete_or_separate(vec, modality="audio", threshold=THRESHOLD, geometry=None)
        if res.is_new:
            ec.register_substrate_node(res.node_id, vec, "audio")
        ids[d] = res.node_id
    stable = total = 0
    for d in DIRECTIONS:
        for _ in range(JITTERS_PER_DIRECTION):
            dj = max(-1.0, min(1.0, d + rng.gauss(0, JITTER_SD)))
            vec = _sensor_embed(place_code(dj, prefix="azdir"), ranges=ranges, gain_exponent=gain_exponent)
            res = ec.pattern_complete_or_separate(vec, modality="audio", threshold=THRESHOLD, geometry=None)
            total += 1
            if (not res.is_new) and res.node_id == ids[d]:
                stable += 1
    return {
        "gain_exponent": gain_exponent,
        "distinct_clusters": len(set(ids.values())),
        "directions": len(DIRECTIONS),
        "jitter_stability": round(stable / total, 4),
    }


def raw_scalar_nonzero_fraction(gain_exponent: float) -> dict[str, Any]:
    nonzero = sum(
        1
        for d in DIRECTIONS
        if any(
            x != 0.0
            for x in _sensor_embed({"azimuth": d}, ranges={"azimuth": (-1.0, 1.0)}, gain_exponent=gain_exponent)
        )
    )
    return {
        "gain_exponent": gain_exponent,
        "nonzero_embeddings": nonzero,
        "directions": len(DIRECTIONS),
        "note": "the zeroed direction is azimuth 0.0 — the centered stimulus",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json", default="", help="write results here")
    p.add_argument("--allow-dirty", action="store_true")
    args = p.parse_args(argv)

    provenance = None
    if args.json:
        sys.path.insert(0, str(_REPO_ROOT / "scripts"))
        import maxim
        from _provenance import DirtyTreeError, ProvenanceError, in_process_code_provenance

        try:
            provenance = in_process_code_provenance(
                _REPO_ROOT, maxim.__file__, out_path=args.json, allow_dirty=args.allow_dirty
            )
        except (ProvenanceError, DirtyTreeError) as exc:
            print(f"[FAIL] gated-record preflight: {exc}", file=sys.stderr)
            return 3

    arms = [run_arm(None, args.seed), run_arm(3.0, args.seed)]
    raw = raw_scalar_nonzero_fraction(3.0)
    for arm in arms:
        print(
            f"place code, gain={arm['gain_exponent']}: "
            f"{arm['distinct_clusters']}/{arm['directions']} distinct, "
            f"jitter stability {arm['jitter_stability']:.3f}"
        )
    print(f"raw scalar under gain p=3: {raw['nonzero_embeddings']}/{raw['directions']} nonzero embeddings")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "harness": "scripts/place_code_gain_check.py",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "provenance": provenance,
                    "metric": "distinct clusters over 7 directions + jitter stability at 0.85, real EC",
                    "threshold": THRESHOLD,
                    "place_code_arms": arms,
                    "raw_scalar_under_gain": raw,
                },
                indent=2,
            )
        )
        print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
