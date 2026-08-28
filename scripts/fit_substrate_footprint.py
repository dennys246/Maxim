#!/usr/bin/env python3
"""FIT — measure the Maxim *substrate* memory footprint.

Answers the **FIT** gate ([docs/plans/deferred/reachy_app_maxim_seams.md](../docs/plans/deferred/reachy_app_maxim_seams.md)):
does the substrate (encoder + EC / NAc / Hippocampus / ATL / SCN) fit on the
Reachy Mini's Raspberry Pi when the large LLM tier is **remote** (mesh)? The LLM is
NOT loaded here — this measures the substrate only, which is what stays local on
the robot regardless of where the LLM runs.

Run on any box for a first ballpark; run on the actual Pi for the real answer (see
the runbook: docs/embodiment/reachy_mini/fit_runbook.md). The single biggest lever
is **which encoder backend is active**:

  * real `sentence-transformers` (pulls torch)  → higher semantic quality, HEAVY RSS
  * bag-of-words fallback (no torch)            → lower quality, LIGHT RSS

so this script reports the active backend explicitly. To compare both, run it once
with the ``semantic`` extra installed and once without.

Usage:
    PYTHONPATH=src python scripts/fit_substrate_footprint.py
    PYTHONPATH=src python scripts/fit_substrate_footprint.py --exercise 200 --json
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import tempfile


# ---------------------------------------------------------------------------
# RSS measurement — robust across macOS (dev box) and Linux/aarch64 (the Pi)
# ---------------------------------------------------------------------------
def rss_mb() -> float:
    """Current resident-set size in MB. psutil > /proc > resource(peak)."""
    # 1. psutil — accurate *current* RSS, cross-platform (if installed).
    try:
        import psutil  # type: ignore

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        pass
    # 2. /proc/self/status VmRSS — Linux / the Pi. Current RSS.
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # kB → MB
    except Exception:
        pass
    # 3. resource.ru_maxrss — PEAK RSS, platform-dependent units.
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports kB.
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def snapshot(stages: list[tuple[str, float]], label: str) -> float:
    gc.collect()
    mb = rss_mb()
    stages.append((label, mb))
    return mb


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--exercise",
        type=int,
        default=100,
        help="Number of percept encodes to run for a steady-state reading (default 100).",
    )
    ap.add_argument(
        "--model",
        default="",
        help="Force an encoder model name; empty = production default (torch) / fastembed default (--onnx).",
    )
    ap.add_argument(
        "--onnx",
        action="store_true",
        help="Measure the torch-FREE path: fastembed/onnxruntime instead of sentence-transformers.",
    )
    ap.add_argument("--json", action="store_true", help="Emit a machine-readable JSON summary too.")
    args = ap.parse_args()

    stages: list[tuple[str, float]] = []

    # 0. Baseline — before ANY maxim import.
    baseline = snapshot(stages, "baseline (fresh interpreter)")

    # 1. Import the package (numpy/scipy land here; torch does NOT yet).
    import maxim  # noqa: F401
    from maxim.runtime.bio_stack import build_bio_stack

    snapshot(stages, "after `import maxim`")

    # 2. Build the substrate (LLM stays remote/unloaded — with_default_network=False).
    tmp = tempfile.mkdtemp(prefix="fit_substrate_")
    stack = build_bio_stack(agent_id="fit_probe", persistence_dir=tmp)
    snapshot(stages, "after build_bio_stack (substrate wired)")

    # 3. Force the encoder to load its model — the dominant footprint. This is where
    #    torch + the sentence-transformers weights land (~430 MB), OR (--onnx) the
    #    torch-free onnxruntime path (~a fraction of that), OR the bag-of-words
    #    fallback (~nothing).
    if args.onnx:
        from fastembed import TextEmbedding

        onnx_name = args.model or "BAAI/bge-small-en-v1.5"
        emb = TextEmbedding(model_name=onnx_name)

        def encode_fn(texts: list[str]) -> None:
            list(emb.embed(texts))  # fastembed .embed yields; force it

        backend = f"onnxruntime/fastembed [{onnx_name}] (NO torch)"
    else:
        from maxim.similarity import encoder as _enc

        model = _enc._get_encoder(args.model) if args.model else _enc._get_encoder()
        backend = "sentence-transformers (torch)" if model is not None else "bag-of-words fallback (no torch)"

        def encode_fn(texts: list[str]) -> None:
            if model is not None:
                model.encode(texts)

    try:
        encode_fn(["warmup: the robot turned toward the sound"])
    except Exception as e:  # pragma: no cover - defensive
        print(f"  (warmup encode failed: {e})", file=sys.stderr)
    snapshot(stages, "after encoder warmup (model loaded)")

    # 4. Exercise — encode N percept-like strings to reach a realistic steady state.
    for i in range(args.exercise):
        text = f"percept {i}: a warm hand, a cold draft, a sound at bearing {i % 360} degrees"
        encode_fn([text])
    snapshot(stages, f"after exercise (+{args.exercise} encodes)")

    # keep `stack` alive to the end so its structures count.
    assert stack is not None

    # ---- report -----------------------------------------------------------
    substrate_delta = stages[-1][1] - baseline
    print()
    print("=" * 68)
    print("FIT — Maxim substrate footprint (LLM remote / not loaded)")
    print("=" * 68)
    print(f"  platform     : {platform.platform()} / {platform.machine()}")
    print(f"  python       : {platform.python_version()}")
    print(f"  encoder      : {backend}")
    print("-" * 68)
    prev = baseline
    for label, mb in stages:
        delta = mb - prev
        sign = "+" if delta >= 0 else "-"
        print(f"  {label:<42} {mb:8.1f} MB   ({sign}{abs(delta):6.1f})")
        prev = mb
    print("-" * 68)
    print(f"  SUBSTRATE FOOTPRINT (total − baseline)      {substrate_delta:8.1f} MB")
    print("=" * 68)
    print(
        "  Verdict is against the ACTUAL Pi: compare this delta to the RAM left\n"
        "  after the reachy-mini daemon + GStreamer (measure on-hardware per the\n"
        "  runbook). This dev-box number is a ballpark; aarch64 torch differs.\n"
        f"  Backend note: '{backend}' — rerun with/without the `semantic` extra to\n"
        "  see both footprints; the fallback is the lean-Pi path."
    )

    if args.json:
        print(
            json.dumps(
                {
                    "platform": platform.platform(),
                    "machine": platform.machine(),
                    "python": platform.python_version(),
                    "encoder_backend": backend,
                    "baseline_mb": round(baseline, 1),
                    "substrate_footprint_mb": round(substrate_delta, 1),
                    "stages": [{"label": la, "rss_mb": round(mb, 1)} for la, mb in stages],
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
