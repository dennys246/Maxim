#!/usr/bin/env python
"""P4 Stage 2 — Oxford Flowers-102 CLIP zero-shot calibration sweep.

ONE-SHOT script, NOT a test. Runs the CLIP ViT-B-32 zero-shot classifier
over the full 102 Oxford Flowers classes using the ``test`` split, picks
the 10 classes whose top-1 accuracy lands in the ``[0.50, 0.85]``
headroom band, and emits two artifacts:

- ``docs/experiments/p4_clip_calibration.md`` — human-readable report
  with the full 102-class table (sorted by top-1 accuracy), the 10-class
  selection rationale, and the pinned list.
- ``docs/experiments/results/p4_clip_calibration.json`` — machine-
  readable dump of per-class accuracy + the chosen subset.

The headroom band is intentional: classes where CLIP already scores
≥0.85 saturate the OpenCLIP baseline and collapse ``baseline_std → 0``
at Stage 3, which would render the ``+2σ`` margin vacuous (this was
the Round 1 review fold that killed imagenette). Classes where CLIP
scores ≤0.50 leave so much noise that the head-to-head becomes a
comparison of two weak signals. The ``[0.50, 0.85]`` band is the
substrate's sweet spot for demonstrating hippocampus-mediated
cross-modal binding OVER the raw-cosine baseline.

Determinism:
- ``torch.manual_seed(42)`` + deterministic numpy RNG.
- Per-class 5 test-split samples chosen via fixed seed ordering — the
  same re-run on the same torchvision cache reproduces byte-identical
  accuracies.

Expected wall clock:
- RTX 5080: ~30-60s for all 102 classes × 5 samples = 510 images.
- Mac M-series (Metal or CPU): ~2-5 minutes.
- This is a one-shot — re-running is forbidden under the "no band-aid
  fixture tweaks" rule unless explicitly amended in a follow-up PR.

Usage:
    PYTHONPATH=src python scripts/p4_clip_calibration_sweep.py
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("p4_calibration")


# Calibration samples per class. At 5/class, per-class accuracy takes
# 6 discrete values {0.0, 0.2, 0.4, 0.6, 0.8, 1.0} and the headroom-band
# filter ties — the first sweep attempt tied 10+ classes all at 0.600
# and the choice was arbitrary. At 20/class the resolution is 0.05 so
# the headroom band has useful spread. The Stage 2 test fixture pins
# 5 images per chosen class (the plan's ~50-pair budget), so the
# calibration sample count is independent of the test-time sample count.
CALIBRATION_SAMPLES_PER_CLASS = 20
HEADROOM_LOW = 0.50
HEADROOM_HIGH = 0.85
TARGET_N_CLASSES = 10
SEED = 42


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _load_flowers_test_split() -> Any:
    """Load the Oxford Flowers-102 test split via torchvision. Downloads
    on first run into ``~/.cache/maxim/p4_flowers`` so the cache is
    repo-agnostic and discoverable."""
    from torchvision.datasets import Flowers102

    cache_root = Path.home() / ".cache" / "maxim" / "p4_flowers"
    cache_root.mkdir(parents=True, exist_ok=True)
    logger.info("loading Flowers102 test split from %s", cache_root)

    dataset = Flowers102(
        root=str(cache_root),
        split="test",
        download=True,
    )
    logger.info("loaded %d test images across %d classes", len(dataset), len(Flowers102.classes))
    return dataset, Flowers102.classes


def _sample_per_class(
    dataset: Any,
    n_classes: int,
    k_per_class: int,
) -> dict[int, list[int]]:
    """Return a dict mapping class_idx -> list of k sample indices in
    ``dataset``. Deterministic via sorted-by-first-occurrence pick."""
    by_class: dict[int, list[int]] = {i: [] for i in range(n_classes)}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        by_class[label].append(idx)

    # Deterministic pick: first k samples in dataset order for each class.
    return {cls: indices[:k_per_class] for cls, indices in by_class.items()}


def _run_zero_shot_sweep(
    dataset: Any,
    class_names: list[str],
    samples_by_class: dict[int, list[int]],
) -> dict[int, float]:
    """For each class, classify its k test samples against the full
    102-name text prompt set via CLIP cosine similarity. Return per-
    class top-1 accuracy."""
    from maxim.models.vision.clip_encoder import VisionEncoder

    encoder = VisionEncoder()
    logger.info("encoding %d class prompts via CLIP-text", len(class_names))
    text_prompts = [f"a photo of a {name}" for name in class_names]
    text_embs = np.stack([encoder.encode_text(p) for p in text_prompts])
    # Normalize once up front so cosine similarity becomes a dot product.
    text_norms = np.linalg.norm(text_embs, axis=1, keepdims=True)
    text_embs = text_embs / text_norms

    logger.info(
        "encoding %d image samples (%d classes × %d/class)",
        sum(len(v) for v in samples_by_class.values()),
        len(samples_by_class),
        CALIBRATION_SAMPLES_PER_CLASS,
    )

    per_class_acc: dict[int, float] = {}
    start = time.monotonic()
    for cls_idx in sorted(samples_by_class.keys()):
        indices = samples_by_class[cls_idx]
        if not indices:
            per_class_acc[cls_idx] = 0.0
            continue

        correct = 0
        total = 0
        for sample_idx in indices:
            image, _ = dataset[sample_idx]
            img_emb = encoder.encode_image(image)
            img_emb = img_emb / np.linalg.norm(img_emb)
            # Dot against all normalized text prompts; argmax is the
            # zero-shot prediction.
            scores = text_embs @ img_emb
            pred = int(np.argmax(scores))
            if pred == cls_idx:
                correct += 1
            total += 1
        per_class_acc[cls_idx] = correct / total if total > 0 else 0.0

        if (cls_idx + 1) % 10 == 0:
            elapsed = time.monotonic() - start
            logger.info(
                "classes 0..%d done in %.1fs (running mean acc %.3f)",
                cls_idx,
                elapsed,
                float(np.mean(list(per_class_acc.values()))),
            )

    total_elapsed = time.monotonic() - start
    logger.info("sweep done in %.1fs", total_elapsed)
    return per_class_acc


def _pick_headroom_band(
    per_class_acc: dict[int, float],
    class_names: list[str],
    low: float,
    high: float,
    target_n: int,
) -> list[tuple[int, str, float]]:
    """Pick classes whose accuracy lands in [low, high]. If fewer than
    target_n match, widen the band adaptively (in 0.05 increments) and
    log the relaxation.  If more than target_n match, pick the target_n
    whose accuracy is closest to the center of the band."""
    candidates = [
        (idx, name, acc)
        for idx, (name, acc) in enumerate(zip(class_names, [per_class_acc[i] for i in range(len(class_names))]))
        if low <= acc <= high
    ]

    while len(candidates) < target_n and high - low < 1.0:
        low = max(0.0, low - 0.05)
        high = min(1.0, high + 0.05)
        logger.warning(
            "only %d classes in initial headroom; relaxing band to [%.2f, %.2f]",
            len(candidates),
            low,
            high,
        )
        candidates = [
            (idx, name, acc)
            for idx, (name, acc) in enumerate(zip(class_names, [per_class_acc[i] for i in range(len(class_names))]))
            if low <= acc <= high
        ]

    if len(candidates) > target_n:
        mid = (low + high) / 2.0
        candidates.sort(key=lambda t: abs(t[2] - mid))
        candidates = candidates[:target_n]

    # Round 2 Exec-lens #5 fold: hard-assert that we found enough
    # classes. Before the fold, a pathological sweep (e.g., all
    # accuracies NaN or all zero) could silently return fewer than
    # target_n classes and the fixture would be written with a short
    # list. Downstream code assumes exactly target_n. Loud failure
    # here means an operator caught the fixture regression at
    # calibration time instead of at Stage 3.
    assert len(candidates) >= target_n, (
        f"headroom sweep could only find {len(candidates)} classes in "
        f"[{low:.2f}, {high:.2f}] (relaxed from the initial band), "
        f"target was {target_n}. Investigate sweep inputs — either the "
        f"encoder is broken, the dataset is empty, or the fixture needs "
        f"a class-count reduction."
    )

    # Sort final choice by accuracy descending for readability.
    candidates.sort(key=lambda t: -t[2])
    return candidates


def _write_report(
    out_md: Path,
    out_json: Path,
    per_class_acc: dict[int, float],
    class_names: list[str],
    chosen: list[tuple[int, str, float]],
    elapsed_s: float,
) -> None:
    sorted_rows = sorted(
        [(i, class_names[i], per_class_acc[i]) for i in range(len(class_names))],
        key=lambda t: -t[2],
    )

    md_lines = [
        "# Substrate P4 Stage 2 — CLIP ViT-B-32 Oxford Flowers-102 calibration",
        "",
        f"**Sweep wall clock:** {elapsed_s:.1f}s",
        f"**Samples per class:** {CALIBRATION_SAMPLES_PER_CLASS}",
        f"**Headroom band:** [{HEADROOM_LOW:.2f}, {HEADROOM_HIGH:.2f}]",
        f"**Target class count:** {TARGET_N_CLASSES}",
        "",
        "## Pinned 10-class subset (headroom band)",
        "",
        "| idx | class | zero-shot top-1 |",
        "|---|---|---|",
    ]
    for idx, name, acc in chosen:
        md_lines.append(f"| {idx} | {name} | {acc:.3f} |")

    md_lines.extend(
        [
            "",
            "## Full 102-class accuracy table (descending)",
            "",
            "| idx | class | zero-shot top-1 |",
            "|---|---|---|",
        ]
    )
    for idx, name, acc in sorted_rows:
        md_lines.append(f"| {idx} | {name} | {acc:.3f} |")

    md_lines.extend(
        [
            "",
            "## Selection rationale",
            "",
            f"The headroom band `[{HEADROOM_LOW:.2f}, {HEADROOM_HIGH:.2f}]` was chosen so that",
            "Stage 3's head-to-head against OpenCLIP on these classes has meaningful",
            "signal on BOTH sides — saturated classes (>0.85) collapse the baseline",
            "standard deviation and make the `+2σ` margin vacuous, and impossible",
            "classes (<0.50) compare two weak signals.",
            "",
            "Classes in the target count are pinned verbatim in",
            "`scenarios/substrate/p4_mug_test.yaml`. Re-running this sweep is",
            "forbidden under the 'no band-aid fixture tweaks' rule unless an",
            "explicit follow-up PR amends the protocol.",
        ]
    )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines) + "\n")
    logger.info("wrote report %s", out_md)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "sweep_seconds": round(elapsed_s, 2),
        "samples_per_class": CALIBRATION_SAMPLES_PER_CLASS,
        "headroom_band": [HEADROOM_LOW, HEADROOM_HIGH],
        "target_n_classes": TARGET_N_CLASSES,
        "seed": SEED,
        "per_class_accuracy": {class_names[i]: round(per_class_acc[i], 4) for i in range(len(class_names))},
        "chosen": [{"idx": idx, "name": name, "accuracy": round(acc, 4)} for idx, name, acc in chosen],
    }
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    logger.info("wrote json %s", out_json)


def main() -> int:
    _set_seeds(SEED)

    logger.info("P4 Stage 2 — CLIP calibration sweep")
    logger.info(
        "seed=%d samples_per_class=%d band=[%.2f, %.2f] target_n=%d",
        SEED,
        CALIBRATION_SAMPLES_PER_CLASS,
        HEADROOM_LOW,
        HEADROOM_HIGH,
        TARGET_N_CLASSES,
    )

    dataset, class_names = _load_flowers_test_split()
    samples = _sample_per_class(dataset, n_classes=len(class_names), k_per_class=CALIBRATION_SAMPLES_PER_CLASS)

    start = time.monotonic()
    per_class_acc = _run_zero_shot_sweep(dataset, class_names, samples)
    elapsed = time.monotonic() - start

    chosen = _pick_headroom_band(per_class_acc, class_names, HEADROOM_LOW, HEADROOM_HIGH, TARGET_N_CLASSES)
    logger.info("chose %d classes:", len(chosen))
    for idx, name, acc in chosen:
        logger.info("  %d %s %.3f", idx, name, acc)

    repo_root = Path(__file__).resolve().parent.parent
    out_md = repo_root / "docs" / "experiments" / "p4_clip_calibration.md"
    out_json = repo_root / "docs" / "experiments" / "results" / "p4_clip_calibration.json"
    _write_report(out_md, out_json, per_class_acc, list(class_names), chosen, elapsed)

    return 0


if __name__ == "__main__":
    sys.exit(main())
