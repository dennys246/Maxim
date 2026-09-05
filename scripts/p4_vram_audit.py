#!/usr/bin/env python
"""P4 Stage 2 — VRAM audit.

Measures peak VRAM usage when the P4 Stage 2 vision + text encoders
are co-resident with a running local LLM (Qwen-14B or similar). This
answers the Stage 2 plan's VRAM audit deliverable: "combined footprint
of clip-ViT-B-32 + paraphrase-mpnet-base-v2 + Qwen-14B Q4_K_M @ 12k
context vs the spillover-detection plan's max(1.5, 0.55 * weights_gb)
headroom."

What the audit measures:

1. **Baseline GPU state** — VRAM used by whatever's already loaded
   (typically the LLM leader). Sampled via ``nvidia-smi`` (CUDA) or
   ``torch.mps.current_allocated_memory()`` (Mac Metal, falls back to
   CPU RSS).
2. **After CLIP load** — VRAM delta when ``clip-ViT-B-32`` is loaded
   through the VisionEncoder.
3. **After paraphrase-mpnet load** — additional VRAM when
   ``paraphrase-mpnet-base-v2`` is loaded.
4. **After full mug test run** — peak VRAM during encoding of 50
   images + 10 text prompts + EC pattern-complete + hippocampus
   episode binding.
5. **After explicit CUDA cache empty** — residual VRAM after a
   ``torch.cuda.empty_cache()`` call. Tells us how much is truly
   "held" vs "cached".

All measurements are written to docs/experiments/p4_vram_audit.md +
results/p4_vram_audit.json. The report includes a go/no-go against
the spillover-detection headroom formula.

Usage (on the RTX 5080 leader)::

    PYTHONPATH=src python scripts/p4_vram_audit.py

Usage (on any other GPU or CPU)::

    # Still runs. Reports "no CUDA" or "MPS/CPU baseline" and the
    # absolute allocated numbers for reproducibility.
    PYTHONPATH=src python scripts/p4_vram_audit.py
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("p4_vram_audit")


@dataclass
class Sample:
    label: str
    wall_clock_s: float
    backend: str  # "cuda" | "mps" | "cpu"
    allocated_mb: float
    reserved_mb: float
    smi_used_mb: float | None = None
    smi_total_mb: float | None = None
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def _nvidia_smi_used_total_mb() -> tuple[float | None, float | None]:
    """Return (used_mb, total_mb) for GPU 0 via nvidia-smi, or (None, None)
    if nvidia-smi is not available or this isn't an NVIDIA system."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return (None, None)
        used_str, total_str = result.stdout.strip().split(",")
        return (float(used_str.strip()), float(total_str.strip()))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return (None, None)


def _torch_allocated_reserved_mb() -> tuple[float, float, str]:
    """Return (allocated_mb, reserved_mb, backend) via torch."""
    try:
        import torch
    except ImportError:
        return (0.0, 0.0, "cpu")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        return (allocated, reserved, "cuda")
    # Round 2 Exec-lens #7 fold: use the canonical
    # torch.backends.mps.is_available() instead of the torch 2.x-only
    # torch.mps.is_available() shortcut. The canonical path works on
    # any torch version that has MPS support; the shortcut exists only
    # on torch 2.0+ and is not documented as stable.
    if hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        allocated = torch.mps.current_allocated_memory() / (1024**2)
        # MPS doesn't distinguish allocated vs reserved the same way.
        return (allocated, allocated, "mps")
    return (0.0, 0.0, "cpu")


def _sample(label: str, start: float, notes: str = "", extra: dict[str, Any] | None = None) -> Sample:
    allocated, reserved, backend = _torch_allocated_reserved_mb()
    smi_used, smi_total = _nvidia_smi_used_total_mb()
    return Sample(
        label=label,
        wall_clock_s=round(time.monotonic() - start, 3),
        backend=backend,
        allocated_mb=round(allocated, 1),
        reserved_mb=round(reserved, 1),
        smi_used_mb=smi_used,
        smi_total_mb=smi_total,
        notes=notes,
        extra=extra or {},
    )


def _try_cuda_empty_cache() -> bool:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            return True
    except ImportError:
        return False
    return False


def _load_clip() -> Any:
    from maxim.models.vision.clip_encoder import VisionEncoder

    encoder = VisionEncoder()
    encoder._ensure_model()  # force load
    return encoder


def _load_mpnet() -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("paraphrase-mpnet-base-v2")


def _run_mug_test_encoding(clip_enc: Any, mpnet: Any) -> dict[str, Any]:
    """Run the full P4 mug test encoding pipeline on real Flowers102
    images. Returns a summary of what was encoded; peak VRAM is
    captured by the surrounding sampler, not this function."""
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "tests"))

    from substrate.p4_fixture_loader import load_fixture_descriptor, load_fixture_images

    descriptor = load_fixture_descriptor()
    images = load_fixture_images()

    n_text = 0
    n_image = 0
    for cls in descriptor.classes:
        mpnet.encode(cls.name, convert_to_numpy=True)
        n_text += 1
        for sample_idx in cls.sample_indices:
            image = images[(cls.name, sample_idx)]
            clip_enc.encode_image(image)
            n_image += 1

    return {"n_text_encoded": n_text, "n_image_encoded": n_image}


def _write_report(out_md: Path, out_json: Path, samples: list[Sample], total_wall_s: float) -> None:
    md_lines = [
        "# Substrate P4 Stage 2 — VRAM audit",
        "",
        f"**Total wall clock:** {total_wall_s:.1f}s",
        f"**Backend:** {samples[0].backend if samples else 'unknown'}",
        "",
        "## Summary",
        "",
        "Measurements taken at each stage of bringing up the P4 Stage 2",
        "encoder stack: baseline (whatever was already loaded) → add CLIP",
        "→ add paraphrase-mpnet → run full mug test encoding → drop GPU",
        "cache. On a machine with a running local LLM (RTX 5080 leader +",
        "Qwen-14B-Q4_K_M), the baseline should show ~8-10 GB used by the",
        "LLM; the deltas tell us how much additional VRAM the P4 Stage 2",
        "encoders consume co-resident with inference.",
        "",
        "## Measurement table",
        "",
        "| step | wall | torch alloc MB | torch reserved MB | nvidia-smi used MB | nvidia-smi total MB |",
        "|---|---|---|---|---|---|",
    ]
    for s in samples:
        smi_used_str = f"{s.smi_used_mb:.0f}" if s.smi_used_mb is not None else "n/a"
        smi_total_str = f"{s.smi_total_mb:.0f}" if s.smi_total_mb is not None else "n/a"
        md_lines.append(
            f"| {s.label} | {s.wall_clock_s:.1f}s | {s.allocated_mb:.0f} | "
            f"{s.reserved_mb:.0f} | {smi_used_str} | {smi_total_str} |"
        )

    # Deltas (nvidia-smi is more truthful than torch on CUDA under a
    # co-resident llama-cpp-server process — torch only sees what torch
    # allocated, not what llama.cpp allocated).
    md_lines.extend(
        [
            "",
            "## Deltas (successive steps)",
            "",
        ]
    )
    prev: Sample | None = None
    for s in samples:
        if prev is not None:
            # Round 2 Exec-lens #6 fold: use explicit signed format
            # ({:+.0f}) so negative deltas render as "-4 MB" instead of
            # "+-4 MB" (happens when torch.cuda.empty_cache releases
            # memory back to the driver).
            smi_delta = ""
            if s.smi_used_mb is not None and prev.smi_used_mb is not None:
                smi_delta = f" (nvidia-smi: {s.smi_used_mb - prev.smi_used_mb:+.0f} MB)"
            alloc_delta = s.allocated_mb - prev.allocated_mb
            md_lines.append(f"- **{prev.label} → {s.label}**: torch alloc {alloc_delta:+.0f} MB{smi_delta}")
        prev = s

    # Headroom check — only meaningful on CUDA where nvidia-smi total is known.
    last = samples[-1] if samples else None
    md_lines.append("")
    md_lines.append("## Headroom check")
    md_lines.append("")
    if last is not None and last.smi_used_mb is not None and last.smi_total_mb is not None:
        free_mb = last.smi_total_mb - last.smi_used_mb
        free_gb = free_mb / 1024.0
        md_lines.append(
            f"- **GPU:** {last.smi_total_mb:.0f} MB total, {last.smi_used_mb:.0f} MB used, {free_mb:.0f} MB free ({free_gb:.2f} GB)"
        )
        md_lines.append(
            "- **Spillover-detection headroom rule** (Plan 3.6): "
            "recommended minimum headroom is `max(1.5 GB, 0.55 * weights_gb)`. "
            "For Qwen-14B Q4_K_M (~8 GB weights) that's `max(1.5, 4.4) = 4.4 GB`."
        )
        if free_gb >= 4.4:
            md_lines.append(f"- **VERDICT: PASS** — {free_gb:.2f} GB free exceeds 4.4 GB headroom requirement.")
        elif free_gb >= 1.5:
            md_lines.append(
                f"- **VERDICT: WARN** — {free_gb:.2f} GB free is above the 1.5 GB "
                f"minimum but below the 4.4 GB recommended for Qwen-14B. "
                f"Consider smaller LLM (Qwen-7B or mistral-7b) OR lower n_ctx "
                f"OR run Phase 2D/Stage 3 sweeps on a dedicated worktree with "
                f"MAXIM_LLM_ENABLED=0."
            )
        else:
            md_lines.append(
                f"- **VERDICT: FAIL** — {free_gb:.2f} GB free is below the 1.5 GB "
                f"absolute minimum. KV cache spillover risk is high. Take "
                f"immediate action: drop LLM to a smaller model, lower n_ctx, "
                f"or unload LLM before running P4 Stage 2/3 sweeps."
            )
    else:
        md_lines.append(
            "- No CUDA backend detected — headroom verdict not applicable. This run provides reproducibility data only."
        )

    md_lines.extend(
        [
            "",
            "## How to reproduce",
            "",
            "```bash",
            "# Run ON the RTX 5080 leader (or any machine where you want",
            "# to measure co-residency). Requires the 'semantic' extra and",
            "# the Flowers102 torchvision cache populated by Phase 2B.",
            "PYTHONPATH=src python scripts/p4_vram_audit.py",
            "```",
            "",
            "The script is idempotent — re-running gives the same delta",
            "values as long as the baseline GPU state is unchanged (i.e.,",
            "the LLM is still loaded at the same ctx length). The absolute",
            "baseline MB number will vary depending on what else is loaded.",
        ]
    )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines) + "\n")
    logger.info("wrote report %s", out_md)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "total_wall_clock_s": round(total_wall_s, 2),
        "samples": [
            {
                "label": s.label,
                "wall_clock_s": s.wall_clock_s,
                "backend": s.backend,
                "allocated_mb": s.allocated_mb,
                "reserved_mb": s.reserved_mb,
                "smi_used_mb": s.smi_used_mb,
                "smi_total_mb": s.smi_total_mb,
                "notes": s.notes,
                "extra": s.extra,
            }
            for s in samples
        ],
    }
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    logger.info("wrote json %s", out_json)


def _oom_exceptions() -> tuple[type[BaseException], ...]:
    """Return the exception types that indicate a VRAM OOM condition,
    degrading gracefully if torch isn't installed."""
    excs: list[type[BaseException]] = [MemoryError, RuntimeError]
    try:
        import torch

        if hasattr(torch.cuda, "OutOfMemoryError"):
            excs.append(torch.cuda.OutOfMemoryError)  # type: ignore[attr-defined]
    except ImportError:
        pass
    return tuple(excs)


def _safe_step(
    label: str,
    action,
    start: float,
    samples: list[Sample],
    extra: dict[str, Any] | None = None,
):
    """Run one audit step (load CLIP, load mpnet, run mug test, ...)
    inside a try/except that captures OOM or other fatal errors as a
    Sample with notes='OOM: …'. Returns the action's return value on
    success, or None on failure. On failure the caller should break
    out of the remaining steps but the partial samples list is still
    written to the report — the whole point of the audit is to
    measure "does the encoder stack fit?" and an OOM IS the answer.

    Round 2 Arch-lens #7 fold: before this helper, the script crashed
    on OOM and _write_report was never called, leaving operators with
    zero data from the audit they specifically ran to detect OOM.
    """
    try:
        result = action()
        samples.append(_sample(label, start, extra=extra))
        logger.info("%s: %s", label, samples[-1])
        return result
    except _oom_exceptions() as e:
        notes = f"OOM: {type(e).__name__}: {str(e)[:200]}"
        samples.append(_sample(label, start, notes=notes, extra=extra))
        logger.error("%s FAILED with %s", label, notes)
        return None


def _evidence_args() -> argparse.Namespace:
    """D27: committed-evidence writes are opt-in (see scripts/_provenance.py)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--write-experiment-results",
        action="store_true",
        help="update the COMMITTED report + results record (refuses a dirty tree)",
    )
    ap.add_argument("--allow-dirty", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = _evidence_args()
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "scripts"))
    from _provenance import evidence_out_paths

    out_md, out_json = evidence_out_paths(
        repo_root,
        [
            repo_root / "docs" / "experiments" / "p4_vram_audit.md",
            repo_root / "docs" / "experiments" / "results" / "p4_vram_audit.json",
        ],
        write_experiment_results=args.write_experiment_results,
        allow_dirty=args.allow_dirty,
    )
    logger.info("P4 Stage 2 VRAM audit")

    start = time.monotonic()
    samples: list[Sample] = []

    samples.append(_sample("baseline", start, notes="before any P4 encoder load"))
    logger.info("baseline: %s", samples[-1])

    logger.info("loading CLIP ViT-B-32...")
    clip_enc = _safe_step("after_clip_load", _load_clip, start, samples)
    if clip_enc is None:
        logger.error("aborting audit — CLIP load failed; writing partial report")
        total = time.monotonic() - start
        _write_report(out_md, out_json, samples, total)
        return 2

    logger.info("loading paraphrase-mpnet-base-v2...")
    mpnet = _safe_step("after_mpnet_load", _load_mpnet, start, samples)
    if mpnet is None:
        logger.error("aborting audit — mpnet load failed; writing partial report")
        total = time.monotonic() - start
        _write_report(out_md, out_json, samples, total)
        return 2

    logger.info("running full mug test encoding...")
    mug_stats_holder: list[dict[str, Any]] = []

    def _encode_action():
        stats = _run_mug_test_encoding(clip_enc, mpnet)
        mug_stats_holder.append(stats)
        return stats

    encoded = _safe_step(
        "after_mug_test_encode",
        _encode_action,
        start,
        samples,
        extra=mug_stats_holder[0] if mug_stats_holder else None,
    )
    if encoded is None:
        logger.error("mug test encoding OOM'd; writing partial report")
        total = time.monotonic() - start
        _write_report(out_md, out_json, samples, total)
        return 2

    if _try_cuda_empty_cache():
        samples.append(_sample("after_cuda_empty_cache", start))
        logger.info("after torch.cuda.empty_cache: %s", samples[-1])

    total = time.monotonic() - start
    _write_report(out_md, out_json, samples, total)

    return 0


if __name__ == "__main__":
    sys.exit(main())
