"""P4 Stage 2 — Mug test subprocess round-trip.

Proves the P3.5 ``SessionSnapshot`` atomic-rollback machinery +
``_node_modality`` sidecar + Hebbian binding graph ALL survive a real
serialize → kill interpreter → fresh interpreter → reload cycle on
the real CLIP + Oxford Flowers-102 fixture. This is the one-and-only
test that exercises the full P4 mechanism with real-world geometry.

Gated behind:
- sentence_transformers (``semantic`` extra) via importorskip
- Flowers102 torchvision cache via an explicit skip when missing
- A fresh ATL + NAc + SCN + PerceptTraceBuffer + CrossLayerGraph
  instance so ``SessionSnapshot.capture(strict=True)`` can land the
  full envelope the harness expects
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _build_full_session():
    """Wire every bio-system the strict SessionSnapshot envelope needs.

    Mirrors the P3.5 Stage 2 round-trip test fixture but with a mug-
    test-prepared hippocampus instead of a synthetic one.
    """
    from maxim.decisions.nac import NAc
    from maxim.memory.atl import ATL
    from maxim.memory.cross_layer import CrossLayerGraph
    from maxim.memory.percept_trace_buffer import PerceptTraceBuffer
    from maxim.time.scn import SCN

    atl = ATL()
    nac = NAc()
    scn = SCN()
    ptb = PerceptTraceBuffer()
    cross_layer = CrossLayerGraph()
    return atl, nac, scn, ptb, cross_layer


def _prepare_mug_test_hippocampus():
    """Build a hippocampus with the full P4 mug test state wired in.

    Encodes the 10 × 5 = 50 fixture pairs via real CLIP + paraphrase-
    mpnet, binds each in an episode. Returns the ready-to-snapshot
    hippocampus instance.
    """
    from scripts.p4_mug_test_sweep import _build_and_bind  # reuse orchestrator

    from .p4_fixture_loader import load_fixture_descriptor, load_fixture_images

    descriptor = load_fixture_descriptor()
    images = load_fixture_images()
    _ec, hippocampus, _results = _build_and_bind(descriptor, images)
    return hippocampus


class TestP4MugTestRoundTrip:
    def test_real_clip_mug_test_survives_session_snapshot_round_trip(self) -> None:
        pytest.importorskip("sentence_transformers")
        pytest.importorskip("torchvision")
        pytest.importorskip("PIL")

        cache = Path.home() / ".cache" / "maxim" / "p4_flowers"
        if not cache.exists():
            pytest.skip("Flowers102 cache not present — run scripts/p4_clip_calibration_sweep.py to download")

        import sys

        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from tests.substrate.persistence_harness import run_session_round_trip

        hippocampus = _prepare_mug_test_hippocampus()
        atl, nac, scn, ptb, cross_layer = _build_full_session()

        systems = {
            "atl": atl,
            "hippocampus": hippocampus,
            "nac": nac,
            "scn": scn,
            "percept_trace_buffer": ptb,
            "cross_layer_graph": cross_layer,
        }

        result = run_session_round_trip(
            systems=systems,
            probe="tests.substrate.p4_mug_probe:retrieve_cross_modal_snapshot",
            tolerance=0.0,
            timeout_s=180.0,
        )

        assert result.success, (
            f"P4 mug test round-trip FAILED: {result.error}\n"
            f"child_returncode={result.child_returncode}\n"
            f"child_stderr={result.child_stderr}"
        )

        # The probe returns a dict of {text_node_id: [vision_node_id, ...]}.
        # Post-reload must exactly match pre-shutdown.
        assert result.pre_shutdown == result.post_reload, (
            "retrieve_cross_modal results diverged across round-trip — "
            "the _node_modality sidecar or binding graph did not restore correctly"
        )

        # Sanity: there are 10 text node ids (one per class) and each
        # retrieves at least one vision partner.
        assert len(result.pre_shutdown) == 10, f"expected 10 text nodes in the sidecar, got {len(result.pre_shutdown)}"
        for text_id, vision_hits in result.pre_shutdown.items():
            assert len(vision_hits) > 0, f"text node {text_id} retrieved zero vision partners"
