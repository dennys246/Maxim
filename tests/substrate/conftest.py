"""Pytest fixtures for substrate testing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.substrate.persistence_harness import RoundTripResult, run_round_trip

COMMITTED_RESULTS_DIR = Path(__file__).parent.parent.parent / "docs" / "experiments" / "results"


@pytest.fixture
def publish_sweep_results(request: pytest.FixtureRequest, tmp_path: Path):
    """Write sweep output somewhere safe, and only overwrite evidence on purpose.

    Bugs ledger D25. These sweeps used to `json.dump` straight into
    ``docs/experiments/results/*.json`` as a side effect of running. Those files
    are S4 committed raw records, so any run — including one on a degraded
    apparatus — silently replaced real evidence. Observed 2026-08-20: an offline
    run rewrote ``p1_recognition_sweep.json`` from ``seeds_passing: 7`` to
    ``0`` and ``means_pass: true`` to ``false``. It was caught in ``git status``
    by luck, not by a guard.

    Default: write to the test's tmp_path and print where it went, so a normal
    run is observable but harmless. With ``--write-experiment-results``: also
    write the committed copy, because updating the record is then a deliberate,
    reviewable act rather than a side effect.

    The apparatus check is the caller's job and must happen BEFORE measuring —
    see ``similarity/encoder.require_semantic_encoder`` (D26). This fixture
    refuses to publish results it cannot attribute to a real encoder.
    """

    def _publish(filename: str, payload: dict[str, Any]) -> Path:
        scratch = tmp_path / filename
        scratch.write_text(json.dumps(payload, indent=2))

        if not request.config.getoption("--write-experiment-results"):
            print(
                f"\nResults written to {scratch}\n"
                f"  (committed record at docs/experiments/results/{filename} left untouched — "
                f"pass --write-experiment-results to update it deliberately)"
            )
            return scratch

        from maxim.similarity.encoder import require_semantic_encoder

        require_semantic_encoder(context=f"publishing {filename}")

        COMMITTED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        committed = COMMITTED_RESULTS_DIR / filename
        committed.write_text(json.dumps(payload, indent=2))
        print(f"\nUPDATED COMMITTED RECORD: {committed} (review this diff before committing)")
        return committed

    return _publish


@pytest.fixture
def persistence_round_trip():
    """Fixture that wraps run_round_trip for convenient per-phase usage.

    Usage in test files:
        def test_hippo_round_trip(persistence_round_trip):
            hippo = Hippocampus(config=HippocampusConfig())
            hippo.store(EpisodicMemory(...))
            result = persistence_round_trip(
                state={"hippocampus": hippo},
                probe="tests.substrate.probes:hippocampus_episode_count",
            )
            assert result.success
    """

    def _run(
        *,
        state: dict[str, Any],
        probe: str,
        tolerance: float = 0.0,
        timeout_s: float = 30.0,
    ) -> RoundTripResult:
        return run_round_trip(
            state=state,
            probe=probe,
            tolerance=tolerance,
            timeout_s=timeout_s,
        )

    return _run
