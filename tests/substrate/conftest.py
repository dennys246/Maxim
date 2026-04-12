"""Pytest fixtures for substrate testing."""

from __future__ import annotations

from typing import Any

import pytest

from tests.substrate.persistence_harness import RoundTripResult, run_round_trip


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
