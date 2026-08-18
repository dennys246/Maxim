"""Seeded stimulus-order shuffle (the Exp 48 phase-lock fix, apparatus-v3).

The deterministic stimulus cycle phase-locked with the deterministic greedy
agent: directedness collapsed to exact seed-invariant twelfths at every
explore weight, and the CONTROL moved when the explore weight changed (the
2026-08-18 sweep finding). "shuffled" applies a seeded per-block permutation:
exposure stays balanced (every stimulus once per block) but the order is
unpredictable, so the measurement is dithered back into a graded function of
the learned policy.
"""

from __future__ import annotations

from maxim.simulation.cradle_mother import MotherScaffold, reactive_mother_tick


class _Root:
    def __init__(self) -> None:
        self.sensors = {"azimuth": object()}
        self.vital_metrics = {"azimuth": 0.0}
        self.drive_specs = {}


class _Emb:
    def __init__(self) -> None:
        self.root = _Root()
        self.live_world_set_sensors = set()


STIMS = (-0.7, 0.6, -0.5, 0.8, -0.9, 0.4)


def _sequence(order: str, seed: int, turns: int) -> list[float]:
    emb = _Emb()
    scaffold = MotherScaffold(stimulus_azimuths=STIMS, stimulus_order=order, stimulus_seed=seed)
    out = []
    for t in range(turns):
        tel = reactive_mother_tick(emb, scaffold=scaffold, turn_idx=t)
        out.append(tel["az_stimulus"])
    return out


def test_cycle_default_is_byte_identical_legacy():
    seq = _sequence("cycle", 0, 12)
    assert seq == list(STIMS) * 2


def test_shuffled_preserves_per_block_exposure():
    """Every stimulus appears exactly once per block — the S5 exposure
    contract survives the shuffle."""
    seq = _sequence("shuffled", 7, 18)
    for b in range(3):
        block = seq[b * 6 : (b + 1) * 6]
        assert sorted(block) == sorted(STIMS)


def test_shuffled_is_deterministic_per_seed():
    assert _sequence("shuffled", 7, 24) == _sequence("shuffled", 7, 24)


def test_shuffled_differs_across_seeds_and_blocks():
    """The point of the fix: seeds finally produce different stimulus orders,
    and consecutive blocks differ (no accidental re-lock onto one cycle)."""
    a = _sequence("shuffled", 7, 24)
    b = _sequence("shuffled", 8, 24)
    assert a != b
    blocks = [tuple(a[i * 6 : (i + 1) * 6]) for i in range(4)]
    assert len(set(blocks)) > 1
