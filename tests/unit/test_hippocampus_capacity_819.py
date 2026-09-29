"""#819 (the loud half) -- the Hippocampus capacity cap is never exceeded silently.

`Hippocampus._evict_one` never evicts a long-term memory at store time. When every stored memory is
long-term, an insert at `HippocampusConfig.max_nodes` evicted nothing and proceeded, with no warning:
reproduced with a cap of 3 (4 stored, nothing logged). Batch 2 makes that loud; the byte budget and the
lazy-heap eviction are memory-strength Phase 4.
"""

from __future__ import annotations

import logging

from maxim.memory.hippocampus import Hippocampus, HippocampusConfig


def _full_of_long_term(cap: int = 3) -> Hippocampus:
    h = Hippocampus(HippocampusConfig(max_nodes=cap, auto_save_after_sleep=False))
    for i in range(cap):
        h.store_observation(f"note {i}")
    for record in h._memories.values():
        record.long_term = True
    return h


def _over_capacity_warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING and "capacity" in r.getMessage()]


def test_exceeding_the_cap_with_only_long_term_memories_is_loud(caplog) -> None:
    h = _full_of_long_term()
    with caplog.at_level(logging.WARNING, logger="maxim.memory.hippocampus"):
        h.store_observation("one too many")
    assert len(h) == 4  # the insert still happens: nothing it may evict
    warnings = _over_capacity_warnings(caplog)
    assert len(warnings) == 1 and "4" in warnings[0] and "3" in warnings[0]
    assert h.stats()["over_capacity_inserts_this_process"] == 1


def test_the_warning_repeats_as_the_overage_doubles_not_on_every_insert(caplog) -> None:
    h = _full_of_long_term()
    with caplog.at_level(logging.WARNING, logger="maxim.memory.hippocampus"):
        for i in range(8):
            h.store_observation(f"extra {i}")
            for record in h._memories.values():
                record.long_term = True  # keep everything unevictable
    # overage 1, 2, 4, 8 -> four warnings for eight overflowing inserts
    assert len(_over_capacity_warnings(caplog)) == 4
    assert h.stats()["over_capacity_inserts_this_process"] == 8


def test_normal_eviction_at_the_cap_stays_quiet(caplog) -> None:
    """Anti-vacuity arm: at the cap with an evictable memory, one is evicted and nothing is logged."""
    h = Hippocampus(HippocampusConfig(max_nodes=3, auto_save_after_sleep=False))
    for i in range(3):
        h.store_observation(f"note {i}")
    with caplog.at_level(logging.WARNING, logger="maxim.memory.hippocampus"):
        h.store_observation("replaces the weakest")
    assert len(h) == 3
    assert _over_capacity_warnings(caplog) == []
    assert h.stats()["over_capacity_inserts_this_process"] == 0


def test_a_store_already_over_its_cap_stays_loud(caplog) -> None:
    """Loaded from a larger store (or grown past the cap by memories promoted at birth): one eviction per
    insert leaves it over the cap, and that is not silent either."""
    big = Hippocampus(HippocampusConfig(max_nodes=100, auto_save_after_sleep=False))
    for i in range(10):
        big.store_observation(f"note {i}")
    small = Hippocampus(HippocampusConfig(max_nodes=3, auto_save_after_sleep=False))
    small.load_state(big.dump())
    with caplog.at_level(logging.WARNING, logger="maxim.memory.hippocampus"):
        small.store_observation("one more")
    assert len(small) == 10  # one evicted, one inserted: still over
    assert len(_over_capacity_warnings(caplog)) == 1
    assert small.stats()["over_capacity_inserts_this_process"] == 1


def test_a_second_overflow_after_dropping_under_the_cap_warns_again(caplog) -> None:
    h = _full_of_long_term()
    with caplog.at_level(logging.WARNING, logger="maxim.memory.hippocampus"):
        h.store_observation("over")  # overage 1: warns
        for memory_id in list(h._memories)[:2]:
            with h._rwlock.write():
                h._remove_memory(memory_id)  # back under the cap (as sleep() can do)
        h.store_observation("under again")  # below the cap: resets the threshold
        for record in h._memories.values():
            record.long_term = True
        h.store_observation("over again")  # overage 1 again: warns again
    assert len(_over_capacity_warnings(caplog)) == 2
