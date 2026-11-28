"""#1684: a caller's time of exactly 0.0 is a time, not "no time given".

`x or time.time()` turned 0.0 into the wall clock at five sites, so a caller on a clock that starts at zero (a test
clock, a stepped world such as Overdrive's, a replay) got wall time mixed in on its first tick. Each test passes 0.0
and checks the work was done at 0.0; each is red on the `or` form.

The CostTracker sites were later superseded by D-WC5 (#1769): cost windows run on the wall, and a caller clock is
ignored with a ``DeprecationWarning`` (see the CostTracker tests below).

`SalienceMap` also read `last_visited == 0` as "never visited", so a visit at 0.0 was forgotten as soon as it was
recorded. The visit count is the visited flag now (`test_a_visit_at_zero_still_inhibits_return`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from maxim.attention.salience_map import SalienceMap
from maxim.models.language.cost_tracker import CostTracker, CostTrackerConfig, ModelPricing
from maxim.salience.salience_network import SalienceNetwork

_MODEL = "priced-model"


class _Gaze:
    """The one method `SalienceMap.update` reads from a gaze history."""

    def __init__(self, pos: tuple[float, float]) -> None:
        self._pos = pos

    def get_current_position(self) -> tuple[float, float]:
        return self._pos


def _tracker(tmp_path: Path) -> CostTracker:
    # Its own state file: the tracker loads and saves its spend, and a shared file carries one test's calls into the next.
    pricing = {_MODEL: ModelPricing(input_price=1_000_000.0, output_price=0.0)}
    return CostTracker(pricing=pricing, config=CostTrackerConfig(state_path=str(tmp_path / "cost_state.json")))


def _record(tracker: CostTracker, timestamp: float) -> float:
    return tracker.record(provider="p", model=_MODEL, input_tokens=1, output_tokens=0, timestamp=timestamp)


def _visited_cell(smap: SalienceMap, pos: tuple[float, float]):
    row, col = smap._pixel_to_cell(*pos)
    return smap._grid[row][col]


def test_salience_map_update_visits_at_zero() -> None:
    smap, pos = SalienceMap(), (10.0, 10.0)
    smap.update([], gaze_history=_Gaze(pos), current_time=0.0)
    cell = _visited_cell(smap, pos)
    assert cell.visit_count == 1
    assert cell.last_visited == 0.0


def test_a_visit_at_zero_still_inhibits_return() -> None:
    smap, pos = SalienceMap(), (10.0, 10.0)
    smap.update([], gaze_history=_Gaze(pos), current_time=0.0)
    # Just visited, so inhibition of return holds the cell well below "fully unexplored" (1.0).
    assert _visited_cell(smap, pos).unexplored < 0.5


def test_salience_network_detections_seen_at_zero() -> None:
    net = SalienceNetwork()
    net.update_from_detections(
        [{"track_id": 7, "class_id": 0, "label": "cup", "confidence": 0.9, "bbox": (1, 1, 2, 2)}], timestamp=0.0
    )
    (obj,) = [o for o in net.query() if o.get("track_id") == 7]
    assert obj["last_seen"] == 0.0


# The CostTracker half of #1684 is superseded by D-WC5 (#1769, world clock A-pre): cost windows run on the wall,
# so a caller's clock, 0.0 included, is ignored with a DeprecationWarning rather than honoured. A caller on a clock
# starting at 0.0 used to reset the month and prune its own spend; these pin that it can no longer reach the windows.


def test_cost_tracker_ignores_a_caller_timestamp_of_zero(tmp_path: Path) -> None:
    tracker = _tracker(tmp_path)
    with pytest.warns(DeprecationWarning, match=r"record\(timestamp=\.\.\.\) is deprecated and ignored"):
        assert _record(tracker, 0.0) > 0
    (stamp,) = [ts for ts, _ in tracker._hourly.events]
    assert stamp > 1_000_000_000.0, f"booked on the caller's clock, not the wall: {stamp}"


def test_cost_tracker_totals_ignore_a_caller_now(tmp_path: Path) -> None:
    tracker = _tracker(tmp_path)
    cost = tracker.record(provider="p", model=_MODEL, input_tokens=1, output_tokens=0)
    # Read at a far-future caller clock, the call would be pruned as decades old if the clock were honoured.
    with pytest.warns(DeprecationWarning, match=r"get_totals\(now=\.\.\.\)"):
        assert tracker.get_totals(now=4_000_000_000.0)["hourly"] == cost


def test_cost_tracker_per_provider_totals_ignore_a_caller_now(tmp_path: Path) -> None:
    tracker = _tracker(tmp_path)
    cost = tracker.record(provider="p", model=_MODEL, input_tokens=1, output_tokens=0)
    with pytest.warns(DeprecationWarning, match=r"get_per_provider_totals\(now=\.\.\.\)"):
        assert tracker.get_per_provider_totals("p", now=4_000_000_000.0)["hourly"] == cost


def test_none_still_means_the_wall_clock(tmp_path: Path) -> None:
    tracker = _tracker(tmp_path)
    cost = tracker.record(provider="p", model=_MODEL, input_tokens=1, output_tokens=0)
    assert tracker.get_totals()["hourly"] == cost
    assert tracker.get_per_provider_totals("p")["hourly"] == cost


def test_salience_network_reads_on_the_callers_clock() -> None:
    """A detection made at 0.0 and read at 0.0 is brand new: fully novel and among the top salient."""
    net = SalienceNetwork()
    det = {"track_id": 7, "class_id": 0, "label": "cup", "confidence": 0.9, "bbox": (1, 1, 2, 2)}
    net.update_from_detections([det], timestamp=0.0)
    assert net.get_novelty(7, now=0.0) == 1.0
    assert [o["track_id"] for o in net.get_top_salient(1, now=0.0)] == [7]
    assert "cup" in net.to_context_str(now=0.0)
    (near,) = net.get_objects_at_position((2.0, 2.0), now=0.0)
    assert near["salience"] == net.get_salience(7, now=0.0)


def test_spend_rate_starting_at_zero_matches_any_other_start() -> None:
    """SpendRate read last_update == 0.0 as "no earlier sample", so a first sample at 0.0 doubled the next rate."""
    from maxim.models.language.cost_tracker import SpendRate

    def run(t0: float) -> float:
        rate = SpendRate()
        rate.update(1.0, t0, alpha=0.5)
        rate.update(1.0, t0 + 3600.0, alpha=0.5)
        return rate.value

    assert run(0.0) == run(100.0)
