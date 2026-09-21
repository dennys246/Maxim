"""#839 — a memory store that EXISTS is truthy even when EMPTY.

Hippocampus, ATL, AngularGyrus, EC, NAc and SCN define `__len__`, so without `__bool__` an empty store was falsy and
every `if store:` / `if not store:` (21 sites) skipped it until something else wrote its first
entry. The fix is at the type, so the next such site is covered too.
"""

from __future__ import annotations

import pytest

from maxim.decisions.nac import NAc
from maxim.math.angular_gyrus import AngularGyrus
from maxim.memory.atl import ATL
from maxim.memory.hippocampus import Hippocampus
from maxim.similarity.ec import ECConfig, EntorhinalCortex
from maxim.time.scn import SCN

# Hippocampus, ATL and AngularGyrus get __bool__ from MemoryLayer; EC, NAc and SCN define their own.
EMPTY_STORES = {
    "Hippocampus": Hippocampus.empty,
    "ATL": ATL,
    "AngularGyrus": AngularGyrus,
    "EC": lambda: EntorhinalCortex(ECConfig()),
    "NAc": NAc,
    "SCN": SCN,
}


@pytest.mark.parametrize("name", sorted(EMPTY_STORES))
def test_a_fresh_store_is_truthy_whatever_its_length(name: str) -> None:
    store = EMPTY_STORES[name]()
    # Most start empty (len 0); AngularGyrus starts with built-in entries. Either way a present
    # store is truthy -- emptiness is observed with len(), never with bool().
    assert bool(store) is True


@pytest.mark.parametrize("name", ["Hippocampus", "ATL", "EC", "NAc", "SCN"])
def test_the_stores_that_start_empty_are_still_truthy(name: str) -> None:
    store = EMPTY_STORES[name]()
    assert len(store) == 0 and bool(store) is True


def test_memory_layers_share_one_definition() -> None:
    from maxim.memory.layer import MemoryLayer

    for cls in (Hippocampus, ATL, AngularGyrus):
        assert cls.__bool__ is MemoryLayer.__bool__, cls.__name__


def test_memory_agent_captures_into_an_empty_hippocampus() -> None:
    """The write path the bug broke: MemoryAgent's guard was `if not self._hippocampus: return None`."""
    from maxim.agents.bus import AgentBus
    from maxim.agents.memory_agent import MemoryAgent

    agent = MemoryAgent(AgentBus())
    hippocampus = Hippocampus.empty()
    agent.connect_hippocampus(hippocampus)

    memory_id = agent._add_memory({"note": "the very first memory"}, 0.7, 0.05, "percept")

    assert memory_id is not None, "the first capture into an empty hippocampus was dropped"
    assert len(hippocampus) == 1
