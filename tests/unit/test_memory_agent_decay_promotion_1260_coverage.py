"""MemoryAgent's salience decay and long-term promotion, exercised directly (#1260 coverage).

Both are wall-clock gated: ``_apply_decay`` runs once per ``decay_interval`` and drops a memory's local tracking when
its salience falls to the threshold; ``_check_promotions`` runs at most every 10 s and marks a Hippocampus record
``long_term`` when its tracked salience is at least 0.7. Before #1260 the orchestrator golden's runs were long enough
(each AUT ``respond`` was followed up) that both fired by chance, and that was their only coverage. These pin them on
purpose, with a real ``MemoryAgent`` and a real ``Hippocampus``, by moving the agent's last-run stamps back instead of
waiting. The last two tests pin two neighbours no test reached at all (the Hippocampus deletion callback's cleanup and
the person-in-view salience bonus), so ``maxim/agents`` gets back to its floor's pinned missing count.
"""

from __future__ import annotations

import time

import pytest

from maxim.agents.bus import AgentBus
from maxim.agents.memory_agent import MemoryAgent
from maxim.memory.hippocampus import Hippocampus


@pytest.fixture
def agent() -> MemoryAgent:
    memory_agent = MemoryAgent(AgentBus(), salience_threshold=0.2, decay_interval=1.0)
    memory_agent.connect_hippocampus(Hippocampus())
    return memory_agent


def _capture(agent: MemoryAgent, text: str, salience: float, decay_rate: float = 0.1) -> str:
    memory_id = agent._capture_to_hippocampus({"text": text}, salience, decay_rate, "test")
    assert memory_id is not None
    return memory_id


def test_decay_drops_a_memory_that_falls_to_the_threshold_and_keeps_a_salient_one(agent: MemoryAgent) -> None:
    faint = _capture(agent, "a faint hum somewhere", salience=0.25, decay_rate=0.1)
    vivid = _capture(agent, "the kettle is boiling over", salience=0.9, decay_rate=0.1)
    agent._last_decay_time = time.time() - 2.0  # two seconds since the last decay: 0.2 off each

    agent._apply_decay()

    assert faint not in agent._salience and faint not in agent._decay_rates  # 0.25 - 0.2 <= 0.2: dropped
    assert faint not in [mid for mid, _ in agent._association_index.find_similar("faint hum")]
    assert agent._salience[vivid] == pytest.approx(0.7, abs=0.02)  # decayed, still tracked
    assert vivid in [mid for mid, _ in agent._association_index.find_similar("kettle boiling")]
    assert agent._hippocampus.get(faint) is not None  # decay drops local tracking only, never the record


def test_decay_waits_for_its_interval(agent: MemoryAgent) -> None:
    faint = _capture(agent, "a faint hum somewhere", salience=0.25, decay_rate=0.1)
    agent._last_decay_time = time.time()  # just decayed

    agent._apply_decay()

    assert agent._salience[faint] == 0.25


def test_promotion_marks_a_salient_record_long_term_and_leaves_the_rest(agent: MemoryAgent) -> None:
    salient = _capture(agent, "the dog bit my hand", salience=0.9)
    mild = _capture(agent, "a grey wall", salience=0.5)
    agent._last_promotion_time = time.time() - 11.0
    before = time.time()

    agent._check_promotions()

    record = agent._hippocampus.get(salient)
    assert record.long_term is True and record.consolidated_at >= before
    assert agent._hippocampus.get(mild).long_term is False
    assert agent._last_promotion_time >= before  # the next pass waits another 10 s


def test_promotion_waits_ten_seconds_between_passes(agent: MemoryAgent) -> None:
    salient = _capture(agent, "the dog bit my hand", salience=0.9)
    agent._last_promotion_time = time.time() - 5.0

    agent._check_promotions()

    assert agent._hippocampus.get(salient).long_term is False


def test_promotion_without_a_hippocampus_only_advances_its_clock() -> None:
    memory_agent = MemoryAgent(AgentBus())
    memory_agent._salience["orphan"] = 0.95
    memory_agent._last_promotion_time = time.time() - 11.0
    before = time.time()

    memory_agent._check_promotions()

    assert memory_agent._last_promotion_time >= before


def test_a_hippocampus_deletion_clears_the_agents_tracking(agent: MemoryAgent) -> None:
    """``MaximAgent.wire_memory_hub`` registers ``_on_memory_deleted`` on the Hippocampus; a pruned record must not
    stay in the agent's salience, decay or keyword tracking."""
    agent._hippocampus.register_deletion_callback(agent._on_memory_deleted)  # as MaximAgent.wire_memory_hub does
    gone = _capture(agent, "a broken cup on the floor", salience=0.8)
    kept = _capture(agent, "the window is open", salience=0.8)

    agent._hippocampus.remove(gone)

    assert gone not in agent._salience and gone not in agent._decay_rates
    assert gone not in [mid for mid, _ in agent._association_index.find_similar("broken cup floor")]
    assert kept in agent._salience


def test_a_person_in_view_raises_a_percepts_memory_salience_once() -> None:
    from maxim.agents.bus import Percept

    memory_agent = MemoryAgent(AgentBus())
    people = [{"class_id": 2}, {"class_id": 0}, {"class_id": 0}]  # a car, then two people: one bonus, not two
    assert memory_agent._compute_memory_salience(
        Percept(timestamp=0.0, source="t", salience=0.5, detections=people)
    ) == (pytest.approx(0.6))
    no_people = Percept(timestamp=0.0, source="t", salience=0.5, detections=[{"class_id": 2}])
    assert memory_agent._compute_memory_salience(no_people) == pytest.approx(0.5)
