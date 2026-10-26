"""#1454 (interim, owner decision 2026-10-26): the post-sim ``/new`` prompt does not promise what the run does not do.

The interactive post-sim prompt read "Tell me a new simulation to imagine (memory carries over)", but the re-entered
run builds a fresh throwaway AUT in its own ``sim_agent_*`` home, so the first run's learning does not reach it (pinned
as current behaviour by the ``new_goal`` orchestrator characterization arm). 1.3.4 drops the claim. The 1.3.5 ``/new``
slice makes the learning carry (owner decision on #1454) and restores the text in the same PR, so this pin is then
re-pointed at the behaviour, deliberately, in that reviewed change.
"""

from __future__ import annotations

import pytest

from tests.unit._orchestrator_source import orchestrator_source


@pytest.mark.xfail(strict=True, reason="#1454: the post-sim prompt promises '(memory carries over)'")
def test_the_post_sim_prompt_does_not_promise_carried_memory_1454() -> None:
    src = orchestrator_source()
    assert "Tell me a new simulation to imagine" in src  # the prompt is still there, and read from its new home
    assert "memory carries over" not in src
