"""#1454 (owner decision 2026-10-26): the post-sim ``/new`` prompt promises only what the run does.

1.3.4 dropped "(memory carries over)" because the re-entered run built a fresh throwaway AUT. The 1.3.5 ``/new`` slice
makes the learning carry, so the promise is back: this pin was re-pointed at the constant, deliberately, in that change.
What makes it true is pinned by ``test_new_slice_1453_1454_1457.py``.
"""

from __future__ import annotations

from maxim.simulation.tty_stdin import POST_SIM_PROMPT


def test_the_post_sim_prompt_promises_the_carried_memory_1454() -> None:
    assert POST_SIM_PROMPT.startswith("Tell me a new simulation to imagine (memory carries over)")
