"""Three-mode multi-agent fixture for per-agent state isolation tests.

Companion to the ``multi_agent_modes`` pytest marker registered in
``pyproject.toml``. Lives outside ``conftest.py`` so the fixture is
import-locatable (lint script greps for it) and the conftest stays
focused on cross-cutting test setup.

The fixture turns the P4 multi-agent rule (CLAUDE.md L43 — *"any
per-agent runtime stash MUST be a ``dict[agent_id, value]`` from day
one"*) into a forcing function on every new test that touches per-agent
state. Tests marked ``@pytest.mark.multi_agent_modes`` and that use the
``multi_agent_modes`` fixture parameter are auto-parametrized over
three modes:

1. ``single_agent`` — baseline; one agent, default path.
2. ``two_agents_shared_instance`` — **TRIPWIRE**. Two agents share ONE
   bio-system instance. **NOT a production case.** Production uses
   separate instances per ``AgentFactory``. The mode exists ONLY to
   catch the silent-merge bug class: if per-agent state isolation
   accidentally regresses (someone forgets the ``agent_id`` key, drops
   the ``_check_agent_id`` validator, etc.), the shared-instance
   trajectory will smear state across agents and the test will see
   wrong attribution. Do NOT design features around this mode's
   behavior — it is a diagnostic surface, not an API contract.
3. ``two_agents_isolated_instances`` — production case. Each agent owns
   its own bio-system instance. Isolation is structurally guaranteed
   by the separate objects; the test confirms attribution still flows
   through correctly even when the structural guarantee is the only
   thing keeping the agents apart.

If you're writing a NEW test that touches per-agent state, declare the
marker on the test:

.. code-block:: python

    @pytest.mark.multi_agent_modes
    def test_my_new_per_agent_thing(multi_agent_modes):
        for agent_id in multi_agent_modes.agent_ids:
            nac = multi_agent_modes.nac_for(agent_id)
            ...

If your test is genuinely single-agent and shouldn't run 3×, use the
opt-out marker:

.. code-block:: python

    @pytest.mark.single_agent_only
    def test_genuinely_single_agent_thing():
        ...

The CI lint at ``scripts/lint_multi_agent_marker.py`` rejects new tests
that touch per-agent state but declare neither marker. The lint is
heuristic (grep on field references); false positives are addressed by
declaring ``single_agent_only`` explicitly.

See CLAUDE.md L43 (P4 multi-agent rule) and
``docs/plans/archive/structural_invariant_tests.md`` Stage 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

# Three-mode identifiers. Kept module-level so callers + the CI lint can
# reference them without re-deriving.
MODE_SINGLE = "single_agent"
MODE_SHARED = "two_agents_shared_instance"
MODE_ISOLATED = "two_agents_isolated_instances"

ALL_MODES = (MODE_SINGLE, MODE_SHARED, MODE_ISOLATED)


@dataclass
class MultiAgentSetup:
    """Yielded by the ``multi_agent_modes`` fixture.

    ``mode`` is one of ``MODE_SINGLE``, ``MODE_SHARED``, ``MODE_ISOLATED``.
    ``agent_ids`` lists the agents in this mode (1 for single, 2 otherwise).
    ``nac_for(agent_id)`` returns the NAc instance backing that agent — in
    ``MODE_SHARED`` this returns the same instance for both agent_ids; in
    ``MODE_ISOLATED`` each agent has its own NAc.
    """

    mode: str
    agent_ids: tuple[str, ...]
    _nacs: dict[str, Any]

    def nac_for(self, agent_id: str) -> Any:
        return self._nacs[agent_id]

    @property
    def is_shared_instance_tripwire(self) -> bool:
        """True iff this is the diagnostic shared-instance mode.

        Tests can branch on this to apply different assertion shapes:
        in shared-instance mode, isolation is proven via
        ``event_context["agent_id"]`` filtering on a single underlying
        accumulator; in isolated-instances mode, isolation is
        structurally trivial.
        """
        return self.mode == MODE_SHARED


def _build_nac() -> Any:
    """Lazy import — keeps `pytest --collect-only` cheap."""
    from maxim.decisions.nac import NAc, NACConfig

    return NAc(NACConfig(temporal_window_seconds=60, max_pending_events=2000))


@pytest.fixture(params=ALL_MODES, ids=lambda m: m)
def multi_agent_modes(request: pytest.FixtureRequest) -> MultiAgentSetup:
    """Parametrize a test across single_agent / shared / isolated modes.

    See module docstring for the load-bearing TRIPWIRE warning on the
    ``two_agents_shared_instance`` mode.
    """
    mode = request.param

    if mode == MODE_SINGLE:
        nac = _build_nac()
        return MultiAgentSetup(
            mode=mode,
            agent_ids=("agent_solo",),
            _nacs={"agent_solo": nac},
        )

    if mode == MODE_SHARED:
        # ONE NAc instance shared by both agents — the tripwire.
        # Isolation depends entirely on (agent_id, ...) keyed access.
        shared = _build_nac()
        return MultiAgentSetup(
            mode=mode,
            agent_ids=("agent_alpha", "agent_bravo"),
            _nacs={"agent_alpha": shared, "agent_bravo": shared},
        )

    if mode == MODE_ISOLATED:
        # Each agent gets its own NAc — production AgentFactory shape.
        return MultiAgentSetup(
            mode=mode,
            agent_ids=("agent_alpha", "agent_bravo"),
            _nacs={"agent_alpha": _build_nac(), "agent_bravo": _build_nac()},
        )

    raise ValueError(f"unknown mode {mode!r}")
