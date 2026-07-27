"""MaximHandle — one persistent agent behind the Console (HANDLE seam).

The headless flavor of the HANDLE seam ([reachy_app_maxim_seams.md] § HANDLE,
design in [console_handle_campaign_injection.md]): a small wrapper over ONE
persistent ``AgentInstance`` built via ``AgentFactory.create_full_agent``
with ``auto_load=True`` over a ``~/.maxim`` home. Modes are methods:

* ``play_campaign(path)`` — run a DM campaign **as the persistent agent**
  (the campaign-injection surface: ``start_simulation_mode(dm_campaign=…,
  persistent_agent=self.instance)``). Adventure teaches Talk — the episodes
  land in the persistent agent's own Hippocampus/NAc, not a throwaway.
* ``stop(consolidation="full")`` — clean shutdown: full sleep/replay
  consolidation + hippocampus/NAc/cerebellum saves (#427's explicit flavor).

``talk(...)`` / ``rest(...)`` live-loop modes and the ``/ws`` ``api.on()``
stream are Phase 3 — this module deliberately ships only injection + stop.

The constructor is **body-agnostic**: the embodied (Reachy) flavor is the
same interface with ``body="bodies/reachy_mini"`` — ``RunSurface`` drives a
HANDLE, not "a robot".
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


class MaximHandle:
    """A live handle on one persistent Maxim agent.

    Args:
        agent_id: The persistent identity episodes attribute to. Must NOT be
            ``"sim_aut"`` (that id marks the throwaway sim AUT).
        home: Persistence directory for the agent. Default: the factory's
            ``~/.maxim/agents/<agent_id>`` home.
        body: Optional SEM component ref (e.g. ``"bodies/reachy_mini"``) for
            the embodied flavor. The handle wires it at CONSTRUCTION
            (``AgentConfig.embodiment_ref``) — never via the sim's
            ``entity_ref``, which is incompatible with injection.
    """

    def __init__(
        self,
        agent_id: str = "console_agent",
        *,
        home: str | Path | None = None,
        body: str | None = None,
    ) -> None:
        from maxim.runtime.agent_factory import AgentConfig, AgentFactory
        from maxim.runtime.bootstrap import build_tool_registry

        if agent_id == "sim_aut":
            raise ValueError("agent_id 'sim_aut' is reserved for the throwaway sim AUT")

        component_registry: Any | None = None
        if body is not None:
            from maxim.embodiment.component_registry import ComponentRegistry

            component_registry = ComponentRegistry()

        config = AgentConfig(
            agent_id=agent_id,
            role="pc",
            persistence_dir=str(home) if home is not None else None,
            with_bio_stack=True,
            with_executor=True,
            # Mirror the sim AUT's pain-bridge decision: subscription only
            # matters when an embodiment can publish SEM pain.
            with_pain_bridge=body is not None,
            with_fear_gate=False,
            embodiment_ref=body,
        )
        factory = AgentFactory(component_registry=component_registry)
        tool_registry = build_tool_registry(operational_mode="active")
        self.instance = factory.create_full_agent(
            config,
            tool_registry=tool_registry,
            auto_load=True,
        )
        self.agent_id = agent_id
        self._stopped = False
        # One campaign at a time — the persistent substrate is not
        # re-entrant across concurrent sims (shared Hippocampus/NAc).
        self._campaign_lock = threading.Lock()

    # ── modes ───────────────────────────────────────────────────────────

    def play_campaign(self, path: str | Path, *, max_turns: int = 100) -> Any:
        """Run a DM campaign as the persistent agent; blocks until it ends.

        Returns the ``SimulationResult``. The campaign's episodes are
        recallable from ``self.instance.hippocampus`` afterwards — that is
        the seam's whole point.
        """
        if self._stopped:
            raise RuntimeError("MaximHandle is stopped — build a new handle to play again")
        campaign_path = Path(path)
        if not campaign_path.exists():
            raise FileNotFoundError(f"Campaign not found: {campaign_path}")

        from maxim.embodiment.component_registry import ComponentRegistry
        from maxim.simulation.dm_schema import load_campaign, validate_campaign

        registry = ComponentRegistry(campaign_dir=str(campaign_path.parent))
        campaign = load_campaign(campaign_path, registry=registry)
        errors = validate_campaign(campaign)
        if errors:
            raise ValueError(f"Campaign validation failed ({len(errors)} errors): {errors}")

        if not self._campaign_lock.acquire(blocking=False):
            raise RuntimeError("A campaign is already running on this handle (one at a time)")
        try:
            # Headless surface: force non-interactive so the sim never grabs
            # stdin (the console serves this from a FastAPI worker thread).
            # Prior mode is restored on exit — a library user embedding the
            # handle next to interactive CLI usage keeps their setting.
            from maxim.simulation.sim_logger import (
                InteractiveMode,
                get_interactive_mode,
                set_interactive_mode,
            )

            _prior_mode = get_interactive_mode()
            set_interactive_mode(InteractiveMode.OFF)

            from maxim.simulation.orchestrator import _CampaignToolLease, start_simulation_mode

            # Exception-path safety net (three-lens review, cross-confirmed
            # Executor #2 / Architecture #2): the orchestrator restores its
            # tool lease on the normal path, but a mid-sim raise would
            # otherwise leave the LIVE persistent registry stripped of its
            # own tools + polluted with sim tools for every later mode.
            # restore() is an idempotent snapshot-diff, so running it again
            # after a clean finish is a no-op.
            _safety_lease = _CampaignToolLease.snapshot(self.instance.tool_registry)
            try:
                return start_simulation_mode(
                    goal=f"dm:{campaign.name}",
                    persona="dungeon_master",
                    dm_campaign=campaign,
                    max_turns=max_turns,
                    persistent_agent=self.instance,
                )
            finally:
                dropped, restored = _safety_lease.restore(self.instance.tool_registry)
                if dropped or restored:
                    logger.warning(
                        "Campaign exited without a clean lease restore — safety net dropped %d "
                        "leaked campaign tools and re-registered %d persistent tools",
                        len(dropped),
                        len(restored),
                    )
                set_interactive_mode(_prior_mode)
        finally:
            self._campaign_lock.release()

    # ── lifecycle ───────────────────────────────────────────────────────

    def stop(
        self,
        *,
        consolidation: Literal["full", "lightweight"] = "full",
        campaign_wait_s: float = 60.0,
    ) -> None:
        """Shut the persistent agent down with an EXPLICIT consolidation flavor.

        ``"full"`` (default): blocking sleep/replay consolidation +
        hippocampus/NAc/cerebellum saves — the right flavor for a persistent
        agent. ``"lightweight"`` skips the replay (still persists state).
        Idempotent: a second stop is a no-op.

        Waits up to ``campaign_wait_s`` for a live campaign to finish (post-
        merge review, Exec #4 / Arch #1: stopping under a live loop raced its
        own session-end — the MemoryHub's atomic test-and-clear now guarantees
        single consolidation either way, but stopping mid-campaign would still
        steal the loop's consolidation slot, so we wait). On expiry, proceeds
        LOUDLY — a hung campaign must not make stop() unbounded.
        """
        if self._stopped:
            return
        acquired = self._campaign_lock.acquire(timeout=campaign_wait_s)
        if not acquired:
            logger.warning(
                "MaximHandle.stop(): campaign still running after %.0fs wait — "
                "proceeding with shutdown; the campaign loop's own session-end "
                "becomes a no-op (atomic session flag)",
                campaign_wait_s,
            )
        try:
            if self._stopped:  # settled while we waited
                return
            self._stopped = True
            self.instance.shutdown(consolidation=consolidation)
        finally:
            if acquired:
                self._campaign_lock.release()
