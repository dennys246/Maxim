"""The agent loop's substrate tick, section 6b (1.3.2 decomposition, slice 3).

``substrate_tick`` is the body of ``agent_loop.run_agentic_loop``'s substrate-primary branch, moved
verbatim: the turn-scoped action gate (apparatus standard S6), the proposal from
``substrate_proposal.propose_via_substrate`` (no LLM call), the submit-clock advance, installing the
proposal as ``ctrl.pending_proposal`` with its sim EXEC line, and the Phase 0 telemetry snapshot. The
branch's ``if`` stays at the call site: ``loop_state._substrate_tick_due`` is ONE predicate, shared with
the idle gate (``loop_gates.pre_tick_gate``), and it is visible where it applies.

**Arguments.** Explicit keywords, never the loop's ``LoopRun`` (rule (d) of the roadmap's
import-direction paragraph): exactly the names the block reads, each the loop's own local passed as-is
and named as the body has always named it (``_loop_nac`` is ``LoopRun.nac``, ``_loop_agent_id``
``LoopRun.agent_id``, ``_loop_situation_cue`` ``LoopRun.situation_cue``, ``_loop_sensor_encoder``
``LoopRun.sensor_encoder``), so the body is byte for byte the inline block's. The one edit: the proposer
is called through the module reference, ``_sp.propose_via_substrate(...)``, so a test that replaces it
patches ONE place, ``substrate_proposal.propose_via_substrate``. The tick LOGS on the
``maxim.runtime.agent_loop`` logger (``logger`` below is that same object), so its records keep the
agent loop's logger name.

Characterization: ``tests/unit/test_loop_substrate_characterization.py`` (written before the move,
through the public ``run_agentic_loop``, and kept green by it).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from maxim.runtime import substrate_proposal as _sp

if TYPE_CHECKING:
    from maxim.runtime.loop_controller import LoopController

# The SAME logger object as ``agent_loop.logger`` (logging returns one logger per name), so records keep
# the ``maxim.runtime.agent_loop`` name.
logger = logging.getLogger("maxim.runtime.agent_loop")


def substrate_tick(
    *,
    step_num: int,
    ctrl: LoopController,
    executor: Any,
    sim: Any,
    memory_hub: Any | None,
    substrate_action_gate: Any | None,
    substrate_telemetry: Any | None,
    _loop_nac: Any,
    _loop_agent_id: str,
    _loop_situation_cue: Any,
    _loop_sensor_encoder: Any | None,
) -> None:
    """Section 6b of one pass of ``run_agentic_loop``, on a pass where the substrate cadence is due."""
    now = time.time()
    # Turn-scoped action budget (apparatus standard S6; the Exp 48
    # thrashing fix). A denied tick skips the proposal — the AUT
    # idles until the orchestrator opens the next turn window —
    # but still advances last_llm_submit_time and fires telemetry
    # (proposal=None, gated=True) so the cadence stays observable
    # AND gate-idle is distinguishable from substrate-no-opinion
    # IDLE in the telemetry artifact itself (review fold — the
    # once-per-window sim_log line alone marks the window, not
    # the rows). Drive drift is unaffected: it is wall-clock-lazy
    # and the next propose_via_substrate applies the accumulated dt.
    _substrate_gate_denied = substrate_action_gate is not None and not substrate_action_gate()
    substrate_proposal = None
    if not _substrate_gate_denied:
        substrate_proposal = _sp.propose_via_substrate(
            nac=_loop_nac,
            agent_id=_loop_agent_id,
            executor=executor,
            situation_cue=_loop_situation_cue,
            sensor_encoder=_loop_sensor_encoder,
        )
    ctrl.last_llm_submit_time = now
    if substrate_proposal is not None:
        ctrl.pending_proposal = substrate_proposal
        if sim.is_sim_mode:
            sim.log(
                "EXEC",
                f"substrate-primary proposal: tool="
                f"{substrate_proposal.action.get('tool_name') if substrate_proposal.action else None} "
                f"confidence={substrate_proposal.confidence:.2f} "
                f"reasoning={substrate_proposal.reasoning[:80]}",
            )

    # Phase 0 telemetry — fires every tick (proposal or
    # IDLE). Fail-soft: telemetry exceptions never crash
    # the loop. See simulation/substrate_telemetry.py.
    if substrate_telemetry is not None:
        try:
            _ec_ref = getattr(memory_hub, "ec", None) if memory_hub is not None else None
            substrate_telemetry.snapshot(
                step=step_num,
                nac=_loop_nac,
                ec=_ec_ref,
                executor=executor,
                proposal=substrate_proposal,
                gated=_substrate_gate_denied,
            )
        except Exception:
            logger.debug("substrate telemetry callback raised", exc_info=True)
