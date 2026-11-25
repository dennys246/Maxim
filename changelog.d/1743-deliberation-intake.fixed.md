- **An action-less answer from the PFC deliberation no longer wedges the loop** (#1743): a cycle-1 "ready" answer, or
  the answer cycles 2+ settled on, was installed as the pending proposal past §2's intake checks, so one with no
  action stayed pending forever and the loop never submitted again, not even for a fresh input. Every answer the
  deliberation consumes now goes through §2's one intake (`loop_intake.accept_llm_proposal`): only an answer with an
  action is installed, a stale answer (one an early-stopped deliberation fell back to) or a sim-skipped fallback is
  dropped, the turn is counted for the stall detector, the pending user input is cleared on an answer with an action
  or an error, and the answer's `next_actions` are queued (they were silently dropped). The deliberation's wait now
  also resets the planning transport-failure streak, as §2's poll does, and a wait that brings nothing back closes
  the submit's `inference_start` event.
  - A planning retry budget spent anywhere now aborts the run on the next pass (`loop_gates.pre_tick_gate` reads the
    latch), so a budget spent in the deliberation's intake cannot leave the loop idling. Under planning liveness a
    declined deliberation turn is dropped, not retried; no shipped loop combines liveness with a deliberation today.
  - A turn that ends with no action and no retry (a declined deliberation, or an action-less answer with planning
    liveness off) now records the sim's no-op marker, so the bridge ends that turn after its settle window instead
    of waiting out the 120 s response timeout. It still reports `timed_out` and counts no action; the turn's
    `noop_count` is 1. The marker existed since 0.8 but nothing recorded it.
