- **An action-less answer from the PFC deliberation no longer wedges the loop** (#1743): a cycle-1 "ready" answer, or
  the answer cycles 2+ settled on, was installed as the pending proposal past §2's intake checks, so one with no
  action stayed pending forever and the loop never submitted again, not even for a fresh input. Both now go through
  §2's one intake (`loop_intake.accept_llm_proposal`): only an answer with an action is installed, a stale answer or
  sim-skipped fallback is dropped, the turn is counted for the stall detector, the pending user input is cleared, and
  the answer's `next_actions` are queued (they were silently dropped). Under planning liveness a declined
  deliberation turn is dropped, not retried as a planning failure.
