- **Gating prompts no longer read a refusal as a "yes"** (#1185): the plan-approval parser approved "stop, don't
  run it", "not ok", "that is not fine" and "no good", read "No." as unknown, and flipped mixed replies with
  `PYTHONHASHSEED`. Plan approval, action confirmation and the timeout-retry prompt now share one reader,
  `runtime/approval.py::classify_yes_no`:
  - only a whole reply of `yes`, `y` or `approve` approves, and only `no`, `n`, `reject`, `cancel` or `stop`
    rejects (case and surrounding punctuation ignored, so "Yes." and "No!" work at every prompt);
  - "ok", "sure", "go ahead", "lgtm", "yes please" and similar now ask again, as do negations that are not a
    plain no ("not yet", "don't", "no problem", "wait"), questions ("yes?") and emoji;
  - a re-asked prompt is shown again (in a simulation, on the simulated person's channel too) and the reply is
    not sent to the model as a new turn;
  - the timeout-retry prompt reads yes, a whole number of minutes from 1 to 10, or no, and asks again for
    anything else; it no longer offers "skip", and "skip", "ok" and "sure" are no longer answers. A timeout
    retry is no approval: when nobody can answer it (a non-interactive run, or a sim with no response policy,
    including an interactive scenario turn), the loop answers it itself, "yes" at most twice in a row and then
    "no", which ends the turn visibly ("Giving up on this turn"), counts toward the loop's LLM-failure count and, in
    a sim, records the LLM-unavailable marker. A person in an interactive run (not a sim) is still asked. This
    changes production: a headless run (`maxim.api interactive=False`, the console handle, Roy) used to park a
    timeout and never retry, and a sim with no policy used to retry without limit; both now retry twice, then
    stop. Any LLM turn that is not a timeout resets the count;
  - a tool result that arrives while a prompt is waiting is never read as its answer;
  - in a simulation, the response policy's answer to a pending confirmation is no longer dropped when the
    narrator's next sentence arrives in the same tick; the answer goes first and the sentence is the next turn,
    with its memory record and recall cue kept;
  - inputs that arrive together are taken one per tick, oldest first: newer input no longer overtakes words
    waiting behind a prompt's answer, and a voice turn that arrives with typed input is no longer dropped;
  - a re-asked prompt shown to a simulated person no longer counts as an action. Report action totals, tool
    counts and success rates, the substrate action budget and the stall detector now count only real actions:
    every loop-made marker (`ActionRecord.is_synthetic`: `_gating_prompt`, `_llm_unavailable`,
    `_deliberation_noop`) is left out. A run that recorded any such marker is not comparable on those numbers with
    earlier reports; `actions.jsonl` stays complete.
- **A simulation adapter no longer answers "yes" on its own** (#1723): `SimulationAdapter.resolve_plan_approval`,
  `resolve_confirmation` and `resolve_timeout_retry` answered "yes" whenever the bridge's policy could not be
  reached, which in production was always (the adapter was handed the bridge's percept source, never the bridge),
  and also swallowed any error from the policy into that "yes". They now ask the policy the sim passes explicitly,
  a failing policy's error propagates, and an adapter with no policy answers nothing.
- **`ResponsePolicy` ASK_ORCHESTRATOR is removed** (#1736): `PolicyType.ASK_ORCHESTRATOR` and
  `simulation.response_policy.ask_orchestrator()` never had a route to the orchestrator, so the loop answered in
  their place. Nothing constructed them. A response policy passed to `run_agentic_loop` without a
  `percept_source` is now a `TypeError` instead of being dropped.
- **A sim policy's timeout "yes" no longer runs a parked action**: with a confirmation and a timeout retry both
  pending, the reply was routed to its prompt only when the loop itself made it, so the policy's "yes" ran the parked
  action and left the retry stranded. Every loop reply now carries the prompt it answers. A parse failure
  (`llm_fallback`) counts as a failed LLM turn, and the loop's own "no" is no longer reported as a person's.
