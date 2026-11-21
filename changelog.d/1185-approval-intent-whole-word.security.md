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
    anything else; it no longer offers "skip", and "skip", "ok" and "sure" are no longer answers;
  - a tool result that arrives while a prompt is waiting is never read as its answer;
  - in a simulation, the response policy's answer to a pending confirmation is no longer dropped when the
    narrator's next sentence arrives in the same tick; the answer goes first and the sentence is the next turn.
- **A failing simulation response policy no longer approves** (#1723): `SimulationAdapter.resolve_plan_approval`,
  `resolve_confirmation` and `resolve_timeout_retry` swallowed any error from the bridge's policy and answered
  "yes"; the error now propagates.
