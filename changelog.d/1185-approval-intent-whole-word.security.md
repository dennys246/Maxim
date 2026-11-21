- **Gating prompts no longer read a refusal as a "yes"** (#1185): the plan-approval parser approved "stop, don't
  run it", "not ok", "that is not fine" and "no good", read "No." as unknown, and flipped mixed replies with
  `PYTHONHASHSEED`. Plan approval, action confirmation and the timeout-retry prompt now share one reader,
  `runtime/approval.py::classify_yes_no`:
  - only a whole reply of `yes`, `y` or `approve` approves, and only `no`, `n`, `reject`, `cancel` or `stop`
    rejects (case and surrounding punctuation ignored, so "Yes." and "No!" work at every prompt);
  - "ok", "sure", "go ahead", "lgtm", "yes please" and similar now ask again, as do negations that are not a
    plain no ("not yet", "don't", "no problem", "wait"), questions ("yes?") and emoji;
  - at a plan or confirmation prompt the re-ask is shown again and the reply is no longer sent to the model as a
    new turn; free text with no yes/no word is still a change request, and a plan change request now reaches
    the model with the original action attached;
  - at the timeout prompt "skip", "ok" and "sure" are no longer read as an answer.
- **A failing simulation response policy no longer approves** (#1723): `SimulationAdapter.resolve_plan_approval`,
  `resolve_confirmation` and `resolve_timeout_retry` swallowed any error from the bridge's policy and answered
  "yes"; the error now propagates.
