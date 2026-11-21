The agent loop's LLM proposal intake (section 2 of `run_agentic_loop`'s body) moved verbatim into
`runtime/loop_intake.py` (`take_llm_proposal`, `take_queued_next_action`), with `_drop_stale_proposal` and
`_proposal_without_action_reason`; behaviour is unchanged and pinned by a new characterization through the
public loop. The body shrinks from 2,140 to 1,934 lines (1.3.6 loop phase 2, slice 6; #1323).
