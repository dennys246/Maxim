- **The first half of the agent loop's LLM submit moved into `runtime/loop_submit.py`** (#1323): section 6 of
  `run_agentic_loop`'s body (the tick's context, its Wire-A and grayscale annotations, and the gate that decides
  whether anything is worth a submission) moved verbatim into `prepare_submission`, with `_take_deferred_inputs` and
  `_followup_synthetic_input`. Behaviour is unchanged and pinned by a new characterization through the public loop.
  The body shrinks from 1,773 to 1,453 lines (1.3.6 loop phase 2, slice 7a).
