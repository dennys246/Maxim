- **The rest of the agent loop's LLM submit moved into `runtime/loop_submit.py`** (#1323): the request assembly of
  section 6 (the mode's `ModeInfo`, the input filter, the tool roster with the SEM union, Wire 3's filter and the
  permits filter, the tool descriptions with Wire 3's felt phrases and Wire 1's variance, the prefetch, the submit and
  its logs) and the PFC deliberation after it moved verbatim into `submit_tick`, which now holds the whole submit
  `try`. Behaviour is unchanged and pinned by a new characterization through the public loop. The body shrinks from
  1,603 to 1,226 lines (1.3.6 loop phase 2, slice 7b).
