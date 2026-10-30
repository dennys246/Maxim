# #1481 pre-merge code review: Architecture lens

Commit `fc69e312`, branch `fix/1481-inbound-comms-prompt`. Read-only review against docs/CODE_REVIEW.md,
docs/agents/runtime-tools.md, ARCHITECTURE.md and docs/plans/inbound_comms_1481_approach.md (v3.3).

**Verdict: no BLOCKER. 6 SHOULD-FIX, 7 NIT.**

## What was checked and holds

- **Layering.** `maxim --audit-architecture`: "No new, stale, or unreviewed findings." `agents/` gains no `runtime`
  import (the `InboundFeed` Protocol is in `agents/llm_types.py`; `prompt_builder` imports `utils.content_safety`
  only). `loop_external` imports only `maxim.agents.bus` (lazily) and never `agent_loop`; `loop_planning` gains only
  stdlib imports. `llm_worker`'s `runtime.worker_pool` imports predate this diff.
- **A's fence.** The `agent_loop` hunks are: imports, the Wire 1 constants, `_release_unsent_deferred_inputs` (moved
  verbatim, grant 1), the wrapper's `with`, one body parameter, the Wire 1 call, the §6 submit, and the deliberation
  `_submit_kwargs`. None touches the `cli_input` selection, `_take_deferred_inputs`, #1414's tick lifetime or the
  `pending_action_followup` injection. The `try/finally` moved whole (`test_narrator_reliability` pins green).
- **Wire 1 move.** Body is a verbatim move; free reads became keyword parameters (`_loop_nac`, `_loop_agent_id`)
  and the loop's `logger` is passed in (loop_planning's module logger is already named `maxim.runtime.agent_loop`).
  Constants moved; `test_wire_1_risk_annotation.py` import updated; the conftest scrub for
  `MAXIM_DISABLE_VARIANCE_ANNOTATION` is by name, so it still applies.
- **Lints.** `lint_function_length.py` clean (2379 -> 2208, 259 -> 252, 384 -> 374; history note accurate, no
  `touch_exemptions`). `lint_no_silent_swallows.py` clean: `_sim_log_proposal_built`'s moved `except Exception: pass`
  is credited as a verbatim move. `lint_orphan_modules.py` clean (loop_external is imported by agent_loop).
  `lint_claude_md_invariants.py` PASS (the new runtime-tools invariant has its `Regression guard:` line and its
  `file::symbol` citations resolve). `lint_unreleased_on_src_change.py` clean. No new env var, so no scrub owed.
- **Narrow tests.** The 8 touched/adjacent test files: 205 passed, 2 xfailed (O4, O9, expected).
- **CC3.** `LLMProposal` is frozen but in-process only (no `asdict`, pickle, `replace` or persistence found); the new
  fields are `compare=False` with defaults. `LLMRequest` is not frozen.
- **`build_comms_stack`.** Not in `docs/user/stable_api.md` or any public verb; its two callers both changed.

## SHOULD-FIX

**S1. The worker re-implements the frame marker syntax.** `llm_worker._record_inbound_rendered` searches for the
literal `<<TOOL_OUTPUT id={nonce} ` / `<</TOOL_OUTPUT id={nonce}>>`. The frame format is owned by
`utils/content_safety.py` (the runtime-tools follow-up framing invariant names it the one builder). If the format
changes there, no frame is ever found "whole": nothing is acknowledged, `note_rendered` is never called, so K never
evicts, and every held message re-renders every turn until `MAX_HELD` drops it. That is a silent no-op in a second
module. The gate test would catch it, but the coupling belongs at the owner.
*Fix:* add `content_safety.frame_is_whole(text, nonce) -> bool` next to `frame_tool_output` (and use it from the
worker); a unit test there pairs it with `frame_tool_output`'s output.

**S2. `loop_external.py` is not in `scripts/run_mypy_public.sh`.** The script's stated scope is "agent_loop and its
loop_* slices" because #840/#841 were argument mismatches; `submit_tick_context(**shared)` and
`planning_submit_shared_kwargs` are exactly that class. mypy 1.20.0 on the file alone is already clean.
*Fix:* add `src/maxim/runtime/loop_external.py` to `FILES`.

**S3. The body signature's comments are crossed (ruff-format artefact).** In `_run_agentic_loop_body`:
```
    sim_adapter: Any | None = None,
    inbound_inbox: ExternalInbox
    | None = None,  # #1481: ... (wrapper-supplied)  # Pre-built NullSimulationAdapter (Stage 3, ...):
    # lets a live producer (the DoA feed) hold the adapter ...
```
The `sim_adapter` explanation now reads as `inbound_inbox`'s. *Fix:* restore the `sim_adapter` comment to its own
line(s) and put `inbound_inbox: ExternalInbox | None = None,  # #1481: ...` after them, on one line.

**S4. The approach note does not describe what was built.** v3.3 says `build_loop_run` subscribes and stores the
inbox on `LoopRun.inbox`, the deliberation feed is `run.inbox.feed()`, the subscription is entered "inside H's
`bio_session` claim", and "if comms is enabled and `LoopRun.inbox is None`, a WARNING names the defect". The code
has no `LoopRun.inbox`, passes the inbox as a body parameter from the wrapper, has no `bio_session` nesting in the
wrapper (H's PR 0 has not landed here), and emits no such WARNING (`inbound_comms_subscription` yields None
silently; `build_comms_stack` raising without a bus makes that case unreachable for comms, which should be said).
Status still reads "cleared to build". *Fix:* a v3.4 "as built" paragraph recording the wrapper/parameter route and
why the WARNING is unnecessary, the outstanding H `bio_session` nesting (whoever lands second joins the `with`),
and status "built in fc69e312 / PR #...".

**S5. The Wire 1 move's grant is not on record.** `annotate_tool_variance`'s docstring and the baseline history
note cite "Session A's grant 2026-10-28", but neither grant section of the approach note (terms 1-6) mentions Wire 1
or `loop_planning`; the grants cover the submit block, the unsubscribe, the deliberation line and dispatch. The move
is clean and is what pays Rule T, but it is outside the recorded hunks.
*Fix:* record the grant (term 7: Wire 1 block verbatim to `loop_planning.annotate_tool_variance`) in the note, or
have A confirm it in the PR before merge.

**S6. advisor_mode.md's O6 row is stale.** It still says "PARTIAL ... Inbound comms reach the prompt unframed, as
'[User via ...]'", and item 4 still lists O6 open. The O6 red gate was retired from
`test_atonia_open_items_other.py` (owner decision 2026-10-26, re-pointed), so the ledger row is now the only place
O6's status lives. *Fix:* update the row: live loop framed and gated by `test_inbound_comms_1481.py` (#1481);
residuals: #1519 (working-memory thoughts unframed), no self-triggered turn (item 3), Twilio `conversation_context`
not carried. The atonia notes' O6 mentions (the redgates fold tables, "the `triggering_input` route for O6 (#1481)")
are round records and can stay.

## NIT

- **N1.** `loop_external.py` has an orphan comment after `_SEEN_SIDS`: "# What the frame rule says the inbound frames
  hold ..." with no constant under it (it moved to `content_safety.INBOUND_WORDING`). Delete it.
- **N2.** The moved Wire 1 block in `loop_planning` says "Same shape rationale as the Wire 3 regex above"; there is no
  Wire 3 regex in that module. Say "the Wire 3 regex in `agent_loop` (`_WIRE3_PHRASE_RE`)".
- **N3.** Cohesion: `planning_submit_shared_kwargs`, `submit_tick_context` and `_release_unsent_deferred_inputs` are
  planning-submit code housed in the comms-inbox module (`loop_external`), next to `loop_planning`. Acceptable under
  the grant's wording ("ONE `loop_external` helper"); a module docstring line saying why would help a later slice.
- **N4.** Typing: `LLMRequest.inbound_feed: InboundFeed | None` but `LLMProposal.inbound_feed: Any | None`,
  `LLMWorker.submit_context(inbound_feed: Any)`, `submit_tick_context(inbound_feed: Any)`. Use `InboundFeed | None`
  throughout (it is in `agents/`, importable by runtime). `continuation()` is used by the loop but not in the
  Protocol; fine while only `ExternalInbox` is passed.
- **N5.** `build_comms_stack(agent: Any)` reads the private `agent._bus`. Matches `inbound_comms_subscription`, which
  is the point (D1); a one-line `Protocol` with `_bus` or a public `agent.bus` accessor would make the pairing typed.
- **N6.** `LLMProposal`'s docstring has no CC3 line (pre-existing). Add "Runtime-ephemeral (in-process only), so
  CC3 does not apply" as `_Refuser` does, now that it carries a live object with a lock.
- **N7.** CHANGELOG: "(it raised nothing before and took `bus=`)" reads oddly; "it took `bus=` before, and now raises
  `ConfigurationError` for an agent without a bus". `build_comms_stack` is not a stable surface, so listing the
  signature change under Fixed is acceptable. The commit message's "autonomy_level per call" means per submit site:
  cycles 2..N share one read taken after the first proposal. Accurate to the test; say "per submit site" in the
  `planning_submit_shared_kwargs` docstring.
