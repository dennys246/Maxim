# #1519 (b): a voice-sourced turn needs typed confirmation for effectful tools (approach note)

> Gate approach note, written before any code (CLAUDE.md "Faster without lowering the bar" (1)).
>
> - **Status: DRAFT v2. The design pass is complete; a delta design pass on this revision is owed before code.**
>   **Not built.**
>   - v1's adversarial pass returned three DO-NOT-BUILDs (D1–D3) and seven SHOULD-FIXes (S1–S7). They are folded
>     below.
>   - v1's open questions are settled by the owner decisions of 2026-11-21.
> - **Scheduling:** build after #1185 merges, on its typed verdict and its `loop` source, in a PR of its own.
> - **Shipped with #1519:**
>   - (a) a live voice request is labelled in the prompt: `prompt_builder.VOICE_REQUEST_LABEL`, carried as
>     `LLMRequest.triggering_input_source`.
>   - This note is (b): what the model may DO on a voice turn.

## The defect

`embodied_runtime/input_handlers.py::_handle_transcript_record` forwards any room-mic Whisper line that names the
wake word ("maxim", "reachy") as `pending_voice_input`. `loop_perception.take_user_input` makes that line the
turn's request.

Anyone in the room can produce that line, as can a TV or a speakerphone. After (a), the model is told the speaker
is unverified. Nothing stops it acting: at AUTONOMOUS level, `AutonomyController.can_execute_action` applies only
the safety constraints. A label is advice, not a gate.

## Owner decisions (2026-11-21)

- **O1, motion.** Voice may run single bounded motions and speech (`move`, `track_target`, `focus_*`, speaking a
  reply).
  - Multi-step motion tools need typed confirmation: SEM motor programs through `use`, `maxim_command`'s pose,
    antenna and turn subcommands. So does anything above them.
  - Voice-driven actions are rate-capped (default 6 per minute, a config key, counted per source), on top of
    `SafetyConstraints.max_actions_per_minute`.
- **O2, no typed channel.** A headless robot, or a run with no operator terminal, has no typed channel. There, a
  gated voice action is refused with a typed reason. It is booked as a machine refusal (`book_refusal`, outcome
  `refused_no_operator_channel`) and the refusal is spoken aloud. It is never auto-approved and never left waiting.
- **O3, sims.** The gate is exempt only under a `SimulationBridge`, whose "voice" is the scripted narrator.
  - A sim with a live microphone or a robot attached gets the full gate.
  - The exemption is decided by what is attached (`embodiment is None and audio input is None`). It is never
    decided from a mode string.
- **O4, per action.** The gate is applied to each action, not to each turn.
  - Follow-ups from a voice turn inherit its voice source, through `ActionFollowup` (S4).
  - One typed "yes" confirms one action. It does not cover the chain that follows.
- **O5, scheduling.** Build after #1185, on its typed verdict, in its own PR.

## One source type (S3)

The gate reads one enum, `Source`, placed beside `Provenance` in `memory/text.py`. #1185 owns the shared definition;
this gate adds no second one.

| `Source` | Meaning | Gate behaviour |
|---|---|---|
| `OPERATOR` | Typed by the operator: `cli` | May satisfy any confirmation |
| `VOICE` | Room mic: `voice`, `transcript` | Its own turn is capped at the voice ceiling |
| `EXTERNAL` | `comms:*`; never a turn (#1481) | Never a turn and never a confirmer |
| `LOOP` | The loop's own reply: `pending_loop_reply`, the non-interactive auto-"yes", a non-bridge sim's default answer (D1; #1185 adds it) | Never satisfies a confirmation |
| `SELF` | No human input: substrate, drives, autonomous exploration | Governed by autonomy level alone, as today |
| `UNKNOWN` | A human input whose source could not be bound | Fails closed: treated as `VOICE` |

How a source is bound (S2):

- An input's source is `provenance_of(observation.source)`, mapped to `Source`, at `take_user_input`. It is never
  read from which slot the text sat in. That is why `pending_voice_input` and `pending_cli_input` are not themselves
  trusted.
- A deferred input (#1048/#1451) stores its `Source` with its text in the deferred slot. A replanned turn keeps the
  source it arrived with.

## The verdict, in the one function every execute path consults (D2, D3)

`AutonomyController.verdict(action, *, confidence, source: Source)` is the only gate. It returns #1185's typed
verdict:

- `Allowed`;
- `Refused(reason)`;
- `NeedsConfirmation(reasons, satisfied_by: frozenset[Source])`.

**Reason strings are never matched (D3).** A voice-ceiling breach yields `NeedsConfirmation(("voice turn: <effect
names> above the voice ceiling",), satisfied_by=frozenset({Source.OPERATOR}))`. Each call site acts on the
verdict's TYPE: it parks a `NeedsConfirmation`, and it books a `Refused`.

**Every execute path consults it (D2).** #1185 makes the verdict a required input to the dispatch seam:
`tool_dispatch.execute_and_learn(..., verdict: Allowed)`, as a required keyword. A path that has not asked cannot
dispatch; forgetting is a `TypeError`. The paths are:

| Path | Today | With the gate |
|---|---|---|
| Main (`agent_loop` §4) | `can_execute_action` | `verdict(..., source=proposal.source)` |
| Parallel batch (`tool_dispatch.execute_parallel_actions`) | per action, `can_execute_action` | The whole batch is pre-checked. If ANY action is not `Allowed`, the batch is refused with "propose singly" (S6). Batches are never parked. |
| Confirmation (`LoopController.handle_confirmation`) | runs the parked action | `verdict` again, with the confirmer's source (S1, S7) |
| Plan approval (`LoopController.handle_plan_approval` and the plan run it starts; the review named it `_run_approved_plan`, a name not in the tree at 5a0c4a97, so the build first locates every function that plan approval reaches) | never asks (D2) | each step asks `verdict` with the PLAN's source |
| PLANNING drain (`loop_planning.drain_approved`) | `approval_blocker` + `hard_deny` | `verdict` with the approval's source |
| Agent fallback (#1147) | dispatches on its own | gets the same required verdict, or is refused |

**Placement against `ALWAYS_ALLOWED_TOOLS` (S5).** The voice-ceiling check runs BEFORE the always-allowed bypass,
because that list is a level bypass, not an effect exemption. Nothing is exempted by name. `respond` declares
`READ | WRITE_LOCAL | ACTUATE`, because its file output mode writes files. Its declaration is narrowed by call:
`respond` gains `effect_for(params)`, which returns `READ | ACTUATE` unless `output_type="file"`. The gate reads the
call's effect: `effect_for` when a tool defines it, `effect_of` otherwise. A spoken or printed reply then passes
the ceiling by its effect, and a file write does not.

## The voice ceiling, and the motion subset (O1)

The rule is `exceeds(call_effect, VOICE_CEILING)`, with `VOICE_CEILING = READ | SELF_SCOPE | ACTUATE`, plus the
bounded-motion declaration below. It is gated through `NeedsConfirmation`, and an undeclared tool is `ALL`.

**Choice: a declared subset, not an effect split.** A new `Effect` flag would change the atonia gate's
seven-flag vocabulary (V2) and every ceiling that names `ACTUATE`. Instead, a tool whose `ACTUATE` is one bounded
motion or speech declares `bounded_actuation: ClassVar[bool] = True`. That is a declaration on the class, read
through the same recorded-at-class-creation mechanism as `effect` (`_EFFECT_RECORD`), not a name list.

- A voice call with `ACTUATE` passes only if its tool declares `bounded_actuation`. Today that is `move`,
  `track_target`, `focus_on_sound`, `novelty_track`, and the speech of `respond`/`speak`.
- `focus_interests` declares `Effect.ALL`, because its display path starts a process (#1544). The O1 list names
  `focus_*`, so the build either narrows that declaration (with `effect_for`, as for `respond`) or leaves it gated.
  That is the owner's call, not this gate's: open question Q1.
- `use` (SEM motor programs), `maxim_command` (its effect includes `GOVERN`) and any multi-step motion tool do not
  declare it, so they need typed confirmation.
- Motion stays bounded by `ReachyMiniController.goto_target`'s clamp either way.
- The voice rate cap (O1) sits in the same verdict. Over the cap, the verdict is `Refused("voice rate cap")`.

## Who can confirm (S1, S7)

- `handle_confirmation(text, *, source: Source)` takes the confirmer's source as a required keyword. It is never
  read from `pending_user_input_source`, which can be stale (S1).
- A parked `NeedsConfirmation` runs only when `source in verdict.satisfied_by`. That set holds `OPERATOR` for any
  effect above the voice ceiling, WHATEVER turn the action came from (S7). A TV's "yes" cannot confirm a typed
  turn's write.
- A `VOICE`, `LOOP` or `EXTERNAL` "yes" leaves the action parked. The robot says it needs a typed confirmation;
  under O2 it is refused at once instead.
- `detect_approval_intent` still reads WHAT the answer is. It never decides WHO may give it.

## Every input the gate consults, and how each could lie

| Input | How it could lie | Answer |
|---|---|---|
| The action's source (`LLMProposal.source`, `ActionFollowup.source`; S4) | Stale state; a follow-up with no source; a replan | It is stamped when the proposal is built, from the request's bound source. Follow-ups copy their parent's (O4). `UNKNOWN` fails closed. `SELF` is set only by producers with no human input. |
| The confirmer's source | A room-mic "yes"; the loop's own "yes" (D1) | A required keyword. Only `OPERATOR` satisfies. `LOOP` is its own source and never satisfies. |
| `cli` itself | `/api/command`; the sim narrator; a remote shell | `/api/command` is `comms:command` and never `cli` (#1481); a test pins that no comms producer writes the CLI slot. The narrator is exempt only under a bridge (O3). A terminal user is the operator, by definition. |
| The call's effect | Under-declaration; a `respond` file write | The same declaration the atonia gate trusts. Undeclared is `ALL`. `effect_for` narrows only from what the declaration names. |
| `bounded_actuation` | A multi-step tool declares it | It is recorded at class creation, like `effect`. A test pins the declared set against the tool registry, so adding one needs a reviewed diff. |
| The tool name | An alias (`reflect` → `think`) | Resolved through the executor's one alias resolver before the effect is read. |
| A parallel batch | One gated action hidden among allowed ones | The batch is pre-checked and refused whole (S6). |
| Wake-word text | It reads as operator | Never read: only the bound source is. |
| The attached devices (O3) | A sim with a real mic claims exemption | The exemption is computed from attached embodiment and audio, never from config text. |

## Open question

- **Q1:** `focus_interests` (`Effect.ALL`): narrow its declaration, or keep it gated on voice turns?

## Notes and scope

- **N1.** The `write_file` overwrite retry (`execute_and_learn`) re-issues a confirmed write with `overwrite=True`.
  That widens what was confirmed. Under the gate, the retry is a NEW action with its own verdict and inherits the
  action's source. It is never covered by the earlier "yes".
- **N2.** Phrase-spec actions (`input_handlers._run_action_spec`: a phrase mapped straight to a tool, with no
  LLM) are out of scope. They are operator-configured. A follow-up issue can give them the same verdict.
- **Not in scope:** speaker verification; changing the wake-word forwarding; comms text, which is never a turn;
  stored or recalled voice text, which is framed (#1519).

## Regression guards (owed with the build)

Real-loop tests:

- A voice wake-word turn proposing `write_file` is parked. A typed "yes" runs it. A voice "yes" and a `LOOP` "yes"
  do not.
- A voice-proposed plan approved by voice runs no gated step.
- A parallel batch holding one gated action is refused whole.
- A headless run refuses at once, books `refused_no_operator_channel`, and speaks the refusal.
- A follow-up from a voice turn is gated again.
- A voice turn's `move` runs, and `use` is parked.

Deletion probes:

- the verdict's voice check;
- the required `verdict` on `execute_and_learn`;
- `satisfied_by`;
- the `LOOP` source;
- the batch pre-check;
- `bounded_actuation`.

Ledger: none (no experiment row reads confirmation).
