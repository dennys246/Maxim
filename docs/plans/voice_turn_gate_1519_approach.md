# #1519 (b): a voice-sourced turn needs typed confirmation for effectful tools (approach note)

> Gate approach note, written before any code (CLAUDE.md "Faster without lowering the bar" (1)).
>
> - **Status:** DRAFT v1, for its adversarial design pass. **Not built.**
> - **Owner decision (2026-11-21):**
>   - (a) Live voice is labelled. This shipped with #1519: `prompt_builder.VOICE_REQUEST_LABEL`, carried as
>     `LLMRequest.triggering_input_source`.
>   - (b) Risky tools on a voice-sourced turn need typed confirmation. That is this note.

## The defect

`embodied_runtime/input_handlers.py::_handle_transcript_record` forwards any room-mic Whisper line that names the
wake word ("maxim", "reachy") as `pending_voice_input`. From there:

- `loop_perception.take_user_input` takes it as the turn's request;
- `LoopController.store_user_input` records it with `pending_user_input_source = "voice"`.

Anyone in the room, or audio from a speaker or a speakerphone, can produce that line. After (a), the model is told
the speaker is unverified, but nothing stops it acting: at AUTONOMOUS level, `AutonomyController.can_execute_action`
applies only the safety constraints. A label is advice to the model, not a gate.

## What is gated

Gating is by declared effect, never by a new tool list. A tool's effect is `tools/base.py::effect_of(tool)`, a
`tools.base.Effect` set, and every tool already declares one. A missing or invalid declaration counts as `ALL`
(fail-closed, V15).

On a voice-sourced turn, an action needs typed confirmation when:

```
exceeds(effect_of(tool), VOICE_CEILING)
```

and `VOICE_CEILING = Effect.READ | Effect.SELF_SCOPE | Effect.ACTUATE`.

- **Allowed:** reading, the agent's own memory, and physical output (motion, speech, display), so voice still
  commands the robot.
- **Gated:** `WRITE_LOCAL`, `NETWORK`, `COMMS` and `GOVERN`, plus anything declaring `ALL`.
- `ACTUATE` is in the ceiling by design. It is the robot's purpose, and motion is already bounded by
  `ReachyMiniController.goto_target`'s clamp. Motion is a candidate for review: open question Q1.
- The ceiling is one constant beside `OPERATOR_SOURCES` in `memory/text.py`, and the same subset check as the
  atonia gate (`exceeds`). It is not a second effect vocabulary.

## How it reuses the existing path

The gate gets no new approval UI or state machine. Instead, it extends the existing confirmation path:

1. `SupervisionPolicy.can_execute` already returns `(False, "requires confirmation")` for names in
   `requires_confirmation`. The gate adds one input: the turn's source.
   `AutonomyController.can_execute_action(action, *, confidence, turn_source)` checks the voice ceiling before the
   level branches. It returns the same shape (`False, "voice turn: <effect> needs typed confirmation"`), at every
   level, including AUTONOMOUS. The keyword is required (a `TypeError` if a call site forgets it), per CLAUDE.md
   "push silent-no-op invariants into types". All three call sites pass it:
   - `agent_loop` §2 and §4;
   - `tool_dispatch.execute_parallel_actions`.
2. A refused action takes the path a SUPERVISED refusal takes today. It is parked as
   `LoopController.pending_confirmation`, and the user is asked. `handle_confirmation` runs a "yes" through the
   run's one `execute_and_learn` (#1133), so a confirmed action learns exactly what an autonomous one would.
3. **What counts as confirmation:** only input whose recorded source is typed `cli`, via the
   `pending_user_input_source` of the confirming input. A "yes" heard on the room mic is not confirmation, because
   the same unverified speaker could say it. `handle_confirmation` gains a `source` argument. A voice "yes" on a
   voice-parked action leaves it parked and says so aloud.

## Every input the gate consults, and how each could lie

| Input | Where from | How it could lie | Answer |
|---|---|---|---|
| The turn's source | `pending_user_input_source`, set by `store_user_input`; carried as `LLMRequest.triggering_input_source` → `LLMProposal` | (i) A voice line arrives while a typed one is stored, and the proposal inherits "CLI". (ii) A follow-up or replan turn carries no source. (iii) Stale state from an earlier input. | Bind the source to the TEXT it was stored with, as `_triggering` does now (equal text, or ""). Unknown ("") is treated as voice-sourced (fail-closed). The proposal carries the source it was built from; dispatch never re-reads state. Follow-ups inherit their originating turn's source through `pending_proposal`. |
| The confirming input's source | Same field, at the confirming tick | The room mic says "yes". | Only `cli` confirms (above). A voice "yes" is never approval, whatever `detect_approval_intent` reads. |
| `cli` itself | Typed terminal, the robot/API mailbox `pending_cli_input`, the sim narrator | (i) The `/api/command` path. (ii) A sim narrator is an LLM playing the user. (iii) A remote shell user. | (i) `/api/command` is `comms:command`, never `cli` (#1481); a test pins that no comms producer writes `pending_cli_input`. (ii) The sim narrator stays operator by D5, and the owner has accepted that: sims are not a trust boundary. (iii) Whoever can type into the terminal is the operator, by definition. |
| The tool's effect | `effect_of(tool)`, recorded at class creation | A tool under-declares (a `NETWORK` tool says `READ`). | The atonia gate trusts the same declaration (V2–V15). This gate does not make it worse. An undeclared tool is `ALL` and is gated. Under-declaration is the effect audit's concern (atonia notes §2.1). |
| The alias the model named | The proposal's `tool_name` | A `reflect`-style alias resolves to another tool. | Resolve through the executor's one alias resolver (the same as `untrusted_output`, tool_dispatch) before `effect_of`. |
| A parallel batch | `execute_parallel_actions` | One gated action hidden in a batch. | The check runs per action. If any action in a batch is gated, the whole batch is parked. |
| Wake-word text | `_handle_transcript_record`'s "maxim"/"reachy" test | The text sniffs as operator. | Never read: the gate reads the recorded source only. |

## Not in scope

- Speaker verification and voice biometrics.
- Changing `input_handlers`' wake-word forwarding.
- Comms text, which is never a turn (#1481).
- Stored or recalled voice text, which is framed (#1519).

## Open questions for the owner

- **Q1:** Should `ACTUATE` stay in the ceiling? A body turn toward a door is harmless, but a long motion program is
  not.
- **Q2:** Is "unknown source = voice" acceptable for follow-up turns, or should follow-ups inherit only?
- **Q3:** Should the voice refusal be spoken (TTS) or only shown?

## Regression guards (owed with the build)

- A real-loop test: a voice wake-word turn proposing `write_file` is parked; a typed "yes" runs it; a voice "yes"
  does not.
- A deletion probe on each of: the ceiling check, the required `turn_source` keyword, and the `cli`-only
  confirmation.
- Ledger: none (no experiment row reads confirmation).
