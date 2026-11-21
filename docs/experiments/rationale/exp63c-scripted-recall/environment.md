# Exp 63c: ENVIRONMENT lens (design review, 2026-11-09)

Reviewer: environment lens (DESIGN_REVIEW.md: "does the world game-natively afford it (D1), are the needed states and
acts reachable and measurable, does the bridge or world behave?"). I read the draft
(`exp63c_scripted_recall_prereg.md`, DRAFT v1), its base (`exp63b_carried_recall_prereg.md`), the 63b rationale dir,
`docs/agents/simulation-experiments.md`, and the code on this worktree that the scripted path runs through. I tabulated
63b's committed data (`docs/experiments/data/63b_carried_recall/`, both sessions) with offline probes on this tree's
tokenizer. I did not read the other lenses' files.

**Why 63b had no environment file.** 63b was a derivative of Exp 63 with the same world (goal, LLM narrator, argv), so
it got the floor (confounding, bio-faithful) plus wiring. 63c changes the stimulus, which is the world here, so this
lens is owed now.

## Verdict

**DO-NOT-BUILD on one finding (F1). The rest is buildable with folds.** The pacing, timeout, context and model
behaviour are well inside budget: 63b's AUT was purely reactive, made one call per arrival, and never came near the
120 s wait or the 8192 context. D1 holds: scripted lines take the real addressed path and the world stays live. The
problem is that the draft treats "the phase-1 lines" as the carried store. In the real run, the carried store also holds
text the AUT wrote and text the world wrote: the AUT's intent, stored as `context.active_goal`, and tool outputs. Both
reach `content_tokens`. That text already holds one negative-control word in 63b's data. Under the draft's rule, the
result would be a false C5 that burns an attempt of a campaign the release is waiting on.

---

## DO-NOT-BUILD

### F1. The negative controls are not negative in the store the AUT actually builds

**Claim in the draft.** Turns 3 and 6 are controls because "their non-goal words appear in no phase-1 line". A decisive
turn there is C5 (an instrument fault).

**What the store really holds.** `EpisodicMemory.content_tokens` (`memory/types.py`) keys `text_fields()` (cli_input,
transcript, observations, rationale, reasoning, `outcome.result`, `outcome.error`) plus `context.active_goal`, the tool
name and detections. Each turn in 63b stored exactly two records: the line's own capture and one tool record
(`hippocampus_size` rose by exactly 2 on all 16 traces). The tool record carries:
- **The AUT's free-text intent, as `active_goal`.** Examples from 63b: "I'll explore my options to find out what I can do
  with the guard.", "I remember waking someone up required a loud noise. I'll use sense_tools to find the tool for
  making a loud noise."
- **Tool-output templates.** `sense_tools` ends with "USE one of them as your next action — pick the most relevant tool
  **above** and call it." `sense_presence` lists scene entities (heavy_wooden_door, rusty_key, flickering_lantern,
  guard) and is cut at 1000 characters.

**Measured on 63b's committed stores (this tree's `key_tokens`).** The phase-1 carried `sense_tools` record holds `above`
and `guard`. In phase 2, 3 of the 7 tool records hold `above`. `above` is a control-3 key. Plausible 7B intents also key
to control words: "I'll look around to **see** what I can find" → {around, find, look, see}; "I'll listen to **hear**
whether the guard is awake" → {awake, guard, hear, listen, whether}. 63b's AUT called sense_tools on 1 of 8 phase-1
turns and 5 of 7 real phase-2 turns.

**Why this fires as a decisive turn, not as noise.** On control turn 3, the cue's keys outside the goal are {above,
blue, glass, see, window}, plus `guard`. A carried sense_tools record scores 1 on the non-goal keys and 1 on guard. The
many records that hold only `guard` score 0 on the non-goal keys. Remove the carried ids and one of those guard-only
records enters the cue's top 3. The carried record beats it strictly on the non-goal keys. G4's three conditions hold,
so this is a legitimate shared-word decisive turn. The draft would score it C5, and that makes the attempt incomplete.
Under 63b's rules (at most 3 attempts; an incomplete attempt showing a not-passed gate ends the campaign), one ordinary
sense_tools call in phase 1 can ABORT the experiment the owner held 1.3.4 for.

**Structural point.** G4 condition 3 requires a carried record that holds a control key. So a decisive control turn
cannot show that "the gate credits something other than shared non-goal words". It shows only what the carried store
contains. The control tests the store, not the gate, unless the judge's port is wrong.

**Fold.**
1. Decide a control turn's status from the **judge-recomputed carried store** (phase 1's saved bytes, G2's port), not
   from the script. If any carried record holds a control key, the turn reads `control_contaminated` (descriptive,
   reported with the holding record's ids and fields), counts toward neither R3d nor C5, and the prereg says so. C5
   fires only on a decisive control turn where **no** carried record holds a control key. That case is a port or
   simulator inconsistency, which is a real instrument fault.
2. Exclude control turns from R3d's count of decisive turns. Otherwise a contaminated control could PASS the claim on a
   word the script never planted.
3. Choose control words with a recorded vocabulary check before freeze: not in any tool description or output template
   in the AUT's registry (`sense_tools`, `sense_presence`, `sense`, the base_humanoid tools, `respond`), not in the
   manifest's entity names, not in 63b's AUT intent vocabulary, and not common perception or motion verbs. Drop `above`,
   `see`, `hear`, `play`, `somewhere` and `near` (`near` is also a cue-A and cue-D key). Keep concrete nouns or
   adjectives that a dungeon tool-use turn will not produce (e.g. `blue`, `glass`, `window`, `flute`), after checking
   them against the same sources. Put the check's sources and output in the frozen prereg.

---

## SHOULD-FIX

### F2. The power replay's stub policies do not match how this AUT writes text

**Evidence (63b, both phases).** `response_texts` is `[]` in both reports. All 15 real turns made exactly one tool
action. Across 16 turns there was one `respond`, and it was the turn-8 `llm_fallback`. The AUT never answers in words.
Its text reaches the store through two channels: the intent (`active_goal`) and tool outputs. Policy (a) "quiet:
outcomes holding no planted word" does not describe this world: every `sense`/`sense_presence` record in 63b holds
`door`, `key` and `rusty` (from the manifest's entities), which are cue-A and cue-C keys. Policies (b) and (c) model echo
as "replies", which this AUT does not produce.

**Fold.** Drive the stubbed AUT's echo through the proposal's intent, so the real loop stores it as `active_goal`, and
use 63b's real tool-output texts (`actions.jsonl.gz`) as outcomes. Add a policy (d) "world": sense_presence outputs that
list the manifest's entities plus any entity the script's phrases instantiate (F3). Keep the go rule, and add to it: (d)
yields at least one decisive turn, and the controls stay clean under all four policies once F1's fold is applied. If
the `63b` mock does not run the AUT's real tool outputs and ImaginationTrigger, say so in the replay's report.

### F3. The world reacts to the script, and that reaction has to be measured

**Evidence.**
- **The scene manifest runs before turn 1 whenever `fixture_path is None`.** That is `orchestrator.py`'s manifest
  block, on the narrator's `llm_router`, independent of `pre_campaign_turns`. In 63b it made 4 calls per phase (430,
  1028, 1030 and 1031 input tokens; identical outputs in both phases) and built `guard`, `heavy_wooden_door`, `rusty_key`
  and `flickering_lantern`.
- **The AUT-side ImaginationTrigger runs on every arrival**, before the AUT's LLM call. 63b logged `imagination
  index_hit` at each ENTER. Every 63b line named only the guard. 63c's lines name objects: `extract_entity_phrases` on
  the script returns 'brass key', 'hook beside the torch', 'cell door', 'has a rusty lock', 'does the brass key', 'glass
  above the guard', 'does the rusty lock', 'on the cell door'.
- **Index hits instantiate live entities, and novel phrases above the mention threshold call the designer.** Those are
  synchronous LLM calls on the percept path, 6–9 s each at startup in 63b. Live entities appear in later sense_presence
  outputs, so the world writes planted words into new records from then on.

This is game-native (D1 is fine): a narrator's lines would do the same. It is an echo channel the design does not name,
though. It also makes the script's world inconsistent with the generated scene: the script plants a "brass key on an
iron hook", while the scene holds a `rusty_key`.

**Fold.** The prereg names this channel and states that the scripted objects may not exist as scene entities. The smoke
reports, per turn and per phase: imagination events (cache, index, design, gate), designer calls and their latency, and
the sense_presence entity list. The manifest's phrases must be identical in both phases: they were in 63b, but the
manifest prompt takes the AUT NAc's tool biases, which phase 1 trains. If the phrases differ, the two phases ran in
different worlds, which confounds carried-versus-new competition. Report that as a finding before any marker.

### F4. S2's premise ("the narrator makes no LLM call") is false on this path

**Evidence.** Even with no narrator turns, the run's one server sees these non-AUT calls:
- `generate_scene_manifest(llm_router, goal, …)`, on the narrator's router (F3).
- The designer's entity calls.
- `report.py::analyze_simulation`, on `llm_router` with `request_context={"agent": "sim_roundup"}`. 63b logged one
  `sim_roundup` call per phase (551 and 432 input tokens).

So "the narrator's router made zero calls" never holds, and S2 as written can never stamp `sole_consumer`. Also,
`peer_backend_call` stamps `agent_id: "llm_worker"` on both the AUT's ~6k-token calls and the narrator's ~1.2k-token
retries in 63b, so that field cannot separate consumers (this is #1227's problem, seen in the data).

**Fold (an input to OD-c5, not a choice).** If S2 is taken, define its window as first ENTER to the last turn's settle,
and assert from timestamps that the manifest, designer-at-startup and roundup calls fall outside it. Or name every
known consumer by its call site and require each to be accounted for. Mid-run designer calls (F3) are inside any turn
window and are AUT-side, so the stamp's wording has to cover helpers ("everything served came from model X", which the
draft already says). With one owned server, `served_id` equality is close to trivially true; the strength of S2 rests
on the consumer accounting, so write that part precisely.

### F5. Pacing without the narrator's buffer: assert the turn windows, don't infer them

**Evidence (63b).** The AUT's per-arrival call took 21.3–24.8 s (input 5.5k–6.4k tokens, output 29–124), then exactly
one action, then the 3 s settle. A 5–30 s narrator gap sat between turns; between phase-2 turns 3 and 4 it held 14
narrator retries. Under the script, `run_precampaign_turns` sends line k+1 as soon as `send_and_wait` returns, about 3 s
after the action. Turn 1 goes out 1 s after the pre-campaign block starts; in 63b turn 1 arrived about 16 s after the
AUT thread started. 63b showed no AUT follow-up actions, but the buffer that would have hidden one is gone.

**Budget.** About 28 s per turn, so 8 turns take about 4 min, plus about 45 s of startup (manifest and designer) and
about 6 s of roundup per phase. The 120 s wait has roughly 5× margin on the slowest call seen. 8 turns are enough for
R3a's view, four cues, two controls and a closing line, because the script fixes their order and each cue's query runs
at arrival. A turn whose AUT fails or times out still yields its cue's trace; only that turn's tool record is lost.

**Fold (smoke checks, before any marker).** Per phase:
- Exactly 8 `Bridge.send_and_wait ENTER` lines, whose `text_len` values equal the script's line lengths in order.
- Every AUT action timestamped inside its own `[ENTER_k, ENTER_k+1)` window.
- No AUT LLM call in flight at any ENTER.
- A turn-1 trace with non-empty `own_ids`.
- `hippocampus_size` progressing as the record count predicts (63b: +2 per turn).
- Zero `sim_orchestrator` LLM submits inside the turn window.

### F6. #1616 (the served reader's first-read race): decide it now

63b's row shows the first `/v1/models` read at +0 s failing with HTTPConnectionError; every later read matched. The
issue says its fix "lands only between campaigns", and 63c is a new campaign, so this is the window. **Fold:** add
#1616 to the build order, either fixed before the 63c smoke or explicitly carried with the expected first-read error
named in the prereg. Do not leave it to whoever reads the run log.

---

## NIT

- **N1. Tag ruleset: already covered. Replace "the tag ruleset must cover them" with the verified fact.** Ruleset
  24345734 `o19-markers`: target `tag`, include `refs/tags/o19/**`, rules `deletion` and `update`, enforcement
  `active`. It covers `refs/tags/o19/63c/` with no change.
- **N2. Turn 8 settling is real but harmless to R3d; state its side effects.** In 63b neither phase's turn 8 was a real
  AUT turn. In phase 1, a `respond` with intent `llm_fallback` came 0.6 s after ENTER, and that record is in the carried
  store. In phase 2 the turn "timed out (1.6s)", and its in-flight call ended `dispatch_exhausted` ("down") at +25 s. In
  63c turn 8 settles, so:
  - phase 1's carried store gains a real tool record instead of the fallback;
  - phase 2 loses the cancelled-call artefact;
  - each phase takes about 25 s longer.
  Turn 8's query runs at arrival, so R3d is unchanged. The descriptive "new records per input" and any
  provider-failure reading will differ from 63b.
- **N3. Context headroom is fine; make it a smoke number.** The AUT's prompt reached at most 6446 tokens of 8192 (phase 2
  turn 7) and grew about 120 tokens per turn. The script's lines (24–80 characters) are shorter than 63b's narrator lines
  (37–181). Smoke check: max `input_tokens` + the request's `max_tokens` < `n_ctx`.
- **N4. Behaviours that would make turns unmeasurable were not observed for this model.** In 63b there were no
  refusals, no loops, no long replies (output ≤124 tokens) and no AUT parse failures. 63b's `planning_failed` stop (the
  smoke at b1120188, #1592) was the narrator's. The script removes that failure mode, which is a real gain for the
  environment.
- **N5. A known escape from the fresh data home is unrelated to recall.** Both 63b phases logged a swallowed
  `FileNotFoundError` from `loop_state.py::_persist_state_json`, writing
  `/Users/dennys/RMSrv/scripts/Maxim/data/agents/MaximAgent/runtime/state_*.json` (relative to the working directory,
  outside `/tmp/o19-home-*`). This is #1264/#1567. Nothing on the recall path reads it. Name it in the prereg's
  environment note rather than working around it.
- **N6. The goal does not depend on the script's wording.** All 16 of 63b's traces carry the configured goal, including
  traces whose line lacked it. So script line 1 needs no "SIMULATION GOAL:" prefix. Confirm it on the smoke's first
  trace.
- **N7. Rig hygiene stands as in 63b.** `o19_rerun.py` preflight checks that `llm.profile` and `llm.n_ctx` resolve from
  `config` (not env) and that port 8100 is free. Beyond that, "quiet" (no suite, no Minecraft stack, no second consumer)
  is on the operator. Rig Tailscale-SSH exit codes are not trustworthy, so judge rig commands by their printed output.
  The harness runs on big-mac-mini, never on the leader PC. Provenance (`executed_maxim_file` under the rig repo's
  `src/`) guards against the stray-pymaxim class.

## D1 (no synthetic sensor or reward): acceptable for this claim

Scripted lines enter through `bridge.send_and_wait`, then `inject_cli`, then the `"cli"` addressed source: the same call
a narrator or a person makes. Nothing marks them. No sensor, reward or tool output is fabricated: the manifest,
ImaginationTrigger, tools and Hippocampus all stay live and react. The claim is about how text arriving on that path is
ranked against memory, so where the text comes from is outside the mechanism. The prereg already states the stimulus
change. It should add F3's caveat: the script describes objects the generated scene may not contain.

## Missing from the build order

1. The control-vocabulary check and the judge-side contamination rule (F1), at step 2, before freeze.
2. The power replay on real tool outputs and intent-borne echo, plus policy (d) (F2), at step 2.
3. The #1616 decision (F6), before step 6.
4. S2's consumer enumeration (F4), at step 4.
5. The scripted-run smoke checks (F3, F5, N3, N6) written into step 6's "must show" list.

## What I verified

- **63b timing per call and turn** (`run_log.jsonl.gz`, both sessions):
  - AUT call latency 21.3–24.8 s, input 5467–6446 tokens, output 29–124.
  - One action per turn on all 15 real turns.
  - Narrator calls about 1.2k tokens, up to 14 between turns.
  - 4 pre-turn manifest and designer calls, plus 1 `sim_roundup` call, per phase.
  - Turn 8: a fallback in phase 1, a timeout in phase 2 with a `dispatch_exhausted` call.
- **Records**: 16 and 31 records; +2 per turn on every trace; `active_goal` holds the AUT's intent prose;
  `response_texts` is empty in both phases.
- **Script tokens** on this tree's `key_tokens`: the draft's key tables match (goal keys {dungeon, escape, guard,
  sleep}). Control-3 key `above` is in 1 phase-1 and 3 phase-2 tool records. `door`, `key` and `rusty` are in every
  sense/sense_presence record. Underscored entity names split (`rusty_key` → rusty, key).
- **`extract_entity_phrases`** on all 16 script lines: listed in F3.
- **Code paths**:
  - `campaign_runner.py::run_precampaign_turns`: no inter-turn delay; 1 s initial sleep.
  - `bridge.py::_send_and_wait`: settle and timeout loop.
  - `orchestrator.py`: AUT restore before thread start; manifest block before AUT launch, gated only on `fixture_path`
    and imagination; pre-campaign block, then `_finish_runner`.
  - `report.py::analyze_simulation` on `llm_router`.
  - `imagination/trigger.py::process_percept` / `_resolve_phrase`: index, threshold, designer.
- **GitHub**: ruleset 24345734 (via `gh api`); issues #1616, #1620, #1227 and #1228 are OPEN.
- **Not run**: no sim, no suite, no LLM, no rig access. Probes ran offline in WSL with a scratch HOME and
  `MAXIM_OFFLINE=1`.
