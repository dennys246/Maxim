# The embodied device: one owner per device contract

Status: **DRAFT v2**. Owner direction and decisions 2026-10-24. Review round 1 (Executor, Architecture, Wire integrity)
folded. Input: maxim-pulse's device review, [Maxim-pulse #58](https://github.com/dennys246/Maxim-pulse/pull/58)
(`docs/plans/voice_and_install.md`), from the first real-robot session (Reachy Mini, pymaxim 1.3.3, reachy_mini SDK
1.12). Pulse's document is the evidence and a proposed shape. This one is pymaxim's plan: what pymaxim owns, what
rides on what already exists, and in what order.

**Home (owner decision 2026-10-24): its own clock, in parallel.** Like the atonia gate, it gets its own session. It
ships as a patch when it is ready and holds nothing. It touches no decomposition-fenced file, so 1.3.5, 1.3.6 and
1.4.0 keep their order.

## 1. Why

Every robot failure sat at a seam where each caller re-implemented a contract nobody owned, or where a test faked
exactly the thing that broke:
- the SDK's audio rules;
- the daemon's launch, stop and user;
- pymaxim's config schema;
- "which agent";
- "is the voice actually working".

This is D43's lesson: a defect in a composition is fixed in the seam, as one callable thing.

**Pulse's findings checked against pymaxim main (2026-10-24):**

| Pulse finding | Checked here | Tracked |
|---|---|---|
| R5: voice input dead since April | **Confirmed**: `embodied_runtime/media_loop.py` lazily imports a module that `db274547` (2026-04-03) deleted, behind one warning | #1436 |
| `maxim --tts` does nothing | **Confirmed**: `cli.py` passes `speaker_fn=None`, so `ResponseOutput.audio_enabled` is always False. The embodied path has its own `MAXIM_TTS_ENABLED` toggle, which does wire `speak` and is the path that clips (next row) | #1437 |
| R6: a dropped push reads as played; no resample to the device rate | **Confirmed**: `utils/audio.py::push_audio_to_device`, and the same lie in `hardware/reachy/streams.py::ReachyAudioStream.push_sample` | #1435 |
| R6: the agent's resample clips | **Confirmed and measured**: `response_output.py::_speak_response` → `resample_audio` (`resample_poly` returns float64 on the int16 scale) → `_to_device_float32` only casts. An int16 sine at 20000 arrives with max 20011.55, a hard clip at ±1.0, on the device and on the local fallback | #1435 |
| `speak` tool honesty | `tools/response.py` returns `spoken: True` whatever the sink did | #1435 |
| R1/R4: Talk reaches the body-less `console_agent`; the robot's agent idles | Consistent with `console/server.py::_DEFAULT_HANDLE_AGENT_ID` and its module-global `_handle` | #1439 |
| R2: the stop outlives the daemon's 20 s kill | **Reproduced from code**: `server.py::_drain_and_stop_handle` defaults to 30 + 60 + 20 s, up to ~110 s | #1438 |
| R8: two readers of one mic stream | **Confirmed**: `CaptureManager._audio_capture_loop` and `workers.py::audio_capture_worker` both call `get_audio_sample()`; `ReachyAudioStream` is a third possible reader. Video has the same pair (`frame_capture_worker` and `_frame_capture_loop`) | this plan |
| R3, R7, R9, R10 | Pulse's evidence, not yet re-checked here | this plan |

**Pushing the invariant into a type** passes CLAUDE.md's three-silent-misses bar on confirmed misses alone. Four have
been confirmed:
- a dropped push reported as played;
- audio pushed at the wrong rate;
- a resample left on the int16 scale, and clipped;
- `--tts` never speaking.

## 2. Front gate: ride, or a new mechanism?

A piece rides on existing infrastructure unless its row names why that cannot work. **No new top-level package.**
Any new package would need an ARCHITECTURE.md entry and an architecture-audit rule first.

| Piece (pulse §) | Existing infrastructure | Verdict and where it lives |
|---|---|---|
| One agent: `build_app(handle_factory=)` (A) | `console/server.py`'s module-global `_handle` / `_get_handle`, read by recall, rest, run, talk and `_drain_and_stop_handle` | **Ride, but bigger than a parameter.** The handle moves from a module global onto `app.state` (one handle per app). Every reader, and conftest's reset, moves with it. **Identity rule:** the injected handle's `agent_id` wins. `build_app` refuses a conflicting `console.agent_id`, and refuses `handle_factory` together with sandbox, so `/api/recall` can never read one home while Talk writes another. **Stop ownership:** the embedder (`run_device`) owns an injected handle's stop; the lifespan drains only a handle it built. |
| The turn entry point (A, "`submit_utterance`") | `console/handle.py::MaximHandle.talk(text, *, timeout)`, a synchronous turn whose only caller is `server.py`'s talk route | **Ride: extend `talk`, no second path.** A voice turn has no HTTP caller waiting on its dict, so `talk` splits into submit plus await, with `source="typed"\|"voice"`. It **must be advisor item 4's single ingestion entry point, with inbound text framed as untrusted**. Otherwise voice is a new unframed inbound path. `source` on USER events is a **payload change** (see the status-seam row). |
| The bounded stop (A) | advisor item 9's bounded yield on SIGTERM (#1344, Session H) | **Ride, one mechanism, the caller sets the deadline** (design: [bounded_stop.md](bounded_stop.md), amended 2026-10-25/26). The library saves within the deadline and RETURNS; the embedder exits. pymaxim installs no signal handler in an embedder. The device passes its budget through ONE seam, `build_app(stop_budget_s=)` (the console's default `console.stop_budget_s` is 15 s, under the daemon's ~20 s kill; the CLI's `runtime.yield_timeout_s` is 30 s), or `handle.stop(deadline=)` for a handle it owns. Pulse is asked for the daemon's stop signal, kill delay and start window. |
| `Pcm` value type (B) | `utils/audio.py` free functions with disagreeing scales: `to_int16` clips floats as [-1, 1]; `_to_device_float32` assumes floats are already [-1, 1]; `resample_audio` silently promotes int16 to float64 on the int16 scale | **New type, in `utils/audio.py`.** The free functions become its methods. This is the type the four misses in §1 justify. |
| The output port (B, "`AudioOutput`") | **`hardware/streams.py::AudioStream`** (`push_sample -> bool`, `input_sample_rate`/`output_sample_rate`, `start`/`stop`), with adapters `hardware/reachy/streams.py::ReachyAudioStream` (`update_sample_rates()` reads the device) and `hardware/simulation/streams.py::MockAudioStream`; also `make_device_speak_sink`, `push_audio_to_device`, `LocalSpeaker` | **Reshape `AudioStream`; no parallel port.** `push_sample` returns a `PlayReport(ok, reason)` **whose `__bool__` is `ok`**, so an old bool-checking caller cannot read a failure as success. The free sinks become thin callers of the port, kept for one deprecation window (see "Pulse's Python API" below). |
| `Voice` (B) | `models/audio/tts.py::VoiceProvisioner`, `TTSEngine`, `PiperTTS` | **Compose, don't push hardware into models.** `Voice`, at the device and console layer, composes `VoiceProvisioner` (provisioning states, unchanged layer) with an `AudioStream`, and adds the PLAYS state. Names stay, because renames ripple to pulse. |
| Listening (C, "`AudioInput` + `AudioHub` + `Listener`") | `AudioStream.get_sample`; `runtime/capture.py::CaptureManager` (already consumes `AudioStream`); `RobotController.get_doa_reader()` + `embodiment/audio_localization.py::DoAReader`; STT in `models/audio/transcription.py`; the dead file-IPC path | **Ride on the existing ports**, with one rule: **one consumer per mic**, CaptureManager, which fans out. `audio_capture_worker` goes, and so does the video duplicate. The dead path is deleted, not wrapped. The design review places the endpointer and Listener composition next to `runtime/capture.py` or in `embodied_runtime/`. **DoA is on-robot only** (embodiment.md §1 forbids a client-side `get_DoA()` for off-robot consumers). |
| `ReachyMiniController.adopt(sdk)` + `MaximHandle.embody(sdk)` (A) | `hardware/reachy/controller.py::ReachyMiniController` (the single clamped and locked dispatch point; `connect` opens its own `ReachyMini(...)`) | **Ride, and mandatory**: all motion stays on the controller (embodiment.md §1). `adopt` opens no connection. Second connections that exist today must go or be adopted: `embodiment/audio_localization.py` builds `ReachyMini()` when `mini is None`; the CLI's selfy path connects through the controller. **Guard:** a grep for `ReachyMini(` outside the controller. |
| `maxim.device.run_device(...)` (§2) | F4, `console.device.build_device_app` (#1292, "design review first") | **The same thing, in `console/device.py`.** F4's design review is this plan's step 6 review; #1292 gets no second one. |
| The device status seam (D) | `SeamStatus`, `/api/hello`'s `pairing`, `HealthResponse` | **Ride.** It is a contract change. It lands with R3's 0.7.0 bump (#1336) **if that bump has not shipped yet**; otherwise it takes its own minor bump and its own pulse tag (the R4 cut rule). |
| The `robot` extra (E) | the `pi` extra (`reachy, console, llm-anthropic, tts`) | **Ride: rename to `robot`.** `pi` stays an alias for one release. The aarch64 no-torch guard (`test_pkg_aarch64_guard.py`), `test.yml`'s aarch64 jobs and `check_aarch64_install.py --extra` all move to `robot`, so the alias cannot make the guard vacuous. `optional_deps` fix hints name `robot`. **Order:** M69 ("a new extra is born in the matrix") means R2's reachy cells come first. **`audio` stays out of `robot`**: STT runs on the leader (decision 1). |
| A device log (E) | `MAXIM_LOG_FILE` and the JSONL handler | **Ride**: a rotating default path on a device. |
| mDNS leader discovery, peer↔leader pairing by code (E, 1.4) | the spoken-code pairing machinery; `peer/` | **Later**, with its own security design pass: it changes how a peer earns the leader key. |

**Pulse's Python API.** pulse imports, from the documented embedder recipe in `models/audio/tts.py`'s docstring:
- `VoiceProvisioner` (`.engine`, `.is_ready`);
- `make_pairing_announcer`;
- `make_device_speak_sink`;
- `build_app(pairing_announcer=, pairing_ready=)`.

None of these is covered by `contract_version` or the OpenAPI fingerprint, which cover HTTP only. Each one stays, or
gets one release of deprecation with a warning, and the note to pulse names it (M70). That matters more because pulse
proposes a floating `>=1.3.3,<2` pin.

## 3. Atonia and the device are independent

Owner decision 2026-10-24: **the atonia gate ships as designed.**
- Its closed builder refuses any embodiment or robot. That is advisor_mode.md's Actuation row in §The guarantee by
  effect channel, guarded by the builder-refusal tests.
- Under it no SDK handle exists, so nothing on the robot can actuate. #1435 is not a hole in that gate.
- The gate ships on its own clock, accepted by its own red gates. **The device plan does not gate it, and it does not
  gate the device plan.**

"Atonia within embodiment" would be a later extension of the gate: a body attached through `embody()`, with motor
output inhibited and the report open. It reverses the Actuation row and would need its own design passes. This plan
prepares for it without deciding it:
- **`embody()` is the one place a body attaches.**
- **The embodied census is this plan's acceptance criterion.** It enumerates every path that can move a motor or emit
  sound through `embody()`: the controller, `Voice`'s output, SDK side effects (`start_playing()`, the wobbler) and
  anything the daemon starts. CI enforces it with an SDK-faithful fake that records every actuation call.
- **If the extension is ever taken,** that census becomes its acceptance test, and one operator-attested real-robot
  drill becomes its supporting evidence, never its only evidence.

**The head wobbler (owner decision 2026-10-24): disabled explicitly.**
- In reachy_mini 1.8.3 the wobbler is opt-in: `enable_wobbling()`, on the SDK object or by a daemon command.
  `start_playing()` starts it only if it already exists. The 1.12 default still needs checking. Pulse found
  (2026-10-24) that `reachy_mini.disable_wobbling()` turns off both the SDK-side and the daemon-side wobbler. Its
  smoke starts a fake robot with wobbling ON and asserts that the call turns it off.
- The device lifecycle owner calls `disable_wobbling()` when it starts, so speech never moves the head and all motion
  stays on `goto_target`.
- A test pins it off.
- Opting in later is its own hardware-safety decision: a third motion primitive, an amendment to embodiment.md §1,
  and a guard that it never races `_motion_lock` or the retained head target.

## 4. Decisions

**Owner, 2026-10-24.** Pulse's four questions, decided as recommended:
1. **Speech-to-text runs on the leader first,** with on-robot voice-activity detection and endpointing. On-robot STT
   comes later, as an offline fallback (advisor item 10 points there).
2. **Listening starts with the wake word "Maxim"**; pulse's draft defaulted to always on. The owner has since asked
   for **dynamic listening**, so the wake word is the Listener's first state, not its only one. A listening state
   machine follows on the same Listener: follow-up turns without the wake word, and closing on silence or on a
   topic end. Each new state is an inbound gate, so it gets its own design pass.
3. **A private `-dev` Space first.** It goes public when R2's reachy cell and the one-command on-robot check pass on
   a release candidate.
4. **The 1.4 memory-folder merge rides `runtime/agent_factory.py::migrate_agent_state` (#932):** whole or nothing,
   refusing when the target already holds an agent, with a backup.
   - Its unnamed consumers move with it: `maxim hive merge`'s default `user_memory()/nac.json`,
     `tools/learned_index.py`, `agentic_runtime.py`'s `persistence_dir`, `bio_stack.py`'s Reachy recipe, and
     `AGENT_HOME_PATTERNS`.
   - **Interim rule from step 1 to 1.4:** on a device whose live agent is `agents/<id>`, `maxim hive merge` with no
     `--into` refuses and names both homes. It never silently writes into the idle CLI home.

**Also decided 2026-10-24:**
- the gate stays as designed (§3);
- the wobbler is disabled (§3);
- the home: own clock, in parallel (top);
- **the defects' owners.** Each one ships in 1.3.4 only if ready, and never holds the atonia gate's cut:
  - Session S takes #1435, #1436 and #1437, each through named fence exceptions:
    - #1435 reaches `utils/audio.py` (in S's fence), `embodied_runtime/media_loop.py` (its interim player start;
      no session owns it) and `hardware/reachy/streams.py` (the second sink; no session owns it). Its
      `tools/response.py` honesty hunk is Session V's file, so it is a shared hunk.
    - #1436 is `media_loop.py`.
    - #1437 is a one-line `cli.py` wiring, Session A's file.
  - Session H takes #1438's bounded stop, with #1344, as one mechanism whose deadline the caller sets.
  - #1439 is this plan's step 1.
  - **#1435's interim lifecycle owner:** until `embody()` exists, whoever holds the SDK handle calls
    `disable_wobbling()` and then `start_playing()` once, through `maxim.utils.audio.start_device_player`: `MediaLoopMixin` at media start, and a documented call for
    bare-SDK embedders (pulse's bootstrap). The sink never starts the player; it refuses loudly when the player is
    not started. **The sink normalizes the scale before resampling**, and `response_output`'s own pre-resample goes in
    the same PR, so R6's clip cannot survive the fix.

**The board (read by pulse on the owner's robot, 2026-10-24):** a Raspberry Pi Compute Module 4 Rev 1.1: 4 ×
Cortex-A72 cores and 3.7 GiB of RAM, about 3.0 GiB free with the app running. On-robot defaults:
- speech-to-text runs on the leader by default (decision 1);
- the offline fallback is `tiny.en`, `base.en` at most, int8, CPU, 4 threads. Never pulse's earlier
  `distil-large-v3`, and never forced to CUDA.

RAM is tight, so step 1 (one agent) also frees the idle second agent's memory: a second reason it comes first.

**Pulse's interim owner (2026-10-24):** pulse #57's bootstrap is #1435's interim lifecycle owner for a bare-SDK
embedder: `start_playing()`, then `disable_wobbling()` once at start, and `stop_playing()` last, pinned by pulse's
tests. When `embody()` owns this, pulse drops its copy, and a test on each side pins exactly one owner.

**Superseded by #1435's owner decisions (2026-10-24, [speaker_sink_1435.md](speaker_sink_1435.md)):**
- the sink can only trust a start it recorded, so pulse's bootstrap calls maxim's pair instead of the SDK:
  `maxim.utils.audio.start_device_player(reachy_mini)` once at start, `stop_device_player(reachy_mini)` last;
- the order is **`disable_wobbling()` first, then `start_playing()`**, so a pre-existing wobbler is never started;
  a missing or failing disable means the player is not started;
- **pulse ships that release first**: 1.3.4 does not publish before it (mechanization backlog M79);
- a WebRTC (off-robot) handle plays on the robot only once `start_device_player` ran and its send chain has been
  built, else on local speakers (known gap: the SDK never clears the chain, so after a mid-session peer drop a push is a silent drop reported as played; the device plan's step 3 closes it);
- #1435's named fence exceptions also cover `embodied_runtime/connection.py` (drop the record before every media
  close, restart the player after a soft reconnect) and `embodied_runtime/selfy.py` (`Maxim.push_audio_sample`'s
  fallback).

## 5. Order

1. **One agent:** the handle on `app.state`, the identity rule, `talk` split into submit plus await (advisor item 4's
   framed entry). This closes R1, one agent. **R4, a body-less agent, closes at step 5.** `source="voice"` is a
   payload change: if R3's typed payloads have landed it is a schema bump; if not, the note to pulse names the new
   untyped field.
2. **The bounded stop**, the caller's deadline, through Session H's #1344 mechanism. #1438 lands it earlier as the
   defect fix; this step wires `run_device`'s budget.
3. **`Pcm`, the reshaped `AudioStream`, `Voice`.** The speech callers collapse onto `voice.say`:
   - the pairing announcer;
   - the `speak` tool, reporting `spoken = result.ok`;
   - `--tts` and `MAXIM_TTS_ENABLED`, converging on one toggle;
   - Console replies on a device.

   **Readiness changes meaning here, and supersedes R1's in-memory "has spoken":** `pairing_ready` becomes
   `voice.ready(PLAYS)`. A device whose output cannot play (WebRTC, the local fallback) never arms spoken pairing and
   shows the paste path; the note to pulse says so. PLAYS is reachable here because #1435's interim owner starts the
   player (§4). Before step 5 it is proven against the SDK-faithful fake.
   **It absorbs #1435's interim record** (`utils/audio.py::_PLAYERS` and `start_device_player`/`stop_device_player`):
   the reshaped port holds "started" as its own state, the free sinks (`push_audio_to_device`,
   `make_device_speak_sink`) become thin callers of it, and no second "started" record is built. #1435 routed
   `ReachyAudioStream.push_sample` through the free function, the reverse of this direction; step 3 flips it back. A
   connected WebRTC send chain plays (owner decision 2026-10-24), so "cannot play" above means an unconnected one;
   PLAYS must also see a peer drop, which #1435's probe cannot (the SDK never clears `_appsrc`).
4. **`ReachyMiniController.adopt` and `MaximHandle.embody`**, under embodiment.md §1. The single-connection guard and
   **the embodied census** land here.
5. **Listening**, attached through `embody()`: CaptureManager as the one consumer, the wake word as the first state
   (decision 2), STT on the leader, and half-duplex while `Voice` speaks. (It comes after `embody`, so it has a body
   to listen through.)
   - **The leader's STT endpoint does not exist yet.** No `/v1/audio/transcriptions` route exists in `src/maxim`, but
     decision 1 needs one, and so does pulse's planned STT-brokering peer (the owner's Mac). It is a new leader-proxy
     route, so it gets **its own design pass**: bearer-gated like every proxy route; the 1.3.3 key rules; an audio size
     and duration cap; no audio logged or persisted; client calls through `utils/http.py`; read
     docs/agents/llm-routing.md first. It is an OpenAI-shaped route, so a brokering peer serves the same contract.
6. **`run_device`** (F4, `console/device.py`), whose design review is this step's.
7. **The status seam:** with 0.7.0 if 0.7.0 is unreleased, else its own minor bump. The tag plan follows the R4 cut
   rule every time: pulse tags a release built for the new contract, pymaxim vendors it, then pymaxim releases. The
   concrete plan is proposed when R3 (#1336), the first 0.7.0 rider, is scheduled.
8. **The `robot` extra**, after R2's reachy cells, **and the device log.**

**Pulse, alongside:**
- the Space publishing and the private `-dev` Space;
- the daemon-faithful harness;
- the SDK-generated contract tests;
- `robot-check.sh`;
- the status UI;
- the app shrinking to SDK glue once `run_device` lands.

**1.4:**
- `selfy.Maxim` re-based on `MaximHandle`, with the migration (decision 4);
- the remaining raw head moves onto the controller;
- mDNS discovery and pairing by code;
- an architecture test refusing a second agent builder.

## 6. Done when, and what each invariant cites

- **SDK-faithful fakes,** built on the SDK's own `MediaManager`, assert what would be heard: duration, pitch, no
  clipping, not silence. They never assert on a returned bool.
- **A daemon-faithful harness** launches the app the way the daemon does and stops it with SIGINT. The process exits
  within the budget read from the SDK, with the stores saved and consistent.
- **The R2 cells** for reachy and robot pass at floor and latest.
- **Pulse is sent** every contract and Python-API change under M70.

**The new invariants enter as `[engineering]`:**
- **One agent per device:** a Talk turn and the embodied agent are the same agent, and its home grows. Regression
  guard: the harness test above, plus a mechanization-backlog row for the 1.4 architecture test refusing a second
  agent builder. Add that row when step 1 lands.
- **One SDK connection:** guard is the `ReachyMini(` grep outside the controller.
- **One mic consumer:** guard is a test that `get_audio_sample` has one caller.
- **No speech-driven motion:** guard is the wobbler-off test.
- **The embodied census:** guard is the recording-fake census test.

Review: round 1 (Executor, Architecture, Wire integrity, 2026-10-24) and a delta round (two wording folds: #1435's
fence exceptions, F4's last two references) are folded into this v2. Another round is owed only if a later edit
changes a decision or a step. Each build step gets its own
design and code rounds.
