# #1435 approach note — the Reachy speaker sink (Session S, 1.3.4)

Status: ADOPTED 2026-10-24 (adversarial design pass folded below; owner decisions at the end). Built in the #1435 PR. Decided by the owner (2026-10-24), not re-opened here: the sink never
starts the player; whoever holds the SDK handle starts it once (MediaLoopMixin at media start + a documented call for
bare-SDK embedders) and disables the wobbler; the sink refuses loudly when not started; it normalizes scale, then
resamples to `get_output_audio_samplerate()`; `response_output`'s pre-resample and `tools/response.py`'s unconditional
`spoken: True` go in the same PR; both sinks are covered; a test pins the wobbler off.

## Facts established against the code (SDK floor reachy-mini 1.8.3, installed locally)

1. `MediaManager.push_audio_sample` returns `None` always. With `media.audio is None` it logs "Audio system is not
   initialized." and drops. `GStreamerAudio` (LOCAL) builds `_appsrc` in `__init__`, so before `start_playing()` the
   push reaches a NULL-state pipeline and drops with "push_buffer dropped: FLUSHING" — the issue's "AppSrc is not
   initialized" text is the WebRTC client (`_appsrc` set only on connection; its `start_playing()` is a no-op).
   **So option (b), reading `_appsrc`, is unsound at our floor**; the only private truth would be
   `_pipeline_playback.get_state()` (LOCAL) and `_appsrc` (WebRTC) — two private shapes per backend per SDK version.
2. `ReachyMini.release_media()` / `acquire_media()` close and RE-CREATE `media_manager`; maxim's soft reconnect
   (`embodied_runtime/connection.py`) replaces `mini` entirely. A started player is a property of one
   `MediaManager` object, not of `mini`.
3. `ReachyMini.disable_wobbling()` (exists at 1.8.3) = `media_manager.disable_wobbling()` + daemon
   `SetWobblingCmd(enabled=False)`; `MediaManager.disable_wobbling()` alone is SDK-side only. Neither commands motion.
   `start_playing()` starts the SDK wobbler only if one exists.
4. Sinks that reach `push_audio_sample`: (i) `utils/audio.py::push_audio_to_device` (callers:
   `MediaLoopMixin.speak` → `ResponseOutput` speaker_fn in `agentic_runtime.py` under `MAXIM_TTS_ENABLED`;
   `make_device_speak_sink` → pulse's pairing announcer); (ii) `hardware/reachy/streams.py::ReachyAudioStream.push_sample`
   (only caller: `Selfy.push_audio_sample`, itself with no `src/` caller — public surface only); (iii)
   `Selfy.push_audio_sample`'s direct-SDK fallback (same lie, third copy).
5. R6 (measured): `ResponseOutput._speak_response` pre-resamples int16 22,050→16,000 with `resample_poly`, which returns
   float64 on the int16 scale; `_to_device_float32` passes floats through unscaled → device sees peak ≈ 20011.
   The pre-resample exists because the speaker contract is `speaker(audio)` with no rate.
6. Pulse #57 bootstrap (pulse origin/main `apps/reachy/maxim_reachy_app/app.py`) calls `media.start_playing()`,
   then `reachy_mini.disable_wobbling()`, then later `media.stop_playing()`, directly on the SDK, and pins that order.
7. `MediaLoopMixin.live` is pinned at 587 lines (rule T).

## Design

**D1 — "started" is a record maxim's lifecycle helper writes, keyed on the MediaManager object (option a).**
New in `utils/audio.py` (one module owns the sink and its lifecycle):
- `start_device_player(mini) -> bool`: `mini.disable_wobbling()` (fall back to `mini.media.disable_wobbling()` for a
  handle without it, WARNING) → `mini.media.start_playing()` → read `get_output_audio_samplerate()`; on `-1` or
  `media.audio is None` record nothing, WARNING, return False. Else record `{media: rate}` in a lock-guarded
  `weakref.WeakKeyDictionary` (dies with the manager, so `acquire_media`/reconnect invalidate it with no hook).
  Idempotent per manager: a second call on the same manager is a no-op returning True (no double start).
- `stop_device_player(mini) -> None`: pop the record, then `mini.media.stop_playing()`.
- The sink reads the record for `mini.media`: absent → False + WARNING naming `start_device_player` (rate-limited,
  one per manager, so a stopped player during shutdown does not flood).
Rejected: (b) — fact 1. (c) ask Pollen — worth filing upstream, cannot gate 1.3.4. A typed `DevicePlayer` token
passed to the sink instead of `mini` (push the invariant into the type): the embodied-device plan's step 3 owns that
type (the reshaped `AudioStream` holding PLAYS), and a token now would be a third API shape to deprecate. (The
draft's reason, keeping `make_device_speak_sink(mini)` unchanged for pulse, did not hold: pulse changes anyway, Q1.)
Every path fails loudly meanwhile: False, a WARNING, `spoken: False`.

**D2 — scale, then rate, in one conversion.** `_to_device_float32` becomes strict: int16 → /32768-style scale;
float input must already be in [-1, 1] (peak > 1.0 + epsilon → refuse, False + WARNING, never guess the scale).
Then `resample_poly` to the recorded device rate, clip to [-1, 1], float32. Duration preserved within one sample
period per 1/gcd block.

**D3 — the speaker contract carries the rate and the truth.** `ResponseOutput` calls
`speaker(audio, sample_rate=tts.sample_rate)` and stops pre-resampling; `_speak_response` returns the speaker's bool;
`tools/response.py` reports `spoken` from it (V's file: one hunk). `LocalSpeaker.__call__` accepts `sample_rate`.
`MediaLoopMixin.speak` already takes it.

**D4 — the second and third sinks.** `ReachyAudioStream.push_sample` and `Selfy.push_audio_sample`'s fallback route
through `push_audio_to_device`'s device half (one implementation; no local fallback for the stream: a stream is the
device or nothing), so all three refuse on the same record.

**D5 — the runtime's owner.** `MediaLoopMixin.live` calls `start_device_player(self.mini)` at media start (the block
that reads the audio rates, extracted into a helper so `live` shrinks) and `stop_device_player` in its `finally`
before `_release_media` (also through an extracted teardown helper). Only when `self.mini` exists and the media
backend has audio. The soft reconnect (`connection.py`) gets a new manager → the record is gone → the sink refuses
loudly until restarted: **needs a restart hook in `connection.py::_soft_reconnect` (no session owns it; fence
question below).**

**D6 — tests.** Strict red gate first (xfail strict, flipped in place): an SDK-faithful fake whose push before
`start_playing()` drops → sink returns False with a WARNING. Then: 22,050 Hz int16 reaches the fake at its reported
rate, duration within tolerance, peak ≤ 1.0 (R6); the wobbler is off after `start_device_player` on a fake with
wobbling ON, and `start_playing` was never called with a live wobbler (order pinned); `live()`'s helper starts the
player exactly once and `speak` never calls `start_playing` (a fake that counts); record dies on `acquire_media`-style
manager swap; `spoken` false when the sink fails. Prove each by deleting the mechanism.

## Owner questions (asked together)
Q1. Pulse's bootstrap starts the SDK directly, so under D1 maxim's sink refuses pulse's audio until pulse calls
    `maxim.utils.audio.start_device_player(reachy_mini)` instead. Rec: yes — pulse replaces its three SDK calls with
    our start/stop pair (the "documented call"), told in the note to pulse with the 1.3.4 candidate; pulse's own
    order test moves to assert it calls ours. Alternative: the sink trusts an embedder flag
    (`make_device_speak_sink(mini, player_started=True)`) — keeps pulse unchanged, but the flag is an attestation, not
    a record.
Q2. Order: disable the wobbler BEFORE `start_playing()` (rec: then start_playing can never start a wobbler, even for a
    moment), vs the decided/pulse-pinned start-then-disable.
Q3. The soft-reconnect restart hook in `embodied_runtime/connection.py`: a named fence exception for one call (rec), or
    leave speech refused-loudly after a reconnect until the embodied-device plan's `embody()`.
Q4. `Selfy.push_audio_sample` + `ReachyAudioStream.push_sample` have no `src/` caller: fix both as D4 (rec — the owner
    said both sinks), or mark the pair Dormant and fix only the reachable path.

## Folds from the adversarial design pass (2026-10-24; findings F1–F16)

- Fact 1 corrected: at 1.8.3 the WebRTC push is a SILENT return; "AppSrc is not initialized" comes from a newer SDK
  (the robot's). D1 depends on no SDK internals, so it holds across versions (F15).
- Fact 2 corrected: `release_media()` and maxim's `media.close()` leave the SAME manager with `audio=None`; only
  `acquire_media` re-creates. Fold (F1): the sink requires record AND `media.audio is not None`; `_release_media` and
  the reconnect close pop the record (connection.py, fence Q3).
- Fact 3 corrected: `disable_wobbling`/`stop_playing` send `SetSpeechOffsetsCmd(zeros)` (a return to zero offset,
  benign). Fold (F4/F5): disable FIRST, then start; if `ReachyMini.disable_wobbling` is absent or raises, record
  nothing and do not start (False + WARNING); repeat calls re-issue the disable. Daemon-side re-enablement by
  another client is not observable (documented).
- F2: non-`GStreamerAudio` backend (WebRTC) → no record (plan step 3: WebRTC cannot play). Decision table Q4.
- F3: released pulse (`>=1.3.3,<2.0`) starts the SDK directly → under D1 its announcer goes silent on upgrade
  (the announcer ignores the bool; `pairing_ready` is synthesis-only). Rollout is Q1 — do not publish 1.3.4 before it.
- F6: start the player in an extracted helper BEFORE `_start_agentic_runtime`, gated on output capability
  (`has_audio`), not the mic flag `self.audio`.
- F7: one private device-push primitive used by both sinks; `ReachyAudioStream.push_sample` contract is "float32
  at `output_sample_rate`", checked and refused otherwise. Records the inversion of plan §2; step 3 reshapes it.
- F8: the old-behaviour pins that flip: test_device_speak_sink.py (5 tests named in the review) + pulse
  test_app.py:527 and scripts/smoke_reachy_release.py (named in the note to pulse). The dead "webrtc"/"not
  implemented" string-match branch is deleted.
- F9: decision table written and pinned row by row (Q4).
- F10: `speaker_fn` typed `Callable[..., bool]`, CHANGELOG line for the contract change, `output_response` surfaces
  the bool.
- F12: refuse non-finite and non-int16 integer dtypes; scale int16 by /32768. F13: keep the clip; full-scale-square
  case in D6. F14: hold the record lock across check → push (non-blocking). F16: close docs/bugs D87(a).

## Owner decisions 2026-10-24 (after the design pass)
- Q1 rollout: PULSE SHIPS FIRST. Pulse releases a build calling start/stop_device_player when present (getattr
  fallback to the SDK) and flips its order pin; 1.3.4 does not publish until that pulse release is out. Sink strict.
- Q2 order: DISABLE the wobbler, THEN start_playing; absent/raising disable → never start (False + WARNING); repeat
  starts re-issue the disable.
- Q3 fence: named exception for embodied_runtime/connection.py — pop on close, restart after a soft reconnect.
- Q4 (the decision table; the draft's Q4 about the caller-less sinks was already answered by "cover both sinks"):
  LOCAL-but-not-started → refuse (False + WARNING); WebRTC/non-LOCAL → local speakers + one WARNING (spoken = played
  locally); MAXIM_TTS_LOCAL and mini=None → local. **The WebRTC row is REVISED below.**

## The three-lens review round (2026-10-24, on staged tree 6c05039e) and its folds
No BLOCKER from Executor, Architecture or Wire integrity. Owner decisions taken from it:
- **WebRTC row revised** (Wire integrity SF-1: at SDK 1.8.3 a connected WebRTC push reaches the robot, so the
  draft's "WebRTC cannot play" was wrong): the robot when `start_device_player` ran AND the send chain has been built,
  read through one typed private probe (`_webrtc_send_chain_ready`: `media.audio._appsrc is not None`) pinned by a
  contract test against the installed SDK; otherwise local speakers with one WARNING per state. `no_media` → local;
  any other or missing backend → refuse (SF-4), never a silent re-route. `start_device_player` runs for WebRTC too:
  its `disable_wobbling()` also turns off the daemon-side wobbler that incoming WebRTC audio drives.
- **`selfy.py` named fence exception granted.**
- D1's rejection of option (b) stands for LOCAL (`_appsrc` exists before `start_playing`); for WebRTC `_appsrc` is
  the only signal there is, read in one place. The delta round (Architecture SF-1) found it means "built", not
  "connected": the SDK never clears it, so **after a mid-session peer drop a WebRTC push is a silent drop reported as
  played. Known gap, recorded, not closable without more private state**; the device plan's step 3 (PLAYS) owns it,
  and the contract test fails if the SDK starts clearing `_appsrc`, forcing the docs to be revisited.

## The delta round (2026-10-24, staged tree 42e4ea56 against 6c05039e)
No BLOCKER. Wire integrity: no further round owed. Folds:
- Executor SF-1: `start_device_player` probes `weakref.ref(media)` BEFORE `start_playing` (the old guard was dead:
  `WeakKeyDictionary.__contains__` returns False instead of raising, so the player started and then the record raised).
- Executor N-2/N-6: the re-check under the lock compares the recorded rate and checks closed audio again.
- Executor N-3: a non-numeric `sample_rate` is refused, not raised.
- Executor N-4/N-5 and Wire N-1..N-5: the "never started but chain built" WebRTC row, the defensive lines, the
  `selfy` rate contract, and "one WARNING per manager until its next start" (not "per state").
- Architecture SF-1/SF-2, N-1/N-2: "built", not "connected", everywhere, with the gap above; the SDK contract test
  skips in CI until the R2 reachy cell exists (M69), so it is a local check today; the brief's refusal is LOCAL's.
- Architecture SF-3: the release-guide line for M78 is `docs/publication_guide.md`'s (the process session's file);
  handed off as a stub in the PR description.

Folds:
- Executor SF1: `on_failure` (which may soft-reconnect under the runtime's media lock) runs after the record lock is
  released; the conversion runs outside it, and the record is re-checked under it before the push (F14).
- Executor SF5 / Architecture SF4 (corrects F7): `ReachyAudioStream.push_sample` resamples from the rate the stream
  advertises (`output_sample_rate`) to the recorded device rate; F7's "checked and refused" was never built.
- Architecture SF5: `sample_rate` is required on `push_audio_to_device`, and keyword-only on the
  `make_device_speak_sink` sink and `MediaLoopMixin.speak` (a silent 16 kHz default replayed #1435).
- Architecture SF7: the source scan also forbids a direct `.push_audio_sample(` outside `utils/audio.py`.
- Architecture SF6: a contract test binds the fake's assumptions and the WebRTC probe to the installed SDK.
- Wire SF-3 / Executor SF6: the `make_device_speak_sink` and `models/audio/tts.py` docstrings show the start/stop pair.
- Wire N-1: the pairing announcer logs (type-free, never the code) when the sink returns False.
- Executor SF2–SF4: tests for the lock (a stop waits for an in-flight push; a stop during the conversion refuses),
  stop-before-close on reconnect, and every refusal row of the lifecycle calls.
- Release gate (Architecture SF1, Wire SF-2): mechanization backlog **M78**, and the gate line in roadmap_1_3_x.md
  §1.3.4 (added after PR #1448 merged).
- Not folded, recorded: N-3 (`live` starts the player before its `try:`; the CLI's `shutdown()` covers it, and the
  record stays true); N-5 (`speak` returns `success=True` with nothing delivered when `also_print=False`; it
  predates this, Session V's file); `ReachyMini.cancel_move()` stops the player behind the record (no maxim caller;
  noted in the `_PLAYERS` comment); one owner per device is guarded only at `embody()` (device plan §4).
- **Deletion probes:** 27 mechanisms, each deleted on its own in place and restored; all 27 turn a test red (the
  probe list is in the PR description).

## Built (2026-10-24), and what stays open
- `utils/audio.py`: `start_device_player`, `stop_device_player`, `push_to_started_player` (resamples from the given
  `sample_rate`), the strict `_to_device_float32`, and the decision table in `push_audio_to_device`.
- Callers: `MediaLoopMixin._start_media_audio` (extracted from `live`, before `_start_agentic_runtime`; `live`'s pin
  587 → 566), `ConnectionMixin._release_media` and `_soft_reconnect` (pop before close, restart after), and
  `ReachyAudioStream.push_sample` plus `Maxim.push_audio_sample`'s fallback (`selfy.py`; its named fence exception
  was granted in the review round).
- F10: `ResponseOutput.output_response` keeps returning the saved file's path (its callers use it); a refused
  utterance there logs a WARNING. The `speak` tool, the one caller that reports speech, returns the truth.
- Open: the R2 reachy cell's device smoke (audio reached a STARTED player at the device rate, floor and latest) is
  #1435's hardware check; R1's in-memory "has spoken" stays the readiness signal until the device plan's step 3.
