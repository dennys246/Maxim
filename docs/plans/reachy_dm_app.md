# Reachy Adventure Mode — the embodied game master (flagship experience)

**Status:** Shell plan, drafted 2026-07-23. Companion to [reachy_mini_app.md](reachy_mini_app.md). Post-1.1 flagship track — *not* an MVP dependency.
**Scope:** Reachy Mini as a voice **Dungeon Master** with a personality, guiding human players through a campaign — narrating aloud (TTS), listening (STT), facing whoever speaks (orient/DoA), and **carrying what it learns into every other mode** via one persistent agent. This is the app's headline experience; Talk and Rest are the supporting cast.
**Target:** Post-1.1. Rides the existing `DMRuntime` + audio stack; the genuinely new work is the real-time conversational loop, voice placement, the same-agent memory wiring, and content.
**Gates:** Depends on the MVP app shipping first (setup + Talk + memory view) and on **P0 (substrate fits the Pi)** from the companion plan. The main unknown is voice-loop *latency* on the Pi (STT+LLM+TTS round trip) — **not** turn-taking, which push-to-talk reduces to the DM's ordinary response (see Risks).

**Why this is the flagship:** "Talk to a robot" is a talking head with a face — the embodiment is decoration. A DM is the first experience where the body is *load-bearing*: it needs a voice, expressive motion, to face the speaker, a personality, and to remember your party across sessions. Every research + hardware investment (orient/Exp 45-48, persona, TTS, cross-session memory) gets a natural job, and none is decorative. It is also the most *legible* showcase of the 1.0 cross-session thesis — vivid story memory lands harder than remembered facts.

**In the shared-UI stack** ([maxim_console.md](maxim_console.md)): Adventure is a **RunSurface mode** (Layer 2) over the **embodied `HANDLE`** (Layer 1). The general console gets the same DM as a headless RunSurface mode — so building Adventure well here also delivers "run a DM campaign" in the console for free.

**Adventure's UI ports the interactive `Live` display's spread** ([interactive/display.py](../../src/maxim/interactive/display.py)): narrative/dialogue bright, bio-stack activity (nac/hippo/ec/pain) dimmed *underneath*, plus a thinking panel + `DisplayExtension`-style DM panels (character sheet / inventory / encounter). This makes cross-session learning **visible in the moment** — you watch it remember your rogue's betrayal *as the DM narrates* — which is the strongest legibility surface for the thesis. It's a renderer over the existing `sim_log`/`observe` event model, not new machinery.

---

## The load-bearing invariant: ONE persistent agent across all modes

**Adventure, Talk, and Rest are modes of a single persistent Maxim agent — not separate agents.** Same `agent_id`, same bio-stack, same `~/.maxim/` home. This is the whole reason cross-mode learning works, and it is **not** how the stock sim path behaves.

**The trap (confirmed in code):** the sim/DM orchestrator builds a throwaway AUT — `agent_id="sim_aut"` ([`simulation/orchestrator.py:534`](../../src/maxim/simulation/orchestrator.py)) — whose Hippocampus/NAc persist to *session-scoped* files (`~/.maxim/sessions/{id}/aut_hippocampus.json`), with resume reading the previous *session* dir. On that plumbing, everything Adventure learns lands in a per-campaign scratch agent the Talk agent never reads. **The headline promise ("what it learns in Adventure comes to Talk") silently fails.**

**The requirement:** Adventure runs the `DMRuntime` against the app's **persistent conversational agent**, not a fresh `sim_aut`. The DM is the *orchestrator/narrator*; the persistent agent is the *learner* (it learns about the human player + the shared story). This likely means a pymaxim seam — a way to drive `DMRuntime` with an externally-owned agent + memory home rather than the orchestrator's self-constructed AUT. Scope that seam in pymaxim; do not fork `DMRuntime` in the app repo (keep-it-thin rule).

**Regression-guard shape (for the eventual build):** a test that plays one campaign turn, then asserts the resulting episode is recallable from the *persistent* agent's Hippocampus (same `agent_id`, same home) — not just from a session-scoped AUT file.

## Selective transfer by provenance (fiction must not become fact)

A DM operates in fiction. Transfer must be *selective*:

| What | Provenance | Fate |
|---|---|---|
| "We played this campaign; your rogue betrayed the party" | real episodic | **persists** — this is the delightful cross-session memory |
| Model of the human player (cautious? roleplayer? likes combat?) | real preference | **persists** — carries into how Talk converses |
| In-fiction world-facts ("dragons breathe fire", "this NPC is evil") | `imagined=True` | **decays** — must not leak into Talk as grounded truth |

The machinery already exists and is already used: `Episode.imagined` / `CausalLink.imagined`, plus `tag_imagined_links` + `decay_imagined_links(0.5)` at session end ([`orchestrator.py:2740-2743`](../../src/maxim/simulation/orchestrator.py)). The work is pointing that selectivity at the *persistent* agent: fiction fades, the shared experience and player-model stay. **Do not disable the imagined-decay when moving to the persistent-agent design** — it is what keeps the transfer safe.

---

## What already exists (~60% is wiring, not greenfield)

| Piece | Where | Status |
|---|---|---|
| DM turn loop w/ human-in-the-loop | [`simulation/dm_runtime.py`](../../src/maxim/simulation/dm_runtime.py) (`DMRuntime`, `_prompt_human_choice`, `_deliver_and_wait`) | exists |
| Free-text → choice classification | `dm_runtime._get_choice(response, encounter)` (LLM fallback) | exists — voice answers ride this |
| TTS (on-device) | `TTSEngine` (piper) [`models/audio/tts.py`](../../src/maxim/models/audio/tts.py); `speak` affordance + `speaker_fn` | exists |
| STT | `faster-whisper` (`audio` extra); `agent_loop` consumes `pending_voice_input` ([agent_loop.py:1999](../../src/maxim/runtime/agent_loop.py)) | exists |
| DM display panels (character sheet, inventory) | [`interactive/display.py:54`](../../src/maxim/interactive/display.py) | exists |
| Personas | ~~`simulation/personas.py`~~ | DELETED in 1.1 (persona hard-remove — label-only dead code) |
| Orient-to-speaker (DoA + head-frame) | Exp 45/48, perception placement | exists |
| Cross-session memory | bio-stack in `~/.maxim/` | exists |

## The genuinely new 40% (where the work + risk live)

1. **The voice loop — simpler than a general voice assistant, because Adventure is turn-based.** With **push-to-talk** (antenna tap / dashboard button) the human sets the utterance boundary explicitly — no VAD, no endpointing, no barge-in, no open-mic diarization; whoever taps is the speaker. That collapses "turn-taking" to what the DM already does: hand the transcript to `DMRuntime`, and the DM either advances the story *or* asks a clarifying question **as part of its normal narration response** (`_get_choice` + a "didn't catch that" branch) — **no dedicated turn-taking module.** Push-to-talk also cuts TTS on press, so interrupting a long narration is free. Residual work is plumbing (STT invoke → transcript → DM → TTS playback) — engineering, not an AI problem. **VAD / hands-free open-mic is an optional post-MVP "magic" upgrade, explicitly out of scope.**
2. **Voice-loop latency on the Pi — the real unknown (measure, don't architect around).** A turn is STT → LLM narration → TTS. Whisper on a Raspberry Pi is not fast; piper TTS is okay; the LLM is remote (mesh/cloud). **But Adventure has high latency tolerance** — a DM is *expected* to deliberate, so a 4–8s beat reads as immersive, not broken (the same gap would break casual chat). So this is a *spike-and-measure* task, not a real-time-architecture problem. If Pi-side STT is the bottleneck, it becomes a **placeable stage** — exactly what [perception_pipeline_placement.md](perception_pipeline_placement.md) exists for: run STT on the owner's leader. That plan now has a concrete consumer driver.
3. **The orient synthesis (the magic).** "The DM turns to face whoever speaks" = DoA + Exp 45/48 orient + head-frame fix, in service of a social moment. Most delightful piece; a real integration, not free. (Push-to-talk already tells us *who* the speaker is; orient makes it *feel* alive.)
4. **Content + safety.** A few genuinely good seed campaigns, plus guardrails for improvised LLM narration to families/kids. Content is a product surface. The imagination system can extend/generate campaigns, but the floor is a couple of hand-authored ones.
5. **The persistent-agent + provenance wiring** (above) — the seam that makes the flagship claim true. **This — not turn-taking — is the highest-risk piece**, because it fails silently (learning lands in a throwaway AUT the Talk agent never reads).

---

## Relationship to the deferred DM Extensions plan

Keep these separate:
- **This plan** = a new *voice/embodiment front-end* over the existing `DMRuntime`. Generates real usage.
- **[deferred/dungeon_master_extensions.md](deferred/dungeon_master_extensions.md)** = *content-depth follow-ons* (encounter library, adaptation engine, sub-sim isolation…), explicitly gated on "revive when real DM usage produces a signal." **Do not revive that list preemptively** — this front-end is what produces the signal that tells you which extension matters. (E.g. "users complain about encounter repetition" → encounter library; "the DM steamrolls players" → adaptation engine.)

## Main-page placement (updates the companion plan)

The consumer main page becomes three flagship actions over one persistent agent:

```
 ├─ [ 🎲 Adventure ]   — voice DM campaign (flagship)
 ├─ [ 💬 Talk ]        — free conversation; recalls what Adventure taught it
 ├─ [ 😴 Rest ]        — ProcessingState.SLEEP (not app-kill)
 ├─ ✦ What Maxim remembers about you   — now full of story: characters, choices, party
 └─ ▸ Developer drawer: AUT simulation · logs · doctor
```

Note the AUT/substrate-primary *simulation* (fake percepts, debug) stays in the developer drawer — it is a different thing that shares the word. Adventure is human-facing; the AUT sim is not.

## Sequencing

1. **MVP first** (companion plan): setup + Talk + memory view. De-risks the platform + P0 (Pi resources). Ships without Adventure.
2. **Spike the voice-loop latency** on the Pi (Risk #2) with push-to-talk — the real residual unknown. Turn-taking itself needs no prototype: push-to-talk + the DM's normal response handles it (Risk #1).
3. **Adventure track** post-1.1: the persistent-agent seam (Risk #5, the real risk) + voice-loop plumbing + voice placement + orient integration + 2 seed campaigns.
4. **Extensions**: demand-driven, per the deferred plan's revive gate, from real Adventure usage.

## Open questions

1. **pymaxim seam for "drive `DMRuntime` with an externally-owned persistent agent."** Design it so the app never forks `DMRuntime`. First DM-side pymaxim addition.
2. **Turn-taking model — RESOLVED for MVP: push-to-talk, two press surfaces.** The human sets the utterance boundary; no VAD/endpointing needed; the DM's normal response absorbs the "complete vs clarify" decision. VAD/hands-free is a post-MVP magic upgrade, not required.
   - **On-screen button (dashboard/phone) = the guaranteed baseline.** Always works, accessible, no hardware dependency — the loop is never gated on unverified input hardware.
   - **Antenna tap = delightful enhancement, layered on top** when the hardware affords a touch/capacitive read (verify against the SDK/hardware; degrade silently to button-only if not).
   Both map to the same "start/stop capture" signal into the voice loop, so supporting two surfaces is one event, two triggers — not two code paths.
3. **Where STT runs:** Pi-local (simple, maybe too slow) vs leader-placed (fast, needs the perception-placement cut point). Measure Pi whisper latency first.
4. **TTS voice + persona binding:** one DM voice, or persona-selectable? Piper is on-device (private, free, slightly robotic — arguably on-brand for Reachy). Cloud TTS = better voice, more cost/latency/privacy surface.
5. **Safety posture for improvised narration** to families — content guardrails + a documented stance in the store listing.

## References

- [reachy_mini_app.md](reachy_mini_app.md) — the MVP app + setup + memory view (this plan's prerequisite)
- [`simulation/dm_runtime.py`](../../src/maxim/simulation/dm_runtime.py) · [`simulation/campaign_runner.py`](../../src/maxim/simulation/campaign_runner.py) · [`simulation/orchestrator.py`](../../src/maxim/simulation/orchestrator.py) (the `sim_aut` trap + imagined-decay)
- [`models/audio/tts.py`](../../src/maxim/models/audio/tts.py) · `audio` extra (faster-whisper) · [`interactive/display.py`](../../src/maxim/interactive/display.py)
- [perception_pipeline_placement.md](perception_pipeline_placement.md) — STT as a placeable stage
- [deferred/dungeon_master_extensions.md](deferred/dungeon_master_extensions.md) — demand-driven depth follow-ons
