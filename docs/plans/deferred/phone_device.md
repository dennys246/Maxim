# Phone device — a phone (later a watch) as a sensing device and remote UI

**Status:** PROPOSED 2026-10-29, plan only. **Phase 0 is homed in 1.3.5** (owner decision 2026-10-29, §Phase 0).
Phases 1–3 start **after the 1.4.0 cut**; nothing here holds 1.4.0. **Revive trigger for Phase 1:** 1.4.0 is
published, Phase 0 has landed, and [embodied_device.md](../embodied_device.md) has built `embody()` (its step 1)
and the leader STT route (its step 5).

**Input:** a four-lens read-only review on 2026-10-29 (security, architecture and wire integrity, pulse
feasibility and testing, bio-faithful value and sequencing). Its reports live only in that session's transcript,
the same disclosed gap as O6. Every load-bearing code claim was re-checked before this file was written: the
loopback bind (`console/server.py::run_serve`), the control fields in the wire whitelist
(`agents/bus.py::_PERCEPT_WIRE_FIELDS`), and the phone app as the motivating goal of
[console_tunnel_hardening.md](console_tunnel_hardening.md). One reviewer claim was wrong and is not used: there is
no LAN bind; A9 trusts extra origins, while the server still binds loopback only.

**Companion plans, authoritative in their own areas:** [embodied_device.md](../embodied_device.md) (one owner per
device contract; `embody()`; the embodied census) · [console_tunnel_hardening.md](console_tunnel_hardening.md)
(the remote trust boundary; PR 3 and PR 4) · [advisor_mode.md](../advisor_mode.md) (the atonia gate) ·
[thalamic_relay.md](../thalamic_relay.md) (receptors, provenance kinds) ·
[mesh_perception_transport.md](mesh_perception_transport.md) (peer-to-leader percepts) ·
[maxim_ui.md](../maxim_ui.md) (the pulse kit and its shells) · [maxim_hivemind.md](../maxim_hivemind.md).

---

## Owner decisions (2026-10-29)

| # | Decision |
|---|---|
| PD1 | **The phone attaches through `embody()` as a device, and is not a second robot body.** Its outputs (speaker, haptics, notifications) are in the embodied census, so the atonia gate can inhibit them. It never registers a `maxim.robots` controller: it has no motor command whose consequence Maxim senses. So M76's second-body tripwire, [second_body_staging.md](second_body_staging.md) and the microduck plan stay asleep. |
| PD2 | **Remote reach is cloudflared only** for v1: one path to harden, and the one the tunnel plan was written for. `tailscale serve` is recorded as "not now, additive later", not as rejected. `maxim serve` keeps its loopback bind as an invariant; there is no LAN bind. |
| PD3 | **Atonia with a device attached is planned now** (§Atonia with a device attached), so `embody()` and the gate are not redesigned later. It is designed now and built after the atonia gate ships and `embody()` exists. |
| PD4 | **Phase 0 is homed in 1.3.5.** This re-homes console_tunnel_hardening's PR 3 (admission control), a recorded 1.3.3 move that did not ship. No remote reach ships before it. |
| PD5 | **The camera stays in scope** as user-directed perception ("look at this"), not as an orienting sensor: the agent cannot point it. Frames reach the operator's own leader only. |

## What the phone is, and what it is not

- **A console client, not a mesh peer.** It authenticates to `maxim serve` with a per-device console token. It
  never holds the mesh `api_key`: one credential must not grant both inference and console access
  ([runtime-tools.md](../../agents/runtime-tools.md), the console-auth invariant). So the cluster-keyed
  `/v1/mesh/percepts` family in [mesh_perception_transport.md](mesh_perception_transport.md) is the wrong ingress
  for a phone (amended there).
- **Not an inference peer.** It does not serve an LLM to the mesh (thermal limits, OS background limits, tiny models).
- **A device under `embody()`** (PD1), with the device contract owned by pymaxim. maxim-pulse owns capture and UI only.
- **Its percepts enter through the relay** as registered receptors with `reported` provenance (M39: every percept
  reaches the EC through the relay's one registration point). No `PerceptSource` side door.
- **Its device id comes from the token, never from the body** of a request.

**The hivemind carries what was learned, not what was seen.** [maxim_hivemind.md](../maxim_hivemind.md) shares
substrate bundles (NAc weights, EC concepts, ATL relations, reflexes), and `hive contribute` is write-only. The
path is: the phone's camera, then the operator's leader perceives, then the substrate learns, then what was learned
may be contributed. Raw frames never reach the hivemind, so another person's room never crosses into someone
else's Maxim.

## Front-gate scope pressure

| Piece | Rides on | Verdict |
|---|---|---|
| The UI | the pulse kit: `DeviceProfile`, `bootConsole`, the auth flow, `HttpFacade`, `EventClient` | **rides**: a third shell, from the Reachy profile template |
| Auth | `_GuardMiddleware`, bearer auth, A9.1 pairing | **rides, reshaped**: a principal seam (P0-3), then PR 4's per-device tokens |
| Speech | embodied_device step 5's leader STT route; step 1's framed `talk(source=...)` | **rides**: no second STT route |
| Sensor ingress | the relay's receptor registry (GL3.B1) | **rides**: a remote-receptor row, `reported` provenance |
| Outputs | `embody()` and the embodied census | **rides**: a phone device adapter |
| Remote transport | cloudflared, console_tunnel_hardening | **rides** |
| A phone-specific transport or package | none | **not built**: embodied_device §2 forbids a new top-level package |

## What each sensor is worth

| Modality | Consumer | Verdict |
|---|---|---|
| Location + activity (GPS, IMU collapsed to an activity label) | SCN (`time/oscillator.py` runs circadian to annual scales, and every run so far is minutes long); Hippocampus `situation=` context | **The strongest case.** Weeks of input at the scale the SCN was built for. Phase 2's experiment. |
| Mic → transcript | the talk path; the word world of [grounded_language_acquisition.md](../grounded_language_acquisition.md) | **Real.** Phase 1's slice. Enters as `reported` provenance; language stays off the E-ladder (G5). |
| Camera | perception, after placement and the fabric revive | **User-directed perception only** (PD5). Earns no orienting claim. Phase 3. |
| IMU as proprioception | none | **Not built.** It measures the human, not the agent; the Cerebellum needs the agent's own motor command. |
| Watch heart rate | none | **Never interoception.** It is the user's physiology, an observation of another agent. Never written into drives, `vital_metrics` or `InteroceptiveOutcome` (G16, M78). |
| Speaker, haptics, notifications | the social loop | Outputs under `embody()`'s census (PD1); inhibited under atonia. |

## Phase 0 — 1.3.5 (PD4)

Ranked. "Lenses" names the reviewers that raised it independently.

| # | Item | Why now | Severity | Lenses |
|---|---|---|---|---|
| P0-1 | **`Percept.from_wire_dict` becomes a trust boundary.** The control fields in `_PERCEPT_WIRE_FIELDS` (`hard_override`, `has_voice_command`, `has_maxim_keyword`, `cli_input`, `explore_command`) are fixed server-side, as `comms/gateway.py::CommunicationGateway.receive_inbound` already does; `sensory` is rehydrated through `SensoryTag.from_dict` (a raw dict today reads as non-SOUND in `sim_adapter.py`, skips entity novelty in `default_network/network.py` and raises in `memory_agent.py`); `PerceptContext.channel` is validated at runtime. `tests/unit/test_percept_wire_format.py` changes with it. | No producer exists yet, so it is cheap; a latent defect with or without a phone | SHOULD-FIX | architecture |
| P0-2 | **console_tunnel_hardening PR 3, admission control**: `limit_concurrency`, `ws_max_size`, an input cap outside the sandbox, a rate limit on `/api/pair/claim`. | Needed for any remote reach | SHOULD-FIX | security, pulse |
| P0-3 | **A principal seam in `_GuardMiddleware`.** `_bearer_authorized` returns `{device_id, scopes}` in `scope["state"]`, not a bool; a path→scope table defaults unknown routes to `admin`. Today's token is the operator entry, so behaviour is unchanged. | PR 4's per-device tokens become additive, not a refactor | SHOULD-FIX | security |
| P0-4 | **M3: the Talk registry is built without `bash`, write, edit and execute tools**, rather than kept out by `handle.py::talk_autonomy`'s deny derivation. | Before any remote turn reaches the loop | SHOULD-FIX | security |
| P0-5 | **Reserve the `reported` provenance kind and an "other body" source class** at GL3.B1 ([thalamic_relay.md](../thalamic_relay.md) §11, TR11), with a remote-sensor row in `tests/unit/test_receptor_census.py::CENSUS`. | Keeps a user's speech and physiology out of interoception, and gives the phone its way in | SHOULD-FIX | research, architecture |
| P0-6 | **Device identity on the console contract**: a device/origin id beside embodied_device step 1's `source=`, in the same bump as R3's 0.7.0 ([#1336](https://github.com/dennys246/Maxim/issues/1336)). M70's note to pulse covers the interim. | One contract bump instead of two | SHOULD-FIX | architecture |
| P0-7 | **embodied_device steps 1 and 5 are body-agnostic**: the framed talk ingestion and the leader STT route take a device id and are not tied to the Reachy SDK. A design-review note on that plan. | The phone becomes an adapter, not a fork | SHOULD-FIX | architecture, pulse, research |
| P0-8 | **O2 / [#784](https://github.com/dennys246/Maxim/issues/784)**: the A4 gain inverts a place code. | Location is exactly a place-coded sensor | SHOULD-FIX | research |
| P0-9 | **A privacy design for evidence**: a derived, quantized record schema for location and activity, and `scripts/lint_secrets.py` (M45) extended to coordinates. | Must precede the first capture, or M1's committed-evidence rule and privacy deadlock | SHOULD-FIX | research |
| P0-10 | **Every declared SCN `event_type` has a producer or is marked reserved**: item 3 of [scn_event_producer_gap.md](scn_event_producer_gap.md). | Tests only; a tripwire before a phone producer arrives | SHOULD-FIX | research |
| P0-11 | **Pulse kit groundwork** (a note to pulse; pulse owns it): reconnect on `visibilitychange` and `online` with catch-up from `/api/runs` in `facade/events.ts`; a size budget per profile (generalize `scripts/size-reachy.mjs`); an e2e device server that takes an origin and a TLS argument. | A hidden laptop tab has the reconnect problem today | SHOULD-FIX | pulse |
| P0-12 | **Doc reconciliation** (this PR): `maxim_ui.md`'s localhost line; mesh_perception_transport's ingress for phones; M76's wording (PD1); the tunnel plan's PR 3 homing. Pulse's own `docs/plans/maxim_ui.md`, `CLAUDE.md` and `AGENTS.md` carry the same localhost line: a note to pulse. | Otherwise three revive triggers misfire | SHOULD-FIX | all four |
| P0-13 | **NITs**: loopback-origin trust in `_origin_allowed` becomes scheme-aware (http/https only; `capacitor://localhost` named explicitly); an additive-only check in pulse's `gen:facade:check`, warn-only until a store binary exists; responsive kit layout. | Cheap | NIT | security, pulse |

**Not built before 1.4.0 (DO-NOT-BUILD):** a sensor-ingest endpoint, a service worker or manifest, a Capacitor
scaffold, per-device tokens (PR 4 has its own trigger), watch work, any phone `src/` behind the decomposition
fence, a phone flag in any E-arm or on the rig's agent home (M10). Each would freeze a contract before `embody()`,
the Listener and GL3.B1 settle.

## Phase 1 — push-to-talk speech (after 1.4.0)

- **pymaxim:** the phone device adapter under `embody()`; PR 4 (per-device scoped tokens, QR pairing); speech
  through embodied_device step 5's route into `talk(source="voice", device=...)`.
- **pulse:** a phone shell as a PWA, from `apps/reachy/ui/src/profile.tsx`. Foreground only. HTTPS through the
  tunnel only (browsers allow mic and camera only on a secure origin).
- **Pairing:** a QR code shown in the localhost console carries a single-use code of at least 128 bits with a short
  TTL; the claim mints a per-device token with the `talk` scope. The phone stores it; the operator can revoke one
  device without logging out the rest.
- **Tests:**
  - CI, in pulse's harness: a Chromium mobile context, `--use-fake-device-for-media-stream`, scripted
    `DeviceMotionEvent` and geolocation, CDP `Page.setWebLifecycleState frozen` for backgrounding, behind a
    self-signed TLS proxy. Assertions read what pymaxim received, through its own writers.
  - A manual real-device checklist (pulse's `docs/release-checklist.md`): iOS Safari's motion-permission gesture,
    mic loss on lock, background socket kill, Wake Lock (16.4+), the Add-to-Home-Screen storage split; Android's
    install prompt; token rotation mid-session; a cell network. Playwright's WebKit is not iOS Safari, so CI
    claims no iOS coverage.
  - Release: a `phone` step in pulse's `release-acceptance.sh` and a `--phone` option in `live.sh`.

## Phase 2 — location and activity, as an experiment

A new claim, so it takes the four-lens design review ([DESIGN_REVIEW.md](../../experiments/DESIGN_REVIEW.md))
before any harness is built.

- **Sketch:** place and activity transitions, fed to the SCN as one new event category, let the oscillator and
  `anticipatory_pre_activate` predict the next transition (type and time) on held-out days better than three
  floors: an hour-of-week frequency table, a first-order Markov model over places, and `random`. Scored offline on
  a recorded trace.
- **Likely outcome:** a frequency table is strong on routines, so `PARTIAL (floor matched)` under
  [baseline_floors.md](../baseline_floors.md) is a real possibility.
- **Prerequisite:** the GL4 S0a verdict says whether the timed predictor can work at day scale. If it cannot, this
  is not worth a prereg.
- **Confounds:** the human carries the phone, so every move is the human's act and no claim may say "the agent
  acts"; the user adapts to the agent's outputs, so scored data must precede any output reaching the user; N=1
  with the owner as subject and experimenter; OS throttling makes missing data correlate with activity.
- **To [behavioral]:** a second prereg in which the prediction changes an agent act (a proactive utterance or
  haptic before the transition), scored against a clock-schedule floor.

## Phase 3 — only if earned

- **Capacitor wrapper** when background or locked-screen sensing, iOS push or keychain token storage is required;
  after PR 4.
- **Camera snapshots** to the leader: perception_pipeline_placement's "cut after capture" point, under a size cap;
  segmented on the leader.
- **The watch**: native only (watchOS and Wear OS have no web runtime), in its own repo, manual tests until it has
  its own harness. Heart rate only as an observation of another agent.

## Atonia with a device attached (PD3)

[embodied_device.md](../embodied_device.md) §3 names "atonia within embodiment" as a later extension that reverses
advisor_mode's Actuation row and needs its own design passes. This section is the approach note for that pass. It
is a GATE, so it gets an adversarial design pass before any code (CLAUDE.md, "Faster without lowering the bar"),
on the atonia gate's own clock ([advisor_mode.md](../advisor_mode.md)).

**The rule, by channel:**

| Channel | Under atonia | Why |
|---|---|---|
| Phone receptors (camera, mic, location) | **Open**, through item 4's single framed entry, tagged untrusted and `reported` | REM atonia blocks motor output, not perception |
| Speaker, haptics, notifications | **Inhibited**, through the `embody()` census | Actuation |
| The screen showing the report and proposals | **Open**: the report channel | The gate is "act inhibited, report open" |
| Reading the report aloud | **Inhibited** by default (proposed; owner to confirm) | Sound is actuation under the census |
| Frame upload anywhere but the local leader; hivemind contribution | **Refused** (item 10, offline) | Egress |

**What changes in the gate.** The closed builder's refusal of `percept_source` and embodiment narrows rather than
reverses: it accepts a device whose census shows every output inhibited, and still refuses a robot.

**Acceptance.** An SDK-faithful fake phone device that records every output call; CI asserts it records zero under
atonia. One operator-attested real-phone drill is supporting evidence, never the only evidence (embodied_device §3).

**The inputs the gate consults, and how each could lie** (the pass extends this list):

| Input | How it could lie |
|---|---|
| The census itself | an output path added after the census was written (a new notification API, an SDK side effect) |
| The device's declared capabilities | a device that under-declares its outputs |
| The atonia flag on the handle | set after a device was attached; a second handle on the same home |
| The relay's provenance on inbound percepts | a transcript or OCR text carrying instructions; a body-supplied device id |
| The pulse UI | the screen as a side channel: a proposal rendered as a one-tap action |
| Speech output | TTS reached through `Voice` outside `embody()` |
| Egress | a frame upload that reaches a cloud lane through the vision stage |

## Security shape

- Loopback bind stays an invariant; all remote reach is a TLS proxy to loopback (PD2).
- Phones never hold the operator token. Scopes: `sensor` (percept ingest only), `talk` (talk and the clean `/ws`
  tier), `admin` (setup, probe, diagnose; host only). An unknown route defaults to `admin`.
- Phone-supplied fields are ignored for identity and trust: the device id, scope and trust come from the token; the
  server's receive time is authoritative, and a phone's `ts` never feeds the experience clock or memory strength.
  Replay inside a session is refused by a per-device monotonic sequence.
- Every remote percept carries a server-stamped provenance/trust field and enters the loop as data, never as an
  instruction.
- Consent: per-sensor opt-in at pairing, a server-side per-sensor enable per device, an app kill switch, and a
  bystander and retention statement in `PRIVACY.md`, as gate P3 does for Reachy.
- On the tunnel host, `/api/pair/*` is refused unless the pairing code is high-entropy.

## Open questions

1. Is reading the atonia report aloud an output (proposed: yes, inhibited)?
2. Does a phone percept ever trigger the perception fabric's revive (re-keyed to GL3), or only through a camera
   placement decision?
3. Phase 2's record schema: how coarse must location be before it is committable evidence?
