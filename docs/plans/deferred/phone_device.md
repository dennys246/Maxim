# Phone device — a phone (later a watch) as a sensing device and remote UI

**Status:** PROPOSED 2026-10-29, plan only. **Phase 0 is homed in 1.3.5** (owner decision 2026-10-29, §Phase 0).
Phases 1–3 start **after the 1.4.0 cut**; nothing here holds 1.4.0. **Revive trigger for Phase 1:** 1.4.0 is
published, Phase 0 has landed, and [embodied_device.md](../embodied_device.md) has built `embody()` (its step 4)
and the leader STT route (its step 5).

**Input:** a four-lens read-only review on 2026-10-29 (security, architecture and wire integrity, pulse
feasibility and testing, bio-faithful value and sequencing). Its reports live only in that session's transcript,
the same disclosed gap as O6. Every load-bearing code claim was re-checked before this file was written: the
loopback bind (`console/server.py::run_serve`), the control fields in the wire whitelist
(`agents/bus.py::_PERCEPT_WIRE_FIELDS`), and the phone app as the motivating goal of
[console_tunnel_hardening.md](console_tunnel_hardening.md). **Corrected 2026-10-29 (pulse):** the first draft called a reviewer's "LAN bind (A9)" wrong. It was right:
`maxim serve` binds loopback only, and a device EMBEDDER (the Reachy app) binds its own LAN interface under A9
(console_tunnel_hardening.md §Non-goals and A9). PD2 now says so. **Pre-merge round (Executor,
Architecture, Wire integrity) folded 2026-10-29;** the owner decisions the folds took are listed in §Fold decisions.

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
| PD2 | **Remote reach is cloudflared only** for v1: one path to harden, and the one the tunnel plan was written for. `tailscale serve` is recorded as "not now, additive later", not as rejected. The bind rule is console_tunnel_hardening's, unchanged: **desktop `maxim serve` binds loopback only; a device embedder (the Reachy app) binds its own LAN interface under A9**, behind the Host/Origin allowlists (`build_app(extra_trusted_origins=…)`) and bearer auth, with Pollen's dashboard as its trust boundary. **A phone is always a client, never an embedder:** it reaches a desktop leader only through the tunnel. It may reach a robot's LAN console as a plain-HTTP client for the UI, but its sensors need a secure origin, so sensing waits for a TLS path on that console. Moving the robot behind the tunnel is not intended. |
| PD3 | **Atonia with a device attached is planned now** (§Atonia with a device attached), so `embody()` and the gate are not redesigned later. It is designed now and built after the atonia gate ships and `embody()` exists. The pre-merge round found it is a larger reversal than first drafted (the open problems are recorded there); nothing here changes the gate as shipped. |
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
  This reshapes embodied_device.md's step 4 (today `embody(sdk)` adopts a Reachy SDK, and its census covers motors
  and sound): `embody()` must take a device adapter, and the census must cover every effector (sound, haptics,
  notifications) across all attached devices. Whether `embody()` holds a Reachy and a phone at once is open (P0-7).
- **Its percepts enter through the relay** as registered receptors with a non-`experienced` provenance kind (`reported`,
  or `narrated` + `caused_by`, whichever thalamic_relay's open TR11 decides; P0-5). M39: every percept reaches the EC
  through the relay's one registration point. No `PerceptSource` side door.
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
| Sensor ingress | the relay's receptor registry (GL3.B1) | **rides**: a remote-device receptor, a non-`experienced` kind (TR11) |
| Outputs | `embody()` and the embodied census | **rides, reshaped**: `embody()` takes a device adapter; the census widens to every effector (P0-7) |
| Remote transport | cloudflared, console_tunnel_hardening | **rides** |
| A phone-specific transport or package | none | **not built**: embodied_device §2 forbids a new top-level package |

## What each sensor is worth

| Modality | Consumer | Verdict |
|---|---|---|
| Location + activity (GPS, IMU collapsed to an activity label) | SCN (`time/oscillator.py` runs circadian to annual scales, and every run so far is minutes long); Hippocampus `situation=` context | **The strongest case.** Weeks of input at the scale the SCN was built for. Phase 2's experiment. |
| Mic → transcript | the talk path; the word world of [grounded_language_acquisition.md](../grounded_language_acquisition.md) | **Real.** Phase 1's slice. Enters with a non-`experienced` kind (TR11); language stays off the E-ladder (G5). |
| Camera | perception, after placement and the fabric revive | **User-directed perception only** (PD5). Earns no orienting claim. Phase 3. |
| IMU as proprioception | none | **Not built.** It measures the human, not the agent; the Cerebellum needs the agent's own motor command. |
| Watch heart rate | none | **Never interoception.** It is the user's physiology, an observation of another agent. Never written into drives, `vital_metrics` or `InteroceptiveOutcome` (G16, M78). |
| Speaker, haptics, notifications | the social loop | Outputs under `embody()`'s census (PD1); inhibited under atonia. |

## Phase 0 — 1.3.5 (PD4)

Ranked. "Lenses" names the reviewers that raised it independently. "Owner" is **proposed**: 1.3.5 is Session A's
orchestrator release, and these items sit in other sessions' files, so each session confirms when it picks one up.
Nothing here edits `orchestrator.py`, `agent_loop.py`, `cli.py` or NAc's selection path.

| # | Item and done state | Why now | Severity | Lenses | Owner (proposed) |
|---|---|---|---|---|---|
| P0-1 | **The percept wire is a trust boundary inbound.** Split `agents/bus.py::_PERCEPT_WIRE_FIELDS` into an outbound set (what `to_wire_dict` ships) and an inbound set. `from_wire_dict` forces `cli_input`, `has_voice_command`, `has_maxim_keyword` and `explore_command` to None and re-derives `hard_override` as `agents/autonomy.py::check_hard_stop(text, None)` over the transcript or content (the inbound `hard_override` is never passed as `percept_override`), the way `comms/gateway.py::CommunicationGateway.receive_inbound` treats `hard_override`. `context` is no longer taken whole: `agent_id`, `sender` and `channel` are set by the receiver, `channel` is validated at runtime, and the top-level `timestamp` (which `Percept.memory_id` hashes), `context.timestamp` and `context.scn_tag` are receiver-stamped (a sender's clock never feeds the experience clock); `source` is set by the receiver from the authenticated principal, never taken from the body. `sensory` is rehydrated through `SensoryTag.from_dict`, in `from_wire_dict` and in the session path `Percept.from_dict`. **No consumer changes**: `agent_loop.py`'s `check_hard_stop` call and `exec_agent.py`'s `hard_override` branch stay as they are. `tests/unit/test_percept_wire_format.py::test_wire_dict_round_trips_without_loss` is inverted for the control fields, and `test_wire_fields_constant_matches_to_wire_dict_output` follows the split. This changes mesh-peer ingress too (mesh_perception_transport.md), which has no producer yet. | Neither path has a `src/` caller today, so it is cheap; a latent defect with or without a phone | SHOULD-FIX | architecture, wire | V (proposed; `agents/bus.py` is unfenced; it calls V's `autonomy.py`) |
| P0-2 | **console_tunnel_hardening PR 3, admission control**: `limit_concurrency`, `ws_max_size`, an input cap outside the sandbox (today `max_input_chars` is sandbox-only), a rate limit on `/api/pair/claim`. Generalize `runtime/leader_proxy.py::_check_admission` per PR 3's front-gate note rather than building a second limiter. | Needed for any remote reach; it did not ship in 1.3.3 | SHOULD-FIX | security, pulse | S (`console/`) |
| P0-3 | **A principal seam in `console/server.py::_GuardMiddleware`** (pulled forward from tunnel PR 4). `_bearer_authorized` (one caller) returns a principal `{device_id, scopes}` in `scope["state"]`, not a bool; a path→scope table defaults unknown routes to `admin`. Today's token is the operator principal. Defined explicitly: the sandbox principal (bearer auth is off under sandbox by design) and the anonymous principal of `_AUTH_EXEMPT_PATHS` and the static bundle. Guards: `tests/unit/test_console_auth.py`, `tests/unit/test_console_device_handoff.py`. | PR 4's per-device tokens become additive, not a refactor | SHOULD-FIX | security, wire | S |
| P0-4 | **The tunnel plan's M3 (talk's tool gate; not outstanding.md's M3), pulled forward from PR 4:** Talk's registry holds no `bash`, write, edit or execute tool by construction, not by `console/handle.py::talk_autonomy`'s deny derivation. Today Talk shares the persistent instance's `tool_registry`, built `operational_mode="active"`. Open design choice, decided by the picking session with the owner: a separate talk registry in `handle.py`, or a change in `runtime/agent_factory.py`. Guard: a test that fails when one of those tools is added to Talk's registry. | Before any remote turn reaches the loop | SHOULD-FIX | security, executor | S with V (either option) |
| P0-5 | **A plan amendment only in 1.3.5:** thalamic_relay.md §11 records the phone as a named case for the open TR11 (a remote device's percepts carry a non-`experienced` kind: `reported`, or `narrated` + `caused_by`) and a remote-device / observed-other-agent `ReceptorClass`. The `test_receptor_census.py::CENSUS` row and the code land with GL3.B1 in 1.4, with the first producer (a row without a src site fails that test as STALE). | Keeps a user's speech and physiology out of interoception before GL3.B1 freezes its kinds | SHOULD-FIX | research, architecture, executor, wire | doc |
| P0-6 | **Device identity on the console contract**: a device/origin id beside embodied_device step 1's `source=` (#1439), after that step lands. It rides R3's 0.7.0 ([#1336](https://github.com/dennys246/Maxim/issues/1336)) if that bump has not shipped, else its own minor bump (embodied_device.md's own rule). Contract today: `console/ui_bundle.py::CONSOLE_CONTRACT_VERSION` 0.6.0. M70's note to pulse covers the interim. | One contract bump instead of two where possible | SHOULD-FIX | architecture, executor, wire | S |
| P0-7 | **Doc-only: embodied_device.md steps 1, 4 and 5 become device-generic.** The framed talk ingestion and the leader STT route take a device id; `embody()` takes a device adapter, not only a Reachy SDK; the census covers every effector across all attached devices; one owner per device. Decide whether `embody()` holds several devices at once. | The phone becomes an adapter, not a fork | SHOULD-FIX | architecture, pulse, research | doc (embodied_device's session) |
| P0-8 | **Owner decision on [#784](https://github.com/dennys246/Maxim/issues/784)'s option** (the A4 gain inverts a place code), recorded on the issue before 1.3.5 closes. The fix still rides the grounding line, where roadmap_1_3_x.md parks it. | Location is exactly a place-coded sensor | SHOULD-FIX | research, executor | owner |
| P0-9 | **A privacy design for evidence**: a derived, quantized record schema for location and activity, and `scripts/lint_secrets.py` (M45) extended to coordinates. | Must precede the first capture, or M1's committed-evidence rule and privacy deadlock | SHOULD-FIX | research | H |
| P0-10 | **Every declared SCN `event_type` has a producer or is marked reserved**: item 3 of [scn_event_producer_gap.md](scn_event_producer_gap.md). | Tests only; a tripwire before a phone producer arrives | SHOULD-FIX | research | R |
| P0-11 | **Pulse kit groundwork** (a note to pulse; pulse owns it): reconnect on `visibilitychange` and `online` with catch-up from `/api/runs` in pulse: `facade/events.ts`; a size budget per profile (generalize pulse: `scripts/size-reachy.mjs`); an e2e device server that takes an origin and a TLS argument. | A hidden laptop tab has the reconnect problem today | SHOULD-FIX | pulse | pulse |
| P0-12 | **DONE in this PR — doc reconciliation**: `maxim_ui.md`'s localhost line; mesh_perception_transport's ingress for phones; M76's wording (PD1); the PR 3 homing in console_tunnel_hardening.md, roadmap_1_3_x.md and README.md; pointers on embodied_device.md and maxim_console.md. **Owed to pulse:** its `docs/plans/maxim_ui.md`, `CLAUDE.md` and `AGENTS.md` carry the same localhost line. | Otherwise three revive triggers misfire | SHOULD-FIX | all four | doc / pulse |
| P0-13 | **NITs**: loopback-origin trust in `_origin_allowed` becomes scheme-aware (http/https only; `capacitor://localhost` named explicitly), pinned in `tests/unit/test_console_trust_guard.py` beside `test_non_http_scheme_origin_can_be_listed`, with `server.py`'s docstring updated; an additive-only check in pulse's `gen:facade:check`, warn-only until a store binary exists; responsive kit layout. | Cheap | NIT | security, pulse, wire | S / pulse |

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
on the atonia gate's own clock ([advisor_mode.md](../advisor_mode.md)). **Nothing here changes the gate as shipped**
(embodied_device §3: the gate ships as designed, and the device plan does not gate it).

**The intended rule, by channel:**

| Channel | Under atonia | Why |
|---|---|---|
| Phone receptors (camera, mic, location) | **Open**, through advisor item 4's single framed entry, tagged untrusted with a non-`experienced` kind | REM atonia blocks motor output, not perception |
| Speaker, haptics, notifications | **Inhibited**, through the `embody()` census | Actuation |
| The screen showing the report and proposals | **Open**: the report channel | The gate is "act inhibited, report open" |
| Reading the report aloud | **Inhibited** by default (proposed; owner to confirm) | Sound is actuation under the census |
| Frame upload anywhere but the local leader; hivemind contribution | **Refused** (advisor item 10, offline) | Egress |

**What changes in the gate: open, for the extension's own design pass.** The pre-merge round found this is a larger
reversal than "narrowing" one refusal:
- It reverses the **Actuation** row, and with a remote device also the **Environment** and **Egress** rows. A phone
  reaches Maxim only through `maxim serve` and the tunnel, and its speech through the leader STT route. The closed
  builder ignores `MAXIM_ROLE=leader` (proxy, mesh, tunnel), and advisor item 10 refuses the leader proxy and tunnel.
  So the two open rows above need channels the gate closes today.
- **Under V1 the observer runs one bounded turn per `ingest()`, with no loop.** A receptor stream needs a loop or an
  ingest path that the gate does not have.
- The closed builder refuses "any embodiment or robot". (`percept_source=None` is Profile A's recipe, not a builder
  refusal.)
- **The census is a CI acceptance test, not a runtime object.** Which runtime object the builder would consult in
  its place is undecided.
- A local-only variant (a phone on the same host, no tunnel) is not a real deployment, so it does not dodge these.

**Acceptance, if the extension is taken.** An SDK-faithful fake phone device that records every output call; CI
asserts it records zero under atonia. One operator-attested real-phone drill is supporting evidence, never the only
evidence (embodied_device §3).

**The inputs the gate would consult, and how each could lie** (the pass extends this list):

| Input | How it could lie |
|---|---|
| The census itself | an output path added after the census was written (a new notification API, an SDK side effect) |
| The device's declared capabilities | a device that under-declares its outputs |
| The atonia flag on the handle | set after a device was attached; a second handle on the same home |
| The relay's provenance on inbound percepts | a transcript or OCR text carrying instructions; a body-supplied device id |
| The pulse UI | the screen as a side channel: a proposal rendered as a one-tap action |
| Speech output | TTS reached through `Voice` outside `embody()` |
| Egress | a frame upload that reaches a cloud lane through the vision stage; the tunnel itself, which the gate closes |

## Security shape

- Desktop `maxim serve` stays loopback-bound, and remote reach to it is a TLS proxy to loopback; a device embedder's LAN bind follows A9 (PD2).
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

## Fold decisions (pre-merge round, 2026-10-29; owner to confirm)

The folds took these choices; each is the conservative reading and is reversible:
1. **The tunnel plan's M3 (talk's tool gate) and the principal seam are pulled forward from tunnel PR 4 into 1.3.5** (P0-3, P0-4), with Phase 0's homing.
   PR 4's per-device tokens still fire with Phase 1.
2. **P0-5 is a plan amendment in 1.3.5**; its census row and code wait for GL3.B1 (1.4).
3. **P0-8 is an owner decision, not a fix,** in 1.3.5; the fix stays parked with the grounding line.
4. **P0-1 also fixes the session path `Percept.from_dict`** (`sensory` rehydration), and changes the mesh-peer ingress.
5. **PD3's atonia extension is recorded with its open problems**, not as a narrowing of the builder.
