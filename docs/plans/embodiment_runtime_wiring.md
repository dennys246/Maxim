# Wiring the SEM body into the live runtime — so the substrate has a body again

**Status:** DRAFT plan (2026-07-17). Prerequisite for
[orient_runtime_integration.md](orient_runtime_integration.md) Landing 1 (audio as a
runtime percept) and the eventual reflex — but broader than orienting: it closes the
documented gap where **`executor.embodiment` is never populated on any production
agent path**, so the drive/pain/substrate cascade the loop already runs has been dark.
Grounded in a five-track repo audit (2026-07-17); findings folded per track below.

**One-line thesis:** the machinery to give the substrate a live body already exists and
is already ticked by the agent loop — the only missing thing is the *construction*:
instantiate the `reachy_mini` SEM body and hand it to `build_executor`, exactly as the
sim-AUT path already does. The work is making that safe, gated, and honest about what
turns on.

---

## Front-gate: this rides existing infrastructure (audit-confirmed)

Per CLAUDE.md Principle 3. The plumbing is entirely present; nothing new is invented:

| Need | Already in `src/maxim`? |
|---|---|
| A runtime slot for the body | ✅ `Executor.embodiment` (executor.py:93); the loop reads it |
| The loop ticks the body | ✅ `agent_loop.py:851` — `getattr(executor, "embodiment", None)` → `evaluate_failures()` |
| A constructor that builds the Embodiment from a body | ✅ `build_executor(entity_ref=, component_registry=, pain_bus=)` (bootstrap.py) |
| A precedent that already wires it | ✅ the sim-AUT path in `simulation/orchestrator.py` passes `entity_ref` |
| The prompt-facing seam | ✅ `_maybe_wire_body_state` (agent_factory.py:76) routes `executor.embodiment → memory_hub.embodiment` |
| The body declaration | ✅ `bodies/reachy_mini.yaml` — azimuth sensor + centeredness drive + failure modes |

**What's missing is one line of construction:** `agentic_runtime.py` (~line 317)
deliberately passes NO `entity_ref` to `build_executor` — its own comment says *"No
entity_ref because the Reachy runtime already wired..."* — so `executor.embodiment` is
`None` and the whole cascade never fires. This plan changes that construction, behind a
default-off flag, and makes the consequences safe.

**Verdict: no new mechanism.** The risk is not "build something new" — it's "turn on
something that's been dark long enough that its interactions need re-verifying." That is
exactly what the audit tracks below cover.

---

## Design goal: fix in the GENERIC layer, so no future robot redoes this (2026-07-17)

**The operator's requirement: abstract how substrate-primary wires into embodiment so a
future robot (Atlas, Spot, …) inherits it instead of re-patching the runtime.** This is
right, and it reconciles cleanly with the project's "don't abstract from N=1" rule
(porting doc: abstracting from one example bakes its assumptions in) — because the two are
about *different things*:

**Most of the audit's fixes are not abstractions — they are fixes placed in the WRONG
layer today, and moving them to the generic layer IS the generalization.** These are
robot-agnostic by nature and must NOT live in Reachy code:

- **The per-iteration `evaluate_failures()` tick (Track A)** — a *loop* concern. It belongs
  in `agent_loop.py`, gated `embodiment is not None`, so **every** embodied robot's loop
  ticks its body. Putting it in the Reachy runtime would force robot #2 to re-add it. This
  is not speculative — any live SEM body needs it.
- **The `build_executor(entity_ref=…)` wiring (Track B)** — already generic; `build_executor`
  takes `entity_ref`. What's Reachy-specific today is the *hardcoded string* `"bodies/
  reachy_mini"`. The fix: wire from the robot's **declared** body, not a literal.
- **The hybrid substrate-primary-under-LLM loop mode (P1)** — a *runtime mode*, not a robot.
  A robot doesn't implement hybrid; it *declares it wants* a substrate-primary reflex, and
  the shared loop provides it.
- **The exteroceptive-vs-interoceptive encoding split (P2/Track D)** — the general lesson
  ("a signed bearing is exteroceptive; it must not route through the interoception drive
  encoder that folds its sign") is robot-agnostic. The `"audio"` EC modality already exists
  reserved for exactly this; wiring it is generic infra, not Reachy code.

**Only ONE thing is a genuine new interface, and here the N=1 caution DOES apply — so keep
it minimal, declarative, and validated by the one real robot:** *how a robot declares its
body + its substrate-primary reflexes.* Do NOT build a speculative reflex-registry
framework. Instead, extend the EXISTING capability-driven seam:
- Add a `body:` field to `RobotConfig`/`robots.yaml` (the body YAML ref), so the runtime
  reads `entity_ref = robot.body` instead of hardcoding. One field, reuses the pattern
  vision already uses (`has_vision`).
- Reuse `has_audio` (already exists) to gate the audio-orient reflex.
- Declare the reflex as data (sensor → policy → affordance), the minimum Reachy needs — not
  a plugin system. Robot #2 fills the same declaration; the *wiring* is shared.

**The rule for this plan: every fix lands in the most generic layer that's correct, and
Reachy becomes the first CONSUMER of robot-agnostic seams, never the special case.** Each
track below is tagged `[generic]` (goes in the shared runtime/loop) or `[declaration]`
(the minimal per-robot seam) so the placement is explicit and the next robot's path is
"fill the declaration," not "re-patch the runtime."

## The three layers

Landing "sound as a runtime percept" splits into three, and only the first is the real
architectural commitment:

### Layer 1 — wire the SEM body into the live runtime (the unlock)  ✅ LANDED (2026-07-17)

Build the `reachy_mini` Embodiment and pass it to `build_executor` on the live path.
This lights up the tick + drive + pain cascade the loop already runs — for ALL of the
body's sensors, not just azimuth.

**This is broader than audio** — it's the general "the substrate has a body" wiring, and
`body_state`, the Acting Coach layers, and the future reflex all depend on it.

> **SHIPPED (the `[declaration]` seam, opt-in):** the body is declared in `robots.yaml`
> via the free-form `config.body` (e.g. `body: bodies/reachy_mini` — same ref form as
> `--embodiment`), read by `hardware.config.resolve_body_ref`. `agentic_runtime.
> _resolve_body_wiring` resolves it and passes `entity_ref` + `component_registry` into
> `build_executor`, so `executor.embodiment` is live and the Track-A per-iteration tick
> now has a body to advance. **Opt-in by design** — absent a declaration the runtime is
> byte-identical to before (bodiless, tick is a no-op); a declared-but-unresolvable body
> logs loudly and falls back to bodiless (a typo never crashes the live robot). Defaulting
> it ON per robot_type is deferred as a *deliberate* decision, bundled with Layer 3a
> (`body_state`) + Layer 3b (the Acting Coach drive-modulation fix) so drive-pain and the
> coach's exteroceptive-vs-homeostatic rendering land together, not silently. Guard:
> `tests/unit/test_multi_robot.py::TestBodyDeclarationSeam`. (Deferred from the original
> scope: the `has_robot`/`has_audio` + `config.json` flag gating — the declaration seam
> supersedes it as the on/off control; revisit if a non-declaration gate is wanted.)

### Layer 2 — feed live DoA into the body's `azimuth` sensor (the audio-specific bit)

Once the body exists, "azimuth in the runtime" is what the orient scripts already do:
write the built `AzimuthDoASource`'s reading into `entity.vital_metrics["azimuth"]`
before the tick. The centeredness drive fires; the substrate encodes it. Steps 1+2 of
the orient-runtime plan (the reader + builder, merged in #399) plug in here.

### Layer 3 — the prompt half  ✅ LANDED (2026-07-17)

With `executor.embodiment` non-None, `body_state` surfaces through the existing prompt
section. The Reachy runtime bypasses `create_full_agent`'s `_maybe_wire_body_state` seam
(Track E Gap 3), so it needed its own wiring.

> **SHIPPED — Layer 3a (`memory_hub.embodiment`):** `agentic_runtime._start_agentic_runtime`
> now routes `executor.embodiment` into `memory_hub.embodiment` after executor construction,
> gated by the **same** `body_state_prompt_enabled()` flag (`MAXIM_ENABLE_BODY_STATE_PROMPT`)
> as the AgentFactory seam — so the live-robot prompt is unchanged by default (opt-in,
> consistent with the Exp 44 ablation). Only fires when a body was actually wired.
>
> **SHIPPED — Layer 3b (the Acting Coach category-error fix):** `_compose_drive_modulation`
> was rewritten from a hardcoded thermal branch (any homeostatic breach → "seek warmth/
> shelter") to **per-sensor, action-neutral** reporting — it names the specific signal(s)
> outside their comfortable range / deprived / rising and leaves the action to the LLM. Per
> review SF-4 this is the data-driven fix (reflects whatever sensors the body declares), NOT
> a second `if drive == "centeredness"` branch, and it touches **no** shape-frozen spec. So
> when the flag is flipped on, an off-center sound no longer tells the robot to "seek warmth."
> Guard: `tests/unit/test_acting_coach.py::TestComposeDriveModulation` (incl. the azimuth-breach
> anti-regression + a behavioral no-modality-prescription sweep).
>
> Layers 3a+3b shipped **together** (the review was firm: turning on 3a without 3b ships the
> misfire). Pain-anticipation (Layer 2) + drive-modulation (Layer 4) now activate coherently
> when the flag is set. A richer *per-drive declared* guidance string (a `guidance` field on
> the drive specs) is a deliberate follow-up — it touches the shape-frozen `HomeostaticDriveSpec`
> / `EntropicDriveSpec` (CC3 review gate) and is worth its own change; the shipped fix already
> removes the category error without it.

---

## Audit findings (five tracks, 2026-07-17)

### Track A — the tick / wall-clock drift invariant  ✅ LANDED (2026-07-17)
**VERDICT: double-drift is impossible; the real hazard is NO-drift on LLM-primary — fixable with one call.** (audit-confirmed against body.py, agent_loop.py, the CI grep)

> **SHIPPED:** `run_agentic_loop` now calls `tick_embodiment_drift(executor, aut_mode)` once per
> live iteration (after the pause check, before the 0.6 idle gate so a *sitting* body still drifts).
> Extracted as a testable module-level helper next to `propose_via_substrate`; no-op on
> substrate-primary + unembodied; calls the public `evaluate_failures()` (no CI-grep trip). Guard:
> `tests/unit/test_substrate_primary_scene_harm.py::TestTickEmbodimentDriftLLMPrimary`. CLAUDE.md
> embodiment-tick invariant updated.

- **The invariant:** `Body.evaluate_failures()` self-applies wall-clock drift at the top
  (`body.py:149-154`) using elapsed `now - _last_poll`, then resets `_last_poll`. The raw
  `tick_vital_drift(` primitive is CI-locked to `body.py` (grep at
  `.github/workflows/test.yml:551-563` fails the build on any other call site — this is
  what commit `ed8b187f` earned). `evaluate_failures()` is the only sanctioned public tick
  and is deliberately callable from many sites.
- **Double-drift: NO, by construction.** Elapsed-dt + `_last_poll` reset means a second
  `evaluate_failures()` in the same tick sees `dt≈0` → negligible. Safe to call from
  multiple sites.
- **THE REAL HAZARD — no-drift on LLM-primary.** The Reachy runtime is **LLM-primary**
  (`run_agentic_loop` with no `aut_mode` → default; `agentic_runtime.py:757`). The loop's
  only `evaluate_failures()` tick site (`agent_loop.py:854`) is **gated behind
  `aut_mode == "substrate-primary"`** and never runs. So a live body would drift ONLY when
  a `ModulatorAffordanceTool` executes (`tool_bridge.py:413`) — a Reachy driving real robot
  tools with few affordance calls leaves drives **effectively frozen**. This is the exact
  "event-driven, not per-turn" gap CLAUDE.md already flags.
- **The fix (Layer 1 must include it):** add **one** `evaluate_failures()` call per loop
  iteration in `run_agentic_loop`'s `for step_num in step_iter:` body (~`agent_loop.py:1159`),
  guarded `embodiment is not None and aut_mode != "substrate-primary"`, co-located with
  auto-sense (~1306) or the NAc per-tick decay (section 8.5). **MUST call
  `evaluate_failures()`, never a raw `tick_vital_drift(`** (CI grep). This makes the
  LLM-primary body tick once per iteration (~10 Hz) independent of tool execution.

**Consequence for the plan:** Layer 1 is not just "wire the body into `build_executor`" —
it is **also** "add the per-iteration tick for LLM-primary." Both, or the body is dark.

### Track B — `build_executor` + PainBus construction contract
**VERDICT: clean addition, no conflict.** (audit-confirmed against bootstrap.py:384-477)

`build_executor`'s fail-fast checks (bootstrap.py:385-414) make the `entity_ref` path
require the triad **`component_registry` + `pain_bus` + `nac`**, all non-None. The Reachy
runtime's existing `if nac is not None:` branch (agentic_runtime.py:320) already has
`nac` and `_pain_bus` (from `build_bio_stack`) — **it only needs a `ComponentRegistry`
and the string ref added.**

Crucially: when `entity_ref` is set, `build_executor` **reuses the passed `pain_bus`** to
build the Embodiment (`bootstrap.py:456` → `Embodiment(entity, pain_bus=pain_bus, ...)`;
`body.py:65` stores it, never builds its own). **No second bus, no conflict** with the
one `build_bio_stack` already made.

- `entity_ref` expects a **STRING** (`"bodies/reachy_mini"`) — `build_executor` calls
  `component_registry.instantiate()` itself (bootstrap.py:455). Do NOT pre-instantiate.
- The sim-AUT precedent routes through `AgentFactory.create_full_agent` →
  `build_executor(entity_ref=config.embodiment_ref, ...)` (agent_factory.py:454-466) —
  that's the template.
- **Defensive gate (load-bearing per invariant #3):** wire `entity_ref` only when
  `_pain_bus is not None`, falling back to the no-body call otherwise — so a `bio=None`
  edge can't trip the "entity_ref requires pain_bus" ValueError.
- `cerebellum`/`distributor` are optional; adding them matches the factory template and
  enables forward-model training for the generated affordance tools.

**The exact corrected `build_executor` call is settled** (in the Layer-1 implementation
notes below). This track carried the biggest "does it even wire cleanly" risk — and the
answer is yes.

### Track C — `reachy_mini` sensor/drive/failure safety when live (highest risk)
**VERDICT: SAFE to wire live as-is — no sensor drifts toward a pain threshold when unfed.** (audit-confirmed against reachy_mini.yaml + body.py drift logic)

The feared failure mode (body hurts itself because battery/temperature/pose are never
fed) **does not occur**, for a concrete reason:
- A sensor drifts **only if it has a `drive:` block**. Of reachy_mini's 12 sensors, only
  `azimuth` has one — and its `drift_rate` is **0.0** (`step = min(x, 0) = 0` → pinned),
  which is load-bearing-by-design (a world-set sensor must not auto-return). The other 11
  have **no drive**, and none match the legacy hardcoded drift names
  (`fatigue/strain/exhaustion`, `durability/sharpness`), so they sit at `initial:` forever.
- **Every failure threshold is on the far side of `initial:`** and nothing moves the
  sensor toward it: battery 1.0 vs `<0.15`, motor_temp 28 vs `>75`, pose 0.95 vs `<0.5`,
  camera/mic 1.0 vs `<0.2`, azimuth pinned at 0.0 inside its own `±0.1` comfort band. **No
  spurious pain, no phantom learning.**
- Nothing in the production runtime currently feeds any of these (grep-confirmed) — so
  live, they all sit at `initial:`.

**The inverse gap (a monitoring gap, NOT a safety one):** unfed health sensors report
*falsely healthy* forever — a genuinely low battery or hot motor will never fire its
failure_mode. The failure_modes are **inert, not dangerous**. Feeding real telemetry
(`battery`, `motor_temperature`, `pose_confidence` have daemon analogues) makes them
*useful*; it is not required for *safety*.

**Implication for the plan:** Layer 1 can wire the body live **without** feeding the
health sensors first — no substrate-poisoning gate. Telemetry feeds become a nice-to-have
follow-up (real proprioceptive pain), not a prerequisite. Azimuth (Layer 2) is the one
sensor we *do* feed, and it only fires pain on genuine off-center — correct signal.

**Track C also flagged a scope boundary for Track D:** it audited drift/pain only, NOT
whether `potential_diff` relief-crediting reaches NAc on the live path — that learning-
signal question is Track D's.

### Track D — substrate-encoding reconciliation (does live-wired match Exp 45?)
**VERDICT: NO — they are two categorically different substrate representations. A live-wired body routing azimuth through `SensorEncoder` CANNOT use the Exp 45 queen-mind policy.** (audit-confirmed against agent_loop.py, encoder.py, ec.py, live_common.py, nac.py)

This is the finding that reshapes the plan. Three specifics:

1. **Different cluster-key namespace (decisive).** Exp 45 learned into
   `_cluster_reward_bias[(agent, az_bin STRING, tool)]` — hand-binned strings
   (`"near_left"`, `"far_right"`) written by calling `update_cluster_reward` **directly**;
   the orient loop never touches EC or SensorEncoder. The runtime substrate path
   (`agent_loop.py:871`) keys the *same dict* on an **EC node UUID** from
   `encode_sensors → pattern_complete_or_separate`. So a runtime passing `(agent, <uuid>,
   tool)` never looks up the `(agent, "near_left", tool)` the policy learned → every entry
   misses → `recommend_action` sees 0.0 bias → **cold-start despite a trained policy.**
2. **The load-bearing sign is degraded.** `SensorEncoder._normalize_value`
   (`encoder.py:405-424`) folds `[-1,0]` and `[0,1]` both onto `[0,1]`, so left and right
   of equal magnitude collapse toward similar embeddings — **the exact left/right sign the
   orient policy depends on is not preserved** in interoception clustering. So even
   *retraining* in the runtime's interoception space would fight a representation that
   discards the signal.
3. **The drive-pain path (PainBus → NAc) doesn't rescue it** — it gives NAc an aversion
   *gradient* (further off-center = more pain) but no azimuth *state* and no *side*.

Also confirmed: the **`"audio"` EC modality** (frozen-centroid, reserved in `ec.py:216`
for exactly this exteroceptive bearing) **has zero producers** — nothing passes
`modality="audio"` to `encode_sensors` anywhere.

**This is a fourth instance of the state-space-must-travel-with-the-policy failure class**
(the demo bug, the learner/metric mismatch, cross-robot bundles) — but worse: the two
state-space *definitions are categorically different* (signed hand-binned strings vs
unsigned EC-UUID interoception clusters), not a drifted boundary a sidecar fixes.

**The crux:** azimuth is **exteroceptive** (a bearing to something in the world), NOT an
interoceptive drive state. The orient loop chose az_bin strings *deliberately* because the
generic drive→interoception encoder is the wrong representation for a signed bearing. So
"make the runtime use the Exp 45 policy" means the runtime's substrate-primary orient path
must key on the **same az_bin string** (+ boundary sidecar) the policy learned on —
**bypassing `SensorEncoder` for azimuth** — not route it through the interoception drive
path.

---

## Reframe (2026-07-17): the Reachy runtime predates substrate-primary

**The operator's framing, and it's the root cause:** the Reachy runtime was built before
substrate-primary existed, so it is **LLM-primary only** (Track A) and its substrate path
(`propose_via_substrate` + `SensorEncoder`, gated on `aut_mode == "substrate-primary"`) is
never reached. The Exp 45 orient policy is *fundamentally* substrate-primary (no LLM in the
action path). So "run the learned orient policy on a live Reachy" is not a percept-wiring
task — it is **reconciling the Reachy runtime with substrate-primary**, which is a deeper
arc than Landing 1 assumed. Two distinct problems fall out:

- **P1 — the Reachy runtime can't run substrate-primary at all.** `run_agentic_loop` is
  called with no `aut_mode` (defaults LLM-primary; `agentic_runtime.py:757`). Enabling a
  substrate-primary *mode* (or a hybrid where orienting is substrate-primary while higher
  cognition stays LLM) on the robot is the real unlock for using the policy.
- **P2 — even in substrate-primary, azimuth must key on az_bin strings, not
  SensorEncoder UUIDs**, or the learned policy is unreadable (Track D). This is an
  orient-specific representation decision, separate from the generic drive path.

**This makes the plan bigger than "wire a body," and that's the honest finding.** The body
wiring (Layers 1–3) is still correct and safe (Tracks A/B/C), but it only lights up the
*interoceptive* drive/pain/prompt cascade — it does **not** by itself make the Exp 45
orient policy runnable. That needs P1 + P2.

### Track E — prompt / body_state activation + auto-sense coexistence
**VERDICT: three real gaps — Layer 3 is NOT free, and two are honest bugs to fix or scope out.** (audit-confirmed against body.py, acting_coach.py, agent_loop.py, agent_factory.py)

**Gap 1 — azimuth does NOT become "a voice, to your left".** `format_body_state_for_prompt`
(`body.py:559-584`) emits raw numerics: `reachy_mini.azimuth: 0.3normalized (DRIVE:
outside comfort band, discomfort 0.20)`. The `-1=left/+1=right` convention is never
translated to English. **The Landing-1 pitch overstated this** — wiring the body gives
the LLM a number + a drive tag, not readable direction. If we want "to your left" text,
that is a small explicit renderer, not free.

**Gap 2 (a bug) — the Acting Coach misfires the centeredness drive as thermal.** Layer 4
`_compose_drive_modulation` (`acting_coach.py:288-318`) DOES fire on the azimuth breach —
but its hardcoded `"outside comfort band"` branch emits *"Your body temperature or
pressure is outside the comfortable range… seek shelter/warmth."* It has no concept of
"off-center → turn toward the source." **So a live-wired Reachy would tell the LLM to seek
warmth when a sound is off to the side.** Layer 2 (pain anticipation) stays inert (looks
for "anticipated"/"anxiety" substrings azimuth never emits). **Fix required before Layer 3
ships with the coach on.**

> **Review correction (SF-4):** the fix is NOT "teach `_compose_drive_modulation` the
> centeredness case" — that bakes a *second* drive-name-specific hardcoded branch into the
> generic coach (the thermal branch is the first), so robot #3's new drive re-triggers the
> identical bug in the `[generic]` layer. The robot-agnostic fix is a **data-driven
> drive→guidance mapping**: each drive *declares* its own modulation text (or a small template)
> in its spec, and `_compose_drive_modulation` renders whatever the drive declares — no
> `if drive == "..."` chain. This also fixes the deeper category error (rendering an
> *exteroceptive* bearing through a *homeostatic/interoceptive* template). Interim, if the
> data-driven mapping is more than this plan wants to carry, **gate the coach's drive layer off
> for the azimuth drive** rather than add the second hardcoded branch.

**Gap 3 — the live Reachy runtime BYPASSES `_maybe_wire_body_state`.** `agentic_runtime.py`
builds its agent **directly**, not via `create_full_agent` (where the seam lives,
agent_factory.py:471). So `MAXIM_ENABLE_BODY_STATE_PROMPT` will **not** auto-enable the
prompt path here. Layer 3 must **explicitly** set `memory_hub.embodiment =
executor.embodiment` in `agentic_runtime.py` after the (Layer-1) build_executor change.

**Gap 4 — auto-sense + body_state DOUBLE-UP.** Both fire per percept tick; the prompt
carries azimuth twice — once as a bare `{value, unit}` dict (`auto-sense`, agent_loop.py
section 1.15, `=== What you perceive right now ===`) and once as the richer
`(DRIVE: …)` line (`body_state`, `=== Body State ===`). Not a crash (different fields,
different sections), and **deliberate per the Exp 44 ablation** (auto-sense stays ON in all
arms so body_state is the only *added* variable). But for a live Reachy it's redundant, and
the ablation doc explicitly warns **NOT** to "fix" the shared no-tick lag by adding
`evaluate_failures()` in the enrich path — which interacts with Track A's required tick
(the tick belongs in the loop body, NOT the enrich path).

**Implication for the plan:** Layer 3 splits into (3a) explicit `memory_hub.embodiment`
wiring, (3b) the coach centeredness fix or gate [BUG], (3c) an optional English azimuth
renderer, and a decision on the double-up. None are blockers for Layer 1, but the coach
misfire (Gap 2) must not ship silently.

---

## Sequencing + gates (post-audit)

Two tracks that were conflated in the original Landing-1 framing, now separated by the
audit. **The body wiring is safe and cheap; the substrate-primary reconciliation is the
real architecture work.**

### Track 1 — wire the SEM body live (interoceptive cascade). SAFE, offline-testable.

*Placement tags: `[generic]` = shared runtime/loop (no future robot redoes it); `[declaration]` = the minimal per-robot seam.*

- **Layer 1a `[generic]`** — the per-iteration `evaluate_failures()` tick in `agent_loop.py`,
  gated `embodiment is not None` (Track A: without it a live body is frozen). Robot-agnostic;
  `evaluate_failures()` only, never raw `tick_vital_drift` (CI grep). Every embodied robot
  inherits it.
- **Layer 1b `[generic]`** — `build_executor(entity_ref=<robot.body>, component_registry=…)`
  wired from the DECLARED body, not the literal `"bodies/reachy_mini"` (Track B: clean,
  reuses the pain_bus, gate on `_pain_bus is not None`). Behind a default-off `config.json`
  flag. Ships with the tick-fires-once integration test.
- **Layer 1c `[declaration]`** — declare `body:` in `robots.yaml`; Reachy declares
  `bodies/reachy_mini`. `RobotConfig` has no `body` field today (NH-2) — ride the existing
  free-form `config["body"]` rather than a schema change unless a typed field earns itself.
  Reuses the `has_vision`/`has_audio` capability pattern.
- **Gate:** none for safety (Track C — no spurious pain). Gate is the tick test + review.
- **Layer 3a** — explicit `memory_hub.embodiment = executor.embodiment` in
  `agentic_runtime.py` (Track E Gap 3: the runtime bypasses `_maybe_wire_body_state`).
- **Layer 3b [BUG, must not ship silently]** — data-driven drive→guidance mapping so
  `_compose_drive_modulation` renders each drive's *declared* text (not a second hardcoded
  branch), OR interim-gate the drive layer off for the azimuth drive (Track E Gap 2 / review SF-4).
- **Layer 3c [optional]** — an English azimuth renderer ("a voice, to your left") if we
  want readable direction (Track E Gap 1: today it's a raw number). Plus a decision on the
  auto-sense double-up (Track E Gap 4).

This delivers: the substrate has a body, drives tick, real proprioceptive pain becomes
possible (once telemetry is fed), and body_state reaches the prompt. **It does NOT make the
Exp 45 orient policy runnable** — that's Track 2.

### Track 2 — reconcile the Reachy runtime with substrate-primary (the real unlock for the policy)

- **P1** — give the Reachy runtime a substrate-primary path (full mode, or a hybrid where
  orienting is substrate-primary under LLM cognition). This is the architecture decision
  the operator named; it's a plan of its own, not a sub-step.
**Track 2 is now drafted: [hybrid_substrate_reflex_runtime.md](hybrid_substrate_reflex_runtime.md)** — the reflex-as-DN-Behavior design (P1 hybrid + P2 az_bin keying), which merges with the runtime plan's Landing 2.

- **P2** — in that path, key azimuth on the **az_bin string** (+ boundary sidecar from
  Steps 1/2), bypassing `SensorEncoder`, so the queen-mind policy is directly usable
  (Track D). Alternatively, wire the reserved **`"audio"` EC modality** as the exteroceptive
  bearing encoder — but that is a *new* representation the current policy wasn't trained on,
  so it means retraining, and Track D's sign-collapse caution applies to the encoder design.

**Do not conflate Track 1 and Track 2.** Track 1 is safe, small, and independently
valuable (it's the general "substrate has a body" wiring everything else needs). Track 2 is
the substrate-primary reconciliation that actually runs the learned policy, and it deserves
its own design pass — likely its own plan doc — because P1 (substrate-primary on the robot)
is a significant runtime change.

## Open questions

1. ~~P1 shape~~ **DECIDED (2026-07-17): HYBRID** — orienting reflex substrate-primary,
   high-level cognition LLM-primary. Matches the biology (reflex under deliberation) and the
   runtime plan's reflex Landing. The open sub-question is motor arbitration: the two action
   paths must share the head without fighting — and DN's `PriorityArbiter` + `InhibitionMixin`
   (from the runtime-integration audit) are the existing seam for exactly that. As a
   `[generic]` runtime mode, "hybrid" is declared by any robot, not implemented per-robot.
2. **P2 representation:** az_bin-string bypass (uses the trained policy directly) vs wiring
   the `"audio"` EC modality (principled exteroceptive encoder, but retrain-from-scratch and
   must not repeat the sign-collapse). The first ships the existing result; the second is
   the "right" long-term substrate but unproven.
3. **The generic interoception encoder folds sign** (`_normalize_value`, Track D #2) — is
   that correct for *any* signed drive, or a latent bug that also affects other bipolar
   drives? Worth a separate look.
4. **Telemetry feeds** (battery/temp/pose) are safe-to-skip (Track C) but make the
   failure_modes real — a small, high-value follow-up for genuine proprioceptive pain.
5. **The double-up** (Track E Gap 4): for a live Reachy, suppress auto-sense's interoception
   once body_state is on, or keep both per the Exp 44 ablation design?

## Pointers

- Orient-runtime plan (Landing 1 this unblocks): [orient_runtime_integration.md](orient_runtime_integration.md)
- The Exp 44 body_state ablation + the `_maybe_wire_body_state` seam: [acting_coach_body_state_ablation.md](acting_coach_body_state_ablation.md)
- `build_executor` invariant: [CLAUDE.md](../../CLAUDE.md)
- The body: [bodies/reachy_mini.yaml](../../src/maxim/_data/components/bodies/reachy_mini.yaml)
- Layer-1 result the body serves: [substrate_native_orienting.md](substrate_native_orienting.md)
