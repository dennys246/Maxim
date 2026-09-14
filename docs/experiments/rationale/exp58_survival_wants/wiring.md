# Exp 58 — WIRING lens (four-lens design review, 2026-09-14)

Reviewer charter: real consumers + real credit path (D43 family), right encoding/seams, no
hand-composed shortcut that passes while the real loop fails. Read: the prereg draft, all of
`docs/wiring/`, `scripts/survival_world/dark_danger_probe.py`; verified against
`src/maxim/proprioception/pain_bus.py`, `src/maxim/runtime/agent_loop.py`
(`propose_via_substrate`, `_encode_current_clusters`, `_read_drive_states`,
`_loop_bio_tick_maintenance`), `src/maxim/decisions/nac.py` (`recommend_action`,
`_DRIVE_TOOL_AFFINITIES`, `update_cluster_reward`, `dump`/`load_state`/`save`),
`src/maxim/hivemind/bundle.py` + `merge.py`, `src/maxim/simulation/minecraft_harness.py`,
`src/maxim/_data/components/bodies/minecraft_player.yaml`, `src/maxim/runtime/tool_dispatch.py`.

---

## DO-NOT-BUILD (as written)

### W-1. The read path is dead on this body: no affordance matches the "threat" affinity, and `recommend_action` cannot emit params — the primary DV mechanically nulls

The prereg's read design is: inject a normalized "threat" need; "selection then rides the
existing drive-prior affinity machinery toward avoidance/move affordances — no new scoring
term." Verified against the code, that machinery has **nothing to land on**:

- `nac.py::_DRIVE_TOOL_AFFINITIES["threat"] = ("flee", "hide", "retreat", "escape",
  "withdraw", "defend", "shelter")`. The minecraft body's full affordance roster
  (`bodies/minecraft_player.yaml`) is `move_to`, `turn`, `mine_block`, `place_block`,
  `eat`, `attack_nearest`. **Zero keyword matches**, and no tool name contains the
  substring "threat" for the direct name-match branch. The injected need therefore adds
  0.0 to every tool's score, and `drive_relevant` stays empty so the B7 drive gate never
  engages either. The threat need is arithmetic that touches nothing.
- Even if `move_to`/`turn` were made threat-affine, `recommend_action` returns
  `"params": {}` **hardcoded** (nac.py ~line 2381). A substrate-selected `move_to` executes
  with no x/z; directed avoidance is inexpressible through this path. (This is also why
  break-3 worked: `eat` is the roster's only param-free corrective affordance.)
- Same gap on the **other side of the DV**: `P(enter dark)` needs a nonzero pre-training
  baseline, i.e. the naive agent must sometimes *walk into the dark by its own selection*.
  With no substrate-selectable locomotion, both arms' P(enter dark) is whatever the
  harness stages, not behaviour — the gates compare two staged numbers.

**Failure scenario:** mechanism built exactly per prereg, write path perfect, dark-cluster
valence −0.9 — FEAR ≈ ABLATED on the primary DV in every trial. The falsifier fires and
ships "a null naming the read-path as the break" for a break that was knowable from two
greps before building. That is a forced null, the exact shape
`docs/wiring/substrate-learning-channels.md` warns designs away from.

**Fix shape:** the mechanism PR must ship the read path's real consumer: a substrate-
selectable, **param-free** avoidance affordance on the minecraft body + bridge, game-native
per D1 (e.g. a `flee` / `retreat` macro that pathfinds away from the nearest hostile or
toward higher light — mineflayer pathfinding makes this a body/bridge affordance, not a
harness injection), **named to match an existing threat keyword** so the affinity table
lands on it. Whatever affordance lets the agent *enter* the dark under its own selection
(a param-free `explore`/wander, or a small set of fixed-direction moves) is equally load-
bearing for the baseline. Then verify the whole read path offline via
`propose_via_substrate` (see W-2) before any live trial. Until the prereg names these
affordances and their selection path, Claim B is not buildable as designed.

### W-2. The probe re-run as the mechanism gate will not exercise the new seam — it hand-composes the loop the mechanism lives in

`dark_danger_probe.py` never calls `propose_via_substrate`: it calls
`_encode_current_clusters` + `executor.execute` + `record_outcome` **by hand** (deliberately
— it was a wiring probe for the *existing* channels). But the new mechanism's write path
depends on a loop-side caller noting the current clusters onto the provider the subscriber
reads, and its read path lives at/around `propose_via_substrate`. Neither runs in the probe:

- **Write half:** the noting caller never executes → the subscriber sees no clusters → the
  B readout stays 0.0 *with a fully working mechanism* (gate fails spuriously). The
  tempting fix — probe calls `nac.note_active_clusters(...)` itself — re-points the ship
  gate at a hand-composed sequence, the precise D43/D44 anti-pattern CLAUDE.md forbids
  ("that turns a ship gate into a test of a recipe").
- **Read half (readout C):** the probe calls `nac.recommend_action` directly with
  `_read_drive_states(...)`. The anticipatory threat need cannot live in
  `_read_drive_states` (it has no NAc access — signature is `(executor)`), so it will live
  in `propose_via_substrate` (which holds nac + clusters + drives in the right order).
  The probe's direct call bypasses it → readout C can never show the avoidance flip, even
  when everything works.

**Failure scenario:** either the gate blocks a working mechanism, or the probe is quietly
edited into a recipe test and the first place the real composition runs is the live FEAR
arm — where a wiring miss is indistinguishable from a behavioural null.

**Fix shape:** the mechanism instrument check must drive the REAL seam: a probe variant
whose per-tick body is `propose_via_substrate(nac=..., executor=..., sensor_encoder=...)`
against the scripted bridge (the production caller then does the noting, the injection,
and the selection), staging damage between ticks. Gate on (a) B flipping negative on the
dark world cluster with lit ≈ 0, AND (b) the selection flip appearing through
`propose_via_substrate`'s own return (or the emitted `recommend_action` event), never
through a hand-assembled `recommend_action` call. `run_minecraft_aut` → `run_agentic_loop`
(`_loop_kwargs` is already a pure function for exactly this D43 reason) is the live
composition this offline check must mirror.

---

## SHOULD-FIX

### W-3. The cluster-provider seam is NEW — and the clean home is NAc itself, so `build_pain_bus` can auto-wire the subscriber

The prereg says the loop "will note them onto a provider the subscriber reads (same pattern
as the loop's other per-tick state)." Verified: **no such per-tick provider exists.**
Clusters are locals in `propose_via_substrate` and fields on the proposal
(`LLMProposal.clusters`); `_encode_current_clusters` exists precisely because there is no
standing stash to read at outcome time. The nearest real precedent is NAc-owned one-step
state: `NAc.set_pending_operant_action` (noted from `tool_dispatch.record_outcome` ~line
424, read later by `credit_operant_reward`).

That precedent is also the answer to the D43 attachment question. If the provider is a new
loop-side object, `create_pain_cluster_valence_subscriber` cannot be constructed inside
`build_pain_bus` (which has only `hippocampus`/`nac`) and must be attached per-entry-point
via `additional_subscribers` — recreating the exact three-CLI-sites bug class
`build_pain_bus`'s docstring exists to kill, and making "ABLATED" the accidental default at
every entry point that forgets. **Fix shape:** put the noted-clusters state ON NAc
(`nac.note_active_clusters(agent_id, clusters)` / read under the same lock), have the
subscriber close over `nac` alone, and auto-wire it in `build_pain_bus` beside
`create_pain_nac_subscriber` and `create_percept_valence_subscriber` (`nac is not None` →
subscribed; the Wire-2 comment says why). The ablation arm then detaches it explicitly at
the harness, as declared.

### W-4. Pain fires against tick-stale clusters — a systematic boundary misattribution aimed straight at the specificity gate

Ordering, verified in `propose_via_substrate`: `evaluate_failures()` (the drive-pain
publish site, ~line 1459) runs **before** that tick's channel read + encode (~1467+). So
out-of-execute drive pain publishes while the provider still holds the *previous* tick's
clusters; in-execute pain (tool_bridge post-effect, fed by `MinecraftSyncPump`'s async
`sync_world_sensors`) fires against the *pre-action* clusters noted at propose time.

**Failure scenario:** the agent steps from lit into dark and takes the first mob hit on the
transition tick — the fear books onto the **LIT** cluster. Over an exposure schedule the
deep-in-dark bookings dominate, but boundary hits put a real negative valence on lit, and
the prereg's mechanism gate is "lit-cluster valence ≈ 0 (specificity)" plus a behavioural
"lit-area activity unchanged" gate. Both can fail for wiring reasons while the mechanism is
correct — or pass only because the harness staged episodes to avoid transitions, which the
prereg should then say out loud.

**Fix shapes** (pick one, name it in the prereg): (a) note clusters *before* the pain
evaluation — hoist the encode above the `evaluate_failures()` tick in
`propose_via_substrate` (check `docs/plans/deferred/transition_based_drive_pain.md` first:
its revival trigger is "before any change to evaluate_failures cadence"); (b) the
subscriber encodes fresh sensors at pain time (`vital_metrics` are pump-fresh; costs an EC
write inside `PainBus.publish`'s synchronous dispatch — `build_pain_bus` says keep
subscribers cheap, and an encode can mint a novel singleton cluster mid-pain); (c) accept
the one-tick window, declare a lit-valence tolerance derived from it, and stage damage
episodes off transition ticks. (a) is cleanest; (c) is honest-but-fragile.

### W-5. No failure-mode filter: hunger pain writes the fear store too

The subscriber as designed books `-intensity · alpha` onto the co-active world cluster for
**every** PainSignal above threshold. The survival world's hunger drain is live in both
claims; a food breach past the band publishes `drive:food` pain (the probe verified drives
publish with no declared failure modes needed) — while the agent stands in the **lit**
dining hall or recovery area. Both arms of Claim B accumulate negative valence on lit
clusters from hunger; Claim A's deficit induction does the same to the dining-hall cluster.

**Failure scenario:** specificity gate (lit ≈ 0) fails in both arms; worse, the agent
acquires aversion to the *food area*, coupling Claim B's mechanism into Claim A's
apparatus. The prereg's store key `(agent_id, cluster_id)` **collapses all pain sources
onto one number**, so this is unrecoverable post-hoc — Wire 2 keeps `failure_mode` in its
key for exactly this reason.

**Fix shape:** either key the store `(agent_id, cluster_id, failure_mode)` (analysis can
then marginalize, and "dark = mob damage" stays separable from "dining hall = hunger"), or
give the subscriber a declared failure-mode allowlist (`drive:health` for v1). Name the
choice in the prereg; it changes the dump/merge shapes in W-6.

### W-6. "Included in hivemind bundle export" is five wiring items, two of which fail SILENT — either wire them all now or scope the claim down

Verified current state: `NAc.dump()` carries `cluster_reward_bias` + `percept_valences`
(`_format_version` 1.1); `bundle.py::scrub_nac_state_for_bundle` and `merge.py::nac_merge`
handle both; `rekey_nac_state` re-keys the cluster-keyed fields to receiver cluster ids.
Adding the new valence map entails ALL of:

1. `dump()`/`load_state()` + `_NAC_FORMAT_VERSION` 1.1 → 1.2 (the documented migration
   pattern at nac.py ~line 129) — persistence proper.
2. `scrub_nac_state_for_bundle` — unknown keys actually SHIP by default (`dict(nac_state)`
   copy), so this one is a review item, not a blocker.
3. **`nac_merge._merge_state` builds its output field-by-field — an unlisted field is
   silently DROPPED on merge.** The bundle carries the fear; the receiver merges it to
   nothing; Phase 2 discovers the transfer null a release later. Classic D43 silent shape.
4. **`rekey_nac_state` (+ the prune helper ~line 1072) lists cluster-keyed fields
   explicitly** (`cluster_reward_bias`, `cluster_reward_source`). Cluster ids are
   sender-local; an un-rekeyed map arrives keyed to clusters that don't exist on the
   receiver — the exact dead-key bug memorialized in merge.py's own comment (~line 591)
   for the want-transfer. Silent again.
5. `nac_merge_many`'s clamp list (~line 873), plus `mypy` on `hivemind/` (CLAUDE.md: the
   bundle format is a wire boundary).

**Fix shape:** either the mechanism PR wires 1–5 with a two-agent round-trip test
(export → ingest → `get_cluster_valence` nonzero on the receiver's re-keyed cluster), or
the prereg scopes the claim to "persisted with NAc state; bundle travel is a named Phase-2
wiring item." Claiming export while merge drops the field is the shipped-the-pieces shape.

### W-7. Decay policy unstated — it decides whether the falsifier can pass at all

Every sibling map decays per tick via `_loop_bio_tick_maintenance`
(`decay_cluster_reward_biases`, `decay_percept_valences`; inherent keys decay-exempt), and
live trials run the real loop (`run_minecraft_aut` → `run_agentic_loop`), so the loop WILL
tick whatever decay the new map declares — or fossilize it forever if it declares none
(Wire 2's docstring names that failure). The prereg specifies alpha and cap but not decay.

**Failure scenario:** decay like `cluster_reward_bias` → at loop rate (~4 Hz target) the
trained valence can wash below −θ between the last damage episode and the healthy probe
windows → FEAR ≈ ABLATED, a mechanically manufactured null. Decay-exempt with no statement
→ the graduation row later inherits an undeclared permanence property.

**Fix shape:** prereg declares the decay policy (per-tick factor or exempt-with-rationale —
fear persistence is a defensible bio argument for slow/no decay in v1), adds the decay
caller to `_loop_bio_tick_maintenance` in the mechanism PR if decaying (a D43 caller —
name it), and shows the arithmetic: valence after K episodes at alpha, minus decay over
the training→probe gap, still ≤ −θ.

### W-8. Claim A's "read at the recommendation stage" must come from the emitted event stream — `recommend_action` is NOT read-only when exploration is on

`recommend_action` mutates `_ever_selected` and `_visit_count` on every passing call
(nac.py ~2357) when `substrate_explore_bonus_weight > 0`, and the explore-FIRST hard gate
reads `_ever_selected`. A harness that makes 30 × 2 states × 5 seeds of *side*
`recommend_action` calls against the live agent's NAc perturbs the very selection dynamics
under test (r2_learned_bias.py's "recommend_action is read-only" comment is true only at
explore weight 0). **Fix shape:** read the selection DV from
`_emit_recommend_action_event`'s stream (it already carries `best_tool`, `passed_gate`,
per-component decomposition, per-modality consulted bias — everything the DV needs), or
pin `substrate_explore_bonus_weight = 0` in the frozen-apparatus fingerprint (the
`frozen-apparatus-hygiene` wiring stub names explore-bonus as a silent selection-changer).
Say which in the prereg.

---

## NIT

- **N-1. Interactive-mode gate:** the new subscriber should mirror `create_pain_nac_subscriber`
  / `create_percept_valence_subscriber`'s interactive-mode suppression, for the same
  contamination reason.
- **N-2. Refractory + latch, restated for the harness:** PainBus drops repeats within 0.5 s
  per `(entity, failure_mode)`, and drive pain re-fires only on deepening breach, clearing
  only when an evaluation observes recovery. The prereg's interleaved lit recovery covers
  the latch; the harness must also keep the K ≥ 10 episodes spaced past the refractory and
  tick `evaluate_failures` in the healthy state (the probe's run-#2 lesson: 1 publish
  across 8 breaches without it) — put both in the stop-rule instrumentation.
- **N-3. Injection ordering:** inject the anticipatory threat AFTER the interoception encode
  in `propose_via_substrate` (post-encode, pre-`recommend_action`), so learned fear does
  not perturb interoception cluster identity between arms (`_read_drive_states`' derived
  needs DO feed the encode, by documented intent); combine with the innate reactive
  `health→threat` via max (the `derived_needs` convention), never sum.
- **N-4. Naming:** keep the new map's name clearly distinct from Wire-A's
  `cluster_reward_bias` (per-(agent, cluster, TOOL)) — a reviewer folding the tool-less
  valence into that map under a fake tool key would inherit the wrong decay, cap, and
  merge semantics.

---

## Verified clean

- **`_read_drive_states`' shape admits the injected need.** The dict is
  `{name: value in [0,1]}`; the affinity machinery fires on values > 0.5, the raw-sensor
  guard skips values > 1.0, "threat" is already a table key, and the max-combine with the
  innate `health→threat` composes correctly at full health (innate contributes 0). The
  *vector* is fine — W-1 is about what the vector lands on.
- **Drive pain reaches subscribers with rich context through the real path** — probe-measured
  (intensity 1.0, 8-key context; `.recent`'s lossy view is a red herring, per the wiring doc).
- **`build_pain_bus` is the right door**, and the auto-wire precedent (Wire 2 subscribed
  whenever `nac is not None`) is exactly the shape W-3 asks the new subscriber to take.
- **No double-attribution risk:** B8 keeps action-blame suppressed for bystander damage; the
  new store, Wire 2 (`_percept_valences`), causal links (`_links`), and Wire-A
  (`_cluster_reward_bias`) are four disjoint maps with disjoint keys. The dark=danger fear
  correctly does not ride action-blame.
- **The ABLATED arm's harness-level detach isolates only the new mechanism** — pain still
  publishes, Wire 2 and the NAc pain subscriber still fire, matching the prereg's stated
  ablation semantics.
- **Persistence precedent is real:** `NAc.save` → `atomic_write_json` +
  `with_format_version`; the 1.0→1.1 `percept_valences` migration is the documented
  template for adding the new map (W-6 is about the *hivemind* half, not local persistence).
- **Live trials run the real composition:** `run_minecraft_aut` → `run_agentic_loop` with
  `aut_mode="substrate-primary"` via the pure `_loop_kwargs` (kept pure for D43 pinning),
  so an end-to-end mechanism-in-the-loop test is genuinely available — W-2's ask is to use
  it offline before going live.
- **Phase-0 separability stands:** dark/lit encode to distinct world clusters through the
  production encoder (Phase-0 record + the probe's own instrument guard), so the store's
  key material exists.

## D43 consumer roster (every piece, with its caller — none may ship caller-less)

| Piece | Real consumer / caller | Status |
|---|---|---|
| `note_active_clusters` write | `propose_via_substrate` (substrate-primary); the llm-primary outcome-time encode site (~agent_loop 3466) | NEW callers — name in mechanism PR |
| `create_pain_cluster_valence_subscriber` | `build_pain_bus` auto-wire (W-3) | NEW — auto-wire, not per-harness |
| valence store write | the subscriber (above) | NEW |
| `get_cluster_valence` read | the threat-injection site in `propose_via_substrate` | NEW caller |
| injected threat need | `recommend_action` affinity machinery → **a threat-keyword avoidance affordance** | consumer MISSING (W-1) — the load-bearing gap |
| decay | `_loop_bio_tick_maintenance` | NEW caller or declared exempt (W-7) |
| bundle travel | `nac_merge` + `rekey_nac_state` + prune + `nac_merge_many` | silently dropped today (W-6) |
| mechanism gate | probe variant driving `propose_via_substrate` | probe as-written does NOT consume the seam (W-2) |
