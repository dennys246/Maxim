# Exp 61 — ENVIRONMENT lens (four-lens design review, 2026-09-16)

**Charter (docs/experiments/DESIGN_REVIEW.md):** does the world game-natively afford it (D1), are the
needed states/acts reachable and measurable, does the bridge/world behave — the lens that would have
caught hunger-drains-too-slowly, food-caps-at-20, the eat-lag. Read first:
`docs/wiring/harness-loop-must-be-proven-live.md`, `docs/wiring/sensor-range-clamps.md`,
`docs/experiments/exp60_drowning_avoidance_prereg.md` (§Apparatus, §Gate (ii), §Design (iii), the
runbook, Amendments 1–7, §Outcome), `docs/experiments/rationale/exp60-drowning/environment.md`.

**Verdict line: FIX-THEN-BUILD.** The world affords everything the design asks for and Exp 60 already
proved the apparatus, the states, the act and the measurement live. Nothing here is a design flaw of
the Slice-2 kind. But six things the prereg asserts about the apparatus are not true of the apparatus
as it exists — one of them (S1) would make every arm-2 pair REFUSE at the ingest gate, and the
sentence "B has never been underwater" (S2) is false under the preflights the prereg declares it
reuses. All six are fixable in the harness PR and the prereg fold, none needs a new mechanism.

---

## Verified first (citations are `file::symbol`)

**Apparatus and edges (Exp 60, unchanged).**
- Classroom: `scripts/survival_world/setup_world.py::water_classroom_geometry` — shore (ax−1, 40, az),
  submerged target (ax+4, 35, az) with the head at y=36 in water, 45 SOURCE blocks, stone shell,
  `forceload` first; `::water_classroom_commands` sets `doMobSpawning false`, the shore spawnpoint
  and the `exp60_deaths` objective; `::water_classroom_verifications` (10 `execute if block` asserts
  after a 2 s fluid pause). Anchor record `~/.maxim/exp60_water_classroom.json`
  (`::water_anchor_record`). Built at shore (−393, 40, −312), submerged (−388, 35, −312), depth 5
  (`docs/experiments/data/exp60_water_apparatus.json::apparatus`).
- Measured edges (`exp60_water_apparatus.json`, 3 cycles): `t_in_water` 0.40–0.79 s, **`t_pain_edge`
  5.085 / 5.186 / 5.443 s**, `t_oxygen_zero` 15.08–15.27 s, `t_damage_onset` 16.07–16.65 s,
  `health_at_rescue` 18, W4 `t_surface` 1.45–1.83 s. W1 baseline: `nearest_hostile_dist` 64,
  `hostile_count` 0, `light_level` 0, `time_of_day` 0.04167 (= `time set day`, 1000/24000),
  `distance_from_spawn` 69.17. **W1 records `is_raining`, `nearest_player_dist`, `saturation`, `food`
  as `None`** — they are neither gated nor recorded (see S4).
- Harness (`scripts/survival_world/exp60_run.py::_run`): `probe_cap_s = min_pain_edge_s(apparatus) −
  FROZEN["probe_cap_margin_s"]` = 5.085 − 0.75 = **4.335 s** (pain edge read from the git-tracked
  apparatus record, NOT the anchor); `train_cap_s` = anchor `measured.t_damage_onset_min_s` − 1.0 =
  15.067 s. Preflights: fingerprint (`::fingerprint_drift`), raw roster
  (`exp60_water_check.py::missing_bridge_sensors`), cadence ≤ 0.15 s (`::_measure_bridge_cadence`),
  loop liveness ≥ 4 ticks / 3 s, five gamerules VERIFIED not toggled (`doMobSpawning false`,
  `doDaylightCycle false`, `doWeatherCycle false`, `doImmediateRespawn true`, `keepInventory true`),
  live cluster-distinct (`_submerge("preflight")` + `agent_loop._encode_current_clusters`), escape
  actuation through `aut.client.call_action("escape_water")` (never the executor),
  `get_positive_outcomes(escape_water) == []`. Placement = `_stop_motion()` (bridge `stop`) → RCON
  `tp` to the floor (window clock) → 4 Hz `_sample` → rescue teleport FIRST then loop stop
  (`::_loop_window`); rescue = teleport + `settle_until(is_in_water < 0.5 ∧ oxygen ≥ 19)` + `_heal()`
  (`effect give instant_health 1 10 true` + `saturation 1 10 true`) + `settle_until(health ≥ 20 ∧ food
  ≥ 16 ∧ saturation ≥ SATURATION_REST(10))` (`::_rescue`). Training: propose-only
  `propose_via_substrate` at 4 Hz until a `drive:oxygen` publish at intensity 1.0 with `is_in_water`
  1 and sensed oxygen ≤ 12, then rescue + 4 healthy ticks; arrival health < 20 or any `drive:health`
  publish REFUSES. Teardown per seed: `aut.bio.on_session_end()` (cerebellum + distributor ONLY —
  `runtime/bio_stack.py::BioStack.on_session_end`), `client.close()`, `shutil.rmtree(persistence_dir)`.
- Bridge (`scripts/minecraft_bridge/index.js`): one client at a time, slot freed on socket `close`;
  `snapshot()` every `STATE_INTERVAL_MS` (must be 100 for the harness); `is_in_water` = eye-height
  block (1.62) is water/bubble_column; `escape_water` holds `jump` until the eye block is air +
  `SURFACE_HOLD_MS` 600, hard cap 8 s, **releases `jump` in `finally`** (no held control survives the
  action), returns "already at surface" (ok:true) when called dry; `flee` throws FAST when
  `bot.entity.isInWater`, `canDig=false` otherwise; `stop` clears the pathfinder goal only. Text
  events only on chat/damage/death/spawn/kicked/error (the live loop needs no event since #732).
- Client lifecycle (`src/maxim/simulation/minecraft.py::MinecraftClient.connect`): `confirm_timeout_s`
  + `retries` (the Exp 56 Amendment-4 fix for the async slot-free race); `::close` drops the socket;
  `latest_state()` is per-client and EMPTY until the first snapshot. Serial AUT sessions on one
  bridge without a restart are PROVEN: 20/20 Exp 60 seeds (runs 1+2) each opened a fresh client
  (`retries=8, backoff 0.5 s`) against a bridge that was never restarted between seeds.
- Exp 60 record (`docs/experiments/data/exp60_trials.jsonl`, run 2 ids `301eb2edff6d`/`eeb92752ee2b`):
  per-seed wall time from the row timestamps **195–200 s (FEAR), 208–211 s (ABLATED)**; every seed
  `bridge_state_interval_s` 0.1008–0.1017, `loop_liveness_ticks` 6, actuation preflight `t_surface`
  1.37–1.59 s, training 10/10 attempts, 20 oxygen / 0 health pain signals, 0 deaths, one episode
  cluster = the probe cluster. FEAR first post placement: `flee` call at 0.87 s (fails), `escape_water`
  call at 1.55 s, surfaced read at **2.92 / 3.28 / 3.16 / 3.24 / 3.34 s** (cap 4.335); placements 2–6
  1.28–2.29 s. Pre-probe ticks ~10 per window with NO proposal (no drive on a fresh dive).
  **FEAR post shore roam: 14 `escape_water` executions on the DRY shore** ("already at surface"
  succeeds → more positive links) — the state-blind positive link the prereg's Trap 1 describes,
  visible in the record.
- Exp 56 serial pairs (`scripts/exp56/run_campaign.py::run_pair_arm`, `common.py`): donor built,
  trained, `close_and_stage_session` (`memory_hub.on_session_end()` THEN `bio.on_session_end()`, copy
  `nac.json`/`ec.json` → `aut_nac.json`/`aut_ec.json`, client closed) → `export_bundle` (the real CLI,
  `--session <stage> --body-ref --body-yaml`) → receiver built, closed at rest, `ingest_bundle_into`
  (real CLI `--apply`, journal entry read) → receiver REBUILT from the same home → probe. World reset
  between sessions = `settle_until_reflected` teleports only (no kill/clear). Campaign record
  (`docs/experiments/data/56_four_arm.jsonl`): **200 rows in 51.0 min, 15.4 s/row**, one bot, one
  bridge; the first live campaign crashed at ~88% on the bridge-busy race (Amendment 4), fixed by the
  confirm+retry connect Exp 60 already uses.
- Export/ingest surfaces (`src/maxim/hivemind/cli.py`): `::_run_export` reads ONLY
  `aut_nac.json`/`aut_ec.json` from `--session` (`::_expand_session_dir` accepts an absolute dir);
  `::_resolve_receiver_pair` accepts EITHER `aut_*` or the bio_stack's `nac.json`/`ec.json` pair, so
  ingest works directly into a `build_minecraft_aut` persistence dir; `_run_ingest` docstring: "MUST
  NOT run against a receiver a live session currently owns". `build_bio_stack(persistence_dir=…)`
  loads the persisted NAc/EC at build (`runtime/bio_stack.py::build_bio_stack`), so Exp 56's
  close → ingest → rebuild sequence carries over unchanged.
- Persistence gap (load-bearing for S1): `integration/memory_hub.py::MemoryHub.on_session_end` returns
  `{}` WITHOUT saving when `_session_active` is False and a session was ever started — its own comment:
  "mutations made AFTER the loop closed and before shutdown() are not persisted by this call."
  `run_minecraft_aut` opens+closes a hub session per loop run (`consolidation="full"`); Exp 60's
  propose-only training runs with NO loop, i.e. after the liveness loop's close. Exp 56 avoids this by
  `aut.bio.memory_hub.on_session_start()` at `common.py::build_bench_session`.
- `NAc.decay_all` (called 0.95× by the hub close) decays CAUSAL LINKS only
  (`decisions/nac.py::NAc.decay_all`); `cluster_fear` is untouched by the donor's export close and by
  the receiver's liveness close. Verified so the −1.0 survives the (at least) two closes between
  booking and B's read.
- H2 status: the commit "a GAINED modality's geometry tag carries the declared range VALUES (H2,
  Option A)" is `243cee9a` on branch `feat/h2-geometry-tag-ranges`; **main HEAD is `adcd6808` (#739)
  — H2 is NOT on main** as of this review. The prereg cites it as "#740" merged.
- `~/.maxim` sharing: the harness passes a `persistence_dir`, so `user_memory()` is never the home;
  `EC.config.enable_semantic` defaults False (`similarity/ec.py`), so the shared
  `~/.maxim/util/semantic_embeddings.npz` is never written; the only `~/.maxim` read is the anchor.
  The loop's CWD-relative `data/agents/<name>/runtime/state_*.json` lands inside whatever dir the
  harness `os.chdir`s to.
- LLM co-location: `minecraft_harness.py::_loop_kwargs` passes no LLM worker; `aut_mode`
  substrate-primary; export/ingest are file operations. This harness is LLM-free exactly as Exp 60 was
  and is exempt from CLAUDE.md's leader/harness co-location rule on the same grounds; a co-tenant CPU
  spike would surface as a per-pair cadence/liveness REFUSAL, not as a silent latency.
- Replay of cos(A's water node, B's contact reading) on the shipped embed (`similarity/encoder.py::
  _sensor_embed`, gain exponent 3.0, the exact method of `docs/experiments/data/
  exp60_saturation_rest_check.py`; A's node = its first-touch dive-second-0 vector: the frozen-centroid
  EC's centroid is the first-touch reading; threshold 0.85 for completion). Identical reading: 1.000.
  Per-sensor deviations of B's reading from A's node:

  | deviation on B's side | cos | completes (≥ 0.85)? |
  |---|---|---|
  | `doDaylightCycle` on: time 0.10 / 0.25 / 0.50 / 0.90 / 0.99 | 0.987 / 0.920 / 0.892 / 0.884 / **0.799** | mostly, but NOT at 0.99 — and none of it is stable |
  | `time set noon` (6000) vs A at `day` (1000) | 0.920 | yes, thin |
  | a player (spectator) at 32 / 16 / 8 / 4 blocks | 0.998 / 0.974 / 0.931 / **0.896** | yes, but eats the margin |
  | `is_raining` 1 (froze while raining) | **0.8525** | NO |
  | food 16, saturation 0 (not re-satiated) | **0.8499** | NO |
  | saturation 5 / 2 with food 20 | 0.998 / 0.962 | yes |
  | oxygen 12 / 8 / 3 / 0 (A's conditioning moment vs B's second 0) | 0.9995 / 0.994 / 0.941 / 0.847 | yes down to the pain edge (gate (ii)'s early/late bin, measured) |
  | health 18 / 16 (damage carried in) | 1.000 / 1.000 | yes (rest 20 of [0,40] barely moves) |
  | `look_pitch` 0.3 / 0.8 / 1.2 rad | 1.000 / 0.998 / 0.971 | yes |
  | `on_ground` 0 (sinking) | 0.998 | yes |
  | `hostile_count` 1, `nearest_hostile` 48, xp 1–5 | 1.000 | yes |

  Reading: with the five gamerules verified and the rescue bar (food ≥ 16, saturation ≥ 10) enforced
  at every teleport, EVERY sensor B could differ on is either frozen, re-set by the rescue, or
  separability-inert. The two that can break completion (`is_raining`, an un-satiated bot) are
  frozen/reset by the existing protocol; the two that are NOT GATED anywhere (`nearest_player_dist`,
  `is_raining`) are exactly the ones a 3 h campaign invites (S4).

---

## Findings

### DO-NOT-BUILD — none

The environment half is sound: the water, the drowning pain, the act, the rescue, the serial
donor→receiver swap on one bot and the export/ingest on the box are all either measured live (Exp 60,
Exp 56) or verified in source above. Nothing requires a synthetic sensor or reward.

### SHOULD-FIX

**S1. The donor's fear is booked in a phase no hub session covers, so the STAGED `aut_nac.json` would
not carry it — every arm-2 pair refuses at the ingest gate.** Evidence: Exp 60's training is
propose-only with the loop STOPPED (`exp60_run.py::_run`, the training `while` loop calls
`propose_via_substrate` directly); the hub session opened by the liveness/pre-probe loops was closed
by `run_agentic_loop`'s own end-of-run; `MemoryHub.on_session_end` on an already-closed hub returns
`{}` without `nac.save` (its comment names this exact residual gap). Exp 56's `close_and_stage_session`
would therefore copy the LIVENESS-close `nac.json` (fear 0). The prereg's "donor sanity" reads the
in-memory NAc (as Exp 60's G2 does) and would PASS while the bundle ships nothing; the prereg's own
`fear_rekeyed == 1` gate would then REFUSE the pair — loud, not silent, but 100% of arm 2. Concrete
change: (a) the donor flow opens a hub session around training (`memory_hub.on_session_start()` before
the first episode, as `exp56/common.py::build_bench_session` does) and closes it via
`close_and_stage_session`; (b) donor sanity reads the STAGED `aut_nac.json` (exactly one
`cluster_fear` key, value −1.0, `drive:oxygen`, world cluster id ∈ the training episodes' noted ids,
zero positive `escape_water`/`flee` links) — the file the export reads, never the object; (c) the
export-before-probe rule becomes a file fact: the stage is written BEFORE any loop is started after
training. Also drop Exp 60's per-seed `shutil.rmtree(persistence_dir)`: the donor stage must outlive
the pair (arm 4 reuses it — Exp 56's `--resume requires a durable --workdir` refusal exists for this
reason).

**S2. "B has never been underwater" is false under the preflights the prereg says it reuses
unchanged — and the submersion it involves is the design's best pre-placement gate, so name it.**
Evidence: `exp60_run.py::_run` submerges the bot TWICE before the pre-probe with the AUT built and the
sync pump writing the body: the live cluster-distinct preflight (`_submerge("preflight")` → the
harness ENCODES B's submerged reading through `_encode_current_clusters` on B's EC) and the escape
actuation check (`_submerge` + bridge `escape_water`, ~1.4–1.9 s submerged). Neither is a US (oxygen
stays ≥ ~17; the pain edge is 5.085 s; no `drive:oxygen` publish — the subscriber proves it), so "B
never experienced the US" is true; "B's FIRST submersion" as the DV placement is not. Concrete change:
(a) reword the claim to "B has never experienced the air-hunger pain; on its first LOOP-LIVE
submersion …" and disclose the two preflight submersions with their recorded min oxygen and zero pain
publishes; (b) turn the cluster-distinct preflight into the receiver's explicit REPRESENTATION gate:
B's submerged preflight encode must return the TRANSFERRED node's id (post-`id_map`) in arms 2/3 and
a fresh id in arm 1 — a mismatch in arm 2 is a REFUSAL class of its own ("the reading did not
complete into the imported node"), distinct from the behavioural null, and it is measured BEFORE the
one-shot placement is spent; (c) if the review prefers B to be literally never submerged before the
placement, run the actuation check on the BOT before B's AUT is built (it is a bot/bridge property,
not an agent property; the raw client + RCON suffice) and keep only the encode gate — but do not
drop the live cluster-distinct gate, it is gate (ii)'s necessary partner.

**S3. A 1.0–1.4 s margin on a ONE-SHOT binary, over a 3 h campaign, needs a censoring class and a
drift refusal — Exp 60 cannot tell trend from noise at n = 5.** Evidence: first-placement surfaced
reads 2.92, 3.28, 3.16, 3.24, 3.34 s across seeds 11–15 (mean 3.19, sd 0.17) against the 4.335 s cap;
decomposition from the tick record: tie-break `flee` call at ~0.87 s, `escape_water` call at ~1.55 s,
escape 1.37–1.89 s (`t_surface` across runs), sample read +0.25 s, snapshot +0.10 s; worst-case tick
alignment (+0.5 s on each of two proposals) plus the slowest observed escape sums to ≈ 4.8 s > cap.
Exp 60 absorbed this with 6 placements per probe; Exp 61 has one per receiver, so one censored true
surface = 0.083 on the arm-2 rate. Actuation `t_surface` showed no trend (1.53, 1.44, 1.54, 1.59,
1.37) but the loop-side first-placement latency rose 2.92 → 3.34 over 17 min — indistinguishable from
noise at n = 5, and a real +0.08 s/pair would cross the cap by pair ~15. Concrete change: (a) record
per placement `t_flee_call`, `t_escape_call`, `t_first_air` and name "escape CALLED before the cap,
head NOT in air by the cap" as its own censoring class (reported, and re-runnable on a fresh seed as an
instrument refusal if the escape itself exceeded its measured max); (b) per-pair refusal bounds on the
apparatus numbers the harness already stamps — actuation `t_surface` ≤ 2.5 s, cadence ≤ 0.15 s,
liveness ≥ 4 — plus a CAMPAIGN-level check in the verdict: refuse INCOMPLETE if the actuation
`t_surface` or the arm-2 first-placement latency drifts monotonically across pairs (e.g. last-quartile
median − first-quartile median > 0.5 s); (c) keep Exp 60's shore warm-up (1 s) and liveness preflight
(3 s) for the receiver verbatim — the loop must be proven live BEFORE the one placement, exactly the
wiring lesson.

**S4. Two full-weight sensors are gated NOWHERE and are the two a 3 h campaign invites: a joined
player and the frozen weather state.** Evidence: W1 (`exp60_water_check.py`) gates
`nearest_hostile_dist` and `distance_from_spawn` and records `hostile_count`/`light_level`/
`time_of_day`; the apparatus record shows `is_raining: None`, `nearest_player_dist: None`; the
harness fingerprint gates config, not readings. Replayed: a spectator at 4 blocks → cos 0.896 (still
completes, margin gone); `is_raining` 1 → 0.8525 (does NOT complete). `doWeatherCycle false` freezes
whichever weather was current — if it was raining when the world was prepared, every situation
carries a full-weight constant (the saturation lesson, Amendment 1) and the live cluster-distinct
preflight would refuse LOUDLY, so the failure is not silent — but it would refuse the whole campaign
at pair 1, not a pair. Exp 56 had a `--spectator` mode precisely because a human peeks at long runs;
this design has no equivalent and `nearest_player_dist` (range [0,128], rest 64 = the bridge cap) is
the sensor that would carry the peek. Concrete change: gate `is_raining == 0` and `nearest_player_dist
== 64` (a) in the campaign preflight, (b) at every rescue settle (`_rescue`'s predicate) — the
teleport instant of every encode-bearing placement — and (c) refuse the pair if either flips inside a
window; forbid any second player on the server for the campaign in the runbook (or op them as
spectator ≥ 64 blocks away, which the bridge reads as 64).

**S5. One process, tmpdirs, no resume — the Exp 56 88%-crash shape, at six times Exp 60's length.**
Evidence: `exp60_run.py::_run` runs all seeds in ONE Python process (5 per invocation to date), mints
a `tempfile.mkdtemp` home per seed and `rmtree`s it; each seed builds a full AUT (bio stack, executor,
client reader thread, telemetry writers, an executor spy closure, a bus subscriber). Exp 61 needs ~24
donor builds + ~96 receiver builds (pre-ingest + post-ingest) + ~144 bridge connects in one campaign —
never exercised; and a crash at pair 30 loses nothing only if rows are flushed (they are, per seed)
AND the donor stages survive AND the harness can resume by (pair, arm). Exp 56 ran 200 rows / 51 min
to completion only after the busy-race fix and with `--resume` + durable `--workdir`. Concrete change:
durable `--workdir` with per-pair subdirs (donor stage, bundles, receiver homes kept; `--keep` for the
first pair's anti-vacuity kit as Exp 56); `--resume` keyed on (pair, arm) with Exp 56's refusal on a
redirected output; either one subprocess per pair (cleanest — provenance stamped per row anyway) or a
per-pair RSS line in the record so growth is visible; connect with `confirm_timeout_s`/`retries` as
Exp 60 (already there). Also: interleaving arms within a pair requires ONE invocation across arms —
Exp 60 mints one `run_id` per ARM invocation and its verdict takes `--run-id` per arm; Exp 61's
verdict must select by a campaign/pair id, not per-arm run ids.

**S6. H2 is not on main; the campaign must run at ONE code hash at or after its merge, and the
verdict must refuse a campaign whose rows span two.** Evidence: `243cee9a` (H2, Option A) sits on
`feat/h2-geometry-tag-ranges`; main HEAD `adcd6808` is #739 (the encoder golden pin, which proves the
VECTORS are unchanged by H2 — only the tag). Environment consequences: (a) every agent in Exp 61 is
fresh, so no stale-geometry node or deduped mismatch warning (`similarity/ec.py::_note_geometry_
mismatch`) can appear on the box — UNLESS the operator pulls mid-campaign, after which the live tag
differs from every earlier donor's stamp and the strict ingest refuses every later fold (and any
receiver built before the pull carries pre-H2 nodes that the post-H2 live encode masks out of
completion); (b) `maxim substrate invalidate` is not needed and must not be run on any Exp 61 home;
(c) the gate-(ii) record's `code_hash` predates H2 — the harness compares the fingerprint (ranges),
not the tag string, so that record stands, but the prereg should say so. Concrete change: state H2 as
a merged-on-main dependency in the prereg's freeze conditions; the verdict returns INCOMPLETE if
`provenance.code_hash` differs across any two rows of the campaign; the runbook forbids `git pull`
between the first and last row.

### NIT

**N1. Re-measure the pain edge on campaign day.** The cap derives from the 2026-09-15 apparatus
record (three cycles: 5.085–5.443 s; the harness takes the MIN). The edge is game physics (air 300
ticks; pain below 14 of 20 bubbles) and should not move, but the record predates the eye-height
`is_in_water` change (#730) and the campaign is a different day on a different checkout;
`exp60_water_check.py` costs ~3 min. Run it as the campaign's first gated record and have the harness
read `t_pain_edge_min_s` from THAT record (data PR alongside), so the anchor and the record are the
same day.

**N2. Teleports carry no rotation.** `exp56/common.py::RconControl.teleport` sends `tp <bot> x y z`,
so yaw/pitch are whatever the last motion left — `look_pitch` is a full-weight world sensor
(range ±1.5708, rest 0). Replayed impact is benign (0.998 at 0.8 rad) and nothing in the protocol
moves the head (`flee` fails before any look in water; `escape_water` does not look), but it is a free
variable across 3 h for zero cost: `tp <bot> x y z 0 0` on every placement/rescue pins it.

**N3. Budget arithmetic.** "48 pairs × 3–4 min" double-counts: arms 1 and 4 train no donor. Measured
composition: Exp 60 seed ≈ 195–211 s WITH two 6-placement probes + two roams; a training-only donor ≈
preflights ~30 s + 10 episodes × ~13 s (≈ 6.5 s to the saturating publish + rescue/heal settle + 4
healthy ticks) ≈ 2.5–3 min; a receiver ≈ build/close/ingest/rebuild + preflights (liveness 3 s,
cadence 3 s, actuation ~5 s, four rescues ~8 s each) + one placement ≈ 60–90 s. 24 trainings + 48
receivers ≈ 2.4–2.8 h. The prereg's 3 h is right; the per-pair figure is not.

**N4. CLI paths and CWD.** Exp 60 `os.chdir`s into the per-seed home so the loop's CWD-relative
`data/agents/…` state lands there. The export/ingest CLI runs in-process
(`hivemind.cli.run_substrate_subcommand`) and resolves a relative `--session` against CWD — pass
absolute paths only, and never `chdir` into a receiver home that a live session owns while ingesting
(`_run_ingest`'s contract). The `data/agents/` residue inside a stage dir is harmless to
`_run_export` (it reads two named files) — say so, or stage into a clean dir as Exp 56 does.

**N5. Any loop-live shore window AFTER a surfaced placement books positive links on the dry shore.**
Exp 60's FEAR post roam executed `escape_water` 14× on the shore ("already at surface" = ok:true).
Irrelevant to the first-contact DV, but Exp 61's "second placement (fear + own link)" secondary and
any reused shore roam must run BEFORE the placement or be dropped; a roam after it measures the
snowball, not the fear.

---

## Answers to the six questions, in one place

1. **Bot state across A → B.** Carries: position/rotation (bot entity), health/food/saturation/oxygen
   (game), potion effects (1 s, expire), XP (0 throughout; `keepInventory` also keeps XP on death),
   deaths objective (delta-read per seed), inventory (64 bread; `eat` never proposed at food 20),
   pathfinder goal/movements (`stop` clears the goal; `flee` sets canDig=false movements that
   persist harmlessly), the bridge's mineflayer `bot` (live). Does NOT carry: the Python client's
   `latest_state` (per client, empty until the first snapshot; `connect(confirm)` waits for it), any
   held control (`escape_water` releases `jump` in `finally`). Resets already in Exp 60's flow and
   sufficient: `_rescue` (teleport + observed oxygen ≥ 19 + heal/satiate settle to health 20 / food ≥
   16 / saturation ≥ 10) before EVERY encode-bearing moment, `_stop_motion` at phase boundaries, the
   per-seed `finally` (teleport + heal). Missing: S4's two gates, N2's rotation. Of the carried
   items only an un-satiated or damaged bot or a joined player could move B's contact reading, and
   the replay shows the rescue bar already keeps the first two inside completion.
2. **Same reading?** Yes, given the frozen gamerules and the rescue bar: every sensor is either
   frozen (`time_of_day` 0.0417, `light_level` 0, `is_raining`, `distance_from_spawn` 69,
   `nearest_hostile_dist` 64 by the ≥ 72 clearance), reset (health/food/saturation/oxygen), or inert.
   A 3 h `time_of_day` drift would be fatal (cos down to 0.80) but cannot happen: `doDaylightCycle
   false` is set by `setup_world.py::_prepare` (`_GAMERULES`, plus `time set day`) and VERIFIED, not
   toggled, per seed by the harness. The campaign does not need a new time rule; it needs the two
   S4 gates and a stated dependency that `prepare` ran on the campaign world.
3. **Duration and stability.** ~500 dives/rescues: no blocks change (`escape_water` = jump control;
   `flee` canDig=false; `mobGriefing false`; the three block affordances need params the substrate
   never emits), no entities (`doMobSpawning false` verified; effects are 1 s; deaths 0/20 in Exp 60),
   chunks forceloaded. Exp 60 showed zero drift in cadence/liveness/actuation across 10 seeds; the
   loop-side first-placement latency is the only number that MIGHT trend (S3). LLM-free → exempt from
   the co-location rule as Exp 60 was. The real 6× risks are process-level (S5), not world-level.
4. **Export/ingest on the box.** Shape matches Exp 56 once the donor is STAGED (`aut_*` names; the
   export reads only those); ingest accepts the bio_stack `nac.json`/`ec.json` pair directly; rebuild
   from the same home loads it. The `~/.maxim` shared state does not interfere (anchor read-only;
   semantic store off; `user_memory()` unused). S1 is the one real gap (the hub session around
   training); S6 covers H2 (not merged; no stale warning on fresh homes; a mid-campaign pull would
   refuse every later fold).
5. **DV measurability.** Margin ≈ 1.0–1.4 s on Exp 60's first placements, worst-case alignment can
   exceed the cap (S3). The cap is NOT re-measured per campaign (read from the merged 2026-09-15
   record; N1). The receiver must get the identical shore warm-up AND the 3 s liveness preflight
   (S3c).
6. **D1.** "B never drowned" is verifiable from bridge truth and the bus: B's pain subscriber attached
   at build (before any submersion) records zero `drive:oxygen`/`drive:health` publishes before the
   placement; health 20 at every settle; deaths delta 0; min sensed oxygen during the two preflight
   submersions recorded (expect ≥ 17); B's pre-ingest `nac.json` carries no `cluster_fear` and the
   post-ingest one exactly the transferred key. Synthetic steps: none beyond Exp 60's accepted
   apparatus (teleport placement, RCON heal/satiate, harness-scheduled propose-only training, rescue
   at the cap) and the shipped export/ingest path; the one hand-built INPUT is arm 4's nac-only
   re-compose, Exp 56's accepted falsifier. No synthetic sensor, no synthetic reward.

---

## Verdict

**FIX-THEN-BUILD.** The world affords the experiment: the drowning pain, the separable in-water cue,
the `escape_water` act and the rescue are measured and EARNED on this exact apparatus, one bridge
serves serial AUT sessions without restarts (20/20 Exp 60 seeds; 200 Exp 56 rows in 51 min), and the
export → ingest → rebuild sequence exists end to end with a real precedent. Nothing in the design
needs a synthetic sensor or reward, and the receiver's contact reading provably completes into the
donor's node under the frozen gamerules and the rescue bar. But the prereg's "Exp 60's preflights,
unchanged" and "Exp 56's export, exactly" do not compose as written: the donor's fear is booked in a
phase no hub session persists (S1 — every arm-2 pair would refuse at the ingest gate), the reused
preflights submerge B twice before the "first submersion" (S2 — reword, and promote the encode into
the representation gate), the one-shot binary sits ~1 s inside a cap with no censoring class or drift
refusal (S3), two full-weight sensors a 3 h run can move are gated nowhere (S4), the harness shape
(one process, tmpdirs, no resume, per-arm run ids) is Exp 56's pre-Amendment-4 crash shape at six
times Exp 60's length (S5), and H2 is cited as merged when it is not (S6). All six are harness-PR and
prereg-fold changes; fold them and build.

## What I did NOT verify

- Nothing was run against the live server: no RCON, no bridge, no big-mac-mini state (memory/CPU
  headroom, whether a qwen32b server is resident, spigot `entity-tracking-range`, the current weather
  state of the campaign world, whether `prepare` ran on it). All timing claims are from the Exp 60/56
  committed records; all cos numbers are offline replays on the shipped embed with reconstructed
  normalized values (the same method as `exp60_saturation_rest_check.py`), not live captures.
- The wiring lens's questions — whether the aligned EC merge inserts A's water node under an id the
  fear key survives the re-key to, what `strict_geometry` does on ingest, whether `_validate_nac_
  payload` or the V1–V10 adapter touches `cluster_fear` — I read the CLI surfaces only, not
  `hivemind/merge.py`/`ingest.py` internals.
- Whether `NAc.load()`'s decay-on-load (`bio_stack.py` comment) touches `cluster_fear` when B is
  REBUILT from its ingested home (a load-path sibling of the `decay_all` check I did make); the
  receiver's G2 read at contact (> 0.5) is the belt either way, and the prereg gates it.
- The loop-side latency trend (S3) is a 5-point observation; I could not distinguish it from noise
  and did not claim to.
- Java/Paper long-run behaviour (GC pauses at hour 2–3) — unmeasured; the per-pair cadence/liveness
  refusals are the only instrument the design has for it.
