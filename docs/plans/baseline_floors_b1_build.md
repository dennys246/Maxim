# Baseline floors B1 — build plan v2 (reviewed; for the owner before code)

> **SHIPPED (2026-10-24 plans archive; it stays at this path because `src/` or the code's docs cite it):** PR-A #1236, PR-B #1238, PR-C #1245, fix #1258 and the run #1283 shipped; it is B1's build record.

> **Build plan v2, 2026-10-11**, for [../experiments/b1_exp60_floor_prereg.md](../experiments/b1_exp60_floor_prereg.md)
> (v3: its §Frozen implementation rules come from this review). v1 had a four-lens parallel review (Executor,
> Architecture, Wire integrity, Experiment integrity); every lens returned DO-NOT-BUILD items, all folded here
> (§Review record). **BUILT and RAN (2026-10-18 status):** PR-A #1236, PR-B #1238, PR-C #1245, CLI fix #1258;
> the run (#1283) returned `FLOOR_MATCHED_Q`, so T1-13 is PARTIAL (narrow: floor matched). This plan is the
> build record.

## What exists (the build fits into it)

- `exp60_run.py::_run` is one monolithic function: one arm per invocation, a prefix of the frozen seeds, one
  `run_id` per invocation. It is a T1-13 regression guard and is NOT edited.
- `water_trial.py::WaterTrial` is shared by Exp 60, 61, 62, R3, `exp62_precheck` and `test_water_trial_smoke`.
  Its executor spy already records each call's `t` and `t_return`. `train()` calls `propose_via_substrate`
  directly at 4 Hz (no loop). `loop_window` runs the loop at `target_hz = 4.0`, proposing at 2 Hz
  (`llm_submit_interval = 0.5`).
- `propose_via_substrate` (function-length pin 250) runs the filter, encode, `note_active_clusters`, situation
  cue and pain tick, then one `nac.recommend_action(...)` call, then builds the proposal. `min_confidence` is
  applied only inside `recommend_action`.
- Required affordance params are validated in `Tool.run` before `execute` (legacy schema: `move_to {x, z}`,
  `turn {degrees}`, `mine_block`/`place_block {x, y, z}` are required), so a `{}` call fails with "Missing
  required input" before the bridge. (The Executor lens's contrary finding was checked and is wrong.)
- Evidence: `seeds_per_run_arm` refuses a `run_id` that spans more than one arm; a kind with a complete-run rule
  supports exactly one scope; a new rule kind needs `COMPLETE_RULES` + a FROZEN pin test.
- The 1.4 campaign core (Phase 0 item 2) is planned, not built; `exp62_run.py` already exists, so B1 is the
  fifth survival harness.

## PR split

### PR-A — the `src/` hook (generic; capability, no non-test caller until PR-B)

- `runtime/substrate_proposal.py`: `SubstratePolicy` Protocol,
  `select(*, agent_id, available_tools: Sequence[str], clusters: Mapping[str, str] | None, observation: Mapping[str, float]) -> Mapping[str, Any] | None`.
  `observation` is a read-only copy of `channel_values["world"]`, the exact values the encode used (no fear term,
  no NAc, no interoception). Returns `recommend_action`'s dict shape or `None`; `None` means idle, never a
  fallback to `recommend_action`.
- `propose_via_substrate(..., policy=None)`: the selection is extracted into a small helper so the pinned
  function does not grow; with a policy, the helper calls `policy.select` instead of `recommend_action`. The hook
  VALIDATES the return and raises a typed error on violation: `tool_name ∈ available_tools`, `params` a Mapping,
  `confidence` absent or finite. A policy bypasses `min_confidence` by design (pinned in a test). The proposal
  carries `strategy_used="substrate-primary"` and the policy's name in `reasoning`.
- Threaded: `run_agentic_loop(substrate_policy=None)` → `loop_substrate.substrate_tick(substrate_policy=…)`
  (required keyword, like the slice's others) → `minecraft_harness._loop_kwargs` / `run_minecraft_aut`
  (forwarded only when given). `run_agentic_loop`'s signature grows, so a function-length exception entry with a
  reason is added.
- `policy=None` is byte-identical to today.
- Tests: the characterization tests unchanged; an always-`escape_water` policy through `run_agentic_loop` on
  `FakeBridgeServer(events=False)` executes it (RED when the threading or the helper call is deleted); the
  validation errors; `min_confidence` bypass; `None` is idle. mypy over the composition files, `[Unreleased]`,
  three-lens review.
- Trigger walk recorded in the PR: T1-6's "substrate exploration-policy change" fires by its wording; discharged
  structurally (`policy=None` byte-identical; `recommend_action` and fear untouched), with the `None`-path guard.

### PR-B — the floor policies and the driver (`scripts/`)

- `scripts/survival_world/floor_policies.py` (arms live in the harness; they move to `src/` only when E1–E3
  consume them): `RandomPolicy`, `TabularQPolicy` (keys, cost, update order, tie-break, terminals per the
  prereg's frozen rules), `ArmedPolicy` (idle unless armed; records a params violation for the harness to refuse
  on, never raises inside the loop). One lock guards `select`, `end_episode`, `arm` and `disarm`, which run on
  different threads. Q credits the EXECUTED action: it reconciles against the trial's executor spy and skips the
  update for a pick that never executed (same-tool cap, a pending proposal at stop), recording each drop.
- `WaterTrial` changes are dormant: one attribute `substrate_policy = None` that `loop_window` forwards only when
  not None, and an optional `before_rescue=` callback (default None) threaded through `loop_window`, `placement`
  and `probe`. Exp 60/61/62/R3 rows are unchanged. Everything else lives in `FloorWaterTrial(WaterTrial)` in the B1
  module: arming just before the teleport (t_arm stamped), `end_episode` in every placement and episode in the
  frozen order (disarm → `end_episode` with the final interval's cost → RCON rescue → loop stop → bridge `stop`),
  the armed shore roam, and `train_executed(policy)` (K = 10 executed episodes through real loop windows,
  harness-clock terminals, cap = measured damage onset − 3 s).
- `scripts/survival_world/b1_floor_run.py`: the per-seed preflight call sequence is COPIED from `exp60_run._run`
  (about 15 `WaterTrial` calls; a test pins its order against Exp 60's); it imports only `FROZEN` and pure helpers
  (`load_apparatus_record`, `committed_record_problem`, `anchor_record_mismatch`, `min_pain_edge_s`,
  `exact_permutation_p`). Its own `FROZEN` (seeds 11–18, the arm order, every frozen threshold). One `run_id` PER
  ARM within a session plus a shared `session_id`, so `seeds_per_run_arm` works unchanged. Pain is read from
  `WaterTrial.signals` with a cursor (no second subscriber; each pain counted once). Telemetry ticks are kept in
  the row (the persistence dir is deleted). Provenance as `exp60_run` does (`in_process_code_provenance`,
  `preflight_gated_record`, `stamp_harness_row`, `record_kind`). Data names `b1_*` only (`exp60_*`/`rerun_exp60_*`
  would fall under the frozen Exp 60 prereg).
- Apparatus re-verify: `setup_world.water_classroom_verifications(geom)` (10 `execute if block` checks, < 1 s),
  rebuilt from stamped build params, plus `check_surface_cell_air`. "Shell faces" is dropped: with `params={}`
  no action can mine or place.
- Rig check through `maxim.utils.http`: the leader proxy's `/v1/debug/status` (`infer_lane.in_flight == 0`) and
  `/v1/debug/last-requests` (no request in 2 min); a missing server passes and is recorded; server TPS by RCON
  `tick query`, refused below 19.5. No auth header or URL in the data.
- Tests: the policies (update order, interval cost, executed-action credit, terminals, truncation, tie-break not by
  name, armed/disarmed, lock), the Williams order, the verdict on synthetic rows (incl. INCOMPLETE and the FEAR
  sanity gate), `train_executed` terminals, one offline test that every `minecraft_player` affordance with
  required params fails before `execute` on `{}`. A smoke that runs one arm-seed of every arm end to end on
  `scripted_water.ScriptedWaterBridge` + `ScriptedWaterControl` (which model oxygen, surfacing and `flee`
  failing; `ScriptedWaterControl` gains `execute if block` replies), so Q learning is shown offline, not only
  wiring.

### As built (PR-B, 2026-10-11) — changes from this plan

- Data: one CAMPAIGN per file (`campaign_id`), a `session` header (`stamp_harness_header`, never counted) and one
  `row` per arm-seed ATTEMPT, each with its arm's `run_id`, `attempt` and `refusal_cause`. Rule 6's rerun puts a
  refused row and a clean row under one key, which `seeds_per_run_arm` refuses ("repeats an (arm, seed)"), so the
  complete-run rule is PR-C's decision (below), not `seeds_per_run_arm` unchanged.
- The learner's constants changed after a toy model (owner decision OD5: γ = 1, Q₀ = −2), and the scripted smoke
  found two credit defects (prereg rules 16–17). A delta design review (confounding + bio-faithful,
  `rationale/b1-exp60-floor/*-v4.md`) ran on v4 and is folded.
- The rig check waits (≤ 10 min) for a quiet rig before an attempt, so rule 11 never spends an attempt; transport
  faults (`OSError`) are recorded instrument refusals; programming errors crash (resume with `--resume`).

### PR-C as built (2026-10-11): the evidence half; the freeze is its own step

Design: [archive/baseline_floors_b1_evidence.md](archive/baseline_floors_b1_evidence.md) (five adversarial rounds, owner decisions
D1–D2). Built:
- the `b1_verdict` pass-table entry (PARTIAL at scope `narrow` from a floor-matched verdict only);
- the `b1_campaign` complete-run rule (`scripts/_evidence_records.py`), on top of `_campaign_rule`;
- the BF2 obligation lint (`scripts/lint_bf2_obligation.py`, in the CI lint job), sharing the gate's verdict-data
  binding helpers and a `build_ctx` extracted from `lint_evidence_gate.gate()`;
- the harness's `attempt_start` headers and resume closing rows (`unrecorded`, never rerun);
- the pin test, and backlog rows M48–M50.

**The prereg freeze is a separate, later PR (done 2026-10-13, FROZEN v5, after the seed 99 diagnostic):** it follows the seed 99 rig diagnostic, so anything the diagnostic finds
changes the rules before they freeze, not by amendment.

### PR-C — freeze + evidence (the original plan; superseded by the section above)

- **Complete-run rule (decide with the approach note).** A new `_campaign_rule("seed", "row", allowed={"row",
  "session"}, required={"session"})` instance fits the data. Open: (a) its frozen sets name all five arms × 11–18
  (40 keys) though the verdict gates on three; (b) `_campaign_rule` refuses only two CLEAN rows per key, so the
  rerun discipline (attempt ≤ 2, a rerun only after an instrument-cause refusal, one `run_id` per arm) lives only
  in `compute_verdict` — the recommended strict option puts it in the rule; (c) keep the verdict's scope
  `{"all_rows": True}` with no `campaign_id`.

- The prereg frozen on main before the first gated row.
- `evidence_pass_table.json` gains `b1_verdict`: rows `T1-13`, `scopes: ["narrow"]`, targets `PARTIAL` only, with
  the verdict values frozen in the prereg (`FLOOR_MATCHED_Q`, `FLOOR_MATCHED_CHANCE`; `FLOORS_BEATEN` moves no
  status and becomes a "floor: beaten" annotation on the EARNED row). The status change re-dates the row, so it
  needs this newly cited, established, prereg-PASS verdict. The complete-run rule (above), with its FROZEN pin
  test. A pass-table edit is a gate change: its own
  adversarial approach note first.
- Roadmap 1.4 item 2 gains one line: B1 is a consumer the extraction absorbs, and `b1_floor_run verdict` over
  committed rows joins its acceptance check (i).

### Then (operator)

1. One armed diagnostic per new arm on the rig, seed 99 (never pooled): the policy log, telemetry proposals and
   executor calls identical except recorded cap drops; RANDOM shows calls other than `escape_water`; Q shows one
   update per executed pick and nonzero costs in water; every stamped tool called once with `{}` through the
   executor, shore and submerged, with outcome, duration and connection survival recorded.
2. PR-C merged.
3. The run (≈ 2.5–3 h, no LLM consumer), the verdict, a merge-commit data PR, §Outcome, the T1-13 change.

## Owner decisions (taken 2026-10-11, both as recommended)

- **BD1. Name: the `b1` token is kept** (B1 is a floor on an existing claim, `r2_`/`r3_` set the precedent, and
  the lint has no collision), not `exp64`.
- **BD2. B1 is not Exp 60's trigger re-run** (no ABLATED arm and different seeds, so the gate could not use it);
  no ABLATED arm is added.

## Review record (v1 → v2, 2026-10-11)

- Evidence (Architecture, Wire): one `run_id` per session fails `seeds_per_run_arm` → one per arm + `session_id`;
  a ruled kind supports one scope → `b1_verdict` supports PARTIAL only; the date change needs new support.
- No Exp 60 seed body to import (Architecture, Executor, Wire) → copied call sequence, pure-helper imports only.
- The policy cannot read the cost or RAW key without a second racing read (Executor, Architecture, Wire,
  Experiment) → `observation` = the encoder's own world values.
- Interval cost (Executor): one reading per select rewards slow actions → time-integrated.
- Silent policy failure (Executor, Experiment) → a loop thread that exits before `stop` refuses; violations are
  recorded and refused by the harness.
- Executed-action credit, thread safety, episode ends in probes (Executor, Wire, Experiment) → folded.
- Tie-break, usable-episode rule, unclean first placement, rerun rule, FEAR sanity gate, arming before the
  teleport, armed roam, INCOMPLETE scope, Williams order, frozen thresholds (Experiment) → the prereg's frozen
  rules.
- `WaterTrial` blast radius (Architecture) → a subclass plus one dormant attribute and one optional callback.
- Function-length pins (Wire) → extracted helper + one exception entry.
- Smoke harness (Executor) → `ScriptedWaterBridge`, not `FakeBridgeServer`.
- Rejected: the Executor lens's "params `{}` reach the bridge" (checked against `Tool.run`; kept as a test and a rig
  check anyway).
