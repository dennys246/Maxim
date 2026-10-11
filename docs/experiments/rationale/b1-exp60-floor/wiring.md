# B1 design review — wiring lens (2026-10-11, on the DRAFT prereg)

Preserved verbatim from the reviewer.

I read the code and made no edits. The only thing I ran was an offline FakeBridge tool count, done in the scratchpad.

## DO-NOT-BUILD

**DNB-1: Training never runs the loop, so the hook has to live inside `propose_via_substrate`.**
- `WaterTrial.train()` (`scripts/survival_world/water_trial.py`) calls `propose_via_substrate(...)` directly at 4 Hz and throws away the return value. No `run_agentic_loop` runs during training.
- If the hook sits in `substrate_tick` or the loop, Q_YOKED's learner is never called in training. Its table stays empty and it ties RANDOM by construction, which makes H2 vacuous.
- **Prereg change:** the hook replaces only the `nac.recommend_action(...)` call inside `substrate_proposal.propose_via_substrate`. Suggested signature: `propose_via_substrate(*, nac, agent_id, executor, situation_cue, min_confidence=None, sensor_encoder=None, policy: SubstratePolicy | None = None)`, with `SubstratePolicy.select(*, agent_id, available_tools: Sequence[str], clusters: Mapping[str,str] | None) -> Mapping | None`. It returns recommend_action's dict shape (`tool_name`, `params`, `confidence`, `reasoning`), so the code that builds the `LLMProposal` (clusters, margins, strategy) stays a single path.
- Thread it through `run_agentic_loop(substrate_policy=)` → `loop_substrate.substrate_tick` → `_loop_kwargs` (forwarded only when given, like `substrate_telemetry`) → `run_minecraft_aut` → `WaterTrial.loop_window/train`.
- **Must stay identical across arms:** the tool filter (introspection list plus `MAXIM_SUBSTRATE_TOOL_WHITELIST`, which must be asserted unset and stamped), the per-channel encode, `note_active_clusters`, `situation_cue`, `evaluate_failures` (the pain tick), the cadence, `_attach_live_situation`, the same-tool cap, `can_execute_action`, `execute_and_learn`, and telemetry.
- The policy gets no `nac` and no `drives`. That makes "the floor never reads Maxim's learning" a structural property, not a convention.
- `policy=None` must be byte-identical to today. `test_loop_substrate_characterization.py` stays green, and the change carries mypy, CHANGELOG and three-lens obligations.

**DNB-2: The reward is credited to the wrong action as drafted.**
- Pain publishes inside `propose_via_substrate` (from `evaluate_failures`, after the encode and before selection). It also publishes inside an executed affordance (`tool_bridge` calls `evaluate_failures`).
- So the pain seen on call t was caused by the state and action before it. "Booked with the tick's reward" would credit the action chosen after the pain.
- **Freeze the update order:** on select at t, read r_t = −max intensity of pain published since the select at t−1 (PainBus, allowlist {drive:health, drive:oxygen}), update Q(s_{t−1}, a_{t−1}), then choose a_t.
- Collect the reward through a per-seed PainBus subscriber. `WaterTrial._record_pain` / `self.signals` already exists. Do not use telemetry: it only exists inside loop windows, not in training's direct calls.

**DNB-3: Episode boundaries must be explicit, because surfacing cannot be seen from the tick.**
- The harness polls bridge state at 4 Hz and teleports on the first head-in-air read. The 2 Hz loop usually never encodes a surfaced state.
- The rescue teleport produces the same shore-cluster transition whether the episode ended in surfacing, saturating pain or the cap. Bootstrapping across it would credit all three alike.
- **Prereg change:** `policy.end_episode(reason ∈ {surfaced, pain, cap})` is called by the harness before every rescue.
  - Terminal update: r = 0 for surfaced; r = the saturating pain for pain; no terminal for cap (truncated).
  - The 4 shore recovery ticks after training and the shore warm-up run with the policy disarmed (it returns None).

**DNB-4: The arms are not idle on the shore.**
- FEAR's recommend_action returns None on the shore. RANDOM and Q always propose, so during the 1 s warm-up and during liveness they execute actions such as `move_to` and `mine_block`.
- `action_timeout_s` is 15 s and the probe cap is 4.3 s. An action started during warm-up can block the loop thread through the whole window, or fight the teleport.
- **Prereg change:** the policy is armed at `enter()` (t0) and disarmed at rescue. Loop liveness runs disarmed, which is the same measurement as FEAR. Every call's `t` and `t_return` is recorded, and so are calls still in flight at t0.

## SHOULD-FIX

1. **"About 15 selectable actions" is wrong.** The live registry minus the introspection tools is **26**: 18 `read_*`/`sense_*` sensor tools plus 8 motor actions (`move_to`, `turn`, `mine_block`, `place_block`, `eat`, `attack_nearest`, `flee`, `escape_water`). That gives 1 − (25/26)^8 ≈ 0.27 before any action duration. Re-derive the H1 expectation. Also freeze a parameter rule: `move_to`, `turn`, `mine_block` and `place_block` need params. Either use `{}` (and record how the failure behaves) or a frozen seeded distribution. The cap and the Q keys should hash the same params.
2. **The selectable set is defined by the proposer's filter, not by "AUTONOMOUS autonomy".** At AUTONOMOUS, `can_execute_action` passes everything except safety blockers. Name the set as "`propose_via_substrate`'s `available_tools`" and stamp it per seed.
3. **State key.** Use the world cluster this tick's proposer already encoded (`clusters["world"]`). Do not call `_encode_current_clusters` a second time: a second `encode_sensors` call adds EC observations and changes first-touch ids relative to FEAR.
4. **NAc still books learning in the floor arms, and that is acceptable if it is disclosed.** `execute_and_learn` → `record_outcome` books causal links, the `(cluster, tool)` reward bias and negative links from ToolPainBridge. Wire-4 books fear in the Q_YOKED and Q_EXPLORE training. None of it reaches selection, because RANDOM and Q never call `recommend_action` or `anticipatory_threat_need`, and the Minecraft harness builds no `FearGatedExecutor`.
   - Keep Wire-4 attached: detaching it changes nothing the policy reads and only adds one more difference between arms.
   - Record `positive_escape_links` and the fear dump per arm to show these values are inert.
5. **Learning during probes.** Freeze whether Q updates during the probes. I recommend OFF. FEAR does learn in probes (the positive-link caveat), so pre-register first-placement P(surface) as a secondary that compares the arms fairly.
6. **Q_EXPLORE needs a new training method.** It must run real loop windows (2 Hz, executed actions) from the same `loop_window` machinery, not `train()`'s direct 4 Hz calls. State that the cadence differs: Q_YOKED trains at 4 Hz, the probes run at 2 Hz.
7. **Liveness ladder per arm**, before any data:
   - (a) Inherit every Exp 60 preflight: roster, cadence ≤ 0.15 s, liveness ≥ 4 ticks in 3 s (disarmed), clusters distinct, backend actuation, no positive escape link.
   - (b) A new one-seed armed diagnostic per new arm: telemetry proposals equal the policy's own log of its picks equal the executor-spy calls (an identity check, not counts). For RANDOM, show at least one call outside `escape_water`.
   - (c) Q_YOKED training must show updates equal to propose calls in water, and nonzero-reward updates equal to the oxygen signals (about 20 per seed).
   - (d) A unit guard: a scripted always-`escape_water` policy through `run_agentic_loop` on `FakeBridgeServer(events=False)` executes `escape_water`, and goes RED when the hook wiring in `substrate_tick` or `propose_via_substrate` is deleted.
8. **Disclose the same-tool cap.** A greedy (ε = 0) Q policy repeats its argmax, so the cap blanks one tick in six for it exactly as it does for FEAR. It is identical across arms, but disclose it.

## NIT

- Seed the policy RNG per seed × arm, and stamp it along with the policy class and its frozen α, γ, ε.
- The cheaper alternative with no `src/` change is to replace `aut.bio.nac.recommend_action` on the instance. Name it as considered and rejected (an untyped monkeypatch of a bio class, D43 opacity) to satisfy the front-gate scope check.

## Verified

- `loop_substrate.substrate_tick` is the only loop caller and goes through `_sp.propose_via_substrate`.
- `_substrate_tick_due` is the shared wake/branch predicate.
- `_loop_kwargs` uses AUTONOMOUS.
- §4 runs the cap, then `can_execute_action`, then `execute_and_learn`. The cap is content-hashed, with `_MAX_CONSECUTIVE_SAME_TOOL`.
- `train()` is propose-only by direct call.
- The Wire-4 subscriber reads `nac.active_clusters` noted before the pain tick.
- `tool_bridge` also publishes pain on execution.
- The live tool roster is 26 tools.
