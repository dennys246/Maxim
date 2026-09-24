# Wiring / consumer-audit lens — nociception_layer.md

**Verdict: SHOULD-FIX before revive.** The thesis (pain needs one vocabulary) holds up, and so does the F1 core. But the counts are wrong, the consumer table leaves out 3 consumers and misstates 4 rules, F1 is wider than the plan says, F2 names the wrong overlap, and step 3's "same set" claim does not hold. Checked on `main` @ `d298d4f6`.

## Claims

- **Step 1 "SHIPPED in 2S-c": NOT ON MAIN.** `PainKind`, `classify_pain` and `failure_pain_kind` don't appear anywhere in `src/`, and `feat/memory-2sc-survival-signals` is identical to main. Say "ships with 2S-c" until it merges.
- **"PainSignal constructed in 4 modules": WRONG, it is 6.** The plan's 4, plus `proprioception/perceived_pain.py::PerceivedPainAssessor` (2 sites, ANTICIPATED) and `proprioception/pain.py::PainDetector`, which the plan lists separately. `pain_bus.py`'s one site is the lossy rebuild `_reaction_to_pain_signal`, not a producer.
- **"Reaction(kind="pain") in 9 modules": WRONG, it is 7.** `reactions/bus.py` and `pain_bus.py` only mention it in docstrings; `pain_bus.py` converts through `compat.pain_signal_to_reaction`. Also missed: the scripts `sem_learning_loop_poc.py` and `valence_annotation_poc.py`.
- **Two buses / `_bridge_reaction_to_pain_subs`: VERIFIED, with an omission.** The rebuilt PainSignal's context is `{}` or `{entity_path}` only. That drops `agent_id`, `source` and `failure_mode`, so which consumers still fire depends on which keys each one needs (see F1).
- **Memory subscriber, "nociception only": WRONG.** `create_pain_memory_subscriber` captures an episode for EVERY pain at intensity ≥ 0.4. Only the `pain` encoding field is nociceptive (`_pain_encoding`). It is not `_human_is_driving`-gated (on purpose).
- **Percept-valence subscriber: INCOMPLETE.** Threshold 0.3 plus `_human_is_driving`, and it requires non-empty `agent_id`, entity name/type and `failure_mode`. So Reactions arriving via the bridge never reach it.
- **NAc subscriber: INCOMPLETE.** Threshold 0.3 plus `_human_is_driving`, with no `agent_id` requirement. Bridged Reactions (including anticipated ones) DO reach `record_outcome_full`.
- **Cluster-fear subscriber: WRONG on where the rule lives.** `create_pain_cluster_fear_subscriber` passes any `failure_mode` through; the allowlist is enforced in `NAc.record_cluster_fear` against `NACConfig.cluster_fear_failure_modes`, which is configurable. The value `{drive:health, drive:oxygen}` is VERIFIED (`decisions/nac.py::DEFAULT_CLUSTER_FEAR_FAILURE_MODES`). It is also consumed by `hivemind/bundle.py` and `hivemind/ingest.py`, a wire boundary the plan does not mention.
- **Reward distributor, "no kind filter": VERIFIED, and wider than stated.** It takes every Reaction of every kind, positive or negative, with `agent_id ∉ {None, WORLD_AGENT_ID}`.
- **`ToolPainBridge`: INCOMPLETE.** It takes TOOL_FAILURE/TIMEOUT/INVALID_INPUT, or `source=="embodiment"` when no tool is pending. It has no intensity threshold and no `_human_is_driving` gate. It also writes to SCN (`register`) and the distributor (`record_event`), so it is a temporal-credit consumer too.
- **`PainCircuitBridge`, "motion pain": WRONG.** `bridges/pain_bridge.py::PainCircuitBridge._on_pain` has no type filter: it takes ANY pain while a movement is pending (≤ 5 s). Pending is set only by `embodied_runtime/movement.py` (`record_action_start`), so it is inert in sim.

## Missed consumers

1. **`hippocampus.capture_reaction`** (`runtime/bio_stack.py`, `subscribe_all`): every Reaction goes into the pending episode, and `memory/episode.py::finalize` sums them into `net_valence`. Anticipated pain lowers episode valence. This is a second F1-shaped defect.
2. **`pain_bus.py::_sim_log_reaction`** (`subscribe_all`, telemetry).
3. **`NAc.record_cluster_fear` plus the hivemind ingest filter**: the actual allowlist consumers (above).
4. **Harness subscribers:** `scripts/survival_world/water_trial.py` (`_record_pain`, the Exp 62 instrument) and `scripts/sem_learning_loop_poc.py`. Step 3's migration must keep these byte-identical, or it silently changes an EARNED instrument.
5. **Channel-1 direct attribution** (`ToolPainBridge.record_tool_embodiment_failure`, not on the bus): this is `failure_pain_kind`'s consumer and needs a row in the table.

Nothing else in `src/` subscribes: DefaultNetwork only by way of `PainCircuitBridge`, and nothing in the fear gate, SCN or reactions.

## F1: LIVE, on the `maxim --sim` orchestrator AUT path only

- **Construction.** `simulation/orchestrator.py` builds `PerceivedPainAssessor(nac=aut_nac, pain_bus=aut_pain_bus, agent_id=_aut_agent_id)`, where `_aut_agent_id` is `"sim_aut"` or the adopted agent's id. It is wired both as `AnticipatoryPainExecutor` and as `bridge.percept_anxiety_hook`.
- **Why the reward fires.** `aut_pain_bus` is passed to `create_full_agent` → `build_bio_stack(pain_bus=...)`, which subscribes `_distribute_reward_from_reaction` on `aut_pain_bus.reaction_bus`. The assessor publishes `Reaction(valence=NEGATIVE, agent_id="sim_aut")` directly on that bus. The id is not `WORLD_AGENT_ID` and not `None`, so the handler calls `distributor.distribute("sim_aut", -intensity)`.
- **Refractory.** The ReactionBus gate (0.5 s on `pain:perceived_pain:anticipated`) only rate-limits; it doesn't gate.
- **Not live** on `simulation/minecraft_harness.py`, `embodied_runtime/agentic_runtime.py` or `bio_stack.build_default` (no assessor there).
- **Extra anticipated → learning paths the plan missed:**
  - (a) episode `net_valence`, via `capture_reaction`;
  - (b) the lossy bridge → `create_pain_nac_subscriber` records an outcome `pain:unknown:unknown:ANTICIPATED`. Its context is empty, so it links rarely.

**Step 4 scope risk.** The distributor also pays out drive pain. `embodiment/body.py::_publish_drive_pain` stamps `agent_id`, and `minecraft_harness.py` says outright that the survival loop depends on body pain reaching the distributor. "Only NOCICEPTIVE, owner decides DRIVE" is therefore a survival-loop reward change. Step 4's ledger must add Exp 60/61/62 and R3, not just 52/56.

## F2: the named pair is effectively disjoint; the real overlap is elsewhere

**Both bridges do share a bus:**
- `--sim --embodiment`: `ToolPainBridge` via `build_executor`, `PainCircuitBridge` via DN on `aut_pain_bus`.
- Reachy: both on `bio.pain_bus`.

**Their effective inputs barely intersect.** `PainCircuitBridge` needs a pending movement, which never happens in sim. `ToolPainBridge` ignores motion PainTypes. No double attribution between the two was found.

**The overlap F2 misses** is `create_pain_nac_subscriber` plus `ToolPainBridge._on_embodiment_pain`. Both call `nac.record_outcome_full` on the same out-of-band embodiment pain, in the Minecraft harness and in `--sim --embodiment`, with different signatures (`pain:embodiment:…` vs `embodiment:…`). NAc removes linked pending events (`NAc.record_outcome_full`, the `remaining_events` rebuild). The subscriber is registered first (`build_pain_bus`), so the bridge's call mostly finds nothing to link: **order-dependent, not measured.** On Reachy, motion pain likewise reaches both `create_pain_nac_subscriber` and `PainCircuitBridge`.

Step 6 should name the one NAc path among **three** writers, not two.

## Allowlist as kinds: NOT lossless

`drive:health` classifies as NOCICEPTIVE (`_TISSUE_DAMAGE_DRIVES`) and `drive:oxygen` as DRIVE. A kind filter `{NOCICEPTIVE, DRIVE}` would let in every nociceptive failure mode and every drive, hunger included. That reopens wiring W-5, where hunger pain writes fear onto the lit/dining clusters.

"`{NOCICEPTIVE(health), DRIVE(oxygen)}` — same set" is only true if the filter is on **(kind, failure_mode)** pairs. The `failure_mode` allowlist in `NAc.record_cluster_fear` and at hivemind ingest has to stay the authority; kind can only be a coarse pre-filter. Step 3 should say so.
