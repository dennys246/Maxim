# GL1 design review — WIRING lens (grounding.md + autonomic_layer.md)

Reviewer lens: real consumers and the real credit path (D43); right encoding and seams; no
hand-composed shortcut that passes while the loop fails. Read at worktree `c80d619b` (origin/main
with #1167 + GL0 #1186). Read: `docs/wiring/` (README, substrate-learning-channels,
harness-loop-must-be-proven-live, engram-formation), the two plans, DECISIONS 2026-10-07 (G1–G8,
not re-litigated), the GL2a owner decisions of 2026-10-09 as given in the brief. Every code claim
below was read at `file::symbol`; one was probed at runtime (D3); anything else is marked UNVERIFIED.

**Verdict: 3 DO-NOT-BUILD / 10 SHOULD-FIX / 4 NIT.** All three DO-NOT-BUILDs are on GL2a. Each
can be folded with a plan edit before code. None re-opens G1–G8. D1 asks the owner to widen G1's
`executor.py` scope wording from one method to two, inside the same file, which the owner may
prefer to treat as a decision. Recommended reading order: D1, D3, D2.

---

## DO-NOT-BUILD

### D1 — GL2a: the tool-path record's window is undefined and self-contradictory, and under the strict default it still stamps narrator writes `experienced`

**Stage:** GL2a (tool path, §3.1, §3.1.4, §5.2 guards).

**The issue.** The plan never says over what window, and over which drives, the tool-path
`drive_delta` / `relief` / `harm` / `satiated` are computed. Every reading it allows breaks a
stated rule.

1. **The guard contradicts the derivation rule.** §3.1 says the guard pins
   `relief == max(drive_relief)`. §3.1.4 says `drive_delta` must not come from
   `side_effects["drive_progress_by_drive"]` "while #1161 keeps that side effect blind". But
   `ToolOutput.drive_relief` comes from that very side effect
   (`runtime/executor.py::Executor._drive_relief` reads `drive_progress_by_drive`). That side effect
   reads bare names (`embodiment/tool_bridge.py::_drive_progress_by_drive`:
   `metrics.get(name)` on `body.vital_metrics`), so it never contains `arms.thermal`. A record
   that "never omits `arms.thermal`" therefore fails `relief == max(drive_relief)` on exactly the
   burn and warm invocations the gate scripts (`warm_self` ×2, `touch`).
2. **"The existing trio is filled from the same computation" would break record-only.** If the
   trio is refilled to match, `ToolOutput.drive_relief` changes. Through
   `runtime/bio_integration.py::capture_episodic_memory` it lands in `EncodingSignals.drive_relief`,
   and `memory/encoding.py::encoding_tag` reads it (relief channel + pressure weight). That moves
   `storage_strength` and the retro-tag inputs, which is a memory-strength behaviour change, not a
   record.
3. **Any honest derivation needs state that `_stamp_invocation` does not have.** A signed delta
   needs RAW before-values. Today the executor keeps only the unsigned pressure
   (`Executor._drive_pressure_snapshot`, taken in `Executor._run_started` before `tool.run`), and
   homeostatic pressure cannot be inverted back to a signed value. A satiation also needs the latch
   state before the call. Both snapshots have to be taken in `_run_started`, but G1's exempt-set
   wording is "`runtime/executor.py` (`Executor._stamp_invocation` only)". GL4 S1's `ActionContext`
   has the same scope.
4. **A body-wide before/after window breaks G6 under the strict default.** A naive whole-body
   `_run_started` → `_stamp_invocation` window takes in every write made while `tool.run` runs:
   - In `--sim`, the narrator tools write the AUT body from the orchestrator thread while the AUT
     executor runs on `sim.aut` (`simulation/orchestrator.py`: `aut_thread = threading.Thread(...,
     name="sim.aut")`; `OrchestratorActorTool` / `DamageComponentTool` / `SetEntitySensorTool` on
     `orch_registry`).
   - On Minecraft, `MinecraftSyncPump._run` writes the body's sensor VALUES on its own `mc-sync-*`
     thread through `backend.sync_world_sensors()`.

   So a narrator write that lands mid-invocation reaches the AUT's tool-path record as an
   `experienced`, tool-caused delta. The strict default exists to prevent exactly that. Satiation
   detected by diffing the latch has a second hole: `Embodiment.evaluate_failures` also pops the
   latch when a sensor is unreadable (`ent.drive_breach_severity.pop(ds_name, None)` in the
   `current is None` branch), and a diff would read that as satiation.

**Why it matters.** GL2a's whole value is an honest target for the forward model and an honest
input for GL2c's relief store. A record whose window takes in other threads' writes is positive
bystander credit: the mirror of the Exp 42 pollution that B8 exists to prevent, now landing in the
record that GL2c turns into reward.

**Concrete fix (plan text before code):**
- Define the tool-path consequence as **action-scoped**: only the drives named in the affordance's
  own declared effect (the same key set `_drive_progress_by_drive` iterates). Read them with the
  #1125 resolver (`embodiment/sem.py::_read_sensor_value`). Body-wide change not in that set is NOT
  on the tool-path record; it belongs to the out-of-band producer, which comes after the fence with
  the narrated scope.
- Amend G1's executor wording to "`Executor._run_started` (raw before-snapshot) +
  `Executor._stamp_invocation`". Same file. Say so in DECISIONS.
- Keep the existing trio exactly as computed today; never refill it.
- Replace the guard with `relief == max(drive_relief)` over the drives present in both, and add a
  separate `xfail(strict=True)` gate for the qualified drives that the trio misses until #1161
  lands.
- Detect satiation only at the two `elif cleared:` sites, recorded by the body for the calling
  thread's evaluation inside this invocation. Never detect it from a latch diff, and never from the
  unreadable-sensor pop.

### D2 — GL2a: "seq persists per agent from GL2a" cannot be wired inside the exempt set, and it resumes from a store that loses traces

**Stage:** GL2a (§3.1.2 "seq persists…", §5.2 build + guard "resumed past the saved maximum after
a save/load round trip"); inherited by GL3.B3's handover and GL4's one-join rule.

**The issue.**
1. **The load seams are outside the exempt set, and there are two of them.**
   - `runtime/bio_stack.py::build_bio_stack` restores the Hippocampus through
     `hippocampus.load_with_recovery()` before `build_executor` constructs the `Embodiment`
     (`runtime/bootstrap.py`: `Embodiment(entity, pain_bus=pain_bus, agent_id=agent_id)`).
   - On `--resume-sim`, `simulation/orchestrator.py::_restore_aut_from_session` calls
     `aut_hippocampus.load(p)` AFTER the AUT executor and embodiment already exist
     (`aut_executor = _aut_instance.executor` precedes the `_restore_aut_from_session(` call).

   A sequencer seeded when the `Embodiment` is constructed misses the second path entirely. Both
   seams sit in fenced files (`bio_stack.py`, `bootstrap.py`, `orchestrator.py`), and none is in
   G1's exempt list. As scoped, GL2a either ships the resume uncalled (capability, not a fix) or
   edits outside the fence.
2. **The resume source is lossy.** "Resume past the max seq found in
   `EncodingSignals.extra["interoception"]`" assumes every minted pid is persisted there. It is not:
   - the capture queue drops its oldest entry when full
     (`Hippocampus.capture_from_loop_async`, `queue.Full` branch);
   - the store evicts at capacity (`Hippocampus._evict_one`) and compresses at sleep;
   - many executor paths never reach `capture_loop_action`: the PLANNING-approved path
     (`runtime/agent_loop.py`, "CHECK FOR APPROVED PROPOSALS"), parallel actions and retries
     (`runtime/tool_dispatch.py`), and harness preflights that call `executor.execute` directly;
   - write-but-don't-read agents (`load_persisted=False`) never resume.

   This is harmless while the Hippocampus is the only store that holds a pid. It stops being
   harmless at GL4 S1 (pid on `ActionContext`, `extra["context"]`, Cerebellum payload `"1.2"`) and
   at GL3: a resumed seq then collides with pids persisted in other stores, and "one join rule
   everywhere" breaks silently.
3. **The proposed guard is a hand-composed shortcut.** A unit round-trip (build a store, save,
   load, seed the sequencer) passes while both production seams stay unwired. That is the D43
   family.

**Concrete fix (choose one in the plan):**
- **(a)** The sequencer persists its OWN high-water mark (e.g. a top-level key in an existing
  per-agent persisted file that already carries `_format_version`), written at every save.
  Resume = max(own HWM, max pid seq seen in any pid-bearing store). Name both seams now,
  `build_bio_stack` (persistence_dir) and `_restore_aut_from_session` (`--resume-sim`), and put
  them in the exempt set or move resume to after the fence. The gate test drives both real load
  paths (the `test_water_trial_smoke.py` style, through `run_minecraft_aut` / the orchestrator
  restore). It must not hand-seed the sequencer.
- **(b)** GL2a ships pids unique within a session only, says so on the type and in the trace, and
  the persisted resume lands after the fence, before GL4 S1 persists a pid anywhere. GL4 S1's
  dependency then becomes "GL2a + the resume stage", not "GL2a only".

Either way, T1-16 also fires through its "Hippocampus save/restore or the resume path" clause if
the resume touches the Hippocampus load or `simulation/report.py::RESUME_STORES` (S9).

### D3 — GL2a: "no reader acts on it" is false. Every captured action stores the `ToolOutput` repr, and the substring recall path reads it

**Stage:** GL2a (§3.1.4 "Consumers in GL2a … No reader acts"; §5.2 deletion probe "changes only
the trace and the extra key"; the T1-16 structural discharge "`_rank_by_relevance` and
`_query_hippocampus` read no `extra` key").

**The issue.** The plan puts the record on `ToolOutput` as a new field. `ToolOutput` is
`@dataclass(slots=True, frozen=True)` with no `to_dict`, and every field is in its default repr.
`Hippocampus.capture_from_loop` stores the `ToolOutput` object itself in `Outcome.result`.
`Outcome.to_dict` passes it through. `utils/atomic_io.py::atomic_write_json` serialises it with
`default=str`, so `hippocampus.json` holds the full repr string. `Hippocampus.search_by_content`
matches `query in str(value).lower()` over `_searchable_values`, which includes `outcome.result`.
That substring match is **Path 3 of `integration/bio_enrichment.py::_query_hippocampus`**, the
ranker T1-16 names. It is also what `tools/narrative.py`, `agents/exec_agent.py` (similar-memory
search) and `simulation/introspection.py` call.

**Probed** (this worktree, read-only, `PYTHONDONTWRITEBYTECODE=1`): a capture whose ToolOutput
carried `drive_relief=(('oxygen', 0.5),)` is returned by `search_by_content("oxygen")` and by
`search_by_content("relief")`. Its persisted `outcome.result` is the string
`"ToolOutput(success=True, output='ok', … drive_relief=(('oxygen', 0.5),), pain=None)"`.

An `interoceptive_outcome` field with its default repr would add `relief`, `harm`, `urgency`,
`nociception`, `satiated`, `experienced`, drive names (`oxygen`, `health`, `thermal`) and cause
nouns (`fire_pit`) to the searchable text of every captured action, in-session and after reload.
That changes recall. The deletion probe would show more than "the trace and the extra key", and
the proposed T1-16 discharge would be false.

**Pre-existing, file separately (no band-aid).** The 2b-ii / 2S-c stamps already leak this way:
every captured action's repr contains the field names `pain=` and `drive_relief=`, so
`search_by_content("pain")` (`simulation/introspection.py`'s pain-memory query) matches every
action memory. The root cause is that `Outcome.result` stores the `ToolOutput` object and it is
stringified at save. The proper fix is a declared projection at capture, which changes the
persisted shape and owes its own T1-16 walk. That is an owner decision, not a GL2a fold.

**Concrete fix for GL2a:**
- Declare `interoceptive_outcome: InteroceptiveOutcome | None = field(default=None, repr=False)`.
  GL4 S1's `ActionContext` gets the same.
- Add a guard: `str(ToolOutput)` is byte-identical with and without the record.
- Extend the T1-16 byte-identical ranking test to the substring path, with a query that would match
  a record token (e.g. `"relief"`, `"oxygen"`).
- Correct the plan's discharge wording.

---

## SHOULD-FIX

**S1 — GL2a: #1125 is merged; the plan's dependency text is stale.** #1164 (`ee234cb3`, "the drive
records read modulator drives through the one resolver") is on main.
`Executor._drive_pressure_snapshot` now reads through `embodiment/sem.py::_read_sensor_value`. The
plan still says "#1125 (OPEN)" (§1.1, §3.1.4, §5.2, §8 first row). §3.1.4 also cites
`tool_bridge.py::_resolve_sensor_slot`, but the resolver lives in
`embodiment/sem.py::_resolve_sensor_slot` / `_read_sensor_value`; tool_bridge only imports it. Fix:
drop the #1125 owner decision and keep #1161 as the only relief-side blindness (it is still open:
`_drive_progress_by_drive` reads `vital_metrics` by bare name).

**S2 — GL2a: under the strict default, the snapshot, queue, bound and `evaluate_failures` lock
have no producer.** The 2026-10-09 decision is tool-path-only. §5.2 still builds "the per-entity
snapshot, the RLock and the bounded `Embodiment.drain_outcomes()`… either way":
- the queue has no producer, so the "measured drop-oldest bound" measures 0;
- the two-thread test's "seq stays unique" is vacuous, because narrator calls mint nothing;
- an RLock around `evaluate_failures` changes how narrator-thread pain publication interleaves,
  with no GL2a consumer.

Fix: GL2a builds the sequencer with its own private lock. Tool-path minting runs wherever the AUT
executor runs, so it needs no `evaluate_failures` lock. The snapshot, queue, bound and
`evaluate_failures` lock move to the post-fence out-of-band stage. If they are kept, their bound
measurement and thread test must name the out-of-band producer they exercise.

**S3 — GL2a: when a pid is minted is unspecified.** `_stamp_invocation` also runs on the
inactive-scene, unregistered-tool and exception paths (`pressure_before is None` there). Specify:
mint only when the invocation reached `tool.run` AND `self.embodiment.agent_id` is non-empty.
Reword the guard "record present iff a body is attached" to "iff an agent-bound body is attached
and the tool ran".

Bodies with `agent_id == ""` mint nothing: `maxim.create.embodiment()` (`create.py`), the foundry
wrappers, and every `scripts/orient_substrate/*` probe (`Embodiment(root=...)`). Public-API users
therefore never get a record, which should be stated. Synthetic outputs from wrappers
(`runtime/fear_gate.py::FearGatedExecutor` returns a new `ToolOutput` on a block) carry no record.
That is correct, and should also be stated.

**S4 — GL2a: the falsifiable gate must run through the real loop capture.** "On a scripted cradle
sequence the records match a committed table" can be met by calling `executor.execute` directly.
That checks the stamp but not the deliverable: the pid and core in `extra["interoception"]` on the
captured trace. Drive it through `run_agentic_loop` → `tool_dispatch.execute_and_learn` →
`capture_loop_action`, as `tests/unit/test_water_trial_smoke.py` does ("Through the real loop, not
a hand-composed capture"). Read the record back from the Hippocampus.

**S5 — GL2a / Minecraft: world-owned drives change outside the invocation window.** On
`minecraft_player`, oxygen/health/food are world-owned (`live_world_set_sensors`;
`MinecraftWorldBackend` writes measured state, and a timeout books an UNKNOWN outcome). The latch
for oxygen clears at the next loop-thread `evaluate_failures` (`runtime/substrate_proposal.py`,
`runtime/loop_gates.py`), not inside `tool.run`, because `MinecraftSyncPump._run` never evaluates.
So on the survival world the tool-path record systematically misplaces relief and satiation for
`escape_water` / `eat`. Red gate (d) under the strict default can flip only on cradle-style drives
the affordance writes itself.

Fix: state this. Mark world-owned drives' tool-path deltas as window-bounded (or omit them). Note
that GL4 S0b's Minecraft graded-Δoxygen floor, and GL2c's "satiation at `escape_water`" argument,
depend on the post-fence out-of-band producer, not on GL2a.

**S6 — GL2b(iii): `evaluate_failures(*, cause=)` cannot gate the cause per sensor as specified.**
Channel 2 publishes inside `evaluate_failures` (`_publish_drive_pain` under the latch). But
`tool_bridge.py` computes B8's set (`_intrinsically_harmful_sensors`) AFTER `evaluate_failures()`
returns, in `ModulatorAffordanceTool.execute`. B8 depends only on the effect dicts and the specs,
so it can be computed first. The keyword must therefore carry the gated set, e.g.
`cause_sensors=frozenset(...)`, or a body-wide `cause` leaks onto lingering breaches. Note also
that PainBus's `(entity, failure_mode)` refractory ignores the cause: a second cause within 0.5 s
on the same mode is dropped (state it).

**S7 — GL2c option A/C: the distributor's eligible set is wider than `tool:<name>`.**
`TemporalCreditDistributor.distribute` credits every `(agent, node)` in `NAc._eligibility`. Its
writers include:
- `similarity/encoder.py` `LinguisticEncoder` paths: EC **text** nodes;
- `SensorEncoder`: **sensor-cluster** nodes;
- `ToolPainBridge` temporal events: `tool:<name>`.

(`anticipatory_pre_activate` has no production caller; verified.) Option A therefore pays positive
`_reward_bias` to text nodes, so E4's overreach set is **mandatory** under A/C, not conditional
("if any option ever lets…"). It also pays sensor nodes, whose `reward_bias` has no reader (#911)
but trips `donor_sanity_staged` / `_R3._boundary`. And it pays whatever bystander `tool:` keys are
still eligible, which is positive-direction B8 pollution that the pid-keyed channel-3 dedupe does
not cover. Fix: put this census in §3.5, and either gate satiation credit to the causing `tool:`
key, or record the eligible-set composition as an input to the joint review.

**S8 — GL2c option A/C: T1-16 is missing from the rows.** `hippocampus.capture_reaction` is
`subscribe_all` on the ReactionBus (`runtime/bio_stack.py`). It appends every Reaction to the
pending episode, so satiation Reactions change persisted episode valence/content, which is "the
memory record shape". Add it to §5.4 / §6. Also: the ReactionBus default refractory (0.5 s, keyed
`satiation:drive:<name>:satiation`) silently drops a second crossing of one drive inside the
window. State it.

**S9 — GL2a: T1-16 fires a second way if the resume touches the load path.** If D2's resume hooks
into the Hippocampus load or `_restore_aut_from_session` / `RESUME_STORES`, T1-16's "Hippocampus
save/restore or the resume path" clause fires by its letter, in addition to "the memory record
shape". The walk must cover both.

**S10 — GL5 / GL4 S0b: name the production entry point that produces each arm's records.** GL5's
cradle arm is "substrate-primary, no LLM in the action path". The existing cradle runs are:
- Exp 42 via `scripts/benchmark_exp42_preference.py` → `maxim --sim` subprocess (LLM narrator on
  the orchestrator thread, so D1 applies);
- the `scripts/orient_substrate/*` probes, which build `Embodiment(root=...)` with `agent_id ""`
  and so mint no record under the identity contract.

No runtime is named that yields pid-bearing, captured cradle records without an LLM. The risk is
a hand-composed harness that calls the factory directly (D43). GL4 S0b's "100 % joinable" floor
also requires capture of every counted invocation, and the PLANNING-approved, parallel and retry
paths do not capture (D2). Fix: name the runtime per arm, and count only invocations that reach
`capture_loop_action`.

---

## NIT

**N1 — §1.2a census wording.** "The Reactions put straight onto ReactionBus … never become a
`PainSignal`, so those subscribers never see them" is true for the `api.py` agent-bus subscribers.
But PainBus's own direct subscribers (memory, NAc, Wire 2, Wire 4) DO receive such pain Reactions,
through `PainBus._bridge_reaction_to_pain_subs` (lossy), whenever they are published on that
`pain_bus.reaction_bus`. Clarify, so that GL3.B0's census does not misread it.

**N2 — `PainBus._suppress_bridge` is a plain instance flag, not thread-local.** With the narrator
thread and the loop thread both publishing, one thread's flag can suppress the other's bridge
dispatch, or fail to suppress it. Pre-existing. Add it to GL3.B0's two-thread census next to the
latch race.

**N3 — Runtime list.** The fixture orchestrator (`simulation/fixture_orchestrator.py`,
`--sim scenarios/substrate/*.yaml`) is absent from §1.2a. Verified: it calls no
`evaluate_failures`. Add the line so the census is complete.

**N4 — "Primary `Embodiment`" has no code meaning.** Define it as the `Embodiment`
`runtime/bootstrap.py::build_executor` constructs with a non-empty `agent_id`. Assert at most one
live sequencer per `agent_id` per process, so a harness that builds two AUTs (or rebuilds one)
fails loudly instead of running two seq authorities.

---

## What I verified (code, at c80d619b)

- `Executor._stamp_invocation` is the single writer of `rpe` / `drive_pressure_before` /
  `drive_relief` / `pain`. It runs on every exit path of `_run_started`. The pressure snapshot is
  taken before `tool.run` in `_run_started`, not in `_stamp_invocation`.
- #1125 is merged (#1164). `_drive_pressure_snapshot` uses `sem._read_sensor_value`.
  `_drive_relief` still reads the `drive_progress_by_drive` side effect, and
  `tool_bridge.py::_drive_progress_by_drive` reads `vital_metrics` by bare name (#1161 still
  blind). `_drive_progress_by_drive` iterates the affordance's own effect keys (action-scoped).
- `capture_loop_action` → `capture_episodic_memory` builds `EncodingSignals` from the ToolOutput
  stamps and queues `capture_from_loop_async` (drop-oldest on full).
  `EncodingSignals.to_dict` flattens `extra`, and `__post_init__` rejects collisions and non-JSON
  values. `encoding_tag` reads only declared fields, not `extra`. Callers:
  `tool_dispatch.execute_and_learn` and the agent_loop agent-fallback path. The PLANNING-approved,
  parallel and retry `executor.execute` sites do not capture.
- ToolOutput persistence and substring recall (D3): probed; see D3.
- Hippocampus episodes never ship in bundles (`hivemind/bundle.py`: "Episodes NEVER ship").
- `_resume_capture_seq` runs only inside `load_state` (`hippocampus_persistence.py`). Two load
  paths: `bio_stack.build_bio_stack` and `orchestrator._restore_aut_from_session`. The latter runs
  after the AUT executor exists.
- `Embodiment` has `agent_id`. `build_executor` passes it. `create.embodiment`, the foundry and the
  orient probes construct with `agent_id ""`.
- Narrator tools are registered on `orch_registry`, with separate reflex instances, and the
  `sim.aut` thread exists. `MinecraftSyncPump._run` writes sensors through `sync_world_sensors` on
  `mc-sync-*` and never calls `evaluate_failures`. The `evaluate_failures` callers match §1.2a,
  plus `percepts.py` / `foundry.py`, which the plan already lists.
- `evaluate_failures`: the latch clears only at the two `elif cleared:` sites, plus the silent pop
  on an unreadable sensor. The latch is set regardless of `pain_bus`.
- PainBus: direct subscribers are dispatched before the ReactionBus forward. The refractory key is
  `(entity, failure_mode)`. ReactionBus refractory is `f"{kind}:{source}"`, `"pain": 0.5`, default
  0.5. `pain_signal_to_reaction` sets the coarse source, so the §3.2 collision is confirmed.
- `build_pain_bus` auto-wires memory, NAc, Wire 2 and Wire 4 whenever `nac` is present, so they are
  reachable on every production entry that builds the bio stack (orchestrator AUT via the factory,
  `minecraft_harness`, `embodied_runtime`).
- `_distribute_reward_from_reaction` pays `+intensity` for POSITIVE and skips `None` / `WORLD`
  agent ids. `distribute` iterates all `NAc._eligibility`; its writers are listed in S7.
  `anticipatory_pre_activate` has no production caller.
- `capture_reaction` appends to the pending episode.
- No `relief_store` / `ReliefStore` exists in `src/` or `scripts/` (Phase 5 is unbuilt; GL2c
  option B has no consumer yet).
- The named guard tests exist: `test_agent_loop_selection_golden.py`,
  `test_decision_provenance.py`, `test_encoder_golden_v1.py`, `test_water_trial_smoke.py` (drives
  the real loop and reads the capture), `test_exp61_run.py`, `test_transition_drive_pain.py`,
  `test_cluster_fear.py`, `test_nac.py`.
- T1-16's `Re-run on:` wording (ledger): "`_rank_by_relevance` or `_query_hippocampus`;
  Hippocampus save/restore or the resume path (`RESUME_STORES`); the memory record shape".

UNVERIFIED:
- Whether any shipped scripted sequence stacks `warm_self` past a nociceptor threshold. That is the
  §3.2 census, owed by GL2b(ii).
- Exact Minecraft action/latch timing beyond the code reading in S5 (no rig run).
- Whether `actions.jsonl` serialises `str(ToolOutput)`. Not found by grep; check during the D3 fold.
