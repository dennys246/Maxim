# GL4–GL5 design review — WIRING lens (latent_forward_model.md)

Reviewer lens: real consumers and the real credit path (D43); right encoding and seams; no
hand-composed shortcut that passes while the shipped loop fails. Read at worktree `.worktrees/e-t8b`,
origin/main `d9dffe6e` (the plan's §3 was written at `f1c833ab`; 702 commits later, GL2a has merged as
`660d2806`/`485225d7`). Read: the whole plan, `docs/experiments/DESIGN_REVIEW.md`, `docs/CODE_REVIEW.md`,
`docs/plans/grounding.md` (G1–G20 not re-litigated), `docs/plans/reviews/grounding_gl1/wiring.md`
(format), all of `docs/wiring/`, the bio-memory / runtime-tools / persistence-config briefs,
`ARCHITECTURE.md`, `docs/plans/thalamic_relay.md` §5.7, §GL3.B1, §GL3.B2 and §8, and
`docs/plans/autonomic_layer.md` §3.1–3.1.1. Every code claim below was read at `file::symbol` at
`d9dffe6e`; anything not read is marked UNVERIFIED. No sim, no LLM, no file edited but this one.

**Verdict: 3 DO-NOT-BUILD / 8 SHOULD-FIX / 6 NIT.** Each DO-NOT-BUILD folds as a plan edit before code;
none re-opens G1–G20. D2 and D3 move work across the decomposition fence or change what S1 records,
so the owner may want to treat them as decisions. Recommended reading order: **D2, D1, D3**, then S8
(the GL3 handoff), S2 (caused-only nociception), S7 (the S4 encode door).

---

## DO-NOT-BUILD

### D1 — S1: the Cerebellum target fix cannot happen "at the existing observe call site"; the consequence does not exist there yet

**Stage:** S1 (§4.2 item 2, §7 S1 What/Guards).

**The issue.** The plan says that at the existing `observe_from_action` call site, `actual` becomes "the
consequence dims (the 6-d core plus the body's per-drive block)". That call site is inside the tool:
`embodiment/tool_bridge.py::ModulatorAffordanceTool.execute` calls `self._cerebellum.observe_from_action(
entity=self._entity.full_path, …, actual=entity_state, …)` and then `get_confidence` → `sim_cerebellum`,
all while `tool.run` is still running. The consequence record is built **after** the tool returns, in the
executor:
- `runtime/executor.py::Executor._stamp_invocation` pops the invocation's pain
  (`ToolPainBridge.pop_invocation_pain(invocation_id)`) and only then calls
  `Executor._interoceptive_outcome`, which reads the after-values, nets the drift from the outcome window
  (`window.drift`, `window.cleared`) and calls `sem.py::interoceptive_outcome`.
- None of the pain, the drift window or the record is visible to `tool_bridge`.

So S1 as written has two ways to build, and both break a rule already decided:
1. Compute a consequence inside `tool_bridge`. That is a second record of the same event, without the
   caused/felt pain and without the drift netting (G14). autonomic_layer.md §3.1.1 and this plan's §4.1
   both say "one record; the target is its projection" (the dropped `ConsequenceCode`).
2. Keep the call where it is and pass whatever `tool_bridge` has. That does not fix the target.

The guards inherit the problem. `tests/unit/test_cerebellum_wiring.py::_rig` drives
`ModulatorAffordanceTool.execute` directly, with no executor, so an "extended" test on that rig can never
see an `InteroceptiveOutcome`. Its `test_every_call_site_kwarg_exists_in_the_signature` parses
`tool_bridge` and asserts the call is there ("did it move?"). The plan says S1 leaves the test files'
structure alone, but the correct fold has to move the call.

There is a scope fact here too: `_outcome_start` returns `None` for a body with `agent_id == ""`. A
Cerebellum fed from the record therefore stops training wherever no record is minted. Today
`tool_bridge` trains on every SEM call that has a Cerebellum wired. Which non-agent-bound executors have a
Cerebellum is **UNVERIFIED**. The ephemeral `OrchestratorActorTool` tool is built without `cerebellum=`
(`simulation/tools.py`), so it is not one of them.

**Fold (plan edit; strict first).**
- **(a, recommended)** Move the observe, and the `get_confidence` → `sim_cerebellum` line, out of
  `tool_bridge` into `Executor._stamp_invocation`, right after `_interoceptive_outcome` returns a record.
  The executor already holds the Cerebellum (`Executor._cerebellum`, threaded through
  `Executor.generate_entity_tools`). It reaches the key through the tool's #1195 read-only accessors
  (`affordance_name`, `acting_entity`, `modulator`, `affordance_schema`), which `_outcome_start` already
  uses. Delete the `tool_bridge` block. Rewrite `test_cerebellum_wiring.py`: drive the guard through
  `build_executor` → `Executor.execute`, point the AST call-site check at the new site, and keep the
  deletion probe ("remove the observe → red"). Add one sentence on the scope change above, with the count
  of affected executors. This stays inside G1/G14's `executor.py` wording (`_stamp_invocation`).
- (b) Keep the observe in `tool_bridge` and record nothing new there. S1's Cerebellum half then moves to
  the post-fence out-of-band stage, and S1 is record-only for the pair.
- Never (c): a second consequence computation in `tool_bridge`.

### D2 — S3 (and S1's persistence claim): the cradle seam edits the ORCHESTRATOR's instance; the `--sim` AUT never reaches `BioStack.on_session_end`, and `--resume-sim` carries neither file

**Stage:** S3 (§4.2 item 3, §7 S3 "The cradle seam", Gate "caller grep"), and §3.1's persistence row as
it applies to the cradle.

**The issue.** §4.2 item 3 says: "On the `--sim` orchestrator path … the Cerebellum is saved by
`runtime/agent_factory.py`'s shutdown, which calls `bio_stack.save_cerebellum()`." The method,
`AgentInstance.shutdown`, does call `save_cerebellum`. On `--sim`, though, it is called on the wrong
instance:
- `simulation/orchestrator.py::start_simulation_mode` calls `_orch_instance.shutdown()` and nothing else.
  `_orch_instance` is the narrator (`agent_id="sim_orchestrator"`, `load_persisted=False`,
  write-but-don't-read). Its tools are the narrator tools, and the actor tool's ephemeral
  `ModulatorAffordanceTool` gets no Cerebellum, so that Cerebellum never trains.
- The AUT (`_aut_instance`, `agent_id="sim_aut"`, built by `create_full_agent` with
  `persistence_dir=sim_tmpdir`, a fresh `tempfile.mkdtemp` per run) is never shut down. A grep for
  `_aut_instance.shutdown`, `save_cerebellum` and `on_session_end` across `simulation/` finds none for the
  AUT. The AUT's session end is `runtime/agent_loop.py::run_agentic_loop` →
  `runtime/bio_integration.py::end_bio_session`, which flushes and saves the Hippocampus and ends the
  MemoryHub session, and never touches the BioStack.
- So on `--sim` today the AUT's Cerebellum is **never saved**, and its `distributor.cleanup_session`
  never runs. The bio-memory brief's SCN entry already says "`--sim` does not call it". The plan's version
  of why is wrong.
- `simulation/orchestrator.py::_restore_aut_from_session` (`--resume-sim`) restores `aut_hippocampus`,
  `aut_nac`, `aut_ec` and `aut_atl` only. It restores no Cerebellum and no predictor.

Consequences for the plan:
1. The S3 edit as written ("agent_factory shutdown calls `BioStack.on_session_end` instead") runs the
   refit on the narrator's untrained stack. S3's gate, "a caller grep shows the refit is reached from
   `on_session_end` on the cradle (`--sim`, through the agent_factory seam)", **passes on that grep while
   the AUT path never reaches it**. That is the D43 / "a fix ships with a caller" shape exactly.
2. "Together with the persisted training set, it refits" has no cross-session carrier on the cradle path
   that S0b and S5 use. The AUT home is per-run, and a resume does not restore the Cerebellum's payload
   `"1.2"` or `consequence_predictor.json`.
3. The real AUT seams are `start_simulation_mode`'s AUT teardown (after the AUT thread joins) or
   `run_agentic_loop`'s end. Both sit inside Session A's fence, and both functions are pinned by the
   function-length baseline (`start_simulation_mode`, `run_agentic_loop`), so Rule T applies.
4. The T1-13 walk premise changes. `cleanup_session` would start running on the **AUT's** distributor
   on `--sim`, not the orchestrator's.

**Fold (plan edit + owner decision at S3 start; strict first).**
- **(a, recommended)** Rewrite §4.2 item 3 and S3's cradle seam to name the AUT's seam: the
  orchestrator's AUT teardown calls `_aut_instance.bio_stack.on_session_end()` after the AUT thread joins
  and the Hippocampus has been flushed. Build it as a Session-A-coordinated slice after the fence, with
  Rule T accounted for. In the same stage, add `cerebellum` and `consequence_predictor` to
  `_restore_aut_from_session`, and to the `aut_*.json` save that the resume reads, with
  `test_persistence_compat.py` round-trips. Replace the grep gate with a test that drives
  `start_simulation_mode`'s teardown (fixture runtime, mocked LLM) and asserts that the **AUT's**
  `BioStack.on_session_end` ran, with a deletion probe.
- (b) Declare the cradle arms single-session, or persistent-HANDLE only (`console/handle.py` calls
  `instance.shutdown` on the adopted agent). The S3 refit then runs only on the Minecraft harness and on
  HANDLE paths. State that S5's cradle arm cannot accumulate across `--resume-sim`.
- Whichever is chosen, the orchestrator-instance edit is struck. If it is wanted for the narrator's own
  `cleanup_session`, it is a separate change with its own justification.

### D3 — S1: two of the four `ActionContext` blocks cannot be produced at the planned assembly site without a fenced edit or an EC side effect

**Stage:** S1 (§4.1 Context, §4.2 item 1), with a knock-on to S1's "zero selection-path diffs" gate.

**The issue.** §4.1 defines the blocks as `identity` (LinguisticEncoder raw embed), `object` (the Receptor
embedding of the target's readings), `situation` ("the active body-world Receptor vector(s) at decision
time") and `params`, assembled in `tool_bridge` and attached in `_stamp_invocation`.
- **`situation`: no vector reaches the tool path.** The decision-time encode happens in
  `runtime/substrate_proposal.py::propose_via_substrate` / `_encode_current_clusters`, which is fenced.
  What travels on is `LLMProposal.clusters` (ids) and `cluster_margins` (floats); that is all
  `bio_integration.py::capture_loop_action` receives (`situation=getattr(proposal, "clusters")`). Getting
  the vector to the executor means editing the fenced proposal path. Re-encoding at tool time reads a
  different moment (in substrate-primary the proposal was encoded the tick before).
  `capture_loop_action`'s own docstring warns that re-encoding at capture "would read the POST-action
  state (the wrong key) and write the EC".
- **`identity`: the only embed door writes the EC.** `similarity/encoder.py::LinguisticEncoder.embed`
  calls `self.ec.record_encoder_provenance("linguistic", …)`, which is persisted in `ec.json`
  (`similarity/ec.py::EntorhinalCortex.record_encoder_provenance`). An identity embed per invocation
  therefore changes the saved EC of a substrate-primary cradle run that today has no linguistic stamp. It
  also loads mpnet on a hot path that otherwise never loads it, and `tool_bridge` holds no
  LinguisticEncoder handle at all. S1's gate greps only for "EC encode" and `_sensor_embed`, so it would
  miss this. GL3.B1 (thalamic_relay.md §5.7 item 3) will also make `provenance=` a required keyword on
  `LinguisticEncoder.encode*`.
- **`object` "through the registered Receptor": no such receptor exists.** GL3.B1 registers three
  body-channel receptors (`body.drives`, `audio.doa`, `world.state`; thalamic_relay.md §5.8). The acted-on
  scene entity's own sensors are not among them, so "after GL3.B1 they arrive through the registered
  `Receptor`" does not hold for this block.
- **The size cost is paid on the loop thread.** `memory/encoding.py::EncodingSignals.__post_init__` runs
  `json.dumps(self.extra)` at construction, and `capture_episodic_memory` builds the `EncodingSignals`
  synchronously, before the queue. That is about 1.5k floats per invocation.

**Fold (plan edit, recommended).** Make S1 record **raw inputs, not embeddings**, and compute every
block offline (S2) or at the refit (S3), each with the geometry tag it was computed under:
- `affordance` and the entity name as strings, which feed `identity`;
- the target entity's numeric readings and declared ranges, which feed `object`. `tool_bridge`'s
  `entity_state` is exactly this, and it is already in `output["entity_state"]`;
- the AUT's pre-action readings from `Executor._outcome_start`'s snapshot, widened to the body's sensors,
  inside G14's `_run_started` scope. This is `situation` at execution time, stated as such and kept
  distinct from the credit key's decision-time clusters, whose ids are already on the trace;
- params.

`_sensor_embed` is pure and deterministic (`similarity/encoder.py::_sensor_embed`), so the offline
encode is byte-identical to a live one under the same tag. S1 then has no EC side effect, no mpnet on the
tool path and no fenced edit, and the T1-16 trace growth drops by about two orders of magnitude. The
alternative, carrying vectors from `propose_via_substrate` on the proposal, is a fenced edit that waits
for 1.3.6, and identity would still need an EC-free embed door. That is the non-strict option, and the
owner should decide it at S1 start if (a) is refused.

---

## SHOULD-FIX

### S1 — S1/S2: the target projection the plan calls is not in the shipped record, and the record's drive block is per-affordance, not per-body

**Stage:** S1, S2, S3 (§4.1 Target).

`embodiment/sem.py::InteroceptiveOutcome` (GL2a, merged) has no `as_vector`, no `schema_id` and no
`valence`. Its core is five fields (`_OUTCOME_CORE = nociception, drive_pain, relief, harm, urgency`).
`autonomic_layer.md` §3.1.1 specifies `as_vector(schema_id="ans-v1")`, but neither GL2a nor this plan's
stage list owns building it. And by G14, the record covers only **the invoked affordance's own declared
drives** (`sem.py::affordance_declared_drives`, `Executor._outcome_start`), so `drive_delta` /
`deviation_after` are ragged across keys: a `touch` and a `turn` record different drive sets. §4.1's "per-body
block: `deviation_after ‖ drive_delta` in the body's declared drive order" does not exist in the record,
and the docstring says absent drives are "never imputed".
**Fold:** name the owner of the projection: a pure function in S2's script, moved into
`memory/consequence_predictor.py` in S3, never a method added to the SHAPE of the GL2a record without
GL2's own review. Define how absent drives are handled: a mask, not a zero (zero would claim "no change"
for a drive the action never touched). Say where `ans-v1` is stamped. The record carries no schema id, so
either the projection stamps it on every pair or GL2 adds `extra["schema_id"]`.

### S2 — S0b/S2: "caused-only nociception" cannot be trained before the bridge split lands, and the dependency is missing from §7's table

**Stage:** S0b, S2 (§4.1 "Caused, not felt", S2 owner decision).

The shipped record carries `extra["nociception_basis"] = "caused_or_felt"`
(`sem.py::interoceptive_outcome`). `bridges/tool_pain_bridge.py::ToolPainBridge.pop_invocation_pain`
returns `caused if caused is not None else felt`. autonomic_layer.md §3.1 puts the split at the
out-of-band producer stage, after the fence, and says GL4 "trains on them only after the split". This
plan's dependency table gives S2 only "S0b PASS". `nociception` is one of the three gated dimensions, so
S0b's sign counts and S2's gates on it read a caused-or-felt mix. S0b's "frozen copy in the script"
cannot split it either, because the bridge entry holds both values and returns one.
**Fold (strict first):** (a) add the split stage to S0b's and S2's dependencies for the `nociception`
dimension; until it lands, `nociception` is reported and not gated, and the gate reads `harm` and
`relief`. (b) Non-strict: gate on the mix and state it. Ask at S0b start, together with Q7–Q9.

### S3 — S1: assemble the `ActionContext` in the executor, define `entity_path` as GL2a's `CauseRef.entity`, and list `tools/base.py`

**Stage:** S1 (header file set, §4.1 Context, §4.2 item 1, Q6).

- **No channel exists from the tool to the executor for the `ActionContext`.** Executor-stamped
  `ToolOutput` fields are documented "Tools never set it" (`tools/base.py::ToolOutput`). `_outcome_start`
  already reads everything the context needs through the #1195 accessors, so assemble it there (or in
  `_stamp_invocation`), symmetric with the GL2a record, and keep `tool_bridge` out of S1 except for the
  deletion D1 makes.
- **The file set omits the field's home.** A `ToolOutput.action_context` field (`repr=False`) is an edit to
  `tools/base.py`, which is not in S1's exempt set in the header. GL2a's set has it. Add it.
- **"Key on the TARGET entity" is a no-op or undefined on the shipped YAML.** In every shipped cradle and
  Minecraft item read, the acted-on object owns the modulator: `cradle_fire_pit.yaml` `touch`, and
  `minecraft_bread.yaml` `eat_bread` with `target` defaulting to the bread itself. The body's own
  modulators (`minecraft_player.yaml` `avatar.escape_water`) have no object. The one site where owner
  differs from target, `OrchestratorActorTool`, builds an ephemeral tool with no Cerebellum. And
  `ActionContext.entity_path` ("the TARGET … not the modulator's owner") contradicts GL2a's
  `CauseRef.entity` for the same event (`Executor._outcome_start`: the acting entity, `""` for the body's
  own modulators), so two records of one event would name different entities.
  **Fold:** `entity_path := CauseRef.entity`. For S1, Q6 becomes "no key change; only `actual` changes",
  unless the owner names a `target=` case. Record a resolved `target=` separately if one is ever needed.

### S4 — S1: "`import_state` refuses a newer version" turns into a set-aside-and-overwrite under `load_safe`

**Stage:** S1 (Cerebellum payload `"1.2"`).

`Cerebellum.load_safe` treats any exception in `UNREADABLE_STORE_ERRORS` (`ValueError`, `KeyError`,
`TypeError`, …; `utils/store_ownership.py`) as corruption and calls `start_fresh_keeping_copy`, so the
**next save replaces the file**. A refusal raised as a `ValueError` from `import_state` would copy a newer
file aside and overwrite it. `build_bio_stack`'s `_build_cerebellum` goes through `load_safe`.
**Fold:** the refusal raises a typed exception outside `UNREADABLE_STORE_ERRORS` (or `load_safe` handles
it the way it handles `OSError`), which leaves the file untouched and refuses saves over it. Add a test
that a newer file survives a session end byte-identical. Also say what a `"1.1"` file's `models` become:
dropped, or kept under the old key for the dormant read side.

### S5 — S3: the persisted predictor is a second store of the pairs and needs the store contracts, not only `atomic_write_json`

**Stage:** S3 (§4.2 item 4).

A dual-form kernel ridge model has to persist its training contexts in order to predict. So
`consequence_predictor.json` is the training set. That contradicts "no new store" (lookback decision 6),
and its size is unbounded: about 10³ pairs × about 1.5k dims of JSON floats is tens of MB, which matters
on a Pi. The bio-memory brief requires every persisted memory store to carry `StoreFileOwnership` (#939,
#971: never save over a file it did not read), to honour `load_persisted` (#972), and to copy a corrupt
file aside. §4.2 item 4 lists only `atomic_write_json`, `with_format_version` and `hash_scheme`.
**Fold:** name it a store (or keep only the dual coefficients plus a pointer to the Hippocampus traces
by `pid`, if that survives eviction, which is **UNVERIFIED**). Give it a size bound and an eviction rule
counted in S0b. Mix in `StoreFileOwnership` with a `_reset_store_state`. Add `test_persistence_compat.py`
plus a corrupt-copy test. For the two-process byte-identity test, pin the BLAS thread count (or round
the persisted floats), because a numpy solve is not guaranteed bit-stable across thread counts.

### S6 — S3: name the session-end order and the shadow log's site

**Stage:** S3.

- **Order.** On the Minecraft harness, `aut.bio.on_session_end()` runs after `run_agentic_loop`'s
  `end_bio_session`, which has already flushed, stopped the capture worker, saved, and run MemoryHub
  consolidation (`simulation/minecraft_harness.py::run_minecraft_aut`'s `finally`).
  `AgentInstance.shutdown` runs `memory_hub.on_session_end()` before anything else. A refit that "replays
  the session's loop traces" therefore reads them after consolidation, and the plan does not say whether
  consolidation prunes or merges `"loop"` traces or their `extra` (**UNVERIFIED**). **Fold:** state the
  order (after the flush; before consolidation, or after it with a counted loss), select the traces by
  `run_id`, and count the dropped ones next to S0b's queue-drop count.
- **Site.** "For every candidate tool at decision time" is `propose_via_substrate` / the loop's choose
  step, which is fenced. Name the site and its fence row.

### S7 — S4: the encode door for consequence nodes is unnamed, and the existing sensor door credits and activates; "nothing happens → zero vector" depends on ranges the plan does not declare

**Stage:** S4 (§4.2 item 5).

- **The existing sensor door is not read-only.** `similarity/encoder.py::SensorEncoder.encode_sensors`
  calls `ec.pattern_complete_or_separate`, then `atl.activate_substrate_node`, then
  `nac.update_eligibility`. If S4 encodes predictions through it, every consequence node becomes an ATL
  activation and a creditable NAc eligibility key, which puts it on the credit path. The plan's guards
  (other modalities byte-identical) cannot see that, and "read-only reuse" would be false.
  **Fold:** name a new EC-only encode function (`_sensor_embed` → `pattern_complete_or_separate`, with no
  ATL activation and no eligibility). Add a guard that no `"consequence"` node id ever appears in NAc
  eligibility, `reward_bias` or ATL activations, proven by deletion.
- **Direction.** Under `_sensor_embed` with the A4 gain, a dimension contributes zero only at the
  normalised midpoint of its declared range (`docs/wiring/cosine-separation-is-directional.md`). With
  the native `[0, 1]` ranges of `relief`/`harm`/`nociception`, "nothing" (0) is an extreme and gets full
  weight on the low basis, so a no-consequence prediction would form a node. And `_sensor_embed`
  *rotates* direction with value by design (its docstring), so "a mild burn and a severe burn share a
  direction" is not given either. **Fold:** declare sign-symmetric ranges (`[-1, 1]`) per dimension, and
  make an offline cosine replay of S0b's predicted vectors through the real encoder (corollary 3 of that
  wiring entry) an S4 gate: nothing maps to the zero vector, mild and severe of one kind co-cluster at
  0.85, burn and relief separate.

### S8 — GL3 handoff: GL3.B2 and this plan describe different producers and different pid spaces for one contract

**Stage:** S1 / S2 against thalamic_relay.md §GL3.B1, §GL3.B2, §8.

- **Producer and pid space.** thalamic_relay.md §GL3.B2 says "the registry exposes
  `(pid, receptor context embedding, action)` per pass". Its gate requires that "every row's pid resolves
  to exactly one `AfferentEvent`". This plan builds the context per **invocation** on the tool path, with
  a tool-path `pid` from `_stamp_invocation`. If both draw from one per-agent `EventSequencer` (G3), a
  tool-path pid is not an `AfferentEvent` pid, and GL3.B2's gate cannot pass as written.
- **Fence row.** thalamic_relay.md §8's fence table says GL3.B2 "waits for GL4 Stage 2", but S2 is
  offline `scripts/`.
- **GL3.B1's consumer.** GL3.B1 "lands only together with a consumer of provenance … or the forward
  model's contamination guard". No stage here says it carries GL3.B1, and §4.1 says S2 "does not wait for
  GL3". Unless S3 or S5 names GL3.B1, it has no landing slot.
- **Receptors.** §4.1's "after GL3.B1 they arrive through the registered `Receptor`" also fails for the
  `object` block (D3).

**Fold (owner decision, asked with S0a; strict first):** (a) one reconciliation edit in both plans: the
pair's pid is the tool-path invocation pid (the `InteroceptiveOutcome.pid`); `AfferentEvent` pids are
receptor passes in the same sequencer space; GL3.B2's gate reads "resolves to exactly one
`InteroceptiveOutcome` or `AfferentEvent`"; the fence row reads "GL4 S1"; S3 states whether it carries
GL3.B1 or leaves it to S5. (b) Move the context to the registry per pass, which puts S1 behind GL3.B1
and the fence and contradicts D3's fold.

---

## NIT

- **N1 — §3 is stale at `d9dffe6e`.** GL2a has merged: the `InteroceptiveOutcome` field on `ToolOutput`
  (`repr=False`), `extra["interoception"]` in `capture_episodic_memory`, and the 2S-c pain stamp. S0b's
  "or its frozen copy in the script if GL2a has not merged" is moot. Re-date §3's header and cite the
  shipped `sem.py::interoceptive_outcome`. Note that `invocation_id` is `compare=False` and not
  persisted (`to_dict`).
- **N2 — the Cerebellum's `"version"` bump.** The persistence brief tombstones the payload-layer
  `"version"` strings and directs a shape change to `_format_version` or the envelope `schema_version`.
  The Cerebellum is not one of the envelope-conformant six, and its `"version"` is still read by
  `Cerebellum.import_state`. Say so in S1, with the reason this string, not `_format_version`, is the one
  that moves.
- **N3 — two provenance fields for one pair.** `ActionContext.provenance` and
  `InteroceptiveOutcome.provenance` can disagree. The pair constructor should derive one from the other,
  or refuse a mismatch (M41's test can carry the case).
- **N4 — "the pid S1 persists in … the Cerebellum payload".** Per-key RW means are aggregates and hold
  no per-sample pid (`cerebellum.py::ForwardModel`). Say what pid, if any, the payload carries (a
  high-water mark for the resume?), or drop the clause.
- **N5 — S4 hivemind: an existing never-exported path.** `hivemind/bundle.py::_filter_ec_nodes_by_domain`
  drops `domain == IDENTITY_DOMAIN_MARKER` nodes unconditionally, so a domain marker on consequence nodes
  would keep them local without a `bundle.py` edit (T1-14's wording). It overloads the "identity"
  meaning, so it is offered under Q4 as an option, not a recommendation. The receiver-side refusal is
  still owed either way.
- **N6 — T4a covers S4 too.** The `"consequence"` modality and the α read have no reader until S5. The
  Dormant-at-1.4.0 clause should name them, not only `ConsequencePredictor`. Rule T also applies to every
  pinned function the stages touch (`ModulatorAffordanceTool.execute` at 502 lines;
  `start_simulation_mode`; `run_agentic_loop`). D1's fold shrinks the first.

---

## Proposed `docs/wiring/` entry

**`docs/wiring/session-end-seams-per-runtime.md`: which session end each runtime actually reaches.**
"If you wire a session-end behaviour (a save, a refit, `cleanup_session`), name the runtime and its
instance. On `maxim --sim`, `AgentInstance.shutdown()` runs on the **orchestrator's** instance
(`sim_orchestrator`, write-but-don't-read); the AUT (`sim_aut`, a per-run `mkdtemp` home) gets only
`run_agentic_loop` → `end_bio_session` (Hippocampus flush and save + MemoryHub), never
`BioStack.on_session_end`, so its Cerebellum is never saved; `--resume-sim` restores hippocampus/nac/ec/atl
only. The Minecraft harness and the survival scripts run `end_bio_session`, then `bio.on_session_end()`.
The console HANDLE runs `instance.shutdown()` on the adopted agent. The trap: a fix at
`AgentInstance.shutdown` passes a caller grep and misses the `--sim` AUT. Measure it with a test that
drives the runtime's teardown and asserts the AUT's own stack ran. Corollary: the per-invocation body
record exists only after `tool.run` returns, in `Executor._stamp_invocation`, so anything that needs the
consequence (a forward-model write, a context) lives in the executor, not in the tool." (GL4 design
review, 2026-10-24; D1, D2.)

---

## Verified

At `d9dffe6e`, by reading the code at `file::symbol`:
- `embodiment/tool_bridge.py::ModulatorAffordanceTool.execute`: `entity_state` is read from
  `self._entity.sensors` after `modulator.execute` and before the `_apply_sensor_deltas(…self_effect)`
  block; `observe_from_action(entity=self._entity.full_path, …, actual=entity_state,
  sensor_ranges=<owner's>)` and `get_confidence` → `sim_cerebellum` on the owner key; pinned at 502 lines
  in `utils/function_length_baseline.json`. The plan's §3.1 rows hold.
- `embodiment/cerebellum.py`: `ForwardModel.update` (RW, first observation = actual), `confidence`
  (`1 − exp(−0.3n)`, cap 0.95), `export_state` `"version": "1.1"`, `import_state` warns and loads an
  unknown version, `load_safe` sets an unreadable file aside and replaces it;
  `utils/store_ownership.py::UNREADABLE_STORE_ERRORS`.
- `runtime/executor.py`: `Executor.execute` uses `uuid.uuid4()`; `_outcome_start` returns `None` for
  `agent_id == ""` and uses the #1195 accessors; `_interoceptive_outcome` runs after `tool.run` with the
  pain and the drift window; `_stamp_invocation` is the single stamp; `generate_entity_tools` threads
  `cerebellum=`.
- `embodiment/sem.py::InteroceptiveOutcome` / `interoceptive_outcome`: a 5-field core, no `as_vector` or
  valence, per-affordance drive blocks (G14), `nociception_basis = "caused_or_felt"`,
  `OUTCOME_PROVENANCE` includes `apparatus`.
- `bridges/tool_pain_bridge.py::ToolPainBridge.pop_invocation_pain` returns caused else felt.
- `tools/base.py::ToolOutput`: `interoceptive_outcome` and `nac_failure_booked` are `repr=False`.
- `runtime/bio_integration.py`: `capture_loop_action` passes `proposal.clusters`/`cluster_margins` only;
  `capture_episodic_memory` builds `EncodingSignals(extra={"interoception": …})` synchronously;
  `end_bio_session` touches no BioStack. `memory/encoding.py::EncodingSignals` runs `json.dumps(extra)`
  at construction and `to_dict` flattens `extra`.
- `runtime/substrate_proposal.py`: `LLMProposal` gets `clusters` and `cluster_margins` (no vectors);
  `_attach_live_situation`.
- `runtime/bio_stack.py`: `BioStack.on_session_end` (save + `cleanup_session` in a `finally`),
  `_build_cerebellum` (`<home>/cerebellum.json`, `load_safe`).
- `runtime/agent_factory.py::AgentInstance.shutdown`: MemoryHub first, then Hippocampus, NAc, and
  `save_cerebellum` last.
- `simulation/orchestrator.py`: AUT `sim_aut` in `sim_tmpdir` (`mkdtemp`); `_orch_instance.shutdown()`
  is the only shutdown; `_restore_aut_from_session` restores 4 stores.
- `simulation/minecraft_harness.py::run_minecraft_aut`: `finally` runs `aut.bio.on_session_end()`. The
  survival scripts call it too (grep).
- `simulation/tools.py`: the actor tool's ephemeral `ModulatorAffordanceTool` gets no `cerebellum=`.
- `similarity/encoder.py`: `LinguisticEncoder.embed` → `ec.record_encoder_provenance` (persisted,
  `similarity/ec.py`); `_sensor_embed` gain/neutral semantics; `SensorEncoder.encode_sensors` → ATL
  activation + `nac.update_eligibility`; `gain_modalities = {"world"}`.
- `similarity/ec.py`: per-modality `_matrix_for`; `pattern_complete_or_separate` does not use the LSH;
  `ECConfig.frozen_centroid_modalities` is pinned equal to `hivemind/merge.py::DEFAULT_FROZEN_CENTROID_MODALITIES`.
- `hivemind/merge.py::SENSOR_MODALITY_THRESHOLDS` has no `"consequence"`;
  `hivemind/bundle.py::_BUNDLE_EC_NODE_FIELDS` and `_filter_ec_nodes_by_domain` (domain-only plus the
  identity marker).
- YAML: `cradle_fire_pit.yaml` (`heat_output`, `fuel`; `touch` `arms.thermal: 0.6`); `minecraft_bread.yaml`
  (`target` defaults to itself); `minecraft_player.yaml` `avatar.escape_water`.
- Cited tests exist: `test_cerebellum_wiring.py` (its rig drives the tool directly; its AST check pins
  the call in `tool_bridge`), `test_cerebellum_dormant_909.py` (scans `src/` and `scripts/`), and
  `test_water_trial_smoke.py::test_water_trial_ticks_acts_and_the_staging_close_persists_fear` (the real
  loop). `test_agent_loop_selection_golden.py`, `test_decision_provenance.py`, `test_encoder_golden_v1.py`,
  `test_stable_hash_two_process.py`, `scripts/lint_prereg_precedes_data.py`, `scripts/grounding_census.py`
  and outstanding.md M41 exist.
- No `PhysicalEventId`, `EventSequencer`, `ActionContext` or `consequence_predictor` exists in `src/` yet
  (consistent with G17 and the plan).

UNVERIFIED: whether Hippocampus consolidation prunes `"loop"` traces or their `extra` (S6); whether trace
eviction keeps `pid`-addressable pairs (S5); which non-agent-bound executors carry a Cerebellum (D1);
the fixture-orchestrator runtime's capture path (left UNVERIFIED as the plan has it, though `sim_aut`'s
`agent_id` is non-empty, so GL2a records mint on that path); `Executor._drive_relief`'s 0.0-for-harm
claim (§3.3, not re-read).
