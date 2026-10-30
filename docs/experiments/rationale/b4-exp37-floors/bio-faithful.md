# B4 design review: bio-faithful lens

Reviewed: b4_exp37_floors_prereg.md DRAFT v1 (2026-10-18)

The question for this lens: does B4 test the real job of T1-2's mechanism, or a caricature of it? Short answer:
not yet. The arm meant to be "resumed, wrong content" (C) actually carries the same fire-pit world as A. The
deceptive scenario's gates point the wrong way. And the metric scores an agent that never goes near the fire as
having learned. Each of these can be fixed on paper. Notes on the other five questions follow the findings.

## What I verified

- **The C prior is not peaceful.**
  - `PEACEFUL_GOAL` ("cradle infant explores the puzzle door and button") routes to the `cradle` arc: the keywords
    `cradle` and `infant` match, and in `simulation/arcs.py::_ARC_KEYWORDS` `cradle` comes first and wins the tie.
  - The narrator's phases advance only by count, never by goal. `simulation/narrator.py::decide` and
    `generate_single` move to the next phase once `_turns_in_phase >= phase.turns_max`, and the decision's `phase`
    field never sets `_phase_idx`.
  - `simulation/generative_runner.py` activates each phase's `world_entities` in order, starting from phase 0.
  - Phase 0 of `cradle` (`exploration`) is `cradle_fire_pit` + `cradle_food` + `cradle_cool_air`, and phase 1
    re-presents the fire pit. The lever door and button arrive only in phase 4 (`tool_discovery`).
  - The phase maxima are 4 + 2 + 3 + 3 = 12, so `--sim-max-turns 12` ends before phase 4 starts.
  - In all 725 committed Exp 37/38 rows (`docs/experiments/data/3[78]_*.jsonl`) there is no lever, door or button
    tool call.
  - So the "peaceful" prior is a 12-turn session in the fire-pit world, with cold air and a goal string naming
    objects that never appear. `docs/experiments/protocols/37_cross_session_graduation_reproduction.md` says the
    peaceful prior "targets cradle phase 3 entities (`lever_door` + `button`)", and the code says otherwise.
  - The prior itself is never written as a row (`benchmark_cross_session.py::run_benchmark` emits only the resumed
    C), so how much it engaged the fire pit has never been measured.
- **What the mechanism shows the LLM is tool-level, not tied to the situation.**
  - Wire-A is `NAc.get_agent_tool_biases(agent_id, top_n=5)`, which its own docstring calls "agent-wide (no
    active-cluster filter)".
  - Wire-1 is `NAc.get_action_risk_profile(min_observations=_WIRE1_MIN_OBSERVATIONS)`, with
    `_WIRE1_MIN_OBSERVATIONS = 5` (`runtime/loop_planning.py` since #1481; it was `runtime/agent_loop.py`).
  - The situation keying (EC cluster × tool) lives in NAc, and neither annotation shows it to the LLM.
- **The resumed AUT has more memory channels than the three named wires.** `simulation/orchestrator.py` (sim AUT
  construction) registers:
  - `MemoryRecallTool(hippocampus)`;
  - `SimilaritySearchTool(ec)`, `TemporalPatternsTool(scn)` and `ConceptQueryTool(atl)` whenever a MemoryHub exists;
  - `PredictOutcomeTool(nac)` and `CausalLinksTool(nac)`;
  - `ThinkTool(pipeline=bio_enrichment)` and `ExamineTool(bridge, hippocampus)`.

  Separately, `prompt_builder` renders the bio-enrichment section "What your experience tells you about this
  situation". Each of these bypasses Wire-A, Wire-1 and the NAc bias.
- **`ExamineTool` recall (#845) is not one-sided.** It recalls by content (`tools/narrative.py::ExamineTool`, Stage
  2: `search_by_content(target)`, past memories only). Because the C prior is a fire-pit world, examining
  `fire_pit` in C recalls prior-session text too. The T1-2 ledger note's assumption that "arm C's peaceful prior
  mostly recalls nothing for those targets" does not hold for `fire_pit`.
- **Decay and drift depend on wall time, not on what the agent lives through.**
  - In-session NAc decay runs every pass (`agent_loop.py::_NAC_PER_TICK_DECAYS`; cluster bias tau 300).
  - In the text sim, the number of passes scales with wall time and LLM latency
    (`docs/plans/experience_clock_decay.md` §Motivation table).
  - Body drift is `time.time()` deltas (`Embodiment.evaluate_failures`).
  - Loads use `apply_decay=False` (`orchestrator.py`, `_restore("nac", …)`).
  - `MemoryHub.on_session_end` applies `decay_all(0.95)`.
  - S3a/S4 (experience-clock decay and drift) are proposed and have not landed.
  - That plan has a T1-2 re-fire note listing the stamps it wants: tick, submit and latency counts, and bias maps at
    save, at load and at end of session.
- **The bodies.**
  - Fire-pit `warm_self` adds `core_temperature +0.2` and `arms.thermal +0.2` (`cradle_fire_pit.yaml`). The deltas are
    additive (`embodiment/tool_bridge.py::_apply_sensor_deltas`), against `arms.thermal` comfort band 0.5 (drift
    0.0008/s) and `core_temperature` comfort band 0.25 (`infant_humanoid.yaml`). So two or three quick warm_selfs
    overheat the arms or the core: warm_self is good when the infant is cold, not every time.
  - The hearth's `warm_self` is `arms.thermal +0.6`, which breaches the band (`cradle_false_hearth.yaml`).
- **The committed engagement numbers are small.**
  - Fire engagement runs 2 to 11 calls per session. In R1 Exp 38 `deceptive_fire`, arm A warmed the hearth about
    once per trial (5 calls in 5 trials), then observed.
  - B had one zero-engagement trial, and C had two (`38_results_r1_distill_qwen_32b.jsonl`).
  - Over those trials C warmed the hearth twice as often as A (10 calls against 5).
- **The ablation flags are sound now.** The NAc-bias ablation is a real ablation again: `MAXIM_NAC_REWARD_BIAS_DISABLED`
  now gates `credit_node`, `reward_bias` and `get_agent_tool_biases` (`decisions/nac.py`, #889). All three Exp 37
  ablation flags have readers in `src/`.
- **What `maxim_all_off` still builds.** `create_agent` with `remembers=False, learns=False` builds no Hippocampus and
  no NAc, but it does build an ATL and a MemoryHub with EC and SCN (`runtime/agent_factory.py`). The sim path does
  more after the factory (pain layers, the DefaultNetwork, introspection tools), and that code reads
  `_aut_instance.memory_hub`, not the loop's `memory_hub=None`.

## DO-NOT-BUILD

### D1. Arm C is not "resumed, wrong content". Its prior is a fire-pit session, so B − C is structurally near zero on `fire_pit` and tilted in our favour on `deceptive_fire`.

**Claim attacked.** The Arms table calls C "Full Maxim resumed from a peaceful-prior session … Resumed, wrong
content: the L8 gate's comparator", and G1 reads B − C as "what the agent carried, not being resumed".

**Evidence.** See the first item of What I verified. The routing, the count-only phase advance, phase 0's entities
and the 12-turn cap together mean the C prior lives through `exploration` and `pain_consequence` with the fire pit
and cold air, then the sharp rock and the blanket. It never meets the door or the button, and no committed row
contains a door or button tool. Its goal string differs from A's, and nothing else about its world does.

**What the mechanism would therefore predict.**
- *`fire_pit`:* B and C both carry fire-pit warmth: hippocampal episodes, examine recall, and a Wire-A bias on
  `fire_pit_warm_self` from relief while cold. G1 is close to a null by construction, even if the substrate works
  perfectly. This is the most economical explanation of the "Arm-C confound" in every June fire (C ≈ B > A at
  Qwen32B and R1). It is not "any resumed prior shifts behaviour": C's prior was the same world.
- *`deceptive_fire`:* the C prior is shared and runs on `cradle`, not `cradle_deceptive`. So C carries "standing near
  the fire felt good", the prior-aligned association for an entity that looks the same (hippocampal recall, and
  `fire_pit_warm_self` surfaced as an out-of-roster tool by the grayscale annotation). That pushes C's hearth
  warming up (R1: C 10 hearth warm_selfs against A's 5). B − C < 0 can then pass on C's inflation alone, with no
  learning in B. That is a comparator unfair in our favour, the strawman class the confounding lens exists to refuse.

**Fix (text for the prereg).**
> **Arm C's prior is content-neutral, and shown to be.**
> - The C prior runs a `cradle_peaceful` arc. It is derived from `cradle` the way `cradle_deceptive` is (same phase
>   count, turn budgets, acts and drives), with every thermal entity (`cradle_fire_pit`, `cradle_false_hearth`,
>   `cradle_cool_air`) removed or replaced by a non-thermal entity of matched affordance count. It must also carry
>   one valenced need episode (for example food under hunger), so C's carried NAc credit-event count is comparable
>   to A's.
> - The prior is written as its own row (`arm = "C_prior"`, not scored) with its tool usage.
> - A structural probe, run in the dry run and asserted on every fire, refuses a C whose prior made any call to a
>   `fire_pit_*` or `hearth_*` tool, or whose Hippocampus returns a match for `fire`, `hearth` or `warm` from
>   `search_by_content`.
> - The reproduction protocol's "targets cradle phase 3 entities" line is corrected.
> - The June "Arm-C confound" is re-labelled in T1-2's prose as "C's prior was a fire-pit session"; it is not
>   evidence that any prior shifts behaviour.

### D2. The deceptive scenario's gates are signed the wrong way: G1–G3 reward the prior-amplification Exp 38 found.

**Claim attacked.** "G1: B − C > 0 on each scored scenario"; "G2: B > `bare_llm`"; "G3: B > `llm_notes`", with
`deceptive_fire` scored on `warm_self_engagement_fraction`.

**Evidence.** In `deceptive_fire` the hearth's `warm_self` is the harmful act (`cradle_false_hearth.yaml`). The
substrate's job there is to make B warm the hearth LESS than a comparator without the hearth-pain memory
(`38_counter_prior_substrate.md` §5: the counter-prior learning direction is B < A). Exp 38 found the opposite at R1:
the substrate amplified the `fire→warm` prior, so B warmed the hearth more (B 0.60 against A's 0.40 or lower), and
Exp 40 found dominance at Qwen32B. Under the draft's uniform sign, that maladaptive amplification passes G1–G3 on
`deceptive_fire` as a "cross-session delta".

**Fix (text for the prereg).**
> **Every gate is signed per scenario.**
> - `fire_pit`: B − X > 0 on `warm_self_engagement_fraction` (see S2 for the state-conditioned form).
> - `deceptive_fire`: B − X < 0, for X ∈ {C, `bare_llm`, `llm_notes`}.
> - Reported beside them, Exp 38's matched-world interaction: (B − C)_fire − (B − C)_deceptive > 0. Its baseline is
>   the same C prior in both worlds once D1 is folded.
> - A B that moves up in both worlds is recorded as "general approach", never as a delta.

### D3. The zero-engagement clamp scores "never approached" as "learned", and the floors are the arms most likely to hit it.

**Claim attacked.** Both metrics are fractions over on-target engagement. The analyzer emits 0.0 when engagement
is 0 (the reproduction protocol §metrics: "denominator clamped to 1").

**Evidence.**
- Engagement is 2 to 6 calls in most committed sessions, and zero-engagement cells exist (R1 Exp 38 deceptive: B 1/5,
  C 2/5).
- Biologically, avoidance learning is approaching and then withholding. An agent that never registered the hearth
  has learned nothing, yet scores perfect avoidance on `deceptive_fire` and total failure on `fire_pit`.
- The bare arms lack `prompt_builder`'s embodied steering (the `is_embodied` suppression of conversational tool
  guidance, and `acting_coach`), so they are the arms most likely to drift into `respond`/`say` and score 0. G2 and
  G3 would then pass on `fire_pit` and fail on `deceptive_fire` for reasons that have nothing to do with memory.

**Fix (text for the prereg).**
> **Engagement is an outcome of its own.**
> - Each arm reports P(≥ 1 on-target engagement) and the engagement count.
> - The two fractions are computed only over sessions with ≥ 1 engagement. A zero-engagement session is excluded
>   from the fraction, never scored as 0.
> - A gate whose arms differ in engagement rate by more than 0.2 (absolute) is reported `ENGAGEMENT_MISMATCH` and
>   not scored.
> - `deceptive_fire`'s primary is Exp 38's **first-contact** measure (`first_contact_warm_self`, already in the
>   record). It is defined only for engaged sessions and is the only measure that separates cross-session transfer
>   from the one-burn, within-session learning arm A shows. The session aggregate is co-reported.

## SHOULD-FIX

### S1. B − C and G3 test "Maxim's whole memory" against notes, not the three wires T1-2 names; without one ablation the row cannot say what it earned.

**Claim attacked.** "The three Exp 37 ablations … do not bear on T1-2 under BF2", and the "Not claimed: which bio
wire carries a delta" line, while FLOORS_BEATEN would move T1-2 to MAINTAINED/EARNED. The row's mechanism column
says "Wire-A annotation + Wire-1 annotation + NAc reward bias → action selection".

**Evidence.** B carries from A, at minimum:
- episodic text: the bio-enrichment section, examine recall, `memory_recall`, `think`'s enrichment;
- NAc tool valuation: Wire-A, the grayscale out-of-roster tools, Wire-1, `predict_outcome`, `causal_links`;
- the EC, ATL and SCN tools.

Wire-A and Wire-1 are tool-level (see What I verified), so the information they give the LLM is the same grain as a
Reflexion note ("warming at the hearth hurt"). `llm_notes` is therefore the true competitor, and a fair one. But a
pass would come from the bundle. Wire-1 needs ≥ 5 observations of a tool, and arm A warms the hearth about once per
trial, so Wire-1 is very likely silent for the key tool at dose 1: one of the three named mechanisms is probably
absent from the test.

**Fix (text for the prereg).**
> **One combined valuation-off arm, `B-valuation-off`:** B with `MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION=1`,
> `MAXIM_DISABLE_VARIANCE_ANNOTATION=1` and `MAXIM_NAC_REWARD_BIAS_DISABLED=1` together (all valid since #889). It is
> reported, not gated: B − `B-valuation-off` reads how much the named wires add over episodic recall.
> - If FLOORS_BEATEN holds but B − `B-valuation-off` ≤ 0, T1-2's mechanism column is re-worded to "substrate-carried
>   episodic recall" before any status move.
> - Either way, each row stamps, at B's first decision, the Wire-A list, the Wire-1 annotated tools, the
>   bio-enrichment section's presence, and the count of `memory_recall`, `examine`-recall, `predict_outcome` and
>   `causal_links` calls.
>
> Cost: +1 sim per trial and scenario (about 12 %). If the owner declines the arm, the mechanism column is
> re-worded before freeze to "the full substrate memory stack (hippocampal recall + Wire-A + Wire-1 + NAc bias)",
> so a pass cannot be read as a per-wire claim.

### S2. `fire_pit`'s metric scores generic approach, not need-conditioned selection.

**Claim attacked.** `positive_approach_engagement_fraction` / `warm_self_engagement_fraction` as the measure of what
carried memory should change on `fire_pit`.

**Evidence.**
- Fire-pit `warm_self` adds `arms.thermal +0.2` and `core_temperature +0.2`, additively. The arms band is 0.5 with
  drift of 0.0008/s, and the core band is 0.25. So repeated warm_self overheats (pain), and only warm_self while cold
  is adaptive.
- A higher share of warm_self among fire engagements is what the LLM prior, a notes arm, and a substrate that learned
  "fire good" with no state would all produce. Wire-A's agent-wide aggregation is exactly that kind of substrate.
- This is the "state-conditioned selection, not eat-sooner" trap that DESIGN_REVIEW.md names.

**Fix (text for the prereg).**
> **Report the need-conditioned split, from the `sim_sensor` log** (or a per-action body snapshot the harness adds):
> - warm_self share at decisions where `core_temperature` is below −comfort_band (cold);
> - warm_self share when it is within band;
> - over-warming events (an `arms.thermal` or core breach caused by `warm_self`).
>
> The gated `fire_pit` measure stays the frozen fraction, but the claim reads "approach share". Wording that implies
> state-conditioned transfer needs the cold-conditioned split to move with it.

### S3. The floors must feel the burn: Body State is an observation, not a bio section.

**Claim attacked.** "The bare arms are likely shorter (no bio prompt sections)"; Information parity "the same
observation text".

**Evidence.** The pain consequence of the hearth reaches the LLM as `Embodiment.format_body_state_for_prompt` ("===
Body State (pain-relevant) ===", with DRIVE and WARN labels), rendered by `prompt_builder` as `body_state`. That is
the environment's feedback, not memory. If B2's bare loop treats it as a bio section, `bare_llm` and `llm_notes`
cannot learn that the hearth hurts. Reflexion's reflection would then have no failure signal (Shinn et al. 2023 need
an evaluator signal in the trajectory), and G3 on `deceptive_fire` would be a floor starved by design.

**Fix (text for the prereg).**
> The bare loop renders the same `body_state` block (sensor values, DRIVE and WARN labels) and the same auto-sense
> percepts. The arm parity test asserts the `body_state` text is byte-identical across arms on the fixture world.
> The Reflexion call receives the full trajectory, observations and body state included, with a generic reflection
> instruction that names no object.

### S4. Body drift and NAc decay depend on wall time, so arm latency becomes an unrecorded variable in the body and in the dose.

**Claim attacked.** "The bare arms are likely shorter"; the dose reading B2 − B; budget matched by turns only.

**Evidence.**
- Body drift is `time.time()` deltas, and NAc decay is per pass, with passes ∝ wall time in the text sim
  (`experience_clock_decay.md` §Motivation). A faster arm gets colder more slowly (less need for warm_self) and
  decays its biases less.
- The prereg predicts the latency asymmetry between bare and full arms itself.
- That plan's T1-2 re-fire note asks for tick, submit and latency stamps and bias maps at save, at load and at end of
  session. The draft omits all of them.
- S3a/S4 are unlanded, and "NAc decay-pacing change" is a T1-2 re-run trigger, so a campaign straddling S3a would
  mix two decay regimes.

**Fix (text for the prereg).**
> Adopt `experience_clock_decay.md`'s T1-2 stamps verbatim:
> - §8.5 tick count, submit count, median and p90 per-turn latency;
> - the save→resume gap and the `apply_decay` used;
> - for the three bias maps: count, sum |v|, max |v| and target-key values at save, after load (hard-fail on a
>   difference) and at end of session;
> - per-turn `core_temperature` and `arms.thermal`.
>
> The whole campaign runs on one side of S3a and S4, stamped by commit. If S4 has not landed, every gate is also
> reported with per-turn latency as a covariate, and a gate whose arms differ in median latency by more than 1.5× is
> flagged.

### S5. The dose reading: biologically defensible only as a reported curve over measured carried content; a monotone trend is not the biological prediction.

**Claim attacked.** Dose = prior same-scenario sessions (A = 0, B = 1, B2 = 2), with a Page's L monotone trend.

**Evidence.** Between B and B2 several things change besides "more experience":
- A's carried biases decay through B's ticks (cluster tau 300 against roughly 27 ticks per turn × 12 turns in Exp 52,
  and more under the 2 Hz text-sim poll), so B2's NAc content is mostly B's own session.
- A second session-end `decay_all(0.95)`.
- Hippocampal episodes, which accumulate without decay, so the episodic channel grows faster than the valuation
  channel.
- Familiarity. `ThoughtGate.should_think` gates bio-enrichment on a combined novelty/salience score, so a
  third pass through the same arc can suppress enrichment: habituation.
- NAc bias saturation (`reward_bias_alpha`).

A working substrate is as consistent with a plateau or an inverted U as with a monotone rise, and the session
count is not the biological dose (that would be the number and size of reinforcement events carried). There is
also no `llm_notes` at dose 2, so B2 − B has no floor.

**Fix (text for the prereg).**
> **The dose is reported, never gated, and read against measured carried content.** The x-axis is the session count
> as the assigned label, plus per row:
> - the target-tool credit events in the carried sessions;
> - |bias| of the target key at load;
> - the hippocampal memories matching the entity;
> - the ThoughtGate pass count.
>
> Page's L is descriptive only, labelled "monotone not predicted". A non-monotone B2 is not evidence against the
> mechanism. If the owner keeps B2, add `llm_notes2` (two reflections) or state that B2 − B has no floor.

### S6. `maxim_all_off` is not shown to be memory-free, and it is a within-session point.

**Claim attacked.** That `maxim_all_off` measures scaffolding against `bare_llm` and bio against A.

**Evidence.**
- Side finding 1 does not bite: it applies only when `with_bio_stack=True`.
- But `create_agent` with both flags off still builds a MemoryHub (EC, SCN) and an ATL. The sim path then registers
  `similarity_search(ec)`, `temporal_patterns(scn)`, `concept_query(atl)` and `SystemStatsTool` from
  `_aut_instance.memory_hub`, not from the loop's `memory_hub=None`.
- The pain layers (`PainInterceptor`, `AnticipatoryPain`) and the DefaultNetwork are built after the factory.
- So "all-off" depends on what the new switch does in `orchestrator.py`, which the draft does not specify. Its
  single-session gap A − `maxim_all_off` also includes within-session bio (enrichment from the current session, NAc
  credit inside the session, reflex and pain layers, the DefaultNetwork), not "bio" in general.

**Fix (text for the prereg).**
> `maxim_all_off`'s sim instance asserts, in a structural probe on the fixture world, that Hippocampus, NAc,
> MemoryHub, `bio_stack`, ThoughtGate and DefaultNetwork are all `None`, and that the AUT tool roster contains none
> of `memory_recall`, `similarity_search`, `temporal_patterns`, `concept_query`, `predict_outcome`, `causal_links`,
> or `examine` with a Hippocampus. The record lists what remains (embodiment drives and Body State, the
> `pfc_preamble` and `acting_coach` choices, the pain-layer choice). A − `maxim_all_off` is labelled "within-session
> bio stack".

## NIT

- **N1. OB1's "ReAct verbatim" is under-specified for this world.** ReAct's published prompts are few-shot and
  task-specific (HotpotQA, FEVER, ALFWorld, WebShop), and the action syntax (`Action: take x from y`) is not
  Maxim's JSON parser. Copying the ALFWorld exemplars verbatim primes the wrong verbs. Name it as "ReAct's
  Thought/Action/Observation scaffold, zero-shot (no exemplars), action line in the shared parser's format; no
  other text", and put the exact frozen prompt in the prereg. Then "verbatim" means something a reviewer can check.
- **N2. Reflexion fairness is fine in kind.** Repeated trials of the same task with a verbal lesson is Reflexion's
  own setting, and the `deceptive_fire` single-burn lesson is its best case. Say that a floor win on
  `deceptive_fire` is the expected boundary result, not a surprise. Record whether the notes block is always in the
  prompt (Reflexion) while Maxim's enrichment is gated (ThoughtGate): that asymmetry favours the floor and should be
  stated.
- **N3. `FearGatedExecutor`.** Say that "OFF on every arm" removes the code-safety reviewer only. NAc's
  `cluster_fear` (carried across sessions, with no tick decay) still runs in the full arms, so it belongs in the
  channel list in S1.
- **N4. The record of what is carried.** Give the prereg a short "what crosses the session boundary" table (store,
  loader, consumer that reaches the LLM), built from `orchestrator.py::_restore_aut_from_session` and the AUT tool
  registration. Every later lens or reader then argues from one list.

## On the six questions, briefly

1. **What T1-2 carries.** Tool-level valuation text plus episodic text, so `llm_notes` is the right competitor (S1).
   B − C as drafted does not measure it (D1).
2. **`maxim_all_off`.** Not shown to be bio-free, and a within-session point (S6).
3. **The floors.** Reflexion is fair (N2) if the floor sees Body State (S3). ReAct needs a named zero-shot form (N1).
4. **The metrics.** The fire-pit metric scores approach, not need-conditioned choice (S2). The deceptive gates are
   wrong-signed (D2). Both need engagement conditioning and first-contact on deceptive (D3).
5. **The dose.** Defensible only as a reported curve over measured content, and confounded by in-session decay,
   session-end decay, enrichment habituation and wall-clock pacing (S4, S5).
6. **The ablations.** Dropping all three is wrong for a row whose mechanism column names them. One combined
   valuation-off arm, or a re-worded mechanism column, is owed (S1).
