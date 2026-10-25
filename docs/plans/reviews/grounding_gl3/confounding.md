# GL3 design review — CONFOUNDING lens (thalamic_relay.md)

Reviewed: `docs/plans/thalamic_relay.md` (all 1,217 lines) in the `e-t8b` worktree at origin/main
`d9dffe6e`. I also read the charters (`docs/experiments/DESIGN_REVIEW.md`), the umbrella's GL3 rows and
owner decisions (`docs/plans/grounding.md`), the GL1 confounding report for format,
`docs/wiring/harness-loop-must-be-proven-live.md`, and the built GL3.B0 code:
`tests/unit/test_receptor_census.py`, `tests/unit/test_afferent_red_gates_units.py`,
`tests/unit/test_afferent_red_gates_loop.py`, `tests/unit/test_afferent_latency_characterization.py` and
its fixture. On the `src/` side I read `runtime/loop_state.py::_substrate_tick_due`,
`runtime/loop_substrate.py::substrate_tick`, `runtime/substrate_proposal.py::propose_via_substrate`,
`runtime/loop_gates.py::tick_embodiment_drift`, `embodiment/body.py::evaluate_failures` (the drive
latch), `embodiment/tool_bridge.py::ModulatorAffordanceTool.execute`, `proprioception/pain.py`
(`classify_pain`, `failure_pain_kind`, `TISSUE_DAMAGE_DRIVES`), `similarity/encoder.py::SensorEncoder.encode_sensors`
(its eligibility write) and `decisions/nac.py::DEFAULT_CLUSTER_FEAR_FAILURE_MODES`. I ran three read-only
probes: the four GL3.B0 test files (9 passed, 8 xfailed, 18 s); a fresh `measure_all()` of the
characterization on today's tree, compared with the frozen fixture; and `run_d1()` with its trace
printed. I did not re-open G1–G20 or TR1/TR2. Where a finding touches one of them, it is about which
number a gate reads, not about the decision.

The question for every gate: could a pass, or a fail, come from something other than the claimed cause?
And can the gate fail at all?

**Verdict: 4 DO-NOT-BUILD / 10 SHOULD-FIX / 5 NIT.**

Recommended reading order: D1 (it puts the plan's headline timing claim in doubt), then D4, D3 and D2
(they block GL3.B3 and GL3.B1, the near-term stages), then S1 and S2 (the GL3.B0 gates that are already
built), then the rest.

- DO-NOT-BUILD **GL3.B4**: the H2 preemption check reads an event that, in the stage's own scenarios,
  does not exist yet when H2 runs. Any latency gain must then come from an undeclared transduction or
  reselect, which is the L1 fix (or a cadence bypass), not preemption.
- DO-NOT-BUILD **GL3.B1**: deletion probe 5 ("bypass the registry and gate 1's golden fails")
  contradicts gate 1 (that golden is byte-identical with or without the registry). That leaves no gate
  showing the registry is the encode route.
- DO-NOT-BUILD **GL3.B1**: the encode-time eligibility change (imagined none, narrated discounted) is not
  behind the flag, the narrated weight does not exist yet, and none of the cited goldens runs that path.
- DO-NOT-BUILD **GL3.B3**: with the flag on, `drive:oxygen` (PainKind DRIVE) leaves the W-4 in-place
  order, or leaves PainBus altogether. That is the Exp 60 learning signal, and no flag-on gate exercises
  drowning.

---

## DO-NOT-BUILD

### D1 (GL3.B4): the preemption check cannot see the breach it is meant to preempt on; the latency gain belongs to whatever transduces first

**Issue.** H2 runs "immediately before §4 executes `ctrl.pending_proposal`" and asks
`preempted_since(proposal.basis_seq, "pending_proposal")`, which is true only if an event on
`nociceptive_fast` arrived after `basis_seq` (§5.5). In both scenarios GL3.B4's gates use, no such event
exists at H2.

- **Who transduces a Minecraft health drop.** In substrate-primary, `evaluate_failures` runs inside
  `substrate_proposal.py::propose_via_substrate` (after encode and `note_active_clusters`), which runs
  only when `loop_state.py::_substrate_tick_due` holds. That predicate requires
  `ctrl.pending_proposal is None` and the 0.5 s cadence. `loop_gates.py::tick_embodiment_drift` returns
  early on substrate-primary. The sync pump never calls `evaluate_failures` (§3.2), and the harness's
  `_ScriptedWorld` has no edges anyway: `clock.on_advance = aut.backend.sync_world_sensors` writes the
  sensors only. The one other caller is the stale action itself: `tool_bridge.py::ModulatorAffordanceTool.execute`
  calls `self._embodiment.evaluate_failures()`, and that happens **inside** §4, after H2.
- **Probe, gate (d)'s world (`run_d1`, today's tree).** Proposal `eat` at pass 0. Health drops in the
  sleep. The stale `eat` executes at pass 1 (success), and the **only** `drive:health` publish is at pass
  1, emitted by that stale execute. At H2 of pass 1 the scheduler would hold no event with
  `seq > basis_seq`.
- **The frozen damage arm** (the gate-3 baseline,
  `tests/fixtures/afferent_latency_characterization_v1.json`): the stale `mine_block` is refused by the
  executor ("Missing required input"), so nothing transduces until the next substrate tick at pass 3
  (`publish_latency_passes: 2`), and `flee` follows at pass 4. At H2 of pass 1 there is again no event.
- **So, as designed, H2 never fires on the L1 case.** Gate 2 ("the pain event's `seq > basis_seq`,
  reselection in the same pass") and gate 3 ("latency strictly lower than the frozen record") can pass
  only if the build adds a transduction before §4 (a second `evaluate_failures`, or a reselect that runs
  `propose_via_substrate` outside its cadence). §4.2 names exactly that as the forbidden special-case
  shape ("re-running `evaluate_failures` in a second spot before §4").
- **Whatever then lowers the latency is not preemption.** It is either the extra transduction, which is
  the L1 fix itself, or the reselect bypassing the 0.5 s cadence (`llm_submit_interval` against the
  harness's `target_hz=4`, which is what sets the frozen +2/+3). Deletion probe 5 ("remove the in-pass
  reselect and (3) goes red") confirms that the reselect, not the check, carries the effect. If #1176's
  L1 fix lands first, as the plan allows ("if the L1 issue has not already flipped it"), it can take the
  whole gain. Gate 3, measured against a record frozen before that fix, would then pass on the L1 fix's
  effect, while deletion probe 5 fails for a reason unrelated to the arm.

**Why it matters.** This is the plan's headline timing claim ("A breach that begins after the substrate
has selected an action cannot stop that action", §Why). A declared rung arm would read a latency drop
that preemption did not cause.

**Fold (plan edit, before GL3.B4's gate design pass).**
1. Say in §5.5/§5.6 which producer emits the event H2 reads, in the damage arm and in gate (d)'s world.
   If the answer is "a transduction step before §4", say that it is the L1 seam (#1176), and decide
   whether GL3.B4 is the L1 fix (then the "special case" language in §4.2 must change) or comes after
   it. Strict option first: **GL3.B4 enters only after #1176 has settled where transduction runs, and
   it is measured against a baseline taken with that fix in place.**
2. Make gate 3 a **paired same-tree comparison**: flag on against flag off, on the GL3.B4 tree. Keep
   the frozen record as a stated drift check. My probe shows today's tree still reproduces the record
   exactly (both arms, every field), but nothing re-checks that once `src/` moves: the test checks shape
   only off the measured tree (`_src_is_the_measured_tree`).
3. Add a **sham control**: drop and reselect at the same pass with no qualifying event (or with an event
   below `preempt_threshold`). If the sham lowers latency as much as the real arm, the gain is the
   reselect or cadence, not preemption.
4. Gate on passes **and** step-clock seconds (the record carries both; idle passes are 0.05 s, running
   passes 0.25 s), so that a change in idle passes cannot fake a pass-count win.

### D2 (GL3.B1): the deletion probe contradicts the byte-identity gate, so nothing shows the registry is the encode route

**Issue.** Gate 1 requires the `(modality, cluster_id, geometry_tag, margin)` sequence to be
byte-identical before and after the registry. Gate 5's first deletion probe says "bypass the registry
and gate 1's new golden fails". If gate 1 holds, bypassing the registry (calling the old direct path)
produces the same bytes, so the probe cannot go red. If the probe goes red, gate 1 is false. The
umbrella repeats the same pair (grounding.md GL3 row: "deletion probe (bypass the relay) fails the
golden").

**Alternative explanation the gates admit.** Suppose a registry is built, writes its `receptor:*`
stamps (gate 2 sees them), and is never on the encode route: `_encode_current_clusters` still calls
`encode_sensors` directly, or calls both. That build passes gates 1–4. The harness-loop lesson
(`docs/wiring/harness-loop-must-be-proven-live.md`: "a mechanism that does not run looks exactly like
one that ran and found nothing") applies directly. M39's lint is the structural guard, but §5.10 leaves
it a process invariant "until GL3.B1 ships".

**Fold.** Re-point the probe at something the registry alone produces. Strict option first:
**(a)** M39 (`scripts/lint_relay_only_encode.py`) lands in the GL3.B1 PR, and the deletion probe is "add
a direct `encode_sensors` call outside `perception/` and M39 fails". Also add a per-receptor route
counter (the registry's own stamp count per `tick`), asserted equal to the number of encodes the
golden's trajectory performs, and a bypass makes it zero. **(b)** The weaker option: the probe deletes
the stamp step and gate 2's golden diff fails. That proves only that the stamp ran, not the route. Fix
the umbrella's GL3 row in the same edit.

### D3 (GL3.B1): an unflagged credit-path change, with an undefined weight, that the byte-identity goldens cannot see

**Issue.** §5.7 item 3: "`update_eligibility` is called at full weight for `experienced`, at the
declared discount for `narrated`, and not at all for `imagined`." Only item 4, the credit **check**,
"ships behind a flag that defaults OFF". Red gate (a) flips at GL3.B1 with no flag set, so the
encode-time change is meant to be live by default. Three problems:

- **The narrated weight does not exist at GL3.B1.** G6 sets its value at GL4 start. The narrator's tools
  write body sensors, and the body channel's pull read goes through `SensorEncoder.encode_sensors`,
  which calls `self._nac.update_eligibility(...)` (verified, `similarity/encoder.py`, the "NAc
  eligibility update" block). Under §5.7 item 1's strict mixed-state rule, every body-channel encode
  after a narrator write is `narrated`. So GL3.B1 either cannot be built as written, or it changes
  eligibility in every `--sim` run with the narrator active, at a value nobody has chosen.
- **The goldens cited as proof of "nothing moved" do not run this path.**
  `test_agent_loop_selection_golden.py` runs only the `_loop_harness` Minecraft arms (`shore`,
  `fear_water`): no narrator, no imagination, and no eligibility in the trace. The encoder golden pins
  `_sensor_embed`. The GL3.B1 gate 1 fixture records `(modality, cluster_id, geometry_tag, margin)`, not
  eligibility. A credit change would land under a "byte-identical" banner.
- **The blast radius is mis-timed.** §6 GL3.B1 and §7 put T1-13's "`TemporalCreditDistributor`
  credit-path change" and T1-14's `credit_node` trigger at "when the check is on". An encode-time
  eligibility change fires them at build. Which EARNED rows actually run imagination or narrator writes
  is **UNVERIFIED**: the Minecraft-harness rows (Exp 60/61/62) likely do not, while `--sim`
  cradle/E-rung runs do.

**Fold (owner decision, strict option first).** **(a)** Record provenance always, but act on it
(eligibility weights and the credit check) only behind the one credit flag, default OFF. Red gate (a)
then flips under flag-on, which is an owner-approved re-point that GL3.B0's banner requires to be
recorded. Add an **eligibility golden**: per-node eligibility after a scripted `--sim` trajectory with
one narrator write and one imagination encode, byte-identical with the flag off. **(b)** Keep imagined
eligibility off unconditionally, as an invariant (TR12 strict), but leave narrated at full weight
until the G6 value exists. State this as a behaviour change, walk T1-13 at build, and add the same
eligibility golden for the narrated half. Either way the plan's header promise ("every behaviour change
ships behind a flag that defaults OFF") must match what the stage does.

### D4 (GL3.B3): flag on, the Exp 60 drowning signal leaves the W-4 order (or leaves PainBus), and no flag-on gate looks

**Issue.** GL3.B3: "NOCICEPTIVE fans out to fast + slow, DRIVE goes to slow". `drive:oxygen` is DRIVE
(`proprioception/pain.py::failure_pain_kind`; `TISSUE_DAMAGE_DRIVES = {"drive:health"}`). It is also
one of the two Wire-4 cluster-fear failure modes (`decisions/nac.py::DEFAULT_CLUSTER_FEAR_FAILURE_MODES`).
`affective_slow` has `latency_passes = 2`, and H3 drains **only latency-0 tracks** in place. So with the
flag on, the drowning breach no longer runs through "encode → `note_active_clusters` → pain → Wire-4
fear write" in one call. Its delivery is two passes later at H4 (§8.5), against whatever clusters were
noted last, which is exactly the Exp 58 W-4 hazard ("or the fear lands on the wrong cluster"). Worse,
the plan contradicts itself on whether DRIVE pain reaches PainBus at all under the flag. §5.4's table
lists `affective_slow`'s destinations as autonomic, credit and prompt, with **no `pain_bus`**, while
§5.8 says PainBus is "the `pain_bus` destination of ... `affective_slow` (DRIVE)". If the table wins,
flag-on drowning publishes no Wire-4 fear at all.

**Which gates see it.** None. Gate 1 is flag-off. Gate 2 is one scripted burn (NOCICEPTIVE). Gate 5 is a
flag-on golden whose content is new by definition. GL3.B4 (which runs with tracks on, M10's
`runtime.afferent_tracks`) compares latency on the **damage** arm only. A rung arm declaring GL3.B4
would therefore mix preemption with re-timed (or deleted) drowning fear, and the drowning arm, the one
built on an EARNED row (T1-13), is the one never measured.

**Fold (plan edit before GL3.B3).** Resolve the §5.4/§5.8 contradiction in writing. Strict option first:
**DRIVE breaches keep their synchronous PainBus publish inside H3 (in place, W-4 order unchanged), and
`affective_slow` adds a timed delivery to the autonomic and prompt destinations only**, mirroring the
T1-4 "tracks add beside, never replace" rule. Add flag-on gates on the drowning arm: the Wire-4 fear
write lands on the same `(agent, cluster)` with the flag on and off, and breach-to-`escape_water`
latency is no worse with the flag on (both in passes and in seconds). The alternative, moving DRIVE
pain timing, is a selection-dynamics change. It needs its own owner decision and the T1-13/14/15 rig
re-runs.

---

## SHOULD-FIX

### S1 (GL3.B0, built): "fails today" proves that a gate can fail, not that only the claimed fix flips it. (a), (d) and (g) have no positive control, and (e) does not pin TR2's action half

**Issue.** The plan offers "All seven red gates fail today, which proves none is vacuous" (GL3.B0 gate
text). That shows each gate can fail. It does not show that a flip means the claim. Gates (e) and (f)
carry controls (`test_d2_control_a_free_loop_publishes_the_drift_breach`,
`test_d5_control_an_open_gate_fires_the_reflex`). The others do not:

- **(a)** reads `nac._eligibility[("aut-1", nid)] > 0` (a private dict). It XPASSes if every encode stops
  writing eligibility (a fix that kills all credit), or if eligibility storage is renamed or re-keyed in
  any NAc refactor. There is no control showing that an `experienced` encode through the same encoder
  still writes a positive trace on the same read surface.
- **(g)** XPASSes if every node is stamped `narrated`. There is no control showing that a drift-only
  encode (no narrator write) is stamped `experienced`.
- **(d)** XPASSes if the pending proposal is dropped for **any** reason (expiry, an approval change, the
  same-tool cap, a decomposition change to §4). There is no no-hurt control showing that, absent damage,
  the same proposal does execute at pass N+1. The probe shows it does today, so the control is free to
  add.
- **(e)** encodes TR2 ("the budget delays only action"), but the body rig is action-free in both runs
  (`assert not trace.calls` in the control). A fix that runs the whole `propose_via_substrate` on a
  denied tick, selection included, flips (e) while breaking TR2's action half: `gated` is computed from
  the predicate in `loop_substrate.py::substrate_tick` and would stay True.

**Fold.** Add a green control beside each: (a′) an experienced (own-sensor) encode has eligibility > 0
on the same surface; (g′) a drift-only encode carries `experienced` and not `narrated`; (d′) with no
hurt, tick k's proposal executes at pass k+1; (e′) on a body whose free run does act, the denied run
installs no proposal and makes no executor call. Change the plan sentence to "each fails today, and
each has a control that a non-claim fix would break". These are test-only edits inside the fence. Where
an assertion inside a strict red gate changes, record it as the owner-approved change that GL3.B0's
banner requires.

### S2 (GL3.B0 → GL3.B4): one claim, two worlds; and gate (d)'s world has no damage-contingent pick

**Issue.** GL3.B4 gate 2 reads red gate (d) (world: `shore` seeded with `eat`, which executes). Gate 3
reads the frozen damage arm (world: `shore` seeded with `mine_block`, which the executor refuses). The
characterization's own docstring says the damage arm's +2 publish latency "depends on the stale
`mine_block` being REFUSED". In (d)'s world, a stale action that really runs transduces the breach
itself at pass 1. My `run_d1` probe also shows that (d)'s world **never picks a protective action**:
ticks at passes 3 and 6, after the `drive:health` pain at pass 1, pick `eat` again, and the seeded
reward dominates `flee`. So a (d) flip shows "the proposal was re-evaluated", never "the damage changed
the action". A reselect that picks `eat` again flips (d), because the new proposal's `proposal_tick`
differs from k.

**Fold.** Define one GL3.B4 scenario in which the stale action is a body affordance that really runs
**and** the post-damage pick differs (protective). Use it for gates 2 and 3, and add a frozen record
for it (a v2 fixture beside v1; v1 stays as history). State in §6 GL3.B0 that (d) pins re-evaluation
only.

### S3 (GL3.B4): the livelock probe counts the wrong unit; a deepening breach mints a new pid on every re-publish

**Issue.** The guard is "at most once per physical event; refractory per locus" with
`refractory_passes = 1` per (receptor, locus), probed by "a standing breach over 20 passes preempts at
most once per physical event". `embodiment/body.py::evaluate_failures` latches drive pain on band entry
and **re-publishes whenever the breach deepens** (`severity > latched + eps`). Drowning deepens by one
bubble a second. Each deepening is a new emit, so a new pid. Each pid may preempt once, and the per-locus
refractory allows that every other pass. The protective action selected by the reselect then has its own
`basis_seq` and is pre-empted by the next deepening. The probe passes (once per pid) while `escape_water`
never executes. With a truly *standing* (non-deepening) breach, only one event is ever published, so the
probe is vacuous.

**Fold.** Probe on a **deepening** breach (the drowning arm). Count preemptions per breach episode, not
per pid. Assert that the protective action executes within N passes of the first preemption. State
whether a preemption may pre-empt a proposal that was itself produced by a preemption reselect (the
strict default: no).

### S4 (GL3.B3): the fan-out can double-count one burn in learning, and deletion probe 3 is ill-posed

**Issue.** `nociceptive_fast` delivers to `pain_bus`, "the learning subscribers, unchanged": memory,
NAc outcome, Wire-2 and Wire-4 cluster fear. `affective_slow` delivers to "credit (cluster fear, memory
tag)". One physical event can therefore write memory and fear twice, once now and once two passes
later. Any later "stronger aversion under tracks" would be a repetition artefact. Gate 2 checks that the
two deliveries share a pid, not that learning is applied once per pid. Deletion probe 3 ("removes the
affective reading only, and the fast path's goldens are unchanged") can fail legitimately, because the
slow track's fear write feeds selection, or pass vacuously if the slow credit destination is empty.

**Fold.** Add an invariant and gate: each learning destination (memory, NAc outcome, cluster fear,
eligibility) applies **at most once per pid**, asserted by a count over the scripted burn. Re-word
probe 3 to name exactly which destinations the slow track owns and what changes when it is removed.

### S5 (GL3.B1): the per-node "ever reinforced" set makes the credit lock rarely bite, and the mixed-node case is ungated

**Issue.** §5.7 item 2 records every kind that has ever reinforced a node. Item 4 refuses only nodes
with no creditable kind and caps "narrated-only" nodes. In a `--sim` run, narrator writes and drift both
reinforce the same frozen-centroid interoception/world clusters (0.85 threshold), so most nodes become
`{experienced, narrated}` and get full credit: the G6 discount is almost never applied. The strict
mixed-state rule ("narrated whenever any declared sensor's last writer was the narrator") leaves "last
writer" undefined for drift (`tick_vital_drift` writes the same slots). Depending on that, one narrator
write either taints every later read in the run or is erased on the next tick. Gate 4 tests only pure
nodes (imagined-only, narrated-only). GL5's "with AND without narrated data" (G6) cannot be built from
node-level unions. The label would then be driven by the labelling rule, not by the event.

**Fold.** Define, before GL3.B1's gate-design item closes: (i) whether drift is a writer; (ii) the
credit rule for mixed nodes, with the strict option first: credit the creditable share by
per-reinforcement counts, not by membership; (iii) a gate on a mixed node; (iv) the fraction of mixed
nodes on a recorded `--sim` trace, reported before the credit flag can flip.

### S6 (GL3.B1): red gate (b) is scheduled to flip with work that is not in GL3.B1's build

**Issue.** GL3.B1 gate 3 requires (a), (b) and (g) to flip. (b) reads the Minecraft event percept's
`agent_id` through `build_minecraft_aut`. GL3.B1's build list does not touch `MinecraftPerceptSource`.
§5.8 assigns it to GL3.B5 ("Text receptors ... and a non-empty agent (GL3.B5)"), and GL3.B5's gate says
"every text percept's agent is non-empty". So GL3.B1 either fails gate 3, or passes it with a one-line
drive-by (`agent_id=` through the builder) that has nothing to do with the registry. Either way the gate
reads something other than the stage's claim.

**Fold.** Re-home (b) to GL3.B5, or to its own F0.5 issue, recorded as an owner-approved change, and
remove it from GL3.B1 gate 3.

### S7 (GL3.B1): gate 1's trajectory must exercise the two things the refactor can silently move

**Issue.** §7 names the risk: "channel order and the active-channel count set `recommend_action`'s
summed cluster term ... plus the empty-read rule". Gate 1's "scripted sensor trajectory" is unspecified.
If every channel reads non-empty on every step, an ordering or empty-read regression is invisible to it.
`minecraft_player` has no azimuth sensor, so the audio channel is always empty there.

**Fold.** Specify the trajectory: at least one step on which each channel goes from reading to empty and
back, on each body; at least one step with all three channels active (`reachy_mini` or a fixture body);
and a deletion probe that permutes the registration order and fails the golden.

### S8 (GL3.B3): gate 6's thread-identity assertion passes if the narrator write is never transduced

**Issue.** "The breach latch and the snapshot slot are touched from the loop thread only" is also true
if the narrator tool simply stops reaching `evaluate_failures`, for example if the inbox post is dropped.

**Fold.** Add the positive half: the narrator-written breach is published exactly once, at the first H0
drain after the write, and stamped `narrated`.

### S9 (GL3.B5): a live-LLM "same as before" comparison cannot isolate the change

**Issue.** "The interactive log check (`MAXIM_LOG_FILE` + `--interactive false`, 3 turns) shows the same
percepts and followups as before". With a live LLM, run-to-run variance alone makes the comparison fail
(different followups), or pass (three turns may never exercise a moved producer).

**Fold.** Make the gate a scripted-source run (`ConversationalSource` + a mocked LLM, or the fixture
runner) that compares the percept sequence byte for byte. Keep the logged interactive session as the
CLAUDE.md smoke (`ACTION_FOLLOWUP` present), not as the gate.

### S10 (GL3.B4): gate 3 tests one arm, in one direction

**Issue.** Gate 3 asks only that damage-arm latency be strictly lower. The flag-on condition also
changes the drowning arm (D4, S3). A stage can pass gate 3 while the drowning arm regresses.

**Fold.** Gate both frozen arms: strictly lower on the arm the claim names, and no worse on the other.
Use both units (passes and seconds), and pre-register both comparisons in the arm's prereg.

---

## NIT

- **N1 (GL3.B0 wording).** "All seven red gates fail today, which proves none is vacuous" → "each fails
  today for its named reason" (see S1 for what non-vacuity also needs).
- **N2 (GL3.B0 (c)).** One reading only. A constant or sentinel pid stamped on both lanes would flip it.
  Use two readings and assert two distinct pids, each shared across its two lanes.
- **N3 (GL3.B0 (g)).** "`experienced` not in provenance" conflicts with §5.7's union semantics on any
  reused node. It holds only because the EC is fresh. Say so in the test docstring, or assert on the
  write's own kind.
- **N4 (GL3.B3 gate 2).** "The slow delivery lands `latency_passes` later" restates `emit`'s definition
  (`due_pass = pass_index + latency_passes`). It pins the scheduler, not a consumer. The joined-record
  half is the part that carries information.
- **N5 (GL3.B0 record).** The drowning arm's `publish_latency_passes: 0` is the cadence's phase (the
  characterization docstring says so), not a mechanism. §6's summary should carry that caveat wherever
  it quotes "+0", so that a later reader does not cite it as "drive pain is immediate".

---

## Verified

- **The GL3.B0 files exist and run as the plan says.** The census (5 tests) passes. Gates (a), (b), (c)
  and (g) are in `test_afferent_red_gates_units.py`, and (d), (e) and (f, two variants) are in
  `test_afferent_red_gates_loop.py`. Every gate is `xfail(strict=True, raises=AssertionError)`, and its
  preconditions raise non-`AssertionError` (`pytest.fail`, `ScenarioBroken`), so a broken fixture fails
  loudly. That is sound design: a broken scenario cannot pass as a red gate. 8 xfailed, 9 passed.
- **The frozen characterization still matches the tree.** It was measured at `7d18347e` (`src_tree
  74a0fa…`), and there are 137 `src/` commits since. A fresh `measure_all()` on `d9dffe6e` reproduces
  every field of both arms exactly. Its test checks equality only on the measured tree, otherwise shape
  only, as the plan says.
- **The (d) stale match is robust.** It matches by object identity and by (pass N+1, tool), and refuses
  ambiguity. A dict copy cannot flip it.
- **The (e) control is real.** The free run publishes the thirst breach on a substrate tick, and the
  substrate cadence is pinned equal across the two runs.
- **The (f) control is real.** With the gate open, `fire_burn` acts on the reflex pass and
  `DamageComponentTool` publishes.
- **§3.4's L1 and L2 mechanics hold.** `_substrate_tick_due` requires `pending_proposal is None` plus the
  cadence. `substrate_tick` skips `propose_via_substrate` whole on a denied tick, but advances
  `last_llm_submit_time` and sends `gated=True` to telemetry.
- **§3.3's W-4 order holds.** `propose_via_substrate` encodes, then calls `nac.note_active_clusters`,
  then `evaluate_failures`, then `anticipatory_threat_need`.
- **`tick_embodiment_drift` is a no-op on substrate-primary.**
- **The drive pain latch is as §3.2 and S3 describe.** It is on the entity
  (`ent.drive_breach_severity`) and fires on band entry, then again on deepening.
- **The PainKind facts hold.** `drive:health` is NOCICEPTIVE (`TISSUE_DAMAGE_DRIVES`). `drive:oxygen` is
  DRIVE. The Wire-4 allowlist is `{drive:health, drive:oxygen}`.
- **`SensorEncoder.encode_sensors` writes NAc eligibility** (so body-channel provenance reaches credit,
  D3).
- **The selection golden covers only the two `_loop_harness` Minecraft arms.**
- **These plan choices are sound for confounding:** `latency_passes` as ordinal and no wall time in
  track decisions (TR7); the sorted single-drain edge inbox; the frozen-record-not-golden rule and its
  provenance blobs; TR3's strict default; `tool:<name>` excluded from the credit check, with a deletion
  probe in each direction; GL3.B2 reporting row counts with no threshold.
- **UNVERIFIED:** which EARNED or PARTIAL rows' runs exercise imagination encodes or narrator writes
  (D3's blast radius). I did not trace Exp 56/57's harness composition.
