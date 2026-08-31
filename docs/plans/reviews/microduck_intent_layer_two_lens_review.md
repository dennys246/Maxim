# Two-lens pre-merge review — microduck_intent_layer.md

**Date:** 2026-08-31. **Lenses:** Executor/correctness, Architecture/maintenance (two parallel
reviewers, per the pre-merge review-round discipline). **Reviewed at:** commit `2a835e24` on
branch `docs/microduck-intent-layer` — the rev 1 draft, docs-only
(`git diff origin/main...2a835e24 -- src/ scripts/ tests/` is empty).

**Headline: the framing survives, both headline recommendations do not.** The document's method —
take operator-supplied constraints, verify them against the code, correct the operator's own
premises where they are wrong, propose zero code — is right, and the Architecture lens called §0's
explicit *input vs. derivation* split the best thing in the diff. But rev 1 had converted a **1.3
design exploration into two decisions binding 1.1.3**, and the roadmap fold recorded them as
decided. Both were withdrawn in rev 2. Separately, the Executor lens found one roadmap claim
**false and self-contradictory**, and found that a bound rev 1 stated correctly rested on a reason
that does not hold.

Findings ranked. Rev 2 folds all BLOCKING and SHOULD-FIX findings and every NIT.

---

## BLOCKING

**B1 — [Architecture] The gate-7 fold pre-decided a question the Oasis case study already decided
the other way, leaving two active plans contradicting each other.**
[oasis_case_study_taught_orient.md](../oasis_case_study_taught_orient.md) §1 had front-gated
exactly this choice, offered both options — (a) typed bundles `manifest.body_ref` +
`manifest.affordance_namespace`, (b) body-agnostic keys on the SEM modulator/affordance — and
picked (a) with a stated reason: *"(b) is the better long-term key but changes what every existing
NAc file means; (a) is the honest first step and is what Exp 53b actually did."* Rev 1's roadmap
fold selected (b) and did not touch the case study, so after merge the roadmap (declared SOLE
SCOPE AUTHORITY) and the case study (where gate 7 says the decision is made) would have given
opposite guidance. It also **expanded 1.1.3**: that release ships on **D44**, a behavioural delta
across a merge between two independent agents; D43 names `agent_id` and `cluster_id` as the axes
that miss and `tool_signature` as the *third* barrier, and two independent agents **on the same
body** produce identical tool signatures — so (b) is not on D44's critical path. And the urgency
was false: `hivemind/bundle.py::register_bundle_migration` / `::migrate_bundle_envelope` already
exist as reserved hooks, documented as making a schema bump the only change needed for older
bundles to load, so (a)-then-(b) is a supported path rather than a shim to fear. *Verified
independently before folding.* **Rev 2:** §2.4 demotes the recommendation to a costing request;
the roadmap row restores (a) as the scheduled choice.

**B2 — [Architecture] §2.3's "promote `MotorStep.sem_key`" was a large refactor described as a
reframing, justified circularly.** `sem_key` has **exactly two references in the tree**, both
inside `embodiment/motor.py` — it is not a load-bearing identity, so "promoting" it means
*building* one. And rev 1 justified it as "work gate 7 already owes", which asserts the thing under
debate. The uncosted blast radius: 20 persisted `53_agents` agent files whose `aut_nac.json` keys
embed the flat string (`links`, `outcome_index`, and unit-separator bias triples);
`hivemind/bundle.py::_scrub_event_signature`, which parses it; `MAXIM_SUBSTRATE_TOOL_WHITELIST`'s
**substring** filter, used by `exp49/run_trials.py` and `benchmark_cradle_mother.py` — harnesses
behind EARNED rows; and `porting_orient_loop.md`, which pins YAML action names as the cross-robot
contract. The lens also noted rev 1 under-cited its own best support:
`runtime/tool_dispatch.py::build_tool_signature` declares itself "the single source of truth for
tool→NAc event signature format", a stronger anchor than `similarity/signature.py`. **Rev 2:**
§2.3 is retitled as an option, carries the blast radius, and cites `build_tool_signature`.

**B3 — [Executor] The roadmap fold's "no raw motion dispatch survives in
`embodied_runtime/movement.py`" is false, and self-contradictory.**
`Movement._enqueue_sdk_look_at` binds `self.mini.look_at_image`; the file's own comment says it
*"bypasses `ReachyMiniController.look_at_pixel`, so the controller's last-commanded head stash
never sees the motion"*, and CI's `RAW_SDK_MOTION` guard carries **two explicit allow-list lines**
for that file. The sentence also refuted itself — "no raw motion dispatch survives … one
`look_at_image` wrap" — where the wrap *is* the dispatch. This mattered because the paragraph's
conclusion ("media-abstraction job, not motion-safety; materially safer and cheaper") is what a
future reader acts on. *Verified independently before folding.* **Rev 2:** the roadmap says the
work is mostly media abstraction but must relocate the allow-listed motion site and its CI
allow-list, and that its review round is a motion-safety round.

---

## SHOULD-FIX

**S1 — [Executor] §5.2's bound on the `focus_on_sound` defect was right in conclusion, wrong in
reason.** Rev 1 said the graduated experiments suppress the tool-success floor via
`drive_relief_only` / `drive_credit_withheld` / `MAXIM_OPERANT_ONLY_CREDIT`. **None of those flags
appear in the Exp 45/53/53b/54 harnesses.** The actual reason is stronger: those harnesses do not
route credit through `record_outcome` at all — Exp 53/53b/54 are readout-only by their own module
docstring, Exp 45 and Exp 52 Phase A call NAc directly, and only Exp 52 Phase B runs the full
dispatch (and does set the flag). Further, `MAXIM_OPERANT_ONLY_CREDIT` nulls **only** the
cluster-reward term; `learn_success`, the `Valence.POSITIVE` causal link and `credit_goal(+1.0)`
fire regardless — so the bound is over the cluster-bias surface, not the substrate as a whole.
**Rev 2:** §5.2 carries the per-experiment table and the narrower bound; the same correction was
applied to the D51 ledger row, which rev 1 had filed with the wrong reasoning.

**S2 — [Architecture] §5.2 routed an unexecutable-intent signal into a *harm* channel and would
have inverted the bug it fixed.** `side_effects["embodiment_failures"]` means the entity's own
components failed; a clamp is the controller correctly **refusing** an out-of-workspace command.
Worse, `learn_success = success and not embodiment_failed` is unconditional, so every clamped
motion would book NEGATIVE — including a turn that clamped at 40° of a requested 60° and
nonetheless centred the sound. The doc's own closing paragraph named the right home
(`proprioception/pain.py::PainDetector::_check_movement_failure`, graded, unwired on the
`agent_loop` path) and buried it under the wrong one. **Rev 2:** the `embodiment_failures` route is
explicitly rejected, the `PainDetector` revival leads, and the credit rule is stated to key on the
*outcome* (`potential_diff`), not on clamp-occurrence. D35 is cited, with its scope pinned as
harness instrumentation rather than a production path.

**S3 — [Executor] §6.3's trial budget was derived from a learning rule the document tells the duck
not to use.** The RW arithmetic is exactly right (α = 0.14 from `base_learning_rate` 0.2 × novelty
0.5, `0.86ⁿ ≤ 0.1` → n ≈ 16), but §5.1 and §5.3 both commit the duck to
`NAc.credit_operant_reward`, a wrapper over `update_cluster_reward` whose rule is a **linear
accumulate-and-clamp** (`+= reward_bias_alpha (0.15) × reward`, cap ±1.0) with no exponential
approach — saturating in ~7 trials, where the behavioural quantity is *separation* between
competing actions, not convergence. The Exp 45 cross-check compounded it: "1.00 in ~10 trials" is
argmax-correctness across bins, not 90% of an asymptotic link value. **Rev 2:** both rules are
given, the duck's committed surface leads, and Exp 45 is demoted to an order-of-magnitude sanity
check.

**S4 — [Executor] §5.1's table implied `credit_operant_reward` and `update_cluster_reward` are
different mechanisms.** The former is a thin wrapper over the latter and inherits the identical
0.15 attenuation and ±1.0 cap. Load-bearing for the duck: a graded reward of 0.02 moves the bias
by 0.003 per trial against a `causal_pos` term near 1.0 — the same drowning failure §5.1 diagnoses
for the sign collapse, one layer down. **Rev 2:** rows merged with the attenuation stated.

**S5 — [Executor] §3's tool-name collision hazard had the direction backwards.**
`_resolve_tool_name` starts at `candidate = base_name` and prepends ancestors only `while candidate
in existing_names`, so the **first** registrant keeps the plain name and the **later** one is
renamed. The hazard is a body that learned keys solo and then registers second. **Rev 2:**
corrected, with "pin registration order" as the mitigation.

**S6 — [Executor] §5.3's "7 of 10 POSITIVE producers" does not reproduce.** Ten assignment sites,
but one is inside a class docstring example, and **5 of the remaining 9** mean "returned without
raising". The qualitative conclusion is unaffected. **Rev 2:** corrected to 5 of 9.

**S7 — [Executor] §5.4's "one tuple entry" understated the work** in a section otherwise devoted to
warning that the seam is not free. A `ModalityChannel` also needs a `read_values`/`read_ranges`
pair, and anything riding the existing extero readers is filtered through the hardcoded
`_EXTEROCEPTIVE_ROOT_SENSORS = ("azimuth",)`. The Minecraft plan names both obstacles; rev 1
carried only the second. **Rev 2:** both named, in §5.4 and the front-gate table.

**S8 — [Executor] §2.1's "zero occurrences" is precise only for the narrow shape.** No
`body:<name>:<verb>` exists — true. But a colon-delimited `skill:`-prefixed key does exist in
`memory/concept_extractor.py` (consumed by `memory/concept_context.py`), in the concept graph
rather than the action namespace. The derived recommendation is unaffected. **Rev 2:** exception
noted.

**S9 — [Architecture] The "Owns (proposed)" line contradicted the document's own conclusions**, and
there was no named front-gate section though every sibling design doc has one. Rev 1 claimed
ownership of three mechanisms the body concludes all ride existing seams. **Rev 2:** a
`## Front-gate scope pressure` table was added, per constraint; "Owns" now names only the reduced
proprioceptive summary.

**S10 — [Architecture] The defect should be filed now, not deferred.** Ledger rule 1 reserves the
plan's open-questions section for *suspicions*; this one is verified against two named symbols, so
the plan doc was the wrong home by the ledger's own rule — and "file after #577" named no trigger
and no owner. **Rev 2:** filed as **D51**, deliberately skipping D49/D50, which the Executor lens
confirmed is exactly where #577 moves the duplicated documentation-truth rows.

**S11 — [Architecture] No revive/schedule trigger**, though the doc describes itself as unscheduled
and blocked on an unanswered operator question — the condition the README's deferral rule exists
for. **Rev 2:** a `Schedule trigger:` line was added to the header.

---

## NIT (all folded)

- **[Arch]** §0 declared itself "not arguable" while containing a premise §2.1 refutes → the clause
  is now marked inline as corrected, preserving the verbatim record.
- **[Arch]** §2.6's per-layer reasoning is sound but lived only in the doc that does not own the
  commitment → a per-layer note was added to
  [porting_orient_loop.md](../../embodiment/porting_orient_loop.md) §"When robot #2 arrives".
- **[Arch]** §5.4's shared recalibration job was described in two plans and owned by neither →
  ownership assigned to the roadmap's 1.1.4 row.
- **[Arch]** The README index stated recommendations as directives → rewritten to describe the doc.
- **[Arch/Exec]** Volatile counts and verbatim code rot → §5.1's quoted line replaced with a
  symbol reference, §6.3 given an explicit re-derive trigger, the roadmap's `.mini` counts dated
  and led by the durable distribution finding.
- **[Exec]** `last_clamped_axes`'s consumer is `tools/reachy.py::MoveTool`, not `FocusOnSoundTool`.
- **[Exec]** The 12-method freeze statement lives in `RobotController.get_doa_reader`'s docstring.
- **[Exec]** `cerebellum_modulator_factory` is a second `attach_backends`-shaped factory (dormant).
- **[Exec]** `MovementSample` has 7 fields (timestamp + 6-DOF pose).
- **[Exec]** BL-1 is a body-rotation/head-matrix defect, not a visual-detection one → "most of
  BL-1..BL-5".
- **[Exec]** `record_percept_valence` accepts an unvalidated float; `[-1, 1]` is the stored range.

---

## Confirmed clean — what both lenses endorsed

Cross-confirmation is the repo's trust signal, and several dimensions came back clean from the
lens that owns them.

- **§0's input/derivation split** — Architecture called it the best thing in the diff and worth
  copying into future plan docs.
- **§2.1's core correction** — independently re-verified by both lenses:
  `tool_bridge.py::generate_tools_for_entity` builds `f"{ent.name}_{aff_name}"` with the modulator
  dropped; `affordance_namespace` is docs-only; `hivemind/bundle.py`'s manifest carries no body or
  namespace field; the `MAXIM_SUBSTRATE_TOOL_WHITELIST` comment does read "BAND-AID (tracked)".
- **§2.2's inventory** — all confirmed: 12 abstract methods, `MotionTarget.extras` with no
  subclassing, `maxim.robots` entry-point discovery, `attach_backends`'s exact factory shape,
  `make_reachy_orient_factory`'s `mod_name != "orient"` gate, `MotorStep.sem_key`'s triple.
- **§2.5's refusal to widen the frozen ABC** — architecturally correct.
- **§2.6's per-layer second-consumer reasoning** — "sound, and the right application of the test,
  not an evasion."
- **§5.2's core finding** — `last_clamped_axes` has one `src/` consumer and **no path to PainBus,
  NAc, or any modality channel**; `PainDetector`/`JointLimitHarmPredictor` are reachable only via
  the legacy Selfy runtime, not the `agent_loop` path.
- **§5.3's structural claims** — no reward bus, `build_pain_bus`'s required kwargs and three
  auto-subscribed learners, zero `subscribe("reward", …)` anywhere, `credit_operant_reward`'s
  producers are harnesses and never the runtime.
- **§5.4's selection-dynamics warning** — the registry comment matches verbatim; `place_code`
  replaces rather than augments; the `AUDIO_TAG`-then-`sorted()` fallback is exact.
- **§6's harness facts** — `scripts/exp54/` does not exist; no `.reset()` anywhere; the
  `YOKED_SEED_OFFSET`, the `MECHANISM SANITY` gate that prints "VOID — apparatus, not a result",
  `JsonlLog`'s preflight, and `LiveRig`/`DryRig`'s location all confirmed. (The microduck doc has
  `LiveRig`/`DryRig`'s current location right where `porting_orient_loop.md` is stale.)
- **§7's "the engineering got easier; the science did not"** — holds; not rationalizing.
- **§8's refusal to allocate an experiment number** — correct, and should be defended.
- **§1.1's calibration on the audio unknown** — "not proof of absence" is the right strength.
- **Convention compliance** — no `file:line` citations anywhere; all cross-doc links resolve.

**Could not verify:** nothing material. The Executor lens declined to adjudicate "~16–17
substantive" `.mini` uses since *substantive* is undefined (its own count lands near 20); rev 2
therefore leads with the distribution rather than the total.
