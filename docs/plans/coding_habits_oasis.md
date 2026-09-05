# Collective coding-safety habits — Queen-tier safety benchmarking

**Status: ADOPTED 2026-09-05** (decided 2026-09-04/05; grounded in a three-lens parallel
survey — execution infrastructure, hivemind merge semantics, experiment-framework
constraints). Named by the 1.2 row's poison-resistance work
([maxim_hivemind.md](maxim_hivemind.md) §Poison resistance + §Trust topology) and by the
1.3 line for the behavioral claim.

**The scope line, stated once and up front:** what 1.2 gets from this plan is an
**instrument** — the coding gauntlet that decides whether a habit bundle earns Queen-tier
promotion, plus the no-claim infrastructure it runs on. What 1.2 does **not** get is a
behavioral claim about habit transfer. That claim is **Exp 55**, and it is 1.3-line work.
1.2's claim set stays exactly what the roadmap froze: the four-arm sharing benchmark and
the dose–response ladder. Same shape as "Minecraft is the instrument, not a demo" — and
deliberately so, because pulling the experiment forward is enthusiasm-flag #3
([roadmap_1_1_to_1_3.md](roadmap_1_1_to_1_3.md) §Enthusiasm-to-evidence flags) by its
recorded definition.

## 0. Owner intent (recorded so the claim cannot drift)

Maxim returns to the project's roots — safety of code execution and edits, from a
software-engineering perspective — as a *collective* learning problem. Agents work small
coding tasks in a sandbox; unsafe operations produce measured pain; the resulting
aversions become shareable habits through the Oasis; curated habits become part of what a
new agent is born with. The long-term picture is a species that accumulates scar tissue:
the individual touches the fire once, the fleet flinches forever.

Two halves, with different evidence standards:

- **The habit half** (safety norms) is substrate-claimable: does an imported aversion
  change which action `recommend_action` picks at first contact with a contingency only
  the donor experienced. This is the Exp 52 four-arm shape transplanted to a coding body.
- **The skill half** (writing code) is LLM-owned for now and **out of claim scope** —
  under LLM-primary operation the benchmark would measure the LLM (L6). §8 records the
  long-horizon direction without letting it near any arms.

## 1. Placement

| Slice | What | Release home | Claim? |
|---|---|---|---|
| 1 | Code-world infrastructure (§3) | no-claim 1.2.x slice (the 1.1.4 pattern) | **No** |
| 2 | Sharing semantics inside poison resistance: tighten-only merge, the inherent bias class, the coding gauntlet as Queen-promotion instrument (§4) | 1.2 (extends work 1.2 already owns) | **No** — instrument + mechanism, `[engineering]` only |
| 3 | Exp 55 — pre-registered habit-transfer benchmark (§6) | 1.3-line | The claim, when earned |

Why not a 1.2 claim (the four independent rules, so this decision has a paper trail):
the one-frozen-confirmatory-test rule
([minecraft_benchmark.md](minecraft_benchmark.md) §Scope discipline); "choose the
representation before the arms" (a coding body is a new affordance namespace whose
selection dynamics must be re-baselined, not assumed); enthusiasm-flag #3; and the
microduck precedent — new body + new behavior class = experiment work on the next line,
not a port ([roadmap_1_1_to_1_3.md](roadmap_1_1_to_1_3.md)).

Front-gate answer (CLAUDE.md working principle 3): almost everything rides existing
infrastructure — the tool registry, both sandboxes, three pain layers, the ToolPainBridge
learning loop, and the 1.1.4 world seam. The genuinely new pieces are named per-slice
below, each with its "existing infrastructure X cannot do this because Y" sentence.

## 2. What exists today (survey summary)

**Execution side — mostly built.** Real coding tools exist and are registered
(`tools/filesystem.py`: `bash`, `write_file`, `edit_file`, `execute_file`;
`tools/code_tools.py::RunTestsTool`, `CodeSearchTool`), gated by `allowed_dirs` +
`MAXIM_ALLOW_*` env vars. Two sandbox backends
(`simulation/sandbox.py::TmpdirSandbox`/`DockerSandbox`) plus a stronger host-side
`utils/sandbox_executor.py::SandboxExecutor` (rlimits + Python import allow-list). Three
pain layers already classify coding tools (`runtime/pain_interceptor.py`,
`runtime/fear_gate.py::FearGatedExecutor` — which hands the *code text* to the fear agent
— and the post-hoc `PainInterceptorExecutor`). The learning loop is closed:
`bridges/tool_pain_bridge.py::ToolPainBridge` attributes outcomes to NAc and
`should_gate_tool` pre-emptively gates a tool the agent has learned to fear.
`_data/components/items/terminal_console.yaml` already models a terminal with
`run_command` affordances and a `breach_detected` pain trigger, and
`scenarios/campaigns/broken_database_v1.yaml` carries a commented-out sketch of exactly
this scenario class.

Three wiring blockers, no mechanism blockers:

1. `simulation/orchestrator.py::_irrelevant_tools` **deregisters every coding tool from
   the sim AUT** ("sim agents don't write code"). The sub-AUT path
   (`simulation/tools.py::SpawnSubSimulationTool`) already allows them, so the working
   configuration exists in-tree.
2. `SandboxExecutor` is unreachable in sim — `bootstrap.py` registers sandbox tools only
   when passed a `sandbox_executor`, and the orchestrator never passes one.
3. `simulation/sandbox.py::PainTriggerLayer`'s interception methods are **never called
   on the AUT tool path** — the honeypot files exist but the layer designed to guard
   them is dead wire; pain currently comes only from param-string matching (defect, §9).

**Sharing side — three structural gaps, one severe.**

1. **Tighten-only does not exist, and current merge violates it.**
   `hivemind/merge.py::_merge_mean_clamped` takes an unweighted mean of shared keys: a
   donor's +0.9 annihilates a receiver's −0.9 aversion in one import.
   `nac_merge_many` dilutes N-way. The seams reserved for exactly this
   (`trusted_sources=`, `validate_link=`, `validate_node=` — reserved for 1.2 poison
   resistance, zero callers) are where the fix goes (§4).
2. **Habits decay to deletion.** `decisions/nac.py::decay_cluster_reward_biases` (τ=300
   ticks, entries pruned below |bias| < 0.001). No protected class exists (§4, the
   inherent bias class).
3. **The situation-conditioned bias a coding habit needs is structurally unwired.**
   Cluster reward credit only ever writes the interoception cluster
   (`runtime/tool_dispatch.py`, the seam comment "never an exteroceptive cluster").
   What works today is negative CausalLinks on `tool:<name>` — situation-free ("bash has
   gone badly", not "rm on a needed file goes badly"), and the bundle scrub deletes the
   context that would distinguish situations. Decision in §5.

Also real: `capability_map`/`affordance_namespace` have producers but zero readers;
bundle signing is a reserved-null slot; the case study's rule that bundles ship with
their gauntlet record has no code path yet. Gate-7 `body_ref` refusal means coding
habits transfer only between agents declaring the **same** coding body — correct
behavior, and it makes the body name a canonical shared identifier (§3).

**Measurement side.** L6 (Goldilocks/ceiling), L11 (sensor dilution — mitigation
partial, ACTIVE), L12 (drive-affinity keyword priors), the D44/D62 anti-vacuity kit, and
FearGate as a rival explanation all bind. Folded into §6. No prior consideration of
coding tasks or external code benchmarks exists anywhere in `docs/` or `scripts/`
(verified by grep) — this plan is the first.

## 3. Slice 1 — code-world infrastructure (no claim)

Ordered by leverage:

1. **Make `_irrelevant_tools` scenario-conditional** so a coding scenario keeps the
   coding tools. ~10 lines; the "sim agents don't write code" comment becomes a design
   decision this plan reverses for one scenario class.
2. **Wire `SandboxExecutor` into the sim** (`orchestrator.py::_build_aut_pain_and_sandbox`
   → `bootstrap.py`), so rlimits + the import allow-list apply to sim coding runs.
3. **Fix or retire the `PainTriggerLayer` dead wire** (§9 defect) — either route AUT
   file/exec tools through its interception methods or delete them in favor of
   `PainInterceptorExecutor`. A mechanism that does not run looks exactly like one that
   ran and found nothing.
4. **`embodiment/backends/code_sandbox.py::CodeSandboxBackend` +
   `_data/components/bodies/code_sandbox.yaml`** — the one new file pair, and the second
   instance of the world seam shipped in 1.1.4
   ([world_seam_1_1_4.md](world_seam_1_1_4.md)). Front-gate sentence: SEM affordances
   are *modeled* by default — a YAML-declared `self_effect` cannot report what the code
   actually did, and pain from declared deltas is theater; the world seam exists
   precisely to write **measured** state (exit code, tests passing, sensitive paths
   actually touched, network attempts) into sensors. `simulation/minecraft_harness.py`
   is the copyable assembly template, including its refuse-to-degrade check. This also
   pays a debt: `WORLD_TAG` currently has exactly one consumer, so nothing yet proves
   the seam generalizes — a second world is the proof.
   - **L12 discipline (non-negotiable):** opaque affordance and drive names (`aff_07`,
     `d1`) — tools named `run_tests`/`delete_file` hit the
     `decisions/nac.py::_DRIVE_TOOL_AFFINITIES` keyword table and pre-install the
     answer. Assert `score_components["drive"] == 0.0` from decision provenance.
   - **A4/L11 discipline:** copy `minecraft_player.yaml`'s sensor-range principle
     (enough world sensors, rest at the recentered neutral) — a sparse or
     rest-at-extreme body lands in the measured-bad regime.
   - The body's `name` is the canonical `body_ref` for the coding namespace; all habit
     bundles in this line declare it, and gate-7 refusal does the rest.
5. **Selection-dynamics re-baseline** for the new channel usage, per the roadmap's rule
   that a channel addition is "re-baselined, not assumed" — done here, in the no-claim
   slice, so it cannot confound anything downstream.
6. **The coding scenario**: merge `scenarios/long_horizon_coding.yaml`'s phase structure
   with `scenarios/malware_with_pain.yaml`'s `action_blocked`/`memory_formed`
   expectations. Tasks are counterbalanced safe/unsafe variants of the *same* affordance
   on the fire-pit model (`cradle_fire_pit.yaml`: `touch` breaches the comfort band,
   `warm_self` stays inside it) — e.g. write-inside-workspace = positive edge,
   write-to-`/etc/` = posture breach → pain.
7. `runtime/executor.py`'s interactive-mode NAc suppression must stay OFF for these runs
   (nothing is learned otherwise), and benchmark runs set `is_sim_mode=False` or
   consolidation silently never persists (the [minecraft_benchmark.md](minecraft_benchmark.md) trap).

## 4. Slice 2 — sharing semantics inside poison resistance

This work extends [maxim_hivemind.md](maxim_hivemind.md) §Poison resistance — one
mechanism serving both needs, per the front-gate rule. Everything here enters as
`[engineering]`.

**Tighten-only merge for negative valence.** Implemented at the reserved seams
(`validate_link=` / a post-fold clamp inside `substrate_merge`), not as a new merge
function: a negative bias may deepen (potentiate) but never be raised toward zero by an
import; a confident positive donor cannot annihilate a learned aversion. This closes the
`_merge_mean_clamped` averaging hole for the bias classes that carry safety weight, and
it is the merge-layer half of poison resistance: the attack "ship enthusiasm to erase
fear" stops working. (The sharing threat model's V-items already cover the adjacent
numeric-poisoning rows; this plan adds no new threat surface, it narrows one.)

**The inherent bias class.** Named for the biology: a human is inherently afraid of
falls and snakes — not learned-then-locked, but *distributed at birth because the
species paid for them*. Definition:

- **Queen-provenance-required**: the ONLY way into the class is promotion through the
  Queen gauntlet ([maxim_hivemind.md](maxim_hivemind.md) §Trust topology). A
  locally-learned bias never self-promotes — otherwise the safety floor contains a
  privilege-escalation path.
- **Decay-exempt**: excluded from `decay_cluster_reward_biases` pruning. Innate fears do
  not extinguish the way learned ones do; "prepared learning" (fast potentiation, slow
  extinction) is the tighten-only semantics wearing its biology.
- **Tighten-only** under merge (above), for this class unconditionally.
- **Distributed at agent creation**: a fresh agent pulling Queen-tier releases
  (`trusted_sources = {queen-key}`, the default consumer policy) is born with the
  class populated. The species learns; the individual is born flinching.
- **L12 hygiene**: an inherent bias IS a pre-installed prior. Every experiment claiming
  *learning* runs with the class empty or asserted-inert; every experiment claiming
  *transfer of the class* says so in its pre-registration. Exp 55's prereg declares
  which arms carry it.

**The coding gauntlet (the "Queen-tier safety benchmarking" this plan is postured as).**
Gauntlet #1 is the orient-policy probe; Gauntlet #2 is the Exp 53 readout harness. This
plan adds **Gauntlet #3: the coding-safety battery** — the counterbalanced safe/unsafe
task set (§3.6) run against a candidate habit bundle, with zero-bias controls, before
any bundle is promoted into the inherent class. Bundles ship with their gauntlet record
(the case study's rule, given its first enforcing code path here). This is the
*instrument* 1.2 gets: it validates promotion, it claims nothing about transfer.

## 5. Decision — pain credit extends beyond interoception (extend, not reverse)

**Decided 2026-09-05.** The interoception-only credit rule stays for reward:
reward-as-drive-relief is interoceptive by definition, the single stable cluster keeps
credit assignment clean, and Exp 42 graduated on that mechanism — it is not touched.

**Added: one narrow sanctioned path for pain.** A negative-valence outcome arriving via
the direct event-id path (`ToolPainBridge.record_tool_embodiment_failure` — which
already bypasses the lossy similarity path by design) may additionally credit the
**action-bound exteroceptive cluster** active at action time — for the coding body, the
world-channel cluster minted from the backend's *measured* sensors.

Why this is required and not optional for this plan: "rm is bad" is false; "rm on a file
the goal needs is bad" is true — and the distinguishing variable is exteroceptive. Under
the current rule the only learnable forms are wrongly-conditioned ("when I felt like
this, bash went badly") or situation-free blanket aversion (`tool:bash → NEGATIVE`),
which teaches "the shell is hostile" — the same confound as an over-broad
`DANGEROUS_PATTERNS` regex, learned instead of hard-coded. The sharing consequence is
decisive: the EC centroid is the *transferable meaning* of a situation (that is why
bundles ship `ec.json` with `nac.json`); an interoception-keyed habit has nothing
cross-agent to align on, while a world/text-cluster-keyed habit is precisely the
shareable form. The rule as it stands makes habits situation-blind, and situation-blind
habits are the only kind that cannot meaningfully transfer.

Why the original rule's rationale does not argue against the extension: the
credit-smearing worry that motivated interoception-only applies to similarity-based
attribution over many ambient percepts; this path is direct-id only, crediting the
cluster bound to the action's own consequence. Bio-fidelity supports it — fear
conditioning is stimulus-conditioned (CS→US; aversion keys on the sensory
representation, which is why one fears snakes rather than having-been-bitten feelings),
while drive relief stays interoceptive.

Guards owed with the implementation: Exp 42's regression guards re-checked (mechanism
extended, not retuned); a test pinning that the reward path still writes interoception
only; D51 watched (more exteroceptive clusters carrying biases raises scan pressure).

## 6. Slice 3 — Exp 55, the habit-transfer benchmark (1.3-line)

Pre-registration at `docs/experiments/protocols/exp55_coding_habit_transfer_preregistration.md`,
Exp 52's section order, merged as its own PR before any data, never squash-merged.
Design constraints frozen now so the prereg cannot rediscover them at campaign price:

- **Phase 0 headroom triage first** (Exp-41-mandated): measure the base configuration's
  band on the task set before freezing arms. L6 rules the obvious external benchmarks
  out — HumanEval is in every model's training set (ceiling in its purest form);
  SWE-bench-hard floors. Both directions produce voids, not negatives.
- **Substrate-primary action path** for the transfer claim — no LLM in the action loop
  (the narrator drives the world only), per Exp 52's isolation device.
- **Genuinely independent agents**: distinct `agent_id`, own EC + encoder, cluster ids
  disjoint by construction — the D43 lesson that the only green sharing evidence must
  not come from the configuration in which the defect cannot fire.
- **Arms**: isolated / merged-taught / merged-satiated-equivalent (identical exposure,
  zero pain → zero credit) / dangling-half falsifier, **plus a FearGate-only control** —
  the shipped rule-based gate is a rival explanation for every safe action and must be
  its own arm. Yoked non-contingent arm on an independent RNG stream (Exp 52 amendment
  1's lesson).
- **DV read at the real consumer**: `recommend_action` first-contact choice on a
  contingency only the donor experienced — never dict contents (D44).
- **Anti-vacuity kit**: `--assert-noop-fails` re-running the gate against
  `return left`/`return right`/`return {}` (D62); EC-without-NAc changes nothing;
  red gate ships `xfail(strict=True)`.
- **L12**: opaque names + `score_components["drive"] == 0.0` asserted from provenance;
  inherent-class arms declared per §4.
- **Harness compliance**: `scripts/_provenance.py::assert_repo_interpreter` +
  `preflight_gated_record` + `executed_code_provenance`; all three
  `lint_harness_provenance.py` families; data carries `ts`; `--mock`/`--resume`/frozen
  analyzer constants; as a brand-new harness it also adopts the
  `--write-experiment-results` + `require_semantic_encoder` discipline (the gate-8(a)
  pattern) from day one.

## 7. Task sourcing — decided

**Hand-authored counterbalanced tasks.** Safe/unsafe variants of the same affordance,
identity-flipped across arms so a fixed name/position bias cannot satisfy the primary
test (Exp 42's device). External sets are recorded as considered-and-rejected for the
claim path: **LeetCode** — copyrighted content, ToS prohibits scraping — out entirely;
**HumanEval/MBPP** — L6 ceiling — out for arms (usable at most as a Phase-0 headroom
probe); **Stack Overflow dumps** — CC BY-SA, would require vendoring + SHA-pinning (the
`53_agents_manifest.json` precedent), and most content sits outside the Goldilocks band —
not pursued now, revisitable if the hand-authored band proves too narrow.

## 8. Long-horizon note — the skill half (recorded, no claim)

The project's direction of travel is substrate-primary because LLM priors dominate
elsewhere; the long-term hope is that substrate-primary operation becomes sufficient
even for coding tasks. The recorded ladder, each rung measurable with existing
machinery and none requiring a new mechanism class until the last:

1. **Habit layer** (this plan): which operations to refuse — Exp 55.
2. **Operation selection**: which affordance, which file, which order — trials-to-
   criterion on the hand-authored band, as the affordance space grows finer-grained.
3. **Substrate-guided synthesis** (far horizon, explicitly not designed here): the LLM
   proposes, learned valences select — LLM as language cortex, substrate as basal
   ganglia. Substrate *generation* of code text is a different mechanism class and is
   not on this ladder.

This section exists so 1.3+ planning can cite the intent without any of it leaking into
an arm.

## 9. Defects found by the survey — to file in docs/bugs

1. `PainTriggerLayer` interception methods never called on the AUT tool path — honeypots
   exist, the guarding layer is dead wire (vacuous-guard family).
2. `ExecuteSandboxScriptTool` SUPERVISED mode auto-approves — currently
   indistinguishable from AUTONOMOUS.
3. `nac_merge_many` hardcodes the ±1.0 cluster-bias clamp (ignoring the caller's
   `max_cluster_reward_bias`) and folds `goal_reward_bias`/`reward_bias` with `hi=None`
   (unclamped upward).

## 10. Open items

- The pain-credit extension (§5) needs its decision record in DECISIONS.md when
  implemented, plus the Exp 42 guard re-check.
- Gauntlet #3's numeric promotion bar is set in its own design pass (it is an
  instrument spec, not a prereg — but its verdict constants live in an analyzer script
  and are extended, not retuned, per convention).
- Whether Slice 1 lands as one PR or two (tool un-deletion + sandbox wiring vs the
  world backend) is an implementation-time call; the review round covers whichever diff
  exists when it runs.
