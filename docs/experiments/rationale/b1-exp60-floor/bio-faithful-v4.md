# B1 delta design review (v3 → v4): bio-faithful lens (2026-10-11)

Scope: the v3 → v4 prereg diff (OD5 γ = 1 / Q₀ = −2, rules 15–17), DECISIONS OD5, `scripts/survival_world/floor_policies.py`, `rationale/b1-exp60-floor/learner_sim.py`. This review was read-only and nothing was run on the rig.

**Verdict: no DO-NOT-BUILD.** v4 is still the right kind of comparison. A stronger floor makes BF2 stricter, not looser. But v4 changes what the floor contrast can show, and the claim text has not caught up. There are two SHOULD-FIXes, both in the prose, to fold before freeze.

## SHOULD-FIX

**S1. Under rule 17 the harness supplies WHEN to the floors, so H2 tests only WHICH (and how fast). The per-outcome claim still reads as if the floor matched both.**
Evidence:
- `ArmedPolicy.select` returns nothing until `in_water(observation)`, and the floors have no idle action. So in every placement and training episode a floor acts on every tick, and only in water.
- Its "when" is the experimenter's gate. FEAR's "when" is the value it learned for the situation, and on the shore at the first post placement FEAR proposes nothing.
- Rule 16 never credits a pick made on the shore. So a floor's shore rows stay at Q₀ = −2 for every tool, and its greedy shore behaviour is chance by construction (uniform tie-break, rule 1).
- That makes the secondary DV "shore-roam affordance calls" structurally fixed for the floors. It cannot show whether a floor tells the situations apart.
- Net effect: no gated or secondary measure in B1 contrasts WHEN between FEAR and a floor. The WHEN evidence is still FEAR vs ABLATED (Exp 60) plus FEAR's silence on the shore. That is unaffected, but B1 adds nothing to it.

The claim sentence "a model-free learner over the full repertoire needs N executed actions to match it" now overstates the floor's task.

Fix:
- Reword the H1-pass/H2-tie claim to: "…It needs no executed training actions. A tabular learner that is armed only in water (the harness supplies WHEN) needs N executed actions to match its first-placement escape over the full repertoire, and fewer (H3) when given the same prior. B1 does not test WHEN against a floor; that rests on FEAR vs ABLATED."
- Add one line to the secondary-DV text: under rule 16 the floors' shore-roam calls are at chance by construction, so they bound FEAR's shore habit and say nothing about floor learning.
- Optional, not required: a floor that has to learn WHEN would need an explicit idle action, armed on the shore with credit. Name it in Not claimed rather than build it.

**S2. The sample-efficiency number depends on the learner's constants, and v4 shows how much. Say "this learner", not "a model-free learner".**
Evidence:
- All rewards are ≤ 0. So v3's Q₀ = 0 was the textbook *optimistic* init, and it over-explored: about 0.10 at K = 10 and about 0.33 at K = 30 (DECISIONS OD5).
- v4's Q₀ = −2 is the minimum return, V_min. With greedy selection it becomes a *satisficing elimination* search:
  - A non-escape tool that has been tried bootstraps to r + max Q(next) ≈ −2 − c, which drops it below the untried tools.
  - `escape_water` updates toward about −0.3 (terminal 0), which locks it in.
- That is a legitimate, competent learner. But an RL reviewer would call init at V_min a tuned choice, made after the toy showed the default lost, not "the standard baseline". It was tuned for the strict outcome before any rig data, and the toy gates nothing, so it is acceptable under the weak-evidence rule. It still has to be disclosed where the number is quoted.

Fix:
- Wherever N or the efficiency gap is stated (H2, the claim, the BF2 row text), name the learner: "tabular Q, α = 0.5, γ = 1, Q₀ = V_min (−2), ε = 0.2".
- Add one clause: "with the optimistic default Q₀ = 0 the same learner did not reach criterion in 10 episodes (toy model, not data)".
- That makes the efficiency gap a statement against a stated competent learner, not against model-free learning in general.

## Answers to the four questions

**(a) Does Q₀ = −2 smuggle in a defensive prior?** No, not in the sense that matters for H3. It is the same for every action and every state: it encodes the *scale* of the worst outcome, not *which* act is defensive. So the H3 contrast (the 26-tool repertoire vs the prior's 2-tool set) still isolates action-identity knowledge.
- Q₀ = −2 is not neophobia either. Neophobia avoids the untried; here an untried tool ranks *above* tried failures.
- Nor is it the situation value FEAR learns, because it is pessimistic on the shore too.
- What it does is shrink the H3 gap to "elimination over 26 vs 2". That is honest, and expected; see S2.

**(b) Is γ = 1 with a per-tick oxygen-deficit cost an honest learner for an air-breather?** Yes.
- It is the homeostatic-RL formulation (Keramati & Gutkin 2014): reward is the drive deficit integrated over time, which under γ = 1 equals drive reduction up to a constant.
- Every episode terminates, so γ = 1 is well defined.
- Rule 16 makes surfacing terminal at the first select that sees air. That is the breath, the right event for an air-breather to treat as the end of the threat, and it removes v3's mis-credit.
- The graded deficit gives the floor a richer training signal than FEAR's pain-booked fear. That favours the floor, so a tie is conservative. See N1.
- The claim does need re-wording: S1 and S2.

**(c) Does water-only arming change what "the prior with vs without the learned situation value" means?** Not for FEAR vs ABLATED, which still stands as written (prereg §What Exp 60 already showed). It does change what the *floor* contrast means; see S1.
- Placements 2–6 are asymmetric under rule 17: FEAR's shore-launched `escape_water` lands about 0.7 s before the teleport, and the floors cannot do that.
- This is harmless only because the primary DV is the first placement. Add one sentence saying no rate over placements 2–6 is compared across arms (N2).

**(d) Is the bio-mapping text still accurate under v4?** Yes, still accurate: Bolles/SSDR (species-specific defence reactions), "context (situation) conditioning", the non-Rescorla–Wagner accumulator, and the disclosure of the task-fitted actuator. Nothing in v4 makes them wrong. The only inaccurate prose is the claim sentence (S1).

## NIT

- **N1.** OD3 or rule 15: state that the floors' reward is a graded interoceptive deficit while FEAR's training signal is the pain publish. The difference favours the floor, and saying so pre-empts the reverse objection.
- **N2.** §Primary DV: "placements 2–6 are not compared across arms: rule 17 forbids floors the shore-launched act that FEAR's habit performs."
- **N3.** H1 rationale: say that RANDOM is wrapped water-only too. The "≈ 4 useful draws from submersion" arithmetic assumes it is (it is, via `ArmedPolicy`).

## Verified

- `floor_policies.py`: `GAMMA = 1.0`, `Q_INIT = PAIN_TERMINAL − SATURATING_PAIN = −2`, the same for every action in `_row`.
- `_close` drops picks with `"made_out_of_water"`, and an air observation after a water pick is terminal (`next_key=None`).
- `ArmedPolicy(water_only)` counts `idle_dry`.
- The floors have no idle or no-op action (`available_tools` only).
- Rule 1's uniform tie-break makes greedy picks from uncredited shore rows uniform.
- DECISIONS OD5 toy numbers: v3 ≈ 0.10, K = 30 ≈ 0.33, v4 ≈ 1.0.
- The prereg's Bolles and context-conditioning text is unchanged by the diff.
