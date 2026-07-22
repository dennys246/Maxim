# Credit on progress, not execution (deferred)

**Status:** Deferred / tracked (2026-07-22). Surfaced by the cradle_mother embodied-sim investigation (experiment [46](../../experiments/46_operant_orient_creche.md)) and **cross-confirmed by both lenses of the PR #-pending pre-merge review**. This is the root cause the `MAXIM_SUBSTRATE_TOOL_WHITELIST` band-aid masks.

## The problem

Substrate-primary action selection credits a tool for **executing successfully**, not for producing **goal/drive progress**. In `runtime/tool_dispatch.py::record_outcome`, a tool that mechanically "succeeds" (`sense_presence`, `examine`, `say`, `sense`, `listen`, …) books a POSITIVE causal link and, absent a drive signal, the tool-success cluster-reward floor (`cluster_reward = 1.0 if learn_success`). Because these tools always succeed, their `causal_pos` snowballs toward the cap and dominates `recommend_action` — the exact pathology the `INTROSPECTION_TOOL_NAMES` filter (`agent_loop.py`) already documents for read-only cognitive tools, but which extends to any always-succeed tool.

**Consequence:** a *specific* operant/drive-relief signal (e.g. the cradle mother's feed reinforcing an orient turn) is one faint signal in a flood of generic "this tool worked" signals. The embodied cradle_mother sim measured at chance for this reason (the infant chose `sense_presence`, causal_pos 0.99, over turning). It is not cradle-specific — it degrades **every** embodied substrate-primary agent's learning: the substrate cannot tell "I did a thing" from "the thing helped."

## Current work-arounds (both band-aids)

1. `INTROSPECTION_TOOL_NAMES` filter — removes read-only cognitive tools from the candidate set (partial: only the introspection subset).
2. `MAXIM_SUBSTRATE_TOOL_WHITELIST` — restricts the candidate set to a hand-listed minimal repertoire (per-experiment; must be re-specified each time = the band-aid signature).

Two band-aids around one root cause is itself the signal (per the "no band-aid" rule) that the fix belongs one level down.

## The fix (sketch)

**Separate two layers** (framing sharpened 2026-07-22): an **action-outcome layer** ("the tool executed" — kept for logging / telling the LLM what happened) that is DECOUPLED from an **internal-reward layer** (drive relief at the NAc — the ONLY thing that trains the policy). Today `record_outcome` conflates them: mechanical execution feeds reward.

Make **drive relief the sole reward driver**, gating BOTH credit surfaces on progress:
- **`learn_success` should mean "produced reward/progress," not "ran without error."** Today `learn_success = success and not embodiment_failed`; it should additionally require measurable drive/goal progress (the `drive_potential_diff` machinery already computes this for drive-touching affordances). A tool that executes but relieves no drive books **neutral**, not positive — on BOTH surfaces:
  - **cluster reward:** drop the `learn_success` floor for no-progress tools (the `MAXIM_OPERANT_ONLY_CREDIT` mode does this HALF today — it kills the cluster floor but NOT the causal link, which is why the whitelist is still needed).
  - **causal valence:** the OTHER half, currently unaddressed — `sense_presence`'s `causal_pos` 0.99 comes from the causal link, not the cluster floor. Causal *valence* must track value/reward, not mechanical execution. A tool still RECORDS that it ran (prediction of *what it does* is preserved); it does not record that running was *good* unless drive relief followed.
- With both surfaces gated, `sense_presence` never out-competes an orient turn (it earned nothing), and **both** the introspection filter AND the whitelist become unnecessary. Food is the sole driver by construction.

**Bio-faithful reframe:** tying reward to a GOAL STRING is the wrong frame for this learning — a baby has no goal, it has drives. Making reward intrinsic (drive relief) rather than goal-derived is more bio-faithful and is what this fix does. (Secondary, still needed but tractable once reward is clean: an explore→exploit schedule to converge on the now-clean signal, and ensuring the orient affordances are actually surfaced by tool activation.)

## Why deferred

It touches the core `record_outcome` credit path for ALL substrate-primary agents — a load-bearing change that needs its own experiment + two-lens review, not a rider on the cradle work. It is the prerequisite to **resurrecting the embodied cradle_mother demo** (so the operant contingency isn't drowned) and would measurably improve embodied affordance learning generally.

## Regression guard (when built)

The cradle_mother embodied sim itself becomes the integration test: with credit-on-progress + no whitelist, the embodied infant should learn to orient (approach the scripted probes' result) instead of measuring at chance. Plus a unit test on `record_outcome` pinning that a no-progress successful tool books neutral (not +1) cluster reward.