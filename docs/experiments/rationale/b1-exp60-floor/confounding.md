# B1 design review — confounding lens (2026-10-11, on the DRAFT prereg)

Preserved verbatim from the reviewer.

Read: the draft prereg; Exp 60 §Design (iii), Amendments 3–7 and §Outcome; baseline_floors.md (BF2, §B1, "tabular_q vs NAc"); `water_trial.py::loop_window/placement/probe`; `substrate_proposal.propose_via_substrate`; `nac.recommend_action` + `_DRIVE_TOOL_AFFINITIES`; `minecraft_player.yaml`; `tool_bridge` tool generation. I also re-read the run-2 rows of `exp60_trials.jsonl` (run ids 301eb2edff6d / eeb92752ee2b).

## DO-NOT-BUILD

**D1. FEAR's placements 2–6, and the shore roam, are a habit that fires anywhere, not a response to water. The draft's specificity expectation contradicts the data.**
- In every FEAR seed, the first `escape_water` call in placements 2–6 lands at **t ≈ −0.74 s, before the teleport**, while the agent is still on the shore during warm-up.
- The post shore roam logged **13–14 `escape_water` actions per FEAR seed**. The draft says FEAR "is expected near 0".
- So after placement 1 the positive causal link drives `escape_water` whatever the cluster. A trivial "always `escape_water`" policy scores the same 1.0.
- A 6-placement P(surface) therefore credits FEAR with something that needs neither the situation value nor the water.

Change:
- Make the **primary DV for H1/H2 the first post placement**. It is the only fear-only read and the only one the claim's "without action-outcome experience" covers.
- Report placements 2–6 as secondary, labelled fear + positive link + shore habit.
- Make shore-roam affordance calls a **reported specificity DV for every arm**.
- Correct the specificity expectation to Exp 60's measured 13–14 calls.
- Power the binary DV (see S4).

**D2. Both Q arms lose by construction under the frozen reward, so "beats tabular_q" is guaranteed and carries no information.**
- Reward is −pain, else 0, and surfacing is terminal 0. Every tick before the pain at about 5–6 s books 0, and so does `escape_water`.
- Q only goes negative for actions that happened to be taken on the 1–2 pain ticks per episode. With about 26 actions (D3), the greedy pick stays a uniform tie over the 0-valued set.
- That holds for Q_EXPLORE too: 10 episodes give about 20 pain-tick samples, so it cannot single out `escape_water`. H3's "finds `escape_water` and matches FEAR" is very likely false for a reward-design reason, not a learning one.
- The draft itself derives that Q_YOKED equals RANDOM. That makes it a strawman, which is what the reviewer would say.

Change:
- Replace Q_YOKED with **Q_PRIOR_YOKED**: the same learner, Q initialised with the same innate affordance bonus (threat-affinity tools +b in every state), and the same propose-only credit. This is the fair same-experience, same-prior floor. It separates Pavlovian state value (Wire-4) from instrumental action credit. Prediction to pre-register: it punishes `flee`/`escape_water`, the actions it was "doing" when the pain hit, and it fires on the shore.
- For Q_EXPLORE, use a game-native per-tick cost (oxygen deficit from the drive, or a constant step cost), declared now. Otherwise label H3 "uninformative by reward design".
- Run the TD(V) + prior "mechanism twin" as a cheap arm. baseline_floors.md already says "if no difference, the floor IS the mechanism". Showing that tie explicitly is the honest BF2 answer; naming it and not running it is not.

## SHOULD-FIX

**S1. Narrowed claim.** "A model-free learner given the same experience cannot do" is self-serving. It was not given the innate prior FEAR uses. Reword to: "learning lives in a Pavlovian situation value; the action is innate; a model-free action-value learner without the prior cannot, with the prior and instrumental credit it learns X (Q_PRIOR_YOKED)". Drop "matches it" for H3 unless an equivalence criterion is declared (S4).

**S2. RANDOM's P(surface) is wrong and is a researcher degree of freedom.**
- The harness registry is built from the body: 1 `sense_minecraft_player`, 17 `read_minecraft_player_*` and 8 affordances, so **about 26 tools**, not about 15. Introspection tools are filtered out; the bootstrap no-op head tools are not in this `ToolRegistry`.
- The only way to surface in time is an `escape_water` call by about 2.8 s after the teleport, since the escape itself takes about 1.5 s. That is about 5–6 useful 2 Hz ticks, fewer if a slow call (`attack_nearest`) blocks the loop.
- Expected P ≈ 1 − (25/26)^6 ≈ **0.2**. Over the 8 affordances only it is ≈ 1 − (7/8)^6 ≈ 0.55.
- Freeze the set by enumerating the live registry at preflight and stamping it.
- Declare how RANDOM gets past `min_confidence`.
- Disclose that RANDOM acts during the 1 s shore warm-up and before the in-water cluster forms. That is more ticks than FEAR, which is conservative.
- RANDOM cannot approach 1.0 per seed: P(6/6) ≈ 0.03 even at p = 0.55.

**S3. Learning-in-probe asymmetry.** FEAR's NAc keeps learning inside probes (the positive link). Say whether the Q tables update during probes. Either freeze all learning in probes, which is impossible for FEAR without changing it, or let the Q arms learn from executed probe outcomes too, and disclose the asymmetry.

**S4. Statistics.**
- H1 and H2 are a family: use Holm at 0.05.
- With a binary first-placement DV, 5 v 5 and one RANDOM seed surfacing gives p = 6/252 = 0.024; two give 21/252 = 0.083 and fail. Use **≥ 8 seeds per arm**, or pre-register the power calculation at RANDOM p ≈ 0.2.
- For H3, a "tie" needs TOST with a declared margin (for example ±0.2), or the prereg says "descriptive only".

**S5. Q_EXPLORE confounds to disclose and record.**
- Unyoked, executed exposure.
- A different number of pain events, because early surfacing cuts them.
- Possible world-mutating or position-changing actions (`move_to`, `mine_block`, `place_block`, `attack_nearest`).
- Exp 60's rule "any `drive:health` pain in training refuses the seed" must apply, or must be explicitly amended for this arm.
- Record pain events, executed actions and terminal causes per episode.

**S6. Carryover between arms.**
- A fixed order (FEAR first in every seed) puts each arm at the same session position every time. Use a Latin-square order across seeds.
- Before each arm, re-run the live geometry and cluster-distinct preflight, plus a pool block check and an inventory, food and mob-count check.
- Assert a separate throwaway persistence dir per arm, and assert that no `~/.maxim` state is read.
- Re-verify the frozen-day gamerule per arm. A session now runs about 36 min per seed.

## NIT

- **N1.** "Selectable actions at AUTONOMOUS" should cite `propose_via_substrate`'s registry minus `INTROSPECTION_TOOL_NAMES`, and the consecutive-same-tool cap (5, identical params), which RANDOM rarely hits.
- **N2.** Owner decision 3: reusing FEAR's existing records is weak evidence by the project's rule. Recommend all arms run together on the same rig day.
- **N3.** State in the prereg that the Q state key never carries "surfaced". Terminal is the harness's first head-in-air read.

## Verified

- Run-2 FEAR placement 1: `flee` at about 0.9 s (fails), then `escape_water` at about 1.55 s; latency 2.9–3.3 s.
- Placements 2–6: first call at −0.68 to −0.78 s, before the teleport.
- Post roam: 13–14 `escape_water` actions per FEAR seed, no water entry.
- Call `t` is measured from the teleport (`loop_window`: `c["t"] - t0`).
- "escape" in `_DRIVE_TOOL_AFFINITIES["threat"]` substring-matches `minecraft_player_escape_water`; ties break by name sort, so `flee` comes first.
- The harness registry is body-generated (sense/read/affordance tools); the oxygen drive has no innate corrective need.
- The 5 v 5 exact permutation floor is 1/252 = 0.004.
