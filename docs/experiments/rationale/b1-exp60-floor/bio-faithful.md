# B1 design review — bio-faithful lens (2026-10-11, on the DRAFT prereg)

Preserved verbatim from the reviewer.

## DO-NOT-BUILD

**D1. Q_YOKED can't fail to lose, so it can't serve as the binding BF2 tabular floor.** In propose-only training the air-hunger pain arrives whatever action is proposed, because nothing executes. Every action value Q_YOKED books is therefore noise. The prereg's own H2 rationale works out the loss from that structure, so the gate measures nothing. It doesn't match any real learner either. In the classic triadic yoked design (Seligman), the yoked animal *emits* responses that have no effect on the outcome. Here the responses are never emitted at all, so this is a credit rule invented for the floor. Making it the binding floor (owner decision 1's recommendation) turns BF2 into a gate that cannot fail, which is the evidence pattern the weak-evidence rule forbids.
Prereg change:
- Drop Q_YOKED as a gate. Keep at most one sentence of analysis: "an action-value learner cannot learn from action-independent reward; shown by construction, not run."
- Make Q_EXPLORE the binding tabular floor, run on an executed-episode budget curve (episodes 1, 2, 5, 10).
- Restate the claim as **sample efficiency**: FEAR needed 0 executed training actions; report how many Q_EXPLORE needs to reach criterion. That is the real gap, and a tie at 10 episodes is a boundary, not a loss.
- Under BF2 this probably makes the row `PARTIAL (narrow: floor matched)` with the efficiency gap stated. Say that up front rather than choosing a floor that guarantees a win.

**D2. The narrowed claim overstates what Wire-4 learned. The innate prior names the answer.** `escape_water` is a pathfinder-bypassing macro built for Exp 60 (#719, 2026-09-15). It includes an eye-height stop rule tuned to the game's breathing criterion. The experimenters named it so the existing `"escape"` keyword in the `threat` affinity would match. The keyword itself was not added for Exp 60: it is in `fear` since 75d11128 (May) and in `threat` since 2744ca27 (2026-09-10). With `drive_gate_enabled`, `recommend_action` narrows the choice to the drive-relevant tools, effectively {flee, escape_water}. That is a 2-way pick against RANDOM's and Q's ~15. "Learns anticipatory escape" credits Wire-4 with the escape. Wire-4 learned *when*; the prior's name match supplied *what*.
Prereg changes:
1. Reword the claim to: "Wire-4 learns from Pavlovian experience alone *when* to act; *which* act comes from a designer-specified defensive prior (keyword `escape` matched to the purpose-built `escape_water` actuator), which the floors do not receive."
2. Add a **prior-matched floor**: Q_EXPLORE limited to the threat-affinity action set. If it reaches criterion in about 1 episode, it shows the prior carries nearly all the action knowledge. Publish that number.

## SHOULD-FIX

**S1. The mechanism section mislabels the theory.** It is Bolles, not two-factor theory. Mowrer's second factor is instrumental avoidance reinforced by fear reduction, and Exp 60 has none: there is no positive escape link, and the preflight asserts `get_positive_outcomes(escape_water)` is empty. Bolles' account fits: fear selects from an innate SSDR repertoire (species-specific defence reactions), and avoidance is fast only when the needed response is already an SSDR. Cite it, and note the gap with real biology. A biological SSDR is a species-wide repertoire. This one includes a task-fitted actuator. Surfacing is a defensible innate response for an air-breather, so the analogy holds, but the fit must be disclosed. Also call it context/situation conditioning, not a discrete CS. `record_cluster_fear` is a clamped accumulator (`valence -= α·intensity`), not a prediction-error (Rescorla–Wagner) update, so "the CS predicts the US" overstates it.

**S2. Add a floor on discretised raw sensors to separate the EC from Wire-4.** Keying Q on EC cluster ids hands the floor Maxim's representation. In Exp 60 that representation is also fragile: shore vs submerged first failed at cos 0.8502 against a 0.85 threshold and needed the saturation-rest fix (#726). Add Q_EXPLORE_RAW, keyed on (`is_in_water` at eye height × oxygen bin). baseline_floors.md already names this gap for Exp 62, and here it is cheap. Otherwise state plainly: "the EC's contribution is not tested; the claim credits Wire-4 given the EC's split."

**S3. The state keys aren't the same.** Wire-4 reads the deepest fear across *all* active modality clusters (`anticipatory_threat_need` takes min over `clusters.values()`). Q is keyed on "the live world-cluster id" only. BF2 requires the same state keys. Key Q on the tuple of the full `clusters` dict that the proposer passes, or justify the single key.

**S4. Learning during probes is asymmetric.** FEAR learns from action outcomes *inside* the probe: `flee` fails fast, leaves a negative link, and the pick moves to `escape_water`. A greedy Q at ε=0 that freezes during probes would repeat a failing action forever. Pre-register whether Q keeps updating online during probes. To match FEAR it should. Also amend "FEAR used ZERO executed actions" to "zero in training; one failed action per probe."

## NIT

- **N1.** H1's 0.42 estimate assumes ~15 uniform actions. FEAR's effective set after the drive gate is ~2. State that RANDOM tests "chance over the full repertoire", which differs from the prior-matched comparison in D2.
- **N2.** The mechanism twin should name what Exp 60 already ran. FEAR vs ABLATED is exactly "prior with vs without learned situation value", and that contrast still stands under any floor result. Say so in the BF2 consequence text so PARTIAL isn't misread as a retraction.
- **N3.** Owner decision 3 (reuse the old FEAR records): weak-evidence rule applies, so recommend the strict option and run all arms.

## Verified

- Affinity table: `"escape"` in `threat` and `fear`, plus the git history above.
- `escape_water` origin: #719, plus the pathfinder-bypass and eye-height stop-rule text in the Exp 60 prereg.
- The drive gate narrows the choice to drive-relevant tools (`nac.py` B7 gate; `drive_gate_enabled`).
- `anticipatory_threat_need` takes min over all active clusters and is combined with `health→threat` by max (`substrate_proposal.py`).
- `record_cluster_fear` is a clamped linear accumulator with no prediction error.
- Exp 60 preflight asserts no positive escape link, so there is no instrumental second factor.
- The EC split was marginal (0.8502 FAIL → 0.7874 after #726).
- `flee` then `escape_water` within the probe is disclosed in the Exp 60 prereg.
- baseline_floors.md already flags the raw-sensor floor for Exp 62 only.
