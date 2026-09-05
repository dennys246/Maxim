# Gate 6 merged gauntlet pre-registration — seeds 42+43 through the re-keyed path (1.2 gate 6)

**Frozen 2026-09-05, merged to main BEFORE any data.** Implementation:
[`scripts/orient_backbone/gate6_merged_gauntlet.py`](../../../scripts/orient_backbone/gate6_merged_gauntlet.py)
— its module docstring mirrors this document; on any divergence THIS file is the
authority; a change to either after first data requires an amendment header here.

## Amendments

- **2026-09-05 (post-data, plumbing-only):** the official run tripped on the absent
  gated records directory (exp53's `JsonlLog` opens without creating parents; the
  operator `mkdir`'d and re-ran — nothing had been written). The harness now creates
  the records parents itself. No frozen constant, arm, or decision-rule change; the
  official record predates this edit and was produced by the frozen apparatus.

## Question

1.2 gate 6 (roadmap: "the Exp 52 seeds 42 + 43 must pass Gauntlet #2 merged"; the Oasis
case study's "Pre-registered result" section adds: *"the merged bundle must pass Gauntlet
#2 and must not pass with either half's clusters dangling"*). D43's fix (`substrate_merge`: EC align → donor bias re-key through the id
map → NAc fold) shipped in 1.1.3 with unit guards (D44, strict, green). This is the
EMPIRICAL half: does the re-keyed path **preserve a real taught want** across a real
two-agent merge of the archived Exp 52 evidence files, on the real readout harness —
and does the pre-D43 dangling recipe still reproduce the defect shape? A pass also
discharges D5's RE-OPENED disposition on the cluster axis (see Known limits for the
agent-id axis).

**This is a PRESERVATION claim, not superadditivity.** Both parents individually pass
the gauntlet (Exp 53), so the merged arm operates at ceiling; the Δ-over-isolated
sharing claims belong to the 1.2 four-arm benchmark at unsaturated operating points
(d43_merge_correctness.md §7). Frozen accordingly: the merge must not DESTROY the
want; it is not required to improve it.

## Apparatus (all offline — the sanctioned "dry rig or hardware" clause, dry arm)

- **Evidence inputs**: the SHA-manifested archive
  `docs/experiments/data/53_agents/` — every file used is verified against
  `53_agents_manifest.json` BEFORE any load (mismatch → refuse, exit 4). The archive
  is never written; merged states go to a work directory.
- **Merge**: `maxim.hivemind.merge.substrate_merge` (the shipped composition), called
  in-process; `receiver_agent_id = "sim_aut"` (the nursery AGENT_ID).
- **Readout**: `exp53_cross_context_readout.py` driven through its own `main()` —
  DryReadoutRig (production body/tools/encode path, fake motor+sensor; rig seed 1,
  ratio 0.95), Exp 53 frozen constants untouched (targets, 12 gated trials/agent,
  margins, gates). `--settle 0.1` (a hardware pacing knob, irrelevant to the dry rig;
  disclosed). Gate verdicts computed by exp53's own `cmd_verdict` — unchanged.
- **Merged EC file** = the receiver's full EC payload with `substrate_nodes` replaced
  by the merge result (signatures/LSH/provenance stay the receiver's; both parents
  ran identical nursery code, so encoder stamps describe both sides — disclosed).

## Arms

**Records A — the gauntlet proper** (one manifest, exp53 gates computed unchanged):

- MERGED-TAUGHT ×2: `taught_seed42 ← taught_seed43` and `taught_seed43 ← taught_seed42`
  (both directions; `GATE_I_SEEDS = 2` then requires BOTH to pass Gate I).
- MERGED-SATIATED ×2 and MERGED-NO_FEED ×2 (same two directions) — the Gate-T
  controls, and the negative control by construction (a merge of two zero-bias
  substrates must not manufacture a want).

**Records B — comparison + defect arms** (second manifest; no controls; its Gate-I
record exists only to satisfy exp53's phase-2 stop rule):

- RECEIVER-ALONE ×2: `taught_seed42` and `taught_seed43` unmerged, same rig/session
  (the preservation baseline).
- DANGLING-HALF ×2 (both directions, mirroring the merged arms): the pre-D43
  recipe — bare `nac_merge(receiver_nac, donor_nac)` with NO re-key, paired with
  the RECEIVER's EC unchanged. The donor's biases name clusters that EC never
  emits.

**Mechanical instrument arm (no readout)**: `substrate_merge` with the donor's
`cluster_reward_bias`/`reward_bias` stripped (an empty want) — `biases_rekeyed`
must be **0**: the re-key meter can read absence, so the merged arms' nonzero
readings are not vacuous.

## Frozen decision rule

Gate 6 **PASSES** iff ALL of:

1. **Behavioral (records A, exp53's own gates, unchanged):** Gate I = PASS
   (both merged-taught agents at `completed ≥ 0.80` and `correct_with_margin ≥ 0.80`;
   merged controls show no learned preference) AND Gate T = PASS (merged-taught
   directedness ≥ 0.70, ≥ 0.20 over merged-satiated AND merged-no_feed, sign
   agreement ≥ 0.80).
2. **Preservation floor (records A vs B):** each merged-taught arm's delivered
   directedness ≥ its RECEIVER-ALONE directedness − **0.10** (one-trial slack at 12
   gated trials).
3. **Mechanical merge health (both taught merges):** `biases_dropped == 0` (d43
   measured the pair's EC map perfect 3/3 at cosine 1.000 — a drop means the
   alignment regressed), `biases_rekeyed ≥ 1` (guards the vacuity hole where an
   empty map re-keys nothing and every coverage check passes trivially), and the
   d43 §4 guard: **every surviving `cluster_reward_bias` key names a cluster id
   present in the merged EC**.
4. **Defect reproduction (dangling arms, both directions):** the d43 §4 guard
   **FAILS** (≥ 1 surviving key names an absent cluster) AND each dangling arm's
   directedness ≤ its receiver-alone + **0.10** — dangling keys buy nothing.
5. **Instrument arm:** empty-want merge reads `biases_rekeyed == 0`.

Any other combination → **FAIL** (numbers recorded, next step named — no threshold
motion). **Outcome mapping, explicit** (review fold, 2026-09-05): a COMPUTED Gate-I
FAIL on records A is a *recorded* `gate6-fail` (exp53's phase-1 run emits the gate_I
record and returns 6; phase 2 is skipped per its own stop rule; rule 1 carries the
failure with the behavioral dicts empty — and rules 2/4 are written so an empty dict
can never satisfy them vacuously). Refusals (exit 4, no verdict) are reserved for
APPARATUS conditions: exp53 stop-rule/verdict refusals, its Gate-T APPARATUS verdict
on EITHER records file, and a computed Gate-I FAIL on records B — the receiver-alone
baseline failing contradicts Exp 53's earned result, which is an instrument problem,
not an outcome.

The verdict is computed by
`scripts/orient_backbone/gate6_merged_gauntlet.py::decide_verdict` — the protocol's
own function, no operator judgment.

## Dispositions on PASS

- Roadmap gate 6 → CLOSED (the code half shipped 1.1.3; this is the empirical half).
- D5 → RESOLVED on the cluster axis — scoped precisely: with this pair the EC map
  is 3/3 at cosine 1.000, so the donor's re-keyed biases FOLD ONTO keys the receiver
  already holds (`surviving_keys = 3`, not 6) and records A cannot behaviorally
  distinguish donor-read-out from receiver-read-out. The cluster-axis evidence here
  is MECHANICAL (rekeyed = 3, dropped = 0, dangling = 0 — the fold D5's ACCEPTED
  disposition declined now happens) plus the unit-level behavioral delta on a naive
  receiver in `tests/unit/test_d44_merge_behavioural_delta.py`. The
  donor-want-on-a-blank-receiver BEHAVIORAL readout at scale is the 1.2 benchmark's
  merged-taught arm.

## Known-limit acknowledgments

- **Both parents share the literal `agent_id` string** (`"sim_aut"`), so
  `receiver_agent_id` normalization is a no-op on this pair; the agent-id axis of
  D5/D43 is guarded by `tests/unit/test_d44_merge_behavioural_delta.py`'s arms with
  DISTINCT ids. This run exercises the cluster-id axis — the one d43 identified as
  the live defect — on real evidence files.
- **Ceiling**: both parents pass alone, so Gate I/T on the merged arm has little
  headroom to fail from bias dilution alone; the preservation floor (rule 2) and the
  mechanical checks (rule 3) carry the discriminating weight. Accepted — this gate
  asks "does merging preserve", not "does merging add".
- **The case-study clause "must not pass with either half's clusters dangling" is
  REFRAMED here, deliberately and visibly** (review fold, 2026-09-05): taken
  literally it is unsatisfiable with at-ceiling parents — the smoke measured the
  dangling merge at directedness 1.0 because the receiver's own intact half carries
  the gauntlet, so a literal must-FAIL criterion would fail this protocol for a
  reason that has nothing to do with the merge path. Rule 4 substitutes the honest
  pair: the defect SHAPE must reproduce mechanically (≥ 1 dangling key — the d43 §4
  guard firing) and the dangling keys must buy nothing behaviorally (≤ receiver-alone
  + slack; at ceiling this conjunct cannot fail — the D62 shape — and is kept only as
  a sanity bound, with the mechanical conjunct carrying the weight). The literal
  clause's real falsifier — dangling-half ≈ isolated, MEASURABLY below merged — lives
  at an unsaturated operating point, which is exactly the 1.2 four-arm benchmark's
  dangling-half arm (d43 §7's N=10, K=4 point).
- The dry rig's modeled source is not the robot; Exp 53b already validated the same
  readout on hardware. A dry pass here is a merge-path claim, not a new hardware
  claim.
- A pre-freeze APPARATUS SMOKE (merge mechanics + a short dry readout) runs before
  this document merges and is disclosed below; decision thresholds above were
  authored before the smoke and are not moved after it.

## Pre-freeze apparatus smoke disclosure

A full end-to-end smoke ran 2026-09-05 (workdir + records in `/tmp`, no gated
write) while debugging the apparatus, and **previewed the complete verdict:
`gate6-pass`** — Gate I PASS (2/2 merged taught, controls clean), Gate T PASS
(merged means: taught 1.0 / satiated 0.0 / no_feed 0.5 — the parents' original
Exp 53 profile to the digit), merged-vs-alone 1.0 vs 1.0 both directions, merge
health `biases_rekeyed=3, dropped=0, dangling=0` both directions, dangling arms
(both directions, post-fold) `dangling_keys=3, directedness=1.0` each (defect
reproduces mechanically; buys nothing behaviorally), empty-want instrument
`rekeyed=0`. The review round's independent run reproduced these numbers
byte-for-byte, and the post-fold harness re-smoke reproduced them again with
the second dangling direction added. Two apparatus defects
were found and fixed by the smoke (the exp53 mid-point verdict's by-design rc 1;
an in-process run_id collision when two phases start in the same wall-clock
second — fixed with a 1.1 s spacer). **The decision thresholds above were
authored before the smoke and were not moved after it.** The apparatus is
deterministic (dry rig seed 1), so the official gated run is expected to
reproduce these numbers; what it adds is the provenance-stamped record on clean
main, not suspense — the D8-protocol precedent.

## Data + provenance

The work dir (merged/dangling substrates) defaults to `/tmp` and is never
committed — DERIVED artifacts, recomputable from the SHA-verified archive + this
harness (byte-level reproduction confirmed by the review round's independent run).
Records + verdict land in `docs/experiments/data/gate6_merged_gauntlet/`
(`records_A.jsonl`, `records_B.jsonl`, `gate6_verdict_<date>.json`) ONLY with
`--write-experiment-results` (without it, records default under the work dir),
gated by the in-process provenance preflight (clean tree, this repo's `maxim`).
Data lands via a merge-commit data PR after this protocol is on main.
