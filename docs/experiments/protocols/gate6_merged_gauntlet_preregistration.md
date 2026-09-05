# Gate 6 merged gauntlet pre-registration — seeds 42+43 through the re-keyed path (1.2 gate 6)

**Frozen 2026-09-05, merged to main BEFORE any data.** Implementation:
[`scripts/orient_backbone/gate6_merged_gauntlet.py`](../../../scripts/orient_backbone/gate6_merged_gauntlet.py)
— its module docstring mirrors this document; on any divergence THIS file is the
authority; a change to either after first data requires an amendment header here.

## Question

1.2 gate 6 (roadmap; sharpened by the Oasis case study §3): *"the Exp 52 seeds 42 + 43
must pass Gauntlet #2 merged … and must not pass with either half's clusters
dangling."* D43's fix (`substrate_merge`: EC align → donor bias re-key through the id
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
- DANGLING-HALF ×1: the pre-D43 recipe — bare `nac_merge(seed42_nac, seed43_nac)`
  with NO re-key, paired with seed42's EC unchanged. The donor's biases name
  clusters this EC never emits.

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
4. **Defect reproduction (dangling arm):** the d43 §4 guard **FAILS** (≥ 1 surviving
   key names an absent cluster) AND the dangling arm's directedness ≤ receiver-alone
   (seed42) + **0.10** — dangling keys buy nothing.
5. **Instrument arm:** empty-want merge reads `biases_rekeyed == 0`.

Any other combination → **FAIL** (numbers recorded, next step named — no threshold
motion). exp53-side refusals (its stop rules, its APPARATUS verdict on Gate T's
sign-agreement/spread check) → **refused, no verdict** (exit 4).

The verdict is computed by
`scripts/orient_backbone/gate6_merged_gauntlet.py::decide_verdict` — the protocol's
own function, no operator judgment.

## Dispositions on PASS

- Roadmap gate 6 → CLOSED (the code half shipped 1.1.3; this is the empirical half).
- D5 → RESOLVED on the cluster axis: a foreign want's cluster biases, re-keyed
  through the alignment map, READ OUT on the receiver (the exact fold D5's ACCEPTED
  disposition declined and gate 6 required).

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
health `biases_rekeyed=3, dropped=0, dangling=0` both directions, dangling arm
`dangling_keys=3, directedness=1.0` (defect reproduces mechanically; buys
nothing behaviorally), empty-want instrument `rekeyed=0`. Two apparatus defects
were found and fixed by the smoke (the exp53 mid-point verdict's by-design rc 1;
an in-process run_id collision when two phases start in the same wall-clock
second — fixed with a 1.1 s spacer). **The decision thresholds above were
authored before the smoke and were not moved after it.** The apparatus is
deterministic (dry rig seed 1), so the official gated run is expected to
reproduce these numbers; what it adds is the provenance-stamped record on clean
main, not suspense — the D8-protocol precedent.

## Data + provenance

Work dir + records: `docs/experiments/data/gate6_merged_gauntlet/` —
`records_A.jsonl`, `records_B.jsonl`, `gate6_verdict_<date>.json`, all gated
(in-process provenance: clean tree, this repo's `maxim`; the run refuses gated
writes otherwise). Data lands via a merge-commit data PR after this protocol is on
main. The merged substrate files themselves are DERIVED artifacts (recomputable from
the archive + this harness) and are not committed.
