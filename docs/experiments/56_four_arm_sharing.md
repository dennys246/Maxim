# Exp 56 — The four-arm sharing benchmark: a taught want transfers between independent agents

**Status: EARNED 2026-09-06.** The 1.2 "Oasis" headline claim. One agent is taught a
want; its learned substrate is exported as a bundle and ingested into a second,
genuinely independent agent through the real 1.2 ingestion path; at the receiver's
FIRST contact with the situation it chooses the taught action — and three controls say
that choice needed the taught representation, not merely the arrival of a bundle.

- **Pre-registration** (frozen before any run):
  [protocols/exp56_four_arm_sharing_preregistration.md](protocols/exp56_four_arm_sharing_preregistration.md),
  merged at `3f9ce733` (#639); amendments 1–4 all PRE-CONFIRMATORY-DATA.
- **Data** (gated, committed): [data/56_four_arm.jsonl](data/56_four_arm.jsonl) (200 rows,
  4 arms × 50 pairs) + [data/56_four_arm_verdict.json](data/56_four_arm_verdict.json)
  (analyzer verdict) + [data/56_phase0.json](data/56_phase0.json) (instrument checks).
  Confirmatory campaign at `main`-reachable `9905d4d8`, clean tree, `mock: false`.
- **Harness:** [scripts/exp56/](../../scripts/exp56/) + [scripts/analyze_exp56.py](../../scripts/analyze_exp56.py)
  (frozen verdict constants) + bodies
  [minecraft_bench.yaml](../../src/maxim/_data/components/bodies/minecraft_bench.yaml) /
  [minecraft_bench_satiated.yaml](../../src/maxim/_data/components/bodies/minecraft_bench_satiated.yaml).

## The question

Agent B has never experienced a contingency. Agent A has: at one specific world
situation, one specific action was rewarded by a teacher (the Exp 52 taught-want
mechanism, transplanted from the audio channel to the world channel). A's substrate is
exported as a bundle and ingested into B through the shipped 1.2 path
(`maxim substrate ingest` — the V1–V10 adapter + `substrate_merge`, aligned re-key +
tighten-only clamp — never bare `nac_merge`). At B's first contact with the situation,
does B choose A's taught action, and does that choice depend on the taught
representation rather than on the mere fact that a bundle arrived?

A and B are independent by construction (D44's definition): different `agent_id`,
separately constructed `EntorhinalCortex` + `SensorEncoder`, cluster ids disjoint. The
two shipped federation results (Exp 45 arm 3, the orient merge) pass only because every
participant shared one agent id and one encoder; this is the first cross-agent transfer
where the boundary the merge must normalize actually exists.

**Lineage:** Exp 42/45 (cross-**session**) → Exp 52 (the want is learned) → Exp 53/53b
(cross-**context**: the want reads out on a physical body) → Exp 56 (cross-**agent**:
*someone else's* substrate drives an agent that never learned it).

## Apparatus (as run)

- **World:** the LIVE Minecraft bridge — the frozen NDJSON protocol against a real
  offline-mode Paper 1.16.5 server on the operator's big-mac-mini, superflat/void,
  daylight cycle off, no mob spawning, the contingency source placed by RCON per the
  seeded script. Not the mock (the mock served harness development and the `--mock`
  smoke only — see Amendments 2 and 4 for the two live-only facts it could not model).
- **Body:** `bodies/minecraft_bench` — opaque affordance names (`aff_a`…`aff_h`,
  per-pair permuted) and opaque drive `d1`, so neither the L12 drive-name-substring
  channel nor an alphabetical tiebreak can pay off; five world sensors (light_level
  removed per Amendment 2); ranges re-centered so rest sits at the A4 neutral. A
  satiated twin (`minecraft_bench_satiated`) with the SAME entity name but drive drift
  0 / initial 0.
- **Teaching (the A-phase):** substrate-primary, no LLM in the action path,
  `MAXIM_OPERANT_ONLY_CREDIT=1`. Donors execute a seeded, exposure-balanced schedule
  (each roster affordance equally often — learning is carried by the teacher's credit,
  not by the donor's own behaviour, so the causal-link channel is preference-neutral).
  A harness-side teacher watches the world: when the donor's pending action is the
  target affordance AND the contingency situation is active, it feeds the donor
  (relieving drive `d1`) and calls `NAc.credit_operant_reward` with the Exp 52
  relief-signed value — landing the credit on `(agent, WORLD-cluster, tool:<target>)`,
  the situation-keyed transferable form. Zero relief mints nothing (which is what makes
  the satiated arm a mechanism check).
- **The four arms:** 1 **isolated** (fresh B, no ingestion — the floor); 2
  **merged-taught** (fresh B + A's bundle — the claim); 3 **merged-satiated** (fresh B
  + a bundle from a donor that ran the identical schedule and identical feeds but with
  drive held at zero → zero credits by mechanism — the want-not-file control); 4
  **dangling-half** (fresh B + a re-compose of arm 2's donor session with `aut_ec.json`
  absent → bias keys without the representation they key on — the D43 falsifier). One
  donor per receiver, seed-paired; no bundle reused across receivers.
- **Selector (identical in every arm):** seeded ε-greedy over the frozen roster, ε =
  0.2, `min_confidence = 0.3`, `substrate_explore_bonus_weight = 0` at the probe. The
  first-contact DV is read at the real consumer — the action the loop proposes, from
  the `NAc_RECOMMEND` decision-provenance record — never from dict contents.

## Phase 0 — instrument checks (all five PASS)

Run gated on the live apparatus at `4cf67cf9` ([data/56_phase0.json](data/56_phase0.json)):

1. **L11/A4 discriminability:** separation **1.0**, stability **1.0** over 20
   rest→situation onsets — the far+high slots (Amendment 2) separate cleanly; the
   situation-blindness the pre-amendment apparatus showed (separation 0.0, one fused
   cluster) is gone.
2. **Transfer pilot + dangling tripwire + no-op kit:** the taught pilot's first contact
   is bias-decisive at `learned_margin` 0.9; the dangling pilot does NOT choose the
   target (link channel neutralized); the two must-collapse no-op variants collapse.
3. **L12 zero-prior:** `score_components["drive"] == 0.0` on every probe decision.
4. **Dithered floor:** 10 isolated probes spread across five affordances, concentration
   0.3 — not seed-invariant, no affordance pre-installed.
5. **Donor at K:** donor sanity passes at the frozen K = 96 (world-cluster bias ≥ 0.4,
   link balance, geometry stamps, no inherent keys).

## Results — n = 50 receivers per arm (95% Wilson intervals)

| arm | first-contact raw rate | bias-decisive rate | winning component (of the 50) |
|---|---|---|---|
| **isolated** (floor) | 0.22 [0.13, 0.35] | 0.00 [0.00, 0.07] | causal ×50 |
| **merged-taught** (the claim) | **0.84 [0.71, 0.92]** | **0.80 [0.67, 0.89]** | learned_bias ×40, causal ×10 |
| **merged-satiated** (want-not-file) | 0.12 [0.06, 0.24] | 0.00 [0.00, 0.07] | causal ×50 |
| **dangling-half** (falsifier) | 0.12 [0.06, 0.24] | 0.00 [0.00, 0.07] | causal ×50 |

The taught arm's successes are carried by the situation-keyed learned-bias channel: 40
of its 50 first contacts are decisive on the learned-bias component, the exact channel
that fires only when the receiver's current world cluster matches the taught one. The
remaining 10 taught successes and every isolated / satiated / dangling choice of the
target are won by the causal component — i.e. floor-rate coincidences, not transfer.

## The four gates (all PASS; the claim requires the conjunction)

| gate | rule | arithmetic | verdict |
|---|---|---|---|
| **TRANSFERRED** | taught bias-decisive ≥ 0.70 | 0.80 ≥ 0.70 | ✅ |
| **ABOVE-FLOOR** | taught raw − isolated raw ≥ 0.20 | 0.84 − 0.22 = 0.62 | ✅ |
| **WANT-NOT-FILE** | taught raw − satiated raw ≥ 0.20 | 0.84 − 0.12 = 0.72 | ✅ |
| **BOTH-HALVES** (falsifier) | dangling raw − isolated raw < 0.10 (one-sided) | 0.12 − 0.22 = −0.10 | ✅ |
| **ANTI-VACUITY** | must-collapse no-op merge variants collapse; donor-alone recorded | receiver-unchanged & empty-state chose nothing; donor-alone persists | ✅ `kit_pass: true` |

`verdict: PASS`, `problems: []`. The falsifier is the load-bearing control: arm 4 ships
A's bias keys WITHOUT `aut_ec.json`, and the ingest report shows those biases DROPPED
(`biases_dropped == donor bias count`, `biases_rekeyed == 0`, the honest D43 indicator)
rather than silently landing — so a bias key with no representation to key on buys
nothing, and does so loudly. Arm 4 sits at the floor (below it, in fact), which is the
mechanism the claim asserts.

## What this shows — and what it does NOT

**Shows:** agent A's learned representation — the world-cluster-keyed operant bias plus
the EC nodes it keys on — changes agent B's first-contact behaviour, through the real
1.2 ingestion path, by the situation-keyed channel, where A and B are independent agents
whose cluster spaces are disjoint by construction. The want-not-file and dangling-half
controls establish that the transfer needs the taught representation, not the arrival of
a bundle and not the causal-link half alone.

**Does NOT (each its own later experiment, by pre-registration):**

- **Scaling.** One donor per receiver. The dose–response ladder (N donors, N ∈ {1,2,4,8}
  at the unsaturated operating point) is its own prereg, frozen only now that this
  apparatus is proven.
- **Hardware.** Substrate-primary in a Minecraft world. The two-Reachy cross-unit
  replication (riding Exp 54's body) is its own prereg.
- **Aversion transfer.** Positive credit only; the tighten-only clamp is exercised here
  only as byte-untouched positive pass-through. Negative-valence transfer is Exp 55
  (1.3-line).
- **Cross-layout generalization.** A trains and B probes against the SAME seeded world
  configuration; "recognizes the situation anywhere" is not claimed.
- **Self-taught wants.** The donor is TAUGHT by a contingent teacher (the Exp 52 shape);
  autonomous acquisition of world-keyed wants needs the credit-path extension that is
  1.3-line work.
- **The LLM-AUT path**, live-server dynamics beyond the controlled world, or multi-agent
  coexistence.
- **Limits declared inapplicable by name** (per the prereg): L4 (`safe_pref` saturation —
  different metric, and the floor arm is measured not assumed); L6 (prior-agreement ceiling
  — no LLM in the action path; the L12 prior channel is the applicable analog and is gated);
  L9/L10 (DoA sweep instruments — no DoA here).

**Honest scope in one line:** one campaign, one world layout, substrate-primary, in
Minecraft, n = 50/arm — with the two-Reachy hardware replication and the dose–response
scaling claim still ahead as their own experiments.

## Provenance and the live-apparatus story

The confirmatory campaign ran ONCE (the stop rule), from a clean tree at `main`-reachable
`9905d4d8`, every record stamped with the executed git hash and `ts`. The path to that
single clean run is disclosed in the pre-registration's four amendments, all before any
confirmatory data existed:

- **Amendment 1** (pre-data): corrected the anti-vacuity gate to per-variant expectations
  — the donor-re-keyed-alone variant is EXPECTED to persist on a fresh receiver (the same
  equivalence D44's kit documents), so asserting its collapse would have made the gate
  unpassable. The two must-collapse assertions were unchanged; no constant moved.
- **Amendment 2** (pre-data, live shakedown): the mock could not model the live world.
  A shakedown found `light_level` DEAD (reads 0 everywhere, a rest-at-extreme sensor) and
  the original slots inside the A4 gain's silence band (cos(rest, situation) = 0.9997, one
  cluster). Fix: drop `light_level`, move slots far + high — cos → 0.19, Phase 0 then
  passed all five. Notably, under the broken apparatus the transfer PILOT still passed
  bias-decisive while check 1 failed — the fused cluster made the bias fire situation-FREE,
  invisible to the pilot's own gates; check 1 is what caught it.
- **Amendment 3** (post-Phase-0-data, disclosure only): the gated Phase 0 readings above;
  no constant retuned.
- **Amendment 4** (harness robustness): the first live confirmatory campaign crashed ~88%
  through on two live-only races the multi-client mock never exposed — a single-client slot
  freed on an async close, and a blind settle observing pre-teleport state. Fixed with
  confirm-first-snapshot + bounded retry on `MinecraftClient.connect` and a
  `settle_until_reflected` poll (its own reviewed PR); the mock was made one-client-faithful
  so the class cannot regress unseen. No encode/teacher/merge/selector/gate path changed;
  the confirmatory campaign started over fresh on the fixed code. This is the lesson the
  world seam keeps teaching: the mock apparatus was too permissive, and live validation on
  a real server is what caught the geometry error and both connection races.

## Regression guard

**Re-run on:** `substrate_merge` / V1–V10 adapter (`hivemind/ingest.py`, `hivemind/merge.py`)
change, `NAc.credit_operant_reward` / `set_pending_operant_action` / world-cluster credit
routing change, `recommend_action(current_clusters=)` change, `SensorEncoder` / EC world-
modality change, `minecraft_bench{,_satiated}` body change, Minecraft bridge protocol
change, minor-version heartbeat.

**Guard:** [scripts/exp56/](../../scripts/exp56/) (`instrument_check.py` / `run_campaign.py`
`--mock`/`--resume`, the teacher + balanced schedule + selector in `common.py`) +
[scripts/analyze_exp56.py](../../scripts/analyze_exp56.py) (`--gate v1 --assert-noop-fails`,
frozen verdict constants) + guard tests
[tests/unit/test_exp56_harness.py](../../tests/unit/test_exp56_harness.py),
[tests/unit/test_hivemind_ingest.py](../../tests/unit/test_hivemind_ingest.py),
[tests/unit/test_inherent_bias_class.py](../../tests/unit/test_inherent_bias_class.py),
[tests/integration/test_oasis_ingest_e2e.py](../../tests/integration/test_oasis_ingest_e2e.py),
[tests/unit/test_minecraft_seam.py](../../tests/unit/test_minecraft_seam.py) + committed
data [data/56_four_arm.jsonl](data/56_four_arm.jsonl) /
[data/56_four_arm_verdict.json](data/56_four_arm_verdict.json) /
[data/56_phase0.json](data/56_phase0.json) + the pre-registration.
