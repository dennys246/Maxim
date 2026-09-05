# Oasis case study — sharing a nursery-taught want (1.2 kickoff, adopted 2026-08-26)

**Status:** ADOPTED 2026-08-26 as the motivating case study for 1.2 Oasis + Hivemind
(owner decision). Design pass, not implementation: this document names the three
contracts the case study forces and the pre-registered result that would earn the
sharing claim. Parent plan: [maxim_hivemind.md](maxim_hivemind.md). Evidence it rides on:
[Exp 52](../experiments/52_nurture.md) (the want is learned) →
[Exp 53/53b](../experiments/53_cross_context_readout.md) (the want reads out on a
physical body, unchanged).

## Why this artifact

The taught infants of Exp 52 are the smallest real learned substrate the project has:
two JSON files per agent (`aut_nac.json` + `aut_ec.json`, ~350 KB), already SHA-manifested
([53_agents_manifest.json](../experiments/data/53_agents_manifest.json)), already
stamped with encoder provenance (`sensor:audio = ["azimuth"]`, range-aware), already
validated cross-context on hardware (taught 1.00 / satiated 0.00 / no_feed 0.50). It
carries no episodes, no identity, no dialogue — the `hivemind/` privacy invariants hold by
construction. And it breaks the shipped §B5 surface in exactly the places Oasis must be
designed around, which is what a case study is for.

The claim ladder it completes: 1.0 cross-**session** (Exp 42/45) → 1.1 cross-**context**
(Exp 53b, one robot) → 1.2 **cross-unit** — *someone else's* Reachy Mini runs a want it
never learned.

## What already exists (audit before building — `src/maxim/hivemind/`, shipped §B5)

- `bundle.py::compose_bundle` — ZIP with `manifest.json` (`_format_version`,
  contributor, domain, reserved signature slots, `encoder_provenance`), `nac.json`,
  `ec.json`; the composition scrub keeps identifier-shaped signatures, so the learned keys
  `tool:infant_operant_turn_left` / `_right` ship intact (verified 2026-08-26).
- `merge.py::ec_merge` — aligns foreign `substrate_nodes` to local ones by cosine
  (threshold 0.44, same modality), count-weighted centroid mean, inserts unmatched nodes
  under their own id. `merge.py::nac_merge` — mean-with-clamp on identical keys.
- `cli.py substrate export|import|inspect|merge-nac` — extract is side-effect-free.
- Gauntlet #1: `scripts/orient_backbone/live_3_learn.py::probe_policy` (tabular az-bin
  policies, no hardware).

## The three contracts the case study forces

### 1. Bundle = the pair + its namespace — **DECIDED 2026-09-01: (a), with (b)'s key emitted alongside** (reconciled 2026-09-04)

A shareable want is `nac.json` **and** the `ec.json` it was formed in (D2: the bias keys
are EC cluster ids; one without the other is dangling). The manifest must additionally
declare the **action namespace** the biases key on: today `tool:<entity>_<affordance>`
from `tools/discovery.py`, i.e. `infant_operant_turn_left`. A receiver running
`bodies/reachy_mini` names the same affordance `reachy_mini_turn_left` and the bias is
invisible. The two resolutions this section front-gated, and the outcome (this section
and roadmap gate 7 now say the SAME thing; full costing:
[d43_merge_correctness.md](d43_merge_correctness.md) §5/§5a):

- (a) **typed bundles** — **CHOSEN, shipped 1.1.3**: `manifest.body_ref` +
  `manifest.affordance_namespace` (`BUNDLE_SCHEMA_VERSION` 2, v1→v2 migration); the
  receiver refuses a mismatch (`assert_bundle_body_compatible` —
  `BundleBodyMismatch`) and refuses an UNDECLARED body (`BundleBodyUnverifiable`)
  unless `allow_unverified=True`. Plus the insurance neither option originally
  included: `manifest.capability_map` carries (b)'s body-agnostic
  `(modulator, affordance)` key beside each body-prefixed signature, so (b) later
  becomes a reader-side change on new bundles, no payload migration.
- (b) **body-agnostic keys** — bias keys on the SEM modulator/affordance
  (`orient/turn_left`), the entity prefix dropped at the credit choke point.
  **Scheduled, not chosen now**: seventeen `f"tool:{...}"` sites bypass
  `build_tool_signature` (one miss = a silent zero shaped exactly like D43), the
  modulator vocabulary is not yet shared across bodies (`listen` under `head` vs
  `capture`), the bundle scrub drops `:`/`/`-shaped keys, and migration would touch
  the 20 SHA-manifested `53_agents` evidence files.

(b) stays the better long-term key; (a) is the honest first step and is what Exp 53b
actually did (explicit δ map, infant body). The 1.2 Oasis ingestion adapter routes
through `assert_bundle_body_compatible` — it does not re-implement the check.
**Prerequisite either way — 1.1.x item 15: a Reachy-native nursery body** so the taught
keys are the robot's own and the S6 δ map disappears.

### 2. Gauntlet #2 = Exp 53's harness, with controls

Promotion to Queen tier for an "orients to voice" want = the
[`exp53_cross_context_readout.py`](../../scripts/orient_backbone/exp53_cross_context_readout.py)
readout, dry rig or hardware, against zero-bias controls: Gate I (percepts complete
into the bundle's clusters; probe correct with `|learned_margin| > 0.11`) and Gate T
(delivered directedness ≥ 0.70 and ≥ 0.20 over controls). It already rejects the right
things — seed 48, a mis-learned nursery graduate, fails 0/12 — which makes it the first
gauntlet with teeth. Bundles ship **with their gauntlet record** (the JSONL + verdict),
not just their files: a three-bin representation that turns the wrong way at +0.2 is
shareable only with that caveat attached.

### 3. Merge = cluster identity across substrates (the real unsolved bit)

Exp 45's `nac_merge` worked because tabular az-bins shared keys. Two nursery runs mint
**different EC cluster UUIDs** for the same azimuth bins (seed 42's RIGHT is `5f6fab30`,
seed 43's is `429e7f5a`). `ec_merge` already aligns the *nodes* by cosine, but nothing
re-keys the *biases*: `nac_merge` averages on identical keys only, so a foreign
`cluster_reward_bias` lands under a cluster id the local EC will never emit — a merged
want that silently reads out as nothing. The gap is precise and small: **`ec_merge`
must return the right→left id map, and the composer/adapter must re-key
`cluster_reward_bias` / `cluster_reward_source` through it before `nac_merge`.** Whether
that rides on existing code (a returned map + a key rewrite) or needs semantics of its
own (what happens when two foreign clusters fold into one local one — mean, max,
count-weighted?) is the audit question for the design pass; the taught seeds 42 + 43 are
the test pair, and the merged bundle must pass Gauntlet #2.

Encoder coupling sits under all three: every bundle is valid only under the exact
`_sensor_embed` bases + normalization it was encoded with. The provenance stamp exists;
Oasis makes it a **hard compatibility gate** (1.2 gate 1, already listed) — an encoder
change orphans the whole Oasis unless migration is explicit.

## Pre-registered result that earns the sharing claim (to be frozen at the 1.2 kickoff)

**Cross-unit readout.** A taught bundle exported from this repo's archive (seed 43 or
44 — the ones that passed 4/4 with margin) is imported on a **second Reachy Mini** by
someone who did not run the nursery, with the Reachy-native body, and read out under
Gauntlet #2 with the satiated bundle as the control. PASS = the 1.2 sharing claim in one
sentence: *a want learned in one nursery drives a robot that never learned it.* Then the
merge arm: seeds 42 + 43 merged through the re-keyed path must still pass the gauntlet
and must not pass with either half's clusters dangling.

## What this does NOT change about 1.1.x / 1.2 scope

Nothing lands in 1.1.0 or 1.1.x from this document except item 15 (the Reachy-native
nursery body, a re-run of Exp 52 on the body users own). Oasis ingestion, the re-keyed
merge, bundle typing and the cross-unit experiment are 1.2, behind the existing "Gates
before 1.2" list plus the two added here (roadmap).
