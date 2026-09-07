# R1 — Cross-layout generalization: a structural null (representation vs cache)

**Status: `CACHE-CONFIRMED` (structural null), 2026-09-07.** The Minecraft survival
ladder's R1 rung ([minecraft_benchmark.md](../plans/minecraft_benchmark.md) Part II)
asks whether Exp 56's shared want carried a generalizable **representation** or a
cached **association**: does a want taught at layout S1 fire when the receiver meets
the contingency at a layout it never trained on? **It does not — and this is knowable
from the substrate mechanism, without a live campaign** (the R2 precedent). The
shared want is an exact-key cluster cache with no cross-layout generalization channel.
A null ships as a null (the Exp 53 / R2 shape).

## Why a live campaign is the wrong instrument

The pre-freeze two-lens review of a *live* R1 design caught that the campaign's
headline outcome was architecturally unreachable, so the campaign could only
re-confirm the source code. The mechanism decides it:

- The cluster-keyed learned-bias readout — the channel Exp 56's transfer runs
  through, and the DV R1 would use — is a **pure exact-key dict lookup**:
  `NAc.cluster_reward_bias` → `self._cluster_reward_bias.get((agent_id, cluster_id,
  tool_signature), 0.0)`. No similarity, no neighbour consultation. `recommend_action`
  reads it once per active cluster with that exact `.get()`.
- A world layout maps to a cluster id by a **0.85 cosine threshold**
  (`SENSOR_MODALITY_THRESHOLDS["world"] = 0.85`, world in the frozen-centroid
  modalities). At ingest, A's S1 node is inserted into the fresh receiver and the
  bias re-keyed 1:1 to that node id (`ec_merge_aligned` + `rekey_nac_state`).
- Therefore: a layout **distinct enough to be a different situation** (cos < 0.85 →
  a different cluster id) necessarily **misses** the key the want was taught on →
  `learned_bias` 0 → no transfer. The only layout that fires is one that collapses to
  S1's own cluster (cos ≥ 0.85) — which is not a different layout at all.

**"Generalizes" and "is a genuinely different layout" are the same 0.85 comparison
with opposite sign; they cannot both hold.** So a live cross-layout campaign, gated
(correctly) on the layouts being pairwise-distinct, would return collapse at every
distinct layout by construction. This is exactly the R2 situation — a structural fact
a fresh-substrate probe isolates exactly — and consistency with how R2 and Exp 53
shipped their nulls demands R1 ship the same way: offline, not as a 20-cohort live
campaign framed as genuinely two-sided.

## The demonstration (end-to-end through the real path)

- **Instrument:** [scripts/r1_cross_layout_probe.py](../../scripts/r1_cross_layout_probe.py).
  A donor is taught the want at S1 (a cluster-keyed operant bias on the target, plus a
  small uniform S1-cluster baseline so every tool scores and the target is
  learned-bias-*decisive* there). The bundle is ingested into a fresh receiver through
  the **real** 1.2 path (`substrate_merge` — `ec_merge_aligned` + re-key + `nac_merge`).
  The receiver is then probed at three layouts, reading learned-bias-decisiveness from
  the `NAc_RECOMMEND` provenance at the **real consumer** (`recommend_action`), never
  the emitted action or dict internals (the D44 rule).
- **Record (gated):** [data/r1_cross_layout.json](data/r1_cross_layout.json).

| probe layout | cluster vs S1 | taught want fires? |
|---|---|---|
| **S1** (trained) | same | ✅ decisive (control — the transfer works, replicating Exp 56) |
| **S_distinct** (unseen, cos < 0.85) | **distinct** | ❌ **collapses** — the want does not transfer |
| **S_same** (1-block perturbation, cos ≥ 0.85) | same | ✅ decisive — but this is the *same situation*, the vacuous case |

`verdict: CACHE-CONFIRMED` — fires at S1, collapses at a genuinely distinct layout,
and "transfers" only when the layout is not actually different. The transfer Exp 56
measured moves a **situation-specific cache entry**, not a concept that recurs.

## What this shows — and what it does NOT

**Shows:** the substrate stores a taught want as an exact-key `(agent, cluster, tool)`
cluster bias. Sharing (Exp 56) faithfully moves that entry to a new agent, and it reads
out wherever the receiver re-encodes the *same* situation — but there is **no
mechanism** by which it reads out at a different situation cluster. Cross-layout
generalization of a learned want is **architecturally absent** in 1.2.

**Does NOT:**
- **Refute Exp 56.** Exp 56's cross-*agent* transfer at the *same* layout stands; R1
  only bounds what that transfer is (a cache entry, not a concept).
- **Claim generalization is impossible in principle.** It is absent *given the current
  readout*. A generalization channel — a similarity-weighted bias read across
  neighbouring clusters, or a coarse "situation-kind" cluster above the fine one — is
  **1.3-line engineering**; R1 cannot test it until it exists, and building one to make
  R1 pass would be engineering the outcome (D1's spirit).
- **Depend on the pairwise-separation risk.** Unlike a live campaign (which needs the
  four frozen slots to separate — the Exp 57 check-1 risk), this offline demonstration
  simply *chooses* two genuinely-distinct layouts (far vs near), so it does not wait on
  that apparatus question.

**Honest scope in one line:** the shared want is an exact-key cluster cache;
cross-layout generalization is absent in 1.2 and is a designed 1.3 mechanism, not a
back-fit. This reshapes the Oasis thesis exactly as the benchmark doc anticipated
("collapse to floor is a publishable result that reshapes the Oasis thesis").

## Regression guard

**Re-run on:** `NAc.cluster_reward_bias` / `recommend_action` learned-bias-readout
change (esp. any similarity-weighted or hierarchical cluster read), `substrate_merge` /
`ec_merge_aligned` change, `SENSOR_MODALITY_THRESHOLDS["world"]` change, `SensorEncoder`
/ EC world-modality change. **If a 1.3 build adds a cross-cluster generalization
channel, this probe flips from `CACHE-CONFIRMED` toward transfer at distinct layouts —
that is the signal R1 becomes a live experiment.** **Guard:**
[scripts/r1_cross_layout_probe.py](../../scripts/r1_cross_layout_probe.py) +
[data/r1_cross_layout.json](data/r1_cross_layout.json) +
[tests/unit/test_r1_cross_layout_probe.py](../../tests/unit/test_r1_cross_layout_probe.py).
