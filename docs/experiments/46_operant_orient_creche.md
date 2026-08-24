# Experiment 46 — Operant orient: a mother teaches, a crèche pools

**Status:** COMPLETE (2026-07-22). Scripted, deterministic, on the real substrate + real Hivemind merge. No LLM in the action path.
**Scripts:** [`scripts/orient_substrate/4_operant_learning_curve.py`](../../scripts/orient_substrate/4_operant_learning_curve.py) · [`scripts/orient_substrate/5_operant_creche_federation.py`](../../scripts/orient_substrate/5_operant_creche_federation.py)
**Foundation:** probe [1](../../scripts/orient_substrate/1_motor_credit_probe.py)/[2](../../scripts/orient_substrate/2_full_path_probe.py) (intrinsic-drive orienting) + probe [3](../../scripts/orient_substrate/3_operant_feed_probe.py) (operant feed mechanism end-point).

## Claim

A hungry infant with **no intrinsic drive to orient** learns to turn toward a sound **purely because a mother feeds it** when its own turn moves toward the sound (operant conditioning). Remove the mother and it never learns. And a **crèche** of sample-limited infants can **pool** what each learned into the full-experience policy, via the real substrate-merge infrastructure.

## Mechanism

- **Body** (`bodies/infant_operant` shape): a `hunger` drive + an `azimuth` sensor with **`drive: null`** — the infant *perceives* sound direction but has no innate reason to orient. Turning relieves nothing intrinsically.
- **`NAc.credit_operant_reward(agent_id, reward)`** (+ `set_pending_operant_action`): an EXTERNAL, caregiver-caused drive relief (the mother feeding the infant *because* it oriented) reinforces the infant's OWN most-recent action on `_cluster_reward_bias` — the action-selection surface `recommend_action` reads. This is the mirror image of the self-effect motor-credit, which deliberately excludes a caregiver's `target_effect` so the caregiver's policy isn't credited by the recipient's relief. Operant conditioning credits the *recipient's* action by the relief *it* experienced.
- **Mother = operant shaper**: each tick she places a sound and, if the infant's turn moved toward it, feeds it (hunger relief) and credits that action.

## Result 1 — the mother teaches (learning curve)

`4_operant_learning_curve.py`, ε=0.2, 8 seeds, directedness = fraction of turns that moved toward the sound:

| | pre-learning baseline (first 5 ticks) | settled |
|---|---|---|
| **taught** | 0.65 | **0.90** |
| yoked (same reward, random action) | — | 0.36 |
| none (no credit) | — | 0.50 |

**LEARNED + MOTHER-TAUGHT PASS.** The infant rises from chance to 0.90; the controls stay flat at chance. `none`/`yoked` at chance is the decisive **"mother is necessary"** proof — an intrinsic-drive body (probes 1/2, contingent 1.000) would rise even in `none`; this one does not.

**It learns in ~10 ticks.** Operant orienting is *not* slow in the substrate — the slowness seen in the embodied sim (below) was machinery, not mechanism.

## Result 2 — a crèche pools its learning (federation)

`5_operant_creche_federation.py`, 12 infants × 2 ticks each, 10 seeds, using the real `hivemind/merge.py::nac_merge`:

| arm | directedness |
|---|---|
| single_partial (1 infant, 2 ticks) | 0.73 |
| single_full (1 infant, 24 ticks) | 1.00 |
| **creche_taught (12 × 2, MERGED)** | **1.00** |
| creche_none (12 × 2, no mother, merged) | 0.51 |

Twelve barely-taught infants (0.73 alone) merge into the **full-experience policy (1.00)**. The no-mother crèche stays at chance — **the merge pools *learning*, not noise.**

**Honest scope of the federation PoC:** the infants share a perceptual encoder, so a left sound maps to the *same* EC cluster id for every infant, and `nac_merge` combines (rather than unions) their `cluster_reward_bias`. This is biologically honest — every infant has the same cochlea, so the same sound clusters the same way; only the *learned policy* differs and is pooled. Fully-independent agents encode to *different* uuid clusters and would need `ec_merge` alignment first — that is now the gated 1.2 Oasis ingestion pipeline (re-scoped 2026-08-19). This PoC proves the pooling *concept* on the real merge function.

## Why scripted, not embodied (the honest methods note)

We first built this end-to-end in the `cradle_mother` embodied sim (all of it shipped + tested: the operant credit path, an operant-only credit mode, the `infant_operant` body, exteroceptive azimuth encoding, a substrate tool whitelist). It measured at **chance**, and chasing that produced a textbook **divergence** — each fix surfaced a new failure mode: the infant chose `sense_presence` (a tool that "always succeeds", causal_pos 0.99) over turning → a tool whitelist; then the explore-bonus weight (1.5) explored far more aggressively than a clean ε=0.2 and pinned it at chance; then lowering it stalled the confidence gate into 30-minute timeouts. At the time this read as "the embodied sim is the wrong instrument" — its LLM narrator (non-deterministic, slow, GPU-contended), gates, tool competition, and turn caps wrap a fast, clean operant signal in machinery that fights it — and the scripted substrate was the right instrument for the *mechanism*: deterministic, no LLM, thousands of ticks.

> **Update (2026-07-23, [Exp 48](48_cradle_mother_seam.md)):** the deeper root cause of the embodied chance result was not the tool-competition machinery but an **exteroception/interoception dilution** — `propose_via_substrate` merged azimuth into the interoception encode, so on the multi-drive `infant_operant` body left/right collapsed onto ONE EC cluster and the infant was structurally blind to direction. The [extero/intero seam](../plans/exteroception_interoception_seam.md) (PR #411) de-diluted it, and re-running THIS embodied sim (12 seeds/arm, same harness + tool whitelist) now **GRADUATES**: taught late-bin directedness **0.875** vs no_feed control **0.448** (+0.427), a clean rise 0.51→0.90. So the embodied instrument was recoverable after all — it needed the perception-layer fix, not just a cleaner substrate. The scripted result below remains the mechanism proof; Exp 48 is the embodied confound-check. (The tool whitelist is still applied — out-competing a snowballing always-succeed tool is the separate, still-open credit-on-progress question.)

## Result 3 — graded orienting (face the correct direction)

[`6_graded_orient_curve.py`](../../scripts/orient_substrate/6_graded_orient_curve.py): the sound comes from one of six directions and the infant has six graded turn actions; to be fed it must pick the turn that actually CENTERS the sound (right direction AND right magnitude). Six distinct (direction → correct-action) associations, chance ≈ 0.17.

**First finding — a perceptual limit.** A single `azimuth` scalar folds to just **2 EC clusters (left/right) at every pattern-separation threshold (0.44→0.93).** The substrate resolves *direction* but not *magnitude*, so the infant cannot perceive −0.9 vs −0.6 to choose a different turn — it plateaus at ~0.30 (learns one magnitude per side, centering ⅓ of positions). This is a *perceptual*, not a learning, limit.

**Fix — a place-cell population code.** Tiling azimuth into narrow direction-tuned cells (Gaussian bumps, width 0.12) gives **6/6 distinct clusters** — the resolution graded orienting needs. This is how brains encode space (a population code, not one number), and it should be the standard direction encoding for spatial tasks going forward.

With the place code (8 seeds): taught **0.19 → 0.82** (0.82 is the ε=0.2 exploitation ceiling — a mastered policy), yoked 0.03, none 0.17. **LEARNED + MOTHER-TAUGHT PASS.** The infant learned to face the sound's direction, taught by feeding alone.

## Result 4 — graded federation (coverage pooling)

[`7_graded_creche_federation.py`](../../scripts/orient_substrate/7_graded_creche_federation.py), 12 infants × 25 ticks, 8 seeds, chance ≈ 0.17. On the graded task a sample-limited infant only hears a few of the six directions, so its policy has *holes*; different infants have different holes.

| arm | centering rate |
|---|---|
| single_partial (1 infant, 25 ticks) | 0.59 |
| single_full (1 infant, 300 ticks) | 1.00 |
| **creche_taught (12 × 25, MERGED)** | **1.00** |
| creche_none (12 × 25, no mother) | 0.16 |

The crèche's coverage pooling recovers the full graded policy that no single partial infant has — this is where federation earns its keep (vs the modest payoff on the trivially-easy 2-alternative task). The merge pools learning, not noise (creche_none at chance).

## Raw data (S4 status, 2026-08-24)

The original 2026-07 runs printed to the terminal and were not captured. These probes are
scripted, seeded, LLM-free and run in seconds, so they were **re-derived on 2026-08-24**
and the full stdout of each is committed under
[`data/scripted_rederivation_2026-08-24/`](data/scripted_rederivation_2026-08-24/README.md)
(`46_4_*` … `46_7_*`), each run at the configuration stated in its Result section (Results
1 and 2 need explicit flags — `--seeds 8`, and `--agents 12 --ticks 2 --seeds 10` — because
the script defaults differ from what the doc reports). Every number above reproduces to
within rounding (the one visible move is Result 1's yoked control, 0.36 → 0.41, at chance
either way) and every pre-registered verdict reproduces. The tables above remain the
original measurement; the README carries the side-by-side and the per-run provenance.

## Next

- **Experiment 47 — habituation (DONE, see 47_habituation_novel_in_noise.md), individual vs collective:** modulate the orient response by novelty so the infant habituates to a constant familiar sound (city traffic) and still orients to novel ones (dishabituation). The novelty-decay machinery already exists (`tools/novelty.py::NoveltyRecord.novelty_score` decays with repetition; `attention/salience_map.py` weights novelty + inhibition-of-return). The new wire: familiarity (poolable EC cluster count) modulates orienting. **The novel question:** run habituation WITH and WITHOUT federation — because `ec_merge` accumulates cluster counts across contributors, a crèche-raised agent may habituate to sounds it never personally heard but the collective did (collective vs individual habituation in a hivemind). The rewarded direction (mother's voice) should *resist* habituation while unrewarded background habituates away — the "cocktail party" effect.
- **Extension (under discussion):** habituation — modulate the orient response by novelty so the infant habituates to constant familiar sounds (city traffic) and still orients to novel ones (dishabituation). Rides on the existing novelty/salience machinery (`attention/salience_map.py`, `tools/novelty.py`) + the place-cell code (a familiar *direction* is a high-count cluster).
- **Extension (under discussion):** habituation — modulate the orient response by novelty, so the infant habituates to constant familiar sounds (city traffic) and still orients to novel ones (dishabituation). Rides on the existing novelty/salience machinery (`attention/salience_map.py`, `tools/novelty.py`).
