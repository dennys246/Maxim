# Experiment 47 — Habituation: a novel sound in a wall of noise

**Status:** COMPLETE (2026-07-22). Scripted. **Script:** [`scripts/orient_substrate/8_habituation_novel_in_noise.py`](../../scripts/orient_substrate/8_habituation_novel_in_noise.py). Builds on the operant orient + crèche work (experiment [46](46_operant_orient_creche.md)).

## Claim

Novelty/salience lives on the **joint `(sound-content × direction)`**, not direction alone — a violated expectation *in context*. A familiar car engine from the road habituates; the *same* engine from the bedroom is a rare joint → surprising. Habituation is not mere forgetting: it is **noise suppression in service of novelty detection**, and it is what lets a faint novel sound win against a wall of familiar noise. A rewarded content (the mother's voice) resists habituation. And a **hivemind pools its familiarity**, so a crèche-raised agent suppresses noise the *collective* found common even if it never personally heard it.

## Model (salience gate)

An agent facing a scene of concurrent sounds attends to the most salient:
```
salience(sound) = novelty(content, direction)          # decays 0.85 / repetition (tools/novelty.py) — habituation
                + reward_weight × value(content)        # learned-important content stays salient
```
Novelty is tracked over the joint `(content, direction)` count. (Honest scope: tracked *directly* here — the substrate's `LinguisticEncoder` is text-based and does not cleanly cluster a numeric content×direction code, so EC-native joint clustering, which would add *generalization* across similar sounds, is a follow-up. The salience model + the questions are faithful.)

## Results (200 seeds/level)

**1. Habituation enables novelty-in-noise.** Catch-rate of the novel sound as the background noise grows:

| noises | habituating | flat (no habituation) | chance |
|---|---|---|---|
| 1 | 1.00 | 0.46 | 0.50 |
| 5 | 1.00 | 0.18 | 0.17 |
| 10 | 1.00 | 0.07 | 0.09 |
| 20 | 1.00 | 0.02 | 0.05 |
| 40 | **1.00** | **0.04** | 0.02 |

The habituating agent catches the novel sound even in a 40-noise wall; the non-habituating control drowns (tracks chance). **Habituation has a function.**

**2. Reward protects salience (cocktail party).** The mother's voice, heard as often as the noise (so equally habituated), is still attended **1.00** — a rewarded content beats habituation. Tune out the hum, snap to your name.

**3. Collective vs individual habituation (federation).** 20 noises; each of 12 infants hears only half:

| | catch-novel |
|---|---|
| solo (heard half the noises) | 0.06 |
| **crèche (pooled counts)** | **1.00** |

Alone, an infant's un-heard noises are not habituated and compete with the novel sound. The crèche pools everyone's familiarity, so collectively every noise is habituated — **a hivemind suppresses noise no single member personally experienced**, and the novel sound triumphs. This is the collective-expectation claim: shared substrate → shared model of "what comes from where is normal" → shared surprise.

## Raw data (S4 status, 2026-08-24)

The original 2026-07-22 run printed to the terminal and was not captured. The script is
seeded, numpy-only and runs in under a second, so it was **re-derived on 2026-08-24** and
the full stdout is committed at
[`data/scripted_rederivation_2026-08-24/47_8_habituation_novel_in_noise.txt`](data/scripted_rederivation_2026-08-24/47_8_habituation_novel_in_noise.txt).
Every number in the tables above reproduces exactly.

## Next

- **EC-native joint novelty:** once the substrate's percept representation supports clean joint `(content × direction)` clustering, move novelty onto the EC cluster count so habituation *generalizes* (a novel sound *similar* to a familiar one partially habituates). Needs an encoder-representation pass (the text-based `LinguisticEncoder` is the current limit).
- **Salience → orient wiring:** feed the salience gate into `attention/salience_map.py` upstream of `recommend_action` so the attended sound drives the (already-validated) orient policy end-to-end.