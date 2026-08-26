# Exp 52 — Nurture: caregiver-taught orienting through hunger relief

**Status:** COMPLETE 2026-08-25 — **PASS on both phases, all pre-registered gates.**
**Pre-registration (frozen, amendments 1–2 pre-data):**
[protocols/exp52_nurture_preregistration.md](protocols/exp52_nurture_preregistration.md).
**Harness:** #543 (`e367f526`). **Runs:** Phase A `main` @ `e367f526` (operator Mac);
Phase B `main` @ `60195a29` (big-mac-mini, `~/exp52/phaseB`).
**Raw data (S4):** [data/52_phaseA_scripted.json](data/52_phaseA_scripted.json) ·
[data/52_phaseB_embodied.jsonl](data/52_phaseB_embodied.jsonl) ·
[data/52_phaseB_runs/](data/52_phaseB_runs/README.md) (per-run provenance; mother logs
archived off-repo) · [data/52_dryrun_nonfrozen.json](data/52_dryrun_nonfrozen.json)
(harness verification, not a result).

## The question

Does an infant with a hunger drive and **no** orient drive learn to turn toward its
mother's voice when the only consequence of turning toward her is being fed — with the
credit the substrate receives derived from the **relief the infant actually experienced**,
not a constant handed to the learner? Three things had to be true: it learns when
hungry and fed contingently (LEARNED); it does **not** learn when fed contingently but
never hungry (HUNGER-NECESSARY — the arm Exp 48 never had); it does **not** learn when
fed on the same schedule non-contingently, or not at all (MOTHER-NECESSARY).

## The mechanism (what changed vs Exp 48)

The mother's feed already wrote a real hunger delta in Exp 48; the credit was a
constant `feed_reward=1.0` — a full infant was credited exactly like a starving one.
Exp 52 makes the operant credit's value **`sign(Σ drive_comfort_progress)`** over the
drives the feed touched (the existing channel-3 value-progress signal, scored on the
recipient via `tool_bridge._drive_potential_diff`), delivered through the existing
one-turn pending-operant trace; **zero relief → no credit.** No new mechanism class;
`cradle_mother.reactive_mother_tick(credit="relief")`, harness #543.

## Phase A — scripted substrate (8 seeds × 600 ticks) — PASS

| arm | baseline (first 5 ticks) | settled (last 4 bins) | feeds / credits per seed |
|---|---|---|---|
| taught | 0.650 | **0.892** | 532–544 / 532–544, all +1 |
| satiated | 0.475 | 0.496 | 274–315 / **0** |
| yoked | 0.575 | 0.496 | = taught seed's feeds / all credited |
| no_feed | 0.475 | 0.496 | 0 / 0 |

LEARNED (+0.24 ≥ 0.15, settled ≥ 0.80), HUNGER-NECESSARY (+0.40), MOTHER-NECESSARY
(+0.40): PASS. Satiated and no_feed curves are **identical to the digit** — with no
credit minted both are the same seeded random walk; the feed without need had zero
effect. Yoked received every one of the taught arm's 4305 credits, decoupled from its
own actions, and stayed at chance: contingency carries the learning, not reward volume.

## Phase B — embodied `cradle_mother`, apparatus v3 (12 seeds/arm, 48 turns) — GRADUATE

Shuffled stimulus order (#514), relief credit, `explore_weight 1.5`, mistral-7b narrator
at temperature 0, exposure-matched (48 turns/seed all arms), ~663 s/run.

| arm | act1 | act2 | act3 | act4 | late (act3+4) | per-seed late SD | fed rate | credited rate |
|---|---|---|---|---|---|---|---|---|
| taught | 0.61 | 0.85 | 0.87 | 0.89 | **0.878** | 0.130 (6 distinct) | 0.73 | 0.73 |
| satiated | 0.34 | 0.43 | 0.43 | 0.45 | 0.441 | 0.079 (8 distinct) | 0.35 | **0.00** |
| no_feed | 0.33 | 0.40 | 0.38 | 0.44 | 0.413 | 0.082 (6 distinct) | 0.00 | 0.00 |

Gate v3 (`analyze_cradle_mother.py --gate v3`, frozen 2026-08-25):

- **LEARNED: PASS** — late 0.878 ≥ 0.65, rose +0.26 from act1 (0.614; not at ceiling).
- **MOTHER-TAUGHT: PASS** — taught − no_feed = +0.465 ≥ 0.20.
- **HUNGER-NECESSARY: PASS** — taught − satiated = +0.437 ≥ 0.20; satiated rise +0.10
  < 0.15; satiated late 0.441 ≤ no_feed 0.413 + 0.20 (amendment-2 cap).
- **APPARATUS (L2): clean** — every arm shows real seed spread (SD 0.08–0.13, 6–8 distinct
  late values). The v2 signature — exact, seed-invariant twelfths — is gone: the shuffle
  broke the phase-lock, and directedness is a graded measure again.
- **APPARATUS (S3): OK** — satiated fed on 35% of turns and credited on 0%; no negative
  reward; no credit without relief. Every act record's *observed* credit mode is `relief`.

**Per-seed taught late bins:** 0.88 0.92 0.96 0.75 1.00 0.96 **0.54** 1.00 0.96 0.88 0.75
0.96. Seed 48 is the low outlier. Its decision provenance (#504, in the archived mother
log) says why: median `|learned_margin|` **0.12** — sitting on the ~0.11 argmax
visibility floor ([L1](../limits/README.md)) — so the exploration term decided **18%** of
its choices, against 3–11% for the other eleven seeds (median margins 0.38–0.62). A weak
learner, not a non-learner. The satiated seed 48, for contrast: margin 0.0, exploration
decides 50% — a pure random walk, as designed.

## What this shows — and what it does not

- **Shows:** on both the clean substrate and the embodied sim, an infant with no orient
  drive acquires "turn toward the voice" from the mother's contingent feeding *only when
  the feed relieves something*. Same feed events, same contingency, zero need → zero
  learning, embodied (satiated 0.441 vs the teacherless control 0.413). That is the
  operational content of "learns to want to orient from a primary reward", and it is the
  claim the 1.1 "Sensorimotor" cut was reopened to test.
- **Does not show (stated in the pre-registration, restated here):** "fed while hungry"
  in the everyday sense — the sign-only credit discriminates nonzero-from-zero relief;
  hunger at feed is ≈ 0.05 (Phase A) to ≈ 0.1 (Phase B, e.g. `relief=0.177` on a typical
  fed turn), far below the deprivation threshold. Not modeled: secondary reinforcement of
  the mother's voice itself, or devaluation (a learned bias reads out identically when
  sated). Nothing about magnitude selection (Exp 45b–e), the LLM-AUT path, loudness, or
  multi-turn credit.
- **Evidential weight:** Phase B is **one session, n = 12/arm**; the cross-session
  replication caveat that sits on the Exp 45 hardware row applies here too until a
  second session is run (the row's *Re-run on:* triggers name it).

## Ledger consequences

- New Earned row in [behavioral_graduation_candidates.md](../plans/behavioral_graduation_candidates.md)
  ("Caregiver-taught orienting through hunger relief"); the Exp 48 row's PARTIAL is
  superseded — Exp 48 stays as the apparatus case study (v1 contest → v2 re-baseline →
  sweep → the shuffle) that made this measurement possible.
- [L2](../limits/README.md): mitigation now **measured** (the shuffle restores seed
  spread). [L1](../limits/README.md): the margin instrumentation explained the one weak
  seed exactly as designed.
- Roadmap item 17 → DONE; item 18's bench ran the same day and its design moved to 1.1.1.
- **2026-08-26 — these infants were read out on the physical Reachy Mini** ([Exp 53/53b](53_cross_context_readout.md)): files unchanged, taught 1.00 / satiated 0.00 / no_feed 0.50 — the cross-context half of the claim.
