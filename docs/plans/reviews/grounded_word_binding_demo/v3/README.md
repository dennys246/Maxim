# v3 review round — grounded_word_binding_demo.md (2026-09-24)

**Why this round exists.** The six reviews one level up (`../`) were written against **v2**
(`6ebcf9b1`). v3 (`00f74b37`, merged via #881 and #882) is the plan that folded them, so v3's
own text had never been reviewed. By CLAUDE.md's scope rule — a round covers the diff as it
existed when it ran — the merged plan was unreviewed. This round reviews v3 as merged.

**The plan has since been split and renamed** in a separate PR: Track S becomes
`docs/plans/social_referencing.md`, and this plan is deferred behind a frozen re-entry gate and
moved to [`docs/plans/deferred/grounded_word_binding.md`](../../../deferred/grounded_word_binding.md). The
reports below name the file as it was.

## The lenses

Five lenses, none duplicating the six v2 lenses (confounding, bio-fidelity, wiring,
environment, architecture, security). Each ran read-only against `main`; each report is kept
**verbatim** so the record shows what was actually said, including what turned out wrong.

| Lens | Question | Verdict |
|---|---|---|
| [Fold verification](fold_verification.md) | Did v3 actually fold the six v2 reviews? | The fold does not hold as claimed |
| [Claim discipline](claim_discipline.md) | Does each claim ceiling match what its design can measure? | Adopt with changes; no prereg until its findings 1–4 land |
| [Demo surface](demo_surface.md) | What does a viewer see, and does it ship? | Supersede the hosted-sandbox plan; neither plan ships anything a visitor can drive |
| [Sequencing](sequencing.md) | Can it run in the proposed order alongside 1.4? | Adopt with changes |
| [Measurement](measurement.md) | Can each experiment's statistics refute its claim? | Adopt with changes; no experiment can currently be pre-registered |

## Verification notes — read these before the reports

Every finding was re-verified against `main` by a second session. Where that disagrees with a
report, **this section wins**; the reports are deliberately not edited.

**Held:** the failed entry gate (the paired-data audit recorded "REDESIGN THE DATA SOURCE — not
a pass" on 2026-09-20 and "EXPLORATORY; not a revival" on 2026-09-21); the bio-fidelity
attribution over v3-only mechanisms; the pre-boot contradiction between decision 2 and J1; the
unbuilt recalled-situation read seam; no `n`, no non-inferiority margin, no frozen location and
no Exp C primary DV anywhere in the plan; both Exp C statistical critiques (yoking breaks
exchangeability; the yoked arm matches consult COUNT, not advice WEIGHT); the look-back buffer
decaying per `tick()` call with `tick_rate` stored and unused; the "model server" rig line;
`scripted_water.py` emitting only `state` / `action_result` frames; 1.3.2's `agent_loop`
decomposition; and the claim-discipline findings in substance.

**Corrected:**
- **The text encoder's model name** (demo surface #6). The weight point holds — Track L needs the
  `semantic` extra (sentence-transformers, torch, spacy) — but `LinguisticEncoder` loads
  **`paraphrase-mpnet-base-v2`**, its configured default in `similarity/encoder.py`.
  `all-mpnet-base-v2` is only the bare `_get_encoder` helper's default.
- **"Exp C reads Exp 62 as unrun"** (sequencing #10) is **false**: v3 cites Exp 62's night result
  (0.799) as a known boundary.
- **The DO-NOT-BUILD count** (fold verification #10) is **unverified**. The total depends on
  whether named items or headings are counted; neither the plan's "14" nor the report's "15" was
  reproduced. Do not repeat either number.
- **"Zero production constructors, tests only"** (sequencing #2). Nothing outside its own module
  constructs a `PerceptTraceBuffer`, which holds — but seven `src/` files reference it, and
  `memory/snapshot.py` and `simulation/fixture_orchestrator.py` take one as a real parameter.
  The reviewing session also first described those references as "only comments", having
  generalised from a truncated search; that was wrong.

**Missed by all five lenses, found at re-verification:**
- Track S's foreign-layer read path goes into `recommend_action` and the threat read — the
  selection code the 1.4 rungs run on — so Track S's `src/` waits for 1.3.2's decomposition, not
  only for being opt-in.
- The look-back buffer's ownership was a three-way dispute that needed one decision recorded
  identically in three places.

## Outcome

The owner split the plan, deferred the language track on a condition rather than a date, kept
E3 as 1.4's only may-fail headline, and stopped calling any plan "the demo". The decisions, the
frozen L0 re-entry gate and the fixes these reports prompted are recorded in the plans
themselves (`social_referencing.md`, `deferred/grounded_word_binding.md`, and the cross-references in
`roadmap_1_4.md`, `grounded_language_acquisition.md`, `public_oasis.md`, `maxim_hivemind.md` and
`memory_strength_and_forgetting.md`), not repeated here.
