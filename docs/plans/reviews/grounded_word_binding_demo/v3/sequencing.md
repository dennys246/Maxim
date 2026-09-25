# Sequencing-and-dependency lens — grounded_word_binding_demo.md v3

**Round:** v3 review, 2026-09-24. **Target:** `docs/plans/grounded_word_binding_demo.md` v3
(`00f74b37`, merged via #881/#882). **Reviewer:** Claude subagent, read-only. The report below
is verbatim. Corrections found during re-verification are recorded in [README.md](README.md),
not edited into the report.

---

## Sequencing-and-dependency review — `grounded_word_binding_demo.md` (v3)

Verified against Maxim `main` (eb09abc1), `roadmap_1_4.md`, `roadmap_1_3_x.md`, `grounded_language_acquisition.md`, `memory_strength_and_forgetting.md`, merged PRs #858–#882.

**1. BLOCKER — the entry condition was never passed, and the plan substitutes a claim-downgrade for the gate.** The parent line's gate is binary: pass → revive with prereg + four-lens; fail → both plans to `archive/`. Recorded dispositions: 2026-09-20 audit = "REDESIGN THE DATA SOURCE (… **not a pass**)"; 2026-09-21 re-audit = "EXPLORATORY; **not a revival**". The demo plan records both in its status table and in §Where this is referenced, but nowhere states that it is downstream of an unpassed gate. Worse, L0's "re-run `scripts/paired_data_audit.py` unchanged" cannot pass: a curated teacher's scripted strings differing by one word ARE templates, so the audit's own template test fails by construction. *Smallest fix:* one header sentence — the audit said "redesign the source", the curated teacher IS that redesign, so the owner's re-gate is L0's exit (lag distribution + vocabulary + separability table), not the unchanged audit — plus one line naming the archive branch (separability fails or vocabulary still templated ⇒ claims say "a label" and the language framing archives).

**2. BLOCKER — the look-back is claimed by three plans and owned by none; the plan's attribution is wrong and its escape hatch is circular.** The plan states `PerceptTraceBuffer` is "designed in R4's design review… R4 builds it". But: roadmap_1_4 §Parallel lines says "whichever line wires it first owns it"; `memory_strength_and_forgetting.md` open question 5 proposes **itself** as owner ("confirm at the Phase 2 kickoff", unresolved) and calls its Phase 2 the buffer's "first production caller"; and R4 sits in 1.4 **Phase 5**, entering "only when a rung names it", with its first audit explicitly "routing, not a new rule" — no look-back design review is scheduled anywhere. So L1/L2 depend on an artifact no plan owns, and "if R4 slips this line builds it from R4's reviewed design" presupposes the design exists. Aggravating: the memory line is what is actually in flight (2S-a/b/c merged 2026-09-24), so it will wire it first. Confirmed unbuilt: zero production constructors, tests only. *Fix:* drop the R4 attribution; make the memory line's Phase 2 the owner and L1/L2 declared consumers, filing the language requirements (~1 s window, text-triggered, several candidate situations) into that review now.

**3. BLOCKER — no slot relative to 1.3.2.** Repo is still 1.3.0; **1.3.1 and 1.3.2 are unshipped**. 1.3.2 decomposes `agent_loop.py` gated on the byte-identical-selection provenance test, 1.4 Phase 0 builds on the decomposed loop, and the divergence rule forbids refactoring while a may-fail experiment runs. L1 adds a new modality to the substrate selection path in exactly that loop. *Fix:* state that no L1/S1 `src/` change lands between 1.3.2's first and last slice (offline/design work is fine), and generalise the existing "#879 rebase" note into that rule.

**4. should-fix — Exp A's apparatus collides with Phase 0 item 4.** Exp A reuses the Exp 60/61/62 water apparatus; Phase 0(4) adds param-free primitive verbs = a **bridge protocol change** that fires Exp 56/60/61's re-run triggers by letter. An Exp A frozen before it collects rows on a bridge 1.4 then changes. *Fix:* fix Exp A's slot explicitly — wholly before Phase 0(4), or after its discharge annotation — and name which.

**5. should-fix — "nothing here powers E1–E3" separates claims, not cost.** T4 requires every Tier-1 EARNED row × every 1.4 `src/`, bridge and harness change be walked FIRED/NOT FIRED with no `Stale` token — and L1 deliberately *adds* a "Re-run on" trigger to Exp 60/61/62's rows. This line therefore adds 1.4.0 release-gate debt while claiming nothing. *Fix:* say so, and file L1's trigger with a discharge plan (the Exp 60 re-run already scheduled in 1.3.2).

**6. should-fix — the S-side security prerequisites must ride 1.3.1's CI lane, not precede it.** Per roadmap_1_3_x, **no lane installs fastapi or cryptography**, so bundle-signing, hive-pull and Oasis-exchange tests are skipped everywhere. Moving `signer_identity` under the signature changes what the signature covers — a compatibility change on the path both the 1.2 and 1.3 headlines travel — and would ship unverified. *Fix:* file the three S1 prerequisites as 1.3.1 items behind that lane, with a stated compatibility answer for existing signed bundles and 1.3.0 receivers.

**7. should-fix — Exp C has an unstated corpus dependency and both experiment budgets are understated.** A "stale Oasis" needs real entries covering the situations the agent meets, but there is no contribution path (promotion stays WRITE-ONLY, explicitly out of scope), so donor agents must be produced per situation — Exp 56/61-style rig time absent from the 3–5 h line. Sizing: Exp C at 5 arms × 2 conditions × ≥20/cell = 200 rows; Exp 62 measured 27 rows in 46.9 min, so ≈6 h of pure row time before pilots, dry runs and resets, with longer episodes than WaterTrial. Exp A is 8 arms × ≥20 = 160 three-phase rows against 3–7 h. *Fix:* relabel budgets "per campaign attempt, excluding donor production, pilots and dry runs"; n stays with each prereg.

**8. should-fix — open question 4 is mis-framed: Exp B is a join, so S-first buys no span.** Exp B needs a donor that PASSED Exp A *and* J1; J2 needs L1 + S2. The demo's critical path runs through Track L in every ordering. *Fix:* one line — Track S fills operator slack, Track L is the critical path, so the look-back owner decision is the schedule's only real lever.

**9. nit — the contract table's "Oasis side" column has no named owner** while there is one operator and one rig; the two "tracks" serialize on that person. *Fix:* say the Oasis-side items interleave, not parallelize.

**10. nit — Exp C reads Exp 62 as unrun** ("until Exp 62's night boundary is understood"); Exp 62 is EARNED 2026-09-20 and the night pool (0.799) is measured but *unowned* (Rung B SHAPE done; keying did not enter). *Fix:* "understood" → "has an owner in R1/keying".

### Contention with 1.4 (finding 5 in full)
Separation is achievable for *claims* and not for *resources*. Both lines use one rig ("big-mac-mini only, once quiet"), one operator who hand-starts the Paper and model servers, the same water column, and the same bridge. 1.4's §Schedule is explicitly designed to keep that rig saturated with Phase 0 pilots, E1, E2, E3. What breaks first, in order: (i) the **operator** — the serializing resource, since every campaign needs a freeze, a dry run and a supervised start; (ii) the **bridge/apparatus** (finding 4); (iii) the **ledger trigger walk** at the 1.4.0 release gate (finding 5); (iv) prereg-freeze discipline, as rebases against in-flight loop work tempt post-freeze harness edits.

### Recommended order of work
1. **Two owner decisions, no code:** the look-back's owner (→ memory line Phase 2, language as consumer) and the re-entry wording for the failed audit gate.
2. **L0's offline half now** — teacher-string separability, text formation threshold, refuse the 384-d fallback, bridge timestamp design. Free, dev-box only, needs neither R4 nor the rig, and it is the line's cheapest kill test.
3. **Ship 1.3.1**, including the three S1 security prerequisites behind its console+crypto lane.
4. **1.3.2 decomposition** — no language/Oasis `src/` change in this window; L0's natural-death capture (separate world) takes the rig while the dev box is busy.
5. **S1 → S2 → Exp C** (donor corpus first), fitted into 1.4's rig slack.
6. **L1 → L2** as consumers of the landed look-back → **Exp A**, slotted relative to Phase 0(4).
7. **J1 → Exp B → J2.**

So: yes, start S before L's *code* — but start L0's offline measurement before either, and do the S security work as 1.3.1 items rather than as Track S.

**Longest pole:** the unowned look-back primitive (`PerceptTraceBuffer`) — it gates L1→L2→Exp A→Exp B→J2, i.e. three of the four demo experiments, and no plan currently owns its design review. (Fixed cost in front of everything: 1.3.2's decomposition.)

**Span:** four release cycles, not one — 1.3.1 (S prerequisites), 1.4.x (Exp C), 1.5 (Exp A), 1.5.x/1.6 (Exp B + J2); nothing in the demo can land inside 1.4.0 without displacing a rung.

**Verdict:** ADOPT WITH CHANGES — the plan is honest that it is multi-release, but it is wrong about its two hardest dependencies: it reads an unpassed entry gate as passed-by-redesign, and it assigns the look-back to an owner (R4) that no roadmap schedules and another live line already claims.
