# Measurement-and-statistics lens — grounded_word_binding_demo.md v3

**Round:** v3 review, 2026-09-24. **Target:** `docs/plans/grounded_word_binding_demo.md` v3
(`00f74b37`, merged via #881/#882). **Reviewer:** Claude subagent, read-only. The report below
is verbatim. Corrections found during re-verification are recorded in [README.md](README.md),
not edited into the report.

---

## Measurement-and-statistics lens — grounded_word_binding_demo.md v3

### Findings

**1 — BLOCKER. Exp A §Arms: the extinction arm and the timing-split ablation discriminate a route that v3 closed by construction — DNB-1 repeating.** Mediated conditioning requires either backward `NAMES` traversal or valence written to the text cluster. Decision 1 forbids both (`text` gets no fear/credit unless an arm declares it; recalled situations are read-only; `NAMES` is stated one-way). Failure: both ablations abolish the effect and the late-ablation arm reads 0/12; the record says "retrieval at test, mediated conditioning refuted" when the architecture forbade the alternative. *Fix:* state per write path, by `file::symbol`, that route (b) is closed; demote both arms to pre-registered **leak checks** with known answers (a surviving effect after late ablation = INCOMPLETE-with-cause, not a mediation finding), and stop listing "the effect runs through the situation, not the word's conditioning" as a measured discrimination.

**2 — BLOCKER. L2 / Exp A: `PerceptTraceBuffer` cannot express a "~1 s look-back window" — the R3 failure mode in the named instrument.** `percept_trace_buffer.py::tick` decays by exp(−1/τ) per *`tick()` call*, τ=10 ticks, `min_activation` 0.01 (≈46 ticks of life); `tick_rate` is stored and never used in decay; `snapshot()` orders by activation and `recent(k)` by insertion — neither by elapsed time. Measured loop cadence is 0.555–0.583 s idle, 1.4–1.8 Hz proposing (R3), and Exp 60 run 1 got **one tick per 4.3 s window**. Failure: a fixed k or activation floor spans 0.5 s in one arm and 20 s in another; binding accuracy then measures tick phase, not the lag. *Fix:* select the window on `TraceEntry.registered_at` versus the bridge's 100 ms event stamp, record **ticks-in-window per binding**, refuse a binding with < 2 in-window ticks, and report the like-for-like idle cadence as covariate (R3 Amendment 2's rule verbatim).

**3 — BLOCKER. Exp A §Phase 3: no representation gate that the test utterance resolves to the bound text cluster.** Text centroids drift and share space with affordance names (plan's own words); the formation threshold is set high to separate one-word-apart strings. Failure: the test word forms a *new* text cluster with no `NAMES` edge, 0/12, recorded as a behavioural null for the claim. *Fix:* add Exp 61 lifecycle-step-4's shape — before the test placement is spent, assert the heard test utterance completes into the *same* text cluster id as phase 1 and that the recalled world id equals the feared id, each miss its own named refusal class (measured before, never a post-hoc filter — Exp 61 F1).

**4 — BLOCKER. Exp A: no named primary DV, no gates, no n per arm.** "The executed choice" is undefined on a dry cell where the response is a no-op; the plan carries only the lens's "floor". Failure: the prereg cannot be written from this text and the thresholds get chosen after the pilot. *Fix:* name the DV as one binary per agent at the **first** word-alone test (later tests are extinction trials — unfolded lens N2), with Exp 56/61 decision provenance (drive component decisive, `causal == 0`, `learned_bias == 0`), and assign n per arm: 24 to every floor arm (Wilson upper for 0/24 = 0.137 < 0.20; 0/12 = 0.239 is above it), 12 to claim arms. Fisher one-sided at 12 v 24 has floor 1/C(36,12) ≈ 8×10⁻¹⁰ — reachable, Holm over ≤ 3 named contrasts included.

**5 — SHOULD-FIX. Exp A: null-shaped arms have no equivalence margin.** "Retrieval predicts the effect drops with extinction" is directional; "it stays" is a null. Failure: 9/12 vs 12/12 read as "stays" at n=12. *Fix:* pre-register the non-inferiority margin both ways, as the plan already does in spirit for Exp B.

**6 — SHOULD-FIX. Exp A: the extinction arm does not remove exactly one thing.** Repeated safe submersions move the EC centroid; `nac.py` has no extinction producer (decay is re-learning, not a timer, Exp 58). Failure: the effect drops because the `NAMES` endpoint is now a different cluster id, read as retrieval-at-test. *Fix:* verify extinction on the staged `nac.json` (target fear below θ) **and** assert the world cluster id unchanged; drop the arm if no measured extinction procedure exists.

**7 — SHOULD-FIX. Exp A: the conditioned-word positive control changes two things (write-path allowlist + word timing).** Failure: an allowlist set globally leaks into the main arms, where the only remaining guard is "text cluster holds no fear". *Fix:* per-arm fingerprint assertion (Exp 60's frozen-fingerprint shape) stamping `text ∉ fear allowlist` on every non-control row.

**8 — SHOULD-FIX. Exp A §Rig: budget is off by ~2×.** 8 arms at 24/12 ≈ 144 agents; Exp 61 measured ≈2.5–3 min training plus ≈60–90 s per receiver, and Exp A adds phase 1 and phase 3 ⇒ ≈11 h against the stated 3–7 h. *Fix:* derive the budget from counted per-agent minutes before freezing n, or cut arms.

**9 — BLOCKER. Exp C: no primary DV, no n, and the permutation test's exchangeability is broken by the yoking.** A yoked agent's consult count is a function of its gated partner, so permuting arm labels freely is not valid. *Fix:* name the DV (R3's survival score / time-alive + reward, per SF-7), make the test a **paired sign-flip** permutation on the difference-of-differences within pair, and state the floor 2⁻ⁿ_pairs — **n ≥ 5 pairs or p < 0.05 is unreachable** (n=4 → 0.0625).

**10 — BLOCKER. Exp C: yoked-random does not hold advice *strength* constant, only count.** The foreign weight is `trust × (1 − own confidence)` and the gate fires on low own confidence; with one source, trust is a constant, so the gate and the weight are the same quantity. Failure: gated admits stale advice at near-full weight, yoked at a discount; a gated-worse result under stale is read as "the gate does not protect" when the arms differ in dose *per entry*. *Fix:* record admitted weight per entry, pre-register weight-matched analysis (or weight-match the yoked arm), report the weight distribution per arm as mediator.

**11 — SHOULD-FIX. Exp C: "damage per corrupted entry admitted" has a random, often-zero denominator that the gate itself sets;** and novelty fires roughly once per agent, so the effective unit is the consult (likely 1–3 per session), not the session. Failure: gated admits 0 corrupted entries in most sessions, the ratio is undefined, two defined rows get read as "protects per entry too". *Fix:* count consults in a counted-out pilot, power on consults, fix episode length so every session has the same denominator, and yoke on a rate over a common window (consults per 1000 ticks), not a raw count over unequal sessions.

**12 — SHOULD-FIX. Exp C: content-null removes two things if it short-circuits selection.** A consult is a local similarity scan (EC `find_similar` is a one-bucket scan) on a ~1 s tick; consulting arms pay a loop cost that `never` and a short-circuited `content-null` do not. Failure: timing DVs differ by arm for instrument reasons — R3's lesson again. *Fix:* content-null performs the full selection and discards the answer; measure per-consult on-loop wall time and freeze a refusal band.

**13 — SHOULD-FIX. Anti-vacuity is the primary contrast in both experiments.** "Remove `NAMES` → the effect collapses" *is* the ablate-early arm; "disable the gate → gated equals yoked" *is* the yoked arm; "empty the foreign layer" *is* content-null. Neither experiment has Exp 61's independent ANTI-VACUITY kit row. *Fix:* require a per-campaign kit row over staged files — the real path must make the read change, no-op variants must read 0 — absent ⇒ INCOMPLETE, failed ⇒ NULL (D62).

**14 — SHOULD-FIX. Gate freezing has no home.** No `FROZEN` location is named for any of A/B/C; Exp B's non-inferiority margin is unstated; L0's text formation threshold is fixed by a pilot and then re-used as J1's merge threshold; S2's thresholds are "hard-coded" with no drift check. *Fix:* name `expA_run.FROZEN` / `expC_run.FROZEN`, pin literal copies of inherited Exp 60/61 numbers by unit test (Exp 61's rule), and state Exp B's margin (e.g. live ≥ pre-boot − 0.15, one-sided) before data.

**15 — NIT. Exp B's n is bounded by Exp A's passing donors, and reuse is unstated** — 12 receivers off one donor is pseudo-replication. *Fix:* Exp 61's no-reuse rule scoped to the measuring arms; floors at n = 24.

**16 — NIT. L2 binding accuracy has no chance level.** With 2–3 occupied world clusters, 50 % is chance. *Fix:* report against the shuffled-control distribution plus cluster-occupancy entropy, not an absolute number. Also unfolded: lens N1 (jitter the teacher's lag), without which T1 reads back the teacher's schedule.

### Could each produce a refutation, as designed?

- **Exp A:** No. The headline discrimination is settled by the write path (1), the DV/gates/n are unnamed (4), and a null is uninterpretable without the text-identity gate (3).
- **Exp B:** Yes, in shape — the stripped and dangling arms can fail — but the non-inferiority comparison cannot, with no margin and no n.
- **Exp C:** No, as stated. The primary test has no DV, no n, an invalid exchangeability (9), and arms that differ in advice dose as well as timing (10); "gated does not beat yoked" is the only recordable refutation and it is confounded.

**Verdict: ADOPT WITH CHANGES — the instrument and the arms are further along than the statistics, and no experiment can currently be prereg'd: Exp A's central discrimination is closed by construction, Exp C has no DV, no n and a broken permutation null, and the one named look-back instrument decays in ticks where the plan specifies seconds.**
