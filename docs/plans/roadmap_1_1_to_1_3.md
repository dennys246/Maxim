# Roadmap 1.1 → 1.3 (scoped 2026-08-07 by four-lens review)

**Status:** ACTIVE. This is the single 1.1 scope authority. It supersedes the
[2026-07-23 release checklist](archive/release_1_1_checklist.md), whose Oasis/Exp 44
critical path no longer matches the evidence.
**Method:** four parallel review lenses — audio front-end, reflex wiring,
bio-fidelity, release scope — each grounded in code rather than plan docs.
**Headline finding:** *the roadmap was being drafted from the plans' ambitions
rather than from the audits' findings.* Every blocker below was already written
down somewhere in this repo by someone being careful.

---

## The three versions

| | Theme | Contains | Risk |
|---|---|---|---|
| **1.1** | **"Sensorimotor"** — *the substrate leaves the simulator, and learns to want* | Already-merged embodiment work + release correctness, contract truth, and the verification debt it incurred (all landed by 2026-08-24, published as `1.1.0rc1`) — **plus, reopened 2026-08-25: a recorded, gated result for caregiver-taught orienting (item 17) — DONE 2026-08-25, PASS — before `1.1.0` final.** The loudness bench tests (item 18) were run the same day and the item itself moved to 1.1.1 (bench done; the design is not part of the sensorimotor-learning claim). **Re-gated 2026-08-26 (item 19): the Exp 52 infants must be READ OUT on the physical robot — cross-context transfer of the learned want — with a recorded outcome before the cut. RECORDED the same day: Exp 53 APPARATUS → Exp 53b PASS (taught 1.00, controls 0.00 / 0.50). Cut unblocked.** | Medium-high: item 17 is the thesis experiment and may fail; a recorded fail still ships (as a fail). |
| **1.2** | **Oasis + Hivemind** | ~1,400 LOC of de-risked engineering with a cleared gate — **motivating case study adopted 2026-08-26: sharing the nursery-taught want ([oasis_case_study_taught_orient.md](oasis_case_study_taught_orient.md)); the claim ladder ends at cross-unit readout on a second Reachy** | Low-medium. Known shape. |
| **1.3** | **Perception fabric + reflex tier** | Cochlear front-end, vision encoder, binding, three-factor calibration, DN-canonical orienting reflex — **plus the microduck (added 2026-08-30; constraints folded 2026-08-31, [microduck_intent_layer.md](microduck_intent_layer.md)), semi-open; possibly JEPA's paired-data source** | **High — contains a pivotal may-fail experiment.** |

### Why this ordering (two corrections to an earlier draft)

An earlier sketch put the perception fabric at 1.2 and Oasis at 1.3. Both moved back:

1. **Put the predictable deliverable near, the may-fail research far.** 1.3's Stage 0c
   is described by its own plan as *"the single most informative outcome"* — a pivotal
   test whose failure means emergent identity needs a different mechanism. A re-plan
   costs least in the far slot. Oasis is de-risked engineering with a known shape.
2. **Respect the plans' own stated targets and published commitments.**
   `cross_modal_perception_fabric.md` and `three_factor_credit_assignment.md` both
   self-target **1.3**; pulling them to 1.2 was against their headers with no new
   evidence. Conversely `maxim_hivemind.md` and this README both commit Oasis to
   **1.1** — a one-minor slip (→1.2) is an honest re-scope; two minors would be a
   quiet abandonment of a published commitment.

The earlier "build the thing worth sharing before the sharing" argument still holds
directionally — Oasis's near-term payload (direction policy) is per-body calibration,
the *least* transferable thing — but it is answered by landing **artifact stamping in
1.1** (below) so the calibration leak can't bake into circulating bundles, rather than
by delaying Oasis two versions.

---

## 1.1 cut line — reconciled 2026-08-19, REOPENED 2026-08-25

> **Ship the work that is already merged, plus the correctness, contract, truth,
> and verification debt that work incurred. Zero new mechanisms.** *(2026-08-19 wording —
> items 1–16 closed under it on 2026-08-24; #537 cut the release commit.)*

> **Reopened 2026-08-25 — operator decision, recorded so it is a decision and not
> drift.** "Sensorimotor learning" as the project owner means it is *learning to want
> to orient from a primary reward* (a caregiver feeding a hungry infant), not
> orienting under a hand-declared centering drive. The Exp 45 row earns the latter on
> real hardware; the former is the caregiver experiment (Exp 48), which is PARTIAL —
> not because learning failed, but because the instrument cannot score it
> (directedness under deterministic alternation measures phase alignment; LEARNED-v2
> missed by 0.001 under a ceiling) and the reward was credited by fiat
> (`MAXIM_OPERANT_ONLY_CREDIT`), never through hunger relief. Three iterations of that is
> the divergence signal, and the divergence rule's answer is an *audit-shaped*
> experiment, not another arm. Releasing "Sensorimotor" while the experiment that
> tests the owner's version of the claim is unresolved would be the weaker claim wearing
> the stronger name. So: **the merged `main` ships now as `1.1.0rc1` (users get the
> correctness fixes; no announcement); `1.1.0` final ships when item 17 has a recorded,
> gated result — PASS or an honestly-named FAIL — and item 18's bench tests are done.** *(Both happened 2026-08-25; item 18's design then moved to 1.1.1 — see its row.)*
> Two mechanism-class additions are admitted by this decision, each with its
> front-gate answer in its row. Oasis + Hivemind stay 1.2.
> **RESOLVED 2026-08-26:** items 17 and 19 recorded (Exp 52 PASS; Exp 53 APPARATUS → 53b PASS), item 18's
> bench done and its design moved to 1.1.1; `1.1.0` published 2026-08-26 (tag `v1.1.0`, GitHub Release).

| # | Item | State / release contract |
|---|---|---|
| 1 | `_cosine` dimension guards + frozen-modality parity | **DONE** (#467). Same-dimension geometry compatibility remains a pre-1.2 gate (D4). |
| 2 | CHANGELOG reconstruction + historical tags for 1.0.0→1.0.6 | **DONE**. Historical interval reconstructed 2026-08-07; the post-1.0.6 record is now the `[1.1.0]` section (with an attribution note for bullets that shipped inside 1.0.7–1.0.9); the release transaction is gate-order step 5. |
| 3 | n_ctx clamp and headroom truth | **DONE**. Re-attest through the Big-Model heartbeat. |
| 4 | Persona hard-remove | **DONE**. Keep compatibility/error behavior covered. |
| 5 | Graduation heartbeat walk | **DONE (2026-08-24)** — Sim-Short complete (records committed #536); Big-Model chapter RAN 2026-08-21 (Qwen32B re-fire → instrument finding L8, row stays PARTIAL); H1 8-rep delivered-shift block DONE (#535); pytest-only rows re-attested at the RC commit `88739318`. |
| 6 | H1 healthy-hardware re-characterization | **DONE** — base session 2026-08-08 + the `_big` delivered-shift block 2026-08-24 (n=8/side in one session, 0.943 of command both sides; `_big` YAML magnitudes stay frozen; the right-side Δaz asymmetry is a follow-up still to be pre-registered; D30/D31 filed). |
| 7 | Artifact stamping | **DONE**. |
| 8 | Orient-vocabulary audit + workspace-limit bypass safety | **DONE** (#472); retain hardware-safety guards in the final RC suite. |
| 9 | Documentation truth pass | **SUBSTANTIALLY DONE, CONTINUOUS THROUGH RC**. Reconcile plans, architecture, decisions, API count/dependencies, and release surfaces. D21's EC performance claim was corrected on 2026-08-19. |
| 10 | Atomic NAc + EC invalidation | **DONE (2026-08-23)** (D2). `MEMORY_PATHS` gained the missing `ec` key and `MEMORY_PAIRS` declares `{nac, ec}` inseparable, so clearing either half pulls in the other and `all` no longer leaves a stale EC behind. |
| 11 | Annotation S4 non-stationarity analysis | **DONE (2026-08-24)** — result recorded in [44b_pilot.md §S4](../experiments/44b_pilot.md) with the analyzer JSON under `docs/experiments/data/44b_s4_nonstationarity/` and the pilot captures under `docs/experiments/data/44b_pilot/`: the band treatment is NON-STATIONARY within a run (4 of 5 tracked tools fall strongly→mildly rewarding by the second half in every arm; the target `warm_self` holds strong), flip rate does not track band tier monotonically, and the identical-prompt determinism probe is 0.000 (11 pairs). Consequence: the 44b freeze must stratify early/late or hold the tier. *(Original note — inputs located 2026-08-24:* the Exp 44b pilot captures are NOT on the operator Mac — they live on big-mac-mini (the pilot machine; `~/exp44b/pilot/arms/<arm>/seed<n>/capture.jsonl` + `requery/`). Run `scripts/exp44/analyze_nonstationarity.py --json` there per capture and commit the JSON under `docs/experiments/data/44b_s4_nonstationarity/` with a results section in [44b_pilot.md](../experiments/44b_pilot.md); that commit closes the gate.)* |
| 12 | Planning-turn liveness + truthful progress state | **DONE** (D13/D14/D22). Bounded recovery or typed terminal abort, observationally true progress display, and non-zero process/harness propagation for unusable results. |
| 13 | Stable Python API contract repair | **1.1 GATE DONE (2026-08-23)**; 1.1.x tail listed below (D15–D18). D15 `goal`/`robot` and D16 `run()` cleanup are DONE in v1.0.9; **D17 complete load semantics and D18 tool-registration lifetime are DONE (2026-08-23)** — `load.agent()` restores ATL before returning and raises `MemoryCorruptionError` naming every unreadable file (explicit `on_corrupt="fresh"` opt-in), and tool registration is persistent with `unregister_tool`/`clear_registered_tools`. `home_dir` completeness plus `imagine()`/`campaign()` cleanup stay 1.1.x. |
| 14 | Hermetic required fast suite | **DONE in v1.0.9** (D20). Unique temporary user/config/cache roots, offline model-hub defaults, explicit pretrained-asset opt-in, per-test path-cache reset, and the wider CI gate are executable. |
| 15 | Architecture-audit enforcement | **DONE (2026-08-24)** (D19). The 33 findings are classified (10 typing-only / 16 function-local lazy imports, none a cycle-break / 7 module-level) in a reviewed baseline (`src/maxim/utils/architecture_baseline.json`, keyed by file + module + scope + accepted symbols); the fast suite fails on additions (incl. a widened symbol list), stale entries, and unreviewed entries. Zero debt is 1.1.x item 11, not a release blocker. |
| 16 | Release, website, and agent-guidance truth | **DONE 2026-08-26 for the cut** (38-route audit + maxim-web #7 live 2026-08-25/26, hardware card #8 + release-notes/video pointers #9, post-publish install check passed; the human browser/accessibility pass is the 1.1.x remainder). Original text: (1.0.9 was published to PyPI on 2026-08-23 at tag `v1.0.9`/`5cb4413b` after the 2026-08-20 maxim-web corrections went live; D24's remaining items now gate the 1.1 cut, not 1.0.9.) The 38-route live content audit is DONE and its cross-repo handoff is recorded; maxim-web corrections, path-preserving `docs.pymaxim.bio` redirects, visual/accessibility verification, exact-wheel command checks, legacy deep-link migration, and post-upload PyPI link verification remain. Repo metadata now points directly to the canonical getting-started route. |
| 17 | **Nurture — hunger-relief-taught orienting (Exp 52)** | **DONE 2026-08-25 — both phases PASS; shipped in `1.1.0`.** (Added as the 1.1 FINAL GATE and pre-registered 2026-08-25:) [exp52_nurture_preregistration.md](../experiments/protocols/exp52_nurture_preregistration.md). The audit-shaped successor to Exp 48. The mother's feed already writes a real hunger delta; what changes is the credit's VALUE — from a constant `feed_reward` to the sign of the relief the infant experienced (`drive_comfort_progress` on the drives the feed touched; zero relief → no credit) — delivered through the existing one-turn pending-operant trace. **Front-gate:** rides existing drive specs, value-progress credit and the operant trace; the only new contract is relief-sourced operant credit (the temporal-credit distributor is NOT needed for a one-turn contingency — an earlier wording of this row said otherwise). Arms taught / **satiated** (fed contingently, never hungry — the arm that separates "learns to want" from "learns to be fed") / yoked / no_feed. **Order:** Phase A scripted (`orient_substrate/9`, seconds; Exp 46 gates + HUNGER-NECESSARY) gates Phase B embodied (`cradle_mother` under gate v3: v2 constants + HUNGER-NECESSARY + an explicit L2 seed-spread apparatus gate, shuffled stimulus order, n = 12/arm, exposure-matched 48 turns). **Stop rules pre-registered:** A fails → B does not run; B runs once; a second divergence ships 1.1.0 with the result named. **DONE 2026-08-25 — both phases PASS.** Phase A @ `e367f526`: taught 0.89 / satiated 0.50 / yoked 0.50 / no_feed 0.50 (satiated ≡ no_feed to the digit). Phase B @ `60195a29` on the mini, apparatus v3: taught late 0.878 / satiated 0.441 (fed 35%, credited 0%) / no_feed 0.413 — GRADUATE under gate v3, L2 apparatus clean, S3 OK. Write-up [52_nurture.md](../experiments/52_nurture.md); new Earned row; Exp 48 row superseded. `1.1.0` final now waits on item 18 only. |
| 19 | **Cross-context readout — the nursery-taught want on the robot (Exp 53)** | **DONE 2026-08-26 — Exp 53 APPARATUS (direction 36/36, δ overshoot on the −0.2 target) → Exp 53b PASS with the one declared change (δ = the robot's own 0.30 rad step): taught 1.00/1.00/1.00, satiated 0.00, no_feed 0.50 — cross-context transfer EARNED; new ledger row; write-up [53_cross_context_readout.md](../experiments/53_cross_context_readout.md). `1.1.0` publication unblocked.** Original row (added 2026-08-26; PRE-REGISTERED 2026-08-26): [exp53_cross_context_readout_preregistration.md](../experiments/protocols/exp53_cross_context_readout_preregistration.md). Owner intent: the claim is learning that carries across sessions AND contexts without fine-tuning; cross-session is earned (Exp 42/45), cross-context never shown — a sim-only "Sensorimotor" is the weaker claim wearing the stronger name. **Design:** the Exp 52 Phase B infants' persisted `aut_nac.json` + `aut_ec.json` (taught 42/43/44; satiated + no_feed 42/43/44 as zero-bias controls; taught 48 exploratory) loaded UNCHANGED (`apply_decay=False`, no credit, SHA-256 before/after) into the production substrate-primary path on the live Reachy: `bodies/infant_operant` (tool names match the learned keys; no innate drive), `DoAFeed` on the same `azimuth` sensor, `_encode_current_clusters` → `recommend_action(current_clusters=)` → `propose_via_substrate`, production `ReachyOrientMotorBackend` with an explicit δ = 0.55 rad map (the infant body declares no `head_yaw` — S6). Exp 45's `--perturb` trial generation; **targets az {−0.3, −0.2, +0.5, +0.6} (amendment 1, pre-data, from the harness dry run: the nursery's three audio clusters partition the axis FAR-LEFT ≤ −0.5 / CENTRE −0.4…+0.3 / RIGHT ≥ +0.4, with `turn_right +0.90` on RIGHT and `turn_left +0.65` on CENTRE — the original ±0.5/±0.6 would have probed the weak far-left bin) + exploratory −0.6 / +0.2 placements (recorded, not gated; +0.2 is predicted wrong-way — the representation's stated limit).** **Gate I (instrument, no motion):** live percepts pattern-complete into the nursery's `audio` clusters and the frozen-policy probe is correct with `|learned_margin| > 0.11` for ≥ 2/3 taught seeds; controls show no learned preference (they may act on persisted causal credit, side-blind) — **fail → instrument STOP, `1.1.0` ships as-is, transfer → 1.1.x.** **Gate T (transfer, 12 gated trials/agent, primary condition explore 0.0 — frozen policy with motion; explore 1.5 secondary, reported):** taught delivered directedness ≥ 0.70 and ≥ 0.20 above each control, sign-rule agreement ≥ 0.80. Runs once; a FAIL ships `1.1.0` with the fail named. Front-gate: rides existing infrastructure end to end — zero new mechanisms; the only declared apparatus is the δ map. Considered/rejected: re-running Exp 52 on a Reachy-shaped body (the transfer claim is strongest when the files are byte-identical). |
| 18 | **Loudness / onset salience** | **BENCH DONE 2026-08-25; DESIGN DEFERRED → 1.1.1 (decision 2026-08-25).** The bench tests ran in one 75 s trace on the live rig — [h2_loudness_bench_2026-08-25.md](../experiments/h2_loudness_bench_2026-08-25.md): the XVF3800 already computes a level (`AEC_SPENERGY_VALUES`, per-beam speech energy, pre-AGC) and the daemon we run serves it over `GET /api/audio/config/parameter/{name}` (reachy_mini ≥ 1.8.0), so neither of the two paths in §Loudness (Pollen's code / onboard PCM) is needed. (a) level IS available — two signals: `PP_AGCGAIN` readback = graded inverse loudness envelope (42–46 quiet → 8.3 loud speech, ~3 s attack / ~15 s release); `AEC_SPENERGY_VALUES[3]` = spiky, VAD-gated speech magnitude (3–4× quiet→loud on a 2 s window-max; 0 for loud non-speech). (b) the AGC does NOT flatten the register (energy peaks exactly where gain bottoms) — it WOULD flatten PCM ~5×, which is why the PCM path was the wrong path. **Why deferred:** loudness never entered the sensorimotor-learning claim that reopened 1.1 (item 17 tests it; Exp 52 PASSED without it); shipping a salience design under the release headline would be scope, not evidence. **1.1.1 plan:** salience in the DoA feed = f(onset, level) from those two REST reads, **riding the existing `percept.salience` field**, no PCM, no `media_backend: default`, no new mechanism; cost inside a live WebRTC-streaming orient session still to be measured. The *forced* startle look (action below deliberation) stays the 1.3 reflex tier. |

The remaining scope is release closure, not a new feature phase. Estimates belong on
the implementation PRs after each item's failing contract test exists; this roadmap
does not convert uncertain debugging into calendar promises.

**Operator-ratified 2026-08-19** (the 9→16 expansion), with this severity split
from the claims-verification round — blocking vs 1.1.x *within* the gated items:

- **Blocking for the 1.1 cut:** ~~D15's `goal`/`robot` and D16 for `run()`~~
  **DONE in v1.0.9**; ~~D17 (partial restore + corrupt-state swallows on the
  stable load path), D18 (documented contract silently one-shot)~~ **DONE
  2026-08-23**;
  ~~item 15's baseline+CI gate (cheap)~~ **DONE 2026-08-24**, and item 16 — which now explicitly
  includes correcting the **false
  "shipped to PyPI" claims**: ~~PyPI's latest release is 1.0.0~~ (true as of
  2026-08-19; 1.0.9 was published 2026-08-23); v1.0.1–v1.0.6 are
  git-tag-only and both CLAUDE.md and the CHANGELOG said otherwise (corrected).
- **1.1.x hardening:** D16 for `imagine()`/`campaign()`, D15's `home_dir` (document
  the partial behavior if not surgical), and D19 zero-debt burn-down.
- **Also folded into item 12:** the #519 abort-clock fix shipped without a guard
  test (scorecard finding) — the D13/D14 fix PR adds it.
- **Also folded into gate-order step 4 (evidence closure):** the S4 backfill gap —
  zero committed raw data exists for Exp 09/10/43/44/46/47/49 including the Tier-1
  EARNED Exp 10 row; each Earned row either gets its data committed (re-run under
  S4 where originals are lost — Exp 10's re-run already happened 2026-08-18/19,
  commit its heartbeat records) or carries an explicit data-lost annotation.

### Release-gate order

1. **DONE — D13/D14/D22:** planning liveness, observationally true display, and
   trustworthy terminal-status propagation now unblock long heartbeat runs.
2. **DONE — Stable API + hermetic tests:** D15, `run()`'s D16 slice, D20, and
   now D17/D18 are all landed with facade, registry-lifecycle, controller,
   temporary-state, offline-model, load-contract, tool-persistence, and
   wider-CI guards.
3. **DONE — Persistence/architecture correctness:** ~~D2 atomic invalidation~~ **DONE
   2026-08-23**; ~~D19's accepted-debt baseline/regression gate~~ **DONE 2026-08-24**.
4. **DONE 2026-08-24 — Evidence closure:** S4 recorded (item 11, #536), Big-Model chapter ran (2026-08-21 → L8), hardware chapter done (#535), cheap rows re-attested at the RC commit `88739318` (37/37, 41/41).
5. **Release transaction — split 2026-08-25:**
   - **5a — DONE 2026-08-25: `1.1.0rc1` published** (tag `v1.1.0rc1` at `eae6559c`, #540). The recipe, kept for 5b:
     bump both version files + the CHANGELOG header to `1.1.0rc1` in one commit, vendor
     the Console UI, build/`twine check`, TestPyPI → PyPI, tag `v1.1.0rc1` on the
     published commit, update the three version sync lines (they NAME the version and
     link PyPI; the "PyPI serves" prose was retired 2026-08-29 by item 16.1 and is now
     rejected by `scripts/lint_version_sync.py`). No announcement.
   - **5b — DONE 2026-08-26: `1.1.0` PUBLISHED** (PyPI latest 1.1.0; tag `v1.1.0` at `df881b87`; GitHub Release with artifacts; site flipped; release notes [release_1_1_0.md](../announcements/release_1_1_0.md)). The recipe as it ran: the bump (#547) and the website audit (#548, maxim-web #7) are DONE 2026-08-26; **as it ran:** the site carried the hardware result (maxim-web #8) before the cut; wheel + sdist built from `main`, `twine check`, TestPyPI → PyPI, `v1.1.0` tagged on the published commit, GitHub Release with the exact artifacts and `docs/announcements/release_1_1_0.md` as notes; sync lines flipped in #554; announcement + video after.

### Agent-guidance single-source decision (1.1 docs gate) — RATIFIED 2026-08-19, inverse direction

Two independent root instruction corpora are not sustainable — but the review round
found the original proposal's mechanism backwards. The diet architecture (#509,
operator-reviewed) already delivers the single source: **`CLAUDE.md` stays the
canonical core** — 9.1K tokens, CI-linted (`scripts/lint_claude_md_invariants.py`:
invariant format, guard citations, token ceiling, link existence), auto-loaded by
the tooling doing most work here — with subsystem knowledge in `docs/agents/`
briefs and incident history in `docs/lessons/`.

**`AGENTS.md` is the pointer-only provider-neutral ADAPTER** (tightened 2026-08-19):
it tells auto-loading tools to read `CLAUDE.md` in full and contains no copied routing
table, checks, or hard-rule summaries. `scripts/lint_claude_md_invariants.py` enforces
the adapter byte-for-byte, so rule accumulation fails CI. The
previous AGENTS.md had silently diverged since 2026-06-06 (wrong Python floor,
dead pointers) — which is itself the evidence for why an adapter must carry no
substantive rules of its own. Its one unique live artifact (the cross-system
naming-conventions table) was rehomed to docs/agents/bio-memory.md §4b.

Rationale for the inversion: demoting CLAUDE.md to an adapter would have (a) made
the invariant lint pass vacuously with no replacement drafted, (b) taxed every
Claude session with an indirection read in a weaker instruction position, and
(c) obsoleted an operator-reviewed artifact six days after its review — all to buy
a filename. The inverse buys the same single-source property (non-Claude agents now
load a truthful adapter) at zero migration risk. If a canonical-filename migration
is ever wanted, the mechanism is content-identity (generated copy + CI byte-check),
never indirection, and the lint retarget ships in the same commit.

### H1 buys three things at once — now four

The DoA re-sweep resolves **Exp 45's staleness**, *is* **1.3's Stage 0a**, and provides
**motor-binding Phase 3's gain calibration**. Buy once. Fold the contingent H2 branch
(magnitude re-probe, only if H1 moves the ≈0.33 decision boundary) into the
pre-registration **before** H1 runs, so its outcome is decided in advance.

### Hardware note (2026-08-07): motors 2 and 3 were broken for the ENTIRE 1.0+ era

**Operator report:** motors in Stewart positions **2 and 3** were broken and have now
been replaced and reflashed from the motor-1 config. All six legs confirmed healthy;
the platform moves cleanly. **The breakage spans essentially all of 1.0+**, repaired
~2026-08-05.

**Root cause, and it is ours:** an earlier Maxim iteration commanded a pose *beyond its
physical capability*; the motors glitched, the head snapped violently to the opposite
extreme, and the robot rotated itself off the table. This is a **workspace-limit
enforcement failure**, and it connects directly to the orient-vocabulary audit (item 8):
**two paths bypass `ReachyMiniController.goto_target` entirely** — `MoveTool` gaze
without a `robot_id`, and `turn_around`, which hand-rolls its own centering. A path that
bypasses the controller plausibly bypasses workspace clamping with it. That elevates
item 8 from a correctness fix to a **safety** fix, and it should be treated as the
highest-priority item in the 1.1 cut line after PR #467.

**Data-quality consequence — the important part.** Every live-hardware measurement in
the 1.0+ era was taken on a degraded platform, including:

- the 2026-07-16 "TRUE characterization" (0.57 az/rad, R²=0.9982) — **also degraded**,
  contrary to the earlier hypothesis that it was the healthy baseline
- the 2026-08-05 contested sweep (~0.19 az/rad)
- Exp 45 / 45b / 45c / 45d / 45e (orient direction + magnitude)
- Exp 46 / 48 (operant orienting)
- every live orient session and smoke test

**This yields a better hypothesis for the contested curve than either previous one:
progressive mechanical degradation.** If motors 2 and 3 were failing *gradually*, then
0.57 (July) and 0.19 (August) are both real measurements of a platform in two different
states of decline — not one good run and one instrument artifact. That predicts
**healthy hardware should now measure ≥0.57, plausibly nearer the geometric 0.637.**
H1 tests it directly.

**What survives and what does not.** Direction findings are likely robust — if the body
turns at all, which way it turned is preserved. **Magnitude findings are not**: delivered
shift is exactly what a degraded platform corrupts, and the magnitude line already
rested on n=1 sessions. Treat every magnitude claim (the ≈0.33 decision boundary, Exp
45b/45c/45d/45e magnitude arms) as **provisional pending re-measurement**, and say so in
the graduation walk rather than assuming clean.

**H1 is therefore the first honest hardware measurement in the project's 1.0+ history**,
and its value is much higher than originally scoped. Run it early.

**Operational:** stock multiple spares — this incident needed more than the one held.
And two failures from one root cause argues for fixing the *cause* (workspace-limit
bypass) rather than only the symptom.

---

## Pulled into 1.1 (adopted 2026-08-13 from the roadmap synthesis)

Two additions to the cut line, both bug-class rather than mechanism-class so the
"zero new mechanisms" discipline holds:

1. **`maxim memory invalidate ec` (bugs ledger D2).** `MEMORY_PATHS` has no `ec` key, so
   an operator cannot invalidate a stale EC substrate — and clearing `nac` alone violates
   the NAc/EC persist-as-a-pair invariant (biases dangle on nodes a fresh EC never
   re-allocates). Invalidation must be NAc+EC in lockstep. Also a place-code
   default-ON gate, so doing it in 1.1 unblocks 1.1.x.
2. **Annotation S4 — non-stationarity analysis** (`annotation_context_and_provenance.md`).
   Pure offline analysis on captures already on disk (zero sim cost); answers Exp 44b
   pilot finding F6 (0.997 → 0.059 within-run signal decay despite τ=1000) and feeds the
   pre-registration freeze. Do it whenever a session has an hour.

## Scorecard → roadmap reconciliation (2026-08-27)

The two 2026-08-19 scorecards ([claude](../limits/score_cards/2026-08-19-claude.md) /
[codex](../limits/score_cards/2026-08-19-codex.md)) were re-read against `main` after the
1.1.0 cut. **Both cards say "re-score at each release cut"; 1.1.0 was cut 2026-08-26 and
no re-score exists — one is owed, at the 1.1.0 commit, before any grade is cited again.**
It should be a real move: most of the Runtime-correctness, Test/CI-truthfulness and
Release-governance upgrade conditions are now *enforced* (the cards' own criterion for a
grade to move) — D13/D14, D15–D21, D25/D26 fixed with guards; the required fast suite is
hermetic (D20) and **CI runs the full documented command, not `tests/unit/`**
([test.yml](../../.github/workflows/test.yml) "Run required fast suite" + MemoryHub gate +
the nightly `requires_model_cache` job); the architecture audit is baselined (D19); PyPI
serves 1.1.0 and every sync line says so; `[Unreleased]` is current; every `fix(` commit
since the cards touches `tests/` (9/9).

What the cards flagged that was on **no** list is folded in below as **item 16** (cheap
enforcement, one PR) and **gate 8** (evidence/ledger coherence before anything is shared),
plus three sharpenings (items 1, 7, 13). Card findings deliberately NOT adopted: reducing
the number of ledgers (the ledgers are what the cards graded A — automate their drift
checks instead), repo-wide mypy, a repo-wide swallow purge (D11 is ACCEPTED with a reasoned
scope), and the ~60 %-mnemonic bio-naming (answered by the docstring taxonomy, #492).

## 1.1.x follow-through (post-cut, pre-1.2 — items 1–10 adopted 2026-08-13; items 11–14 added 2026-08-19 and ratified with the cut-line reconciliation; item 15 added 2026-08-26; item 16 added 2026-08-27 from the scorecard reconciliation)

The 1.1 cut line stays closed; these are the next minor-stream items in rough order.
Each POINTS at its owning plan — stages live there, not here:

1. **Exp 44b pre-registration freeze → confirmatory campaign**
   (`protocols/exp44b_preregistration.md`). The core research claim's power run.
   Freeze blockers: F1 name-copying control (→ Exp 51), determinism measurement
   (#496 built it — run it), invalid-action rule (F7), entangled-axes framing fix (F2).
   Deliberately NOT a 1.1 gate — holding the release for a 10-seed/arm campaign was
   considered and declined; it opens 1.1.x instead.
   **Prerequisite added 2026-08-27 — the [L8](../limits/README.md) record-stamping fix.**
   Run records stamp only the REQUESTED profile name; the 2026-08-22 re-fire showed the
   serving environment moves an LLM-AUT baseline more than the mechanism does and nothing
   in the record can detect it. Before 44b (or any Exp 37-class fire) runs, every run
   record must carry `resolved_model`, `endpoint`, `n_ctx`, quantization and server build,
   and the gate must score `B − C` by position, never against a remembered number. This is
   the Claude card's only upgrade path for Ambition (one at-power confirmatory result) and
   without the stamping a 44b fire lands exactly where the Qwen32B heartbeat did. Item 8
   (cross-process `served_n_ctx`) is the same stamping surface — build them together.
2. **Decision provenance Stages 3+4** (`decision_provenance.md`): make the provenance
   fields queryable + wire them into the S2 apparatus canaries. Stage 4 overlaps the
   canary work — build together.
3. **Annotation S2 (context-aware view) + Exp 51** (`annotation_context_and_provenance.md`):
   the decisive name-copying-vs-learned-content experiment. Sequencing per the plan:
   S4 → A3 channel-obedience probe → S2 → Exp 51 → S3.
4. **Exp 50 — re-adaptation after plant change**
   (`docs/experiments/50_readaptation_after_plant_change.md`). Pre-registered, unblocked
   by H1, needs robot time (3 arms, n ≥ 3 sessions/arm). A PASS adds a graduation row.
5. **Place-code default-ON** (`modality_resolution_and_alignment.md` §7). Gates: D2
   (pulled into 1.1 above), the hivemind merge dim-guard for same-dimension geometry
   changes (D4), `min_confidence` recalibration against the ~0.11 visibility floor
   (instrument = #504's `explore_decisive`/`learned_margin`), then the Exp 48 + Exp 49
   H3/arm-C re-runs the flag's own note requires.
6. **Fail-loud Stages 2–3** (`measurement_path_fail_loud.md`): read the Stage-1
   instrumentation's logs (Stage 2, nearly free), then narrow/propagate per the policy
   (Stage 3). Stage 4's CI lock already shipped 2026-08-13.
7. **God-function decomposition** (`god_function_decomposition.md`) — its own plan
   sequences it after fail-loud Stages 1–2; start once item 6's Stage 2 lands.
   **Sharpened 2026-08-27:** both scorecards' Maintainability finding, and the one axis
   they disagree on hardest (C+ vs D+). The three functions **grew** while the plan waited
   — `run_agentic_loop` 3,331 → 3,546, `start_simulation_mode` 3,226 → 3,342,
   `_main_impl` 1,737 → 1,752 (2026-08-19 → 2026-08-27) — because the D13 fix went
   *inline* into the exact planning-submit/await seam the Claude card named as the first
   extraction. Two consequences: (a) a **no-growth guard ships NOW, ahead of the
   decomposition** (item 16.4 — a function-length baseline in the fast suite, the D19
   pattern: additions fail, shrinkage tightens the baseline); (b) the planning-liveness
   block is now the best-tested code in the function (66 tests in
   `test_planning_liveness.py`), which makes it the *safest* first cut, not the scariest —
   extract it first.
8. **n_ctx leg 3, cross-process** — a `served_n_ctx` handshake readable across processes
   (today `maxim config` alignment is the documented mechanism; acceptable carry, close
   it when touching the lane code anyway). **2026-08-27: folded into item 1's L8
   stamping prerequisite — same record surface, one PR.**
9. **`agents/llm_agent.py` router migration** — retires the CI backend-import
   grandfather clause and its positive control.
10. **Memory-consolidation decision** — `deferred/memory_consolidation_practice.md`'s own
    rule: "if 1.1 ships without touching consolidation, downgrade to archive at the next
    sweep." Ship the decision either way at the 1.1 cut.
11. **Architecture-debt burn-down** (D19) — move shared graph/event contracts out of
    dependency-inverted packages and drive the accepted baseline toward zero. The 1.1
    gate prevents new violations; 1.1.x removes old ones.
12. **Control-loop lifecycle and I/O hardening** — after fail-loud Stage 2, extract the
    planning-failure/lifecycle seams from `start_simulation_mode` and
    `run_agentic_loop`; move durable writes off the 30 Hz thread with a bounded queue,
    and protect final persistence/session shutdown with structural cleanup.
13. **DONE 2026-08-29 (#572) — Dormant-path decisions for D6/D9.** `Dormant since` docstring
    markers on both ends of the D6 percept-binding path (`episode.py::apply_hebbian_on_close`,
    `bio_integration.py::record_substrate_nodes`) and on the D9 drive emitter, each naming the
    1.3 fabric as the resurrection trigger; ledger rows re-dispositioned `DORMANT`. The D9
    emitter's malformed `TemporalEvent` construction — hidden for months behind
    `except Exception: log.debug` — is repaired and now reports at WARNING, but the structural
    fix (a typed distributor entry point) stays with `deferred/scn_event_producer_gap.md`.
    Original text:** — either wire and behaviorally graduate
    Hebbian multi-node binding / temporal-event producers, or mark the unused contract
    dormant and stop implying it learns in production. **Sharpened 2026-08-27: the 1.1.x
    decision is DORMANCY, not building.** D6 (Hebbian binding inert on the percept path)
    is a hard dependency of the 1.3 fabric's orient-windowed binding — wiring it here
    would be 1.3 scope creep without 1.3's experiment; D9's producers are already
    deferred (`deferred/scn_event_producer_gap.md`). Mark both `Dormant since <date>`
    with the 1.3 stage as the resurrection trigger, and pull any docstring that implies
    they learn in production.
14. **Published-support truth** — **PARTIAL in v1.0.9:** lightweight CI
    install/import/CLI lanes now cover Python 3.10, 3.11, 3.13, and 3.14 while
    the full suite covers 3.12; contributor guidance now matches the seven core
    dependencies and 18 API verbs declared by the package (`recall` landed post-1.0.0 and is still undocumented — see below). Keep dependency and
    verb-count drift checks executable as those surfaces change.
15. **Reachy-native nursery body — Exp 54 [Phase A GRADUATE 2026-08-27](../experiments/54_nurture_reachy_body.md) (harness #561; taught 0.858 / satiated 0.472 / no_feed 0.514 on `bodies/reachy_mini_infant`; Phase B/C on the robot pending — item DONE only when all three phases are recorded); pre-registered 2026-08-26 as [Exp 54](../experiments/protocols/exp54_nurture_reachy_body_preregistration.md)** (added 2026-08-26, from Exp 53b + the Oasis case
    study [oasis_case_study_taught_orient.md](oasis_case_study_taught_orient.md)): an
    `infant_operant`-shaped body whose orient affordances carry the Reachy's `head_yaw`
    self-effects and whose tool names are the robot's own, then a re-run of Exp 52 on
    it. Removes the S6 δ map from the readout path and makes the taught files usable on a
    user's robot without a key remap — the prerequisite for sharing them.
16. **Scorecard mechanization** (added 2026-08-27; every piece rides an existing lint or
    CI step — no new mechanism. Shipped 2026-08-29 as the 1.1.1 "enforcement" release in
    four PRs: **A** provenance 16.7–16.9 (#569), **B** ratchets 16.2–16.4 (#570), **C**
    release truth 16.1/16.5/16.6 (#571), **D** the card's Runtime C+ items). Both cards say a grade moves
    only when the normal workflow *enforces* the improvement; these are the enforcement
    gaps the cards named that no item owned. Front-gate: `lint_no_silent_swallows.py`'s
    diff-scoped / grandfathered-count shape and the D19 baseline pattern already exist —
    reuse them.
    1. **DONE 2026-08-29 (1.1.1 Cluster C: policy written in CLAUDE.md §Versioning — `main` is ahead of PyPI, the bump is the release transaction; `scripts/lint_version_sync.py` in CI; the three version lines link PyPI instead of describing it) — Version-bump policy: decide, write it down, enforce it.** CLAUDE.md §Versioning
       says "bump on any change affecting runtime behavior"; practice is "bump at the
       release cut" (≈60 unbumped post-tag commits at scoring time; #561/#562 touched
       `src/` post-1.1.0 unbumped). Written intent vs routine practice diverging is the
       Codex card's *definition* of a D. Pick one — the honest option is "main is always
       ahead of PyPI; the bump happens in the release transaction" — and rewrite the
       CLAUDE.md paragraph to match. Then extend the existing "Version sync" CI step: if
       a diff changes the `pyproject.toml` version, `CHANGELOG.md` must contain
       `## [<that version>]` (`scripts/audit_release_tags.py` covers tag↔changelog, not
       bump↔changelog — the Claude card's stated path to B for Release governance).
    2. **DONE 2026-08-29 (#570: `scripts/lint_fix_touches_tests.py` — the PR TITLE, i.e. the
       subject that squash-merges onto `main` and the one the card counts, against the aggregate
       diff, PLUS every branch commit; CI + fixture tests) — Diff-scoped fix→tests lint:** a commit whose subject starts `fix(` and touches
       `src/` must touch `tests/`. #519 is the incident (a behavioral fix to an abort path
       with zero test changes); every fix since the cards has complied (9/9), so the lint
       ratifies practice rather than changing it. The card's "90 days clean" then
       becomes measurable at the next re-score instead of asserted.
    3. **DONE 2026-08-29 (#570: `scripts/lint_atomic_io_ratchet.py` — per-file AST count of
       hand-rolled atomic renames printed in CI + a no-growth ratchet; CLAUDE.md cites that
       output) — `atomic_io` violations get a guard.** The KNOWN-GAP note *had* admitted "17
       hand-rolled `os.replace` sites, detection-only, needs its own task" — a stale quantified
       confession, the Claude card's specific Documentation-honesty deduction. The 17 came from a
       text grep that counted comments and docstrings and saw only ONE spelling; the AST count
       across all four (`os.replace`/`os.rename`, `Path.replace`/`Path.rename`) is **12 call sites
       in 12 files** as of 2026-08-29. The lint makes the count self-updating; the burn-down is
       still its own task, and needs an `atomic_write_bytes` before the two BYTES writers
       (`hivemind/bundle.py` zip, `models/download.py` GGUF) can move at all.
    4. **DONE 2026-08-29 (#570: `src/maxim/utils/function_length_baseline.json` +
       `tests/unit/test_function_length_baseline.py`, fast suite; growth fails, shrinkage must
       tighten in the same commit). This is HALF the card's Maintainability C+ condition — the
       other half is one merged extraction, which is item 7 / 1.1.2.** Function-length baseline for the three god functions (item 7's no-growth
       guard): `src/maxim/utils/architecture_baseline.json`'s pattern — a reviewed
       ceiling per function, additions fail the fast suite, shrinkage must tighten the
       ceiling in the same commit.
    10. **Post-tag `src/` commits declare themselves** (added 2026-08-29 from the 1.1.1
        Cluster C review). The version policy (16.1) says `main` is ahead of PyPI and
        `CHANGELOG.md` accumulates under `## [Unreleased]` — the first half is enforced by
        `scripts/lint_version_sync.py`, the second is convention: #561/#562 changed `src/`
        post-1.1.0 with no `[Unreleased]` line. The Claude card's Release-governance
        condition (4) is the enforcement: a diff-scoped check that a `src/`-touching commit
        after the last tag either bumps or adds an `[Unreleased]` entry. Rides
        `scripts/_lint_git.py` (#570). **A policy half-enforced is the divergence 16.1
        exists to end.**
    5. **DONE 2026-08-29 (1.1.1 Cluster C) — Delete `docs/CHANGELOG.md`** — the dead duplicate frozen at 0.3.0 that both
       cards named and that survived the 1.1.0 cut.
    6. **DONE 2026-08-29 (1.1.1 Cluster C: rows 1/2/6/7 re-measured — row 7's CI claim had been false since #527, row 6's diet had shipped, row 2's Stage-4 lock had shipped; every row now carries the date its numbers were measured) — Truth pass on [external_critique_response.md](external_critique_response.md)**
       — row 7 still says "CI still runs `pytest tests/unit/` only" (false since D20);
       rows 2 and 6 are stale (the plans README already flags them). The living-scorecard
       label is only true if the rows are.
    7. **Harnesses refuse on a dirty tree** (added 2026-08-27 from the 1.1.0 re-score —
       [lesson](../lessons/experiment-prereg-precedes-data.md)). `scripts/_provenance.py`
       already computes `working_tree_dirty_src_scripts`; a harness writing under
       `docs/experiments/data/` exits 3 when it is true unless `--allow-dirty` is passed,
       and then stamps `allow_dirty: true` into every record. Widen
       `lint_harness_provenance.py` to `scripts/orient_*/` (in-process harnesses — the
       door Exp 53/53b walked through). Guard: unit test on the refusal.
    8. **Pre-registration precedes data, checked by CI** (added 2026-08-27). For every
       `docs/experiments/data/<N>*.jsonl`, the prereg named in `docs/experiments/<N>_*.md`
       must have a `git log --diff-filter=A` timestamp on `main` earlier than the file's
       first `ts`; amendments likewise for the data they govern. Exp 52 passes today;
       53b would have failed at 18:27 on release day. **If only one of 7–9 ships, ship
       this one.**
    9. **EARNED ledger rows need a data citation** (added 2026-08-27). Add
       `behavioral_graduation_candidates.md` to `lint_claude_md_invariants.py`'s doc set:
       any `EARNED` row carries a `Regression guard:` link to `docs/experiments/data/` or
       a dated data-lost annotation. L185 (EC pattern completion) and L186 (SEM pain →
       NAc) fail today. Pairs with gate 8(d).

## The 1.1.x → 1.2 ladder (scoped 2026-08-30 by a four-lens readiness dive)

Written after 1.1.1 merged, from four parallel read-only dives (sharing infrastructure,
spatial representation, second-robot embodiment, world-sim integration). It answers a
concrete proposal — Minecraft simulations for perception/memory sharing, a second robot (a
Hugging Face microduck, arriving ~December), a cross-robot "find what was seen before"
coordination task, and a spatial upgrade — by keeping what the code supports and moving the
rest.

**What the dives established, all verified against `main`:**

1. **The 1.2 blocker is real and worse than gate 6 records it.** A merged foreign want reads
   out as exactly `0.0`, silently, on **two** independent key misses — cluster id *and*
   agent id — with a third barrier behind them (body-prefixed tool signatures, gate 7). The
   agent-id half is in no plan doc. Filed as bugs ledger **D43**; the missing behavioural
   test is **D44**.
2. **The sharing evidence that exists dodges the failure mode.** The two federation
   experiments pass only because every infant shares one `agent_id` and one encoder — the
   script says so in its own comment — and no test asserts a behavioural consequence of a
   merge. So the green evidence comes from the one configuration in which D43 cannot fire.
3. **The Reachy Mini cannot translate.** Rotation only; no odometry, depth, or SLAM. The
   proposed coordination task is not a software gap on this robot — it is physically
   unavailable. Two of the three spatial mechanisms that exist are dead code
   (**D45**): `SpatialMemoryBridge` restores zero because nothing writes the key it reads,
   and `SpatialContext` has zero callers.
4. **Minecraft is the designed-for instrument.** `PerceptSource`/`ActionSink` were frozen as
   CC8 contracts naming Mineflayer; a player body is a drop-in YAML; the engine supplies the
   world coordinates the robot cannot. 800–1500 LOC over existing seams.
5. **The multi-robot hardware layer is ready; the runtime above it is not.** `RobotController`
   (12-method ABC), `RobotRegistry` with entry-point plugin discovery, and a working
   `SimulatedController` proof. But `embodied_runtime/selfy.py::Maxim.mini` exposes the raw
   SDK to ~20 call sites in `movement.py`/`media_loop.py` — the single biggest generalization
   cost, and it touches motion-safety code.

**The ordering principle is unchanged** (§"Why this ordering"): predictable deliverable near,
may-fail research far. The proposal inverted it — vision/binding are 1.3 by their own plans'
headers, spatial mapping is on no roadmap, and a release cannot be gated on hardware that
might arrive in December. The ladder below restores the ordering without dropping the idea.

| Release | Theme | Contents | Ship gate |
|---|---|---|---|
| **1.1.2** | **Decomposition** | fail-loud **Stage 2** (never run — it is the extraction's own stated prerequisite, and the per-PR behaviour gate cites a baseline that does not exist); the first `run_agentic_loop` extraction; the four cheap scorecard conditions (Tier-3 dispositions, `test_api_surface.py`, ARCHITECTURE.md EC rows, item 16.10); the D12 `pytest-timeout` guard; a scheduled `-m slow` lane | The length baseline tightens in the same commit, AND the extraction passes the corrected behaviour gate in [god_function_decomposition.md](god_function_decomposition.md) §"Behavior-preservation gate per PR". **Revised 2026-08-30 by measurement:** the original wording — a JSONL sequence-diff against a pre-refactor capture on the same seed — is UNSATISFIABLE. The generative sim is not reproducible on identical code; three runs of one command, *two of them both pre-extraction*, gave `percept` counts of 51, 8 and 0, so the two pre-extraction runs differ from each other more than the post-extraction run differs from either. Held to the letter it fails every mechanical extraction; held loosely it means nothing. What replaces it: the **substrate probe** is the sequence-diff vehicle (no LLM, bit-reproducible — byte-identical 72,239-record captures before and after the first extraction, digests committed), the generative sim supports a **structural** comparison only, and the swallow gate (`fail_loud_stage2.py check`) plus per-section direct unit tests carry the weight for an agent-loop extraction, since the substrate probe never enters `run_agentic_loop`. |
| **1.1.3** | **Merge correctness** | **D43** both halves (id map out of `ec_merge`; re-key `cluster_reward_bias` + `cluster_reward_source`; N→1 fold semantics; a bias-key identity namespace — the undocumented design gap); gate 1 encoder-provenance validator; gate 2 geometry tag + threshold pin; ~~gate 7 typed bundles (`body_ref` + `affordance_namespace`)~~ **SHIPPED 2026-09-01** — option (a), plus a `capability_map` emitted alongside so a later capability namespace is a reader-side change with no migration ([d43_merge_correctness.md](d43_merge_correctness.md) §5a) | **D44**: a test asserting a *behavioural* delta across a merge — `recommend_action` changes — between two genuinely independent agents. Dict equality does not count |
| **1.1.4** | **The world seam** | The Minecraft bridge, `bodies/minecraft_player.yaml`, the world modality channel **plus its selection-dynamics re-baseline**, the two-AUT-one-world harness ([minecraft_benchmark.md](minecraft_benchmark.md)). **Added 2026-09-01: the sensor-encoding change — a `modality:` declaration on the sensor schema, per-type channels derived from it, and a sensor-count-scaled `pattern_threshold`** ([minecraft_benchmark.md](minecraft_benchmark.md) §"The sensor ceiling is a THRESHOLD artifact"). | Infrastructure only, **no claim**. Smoke benchmark green; `is_sim_mode=False` verified to consolidate. **The encoding change ships HERE and not in 1.2, deliberately** — see the note below |
| **1.2** | **Oasis** | The four-arm sharing benchmark, pre-registered, run in Minecraft at n ≥ 50, replicated on two Reachy Minis at n = 12; gates 3, 4 and 8 | The gate ladder below |
| **1.3** | Perception fabric + reflex tier (**unchanged, semi-open**) | Cochlear front-end, vision encoder, binding, three-factor calibration — **plus the microduck** ([microduck_intent_layer.md](microduck_intent_layer.md)), see below | Its own pivotal experiment. **SEQUENCING: [roadmap_1_3_path.md](roadmap_1_3_path.md)** — Stage A duck baseline (unblocks the mic question) → Stage B engine seam / robot factory (N=2 makes the abstraction honest) → Stage C fabric (Stage 0 gates Stage 1) |

> Instrument limit: **[L11](../limits/README.md#l11--sensor-count-dilution-and-the-discrimination-ceiling-behind-it--mitigated)** (dilution + discrimination ceiling). The 1.1.4 work IS L11's stated mitigation; L11 moves to `RETIRED` only when it is shipped AND re-measured.
>
> **Why the encoding change lands in 1.1.4 and not 1.2** (decided 2026-09-01, after measurement).
>
> The measurements are in [minecraft_benchmark.md](minecraft_benchmark.md): the ~14-sensor
> ceiling is a **threshold artifact**, not an information limit; dimension is not a lever;
> distributional moments help detection and *hurt* discrimination; and **discrimination — not
> detection — is the real ceiling** (two different sensors spiking are already 99% alike at
> N=100 before any moment block). The structural fix is per-type modality channels plus a
> scaled threshold, neither sufficient alone.
>
> **The economics argued for folding it into 1.2, and that part is right.** A minor-version cut
> already fires the heartbeat triggers; grouping + threshold additionally fires the EC-threshold
> row, Exp 42 (SensorEncoder / EC-interoception), Exp 45 (`recommend_action`), Exp 48 (extero/intero
> seam) and Exp 53b (`_sensor_embed` / `pattern_complete_or_separate`). Most of that is being
> re-run regardless, so the marginal trigger cost is low.
>
> **The sequencing is what changed.** Two reasons:
>
> 1. **An unresolved representation confounds the headline claim.** If the encoding is still an
>    open question while the four arms run, a null is uninterpretable (sharing failure or encoding
>    failure?) and a positive is unattributable. This is the "choose the statistic before the arms"
>    rule one level up: **choose the representation before the arms.** Gates frozen before data.
> 2. **The Minecraft run cannot discharge Exp 53b.** That row's trigger names `_sensor_embed` /
>    `pattern_complete_or_separate` and states *"the representation is what transfers"* — it is the
>    whole cross-context claim, and its re-run is **hardware-gated on the Reachy**. Sim does not
>    reach it.
>
> **1.1.4 is the right home and costs nothing extra:** it is already scoped "infrastructure only,
> **no claim**" — where an open design question belongs — and it **already budgets the
> selection-dynamics re-baseline** the encoding change needs. The change rides that re-baseline
> instead of requiring its own. Minecraft stays the test venue; the representation is simply
> frozen *before* 1.2's arms rather than during them.
>
> **Hardware batching:** grouping + threshold re-stales Exp 53b on the Reachy, and 1.2 separately
> needs the n=12 two-robot replication. Those are the same scarce resource — plan them as ONE
> hardware block, not two.

### The 1.2 benchmark (the headline claim)

Full design in [minecraft_benchmark.md](minecraft_benchmark.md). Claim: *A's learned
representation changes B's behaviour, where A and B are genuinely independent agents*
(different `agent_id`, independently encoded EC). Four arms — isolated, merged-taught,
merged-satiated, dangling-half — with the dependent measure being B's **first-contact**
action choice on a contingency only A ever experienced.

Gates: merged-taught ≥ 0.70; merged-taught − isolated ≥ 0.20; merged-taught − merged-satiated
≥ 0.20; and **dangling-half ≈ isolated**, which is the falsifier that proves the effect needs
both halves of the bundle rather than the arrival of a file.

**Second claim, same apparatus — the dose–response ladder** (added 2026-08-30). The four arms
ask *does it transfer*; the ladder asks *does it scale*, which is what Oasis rests on. N ∈
{1, 2, 4, 8} agents at a fixed per-agent budget K, measuring **trials-to-criterion**, with a
matched single-agent control at N×K trials on every rung. Primary gate: median
trials-to-criterion strictly decreases with N (negative rank correlation, p < 0.05); a flat
curve is a result and ships as one. The precedent — `5_operant_creche_federation.py`, which
already carries the make-or-break `single_full` control and passed — establishes that pooling
recovers the full-experience policy, but its measure is **saturated at 1.00** and its infants
**share an encoder by construction**, so it cannot speak to rate or to independent agents.
Full design in [minecraft_benchmark.md](minecraft_benchmark.md) §"The dose–response ladder".

**Running this before D43 lands would produce a confident null with a known cause** — arm 2
is currently guaranteed to read out as arm 1. That sequencing is the point of 1.1.3.

### Spatial — descoped, deliberately

Not a capability. The smallest scientifically meaningful version is **landmark-anchored
symbolic location**: `SpatialContext(room/zone)` already exists and is unused, and Minecraft
supplies coordinates directly. Allocentric mapping needs sensing the Reachy does not have
(depth, fiducials, or a mobile base) and is a research program, not a lead-up item. D45 must
be dispositioned either way — wire the dead bridge or mark it `Dormant since`.

### The microduck — 1.3, and possibly JEPA's paired-data source

> **UPDATED 2026-08-31.** The operator supplied concrete design constraints (intent vocabulary
> as its own layer; policy-granularity arbitration over a shared 61-dim observation contract;
> 50 Hz onboard / 1–5 Hz off-board; pluggable sim and `robotd` reward sources; a headless
> episode loop). They close the "unknown SDK / unknown kinematics" hedge below and are worked
> through against the code in **[microduck_intent_layer.md](microduck_intent_layer.md)**. The
> slot does **not** change — see that doc §7. **Rev 2 (post two-lens review round)** withdrew
> that doc's two headline *recommendations* and demoted them to inputs; what survives, and is
> load-bearing *here*, is narrower: (a) the microduck gives gate 7's body-agnostic option a
> **second, portability-side constituency**, so gate 7's design pass must cost it explicitly —
> **but the case study's typed-bundle option remains the scheduled choice and 1.1.3's D44 gate
> does not need the alternative**; and (b) **`last_clamped_axes` reaches no learner** — its only
> `src/` consumer renders prose for the LLM — so the duck's commanded-vs-applied divergence
> channel can be prototyped on the Reachy, by reviving
> `proprioception/pain.py::PainDetector`'s existing graded comparator on the `agent_loop` path.
> (Rev 1 proposed routing it through `side_effects["embodiment_failures"]`; that was **rejected
> in review** — a clamp is a refusal, not harm, and the unconditional valence flip would book
> NEGATIVE for a clamped turn that nonetheless achieved its goal.)

Pushed to 1.3 (operator's call, 2026-08-30) rather than gating 1.2 on hardware with an
uncertain arrival date and unknown sensing. **Superseded 2026-09-01:** the operator has
decided the duck's value is **locomotion** (second-body pressure on the robot abstraction being
a stated secondary benefit), so directional audio is no longer the decisive unknown — the duck
is a new behaviour class either way, and what gates the pre-registration is now the **valence
design** ([microduck_intent_layer.md](microduck_intent_layer.md) §1.1, §5, §8 items 5–6). The
audio reasoning below is retained because it still holds and still argues for the 1.3 slot; it
simply no longer forks the plan. The 2026-08-31 constraints did **not** close it — they enumerate the
duck's sensing twice (the observation contract; the reward sources) and **no microphone appears
in either list**. Every EARNED behavioural result this project has is sound-orienting
(Exp 45, 52, 53b, 54). A robot without a mic array inherits no analogue of the only validated
behaviour, which makes it a new percept modality — experiment work, not a port.

That is precisely why 1.3 is the right slot, and why the operator's instinct to pair it with
[deferred/jepa_cross_modal_alignment.md](deferred/jepa_cross_modal_alignment.md) is worth
recording: JEPA's motivating problem is bridging a 384-dim sensor space and a 768-dim text
space, and its stated blocker is the absence of **paired cross-modal training data**. A
second body with different sensors observing the same world is a natural source of exactly
those pairs — a duck and a Reachy perceiving one scene through different modalities is the
alignment problem stated physically. This is a **hypothesis to test, not a commitment**: it
raises JEPA's bar (paired data must be *collected*, not assumed) and it inherits the 1.3
perception-fabric risk. Revisit when the duck's sensing is known.

Prerequisite either way, independent of the duck: break `selfy.py::Maxim.mini` so the runtime
talks to `RobotController` rather than a raw SDK handle. That work is worth doing on its own
merits and is the gate on any second robot.

**Corrected 2026-08-31 (measured, then corrected again by the review round).** The "~20 call
sites" figure above was wrong about *location*, which is the part that matters. `.mini` has **29**
references in `src/maxim/`, distributed `media_loop.py` 10 · `movement.py` 7 · `workers.py` 6 ·
`selfy.py` 4 · `capabilities.py` 1 · `segment_vision.py` 1 (counts dated 2026-08-31; the durable
finding is the distribution, not the number). The **bulk is the media/audio path** —
`media_loop.py`'s `get_frame` / `get_audio_sample` / `push_audio_sample` / samplerate reads bypass
`get_video_stream()` / `get_audio_stream()` entirely, and `selfy.py`'s three are documented
backward-compatibility media fallbacks.

**But one raw SDK motion dispatch does survive**, and a first draft of this entry wrongly said
none did: `embodied_runtime/movement.py::Movement._enqueue_sdk_look_at` binds
`self.mini.look_at_image`, which the file's own comment describes as bypassing
`ReachyMiniController.look_at_pixel` so the controller's last-commanded head stash never sees the
motion — and CI's `RAW_SDK_MOTION` guard in [.github/workflows/test.yml](../../.github/workflows/test.yml)
carries **two explicit allow-list lines** for that file to let it through. `movement.py` also hands
the raw handle to `move_head` / `move_antenna`, which is exactly the coupling the break-out must
sever. So: the work is **mostly** media abstraction, but it must relocate that allow-listed motion
site and its CI allow-list with it, and the two-lens round on it is a **motion-safety** round, not
only a media one.

## Gates before 1.2 Oasis + Hivemind

Distribution amplifies silent state errors. Before shared substrate becomes an
execution priority:

1. D1 live encoder-provenance validation must reject or migrate incompatible state.
2. D3/D4 threshold and same-dimension geometry compatibility must be explicit and
   tested, not inferred from vector length.
3. EC read-side mutation (D8) must be measured and accepted or separated from recall.
4. Bundle/version compatibility and the sharing threat model must be frozen.
5. The 1.1 architecture-audit and hermetic-suite gates must remain green.
6. **Cluster identity across substrates** (added 2026-08-26; **CONFIRMED and widened
   2026-08-30 — see bugs ledger D43**). The recorded half is the cluster id. The dive found a
   SECOND, undocumented half: bias keys are the triple `(agent_id, cluster_id, tool_signature)`,
   so a foreign want misses on the agent id too, which additionally kills the prompt-annotation
   path `NAc.get_agent_tool_biases`. A bias-key identity namespace is a genuine design gap in
   no plan doc. Original text: `ec_merge` aligns nodes by
   cosine but nothing re-keys `cluster_reward_bias` through the resulting id map, so a
   merged foreign want reads out as nothing. The re-keyed merge path must exist and the
   Exp 52 seeds 42 + 43 must pass Gauntlet #2 merged — see the case study.
7. **Bundle action namespace** (added 2026-08-26; **sharpened 2026-08-31**): bias keys are
   body-prefixed tool names; a bundle must declare its body/affordance namespace (typed
   bundles) or the keys move to the SEM affordance. Decided in the case study's design pass,
   before the first shared bundle. **Sharpening:** the code map in
   [microduck_intent_layer.md](microduck_intent_layer.md) §2.1 pins what the key actually is
   — `embodiment/tool_bridge.py::generate_tools_for_entity` builds `f"{ent.name}_{aff_name}"`
   (flat, underscore-joined, **modulator dropped**), and `similarity/signature.py`'s
   `f"{tool_name}:{outcome_type}"` is the only real structural key; a colon-delimited
   `body:<name>:<verb>` convention **does not exist anywhere in the tree**, and
   `affordance_namespace`/`body_ref` are docs-only (`hivemind/bundle.py`'s manifest has
   neither). Design the namespace as a **capability** namespace with the body as an attribute,
   taking `embodiment/motor.py::MotorStep.sem_key`'s `(entity, modulator, affordance)` triple
   **minus its first element** as the starting shape.

   > **CORRECTED 2026-09-01 (D43 pre-implementation sweep).** The sentence above previously
   > took the triple whole. `sem_key`'s first element is `entity_path` — **it IS the body
   > dependence this gate exists to remove**, so following the instruction literally
   > reproduces the bug. The capability key is `(modulator, affordance)`. Two further
   > corrections: `sem_key` has exactly **two** references in the tree, so adopting it means
   > *building* an identity, not promoting one; and **this gate contradicts
   > [oasis_case_study_taught_orient.md](oasis_case_study_taught_orient.md) §1**, which
   > front-gated the same choice and picked the **body** namespace for a stated reason, with
   > the microduck two-lens round (rev 2) withdrawing its capability recommendation and
   > restoring the case study's. Whoever implements will read only one of these documents —
   > reconcile them in the same commit as the decision. Both options costed against the code
   > in [d43_merge_correctness.md](d43_merge_correctness.md) §5. Note also that gate 7 is
   > **not** what blocks D44: D43's live axes are `cluster_id` and `agent_id`, and the
   > tool-signature barrier does not fire for two agents on one body — which is exactly the
   > configuration D44 requires.
8. **Evidence and ledger coherence** (added 2026-08-27 from the scorecard
   reconciliation). Gates 1–7 exist because distribution amplifies silent *state*
   errors; this gate applies the same argument to the *evidence* behind the state that
   would be shared and to the ledgers that describe it. Four sub-gates, all
   doc-or-lint-sized except (a):
   - **(a) D27 re-filed here.** The bugs ledger had it as "OPEN — 1.1 gate" for a
     release that has shipped: seven `scripts/` harnesses still overwrite committed S4
     evidence unconditionally and ~18 build an encoder without
     `require_semantic_encoder`. The tests-side fix (D25/D26) exists; extend
     `test_evidence_write_policy.py`'s scan to `scripts/` and give the harnesses the
     `--write-experiment-results` opt-in. The Exp 52/54 files a bundle would carry are
     produced by exactly this class of harness.
   - **(b) D5 vs gate 6 — re-argue D5.** D5 (`nac_merge` never folds cluster biases
     across agents) is ACCEPTED in the ledger; gate 6 (a re-keyed merge so a foreign
     want reads out) *requires* the thing D5 accepts not doing. The case study made D5
     load-bearing; its disposition is re-opened in the ledger and resolves with gate 6.
   - **(c) D28 before the ingestion contract.** `create.agent()` silently restores SCN
     regardless of `auto_load`, so "fresh" is not fresh. The Oasis ingestion contract
     defines what a bundle merges *into*; it cannot be written while the fresh/loaded
     distinction is incoherent. Lands before the contract is drafted, not merely in
     1.1.x.
   - **(d) Tier-3 dispositions.** [behavioral_graduation_candidates.md](behavioral_graduation_candidates.md)
     Tier 3 has had 14 of 20 rows Pending since 2026-05-27 — the item that held
     Research integrity from A on the Claude card and the "Tier-3 gets the dispositions
     the 1.0 readiness review promised" upgrade condition. Doc-only sweep: each Pending
     row becomes scheduled (named experiment), Dropped, or Dormant. Not a code change; a
     release with Stale rows is already forbidden by the ledger's own rule, and Pending
     rows fifteen months old are Stale in everything but name.
   - **(e) A merge test that asserts BEHAVIOUR** (added 2026-08-30, bugs ledger D44). The
     only end-to-end sharing test asserts dict equality on hand-set matching keys, and the two
     passing federation experiments share one `agent_id` and one encoder by construction — so
     every green signal for sharing comes from the configuration in which D43 cannot fire.
   - **Also under this gate, scoped to what 1.2 touches:** `hivemind/` gets added to the
     existing CI mypy step (5 of 508 files today; repo-wide is not worth it, but the
     bundle format is a wire boundary and 1.2 rewrites its merge path — that is where
     typing pays).

---

## Why the reflex tier is NOT in 1.1

The verb "enable" was doing enormous work. Verified against code:

- **The `sim.is_sim_mode` gate does not gate a motor command.** It gates a *modeled
  sensor write* (`world_set_azimuth`). Removing it enables a live **fabrication**, not
  a live reflex — the head-frame failure class that cost a full session and six
  falsified hypotheses. SEM motor binding gave real motion to the *affordance* path;
  the §1.16 reflex has no dispatcher of its own.
- **`world_set_axis` has no `live_world_set_sensors` check.** That guard exists only
  inside `ModulatorAffordanceTool`, so a live reflex write would bypass phantom-credit
  protection entirely and break `DoAFeed`'s single-writer claim.
- **A fabricated centering would null the drive that motivates real orienting.**
  `azimuth → 0.0` sits inside `comfort_band: 0.1` → no breach → no drive pain → the
  policy loses its reason to turn. With `drift_rate: 0` and silence-writes-nothing,
  that persists **indefinitely** in a quiet room.
- **The trigger predicate has no measurement behind it.** `salience`/`novelty` are
  constructor constants from `robots.yaml`, so `is_orienting_reflex` fires *never*
  (0.5 vs 0.9) or *on every speech-gated reading*. Worse: `AzimuthDoASource` gates on
  `is_speech_detected`, so **a clap or a bang never produces a reading at all** — the
  transient stimulus class is filtered upstream.
- **Real scope is ~900–1,400 production LOC**, with an unresolved design decision
  (reflex NAc isolation) and a *mandatory unbuilt prerequisite* (`sem_motor_binding.md`:
  the pending map is required before any non-blocking dispatcher ships).
- ~~**Sign-off is motor-blocked.**~~ **2026-08-07 — a spare motor was swapped in.**
  Four plan docs gated on this. **TWO motors were found broken**, so whether the gate
  is fully open depends on whether both are now healthy — see the hardware note, which
  also raises a more interesting question.

---

## Reflex canonicalization — the decision (lands in 1.1, the behavior does not)

**Decision: the DefaultNetwork is the canonical home for all reflexes.** It inherits
`PriorityArbiter` (arbitration), `BehaviorState.inhibited_behaviors` (one-way
voluntary→reflex suppression — the bio-correct direction; symmetric inhibition
oscillates), and `Behavior.cooldown_seconds` (refractory). The §1.16 inline path
reaches none of these and is invisible to the DN — two controllers, one actuator, the
`head=None` failure class.

**The contract any future reflex must satisfy:**

1. Subclass `Behavior`; propose through `PriorityArbiter`.
2. **Never write sensors directly** — enforced structurally, not by convention.
3. Declare a cooldown (refractory) and take habituation/sensitization **gain** from the
   multiplier, never by mutating a declarative threshold.
4. **Emit efference copy.** A reflex that turns the body without telling higher systems
   makes the LLM read the whole scene shifting and attribute it to the world — a
   corollary-discharge failure, direct descendant of the `head=None` lesson.

**What ships in 1.1** is only the structural half — move the `live_world_set_sensors`
check into `world_set_axis` with an explicit `owner=` opt-out for legitimate writers
(`DoAFeed`, the backend's measured readback), so forgetting becomes a refusal rather
than a silent fabrication. Keep §1.16's branch (dormancy over deletion) but make it
*unable* to lie if someone un-gates it.

---

## The orient-vocabulary audit (1.1 item 8)

There are **seven** paths that command orientation, not three. Two are live bugs:

- **`MoveTool` gaze and `turn_around` bypass `ReachyMiniController.goto_target`** — and
  therefore the head-frame composition, the invariant earned at the cost of a full
  session and a three-doc retraction.
- **SEM `motion` affordances (`look_at`, `goto_pose`, `recenter`, `nod`, `shake_head`)
  are advertised to the LLM but are motorless no-ops** — `make_reachy_orient_factory`
  binds only `mod_name == "orient"`. Same dishonesty class PR #459 fixed for
  `focus_on_sound`.

Also designed-in and uncoordinated: `focus_on_sound` **recommends the SEM turn tool by
name** when clamped, with no coordinator, refractory, or mutual exclusion — a single
~45° sound can produce ~97° of world rotation. Only two of the seven paths call DN
inhibit.

**`ReachyMiniController.goto_target` has no lock**, and reads `get_current_pose()`
before composing the head matrix — a TOCTOU on live kinematic state. Two overlapping
callers each compose against a stale body yaw. A single `RLock` spanning read→compose→
dispatch is ~10 LOC and the highest-value fix in this list.

---

## Loudness — UNBLOCKED 2026-08-25 (the section below is the pre-bench reasoning, kept as the record)

> **Superseded by the bench:** [h2_loudness_bench_2026-08-25.md](../experiments/h2_loudness_bench_2026-08-25.md).
> The level exists (`AEC_SPENERGY_VALUES`, pre-AGC) and the daemon already serves it over
> REST; both paths below were unnecessary. Test (a) as written (PCM) was the wrong test —
> the register answers it without PCM. Test (b): the AGC flattens PCM, not the register.
> Item 18's design is 1.1.1.

`DoAReading = tuple[float, bool]`. The daemon serves `{"angle", "speech_detected"}`.
There is no level anywhere *in what we read*. Two paths, both outside our control or against current
config:

1. **Daemon-side** — a new endpoint or extra fields. That is Pollen's code. Unbounded
   calendar, and the 2026-08-05 SDK/daemon version-skew incident is a fresh reminder of
   what that dependency costs.
2. **Local PCM** — `mini.media.get_audio_sample()` is onboard-only and needs
   `media_backend: default`, abandoning the `no_media` config live orient sessions run
   under specifically to kill the frame thief (#456).

**Two bench tests answer this in under a day and must precede any plan:** (a) does
`media_backend: default` yield non-empty audio samples on the live rig, or is GStreamer
broken there? (b) does the XVF3800's **AGC** flatten RMS to near-constant? If (b) fails,
"loud" is unmeasurable on this hardware and the tier is onset-only forever.

**Consequence for 1.3:** the reflex tier's own trigger is gated on this. Until a
sound's intensity is measured, "loud and sudden" is a config constant wearing a
predicate's clothes.

---

## Bio-fidelity corrections (fold into 1.3's plan before implementing)

1. **Drop "startle" entirely — this is ORIENTING.** Startle is PnC (~5–10 ms,
   non-directional, protective bracing). Orienting is superior colliculus (~70–200 ms,
   directional, information-seeking). The code implements orienting.
   `behaviors/startle.py` is *also* actually orienting (vision-only, proposes
   look-toward) — a third mislabel would compound two existing ones.
2. **A sensitization experiment designed on startle literature would falsify for the
   right biological reason.** Repeated aversive pairing does not produce larger turns
   *toward* — in the defense literature it produces freezing or avoidance.
3. **Do not publish pain on startle.** An ambient sound has no entity, so the publish
   either no-ops or accumulates junk into `_percept_valences` →
   `get_percept_aversions` → `TextSalienceScorer`, boosting every future percept
   containing that token as aversive. Use `Reaction(kind="surprise")` — **declared in
   `ReactionKind` with zero producers today** (every Reaction in the codebase emits
   `kind="pain"`). For genuine nociception, declare a `sound_level` sensor with a
   homeostatic comfort band and let the existing severity-latched drive-pain channel
   handle it.
4. **Habituation ships with or before sensitization, never after.** Dual-process theory:
   response = S-R decrement + state increment. Sensitization alone escalates
   monotonically — the repo already warned: *"NH-5: an audio startle at priority 0.95
   must habituate or it starves orient."*
5. **Front-gate answer: ride `NoveltyTracker`.** It already implements dual-process for
   vision — habituation decay, spontaneous recovery, novelty floor, ceiling-clamped
   sensitization, with an explicit VTA rationale in `MemoryHub._wire_sensitization`.
   The work is generalizing it off COCO class keys, not building from scratch.
6. **Threshold stays declarative; gain is the plastic part.** CeA potentiates PnC's
   *response*, it does not lower its threshold. So:
   `effective = measured × habituation × sensitization_gain` tested against the
   **fixed** YAML threshold. No learned thresholds, no mutable YAML, no new state file.
7. **PPI explicitly out of scope** — the loop tick (33–500 ms) exceeds the PPI lead
   window, there is no sub-threshold acoustic channel, and no graded amplitude.
8. **Halo risk:** graduation Tier-3 #9 reads "Reflex system … EARNED — Experiment 09,"
   but that covers the keyword-matched NARRATIVE percept reflexes only (Exp 09 ran
   `bodies/base_humanoid` against dragon attack/fire narration — this sentence
   originally repeated the row's own "infant thermal contact" mis-description;
   corrected in the 2026-08-07 doc-truth pass). A different modality, trigger,
   circuit, and output **does not inherit it.**
9. **Better framing available:** SC is a *multisensory integration* structure, so
   orienting is arguably the oldest **instance** of binding rather than a layer beneath
   it. That makes 1.1→1.3 continuous rather than stacked.

---

## Free findings (fix during the 1.1 doc-truth pass)

- ✅ **RESOLVED (doc-truth PR, 2026-08-07):** **Graduation Tier-3 #9 cites a file that
  does not exist** (`09_percept_reflex.md`; the real file is `09_percept_reflex_poc.md`)
  **and mis-describes the experiment** (says infant thermal contact; Exp 09 ran
  `bodies/base_humanoid` against dragon/fire narration). A Principle-5 defect on an
  EARNED row. — Row corrected + halo caveat added.
- ✅ **RESOLVED (doc-truth fold, 2026-08-13):** **`CaptureManager` is constructed only
  when `has_vision`** — an audio-only robot gets no audio thread at all. — Gate widened
  to `has_vision or has_audio`; `CaptureManager(has_vision=)` skips the frame +
  segmentation threads for camera-less robots.
- ✅ **RESOLVED (doc-truth fold, 2026-08-13):** **`get_audio_stream()` returns
  non-`None` under `no_media`** — a capability lie at the stream surface. —
  `ReachyMiniController.connect()` now gates both stream wrappers on the SDK's actual
  devices (positive-evidence downgrade, mirroring `derive_media_capabilities`).
- ✅ **RESOLVED (doc-truth fold, 2026-08-13):** **`inhibit_during_tool_execution` is
  dead code** — zero callers, while a plan doc claimed it "covers half the race." —
  `inhibit_for_tool` docstring now states it is UNWIRED; the
  live_audio_orient_wiring.md claim corrected. Wiring it (a behavior change on the
  robot runtime) stays a deliberate decision, not a drive-by.
- ✅ **RESOLVED (doc-truth fold, 2026-08-13):** **`Reaction(kind="reward")` is
  published** from `cerebellum_modulator.py` and was not a member of `ReactionKind`. —
  `"reward"` added to the Literal with a producers-must-be-members note.
- ✅ **RESOLVED (doc-truth PR, 2026-08-07):** **`perception_placement.py`** — 267 LOC,
  zero `src/` callers, claimed "✅ shipped." Wire it or mark it Dormant per Principle 2.
  — Marked Dormant (module docstring, resurrection trigger = the 1.3 fabric actually
  placing stages); plans README claim corrected to PARTIALLY landed.

---

## Enthusiasm-to-evidence flags (recorded so they don't recur)

1. **Loudness scoped as a 1.1 addition without checking the wire format.** It is
   `(float, bool)`; the fix lives in a vendor's daemon. *(Corrected 2026-08-25: the
   vendor's daemon already served it — the flag was right, the diagnosis was one
   register short. Check the vendor's parameter table before declaring a dependency.)*
2. **"Enable the reflex tier"** — the verb implied a flag; the reality is ~1,000 LOC
   against a deliberate correctness fold.
3. **Fabric pulled 1.3 → 1.2** against its own header, with no new evidence.
4. **"Sensorimotor" as a headline** — the *direction* result is robust; the *magnitude*
   line rests on n=1 sessions on a robot with a known motor asymmetry, atop a DoA gain
   contested by 3×. The docs are honest about all of this; the roadmap headline must
   inherit that honesty and claim the loop, not the sensorium.
