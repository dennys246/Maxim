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
| **1.1** | **"Sensorimotor"** — *the substrate leaves the simulator* | Already-merged embodiment work + release correctness, contract truth, and the verification debt it incurred. **Zero new mechanisms.** | Medium. Remaining work is concentrated in liveness, stable API, tests, and release closure. |
| **1.2** | **Oasis + Hivemind** | ~1,400 LOC of de-risked engineering with a cleared gate | Low-medium. Known shape. |
| **1.3** | **Perception fabric + reflex tier** | Cochlear front-end, vision encoder, binding, three-factor calibration, DN-canonical orienting reflex | **High — contains a pivotal may-fail experiment.** |

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

## 1.1 cut line — reconciled 2026-08-19

> **Ship the work that is already merged, plus the correctness, contract, truth,
> and verification debt that work incurred. Zero new mechanisms.**

| # | Item | State / release contract |
|---|---|---|
| 1 | `_cosine` dimension guards + frozen-modality parity | **DONE** (#467). Same-dimension geometry compatibility remains a pre-1.2 gate (D4). |
| 2 | CHANGELOG reconstruction + historical tags for 1.0.0→1.0.6 | **DONE** for the historical interval. The post-1.0.6 unreleased record and final 1.1 release transaction remain. |
| 3 | n_ctx clamp and headroom truth | **DONE**. Re-attest through the Big-Model heartbeat. |
| 4 | Persona hard-remove | **DONE**. Keep compatibility/error behavior covered. |
| 5 | Graduation heartbeat walk | **IN PROGRESS** — Sim-Short complete; Big-Model chapter RAN 2026-08-21 (Qwen32B re-fire → instrument finding L8, row stays PARTIAL); **H1 8-rep delivered-shift block DONE 2026-08-24**; final-RC re-attestation of the pytest-only rows remains (minutes, at the RC commit). |
| 6 | H1 healthy-hardware re-characterization | **DONE** — base session 2026-08-08 + the `_big` delivered-shift block 2026-08-24 (n=8/side in one session, 0.943 of command both sides; `_big` YAML magnitudes stay frozen; the right-side Δaz asymmetry is a follow-up still to be pre-registered; D30/D31 filed). |
| 7 | Artifact stamping | **DONE**. |
| 8 | Orient-vocabulary audit + workspace-limit bypass safety | **DONE** (#472); retain hardware-safety guards in the final RC suite. |
| 9 | Documentation truth pass | **SUBSTANTIALLY DONE, CONTINUOUS THROUGH RC**. Reconcile plans, architecture, decisions, API count/dependencies, and release surfaces. D21's EC performance claim was corrected on 2026-08-19. |
| 10 | Atomic NAc + EC invalidation | **DONE (2026-08-23)** (D2). `MEMORY_PATHS` gained the missing `ec` key and `MEMORY_PAIRS` declares `{nac, ec}` inseparable, so clearing either half pulls in the other and `all` no longer leaves a stale EC behind. |
| 11 | Annotation S4 non-stationarity analysis | **OPEN — 1.1 GATE**. Offline analysis only; record the result, not merely the analyzer. **Inputs located 2026-08-24:** the Exp 44b pilot captures are NOT on the operator Mac — they live on big-mac-mini (the pilot machine; `~/exp44b/pilot/arms/<arm>/seed<n>/capture.jsonl` + `requery/`). Run `scripts/exp44/analyze_nonstationarity.py --json` there per capture and commit the JSON under `docs/experiments/data/44b_s4_nonstationarity/` with a results section in [44b_pilot.md](../experiments/44b_pilot.md); that commit closes the gate. |
| 12 | Planning-turn liveness + truthful progress state | **DONE** (D13/D14/D22). Bounded recovery or typed terminal abort, observationally true progress display, and non-zero process/harness propagation for unusable results. |
| 13 | Stable Python API contract repair | **IN PROGRESS — 1.1 GATE** (D15–D18). D15 `goal`/`robot` and D16 `run()` cleanup are DONE in v1.0.9; **D17 complete load semantics and D18 tool-registration lifetime are DONE (2026-08-23)** — `load.agent()` restores ATL before returning and raises `MemoryCorruptionError` naming every unreadable file (explicit `on_corrupt="fresh"` opt-in), and tool registration is persistent with `unregister_tool`/`clear_registered_tools`. `home_dir` completeness plus `imagine()`/`campaign()` cleanup stay 1.1.x. |
| 14 | Hermetic required fast suite | **DONE in v1.0.9** (D20). Unique temporary user/config/cache roots, offline model-hub defaults, explicit pretrained-asset opt-in, per-test path-cache reset, and the wider CI gate are executable. |
| 15 | Architecture-audit enforcement | **DONE (2026-08-24)** (D19). The 33 findings are classified (10 typing-only / 16 function-local lazy imports, none a cycle-break / 7 module-level) in a reviewed baseline (`src/maxim/utils/architecture_baseline.json`, keyed by file + module + scope + accepted symbols); the fast suite fails on additions (incl. a widened symbol list), stale entries, and unreviewed entries. Zero debt is 1.1.x item 11, not a release blocker. |
| 16 | Release, website, and agent-guidance truth | **OPEN — 1.1 DOCS GATE; D24 blocks the 1.0.9 publication transaction.** The 38-route live content audit is DONE and its cross-repo handoff is recorded; maxim-web corrections, path-preserving `docs.pymaxim.bio` redirects, visual/accessibility verification, exact-wheel command checks, legacy deep-link migration, and post-upload PyPI link verification remain. Repo metadata now points directly to the canonical getting-started route. |

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
  "shipped to PyPI" claims**: PyPI's latest release is 1.0.0; v1.0.1–v1.0.6 are
  git-tag-only and both CLAUDE.md and the CHANGELOG said otherwise.
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
4. **Evidence closure:** record S4, run Big-Model and hardware heartbeat chapters,
   then re-attest cheap rows at the exact RC commit.
5. **Release transaction:** audit the canonical website against the exact release
   artifact, reconcile version policy/docs/PyPI project links, build/check the
   package, cut `1.1.0`, tag it, and publish matching release notes and artifact.

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

## 1.1.x follow-through (post-cut, pre-1.2 — items 1–10 adopted 2026-08-13; items 11–14 added 2026-08-19 and ratified with the cut-line reconciliation)

The 1.1 cut line stays closed; these are the next minor-stream items in rough order.
Each POINTS at its owning plan — stages live there, not here:

1. **Exp 44b pre-registration freeze → confirmatory campaign**
   (`protocols/exp44b_preregistration.md`). The core research claim's power run.
   Freeze blockers: F1 name-copying control (→ Exp 51), determinism measurement
   (#496 built it — run it), invalid-action rule (F7), entangled-axes framing fix (F2).
   Deliberately NOT a 1.1 gate — holding the release for a 10-seed/arm campaign was
   considered and declined; it opens 1.1.x instead.
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
8. **n_ctx leg 3, cross-process** — a `served_n_ctx` handshake readable across processes
   (today `maxim config` alignment is the documented mechanism; acceptable carry, close
   it when touching the lane code anyway).
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
13. **Dormant-path decisions for D6/D9** — either wire and behaviorally graduate
    Hebbian multi-node binding / temporal-event producers, or mark the unused contract
    dormant and stop implying it learns in production.
14. **Published-support truth** — **PARTIAL in v1.0.9:** lightweight CI
    install/import/CLI lanes now cover Python 3.10, 3.11, 3.13, and 3.14 while
    the full suite covers 3.12; contributor guidance now matches the seven core
    dependencies and 18 API verbs declared by the package (`recall` landed post-1.0.0 and is still undocumented — see below). Keep dependency and
    verb-count drift checks executable as those surfaces change.

## Gates before 1.2 Oasis + Hivemind

Distribution amplifies silent state errors. Before shared substrate becomes an
execution priority:

1. D1 live encoder-provenance validation must reject or migrate incompatible state.
2. D3/D4 threshold and same-dimension geometry compatibility must be explicit and
   tested, not inferred from vector length.
3. EC read-side mutation (D8) must be measured and accepted or separated from recall.
4. Bundle/version compatibility and the sharing threat model must be frozen.
5. The 1.1 architecture-audit and hermetic-suite gates must remain green.

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

## Loudness — blocked outside this repo

`DoAReading = tuple[float, bool]`. The daemon serves `{"angle", "speech_detected"}`.
There is no level anywhere. Two paths, both outside our control or against current
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
   `(float, bool)`; the fix lives in a vendor's daemon.
2. **"Enable the reflex tier"** — the verb implied a flag; the reality is ~1,000 LOC
   against a deliberate correctness fold.
3. **Fabric pulled 1.3 → 1.2** against its own header, with no new evidence.
4. **"Sensorimotor" as a headline** — the *direction* result is robust; the *magnitude*
   line rests on n=1 sessions on a robot with a known motor asymmetry, atop a DoA gain
   contested by 3×. The docs are honest about all of this; the roadmap headline must
   inherit that honesty and claim the loop, not the sensorium.
