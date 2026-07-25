# 1.1 release checklist — roadmap audit against actual merge state

**Created:** 2026-07-23. A tracked artifact so "what's left before 1.1" is a list, not a
synthesis. 1.1 theme (from [README.md](README.md)): **embodiment grounding + substrate-primary
validation** — orient-to-center on Reachy Mini, Exp 44 embodied choice, Oasis.

**How to read:** ✅ done/merged · 🔵 in flight · ⬜ not started · ⏸ above-the-line (defer to 1.2).
This audits the cognitive-architecture track (this repo). The **product-surface track** (maxim
serve / Console / website) lives in separate repos on its own clock — see the last section.

---

## Substrate-primary validation gate — ESSENTIALLY MET ✅

The thesis-load-bearing behavioral wins that make "substrate-primary validation" honest:

- ✅ **Exploration policy** — Exp 42 GRADUATE (safe-vs-harm discrimination, mechanism-level, no
  LLM in the action path), PR #380.
- ✅ **Real-hardware sensorimotor learning (orient Layer 1)** — Exp 45 (direction, cross-session,
  merge) + 45b/45c (magnitude) + **45d** (magnitude replication + cross-session-of-magnitude),
  PRs #392/#413. Direction EARNED decisively; magnitude EARNED with characterized ceiling.
- ✅ **Embodied operant orient** — Exp 48 GRADUATE (the cradle_mother sim off chance), PR #412,
  unblocked by the extero/intero seam.
- ✅ **Extero/intero multi-modality seam** — PR #411; de-dilutes direction from the drives.
- ✅ **Perception pipeline placement** — `runtime/perception_placement.py`, `"audio"` EC
  modality, DoA front-end, orient affordances on `reachy_mini.yaml`, PRs #382–#385.

**Read:** the substrate-primary *validation* half of 1.1 is done. The orient line has cleared its
1.1 bar; what remains below is the LLM-primary counterpart (Exp 44) + the Oasis deliverable +
release mechanics.

---

## Remaining before 1.1 can ship

### 1. 🔵 Exp 44 — LLM-primary embodied choice (bounded experiment; NOT fire-and-forget)
- Apparatus shipped: G1 deterministic scene harm (`MAXIM_DETERMINISTIC_SCENE_EMBODIMENT`),
  imagination gate (`MAXIM_DISABLE_IMAGINATION`), the `--aut-mode llm-primary` harness.
- **Blockers before a valid run** (see [exp44_overnight_runcard.md](exp44_overnight_runcard.md)):
  ops stack (model + n_ctx via `maxim config`), a **confirming validation seed** (does the AUT
  act? does harm fire? is arm A below ceiling?), and the **Track-1 drive-pain cadence** change
  that stales prior numbers.
- **Dependency:** wants [transition_based_drive_pain.md](deferred/transition_based_drive_pain.md)
  to land first so the drive-pain it measures is onset-based, not the current state-based cadence.
- **Open decision:** the harness doc + the calibrated A≈B≈C-null prior mean this may resolve as
  "body_state doesn't move LLM-primary behavior" — a legitimate, cheap-to-earn null that corrects
  the B3.1 "shipped" overclaim. Decide **run-the-arms vs invest-in-substrate-native** before
  committing ~48h.

### 2. ⬜ Oasis — the substrate-sharing deliverable (THE LONG POLE)
- The 1.1 back-half from [maxim_hivemind.md](maxim_hivemind.md): the persistent "gathering place"
  where distilled substrate snapshots pool (~800 LOC). Shareability infra already shipped in 1.0
  (B5, `src/maxim/hivemind/`, PRs #305–#311).
- **Gate now met:** it was blocked on "substrate-primary stable enough to host a persistent
  instance" — Exp 42/45/48 have established that. Oasis is unblocked and is the main build
  standing between here and a 1.1 that matches its stated theme.
- Not started. This is the biggest remaining chunk of engineering.

### 3. ⬜ transition_based_drive_pain (small; unblocks a trustworthy Exp 44)
- Revival trigger fired twice (Track 1 per-iteration cadence + the seam's per-tick drive-pain
  noise). Onset/transition-based drive-pain instead of state-based re-publish every tick.
- Bounded change; do it before Exp 44 so the measured drive-pain is causally clean.

### 4. ⬜ Release mechanics
- Version bump (`pyproject.toml` + `src/maxim/__init__.py` in sync), CHANGELOG, docs pass.
- `pymaxim.bio` URL swap in pyproject/README once docs.pymaxim.bio is live.
- Behavioral-graduation-candidates walk: confirm all Earned Tier-1 rows are Maintained (none
  Stale) for the minor-version heartbeat.

---

## Above-the-line — do NOT let these creep into 1.1 scope ⏸

- **S4 / S3 continuous orient magnitude** — Layer-2 research (population-vector readout is being
  hardware-tested now; if it earns, it lands as Exp 45e). Layer 1 is the 1.1 bar and is done.
- **Reachy runtime integration** ("learning ON in production") —
  [orient_runtime_integration.md](orient_runtime_integration.md), gated on S2 gain calibration.
  1.2 unless explicitly pulled.
- **Passive sense discovery**, **sem environmental proximity sensing**, **grounded-language
  Phases 1–3**, **mesh C5–C8** — all post-1.1 / 1.2 per their plan docs.

---

## Parallel track — product surface (separate repos, other session)

Part of the 1.1 *product* surface per [project_maxim_ui_ecosystem]; driven separately, own clock:
- `maxim serve` FastAPI facade (#416) — the OpenAPI contract the UI generates against. Seams fill
  the 501 stubs: PROBE shipped (#419); **next RECALL / SETUP → HANDLE**.
- **maxim-pulse** (Console-localhost + Reachy-HF-Space over a shared kit), **maxim-web**
  (Astro/Starlight → pymaxim.bio). Local-first: Console is 127.0.0.1-only.
- FIT substrate-footprint measurement (#415) — informs the leader-placement story.

Not blocking the *engine* 1.1, but the coordinated product story wants both tracks at a shippable
state. Confirm with the UI session where its critical path stands before calling 1.1 done.

---

## Critical path (recommended)

```
transition_based_drive_pain  →  Exp 44 (validation seed → arms, or the informed null)
                                                    │
                        (parallel) Oasis build  ────┤  ← the long pole
                                                    │
                        (parallel) UI surface  ─────┤
                                                    ↓
                                          release mechanics → 1.1
```

**Oasis is the long pole.** Everything else is a bounded experiment (Exp 44),
a small unblocker (transition_based_drive_pain), or already done. The honest 1.1 gate is:
**substrate-primary validation (done) + Exp 44 resolved (run-or-null) + Oasis shipped.**

## Maintenance
Re-audit this doc when a listed item merges or a new 1.1-scoped plan lands. Each ⬜/🔵 should
either progress or get an explicit 1.2 deferral — no silent scope drift.
