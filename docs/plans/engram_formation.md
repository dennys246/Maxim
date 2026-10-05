# Engram formation — the gaps, and how each gets fixed

**Status:** ACTIVE, opened 2026-09-25. A 1.4 line (see [roadmap_1_4.md](roadmap_1_4.md) §Parallel
lines, "Engram integrity", and release threshold T7). Items E1–E4 are engineering and gate 1.4.0;
E5–E7 ride on rungs that already own them and gate nothing on their own.

**Source.** A four-path code audit on 2026-09-25 (substrate, episodic, semantic, motor), every
finding re-verified at the cited symbol before an issue was filed. The living scorecard of how each
engram family forms is [docs/wiring/engram-formation.md](../wiring/engram-formation.md) — this plan
is the fix list; that page is the state. Update both when an item lands.

## Front-gate scope answer

**No new mechanism.** Every item below fixes, documents, measures or routes an EXISTING path:
a builder argument (E1), docstrings and docs (E2), a dead branch (E3), an offline measurement (E4),
a sensor encoding owned by the Rung B keying line (E5), the parked 2S-e consumer (E6) and the
graded-predictor audit Phase 5 already names (E7). Anything that would add a store, a bus or a
selection term is out of this plan and goes through its owning rung's four-lens review.

## What the audit found — one paragraph

Engrams **form** everywhere they are designed to. The one family that is an engram on all four
counts (forms, stays specific, is recalled, changes behaviour) without the LLM is the **situation
engram** — an EC sensor cluster carrying NAc fear or want — and it has seven EARNED ledger rows.
Its limit is geometric (cosine sees direction; one daily wrap boundary). Episodic and semantic
traces form honestly and are recalled, but reach behaviour only through LLM prompt text; the
substrate-native cue is built and its result discarded. Motor engrams are fully implemented and have
no production caller; the Cerebellum's forward model trains live and was silently never saved
(fixed 2026-10-04, E1/#908 — it now saves to `<home>/cerebellum.json` at session end).

## Items

| # | Issue | What | Kind | Gates 1.4.0? | Ledger triggers fired |
|---|---|---|---|---|---|
| E1 | [#908](https://github.com/dennys246/Maxim/issues/908) | Cerebellum state never saved — **DONE 2026-10-04** | bug, `src/` | **yes (T7)** | none |
| E2 | [#909](https://github.com/dennys246/Maxim/issues/909) | Motor-engram read side: docs overclaim, dormancy undeclared — **DONE 2026-10-04** | docs + docstrings | **yes (T7)** | none |
| E3 | [#910](https://github.com/dennys246/Maxim/issues/910) | `[DANGEROUS]` annotation unreachable | dead branch, `src/` | **yes (T7)** | none (branch never fires) |
| E4 | [#911](https://github.com/dennys246/Maxim/issues/911) | Recognition widening text-only; text drift hazard | scope doc + offline measurement | **yes (T7)** — the measurement, not a fix | none (offline) |
| E5 | [#899](https://github.com/dennys246/Maxim/issues/899) | `time_of_day` linear → daily wrap boundary | substrate geometry | no — Phase 5 keying / Rung B | **Exp 53b, 56, 60, 61, 62** |
| E6 | [#848](https://github.com/dennys246/Maxim/issues/848) | Episodic engrams never reach action (2S-e) | new consumer | no — memory line | Exp 60–62 if it touches selection |
| E7 | #909 (read side) | Motor engrams / forward-model predictions unread | resurrection | no — Phase 5 graded predictor | per its own plan |
| — | R4 (roadmap Phase 5) | node-keyed `reward_bias` not read by selection | credit routing | owned by R4 | per R4 |

### E1 — save the Cerebellum where it loads from (#908)

**Status: DONE 2026-10-04.** `build_bio_stack` binds `<home>/cerebellum.json` (the bio-stack
persistence dir, i.e. the agent home), loads it at session start unless `load_persisted=False`, and
`BioStack.on_session_end` saves it (refusals and write failures logged at ERROR, never raised). It
carries the #971 store guard: it never saves over a file it did not read, and an unreadable file is
kept as `cerebellum.json.corrupt-<UTC>`. The Reachy embodied runtime neither trains nor saves it.
The text below is the plan as written.

**Root cause.** Two sources of truth for one path: `build_bio_stack` hard-codes
`p / "cerebellum.json"` for `load`, while `BioStack.save_cerebellum` reads
`CerebellumConfig.persistence_path`, which nothing sets. Three call sites
(`BioStack.on_session_end`, `agent_factory.py` shutdown, `minecraft_harness.py`) each call a no-op,
and the bio-memory brief's invariant cites the call as its own guard.

**Fix.** Construct `Cerebellum(config=CerebellumConfig(persistence_path=str(p / "cerebellum.json")))`
when a persistence dir exists; `load()` and `save()` both use the config. Remove the
`except ImportError` hand-rolled writer in `Cerebellum.save` (a second writer that cannot fire).
Not the fix: passing a path at each caller.

**Guard (prove by deletion).** Real bio-stack in a tmp home → `observe_from_action` → 
`BioStack.on_session_end()` → file exists → second stack reads the same prediction. Deleting the
`persistence_path=` argument must turn it red. Rewrite the brief's invariant to cite this test
instead of the call.

**Consequences.** `~/.maxim/<home>/cerebellum.json` starts appearing; confirm hivemind bundles still
exclude it (they do today). No behaviour change: predictions have no consumer (E2/E7).
`[Unreleased]` line required (`src/` change).

### E2 — say what runs, mark what doesn't (#909)

**Status: DONE 2026-10-04.** Forward-model training is documented as live (SEM affordances →
`observe_from_action`, confidence in `sim_cerebellum` telemetry). `predict`, program crystallization
and the rest of the motor-engram path (`query_engrams`, `cleanup_program`, `engrams.py`) are marked
`Dormant since 2026-10-04 (#909)`, beside the earlier markers on `CerebellumModulator` /
`cerebellum_modulator_factory` (2026-05-26) and `form_engram` (2026-09-22);
there is no program executor in `src/`. An always-empty motor-programs prompt section is expected.
The text below is the plan as written.

**Docs.** Rewrite `docs/embodiment_guide.md` §Motor Engrams, §Program Executor, §Cerebellum
activation in production and the persistence line; `docs/skills.md` lines 3–8;
`docs/embodiment_yaml_reference.md` "engram matching (Phase 1b)". State: forward-model training is
live (and, after E1, persisted); engram formation/recall, program crystallization, the
`CerebellumModulator` and `predict` have no production caller.

**Dormancy.** `Dormant since 2026-09-25: no production caller` docstrings on
`Cerebellum.query_engrams`, `observe_action_sequence` / `ProgramRegistry.observe_sequence`,
`cerebellum_modulator_factory`, `Cerebellum.predict`. `form_engram` already carries one. Remove the
unread `EngramConfig.gate_tightening_factor` or mark it — it has no reader and no plan.
Dormancy, not deletion (CLAUDE.md principle 2): callers stay, nothing new builds on it.

**Guard.** A caller-scan test (the 2S-d pattern): each dormant symbol has zero non-test callers in
`src/` + `scripts/`; the test fails when one gains a caller, forcing the dormancy marker and this
plan's E7 to be revisited in the same PR.

### E3 — the danger label (#910)

**Root cause.** The annotators were written against the pre-clamp `reward_bias`; negative experience
now lives in `percept_valences` / `cluster_fear` / edge valence, none of which the annotators read.

**Status: DONE 2026-10-04 (option 2, owner decision; extended to a third site found in review).**
The unreachable branch is deleted in `tools/discovery.py::SensePresenceTool._annotate_aff`,
`integration/bio_enrichment.py::_annotate_affordance_valence` and the substrate fallback of
`tools/discovery.py::SenseToolsTool._nac_annotation`; the docstrings say "effective or unlabeled", and
`discovery.py`'s two `except Exception: pass` report through Stage-1 `log_swallowed_exception` (handle-and-log,
not a narrower type). No behaviour change for any state a writer produces: every writer clamps
`reward_bias` to ≥ 0. `NAc.load_state` does not re-clamp: [#1102](https://github.com/dennys246/Maxim/issues/1102).

**Deferred.** Option 1 (read the store that holds the danger) changes the LLM path; open it only if a
measurement shows the LLM needs a learned-danger cue. Revive trigger: a sim/experiment where the LLM
repeats a harmful affordance the substrate has negative valence for. It may be cheaper than "an
affordance → entity-class join" suggests: `SenseToolsTool._nac_annotation` already reaches harm through
the `tool:{entity}_{affordance}` causal links (`caution: …`), a keyed store an affordance can reach.

**Guard.** `tests/unit/test_affordance_danger_label_910.py` (negative experience annotates as unlabeled;
harm after reward removes `[effective]`; both swallows report) and the injected-negative-bias tests in
`test_tool_discovery.py` / `test_bio_enrichment.py`, which fail if a danger branch returns. The
strict-xfail `test_learned_harm_in_the_percept_store_reaches_a_danger_label` is option 1's revive marker:
it seeds harm in `percept_valences`, never through `reward_bias`.

### E4 — widening scope, and measure the text hazard (#911)

**Docs.** The brief's `_reward_bias` invariant and the tracker say widening is **text-only**;
sensor engrams never widen.

**Measurement (offline, no rig).** A script under `docs/experiments/data/` (the
`*_cosine_check.py` pattern): a rewarded-node text fixture, bias ∈ {0, 0.1, 0.2}, isolated vs
sequential EC, cluster purity per arm. Committed with its output.

**Decision tree.**
- Sequential ≈ isolated at bias 0.2 → close #911 with the numbers; no code.
- Sequential collapses → a design entry in this plan: count override-widened matches without
  averaging them into the centroid, or freeze text centroids on override matches. Either is a
  change to `pattern_complete_or_separate` and fires the "EC threshold / centroid-update change"
  trigger on the EC completion ledger row — re-run that row's guard in the same PR.

**Sensor widening** — no action; recorded as an input to the Rung B keying design (it would pull
neighbouring situations into a node that carries fear or want: a generalization mechanism).

### E5 — the daily wrap (#899)

Owned by roadmap 1.4 Phase 5 "Keying / generalization". Candidate: encode `time_of_day` as a
sin/cos pair (two rest-aware sensors), or remove it from the world channel. Either moves world
geometry and **fires the re-run triggers of Exp 53b, 56, 60, 61 and 62 rung A** plus the
retrosplenial §5 registry. Sequence: offline replay on captured vectors first
(`cosine-separation-is-directional.md` corollary 3), then its own plan + four-lens design review,
then the re-runs. Not before E1's instrument work lands (the refactor-during-a-may-fail rule).

### E6 — let an episodic engram act (#848, 2S-e)

Owned by [memory_strength_and_forgetting.md](memory_strength_and_forgetting.md) Phase 2S-e
(candidate (B), generalization by pattern completion, chosen 2026-09-24; plan section and four-lens
review owed). This plan adds one requirement to its design review: the consumer must be measured
through the real `recommend_action` with a recall ablation that holds the situation engram
identical — otherwise an episodic effect is indistinguishable from the cluster fear already there.
The water classroom cannot validate it (two-point situation space); the review names the world that
can.

### E7 — motor engrams: resurrect or retire

Owned by roadmap 1.4 Phase 5 "A graded predictor (anticipation)", whose audit already names
`embodiment/cerebellum.py`. That audit decides whether the forward model can carry "how far pain
is". Yes → its plan wires `predict` into the substrate path (not `CerebellumModulator`'s LLM
fallback) and may revive engram recall as context. No → E2's dormancy stands; at the next release
checkpoint with no reviver, the motor-engram docs move to a "designed, never wired" appendix.

## Sequencing

```
PR 1  docs only (this plan, the tracker, roadmap, brief pointers)        ← now
PR 2  E1 + E2 (small src + docs + caller-scan guard)                      ← DONE 2026-10-04
PR 3  E3 (dead branch + swallow narrowed)                                 ← with PR 2 or after
PR 4  E4 measurement script + result; code only if the tree says so      ← offline, any time
E5    Phase 5 keying plan → four-lens → re-runs                           ← after Phase 0 instrument
E6    memory line 2S-e plan → four-lens                                    ← its own schedule
E7    Phase 5 graded-predictor audit                                       ← when a rung names it
```

Each `src/` PR needs a pre-merge two-lens review round and an `[Unreleased]` CHANGELOG line.

## Done when

- T7 holds: #908, #909, #910 closed; #911 closed with its committed measurement (and a design entry
  here if the measurement said collapse).
- The tracker's §1 table is re-verified in the release PR (caller grep for every ✅ in the
  "Changes behaviour" column).
- E5–E7 each have a named owner document with a status line — not necessarily shipped.
