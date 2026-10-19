# B4 (DRAFT v2, 2026-10-18) — Exp 37 / T1-2 re-fired with floors: does carried substrate memory change behaviour beyond a bare LLM and an LLM with notes?

> **STATUS: DRAFT v2, paper only. Not frozen; no harness change exists.** Session E of 1.3.3. v1 went through the
> four-lens design review on 2026-10-18 (reports verbatim in [rationale/b4-exp37-floors/](rationale/b4-exp37-floors/));
> every lens returned DO-NOT-BUILD (14 in all), and v2 folds them (§What the review changed). The run waits on B2
> (`bare_llm`, [#1352](https://github.com/dennys246/Maxim/issues/1352)), B3 (`llm_notes`, Reflexion first,
> [#1353](https://github.com/dennys246/Maxim/issues/1353)) and the `src/` items in §Build prerequisites, all 1.3.6
> work; [#1354](https://github.com/dennys246/Maxim/issues/1354). Plan: [baseline_floors.md](../plans/baseline_floors.md)
> §B4. Parent: [37_cross_session_graduation.md](37_cross_session_graduation.md) (T1-2, STALE since 2026-10-02).
> Binding limit: [L8](../limits/README.md). T1-2 is resolved by this run before 1.4.0, with no further exception
> (owner 2026-10-16). **Owner decisions OB1–OB7 taken 2026-10-18 (§Owner decisions taken).**

## Why

T1-2 claims a cross-session behavioural delta carried by the bio substrate (Wire-A, Wire-1, NAc reward bias) under an
LLM agent. Its record cannot carry a claim today:
1. **No floor.** Every Exp 37 arm runs full Maxim; BF2 requires beating `bare_llm` and, cross-session, `llm_notes`.
2. **The baseline moved.** L8: arm A 0.420 (June, SD 0.27) → 0.710 (August, SD 0.119), same commit and seeds. B − A
   is anti-correlated with A; only a within-fire difference between resumed arms survives.
3. **N = 5,** against a within-fire arm SD of 0.12–0.33 (B − C's own SD on the anchor: 0.298 / 0.384).
4. **The control was not a control.** v1's review ran `select_arc_for_goal`: `PEACEFUL_GOAL` routes to the same
   `cradle` arc as `fire_pit`, whose narrator advances phases by turn count only, so a 12-turn "peaceful" prior spends
   turns 1–6 at the fire pit and never reaches the door (no door or button call in any of the 725 committed Exp 37/38
   rows). June's "C ≈ B" is that, not isolation.
5. **Uncontrolled channels:** unseeded sampling on the peer path; `examine` recall (#845); imagination's scene
   manifest steered by carried NAc biases; a 30 s wall-clock AUT turn that gives longer prompts fewer actions; every
   committed local fire ran on the hash-embedding fallback (`sentence_transformers_version: "not-installed"`).

## What the review changed (v1 → v2)

| v1 | Problem (lens) | v2 |
|---|---|---|
| C = resumed from `PEACEFUL_GOAL` | It is a fire-pit session (bio D1, env DNB-1) | C resumes from a fire-free arc selected by name, its prior written as its own row, and a probe refuses a C prior that touched fire or hearth (§Arms) |
| G1–G3 "B above X" on both scenarios | `deceptive_fire` should FALL; v1 would score perseveration as a pass (conf DNB-1, bio D2, env DNB-4) | Every gate signed by the scenario's declared direction; headroom read on the side the effect moves toward |
| G1 = B − C | B − C > 0 is also what perseveration, familiarity and #845 recall predict (conf DNB-2) | G1 = the cross-world interaction (OB2) |
| G2/G3 compare levels | Single-session scaffolding/prompt can decide them with zero memory effect (conf DNB-3) | G3 = carry vs carry with `llm_notes_C` (OB3); level comparisons reported |
| Zero engagement scored 0.0 | "Never approached" reads as learned avoidance or total failure (bio D3) | Engagement is its own reported outcome; fractions over engaged sessions; deceptive primary = first-contact choice |
| Arms selectable by the harness | `maxim --sim` always builds the full AUT; `maxim_all_off` as written raises on Exp 37's body (wiring D1, probe in the review) | One `--aut-stack` selector, an arm-keeps/drops table, a per-row `arm_effective` stamp the harness checks |
| B0 stamps the model | On the peer path B0 records an alias only: the AUT's model is attested, not established (wiring D2) | Leader-side attestation is a build prerequisite (OB4: wait, the strict option) |
| `resume_loaded` proves the carry | It accepts `absent` stores; failed saves are swallowed at debug (wiring D3) | Per-row carry proof: bias maps at save vs load, hard fail on mismatch, `saved_at`, `apply_decay`, for every resumed arm including `llm_notes` |
| One `_llm_unavailable` voids the fire | 4/60 rows in the heartbeat: a 300-row fire is near-certain void (env DNB-5) | Row-level failure, the whole trial re-run, every attempt recorded; > 10 % trials failing → INCOMPLETE |
| Holm over G1–G3 × scenarios | Couples the binding gate to the floors; power 0.37–0.44 (conf SF-1) | Fixed-sequence: G1 first at 0.05; G2/G3 as an intersection-union test only if G1 passes |
| Wilcoxon paired by seed | The peer backend sends no seed; 26 % of B − C pairs tie (conf SF-2, env SF-3, wiring S4) | Unpaired permutation test governs; trials are blocks run close in time; chain order randomized within trial |
| FearGatedExecutor off everywhere | It is wrapped unconditionally, no switch, no LLM call on cradle tools (conf, wiring S1) | ON in every Maxim arm (shipped behaviour); absent in the bare arms by construction; recorded |
| Model fixed at Qwen32B | C ≈ 0.80 on today's anchor: no headroom for δ 0.2 (env SF-2) | Model chosen by a pre-registered pilot (OB5) |
| B2 "dose" arm | Confounded by resume count; a rise is not the biological prediction; the term is undefined in the plans (conf, bio S5, wiring NIT) | Dropped (OB6) |
| — | The bare arms must see the environment's feedback (bio S3) | Every LLM arm gets the Body State block with its DRIVE/WARN labels; it is the world's text, not a bio section |
| — | Hash-embedding fallback, imagination steering, wall-clock turn (env DNB-2/3, SF-4, wiring S2) | `require_semantic_encoder` preflight + per-row `using_fallback` stamp; `MAXIM_DISABLE_IMAGINATION=1` on every arm; an action-bounded AUT turn, stamped, timed-out rows failed |

## Owner decisions taken

- **2026-10-14 / 2026-10-16 / 2026-10-17:** B4 homed to 1.4 (L8's design); T1-2 resolved by B4 before 1.4.0 with no
  exception; B3 runs Reflexion first.
- **OB1 (2026-10-18):** `bare_llm` uses a published ReAct format verbatim: zero-shot ReAct (Yao et al. 2023), its
  Thought/Action/Observation scaffold rendered in the shared parser's JSON tool-call format (the parser and retry
  policy are shared code; "verbatim" binds the scaffold's wording, not the wire format), with the same tool list and
  observation text as Maxim.
- **OB2 (2026-10-18):** G1 is the cross-world interaction; `B_valuation_off` is not run, so the verdict names no wire.
- **OB3:** G3 is carry vs carry, with `llm_notes_C`.
- **OB4:** wait for leader-side attestation of the served model; no exception (the strict option).
- **OB5:** the model is chosen by the pre-registered pilot.
- **OB6:** the dose arm (v1's B2) is dropped.
- **OB7:** n = 34 trials per arm.

## Arms (a new experiment id `b4_exp37_floors` with its own arm list; the June arm set and its files are untouched)

| Arm | What runs | Prior | Role |
|---|---|---|---|
| A | Full Maxim, fresh home | — | Position of the fire (L8) |
| B | Full Maxim resumed from that trial's A | A | **The claim's arm** |
| C | Full Maxim resumed from a fire-free prior | `C_prior` (fire-free arc) | Resumed, wrong content: G1's comparator |
| `bare_llm` | The B2 bare loop, ReAct (OB1) | — | BF2 LLM floor (level, G2) |
| `llm_notes` | `bare_llm` resumed with Reflexion notes written at the end of that trial's `bare_llm` row | `bare_llm` | BF2 cross-session floor |
| `llm_notes_C` | `bare_llm` resumed with notes from a bare fire-free prior | `bare_C_prior` | G3's comparator (OB3) |
| `maxim_all_off` | `--aut-stack all_off` (the selector's table; a structural probe proves no hippocampus/NAc/EC/ATL tool is registered; labelled "within-session scaffolding") | — | Scaffolding vs `bare_llm`; reported |

Priors are written as rows (`arm = C_prior`, `bare_C_prior`); the gate never infers a `depends_on`. The C priors are
built per scenario (not shared), so B's and C's save→resume gaps match.

**What crosses the session boundary, per arm** (stamped per row): hippocampus, NAc (biases, causal links,
`cluster_fear`), EC and ATL for the Maxim resumed arms; the notes file for `llm_notes*`; nothing for A, `bare_llm`
and `maxim_all_off`. The narrator is non-resumed in every arm: the cradle campaign's generative path never consumes
the resume prompt (#1226, verified by code reading in the review), and v2 adds a per-row stamp of the path taken plus
a sentinel test whose deletion probe forces the non-generative path.

## Scenarios, metrics, direction, headroom

| Scenario | Direction | Primary metric | Engagement |
|---|---|---|---|
| `fire_pit` | increase | `positive_approach_engagement_fraction`, over engaged sessions | reported: P(engaged) per arm |
| `deceptive_fire` | **decrease** (Exp 38's frozen logic, `analyze_exp37.py::compute_counter_prior`) | first-contact choice at the hearth (warm_self vs not); `warm_self_engagement_fraction` reported | reported |

`sharp_rock` is dropped (the same cradle world as `fire_pit`, zero engagement in every fire). **Headroom** (δ = 0.20):
a scenario is `NO_HEADROOM` when its comparator mean is within δ + 0.10 of the bound the effect moves toward
(≥ 0.70 for `fire_pit`, ≤ 0.30 for `deceptive_fire`). Never scored against June's numbers.

## Gates (within one fire)

Let `dir` = +1 (`fire_pit`) or −1 (`deceptive_fire`).

- **G1 (L8 + isolation; binding; OB2).** The cross-world interaction
  `I = (B − C)_fire_pit − (B − C)_deceptive > 0` (signed so learning from consequence predicts I > 0; perseveration and
  familiarity predict I ≈ 0). Both scenarios are required; a scenario at `NO_HEADROOM` makes G1 unscorable.
- **G2 (BF2 level).** `dir·(B − bare_llm) > 0` per scenario.
- **G3 (BF2 cross-session; carry vs carry, OB3 a).** `dir·[(B − C) − (llm_notes − llm_notes_C)] > 0`.
- **Testing order:** G1 at α 0.05; only if G1 passes, G2 and G3 as an intersection-union test (each at 0.05, all
  must pass). Exact two-sample permutation tests on per-trial values, one-sided; a bootstrap 95 % interval for each
  effect.
- **Reported:** B − A (labelled anti-correlated with A); A − `maxim_all_off`; `maxim_all_off` − `bare_llm`;
  `llm_notes` − `bare_llm`; the level B vs `llm_notes`; P(engaged); #845 recall line counts per row (never a gate
  covariate); the per-row carry proof.
- **Verdicts:** `FLOORS_BEATEN` (G1, G2, G3) → T1-2 may leave STALE per the ledger's rules, worded per OB2: "a
  Maxim-carried memory, by any channel, changed behaviour in the direction its consequence predicts, beyond a bare
  LLM and beyond notes' carry"; no wire is named; `ISOLATED_FLOOR_MATCHED` (G1 passes, G2 or
  G3 not) → `PARTIAL (narrow: floor matched)`; `NOT_ISOLATED` (G1 fails with headroom) → recorded as a negative;
  `NO_HEADROOM`; `INCOMPLETE`.

## Model, pilot, n, cost

- **Model by pilot (OB5, taken).** A pilot (its own data file, never pooled) runs A, C_prior and C, n = 6 trials per scenario,
  for each candidate in order (Qwen2.5-14B on the RTX 5080; Qwen2.5-32B on big-mac-mini), on the confirmatory code
  and stamp. The confirmatory model is the first whose pilot A and C means sit inside the headroom bounds on both
  scenarios. If none does: `NO_HEADROOM (pilot)`, T1-2 stays STALE with that reason, and the owner decides.
  (No Qwen32B `deceptive_fire` data exists, so the pilot is also its first measurement.)
- **n (OB7, taken).** σ = 0.30 (B − C's measured SD), δ = 0.20, one-sided 0.05, 90 % power per gate → **n = 34 trials per
  arm**; the intersection-union's power is about 0.9³ ≈ 0.73.
- **Cost.** Per trial and scenario 9 sims (A, B, C_prior, C, `bare_llm`, `llm_notes`, `bare_C_prior`, `llm_notes_C`,
  `maxim_all_off`). On Qwen2.5-14B (≈ 750 s per full sim) 34 × 2 × 9 ≈ 612 sims ≈ 128 h; on Qwen32B (≈ 1,650 s)
  ≈ 280 h; the bare arms are likely shorter (measured in the pilot). Run in blocks of whole trials with `--resume`;
  comparability is by trial block, each block stamped.
- **One serving identity per block,** refused on change (model file hash, quantization, KV type, n_ctx, endpoint,
  server build, server OS build); LAN topology pinned, OS updates and sleep held off; no second consumer of the
  server (a preflight and a per-block check).

## Row integrity

- A gated row needs: the semantic encoder (no fallback), `_llm_unavailable` = 0, no typed abort, no timed-out turn,
  `arm_effective` = its arm, the carry proof (resumed arms), the narrator-path stamp, `apply_decay` and the save→resume
  gap stamped. A failing row fails its whole trial; the trial is re-run once, mechanically, with every attempt
  recorded. More than 10 % of trials failing in a scenario → `INCOMPLETE`. Weak evidence never gates (CLAUDE.md).
- The sub-sim timeout is raised above the measured p99 (9 of 60 heartbeat sims exceeded 1,800 s).

## Build prerequisites (1.3.6, after B2/B3; each `src/` item through its own three-lens review)

1. `--aut-stack {full, all_off, bare_llm, llm_notes}` at the AUT seam of `start_simulation_mode` (after the 1.3.5
   decomposition), with the keeps/drops table and the `arm_effective` stamp.
2. A fire-free cradle arc selectable by name for the C priors.
3. Leader-side attestation of the served model (file hash, n_ctx, KV type) reaching each sim's provenance (OB4), i.e.
   B0b (#1227/#1228) plus the leader report.
4. The action-bounded AUT turn and its stamp; `require_semantic_encoder`.
5. The per-row carry proof at save and load.
6. The harness: a new experiment id with its arm list (the June `SCHEMA_VERSION` stays 1.0 and its files reproduce
   without a flag), `seed=` threaded to `--seed` (it seeds Python-side RNGs only), `MAXIM_DISABLE_IMAGINATION=1`, the
   recorded-response smoke lane, the per-block serving check.
7. The evidence entry: two pass-table kinds (unqualified and `narrow`), a `COMPLETE_RULES` entry keyed by
   (arm, scenario, trial), numeric `ts` on rows, the `b4_` data-file name, a stamped verdict from the analyzer, the
   BF2 obligation for this LLM-path row (M46's slice for T1-2), all merged on main before the first row.
8. A `docs/wiring/` entry for what crosses the session boundary (the review's proposal).

A notes store added to `RESUME_STORES` would fire T1-16's trigger (EARNED); B3 keeps notes out of it.

## Not claimed

The June fires and the cross-model map (L8); a model other than the pilot's choice; `sharp_rock`; which wire carries
the delta (OB2).

## Open decisions

None open (OB1–OB7 taken 2026-10-18). The prereg freezes after the pilot names the model and the §Build
prerequisites land; the frozen delta gets its own design-review round.
