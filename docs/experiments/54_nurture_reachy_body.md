# Exp 54 — Nurture on the robot's own body

**Status:** Phase A COMPLETE 2026-08-27 — **GRADUATE** (gate v3, all pre-registered gates). Phases B
(readout on the physical Reachy, factory path) and C (the user path) are pending, operator present.
**Pre-registration (frozen; amendment 1 pre-data):**
[protocols/exp54_nurture_reachy_body_preregistration.md](protocols/exp54_nurture_reachy_body_preregistration.md).
**Harness:** #561 (`93887e6e`: bodies `reachy_mini_infant{,_satiated}`, `--embodiment` on the
cradle-mother harness, `sweep`/`--factory`/`--gate C` on the Exp 53 readout harness,
`analyze_exp54_magnitude.py`). **Run:** Phase A `main` @ `93887e6e`, big-mac-mini, `~/exp54/phaseA`,
2026-08-26 19:10 → 2026-08-27 01:50 MDT, 36/36 runs, 0 failed, ~660 s/run, mistral-7b narrator
in-process. Roadmap 1.1.x item 15; the prerequisite for the Oasis case study
([plans/oasis_case_study_taught_orient.md](../plans/oasis_case_study_taught_orient.md)).
**Raw data (S4):** [data/54_phaseA_nursery.jsonl](data/54_phaseA_nursery.jsonl) ·
[data/54_phaseA_runs/](data/54_phaseA_runs/README.md) (per-run provenance; mother logs archived
off-repo) · [data/54_agents/](data/54_agents/) + [data/54_agents_manifest.json](data/54_agents_manifest.json)
(the ten agents Phase B/C load, SHA-256-pinned — the artifact the Oasis case study ships) ·
[data/54_magnitude.json](data/54_magnitude.json) · [data/54_targets.json](data/54_targets.json)
(the sweep: bins, strengths, declared placements) · [data/54_dry_run_nonfrozen/](data/54_dry_run_nonfrozen/README.md)
(harness verification, not a result).

## The question

Exp 52 showed a driveless infant learns to *want* to orient because being fed relieves hunger;
Exp 53b showed those files read out on the physical Reachy — but through an explicit δ map and
the infant body, because the learned keys were `tool:infant_operant_*`. Does the Exp 52 result
hold when the infant **is** a Reachy Mini — its own body component, its own four orient
affordances (0.17 / 0.50 az per step, not the infant's 0.30), its own tool names — so the files a
nursery writes are the files a user's robot reads, with nothing in between? (Phase A.) And do
those files then drive the physical robot through the production factory path with no adapter
(Phase B), and get *consulted* under the plain `bodies/reachy_mini` body a user runs (Phase C)?

## Phase A — nursery on `bodies/reachy_mini_infant` (12 seeds/arm, 48 turns) — GRADUATE

Shuffled stimulus order, relief credit, explore weight 1.5, exposure-matched (48 turns/seed all
arms), `MAXIM_SUBSTRATE_TOOL_WHITELIST=turn_left,turn_right` (substring-matches the `_big` pair:
the 4-tool repertoire, declared S6), actions-per-turn budget unset as in Exp 52 (~23 actions per
mother turn).

| arm | act1 | act2 | act3 | act4 | late (act3+4) | fed rate (act4) | credited rate (act4) |
|---|---|---|---|---|---|---|---|
| taught | 0.68 | 0.78 | 0.86 | 0.85 | **0.858** | 0.85 | 0.85 |
| satiated | 0.50 | 0.53 | 0.49 | 0.45 | **0.472** | 0.45 | 0.00 |
| no_feed | 0.52 | 0.53 | 0.51 | 0.52 | **0.514** | 0.00 | 0.00 |

Gate v3 (`analyze_cradle_mother.py --gate v3`, constants frozen 2026-08-25, unchanged):

- **LEARNED-AT-CEILING: PASS** — the S7 clause fired: taught act1 = 0.682 ≥ 0.65 makes the +0.15 rise
  criterion unattainable (learning is already visible inside act 1's 12 turns × ~23 credited actions);
  late 0.858 ≥ 0.65 and non-degrading. The teaching claim therefore rests on the two control gates,
  as the pre-registration anticipated.
- **MOTHER-TAUGHT: PASS** — taught − no_feed = 0.858 − 0.514 = **+0.344** ≥ 0.20.
- **HUNGER-NECESSARY: PASS** — taught − satiated = 0.858 − 0.472 = **+0.386** ≥ 0.20; satiated rise
  −0.028 < 0.15; satiated late 0.472 ≤ no_feed 0.514 + 0.20.
- **APPARATUS (L2): clean** — per-seed late SD 0.086 / 0.089 / 0.073 (7 / 7 / 6 distinct values); no
  phase-lock signature.
- **APPARATUS (S3): OK** — satiated credited 0 %, no negative reward, no credit without relief.

**Per-seed taught late bins (42→53):** 0.75 0.92 0.83 0.71 0.75 0.96 0.83 1.00 0.83 0.92 0.88 0.92. Exp 52's weak learner, seed 48, learns
cleanly on this body (0.83); the weakest here is seed 45 (0.71) — the Phase B exploratory agent by
the amendment-1 rule.

**Magnitude choice (reported, not gated; `analyze_exp54_magnitude.py`, late acts):** fraction of
*big* turns by |stimulus| bin —

| arm | near (0.4–0.5) | mid (0.6–0.7) | far (0.8–0.9) |
|---|---|---|---|
| taught | 0.27 (n=1986) | 0.27 (n=1972) | 0.29 (n=1921) |
| satiated | 0.50 (n=1989) | 0.50 (n=1970) | 0.50 (n=1915) |
| no_feed | 0.50 (n=1990) | 0.50 (n=1976) | 0.50 (n=1924) |

The controls sit at 0.50 (uniform over the four tools). The taught infant did **not** resolve
magnitude by distance (Exp 45c/d's big-at-far, normal-at-near): it learned a flat preference for
the *normal* step in every bin. The pre-registration allowed exactly this ("a coarse 3-bin
representation may not"); the likely mechanism is the ~23-actions-per-turn regime, in which a
run of big steps overshoots the source within a turn and the normal step is the one more often
standing when the feed lands. Not a gate; a finding to carry into the fabric/reflex-tier work.

### The learned map (the sweep, declared before Phase B)

`sweep` over each gated taught seed's loaded EC (az ∈ [−1, 1] step 0.1, fresh load per value):

| seed | ≤ −0.5 (far-left bin) | −0.4 … +0.3 (centre bin) | ≥ +0.4 (far-right bin) |
|---|---|---|---|
| 42 | `turn_left` **0.514**, `_big` 0.10 | `turn_right` **0.786** | — (no bias) |
| 43 | `turn_left_big` 0.011 | `turn_left` **0.871**, `turn_right` 0.03 | `turn_right_big` **0.471**, `turn_right` 0.12 |
| 44 | `turn_left_big` **0.835**, `turn_left` 0.09 | `turn_right_big` **0.403**, `turn_right` 0.15 | `turn_right` 0.04 |
| 45 (expl.) | `turn_left` **0.703** | `turn_right` **0.579** (bin −0.4…+0.4) | `turn_right_big` 0.10 |

Three `audio` clusters again partition the axis into three bins, but the **map is the mirror of
Exp 52/53's**: there the left bias sat on the centre bin and the right bias on the far-right bin;
here the left bias sits on the **far-left** bin and the right bias on the **centre** bin, with the
far-right bin nearly empty — for seeds 42, 44 and 45. Seed 43 learned the Exp 53 shape instead
(centre → left, far-right → right). The procedure takes the majority.

**Declared placements (frozen procedure, amendment 1 item 1):**
- gated targets **[-0.6, -0.5, 0.2, 0.3]** — left pair in the far-left bin (eligible {−0.6, −0.5} after the
  |az| ≤ 0.6 clamp; the outer neighbour −0.7 is outside the hemisphere, so the inner −0.5 is used —
  a complete two-magnitude set), right pair in the centre bin's right half (eligible {+0.1, +0.2,
  +0.3}, centroid +0.2);
- exploratory placement **[-0.2]** — the centre bin's left half (−0.4 … −0.1) is the
  predicted **wrong-way region**: a source there completes into the centre cluster, whose learned
  bias is `turn_right` for seeds 42/44 — recorded, excluded from every gate.

**Prediction on the record, before Phase B:** seeds 42 and 44 are expected to pass Gate I at all
four gated placements; seed 43's centre-left policy predicts a wrong-way `turn_left` at +0.2/+0.3
(2 of 4 placements) — if so, Gate I passes 2/3 as the gate allows, and the disagreement is the
representational finding. The far-right bin being empty on this body means a source beyond +0.4
is unmapped for two of three seeds — outside the gated set by the procedure, and a stated limit.

## Phase B — readout on the physical robot, factory path — PENDING

`run --host 10.6.0.63 --factory --body-ref bodies/reachy_mini_infant --targets
data/54_targets.json --manifest data/54_agents_manifest.json --phase 1` (Gate I) then `--phase 2
--condition primary` (explore 0, gated) and `--condition secondary` (explore 1.5, reported), Gate T
as Exp 53b with the step size being the agent's own choice. S8 pre-conditions: SDK == daemon,
`motor_control_mode == "enabled"`, `yaw_verify.py` d(head)/d(body) ≈ 1, continuous speech source
1–2 m dead ahead (audiobook), speech-gate floor 0.50 / 30 s.

## Phase C — the user path — PENDING

`run --host 10.6.0.63 --factory --body-ref bodies/reachy_mini --gate C --targets … --phase 1`
(no motion; the user's full tool space, no whitelist — amendment 1 item 9) then `verdict --gate C`.

## What this experiment does NOT claim (unchanged from the pre-registration)

Cross-unit transfer (1.2); loudness or onset salience; learning on the hardware; that the
three-bin representation is adequate (Phase B's exploratory placement and the empty far-right bin
say where it is not); anything about the LLM path; magnitude selection (measured here as *not*
resolved). n = 12/arm in the nursery, one session.
