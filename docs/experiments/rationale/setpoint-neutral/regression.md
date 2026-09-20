# Set-point-aware neutral — REGRESSION lens (four-lens design review, 2026-09-16)

Reviewed: `docs/plans/deferred/setpoint_aware_neutral.md` (DESIGN PLAN DRAFT 2026-09-15) on branch
`docs/exp60-outcome-earned`. Remit: what the change can break among results already EARNED, the
re-run bill, whether the proposed guards (opt-in, byte-identical default) are sufficient, and whether
the plan's DO-NOT-SHIP condition is actually checkable. Read-only except this file.

## Verified first

- **The encoder and its guards.** `src/maxim/similarity/encoder.py::_sensor_embed` — weight
  `w = (|v − 0.5|·2) ** gain_exponent`, `None` = plain sum; the docstring carries the D1 sentence
  ("no set-point plumbing here ... must not be improvised"). `SensorEncoderConfig` —
  `gain_exponent = 3.0`, `gain_modalities = frozenset({"world"})`, `pattern_threshold = 0.85`. No
  production site constructs `SensorEncoderConfig(...)` with non-default values (grep: zero hits
  outside `encoder.py`), so a new config field lands as a default-only knob unless the body walk
  feeds it. `SensorEncoder.encode_sensors` builds the geometry tag from exactly
  `{encoder, modality, declared_sensors, normalization, embedding_dim}` plus `gain="p3.0"` ONLY
  when gain applies; the code comment states the deferral outright: **"the RANGE VALUES / units
  bullet ... are NOT covered ... This stamp records WHETHER ranges were applied, not whether they
  were the right ranges."** The ranges come from `runtime/agent_loop.py::_read_declared_modality_ranges`,
  which reads only `schema["range"]` — neither `initial` nor a drive `set_point`.
- **The "byte-identical" pins that exist today.**
  `tests/unit/test_ec_vectorized_scan.py::TestA4GainEncoding::test_gain_none_is_byte_identical_to_pre_a4`
  re-derives the expected vector IN-PROCESS from the same `_stable_basis` and compares with
  `pytest.approx` — a formula mirror, not a golden. The two tag pins
  (`test_ungained_modality_geometry_tag_is_byte_identical_to_pre_a4`,
  `test_gained_modality_geometry_tag_moves`) compare against `encoding_geometry_tag(...)` called
  in-test, so a change that moves BOTH sides passes. Grep for a literal `"g[0-9a-f]{8}"` tag
  anywhere in `tests/`: **zero hits**. The only true golden pin in the repo is
  `tests/unit/test_decision_provenance.py::test_golden_alternation_sequence_pins_selection`
  (sequence generated from the pre-change commit, exact equality) — the pattern exists, the encoder
  has none.
- **Load-time behaviour for a tag mismatch.** `similarity/ec.py::EntorhinalCortex._note_geometry_mismatch`
  — one WARNING per (modality, stored, live) triple, nodes SKIPPED for completion, remedy
  `maxim substrate invalidate --drop-geometry` (`hivemind/merge.py::invalidate_stale_geometry_nodes`).
  The migrate half (`ec.py`, the `stamp_unstamped`-style derivation) re-derives sensor tags WITHOUT
  the `gain` field and carries the comment "**if `tag_fields` grows again, derive both from one
  helper** — a wrong migrated tag is worse than none." `hivemind/ingest.py` admission refuses
  unstamped foreign nodes and `strict_geometry=True` blocks cross-geometry folds. All of this keys on
  the TAG — a change that does not move the tag is invisible to every one of these guards.
- **The live precedent for a silent, tag-invisible encoding change.** CHANGELOG `[Unreleased]`:
  `minecraft_player` `saturation` range `[0,10]→[0,20]` (Exp 60 gate (ii)); the tag did not move
  (same declared set, same normalization mode), and the committed 2026-09-04 L11 re-measure now
  prints A4 **0.0881/1.0/0.6852** where it printed **0.0566/0.9984/0.6309**, because
  `scripts/l11_real_trace_remeasure.py::_declared_world_ranges` re-reads `minecraft_player.yaml`
  at `analyze` time (dated row in `docs/limits/l11_sensor_dilution.md`). The CHANGELOG's "no
  persisted substrate is re-encoded" is a statement about disk state on one machine, not a guard.
- **Bodies per experiment (the prompt's "one shared body" premise is only half right).**
  `scripts/exp56/common.py::BODY_REF = "bodies/minecraft_bench"` (+ `_satiated`);
  `scripts/exp57/common57.py::BODY_REF57 = "bodies/minecraft_bench57"`;
  `scripts/survival_world/exp58_run.py` and `exp60_run.py` both pass
  `entity_ref="bodies/minecraft_player"`, which is also the PRODUCTION default
  `simulation/minecraft_harness.py::MINECRAFT_BODY_REF`. So Exp 58 and Exp 60 share a body; Exp 56/57
  do not share it with them. Field census (`src/maxim/_data/components/bodies/`): `initial:` on 17 of
  `minecraft_player`'s sensors, 11 on `base_humanoid`, 12 on `reachy_mini`, 7 on `infant_humanoid`;
  drive `set_point:` on `infant_humanoid` (4), `reachy_mini` (2), `minecraft_player` (2: health,
  oxygen), `base_humanoid` (1). `infant_operant` (Exp 48/52/53's body) declares NO `range:` at all —
  its interoception channel is **range-blind** (legacy fold in `_normalize_value`).
- **Committed records that re-read the body at run time.** `l11_real_trace_remeasure.py` (above,
  and it resolves clusters through the real `encoder.encode_sensors(modality="world", ranges=...)`,
  so any encoder change flows into its printed numbers). `scripts/survival_world/exp60_run.py` reads
  `_read_world_ranges(aut.executor)` live and refuses on `fingerprint_drift(...)` against
  `FROZEN["fingerprint"]["sensor_ranges"]` for `is_in_water/oxygen/saturation` — **ranges only; a
  set-point declaration would pass the drift check**. `scripts/survival_world/l11_geometry_probe.py`
  captures live ranges into the record. The four `docs/experiments/data/exp60_*_check.py` and
  `l11_slice2_cosine_check.py` hard-code normalized values and call the SHIPPED `_sensor_embed(...,
  gain_exponent=3.0)` — they follow the default, so they are byte-stable only while the default is.
- **What the offline-replay inputs actually contain.**
  `docs/experiments/data/l11_geometry_2026-09-15.json` and `exp60_geometry_2026-09-15{,b}.json` store
  `per_sensor[{sensor, v_safe, v_dark, gain_w_*}]` — per-situation normalized MEANS (30 samples
  collapsed), not per-sample vectors. Only `l11_world_trace_2026-09-04.jsonl` (1,070 deduped raw
  snapshots with `state` dicts) supports a stability / discrimination replay; the Slice-1 and Exp 60
  records support separation only.
- **Cost evidence (first→last `ts` in the committed JSONL).** Exp 56: 51.0 min for 4 arms × n=50
  (`56_four_arm.jsonl`; `docs/plans/archive/survival_world_1_3.md` says the same). Exp 57: 91.5 min
  (`57_ladder.jsonl`, 9,200 rows). Exp 60: 13.1 min (FEAR) + 14.0 min (ABLATED), 5 seeds each
  (`exp60_trials.jsonl`, run ids `301eb2edff6d` / `eeb92752ee2b`). Exp 42: `--mock` for CI, real
  fire = `smollm-1.7b-instruct` narrator with an LLM-free AUT (`benchmark_exp42_preference.py`).
  Exp 48/52: `benchmark_cradle_mother.py`, substrate-primary, no LLM, 12 seeds/arm, `mock: False`.
  Exp 53: `exp53_cross_context_readout.py` dry-run offline; the real readout is Reachy hardware.
- **Ledger rules.** `docs/plans/behavioral_graduation_candidates.md` §Status check cadence: "If a
  trigger fires and the affected entries haven't been re-run, they go `Stale` and **block the next
  release**"; the A4 ship (2026-09-03) established the discharge shape — "TRIGGER FIRED AND
  DISCHARGED WITHOUT RE-STALE" with a dated annotation, world-only membership, tag pins and a
  decision-equivalence guard. CLAUDE.md §Working principles — divergence rule verbatim: "two
  iterations in a row whose primary criterion fails AND whose post-hoc findings each spawn new
  follow-up plans."

## Findings

### DO-NOT-BUILD

**R1. "Byte-identical default" is a promise, not a checkable claim — there is no golden pin, so the
plan's own DO-NOT-SHIP condition cannot fail.**
Evidence: `test_gain_none_is_byte_identical_to_pre_a4` mirrors the formula in-process with
`pytest.approx`; both tag pins re-derive via `encoding_geometry_tag`; zero literal tags in `tests/`.
A refactor of `_sensor_embed` that changes `_stable_basis`, the accumulation order, or the tag
canonicalisation and its test expectations together stays green. The A4 precedent got away with this
because its default (`None`) skipped the new code path entirely; a set-point that touches the
normalization or "basis mix re-centering" (plan §Approach) does not have that property.
Change to the plan: add **Build-order step 0, landed and merged BEFORE any encoder edit**: a golden
fixture generated at the pre-change commit — e.g. `tests/fixtures/encoder_golden_v1.json` holding, for
a fixed set of sensor dicts × {gain `None`, `3.0`} × {range-aware, range-blind, range-partial}, the
full 384-float vector (exact `==`, not approx) and the LITERAL geometry-tag strings for the three
shipped spaces (world gained / interoception / audio, declared sets from `minecraft_player`,
`infant_operant`, `reachy_mini_infant`); the tag half runs two-process with differing
`PYTHONHASHSEED` (the stable-hash rule); plus an **anti-vacuity arm** that monkeypatches a set-point
onto one sensor and asserts the golden FAILS (the D44 strict-red-gate lesson — a pin that cannot see
the change is not a pin). The DO-NOT-SHIP condition then reads: *"`test_encoder_golden_v1` passes
unchanged on the shipped commit with no body declaring a set-point, and its anti-vacuity arm fails."*
Regenerating the fixture is itself the ship-blocking event and needs a written justification, exactly
as the NAc golden's docstring demands.

**R2. Set-point VALUES must enter the geometry tag, or opt-in is a SILENT mismatch at load and at
fold — the guard the plan leans on (gate 1 skip-and-warn, `strict_geometry`, `invalidate`) never
fires.**
Evidence: the tag hashes the declared NAME set and normalization MODE; range VALUES are an explicit
stated deferral in `encode_sensors`. The `saturation` `[0,10]→[0,20]` change proves the hole is live:
same tag, different mapping, no warning, committed numbers silently re-printed. A per-sensor set-point
is the same class of quantity as a range endpoint (a numeric that changes `v → contribution` without
changing the basis set). Consequences if unaddressed: (a) an OLD `minecraft_player` substrate
(hippocampus/EC snapshot written under midpoint-neutral) loaded by a NEW encoder with set-points
declared completes new readings onto incomparable old nodes — silently, no `_note_geometry_mismatch`,
no `invalidate` path; (b) `hivemind/ingest.py` `strict_geometry` lets an opted-in donor's world
nodes fold into a non-opted receiver's (the Exp 56 seam); (c) the migrate-half derivation in `ec.py`
already emits a tag WITHOUT `gain` — adding a second unrepresented field makes its own warning
("derive both from one helper") a live defect.
Change to the plan: (1) when a set-point is declared, the tag gains a field (e.g.
`setpoints={name: value}`), added ONLY when declared so undeclared tags stay byte-identical — the
exact `gain` pattern; (2) the encoder and the `ec.py` migrate derivation build `tag_fields` from ONE
helper (the obligation the code already records); (3) decide explicitly whether to close the range-value
deferral in the same change (it is the same hole and the cheaper moment to close it; if not, record
why). Without (1) the "persisted-substrate migration for opted-in bodies" in build-order step 4 has
no trigger — nothing can tell an opted-in file from a legacy one. No `_format_version` bump is
needed (additive `encoder_provenance` key + additive tag field, per `docs/agents/persistence-config.md`
practice), but `record_encoder_provenance` MERGES entries, so set-points must be recorded under a
key whose merge is well-defined (last-write or a set), not folded into the `normalization` mode.

### SHOULD-FIX

**R3. The re-run bill is mis-scoped: it names the wrong experiments and omits the rows whose
triggers literally name `_sensor_embed`. Itemized:**

| Row (ledger) | Trigger text that fires | Apparatus | Cost (measured where possible) |
|---|---|---|---|
| Exp 60 drowning (EARNED 2026-09-16) | "`SensorEncoder` / EC world-modality change, `minecraft_player` sensor-range change" | LIVE Paper + bridge | ~13–14 min/arm × 2 + preflights; operator |
| Exp 56 four-arm (EARNED) | "`SensorEncoder` / EC world-modality change" | LIVE Paper, 4 arms × n=50 | **51 min** |
| Exp 57 ladder (PARTIAL) | same | LIVE Paper | **92 min** |
| Exp 53b cross-context readout (EARNED) | "`_encode_current_clusters` / **`_sensor_embed`** / EC `pattern_complete_or_separate` change (the representation is what transfers)" | dry-run OFFLINE; real = Reachy | dry-run minutes; hardware block (already queued with n=12 replication) |
| Substrate-primary / Exp 42 lineage (GRADUATE) | "SensorEncoder / EC-interoception change" | `--mock` offline; real = local narrator LLM, 10 seeds/arm × 2 arms × gating on/off | tens of minutes local |
| Exp 48 operant orienting (EARNED) | "SensorEncoder/EC-modality change" | OFFLINE, substrate-primary, 12 seeds/arm | tens of minutes |
| Exp 52 nurture (EARNED) | body/credit triggers only — fires by the ledger's blanket "Substrate-pipeline change → All Tier 1" row | OFFLINE, 12 seeds/arm | tens of minutes |
| Exp 45 hardware orient (EARNED) | NAc/transport triggers only — blanket row | Reachy hardware | hardware block |
| Exp 10 cross-session (EARNED), EC pattern completion (EARNED), affordance transfer (PARTIAL), Exp 37 (PARTIAL) | "encoder swap" — TEXT encoder by intent; fires only via the blanket row | Exp 10: mistral-7b sim; Exp 37: multi-model fires (L8 caveat) | Exp 10 ~minutes; Exp 37 hours + an un-reproducible serving env |
| L11 limit ledger (`docs/limits/l11_sensor_dilution.md` §Re-measure on) | "`_sensor_embed` change (the encoding equation)" | OFFLINE replay on the committed 09-04 trace | minutes |

The plan's "Exp 56/57 re-baseline" is the wrong pair for the body it will actually opt in: Exp 56/57
run `minecraft_bench*`, not `minecraft_player`. If step 5 (Exp 58 re-test) declares set-points on
`minecraft_player`, the row that goes Stale is **Exp 60** (same body, cheap: ~30 min live) — and Exp
56/57 stay untouched by the letter AND the spirit unless `minecraft_bench*` opt in. Conversely the
plan omits Exp 53b and Exp 42/48, whose triggers name `_sensor_embed` / SensorEncoder directly and
whose spaces (interoception, audio, ungained) must be shown untouched.
Change to the plan: replace "behavioral-graduation re-run + Exp 56/57 re-baseline" with the table
above, split into (i) rows DISCHARGED-WITHOUT-RE-STALE by R1's golden + a dated annotation (every
ungained-space row: 42/48/52/53/45/10; and 56/57 while `minecraft_bench*` is undeclared), and (ii)
rows RE-RUN because their body opts in (Exp 60 first; 56/57 only if their bodies follow). Under the
ledger's letter, every row in (i) is Stale from the merge until its annotation lands — put the
annotations in the SAME PR as the encoder change, as the A4 PR did.

**R4. Opt-in via a key on `bodies/minecraft_player.yaml` is a shared-mutable knob across Exp 58, Exp
60 and PRODUCTION (`MINECRAFT_BODY_REF`). Use a body VARIANT or a per-run override, not the shared
YAML.**
Evidence: `exp58_run.py`, `exp60_run.py` and `minecraft_harness.py` all resolve
`bodies/minecraft_player`; Exp 60's `FROZEN["fingerprint"]` pins three sensor RANGES and would not
notice a set-point. The codebase's own convention for exactly this situation is a variant body
(`benchmark_cradle_mother.py`: "the satiated arm is a BODY variant, not an env flag";
`minecraft_bench57` vs `minecraft_bench`).
Change to the plan: step 5 runs on `bodies/minecraft_player_setpoint.yaml` (or a per-run
`SensorEncoderConfig` override handed through `build_minecraft_aut`), leaving `minecraft_player`
byte-for-byte and Exp 60's fingerprint intact; when/if the production body adopts set-points, that is
a separate, dated re-stale of Exp 60 with the set-points added to its frozen fingerprint
(`fingerprint_drift` must compare them). Also add "set-point declaration" to Exp 60's `Re-run on:`
text the moment the key exists.

**R5. "or the body infers" a set-point from `initial:` (or a drive's `set_point`) is NOT a
byte-identical default — it opts in every existing body implicitly.**
Evidence: `initial:` is present on nearly every sensor of every body (17/11/12/7 above), and drive
`set_point:` exists on `infant_humanoid`, `reachy_mini`, `base_humanoid`, `minecraft_player`. Any
inference rule that reads an existing field flips the interoception/audio spaces of the EARNED
ungained rows. Additionally `infant_operant` (Exp 48/52/53) is range-BLIND — a set-point has no
defined normalized frame there (the legacy fold is bimodal), so "infer" is not even well-formed for
the bodies that carry the most EARNED weight.
Change to the plan: set-point is an explicit NEW key with no inference, read by a new walk beside
`_read_declared_modality_ranges` (lockstep invariant applies), and applied ONLY when
`modality in gain_modalities` — structurally (the `applied_gain is None` branch never consults
set-points), because "silent at rest" is a property of the GAINED sum; on the plain sum a set-point
has no meaning and would only re-stale rows for nothing. Push this into the signature, per the
silent-no-op lesson: a set-point passed for an ungained modality is a `ValueError`, not an ignored
kwarg.

**R6. The committed replay-based records must be PINNED before the primitive exists, or their printed
numbers will move a second time.**
Evidence: `l11_real_trace_remeasure.py::analyze` re-reads the body YAML and encodes through the real
encoder — it already re-printed the 09-04 verdict once. A set-point on `minecraft_player` moves it
again, and this time the delta would be attributed to the wrong cause unless the record says which
body it read. `exp60_run.py` pins ranges but not set-points (R4). The `data/*_check.py` scripts pass
`gain_exponent=P` explicitly and normalized values by hand — they survive an additive kwarg with a
`None` default, and break if the default changes (which R1 forbids).
Change to the plan: (1) `l11_real_trace_remeasure.py` stamps the full declared world ranges AND
set-points into the verdict record and REFUSES (or prints a dated delta) when the body it reads
differs from the stamp in the committed record — the same shape as Exp 60's `fingerprint_drift`;
(2) list in the plan which committed numbers are re-printable (`l11_remeasure_verdict_2026-09-04.json`
via analyze; nothing in `56_*`/`57_*`/`exp60_trials.jsonl` — those are frozen behavioural DVs) so the
"a set-point changes every replay-based record" sentence is scoped to what it actually touches.

**R7. The offline-replay step is under-specified for the bar it sets: the Slice-1 and Exp 60 geometry
records cannot yield the composite `min(sep, stab, disc)`.**
Evidence: those records store per-situation means; only the 09-04 trace (1,070 snapshots) and the
synthetic bake-off can measure stability/discrimination. The plan's step 2 says "the composite bar"
over "Slice-1 + remeasure + bake-off vectors" as if they were interchangeable.
Change to the plan: name the 09-04 trace as the stability/discrimination source (run through
`l11_real_trace_remeasure.py` with a set-point arm beside A0/A4), and add a set-point arm (A6) to
`scripts/encoding_bakeoff.py` at N=6/8/12/30/50/100 — because A4 COLLAPSED at N=6 and a set-point
variant may change the small-N story either way. State in advance what an N=6/8 result would mean:
if A6 rescues small-N stability, that is a pressure to flip interoception into `gain_modalities`,
which is a full re-stale of every interoception row (42/48/52/53) — the plan should say now whether
that door is open or closed for this line.

**R8. Divergence check: B is the FOURTH remedy aimed at "safe and dark do not separate", and the
sharpened divergence rule has already fired. The plan must answer the bird's-eye question before
building, and if the answer is "no consumer", this converts to DO-NOT-BUILD (defer).**
Evidence: the sequence on this one problem — (1) ranges re-centred so rest sits at neutral (prereg
review, 0.926→0.747 offline, still blocked live); (2) three classroom-contrast iterations (light;
depth; depth + adjacent hostile + position gap), each a new discriminator, each refused at the
cluster-distinct preflight (`exp58_survival_wants_prereg.md` §Outcome); (3) Slice-2 channel split —
rejected offline, every variant WORSE (`l11_slice2_channel_split.md`); (4) B. Each failure spawned a
new follow-up plan, which is the rule's sharpened form verbatim. Meanwhile the same contingency
(learned anticipatory avoidance from game-native pain) was EARNED by Exp 60 WITHOUT B, on a full-range
cue, and `docs/plans/roadmap_1_3.md` has re-pointed the flagship to "water = drowning". The
independent variable in Exp 58 was the classroom's contrast, not the encoder — the encoder was the
messenger.
Change to the plan: add a §Consumer that answers, with a name: *"Which want on the 1.3/1.4 roadmap
REQUIRES a small, one-sided sensor move to separate, now that full-range cues are shown to work?"*
If one exists (e.g. a specific survival cue whose sensor cannot swing across neutral by construction),
B has a caller and this line proceeds under R1–R7. If none does, B is capability without a caller
(the shipped-the-pieces lesson) and the honest placement is `docs/plans/deferred/` with that consumer
as the revival trigger; the "honest re-test of Exp 58's dark/safe vectors" is then a diagnostic
re-run of the replay, not a build. Either way the plan should stop describing itself as "the
foundational fix that the L11 line kept pointing at" — L11's ledger says A4 is the SELECTED
mitigation and the scaled threshold the retirement path; B is nominated by one lens of one rejected
slice.

### NIT

**R9.** Plan §Open design questions (3) says the change "changes ALL gained modalities (world) and
the geometry tag" while §Approach promises a byte-identical default — pick one sentence: the tag and
vectors move ONLY for a body that declares set-points.

**R10.** The `_sensor_embed` docstring's D1 sentence and `SensorEncoderConfig`'s membership comment
must change in the same commit as the primitive, and `docs/agents/bio-memory.md` needs the new
invariant with a `Regression guard:` line pointing at R1's golden (the invariant lint will otherwise
report the gap).

**R11.** The four `docs/experiments/data/exp60_*_check.py` / `l11_slice2_cosine_check.py` replays call
`_sensor_embed(..., gain_exponent=P)` positionally-by-keyword; keep the new parameter keyword-only with
a `None` default so those committed checks keep reproducing their recorded numbers.

**R12.** `record_encoder_provenance` stamps `gain_exponent` per modality; add `setpoints` beside it
so a bundle exported from an opted-in body is self-describing (the hivemind export reads provenance).

## Verdict

**FIX-THEN-BUILD — conditional on R8; DO-NOT-BUILD as written.** The plan's two guards are the right
guards, but neither is real yet: "byte-identical default" has no golden to fail (R1), and "opt-in"
produces a tag-invisible encoding change that every load/fold/invalidate guard is blind to, exactly as
the `saturation` range change already demonstrated (R2). Both are cheap to fix and must land BEFORE
the first encoder edit (R1 as build-order step 0; R2 as a design commitment). The re-run bill is
smaller than the plan fears in one direction (Exp 56/57 do not share the body; ~30 min of Exp 60 is
the real live cost of opting `minecraft_player` in) and larger in another (Exp 53b/42/48 name
`_sensor_embed`/SensorEncoder outright and need dated discharge annotations in the same PR). The
gating question is R8: this is the fourth remedy on one problem after the divergence rule fired, and
the contingency it was meant to unlock has since been EARNED without it — if the plan cannot name a
roadmap want that structurally needs a small one-sided move, the build should be deferred with that
want as its trigger.

**What I did NOT verify:** I did not run any test or replay (read-only lens); I did not read
`scripts/encoding_bakeoff.py` to confirm an A6 arm is a small addition; I did not audit
`hivemind/merge.py::ec_merge_aligned` to confirm the fold path honours a tag with an extra field
(R2 assumes it compares tags as opaque strings — verify); I did not check whether any persisted
`~/.maxim` substrate on the operator's machines currently holds `minecraft_player` world nodes (R2's
silent-completion risk is stated for the case where one exists); and the Exp 42/48/52 cost figures are
estimates from seed counts, not measured wall-clock.
