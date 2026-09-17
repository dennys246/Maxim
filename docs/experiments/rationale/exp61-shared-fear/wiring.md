# Exp 61 — WIRING lens (four-lens design review, 2026-09-16)

Reviewed: `docs/experiments/exp61_shared_fear_prereg.md` (DRAFT) against the code on this checkout
(HEAD `adcd6808`, #739). Charter: real consumers + a real credit path (D43 — a fix ships with a
CALLER), the right seams, no hand-composed shortcut that passes while the loop fails. Read first:
`docs/wiring/README.md`, `substrate-learning-channels.md`, `harness-loop-must-be-proven-live.md`,
`cosine-separation-is-directional.md`, `cluster-dilution-blocks-situation-fear.md`,
`pain-needs-declared-failure-modes.md`.

## Verified first (with citations)

**The three sites that enforce "fear does not travel" today, and what each actually does.**

- `src/maxim/hivemind/bundle.py::scrub_nac_state_for_bundle` — `scrubbed.pop("cluster_fear", None)`
  is the FIRST statement of the scrub. The scrub is called from TWO places: compose
  (`bundle.py::compose_bundle`, unconditional, AST-pinned by `tests/unit/test_hivemind_bundle.py`)
  and receipt (`src/maxim/hivemind/ingest.py::_receiver_scrub`, the V4 re-run). Un-popping it in
  the scrub therefore changes both the sender and the receiver side with one edit — a plus — but it
  also means the scrub's fear handling must be idempotent (clamp + allowlist are).
- `src/maxim/hivemind/ingest.py::_validate_nac_payload` — strips `cluster_fear` with a note. This is
  the ONLY admission gate the field has: the validator has **no top-level key allowlist** (unknown keys
  survive the `copy.deepcopy`), and its bounded-field loop enumerates exactly
  `reward_bias`, `goal_reward_bias`, `cluster_reward_bias`, `percept_valences`. Remove the strip
  without adding fear to that loop and a hostile `cluster_fear: {"a\x1fb\x1fc": -1e300}` reaches
  `nac_merge` unvalidated (the fold clamps to `[-1, 0]`, so the magnitude is caught late, but the
  key shape, the node-id charset (`_NODE_ID_CHARSET` on `parts[1]` — the `cluster_reward_bias`
  precedent) and the failure-mode vocabulary are not checked anywhere on the ingest path).
- `src/maxim/hivemind/merge.py::nac_merge` — min-folds `cluster_fear` on EXACT string keys, absent
  side read as `0.0`, clamped `[-1, 0]`. For a fresh receiver (no fear keys) `min(0.0, donor)` =
  donor: the fold itself is correct and needs no change. `nac_merge_many` overwrites only the four
  mean-folded dicts and leaves `cluster_fear` to the pairwise fold — min is associative, so N-way is
  correct by accident; nothing pins that.

**The re-key.** `merge.py::rekey_nac_state` iterates `("cluster_reward_bias", "cluster_reward_source")`
plus `inherent_bias_keys`. `cluster_fear` is untouched: a donor fear key ships as
`A\x1f<A-cluster>\x1fdrive:oxygen` and, if it survived the scrub, would be folded VERBATIM under A's
agent id. The read `src/maxim/decisions/nac.py::NAc.cluster_fear` filters `aid == agent_id`, and
`anticipatory_threat_need` goes through it — so an un-rewritten donor id reads **0.0 silently**
(no warning, no counter). The prereg's site 3 is therefore not optional: without it a shipped fear
is the D43 silent-zero in its exact original shape. `ec_merge_aligned` on an EMPTY left (fresh B):
every donor node takes the `best_id is None` branch and `id_map[nid_r] = nid_r` — identity — so the
re-keyed cluster id is the donor's own uuid, which is now a node in B's EC. Exp 56's
`cluster_reward_bias` followed exactly this path (40/50 bias-decisive first contacts). One caveat,
below (SF-5): the inner loop compares each donor node against nodes already in `merged`, which on a
fresh receiver are the EARLIER donor nodes — so identity holds only if no two donor nodes clear 0.85
against each other.

**The counters.** `merge.py::substrate_merge::_bias_count` counts `cluster_reward_bias` ONLY;
`merge.py::SubstrateMergeResult` (frozen) has `biases_rekeyed/dropped/tightened` and nothing for
fear; `ingest.py::IngestReport` mirrors that; `ingest.py::ingest_bundle` builds `journal_entry` with
the same three counts + `donor_nodes`; `src/maxim/hivemind/cli.py::_run_ingest` prints those and
nothing else. The Exp 56 harness (`scripts/exp56/common.py::ingest_bundle_into`) reads ONLY the
journal's last entry — `report.id_map` never reaches disk.

**The read path on B, end to end.** `src/maxim/runtime/agent_loop.py` substrate branch
(`propose_via_substrate`): `_encode_current_clusters` → `nac.note_active_clusters(agent_id, clusters)`
→ `embodiment.evaluate_failures()` → `nac.anticipatory_threat_need(agent_id, clusters)` →
`drives["threat"] = max(reactive, fear_need)` → `nac.recommend_action(current_drives, current_clusters)`.
`_DRIVE_TOOL_AFFINITIES["threat"]` carries both `"flee"` and `"escape"` at 0.7×; with no causal link
on either the two tie and `best_tool = max(scores, key=lambda t: (scores[t], t))` picks the
lexically LARGER name → `minecraft_player_flee` beats `minecraft_player_escape_water`; the bridge's
`case "flee"` throws when submerged (`scripts/minecraft_bridge/index.js`), the executor books a
negative link on flee, and the next proposal tick is `escape_water`. That is Exp 60's recorded
1.3–3.3 s latency shape and applies to B unchanged. `src/maxim/simulation/minecraft_harness.py::_loop_kwargs`
hands the loop an `AutonomyController(AUTONOMOUS)` and the telemetry writer (#733);
`agent_loop._substrate_tick_due` is the wake source (#732); the cadence/liveness preflights are the
harness's (`exp60_run.py::_run`). All of `harness-loop-must-be-proven-live.md` applies to B verbatim.

**Geometry on THIS checkout.** `src/maxim/similarity/encoder.py` (sensor path, `tag_fields`) hashes
`encoder/modality/declared_sensors/normalization/embedding_dim` + `gain` — **declared sensor NAMES,
not range values.** Commit `243cee9a` ("a GAINED modality's geometry tag carries the declared range
VALUES (H2, Option A)") exists on a branch and is NOT an ancestor of HEAD. The prereg's "after H2
(#740, on main)" is not true of the tree this review read. `ingest.py::_validate_ec_payload` refuses
`geometry is None` (V3) but admits ANY non-None string; `ec_merge_aligned`'s `strict_geometry` refuses
FOLDS between differing/absent tags — on an EMPTY receiver there is nothing to fold against, so a
mismatched donor node is **inserted, not refused**. `src/maxim/similarity/ec.py::_NodeMatrix.scan`
then masks out every node whose tag differs from the live tag, and `EntorhinalCortex._note_geometry_mismatch`
logs ONCE per process. Net: a pre-H2 donor into a post-H2 fresh B is a silent behavioural NULL, not a
refusal. Provenance: `ec_merge_aligned` docstring — severed at merge; a later re-export from B would
stamp A's arrays with B's `EC._encoder_provenance` (same encoder here, so harmless; stated).

**The dangling-half path.** `cli.py::_run_export` reads `aut_nac.json`/`aut_ec.json` independently
and composes whichever exist (`compose_bundle(ec_substrate_nodes=None)` skips the `ec` slice);
`exp56/common.py::export_bundle(dangling=True)` copies only `aut_nac.json` and calls the real CLI.
`ingest_bundle` with no `ec` slice runs `substrate_merge(donor_ec={})` → `id_map == {}` → every
re-keyable key drops. Post-1.2 this path is unchanged and Exp 56 ran it (`run_campaign.py::run_pair_arm`
asserts `biases_rekeyed == 0 and biases_dropped == shipped`). A fear re-key through the same
`id_map` follows the same accounting — once the counters exist (DNB-1).

**Receiver persistence and reboot.** `minecraft_harness.py::build_minecraft_aut` →
`runtime/bio_stack.py::build_bio_stack(persistence_dir)` loads `nac.json` (`load_safe`, decay-on-load
against `saved_at`) and `ec.json` when present; `cli.py::_resolve_receiver_pair` falls back to that
exact `nac.json`/`ec.json` pair when `aut_*` are absent, and `_run_ingest --apply` writes them back
(`with_format_version(report.nac)`; `_validate_nac_payload` drops the DONOR `saved_at`, so B keeps its
own decay clock — `tests/unit/test_hivemind_ingest.py::test_donor_saved_at_dropped_so_decay_clock_survives`).
Fear wall-decays in the 7-day class (`NAc.apply_wall_clock_decay`), so a minutes-long ingest→reboot
gap is negligible. The pair is persisted by the HUB close (`exp56/common.py::close_and_stage_session`
docstring: "the HUB close persists the NAc/EC pair"), not by `bio.on_session_end()` alone.

**Tests that pin "fear does NOT travel."** Exactly one: `tests/unit/test_cluster_fear.py::TestFearHivemindPosture::test_bundle_scrub_excludes_fear`
(`assert "cluster_fear" not in scrubbed`). The two siblings in that class (`test_merge_preserves_receiver_fear_and_commutes`,
`test_merge_clamps_malformed_fear`) pin properties that survive the change. No test asserts the
ingest strip note. The brief `docs/agents/bio-memory.md` Wire-4 invariant (c) states the posture in
prose and is audited by `scripts/lint_claude_md_invariants.py`.

**Other consumers grepped.** `hive contribute` → `store.py::accept_contribution` validates the
MANIFEST only (`read_bundle_manifest_bytes`) and stores bytes; promotion re-runs `ingest_bundle` —
so there is no separate scrub/validate site on the Oasis side. The manifest (`compose_bundle`) has
no per-field NAc roster or counts — only `contents` (slice names) and `observed_embedding_dims` —
so no manifest change is needed. `inherent_bias_keys` name `cluster_reward_bias` keys only; no
interaction with fear. `merge.py::tighten_negative_biases::_TIGHTEN_ONLY_FIELDS` excludes
`cluster_fear` (moot: MIN is already tighten-only). `merge.py::prune_nac_cluster_biases` prunes the
three cluster-keyed surfaces and NOT `cluster_fear`.

## Findings

### DO-NOT-BUILD (as drafted — blocks the mechanism PR until folded)

**DNB-1. The four-site list is incomplete in the one place the design's falsifier depends on: the
fear counters have no carrier from the merge to the harness.** Evidence: `substrate_merge::_bias_count`
counts `cluster_reward_bias` only; `SubstrateMergeResult`, `IngestReport`, `ingest_bundle`'s
`journal_entry`, and `_run_ingest`'s print all carry exactly `biases_rekeyed/dropped/tightened`;
the harness precedent (`exp56/common.py::ingest_bundle_into`) reads the JOURNAL. A PR that changes
scrub + validate + rekey + merge and stops there ships fear WITHOUT `fear_rekeyed`/`fear_dropped`
anywhere a caller can read them — the prereg's arm-2 gate (`fear_rekeyed == 1`), the BOTH-HALVES
gate (`fear_rekeyed == 0`), and the "refuse the pair on `fear_dropped > 0`" stop rule would all read
a MISSING key. That is D43's exact shape (correct pieces, no composition, green tests). Concrete
change — the mechanism PR's site list becomes NINE, moved together:
1. `bundle.py::scrub_nac_state_for_bundle` — stop popping; clamp each value to `[-1, 0]`, keep only
   triple keys whose third part is in the fear allowlist, key format unchanged (`aid\x1fcid\x1ffm`).
2. `ingest.py::_validate_nac_payload` — delete the strip; add `("cluster_fear", -1.0, 0.0, 3)` to the
   bounded-field loop (this also gives it the `MAX_NODES_PER_SLICE` cap and `_check_key_shape`), apply
   the `_NODE_ID_CHARSET` check to `parts[1]` exactly as for `cluster_reward_bias`, and REFUSE
   (`IngestRefused(duty="V2")`) a `parts[2]` outside the allowlist — refusal, not strip, is the
   house posture for a privilege-shaped field (the `inherent` precedent in `ingest_bundle` step 8).
3. `merge.py::rekey_nac_state` — add `"cluster_fear"` to the re-keyed fields (same drop rule).
4. `merge.py::substrate_merge` — count fear before/after the re-key (a second `_count(field)`), and
5. `merge.py::SubstrateMergeResult` — `fear_rekeyed: int = 0`, `fear_dropped: int = 0` (defaults keep
   `exp56/common.py::noop_variant_readout`'s hand-built results constructing).
6. `ingest.py::IngestReport` — the same two fields; `ingest_bundle` fills them and writes them into
   `journal_entry` (the durable surface the harness reads).
7. `cli.py::_run_ingest` — one printed line beside `biases rekeyed`.
8. `merge.py::prune_nac_cluster_biases` — prune `cluster_fear` on the pruned ids (SF-4 below; "all
   sites move together" is the brief's own rule).
9. `nac_merge` — unchanged, but a test must pin that `nac_merge_many` preserves the pairwise min for
   fear (it does today only because min is associative and the N-way overwrite skips the field).
Plus the docs that roster the bounded fields (`docs/user/hivemind_bundle_format.md` V2 bounds,
`docs/plans/oasis_ingestion_contract.md`) and the brief's invariant (c) rewritten in the same PR.

**DNB-2. "Receiver preflights (Exp 60's, unchanged)" contradicts the claim "B has never been
underwater" and, run on B post-ingest, exercises the mechanism BEFORE the DV.** Evidence:
`exp60_run.py::_run`'s preflight sequence submerges the bot twice — the LIVE cluster-distinct check
(`_submerge("preflight")` + `_encode_current_clusters`) and the escape-actuation check (bridge
`escape_water` from the pool floor, up to `SURFACE_WITHIN_S + 2` s in water, against a measured
air-hunger pain edge of ~5 s). On an INGESTED B the first pattern-completes B's live reading into
the transferred node (the "which node did B complete into" mechanism DV, consumed as a preflight),
and the second can publish `drive:oxygen` pain on B while the transferred water cluster is noted —
`create_pain_cluster_fear_subscriber` then books B's OWN fear on that node, and the first-contact
placement measures fear B learned in the preflight, not fear that arrived in the bundle. Run the same
preflights on B PRE-ingest and B's EC acquires its own shore/water nodes, so `ec_merge_aligned` FOLDS
A's water node into B's (cos ≥ 0.85, same tag) — the fear still lands (re-keyed onto B's node), but
the transferred-node DV and Exp 56's S3 independence check (`run_pair_arm`: receiver holds zero donor
ids pre-ingest) both change meaning. Concrete change — write B's lifecycle into the prereg as a
numbered sequence and split the preflights by subject:
- **Throwaway agent (fresh persistence, discarded):** cluster-distinct live check, escape-actuation
  check, gamerules, bridge roster, cadence. These are apparatus properties; they do not need to run
  on B.
- **B, pre-ingest:** `build_minecraft_aut` → NO loop, NO submersion → full close (hub close +
  `bio.on_session_end()`, Exp 56's `close_and_stage_session` shape) so `nac.json`/`ec.json` exist
  for `_resolve_receiver_pair`; assert B's `ec.json` has ZERO world nodes (fresh by construction).
- **Ingest** via the real CLI with `--receiver-agent-id == B's agent_id` (the SAME id the reboot
  uses — `cluster_fear()` filters on it); read the journal entry; gate on the counters.
- **B, post-reboot, shore only:** loop liveness on the shore (`FROZEN["loop_liveness_min_ticks"]`),
  `get_positive_outcomes(escape_water) == []` and `== []` for flee, ingest-report record present;
  record B's live shore tag (from B's shore node in `EC._substrate_node_geometries`) and assert it
  equals every transferred world node's tag (SF-2).
- **First contact** = the FIRST teleport into water B ever receives, with the loop live.
Then the "B has never been underwater" claim is true by construction rather than by omission.

### SHOULD-FIX

**SF-1. `fear_rekeyed == 1` (and donor sanity's "exactly one `cluster_fear` entry") is the wrong
gate: Exp 60's own record allows a jitter-split donor.** Evidence: `exp60_run.py::_run` records
`distinct_episode_clusters` and the LIVE G2 gate requires the need to clear the floor on EVERY
distinct episode cluster ("a jitter-split minority id must not be dead at recall") — so a valid
FEAR donor may carry 2 fear keys, both at −1.0, both on water nodes. Change: stamp the donor's shipped
fear count into `donor_meta.json` (Exp 56's `bias_entries` precedent) and gate arm 2 on
`fear_rekeyed == shipped >= 1 and fear_dropped == 0`, arm 4 on `fear_rekeyed == 0 and fear_dropped == shipped`.
Donor sanity: "≥ 1 fear key, ALL on world nodes noted during training, all at −1.0, none on the
shore node" — not "exactly one".

**SF-2. Geometry equality must be asserted by the harness, on the tag STRINGS, on both sides —
`strict_geometry` cannot refuse an insertion into an empty receiver, and this checkout's tag does not
hash ranges.** Evidence: the "Geometry on THIS checkout" paragraph above. Change: (a) donor sanity
records the set of world-node tags from the donor's `ec.json` (must be a single tag);
(b) post-reboot B records its live tag from its shore encode; (c) refuse the pair if they differ, and
stamp both into the record; (d) the prereg's "H2 (#740) on main" sentence must be re-verified against
`git merge-base` on the run-day commit — if H2 has not merged by then, donor and receiver still share
a tag (same names, same gain), so the experiment is unaffected, but the prereg must not cite a
protection that is not on the tree. Also record `report.id_map`'s identity property (SF-5) here.

**SF-3. Post-ingest key normalization must be asserted for `cluster_fear`, and its ABSENCE must be
proven load-bearing in the red gate.** Evidence: `NAc.cluster_fear` filters `aid == agent_id`
silently; Exp 56 asserts this for `cluster_reward_bias` post-ingest (`run_pair_arm`: every key's
first part equals `recv_id`). Change: extend that assertion to every `cluster_fear` key in B's
`nac.json`; and in the mechanism PR's tests include one arm that ingests WITHOUT
`receiver_agent_id` and asserts `anticipatory_threat_need(B, …) == 0.0` — the test that shows WHY
the rewrite is in the path, not just that it is.

**SF-4. `prune_nac_cluster_biases` leaves fear dangling after `maxim substrate invalidate`.**
Evidence: it prunes `cluster_reward_bias`, `cluster_reward_source`, `reward_bias`, `inherent_bias_keys`
— not `cluster_fear`. Today harmless (fear never enters from outside); once fear travels, a receiver
that later drops a geometry keeps fear keys naming nodes its EC will never emit (D2's dangling shape,
`cosine-separation-is-directional.md` corollary 4). Change: add the field in the same PR, and extend
`tests/unit/test_d44_merge_behavioural_delta.py::test_every_surviving_bias_key_names_a_reachable_cluster`'s
shape to fear.

**SF-5. `id_map` identity on a fresh receiver is an assumption, not a property — assert it.**
Evidence: `ec_merge_aligned` scans `merged`, which on an empty left is populated by EARLIER donor
nodes as the loop proceeds; two donor world nodes at cos ≥ 0.85 (a jitter-split pair CAN sit near the
threshold — Exp 60 measured early/late dive at "same cluster") would fold, and the second node's fear
would be re-keyed onto the first. Not wrong for the claim (fear still lands on a water node), but it
changes the "which node" DV and the count in SF-1. The journal does not carry `id_map`. Change: the
harness compares the donor's `ec.json` ids with B's post-ingest `ec.json` ids (donor ⊆ post, count
unchanged) — a disk-level identity check that needs no CLI change.

**SF-6. The red gate for the composition, and the deliberate flip.** The prereg's build order names
tests but not the gate. Concrete: a new `tests/unit/test_exp61_fear_transport.py` in the
`test_d44_merge_behavioural_delta.py` shape — two `Agent`s (own `SensorEncoder` + `EntorhinalCortex`
+ `NAc`, distinct ids); A encodes a "water" reading, `record_cluster_fear(A, node, "drive:oxygen", 1.0)`
×2 → −1.0; **real** `compose_bundle` → **real** `ingest_bundle` (tmp `IngestionJournal`,
`receiver_agent_id=B`) → `NAc.load_state(report.nac)` + `EC.load_substrate_nodes(report.ec_nodes)`
on B → B encodes the same reading → `anticipatory_threat_need(B, clusters) >= 0.5` and
`recommend_action(..., current_drives={"threat": need})` picks a flee/escape affordance. **RED today
at the read (0.0) because the scrub pops the field.** Land it `xfail(strict=True)` before the
mechanism PR and remove the marker IN the mechanism PR (never re-point it). Arms alongside: ABLATED
donor → 0.0; dangling half (`ec_substrate_nodes=None`) → `fear_dropped == shipped`, `fear_rekeyed == 0`,
read 0.0; no-`receiver_agent_id` → 0.0 (SF-3); anti-vacuity `receiver_unchanged`/`empty_state`
(`exp56/common.py::noop_variant_readout`'s kit) collapse. The ONE existing pin to flip on purpose:
`test_cluster_fear.py::TestFearHivemindPosture::test_bundle_scrub_excludes_fear` → becomes "fear
ships clamped and allowlisted; an out-of-allowlist mode and a positive value do not". The brief's
invariant (c) and its regression-guard line are rewritten in the same commit as the flip (the guard
test lands with the fix — the review-round rule). Drop the prereg's "two-process key stability"
test: the keys are `\x1f`-joined strings, not hashes; there is nothing PYTHONHASHSEED can touch.

**SF-7. Harness reuse: what is liftable, what is not, and what the lints require.** Evidence,
`scripts/survival_world/exp60_run.py`: **pure, module-level, already unit-tested, liftable as-is** —
`classify_placement`, `p_surface`, `exact_permutation_p`, `select_run`, `compute_verdict`,
`fingerprint_drift`, `median_interval_s`, `min_pain_edge_s`, `_telemetry_ticks`,
`_detach_fear_subscriber`, `_f`, plus `FROZEN`. **Entangled** — every preflight, the rescue/submerge
primitives, `_loop_window`, `_probe`, the training loop, the LIVE G2 block, the pain subscriber and the
executor spy are CLOSURES inside `_run` over `aut`, `rcon`, `args.username`, `shore`/`sub`, `signals`,
`calls`, `persistence_dir`, `agent_id`, `encoder`, `probe_cap_s`, `train_cap_s` (≈700 lines). Lifting
them means a seed-context object (an explicit `WaterTrial` class holding those fields, methods for
`rescue/submerge/loop_window/probe/train/live_g2/preflights`), NOT a copy: a second 700-line `_run`
is the "hand-composed second builder" shape `build_minecraft_aut`'s docstring warns against.
Constraints: `scripts/lint_function_length.py::_BASELINES` pins only three `src/` functions, so
nothing ratchets `_run` — the discipline is on the author; `scripts/lint_harness_provenance.py`
Family 1 (`assert_repo_interpreter`) fires only on `subprocess` spawners, and the Exp 61 harness
calls `run_substrate_subcommand` IN-PROCESS (Exp 56's pattern), so the sanctioned guard is Family 3's
`in_process_code_provenance(REPO_ROOT, maxim.__file__, out_path=…)` + `evidence_out_paths_or_exit`
(exactly what `exp60_run.py::_run` does) — and it answers the same question (which `maxim` the CLI
imported). Two obligations the refactor adds: (a) `tests/unit/test_exp60_run.py` imports the pure
functions from `exp60_run` — keep re-exports so those pins stay green; (b) Exp 60 is EARNED on
`docs/experiments/data/exp60_trials.jsonl` — run `exp60_run.py verdict` before and after the
extraction and diff the JSON, so the refactor is proven not to move a shipped verdict.

### NIT

**N-1.** `tighten_negative_biases` excludes `cluster_fear` — correct (MIN is tighten-only by
construction) but undocumented; add a one-line comment and a test that `biases_tightened == 0` on a
fear-only fold (protects Exp 56's sign-scope guarantee from a future "complete the roster" edit).

**N-2.** The fear allowlist would exist in two places after step 2 (`NACConfig.cluster_fear_failure_modes`
and the ingest validator). Invariant (b) says the allowlist lives INSIDE `record_cluster_fear`; the
hivemind layer avoids NAc imports by convention (`merge.py::_cosine` comment). Pin equality with a
test in the `test_hivemind_frozen_modalities_match_ec_default` shape rather than importing across.

**N-3.** `docs/plans/sharing_threat_model.md` gains a row: a trusted contributor can now ship
bounded, allowlisted fear onto any cluster the receiver aligns to, and MIN means no later ingest can
lift it (extinction stays local re-learning). The V1 door is the gate; say so.

**N-4.** The prereg's Trap 1 says training is propose-only so A has "ZERO positive `escape_water`
links" — true, and `exp60_run.py` asserts it after the actuation preflight (`get_positive_outcomes`).
For Exp 61 the actuation preflight must NOT run on A either (it runs through the bridge, not the
executor, so it books nothing — verified in `_bridge_escape` — but keep the post-preflight zero-link
assert on whichever agent ran it).

**N-5.** `journal_entry["donor_nodes"]` is `len(donor_ec)`; add `receiver_nodes_before` so the
fresh-receiver precondition (0) is in the durable record, not only in the harness's memory.

## Verdict

**FIX-THEN-BUILD.** The seams are the right ones — scrub, validate, re-key, min-fold, and an
unchanged read path that already works for the sibling field — and the fresh-receiver identity map
plus the agent-id rewrite give the fear a real route to `anticipatory_threat_need`. But the draft's
four-site list omits the five carriers (`substrate_merge` counting, `SubstrateMergeResult`,
`IngestReport`, the journal entry, the CLI print) that make `fear_rekeyed`/`fear_dropped` readable by
any caller, so the design's own falsifier and stop rules would read a missing key — the
shipped-the-pieces shape, one field over (DNB-1); and "Exp 60's preflights, unchanged" submerges B
twice before its "first" submersion and can book B's own fear on the transferred node (DNB-2).
Both are foldable on paper: enumerate the nine sites, write B's lifecycle with the preflights split
by subject, gate on `fear_rekeyed == shipped`, assert tag equality and key normalization in the
harness, and land the composition red gate `strict=True` before the mechanism PR. Then build.

## What I did NOT verify

- I did not run any code. No live bridge, no sim, no test execution; every claim is from reading
  the cited symbols on HEAD `adcd6808`.
- Whether a zero-node `EntorhinalCortex.save()` writes an `ec.json` that `_resolve_receiver_pair`
  accepts (Exp 56's fresh receiver used the bench body, which may have minted interoception nodes
  at build); the harness should assert the pair exists after B's pre-ingest close.
- Whether `survival_world.common.make_fresh_encoder` binds to the REBOOTED B's EC (it takes `aut`,
  so it should) and whether B's first shore encode after reboot lands in the transferred node set or
  mints a new shore node (either is fine for the claim; only the tag assertion in SF-2 depends on it).
- The actual cosine between a donor's two jitter-split water nodes (SF-5) — measurable offline from
  an Exp 60 donor `ec.json` with the existing `docs/experiments/data/*_cosine_check.py` pattern
  before building.
- Whether H2 (`243cee9a`) will be on `main` by run day; SF-2's tag-string assertion makes the
  harness correct either way.
- The `flee`-first tie-break's cost inside a 4.3 s window on B specifically (Exp 60's latencies were
  measured with fear the agent learned itself; the same read path, so the same expectation, but not
  re-measured here).
