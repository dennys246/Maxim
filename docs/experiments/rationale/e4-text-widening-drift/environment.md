# E4 text-widening drift: environment lens (design review)

Reviewer: environment lens (determinism, provenance stamps, the prereg and evidence lints, the record, exit
codes, cost). Read: `docs/experiments/DESIGN_REVIEW.md`, the draft prereg
`protocols/e4_text_widening_drift_preregistration.md`, `scripts/lint_prereg_precedes_data.py`,
`scripts/lint_evidence_gate.py` + `scripts/_evidence_records.py`, `scripts/_provenance.py`,
`scripts/lint_harness_provenance.py`, `src/maxim/utils/code_tree.py::SCOPE`, `similarity/ec.py`
(`pattern_complete_or_separate`, `_SubstrateMatrix.scan`), `similarity/encoder.py` (`_get_encoder`,
`require_semantic_encoder`, `_get_reward_overrides`), `decisions/nac.py` (`credit_node`,
`get_threshold_overrides`), `integration/memory_hub.py` (decomposer wiring), the Exp 24 script, the fixture,
issue #911, plan `engram_formation.md` §E4.

The owner decisions (one rewarded node decides; the all-nodes arm is a bound; absorption beyond baseline; prereg
on main first) do not stop the measurement answering its question from this lens. Nothing below re-opens them.

---

## DO-NOT-BUILD

### D1. The harness location puts the code under test outside the provenance scope

The prereg fixes the implementation at `docs/experiments/data/e4_text_widening_drift/e4_text_widening_drift.py`.
Three guards cannot see that path:

- **The clean flag and code digest.** `src/maxim/utils/code_tree.py::SCOPE = ("src", "scripts")` (plus the root
  `.gitignore`). `in_process_code_provenance` / `preflight_gated_record` judge "clean" and compute
  `code_tree_sha256` over that scope only. An edit to the harness after review, including an uncommitted one, reads
  CLEAN, and the record's `code_tree_sha256` does not name the harness that produced it. The harness here is
  the instrument: its walk, its crediting and its decision rule are the measurement.
- **`scripts/lint_harness_provenance.py`.** It scans `scripts/**/*.py` only, so family 3 ("names
  `docs/experiments/data` and writes records → must run the gated-record preflight") never inspects this file.
  Forgetting the preflight would pass CI.
- **The precedent the plan cites is unguarded.** The plan's "`*_cosine_check.py` pattern" is
  `docs/experiments/data/l11_slice2_cosine_check.py`. It does `sys.path.insert(0, "src")` (relative, which is the
  Exp 42b hazard), and it stamps no provenance at all. That pattern is acceptable for a top-level `.py`
  analysis instrument the lint classifies NON_GATED. It is not acceptable for a harness whose output closes #911
  and discharges release threshold T7.

**Why DO-NOT-BUILD.** "A result whose code-under-test cannot be established is not a validation" (CLAUDE.md core),
and the prereg text itself sets the location that defeats the stamp.

**Proposed prereg text** (replace the "Implementation" sentence):

> Implementation (follows in a second PR, merged before the run): `scripts/e4_text_widening_drift.py`. Its module
> docstring mirrors this document. It writes records only under `docs/experiments/data/e4_text_widening_drift/`.
> Before encoding anything it calls `scripts/_provenance.py::in_process_code_provenance(repo, maxim.__file__,
> out_path=<record>, allow_dirty=False)`, which refuses with exit 3 when the imported `maxim` is not this repo's
> `src` or when the tree is dirty. It runs from a commit on `main`, so `executed_git_hash` is an ancestor of
> main. The harness is therefore inside `code_tree_sha256`, and `lint_harness_provenance` (family 3) covers it.

---

## SHOULD-FIX

### S1. Pin the model snapshot and run offline; stamping the revision is not enough

`_get_encoder` calls `SentenceTransformer("paraphrase-mpnet-base-v2")` with no `revision` and no offline flag.
Online, the hub resolves `main` and can fetch a newer snapshot. The run would then measure a different geometry,
and it would only be stamped afterwards. `_get_encoder` also catches every `Exception` and returns `None`, which
falls back to hash embeddings. `require_semantic_encoder` is the canonical guard against that fallback.

On the dev Mac the cache holds `models--sentence-transformers--paraphrase-mpnet-base-v2/snapshots/6cc9279c672dc57f94445ef259b28a1b736fec8f`.

**Proposed text (Apparatus, encoder bullet).** The harness sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`
before importing `sentence_transformers`. It calls `maxim.similarity.encoder.require_semantic_encoder(
"paraphrase-mpnet-base-v2", context="E4")`; a `ModelLoadError` refuses with exit 4. It resolves the loaded
snapshot directory and refuses with exit 4 unless the revision is `6cc9279c672dc57f94445ef259b28a1b736fec8f`.
It stamps that revision, the sha256 of the weights file (`model.safetensors` / `pytorch_model.bin`), the
`sentence-transformers` / `transformers` / `torch` / `numpy` versions (5.4.0 / 5.5.4 / 2.10.0 / — on the dev box
today), `platform.platform()`, and `encoder.using_fallback` read after the first embed (must be `False`).

### S2. Device nondeterminism: the determinism check cannot see it, and decisions sit at thresholds

On this Mac, `torch.backends.mps.is_available()` is `True`. A `SentenceTransformer` built with no `device`
therefore runs on MPS, and production on the same box does too. MPS and CPU embeddings differ in the low bits.
The decision rule is set membership against `sims >= thresholds` in `_SubstrateMatrix.scan`, so a cosine within
about 1e-4 of 0.44 or 0.24 can flip between devices or between machines. Exp 24's drift happened in exactly this
band: low matches at 0.42–0.45.

The prereg's determinism check runs twice in one process. It reuses the module-singleton model on the same device
with the same thread count, so it is close to tautological for the encoder half. It checks only that EC and NAc
are deterministic, which they are by construction, given that node ids are canonicalised by formation order.

**Proposed text** (new "Instrument checks" bullet and new metrics):

- Stamp `str(model.device)` and `torch.get_num_threads()`.
- For every encode in every arm, record the `PatternResult.similarity` and `best_similarity` (the scan already
  returns the comparable-best margin) together with the effective threshold applied to the winning or nearest node.
- **Device check (refuse, exit 4).** Re-embed the fixture with the model moved to CPU and replay the R1 sequential
  and isolated matrix on those vectors. If any string's membership in `A(0)`, `A(0.2)` or `I(0.2)` differs from
  the default-device run, refuse with exit 4 and name the string and its margin: the verdict would be a property
  of the float path, not of the substrate. When the default device is already CPU, the check is skipped and the
  record says so.
- Report `min |cos − threshold|` over the decisive comparisons for each of `A(0)`, `A(0.2)` and `I(0.2)`.

### S3. Record what the verdict needs to be re-derived without the model

**Proposed text.** The record carries the 30 whole-string embeddings: float32, base64, plus the sha256 of the
concatenated bytes in walk order. With them, a reader (or a later fix PR) can re-derive every arm's assignment and
the verdict offline, independent of the model, the device and the library versions. Size is about 30 × 768 × 4 B,
roughly 92 KB before base64, which is acceptable. The determinism check then compares the two runs' assignment
digests and the embedding digest.

### S4. Record form, where it lives, and what discharges #911 / T7

The prereg says nothing about the record's shape. As written, the evidence gate and the weak-evidence rule
cannot judge it.

**Proposed text** (new "Record" section):

> One JSON document, `docs/experiments/data/e4_text_widening_drift/diagnosis.json`, written through
> `scripts/_provenance.py::stamp_diagnosis(report, mock=False, code_provenance=<in_process_code_provenance
> block>)`. It carries `record_kind: "diagnosis"` (never support), epoch `ts`, and `status` (`ok`, or `failed`
> with a `refusal` reason). The committed path is written only under `--write-experiment-results`
> (`_provenance.evidence_out_paths_or_exit`); otherwise the record goes to a temp directory and both paths are
> printed. A refused run writes its `failed` record too (disclosed, never evidence). **Only a `status: "ok"`,
> `mock: false` record from a clean tree at a commit on `main` closes #911 or discharges T7. A refusal (exit 3
> or 4) is a typed abort and moves nothing** (CLAUDE.md, weak evidence never gates). The data PR merges with a
> merge commit, never a squash.

Verified against the lints:

- **Prereg lint mapping is correct.** `token_of("e4_text_widening_drift_preregistration.md") == "e4"`, the prereg
  is in `protocols/` with "preregistration" in the name, and `token_of("e4_text_widening_drift") == "e4"`, so the
  directory is GOVERNED. No existing data entry has token `e4`, and `e0` / `e25` in `protocols/` are not
  preregistration-named, so they map nothing. `stamp_diagnosis` sets epoch `ts`, so the entry is timed by `ts`
  rather than the LENIENT commit fallback. The prereg has no `**Amendment` lines, so the header grammar cannot
  fail it.
- **Evidence gate.** It is diff-scoped to ledger rows, and E4 moves no ledger row ("Ledger triggers fired: none").
  If anyone ever cites this record, `judge_non_support` requires `code_provenance.executed_git_hash` on main, a
  known `code_tree_sha256`, `working_tree_dirty_src_scripts: false` and `mock: false`. Per D1, the dirty flag
  means something only when the harness is in `scripts/`. A COLLAPSE result's fix PR is judged on its own EC
  completion row re-run, not on this record.

### S5. Distinguish the refusals; exit 3 is missing

The prereg names exit 4 for both the encoder fallback and the instrument checks, and never mentions exit 3. The
house convention, used by `_provenance` and by `exp_d8_read_mutation`, `l11_real_trace_remeasure` and
`gate6_merged_gauntlet`, is: exit 3 for provenance (wrong checkout, dirty tree on a gated write), exit 4 for an
apparatus or instrument refusal.

**Proposed text.** Exit 0: a verdict (COLLAPSE or NO COLLAPSE, both `status: ok`). Exit 3: provenance refusal.
Exit 4: apparatus refusal, with the specific `refusal` string in the record. The exit-4 reasons are:
`encoder_fallback`, `model_revision_mismatch`, `fixture_sha_mismatch`, `bias_known_answer`, `positive_control`,
`determinism`, `device_sensitivity`. Any other exception exits non-zero with no `ok` record.

### S6. Pin the fixture hash in the prereg, not just in the record

`data/roy_paraphrase_pairs.json` is outside `code_tree.SCOPE`, so neither the clean flag nor the code digest
covers it. "As tracked at the merge commit" plus a stamped sha detects drift only after the fact.

**Proposed text.** "Fixture SHA-256 `9b83311986a4a17ba7815d2e389aab755d436be3bcfcc663b9ecb4fddce345a6` (last
touched by 47c47578). The harness refuses, exit 4, on any other hash."

### S7. Environment leaks that silently zero the manipulation, and a known-answer check

- With `MAXIM_NAC_REWARD_BIAS_DISABLED` set, `credit_node` returns early and `get_threshold_overrides` returns
  `{}`. If that variable leaks into the shell, every R1 bias arm equals `b = 0`. The positive control (RA) would
  catch it, but only indirectly, and as a refusal that names the wrong cause.
- `credit_node` adds `reward_bias_alpha (0.15) × reward` and clamps to 0.20. Hitting `b = 0.1` needs reward
  `2/3`. It happens to land on exactly 0.1 in IEEE doubles, but that is incidental, and the override is
  `base − b`, computed in float (`0.44 − 0.1 = 0.33999999999999997`).
- Overrides are keyed by `agent_id`. Crediting under one id and querying under another (`""` from a
  context-less percept, for example) widens nothing, silently.
- `MAXIM_CONCEPT_DECOMPOSITION=1` is how production opts into the decomposer (`memory_hub`). With a decomposer,
  one string maps to several nodes.

**Proposed text** (Apparatus):

> The harness refuses (exit 4, `bias_known_answer`) if `MAXIM_NAC_REWARD_BIAS_DISABLED` is set to a truthy
> value, or if, after crediting, `nac.reward_bias(agent_id, rewarded_node)` ≠ `b` (|Δ| > 1e-12) or the override
> for that node is absent at `b > 0`. One fixed `agent_id` is used for crediting and for the override query, and
> the realized override float is stamped per bias. `LinguisticEncoder` is built with `decomposer=None` (the
> production default; `MAXIM_CONCEPT_DECOMPOSITION` unset), and the harness stamps the MAXIM_* / HF_* / OMP_*
> environment it ran under. `MAXIM_DATA_HOME` points at a fresh temp directory, so nothing touches the shared
> `~/.maxim` (worktrees share it).

### S8. Dry runs must not land inside the gated directory

`lint_prereg_precedes_data` applies the NON_GATED markers (`dry_run` / `dryrun` / `nonfrozen`) to the TOP-LEVEL
entry name only. `data_facts` then reads every `.json`/`.jsonl` under the directory. Suppose a dry-run file is
written inside `e4_text_widening_drift/` (for example `dry_run.json`). Its earlier `ts`, or a dirty-tree stamp
without `allow_dirty`, becomes the governed entry's data time and dirty count. If it predates the prereg's merge,
the entry FAILS.

**Proposed text.** "Dry runs write to a temp directory (the default without `--write-experiment-results`) or to a
top-level `docs/experiments/data/e4_dry_run_nonfrozen/` entry, never inside `e4_text_widening_drift/`."

---

## NIT

- **N1. The Exp 24 script no longer runs.** `scripts/diagnose_roy_paraphrase_collapse.py::_encode_text` calls
  `ec.pattern_complete_or_separate(embedding=..., modality="text")` without the now-required keyword-only
  `geometry=`, which is a `TypeError` today. The prereg cites only its walk order, which is fine. Say "the walk
  order of `run_cell` (the script itself is not reused)", and require the harness to pass
  `geometry=encoder.geometry_for(emb, "text")` on every call, as production does, and to stamp the tag.
- **N2. The `e4` token is generic.** "E4" also names the 2026-05 tool-failure-hints validation in
  `grounded_language_acquisition.md`, and other plans use an E-series (E0, E2.5 protocols). There is no
  collision today: no other `e4*` data entry or prereg exists. A future `e4_…preregistration.md` or `e4_…` data
  entry would cross-govern, though. Optionally, add a sentence that the token `e4` is reserved for this
  experiment's data.
- **N3. State cost and time.** About 30 unique strings. Sequential: 3 biases × 2 arms × 30 encodes. Isolated: 3
  biases × 28 × 2 encodes. Everything runs twice, plus the CPU replay. That is under 2 minutes on a laptop CPU,
  most of it the model load. No network (offline), no LLM, $0, no rig. Saying so lets a reviewer spot a run that
  took long enough to have been downloading.
- **N4. Name the ATL.** `LinguisticEncoder.encode` requires an `atl`. Say whether it is the Exp 24 stub or a real
  in-memory `ATL` with no persistence path. Either is fine for the EC/NAc measurement, but the record should
  say which, and a real ATL must not write.
