# E4 text-widening drift pre-registration — does reward widening pull foreign strings into a rewarded text node? (#911)

**Frozen 2026-10-05, merged to main BEFORE any data.** Owner decisions 2026-10-05: one rewarded node decides and the
all-nodes arm is a bound; the absorption-beyond-baseline rule; this file on `main` before the harness runs; after
the four-lens design review ([rationale/e4-text-widening-drift/](../rationale/e4-text-widening-drift/)), fold every
finding and measure as a **latent-hazard upper bound**. Implementation (a second PR, merged to `main` before the
run): `scripts/e4_text_widening_drift.py`. Its module docstring mirrors this document; on any divergence THIS file
is the authority, and a change to either after first data needs an amendment header here. The token `e4` is
reserved for this measurement. Plan: [engram_formation.md](../../plans/engram_formation.md) E4.

## Question

NAc reward widens EC recognition on the `LinguisticEncoder` surface: `encode()` passes
`NAc.get_threshold_overrides(agent_id, base_threshold=ec.config.pattern_complete_threshold)`
(`threshold = base − reward_bias`, clamped to `[0.1, base]`; `max_reward_bias` 0.20), so a node rewarded to the
cap accepts matches down to 0.24 against the 0.44 text default. Text nodes keep a running-mean centroid (the exact
unnormalised `(s·n + e)/(n + 1)`), and every admitted match is averaged into it: the recipe the centroid-drift
lesson ([ec-centroid-drift.md](../../lessons/ec-centroid-drift.md)) warns about. **Does a rewarded text node absorb
strings of other concepts in a sequential stream because its centroid drifted, beyond what widening alone
admits?**

**This is a latent hazard today, measured as an upper bound.** No live path gives a text node a positive bias. The
only writer of `_reward_bias` is `TemporalCreditDistributor.distribute` → `NAc.credit_node`. Every reaction
published in production is negative (the one positive emitter, `CerebellumModulator`, is Dormant since
2026-05-26), and `credit_node` clamps at 0. The committed O19 substrate run (`rerun_exp09_o19/20261002_230206/`)
logged 6 credit distributions (each over 34 nodes), all negative, and ended with `reward_bias == {}`. The percept
text path runs only under `MAXIM_SUBSTRATE_PATH=1`, and a text node is credited only when the reaction's
`agent_id` matches the percept's (many percept sites leave it unset, so eligibility lands under `""`; not
exhaustively traced). The measurement asks what happens the day a positive text-credit producer is wired.

**Scope.** Measured: `LinguisticEncoder.encode(percept)` on NARRATIVE text percepts (substrate modality `"text"`)
with decomposition off, the default (`MAXIM_CONCEPT_DECOMPOSITION` unset). **Not measured**, though they widen by
the same mechanism: `"vision"`-modality percepts (also running-mean) and affordance-name chunks through
`encode_decomposed` (which share the `"text"` matrix). Sensor engrams never widen (recorded as an input to the
Rung B keying design).

## Apparatus (all in-process; no LLM, no hardware, no rig; seconds to minutes on CPU, no cost)

- **The production path, not a composition.** Every string is a `make_text_percept(text, agent_id=AGENT)` encoded
  by `LinguisticEncoder(ec=ec, atl=atl, nac=nac, decomposer=None).encode(percept)`, the construction
  `MemoryHub._wire_substrate_encoder` uses. The harness never calls `ec.pattern_complete_or_separate` or
  `register_substrate_node` itself and imports nothing from `scripts/diagnose_roy_paraphrase_collapse.py` (it
  predates the required `geometry` argument and no longer runs); it re-derives the walk order from the fixture.
  Centroids and member counts are read through `ec.substrate_node_metadata(node_id)`.
- **Stores.** Default `ECConfig` (text threshold **0.44**, default `frozen_centroid_modalities`, which leaves
  `"text"` running-mean), default `NACConfig` (`max_reward_bias` 0.20, `reward_bias_alpha` 0.15), a real in-memory
  `ATL`. Every arm and every isolated trial builds a fresh EC, NAc, ATL **and encoder**.
- **Reward.** `nac.credit_node(AGENT, node_id, b / reward_bias_alpha)` with the same `AGENT` as every percept and
  every override query. No decay runs during a stream (production decays bias with τ ≈ 50 ticks; holding it is the
  upper bound).
- **Environment.** `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set before `maxim` or
  `sentence_transformers` is imported; `MAXIM_DATA_HOME` points at a fresh temp directory (nothing touches the
  shared `~/.maxim`). The harness refuses (exit 4) if any `MAXIM_*` variable other than `MAXIM_DATA_HOME` is set,
  and stamps the `MAXIM_*` / `HF_*` / `OMP_*` environment it ran under.
- **Encoder.** `paraphrase-mpnet-base-v2`, snapshot `6cc9279c672dc57f94445ef259b28a1b736fec8f`, loaded through the
  production `require_semantic_encoder("paraphrase-mpnet-base-v2", context="E4")`; the harness then moves the
  shared `_get_encoder` singleton to CPU (`.to("cpu")`), the only deviation from the production load path, and
  recorded (production runs on MPS on Apple hosts). It refuses (exit 4) if the loaded revision is not that
  snapshot, or if any EC's `encoder_provenance["linguistic"]["using_fallback"]` is not `False`. It stamps the
  `sentence-transformers`, `torch` and `transformers` versions, the loaded revision and `torch.get_num_threads()`.
- **Wiring assertions (exit 4, no verdict, on any failure).** After each credit, `nac.reward_bias(AGENT, node_id)`
  equals `b` within 1e-12. On the first encode after a credit, a spy on `ec.pattern_complete_or_separate` records:
  in R1, `threshold_override == {rewarded_id: base − b}`; in RA, `{n: base − b for every node credited so far}`
  (each value within 1e-12); `None` at `b = 0` in both. It also records `geometry ==
  encoder.geometry_for(embedding, "text")`. The realised override float is stamped per bias.
- **Fixture.** `data/roy_paraphrase_pairs.json`, SHA-256
  `9b83311986a4a17ba7815d2e389aab755d436be3bcfcc663b9ecb4fddce345a6`; the harness refuses (exit 4) any other hash.
  10 paraphrase pairs, 5 distractor pairs (Exp 24's fixture). **Walk order**: unique strings in first-seen order,
  every pair's `a` then `b`, pairs before distractors: **22 unique strings, 20 outside pair_01.**
- **Concepts.** A string's concept is the fixture **class** of the first fixture entry containing it in walk order
  (so a string that also appears in a distractor keeps its pair's class). `pair_01_food_detect`'s class is
  `food_overlap_2pc`, shared with pairs 02–04. **Foreign** means a different class; the two distractor-only strings
  ("two people are arguing in the next room.", "the room grows quiet.") are foreign. A within-class absorption is
  reported as within-concept generalization, never as COLLAPSE.
- **Rewarded node (arm R1).** The node `"you sense food nearby."` (pair_01 `a`, first in the walk) forms, credited
  to `b` immediately after it forms. One walk order, one node position (first formed, no competitor), and a
  reward that does not rise during the stream are what is tested.
- **Provenance.** The harness lives in `scripts/` (inside `code_tree_sha256` and `lint_harness_provenance`'s family
  3), calls `scripts/_provenance.py::preflight_gated_record_or_exit(repo, out)` (exit 3 on a dirty tree) and
  `in_process_code_provenance(...)` before encoding anything, and runs from a commit on `main`.

## Arms

For each bias `b ∈ {0.0, 0.1, 0.2}`:

- **R1 SEQUENTIAL (decides).** One EC over the whole walk; only the pair_01 `a` node carries reward.
- **R1 REPLAY-ISOLATED (decides).** For each string `s` outside pair_01, the **reference** is what the EC's centroid
  rule for the arm's own config would hold, given only the strings in the rewarded node in **R1 SEQUENTIAL at
  b = 0 (same config)** that precede `s` in the walk: their running mean for a running-mean modality, the seed
  embedding for a frozen one. It always holds at least the seed, which is first in the walk. `s ∈ I(b)` iff
  `cos(s, reference) ≥` the realised override at `b`. This is widening without reward-driven drift.
- **R1 BARE-ISOLATED (reported; decides only the drift positive control).** A fresh stack per `s`: encode pair_01
  `a`, credit to `b`, encode `s`; record whether `s` completes into the rewarded node.
- **RA SEQUENTIAL (bound; never decides).** Every node is credited to `b` as it forms.

**Instrument checks (refuse, exit 4, no verdict, if any fails):**
- **Replay known answer.** At `b = 0`, for every `s`, `cos(s, replay reference)` equals the EC's recorded cosine of
  `s` to the rewarded node's centroid at encode time within 1e-9 (at `b = 0` they are the same quantity).
- **Drift positive control.** R1 at `b = 0` with the text threshold set to **0.40** (Exp 24's collapsing setting)
  must show at least one foreign string in the sequential rewarded node that the bare-isolated test at 0.40 (seed
  embedding alone) does not admit: Exp 24's drift signature, so the sequential meter can see drift.
- **Negative control.** R1 at `b = 0.2` with `"text"` added to `frozen_centroid_modalities` (both its sequential
  and its replay arm under that config) must yield no COLLAPSE string: under a frozen centroid the replay
  reference is the seed, exactly what the EC compares against, so COLLAPSE is impossible by construction.
- **Identity.** R1 and RA at `b = 0` produce identical assignments (near-tautological, since `credit_node(…, 0)`
  stores nothing; kept as a bookkeeping check, not counted as instrument evidence).
- **Determinism.** The whole matrix runs twice in one process with freshly built stores; the assignment digests,
  every recorded cosine and the embedding digest must be identical.

## Metrics (reported for every arm and bias)

- **Assignment** of every string, with nodes canonicalised by their first-formed member.
- **Per string, at encode time**: cosine to the rewarded node's centroid, that node's effective threshold, the
  winning node and its cosine, and the rewarded node's member count.
- **`E(b)`, eligible but lost**: strings that clear the rewarded node's threshold but complete into a closer node.
- **Rewarded-node absorption** `A(b)`, broken down by class; `A(0.1) ⊆ A(0.2)` reported.
- **Pair purity and distractor collapse**, as deltas from `b = 0`.
- **Rewarded centroid drift**: `drift(b) − drift(0)` (cosine of the seed embedding to the final centroid), plus the
  replay centroid's drift.
- **Pre-EC cosine** of every string to the seed embedding (explains every `I` membership).
- The stored `reward_bias` and the override value actually passed.
- **For each COLLAPSE string, all three clause margins**: `cos_seq − (base − 0.2)` for `∈ A(0.2)`;
  `(base − 0.2) − cos_replay` for `∉ I(0.2)`; and for `∉ A(0.0)`, `0.44 − cos` (or the winning node's lead).

## Decision rule (frozen)

`A(b)`: strings of a **foreign** class in the rewarded node in R1 SEQUENTIAL at bias `b`. `I(b)`: strings admitted
by R1 REPLAY-ISOLATED at bias `b`. The three outcomes are mutually exclusive and checked in this order.

- **COLLAPSE** iff some foreign string `s` has `s ∈ A(0.2)`, `s ∉ A(0.0)` and `s ∉ I(0.2)`. The verdict carries
  **(marginal)** when every such `s` has its smallest clause margin under 0.01.
  → A design entry in engram_formation.md E4: count override-widened matches without averaging them into the
  centroid, or freeze text centroids on override matches. Either changes `pattern_complete_or_separate` and fires
  the EC completion row's "EC threshold / centroid-update change" trigger. Because the hazard is latent, the change
  lands **with or before the first positive text-credit producer**; landing it sooner is preventive and says so.
- **NO HEADROOM** iff no foreign string lies outside both `A(0.0)` and `I(0.2)`: the meter had nothing it could call
  drift. No verdict is claimed. #911 stays open with the widening-overreach list and `I(0.2)` attached, and the next
  step is an owner decision recorded on the issue. Never reported as NO COLLAPSE.
- **NO COLLAPSE** otherwise → close #911 with the numbers; no code. The close note lists every foreign string in
  `I(0.2)` as **widening overreach** (the radius at the cap spans concepts), an input to whoever builds the first
  positive text-credit producer, and states any non-empty `E(0.2)` (drift a closer node may have masked).
- `b = 0.1`, the bare-isolated arm, `E(b)` and every RA number are reported and never decide.

## Record

One JSON document, `docs/experiments/data/e4_text_widening_drift/diagnosis.json`, written through
`scripts/_provenance.py::stamp_diagnosis(report, mock=False, code_provenance=<in_process_code_provenance block>)`.
It carries `record_kind: "diagnosis"` (never support), epoch `ts`, `status` (`ok`, or `failed` with a `refusal`
reason), and the 22 whole-string embeddings (float32, base64) with the SHA-256 of their concatenated bytes in walk
order, so every arm's assignment and the verdict can be re-derived offline, independent of model, device and
library. The committed path is written only under `--write-experiment-results`
(`_provenance.evidence_out_paths_or_exit`); otherwise the record goes to a temp directory and both paths are
printed. Dry runs write to a temp directory or to a top-level `docs/experiments/data/e4_dry_run_nonfrozen/`,
never inside `e4_text_widening_drift/`. A refused run writes its `failed` record too (disclosed, never evidence).

**Exit codes.** 0: an outcome (COLLAPSE, NO HEADROOM or NO COLLAPSE), `status: ok`. 3: provenance refusal. 4:
apparatus or instrument refusal, with one of the `refusal` reasons `encoder_fallback`, `model_revision_mismatch`,
`fixture_sha_mismatch`, `env_toggle_set`, `bias_known_answer`, `wiring`, `replay_known_answer`,
`positive_control`, `negative_control`, `identity`, `determinism`. Any other exception exits non-zero with no `ok`
record.

**Only a `status: "ok"`, `mock: false` record from a clean tree at a commit on `main` closes #911 or discharges
T7. A refusal (exit 3 or 4) is a typed abort and moves nothing** (CLAUDE.md, weak evidence never gates). The data
PR merges with a merge commit, never a squash.

## What this does not claim

One fixture (Roy-register second-person sensation text, not a run's percept stream), one walk order, one rewarded
node in the first-formed position, one encoder on CPU, a bias held at the cap with no decay. Production's
self-reinforcing loop (a widened node earning more eligibility, hence more credit) is absent here and capped at
0.20 anyway. NO COLLAPSE bounds production for this case; it does not cover `vision` percepts, affordance-name
chunks, nodes rewarded by a real run's credit pattern, or reward that rises during a stream. COLLAPSE names the
mechanism to fix; it does not show production reaches that condition, since no current run widens a text node.

Behaviour tier: n/a (an offline measurement of a learned recognition bias's side effect).
