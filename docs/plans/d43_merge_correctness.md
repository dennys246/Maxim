# D43 — merge correctness: the design brief (1.1.3)

**Status:** DESIGN BRIEF, 2026-09-01. Zero code. Written from a three-lens parallel sweep
of the merge mechanics, the identity surfaces, and the evidence surface — run *before*
implementing, on the D53 precedent, where the ledger described one collapse site and the
sweep found four, and the first fix still shipped the defect.

**It found the same shape here.** The ledger describes three barriers; there are **five**.
And the fix the ledger prescribes — return the id map — **ships the defect** if taken
literally, for a reason no plan document names (§2.4).

> **Read this before implementing.** Two roadmap sentences currently instruct an
> implementer to reproduce the bug (§5).

---

## 1. What D43 actually is, measured

Reproduced against the Oasis case study's own designated pair (`taught_seed42` ←
`taught_seed43`, same `agent_id`, same body — the *most favourable* configuration):

| step | result |
|---|---|
| `ec_merge` aligns seed43's audio clusters onto seed42's | **cosine exactly 1.000**, 4+4 → 4 nodes |
| the right→left id map | **computed, then discarded** |
| `nac_merge` folds `cluster_reward_bias` | 3+3 → **6 keys, zero folded** |
| of the 6 surviving keys | **3 name cluster ids absent from the merged EC** — structurally unreachable at readout |
| post-merge policy vs seed42 alone | **byte-identical** |
| what the CLI reports | success, and `len(cluster_reward_bias)` **doubled** |

The success indicator is *inversely correlated with success*: `len()` is `|left ∪ right|`,
maximal exactly when nothing aligns.

And on the crèche apparatus, breaking either key axis collapses the measured federation
benefit from **+0.410 to exactly +0.000 across 8 seeds** — while `nac_merge` returns a
strictly larger dict and reports success.

---

## 2. The five barriers

### 2.1 The discarded EC id map

`hivemind/merge.py::ec_merge` computes `best_id` as a loop-local, reads it once, rebinds it
next iteration. `return merged` returns nodes only.

**Genuinely unrecoverable, not merely unreturned.** Three reasons: the scan iterates
`merged` *while it grows*, so a right node can match a previously-inserted right node —
the alignment is greedy single-pass clustering over `left ∪ right-so-far`, not a right→left
map; `target["embedding"]` is mutated in place by the running mean before later comparisons,
so re-running the scan gives different similarities than the merge saw; and both loops
iterate `sorted()` over uuid4 keys, making the outcome order-dependent. There is no
commutativity test for `ec_merge` (only `nac_merge` has one).

**`ec_merge` has no production caller.** `hivemind/cli.py`'s `merge-nac` imports and calls
`nac_merge` **only** — the shipped import verb merges NAc keys against an EC it has not
aligned. The dangling half is the default path, not a hypothetical.

### 2.2 `agent_id` in the key — on five surfaces, not one

The ledger names `cluster_reward_bias` and `cluster_reward_source`. Also affected:
`reward_bias` (a **double** miss — `agent_id` *and* a uuid node id, the same shape),
`percept_valences`, `event_outcome_welford`.

`agent_id` is never minted — it is an operator string with defaults scattered across
modules (`"sim_aut"`, `"default_agent"`, `"console_agent"`, `"api_agent"`, `agent_id=name`).
It fails **both ways**: two independent agents both defaulting to `console_agent` *falsely
match*; two that chose names *never* match.

### 2.3 The tool signature is scene-dependent, not just body-dependent

`build_tool_signature` is `f"tool:{tool_name}"`; the prefix comes from the tool *name*,
built by `tool_bridge.py::generate_tools_for_entity` as `f"{ent.name}_{aff_name}"`. Worse,
`_resolve_tool_name` prepends **ancestor names on collision**, so the same body in a
different scene mints different keys — it fails to align *with itself*.

The tree already carries a hand-rolled workaround that documents its own hazard:
`bodies/reachy_mini_infant.yaml` sets `name: reachy_mini` (not `reachy_mini_infant`) with a
comment warning never to register both bodies in one `ToolRegistry`. The identity is
already being managed by hand.

### 2.4 ⚠ The threshold mismatch — this defeats the ledger's prescribed fix

**`ec_merge`'s default `cosine_threshold` is 0.44** (`ECConfig.pattern_complete_threshold`,
tuned for paraphrase-mpnet **text**). **Interoception clusters — the ones that key
`cluster_reward_bias` — are formed at 0.85** (`SensorEncoderConfig.pattern_threshold`).

Applied to `_sensor_embed` output, virtually every interoception node pair clears 0.44. So
returning the id map **without retuning the threshold** collapses *all* of B's interoception
clusters onto whichever of A's scores highest — a degenerate all-into-one alignment,
silently, on the exact modality that keys the primary action-selection signal.

**That is a confidently wrong map where today there is an honestly missing one — strictly
worse.** No plan document names this. It is the D53 pattern repeating: the described fix
reaches one term and ships the defect.

Two mitigating facts for the fix design: `_stable_basis` derives bases from the **sensor
name**, so agents sharing sensor names do live in a comparable space — interoception
alignment is structurally achievable, just not at 0.44. And `_cosine` returns 0.0 on
dimension mismatch, so cross-encoder merges fail safe.

### 2.5 The merge deletes state

`nac_merge`'s return dict has **no `cluster_reward_source` key at all**, so `load_state`
resets it to `{}`. Every merge **deletes the receiver's own credit provenance** — local data
loss, a different failure class from "the foreign value didn't land". The S1 "why" clause
disappears from the prompt annotation for biases the agent learned itself.

`nac_merge` also drops `saved_at` (patched back only in the CLI, so every other caller loses
the decay clock), and the CLI calls `nac_merge` without the target's caps, silently
re-clamping a non-default `NACConfig` to the merge function's defaults.

---

## 3. N→1 semantics are not undecided — they are decided wrong

`_merge_mean_clamped` is an **unweighted** mean of shared keys. The only shipped N>2 fold is
the pairwise left-fold in `scripts/orient_substrate/5_operant_creche_federation.py::merge_creche`
(and its twin in `7_graded_creche_federation.py`).

Unweighted pairwise mean is **not associative**. At N=4 the contributor weights are
**1/8, 1/8, 1/4, 1/2** — the last contributor gets half the pooled bias. So the semantics
today are "most-recent contributor dominates, exponentially."

This is internally inconsistent within one `nac_merge` call: `total_observations` sums
exactly, `event_outcome_welford` uses a true order-independent parallel Welford,
`_merge_link_pair` uses observation-weighted means — and only the four bias dicts use the
order-dependent unweighted mean.

**Fixing the keys without fixing the fold makes the federation experiments produce a
wrong-weighted answer instead of a null one.**

---

## 4. The recommended fix — minimal, and it migrates nothing

The identity that blocks D44 is **not** `agent_id` or `tool_signature`. It is `cluster_id`,
and it needs no namespace — it needs the map that already exists to stop being discarded.

1. **`ec_merge` returns `(merged_nodes, id_map)`.** The map is computed today and dropped at
   `return merged`. On the case study's pair it is a perfect 3/3 at cosine 1.000.
2. **Retune the merge threshold per modality** (§2.4) — or the map is worse than useless.
   Sensor modalities must align at their own `pattern_threshold`, not at the text default.
3. **Re-key `cluster_reward_bias` + `cluster_reward_source` through the map** before
   `nac_merge`. The mechanical twin of the scrub's existing unpack/rewrite/fold.
4. **Normalize `agent_id` at the ingestion boundary, not in the key.** The bias key's agent
   field exists for per-agent stash discipline *within* one substrate; across a merge the
   contributor identity already lives in `manifest.contributor_id`. Rewriting the incoming
   `aid` to the local agent at ingest closes the second axis **without touching a single
   persisted file**, and fixes `get_agent_tool_biases`'s filter for free.
5. **Decide the N→1 fold explicitly** (§3). Count-weighted matches `ec_merge`'s own centroid
   rule and `_merge_mean_clamped`'s zero-prior convention.
6. **Restore `cluster_reward_source` and `saved_at`** to `nac_merge`'s return (§2.5).

**None of this migrates a byte of the 20 `53_agents` evidence files.**

### The guard to land with the fix

> **Every `cluster_reward_bias` key surviving a merge must name a cluster id present in the
> merged EC.**

On the case study's own pair that assertion **fails 3/6 today**. It is cheap, mechanical,
and would have caught D43 the day `ec_merge` shipped.

---

## 5. Two roadmap corrections owed before anyone implements

**(a) The gate-7 instruction reproduces the bug.** `roadmap_1_1_to_1_3.md` says to design a
capability namespace "taking `MotorStep.sem_key`'s `(entity, modulator, affordance)` triple
as the starting shape." **That triple's first element *is* the body dependence.** The
capability key is `sem_key` **minus its first element**. Following the sentence literally
preserves barrier 2.3.

Also: `sem_key` has exactly two references in the tree, so adopting it means *building* an
identity, not promoting one.

**(b) Two authorities contradict each other.** `roadmap_1_1_to_1_3.md` gate 7 says capability
namespace; `oasis_case_study_taught_orient.md` §1 front-gated the choice and picked **body
namespace**, reason stated; the microduck two-lens round (rev 2) **withdrew** its capability
recommendation and restored the case study's. Whoever implements will read one of these.
Reconcile them in the same commit as the decision.

**And `register_bundle_migration` is less reversible than claimed.** It takes and returns the
**manifest dict only**; `extract_bundle` copies every other member through untouched. It can
stamp `body_ref` onto an old manifest. It **cannot re-key `cluster_reward_bias` inside
`nac.json`**, and has nothing to do with the receiver's own on-disk state. (a)-first is safe
because the blast radius is small, not because the migration hook covers it.

---

## 5a. Gate 7 — recommendation

**Ship (a) the body namespace now, and emit the capability key alongside it in the same commit.**

**Gate 7 is not on the critical path**, which is the finding that frees this decision. D43's
live axes are `cluster_id` and `agent_id`; the tool-signature barrier does **not** fire for two
agents on one body — exactly D44's configuration. So choose on long-run merit, not urgency.

**(a) costs about a day and migrates nothing.** Two manifest fields (`body_ref`,
`affordance_namespace`), a refusal branch in the ingestion adapter, one `BUNDLE_SCHEMA_VERSION`
bump with a migration that stamps legacy manifests. It converts a *silent* cross-body miss into
a *loud* refusal — this codebase's stated rule for exactly this shape. It does not make
cross-body sharing work; it makes its absence honest.

**(b) is right long-run but its cost lands in the wrong place.** Seventeen `f"tool:{...}"` sites
bypass `build_tool_signature` despite its docstring claiming the monopoly — two of them on the
hot readout path (`nac.py:2013`, `nac.py:2227`). **Miss one and the write key and the read key
diverge into a silent zero indistinguishable from D43 itself.** Plus: the `listen` affordance
sits under different modulators on different bodies (`head` vs `capture`), so `(modulator,
affordance)` is not yet a shared vocabulary; `_IDENTIFIER_TOKEN` in `bundle.py` excludes `:` and
`/`, so a `skill:orient:turn_left` shape would be silently dropped by the scrub; and migration
touches the **20 `53_agents` files** — the repo's only shipped evidence bundle, SHA-manifested,
behind EARNED rows. That drags a namespace refactor into the provenance discipline that Exp
53/53b already cost a grade over.

**The addition neither option includes, and the reason to prefer this over plain (a):** have
`compose_bundle` write `(modulator, affordance)` as a **second field** beside the body-prefixed
signature. Bundles then carry both keys from day one, so (b) later becomes a **reader-side
change with no migration** — new bundles already hold the data, and old ones get (a)'s honest
refusal. That is cheap insurance against precisely the cost that makes (b) unattractive today,
and it is the half of the decision `register_bundle_migration` cannot cover (§5: it migrates the
manifest only, never the keyed payload).

**Schedule (b) explicitly rather than deferring it again.** It now has **three** constituencies:
sharing (this gate), the microduck's portability, and the `bodies/reachy_mini_infant.yaml`
name-lie — a workaround already being paid as interest, with a documented collision hazard. Three
consumers is past this project's own "one silent-failure miss in a critical path → consider
structural enforcement" line.

**One correction that applies whichever way this goes:** the capability key is `(modulator,
affordance)`, **not** `MotorStep.sem_key`'s full triple. See §5.

## 6. D44 — what actually satisfies it

**Independence, mechanically:** distinct `agent_id`; **a separate `EntorhinalCortex` +
`SensorEncoder` object per agent** (independence is *guaranteed* here — `pattern_complete_or_separate`
allocates `str(uuid4())`, so identical input yields disjoint ids); optionally distinct bodies.
Do not share the encoder object and do not copy one.

```
before = measure(B)                       # B never saw the contingency
B.load_state(nac_merge(B.dump(), A.dump()))
after  = measure(B)
assert before <= chance + eps
assert after - before >= 0.20             # the gate: a BEHAVIOURAL delta
```

Three companion arms, all required: a **negative control** (merge a bundle from an agent that
learned nothing → delta ≈ 0, separating "a bundle arrived" from "a want arrived"); a
**dangling-half** arm (merge `nac.json` without `ec.json` → must reproduce the silent zero);
and an **anti-vacuity guard** that monkeypatches `nac_merge` to `return left` and asserts the
delta collapses to 0.

**This test cannot pass on current `main`** — re-keying does not exist. Write it now as
`xfail(strict=True)` referencing D43, and let the fix flip it. **A D44 test that is green
before D43 is fixed is by definition testing the wrong thing.**

---

## 7. The evidence surface is weaker than the ledger records

**Exp 45's merge/fleet arm is a vacuous guard on an EARNED row.** Re-run on the real recorded
NACs still on disk:

| variant | correctness | magnitude | gauntlet |
|---|---|---|---|
| real `nac_merge` | 1.00 | 1.00 | **PASS** |
| no-op `return left` | 1.00 | 1.00 | **PASS** |
| no-op `return right` | 1.00 | 0.50 | **PASS** |
| `return {}` | 0.00 | 0.00 | FAIL |

Only total annihilation fails, and the real merge is **argmax-identical to `return left` in
all four bins**. Three vacuities compound: both inputs already probe 1.00 so the
`≥ max(left, right)` clause is evaluated at ceiling; the gate reads correctness only, never
`magnitude_appropriateness`; and the parents share `agent_id` and a hardcoded 4-bin symbolic
cluster space by construction (`AGENT_ID = "reachy"`, `OFF_CENTER_BINS`), so no encoder or EC
is involved. Filed as **D62**.

The crèche probes (`5_`/`7_operant_creche_federation.py`) are **honest** — they do have a real
no-op falsifier and they disclose the shared encoder in their own docstrings — but their gate
is saturated at both taught arms and seed-fragile through `single_partial`.

**For the dose–response ladder:** an unsaturated operating point exists — probe 7 at roughly
`N=10, K=4` gives `sp=0.22 / sf=0.63 / ct=0.78`, below ceiling with genuine headroom and
superadditive. Better than the script's saturated `12×25` defaults.

---

## 8. Sequence

1. Correct the two roadmap sentences (§5) — before anyone implements.
2. Decide gate 7 — **recommendation in §5a: (a) now, emitting the capability key alongside,
   with (b) scheduled.** **(b) has three
   constituencies** — sharing, the microduck's portability, and the `reachy_mini_infant`
   name-lie already paid as interest — which is past this project's own "consider structural
   enforcement" line. But it is **not** what blocks D44.
3. Write D44's test as `xfail(strict=True)`.
4. Implement §4, in order. Land the §4 guard in the same commit as the fix.
5. Re-run the crèche probes at an unsaturated operating point.
6. Re-attest or downgrade Exp 45's arm 3 (D62).
