# Known defects & limitations — running ledger

**What this is.** The fourth ledger. The repo already tracks *behavioral* claims
([behavioral_graduation_candidates.md](../plans/behavioral_graduation_candidates.md)),
*algorithmic* claims ([bio_faithful_roadmap.md](../plans/bio_faithful_roadmap.md)), and
*engineering rules* (CLAUDE.md invariants). This tracks **what is verifiably wrong or
bounded right now** — the axis on which findings otherwise evaporate into plan-doc asides.
(A fifth ledger, [docs/limits/README.md](../limits/README.md), tracks measured
*instrument* limits — nothing broken, but properties every experiment design must
respect. Defects go here; characterized measurement limits go there.)

**Why it matters more than a normal bug list.** The 2026-08-10 external critique's sharpest
point was that unstated limitations quietly become false claims. A defect that lives only
in one session's memory is indistinguishable, six months on, from a defect that was fixed.

## Rules (these are what stop this becoming a second CLAUDE.md)

1. **Verified only.** Every row cites `file:line` or a measurement. No suspicions — those
   belong in a plan's open-questions section.
2. **Every row has a disposition**, not just a description: `OPEN` (will fix, trigger
   named), `ACCEPTED` (deliberate limitation + why), or `FIXED` (PR + guard, then it
   leaves this file at the next prune).
3. **Rows expire.** A `FIXED` row is deleted once its guard exists — the guard is the
   durable record. An `OPEN` row with no trigger and no movement for two releases gets
   re-argued or becomes `ACCEPTED`.
4. **Claim linkage is mandatory** when a defect bounds a claim: name the graduation row it
   touches. A defect that silently invalidates an Earned row is the worst case this ledger
   exists to prevent.
5. **Deep investigations stay as their own doc** in this directory (the existing
   convention — see `console_seam_findings.md`). This file indexes them and carries
   standalone defects.

## Investigation clusters

| doc | scope | state |
|---|---|---|
| [console_seam_findings.md](console_seam_findings.md) | `maxim serve` console seams, sim_logger, PFC gate, aarch64 packaging | 12 fixed, 6 open |
| [display_print_corruption.md](display_print_corruption.md) | interactive display / print interleaving | see doc |
| [sim_embodiment_followups.md](sim_embodiment_followups.md) | sim embodiment wiring follow-ups | see doc |

## Standalone defects

Seeded 2026-08-11 from a four-lens architecture review; each was verified against the code
during that review and currently lives nowhere else.

### Substrate / persistence

| # | Defect | Disposition | Evidence |
|---|---|---|---|
| D1 | **`encoder_provenance` detects nothing at runtime.** It is recorded, persisted and reloaded, but nothing compares it against live encoder state — its only readers are the hivemind bundle/CLI export, and `record_encoder_provenance` *merges* divergence ("'mixed' is a finding, not an error"). A geometry change loads old-geometry centroids and cosine-scans them against new embeddings: **silently merged.** | **OPEN** — trigger: any encoder/threshold change. Prerequisite of the place-code default-ON gates. | `similarity/ec.py` (record/persist/load), `hivemind/bundle.py`, `hivemind/cli.py` |
| D2 | **There is no way to invalidate a stale EC substrate.** `cli_utils.py::MEMORY_PATHS` has keys for `hippo`, `nac`, `scn`, `atl`, `angular_gyrus` — **no `ec`**, not even under `all`. Clearing `nac` alone violates the NAc/EC pairing invariant. | **OPEN** — blocks any encoding change that requires invalidate-both-in-lockstep. | `cli_utils.py::MEMORY_PATHS` |
| D3 | **`ec_merge`'s cosine threshold is a hardcoded duplicate pinned by no test.** `cosine_threshold: float = 0.44` duplicates the EC default; unlike the frozen-modality set (which *is* pinned by `test_hivemind_frozen_modalities_match_ec_default`), nothing detects drift. The layer deliberately refuses internal imports, so it cannot read a threshold map. | **OPEN** — trigger: per-modality thresholds (plan F-A). | `hivemind/merge.py:557` |
| D4 | **A same-dimension encoding change defeats the merge dimension-guard, invisibly.** `_cosine` returns 0.0 on dim mismatch — but a place code keeps `dim=384` and the same `"audio"` tag, so old- and new-geometry nodes merge whenever partial cosine clears the threshold. Because `audio` is frozen, the corruption is undetectable: counts and contributors inflate, the centroid never moves. | **OPEN** — gate on `MAXIM_PLACE_CODE_EXTEROCEPTION` default-ON. | `hivemind/merge.py` |
| D5 | **`nac_merge` never folds cluster biases across agents.** Keys are `agent\x1fcluster\x1ftool` and cluster ids are per-agent UUIDs, so cross-agent cluster learning is unioned, never merged. | **ACCEPTED for now** — but it makes session-relative spatial bearings unmergeable, i.e. it is the mechanical answer to the RSC plan's "does this need a world anchor?" (yes). | `hivemind/merge.py:506-510` |

### Learning path

| # | Defect | Disposition | Evidence |
|---|---|---|---|
| D6 | **Hebbian episode binding is inert on the main percept path.** `memory_hub` stashes a **1-tuple** of substrate nodes, and `apply_hebbian_on_close` returns early on `len(nodes) < 2` — a silent no-op. So the binding graph never grows from llm-primary percepts, independent of any resolution question. | **OPEN** — trigger: the cross-modal fabric's binding work depends on this path. | `integration/memory_hub.py:348`, `memory/episode.py:795` |
| D7 | **`min_delta = 0.05` is an undocumented second resolution limiter.** `SensorEncoder` short-circuits the EC scan and returns the cached node whenever no sensor moved ≥0.05 — a hard ~4.5% dead zone on a `[-1,1]` sensor **regardless of encoding**. Any population code past ~40 buckets is capped by this gate, not by geometry. | **ACCEPTED** (it is a real efficiency win) — but must be stated wherever resolution is claimed. | `similarity/encoder.py:519, 640-645` |
| D8 | **A read path mutates text centroids.** `bio_enrichment` calls `pattern_complete_or_separate(embedding, "text")` per enrichment query, with an in-code note that the centroid update is intentional reconsolidation (~1/(n+1) shift per query). Consequence not documented anywhere: **querying degrades text-cluster resolution over time.** | **OPEN** — needs measurement, then accept-or-fix. | `bio_enrichment.py:665-667` |
| D9 | **5 of 6 `TemporalEvent` categories have no producer.** Only `tool` emits; the drive emitter is both unwired *and* malformed (raises `TypeError` into a `except Exception: log.debug`). `record_event` even special-cases `deliberation` significance for a producer that does not exist. | **OPEN** — documented in [deferred/scn_event_producer_gap.md](../plans/deferred/scn_event_producer_gap.md); revival requires answering whether per-event-type phase learning earns its keep. | see plan |

### Claim scope

| # | Defect | Disposition | Evidence |
|---|---|---|---|
| D10 | **Exp 45's Earned status does not transfer to an EC-clustered orient policy.** The orient backbone builds state from `az_bin(...)`, a hand-written bin string passed as `current_cluster_id` — it never calls EC. That is hand-curated discretisation *upstream* of the substrate (the interim-contamination pattern), so any future EC-clustered orient policy needs its own experiment. | **OPEN (scope note)** — must be recorded on the Exp 45 graduation row. | `scripts/orient_backbone/live_3_learn.py:680`, `live_common.py:340` |
| D11 | **~432 bare `except Exception: pass` remain repo-wide.** 48 measurement-path sites were instrumented (#487); the rest are grandfathered and un-instrumented, so a silent failure outside that scope is still invisible. | **ACCEPTED** (scoped deliberately) — see [measurement_path_fail_loud.md](../plans/measurement_path_fail_loud.md). | `grep -rEA1 "except Exception:\s*(#.*)?$" src/maxim/` |
| D12 | **Orchestrator LLM calls blocked unboundedly — ROOT-CAUSED (2026-08-18, `sample` capture of a live 2.4h wedge): `router._inference_lock` inheritance deadlock.** The llm_worker timeout path abandons an orphan thread that can still be inside the locked region; its executor-replacement fallback gives new calls fresh threads but nothing frees the lock, so the untimed `with self._inference_lock` parked every subsequent call forever (~75 lock-waiter threads, ZERO network activity, stall detector blind because registry entry happens inside the lock — 265 impotent nudges observed). NOT a network-timeout escape; the model server was healthy and idle throughout. Deceptive symptom: AUT ticks sensors normally while starving for percepts ("bio systems not engaging"). | **FIXED** — bounded lock acquire + loud failure (`MAXIM_INFERENCE_LOCK_TIMEOUT_S`, default 600s) + the #517 sim-level hard-abort (exit 4) as defense in depth. Guards: [tests/unit/test_inference_lock_timeout.py](../../tests/unit/test_inference_lock_timeout.py) (held lock → bounded loud failure; verified the old behavior never returned) + [tests/unit/test_stall_hard_abort.py](../../tests/unit/test_stall_hard_abort.py). Prune at next sweep per rule 3. | `sample` thread-stack capture 2026-08-18 (80 threads: lock-waiters + zero recv); `models/language/router.py::_complete_text`; `agents/llm_worker.py` orphan path |

## Pending, not yet a defect

- **The S1 annotation renderer will break PR #497's S4 parser** (its regex matches the band
  exactly; adding `— why` inside the bracket makes it report "no annotation" silently).
  Not yet a defect because neither has merged — but they must land together with the
  parser fix, or this becomes D12 on the day they don't.