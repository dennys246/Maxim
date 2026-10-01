# Exp 10 re-run, O19 (pre-registered 2026-09-30)

**Scope:** `rerun_exp10_o19`

- **Ledger row:** T1-1 (cross-session memory persistence), [behavioral_graduation_candidates.md](../../plans/behavioral_graduation_candidates.md).
  STALE since 2026-09-30 (M1b PR 3, owner decision): every session of the 2026-09-27 re-run ended `planning_failed`,
  a typed abort, which cannot back a status. This re-run is `outstanding.md` O19, and it blocks the 1.3.2 cut.
- **Claim re-tested (the row, verbatim):** "Cross-session memory persistence (substrate carries memory across
  sessions)", mechanism "Hippocampus persistence + RECALL reconsolidation".
- **Preconditions:** [#935](https://github.com/dennys246/Maxim/issues/935) (Sim-Short runs stopping after 1–5 turns
  on D13) fixed, and the O19 harness merged. The run does not start before then.

## Gate (the row's metric, owner decisions 2026-09-30)

The 2026-04 doc ([10_cross_session_enrichment.md](../10_cross_session_enrichment.md)) states its success criteria
in prose; one ("2+ enrichment sections") was PARTIAL even originally. The owner chose the ledger row's own metric —
exact persistence plus ~3 memories surfaced per resume turn — written here as numbers, and an honest statement of
what the bytes can and cannot show. No threshold changes.

Three sessions, as in the original protocol (goals, turn caps and resume chain unchanged):

| Phase | Goal (verbatim) | Turn cap | Resumes |
|---|---|---|---|
| 1, baseline | `escape a dungeon with a sleeping guard` | 8 | none (a fresh data home) |
| 2, the gate | `escape a dungeon with a sleeping guard` | 8 | phase 1 |
| 3, negative transfer | `you are in a peaceful garden, enjoy the flowers` | 5 | phase 1 |

### Complete-attempt condition

An attempt is the three phases from one fresh data home. It is **complete** only if all of these hold; otherwise
it is an **aborted** attempt (recorded, never a verdict).

- **C1 (ran to the cap).** Every phase's `report.json` reads `finish_reason == "max_turns"` and `turns >=` its cap.
- **C2 (the resume chain loaded).** Phases 2 and 3 record `provenance.resume.resume_loaded == true`, resuming phase
  1's `session_id`, with `stores.hippocampus == "loaded"` and every store phase 1's copied directory holds reading
  `"loaded"`; phase 1 resumes nothing. A missing `aut_hippocampus.json` in any phase aborts the attempt.
- **C3 (one known, unchanged code tree).** Every session reads `working_tree_dirty_src_scripts == false` and
  `code_changed_during_run == false`, and its `code_tree_sha256` equals its end digest and the harness's own
  digest; none reads `unknown`.
- **C4 (the pre-registered apparatus).** Every session's report stamps the language and AUT profile `mistral-7b-instruct-v0.2` (the name `llm.profile mistral-7b` normalizes to; the router stamps the normalized name — amended 2026-09-30, before any data, owner decision)
  and `n_ctx` 8192 (`configured_n_ctx` and each role's `router_n_ctx`), its `goal` equals the table's verbatim, and
  the harness row's recorded sim argv matches (`--interactive false`, `--sim-max-turns` the cap, `--resume-sim`
  phase 1's session for phases 2 and 3), `configured_n_ctx_source == "config"`, and the model the harness
  read from the server's `/v1/models` while the sim ran matches the profile's GGUF (`_served_model_matches`) on every
  read, at the endpoint the report names; the sim's `MAXIM_*` environment is exactly the harness's (data home, run id,
  log file, `MAXIM_LOG_FILE_MAX_BYTES=0`) and the protocol's (the harness drops every other `MAXIM_*` key and records
  what it passed); and the phase's report, run
  log and `aut_hippocampus.json` were copied.

### Persistence (P), all must hold

Let N1 be the number of memories in phase 1's saved `aut_hippocampus.json`.

- **P0.** N1 ≥ 3 (P2 cannot pass on an empty store).
- **P1.** Phase 2's first `enrichment_trace` reads `hippocampus_size >= N1`, and so does phase 3's. (The value at
  that trace is reported; equality is expected but a memory captured before the first enrichment would raise it.)
- **P2.** Every memory id in phase 1's saved store is present in phase 2's saved store, and in phase 3's.

### Recall (R), per resume turn

Traces are attributed to turns by the `sim_exec` `Bridge.send_and_wait ENTER turn=N` lines (enrichment runs from
several sites, so a trace is not a turn).

- **R1.** Each of phase 2's turns 1–8 has at least one `enrichment_trace` with a non-empty `goal`.
- **R2.** The first such trace of every phase-2 turn reads `memories >= 3`.

**What R shows, stated honestly.** 3 is the enrichment cap (`bio_enrichment` keeps the first 3). On phase 2's first
turn, where `hippocampus_size == N1`, every candidate memory is carried, so the 3 surfaced are carried memories:
that trace shows **carried recall**. On later turns phase 2's own new memories can also fill the cap, and the
trace logs no memory ids, so R2 there shows that the cap is filled on every resume turn, not that the memories are
carried. The verdict reports which turns had `hippocampus_size == N1`.

### Negative transfer: NOT MEASURED

The original check ("dungeon memories don't dominate the garden") was never put into numbers, and the text of
surfaced memories is not logged. Phase 3 runs; the verdict reports its agent actions, its new memories and how many
of those mention `dungeon` or `guard`. None of this gates, and the row keeps saying negative transfer is not
measured.

## Attempts and the verdict

- **Stop rule.** At most 3 attempts. An aborted attempt is recorded and may be retried. **The first complete
  attempt decides**; the harness refuses to start a new attempt once the file holds a complete one. After 3
  aborted attempts T1-1 stays STALE and the cause is investigated.
- **Every attempt is declared before it runs (owner decisions 2026-09-30).** Before spawning anything, the harness
  pushes a start marker to `origin`: the annotated tag `refs/tags/o19/<exp>/attempt-<k>-<harness_run_id>`, with
  k one more than the highest k `git ls-remote` lists (k ≤ 3). A tag ruleset on `refs/tags/o19/**` (created by the
  owner before the first attempt; rules `deletion` and `update`, `creation` left allowed, no bypass actors) keeps
  every marker. The verdict reads the ruleset read-only and refuses unless:
  (a) exactly one ruleset targets tags with include exactly `refs/tags/o19/**`; it is `active`, carries both
  rules, has `bypass_actors == []` and `current_user_can_bypass == "never"` (a missing field refuses), and its
  `created_at`, its `updated_at` and every `/history` entry's `updated_at` (an empty history is allowed) all
  precede the earliest marker's tagger date (the rig clock, as for `ts`), so a ruleset edited, disabled and
  re-enabled, or deleted and re-created after the first marker refuses; every marker must be an annotated tag
  object (a lightweight tag has no tagger date and refuses);
  (b) the markers' k values are unique and run 1..n with no gaps;
  (c) the markers match the rows file's attempts one to one by run id.
  A marker with no rows counts as an aborted attempt toward the 3. So an attempt discarded before it was committed
  is still visible, and removing a marker needs an admin to change the ruleset, which its history shows. The
  verdict runs with a token that can read the ruleset and its history; an unreadable ruleset refuses, never skips.
  Harness refusals before the marker is pushed (preflight, model, tree, HEAD not on `main`, the rows file not
  `main`'s, a server already on the sim's port, a failed push) are not attempts. Once the marker is pushed, every phase
  that starts writes a row, an interrupted one too (Ctrl-C, SIGTERM, SIGHUP; a SIGKILL or a power loss leaves the marker
  with fewer rows: an aborted attempt).
- **Attempts are on main before the next one starts.** After each attempt the operator commits its rows (and copied
  sessions) to `main` (a merge-committed data PR) before starting another. **The rig stays at the first attempt's commit** (amended 2026-09-30, before any data,
  owner decision): every attempt runs from a commit on `main`'s history with the first attempt's code tree (in practice
  that same commit), so the rows file keeps one code tree; only the rows and copies move to `main`. The verdict checks, from `main`'s first-parent history
  (`git log --first-parent --format=%cI -- <rows>`), that the rows file only ever grew (each version a prefix of the
  next, the newest the bytes judged), that every attempt's rows landed before the next attempt's marker (its tagger
  date: the rig clock against GitHub's merge time; PR turnaround is minutes), and that each attempt ran on its
  marker's commit, on `main`; and that this prereg and the two O19 scripts are byte-identical at every executed
  commit, on `main` and at the verdict's own commit. **If the
  marker match or the ordering check fails, the verdict refuses: no status change, the row stays STALE.**
- **Scope.** The verdict reads the whole rows file (every attempt), and names the deciding attempt.

| Verdict | Condition | T1-1 |
|---|---|---|
| `PASS` | the deciding attempt is complete, and P and R hold | **MAINTAINED** at the run date, narrow. The row says "carried recall shown on the first resume turn" only if that turn's first trace reads `hippocampus_size == N1`; if it reads N1 + k, it says at least 3 − k of the 3 were carried (or that carried recall is not shown, when k ≥ 3). "Negative transfer not measured" stays in the row |
| `FAIL` | the deciding attempt is complete, and P or R fails | **BROKEN** |
| `ABORT` | no complete attempt within 3 | no change: T1-1 stays STALE (an abort is not data) |

Pass sets per target status (for the M1b evidence gate): `exp10_verdict` → `{PASS}` supports MAINTAINED; nothing else
supports a positive status.

## Apparatus

- **Box:** big-mac-mini, quiet (Paper servers, Minecraft bridges and ollama stopped; operator-attested — the
  harness stamps the hostname). One harness, no second LLM consumer.
- **Model:** `mistral-7b-instruct-v0.2.Q4_K_M.gguf` (`llm.profile mistral-7b`) at `llm.n_ctx 8192`, set through
  `maxim config` and checked from the reports (C4).
- **Code:** a commit on `main` with this prereg and the O19 harness merged, clean tree, a worktree pinned at it.

## Command

```bash
python scripts/o19_rerun.py --exp 10 --write-experiment-results            # one attempt; re-run to retry an abort
python scripts/o19_verdict.py --exp 10 --data docs/experiments/data/rerun_exp10_o19/rows.jsonl \
    --json docs/experiments/data/rerun_exp10_o19/verdict.json --write-experiment-results
```

The harness (`scripts/o19_rerun.py`, its own PR before the run) runs one attempt per invocation from a fresh data
home, hands each sim the harness run id, records every phase as a row (aborts included), and copies each session
directory and its run log into `docs/experiments/data/rerun_exp10_o19/` with the measured files' SHA-256 in the row.
The verdict (`scripts/o19_verdict.py`) stamps this prereg's blob SHA (at the executed commit and at its own) and the
SHA-256 of its own source.
