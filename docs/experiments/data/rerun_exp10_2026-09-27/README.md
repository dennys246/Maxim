# Exp 10 re-run, 2026-09-27 (the 1.3.1 trigger walk)

A re-run of behavioural-graduation **Tier 1 row 1** (cross-session memory persistence,
[Exp 10](../../10_cross_session_enrichment.md)). Its "hippocampus persistence schema change" trigger
fired about eight times after v1.3.0, and the row was STALE, blocking 1.3.1. Protocol:
[heartbeat runbook](../../protocols/heartbeat_rerun_runbook.md) §Sim-Short 4. Verdict and walk entry:
[behavioral_graduation_candidates.md](../../../plans/behavioral_graduation_candidates.md).

## Provenance

| | |
|---|---|
| Rig | big-mac-mini, worktree `.worktrees/heartbeat-exp10`, detached, clean tree (`git status --short` empty) |
| Executed commit | `a1ba1e5d75f144d534134e37c02cc5f92287d319` (`main` with #934). Operator-attested: the records stamp the model and the worktree path (in the telemetry), not the commit, `n_ctx` or the clean-tree check |
| Interpreter | imports the worktree's `src/` (checked before the run: `python -c "import maxim; print(maxim.__file__)"`) |
| Model | `mistral-7b-instruct-v0.2.Q4_K_M.gguf`, `llm.profile mistral-7b`, `llm.n_ctx 8192` (set through `maxim config`; matches the 2026-08-18 heartbeat) |
| Box | quiet: Paper servers, Minecraft bridges and ollama stopped for the run |
| Commands | the runbook's §Sim-Short 4, run as `python -m maxim --sim … --interactive false`, `MAXIM_LOG_FILE` per phase |

## Files

One directory per session: `report.json`, `aut_{atl,ec,hippocampus,nac,scn}.json`, and gzipped
`actions.jsonl`, `bio_telemetry.jsonl`, `run_log.jsonl` (the `MAXIM_LOG_FILE` JSONL) and `console.out`.
All 50 files were checked against the rig by SHA-256 after copying. `SHA256SUMS.uncompressed` pins
each session's `report.json` and `run_log.jsonl` before compression.

## Every run, in order

Every run ended `planning_failed` on the planning liveness of [bugs ledger D13](../../../bugs/README.md)
(`unregistered_tool_proposed`, four times). None ended on the turn cap. In `110748` the log shows the
narrator's model proposing `sense_tools`, a tool only the agent under test has; the other runs were
not traced to the same depth. August's heartbeat ran 8 clean turns per phase on the same model and settings.

| Session | Phase | Role | Turns | Store at open → closed | Recall per enrichment trace |
|---|---|---|---|---|---|
| `20260927_110748` | 1 | **discarded**: stopped at turn 1 (D13) with 8 memories, too thin to gate on | 1 | 0 → 8 | 0 |
| `20260927_112714` | 2 | **disclosed, not gated**: resumed `110748` by mistake before the phase-1 retry | 2 (3 enrichment traces) | 8 → 136 | 3, 0, 3 |
| `20260927_113807` | 1 | **baseline** (retry, unchanged command) | 3 | 0 → 100 | 0, 3, 3 |
| `20260927_115820` | 2 | **the gate**: resumes `113807` | 1 | **100** → 144 | **3** |
| `20260927_121056` | 3 | **negative transfer**: resumes `113807` (garden goal) | 1 | **100** → 136 | **3** |

"Store at open" is the first `enrichment_trace` `hippocampus_size`; "closed" is the saved
`aut_hippocampus.json`. Predictions were 0 on every trace, as in the original Exp 10 record. Which
session resumed which is settled by the memories' `run_id`s: `115820` and `121056` carry all 100 of the
baseline's memory ids and none of `110748`'s; `112714` carries `110748`'s 8. Every run is a typed abort
(`planning_failed`, exit 4); bugs ledger D22 says consumers must not count such runs as data, so
gating on them is the owner's decision below. The cause is tracked in
[#935](https://github.com/dennys246/Maxim/issues/935).

## Result against the row's gate

- **Persistence: shown, including the fields whose schema change fired the trigger.** Both resumes
  opened at exactly 100, the baseline's closing store; the early resume of the thin session opened at
  exactly its closing 8. On all 100 carried memories the fields the schema change added
  (`encoding`, `encoded_at_us`, `capture_seq`, `storage_strength`, `retrievability_anchor_us`,
  `situation`, `retro_tag`, `encoding_tag`) are identical between the baseline and each resume; the only
  changes are access bookkeeping on the surfaced memories (recall reconsolidation). The
  `experience_clock` resumes too: the baseline closed at 15,000,000 µs and each one-turn resume closed
  at 20,000,000 µs (one turn = 5,000,000 µs); a clock that failed to persist would read 5,000,000. All 9
  of the baseline's causal links are present in both resumed sessions (by link id). A broken load
  would look like the fresh baseline's first trace (size 0, 0 memories), not the resumes' (100, 3).
- **~3 memories per turn on resume: shown on every resume turn observed, but on 1 turn per phase.**
  Phase 2 surfaced 3 on its only turn, and so did phase 3. August showed this on 8 of 8 turns. The
  disclosed early resume ran 2 turns and logged 3 enrichment traces (3, 0, 3); the 0 is a trace with an
  empty goal. Note that 3 is the enrichment cap (`bio_enrichment` keeps the first 3), so "3 per turn"
  means the cap was filled from carried memories, not a relevance measure (true of August too).
- **No negative transfer: no evidence of dominance, on thin evidence.** In phase 3 the agent took
  one action (`sense_tools`). "Dungeon" or "guard" appears once in its whole log, in the narrator's
  summary of the resumed session; none of the phase's 36 new memories mentions either. But the 3
  memories surfaced into the garden prompt were necessarily carried dungeon-session memories: when that
  trace fired, the store held only the 100 carried ones. Their text is not logged. One action is too
  thin to test negative transfer.
- **Link accumulation**, which August reported (observation counts growing), is not shown: each run
  was too short for a link to be observed twice.

## Verdict (owner decision, 2026-09-27)

**MAINTAINED, narrow** (independently read before it landed). The persistence mechanism behaves as the
row claims, and the schema-change fields round-trip unchanged: an exact store reload and ~3 memories
surfaced on every observed resume turn. The per-turn half of the gate stands on one
turn per phase, not eight, because every run stopped early on D13. The shorter runs (1–3 turns
today, against 8 in August on the same model and settings) are tracked as their own issue.
