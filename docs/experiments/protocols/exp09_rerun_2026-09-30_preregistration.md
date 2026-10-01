# Exp 09 re-run, O19 (pre-registered 2026-09-30)

**Scope:** `rerun_exp09_o19`

- **Ledger row:** T3-9 (narrative percept reflexes), [behavioral_graduation_candidates.md](../../plans/behavioral_graduation_candidates.md).
  STALE since 2026-09-30 (M1b PR 3, owner decision): the 2026-09-24 re-run ended `planning_failed` after 5 of 8
  turns, a typed abort, which cannot back a status. This re-run is `outstanding.md` O19, and it blocks the 1.3.2 cut.
- **Claim re-tested (the row, verbatim):** "Reflexes fire below deliberation; shape learned avoidance over repeated
  exposure (habituation + sensitization)", metric "reflex fires on keyword match + habituation/sensitization
  trajectories measured".
- **Preconditions:** [#935](https://github.com/dennys246/Maxim/issues/935) fixed and the O19 harness merged.
- **Ceiling, stated before the run:** **T3-9 can reach at most PARTIAL on this code**, because H3 is not measurable
  (below; [#1026](https://github.com/dennys246/Maxim/issues/1026)). PARTIAL does not block a release; STALE does.

## Gate: the ORIGINAL H1–H7 (owner decision 2026-09-30), as numbers

The 2026-04-25 doc ([09_percept_reflex_poc.md](../09_percept_reflex_poc.md) §Success criteria) states H1–H7 as pass
conditions. The owner chose the original criteria, not the post-#870 runbook gate: a changed gate is a new
experiment. Each is a count over the attempt's committed run log (`MAXIM_LOG_FILE`). A hypothesis the run cannot
satisfy as written is NOT MET; one the bytes cannot measure is NOT MEASURED. Neither is reinterpreted.

One session, as in the original protocol (unchanged): goal *"You are an adventurer in a dark cave. A dragon attacks
you repeatedly with claws and fire breath. The dragon roars deafeningly. It slams you against the wall. A freezing
wind blows through the cave."*, `--embodiment bodies/base_humanoid`, `--sim-max-turns 8`, `MAXIM_SUBSTRATE_PATH=1`.

| H | Original pass condition (verbatim) | As a number over the run log |
|---|---|---|
| H1 | "At least one `sim_enrichment("reflex", ...)` log with `attack_flinch`" | ≥ 1 `sim_reflex` record with `reflex == "attack_flinch"` (stricter: since #870 a `sim_reflex` is emitted only for a firing that acted) |
| H2 | "Damage logs targeting at least 2 different components (torso + legs)" | reflex-sourced `sim_sem_damage` records (`source=reflex_…` in the message and `agent_id == "sim_aut"`, since reflex damage runs in the agent's context) name ≥ 2 distinct components (`component damage: <component>.integrity`), one of them `legs`. Damage the narrator issues (`damage_component` with another source) does not count |
| H3 | "Pain signal with `source` containing `reflex`" | **NOT MEASURED.** `DamageComponentTool` hard-codes the published pain signal's `context["source"]` to `"damage_component"` (the reflex tag sits in `failure_mode`), and the reaction log keeps only `pain_detector:external_signal`. No committed byte carries a published pain signal's reflex source (#1026) |
| H4 | "Damage amounts show decreasing trend (3+ data points)" | the `sim_reflex` `attack_flinch` `intensity` sequence (3 dp; the repeated attack narration) has ≥ 3 points and its last is below its first |
| H5 | "At least one reflex firing with sensitization_factor > 1.0" | for ≥ 1 `sim_reflex` record, the LOWER BOUND of the reconstructed factor, `(intensity − 0.0005) × (1 + 0.3·n) / raw_intensity`, exceeds 1.0 (computed exactly, as a fraction of the logged decimals, never in floating point), where n is the number of earlier `sim_reflex` records of the same reflex (habituation's exposure count). `intensity` is logged to 3 dp, so the bound subtracts the full rounding error before amplifying it; pre-emption and the 1.0 clamp only bias the estimate low |
| H6 | "Zero matches for `auto_attack` or `auto_damage` in JSONL" | 0 occurrences of `auto_attack` or `auto_damage` in the run log. (This counts the strings the original names; it does not by itself show that all damage comes from reflexes.) |
| H7 | "`sim_enrichment` logs with `reflex` substring present" | ≥ 1 `sim_enrichment` record with `system == "reflex"` (faithful to the original; it also fires when 0 reflexes acted) |

**Disclosed before the run, so they are not mistaken for results:**
- **H2 depends on the narrator.** `impact_brace` (legs) fires only if a percept the agent receives contains one of
  its keywords ("slam", "fall", "crash", "impact"). The narrator forwards narration that may or may not include
  them; in the 2026-09-24 re-run none did, and H2 was NOT MET.
- **H1 and H4 firings may come from the agent's own reasoning,** because the deliberation enrichment path also
  evaluates reflexes, not only narration.
- **H4 depends on narration wording:** intensity words (e.g. "massive") scale the raw intensity per percept.

**Complete-attempt condition:** the session's `report.json` reads `finish_reason == "max_turns"` and `turns >= 8`;
`working_tree_dirty_src_scripts == false`, `code_changed_during_run == false`, and its `code_tree_sha256` equals its
end digest and the harness's own, none `unknown`; the report stamps the language and AUT profile `mistral-7b-instruct-v0.2` (the name `llm.profile mistral-7b` normalizes to; the router stamps the normalized name — amended 2026-09-30, before any data, owner decision) and
`n_ctx` 8192 (`configured_n_ctx_source == "config"`), its `goal` equals the goal above verbatim, the harness row's
recorded argv and environment (`--interactive false`, `--embodiment`, `--sim-max-turns 8`, `MAXIM_SUBSTRATE_PATH=1`,
and the original command's `MAXIM_BACKEND_TRACE=1`; beyond the harness's own data home, run id, log file and
`MAXIM_LOG_FILE_MAX_BYTES=0`, no other `MAXIM_*` key reaches the sim) match, its report and run
log were copied, and the model the harness read from the server's `/v1/models` while the sim ran matches the profile's GGUF
(`_served_model_matches`).

## Attempts and the verdict (owner decisions 2026-09-30)

- **Stop rule.** At most 3 attempts; an aborted attempt is recorded and may be retried; **the first complete
  attempt decides**, and the harness refuses a new attempt once the file holds a complete one.
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
  marker match or the ordering check fails, the verdict refuses: no status change, the row stays STALE.** After 3 aborts T3-9 stays STALE and the cause is investigated. The verdict reads the whole rows file and names the deciding attempt.
- **Mapping.** The row's own metric is H1 (fire on keyword), H4 (habituation), H5 (sensitization) and H6.

| Verdict | Condition | T3-9 |
|---|---|---|
| `PASS` | complete, and all seven H PASS | **MAINTAINED** (unreachable on this code: H3 is NOT MEASURED) |
| `PARTIAL` | complete, H1, H4, H5 and H6 PASS, and any of H2, H3 or H7 is NOT MET or NOT MEASURED | **PARTIAL** at the run date, naming them |
| `FAIL` | complete, and any of H1, H4, H5 or H6 NOT MET | **BROKEN** |
| `ABORT` | no complete attempt within 3 | no change: T3-9 stays STALE |

Pass sets per target status (for the M1b evidence gate): `exp09_verdict` → `{PASS}` supports MAINTAINED; `{PARTIAL}`
supports a move to PARTIAL (and nothing higher). A `FAIL` or `ABORT` verdict supports neither.

## Apparatus

- **Box:** big-mac-mini, quiet (Paper servers, Minecraft bridges and ollama stopped; operator-attested — the harness
  stamps the hostname). One harness, no second LLM consumer.
- **Model:** `mistral-7b-instruct-v0.2.Q4_K_M.gguf` (`llm.profile mistral-7b`) at `llm.n_ctx 8192`, set through
  `maxim config` and checked from the report.
- **Code:** a commit on `main` with this prereg and the O19 harness merged, clean tree, a worktree pinned at it.

## Command

```bash
python scripts/o19_rerun.py --exp 09 --write-experiment-results            # one attempt; re-run to retry an abort
python scripts/o19_verdict.py --exp 09 --data docs/experiments/data/rerun_exp09_o19/rows.jsonl \
    --json docs/experiments/data/rerun_exp09_o19/verdict.json --write-experiment-results
```

The harness runs one attempt per invocation from a fresh data home, hands the sim the harness run id, records the
attempt as a row (aborts included), and copies the session directory and its run log into
`docs/experiments/data/rerun_exp09_o19/` with the measured files' SHA-256 in the row. The verdict stamps this
prereg's blob SHA (at the executed commit and at its own) and the SHA-256 of its own source.
