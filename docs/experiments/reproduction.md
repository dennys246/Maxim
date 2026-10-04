# Reproducing an experiment: what the O19 re-runs taught us

Who this is for: anyone who re-runs a graduated experiment to re-validate a ledger row: a heartbeat, a trigger re-run, or a release gate. It is distilled from O19, the Exp 10 and Exp 09 re-runs that block 1.3.2, and the M1 evidence-gate work around them (2026-09-27 to 2026-10-02). Each lesson says what happened, the rule it left, and the check that enforces it. A rule with no check is a row in the mechanization backlog ([outstanding.md](../plans/outstanding.md)).

The short version: **a re-run is a new experiment, and its result counts only if the bytes on `main` can show which code ran, that every attempt was declared, and that the instrument worked.**

**Scope of the enforcement.** The checks named below are built into the O19 harness and verdict (`scripts/o19_rerun.py`, `scripts/o19_verdict.py`), which today cover only Exp 10 and Exp 09. For any other experiment the rules hold, but they are followed by attention until a generic re-run harness exists (mechanization backlog M27 in [outstanding.md](../plans/outstanding.md)). The prereg lint and the evidence gate apply to every experiment.

## 1. A re-run is pre-registered like any experiment

- **What happened.** The 2026-09-27 Exp 10 re-run was judged after the fact against prose criteria. Every session had ended `planning_failed` after 1–3 turns, and the row was still marked MAINTAINED. The blind Codex score card graded research integrity three steps below the Claude card for it.
- **Rule.** Write the re-run's pre-registration from [protocols/TEMPLATE_rerun.md](protocols/TEMPLATE_rerun.md). It names its data directory on a `**Scope:**` line, states every gate as a number, and merges to `main` before any data exists.
- **Enforced by** [scripts/lint_prereg_precedes_data.py](../../scripts/lint_prereg_precedes_data.py): a `rerun_*` directory with no governing prereg (Scope line) merged first fails CI. That every gate is stated as a number is attention, checked in review; for O19 it is pinned by tests that read the prereg text.

## 2. A typed abort is not data

- **What happened.** `planning_failed`, `cancel` and other typed aborts (D22) were cited as if they were runs.
- **Rule.** An attempt that does not reach its turn cap with its apparatus intact is *aborted*. It is recorded and never judged. Only a complete attempt decides, and **the first complete attempt decides**: the harness refuses a new attempt once one exists, so a result cannot be chosen. After 3 aborted attempts the row stays STALE and the cause is investigated.
- **Enforced by** the complete-attempt condition (C1–C4) in `scripts/o19_verdict.py::complete_problems` and the attempt cap in `o19_rerun.py`. See also the CLAUDE.md lesson "Weak evidence never gates".

## 3. Pin the code under test, and prove it

- **What happened.** Exp 42b was retracted because the harness's `git_hash` described where the *harness* lived, not what the sub-sims *imported*.
- **Rules.**
  - The harness asserts that its interpreter imports *this* repo's `maxim`.
  - Every row stamps `executed_git_hash` and a code-tree digest.
  - A dirty tree refuses.
  - A session whose code changed mid-run is not complete.
  - The rig stays at the first attempt's commit for the whole campaign, so one campaign has one code tree.
- **Enforced by** `scripts/_provenance.py`, [scripts/lint_harness_provenance.py](../../scripts/lint_harness_provenance.py), and the verdict's `check_bound`: the prereg and both O19 scripts must be byte-identical at every executed commit.

## 4. Declare every attempt before it runs

- **What happened.** Nothing stopped an operator from running three attempts and committing the best one.
- **Rule.**
  - Before spawning anything, the harness pushes an annotated start marker `refs/tags/o19/<campaign>/attempt-<k>-<run_id>`.
  - A tag ruleset on `refs/tags/o19/**` forbids deleting or moving markers, with no bypass actors. It was created before the first marker and is never edited.
  - The verdict reads the ruleset and its history, and refuses if they changed after the first marker, if k has gaps, or if markers and rows do not match one to one.
  - A marker with no rows still counts as an aborted attempt.
- **Enforced by** `o19_verdict.py::parse_markers`, `::ruleset_problems` and `::ordering_problems`.

## 5. Each attempt lands on `main` before the next starts

- **Rule.** After each attempt, its rows and copied sessions go to `main` in a merge-committed data PR before the next marker is pushed. The rows file may only grow. Every copied file is re-hashed against its row.
- **Enforced by** `o19_verdict.py::history_problems` and `::ordering_problems`, and the harness's on-`main` refusal.
- **A closed campaign's data directory is frozen** (#1059). Once a campaign's `verdict.json` is on `main`, any add, edit, delete or rename under its data directory fails the gate, with no exception path (`_evidence_records.py::o19_closed_data_problems`). This applies to every closed campaign, superseded or not (Exp 09 and Exp 63 included). A correction is a new campaign key (§6). A forced redaction has no committed exception path yet ([#1081](https://github.com/dennys246/Maxim/issues/1081)).
- **Rig without a signing key.** The rig cannot sign commits. Its data commit is re-signed on a machine that can: `git commit-tree -S` on the same tree and parent. The rig must not `git pull` mid-campaign.

## 6. Only a stamped verdict is new evidence, and the gate re-judges it

- **Rule.** A status change cites a **verdict record**: stamped with its data's SHA-256, its judge's source SHA-256 and its provenance, and written by `o19_verdict.py --write-experiment-results`. The ledger's evidence gate re-runs the judge that wrote the verdict, bound to its data (the same script at the verdict's commit and at every commit an attempt ran on), on the committed bytes, and refuses a record it cannot reproduce. Raw rows, logs and sim reports are judged, never cited alone.
- **Enforced by** [scripts/lint_evidence_gate.py](../../scripts/lint_evidence_gate.py) and `scripts/_evidence_records.py::judge_o19`, `::o19_judge_edit_problems` and `::o19_history_problems`.
- **A judge edit keeps every old verdict ([#1050](https://github.com/dennys246/Maxim/issues/1050), fixed 2026-10-03).** The gate used to re-judge only with the merge-base judge, so any edit to `o19_verdict.py` stranded every existing O19 verdict. Now each verdict is re-judged with the judge bound to its data, and a PR that edits the judge must re-judge every existing O19 verdict to the same result (strict: new behaviour goes to a new campaign key). The scripts are still frozen while a campaign is in flight (§10), because the writer requires the same scripts at every executed commit and on `main`.

## 7. Verify the instrument before you read the subject

- **What happened.** O19 Exp 10 attempt 1 aborted with `planning_failed`. The tempting reading was "the agent under test failed". The cause was the *instrument*, the simulation narrator ([#1042](https://github.com/dennys246/Maxim/issues/1042)):
  - Its prompt hard-coded tools it does not have (`internet_search`, `write_file`, `respond`), and it proposed them.
  - The stall detector's suppression was dead: the router registered calls under the cost tier while the detector asked for the lane. Stale nudges reached a busy narrator.
- **Rule.** When a run fails, check that the apparatus did what it was meant to do before reading anything about the agent: the narrator, the bridge, the model actually served, the config actually read. Our version of the CLAUDE.md question "did the action I commanded actually happen?" is: did the probe I think was sent actually reach the agent, from a narrator that could act?
- **Enforced by** C4 (the served model, `n_ctx` and environment checked from the reports) and, for the narrator, [tests/unit/test_prompt_names_own_tools.py](../../tests/unit/test_prompt_names_own_tools.py).

## 8. Make sure the smoke run takes the path the gate reads

- **What happened.** A post-fix check ran `--sim "test basic recall"` and finished cleanly. That goal matches a built-in arc, so the CLI routed it to the generative runner, whose narrator is scripted. It never exercised the agentic narrator that broke. The Exp 10 goals match no arc and take the agentic path. A second check with the real Exp 10 goal did exercise it.
- **Rule.** A smoke run uses the protocol's own goal and settings, or it shows that it takes the same code path (`simulation/arcs.py::select_arc_for_goal` decides the routing). A clean run on another path is not evidence about this one.
- **Enforced by:** nothing yet. Mechanization backlog M26 in [outstanding.md](../plans/outstanding.md).

## 9. Configure the model in one place, and read it back

- **What happened.** The server's `n_ctx` and the prompt budgeter's belief resolve through different paths. When they drift, a sim takes 0 real actions behind HTTP 500s ([lesson](../lessons/sim-n-ctx-config-drift.md)). Before config.json 1.2, the rig's `llm.n_ctx 8192` was read back as `default`.
- **Rule.** Set the model and context only with `maxim config set llm.profile …` and `maxim config set llm.n_ctx …`, and check with `maxim doctor`. The harness reads the served model from the sim's port during the run, and every report stamps what it used. Run on a quiet box: no second LLM consumer on the sim's port ([lesson](../lessons/no-harness-on-leader-machine.md)).
- **Enforced by** C4 and the harness's model and port refusals.

## 10. When the instrument was broken, close the campaign; do not keep retrying it

- **What happened.** Campaign 1 for Exp 10 could not continue on the fix: its prereg pins every attempt to the first attempt's code tree. Retrying on new code would have mixed two code trees in one campaign. Quietly starting over would have hidden the abort.
- **Rule** (owner decisions 2026-10-01 and 2026-10-02).
  - Close the campaign with its own stamped verdict, written *before* any O19 script changes: after that change the writer cannot write it again (`check_bound` refuses), so what survives is the committed verdict, pinned by SHA-256 in its successor. The gate can still re-judge it with its bound judge (#1050).
  - Open a successor campaign with its own prereg, data directory and marker namespace. A campaign may be succeeded **only after an ABORT**: a PASS or FAIL is terminal, so a failed result cannot be discarded by opening another campaign.
  - The successor pins its predecessor's closure verdict by SHA-256 and names the owner decision and the fixed cause issue. The closure (dated by when its pinned bytes reached `main`) and the successor's prereg must reach `main` before the successor's first marker.
  - At most one open campaign per experiment, and at most two campaigns. Another ABORT leaves the row STALE.
  - The O19 scripts are frozen from a campaign's first attempt until its verdict lands.
- **Enforced by** `o19_verdict.py::protocol_problems` and `::successor_problems`, the harness's `check_campaign`, and the gate's campaign binding in `_evidence_records.py::rejudge_o19`. The record lives in [protocols/exp10_rerun_2026-10-02_preregistration.md](protocols/exp10_rerun_2026-10-02_preregistration.md).

## 11. Disclose what you know is wrong, and say why the gates do not read it

- **What happened.** The post-fix check showed the narrator sometimes executing a plan made before the agent's latest reply ([#1048](https://github.com/dennys246/Maxim/issues/1048)). It was disclosed in the campaign-2 prereg because its fix sat in code fenced until 1.3.2's decomposition. It is now fixed under an owner fence exception (the narrator keeps one planning request in flight and folds new inputs into the next follow-up), before campaign 3.
- **Rule.** A known defect in the apparatus goes in the prereg, with the argument from structure for why no gate reads it and why it cannot by itself produce the outcome a gate counts. If an attempt fails anyway, it is an abort on the record, not an excuse.

## 12. Time checks catch forgetting, not evasion

The orderings above compare the rig's clock (row `ts`, tagger dates) with GitHub's merge times. They catch a forgotten step, such as an attempt started before the last one landed, or a prereg merged late. They do not stop an operator who sets the clock. The preregs say so, and so does this page.

## 13. A fix to the instrument can touch the subject: check before calling the next campaign a re-run

- **What happened.** Exp 10's campaign 2 aborted on the narrator (#1052), and the investigation found every remaining abort path in the narrator. A third campaign on the fixed code looked like a re-run of the same claim. The design review of a REPRODUCED label (a successor's PASS after support-only fixes) then diffed campaign 1's executed commit against the fixed code. #1047, the campaign-2 fix "an agent's prompt names only tools it has", applies to every agent, so it changed the agent-under-test's prompt too. A pinned test of what the gates read cannot rule that out: a prompt change can make the agent act more, store more memories, and pass P0 and R2 while every contract test still passes.
- **Owner decision 2026-10-03 (strict).** The agent-under-test's prompt and tool dispatch count as the mechanism. A successor campaign's PASS is evidence for the row only if that code is byte-identical to the root campaign's executed commit. So Exp 10 gets no third campaign; T1-1's claim is re-tested by a new pre-registered experiment on current code ([#1060](https://github.com/dennys246/Maxim/issues/1060)), with the confounding and bio-faithful design review a derivative gets. The gate that enforces this for every successor is [#1059](https://github.com/dennys246/Maxim/issues/1059). Exp 09 is not a successor: it had no attempt yet, so its first campaign is a root on the current code (which #1047 also changed; its amended prereg says so).
- **Rule.** Before opening a successor campaign, diff the root campaign's executed commit against the successor's code and name every changed file on the subject's path, not only the files the fix was meant to touch. If any changed, the successor is a new experiment, not a re-run.
- **The label: `REPRODUCED`** (rank 3, positive; owner decision 2026-10-02). A successor campaign's verdict supports `REPRODUCED`, never `MAINTAINED`; a root campaign's supports `MAINTAINED` (where the pass table allows), never `REPRODUCED`. It means **the subject's code is unchanged since the root campaign**, not that a judged result was reproduced: the root may hold only aborts. It is a positive target, so it may retire a `SUPERSEDED` ledger row like any other; that ledger-row succession is not campaign succession.
- **The subject** (owner decision 2026-10-04): `src/maxim/**`, `pyproject.toml`, any Python lockfile, `scenarios/` and `data/`, everything the sim runs or reads. The one standing exclusion is non-runtime lint data, `src/maxim/utils/function_length_baseline.json`. Installed library versions, llama.cpp, the model weights (GGUF hash), the encoder weights and `.python-version` lie outside the repo and are not compared: a disclosed gap, mechanization backlog row M28 ([outstanding.md](../plans/outstanding.md)).
- **Enforced by the evidence gate** (`_evidence_records.py::o19_succession_problems`, the authority), for every status a successor's verdict supports, `PARTIAL` included:
  - whether a campaign has a predecessor is read from the verdict's bound judge's table, and that entry must equal the merge-base table's. It is never read from the record;
  - the campaign table is append-only for a campaign whose rows or verdict are on `main` or that another entry names in `supersedes`, and each verdict kind belongs to exactly one experiment (`o19_table_problems`);
  - the successor's bound judge emits the leaked-gate bar over each predecessor's data as `main` holds it, checked against the pinned closure and its `data_sha256`, and it is empty: no attempt of any predecessor committed `ok` phases that show a FAILED gate (`o19_verdict.py::leaked_gate_problems`; a committed phase that cannot be judged also bars the successor, the strict reading recommended by the design pass and adopted 2026-10-04). A bound judge without the bar supports nothing;
  - every commit any campaign of the chain ran on (the rows and the markers of each predecessor's pinned closure verdict, and the successor's own, aborted attempts included) exists, is on `main` and holds the same subject listing (`git ls-tree`: mode, object id and path, with literal paths; never the verdict writer's commit). A missing commit or an unreadable listing refuses;
  - the bound judges' phases, `HARNESS_ENV` and model pins (`MODEL_PROFILE`, `MODEL_PROFILE_STAMPED`, `MODEL_GGUF`, `N_CTX`: set through `maxim config`, so no argv or env shows them), and every recorded sim argv (less the `--resume-sim <id>` pair) and `MAXIM_*` env per phase, are the root's. A mechanism switched by a flag or an env var counts as a change, and a successor that needs a new harness flag is a new experiment.
- **The pin chain is immutable on `main`** (#1059 delta review), so a successor cannot be pointed at a rewritten predecessor:
  - once a campaign's `supersedes` (its predecessor and the pinned closure SHA-256) is on `main`, a change to it fails the gate. Only that object is frozen: a not-yet-run successor's prereg and phases may still be amended before data (`o19_table_problems`);
  - a predecessor's data directory is frozen once it is closed (§5). A successor's verdict also supports nothing if that directory changed on `main` after its closure landed (`closed_history_problem`, which reads first-parent history only and needs git ≥ 2.31; on older git it cannot find the closure and refuses). It covers a rewrite that reached `main` before the rule ran, and runs when a successor verdict is newly cited; rows that already cite one are held by the §5 freeze from then on.
- The harness repeats the leaked-gate bar and the subject check before each attempt's marker (`o19_rerun.py::check_campaign`), and the verdict writer repeats the bar (`check_apparatus`). A `ledger` exception clause stays the visible override.

## Checklist for the next re-run

1. The re-run prereg is merged to `main` from the template, with its Scope line, numeric gates and the stop rule.
2. The tag ruleset exists and predates the first marker. Do not edit it.
3. `maxim config` holds the model and `n_ctx`, `maxim doctor` agrees, and the box is quiet.
4. Run a smoke check on the protocol's own goal, then read its log for the instrument: the narrator's proposals, nudges, `finish_reason`.
5. Run `o19_rerun.py preflight --exp <campaign>`, then one attempt at a time. Each attempt goes to `main` in a merge-committed data PR before the next.
6. Write the verdict with `--write-experiment-results`, and cite it on the ledger row. The evidence gate re-judges it.
7. If the instrument broke: close the campaign with its own verdict, record the cause issue and the owner decision, and open a successor only after an ABORT, and only if the subject's code is unchanged since the root campaign (§13). Its verdict can support `REPRODUCED` at most.
