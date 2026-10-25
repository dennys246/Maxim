# Baseline floors B1: the `b1_verdict` evidence entry, its complete-run rule and the BF2 obligation (approach note)

> **ARCHIVED (2026-10-24 plans archive):** Design pass complete; built in #1245. The BF2-obligation rule now lives in docs/agents/simulation-experiments.md; `lint_bf2_obligation.py` keeps citing it as its spec.

> **Approach note v5, 2026-10-11**, for [baseline_floors_b1_build.md](../baseline_floors_b1_build.md) PR-C.
>
> **Why it got a design pass.** It changes a gate: the merge-base pass table
> (`docs/experiments/evidence_pass_table.json`), a new complete-run rule in `scripts/_evidence_records.py`, and a new
> obligation lint. So it got an adversarial design pass BEFORE code (CLAUDE.md "Faster without lowering the bar" (1)).
>
> **Status: design pass COMPLETE.** Five rounds found no DO-NOT-BUILD; each round's real gaps are folded here
> (§Review record). The last round's findings were all fail-closed and in covered classes, so no further round is owed
> (the stop rule). Owner decisions D1–D2 were taken 2026-10-11.

## 1. The pass-table entry

```json
"b1_verdict": {
  "rows": ["T1-13"],
  "scopes": ["narrow"],
  "targets": {"PARTIAL": ["FLOOR_MATCHED_Q", "FLOOR_MATCHED_CHANCE"]},
  "require": {},
  "complete": {"rule": "b1_campaign", "sets": {"<arm>": [11, ..., 18] for each of the five arms}}
}
```

**What the entry supports.**
- **A floor-matched verdict moves T1-13 to `PARTIAL (narrow: floor matched)`** (owner rule BF2).
- **The move re-dates the row.** Ledger convention: the date is when the status was SET. So the cited `b1_verdict` is
  the move's new support, and the gate judges it: ESTABLISHED, prereg PASS, its runs after the previous date.
- **Why the re-date must be stated.** Without one, an EARNED → PARTIAL move with an added qualifier needs no new
  support (`needs_new_date` is false for a demotion, and `qualifier_widens` is false for an added qualifier). The
  BF2 lint below (§3) also requires the re-date.
- **A raise from STALE or BROKEN straight to `PARTIAL (narrow)` on a floor-matched verdict is allowed.**
  `support_problem` never checks the old token. That is acceptable: the verdict's FEAR sanity gate (≥ 6/8) means FEAR
  was re-measured in the same campaign.

**What it does not support.**
- **`FLOORS_BEATEN` supports nothing.** The row stays as it is, with a "floor: beaten" annotation in prose (or "floor
  under-explored", when the verdict carries that).
- **`INCOMPLETE` supports nothing.**
- **No B1 verdict can support EARNED.** A ruled kind has exactly one scope, and EARNED is the unqualified claim. The
  way back up is §3's clearing rule plus an ordinary Exp 60 verdict.

## 2. The complete-run rule `b1_campaign`

**What it builds on.** `_campaign_rule("seed", "row", allowed={"row", "session", "attempt_start"},
required={"session"})`. That already checks:
- one campaign per file;
- the keyed rows' (arm, seed) set equals the frozen sets;
- no key has two clean rows;
- the verdict's scope and campaign id match the file.

A probe confirmed it accepts a `harness_header` carrying `kind: "session"` alongside the rows.

**What `b1_campaign` adds**, each check derived from the bound bytes:
1. **Kinds are tied to record kinds.** Exactly one `session` row, stamped `harness_header`. Every `attempt_start` is
   a `harness_header`. Every `row` is a `harness_row`. (v1 probe: a `kind: "row"` line stamped `harness_header`
   counted as a present clean key.)
2. **The session header is coherent with the rows.**
   - its `run_ids` keys equal the frozen arms;
   - `diagnostic` is `false`;
   - every row's `run_id` equals `run_ids[row.arm]`. This gives one `run_id` per arm, and one arm per `run_id`.
3. **Every row's fields are well formed.**
   - `type(attempt) is int` (a bool is an int, so a plain isinstance check would let one through);
   - `refusal_cause` is in `{null, "instrument", "behavioural", "unrecorded"}`;
   - `refusal is None` exactly when `refusal_cause is None`.
4. **Attempt numbering.** Per key, sorted by `attempt` (never by file order), the attempts are exactly 1..n, with n
   at most `MAX_ATTEMPTS = 2`, a constant in the rule.
5. **Reruns.** Every non-final attempt has `refusal is not None` and `refusal_cause == "instrument"`. This rules out a
   rerun after a clean row, a rerun after a behavioural refusal, and a rerun after an `unrecorded` one.
6. **Lost attempts (owner decision D2: consumed, never rerun).**
   - Each `attempt_start` header carries `campaign_id`, arm, seed and attempt.
   - Per key, the starts are exactly 1..m with m ≤ `MAX_ATTEMPTS`, and each start has exactly one row with the same
     (arm, seed, attempt).
   - A start with no row fails the file. So does a row with no start.
   - The harness never leaves an orphan behind. On `--resume`, it appends a row for every orphan start: the same
     (arm, seed, attempt), a refusal, and `refusal_cause: "unrecorded"`. That cause is never rerun-eligible, so the
     key ends refused.
   - It writes those closing rows right after the provenance stamp and `append_refusal`, so a resume on a changed
     tree is refused and its orphan fails the file. It adds each closing row to its in-memory rows before
     `attempts_done` runs.
   - Consequences:
     - the file still has every key present (`_campaign_rule`);
     - a gated arm that lost a key is INCOMPLETE through the verdict's existing `too_few_clean_seeds`;
     - a deleted row is detectable (its start has no row);
     - a crash at attempt 1 and a crash at attempt 2 count the same way.
   - The gate never counts an orphan start as a present key. That would let someone delete a bad PRIOR or RAW row
     and keep its start.

**The sets name all five arms × 11–18 (40 keys).** Every key must be attempted; a refused row counts. That way a
reported arm (PRIOR, RAW) can never be left out of the file. The verdict's clean-seed rule still decides what is
gated.

**The pin test** ties the table's sets to `b1_floor_run.ARMS` × `SEEDS`. It also pins `MAX_ATTEMPTS` to
`b1_floor_run.FROZEN["max_attempts"]`, and the cause vocabulary to the harness's.

**The harness change.** The harness:
- writes the `attempt_start` header before each attempt;
- on `--resume`, closes every orphan start with an `unrecorded` row;
- makes `attempts_done`, `_after_attempt` and `compute_verdict` read the same rule, with `unrecorded` never
  rerun-eligible. Tests pin all three to the gate rule's predicates.

This lands in PR-C, because #1238 is open and frozen. It is safe because no B1 data can exist before PR-C merges:
the prereg is frozen in PR-C.

## 3. The BF2 obligation lint (owner decision D1; a partial mechanization of M46)

**Rule.** If the newest committed `b1_verdict` has a `FLOOR_MATCHED_*` value, then T1-13's HEAD status must satisfy
one of two conditions:
- rank < 2; or
- `PARTIAL` with qualifier head `narrow` AND a date at or after that verdict's data time. An unqualified PARTIAL is
  a wider claim and does not count.

**Definitions.**
- **Discovery:**
  - The lint scans the tracked data root at HEAD AND at the merge-base for records with `record_kind: "verdict"`
    and `kind: "b1_verdict"`.
  - **B1 verdicts are append-only by identity.** A verdict's identity is the pair (`data_sha256`, verdict value).
    Every pair present at the merge-base must still be present at HEAD, at any path under the data root, or the lint
    fails. A verdict moved outside the data root counts as missing; a moved file inside it keeps its identity.
    Deleting a verdict, or rewriting its value (for example to INCOMPLETE over the same data), fails in the PR that
    does it. That closes the two-step route: delete the verdict in one PR, re-raise the
    row in the next.
  - No redaction path exists for B1 today. A well-formed redaction record needs an O19 campaign key
    (`redaction_record_problem`), so a B1 verdict cannot be retired. Adding that route is a future gate change.
  - Identity leaves out the other fields (`data` path, `scope`, `mock`, `status`), so those can change while the
    pair is kept. That is harmless here, and only because floor-matched verdicts count whatever their status, and a
    clearing `FLOORS_BEATEN` is judged by `judge_entry`. M46 must not inherit this as an unstated assumption.
  - INCOMPLETE verdicts are append-only too, so a stray smoke verdict under the data root can never be deleted.
  - **Escape from a mistaken floor-matched verdict.** There is no retirement route today, and no exception record
    reaches this lint:
    - `ledger` clauses act only on row support;
    - `superseded` works only because it lowers the row's rank;
    - `redaction` needs an O19 campaign key.

    The only way out is a fresh campaign (a new data sha) whose ESTABLISHED `FLOORS_BEATEN` becomes the newest
    verdict. A bug in the verdict code over good data cannot be fixed by re-judging, because a second value over the
    same sha fails the grouping. That costs rig time. A B1 retirement route is backlog row **M50**.

    One more state has no escape: a floor-matched verdict whose data is lost after merge stays red permanently, and
    a fresh campaign cannot clear it. A green merge cannot reach that state, because the data PR must carry the data
    and deleting the data fails in the PR that deletes it. Only an admin bypass gets there. Recorded under M50.
  - The lint never relies on what the ledger row happens to cite.
- **Data time:** the latest `ts` over the `harness_row` lines of the bound data file, read from the git tree at that
  revision, never the working tree.
  - **The bytes must match the verdict's binding.** PR-C extracts `judge_verdict`'s data binding into one shared
    helper, used by both the gate and the lint, so the two cannot drift. The helper checks, in order:
    - the path is under the data root, with no `..`;
    - it is not a retired path;
    - it is tracked at the ref and is not a symlink;
    - sha256 of the raw blob equals `data_sha256`.

    Only then are the bytes decompressed and their `ts` read. Any failure makes the data unreadable, and the lint
    fails.
  - **Scope of the read:** counted verdicts only, after INCOMPLETE is dropped. A stray INCOMPLETE whose data was
    never committed therefore cannot keep the lint red.
  - Floor-matched verdicts are counted without being judged, so this check is the only thing stopping someone from
    editing a verdict's data timestamps backward so that an older `FLOORS_BEATEN` becomes the newest. A test pins
    it.
  - This is deliberately later than the gate's own time (the earliest unit time). An `unrecorded` row's resume-time
    `ts` pushes it later still, which is stricter.
  - Verdicts are ordered as "newest" by that float. Only exact equality is a tie.
  - Data time is compared as a UTC calendar day only against T1-13's ledger date.
- **Which verdicts count, and which is newest:**
  - Floor-matched verdicts count whatever their status. They fail closed: a sloppy record cannot drop the
    obligation.
  - `FLOORS_BEATEN` counts only when ESTABLISHED with prereg PASS. A clearing verdict is support, so it is judged.
    The lint judges it with the gate's own `judge_entry`, using a `Ctx` built by a `build_ctx(repo, base)` helper
    extracted from `lint_evidence_gate.gate()`. It never copies the gate's code.
  - `INCOMPLETE` is ignored.
  - INCOMPLETE is dropped before grouping. The remaining verdicts are grouped by `data_sha256`. Re-running
    `verdict` on the same data gives one group: identical values count once, and differing counted values for one
    data file fail. So once a floor-matched verdict exists, no other counted value over the same sha can ever land.
  - Two different data files with the same data time fail.
  - An unreadable record fails.
- **Cleared:** the newest counted verdict is `FLOORS_BEATEN`.
- **Qualifier and rank:** the predicate is `L.qualifier_head(q) == "narrow"` on token `PARTIAL`. An unparsed
  qualifier fails. RE-VALIDATED-BY-TESTS and LEGACY, which are also rank 2, do not satisfy it. The token check is
  explicit.
- **The data PR carries the demotion.** The floor-matched verdict and T1-13's move land in the same PR. The lint
  passes then, because HEAD sees both. It fails if the verdict lands first, and a test pins that branch.
- **Git plumbing:** the lint uses the gate's `base_ref`, `must_not_skip` and `GitUnavailable` pattern. A push compares
  against the last green push, and a shallow clone is an error, never a skip.
- **Routes around the lint:**
  - Superseding T1-13 needs an owner exception record, which is the sanctioned override.
  - A new row making the same claim is M46's scope.

**What it closes.** v1's S1. After a floor-matched demotion, a FEAR-only Exp 60 verdict, such as Amendment 9's
scheduled re-run, could otherwise support PARTIAL → EARNED, undoing BF2 without any floor being run.

**Where it runs.** In the lint job, as `scripts/lint_bf2_obligation.py`, with a test file that drives each branch.
The obligation is pass-table data: an `obliges` block (`{row, values, clears}`) on the `b1_verdict` entry, read from
the MERGE-BASE like the gate's own rule data, so a PR cannot loosen its own obligation (code review, PR-C). The
evidence gate checks the block's shape (`obliges_problem`), and a block on main may only tighten
(`obliges_change_problems`: no removal, no dropped obliging value, no added clearing value, no re-pointed row), so the
rule cannot be disarmed in two steps either. M46 generalises it.

## 4. Every input, and how each could lie

| Input | How it could lie | Answer |
|---|---|---|
| The pass-table entry | A PR loosens its own sets or targets | The gate reads the merge-base table. The entry lands on main in PR-C, before any data. The pin test catches drift. |
| `COMPLETE_RULES["b1_verdict"]` | A later PR drops or weakens the rule | `complete_problem` fails a `complete` block that has no rule. The rule code is gate code, under review; its tests drive every refusal. |
| `arm` / `seed` | Seed 99 or a typo'd arm in the gated file | Outside the frozen sets → fail (`_keyed`, int seeds). |
| `record_kind` vs `kind` | A row stamped as a header, or the reverse, slips past the counting | Rule item 1. |
| The session header | Missing, duplicated, diagnostic, or its `run_ids` not the rows' | Rule items 1–2. |
| `attempt` | Gaps, duplicates, a bool, a third attempt | Rule items 3–4. |
| `refusal` / `refusal_cause` | Inconsistent, or "clean" with a cause | Rule item 3. |
| A crash mid-attempt | Resume restarts the key at attempt 1 after the outcome was seen | Rule item 6 and the harness's `attempt_start`. |
| `refusal_cause` label | A behavioural refusal labelled "instrument" to buy a rerun | **Not caught.** The label is writer-stamped by reviewed code: `run_arm_seed` classifies by exception class, and the verdict lists every cause and step. Backlog row **M48**. |
| Rows from another code tree | Mixed after a fix | `append_refusal` refuses at write time; the gate's one-code-tree judgment refuses at the gate. |
| The verdict record | A wrong value, or computed over other rows | The value is trusted from its stamped writer. It is bound to the data by path + sha256. Scope `{"all_rows": true}` with no `campaign_id`. |
| The ledger qualifier | PARTIAL at another scope, or unqualified | Exactly `narrow` (`support_problem`), and §3. |
| Timing | Data predates the prereg | The prereg lint and the gate's `prereg == PASS`. |
| A second campaign file | Run campaigns until one looks good, commit only that one | **Not caught:** the gate cannot see files that were never committed. The process rule is that every campaign file started is committed. Backlog row **M49**. |
| A deliberate kill | Kill the run after seeing a bad gated-arm result, so the key turns `unrecorded` and the verdict becomes INCOMPLETE, which avoids the demotion | **Not caught.** The kill cannot buy a rerun, and it shows up in the verdict's `refused` list. It is no stronger than not committing the file. Recorded under **M49**. |
| A later Exp 60 verdict | Re-earns EARNED over a floor-matched B1 | §3. |
| Deleting or rewriting a B1 verdict | Drops the obligation in one PR, then re-raises the row in the next | Append-only by (`data_sha256`, value) (§3). |

## 5. What is not changed

- `seeds_per_run_arm`, `_campaign_rule` and the Exp 60/61/62 entries are untouched. The v1 probe confirmed the shape
  checks accept a new ruled kind with scope `["narrow"]`.
- No exception record is added, and `require` stays empty.

## Review record

**v1, 2026-10-11: one adversarial pass, with probes of `_campaign_rule`, `counted_rows` and `judge_row_file`.**

No DO-NOT-BUILD. Folded:
- **S1:** a FEAR-only Exp 60 re-run could undo BF2. → §3 (D1).
- **S2:** the obligation lint was underspecified. → §3's definitions.
- **S3:** a crash was an unrecorded attempt. → rule item 6 and `attempt_start` (D2).
- **S4:** item 1 was missing from the base rule; kind ↔ record_kind; header coherence. → items 1–2.
- **S5:** exact predicates. → items 3–4 and the pins.
- **S6:** the blind spots needed backlog rows. → M48, M49.
- **N1:** a demotion does not re-date by itself. → §1 states the re-date, and §3 enforces it.
- **N2:** STALE → PARTIAL is allowed. → stated in §1.
- **N3:** the annotation is not re-checked. Agreed.
- **N4:** the 40-key requirement stays.

**v2 delta round, 2026-10-11.** No DO-NOT-BUILD. Folded:
- **S1:** an orphan start voided the file, the same crash counted two ways, and the harness did not implement D2.
  → item 6: an explicit `unrecorded` row on resume, one row per start, never rerun.
- **S2:** deleting the verdict bypassed the lint. → discovery at the merge-base and HEAD, redaction-aware, identity
  by `data_sha256`.
- **S3:** clearing was unguarded, and a regenerated verdict tied. → clearing needs ESTABLISHED; grouping by data
  sha.
- **F1:** data time was undefined. → defined.
- **F2:** the demotion must land with the verdict. → stated, with a test.
- **F3:** the qualifier predicate. → stated.
- **F4:** starts carry `campaign_id`. → item 6.
- **N1:** starts are not compared by code tree; `append_refusal` covers that at write time.
- **N2:** routing around the lint by superseding (needs an owner exception) or by a new same-claim row (M46's
  scope). → stated in §3.
- **N3:** a forged start cannot satisfy item 6 once each start needs exactly one row.

**v3 delta round, 2026-10-11.** No DO-NOT-BUILD. Folded:
- **S1:** base-and-HEAD discovery bound the obligation for one PR only, so a two-step delete-then-re-raise worked.
  → B1 verdicts are append-only by (`data_sha256`, value).
- **N1:** redaction is unreachable for B1. → stated.
- **N2:** ESTABLISHED is checked through the gate's `judge_entry` and a `build_ctx` helper. → stated.
- **N3:** git plumbing. → stated.
- **N4:** data time vs the gate's time; float ordering. → stated.
- **N5:** where the resume closing rows are written. → item 6.
- **N6:** the in-memory rows after closing. → item 6.
- **N7:** a deliberate kill acts as a veto. → §4 and M49.

**v4 round, 2026-10-11.** No DO-NOT-BUILD. Folded:
- **S1:** the data-time bytes were not tied to `data_sha256`, so backdating the data could make an older clearing
  verdict "newest". → bytes must match the binding, with a test.
- **S2:** the escape from a mistaken verdict was undocumented. → stated in §3, plus backlog row M50.
- **N1:** "any path" meant any path under the data root. → stated.
- **N2:** identity omits the other fields. → stated.
- **N3:** INCOMPLETE verdicts are append-only too. → stated.
- Grouping order: INCOMPLETE is dropped first. → stated.

**v5 delta check, 2026-10-11.** It closes the backdating route. Stop: every finding is fail-closed and in a covered
class. Folded as wording:
- reuse `judge_verdict`'s binding through one shared helper;
- the data-time read covers counted verdicts only;
- the hash is over the raw blob;
- lost data after merge is reachable only by an admin bypass (M50).
