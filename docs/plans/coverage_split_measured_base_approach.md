# Approach note: the split rule charges main's MEASURED `missing`, not a stale pin

Status: DRAFT v3, 2026-11-25, gates lane (maxim-c9). Owner decision 2026-11-24: "split rule uses measured". The other
options were gating `missing` on every run, a one-off re-pin, or #1786's data first. This is a gate change, so it
gets an adversarial design pass before any code ([CLAUDE.md](../../CLAUDE.md), "Faster without lowering the bar"
(1)). v1's pass returned DO-NOT-BUILD and v2 folded every finding; round 2 found no DO-NOT-BUILD, its stop rule applied (every
finding fail-closed), and v3 folds its corrections (both change lists at the end).

## The defect

`scripts/lint_coverage.py` pins each floor as `{percent, missing}`. No every-run rule compares measured `missing`
with the pin: a pin is verified only when its floor is NEW or CHANGED. The docstring's own known gap says so: "the
pinned `missing` goes stale between floor changes". So `main` can drift above a pin with nothing failing, as long as
the percent holds. Example: `src/maxim/simulation/orchestrator.py` is pinned at `{76.8, 241}`, and main's last green
coverage (run 45325730732 at 2fb497a1) measures 243 of 1049.

The split rule (SF-2) and the package sum rule both charge the stale pin. The sum is measured `missing` of source plus
successors, and it may not exceed their BASE PINS, with no tolerance. So a PR that moves code out of a drifted scope
pays for drift it did not cause: #1777 covered an extra statement for it, 6cfb90b3.

## The change

In the two SUM rules only (SF-2 for files, the namespace pass's `_sum_rule` for packages), the source's allowance
becomes `max(base_pin, base_measured)`. `base_measured` is the source scope's `missing` in main's own measurement at
the base commit (the merge-base on a pull request, `_lint_git.push_base` on a push), keyed on the source's BASE path
(`f_key` at base, so a renamed source is looked up under its old name). Successors keep their base pins.

**Not changed, on purpose:**
- **LOWERING (`m.missing <= b.missing`) and the CHANGED cap keep the pin.** Giving them `max` would reopen the
  walk-down class the docstring's three review rounds closed: a floor stepped down by drift. **Residual:** a split
  that moves covered code out while the drift stays in the source is still refused by the source's own LOWERING rule.
  The remedy is to cover the drift in the split PR or re-pin first (below). #1777 escaped this residual only by luck.
- The every-run percent rule, NEW floors, the k tolerance.

**The base measurement: its own small qualifier, built from shared pieces.** `_qualify` cannot be reused: it requires
a `pull_request` event, the PR branch, an age under 48 h, a matching tree stamp, and
`floors_only_beyond(sha, pr_head)`, so it refuses every main push run. The shared pieces are `_lint_git.push_base`'s
lookup of main push runs by `head_sha`, `_unit_tests_steps`, `_artifact_names`, `_download_artifact`, and
`load_coverage(root, path)` in its foreign mode, which accepts files the checkout lacks. That last one matters
because `_validate_artifact` refuses an artifact naming files the checkout lacks, which is exactly the split-by-move
case.

A run qualifies only when all of these hold:
1. it is a `push` event on `refs/heads/main` of the Tests workflow, with `head_sha == base` exactly;
2. its `unit-tests` job's `RUN_SUITE_STEP` AND its "Coverage ratchet (roadmap 1.3.2 item 5)" step (the exact pinned
   name) are both `success` IN THE SAME attempt, mirroring `run_measurement`. A passed ratchet does NOT bound `missing`
   by the pin (that is this note's defect); it gives the percent-derived bound
   `missing <= statements x (1 - floor%) + k` (orchestrator.py: about 243 + k), and with the suite step passed the
   sample is a whole measurement (the 4 `-x` shards are reconciled as sets, so a lost shard fails the step);
3. the artifact is that attempt's own `coverage-json-<attempt>`. Other attempts' artifacts are ignored: the upload
   runs `always()`, so a failed attempt stopped by `-x` leaves an artifact with `missing` inflated by hundreds.

`base_measured` is the **minimum** `missing` over every qualifying (run, attempt) at that sha. That is the strict
choice: re-running main until a high sample appears fishes nothing. Push runs never re-use a measurement
(`reuse_decision` needs `pull_request`), so no re-use chain needs following.

## Every input, and how it could lie

| Input | How it could lie | Answer |
| --- | --- | --- |
| base sha | a stale `origin/main`; on a push, the wrong base | the merge-base SF-2 already uses (PR) or `push_base` (push); one value per event for the diff and the lookup |
| run picked | a PR run; another workflow; a failed or cancelled attempt; a partial artifact | qualifier rows 1-3 above |
| re-runs | fishing for a high sample | the MINIMUM over every qualifying attempt and run |
| artifact | a fork (artifact poisoning); another attempt's | only `push` runs on `main` qualify, and a fork cannot create one; the name must match the attempt |
| content | another coverage version or config; truncated; scope absent | `load_coverage` checks only `meta.version` and `branch_coverage`, not `[tool.coverage]` or the exclusions; harmless, because the base was measured under the base commit's own config, exactly as its pin was. Truncated or scope absent: fall back. The unsuffixed legacy `coverage-json` artifact is refused (such runs are past retention anyway) |
| renamed source | keyed by HEAD path, so nothing is found | keyed on the base path (`f_key` at base) |
| retention | artifacts expire after 14 days (`retention-days: 14`) | rarely the cause on a PR: `actions/checkout` checks out the merge ref, so `_lint_git.base_ref` is main's tip when that ref was built, not the branch point. The COMMON fallbacks: main's tip run still in flight (its artifact not yet listed, so a PR can fail now and pass on a re-run), a pending main run cancelled by the `concurrency` group (that commit never measured), and on a push a `push_base` accepted by `_accepted` with no green run. All fall back to the pin |
| intermediate commits | a multi-commit push measures only its head | a merge-base on an unmeasured intermediate commit falls back |
| API / network | down, rate-limited | FAIL CLOSED to the pin (today's rule), printed. Never looser |
| time budget | `_download_artifact` can block 300 s per call | one overall budget for the lookup, like `--reuse-decision`'s 5 minutes; on expiry, fall back to the pin |
| error routing | a `GitUnavailable` from the lookup reaching `must_not_skip`, which exits 2 on pull_request and push | the lookup catches its own errors and falls back to the pin; it never raises into the gate's skip path, so "never a red job" holds |
| no `gh` | local runs, `prepr`, `suite_at_commit.sh` | fall back to the pin: stricter than CI, and printed |
| noise | the base sample is a few statements off | bounded by the minimum over samples (an extra sample can only lower it, and only someone with write access can re-run main) and by the percent-derived bound of qualifier row 2 |
| the operator | a PR raises the pin itself | unchanged: the CHANGED rule forbids a pin rising |

Fail-closed direction: every doubt falls back to the pin, which is STRICTER. A lie can at most restore today's
behaviour; it can never make the allowance looser than main's own measured `missing`.

## Residuals, stated

- **Erosion laundered through a split, up to k per cycle.** PR A erodes scope F's tests within k, and main passes. PR B
  then splits F, is charged at the eroded measurement, and pins its successors' NEW floors at it. One PR doing A+B
  fails today. Each cycle is bounded at k and costs a real 25+-statement move under 80% diff coverage. The module
  docstring's claims ("any number of PRs together can drop at most k statements under an unchanged floor";
  "laundering test erosion through a move") are amended to say "up to k per split" in the build PR.
- The LOWERING residual above.
- Scopes that are never split keep drifting silently; gating that is the option the owner did not choose (#1786).
- Drift that landed between the base and current main is still charged to the PR: stricter, accepted.

**No stale-pin report.** v1 proposed one with a `--pins-from-run` remedy, but that command writes pins only for
scopes MORE than the band above their floor. For drift up, the CHANGED cap forbids a pin rising, so no command helps.
For drift down it only prints a note. A report naming a command that writes nothing is dropped. If wanted, writing a
LOWER `missing` is a separate small mode, as a follow-up.

## Tests the build owes (each deletion-probed)

- A drifted file source (pin 241, measured 243) passes a split charged 243 and fails at 244.
- The same holds for a package source.
- A source measured below its pin keeps the pin.
- Each qualifier row refuses and falls back, printing why: a PR event, wrong sha, a failed attempt's artifact, another
  attempt's artifact, the scope missing, the API down.
- The minimum is taken across two qualifying runs.
- A renamed source is found under its base path.
- LOWERING still uses the pin.
- A failed ratchet step on the base attempt refuses.
- A push base accepted only by `_accepted` falls back.
- A lookup error falls back with exit 0, not 2.
- The budget expiring falls back.
- The legacy unsuffixed `coverage-json` artifact is refused.
- Successors keep their base pins under `max()`.
- A renamed PACKAGE source is looked up under its base key.

## v1 → v2 (adversarial pass, 2026-11-25)

- DO-NOT-BUILD: `_qualify` refuses main push runs, so v2 has its own qualifier from shared pieces.
- DO-NOT-BUILD: "fail closed" was false for a failed attempt's artifact, so v2 requires the same-attempt suite and
  ratchet steps to pass.
- Re-run fishing: v2 takes the minimum.
- Re-use row corrected: push runs never re-use.
- The erosion residual is stated, and the docstring amendment is owed.
- The package sum is covered too, as the same root cause.
- LOWERING is excluded on purpose, as a stated residual.
- The stale-pin report is dropped.
- New input rows: retention, push base, intermediate commits, renamed source.
- `load_coverage` named correctly.
- Only the unit-tests steps are required, not the whole workflow.

## v2 → v3 (round 2, stop rule met, 2026-11-25)

- The ratchet-step bound restated as the percent-derived one, with the step's exact name.
- Retention corrected: the merge ref makes the base main's tip, and the real fallbacks are listed.
- The `load_coverage` claim corrected.
- Time budget, error routing and no-`gh` rows added.
- Seven tests added.
- The stricter charge for drift since the base is stated.

