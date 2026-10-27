# M60a: local floors are notes, and pins come from CI's artifact in one command (approach note)

> Gate approach note (CLAUDE.md "Faster without lowering the bar" (1)). Status: **v3.1, narrowed**, with the
> [delta2](reviews/m60/delta2.md) folds: P1, slack is `K_MIN`, not k; P2, the pinner does not require that main's
> new files exist; P3, the pinner reports branch pins CI will refuse; plus a workflow and event check.
>
> Owner decisions on 2026-10-27:
> 1. Do M60.
> 2. After two adversarial rounds, ship the safe half now and make the "compare to main" fall check its own item.
>
> The two rounds:
> - [reviews/m60/adversarial.md](reviews/m60/adversarial.md) rejected v1, which demoted a rise to a note
>   (D1: a rise followed by a fall goes unseen; D2: stale missing pins).
> - [reviews/m60/delta1.md](reviews/m60/delta1.md) rejected v2, the unchanged-lines comparison against main
>   (X1: touched uncovered lines become headroom; X2: the diff and the baseline are taken at different commits).
>
> Both files are the requirements list for **M60b**.

## What changes (and what does not)

**CI's floor rules do not change.** A rise of more than band still fails, and so does a fall of more than k. Every
pin rule (NEW, CHANGED, LOWERING, the split rule) still gates. The one exception is in rule 2 below. The time this
saves comes from removing manual steps, not from loosening any gate.

1. **`--local` gates no floor.** This is the pre-push run in `scripts/suite_at_commit.sh`. Today it gates the floors
   of the packages a diff touches, but this Mac reads 1–2 pt off Linux, so those results are false reds or false
   greens. After the change it prints every floor result, in both directions, as a note. It still gates:
   - diff coverage (80% of changed lines);
   - `pin_rises`, which reads no measurement;
   - the floors file's format and fixed fields;
   - the config and exclusion rules.

   The v1 pass checked the history: #1201 and #1240 were rises, #1270 stays gated, and #1213 and #1232 were diff
   coverage. `local_touched_packages` has no other caller, so it is deleted along with its tests.
2. **A pin may sit up to `K_MIN` above the measured `missing` (S3 of delta1).** In the CHANGED rule, the cap becomes
   `min(m.missing + K_MIN, b.missing)` instead of `min(m.missing, b.missing)`. `K_MIN` is 5 statements, the absolute
   noise. It is not the size-scaled k, which is 96 statements on `overall` (delta2 P1). The rule is shared through
   `changed_entry`, so file floors and rename targets get it too. The reason: a pin taken from run A is
   verified on run B, the pin commit's own run, and runs differ by ±2 statements. Without the change a correct pin
   fails on noise about half the time.
   - **Bound.** The base side stays strict: a pin can never rise above the previous pin. A pin is therefore at most
     `K_MIN` above the lowest measurement it was taken from, and that slack cannot accumulate across pins.
   - **What it costs.** The LOWERING rule reads that pin. A later PR can therefore lower a floor with up to `K_MIN`
     more uncovered statements than the measurement allowed, a 5-statement escape in total, which does not
     accumulate. NEW floors stay exact (two runs, `--merge-floors`).
3. **The pinner: `python scripts/lint_coverage.py --pins-from-run <run-id>`.** It replaces reading CI's log and
   hand-editing the file.
   - **Refuses unless:**
     - the run is a Tests (`test.yml`) run of a `pull_request` or `push`;
     - the run's `head_sha` equals the checkout's `HEAD`;
     - the jobs API reports the run's latest attempt;
     - in that attempt, the `unit-tests` job's "Run required fast suite" step concluded `success`, so no partial
       `-x` data is used.
   - **Reads** that attempt's artifact, `coverage-json-<attempt>`; the upload step is renamed to bind the artifact
     to its attempt (S1 of delta1). It accepts the legacy name `coverage-json` only on attempt 1.
   - **Normalises** the runner's absolute paths to `src/maxim/...`. A PR run measures the merge with main, so a
     measured file this branch lacks is accepted. Those pins are for the merge, and CI re-measures it on the pin
     commit (delta2 P2).
   - **Rewrites** only the scopes more than band above their floor, to the same pin the CI hint prints:
     `{round_down(m.percent), min(m.missing, f.missing)}`. It prints each change, and the file stays byte-stable
     JSON with indent 2. Scopes below their floor, NEW floors and nulls are left alone and reported. So is a pin already on the branch
     that sits more than `K_MIN` above the run's measurement, which CI will refuse (delta2 P3).
   - **Fails closed** when `gh` is missing or any API call fails (exit 2).

## Inputs and how each could lie

1. **The run id the operator gives.** The run is bound by SHA, attempt and step conclusion. A wrong run is refused.
   A right run of a merge ref is what CI measures, and CI re-verifies the pin commit on its own run anyway.
2. **The artifact.** It is bound to its attempt. A partial artifact from a failed attempt is refused by the step
   check.
3. **The local measurement.** It no longer gates floors in either direction. CI remains the only floor authority, as
   the brief already says.
4. **The CHANGED `K_MIN` slack.** Bounded at `K_MIN` per pin (rule 2). It never applies to NEW floors. LOWERING and
   the split and rename SUM rules read the pin, so a sum over N re-pinned scopes can absorb `K_MIN` × N. That residual
   is now stated in the lint's docstring (code review A3).

## Not in M60a

- A rise above band still fails on a PR, and costs one CI round trip plus one pinner command.
- The fall-against-main check is M60b.
- Local Linux measurement (#1313, H) is the bigger time win.
- Deterministic coverage (#1103, #1359) unblocks xdist (M22) and tighter tolerances.

## Wiring and tests

- **`scripts/lint_coverage.py`:** `main()`'s `--local` branch; `changed_entry`'s cap; the new `--pins-from-run`
  mode; the docstring.
- **`.github/workflows/test.yml`:** the artifact name gets `-${{ github.run_attempt }}`.
- **`tests/unit/test_lint_coverage.py`:**
  - the `--local` tests are re-pointed: a touched package's breach becomes a note, diff coverage still gates, and
    `pin_rises` still gates;
  - the +k cap passes at measured + k, fails at measured + k + 1, and LOWERING stays exact;
  - the pinner runs against a stubbed `gh`, with every refusal branch, path normalisation and the rewrite;
  - deletion probes.
- **Docs:** `docs/agents/ci-testing.md` (the PR-fails bullet and the `suite_at_commit` paragraph); outstanding.md
  (M60 split into M60a done and M60b open, plus the M22 note).
