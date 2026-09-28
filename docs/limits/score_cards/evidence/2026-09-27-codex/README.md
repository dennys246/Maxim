# Codex audit evidence — v1.3.1

Scope: `v1.3.1 @ 7e695a58cb596bafb165dec03286635826c38b59`.
Audited on 2026-09-27 America/Denver. UTC timestamps can cross into September 28.
The output branch starts at `origin/main @ 8d6a2c85c90b8341f50910eb5ecccc71e87207b6`.
No other score card or excluded issue body is in this evidence set.

## Environment and isolation

All Python was run from the firewalled tag worktree with these separate setup lines:

```bash
export PATH=/Users/dennyschaedig/Scripts/Maxim/.venv/bin:$PATH
export PYTHONPATH="$PWD/src"
export MAXIM_DATA_HOME=$(mktemp -d)
```

The existing environment supplies Python/dependencies only. `environment.txt` proves that
`maxim.__file__` resolves inside the detached audit copy. `tool-versions.txt` records versions;
mypy 1.20.0 and Ruff 0.14.14 match CI's pins. Each independent shell gets a fresh data home;
pytest additionally isolates HOME and caches through its own conftest. No real LLM, simulation
campaign or hardware run was requested. Fake-bridge tests are part of the fast suite.

`firewall-check.txt` verifies removed paths, valid redacted JSON and the remaining label matches.
The unread leak-scan buffer and unread worktree-creation output were deliberately not copied.
The baseline JSON comment redaction changes no code or numerical pin. No firewall changes
are part of the output branch.

## Executed checks

`run_offline.py` records exact commands, UTC starts, output and exit status for the verdicts,
prereg lint, silence/length lints, architecture audit, scoped and whole-source mypy, Ruff checks,
CLI help and collection. Exceptions/additional checks:

```bash
python scripts/survival_world/exp60_run.py verdict \
  --data docs/experiments/data/exp60_trials.jsonl \
  --run-id 301eb2edff6d --run-id eeb92752ee2b

export COVERAGE_FILE=/tmp/maxim-codex-audit-131/.coverage
python -m pytest tests/ -x -q -m "not slow" \
  --ignore=tests/integration/test_memory_hub.py \
  --cov=src/maxim \
  --cov-report=json:/tmp/maxim-codex-audit-131/coverage-fast.json \
  --cov-report=term:skip-covered

python -m pytest tests/integration/test_memory_hub.py -q
python -m ruff format src/ tests/

python -m pytest tests/unit/test_sandbox_containment_and_content.py \
  tests/unit/test_sandbox_approval_fails_closed.py tests/unit/test_mode_switch_gate.py \
  tests/unit/test_tool_output_framing.py tests/unit/test_fetch_byte_cap.py \
  tests/unit/test_public_format_freeze.py tests/unit/test_load_agent_restore_contract.py -q

python -m pytest tests/unit/test_mode_tool_gate.py tests/unit/test_network_guard.py \
  tests/unit/test_http_public_only.py -q

python scripts/audit_release_tags.py --check-releases
python scripts/check_nightlies.py
python scripts/audit_release_build.py --dist-dir /tmp/maxim-codex-audit-131/artifacts
git verify-tag v1.3.1
```

The first sandboxed fast-suite attempt stopped at a prohibited loopback bind (270 passed).
The first focused network-guard attempt also lacked socket permission. Their `*-sandbox-refusal.txt`
files are environment failures, not project failures. The authorized reruns retained the
repository's network guard and passed. The full fast run blocked 98 outbound calls; the
focused guard/public-fetch run reports five deliberately tested attempts. These are blocked
attempts, not successful external requests. Raw captured ANSI spinner output is retained;
that existing hygiene defect is ledger D23, not a new finding.

## Research and API

- `exp60-selected.txt`, `exp61.txt`, `exp62.txt`, `exp56.txt`, `r3.txt` and `r3-amended.txt`:
  complete outputs. The unfiltered `exp60.txt` correctly refuses duplicate rows. Its exit code
  is zero despite INCOMPLETE; no successful exit is treated as an earned verdict.
- `analysis-comparison.txt`: shared fields compared with committed verdict JSON. Plain R3's
  cadence-field difference is expanded in `r3-differences.txt`.
- `deep_checks.py` / `deep-checks.txt`: artifact digests/source equality, live protection/jobs,
  first-parent prereg times/data stamps, ten Exp 10 hashes and all session finish reasons.
- `exp10-verified.txt`: baseline/resume JSON compared by memory `id`. Equality is checked for
  `encoding`, `encoded_at_us`, `capture_seq`, `storage_strength`, `retrievability_anchor_us`,
  `situation`, `retro_tag`, `encoding_tag`. An initial search of `bio_telemetry.jsonl.gz` found
  no enrichment events; the actual events are in `run_log.jsonl.gz`, retained in `exp10-traces.txt`.
- `api_probe.py` / `api-probe.txt`: standalone documented operations, persisted identifier/count
  checks and real create/shutdown/load. No router is invoked; an in-process network guard is on.
- `known-defect-probes.txt`: `inspect.signature(nac.record_event).bind(..., metadata=...)`
  rejects the bridge call; a real `hippocampus.recall_similar(str, limit=3)` raises; a
  `ModeSwitchTool` with an in-memory recording callback admits passive→active and refuses
  singularity. No host mode is changed. Source call sites are in `production-callers.txt`.

## Metrics

`size-metrics.json` counts `*.py` under `src/maxim` and `tests`, physical `splitlines()` lines,
and every `ast.FunctionDef`/`ast.AsyncFunctionDef` span (`end_lineno - lineno + 1`, including
nested functions). Test/source ratio uses physical lines. This is not executable LOC.

`coverage-fast.json.gz` is the full unedited pytest-cov JSON compressed with gzip (mtime 0).
`coverage-summary.json` selects risky modules and computes line coverage as
`covered_lines / num_statements`, branch coverage as `covered_branches / num_branches`.
The combined percentage is coverage.py's own report. The configuration excludes
`embodied_runtime/selfy.py`; these numbers do not combine separate focused, integration,
nightly, subprocess, provider or hardware runs. `mypy-all.txt` is the full 1,071-error report,
not a filtered tally.

## Live read-only evidence

Captured with `gh release view v1.3.1 --json tagName,createdAt,publishedAt,targetCommitish,body,assets,url`,
`gh run list --workflow test.yml` (100 runs, no titles), a separate 30-run scheduled query,
and `gh run view <id> --json databaseId,headSha,event,createdAt,updatedAt,status,conclusion,jobs,url`.
Repository: `dennys246/Maxim`. No PR page or PR body was queried during grading.

Protection endpoints: `repos/dennys246/Maxim/branches/main/protection`,
`repos/dennys246/Maxim/rulesets`, `repos/dennys246/Maxim/rulesets/13705164`.
PyPI metadata: `https://pypi.org/pypi/pymaxim/json`. GitHub wheel/sdist downloads were hashed
locally; their binary files are not copied here because their URLs and matching hashes are
retained. No package was published or release state changed.

`issues-filtered.json` came from `gh issue list --state open --limit 500 --json
number,title,body,url,createdAt`. Before display or retention, entries matching grade-like
text or score-card/re-score references were discarded; only their numbers (#939/#940) remain.
Thus an absence claim across every open issue is deliberately not made.

`nightlies-check.txt` is a live refusal because main advanced after the tag. The exact-tag
nightly job evidence separately proves successful model/slow lanes before publication;
the aggregate workflow failure is the already-filed release-gate timing defect #938.

## Findings and limits

The main card distinguishes code facts, reproduced outputs, recorded claims and unverified
properties. No prior grade is used as a baseline. The prereg lint reports historical exceptions
and unjudged in-place edits; an exit-zero lint is not proof that those historical records are clean.
No new runtime defect is asserted from the already-known #840/#841/#924 probes. The three
documentation discrepancies are absent from the inspected blind-safe issue/ledger corpus;
their presence in the excluded issues remains unverified.
