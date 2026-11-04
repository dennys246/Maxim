# #1315 item 10b — Architecture lens

Diff: `eeb9cef2..b6015366` (feat/1315-item10b-offline-call-sites). Probes ran in a copied tree
(`scratchpad/review10b/copy`, no `.git`; the 4 `test_offline_latch.py` provenance failures there are the copy's missing
`.git`, `ScenarioBroken: ... no .git`, not code). Targeted: `test_offline_call_sites_10b.py` + `test_offline_latch.py`
75 pass; `lint_http_surface`, `lint_no_silent_swallows`, `lint_function_length` (touch rule skipped, no git),
`lint_orphan_modules`, ruff on touched files: clean.

## Findings

### S1 SHOULD-FIX: the widened CI grep duplicates `scripts/lint_http_surface.py`, and is weaker, with a second allow-list
- **Where:** `.github/workflows/test.yml` (the new `urllib3|requests|httpx.Client(|socket.socket(` grep step); changelog
  fragment's last sentence; notes §6 10b status ("the CI grep widened").
- **What:** `scripts/lint_http_surface.py` (in CI at the lint job, `dab92b7d`/`fdf105c4`) already enforces a strict
  superset. Rule A bans `urllib3`, `requests`, `httpx`, `httpcore`, `aiohttp`, `http.client`, `urllib.request`
  outside `utils/http.py` with **no allow-list** (all import forms, plus dynamic-import literals). Rule B pins **every** raw
  dial (`socket.socket`, `create_connection`, `socketpair`, `create_server`, `wrap_socket`, `asyncio.open_connection`)
  per file, exact both ways. Rule C registers the egress SDKs. The new grep is weaker on every axis. It bans only
  `httpx.Client(`, so `from httpx import Client` and an alias pass. It misses `socket.create_connection` (7 live sites:
  doctor/checks.py ×3, reachy controller, leader_proxy, local_server_spawner, minecraft.py). It matches docstrings, so
  it exempts the whole latch module `_offline.py`. And its exemption is file-wide, where the lint pins counts. So CI
  now carries two allow-lists for one rule, and they already disagree: 4 files in the grep, 9 pinned in the lint.
- **Consequence:** two mechanisms for one invariant, and they will drift. The changelog's "CI now refuses ... (with the
  four non-HTTP socket sites named)" reads as new enforcement, but it already existed, and the named list is
  incomplete next to the real one. Separately, 10c's site census would become a third roster.
- **Verified:** read `lint_http_surface.py` (docstring and `RAW_DIAL_PINS`), ran it (exit 0, "11 raw dial(s) in 9
  file(s); 17 egress-SDK import site(s)"), and grepped for `create_connection` sites.
- **Fix:** drop the grep step. Mark §6's "the CI grep widens to urllib3, requests, httpx.Client( and socket.socket("
  line as already met by `lint_http_surface.py` (rules A and B), and say in the notes that 10c's site census extends that
  lint's registers (`RAW_DIAL_PINS`, `EGRESS_SDK_IMPORTERS`) rather than building a new roster. Correct the changelog
  sentence. Replace `test_no_http_client_outside_utils_http_widened_grep` (see N4) with nothing, or with an assertion
  that `lint_http_surface` is in the lint job.

### S2 SHOULD-FIX: "the fallbacks become loud" is half met; the `degraded` record has no reader
- **Where:** `_offline.py::degraded`, `_offline.py::mark_degraded`; `simulation/report.py::_code_stamp`;
  `runtime/decision_log.py`; notes §6 ("records a `degraded` reason ... decision_log stamps it"; the `decision_log:
  offline: bool` paragraph).
- **What:** the WARNING half ships (one per site per process). The record half has zero non-test callers.
  `grep _offline.degraded()` over `src/` and `scripts/` finds only the definition. `report.json` stamps "unknown:
  offline" for the code hash, but it carries neither the degraded list nor the refusal ledger (§6 "Startup order" also
  says the report "records the refusal"). `decision_log.py` has no `offline` field. The observer half is N/A by §1A.8
  (no producer), which is fine. The 10b status block claims `mark_degraded()`/`degraded()` as built, without saying the
  consumers are deferred. `test_offline_sim_report_says_offline`'s docstring also says "the item 10 PR owes that
  end-to-end run", and V22 makes 10a+10b that PR.
- **Consequence:** an offline run whose encoder fell back to hash embeddings leaves no machine-readable trace in its
  report. Only a log line shows it, which is exactly the "followed by attention" gap. This is capability, not a fix (the
  CLAUDE.md caller rule), and V22's "main never carries a half-done offline mode" is not met as written.
- **Verified:** caller grep; read `report.py::_code_stamp` and `decision_log.py`.
- **Fix:** either (a) have the report writer add `"offline": {"degraded": [...], "refusals": [...]}` from
  `_offline.degraded()`/`ledger()` beside the code stamp, and add decision_log's defaulted `offline` field, or (b) state in
  the 10b status block that those consumers and the owed e2e offline sim move to a named later item, with a backlog
  row. (a) matches the plan; (b) needs an owner nod under V22.

### S3 SHOULD-FIX: offline, ANY model-load error is reported as "not cached locally"
- **Where:** `inference/transcribe_audio.py::WhisperTranscriber` (the new `except Exception` → `refuse`);
  `similarity/encoder.py::_get_encoder`, `similarity/semantic.py::_get_or_load_model` (the `mark_degraded` reason).
- **What:** offline, the whisper handler turns every exception into
  `OfflineRefusal("... is not cached locally")`, including errors that have nothing to do with the cache.
- **Probe:** under the latch, a `WhisperModel` raising
  `ValueError("Requested float16 compute type, but the target device does not support it")` for a local path
  `/abs/local/whisper-dir` came out as
  `OfflineRefusal: faster_whisper.WhisperModel refused: /abs/local/whisper-dir is not cached locally`, booked in the
  ledger (cause chained).
- **Consequence:** a device or compute-type error, OOM, or corrupt cache becomes a security refusal with a false reason.
  The CLI renders that as exit 2 ("the operator must act: offline"), and the operator is sent to fetch a model they
  already have. The refusal ledger, which §6 makes the audit record, now contains entries the latch never decided. The
  encoder and semantic degraded reasons carry the same misattribution, though only at "degraded".
- **Fix:** classify by type. Refuse only on the hub's local-miss errors (`huggingface_hub.errors.LocalEntryNotFoundError`,
  `OfflineModeIsEnabled`, or an `OfflineRefusal` from the latch), and re-raise everything else unchanged. Narrow the
  degraded reason the same way, or say "failed to load offline (Type)" without asserting the cause. Optionally give
  `_offline.refuse` a `cause=` parameter, so the nested `try: refuse() except OfflineRefusal as r: raise r from e` is
  not copied to the next site.

### S4 SHOULD-FIX: the doctor wrapper demotes unrelated `fail`/`warn` rows of a list check
- **Where:** `doctor/checks.py::_refused_offline_row` (`mark`, applied to every non-ok row of a list result).
- **What:** when this thread booked any refusal during a list-returning check, **every** non-ok row becomes
  `info` / "refused (offline)". The 10b checklist asked for "marking only what the latch refused". The real case is
  `check_robot_reachable`, which returns one row per robot (`_check_one_robot`): robot B's refused probe relabels robot
  A's era-coherence `fail`. `check_resolved_config` and `check_env_config` are list checks too.
- **Probe:** a wrapped `check_fake` returned [`fail` "n_ctx 0 is invalid", refused-probe `warn` "nothing responding"],
  and both came back `info` "refused (offline): socket.connect: 127.0.0.1:8000 — ...".
- **Consequence:** an offline `maxim doctor` hides a genuine failure as info, so the diagnostic fails open.
- **Fix:** attribute per row. Wrap the per-row producer (`_check_one_robot`, and any row helper that probes), or snapshot
  the ledger length per row inside list checks. At minimum, never demote a `fail` whose row did not book a refusal. Add
  a test with a mixed list.

### S5 SHOULD-FIX: the benchmark aborts the whole run on one refused model, and the built behaviour diverges from the checklist
- **Where:** `simulation/benchmark.py::BenchmarkRunner.run` (`except SECURITY_REFUSALS: raise`).
- **What:** the checklist said "benchmark scores a refused scenario as refused, not failed". As built, it re-raises
  out of the model loop. `maxim benchmark --models <in-process>,<cloud>` offline runs the in-process model, then the
  cloud model's refusal discards every completed result (no report is written). The re-raise also changes
  `LeaderRefusal` from "scored failure" to "abort", which is defensible but not stated in the notes or the changelog.
- **Consequence:** completed results are lost, and the docs and the code disagree on what a refused benchmark
  produces.
- **Fix:** pick one shape and state it. Either record `RunResult(error=..., refused=True)` (or a per-model "refused"
  status that `_aggregate_runs` excludes from the score) and continue, or keep the abort and say in the notes and
  changelog that a refusal stops the benchmark (V-decision if it changes `LeaderRefusal` behaviour). The first matches
  the checklist.

### N1 NIT: one helper, not ten copies plus a private one
`if _offline.is_offline(): _offline.refuse(site, detail)` appears at 10 sites, and `models/download.py::_refuse_download_offline`
is exactly that helper, private to one module. Add `_offline.refuse_if_offline(site, detail)` to the leaf (it is
stdlib-only), use it at every site, and delete the private copy. It also gives 10c's census one symbol to look for.

### N2 NIT: the engage-first prologue is duplicated
`cli.py::main` and `models/download.py::main` each carry `try: engage_offline_if_configured() except ConfigurationError:
print; return 2`. `selfy.__main__` calls it without the handler, so a bad config gives a traceback (still fail-closed).
One `config_loader` helper that returns an exit code, or None, would serve all three.

### N3 NIT: transformers silently flips a configured `hf_trust_remote_code`
`transformers_backend._offline_trust_remote_code` turns an operator's `True` into `False`, so a model that needs remote
code then fails deep in transformers. A typed `refuse("transformers.trust_remote_code", model_id)` when it is configured
and offline would be the call-site-first shape §6 uses everywhere else. Also, the status block says
`local_files_only_kwargs()` covers "every HF load", but transformers uses its own predicate (it ORs the config), so the
text should say so. The `_review_remote_code` `snapshot_download` override is dead offline (reached only when trust is
on), which is harmless.

### N4 NIT: the grep "test" only checks that a string is in test.yml
`test_no_http_client_outside_utils_http_widened_grep` asserts that the pattern text is present, and never runs it. It
falls away with S1.

### N5 NIT (informational, for 10c): swallow sites seen around the new refusals
`tools/internet_search.py::InternetSearchTool.execute` has a catch-all that turns the ddgs `OfflineRefusal` into
"Search failed: ..." (still in the ledger). The encoder and semantic failures are not cached, so each new instance
retries the load and appends another `Degraded` record (bounded by instance count). These are for 10c's swallow census,
not this PR.

## What's sound
- `_offline.py` stays a stdlib-only leaf: `mark_degraded` uses stdlib `logging`, and there are no maxim imports.
  `Refusal.thread` is set by `_book` for every entry (the default 0 is only for foreign construction), and the only
  reader (the doctor) filters correctly. Doctor runs no worker threads, so the filter is enough.
- The refusal placements are at the right chokepoints. `LocalDockerRunner._run_docker` (every docker call, not only
  `ensure_image`) is better than §6's text. Ultralytics refuses at the last call before `YOLO(...)` and before
  `attempt_download_asset`. `ensure_available`/`ensure_tts_voice` refuse before consent and preflight. The RTM refusal
  sits outside its warning catch. The twilio refusal comes before anything is recorded as sent. The ddgs refusal is
  justified as the only in-process defence (native primp).
- HF coverage is complete by grep: every `SentenceTransformer(`, `from_pretrained(`, `snapshot_download(` and
  `WhisperModel(` in `src/` is covered.
- The `SECURITY_REFUSALS` reuse in `cli.py::_run_stopped_on` and `agent_pool._generate_llm_response` is one tuple, used
  consistently with 10a.
- No layer violations. `lint_no_silent_swallows`, `lint_orphan_modules`, `lint_function_length` and ruff are clean, and
  the census red gate flipped honestly (xfail removed, test passes).

## Verdict
SOUND WITH CONDITIONS: no blockers. Fold S1 (drop the duplicate grep, credit `lint_http_surface`), S3 (classify the
load error) and S4 (per-row doctor marking), and decide S2/S5 (wire the degraded/refusal record into the report and
pick the benchmark shape, or state the deferral) before merge.
