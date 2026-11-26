# #1315 item 10b — EXECUTOR lens

Diff: `eeb9cef2..b6015366` (feat/1315-item10b-offline-call-sites). Probes ran in a copied tree
(`scratchpad/review10b/copy`, PYTHONPATH pinned and verified via `maxim.__file__`), plus a `git archive eeb9cef2`
copy (`scratchpad/review10b/base`) for before/after comparisons. Targeted tests: `test_offline_call_sites_10b.py` +
`test_offline_latch_unit.py` 77 passed; `test_offline_latch.py -k no_effect_subprocess` (the census gate) passed;
ruff clean on the touched files. No suite, no sims, no network.

Counts: BLOCKER 0 · SHOULD-FIX 7 · NIT 9

---

## SHOULD-FIX

### S1. An optional pose weight that is missing makes the whole YOLO engine refuse offline
`models/vision/ultralytics_engine.py::UltralyticsEngine.__init__` / `::_ensure_weight_file`

- **Scenario:** offline, `yolov8m-seg.pt` on disk, `yolov8m-pose.pt` not on disk, `UltralyticsEngine(load_pose=False)`.
  `__init__` resolves the pose path with `_ensure_weight_file` straight away, and that function now refuses offline
  when the file is missing. So construction raises `OfflineRefusal` even though pose is lazy and optional. Before 10b
  the same run built the engine: the downloader's socket was refused, the broad catch swallowed that,
  `_ensure_weight_file` returned the bare asset name, and pose failed later, by itself.
- **Verified:** `probe2.py` §1 printed `engine REFUSED at construction with load_pose=False: ... yolov8m-pose.pt: no
  local weight`.
- **Knock-on:** `selfy._ensure_segmenter` catches it with `except Exception` and falls back to rtm, so the yolo engine
  can never run offline without a pose weight.
- **Fix:** do not call the refusing `_ensure_weight_file` for the pose path offline. Keep the path when the file exists,
  and otherwise let `_try_load_pose` → `_yolo_weight_path` refuse when pose is actually wanted. Or treat a missing pose
  weight as `mark_degraded("vision.pose", ...)`. Add a test that builds the engine with seg present and pose absent.

### S2. faster-whisper: every offline init failure is relabelled "not cached locally" and booked as a refusal
`inference/transcribe_audio.py::WhisperTranscriber._ensure_model` (the new `except Exception: ... refuse(...) from e`)

- **Scenario:** offline, `model_size_or_path` is a local directory holding a corrupt or partial CTranslate2 model.
  ctranslate2 raises `RuntimeError: Unable to open file 'model.bin'`, and the wrapper turns that into
  `OfflineRefusal("faster_whisper.WhisperModel refused: <dir> is not cached locally")`, with a ledger entry.
- The same happens to a CUDA init error, a bad `compute_type`, or an out-of-memory error. A local defect reads as an
  offline refusal, and the ledger and observer count it as one.
- **Verified:** `probe_whisper.py`, with a real faster-whisper 1.1.1 and an empty HF_HOME:
  - a genuine cache miss raises `huggingface_hub.errors.LocalEntryNotFoundError`, a subclass of `FileNotFoundError`,
    and is wrapped correctly;
  - the corrupt-dir case printed `mislabel: OfflineRefusal ... notamodel is not cached locally | cause: RuntimeError
    Unable to open file 'model.bin'`.
- **Fix:** convert only the cache miss: `except Exception as e: if offline and isinstance(e, FileNotFoundError)` (or
  `LocalEntryNotFoundError` through `optional_dependency_available`), otherwise re-raise. Pin it with a negative test
  (a `RuntimeError` stays a `RuntimeError`).
- **Note:** both callers (`transcribe_audio.py` worker loops, around lines 439/459 and 639/652) end in
  `except Exception: warn(...); return`. The typed refusal still ends as a warning in the worker, which is acceptable
  for a worker process but is 10c swallow-census material.

### S3. The doctor wrapper marks EVERY non-ok row of a list check, including rows the latch did not refuse
`doctor/checks.py::_refused_offline_row` (`mark` applied to the whole list)

- **Scenario:** `check_mesh_nodes` offline. Each node probe is refused, so the ledger grows in this thread. The
  network-independent `Drain orphan <name>` **warn** row (a config defect) is then also rewritten to
  `status="info"` with the message "refused (offline): socket.connect: … — drain state entry … has no matching node".
- `check_robot_reachable` (many robots) and `check_resolved_config` have the same shape: on Linux, a keyring DBus
  `AF_UNIX` connect is refused, and every WARN row in the section would then be downgraded.
- **Why it matters:** the 10a delta-2 checklist that this diff carries says "marking only what the latch refused". As
  built, a real warn or fail is hidden as info.
- **Verified:** `probe4.py`. A synthetic list check with one config-defect `fail` row and one refused-probe row came
  back with both rows `info / refused (offline)`.
- **Fix:** apply the ledger-delta marking to the per-item probes (`_probe_mesh_node_to_check`, `_check_one_robot`).
  The list-level wrapper should only handle a raised refusal, or should mark only a single-row result. Add a test with
  a mixed list.

### S4. Lane preflight: offline, a missing GGUF now aborts the router build instead of re-walking to a local model
`runtime/lane_backends.py::_ensure_lane_profiles_available` (the caller of `download.ensure_available`)

- **Scenario:** offline, and a lane's profile (explicit `llm.profile`, a tier override or `--llm`) is in `LLM_MODELS`
  but not on disk, while a smaller profile is.
  - Before 10b: `ensure_available` returned False (no consent, or the download was refused at the socket and
    swallowed by `download_file`), and the preflight re-walked to the local model or cleared the profile.
  - Now: `OfflineRefusal` escapes `build_primary_router`.
- **Verified:** `probe3.py` on both trees.
  - 10b: `preflight raised OfflineRefusal ... smollm-135m-instruct is not on disk`.
  - base: it took the False → `dataclasses.replace` path, the re-walk branch. The probe's namespace stand-in errored
    there, which proves the branch.
- **Why SHOULD-FIX:** this is a behaviour change at a consumer the diff never mentions. The preflight's own contract
  (docstring, and the comment above its call: "If the user is offline + non-tty … this is a no-op") is now stale.
  Arguably the refusal is the better answer: an explicit profile is missing, and a silent substitution would be the
  fallback the codebase dislikes. But that is an owner call, not a side effect.
- **Fix:** either (a) keep the refusal and update the docstring, the comment and the changelog, plus a test, or
  (b) offline, skip `ensure_available` and re-walk local files only, with `mark_degraded("lanes.preflight", ...)`.
  Recommend (a) as the strict option.

### S5. llama_cpp lazy fetch swallows the refusal, never remembers it, and books it on every completion
`models/language/llama_backend.py::_fetch_missing_model` (`except Exception as e: warn(...); return False`)

- **Scenario:** offline, an in-process llama lane whose GGUF is missing and which the preflight did not cover: a custom
  profile not in `LLM_MODELS`, a bare `LLMRouter`, or the warmup thread.
  - `ensure_available` now raises `OfflineRefusal`, and it is caught as a generic "download failed".
  - `_download_refused` is not set on the exception path, so every `complete()` refuses again: one WARN plus one ledger
    entry per call. The ledger is unbounded.
  - `complete()` returns `""`, so the agent runs on empty completions instead of stopping on a typed refusal.
- **Verified:** `probe2.py` §2: three calls gave `[False, False, False]`, `ledger delta 3`, `remembered False`.
- **Fix:** `except SECURITY_REFUSALS: self._download_refused = True; raise`, or at least set `_download_refused` and use
  `warn_unless_refusal`. The 10a router dispatch already maps an `OfflineRefusal`.

### S6. `sentence-transformers>=2.2.0` floor vs the new `local_files_only` kwarg
`similarity/encoder.py::_get_encoder`, `similarity/semantic.py::_get_or_load_model`,
`models/vision/clip_encoder.py::_load_model`

- **Scenario:** `semantic` extra at its declared floor. The 2.2.x `SentenceTransformer.__init__` has no
  `local_files_only` parameter: it takes `model_name_or_path, modules, device, cache_folder, use_auth_token`.
  - Offline, the kwarg raises `TypeError`.
  - The encoder and semantic sites catch it and `mark_degraded("... is not cached locally (TypeError)")`, then fall
    back to hashing even when the model IS cached.
  - `clip_encoder` raises a raw `TypeError`.
  - Online is unaffected, because the kwargs are empty.
- **Verified:** partly. The installed 5.4.0 accepts the kwarg (checked by signature). The 2.2.x signature is from
  memory and could not be fetched offline, so the author should confirm it on PyPI. The extras matrix covers only
  `console` and `tts`, so CI never exercises the floor, and the unit tests mock `SentenceTransformer`.
- **Fix:** raise the floor to the first release that accepts `local_files_only`, or pass the kwarg only when the
  signature has it. `HF_HUB_OFFLINE`, set by `engage()`, already gives the hub-level guarantee.

### S7. Benchmark: a refusal now aborts the whole multi-model run (losing earlier models' results), unlike the checklist
`simulation/benchmark.py::BenchmarkRunner.run` (`except SECURITY_REFUSALS: raise`)

- **Scenario 1:** `models=[local, cloud]` offline. The local model's scenarios complete, then the cloud model's first
  scenario raises `OfflineRefusal`. The whole `run()` unwinds, no `BenchmarkReport` is built or written, and the local
  results are gone.
- **Scenario 2, online:** a `LeaderRefusal` (a leader with no key), which used to be a recorded failed scenario, now
  aborts the run as well. The changelog fragment mentions only offline.
- **Contract:** the 10a checklist, quoted in the notes, says "benchmark scores a refused scenario as refused, not
  failed". The build aborts instead.
- **Verified:** by code reading. There is no outer handler between the loop and the report build.
- **Fix:** an owner decision.
  - (a) Record a `RunResult` marked refused, excluded from the score and the rankings, and keep going. This matches
    the checklist.
  - (b) Keep the abort, but say so in the notes and the changelog (including the online `LeaderRefusal` change), and
    consider writing the partial report first.

---

## NIT

1. **`rtm_engine.py::RTMEngine._load_pose`:** `_resolve_model_path` sits inside `try/except Exception`, so a refused
   pose weight becomes "Failed to load pose model". That is the opposite of ultralytics' explicit "typed refusal, not
   'no pose'" choice. Pick one policy for optional pose (see S1). Most likely: degrade, plus `mark_degraded`.
2. **Pre-existing broad catches above the new refusals** (10c swallow census):
   - `segmentation.py::YOLO8.ensure_pose_model`;
   - `inference/segment_vision.py`, the per-frame `pose_targets_for_box` in `try/except: pass`;
   - `selfy._ensure_segmenter` (yolo → rtm fallback);
   - `tts.py::VoiceProvisioner._run`, which retries the refusal with backoff forever, logged only as
     "provisioning attempt raised OfflineRefusal; retrying" and booking one ledger entry per retry. Harmless, since a
     voice copied in later still verifies, but it is not said as "offline".
3. **`python -m maxim.models.download --llm/--tts/--vision` offline:** the CLI now engages first, but these paths call
   `download_llm` / `download_file` directly. Its `except BaseException` prints `Error: offline (...) refused` and the
   CLI exits **1**, not the refusal exit 2, and there is no call-site refusal. Add `_refuse_download_offline` at the
   top of `download_file`/`download_vision`, or map refusals in `main`.
4. **`embodied_runtime/selfy.py` `__main__`:** `engage_offline_if_configured()` raising `ConfigurationError` (V20)
   ends in a traceback. `download.main` handles the same error with `✗ … ; return 2`. Mirror that.
5. **`container_runner.check_docker_available` and `_atexit_cleanup` / `_remove_detached`** spawn docker without
   `_run_docker`. Offline they are refused by the hook (`subprocess.Popen`, still typed) rather than the call site. As
   a result, `create_sandbox(backend="auto")` raises the refusal instead of falling back to tmpdir. Probably fine
   (V8), but state it, or route `docker info` through the call-site refusal with a tmpdir fallback.
6. **`mark_degraded` reasons** say "not cached locally" for ANY exception (an out-of-memory error, a CUDA error), with
   only `type(e).__name__` as the clue. Say "could not load offline (<type>)" unless the type is a cache miss.
7. **Download docstrings and messages:**
   - The docstrings of `ensure_available` and `ensure_tts_voice` still say "False on any failure"; they now raise
     offline.
   - The `ensure_tts_voice` detail says "not on disk" for a voice that is present but fails verification.
   - A custom profile not in `LLM_MODELS` now gets the refusal, where it used to get the "custom profiles must be
     downloaded manually" message.
8. **CI grep, `.github/workflows/test.yml`.** The shell is correct: under the default `bash -e`, the
   `if pipeline; then` exit code comes from the `grep -v`. There are no false positives today; I ran it on the tree
   and it exits 1. Gaps:
   - False negatives: `from httpx import Client` then `Client(`, top-level `httpx.get(`/`httpx.stream(`,
     `from socket import socket`, `socket.create_connection(` (7 sites, all latch-refused at connect, but not HTTP-only
     either).
   - The four socket files are whole-file exemptions, so a new HTTP socket in `utils/net.py` passes.
   - `test_no_http_client_outside_utils_http_widened_grep` only checks that the pattern is present. It never runs the
     pattern against positive and negative fixtures.
9. **Whisper calls the library before refusing.** The refusal comes after `WhisperModel(...)` fails, not before it.
   This is safe only because `local_files_only=True` and `HF_HUB_OFFLINE=1` keep the load local; the comment could say
   so. The `raise refusal from e` construction itself is correct: it books once and chains `__cause__`, verified.

---

## Sound (checked, no finding)

- **Online byte-identity:**
  - `local_files_only_kwargs()` returns `{}` online.
  - `_offline_local_files_only` / `_offline_trust_remote_code` reduce to the configured bools online.
  - `_yolo_weight_path` is the identity online.
  - Each new refusal is gated on `is_offline()`.
  - The single online change is S7's `LeaderRefusal`.
- **Refusal before the side effect:**
  - docker: `_run_docker` refuses before `subprocess.run`, and `list(argv)` is consumed only on the raising path.
  - twilio: `send` and `call` refuse before `record_outbound`, the twilio client and the TwiML. The tool path ends in
    the executor's `OfflineRefusal → ToolErrorKind.OFFLINE` mapping, so there is no pain or NAc.
  - ddgs: refused before the import. The tool path is already policy-blocked offline (`source="offline"`), so the
    tool's own `except Exception` is not reached.
  - `ensure_available` / `ensure_tts_voice`: refused after the local check and before `can_download`, consent, the
    tty prompt and the lock (the unit test proves consent is never read).
  - rtm: refused before `download_vision`.
  - ultralytics: refused before `attempt_download_asset` and before `YOLO(name)`.
- **faster-whisper 1.1.1** (the floor) accepts `local_files_only`, verified by signature. transformers'
  `from_pretrained` and `snapshot_download` accept it at the >=4.40 floor.
- **`Refusal.thread`:** a defaulted 4th field. No consumer unpacks the tuple positionally; every reader in
  `src/`, `scripts/` and `tests/` uses `.event` / `.reason` / `.thread`.
- **The doctor thread filter** is correct for today's doctor: the checks run sequentially on the caller's thread, and
  nothing in `doctor/` uses a thread pool. It is a latent false negative only if a check ever probes from a worker
  thread.
- **`_run_stopped_on` and `AgentPool._generate_llm_response`** re-raise `SECURITY_REFUSALS`, and the unit tests pin
  both. The AST test pins the benchmark handler order.
- **`mark_degraded`:** stdlib-only, so the leaf-import guard holds. It locks the record and logs outside the lock, and
  `_reset_for_tests` clears both sets.
- **The census red gate `test_no_effect_subprocess`** passes un-xfailed in the copy.

---

**Verdict:** SOUND WITH CONDITIONS. There are no blockers, because every refusal fires before its spawn, socket,
consent prompt or record-as-sent. Fold S1–S7 before merge: two over-reaching refusals (S1, S4), one mislabel (S2), one
over-broad doctor mark (S3), one swallowed and re-booked refusal (S5), one dependency floor (S6), and one
benchmark-contract mismatch (S7).
