# #1315 item 10b — Wire integrity lens

Diff: `eeb9cef2..b6015366` (branch `feat/1315-item10b-offline-call-sites`). Probes ran in a copied tree
(`scratchpad/review10b/copy`), scripts in `scratchpad/review10b/probes/`. Targeted tests in the copy:
`tests/unit/test_offline_call_sites_10b.py` + `test_offline_latch.py::test_no_effect_subprocess`: 23 passed (twilio
not installed here, so that site was reported not-applicable).

**Probe incident (disclosed):** probe p2's first version also ran an ONLINE leg of
`lane_backends._ensure_lane_profiles_available`. This box's config grants auto-download, so it started a real
`phi-2` GGUF download from huggingface.co. I killed it after about 570 MB and deleted the partial file I had created
(`~/.maxim/models/LLM/phi-2.Q4_K_M.gguf.partial`). Nothing else in `~/.maxim` changed. The probe was then re-run
offline only.

---

## Contract 1: functions that now raise `OfflineRefusal`, and every caller

| Producer | Caller (outside diff) | What happens to the refusal |
|---|---|---|
| `LocalDockerRunner._run_docker` | `launch`, `exec`, `ensure_image`, `stop` (same file); `write_file`/`read_file` via `exec` | propagates out of each method |
| ↳ | `simulation/sandbox.py::DockerSandbox.start` → `create_sandbox` → `orchestrator.py::_setup_sim_sandbox` | **swallowed**: `except Exception: logger.warning("Sandbox creation failed")`, so the sim runs with `sim_sandbox=None` (F1) |
| ↳ | in practice never reached: `create_sandbox` first calls `container_runner.check_docker_available()`, whose raw `subprocess.run(["docker","info"])` is refused by the LATCH (not this call site), and `except (FileNotFoundError, TimeoutExpired, OSError)` does not catch it | same swallow (F1, probe p1) |
| ↳ | `_atexit_cleanup` / `_remove_detached` use raw `subprocess` (latch). No handle can exist offline, so this is moot | — |
| `internet_search._search_with_ddg_package` | `_search_duckduckgo` (no try) → `InternetSearchTool.execute` | re-typed: its `except Exception` → `ToolResult(success=False, "Search failed: offline …")`. Unreachable in production: `execute` returns earlier, because `access.enabled` is False offline (`internet_access.py`, `source="offline"`). OK |
| `ultralytics_engine._ensure_weight_file` | `UltralyticsEngine.__init__` (seg and **pose**) ← `registry._build_engine` ← `build_segmentation_model` ← `selfy._ensure_segmenter` | selfy: **swallowed into a fallback** (`except Exception: warn(... falling back to rtm)`), then the rtm build refuses typed if its weight is missing too. The pose call in `__init__` refuses even with `load_pose=False` (F5) |
| `ultralytics_engine._yolo_weight_path` | `__init__` (seg), `_try_load_pose` ← `__init__`/`estimate_pose` | propagates. Unreachable offline in practice, because `_ensure_weight_file` already returned an existing path |
| `TwilioChannel.send` / `.call` | `comms/gateway.py::send/call` ← `tools/comms.py` (`send_message`, `call_user`) with no try | typed to the executor, which maps it to `ToolErrorKind.OFFLINE` (no pain, no NAc). The tools are COMMS, dropped offline by V7 anyway. OK. (`build_comms_stack` → `start_api_server` binding is refused by the latch and swallowed by `logger.exception("Failed to build comms stack")`: 10c swallow census) |
| `download.ensure_available` (was bool) | `cli.py::_ensure_agentic_model` (no try) | typed, exits 2 through main. OK |
| ↳ | `lane_backends.py::_ensure_lane_profiles_available` ← `build_primary_router` (no try) | typed, **but the tier re-walk to a model already on disk is lost** (F2, probe p2) |
| ↳ | `llama_backend._fetch_missing_model` ← `_ensure` (every `complete()`/warmup) | **swallowed**: `except Exception: warn("Model download … failed"); return False`, not remembered, so it books a new ledger entry on every call (F4) |
| `download.ensure_tts_voice` (was bool) | `tts.py::VoiceProvisioner._attempt` ← `_run` | **swallowed into an endless retry** (`except Exception` → backoff up to 300 s, forever). `VoiceProvisioner` has no production caller today, so this is latent (N3) |
| ↳ | `scripts/extras_matrix.py:267` | propagates (CI harness, not offline) |
| `rtm_engine._resolve_model_path` (was `FileNotFoundError`) | `RTMEngine.__init__` (det) | propagates typed. No caller catches `FileNotFoundError` specifically; selfy's rtm fallback has no try. OK |
| ↳ | `RTMEngine._load_pose` | **swallowed** into `_pose_load_error` ("no pose"). This is the opposite of ultralytics, whose comment says pose is "refused before the try: a typed refusal, not 'no pose'". RTM is the DEFAULT engine (N1) |
| `WhisperTranscriber.__init__` | `transcribe_audio.py` worker functions (4 sites; spawn-mode children) | **swallowed**: every site does `except Exception` → CPU retry → `warn("Whisper transcriber unavailable")` and the worker returns. The parent sees nothing typed. Also **mis-typed**: any init failure offline is reported as "not cached locally" (F3, probe p4) |
| `download.main` | `python -m maxim.models.download` | engage plus V20 typed. Downloads under the latch go through `download_file`, whose `except BaseException` prints "Error: offline…" and returns False, so the exit is 1, not 2 (N4) |
| `selfy.__main__` | entry point | engage-first. But `_ensure_segmenter` → `preflight_matplotlib_fonts` spawns a subprocess, which the latch refuses typed (no try), so offline vision never starts unless `MAXIM_SKIP_MPL_PREFLIGHT`. This is consistent with V7/V8 ("the robot path is refused anyway"). Noted, no finding |

## Contract 2: census completeness

- **HF loads.** `SentenceTransformer` (encoder, semantic, clip_encoder), `WhisperModel`, `AutoTokenizer`/
  `AutoModelForCausalLM.from_pretrained` and `snapshot_download` all carry `local_files_only`. A grep for
  `from_pretrained|pipeline(|hf_hub_download|snapshot_download|torch.hub|CrossEncoder|open_clip|load_dataset|urlretrieve`
  finds no other site in `src/`. Complete.
  - transformers (the in-process backend V8 requires offline) failing on a model that is not cached is swallowed by
    `load()`'s `except Exception: logger.error(...); return False`. It is untyped and not degraded (N2).
  - `clip_encoder._load_model` raises the raw HF `OSError`. It has no `src/` caller (NIT, no finding).
- **Spawns.** 27 files spawn. Only `container_runner` refuses at the call site; the rest rely on the latch, which
  raises typed `RuntimeError`, so narrow `except OSError` sites cannot absorb it. Broad catches around a spawn
  (`TmpdirSandbox.execute` → `exit_code=1`, `prefetch` find → debug, `system_metrics`, `gpu_detect`,
  `container_runner.check_docker_available` (uncaught), `orchestrator._setup_sim_sandbox`) are the **10c swallow
  census**. One of them is on the DEFAULT sim path today (F1).
- **Sockets / clients.**
  - Python-level, covered by the latch: `socket.create_connection` (`hardware/reachy/controller.py`, `doctor/checks.py`
    ×3, `leader_proxy.py`, `local_server_spawner.py`, `simulation/minecraft.py`); reachy_mini's `ws_client`
    (`requests`, pure Python).
  - Native, which the latch cannot see: ddgs/primp, now refused at the call site; reachy_mini's GStreamer/WebRTC media,
    which sits behind the already-refused robot connect.
  - The anthropic/openai SDKs are imported only in `api.py` (probe-only `import`) and inside `models/language` (refused
    by 10a's router predicate).
  - No `aiohttp`, `websockets`, `http.client`, `urllib.request` or `zeroconf` outside `utils/http.py`.
- **Census gate blind spot.** `CENSUS_SITES` drives `simulation.container_runner.ensure_image` directly, so it passes,
  while the shipped path (`create_sandbox`) never reaches `_run_docker` and swallows the latch's refusal instead.

## Contract 3: `Refusal.thread`

- Constructor: `_offline._book` (the only construction site in `src/`). It is positional, and `thread` defaults to
  `0`, so a 3-arg construction still works.
- Consumers:
  - `doctor/checks.py::_refused_offline_row` filters on `r.thread == threading.get_ident()`. No doctor check uses
    threads (grep for `ThreadPoolExecutor`/`threading.Thread` in `doctor/` finds none), so the filter is sound.
  - Tests read `.event`, `.reason`, `.detail`, and `.thread` (`test_offline_call_sites_10b.py:55`).
  - `tests/unit/test_offline_latch_unit.py:88` unpacks `(entry,) = ledger()` and reads attributes, not a tuple
    destructure, so it is unaffected.
  - No serializer, JSON writer or report reads the ledger.
- Clean.

## Contract 4: `degraded()` / `mark_degraded`

- **Producers:** `similarity/encoder.py::_get_encoder` and `similarity/semantic.py::_get_or_load_model`.
- **Readers:** `degraded()` has **no production reader**. Its only readers are `tests/unit/test_offline_call_sites_10b.py`
  and the census child.
  - `simulation/report.py` stamps offline (`_code_stamp`) but not degraded.
  - `runtime/decision_log.py` has no offline or degraded field.
  - The observer has no degraded producer, by design (§4.2: "no producer in this release").
- So the changelog's "the run now says 'degraded' once" is true only as the one WARNING on logger `maxim.offline`. A
  sim's `report.json` cannot tell a degraded offline run from a clean one. §6's "records a degraded reason … decision_log
  stamps it" is unbuilt, and the 10b "As built" block does not say so (F6).
- Neither producer caches the failure. Every encode call retries the load and appends one more `Degraded` record. The
  WARNING is deduped, but the list grows without bound in a long sim (N5).

## Contract 5: the CI grep

- It is in `jobs.lint`, the step "CI grep invariants (urllib + …)". `scripts/prepr.py` runs every `run:` step of
  `LINT_JOB = "lint"` from `test.yml`, so prepr mirrors it.
- Run verbatim on the branch: **PASS** (no matches).
- Scope, NIT only:
  - `socket.create_connection(` (7 sites) is not in the pattern. That is fine, because they are not HTTP and the latch
    refuses them. But the step comment says the allowed raw sites are "the four" `socket.socket(` sites; it could say
    `create_connection` is deliberately out of scope.
  - The exemptions are whole-file, so a future `import requests` in e.g. `minecraft_harness.py` would pass.

---

## Findings

**F1: SHOULD-FIX. `orchestrator.py::_setup_sim_sandbox` + `container_runner.py::check_docker_available`. An offline
sim on the default `--sandbox auto` (or `docker`) silently runs with NO sandbox.**
- Scenario:
  1. `llm.offline` is set and the operator runs `maxim --sim … --interactive false` (the default `auto`).
  2. `create_sandbox` → `check_docker_available()` → `subprocess.run(["docker","info"])` is refused by the latch.
     `OfflineRefusal` is not in its `except (FileNotFoundError, TimeoutExpired, OSError)`.
  3. It propagates to `_setup_sim_sandbox`'s `except Exception` → `WARNING "Sandbox creation failed"` →
     `sim_sandbox=None, sandbox_root=None`.
  4. The AUT runs with no pain-trigger layer. There is no tmpdir fallback and no degraded record. `auto`'s documented
     contract ("fall back to tmpdir") is broken, and `docker`'s ("raise if unavailable") is swallowed.
  5. 10b's `_run_docker` refusal is never reached on this path. The census passes only because it calls `ensure_image`
     directly.
- Verified: probe p1. Both `auto` and `docker` gave `sandbox None root None`, three `subprocess.Popen` ledger entries,
  and two "Sandbox creation failed" warnings.
- Fix:
  - `check_docker_available()` returns False offline without probing, with `_offline.mark_degraded("simulation.sandbox",
    "docker is refused offline; tmpdir")`, so `auto` falls back to `TmpdirSandbox` as documented.
  - `backend="docker"` offline refuses at the call site.
  - `_setup_sim_sandbox` re-raises `SECURITY_REFUSALS`.
  - Point the census entry at `create_sandbox(backend=…)` (the shipped path) instead of `ensure_image`.

**F2: SHOULD-FIX (owner call). `lane_backends.py::_ensure_lane_profiles_available`. Offline, a lane whose detected
profile is not on disk now aborts `build_primary_router` instead of re-walking to a model already on disk.**
- Scenario:
  1. Auto-detected tiers pick a GGUF that is not downloaded, while a smaller one is on disk.
  2. Before 10b, `ensure_available` returned False (no consent, or its download was refused inside `download_file`'s
     `except BaseException`), and the re-walk landed on the present model.
  3. Now `_refuse_download_offline` raises before that, and `_ensure_lane_profiles_available` has no handler. So the
     offline run refuses to start, though an in-process model it could use is on disk. That is exactly the offline
     case the re-walk exists for.
- Verified: probe p2 (offline leg) raised `OfflineRefusal: models.download.ensure_available refused: phi-2 is not on
  disk`, with smollm/mistral/qwen present.
- Fix:
  - In `_ensure_lane_profiles_available`, offline, treat "missing" as declined without calling `ensure_available` (a
    `mark_degraded` naming the swap), and keep the re-walk.
  - Refuse only when the re-walk finds nothing, or for an explicitly pinned `--llm` profile.
  - Owner decision: does an explicit profile that is not on disk refuse, or swap?

**F3: SHOULD-FIX. `inference/transcribe_audio.py::WhisperTranscriber.__init__`. Every offline init failure is re-typed
as "not cached locally".**
- Scenario:
  1. The model is cached, but CUDA init fails (the Blackwell/int8 path this code exists for), or `compute_type` is
     invalid.
  2. Offline, the `except Exception` books a ledger refusal `faster_whisper.WhisperModel: … is not cached locally` and
     raises `OfflineRefusal`.
  3. The callers' GPU→CPU retry still runs, because they catch `Exception`. But the ledger carries a false refusal, the
     log blames the cache, and a doctor or report reader of the ledger is misled. If the CPU retry also fails for a
     device reason, the operator is told to fetch a model they already have.
- Verified: probe p4. A fake `WhisperModel` raising `RuntimeError("CUDA failed …")` gave `OfflineRefusal … not cached
  locally`, with cause `RuntimeError`, and was booked in the ledger.
- Fix: re-type only HF's not-found/offline errors (`huggingface_hub.errors.LocalEntryNotFoundError`,
  `OfflineModeIsEnabled`, or `FileNotFoundError`/`OSError` from the snapshot lookup) and re-raise everything else
  unchanged.
- Note also: all four worker call sites then swallow the refusal into `warn(...); return`. That is a spawn child, so
  the parent never sees the refusal. This is 10c swallow census.

**F4: SHOULD-FIX (low). `models/language/llama_backend.py::_fetch_missing_model` swallows the new `ensure_available`
refusal on every call.**
- `except Exception as e: warn(...); return False`, and the exception path is "not remembered". So under offline every
  `complete()` re-enters `ensure_available`, which books a new ledger entry each time (an unbounded `_ledger`) and
  warns "Model download for X failed". The typed refusal becomes a generic "no local model".
- Verified by reading. The path is reachable offline when `model_path` is missing for a profile the build did not catch
  (a custom profile, or a backend built through a bare `LLMRouter`). Note that `ensure_available` now refuses BEFORE its
  "profile not in registry" check, so a custom profile gets the refusal instead of that message.
- Fix: `except SECURITY_REFUSALS: self._download_refused = True; raise`, or remember it, before the broad catch. Also
  move `_refuse_download_offline` after the registry check in `ensure_available`.

**F5: NIT (borderline SHOULD-FIX). `ultralytics_engine.py::UltralyticsEngine.__init__`. A missing POSE weight refuses
the whole engine even with `load_pose=False`.**
- `__init__` calls `_ensure_weight_file` for the pose weight eagerly. That was harmless before, because it returned
  `asset_name` on failure. Now, offline, it refuses construction, so a seg-only user loses segmentation.
- The only production caller (selfy) asks for pose and falls back to rtm, so the impact is library callers.
- Verified: probe p3 (seg weight present, pose absent, `load_pose=False`) raised `OfflineRefusal … yolov8m-pose.pt`.
- Fix: offline, resolve the pose path without refusing in `__init__`, and let `_try_load_pose` refuse when pose is
  actually needed (set `_pose_load_error` so it does not re-refuse per frame).

**F6: SHOULD-FIX (claim). `changelog.d/1315-item10b-offline-call-sites.added.md` + notes §6 status block. "The run now
says 'degraded'" is a log line only; `degraded()` has no production reader.**
- `report.json` (`simulation/report.py`) and `decision_log` do not carry it, so an offline run whose encoder fell back
  to hash vectors produces a report indistinguishable from a full one. §6's "decision_log stamps it" is unbuilt and not
  listed as deferred.
- Fix, either of:
  - (a) add `"offline_degraded": [d._asdict() for d in _offline.degraded()]` (deduped by site) next to
    `_code_stamp`'s offline branch in `report.py`;
  - (b) narrow the changelog to "logs one WARNING naming 'degraded'" and list report/decision_log stamping as 10c or a
    follow-up in the §6 block.

**N1: NIT. `rtm_engine.py::RTMEngine._load_pose`** swallows the new pose refusal into `_pose_load_error` ("no pose").
This is inconsistent with ultralytics' stated "typed refusal, not 'no pose'". Also, the `_resolve_model_path`
docstring still says it raises only `FileNotFoundError`. Pick one rule for lazy pose under offline and apply it to both
engines.

**N2: NIT. `transformers_backend.py::load`.**
- Offline, a model that is not cached fails as `logger.error("Failed to load PyTorch model")` / False, untyped.
  transformers is one of the two backends V8 allows offline, so a typed "not cached locally" refusal (as for whisper)
  would help most here.
- Forcing `trust_remote_code` off is silent. A cached remote-code model then fails with a buried error; log one line
  saying offline forced it off.

**N3: NIT. `models/audio/tts.py::VoiceProvisioner._run`** turns the `ensure_tts_voice` refusal into an infinite retry
loop. Each attempt books a ledger entry, and only the first is logged at WARNING. It has no production caller yet, so
this is latent. Re-raise `OfflineRefusal`, or stop retrying, when it fires.

**N4: NIT. `models/download.py::download_file`** (`except BaseException` → print + False) makes `python -m
maxim.models.download --llm X` under offline exit 1 with "Error: offline…", not exit 2. Either refuse up front in
`main()` when offline (every verb there is a fetch), or re-raise `OfflineRefusal` in `download_file`.

**N5: NIT. `_offline.mark_degraded`.**
- It appends on every call. The encoder and semantic loaders do not cache their failure, so a long offline sim appends
  one `Degraded` per encode. The `_ledger` has the same unbounded shape for any per-call refusal (F4, N3).
- Fix: keep the first record per site (with a count).

**N6: NIT. `.github/workflows/test.yml` grep step.**
- The exemptions are whole-file.
- `socket.create_connection(` is out of the pattern. That is acceptable, because those connects are not HTTP and the
  latch refuses them, but the comment's "four non-HTTP socket sites" reads as a complete list of raw sockets, and it
  is not.

## Verdict

**SHOULD-FIX (no BLOCKER).** Every `src/` caller of every new refusal is mapped. Two of them break a documented fallback
on a default path (F1: the sim sandbox, F2: the tier re-walk). One re-types unrelated failures as refusals (F3). The
"degraded" claim has no production reader (F6). The `Refusal.thread` and CI-grep contracts are clean.
