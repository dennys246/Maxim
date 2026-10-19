# Readiness, extras and the console contract: four standards from the pulse seam (#1332–#1336)

Status: DESIGN NOTE, no code. Drafted 2026-10-17 from maxim-pulse's end-to-end test of the 1.3.2 wheel, and revised
the same day after a three-lens review. Both defects were verified against `v1.3.2` and the published wheel.

The standards are named **R1–R4**. That avoids a clash with the B-labels `roadmap_1_3_x.md` already uses for the
baseline floors.

Target: the two defects, plus the first slice of each standard, in 1.3.3. Pulse #42 pins `>=1.3.3` and ships as
pulse v0.4.2.

## Why

Two defects shipped in 1.3.2 that no check could catch:
- **#1332: spoken-code pairing is silent.** `PiperTTS` calls the piper API from before 1.3, while the `tts` extra's
  floor is 1.3.0. Synthesis therefore fails on every installable version; I reproduced it on 1.3.0 and 1.8.0.
  `pairing_ready` turns true when the voice merely *loads*, so the page offers "listen" over silence. No test touches
  the real library.
- **#1333: the wheel vendored console bundle v0.4.0.** That bundle predates #1279, which renamed the stall record's
  fields inside an untyped `data` payload. Nothing checks that the vendored tag is the newest for the contract.

The shared shape is a check that passed while the capability was broken:
- readiness reported on a precondition;
- tests run against fakes;
- a contract that cannot see payload fields;
- a release step that cannot see which tag it vendored.

Each standard below names the check that enforces it. Where that check does not exist yet, it has a row in the
mechanization backlog (M68–M71), per CLAUDE.md "Enforced, or on the backlog".

## The #1332 fix and its consumers

**The fix:** `synthesize_wav(text, wav_file)` for WAV output, or concatenate `c.audio_int16_array` over
`voice.synthesize(text)` for raw samples.

**`PiperTTS.synthesize_to_file` has the same bug.**

**Consumers that read `synthesize()` and `sample_rate`,** which must still work if the dtype or rate changes:
- `utils/response_output.py`, which resamples to 16 kHz;
- `console/announcer.py`.

**`TTSEngine.is_available`** is read by `cli.py` (Session A's file) and by `embodied_runtime/agentic_runtime.py`.

**`CHANGELOG.md`'s 1.3.2 line "verified Piper voice provisioning" overclaims.** The 1.3.3 entry corrects it.

## R1: a readiness flag proves its capability (#1334)

**The standard.** Any readiness or "live" flag exercises its capability once, end to end, before it reports true.
If it cannot, it is renamed to say what it does check (`loaded`, `configured`).

**The first instance.** `VoiceProvisioner` synthesizes a short phrase, in its provisioner thread, and requires
non-empty, finite audio. `is_ready()` then returns that **cached** result. It never synthesizes per call, because
`console/server.py::_pairing_of` calls it on every unauthenticated `/api/hello`. A broken synthesis keeps
`pairing_ready` false, so the page offers only the paste path.

`VoiceProvisioner` has no in-repo caller; pulse wires it.

**The audit (origin/main, 2026-10-17).**

Already exercising their capability:
- `LLMRouter.wait_ready` (a warmup completion);
- `_MaximPeerBackend.health_check` stage 2.

Precondition only. Each gets a decision:
- `VoiceProvisioner.is_ready`
- `PiperTTS.is_available` / `TTSEngine.is_available`
- `utils/audio.py::LocalSpeaker.is_available`
- `runtime/local_server_spawner.py::_health_check`, which counts a 401 as ready. That decision belongs to Session H,
  since #1314 changes the same path.
- the console's `SeamStatus.live` (configuration only)
- `HealthResponse.monitoring` / `stall_detection`
- `ModelInfoWire.ready`

**The check.**
- A test per audited flag that breaks the capability underneath the precondition and asserts the flag stays false.
- Backlog **M68**: a lint that refuses a new `*_ready`/`is_ready` unless it has such a test.

## R2: extras are tested against the real library, at the floor and the latest (#1335)

**The standard.**
- One test per extra drives the real library. For tts: synthesize a phrase with a small voice.
- That test runs at the extra's floor pins and at the latest versions they resolve to.
- A fake of a third-party class is checked against the real signature, by `autospec` or by an `inspect.signature`
  assertion.

**Today.**
- Only console and sign are installed from `pyproject.toml` in CI, in the extras lane.
- Every other extra comes from an unpinned ad-hoc list, or is not installed at all.
- No lane installs at the floors.

**Cost.** The `{floor, latest} × extra` matrix runs nightly and before a release. A pull request runs only the
extras its diff touches.

**The console exemption.** Its "latest" lane would move FastAPI past `.github/constraints/console.txt` (0.127.0),
which pins the OpenAPI generator. So that lane skips the snapshot and fingerprint tests:
- `test_openapi_snapshot_is_fresh`;
- `TestContractSurfaceCannotDriftSilently`.

**The second instance: #1347.** The `console` extra installs plain `uvicorn`, which has no WebSocket library, so `/ws`
returns 404 from a clean install. The `/ws` tests use Starlette's `TestClient`, which never goes through uvicorn, so
they could not see it. The R2 test for `console` runs a real uvicorn server from a venv holding only that extra, and
upgrades `/ws`.

**Order.**
1. tts, together with the #1332 fix, and console's `/ws`, together with #1347.
2. The extras the robot depends on: reachy, audio.
3. The rest, over the following patches.

**The check.** Backlog **M69**: a lint that every `[project.optional-dependencies]` key has a real-library test
marked for the matrix.

## R3: the console's `data` payloads are typed in the contract (#1336)

**The standard.** Each record kind gets a schema, as a discriminated union on `kind`:
- `StallData`;
- `HealthSample`'s allow-listed snapshot;
- the degraded-site fields.

A field change then shows up as a real OpenAPI diff, and it trips pulse's `gen:facade:check`.

**Untyped today (`console/schemas.py`):**
- `ConsoleEvent.data`
- `HealthRecord.data`
- `HealthResponse.latest`
- `ActionRecord.params`
- `DiagnoseSection.extra`
- `IdentityResponse.ui_manifest`

**Producers and persisted consumers.**
- The stall and health fields are produced by `runtime/heartbeat.py::HeartbeatMonitor`. The stall dict is also
  written to the JSONL log as `extra.data`.
- `console/health.py` persists the records under `HEALTH_LOG_FORMAT_VERSION` and replays them. Its replay must
  tolerate pre-#1279 field names, or that version is bumped.

**Ordering with #1333 (a cut rule).** `vendor_console_ui.py` refuses a bundle whose contract version differs from
`CONSOLE_CONTRACT_VERSION`. So contract 0.7.0 lands in a release only together with a pulse tag built for 0.7.0.
Otherwise R3 slips past that cut, and the release vendors v0.4.1 at contract 0.6.0. Session P vendors last.

**Order.**
1. The stall and health kinds.
2. The `ConsoleEvent` meta-kinds.
3. The 30+ sim-log subsystem kinds.

**The check.**
- The existing contract fingerprint, once `data` is typed.
- Until then, a note to pulse names every field change to an untyped `data` record. That rule is backlog **M70**.

## R4: the release checks the vendored console (#1333)

**The standard.** A release vendors the newest pulse tag for its contract, then asserts it. The vendored tag is named
in the release notes and in the note to pulse.

**Built (2026-10-18, #1333):** a committed pin plus a live check (owner decision 2026-10-18); the approach note and
its folded design pass are [r4_console_vendor_gate.md](r4_console_vendor_gate.md). **"The newest pulse tag for a
contract"** means the highest stable `vX.Y.Z` tag, ordered as integers, whose `packages/kit/openapi.json`
`info.version` at the tag's commit is that contract. Tag dates are never used. Pulse's `check_pymaxim_vendor.py`
orders by creator date and includes pre-release tags; the 1.3.3 note to pulse proposes this definition for it too.

**This is a gate, so it gets a design pass before code.** "The newest pulse tag carrying the same contract version"
is a new input: a query against another repository's tags, made by a script that is offline today. The approach note
names the source and how it fails closed. Two candidates:
- a pinned list of expected tags, committed and updated in the release PR;
- pulse's `scripts/check_pymaxim_vendor.py` logic run in the release PR.

**What it touches.**
- `scripts/vendor_console_ui.py` gains the refusal.
- `scripts/audit_release_build.py` reads `maxim-ui.json`'s `describe` and fails on a mismatch.
- The `release-build` job in `test.yml` writes a stub manifest with no `describe` (`contract_version: "ci-stub"`),
  so that stub changes in the same PR.
- Three pinned tests in `tests/unit/test_vendor_console_freshness.py` are reversed, with the reason stated:
  - `test_stale_bundle_passes_by_default`;
  - `test_untagged_describe_is_surfaced_not_refused`;
  - `test_missing_commit_date_is_reported_not_refused`.
- `test_audit_release_build.py` and `test_console_ui_bundle.py`, which loads the vendor script, are further pinned
  consumers.

**The check.**
- That refusal, plus the audit.
- Naming the tag in the release notes is backlog **M71** until the audit can check the notes.

## Verifying 1.3.3 before the cut

From a maxim-pulse checkout on Linux, or in its CI job:

```bash
PYMAXIM_SPEC=/path/to/pymaxim-1.3.3-py3-none-any.whl scripts/smoke-reachy-release.sh
```

It must print `SMOKE PASSED`: a pairing request puts real float32 audio on the robot's speaker.

Pulse's browser end-to-end suite must also pass against the same wheel. It needs `/ws` (#1347).

```bash
PYMAXIM_SPEC=/path/to/pymaxim-1.3.3-py3-none-any.whl scripts/e2e.sh   # must print E2E PASSED (59/59)
```

After the release, pulse runs `scripts/check_pymaxim_vendor.py 1.3.3`.

A Mac cannot run the smoke: piper-tts 1.8.0's macOS arm64 wheel cannot find its espeak-ng-data.
