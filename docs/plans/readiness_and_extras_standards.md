# Readiness, extras and the console contract: four standards from the pulse seam (#1332–#1336)

Status: DESIGN NOTE, no code. Drafted 2026-10-17 from maxim-pulse's end-to-end test of the 1.3.2 wheel. Both defects were verified against `v1.3.2` and the published wheel. Target: the two defects and the first slice of each standard in 1.3.3. Pulse #42 pins `>=1.3.3` and ships as pulse v0.4.2.

## Why

Two defects shipped in 1.3.2 that no check could catch:

**#1332: spoken-code pairing is silent.**
- `PiperTTS` calls the piper API from before 1.3. The `tts` extra's floor is 1.3.0, so it fails on every installable version (reproduced on 1.3.0 and 1.8.0).
- `pairing_ready` turns true when the voice *loads*, so the page offers "listen" over silence.
- No test touches the real library.

**#1333: the wheel vendored console bundle v0.4.0.**
- That bundle predates #1279, which renamed the stall record's fields inside an untyped `data` payload.
- Nothing checks that the vendored tag is the newest one for the contract.

The shared shape is a check that passed while the capability was broken: readiness on a precondition, tests on fakes, a contract that cannot see payload fields, and a release step that cannot see which tag it vendored. Each standard below names the check that enforces it (CLAUDE.md "Enforced, or on the backlog").

## B1: a readiness flag proves its capability (#1334)

**The standard.** Any readiness or "live" flag exercises its capability once, end to end, before it reports true. If it cannot, it is renamed to say what it does check: `loaded`, `configured`.

**The first instance.** `VoiceProvisioner` synthesizes a short phrase and requires non-empty, finite audio before `is_ready()` turns true. A broken synthesis keeps `pairing_ready` false, so the page offers only the paste path.

**The audit (origin/main, 2026-10-17).** Two flags already exercise their capability: `LLMRouter.wait_ready` (a warmup completion) and `_MaximPeerBackend.health_check` stage 2. These check only a precondition and each gets a decision:
- `VoiceProvisioner.is_ready`
- `PiperTTS`/`TTSEngine.is_available`
- `LocalAudioPlayer.is_available`
- `local_server_spawner._health_check` (a 401 counts as ready)
- console `SeamStatus.live` (configuration only)
- `HealthResponse.monitoring`/`stall_detection`
- `ModelInfoWire.ready`

**The check.** A test per audited flag that breaks the capability under the precondition and asserts the flag stays false. Plus a backlog row for a lint that refuses a new `is_ready`/`*_ready` without such a test.

## B2: extras are tested against the real library, at the floor and the latest (#1335)

**The standard.**
- One test per extra drives the real library. For tts, synthesize a phrase with a small voice.
- CI runs it at the extra's floor pins and at the latest versions they resolve to.
- A fake of a third-party class is checked against the real signature (`autospec`, or an `inspect.signature` assertion).

**Today.** Only console and sign are installed from pyproject in CI (the extras lane). The rest come from an unpinned ad-hoc list, or not at all. No lane installs at the floors.

**Order.** tts first, with the #1332 fix. Then the extras the robot depends on (reachy, audio). The rest are spread over 1.3.4 and 1.3.5.

**The check.** A lane matrix of `{floor, latest}` × extra. A lint that every `[project.optional-dependencies]` key has a real-library test marked for that lane.

## B3: the console's `data` payloads are typed in the contract (#1336)

**The standard.**
- Each record kind gets a schema, as a discriminated union on `kind`: `StallData`, `HealthSample`'s allow-listed snapshot, the degraded-site fields.
- A field change is then a real OpenAPI diff, and it trips pulse's `gen:facade:check`.

**Untyped today (`console/schemas.py`):**
- `ConsoleEvent.data`
- `HealthRecord.data`
- `HealthResponse.latest`
- `ActionRecord.params`
- `DiagnoseSection.extra`
- `IdentityResponse.ui_manifest`

**Order.** The stall and health kinds first, as contract 0.7.0, or folded into the events-trail 0.7.0. Then the `ConsoleEvent` meta-kinds. The sim-log subsystem kinds come last; there are more than 30 of them.

**Until it lands.** Every note to pulse names every field change to an untyped `data` record.

**The check.** The existing contract fingerprint, once `data` is typed.

## B4: the release PR checks the vendored console (#1333)

**The standard.** The release vendors the newest pulse tag for the release's contract, then asserts it. `scripts/vendor_console_ui.py` refuses a bundle whose commit is older than the newest pulse tag carrying the same `contract_version`, or the release PR runs pulse's own `scripts/check_pymaxim_vendor.py` logic.

**Process.** The vendored tag is named in the release notes and in the note to pulse.

**This reverses a pinned design.** `tests/unit/test_vendor_console_freshness.py` pins "staleness is NOT refused by default". That test changes in the same PR, with the reason given.

**The check.** That refusal, plus `scripts/audit_release_build.py` reading `maxim-ui.json`'s `describe` and failing on a mismatch with the expected tag.

## Verifying 1.3.3 before the cut

From a maxim-pulse checkout on Linux, or in its CI job:

```bash
PYMAXIM_SPEC=/path/to/pymaxim-1.3.3-py3-none-any.whl scripts/smoke-reachy-release.sh
```

It must print `SMOKE PASSED`: a pairing request puts real float32 audio on the robot's speaker. After the release, pulse runs `scripts/check_pymaxim_vendor.py 1.3.3`. A Mac cannot run this smoke, because piper-tts 1.8.0's macOS arm64 wheel cannot find its espeak-ng-data.
