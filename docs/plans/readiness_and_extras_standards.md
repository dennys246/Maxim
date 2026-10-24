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
non-empty, finite audio that is not silence (owner decision 2026-10-18, from the review: piper casts NaN samples to
int16 zeros, so "finite" alone could not fail; the peak must exceed `tts.py::_READINESS_MIN_PEAK`). `is_ready()` then
returns that **cached** result. It never synthesizes per call, because
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

**No console exemption is needed** (design pass, fold 5, below). A console cell at either resolution moves FastAPI off
`.github/constraints/console.txt` (0.127.0, the OpenAPI generator's pin), so the snapshot and fingerprint tests
(`test_openapi_snapshot_is_fresh`, `TestContractSurfaceCannotDriftSilently`) would fail there. They never run there:
only the real-library tests carry the `extras_console` marker, and those tests stay in `unit-tests` under the pin.

**The second instance: #1347.** The `console` extra installed plain `uvicorn`, which has no WebSocket library, so `/ws`
returned 404 from a clean install (fixed in 1.3.3 by declaring `websockets` in the extra, owner decision 2026-10-18). The `/ws` tests use Starlette's `TestClient`, which never goes through uvicorn, so
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
**Since R4 (2026-10-18):** a contract bump MERGES green (CI prints a `CONTRACT DRIFT` warning), but from then on
`main` cannot be released until the console pin moves to a pulse tag built for the new contract. At the cut, either
pulse has tagged one, or the bump is reverted. The release is blocked, not silently slipped
([archive/r4_console_vendor_gate.md](archive/r4_console_vendor_gate.md), fold 1).

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
its folded design pass are [archive/r4_console_vendor_gate.md](archive/r4_console_vendor_gate.md). **"The newest pulse tag for a
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

## R2's lane: approach note (a gate; adversarial design pass before code, 2026-10-18)

*As built (the table and paragraphs below are revised to the design pass's folds, which follow them).*

**What it is.** One `extras-matrix` job in `test.yml`, a `{extra} × {floor, latest}` matrix (1.3.3: `tts`; `console`
with #1347). Owner decision 2026-10-18: the full matrix runs nightly, on `workflow_dispatch` and on a push to main; a
pull request runs only the cells whose extra its diff touches. The job names carry `(nightly)`, so `scripts/check_nightlies.py` (the
release gate) refuses a release while any cell is red: that is the "before a release" half. It is not a required
check: a cell may skip on a PR, and a skipped required check reads as passing.

**Each cell:** a fresh runner with ONLY `pip install -e ".[<extra>]"` plus pytest, under a generated `==` constraints
file (`floor`: each direct requirement's declared lower bound; `latest`: what an unconstrained dry-run resolve picks
today); a verify step asserts each pin is what got installed and that nothing outside the extra's closure is; the extra's assets are provisioned before the network namespace (tts: the pinned
voice, through `ensure_tts_voice`, from an `actions/cache` keyed on `download.py`, where the sha256 pins live); then
the files applying the marker run inside `scripts/ci_netns.sh` with `--require-extras=<extra>` and the lane's exact roster is checked
(`scripts/lane_rosters/extras-<extra>.json`, `scripts/check_lane_roster.py`).

**One helper, `scripts/extras_matrix.py`,** owns the table (extra → the paths that make a PR run it) and five
subcommands: `scope` (a PR only: should this cell run), `pins` (the `==` constraints), `verify` (installed = pinned,
and only the extra), `tests` (the files applying `extras_<extra>`) and `assets` (tts: the pinned voice). Whether a step
runs is decided in the YAML, never by the helper (fold 3).

**Inputs, and how each could lie:**

| Input | Lie | Response |
|---|---|---|
| `GITHUB_EVENT_NAME` | an event this code did not foresee | anything but `pull_request` and `push` runs the cell |
| the PR's base (merge-base with `origin/main`) | unavailable (shallow clone, a fetch failure) | the cell RUNS; never "no diff, so skip" |
| `push` to main | no PR to scope by | runs every cell, like a nightly: `scope` is consulted only on `pull_request` (fold 3) |
| changed paths | a rename hides the old path; a deletion is not a "change" | `--name-status -M`, both sides of a rename, deletions count |
| the path table | a file the cell exercises, or a consumer of the extra, changes outside the table | the table names the extra's code, what its marked tests import (tts: `optional_deps.py`, `console/server.py`, the download path's utils), its consumers, and the lane itself (`pyproject.toml`, the conftest with its guards and plugins, the workflow, the helper with `_lint_git.py`, the roster script, `ci_netns.sh`); the residual (`cli.py`, touched by nearly every PR) is caught by the next push to main, the nightly and the release gate |
| `pyproject.toml`'s extra | a requirement with no `>=` floor; extras or markers in it (`uvicorn[standard]>=0.29`) | `pins` refuses a requirement without a floor (exit 2); strips `[extras]` (pip refuses constraints with extras), keeps the marker |
| the installer | a later install moves a pin; a cached wheel; `.github/constraints/console.txt` leaking in; something else installed that satisfies the test (#1347's `websockets`) | the install runs under the `pins` file, and `verify` reads `importlib.metadata` after it: each pin must equal what is installed (`packaging.Version`), and the installed set must be exactly the metadata closure of `pymaxim[<extra>]` + pytest (fold 4, fold 6) |
| the matrix in `test.yml` | an extra added to the table but not the matrix, or the reverse; a new extra with no cells | fast-suite tests pin matrix == table == lanes == markers, and that every extra is in the matrix or on a not-yet list that may only shrink (M69, partly mechanized) |
| the tests themselves | they skip (extra missing, asset missing), or a marked test drops out | `--require-extras=<extra>` turns a skip for the extra or its asset into a failure; the exact roster check, in its own `always()` step |
| the asset cache | stale or corrupted voice | `ensure_tts_voice` re-hashes against the sha256 pins and re-fetches; a failed fetch fails the provision step, before the tests |

**Design pass (2026-10-18): no DO-NOT-BUILD; five SHOULD-FIX, folded here.** Finding 3 was not fail-closed, so the
built lane gets a further round.
1. `--require-extras=tts` needed `piper-tts` → `piper` in `extra_import_names` and a `tts` entry in the skip reasons
   (both reasons: the extra and the voice). The tts roster's `allowed_skips` is pinned at 0.
2. A lane is one marker: `extras_<extra>` (registered in `pyproject.toml`), lane `extras-<extra>` in
   `check_lane_roster.LANES`, one roster per extra serving both cells (see 5).
3. **The run/skip decision lives in the YAML, not the helper.** A job-level `if:` cannot read `matrix.*`, so cells gate
   their steps, and a cell whose steps all skip still concludes `success`. If the helper decided on a nightly, a helper
   bug would make every cell green with nothing run, and the release gate would read that as passing. So every step
   after `scope` carries `if: github.event_name != 'pull_request' || steps.scope.outputs.run == 'true'` (the roster step
   adds `always() &&`). `scope` runs only on a pull request; a push to main and every nightly run every cell. A scoped-out
   cell writes why to the step summary.
4. **Floor and latest are one mechanism: a pinned set, then equality.** `pins` writes `==` constraints (floor: each
   requirement's `>=` bound; latest: what an unconstrained `pip install --dry-run --ignore-installed --report` of the
   extra resolves today), the install uses them, and `verify` checks equality (`packaging.Version`; requirements whose
   marker is false on the runner are skipped). A holdback then fails at install, as a resolver error, which is signal;
   `pip index versions` is not used.
5. **No console exemption is needed.** The `extras_*` marker goes only on the real-library R2 tests; the snapshot and
   fingerprint tests stay in `unit-tests` under `console.txt`, so no cell selects them, at the floor or the latest.
6. (NIT, taken) **"Only the extra" is checked:** `verify` also requires the installed distributions to be exactly the
   metadata closure of `pymaxim[<extra>]` plus `pytest` (plus `pip`/`setuptools`/`wheel`), so a cell that drags in
   `websockets` some other way cannot go green over #1347.
7. (NIT, owner question) Scoping PRs adds most of the lane's surface for a non-required result; running every cell on
   every PR would delete `scope` entirely. Kept as decided (owner 2026-10-18), and offered back to the owner.
8–9. (NITs, taken) `changed_files(..., suffix="")` so `pyproject.toml`, the workflow and the roster count; `fail-fast:
   false`; Python pinned per cell; the cell installs no pytest-timeout or pytest-cov (`-p no:cov` stays banned).
10. **A release can now wait on Hugging Face and PyPI** (a voice-fetch failure on a cache miss, a breaking upstream
   release). That is intended. The response is to cap the requirement in `pyproject.toml` or fix the code; never a
   one-off override (CLAUDE.md "Weak evidence never gates").

**Floor scope (review, 2026-10-18).** A floor cell pins the extra's DIRECT requirements to their bounds; their own
dependencies resolve as pip picks today. So when an upstream drops an API a floor still uses (for example websockets
removing `websockets.legacy`, which uvicorn 0.29 imports), the floor cell turns red and holds the release. That is the
cell working: the response is to raise the floor in `pyproject.toml`, not to pin the transitive package.
