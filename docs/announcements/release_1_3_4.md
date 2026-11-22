# pymaxim 1.3.4 — "Robot and recall"

**Released 2026-11-22 (UTC).** `pip install --upgrade pymaxim`

1.3.4 ships the defects maxim-pulse (the separate project that builds the Console and the Reachy app) found on a real
Reachy Mini running 1.3.3, Talk's recall across sessions, and a stop that saves before it is killed. Some changes
break on purpose; each is marked in the
[CHANGELOG](https://github.com/dennys246/Maxim/blob/v1.3.4/CHANGELOG.md) with what to do.

## Security

Read the Security section of the
[CHANGELOG](https://github.com/dennys246/Maxim/blob/v1.3.4/CHANGELOG.md) first. In short:

- **Sim records no longer keep a URL's credentials, path or query, and HTTP logs no longer keep its username and
  password** ([#1404](https://github.com/dennys246/Maxim/issues/1404)).
- **A page the model fetches can no longer hold the call open by sending slowly**: `fetch_url` takes a whole-request
  deadline ([#1715](https://github.com/dennys246/Maxim/issues/1715)).
- **Cloud redaction renders private prompt sections by identity.**

## The robot

- **It speaks, at the right speed.** The Reachy speaker sink no longer reports a silent drop as played, resamples to
  the device's rate, and refuses loudly when the player is not started
  ([#1435](https://github.com/dennys246/Maxim/issues/1435)). `maxim --tts` speaks, or refuses at start
  ([#1437](https://github.com/dennys246/Maxim/issues/1437)).
- **It hears again.** Voice input had been off since April behind one warning
  ([#1436](https://github.com/dennys246/Maxim/issues/1436)).
- **Motion safety.** No non-finite, out-of-range-duration or instant command reaches a motor
  ([#1496](https://github.com/dennys246/Maxim/issues/1496)), and no too-fast, wrongly-referenced or mid-path-unsafe
  one either: a turn that would sweep the neck through about 180° mid-move is split or refused
  ([#1499](https://github.com/dennys246/Maxim/issues/1499)). A move's outcome reaches the model, not just "queued".
- **Inbound comms reach the robot**: every comms endpoint had answered 422, so no inbound SMS reached it
  ([#1466](https://github.com/dennys246/Maxim/issues/1466)), and a message now reaches the model framed as untrusted
  data ([#1481](https://github.com/dennys246/Maxim/issues/1481)).
- **The controller cleans up after itself.** It gives the daemon's camera and audio back after a `no_media`
  connection and closes its SDK client on disconnect
  ([#1704](https://github.com/dennys246/Maxim/issues/1704), [#1712](https://github.com/dennys246/Maxim/issues/1712)),
  and a fresh pose read gets its full deadline ([#1706](https://github.com/dennys246/Maxim/issues/1706)).
- **Known oddity, under investigation.** On our test unit the head stops near +56° of yaw on one side and about −50°
  on the other, short of the ±65° clamp ([#1707](https://github.com/dennys246/Maxim/issues/1707)). The motion code
  behind it is sound by its tests: its clamp matches Pollen's documented ±65° limit, the SDK's default (analytical)
  kinematics solve to 65°, and the clamp and path tests pass. Whether the shortfall is this unit's mechanism (two of
  its six head motors were replaced, and the errors behave like friction) or the head's real range with the body held
  (Pollen's Placo model caps head-to-body yaw at 55°) is an open question with Pollen. The full embodied run on the
  robot is still owed ([#1708](https://github.com/dennys246/Maxim/issues/1708)).

## Recall

- **What you tell Talk is now stored, and a later question that shares a word with it can recall it**, across a
  restart and past compression ([#1444](https://github.com/dennys246/Maxim/issues/1444),
  [#1163](https://github.com/dennys246/Maxim/issues/1163), [#1064](https://github.com/dennys246/Maxim/issues/1064),
  [#1447](https://github.com/dennys246/Maxim/issues/1447), [#1137](https://github.com/dennys246/Maxim/issues/1137)).
  Recalled words are framed as untrusted content, because text replayed into later prompts is a stored-injection
  channel.
- **What this does not claim.** Recall matches words (stemmed keys), not meaning. A question that shares no word with
  what was said, or only a spelling variant, recalls nothing
  ([#1516](https://github.com/dennys246/Maxim/issues/1516)). The behavioural evidence is narrow: ledger row T1-16 is
  EARNED (narrow) on [Exp 63c](https://github.com/dennys246/Maxim/blob/v1.3.4/docs/experiments/exp63c_scripted_recall_prereg.md),
  one scripted 8-turn resume on one model whose cues reuse the words planted before it. The raise is an owner reading
  of BF2, and the agent's own model slot is not machine-established. Under an LLM narrator (Exp 63b) no decisive turn
  occurred, and the bare-LLM floor for this row is not yet measured.

## Stopping

- **A stop saves memory first and finishes within the caller's deadline**, so a Reachy stop no longer loses the
  session-end save to the daemon's ~20 s kill ([#1438](https://github.com/dennys246/Maxim/issues/1438)).
- **SIGTERM saves too**, and a forced stop kills every LLM server and tunnel maxim started and nothing it did not
  ([#1344](https://github.com/dennys246/Maxim/issues/1344)). `maxim llm reap` cleans up a leftover server, at a
  terminal only.

## The atonia gate: first pieces, Experimental

- `maxim.advisor`, an observer that judges a batch of events and proposes but cannot act; every tool declares what it
  can cause; and every executor takes a required `ToolPolicy` (**breaking**: `permissions=` is removed)
  ([#1315](https://github.com/dennys246/Maxim/issues/1315)).
- **The gate is not complete.** The offline latch (`MAXIM_OFFLINE`) ships but is not yet safe for operators
  (`MAXIM_OFFLINE=1 maxim doctor` crashes), and this release has no `llm.offline` mode.

## The Console

- **The wheel bundles the maxim-pulse Console v0.4.4** (commit `9af8a57`; contract 0.6.0, unchanged).
- **Correction to the 1.3.3 notes:** they said typed payloads would come in 1.3.4. They did not. Typed payloads (R3,
  [#1336](https://github.com/dennys246/Maxim/issues/1336), contract 0.7.0) wait for a pulse release built for 0.7.0.

## Evidence

- **T1-16** moved from STALE to EARNED (narrow) on 2026-11-13, as above.
- **T1-2** (Exp 37) ships STALE under the committed exception `release-1.3.4-T1-2`; it carries no EARNED claim.
- **T1-13, T1-14 and T1-15** stay PARTIAL (floor matched under BF2).

## Upgrading

- **Breaking changes**, each with its migration in the CHANGELOG: a required `ToolPolicy` on every executor;
  `maxim.run()`'s path-taking file tools reach only the run's workspace; `TimeoutPolicy.total_s` is renamed `pool_s`;
  `create.pool` needs an LLM router, and `create.router` raises `ConfigurationError` instead of returning `None`;
  custom `ResponseOutput` speakers.
- **Hippocampus files** move to `_format_version` 1.2 (1.1 stores a tool result as data, not its repr; 1.2 keeps
  your words through compression). Older files load.

Full list: [CHANGELOG.md](https://github.com/dennys246/Maxim/blob/v1.3.4/CHANGELOG.md).
