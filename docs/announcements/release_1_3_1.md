# pymaxim 1.3.1 — "Hardening"

**Released 2026-09-27 (UTC — PyPI `upload_time`).** `pip install --upgrade pymaxim`

1.3.1 makes **no new behavioural claim**. It fixes what the v1.3.0 re-score found, and each fix ships
with the guard that holds it (a test, a lint, a CI lane or a required check), because the score card
credits only what is enforced. If you share substrate, run sandboxed tools, or call the Python API,
read **Upgrading** below: several fixes change behaviour on purpose.

## Security

- **The sandbox runs what was approved, and only inside the sandbox** (#800–#802).
  - Python scripts ran at all for the first time: the restricted wrapper used to block its own imports.
  - Containment compares resolved paths, not a string prefix, and no longer follows a symlink out.
  - What runs is the content that was approved, not whatever is at the path later.
  - `SUPERVISED` approval **fails closed**: with no approver attached, a script that needs approval
    does not run.
- **A mode's limits are enforced when a tool runs** (#826). They used to be applied only to the
  prompt: a tool the mode excludes still ran when the model named it. The executor now checks the
  live mode at every dispatch. Passive mode, which observes and proposes, refuses tools that act on
  the host.
- **The agent can't switch itself into singularity** (#821, #828). The mode tool and the CLI refuse
  a self-requested switch into any mode flagged to execute code (today, singularity). So does a spoken
  phrase, since any audio in the room could say it. Passive → active is unchanged: active runs shell
  and sandbox tools under approval, which non-interactive runs auto-approve, so a passive agent can
  still leave passive on its own. That gap is tracked with the approval surface (#924, #922).
- **Autonomy requests no longer pretend** (#827). With no human approver attached, a request fails
  with that reason instead of waiting forever. The approval surface itself is #922.
- **A model-chosen web fetch connects only to the public address it checked** (#824), which closes
  DNS rebinding. The fetch byte cap now bounds the download itself (#825).
- **Tool output reaches the model fenced as untrusted data** (#823). A fetched page's own
  instructions block no longer looks like the prompt's. Fencing does not stop a model from choosing to
  follow injected text.

## Sharing substrate

- **Public bundle format 1 is frozen.** A downloaded release is a compatibility promise: every 1.x
  reads both bundle shapes as published, and changing the format needs a recorded decision.
  `docs/plans/public_format_freeze.md` says what is promised and what is not.
- **Oasis release format v2** (bundle schema 3, signed releases only).
  - A signed release carries a detached signature over every member, a signed entry index, its
    signer, a per-key release sequence and its license.
  - A receiver journals what it admitted. It refuses a second payload under the same key and sequence
    (equivocation), and a legacy v1 bundle from a key whose v2 release it already accepted (downgrade).
  - Unsigned bundles stay schema 2, readable by 1.3.0.
- **Merging no longer destroys your own causal links** (#913). Ingest paired links by outcome alone,
  so a receiver's links that differed only in context overwrote each other: one real state went from
  607 to 443 links with an *empty* donor. Donor situations that align with yours now fold together
  instead of the last one winning (#914).
- **Export privacy.** The export scrub is an allowlist at every level, and a signed release carries no
  local agent id.

## The Python API does what its docs say

- `AgentInstance.export_memories()` counts what the store holds. It reported 0 for every agent.
- The `maxim.create.agent` example runs as written, and `capture()` names the fix for a wrong
  argument type.
- `maxim.diagnose()` runs the same checks as `maxim doctor`.
- `maxim.campaign(prompt_handler=...)` works.
- A pip install gets the foundational preamble. It is read from the Constitution, which now ships in
  the wheel, and it carries every §1 hard constraint, including the actuator-speed one the old
  preamble had omitted.
- `--session <id>` on `maxim substrate`, `maxim hive pull` and `maxim roy diff` finds a simulation's
  session, and for `ingest` / `hive pull` a `maxim.create.agent()` name. Before, a bare ID was looked
  up in a directory nothing writes to.

## Release and research integrity

- **A release waits for green nightlies.** The release build refuses unless a nightly passed on the
  exact commit being released. The model-cache nightly, red for 16 nights, is green again.
- **The test process cannot reach the network.** A measured run made 52 real outbound attempts; a
  conftest guard now makes them fail (a subprocess a test spawns is not covered).
- **The prereg lint governs 1.3's own experiments.** Exp 58–62, R2 and R3 were outside it; it now
  checks 52 records, all passing.
- **The ledger's 1.3.1 trigger walk** (window `v1.3.0..042b7d90`; the one later merge, #933, fires no
  trigger). Every graduation-ledger row whose re-run trigger fired carries a dated note:
  - **Exp 10 (cross-session memory) was re-run on the rig: MAINTAINED, narrow.** Both resumes reloaded
    the saved store exactly (100 memories). The fields the persistence change added came back unchanged
    on all 100, and 3 memories surfaced on every resume turn observed. But every run stopped early on
    a known planning defect (D13, #935): one turn per resumed phase (1–3 across all five sessions),
    against eight in August. The record
    says so, and discloses the attempts that were not used.
  - **Exp 42, 45, 53b, 56, 60, 61 and 62**, the pain cascade and the reflexes were discharged with
    offline evidence. That includes checking that the new mode gate cannot refuse anything in the
    survival experiments, which run in `active` mode.
  - **Exp 37's prompt trigger fired**, since the preamble text changed. It was not re-fired; the row
    carries no claim.

## Memory strength: recorded, not yet used

The memory-strength phases land as **recording**. Under the default retention strategy they change
nothing:
- every memory records when it happened, how strongly it encoded, the body's drive pressure and
  relief at the time, and the situation it happened in;
- a strong moment tags the memories just before it in the same situation;
- situation recall is wired but nothing consumes it yet.

`maxim config set memory.strategy strength` opts into the storage-strength model. It is an
uncalibrated placeholder, not campaign-ready.

## Corrections to 1.3.0

- **Exp 61's bundles were unsigned.** The 1.3.0 notes said the fear transferred "through the shipped
  signed-bundle path". The harness exported without `--sign`. The claim rests on the shipped export and
  ingest path, not on signing; signing is covered by its own tests.
- **The `maxim substrate invalidate` upgrade step was incomplete.** Corrected on the v1.3.0 release and
  in its notes: `maxim substrate invalidate --session <ID>` prints the census, then
  `maxim substrate invalidate --session <ID> --modality world --drop-geometry <stale-tag> --apply`.
- **Exp 37/38's NAc-bias-off arm is void** (#889). It left the reward bias on. No EARNED claim rested
  on it.

## Not claimed

**Exp 62**, the fear carried by the same body into a different pool, is EARNED on the ledger
(2026-09-20). It is not a 1.3.1 claim until its different-reader pass is recorded.

## Upgrading

`pip install --upgrade pymaxim`. Behaviour that changes on purpose, most likely first:

- **Python memory API:** `Hippocampus.capture()`, `capture_from_loop()` (and its async form) and
  `store()` now **require** `encoding=` (keyword-only); `capture_from_loop` also requires
  `situation=`. A 1.3.0 call without it raises `TypeError`. Pass what you measured, or
  `encoding=EncodingSignals.unmeasured("api")` (`from maxim.memory.encoding import EncodingSignals`).
  `capture()` also raises `TypeError` for a wrong argument type.
- **The plain CLI agent and `maxim.run()` run in passive mode by default** (unchanged), and passive is
  now enforced: `bash`, `edit_file`, `git_commit`, `run_tests`, `execute_file` and the other
  host-acting tools are refused there. Say or type "maxim active" to switch.
- **Sandbox:** a script that needs approval and has no approver attached is refused
  (`APPROVAL_UNAVAILABLE`), under `SUPERVISED` and whenever no autonomy controller is present. A
  `working_dir` outside the sandbox, or a script over 120 KiB, is refused.
- **Modes and autonomy:** the agent cannot switch itself, or be switched by a phrase, into
  `singularity`; autonomy requests with no approver fail.
- **Internet policy now takes effect** (#822): a previously-off internet toggle or a hand-written
  `util/internet_policy.json` now applies, and an unreadable or ill-typed policy fails closed.
  Model-chosen fetches no longer use `HTTP(S)_PROXY` (#824), so they fail behind a mandatory proxy.
- **Signing releases:** `maxim substrate export --sign` needs `--license SPDX-ID`, and takes its
  sequence from the signing key's per-host counter. **A key you signed with before 1.3.1 needs
  `--release-sequence N` once**, since the counter has never seen it.
- **Receiving and contributing:**
  - a v2 release needs `maxim hive pull --receiver-agent-id`;
  - `hive pull` / `hive contribute` to a remote or LAN Oasis need `--api-key` (the local leader key is
    used for a loopback Oasis only);
  - a newly added Oasis refuses legacy v1 signatures by default (`maxim hive trust --accept-v1` to
    allow), and `hive add` refuses a non-canonical or duplicate key;
  - `maxim oasis publish` needs `--queen-key`, and release ids are now signed-payload digests (re-pin
    any `--release <old id>`).
- **Exporting:** an unsigned `substrate export` ships only your own learning, not material you
  ingested; `--no-identity-filter` no longer keeps free-text signatures; `contributor_id` /
  `signer_identity` must match `[A-Za-z0-9_.@:-]{1,128}`.
- **`--session`:** a bare ID is looked up in `~/.maxim/sim_reports/` (and, for `ingest` / `hive pull`,
  `~/.maxim/agents/`); an ID in more than one place, or an empty argument, is refused.
- **Validation now refuses bad inputs:** a pain or reaction intensity outside [0, 1] (or non-finite),
  a sensor reflex declaring an absolute `value:` (custom reflex YAMLs use `delta:`), a negative
  `damage_component` amount, and an unknown `memory.strategy`.
- **Doctor and diagnose:** on a machine configured as a peer, `diagnose()` now runs the peer probes,
  which make real network calls. `maxim doctor --as peer <url>` and `diagnose(peer=url)` send your
  configured key only to its own leader URL, so another URL now fails the auth check.
- **Python API:** `export_memories()` raises when the store can't be read;
  `campaign(interactive=True, prompt_handler=...)` raises `ValueError`.
- **Removed or now-required internals:** `Executor.get_last_rpe` and
  `SupervisionPolicy.allowed_mode_transitions` are removed; `propose_via_substrate` requires
  `situation_cue=`.
- **Config downgrade:** a `config.json` written by 1.3.1 carries a `memory` section that older builds
  refuse.
- **Every agent's prompt** carries the Constitution's four hard constraints word for word, and the
  preamble header lost its "(from AGENTS.md)" suffix.

The full list, with each fix's guard, is in the
[CHANGELOG](https://github.com/dennys246/Maxim/blob/main/CHANGELOG.md#131---2026-09-27--hardening).

## Score card

A blind re-score is taken at the `v1.3.1` tag, as for 1.3.0. Its evidence agents work from a
firewalled copy with the earlier cards removed. The card lands in `docs/limits/score_cards/` beside
the 1.3.0 cards.
