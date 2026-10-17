# pymaxim 1.3.2 — "Decomposition"

**Released 2026-10-16 (UTC — PyPI `upload_time` 20:47Z).** `pip install --upgrade pymaxim`

## Security: upgrade every leader on its own machine

**Security advisory [GHSA-7qh4-mp34-88v9](https://github.com/dennys246/Maxim/security/advisories/GHSA-7qh4-mp34-88v9)
(high):** before 1.3.2, a Maxim leader ran unauthenticated by default, and its debug routes were readable
through a Cloudflare tunnel.
- **Who is affected:** anyone who ran a machine as a leader, whether chosen on purpose or reached by default. A
  leader behind a `cloudflared` tunnel is the most exposed.
- **How it was exposed:**
  - A machine with no role signal became a leader bound to every interface.
  - A leader with no mesh key accepted every request.
  - Remote-admin routes (`/v1/admin/update|install|restart|llm-swap`) were open whenever auth was.
  - The tunnel delivers every request from `localhost`, and debug routes trusted `localhost`.
- **The fix:** 1.3.2 fails closed instead.

**Upgrade each leader ON ITS MACHINE, with a cold restart.** Do **not** use `maxim peer update && maxim peer restart`
for this release. A remote restart re-executes with the old process's environment, and a leader whose key is missing
or too open now exits with code 2, leaving no remote way back.

On each leader:
1. `pip install --upgrade pymaxim` (or pull your checkout), then run **`maxim doctor`** before starting it.
2. If the machine was a leader only by default: `maxim config set role leader`.
3. If it has no mesh key: `maxim tunnel key rotate`, then `maxim peer key set <key>` on each peer.
4. If the key file is readable by group or others: `chmod 600 ~/.config/maxim/api_key`.
5. If you want to keep `maxim peer update|install|restart|llm`: `maxim config set proxy.allow_remote_update true`.
   The old `MAXIM_ALLOW_REMOTE_UPDATE` is now ignored.
6. Stop the old process and start the leader again on that machine.

On each peer: point any direct `http://<leader>:8100` URL at **`:8099`**, the leader's one network entry point.

**What changed:**
- **The default role is `solo`.** A leader needs `MAXIM_ROLE`, `config.json::role` or a `cloudflared` config.
- **A leader without a usable key refuses to start** (exit 2). The key must be present, long enough and mode
  600; the error names the path and the fix.
- **Every proxy route needs the key.** Loopback is no longer treated as authentication, so the debug routes need it
  too.
- **Remote admin is off by default.** It needs the key and the `proxy.allow_remote_update` opt-in (`config.json`
  format 1.3; `maxim config downgrade` is the documented way back to 1.3.1).
- **The leader's llama-cpp-server listens on `127.0.0.1` only**, with its key in the environment rather than the
  command line. A server answering without its key, or reachable on the LAN without auth, is refused at start.
- **`maxim oasis serve` without a key refuses on any bind** unless you pass `--insecure`.

Details and every guard: the
[Security section of the changelog](https://github.com/dennys246/Maxim/blob/v1.3.2/CHANGELOG.md#security--a-leader-fails-closed-breaking-with-a-migration)
and [SECURITY.md](https://github.com/dennys246/Maxim/blob/v1.3.2/SECURITY.md).

## The decomposition

1.3.2 makes **no new behavioural claim**. Its work is to take the agent loop apart without changing what it does,
and to add the checks that keep it apart.
- **The loop is split.** `runtime/agent_loop.py` went from 5,543 lines to 3,363. Setup, the pre-tick gate,
  perception, planning, the substrate tick and the substrate proposer now live in their own modules
  (`runtime/loop_*.py`, `runtime/substrate_proposal.py`).
- **Each move was pinned first.** Before a slice moved, a characterization golden recorded what the old code did,
  byte for byte, and the moved code had to reproduce it.
- **Ratchets that only turn one way:** a function-length limit, per-package and per-file coverage floors, mypy on
  the composition layer, and a lint against silently swallowed exceptions.
- **1.4.0 is hard-gated** on finishing the decomposition and its findings
  ([#1300](https://github.com/dennys246/Maxim/pull/1300)).

## Defects fixed at their seams

- **A failed tool call is booked once, with its surprise.** Before, a later success could book it as a success,
  or it could be booked twice.
- **The Cerebellum's forward models are saved** at session end, where they are loaded from.
- **The SCN and the Hippocampus survive concurrent capture** during sleep and save.
- **A simulation stops the DefaultNetwork it started.**
- **The narrator hears what the agent said, and the agent is offered only its own tools**
  ([#1260](https://github.com/dennys246/Maxim/issues/1260), [#1261](https://github.com/dennys246/Maxim/issues/1261)).
- **The internet policy fails closed** on an unreadable robots.txt, checks every redirect hop, and normalizes
  hostnames with IDNA 2008 (#966–#968).
- **Mechanisms that never earned behavioural weight are marked Dormant**, rather than left implying they work.

## The Console

- **Run history and health:** past runs and machine health in the Console (contract 0.6.0).
- **Stall detection** for a planning turn that is stuck.
- **Per-action credit** recorded in `actions.jsonl` 1.2.
- **Verified Piper voice provisioning** for spoken-code pairing on a device.
- The wheel ships the maxim-pulse v0.4.0 Console.

## Evidence status

The [ledger](https://github.com/dennys246/Maxim/blob/v1.3.2/docs/plans/behavioral_graduation_candidates.md) is
authoritative. Since 1.3.1:
- **Exp 63 (T1-16): EARNED, narrow.** Memories carried from one session take part in recall in the next, on the
  current goal-keyword ranker (one run, mistral-7b). This is storage into recall, not a change in behaviour. It
  supersedes Exp 10.
- **Exp 60, 61 and 62 (T1-13, T1-14, T1-15): PARTIAL.** Every claim must now beat a learner without the bio stack.
  - In the [baseline-floors](https://github.com/dennys246/Maxim/blob/v1.3.2/docs/plans/baseline_floors.md) run B1,
    a tabular learner escaped the water 8 of 8 times at the first probe, matching the fear pathway. Random escaped
    2 of 8.
  - Exp 61 and Exp 62 are matched analytically; they were not run against the floor.
  - Each comparison against its own ablated arm still stands. What no longer stands is behaviour beyond a non-bio
    learner.
- **Exp 37 (T1-2): STALE.** It ships under the committed exception `release-1.3.2-T1-2`. The cross-model
  "Goldilocks" map may only be cited with its caveat (unverifiable, not refuted). The re-run moves to 1.4 as a
  dose-response design.

## Not claimed

No new behavioural claim. The memory, fear and drive mechanisms are bio-inspired models, not neuroscience
simulations.

## Upgrading

Besides the security changes above, these break on purpose. Each names its migration in the changelog:
- **`create.*` and `load.*` guard their stores**, and `create.*` refuses an existing store at construction
  (#939, #950, #1071).
- **Run directories resolve through `resolve_run_dir`** (#932).
- **No memory store saves over a file it never read** (#971).
- **`export_memories()["memory_summaries"]` items no longer carry `valence`.**
- **Private `agent_loop` helpers moved** to `maxim.runtime.loop_setup` and `maxim.runtime.substrate_proposal`,
  with no re-export.

The full list, with each fix's guard, is in the
[CHANGELOG](https://github.com/dennys246/Maxim/blob/v1.3.2/CHANGELOG.md).

## Files

| File | sha256 |
|---|---|
| `pymaxim-1.3.2-py3-none-any.whl` | `822e2c8f7cf449db08b90915e970e2581721a077d2b3da8c968d943e478e90de` |
| `pymaxim-1.3.2.tar.gz` | `468eb44075efc06233cb5864e6f4ea12259adf843a3a637208d4d2e74d156a25` |

[PyPI](https://pypi.org/project/pymaxim/1.3.2/) ·
[GitHub Release](https://github.com/dennys246/Maxim/releases/tag/v1.3.2) ·
Author: [Denny Schaedig](https://www.dennyschaedig.com)
