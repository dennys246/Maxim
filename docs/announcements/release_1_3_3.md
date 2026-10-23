# pymaxim 1.3.3 — "Release pipeline"

**Released 2026-10-24 (UTC).** `pip install --upgrade pymaxim`

1.3.3 ships the fixes maxim-pulse found in the Console, the Console's live stream from a clean install, and the
first pymaxim release CI publishes from a signed tag. It makes **no new behavioural claim**. Some changes break on
purpose; each is marked below with what to do.

## Security: upgrade every maxim on a host together

- **Every llama-cpp-server maxim spawns now requires a key, solo included**
  ([#1314](https://github.com/dennys246/Maxim/issues/1314)). Before, only a leader's server was keyed, so any local
  process or user on a shared host could use a solo's inference. A non-leader's server now uses a per-host key,
  `~/.config/maxim/llm_server_key` (0600; `%APPDATA%\maxim\llm_server_key` on Windows), created on first start.
  - **Upgrade every maxim on a host together.** An older maxim reuses the keyed server and gets 401s on every call.
  - **Stop any llama-cpp-server an older maxim started.** 1.3.3 refuses a server that answers without a key; the
    next start spawns a keyed one. Nothing is killed automatically.
  - **A lane you pointed by hand at a server on this host** needs `MAXIM_LANE_<TIER>_REMOTE_API_KEY`. `maxim doctor`
    prints the command.
  - **A container with a read-only home** needs a writable `~/.config/maxim` for the key file.
- **The agent can no longer change the operator's internet switch**
  ([#1345](https://github.com/dennys246/Maxim/issues/1345)). `internet_access_toggle` can only lower internet for
  its own session; only the operator turns it on.
- **The file tools refuse to write git configuration** ([#957](https://github.com/dennys246/Maxim/issues/957)):
  anything under `.git`, a git dir by identity, and the user- and system-level git config. This narrows a
  code-execution path through git filters; it does not close it, because `bash` can still write there.
- **Sim reports no longer record an endpoint's credentials, path or query**
  ([#997](https://github.com/dennys246/Maxim/issues/997)). `report.json`'s `language_endpoint` and `aut_endpoint`
  keep `scheme://host[:port]` only.
- **An unusable leader key is a typed refusal on every Python**, never a traceback or a silent "no key"
  ([#1311](https://github.com/dennys246/Maxim/issues/1311), [#1312](https://github.com/dennys246/Maxim/issues/1312)).

## The Console

- **`pymaxim[console]` serves the live stream from a clean install.** The extra now installs `websockets`, so `/ws`
  upgrades instead of returning 404 ([#1347](https://github.com/dennys246/Maxim/issues/1347)).
- **Piper speaks again** ([#1332](https://github.com/dennys246/Maxim/issues/1332)). In 1.3.2 every synthesis failed
  on every installable piper version, so spoken-code pairing played nothing. **Correction to the 1.3.2 notes:**
  "verified Piper voice provisioning" overclaimed. 1.3.2 verified the voice's files and that it loaded, not that it
  could speak.
- **Pairing arms only on a voice that speaks** ([#1334](https://github.com/dennys246/Maxim/issues/1334)). A robot
  whose voice loads but cannot speak stays on the paste sign-in.
- **Talk is offered only the tools it may use**, and a refused tool call goes back to the model instead of idling to
  the timeout ([#1412](https://github.com/dennys246/Maxim/issues/1412),
  [#1413](https://github.com/dennys246/Maxim/issues/1413)). One input is observed once, not once per tick
  ([#1414](https://github.com/dennys246/Maxim/issues/1414)). A setup change in a running `maxim serve` reaches the
  next talk without a restart ([#1411](https://github.com/dennys246/Maxim/issues/1411)).
- **`maxim serve` no longer hangs up to 15 s on shutdown** after a Console tab closes.
- **The wheel vendors the maxim-pulse Console v0.4.2** (commit `bcd0d1e`, backend contract 0.6.0). 1.3.2 shipped
  v0.4.0. Typed console payloads (contract 0.7.0) moved to 1.3.4.

## Release integrity

- **1.3.3 is the first release CI publishes.** A signed tag starts the release workflow. It requires a green nightly
  on that exact commit, smoke-tests the exact wheel in a clean environment, waits for the owner's approval, and
  publishes through PyPI trusted publishing, with no stored token.
- **The vendored Console must be the pinned, newest pulse release** for its contract; the build refuses any other
  bundle.
- **The optional extras run against their real libraries.** `tts` and `console` are installed alone, at their
  lowest supported versions and their latest, in every nightly, and the release waits for them.

## Behaviour changes and what to do

- **NAc no longer forgets while switched off.** A saved agent resumes with its learning intact, however long it was
  off. To restore the old offline decay: `NACConfig(offline_wall_decay=True)`. `NAc.load(apply_decay=...)` is
  deprecated and is removed in 1.5.0; `--resume-sim` still passes it internally, so ignore that warning.
- **A missing local model downloads only with consent, on every path** (#1343). A first `maxim.run()` in a notebook
  no longer downloads a model silently: `maxim config set llm.auto_download true`, or call
  `maxim.download_model(name)`.
- **`maxim --clear-memory` clears the whole data home and asks first** (#1337). Off a terminal it refuses with exit
  2 unless you pass `--yes`.
- **A benchmark passes only strictly** (#1310): every expectation met and at least one threshold applied. The
  shipped `cognitive_suite` cannot pass yet, for any model, and now says so.
- **A refused tool call is fed back to the model** (#1413). An LLM-driven sim can make up to two more model calls
  per refusal chain, so action counts, cost and blocked counts are not comparable to 1.3.2.
- **An agent id holding `:` is refused when the agent is named** (`InvalidAgentIdError`). NAc already refused such
  an id on every write, so the agent's learning was being lost. An agent saved under such a name by an earlier
  version must be renamed before `maxim.load.agent` will open it.
- **The model is no longer offered `execute_file`**, which was always refused. Account for it when comparing
  prompts across versions.

## Evidence status

No ledger row changed status. **Exp 37 (T1-2) stays STALE** and ships under the committed exception
`release-1.3.3-T1-2`; the cross-model "Goldilocks" map may only be cited with its caveat. Its re-run is the
dose-response design B4, which 1.4.0 waits for. The
[ledger](https://github.com/dennys246/Maxim/blob/v1.3.3/docs/plans/behavioral_graduation_candidates.md) is
authoritative.

## Not claimed

No new behavioural claim. The memory, fear and drive mechanisms are bio-inspired models, not neuroscience
simulations.

The full list, with each fix's guard, is in the
[CHANGELOG](https://github.com/dennys246/Maxim/blob/v1.3.3/CHANGELOG.md). The file digests are on the
[GitHub Release](https://github.com/dennys246/Maxim/releases/tag/v1.3.3) and
[PyPI](https://pypi.org/project/pymaxim/1.3.3/).

Author: [Denny Schaedig](https://www.dennyschaedig.com)
