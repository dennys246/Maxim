# GitHub Repo Management Plan — Fork-Based Workflow

## Problem

Currently, `maxim peer update` only pulls from `origin/main`. This means:

- Users running experiments can't isolate their changes from main
- Runtime state files (like `active_llm_model.txt`) dirty the tree and block updates
- There's no way to push local experiment changes to a fork remotely
- Branch preferences are ephemeral (per-command `--branch` flag, not persisted)

Users who fork Maxim for their own experiments need a workflow where:
1. They work on a fork branch (e.g., `experiments/denny`)
2. The leader tracks that branch across restarts
3. They can push/pull between Mac and leader without touching main
4. They can periodically rebase onto upstream main for new features

## Architecture Overview

```
GitHub
├── dennys246/Maxim (origin)          ← upstream, main branch
└── user/Maxim-experiments (fork)     ← user's fork
    └── experiments/denny             ← working branch

Mac (peer)                            Leader (RTX 5080)
─────────────                         ──────────────────
git push fork experiments/denny  →    maxim peer update --remote fork --branch experiments/denny
                                      (leader pulls from fork branch)

maxim peer push                  →    POST /v1/admin/push
                                      (leader pushes its state to fork branch)
```

## Current State

### What exists today
- `maxim peer update [--branch NAME] [--force]` — pulls from `origin/<branch>`
- `maxim peer restart` — reloads code via `os.execv`
- `PeerConfig` in `~/.config/maxim/peer.yml` — stores `url`, `api_key`, `model`, `is_cloud`
- Branch is a per-command flag, not persisted
- All git commands hardcode `origin` as the remote

### What's missing
- No `remote` field in PeerConfig
- No `branch` field in PeerConfig (persistent)
- No `maxim peer push` command
- No fork setup workflow
- Leader can't add/manage git remotes
- No protection against divergent branches

## Proposed Features

### Phase 1: Persistent branch + remote config (~150 LOC)

**Goal:** `maxim peer update` remembers the user's preferred branch and remote.

#### 1a. Extend PeerConfig

```python
@dataclass(frozen=True)
class PeerConfig:
    url: str
    api_key: str
    model: str | None = None
    is_cloud: bool = False
    branch: str = "main"           # NEW
    remote: str = "origin"         # NEW
```

- `to_yaml()` and `read_peer_config()` updated for new fields
- Backward-compatible — missing fields use defaults
- `maxim peer connect` gets `--branch` and `--remote` flags

#### 1b. Update leader_proxy.py

- `_handle_admin_update()` reads `remote` from JSON body (default: `"origin"`)
- Replace hardcoded `"origin"` with variable in git fetch/pull/log commands
- Validate remote exists: `git remote get-url <remote>` before proceeding

#### 1c. Update peer CLI

- `maxim peer update` reads `branch` and `remote` from PeerConfig
- CLI `--branch` and `--remote` flags override config values
- `maxim peer show` displays branch and remote

**Files:** `src/maxim/peer/config.py`, `src/maxim/peer/cli.py`, `src/maxim/runtime/leader_proxy.py`

---

### Phase 2: Fork setup command (~200 LOC)

**Goal:** `maxim peer fork setup <url>` configures the leader to track a fork.

#### 2a. New command: `maxim peer fork setup <fork-url>`

Sends to leader:
```
POST /v1/admin/fork-setup
{"fork_url": "https://github.com/user/Maxim-experiments.git", "branch": "experiments/denny"}
```

Leader-side:
1. `git remote add fork <url>` (or update if exists)
2. `git fetch fork`
3. `git checkout -b <branch> fork/<branch>` (or create if doesn't exist)
4. Updates PeerConfig with `remote=fork`, `branch=<branch>`

#### 2b. New command: `maxim peer fork status`

Shows:
- Current remote and branch
- Commits ahead/behind upstream main
- Dirty file count

#### 2c. New command: `maxim peer fork sync`

Rebases the fork branch onto upstream main:
1. `git fetch origin main`
2. `git rebase origin/main`
3. `git push fork <branch> --force-with-lease`

**Files:** `src/maxim/peer/cli.py`, `src/maxim/runtime/leader_proxy.py`

---

### Phase 3: Remote push command (~150 LOC)

**Goal:** `maxim peer push` pushes the leader's current state to the fork.

#### 3a. New command: `maxim peer push [--message "..."]`

Sends to leader:
```
POST /v1/admin/push
{"remote": "fork", "branch": "experiments/denny", "message": "experiment results from run 20260406"}
```

Leader-side:
1. `git add -A` (or specific paths like `data/sim_reports/`)
2. `git commit -m "<message>"`
3. `git push <remote> <branch>`

#### 3b. Configurable push paths

Not everything should be pushed. Config option for which paths to include:

```yaml
# In peer.yml or a new fork.yml
push_paths:
  - data/sim_reports/
  - data/util/active_llm_model.txt
  - scenarios/experiments/
```

Default: push everything not in `.gitignore`.

**Files:** `src/maxim/peer/cli.py`, `src/maxim/runtime/leader_proxy.py`

---

### Phase 4: Branch preservation across restarts (~50 LOC)

**Goal:** Leader stays on the fork branch after `maxim peer restart`.

Currently `os.execv` re-runs the same command, which re-enters the same git state. This should "just work" as long as we don't checkout main during restart. Verify:

- Auto-spawn reads persisted model (already done)
- No code in startup path does `git checkout main`
- `maxim peer update` respects the persisted branch

**Risk:** If the leader's crontab or systemd service does `git checkout main && python -m maxim`, the fork branch gets lost. Document this.

---

## Open Questions

### Design decisions needed

1. **Should the leader auto-add the fork remote, or require SSH setup first?**
   - HTTPS fork URLs work for public repos (read-only) but need a token for push
   - SSH fork URLs need the leader to have the user's SSH key
   - Option: support `GITHUB_TOKEN` env var for HTTPS push auth

2. **What happens when fork branch diverges from main?**
   - Rebase (Phase 2c) can conflict — how do we handle merge conflicts remotely?
   - Option: fail and tell user to SSH in for manual resolution
   - Option: `--abort` flag to cancel and stay on current state

3. **Should `maxim peer push` auto-commit, or require explicit commits?**
   - Auto-commit is convenient but creates messy history
   - Explicit commits require the user to SSH in
   - Hybrid: `maxim peer commit "message"` as a separate command, `push` only pushes

4. **Multi-user fork scenario: can multiple peers share a fork?**
   - If two peers push to the same fork branch, they'll conflict
   - Each peer could get a sub-branch: `experiments/denny/mac`, `experiments/denny/desktop`
   - Or: single-user assumption (simpler)

5. **Should sim reports be committed to the fork automatically?**
   - Pro: preserves experiment history in git
   - Con: large binary-ish files (JSON, JSONL) bloat the repo
   - Option: git LFS for `data/sim_reports/`
   - Option: separate data repo

6. **How to handle `pip install -e .` on fork branches that change dependencies?**
   - Currently runs after every pull — this is correct for forks too
   - But: if the fork adds a dependency that fails, rollback needs to work
   - Already handled: current code does `git checkout HEAD~1` + reinstall on pip failure

7. **Should there be a `maxim peer fork detach` to go back to tracking main?**
   - `git checkout main && git remote remove fork`
   - Clean separation from experiment state
   - What about uncommitted experiment data?

### Technical risks

- **Git auth on leader:** Leader may not have push credentials for the fork. Need to handle gracefully.
- **Rebase conflicts:** Can't resolve interactively over HTTP. Must fail cleanly.
- **Large repos:** If experiments generate lots of data, fork branches get heavy. Consider `.gitignore` patterns or git LFS.
- **Branch protection:** If the fork has branch protection rules, `--force-with-lease` push will fail.

### Nice-to-haves (future)

- `maxim peer fork diff` — show diff between fork branch and upstream main
- `maxim peer fork cherry-pick <commit>` — pull specific commits from main
- Integration with GitHub API (`gh`) for PR creation from fork branch
- Webhook on fork push to auto-update leader (eliminates `maxim peer update` step)

## Dependencies

- Phase 1 has no external dependencies (extends existing code)
- Phase 2-3 depend on git remote/push access from the leader
- Phase 4 depends on verifying the restart path

## Estimated Scope

| Phase | LOC | Complexity | Priority |
|-------|-----|-----------|----------|
| 1: Persistent branch/remote | ~150 | Low | High |
| 2: Fork setup + sync | ~200 | Medium | Medium |
| 3: Remote push | ~150 | Medium | Medium |
| 4: Branch preservation | ~50 | Low | High |
| **Total** | **~550** | | |

## Related Files

- `src/maxim/peer/config.py` — PeerConfig dataclass, read/write
- `src/maxim/peer/cli.py` — all peer subcommands
- `src/maxim/runtime/leader_proxy.py` — admin endpoints
- `docs/troubleshooting/remote_update.md` — existing troubleshooting guide
- `CLAUDE.md` — needs command reference updates
