# GitHub Repo Management Plan — Agent-Driven Experiment Workflow

> **Status:** Phase 0 done. Remaining phases not started.
> **Goal:** Give Maxim agents the ability to reason about, manage, and learn from git branches as part of their experimental methodology — not just CLI plumbing, but a bio-integrated experiment management system.
> **Estimated scope:** ~1,000 LOC across 7 phases (Phase 0 shipped)
> **Sequence:** Security fixes (0 ✓) → Config (1) → Git tools + hardening (2) → Provenance injection (3) → Scientist persona (4) → Fork CLI (5) → Campaign (6)

---

## Problem (Revised)

The original plan treated git fork management as pure CLI infrastructure — scripted commands piped through peer endpoints. That misses the bigger opportunity.

Maxim already has:
- 40+ tools the agent can call
- A fear/pain system that gates risky actions
- A hippocampus that remembers what happened in past experiments
- A NAc that learns which configurations lead to better outcomes
- A provenance system that traces every decision

Git branching is a **cognitive operation** — the agent should reason about when to branch, what to commit, and which experiments worked, the same way it reasons about any other domain. The fork workflow is a side effect of giving the agent proper experiment management capabilities.

### What still exists from the original plan

The peer CLI extensions (persistent branch/remote, fork setup, remote push) are still needed for **headless leader management**. But they become Phase 6 — scaffolding around the agent's own capabilities, not the core.

---

## Security Fixes (Phase 0 — do first)

These bugs exist in the current peer/leader code and must be fixed before adding any new git surface area.

### 0a. Branch parameter validation

**Problem:** `leader_proxy.py:575` accepts any string as branch name from the peer HTTP request with zero validation. Git refspec parsing can be exploited with paths like `../../../refs/heads/something`.

**Fix:** Add validation before any git command:
```python
import re
_BRANCH_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._/-]*$")

def _validate_branch(branch: str) -> str:
    """Validate and sanitize a branch name."""
    branch = branch.strip()
    if not branch or not _BRANCH_RE.match(branch):
        raise ValueError(f"Invalid branch name: {branch!r}")
    if ".." in branch or branch.startswith("-"):
        raise ValueError(f"Unsafe branch name: {branch!r}")
    return branch
```

Call at the top of `_handle_admin_update()` before any subprocess calls.

**Files:** `src/maxim/runtime/leader_proxy.py`

### 0b. Sanitize error responses

**Problem:** `leader_proxy.py:702-703` returns raw git stderr to the peer, which can contain repo paths, usernames, SSH config. The `[-500:]` truncation reduces but doesn't eliminate exposure.

**Fix:** Strip paths and sanitize before returning:
```python
def _sanitize_git_output(text: str, max_len: int = 300) -> str:
    """Remove file paths and limit length for safe error reporting."""
    import re
    text = re.sub(r"/[\w/.-]+", "<path>", text)
    return text[-max_len:] if len(text) > max_len else text
```

### 0c. Fix stash detection

**Problem:** `leader_proxy.py:611` detects stash success via `"No local changes" not in stash_result.stdout` — fragile string matching that breaks if git changes its message format.

**Fix:** Check return code instead: `stashed = stash_result.returncode == 0 and stash_result.stdout.strip()`

### 0d. Validate rollback success

**Problem:** `leader_proxy.py:731-742` does `git checkout HEAD~1` after pip failure but never checks if the rollback succeeded.

**Fix:** Check `returncode` and report failure explicitly if rollback also fails.

### 0e. Admin endpoint tests

**Problem:** Zero test coverage for `/v1/admin/update`, the most security-sensitive endpoint.

**Add:** `tests/unit/test_leader_admin.py` with:
- Branch validation (valid names, injection attempts, refspec traversal)
- Auth required for admin endpoints
- Error response sanitization
- Dirty tree stash + restore round-trip

**Estimated:** ~100 LOC

---

## Phase 1: Persistent Branch + Remote Config (~100 LOC)

Same as the original plan's Phase 1, with minor adjustments.

### 1a. Extend PeerConfig

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

### 1b. Update leader_proxy.py

- `_handle_admin_update()` reads `remote` from JSON body (default: `"origin"`)
- Replace hardcoded `"origin"` with variable in git fetch/pull/log commands (lines 631, 646, 682)
- Validate remote exists: `git remote get-url <remote>` before proceeding
- Apply same `_validate_branch()` to remote name

### 1c. Update peer CLI

- `maxim peer update` reads `branch` and `remote` from PeerConfig
- CLI `--branch` and `--remote` flags override config values
- `maxim peer show` displays branch and remote

**Files:** `src/maxim/peer/config.py`, `src/maxim/peer/cli.py`, `src/maxim/runtime/leader_proxy.py`

---

## Phase 2: Git Tool Surface (~200 LOC)

Give the agent git capabilities through the standard tool registry.

### 2a. New file: `src/maxim/tools/git.py`

Six tools, registered in `ToolRegistry`:

```python
class GitStatusTool(Tool):
    """Report current branch, dirty files, ahead/behind counts."""
    name = "git_status"
    # Read-only. No safety gate needed.

class GitBranchTool(Tool):
    """Create or switch branches."""
    name = "git_branch"
    input_schema = {"action": str, "name": str}  # action: create|switch|list
    # Safety: refuse to delete branches, refuse to switch during dirty state

class GitCommitTool(Tool):
    """Stage specific paths and commit with a message."""
    name = "git_commit"
    input_schema = {"paths": list[str], "message": str}
    # Safety: NEVER git add -A. Only stage paths the agent explicitly names.
    # Refuse to commit .env, credentials, API keys.

class GitPushTool(Tool):
    """Push current branch to remote."""
    name = "git_push"
    input_schema = {"remote": str, "force": bool}
    # Safety: force=True triggers pain signal. Push to main triggers fear gate.

class GitPullTool(Tool):
    """Fetch and merge/rebase from remote."""
    name = "git_pull"
    input_schema = {"remote": str, "branch": str}

class GitLogTool(Tool):
    """Show recent commit history."""
    name = "git_log"
    input_schema = {"count": int}
    # Read-only.
```

### 2b. Safety integration

Git tools are inherently risky. Wire them through the existing safety systems:

**FearGate rules** (in `runtime/fear_gate.py`):
- `git_push` with `force=True` → fear signal, require explicit confirmation
- `git_push` to `main` or `master` → elevated fear signal
- `git_branch` with `action=delete` → fear signal

**PainBus signals** (in `proprioception/`):
- Failed `git_push` (auth error, rejected) → `PainType.TOOL_EXECUTION_FAILED`
- Merge conflict on `git_pull` → `PainType.EXTERNAL_SIGNAL` with conflict details
- The agent learns to avoid operations that cause pain

**Tool aliases** (in `runtime/executor.py`):
- `check_repo` → `git_status` (hallucination redirect)
- `save_changes` → `git_commit`
- `upload_code` → `git_push`

### 2c. Sensitive path blocklist

`GitCommitTool` refuses to stage files matching:
```python
_SENSITIVE_PATTERNS = {
    ".env", "credentials", "api_key", "secret",
    "*.pem", "*.key", "id_rsa", "*.p12",
}
```

Any match raises `PainType.TOOL_INVALID_INPUT` with an explanation.

All path checks must use `Path.resolve()` to defeat symlink confusion (see Security Review).

### 2d. Security hardening (non-negotiable, ships with Phase 2)

These are **mandatory** for Phase 2 to be considered complete. See Security Review section for full rationale.

1. **Remote URL validation** — `_validate_remote_url()` blocks `file://`, `ftp://`, localhost, cloud metadata endpoints. Called before any push/pull.
2. **Git config isolation** — all git subprocesses run with `GIT_CONFIG_NOSYSTEM=1` and `GIT_CONFIG_GLOBAL=/dev/null` env vars. No tool may write to `.git/config` or `.git/hooks/`.
3. **VCS action category in FearGate** — new `"vcs_operation"` category with per-tool review levels.
4. **Commit message sanitization** — strip control chars, null bytes, cap at 2000 chars.
5. **Freeze tool aliases** — `register_aliases()` must refuse after agent loop starts.
6. **Implicit state tracking** — after git_commit/git_pull, run `git diff --name-only HEAD~1` and feed affected files to pain interceptor retroactively.

---

## Phase 3: Git Provenance Injection (~150 LOC)

Every memory, decision trace, and experiment record gets tagged with git context automatically.

### 3a. Git context utility

New function in `src/maxim/utils/git_context.py`:

```python
def get_git_context() -> dict[str, str]:
    """Return current git state for provenance tagging.

    Returns dict with: branch, commit, remote_url, dirty (bool).
    Fast — cached for 30s (git state doesn't change mid-experiment).
    """
```

Cached with `@lru_cache` + TTL to avoid subprocess overhead on every memory capture.

### 3b. Hippocampus integration

In `memory/hippocampus.py`, at capture time, inject git context into `EpisodicMemory.metadata`:

```python
# In Hippocampus.capture() or _form_memory()
from maxim.utils.git_context import get_git_context
memory.metadata["git_context"] = get_git_context()
```

No dataclass changes needed — `metadata: dict[str, Any]` already exists at `types.py:388`.

This means every memory the agent forms knows which branch and commit it was formed under. When the agent recalls memories across experiments, it can distinguish "I learned this on branch `recall-v3`" from "I learned this on main."

### 3c. Provenance integration

In `provenance/collector.py`, when starting a trace:

```python
# In ProvenanceCollector.begin_trace()
trace.entries[0].metadata["git_context"] = get_git_context()
```

Again, no schema changes — `ProvenanceEntry.metadata` at `types.py:87` is already a dict.

### 3d. NAc integration

In `decisions/nac.py`, when recording outcomes:

```python
# In NAc.record_outcome_full()
link.event_context["git_context"] = get_git_context()
```

`CausalLink.event_context` at `causal_link.py:137` is already a dict. This means the NAc can learn "experiments on branch X → better recall scores" as a causal pattern.

### 3e. Experiment log integration

In `simulation/research_tools.py`, extend `ExperimentLog.record()`:

```python
# Add to the entry dict in record()
entry["git_context"] = get_git_context()
```

### 3f. Simulation report

In `simulation/report.py`, add git metadata to the session-level report:

```python
# In save_report() or equivalent
report_data["git_context"] = get_git_context()
```

This creates a `git_metadata` section in every `data/sim_reports/{session}/report.json`.

**Key principle:** All integration uses existing extensible dicts. Zero dataclass changes. Full backward compatibility — old memories load fine without git context.

---

## Phase 4: Scientist Persona (~100 LOC)

A new simulation persona that uses git branching as part of its experimental methodology.

### 4a. New persona: `scientist`

In `simulation/personas.py`:

```python
SCIENTIST_SYSTEM = """You are a methodical experiment scientist.

Your workflow:
1. BEFORE any experiment: create a git branch for isolation (git_branch tool)
2. CHECK current state: use git_status to understand your working context
3. RUN the experiment with clear hypotheses
4. OBSERVE results through the bio-system introspection tools
5. COMMIT results to the branch (git_commit — only data/sim_reports/ paths)
6. COMPARE against baseline: recall memories from other branches
7. If results are significant: push to remote for preservation

You NEVER push to main directly. You NEVER force push. You ALWAYS
commit with descriptive messages explaining what the experiment tested.

You reason about branching the way you reason about any domain:
- "Should I branch here?" = "Is this experiment distinct enough to isolate?"
- "Should I merge?" = "Did this experiment produce learnings worth keeping?"
- "Which branch was better?" = "What does my causal memory say about outcomes?"
"""
```

### 4b. Persona behavior hooks

The scientist persona gets additional context injected by the orchestrator:
- Current branch and dirty state (via `git_status` at sim start)
- Previous experiment branches (via hippocampus recall of git-tagged memories)
- NAc confidence about "which branches produced the best outcomes"

This makes the persona **learn across sessions** — after 5 experiments, it has causal models about branching strategy.

---

## Phase 5: Fork CLI Commands (~200 LOC)

For remote leader management. Same as original plan's Phases 2-3.

### 5a. `maxim peer fork setup <fork-url> [--branch NAME]`

Sends to leader:
```
POST /v1/admin/fork-setup
{"fork_url": "https://github.com/user/Maxim-experiments.git", "branch": "experiments/denny"}
```

Leader-side:
1. Validate URL format
2. `git remote add fork <url>` (or update if exists)
3. `git fetch fork`
4. `git checkout -b <branch> fork/<branch>` (or create)
5. Update PeerConfig with `remote=fork`, `branch=<branch>`

### 5b. `maxim peer fork status`

Shows: current remote/branch, commits ahead/behind upstream main, dirty file count.

### 5c. `maxim peer fork sync`

Rebases fork branch onto upstream main:
1. `git fetch origin main`
2. `git rebase origin/main`
3. On conflict: **fail cleanly**, print the conflicts, tell user to SSH in. Don't try to be clever.
4. On success: `git push fork <branch> --force-with-lease`

### 5d. `maxim peer push [--message "..."]`

Push leader state to fork. Uses configurable `push_paths`:
```yaml
# In peer.yml
push_paths:
  - data/sim_reports/
  - scenarios/experiments/
```

Default: push only experiment data, never source code modifications.

### 5e. `maxim peer commit "message"`

Separate from push. Stages specific paths (from `push_paths`) and commits. The hybrid approach — explicit commit + explicit push — avoids messy auto-commit history.

**Files:** `src/maxim/peer/cli.py`, `src/maxim/runtime/leader_proxy.py`

---

## Phase 6: Database Recovery Campaign (~1 campaign YAML)

A DM campaign scenario where the agent must diagnose and fix a broken database, exercising git tools, diagnostic reasoning, and memory systems together.

**File:** `scenarios/campaigns/broken_database_v1.yaml`

### Sleep → Wake via SEM Comms

The campaign is designed for the agent to start in **hibernate mode** (no-LLM sleep) and be woken by a PagerDuty alert delivered through a SEM comms sensor. The full SEM entity tree (workstation with pager, terminal, and comms modulators; world entities for PagerDuty and production_db) is spec'd in the YAML as comments, ready to wire when prerequisites land.

**Prerequisites (not yet implemented):**
1. **Hibernate mode** — a no-LLM sleep state where the agent consumes zero inference cost. The agent loop monitors only SEM sensors and wake keywords. This is tracked in `future_plans.md` and must ship before the sleep→wake arc works at runtime.
2. **DM schema `embodiment:` key** — extend `dm_schema.py` to parse an `embodiment:` top-level key and pass the Entity tree to the campaign runtime.
3. **DM schema `initial_state:` key** — extend `dm_schema.py` to set `processing_state`, `operational_mode`, and `time_of_day` at campaign start.

Until these prerequisites land, the campaign runs with the agent starting awake and the sleep→wake transition conveyed through narrative only. The SEM spec is preserved in comments so it can be activated with zero rewriting.

### What the campaign tests
- Git tool usage (branch before fixing, commit after fixing)
- Diagnostic reasoning (read logs, identify root cause)
- Memory systems (recall similar past incidents)
- Causal learning (NAc links: "which fix strategy works?")
- Pain/fear gating (don't force-push, don't drop tables without confirmation)
- Sleep→wake transition (when hibernate mode + SEM schema extensions land)
- SCN temporal awareness (2 AM incident, time pressure)

---

## Open Questions (Resolved)

| Original Question | Resolution |
|---|---|
| Auto-add fork remote or require SSH setup? | Support `GITHUB_TOKEN` env var for HTTPS push. SSH is optional. |
| What happens when fork branch diverges? | Fail cleanly on rebase conflict. Print conflicts. Tell user to SSH in. |
| Auto-commit or explicit commits? | Hybrid: `maxim peer commit "msg"` + `maxim peer push`. Separate operations. |
| Multi-user fork scenario? | Single-user assumption. Multi-user is a mesh problem, not a git problem. |
| Should sim reports be committed automatically? | No. Agent decides via scientist persona. Configurable `push_paths` for CLI. |
| `pip install -e .` on fork branches? | Already handled — current code does rollback on pip failure. Phase 0d validates rollback. |
| `maxim peer fork detach`? | Yes. `git checkout main && git remote remove fork`. Warn about uncommitted experiment data. |

---

## Dependencies

- Phase 0 (security): No dependencies. Do first.
- Phase 1 (config): No dependencies.
- Phase 2 (git tools): Depends on tool registry (exists), fear gate (exists), pain bus (exists).
- Phase 3 (provenance): Depends on hippocampus (exists), NAc (exists), provenance system (exists).
- Phase 4 (persona): Depends on Phase 2 (git tools exist for persona to use).
- Phase 5 (fork CLI): Depends on Phase 1 (persistent config).
- Phase 6 (campaign — narrative only): Depends on Phase 2 (git tools) and Phase 4 (scientist persona).
- Phase 6 (campaign — full SEM + hibernate): Depends on **hibernate mode** (future_plans.md), DM schema `embodiment:` extension, and DM schema `initial_state:` extension. These are external prerequisites, not part of this plan.

Phases 1, 2, 3 can run in parallel after Phase 0.

---

## Estimated Scope

| Phase | LOC | Complexity | Priority |
|-------|-----|-----------|----------|
| 0: Security fixes + tests | ~100 | Low | **DONE** |
| 1: Persistent branch/remote config | ~100 | Low | High |
| 2: Git tool surface + safety + security hardening (2d) | ~350 | High | High |
| 3: Git provenance injection | ~150 | Low | Medium |
| 4: Scientist persona | ~100 | Low | Medium |
| 5: Fork CLI commands | ~200 | Medium | Low |
| 6: Database recovery campaign | ~1 file | Low | Fun |
| **Total** | **~1,000** | | |

---

## Security Review (2026-04-08)

Deep review of the existing safety infrastructure against what git tools would introduce. Phase 0 fixes are shipped. These are the **remaining gaps** that must be addressed during Phase 2 implementation.

### Critical: Remote URL validation (Phase 2)

`git_push` and `git_pull` accept a remote name which resolves to a URL. There is no URL validation anywhere in the codebase — `FilesystemPolicy` gates file I/O but has no `check_network_permission()`. An agent could:
- Push to an arbitrary URL (data exfiltration)
- Pull from a malicious repo (supply chain attack via hooks)
- Use `file://` protocol to read local paths outside the repo

**Fix (must ship with Phase 2):**
```python
_ALLOWED_PROTOCOLS = {"https", "ssh", "git"}
_BLOCKED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "169.254.169.254"}  # metadata service

def _validate_remote_url(url: str) -> str:
    """Reject file://, ftp://, localhost, metadata endpoints."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme and parsed.scheme not in _ALLOWED_PROTOCOLS:
        raise ValueError(f"Protocol {parsed.scheme!r} not allowed")
    if parsed.hostname and parsed.hostname in _BLOCKED_HOSTS:
        raise ValueError(f"Host {parsed.hostname!r} blocked")
    return url
```

Call this when resolving the remote URL from `git remote get-url <remote>` before any push/pull.

### Critical: Git config injection (Phase 2)

`git config` can set `core.hooksPath`, `core.fsmonitor`, `diff.*.textconv`, and other options that execute arbitrary commands on subsequent git operations. The agent must NEVER be able to modify `.git/config` or `.git/hooks/`.

**Fix:**
- `GitCommitTool` and all git tools: run with `GIT_CONFIG_NOSYSTEM=1` and `GIT_CONFIG_GLOBAL=/dev/null` env vars to prevent reading global/system configs
- Block any tool param that contains `.git/` as a path
- FearGate: classify any attempt to write inside `.git/` as `code_execution` severity

### High: No VCS action category in FearGate

`fear_gate.py` classifies actions as `shell_exec`, `file_write`, `network_request`, or `tool_call`. Git operations span all four but aren't explicitly categorized.

**Fix:** Add a `"vcs_operation"` category:
```python
_VCS_TOOLS = {"git_status", "git_branch", "git_commit", "git_push", "git_pull", "git_log"}

# In _classify_action():
if tool_name in _VCS_TOOLS:
    return "vcs_operation"
```

FearAgent reviews:
- `git_status`, `git_log` → auto-allow (read-only)
- `git_branch` → allow with logging
- `git_commit` → review paths being staged
- `git_push` → elevated review (network + irreversible)
- `git_pull` → review (could introduce malicious code)

### High: Commit message injection

`git commit -m <message>` where `message` comes from the agent's LLM output. While `subprocess.run(list_args)` prevents shell injection, the commit message itself could contain:
- Escape sequences that corrupt terminal display
- Extremely long messages that exhaust disk
- Encoded payloads smuggled into git history

**Fix:** Sanitize commit messages:
```python
def _sanitize_commit_message(msg: str, max_len: int = 2000) -> str:
    msg = msg.replace("\x00", "")  # null bytes
    msg = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f]", "", msg)  # control chars
    return msg[:max_len]
```

### High: Tool alias injection

`executor.py` `register_aliases()` has no validation. An agent could theoretically register aliases that redirect git tool calls to other tools or bypass safety checks.

**Fix:** Aliases should be frozen after tool registry initialization. Add a `_frozen: bool` flag to the alias system and refuse runtime registration after the agent loop starts.

### Medium: Pain interceptor doesn't track implicit git state

`pain_interceptor.py` extracts paths from tool params to fire pain signals. But git commands modify implicit repo state — `git commit` affects every staged file, `git pull` can change arbitrary files. The interceptor doesn't see these.

**Fix:** After `git_commit` and `git_pull` execute, run `git diff --name-only HEAD~1` to get the actual files affected, then retroactively evaluate pain signals. This is a post-execution check, not a gate — but it feeds the NAc so the agent learns consequences.

### Medium: Symlink confusion in .git/

`.git/` can contain symlinks (git worktrees, submodules). `os.path.realpath()` resolves them but the sensitive path blocklist in `GitCommitTool` (Phase 2c) must check the resolved path, not the input path.

**Fix:** Always `Path.resolve()` before checking against `_SENSITIVE_PATTERNS`.

### Low: Branch proliferation (long-term)

Over time, the scientist persona will create many experiment branches. Without cleanup, the repo accumulates dead branches.

**Fix (Phase 5, not Phase 2):**
- `maxim peer fork prune` — delete merged/stale branches older than N days
- Scientist persona: after pushing results, create a causal memory that branches should be cleaned up
- NAc learns: "too many branches → longer fetch times → pain signal"

---

## Long-Term Repo Management

### Branch naming convention

Enforce a prefix convention so experiment branches are distinguishable from feature work:
```
experiments/<date>/<topic>     — scientist persona branches
fix/<issue>                    — incident fix branches (broken-database campaign)
research/<hypothesis>          — research protocol branches
```

The `git_branch` tool should enforce the prefix for agent-created branches. Human branches are unrestricted.

### Repo size growth

Experiment data (`data/sim_reports/`, `*.jsonl`) committed to branches grows the repo. Mitigation options:
1. **`.gitignore` experiment data by default** — only the scientist persona explicitly stages it
2. **Branch-level `.gitattributes`** — git LFS for `*.jsonl` and `*.json` over 1MB
3. **Prune old branches** — merged experiment branches get deleted after results are extracted
4. **Separate data repo** — experiment data in a separate repo, main repo stays lean

Recommendation: Option 1 (`.gitignore` by default) + Option 3 (prune) for v1. Revisit LFS if repo exceeds 1GB.

### Fork divergence strategy

When a fork branch falls behind main by many commits, rebase becomes painful. Strategy:
1. **< 20 commits behind:** auto-rebase via `maxim peer fork sync` (Phase 5c)
2. **20-100 commits behind:** warn user, suggest rebase, don't auto-execute
3. **> 100 commits behind:** suggest starting a fresh branch from main, cherry-pick experiment data only

The scientist persona should track divergence via `git_status` and form a causal memory: "large divergence → painful rebase → branch from main more often."

### Secret scanning

Even with `_SENSITIVE_PATTERNS` in `GitCommitTool`, secrets can slip through in unexpected ways (hardcoded in source, pasted in commit messages, base64-encoded). Long-term:
1. **Pre-commit hook integration** — if the repo has a `.pre-commit-config.yaml`, `GitCommitTool` should run `pre-commit run --files <staged>` before committing
2. **Entropy scanning** — flag strings with high entropy (API keys, tokens) in staged diffs
3. **Pain signal on secret detection** — even after commit, retroactive pain signal so the agent learns

### Audit trail

Every git operation the agent performs should be logged to the provenance system (Phase 3 handles this). But for long-term repo management, also consider:
1. **Git notes** — attach provenance trace IDs to commits via `git notes add`
2. **Commit message metadata** — append `[maxim:trace_id=<id>]` to commit messages so humans can trace agent decisions from git history
3. **Session-level git log** — `data/sim_reports/{session}/git_operations.jsonl` recording every git command the agent ran, with timestamps and outcomes

---

## Related Files

- `src/maxim/tools/` — tool registry, existing tools
- `src/maxim/peer/config.py` — PeerConfig dataclass
- `src/maxim/peer/cli.py` — peer subcommands
- `src/maxim/runtime/leader_proxy.py` — admin endpoints
- `src/maxim/runtime/fear_gate.py` — safety gating
- `src/maxim/proprioception/` — pain detection
- `src/maxim/memory/hippocampus.py` — episodic memory
- `src/maxim/decisions/nac.py` — causal learning
- `src/maxim/provenance/` — decision tracing
- `src/maxim/simulation/personas.py` — orchestrator personas
- `src/maxim/simulation/research_tools.py` — experiment logging
- `scenarios/campaigns/` — existing campaign YAMLs
