# Troubleshooting: Remote Update & Restart (`maxim peer update` / `maxim peer restart`)

> **Audience:** Humans or AI assistants debugging remote update failures from a peer machine.
> **Companion docs:** [peer_leader_connectivity.md](peer_leader_connectivity.md), [maxim_doctor.md](maxim_doctor.md)

## How remote update works

The update endpoint auto-detects the leader's install mode:

**Pip mode** (leader installed via `pip install pymaxim`):
```
Peer runs: maxim peer update
  │
  ├── Reads URL + API key from ~/.config/maxim/peer.yml
  ├── POST /v1/admin/update {"mode": "auto"} to leader
  │
  ▼ Leader detects no .git directory → pip mode:
  ├── Checks MAXIM_ALLOW_REMOTE_UPDATE=1
  ├── Checks disk space (6GB for torch extras, 1GB otherwise)
  ├── Pre-caches current version locally (for network-independent rollback)
  ├── pip install --upgrade --index-url https://pypi.org/simple/ pymaxim[detected-extras]
  ├── Compares version before/after to distinguish "updated" from "already current"
  └── Returns result (with from_version, to_version, extras_preserved)
```

**Dev mode** (leader running from git checkout):
```
Peer runs: maxim peer update --dev
  │
  ├── POST /v1/admin/update {"mode": "dev"} to leader
  │
  ▼ Leader detects .git directory → dev mode:
  ├── Checks MAXIM_ALLOW_REMOTE_UPDATE=1
  ├── Checks working tree is clean (git status --porcelain)
  ├── git fetch origin main
  ├── git -c pull.rebase=true pull origin main
  ├── pip install -e .
  └── Returns result (with commits_applied)
```

After a successful update, run `maxim peer restart` to soft-restart the leader process and load the new code. The restart uses `os.execv` to replace the process in-place (same PID, clean Python import cycle).

## Quick commands

```bash
# ── Pip-installed leaders (default for PyPI installs) ──────────────

# Preview available version (no changes applied):
maxim peer update --dry-run

# Upgrade to latest PyPI release:
maxim peer update

# Pin to a specific version:
maxim peer update --version 0.3.1

# ── Git-checkout leaders (for development) ─────────────────────────

# Preview pending commits:
maxim peer update --dev --dry-run

# Pull latest from origin/main:
maxim peer update --dev

# Pull from a specific branch:
maxim peer update --dev feat/my-feature

# Force-update when leader has dirty tree (stashes + restores):
maxim peer update --dev --force

# ── Common to both modes ──────────────────────────────────────────

# Soft-restart leader to load new code:
maxim peer restart

# Full update + restart workflow (dev mode):
git push origin main && maxim peer update --dev && maxim peer restart

# Provide URL explicitly (instead of peer.yml):
maxim peer update https://maxim.yourdomain.com/v1

# Hot-swap the LLM model without restarting Maxim:
maxim peer llm qwen2.5-14b

# Check what model is running:
maxim peer llm --status

# Mesh-aware versions (use node names from mesh.yml):
maxim peer --node mac-studio update                  # drain → update → resume (auto-detect mode)
maxim peer --node mac-studio update --version 0.3.1  # drain → pip update → resume
maxim peer --node mac-studio update --dev feat/foo   # drain → git update → resume
maxim peer --node mac-studio update --dry-run        # preview only, no drain
maxim peer --node mac-studio restart                 # drain → restart → resume
maxim peer --node mac-studio llm qwen2.5-14b         # drain → swap → resume
```

## Decision tree

```
maxim peer update
│
├── "Connection failed" ──────────── Tunnel/network issue. Run: maxim peer test <url>
├── "Remote update is disabled" ──── Leader not in leader mode, or MAXIM_ALLOW_REMOTE_UPDATE=0
├── HTTP 404 ─────────────────────── LeaderProxy not running, or tunnel pointing at port 8100
│
├─── Pip mode (auto-detected or --version):
│    ├── "Already at latest version" ── Nothing to upgrade
│    ├── "0.3.0 → 0.3.1 available" ─── Dry-run preview. Run without --dry-run to apply
│    ├── "Updated! 0.3.0 → 0.3.1" ──── Success! Run: maxim peer restart
│    ├── "Insufficient disk space" ──── Free space on leader (need ~6GB for torch extras)
│    ├── "pip upgrade timed out" ────── Slow network. Check: pip show pymaxim on leader
│    └── "pip upgrade failed, rollback" ── Dependency issue. Check stderr for details
│
├─── Dev mode (auto-detected or --dev):
│    ├── "Leader has dirty working tree" ── Commit/stash files, or use --force
│    ├── "No git repository found" ──── Leader is pip-installed. Drop --dev flag
│    ├── "Already up to date" ────────── Nothing to pull
│    ├── "N commit(s) pending" ──────── Dry-run preview. Run without --dry-run to apply
│    ├── "Updated! N commit(s) applied" ── Success! Run: maxim peer restart
│    └── "git pull failed" ────────────── Divergent branches. Fix on leader: git pull --rebase
│
└─── "Leader does not support pip update mode" ── Leader is running < 0.3.1. Upgrade manually first

maxim peer restart
│
├── "Connection failed" ──────────── Tunnel/network issue. Run: maxim peer test <url>
├── "Remote restart is disabled" ──── MAXIM_ALLOW_REMOTE_UPDATE=0 on leader
├── "Leader is restarting" ────────── Success! Leader will be back in ~2s.
└── HTTP 404 ─────────────────────── LeaderProxy not running, or old version without restart endpoint
```

## Common failures

### HTTP 404 — LeaderProxy not reachable

**Symptom:** `Update failed (404): HTTP Error 404: Not Found`

**Causes (in order of likelihood):**
1. **Tunnel points at port 8100** (llama-cpp-server) instead of 8099 (LeaderProxy)
   - Check on leader: `grep service /etc/cloudflared/config.yml`
   - Should say `service: http://localhost:8099`
   - Fix: edit config, restart cloudflared
2. **cloudflared not restarted** after config change
   - `sudo systemctl restart cloudflared` (Linux/systemd)
   - `pkill -f cloudflared && cloudflared --config /etc/cloudflared/config.yml tunnel run &` (WSL2/manual)
3. **LeaderProxy didn't start** — port 8099 held by stale process
   - Check on leader: `ss -ltnp | grep 8099`
   - Restart maxim

**Verification:** `curl -s -H "User-Agent: maxim-peer/1.0" https://yourhost/v1/debug/status` should return JSON with GPU info. If it returns `{"detail":"Not Found"}`, the tunnel is hitting llama-cpp-server, not the proxy.

### HTTP 403 — Remote update disabled

**Symptom:** `Remote update is disabled on the leader.`

**Cause:** `MAXIM_ALLOW_REMOTE_UPDATE` is not set to `1` on the leader process.

**Fix:** Leaders in leader mode auto-enable this. If it's still disabled:
- Leader may not be detected as leader. Check: `maxim doctor` → Role section
- Need either `MAXIM_ROLE=leader` env var or `/etc/cloudflared/config.yml` present
- Explicitly: `MAXIM_ALLOW_REMOTE_UPDATE=1 maxim`

### HTTP 409 — No git repository (dev mode on pip leader)

**Symptom:** `Leader has no git repository.`

**Cause:** You used `--dev` but the leader was installed via `pip install pymaxim`, not from a git checkout.

**Fix:** Drop the `--dev` flag — the leader will auto-detect pip mode:
```bash
maxim peer update           # auto-detects pip mode
maxim peer update --version 0.3.1  # pin specific version
```

### HTTP 507 — Insufficient disk space

**Symptom:** `Insufficient disk space: X.Y GB free, need ~N GB.`

**Cause:** The leader doesn't have enough disk space for the upgrade. Torch extras need ~6GB; basic pymaxim needs ~1GB.

**Fix:**
```bash
# Free space by removing unused models:
maxim --delete-model <model-name>

# Or check disk usage on the leader:
ssh leader df -h /
```

### Pip upgrade timed out (600s)

**Symptom:** `pip upgrade timed out (600s). Environment may be inconsistent.`

**Cause:** Slow network during a large download (torch can be 2GB+). The leader's environment may be partially upgraded.

**Fix:**
```bash
# Check the current state on the leader:
pip show pymaxim

# If the version is wrong, manually fix:
pip install pymaxim==<correct-version>

# Then restart:
maxim peer restart
```

### HTTP 409 — Dirty working tree

**Symptom:** `Leader has dirty working tree:` followed by file list

**Cause:** The leader has uncommitted or untracked files.

**Fix (on leader):**
```bash
# Commit changes:
git add -A && git commit -m "wip"

# Or stash:
git stash

# Or ignore specific files:
echo "filename" >> .gitignore
```

### HTTP 500 — git pull failed

**Symptom:** `Update failed (500): git pull failed` with stderr showing git error

**Common causes:**
- **Divergent branches:** leader has local commits not on origin
  - Fix on leader: `git pull --rebase origin main`
- **Merge conflicts:** local changes conflict with incoming commits
  - Fix on leader: resolve conflicts manually, then `git rebase --continue`

### Cloudflare 1010 — Bot Fight Mode

**Symptom:** `Remote update is disabled on the leader.` (misleading — actually a Cloudflare block)

**Cause:** Cloudflare's Bot Fight Mode blocks Python's default `User-Agent`. The CLI uses `maxim-peer/1.0` to avoid this, but if you see this with a recent CLI version, Bot Fight Mode may have been re-enabled.

**Diagnosis:** Run with curl to compare:
```bash
# This should work (curl UA):
curl -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
     -X POST https://yourhost/v1/admin/update -d '{"dry_run":true}'

# This might fail (Python UA):
python3 -c "import urllib.request; urllib.request.urlopen('https://yourhost/v1/debug/status')"
```

**Fix:** Disable Bot Fight Mode on the Cloudflare zone, or add a WAF exception for the tunnel hostname.

### POST works with curl but not CLI

**Symptom:** `curl -X POST` works but `maxim peer update` fails

**Cause:** Stale `.pyc` bytecode cache. The CLI is running old code.

**Fix:**
```bash
find src/ -name "*.pyc" -delete
pip install -e .
maxim peer update
```

## Commands safe for autonomous Claude agents

These are read-only and have no side effects:

```bash
# Check connectivity:
maxim peer test <url>
maxim peer show

# Preview update (dry run, no changes):
maxim peer update --dry-run

# Compare local vs leader version:
maxim peer version

# View recent leader logs:
maxim peer logs

# Follow leader logs in real time:
maxim peer logs -f

# Check leader health:
curl -s -H "Authorization: Bearer $KEY" -H "User-Agent: maxim-peer/1.0" \
     https://yourhost/v1/debug/status
curl -s -H "Authorization: Bearer $KEY" -H "User-Agent: maxim-peer/1.0" \
     https://yourhost/v1/debug/heartbeat
curl -s -H "Authorization: Bearer $KEY" -H "User-Agent: maxim-peer/1.0" \
     https://yourhost/v1/debug/metrics
curl -s -H "Authorization: Bearer $KEY" -H "User-Agent: maxim-peer/1.0" \
     https://yourhost/v1/debug/version
curl -s -H "Authorization: Bearer $KEY" -H "User-Agent: maxim-peer/1.0" \
     https://yourhost/v1/debug/logs?limit=50
# Check VRAM pressure (returns live nvidia-smi + projected model footprint):
curl -s -H "Authorization: Bearer $KEY" -H "User-Agent: maxim-peer/1.0" \
     https://yourhost/v1/debug/vram
```

The following **modifies leader state** — use only when explicitly asked:

```bash
# Pull + install on leader:
maxim peer update

# Soft-restart leader (reloads code):
maxim peer restart
```

## Switching between pip and git install modes

The update command auto-detects the leader's current install mode and preserves it. To **switch** between modes, run the following directly on the leader (SSH or local terminal). These are one-time setup steps, not something toggled via `maxim peer update`.

### Git checkout → pip releases

Use this when you want the leader to track stable PyPI releases instead of a git branch.

```bash
# On the leader:
pip install pymaxim[semantic,llm-llama]   # adjust extras to match your setup
maxim peer restart                         # picks up the pip-installed version

# The git checkout stays on disk but is no longer the active install.
# Future `maxim peer update` from a peer will auto-detect pip mode.
```

### Pip releases → git checkout

Use this when you want the leader to track a development branch.

```bash
# On the leader:
git clone https://github.com/dennys246/Maxim.git ~/Maxim
cd ~/Maxim
pip install -e .[semantic,llm-llama]      # editable install from checkout
maxim peer restart                         # picks up the git version

# Future `maxim peer update` from a peer will auto-detect dev mode.
# Use `maxim peer update --dev feat/foo` to pull a specific branch.
```

### How auto-detection works

The leader checks for a `.git` directory at its install root. If present, it's dev mode (git pull + pip install -e). If absent, it's pip mode (pip install --upgrade pymaxim). The detection runs on every update request — there's no stored preference to reset.
