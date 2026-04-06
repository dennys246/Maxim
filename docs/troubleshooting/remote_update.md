# Troubleshooting: Remote Update & Restart (`maxim peer update` / `maxim peer restart`)

> **Audience:** Humans or AI assistants debugging remote update failures from a peer machine.
> **Companion docs:** [peer_leader_connectivity.md](peer_leader_connectivity.md), [maxim_doctor.md](maxim_doctor.md)

## How remote update works

```
Peer runs: maxim peer update
  │
  ├── Reads URL + API key from ~/.config/maxim/peer.yml
  ├── POST /v1/admin/update to leader via Cloudflare tunnel
  │     (with User-Agent: maxim-peer/1.0 to avoid Bot Fight Mode)
  │
  ▼ Leader's LeaderProxy (:8099) handles the request:
  ├── Checks MAXIM_ALLOW_REMOTE_UPDATE=1 (auto-enabled in leader mode)
  ├── Checks working tree is clean (git status --porcelain)
  ├── git fetch origin main
  ├── git -c pull.rebase=true pull origin main
  ├── pip install -e .
  └── Returns result to peer
```

After a successful update, run `maxim peer restart` to soft-restart the leader process and load the new code. The restart uses `os.execv` to replace the process in-place (same PID, clean Python import cycle).

## Quick commands

```bash
# Preview pending commits (no changes applied):
maxim peer update --dry-run

# Pull + install:
maxim peer update

# Soft-restart leader to load new code:
maxim peer restart

# Full update + restart workflow:
git push origin main && maxim peer update && maxim peer restart

# Target a specific branch:
maxim peer update --branch dev

# Provide URL explicitly (instead of peer.yml):
maxim peer update https://maxim.yourdomain.com/v1
```

## Decision tree

```
maxim peer update
│
├── "Connection failed" ──────────── Tunnel/network issue. Run: maxim peer test <url>
├── "Remote update is disabled" ──── Leader not in leader mode, or MAXIM_ALLOW_REMOTE_UPDATE=0
├── "Leader has dirty working tree" ─ Untracked/modified files on leader. Commit or stash them.
├── "git pull failed" ────────────── Divergent branches. Run on leader: git pull --rebase origin main
├── "pip install failed, rolled back" ── Dependency issue. Check stderr output for details.
├── "Already up to date" ────────── Nothing to pull. Leader has latest code.
├── "Updated! N commit(s) applied" ── Success! Run: maxim peer restart
└── HTTP 404 ─────────────────────── LeaderProxy not running, or tunnel pointing at port 8100

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
   - Check on leader: `grep service ~/.cloudflared/config.yml`
   - Should say `service: http://localhost:8099`
   - Fix: edit config, restart cloudflared
2. **cloudflared not restarted** after config change
   - `sudo systemctl restart cloudflared` (Linux/systemd)
   - `pkill -f cloudflared && cloudflared --config ~/.cloudflared/config.yml tunnel run &` (WSL2/manual)
3. **LeaderProxy didn't start** — port 8099 held by stale process
   - Check on leader: `ss -ltnp | grep 8099`
   - Restart maxim

**Verification:** `curl -s -H "User-Agent: maxim-peer/1.0" https://yourhost/v1/debug/status` should return JSON with GPU info. If it returns `{"detail":"Not Found"}`, the tunnel is hitting llama-cpp-server, not the proxy.

### HTTP 403 — Remote update disabled

**Symptom:** `Remote update is disabled on the leader.`

**Cause:** `MAXIM_ALLOW_REMOTE_UPDATE` is not set to `1` on the leader process.

**Fix:** Leaders in leader mode auto-enable this. If it's still disabled:
- Leader may not be detected as leader. Check: `maxim doctor` → Role section
- Need either `MAXIM_ROLE=leader` env var or `~/.cloudflared/config.yml` present
- Explicitly: `MAXIM_ALLOW_REMOTE_UPDATE=1 maxim`

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

# Check leader health:
curl -s -H "Authorization: Bearer $KEY" -H "User-Agent: maxim-peer/1.0" \
     https://yourhost/v1/debug/status
curl -s -H "Authorization: Bearer $KEY" -H "User-Agent: maxim-peer/1.0" \
     https://yourhost/v1/debug/heartbeat
curl -s -H "Authorization: Bearer $KEY" -H "User-Agent: maxim-peer/1.0" \
     https://yourhost/v1/debug/metrics
```

The following **modifies leader state** — use only when explicitly asked:

```bash
# Pull + install on leader:
maxim peer update

# Soft-restart leader (reloads code):
maxim peer restart
```
