# Troubleshooting: LeaderProxy Not Reachable (404 / Connection Refused)

> **Symptom:** `maxim peer update` or `maxim peer version` returns 404 or "connection refused"
> **Root cause:** The Cloudflare tunnel is reaching port 8099, but either the LeaderProxy isn't running, or it started but the tunnel is routing to port 8100 (llama-cpp-server) instead.

## Quick Diagnosis

Run these on the **leader machine**:

```bash
# 1. Is the proxy listening?
ss -ltnp | grep 8099

# 2. Is it the LeaderProxy or something else on 8099?
curl -s http://localhost:8099/v1/debug/version
# Expected: {"version": "0.1.0", "git_hash": "..."}
# If you get {"detail": "Not Found"} → that's llama-cpp-server, not the proxy

# 3. What does the tunnel point at?
grep -A1 service ~/.cloudflared/config.yml
# Should show: service: http://localhost:8099

# 4. Did the early proxy boot run?
# Check maxim's output for this line:
# "LeaderProxy listening on 0.0.0.0:8099 → upstream http://127.0.0.1:8100"

# 5. Is leader mode detected?
python3 -c "from maxim.runtime.leader_mode import detect_role; print(detect_role())"
# Should show: RoleDecision(role='leader', ...)
```

## Common Issues

### A. Leader mode not detected → proxy doesn't start

The early proxy boot in `cli.py` runs `detect_role()` which checks:
1. `MAXIM_ROLE=leader` env var (highest priority)
2. `~/.cloudflared/config.yml` exists (auto-detect)

**Fix:** If neither is set:
```bash
# Option 1: Set env var explicitly
export MAXIM_ROLE=leader
maxim

# Option 2: Verify cloudflared config exists
ls -la ~/.cloudflared/config.yml
```

### B. API key not found → proxy starts without auth

The early boot reads the key from `~/.config/maxim/api_key`. If that file doesn't exist, the proxy starts with `api_key=None` (no auth enforced).

```bash
# Check key exists
cat ~/.config/maxim/api_key

# If missing, generate one
maxim tunnel key rotate
```

### C. Port 8099 already in use → proxy silently skips

If a stale process holds port 8099, `start_leader_proxy()` logs a warning and returns None.

```bash
# Find what's on 8099
ss -ltnp | grep 8099

# Kill stale process if needed
kill $(ss -ltnp | grep 8099 | grep -oP 'pid=\K\d+')

# Restart maxim
maxim
```

### D. Tunnel pointing at wrong port

The cloudflared config must route to 8099 (LeaderProxy), NOT 8100 (llama-cpp-server).

```bash
# Check
grep service ~/.cloudflared/config.yml

# Fix if wrong
sed -i 's|service: http://localhost:8100|service: http://localhost:8099|' ~/.cloudflared/config.yml

# Restart cloudflared
sudo systemctl restart cloudflared
```

### E. Early boot code not loaded (old version running)

If the running maxim process was started before the early-proxy-boot commit, it won't have the code. Verify:

```bash
# Check which commit is running
git log --oneline -1
# Should show: 70df927 or later

# If old, pull and restart
git pull --rebase origin main
pip install -e .
pkill -f "python.*maxim"
maxim
```

### F. Import error in early boot (silently caught)

The early boot is wrapped in `try/except Exception: pass` so import failures are silent. Test manually:

```bash
python3 -c "
from maxim.runtime.leader_mode import detect_role
print('Role:', detect_role())

from maxim.tunnel.keys import read_key
print('Key:', read_key()[:6] if read_key() else 'None')

from maxim.runtime.leader_proxy import start_leader_proxy
print('Proxy import OK')
"
```

If any import fails, that's the blocker. Common causes:
- Missing dependency
- Circular import (check traceback)
- File permission issue on key file

## Debugging the Early Boot Code

The relevant code is in `src/maxim/cli.py` right after arg parsing:

```python
# ── Early leader proxy bootstrap ─────────────────────────────────
try:
    from maxim.runtime.leader_mode import detect_role
    _role = detect_role()
    if _role.role == "leader":
        import os as _os
        _os.environ.setdefault("MAXIM_ALLOW_REMOTE_UPDATE", "1")
        from maxim.runtime.leader_proxy import start_leader_proxy
        _api_key = None
        try:
            from maxim.tunnel.keys import read_key
            _api_key = read_key()
        except Exception:
            pass
        start_leader_proxy(api_key=_api_key, bind_host=_role.bind_host)
except Exception:
    pass  # Not a leader, or deps missing — fine, skip silently
```

To debug, temporarily replace the outer `except Exception: pass` with:

```python
except Exception as e:
    import traceback
    traceback.print_exc()
```

Then restart maxim and check stdout for the traceback.

## Nuclear Option: Force Proxy Start

If detection isn't working, force-start the proxy by setting the env var:

```bash
MAXIM_ROLE=leader maxim
```

This bypasses cloudflared config detection and forces leader mode, which triggers the early proxy boot.

## Verification After Fix

```bash
# From the leader:
curl -s http://localhost:8099/v1/debug/version
# Should return: {"version": "0.1.0", "git_hash": "70df927", ...}

# From a peer:
maxim peer version
# Should show matching versions with no error

# Full test:
maxim peer update --dry-run
maxim peer logs
```
