# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.3.x   | Yes (current) |
| < 1.3   | No — upgrade  |

## Reporting a Vulnerability

If you discover a security vulnerability in Maxim, please report it responsibly:

1. **Do NOT open a public issue** for security vulnerabilities
2. Open a private security advisory at https://github.com/dennys246/Maxim/security/advisories/new
3. Include: description, reproduction steps, potential impact, and suggested fix if any
4. Expected response time: 48 hours for acknowledgment, 7 days for initial assessment

## Security-Critical Components

Maxim includes several safety-critical systems:

### Agent Safety
- **FearAgent** — pre-execution safety review of all tool calls
- **PainDetector** — detects aversive patterns and harmful actions
- **HarmRegistry** — zero-latency prediction of harmful outcomes
- **AutonomyController** — gates agent actions by autonomy level

### Data Safety
- **Atomic file writes** — all persistence uses fsync + tmp + replace pattern
- **Internet access policy** — configurable whitelist/blacklist for web access
- **Filesystem policy** — sandboxed file access with configurable boundaries
- **Cloud redaction** — PII filtering before cloud LLM dispatch

### Deidentification (Mother Maxim — PLANNED, post-1.0)

> **Not yet implemented.** The features below are design goals for the Mother Maxim / Oasis layer
> (targeted for 1.1+). The only current privacy mechanism for substrate sharing is a heuristic
> identity-bearing label filter in `hivemind/identity.py` (used to quarantine identity-shaped
> NAc links and EC nodes from substrate bundles). The full ATL+SEM deidentification pipeline
> described below does not exist in 1.0.

- **Bio-system-aware deidentification** *(planned)* — ATL + SEM identity map extracts names/locations deterministically
- **Dual-pass pipeline** *(planned)* — client-side deidentification + server-side verification
- **Model tier gate** *(planned)* — contributions declare deidentification model; weak models rejected

### Network exposure (leader role)
- **A leader is never a default.** A machine becomes a leader only through `MAXIM_ROLE`, `config.json::role` or a
  cloudflared config. With no signal it is `solo` and binds loopback only.
- **A leader refuses to start without a usable mesh key** at `~/.config/maxim/api_key`. The key must be present,
  at least 16 ASCII characters, and mode 0600 on POSIX.
- **The LeaderProxy (:8099) is the one network ingress.** Every route needs `Authorization: Bearer <key>`,
  including `/v1/debug/*` from loopback, because a cloudflared tunnel delivers internet requests from `localhost`.
  The leader's llama-cpp-server (:8100) binds `127.0.0.1`; its key is passed in the child's environment, never argv.
- **Remote admin (`/v1/admin/{update,install,restart,llm-swap}`) is off by default.** It requires the bearer key
  AND the operator's opt-in, `maxim config set proxy.allow_remote_update true`.
- **`maxim oasis serve --insecure`** is the only way to serve without a key. It never serves remote admin.

## Best Practices

- Set appropriate autonomy level (`--autonomy planning|supervised|autonomous`)
- Use `MAXIM_LLM_REDACTION_POLICY=strict` for cloud providers
- Review `maxim doctor` output before exposing to network (its API key section fails a leader that could not start,
  and shows whether remote admin is on and why)
- For Mother Maxim: run security stress campaign before going public
