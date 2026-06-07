# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.9.x   | Yes (current pre-release) |
| 1.0.x   | Yes (forthcoming)         |
| < 0.9   | No                        |

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

## Best Practices

- Set appropriate autonomy level (`--autonomy planning|supervised|autonomous`)
- Use `MAXIM_LLM_REDACTION_POLICY=strict` for cloud providers
- Review `maxim doctor` output before exposing to network
- For Mother Maxim: run security stress campaign before going public
