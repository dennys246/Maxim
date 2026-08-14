# HTTP call sites must use `maxim/utils/http.py`

**Archived from CLAUDE.md on 2026-08-13** (claude_md_diet Stage 1). The enforced rule
survives as a compressed stub — in the slim CLAUDE.md core or in the owning
`docs/agents/<subsystem>.md` brief (see CLAUDE.md's routing table). This file preserves
the full original narrative: incident history, dates, PR numbers, dead-end hypotheses.

---

**[engineering] HTTP call sites must use `maxim/utils/http.py`:** Plan 1 R1 consolidated ~11 scattered `urllib.request` call sites into one registry-backed module. The invariant is CI-enforced: `grep -r "urllib.request.urlopen" src/maxim/ | grep -v "utils/http.py"` must return zero matches. The 2026-04-12 Cloudflare Bot Fight Mode incident (commit `8b52cbd`) was a missing `User-Agent` header in one of those call sites — the `_external` endpoint in `utils/http.py` sets it once at registration, so that class of bug is structurally impossible now. When adding a new outbound HTTP call: pick `http.get`/`http.post` (registered endpoint), `http.fetch_url` (arbitrary URL), or `http.download_to_file` (streaming file). The `raw_proxy_forward` escape hatch is reserved for `leader_proxy._proxy_request` ONLY — do not use it elsewhere. Regression guard: CI grep `grep -r "urllib.request.urlopen" src/maxim/ | grep -v "utils/http.py"` must return zero matches; enforced in [.github/workflows/test.yml](.github/workflows/test.yml).
