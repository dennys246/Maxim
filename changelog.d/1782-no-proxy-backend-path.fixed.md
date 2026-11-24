- **A backend can refuse every proxy, and `NO_PROXY` on the backend path is pinned** (#1782). A `maxim_peer` or
  `openai` provider entry can now set `strict_address_check: true`; no provider entry read the setting before. That
  backend's model calls and its own probes (warmup's model discovery, `health_check`) then connect directly, even
  with `HTTP(S)_PROXY` set. A custom CA in `SSL_CERT_FILE` / `SSL_CERT_DIR` still applies. Separately, an `https://`
  endpoint used to ignore `strict_address_check` entirely, and offline `https://` clients dropped the env CA along
  with the proxy; both are fixed. `NO_PROXY` was already honoured on the backend path, and tests now pin it on each
  backend's real client. Maxim's own operator commands still go through a configured proxy (owner decision D16).
