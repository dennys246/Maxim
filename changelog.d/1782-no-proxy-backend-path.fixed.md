- **A backend can refuse every proxy, and `NO_PROXY` on the backend path is pinned** (#1782). A provider entry's
  `strict_address_check: true` now reaches both the Maxim-peer and the OpenAI-compatible backends. Before, an
  `https://` backend ignored the setting and still read `HTTP(S)_PROXY`. `NO_PROXY` was already honoured there, and
  tests now pin it on each backend's real client. Maxim's own operator commands still go through a configured proxy
  (owner decision D16).
