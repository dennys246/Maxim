- **A web page the model fetches can no longer hold the call open by sending slowly** (#1715): `fetch_url` takes a
  `deadline_s` that bounds the whole request. Every socket read, write, TLS handshake and connect, and the wait for
  a pooled connection, is cut to what remains of it. Before, httpx timed each read separately, so a server that
  dripped its body or headers, or sent `100 Continue` forever, could hold `HttpFetchTool` indefinitely.
  - `HttpFetchTool` gives each call one budget (its per-read timeout plus 2 s), shared by the robots.txt redirects
    and the page. `internet_search` uses the same bound.
  - Not bounded: the DNS lookup, and HTTPS through an `https://` proxy (design: `docs/plans/http_deadline_1715.md`).
