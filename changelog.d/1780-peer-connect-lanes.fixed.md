- **`maxim peer connect --cloud` is accepted** (#1780). The code read the flag from the command line, but the parser
  had already rejected it as an unknown option. Its banner no longer claims a cloud gate the flag never sets.
- **`maxim peer connect` says what the leader serves** (#1780): the large lane, which every model call goes through,
  and that it falls back to a local model when the leader is unreachable at startup. New troubleshooting page:
  `docs/troubleshooting/tailscale_peer.md`.
