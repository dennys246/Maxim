- **`maxim peer connect` says where every lane goes, and `--all-lanes` sends them all to the leader** (#1780): connect
  still points only the large lane at the leader by default, because the leader serves one generation at a time. It
  now prints each lane's destination: the leader, a local model, or none on this machine. `--all-lanes` points large,
  medium and small at the leader. A later connect without it moves medium and small back to local, and
  `maxim peer forget` clears every lane. New troubleshooting page: `docs/troubleshooting/tailscale_peer.md`.
