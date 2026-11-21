- **A leader that reuses the LLM server already on its port now comes up with its cloudflared tunnel and heartbeat**
  (#1610): they used to start only after a fresh spawn. A leader's proxy, tunnel and heartbeat now have one owner,
  the `maxim` CLI process that binds the leader proxy. It starts the tunnel and heartbeat when it builds its LLM,
  whatever the LLM path decides: a fresh spawn, a reused server, an opted-out spawn, or a build that fails (server
  mode stays reachable for remote recovery). A sub-sim on the leader box shares the leader's proxy and never starts
  or stops the tunnel, and a tunnel that cannot start (an unreadable `/etc/cloudflared`) is logged rather than
  failing the LLM build.
