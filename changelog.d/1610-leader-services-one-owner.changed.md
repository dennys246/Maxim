- **Only the `maxim` CLI leader starts a leader's proxy and cloudflared tunnel** (#1610, owner decisions
  2026-11-09). `maxim serve` (the console) and Python-API processes (`maxim.create`, `api.run`, research) on a
  leader used to start them, with the leader heartbeat, after a fresh LLM spawn, and no longer do. They share the
  proxy a running `maxim` leader holds; the console still runs its own health sampling. On a leader box, run `maxim`
  to bring the leader up.
