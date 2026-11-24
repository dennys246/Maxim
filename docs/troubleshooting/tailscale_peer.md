# Tailscale peer: a laptop using a leader's model over Tailscale

Use this page when a machine on your tailnet (the **peer**, usually a laptop) runs Maxim and sends its model calls to
another machine (the **leader**, such as a Mac mini) that serves the model. No tunnel is needed: Tailscale's
WireGuard already encrypts the traffic, so plain `http://` to the leader's tailnet address is fine.

For any other peer-to-leader failure, see [Peer → Leader Connectivity](peer_leader_connectivity.md) and the
[Peer Diagnosis Runbook](peer_diagnosis_runbook.md).

## The setup

**On the leader:** run `maxim` as a leader. It starts two things:
- the model server (`llama_cpp.server`) on `127.0.0.1:8100`, which only the leader itself can reach;
- the **leader proxy** on `0.0.0.0:8099`. It checks the mesh key and, when it knows the model's context window, rejects a
  prompt larger than it. Peers talk to this port, never to 8100.

Its key is the mesh key: `maxim tunnel key show` on the leader prints it.

**On the peer:**

```bash
maxim peer connect http://<leader-tailnet-name-or-100.x-address>:8099/v1
maxim peer test http://<leader>:8099/v1
maxim doctor --as peer http://<leader>:8099/v1
```

`connect` asks for the key (or takes `--key`), tests the connection, and saves it.

## What the leader serves

Maxim sends its model calls through one lane, the **large** lane. `peer connect` points that lane at the leader, so
the leader serves every model call this machine makes. Maxim also defines medium and small lanes, but no model call
uses them today, so there is nothing to point at the leader there.

If the leader is unreachable when `maxim` starts, the large lane falls back to local: a local model this machine can
run, if it has one. `maxim doctor --as peer <url>` tells you whether the leader answers now; restart `maxim` once it
does.

## Treat the leader as one generation at a time

The leader spawns its model server with no parallel slots, so Maxim treats it as running one generation at a time.
Release 1.3.5's line 6 item 1 measures this under load. The leader proxy lets up to 4 requests in at once
(`maxim config set proxy.max_concurrent N` on the leader; env `MAXIM_PROXY_MAX_CONCURRENT`) and answers anything
past that with an immediate **HTTP 429**. It does not queue them. A per-peer rate limit (`proxy.rate_limit_rpm`) also
answers 429, with "Rate limit exceeded". So:
- two peers, or a peer and an experiment on the leader, share that one generation;
- a 429 means "busy", not "broken". Retry later rather than reconfiguring.

Line 6 item 3 replaces the instant 429 with a bounded queue ([release_1_3_5.md](../plans/release_1_3_5.md)).

**Don't run a game or playtest against the leader while an experiment holds it.** The experiment's numbers depend on
having the server to itself (decision D12 in the same plan).

## When it fails

| Symptom | Likely cause | Check |
|---|---|---|
| `connect` says the connection test failed | wrong port (8100 instead of 8099), the leader is not running, or Tailscale is down on one side | `tailscale ping <leader>`; `curl -s -o /dev/null -w '%{http_code}' http://<leader>:8099/v1/models` should print `401` without a key |
| `401` from every call | wrong or rotated key | `maxim peer key set` with the leader's current `maxim tunnel key show` |
| `429` under load | more requests than the proxy's cap, or this peer's rate limit | the section above; lower your own concurrency |
| calls run locally although the leader is configured | the leader was unreachable at startup, so the large lane fell back to local | `maxim doctor --as peer`; restart `maxim` once the leader answers |
| `doctor --as peer` says `inference_broken` while the leader is busy | a long generation held the one slot past the probe's timeout | retry when the leader is idle |
