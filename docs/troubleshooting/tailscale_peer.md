# Tailscale peer: a laptop using a leader's model over Tailscale

Use this page when a machine on your tailnet (the **peer**, usually a laptop) runs Maxim and sends its model calls to
another machine (the **leader**, such as a Mac mini) that serves the model. No tunnel is needed: Tailscale's
WireGuard already encrypts the traffic, so plain `http://` to the leader's tailnet address is fine.

For any other peer-to-leader failure, see [Peer → Leader Connectivity](peer_leader_connectivity.md) and the
[Peer Diagnosis Runbook](peer_diagnosis_runbook.md).

## The setup

**On the leader:** run `maxim` as a leader. It starts two things:
- the model server (`llama_cpp.server`) on `127.0.0.1:8100`, which only the leader itself can reach;
- the **leader proxy** on `0.0.0.0:8099`. It checks the mesh key and rejects a prompt larger than the model's
  context window. Peers talk to this port, never to 8100.

Its key is the mesh key: `maxim tunnel key export` on the leader prints it.

**On the peer:**

```bash
maxim peer connect http://<leader-tailnet-name-or-100.x-address>:8099/v1
maxim peer test http://<leader>:8099/v1
maxim doctor --as peer http://<leader>:8099/v1
```

`connect` asks for the key (or takes `--key`), tests the connection, and saves it.

## Which lanes go to the leader

Maxim has three lanes: large, medium and small. `peer connect` points **only the large lane** at the leader, and
prints where every lane goes. For example, on a laptop with no usable GPU:

```text
Lanes:
  large   → leader (http://mini:8099/v1)
  medium  → local, phi-3-mini-4k-instruct on cpu
  small   → local, smollm-1.7b-instruct on cpu
  Use --all-lanes to send every lane to the leader (they share its one generation slot).
```

A lane marked **local** runs a model on this machine. A local model that is not on disk yet may be downloaded the
first time a call needs it. To send every lane to the leader instead:

```bash
maxim peer connect http://<leader>:8099/v1 --all-lanes
```

A later `connect` without `--all-lanes` moves medium and small back to local. `maxim peer forget` clears every lane.
`maxim peer show` prints where medium and small go.

You can also point one lane by hand, for example `maxim config set lanes.medium.remote_url http://<leader>:8099/v1`.
That works even on a machine whose hardware has no medium tier: the remote URL creates the lane.

## The leader serves one generation at a time

The leader's model server runs one generation at a time. The proxy lets up to 4 requests in
(`MAXIM_PROXY_MAX_CONCURRENT`, default 4), and answers anything past that with an immediate **HTTP 429**. It does not
queue them. So:
- every lane you send to the leader waits behind the others; that is why `--all-lanes` is not the default;
- two peers, or a peer and an experiment on the leader, share that one slot;
- a 429 means "busy", not "broken". Retry later rather than reconfiguring.

Release 1.3.5's line 6 measures this under load and replaces the instant 429 with a bounded queue
([release_1_3_5.md](../plans/release_1_3_5.md), line 6 items 1 and 3).

**Don't run a game or playtest against the leader while an experiment holds it.** The experiment's numbers depend on
having the server to itself (decision D12 in the same plan).

## When it fails

| Symptom | Likely cause | Check |
|---|---|---|
| `connect` says the connection test failed | wrong port (8100 instead of 8099), the leader is not running, or Tailscale is down on one side | `tailscale ping <leader>`; `curl -s -o /dev/null -w '%{http_code}' http://<leader>:8099/v1/models` should print `401` without a key |
| `401` from every call | wrong or rotated key | `maxim peer key set` with the leader's current `maxim tunnel key export` |
| `429` under load | more requests than the proxy's cap | the section above; lower your own concurrency |
| a lane you expected on the leader runs locally | it was not pointed at the leader | the `Lanes:` block from `connect`, or `maxim peer show` |
| `doctor --as peer` says `inference_broken` while the leader is busy | a long generation held the one slot past the probe's timeout | retry when the leader is idle |
