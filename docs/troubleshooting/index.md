# Troubleshooting Guides

In-depth troubleshooting docs for specific subsystems. For quick-reference tables, see [docs/user/troubleshooting.md](../user/troubleshooting.md).

## Guides

| Guide | When to use |
|-------|-------------|
| [maxim doctor](maxim_doctor.md) | Understanding `maxim doctor` output, fixing each check, debug flags |
| [Peer → Leader Connectivity](peer_leader_connectivity.md) | Peer can't reach the leader's GPU, tunnel issues, auth failures, lane routing bugs |
| [Peer Diagnosis Runbook](peer_diagnosis_runbook.md) | Step-by-step bisection ladder (DNS → edge → tunnel → auth → lane → GPU). Copy-pasteable commands, safe for autonomous Claude agents |
| [Remote Update](remote_update.md) | `maxim peer update` failures — 404 (tunnel routing), 403 (disabled), 409 (dirty tree), 500 (git conflicts), Cloudflare 1010 (bot block). Decision tree + autonomous agent commands |
| [Tool Aliases](tool_aliases.md) | AUT hallucinating tool names in simulation mode. How the alias system works, how to add new aliases, how to diagnose redirect issues |
| [Mesh Debug](mesh_debug.md) | **Plan 4 Stage C operator runbook.** `mesh.yml` declarative config + `init-mesh` / `add-node` / `remove-node` setup verbs + `drain` / `resume` / `list-drained` runtime state. First place to look for "I added a node and it isn't routing" or "I drained a peer and it didn't come back." |
| [Agent Mesh (historical)](mesh.md) | Post-mortem of the deleted R0 mesh scaffolding (`PeerRegistry`, `PeerChannel`, `TaskDelegator`, `ExperienceBroker`). Read this if you arrived from a stale link — current surface is in [Mesh Debug](mesh_debug.md). |
| [Benchmarks](benchmarks.md) | Benchmark runner issues, metric comparison failures, scenario validation |
| [Embodiment](embodiment.md) | YAML loading, SEM entity issues, motor program failures, pain bus, cerebellum |
| [Bio-Systems](biosystems.md) | Diagnosing bio-inspired subsystem issues using DM campaigns. Hippocampus recall, NAc learning, SCN temporal bins, ATL concepts, PainBus, Cerebellum, SensoryGate, ChooseTool. Pipeline audit script usage. |
