- **Dispatch checks the tool it runs, and a refusal is typed and teaches nothing** (#1315 item 3).
  - The executor looks the tool up once, checks that object against the policy's permissions, its effect ceiling
    (`Effect`, ANDed with the offline ceiling while the offline latch is engaged) and any pin, then runs that same
    object.
  - Every gate refusal carries a `ToolErrorKind` (`not_registered`, `inactive_scene`, `policy_ceiling`,
    `policy_permissions`, `policy_pin`, `mode`, `offline`) and `metadata["refusal_layer"]`.
  - A gate refusal is booked to the model's carryover, the recent outcomes and the autonomy audit log, but no longer
    to NAc, goal credit, cluster reward or the plan bridge. The `tools.allow` / `tools.deny` denials, unregistered
    names and inactive scene tools used to book a negative outcome. Unregistered and inactive-scene calls no longer
    reach the pain detector.
  - `record_outcome` takes a required `refused=`. The parallel batch now writes an autonomy audit entry per executed
    action.
  - Under the offline latch the tool-pain bridge books nothing.
  - `SpawnSubSimulationTool` builds its child from the dispatching executor's policy, and refuses outside a dispatch.
  - `auto_sense` runs an auto-fire tool, or `sense`, only when the executor's gates would.
- **`maxim.run()`'s filesystem tools reach only the run's home**, never the process working directory (#1342, first
  half; the prefetch half stays open).
