# Architecture / front-gate lens — nociception_layer.md

**Verdict: ADOPT WITH CHANGES.** The vocabulary (step 1) is right, and so is the front-gate answer for a
kind on `Reaction`, though its reason is stated wrongly. Four changes are needed: step 2 needs a module
move the plan does not mention, the cluster-fear migration in step 3 has to be dropped, the fix for F1
should come before the migration, and step 6's goal cannot be reached as written. The consumer table
leaves out a pain consumer that has the F1 defect.

## DO-NOT-ADOPT

1. **Step 3 does not keep cluster fear's set; it widens it.** Fear's allowlist lists failure modes:
   `decisions/nac.py::DEFAULT_CLUSTER_FEAR_FAILURE_MODES = {drive:health, drive:oxygen}`. That set is
   enforced in `NAc.record_cluster_fear`. It is also the **hivemind ingest gate**
   (`hivemind/ingest.py`, `hivemind/bundle.py` import it), which makes it a wire boundary. The set
   `{NOCICEPTIVE, DRIVE}` is not the same set. It also takes every motor strain, every world hit and
   hunger, and fear's own docstring rules hunger out. `DRIVE(oxygen)` can only be written by adding the
   drive name to the kind, which moves the finer rule into the enum. Keep cluster fear on its
   failure-mode allowlist and take it out of step 3. A kind filter is at most a coarse filter applied
   before that allowlist.
2. **The goal in step 6, "make the one pain path into NAc explicit", is wrong.** Five PainBus
   subscribers write to NAc, each into its own map, and that is by design (see the comments in
   `pain_bus.py::build_pain_bus`): `create_pain_nac_subscriber`, `create_percept_valence_subscriber`,
   `create_pain_cluster_fear_subscriber`, `ToolPainBridge`, `PainCircuitBridge`. There is no single
   path to make explicit. Renaming two bridges behind deprecation aliases is a naming change dressed as
   architecture. A docstring does the same job at no risk. Drop the rename and keep the real F2
   defect (SHOULD-FIX 4).

## SHOULD-FIX

1. **Step 2 cannot import `PainKind` where it lives.** `proprioception/pain.py` already imports
   `reactions.types` (`require_unit_intensity`), so a `pain_kind: PainKind | None` field on
   `reactions/types.py::Reaction` would create an import cycle. The plan has to say that `PainKind`
   (and possibly `classify_pain`) moves to `reactions/types.py` or to a leaf module, with a re-export
   from `proprioception/pain`. The `__post_init__` should reject a `pain_kind` on any reaction whose
   `kind != "pain"`.
2. **The front-gate reason (i) is only half true, and that changes the step order.** ReactionBus
   consumers *can* classify most pain today. Every direct producer puts the `PainType` in the source
   suffix (`perceived_pain:anticipated`, `pain_interceptor:external_signal`), and
   `pain_bus.py::_reaction_to_pain_signal` already parses it. What is really lost is the drive line:
   `reactions/compat.py::pain_signal_to_reaction` drops `context["source"]`, so `drive:oxygen` and
   `drive:health` both arrive as `pain_detector:external_signal`, which is NOCICEPTIVE. That is the
   honest justification for step 2. It also means F1 depends only on step 2, not on step 3. Order the
   steps **1 → 2 → 4 (F1, a live defect) → 3 (a refactor)**. A defect fix should not wait behind seven
   consumer migrations.
3. **The consumer table leaves out `hippocampus.capture_reaction`** (`runtime/bio_stack.py`, step 4b,
   `subscribe_all`). It feeds `memory/episode.py::PendingEpisode.finalize`'s `net_valence` with every
   negative reaction, anticipated pain included. That is **the same defect as F1, in memory**, and the
   plan's step 4 does not cover it. Also missing: `_sim_log_reaction` (harmless) and the public
   `api.py` `on("pain_signal")` → `PainSignalEvent` surface (CC2, field-additive). That surface needs a
   `kind` if it is live; I found no agent-bus `PainSignal` publisher, so verify whether it is dead.
4. **The F2 description is wrong.** `bridges/pain_bridge.py::PainCircuitBridge._on_pain` filters on
   nothing except `enable_learning` and a pending action. It takes **every** kind, not "motion pain".
   So air hunger, a failing tool or anticipated pain that arrives while a DefaultNetwork motion is
   pending gets attributed to that motion. That is the double-attribution path the plan calls
   unobserved, and it is structural. Fix it by giving the bridge a kind filter, not by renaming it.
5. **The lossy bridge back to PainBus turns unclassified reactions into nociception.**
   `_reaction_to_pain_signal` rebuilds a signal with no `source`. Any direct producer whose suffix does
   not parse falls back to `EXTERNAL_SIGNAL` and therefore to NOCICEPTIVE:
   `cerebellum:<entity>.<affordance>`, `sim_adapter:<meta>`, and `conversational_source:<free text>`.
   So step 1's memory capture and `ToolPainBridge._note_felt_pain` count those as pain. Because
   `PainSignal.kind` is a *derived* property, it cannot survive the Reaction round trip. Step 2 should
   make kind **stored** (set in `__post_init__`, defaulting to `classify_pain`) so the reconstruction
   can carry `reaction.pain_kind` across.
6. **Push the invariant into the type (CLAUDE.md), which cuts both ways.** (a) `PainSignal.kind`
   raises `ValueError` *when read*, inside subscribers, and `PainBus.publish` catches that and logs a
   warning. An unclassified type becomes a WARNING per consumer, not a failure at construction. Compute
   it in `__post_init__`, and add a test that `classify_pain` covers every `PainType` member. (b) The
   2S-c incident was a consumer that *forgot* the filter. An optional `subscribe(cb, kinds=...)` would
   not have prevented it, because leaving it out means "all". To earn its place in the front-gate, make
   `kinds=` **required and keyword-only**, as `build_executor(pain_bus=)` is. Otherwise a
   `signal.kind in X` guard in each handler is enough, and it also covers the legacy
   `PainDetector.add_pain_callback` path that both bridges still fall back to. A subscribe-level
   filter would not cover that path.
7. **`kind` on a mutable dataclass.** `PainSignal` is a non-frozen `@dataclass` with a mutable
   `context`. A derived `kind` can change after publish if a subscriber mutates `context["source"]`.
   Storing kind at construction (item 5) fixes this too.

## NIT

- The claim that `PainSignal` is constructed in 4 modules is wrong: it is 6. The list is missing
  `proprioception/perceived_pain.py` and `proprioception/pain.py` (PainDetector).
- `pain.py::failure_pain_kind` parses `drive:<sensor>:<band>` the same way
  `embodiment/tool_bridge.py::_drive_failure_sensor` does, and they disagree on `drive:health` with no
  band. Proprioception cannot import embodiment, so record the lockstep in a comment.
- `pain_bus.py` imports `PainKind` with `# noqa: F401`. Re-export through `__all__` instead.
- ReactionBus refractory is keyed on `kind:source`, so all body pain shares
  `pain:pain_detector:external_signal`. A `drive:oxygen` and a `drive:health` within 0.5 s collapse
  into one before reward and capture see them. This is out of scope, but it is another reason to put
  kind in a field rather than in `source` (changing `source` would change refractory behaviour).
- Keep `PainKind` a flat enum. The drive name already lives in `failure_mode` / `source`, and the
  consumer that needs it (fear) keys on that string.

## Verified fine

- **CC3, step 2:** `Reaction`'s `SHAPE-FROZEN at 1.0` marker explicitly allows appending fields with
  defaults. `Reaction.to_dict` / `from_dict` have no non-test callers, and `from_dict` must use
  `.get`. No hivemind or persistence path serializes `Reaction`: episodes keep reactions in memory and
  persist only `net_valence`. The isolation rule covers `ReactionContext`. A pain class on `Reaction`
  does not leak cross-agent intent.
- The no-third-bus stance and adaptation living at the producer (step 5) match the pain_bus
  unification record.
- Step 1 as shipped: `_pain_encoding` gives the same output. Every `PainType` member is classified
  today (14 of 14).
