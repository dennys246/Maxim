# Authoring Default Network behaviors

A **Behavior** is a lightweight reactive module in the Default Network (DN) — it looks at the
current perceptual input each DN tick and optionally proposes one motor action. Behaviors are how
Maxim turns toward a novel object, tracks a face, startles at a sudden appearance, or (the work in
progress) orients toward a sound — all **without** LLM deliberation, at the DN thread's ~30 Hz.

This folder documents how to write them:

- **[README.md](README.md)** (this file) — the `Behavior` contract, lifecycle, arbitration,
  inhibition, and the gotchas that bite. Modality-agnostic.
- **[vision_behaviors.md](vision_behaviors.md)** — the **detection-driven** path (vision). This is
  the mature, shipped path: every behavior in `default_network/behaviors/` today is visual.
- **[audio_behaviors.md](audio_behaviors.md)** — the **audio-orient** path. **Not yet buildable** —
  it needs five generic-DN additions the visual path already has for free. This doc records exactly
  what's missing and why, so the audio reflex (the Exp 45 learned orient policy) can be implemented
  without re-discovering the seams.

See also: [../default_network.md](../default_network.md) (DN architecture overview),
[../plans/hybrid_substrate_reflex_runtime.md](../plans/hybrid_substrate_reflex_runtime.md) (the audio
reflex plan), and [../plans/reviews/hybrid_runtime_two_lens_review.md](../plans/reviews/hybrid_runtime_two_lens_review.md)
(the three-lens review that surfaced the audio gaps).

---

## The contract

A behavior subclasses `Behavior` ([`behaviors/base.py`](../../src/maxim/default_network/behaviors/base.py)):

```python
from maxim.default_network.behaviors.base import Behavior, BehaviorState
from maxim.default_network.messages import ActionProposal

class MyBehavior(Behavior):
    name = "my_behavior"          # unique identifier; also the inhibition/arbitration key
    base_priority = 0.6           # 0.0–1.0; scaled per-proposal
    cooldown_seconds = 0.5        # minimum time between activations (0 = none)
    enabled = True

    def evaluate(self, detections: list[dict], state: BehaviorState) -> ActionProposal | None:
        if not self.can_activate():          # respect the cooldown
            return None
        # ... decide, cheaply (<10 ms, non-blocking) ...
        if not_triggered:
            return None
        return self._create_proposal(         # records activation + stamps base_priority
            action_type="look_at",
            target=(u, v),                     # (see "targets" below — load-bearing)
            priority_scale=1.0,
            confidence=0.9,
            some_metadata="value",
        )
```

`evaluate()` is the only abstract method. Everything else (`can_activate`, `record_activation`,
`reset`, `time_since_activation`, `_create_proposal`) is provided.

### `evaluate()` — the one rule that dominates all others

> **It must run in `<10 ms` and must never block.** It runs on the DN loop thread, called for every
> enabled behavior every tick. A blocking call (network I/O, `sleep`, a motor command that waits for
> completion) freezes the entire reactive layer. If you need external data (head pose, a sensor
> reading, a NAc reference), it is **pushed in by DN via a setter** between ticks — never pulled
> synchronously inside `evaluate()`. See "Getting external state in" below.

Exceptions raised in `evaluate()` are caught and logged, then the behavior is skipped for that tick
([`network.py` `_evaluate_behaviors`](../../src/maxim/default_network/network.py)). So a bug won't
crash DN — but it also won't be loud. Don't rely on the swallow; validate your inputs.

### `ActionProposal` — what you return

Frozen dataclass ([`messages.py`](../../src/maxim/default_network/messages.py)):

| field | meaning |
|---|---|
| `behavior_name` | your `name` (set for you by `_create_proposal`) |
| `action_type` | dispatch key — must have a branch in `_dispatch_action_to_motor` (see below) |
| `target` | `(u, v)` pixel coords, or `None`. **Load-bearing** — see the targets gotcha |
| `priority` | `base_priority * priority_scale` |
| `confidence` | 0.0–1.0 |
| `metadata` | free-form dict the motor dispatch reads (e.g. `turn_angle`, `duration`) |

`effective_score() = priority * confidence` — this is what the arbiter ranks on.

---

## Lifecycle: how a proposal becomes motion

Per DN tick, in [`network.py::_process_tick`](../../src/maxim/default_network/network.py):

1. **Idle-exploration short-circuit** — if nothing interesting recently, DN may run idle scan and
   return early.
2. **Input gate** — `if not detections: return`. *(Today this is the **visual** gate; it is why an
   audio behavior can't fire yet — see [audio_behaviors.md](audio_behaviors.md).)*
3. **Perception subsystems update** — novelty / spatial / salience / movement / scene.
4. **State push into behaviors** — `_update_head_position_for_behaviors` pushes current head pose and
   interests into behaviors that need them (via `isinstance` checks + setters).
5. **`_evaluate_behaviors`** — builds a `BehaviorState`, loops enabled+un-inhibited behaviors, collects
   proposals, applies per-behavior priority modifiers.
6. **`PriorityArbiter.select`** — picks one winner across all proposals (with hysteresis).
7. **`_dispatch_action_to_motor`** — routes the winner's `action_type` to a `self._maxim` method.

### Registration

Behaviors are **constructed and passed as a list** to `DefaultNetwork(behaviors=[...])`
([`network.py:271`](../../src/maxim/default_network/network.py); default set at ~175). There is no
dynamic `register_behavior()` — you add your class to the constructed list (in the runtime's DN
build, gated on capability/flags as appropriate). Export it from
[`behaviors/__init__.py`](../../src/maxim/default_network/behaviors/__init__.py).

### Getting external state in (the setter pattern)

`evaluate()` receives only `detections` + `state`. Anything else your behavior needs — head yaw, a
novelty tracker, a sensor reading, a `NAc` reference — is injected by DN **between ticks** via a
setter you define, called from `_update_head_position_for_behaviors` (or a sibling). Precedent:
`TurnAround.set_head_yaw()` / `set_novelty_tracker()` / `set_interests()`
([`turn_around.py:88-98`](../../src/maxim/default_network/behaviors/turn_around.py)), pushed at
[`network.py:871-874`](../../src/maxim/default_network/network.py). **A new external input means a new
`isinstance` branch in that push loop** — that is generic-DN plumbing, not "just a behavior." Budget
for it.

### `action_type` needs a motor branch

`_dispatch_action_to_motor` ([`network.py:1227-1281`](../../src/maxim/default_network/network.py))
handles exactly `look_at`, `scan`, `track`, `turn_around` today. A **new** `action_type` (e.g.
`turn_body`) is a silent no-op until you add its branch. Dispatch runs on the DN thread, so the motor
method it calls **must return promptly** — a method that blocks for the duration of the motion (as
`turn_around` does) stalls the loop.

---

## `BehaviorState` — what you can read

([`base.py:21-74`](../../src/maxim/default_network/behaviors/base.py))

| field | use |
|---|---|
| `inhibited_behaviors: frozenset[str]` | names DN has suppressed this tick (you're already skipped if inhibited) |
| `priority_modifiers: dict[str, float]` | per-behavior multipliers (mode config) |
| `current_goals: list[str]` | active goal keywords |
| `interests: frozenset[int]` | class IDs currently of interest |
| `fear_level: float` | caution from FearAgent (0–1) |
| `frame_timestamp: float` | this tick's time |
| `salience_map: SalienceMap \| None` | spatial attention; `get_tracking_bonus` / `record_tracking_target` |
| `focus_learner: FocusLearner \| None` | adaptive movement-gain correction |

---

## Arbitration & inhibition

- **Within DN**, `PriorityArbiter.select` picks one winner across behaviors by `effective_score()`
  with hysteresis. Set `base_priority` relative to siblings deliberately (e.g. startle 0.95 >
  orienting 0.8 > turn_around 0.3). If you add a high-priority reflex, make sure it **habituates or
  cools down** or it starves the others.
- **Inhibition** is expressed by `name`: a behavior whose name is in `state.inhibited_behaviors` is
  skipped. Today `inhibited_behaviors` is populated from **mode config** (`priority_modifier <= 0`),
  **not** from live per-tool deliberative suppression. If you want "the LLM doing a voluntary head
  move suppresses my reflex," that hook does not exist yet — it is net-new wiring (see the audio doc
  and review SF-1). Inhibition is one-way by design: cognition suppresses reflex, never the reverse.

---

## Gotchas (learned, load-bearing)

1. **`<10 ms`, non-blocking, no I/O in `evaluate()`.** The single most important rule. Push external
   data in via setters between ticks.
2. **Target-less proposals skip the salience / IOR / fear gates.** Those gates `return True` when
   `proposal.target` is falsy, and IOR runs in visual `(u, v)` space. A behavior that proposes a bare
   action with `target=None` (a body turn, say) is **not** novelty/IOR-gated — "don't re-fire on the
   same thing" is silently absent. Give the proposal a spatial target, or add an explicit gate.
   *(This is BL-5 for the audio reflex.)*
3. **A new `action_type` is a no-op until it has a motor branch**, and that motor method must not
   block the DN thread.
4. **The head-frame invariant applies to any body-rotation dispatch.** On Reachy, commanding
   `body_yaw` with `head=None` counter-rotates the head in world frame, so head-mounted sensors
   (camera, mics) don't turn. Any behavior that rotates the body must dispatch through a path that
   ships an explicit head matrix. See the CLAUDE.md Reachy head-frame invariant.
5. **Exceptions are swallowed** — validate inputs; a silently-skipped behavior looks like "never
   triggers."
6. **Respect `can_activate()`** — call it first; call `record_activation()` (or use
   `_create_proposal`, which does) when you emit.

---

## Checklist for a new behavior

- [ ] Subclass `Behavior`, set `name` / `base_priority` / `cooldown_seconds`.
- [ ] `evaluate()` is `<10 ms` and non-blocking; external state arrives via a setter.
- [ ] If it needs new external state → add the `isinstance` push branch in DN (generic-DN change).
- [ ] If it uses a new `action_type` → add a non-blocking motor branch; ship a head matrix if it turns the body.
- [ ] Give proposals a `target` if you want salience/IOR gating (or add an explicit gate).
- [ ] Set `base_priority` relative to siblings; add cooldown/habituation for reflexes.
- [ ] Register it in the DN `behaviors=[...]` list (capability/flag-gated) and export from `__init__.py`.
- [ ] Offline test on a fake input + assert the proposal / cooldown / inhibition paths.
